//! A collection of lazy initialized values that are created by `Future`s.
//!
//! [OnceCell]'s API is similar to the [`once_cell`](https://crates.io/crates/once_cell) crate,
//! [`std::cell::OnceCell`], or [`std::sync::OnceLock`].  It provides an async version of a cell
//! that can only be initialized once, permitting tasks to wait on the initialization if it is
//! already running instead of racing multiple initialization tasks.
//!
//! Unlike threads, tasks can be cancelled at any point where they block.  [OnceCell] deals with
//! this by allowing another initializer to run if the task currently initializing the cell is
//! dropped.  This also allows for fallible initialization using [OnceCell::get_or_try_init] and
//! for the initializing `Future` to contain borrows or use references to thread-local data.
//!
//! [Lazy] takes the opposite approach: it wraps a single `Future` which is cooperatively run to
//! completion by any polling task.  This requires that the initialization function be independent
//! of the calling context, but will never restart an initializing function just because the
//! surrounding task was cancelled.  Using a trait object (`Pin<Box<dyn Future>>`) for the future
//! may simplify using this type in data structures.
//!
//! # Overhead
//!
//! Both cells use two `usize`s to store state and do not retain any allocations after
//! initialization is complete.  Allocations are only required if there is contention.
//!
//! Accessing an already-initialized cell is as cheap as possible: only one atomic load with
//! Acquire ordering.
//!
//! # Features
//!
//! ## The `critical-section` feature
//!
//! If this feature is enabled, the [`critical-section`](https://crates.io/crates/critical-section)
//! crate is used instead of an `std` mutex.  You must depend on that crate and select a locking
//! implementation; see [its documentation](https://docs.rs/critical-section/) for details.
//!
//! ## The `std` feature
//!
//! This is currently a no-op, but might in the future be used to expose APIs that depends on
//! types only in `std`.  It does *not* control the locking implementation.

// How it works:
//
// The basic design goal of async_once_cell is to make the simpler, more common cases as fast and
// efficient as possible while reverting to a reasonably performant implementation when that's not
// possible.
//
// The fastest path is "access an already-initialized cell": this takes one atomic load with
// acquire ordering, and doing it with any less is not possible without extreme, platform-specific
// mechanisms (for example, the membarrier system call on Linux) which would make filling the cell
// significantly more expensive.
//
// The fast path for filling a cell is when there is no contention.  The types in this crate will
// not allocate in this scenario, which proceeds according to this summary:
//
//  1. A single task runs get_or_try_init, which calls Inner::initialize(true)
//  2. Inner::state transitions from NEW to QINIT_BIT, and a QuickInitGuard is returned
//  3. The init future is run and completes successfully (possibly after yielding)
//  4. The value is written to the UnsafeCell
//  5. Inner::state transitions from QINIT_BIT to READY_BIT during QuickInitGuard's Drop
//
// If the init future fails (due to returning an error or a panic), then:
//  4. The UnsafeCell remains uninitialized
//  5. Inner::state transitions from QINIT_BIT to NEW during QuickInitGuard's Drop
//
// The fast path does not use Inner::queue at all, and only needs to check it once the cell
// transitions to the ready state (in order to handle the edge case where a queue was created but
// was not actually needed).
//
// Slow path:
//
// If a second task attempts to start initialization, it will not succeed in transitioning
// Inner::state from NEW to QINIT_BIT.  Instead, it will create a Queue on the heap, storing it in
// Inner::queue and creating a QueueRef pointing at it.  This Queue will hold the Wakers for all
// tasks that attempt to perform initialization.  When a QuickInitGuard or QueueHead is dropped,
// all tasks are woken and will either proceed directly to obtaining a reference (if initialization
// was successful) or race to create a new QueueHead, with losers re-queuing in a new Waker list.
//
// Once a Queue has been created for an Inner, it remains valid as long as either a reference
// exists (as determined by the reference count in Inner::state) or the state is not ready.  A
// QueueRef represents one reference to the Queue (similar to how Arc<Queue> would act).
//
// The wake-up behavior used here is optimized for the common case where an initialization function
// succeeds and a mass wake-up results in all woken tasks able to proceed with returning a
// reference to the just-stored value.  If initialization fails, it would in theory be possible to
// only wake one of the pending tasks, since only one task will be able to make useful progress by
// becoming the new QueueHead.  However, to avoid a lost wakeup, this would require tracking wakers
// and removing them when a QueueRef is dropped.  The extra overhead required to maintain the list
// of wakers is not worth the extra complexity and locking in the common case where the QueueRef
// was dropped due to a successful initialization.

#![cfg_attr(feature = "critical-section", no_std)]
extern crate alloc;

#[cfg(any(not(feature = "critical-section"), feature = "std"))]
extern crate std;

use alloc::{boxed::Box, vec, vec::Vec};

use core::{
    cell::UnsafeCell,
    convert::Infallible,
    fmt,
    future::{Future, IntoFuture},
    marker::{PhantomData, PhantomPinned},
    mem::{self, ManuallyDrop, MaybeUninit},
    panic::{RefUnwindSafe, UnwindSafe},
    pin::Pin,
    ptr,
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering},
    task,
};

#[cfg(feature = "critical-section")]
struct Mutex<T> {
    data: UnsafeCell<T>,
    locked: core::sync::atomic::AtomicBool,
}

#[cfg(feature = "critical-section")]
impl<T> Mutex<T> {
    const fn new(data: T) -> Self {
        Mutex { data: UnsafeCell::new(data), locked: core::sync::atomic::AtomicBool::new(false) }
    }
}

#[cfg(not(feature = "critical-section"))]
use std::sync::Mutex;

#[cfg(feature = "critical-section")]
fn with_lock<T, R>(mutex: &Mutex<T>, f: impl FnOnce(&mut T) -> R) -> R {
    struct Guard<'a, T>(&'a Mutex<T>);
    impl<'a, T> Drop for Guard<'a, T> {
        fn drop(&mut self) {
            self.0.locked.store(false, Ordering::Relaxed);
        }
    }
    critical_section::with(|_| {
        if mutex.locked.swap(true, Ordering::Relaxed) {
            // Note: this can in theory happen if the delegated Clone impl on a Waker provided in
            // an initialization context turns around and tries to initialize the same cell.  This
            // is an absurd thing to do, but it's safe so we can't assume nobody will ever do it.
            panic!("Attempted reentrant locking");
        }
        let guard = Guard(mutex);
        // Safety: we just checked that we were the one to set `locked` to true, and the data in
        // this Mutex will only be accessed while the lock is true.  We use Relaxed memory ordering
        // instead of Acquire/Release because critical_section::with itself must provide an
        // Acquire/Release barrier around its closure, and also guarantees that there will not be
        // more than one such closure executing at a time.
        let rv = unsafe { f(&mut *mutex.data.get()) };
        drop(guard);
        rv
    })
}

#[cfg(not(feature = "critical-section"))]
fn with_lock<T, R>(mutex: &Mutex<T>, f: impl FnOnce(&mut T) -> R) -> R {
    f(&mut *mutex.lock().unwrap())
}

/// A cell which can be written to only once.
///
/// This allows initialization using an async closure that borrows from its environment.
///
/// ```
/// use std::rc::Rc;
/// use std::sync::Arc;
/// use async_once_cell::OnceCell;
///
/// # async fn run() {
/// let non_send_value = Rc::new(4);
/// let shared = Arc::new(OnceCell::new());
///
/// let value : &i32 = shared.get_or_init(async {
///     *non_send_value
/// }).await;
/// assert_eq!(value, &4);
///
/// // A second init is not called
/// let second = shared.get_or_init(async {
///     unreachable!()
/// }).await;
/// assert_eq!(second, &4);
///
/// # }
/// # use std::future::Future;
/// # struct NeverWake;
/// # impl std::task::Wake for NeverWake {
/// #     fn wake(self: Arc<Self>) {}
/// # }
/// # let w = Arc::new(NeverWake).into();
/// # let mut cx = std::task::Context::from_waker(&w);
/// # assert!(std::pin::pin!(run()).poll(&mut cx).is_ready());
/// ```
pub struct OnceCell<T> {
    value: UnsafeCell<MaybeUninit<T>>,
    inner: Inner,
    _marker: PhantomData<T>,
}

// Safety: our UnsafeCell should be treated like an RwLock<T>
unsafe impl<T: Sync + Send> Sync for OnceCell<T> {}
unsafe impl<T: Send> Send for OnceCell<T> {}
impl<T> Unpin for OnceCell<T> {}
impl<T: RefUnwindSafe + UnwindSafe> RefUnwindSafe for OnceCell<T> {}
impl<T: UnwindSafe> UnwindSafe for OnceCell<T> {}

/// Monomorphic portion of the state of a OnceCell or Lazy.
///
/// The top two bits of state are flags (READY_BIT and QINIT_BIT) that define the state of the
/// cell.  The rest of the bits count the number of QueueRef objects associated with this Inner.
///
/// The queue pointer starts out as NULL.  If contention is detected during the initialization of
/// the object, it is initialized to a Box<Queue>, and will remain pointing at that Queue until the
/// state has changed to ready with zero active QueueRefs.
struct Inner {
    state: AtomicUsize,
    queue: AtomicPtr<Queue>,
}

impl fmt::Debug for Inner {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let state = self.state.load(Ordering::Relaxed);
        let queue = self.queue.load(Ordering::Relaxed);
        fmt.debug_struct("Inner")
            .field("ready", &(state & READY_BIT != 0))
            .field("quick_init", &(state & QINIT_BIT != 0))
            .field("refcount", &(state & (QINIT_BIT - 1)))
            .field("queue", &queue)
            .finish()
    }
}

/// Transient state during initialization
///
/// Unlike the sync OnceCell, this cannot be a linked list through stack frames, because Futures
/// can be freed at any point by any thread.  Instead, this structure is allocated on the heap
/// during the first initialization call and freed after the value is set (or when the OnceCell is
/// dropped, if the value never gets set).
struct Queue {
    wakers: Mutex<Option<Vec<task::Waker>>>,
}

/// A reference to the Queue held inside an Inner.
///
/// This is somewhat like Arc<Queue>, the refcount is held in Inner instead of Queue so it can be
/// freed once the cell's initialization is complete.
///
/// Holding a QueueRef guarantees that either:
///  - queue points to a valid Queue that will not be freed until this QueueRef is dropped
///  - inner.state is ready
///
/// The value of QueueRef::queue may be dangling or null if inner.state was ready at the time the
/// value was loaded.  The holder of a QueueRef must observe a non-ready state prior to using
/// queue; because this is already done by all holders of QueueRef for other reasons, this second
/// check is not included in Inner::initialize.
///
/// The creation of a QueueRef performs an Acquire ordering operation on Inner::state; its Drop
/// performs a Release on the same value.
///
/// The value of QueueRef::queue may also become dangling during QueueRef's Drop impl even when the
/// lifetime 'a is still valid, so a raw pointer is required for correctness.
struct QueueRef<'a> {
    inner: &'a Inner,
    queue: *const Queue,
}
// Safety: the queue is a reference (only the lack of a valid lifetime requires it to be a pointer)
unsafe impl<'a> Sync for QueueRef<'a> {}
unsafe impl<'a> Send for QueueRef<'a> {}

/// A write guard for an active initialization of the associated UnsafeCell
///
/// This is created on the fast (no-allocation) path only.
#[derive(Debug)]
struct QuickInitGuard<'a> {
    inner: &'a Inner,
    ready: bool,
}

/// A Future that waits for acquisition of a QueueHead
struct QueueWaiter<'a> {
    guard: Option<QueueRef<'a>>,
}

/// A write guard for the active initialization of the associated UnsafeCell
///
/// Creation of a QueueHead must always be done with the Queue's Mutex held.  If no QuickInitGuard
/// exists, the task creating the QueueHead is the task that transitions the contents of the Mutex
/// from None to Some; it must verify QINIT_BIT is unset with the lock held.
///
/// Only QueueHead::drop may transition the contents of the Mutex from Some to None.
///
/// Dropping this object will wake all tasks that have blocked on the currently-running
/// initialization.
struct QueueHead<'a> {
    guard: QueueRef<'a>,
}

const NEW: usize = 0x0;
const QINIT_BIT: usize = 1 + (usize::MAX >> 2);
const READY_BIT: usize = 1 + (usize::MAX >> 1);
const EMPTY_STATE: usize = !0;

impl Inner {
    const fn new() -> Self {
        Inner { state: AtomicUsize::new(NEW), queue: AtomicPtr::new(ptr::null_mut()) }
    }

    const fn new_ready() -> Self {
        Inner { state: AtomicUsize::new(READY_BIT), queue: AtomicPtr::new(ptr::null_mut()) }
    }

    /// Initialize the queue (if needed) and return a waiter that can be polled to get a QueueHead
    /// that gives permission to initialize the OnceCell.
    ///
    /// The Queue referenced in the returned QueueRef will not be freed until the cell is populated
    /// and all references have been dropped.  If any references remain, further calls to
    /// initialize will return the existing queue.
    #[cold]
    fn initialize(&self, try_quick: bool) -> Result<QueueWaiter, QuickInitGuard> {
        if try_quick {
            if self
                .state
                .compare_exchange(NEW, QINIT_BIT, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                // On success, we know that there were no other QueueRef objects active, and we
                // just set QINIT_BIT which makes us the only party allowed to create a QueueHead.
                // This remains true even if the queue is created later.
                return Err(QuickInitGuard { inner: self, ready: false });
            }
        }

        // Increment the queue's reference count.  This ensures that queue won't be freed until we exit.
        let prev_state = self.state.fetch_add(1, Ordering::Acquire);

        // Note: unlike Arc, refcount overflow is impossible.  The only way to increment the
        // refcount is by calling poll on the Future returned by get_or_try_init, which is !Unpin.
        // The poll call requires a Pinned pointer to this Future, and the contract of Pin requires
        // Drop to be called on any !Unpin value that was pinned before the memory is reused.
        // Because the Drop impl of QueueRef decrements the refcount, an overflow would require
        // more than (usize::MAX / 4) QueueRef objects in memory, which is impossible as these
        // objects take up more than 4 bytes.

        let mut guard = QueueRef { inner: self, queue: self.queue.load(Ordering::Acquire) };

        if guard.queue.is_null() && prev_state & READY_BIT == 0 {
            let wakers = Mutex::new(None);

            // Race with other callers of initialize to create the queue
            let new_queue = Box::into_raw(Box::new(Queue { wakers }));

            match self.queue.compare_exchange(
                ptr::null_mut(),
                new_queue,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_null) => {
                    // Normal case: it was actually set.  The Release part of AcqRel orders this
                    // with all Acquires on the queue.
                    guard.queue = new_queue;
                }
                Err(actual) => {
                    // we lost the race, but we have the (non-null) value now.
                    guard.queue = actual;
                    // Safety: we just allocated it, and nobody else has seen it
                    unsafe {
                        drop(Box::from_raw(new_queue));
                    }
                }
            }
        }
        Ok(QueueWaiter { guard: Some(guard) })
    }

    fn set_ready(&self) {
        // This Release pairs with the Acquire any time we check READY_BIT, and ensures that the
        // writes to the cell's value are visible to the cell's readers.
        let prev_state = self.state.fetch_or(READY_BIT, Ordering::Release);

        debug_assert_eq!(prev_state & READY_BIT, 0, "Invalid state: someone else set READY_BIT");
    }
}

impl<'a> Drop for QueueRef<'a> {
    fn drop(&mut self) {
        // Release the reference to queue
        let prev_state = self.inner.state.fetch_sub(1, Ordering::Release);
        // Note: as of now, self.queue may be invalid

        let curr_state = prev_state - 1;
        if curr_state == READY_BIT || curr_state == READY_BIT | QINIT_BIT {
            // We just removed the only waiter on an initialized cell.  This means the
            // queue is no longer needed.  Acquire the queue again so we can free it.
            let queue = self.inner.queue.swap(ptr::null_mut(), Ordering::Acquire);
            if !queue.is_null() {
                // Safety: the last guard is being freed, and queue is only used by guard-holders.
                // Due to the swap, we are the only one who is freeing this particular queue.
                unsafe {
                    drop(Box::from_raw(queue));
                }
            }
        }
    }
}

impl<'a> Drop for QuickInitGuard<'a> {
    fn drop(&mut self) {
        // When our QuickInitGuard was created, Inner::state was changed to QINIT_BIT.  If it is
        // either unchanged or has changed back to that value, we can finish on the fast path.
        let fast_target = if self.ready { READY_BIT } else { NEW };
        if self
            .inner
            .state
            .compare_exchange(QINIT_BIT, fast_target, Ordering::Release, Ordering::Relaxed)
            .is_ok()
        {
            // Because the exchange succeeded, we know there are no active QueueRefs and so no
            // wakers need to be woken.  If self.ready is true, the Release ordering pairs with
            // the Acquire on another thread's access to state to check READY_BIT.

            if self.ready {
                // It's possible (though unlikely) that someone created the queue but abandoned
                // their QueueRef before we finished our poll, resulting in us not observing
                // them.  No wakes are needed in this case because there are no waiting tasks,
                // but we should still clean up the allocation.
                let queue = self.inner.queue.swap(ptr::null_mut(), Ordering::Relaxed);
                if !queue.is_null() {
                    // Synchronize with both the fetch_sub that lowered the refcount and the
                    // queue initialization.
                    core::sync::atomic::fence(Ordering::Acquire);
                    // Safety: we observed no active QueueRefs, and queue is only used by
                    // guard-holders.  Due to the swap, we are the only one who is freeing this
                    // particular queue.
                    unsafe {
                        drop(Box::from_raw(queue));
                    }
                }
            }
            return;
        }

        // Slow path: get a guard, create the QueueHead we should have been holding, then drop it
        // so that the tasks are woken as intended.  This is needed regardless of if we succeeded
        // or not - either waiters need to run init themselves, or they need to read the value we
        // set.
        //
        // The guard is guaranteed to have been created with no QueueHead available because
        // QINIT_BIT is still set.
        let waiter = self.inner.initialize(false).expect("Got a QuickInitGuard in slow init");
        let guard = waiter.guard.expect("No guard available even without polling");

        // Safety: the guard holds a place on the waiter list, and we know READY_BIT was not yet
        // set when Inner::initialize was called, so the queue must be present.  It will remain
        // valid until guard is dropped.
        debug_assert!(!guard.queue.is_null(), "Queue must not be NULL when READY_BIT is not set");
        let queue = unsafe { &*guard.queue };

        with_lock(&queue.wakers, |lock| {
            // Creating a QueueHead requires that the Mutex contain Some.  While this is likely
            // already true, it is not guaranteed because the first concurrent thread might have
            // been preempted before it was able to start its first QueueWaiter::poll call.  Ensure
            // that nobody else can grab the QueueHead between when we release QINIT_BIT and when
            // our QueueHead is dropped.
            lock.get_or_insert_with(Vec::new);

            // We must clear QINIT_BIT, which will allow someone else to take the head position
            // once we drop it.
            //
            // If our initialization was successful, we also need to set READY_BIT.  These
            // operations can be combined because we know the current state of both bits (only
            // QINIT_BIT is set) and because READY_BIT == 2 * QINIT_BIT.
            //
            // Ordering for QINIT_BIT is handled by the Mutex, but ordering for READY_BIT is not;
            // it needs Release ordering to ensure that the UnsafeCell's value is visible prior to
            // that bit being observed as set by other threads.
            let prev_state = if self.ready {
                self.inner.state.fetch_add(QINIT_BIT, Ordering::Release)
            } else {
                self.inner.state.fetch_sub(QINIT_BIT, Ordering::Relaxed)
            };
            debug_assert_eq!(
                prev_state & (QINIT_BIT | READY_BIT),
                QINIT_BIT,
                "Invalid state during QuickInitGuard drop"
            );
        });

        // Safety: we just took the head position
        drop(QueueHead { guard })
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let queue = *self.queue.get_mut();
        if !queue.is_null() {
            // Safety: nobody else could have a reference
            unsafe {
                drop(Box::from_raw(queue));
            }
        }
    }
}

impl<'a> Future for QueueWaiter<'a> {
    type Output = Option<QueueHead<'a>>;
    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut task::Context<'_>,
    ) -> task::Poll<Option<QueueHead<'a>>> {
        let guard = self.guard.as_ref().expect("Polled future after finished");

        // Fast path for waiters that get notified after the value is set
        let state = guard.inner.state.load(Ordering::Acquire);
        if state & READY_BIT != 0 {
            return task::Poll::Ready(None);
        }

        // Safety: the guard holds a place on the waiter list and we just checked that the state is
        // not ready, so the queue is non-null and will remain valid until guard is dropped.
        let queue = unsafe { &*guard.queue };
        let rv = with_lock(&queue.wakers, |lock| {
            // Another task might have set READY_BIT between our optimistic lock-free check and our
            // lock acquisition.  Don't return a QueueHead unless we know for sure that we are
            // allowed to initialize.
            let state = guard.inner.state.load(Ordering::Acquire);
            if state & READY_BIT != 0 {
                return task::Poll::Ready(None);
            }

            match lock.as_mut() {
                None if state & QINIT_BIT == 0 => {
                    // take the head position and start a waker queue
                    *lock = Some(Vec::new());

                    task::Poll::Ready(Some(()))
                }
                None => {
                    // Someone else has a QuickInitGuard; they will wake us when they finish.
                    let waker = cx.waker().clone();
                    *lock = Some(vec![waker]);
                    task::Poll::Pending
                }
                Some(wakers) => {
                    // Wait for the QueueHead to be dropped
                    let my_waker = cx.waker();
                    for waker in wakers.iter() {
                        if waker.will_wake(my_waker) {
                            return task::Poll::Pending;
                        }
                    }
                    wakers.push(my_waker.clone());
                    task::Poll::Pending
                }
            }
        });

        // Safety: If rv is Ready/Some, we know:
        //  - we are holding a QueueRef (in guard) that prevents state from being 0
        //  - creating a new QuickInitGuard requires the state to be 0
        //  - we just checked QINIT_BIT and saw there isn't a QuickInitGuard active
        //  - the queue was None, meaning there are no current QueueHeads
        //  - we just set the queue to Some, claiming the head
        //
        // If rv is Ready/None, this is due to READY_BIT being set.
        // If rv is Pending, we have a waker in the queue.
        rv.map(|o| o.map(|()| QueueHead { guard: self.guard.take().unwrap() }))
    }
}

impl<'a> Drop for QueueHead<'a> {
    fn drop(&mut self) {
        // Safety: if queue is not null, then it is valid as long as the guard is alive, and a
        // QueueHead is never created with a NULL queue (that requires READY_BIT to have been set
        // inside Inner::initialize, and in that case no QueueHead objects will be created).
        let queue = unsafe { &*self.guard.queue };

        // Take the waker queue, allowing another QueueHead to be created if READY_BIT is unset.
        let wakers =
            with_lock(&queue.wakers, Option::take).expect("QueueHead dropped without a waker list");

        for waker in wakers {
            waker.wake();
        }
    }
}

impl<T> OnceCell<T> {
    /// Creates a new empty cell.
    pub const fn new() -> Self {
        Self {
            value: UnsafeCell::new(MaybeUninit::uninit()),
            inner: Inner::new(),
            _marker: PhantomData,
        }
    }

    /// Creates a new cell with the given contents.
    pub const fn new_with(value: T) -> Self {
        Self {
            value: UnsafeCell::new(MaybeUninit::new(value)),
            inner: Inner::new_ready(),
            _marker: PhantomData,
        }
    }

    /// Gets the contents of the cell, initializing it with `init` if the cell was empty.
    ///
    /// Many tasks may call `get_or_init` concurrently with different initializing futures, but
    /// it is guaranteed that only one future will be executed as long as the resulting future is
    /// polled to completion.
    ///
    /// If `init` panics, the panic is propagated to the caller, and the cell remains uninitialized.
    ///
    /// If the Future returned by this function is dropped prior to completion, the cell remains
    /// uninitialized, and another `init` function will be started (if any are available).
    ///
    /// Attempting to reentrantly initialize the cell from `init` will generally cause a deadlock;
    /// the reentrant call will immediately yield and wait for the pending initialization.  If the
    /// actual initialization can complete despite this (for example, by polling multiple futures
    /// and discarding incomplete ones instead of polling them to completion), then the cell will
    /// successfully be initialized.
    pub async fn get_or_init(&self, init: impl Future<Output = T>) -> &T {
        match self.get_or_try_init(async move { Ok::<T, Infallible>(init.await) }).await {
            Ok(t) => t,
            Err(e) => match e {},
        }
    }

    /// Gets the contents of the cell, initializing it with `init` if the cell was empty.   If the
    /// cell was empty and `init` failed, an error is returned.
    ///
    /// Many tasks may call `get_or_init` and/or `get_or_try_init` concurrently with different
    /// initializing futures, but it is guaranteed that only one of the futures will be executed as
    /// long as the resulting future is polled to completion.
    ///
    /// If `init` panics or returns an error, the panic or error is propagated to the caller, and
    /// the cell remains uninitialized.  In this case, another `init` function from a concurrent
    /// caller will be selected to execute, if one is available.
    ///
    /// If the Future returned by this function is dropped prior to completion, the cell remains
    /// uninitialized, and another `init` function will be started (if any are available).
    ///
    /// Attempting to reentrantly initialize the cell from `init` will generally cause a deadlock;
    /// the reentrant call will immediately yield and wait for the pending initialization.  If the
    /// actual initialization can complete despite this (for example, by polling multiple futures
    /// and discarding incomplete ones instead of polling them to completion), then the cell will
    /// successfully be initialized.
    pub async fn get_or_try_init<E>(
        &self,
        init: impl Future<Output = Result<T, E>>,
    ) -> Result<&T, E> {
        let state = self.inner.state.load(Ordering::Acquire);

        if state & READY_BIT == 0 {
            self.init_slow(state == NEW, init).await?;
        }

        // Safety: initialized on all paths
        Ok(unsafe { (*self.value.get()).assume_init_ref() })
    }

    #[cold]
    async fn init_slow<E>(
        &self,
        try_quick: bool,
        init: impl Future<Output = Result<T, E>>,
    ) -> Result<(), E> {
        match self.inner.initialize(try_quick) {
            Err(mut guard) => {
                // Try to proceed assuming no contention.
                let value = init.await?;
                // Safety: the guard acts like QueueHead even if there is contention.
                unsafe {
                    (*self.value.get()).write(value);
                }
                guard.ready = true;
                drop(guard);
            }
            Ok(guard) => {
                if let Some(init_lock) = guard.await {
                    // We hold the QueueHead, so we know that nobody else has successfully run an init
                    // poll and that nobody else can start until it is dropped.  On error, panic, or
                    // drop of this Future, the head will be passed to another waiter.
                    let value = init.await?;

                    // Safety: We still hold the head, so nobody else can write to value
                    unsafe {
                        (*self.value.get()).write(value);
                    }
                    // mark the cell ready before giving up the head
                    init_lock.guard.inner.set_ready();
                    // drop of QueueHead notifies other Futures
                    // drop of QueueRef (might) free the Queue
                } else {
                    // someone initialized it while waiting on the queue
                }
            }
        }
        Ok(())
    }

    /// Gets the reference to the underlying value.
    ///
    /// Returns `None` if the cell is empty or being initialized. This method never blocks.
    pub fn get(&self) -> Option<&T> {
        let state = self.inner.state.load(Ordering::Acquire);

        if state & READY_BIT == 0 {
            None
        } else {
            Some(unsafe { (*self.value.get()).assume_init_ref() })
        }
    }

    /// Gets a mutable reference to the underlying value.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        let state = *self.inner.state.get_mut();
        if state & READY_BIT == 0 {
            None
        } else {
            Some(unsafe { self.value.get_mut().assume_init_mut() })
        }
    }

    /// Takes the value out of this `OnceCell`, moving it back to an uninitialized state.
    pub fn take(&mut self) -> Option<T> {
        let state = *self.inner.state.get_mut();
        self.inner = Inner::new();
        if state & READY_BIT == 0 {
            None
        } else {
            Some(unsafe { self.value.get_mut().assume_init_read() })
        }
    }

    /// Consumes the OnceCell, returning the wrapped value. Returns None if the cell was empty.
    pub fn into_inner(mut self) -> Option<T> {
        self.take()
    }
}

impl<T: fmt::Debug> fmt::Debug for OnceCell<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let value = self.get();
        fmt.debug_struct("OnceCell").field("value", &value).field("inner", &self.inner).finish()
    }
}

impl<T> Drop for OnceCell<T> {
    fn drop(&mut self) {
        let state = *self.inner.state.get_mut();
        if state & READY_BIT != 0 {
            unsafe {
                self.value.get_mut().assume_init_drop();
            }
        }
    }
}

impl<T> Default for OnceCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> From<T> for OnceCell<T> {
    fn from(value: T) -> Self {
        Self::new_with(value)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use alloc::sync::Arc;
    use core::pin::pin;

    #[derive(Default)]
    struct CountWaker(AtomicUsize);
    impl alloc::task::Wake for CountWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    struct CmdWait<'a>(&'a AtomicUsize);
    impl Future for CmdWait<'_> {
        type Output = usize;
        fn poll(self: Pin<&mut Self>, _: &mut task::Context<'_>) -> task::Poll<usize> {
            match self.0.load(Ordering::Relaxed) {
                0 => task::Poll::Pending,
                n => task::Poll::Ready(n),
            }
        }
    }

    impl Drop for CmdWait<'_> {
        fn drop(&mut self) {
            if self.0.load(Ordering::Relaxed) == 6 {
                panic!("Panic on drop");
            }
        }
    }

    async fn maybe(cmd: &AtomicUsize, cell: &OnceCell<usize>) -> Result<usize, usize> {
        cell.get_or_try_init(async {
            match dbg!(CmdWait(cmd).await) {
                1 => Err(1),
                2 => Ok(2),
                _ => unreachable!(),
            }
        })
        .await
        .map(|v| *v)
    }

    async fn never_init(cell: &OnceCell<usize>) {
        let v = cell.get_or_init(async { unreachable!() }).await;
        assert_eq!(v, &2);
    }

    #[test]
    fn slow_path() {
        let w = Arc::new(CountWaker::default()).into();
        let mut cx = std::task::Context::from_waker(&w);

        let cmd = AtomicUsize::new(0);
        let cell = OnceCell::new();

        let mut f1 = pin!(maybe(&cmd, &cell));
        let mut f2 = pin!(never_init(&cell));

        println!("{:?}", cell);
        assert!(f1.as_mut().poll(&mut cx).is_pending());
        println!("{:?}", cell);
        assert!(f2.as_mut().poll(&mut cx).is_pending());
        println!("{:?}", cell);
        cmd.store(2, Ordering::Relaxed);
        assert!(f2.as_mut().poll(&mut cx).is_pending());
        assert!(f1.as_mut().poll(&mut cx).is_ready());
        println!("{:?}", cell);
        assert!(f2.as_mut().poll(&mut cx).is_ready());
    }

    #[test]
    fn fast_path_tricked() {
        // f1 will complete on the fast path, but a queue was created anyway
        let w = Arc::new(CountWaker::default()).into();
        let mut cx = std::task::Context::from_waker(&w);

        let cmd = AtomicUsize::new(0);
        let cell = OnceCell::new();

        let mut f1 = pin!(maybe(&cmd, &cell));
        let mut f2 = pin!(never_init(&cell));

        println!("{:?}", cell);
        assert!(f1.as_mut().poll(&mut cx).is_pending());
        println!("{:?}", cell);
        assert!(f2.as_mut().poll(&mut cx).is_pending());
        println!("{:?}", cell);
        cmd.store(2, Ordering::Relaxed);
        f2.set(never_init(&cell));
        println!("{:?}", cell);
        assert!(f1.as_mut().poll(&mut cx).is_ready());
        println!("{:?}", cell);
        assert!(f2.as_mut().poll(&mut cx).is_ready());
    }

    #[test]
    fn second_try() {
        let waker = Arc::new(CountWaker::default());
        let w = waker.clone().into();
        let mut cx = std::task::Context::from_waker(&w);

        let cmd = AtomicUsize::new(0);
        let cell = OnceCell::new();

        let mut f1 = pin!(maybe(&cmd, &cell));
        let mut f2 = pin!(maybe(&cmd, &cell));
        let mut f3 = pin!(maybe(&cmd, &cell));
        let mut f4 = pin!(maybe(&cmd, &cell));

        assert!(f1.as_mut().poll(&mut cx).is_pending());
        assert_eq!(cell.inner.state.load(Ordering::Relaxed), QINIT_BIT);
        assert!(f2.as_mut().poll(&mut cx).is_pending());
        assert!(f3.as_mut().poll(&mut cx).is_pending());
        assert!(f4.as_mut().poll(&mut cx).is_pending());
        assert_eq!(cell.inner.state.load(Ordering::Relaxed), QINIT_BIT | 3);

        cmd.store(1, Ordering::Relaxed);
        // f2 should do nothing, as f1 holds QuickInitGuard
        assert!(f2.as_mut().poll(&mut cx).is_pending());
        assert_eq!(waker.0.load(Ordering::Relaxed), 0);

        // f1 fails, as commanded
        assert_eq!(f1.as_mut().poll(&mut cx), task::Poll::Ready(Err(1)));
        // it released QINIT_BIT (and doesn't still hold a reference)
        assert_eq!(cell.inner.state.load(Ordering::Relaxed), 3);
        // f1 caused a wake to be sent (only one, as they have the same waker)
        assert_eq!(waker.0.load(Ordering::Relaxed), 1);

        // drop one waiting task and check that the refcount drops
        f4.set(maybe(&cmd, &cell));
        assert_eq!(cell.inner.state.load(Ordering::Relaxed), 2);

        // have f2 start init
        cmd.store(0, Ordering::Relaxed);
        assert!(f2.as_mut().poll(&mut cx).is_pending());

        // allow f2 to actually complete init
        cmd.store(2, Ordering::Relaxed);

        // f3 should add itself to the queue again, but not complete
        assert!(f3.as_mut().poll(&mut cx).is_pending());
        assert_eq!(waker.0.load(Ordering::Relaxed), 1);

        assert_eq!(f2.as_mut().poll(&mut cx), task::Poll::Ready(Ok(2)));

        // Nobody else should run their closure
        cmd.store(3, Ordering::Relaxed);

        // Other tasks can now immediately access the value
        assert_eq!(f4.as_mut().poll(&mut cx), task::Poll::Ready(Ok(2)));

        // f3 is still waiting; the queue should not be freed yet, and it should have seen a wake
        assert_eq!(waker.0.load(Ordering::Relaxed), 2);
        assert_eq!(cell.inner.state.load(Ordering::Relaxed), READY_BIT | 1);
        assert!(!cell.inner.queue.load(Ordering::Relaxed).is_null());

        assert_eq!(f3.as_mut().poll(&mut cx), task::Poll::Ready(Ok(2)));
        // the cell should be fully ready, with the queue deallocated

        assert_eq!(cell.inner.state.load(Ordering::Relaxed), READY_BIT);
        assert!(cell.inner.queue.load(Ordering::Relaxed).is_null());

        // no more wakes were sent
        assert_eq!(waker.0.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn lazy_panic() {
        let w = Arc::new(CountWaker::default()).into();

        let cmd = AtomicUsize::new(6);
        let lz = Lazy::new(CmdWait(&cmd));

        assert_eq!(std::mem::size_of_val(&lz), 3 * std::mem::size_of::<usize>(), "Extra overhead?");

        // A panic during F::drop must properly transition the Lazy to ready in order to avoid a
        // double-drop of F or a drop of an invalid T
        assert!(std::panic::catch_unwind(|| {
            let mut cx = std::task::Context::from_waker(&w);
            pin!(lz.get_unpin()).poll(&mut cx)
        })
        .is_err());

        assert_eq!(lz.try_get(), Some(&6));
    }
}

union LazyState<T, F> {
    running: ManuallyDrop<F>,
    ready: ManuallyDrop<T>,
    _empty: (),
}

/// A value which is computed on demand by running a future.
///
/// Unlike [OnceCell], if a task is cancelled, the initializing future's execution will be
/// continued by other (concurrent or future) callers of [Lazy::get].
///
/// ```
/// use std::sync::Arc;
/// use async_once_cell::Lazy;
///
/// # async fn run() {
/// struct Data {
///     id: u32,
/// }
///
/// let shared = Arc::pin(Lazy::new(async move {
///     Data { id: 4 }
/// }));
///
/// assert_eq!(shared.as_ref().await.id, 4);
/// # }
/// # use std::future::Future;
/// # struct NeverWake;
/// # impl std::task::Wake for NeverWake {
/// #     fn wake(self: Arc<Self>) {}
/// # }
/// # let w = Arc::new(NeverWake).into();
/// # let mut cx = std::task::Context::from_waker(&w);
/// # assert!(std::pin::pin!(run()).poll(&mut cx).is_ready());
/// ```
///
/// Using this type with an `async` block in a `static` item requries unstable rust:
///
/// ```no_run
/// #![feature(const_async_blocks)]
/// #![feature(type_alias_impl_trait)]
/// use async_once_cell::Lazy;
/// use std::future::Future;
///
/// type H = impl Future<Output=i32>;
/// static LAZY: Lazy<i32, H> = Lazy::new(async { 4 });
/// ```
///
/// However, it is possile to use if you have a named struct that implements `Future`:
///
/// ```
/// use async_once_cell::Lazy;
/// use std::{future::Future, pin::Pin, task};
///
/// struct F;
/// impl Future for F {
///     type Output = i32;
///     fn poll(self: Pin<&mut Self>, _: &mut task::Context) -> task::Poll<i32> {
///         return task::Poll::Ready(4);
///     }
/// }
///
/// static LAZY: Lazy<i32, F> = Lazy::new(F);
/// ```
///
/// And this type of struct can still use `async` syntax in its implementation:
///
/// ```
/// use async_once_cell::Lazy;
/// use std::{future::Future, pin::Pin, task};
///
/// struct F(Option<Pin<Box<dyn Future<Output=i32> + Sync + Send>>>);
/// impl Future for F {
///     type Output = i32;
///     fn poll(mut self: Pin<&mut Self>, cx: &mut task::Context) -> task::Poll<i32> {
///         Pin::new(self.0.get_or_insert_with(|| Box::pin(async {
///             4
///         }))).poll(cx)
///     }
/// }
///
/// static LAZY: Lazy<i32, F> = Lazy::new(F(None));
/// ```

pub struct Lazy<T, F> {
    value: UnsafeCell<LazyState<T, F>>,
    inner: Inner,
}

// Safety: our UnsafeCell should be treated like (RwLock<T>, Mutex<F>)
unsafe impl<T: Send + Sync, F: Send> Sync for Lazy<T, F> {}
unsafe impl<T: Send, F: Send> Send for Lazy<T, F> {}
impl<T: Unpin, F: Unpin> Unpin for Lazy<T, F> {}
impl<T: RefUnwindSafe + UnwindSafe, F: UnwindSafe> RefUnwindSafe for Lazy<T, F> {}
impl<T: UnwindSafe, F: UnwindSafe> UnwindSafe for Lazy<T, F> {}

impl<T, F> Lazy<T, F>
where
    F: Future<Output = T>,
{
    /// Creates a new lazy value with the given initializing future.
    pub const fn new(future: F) -> Self {
        Self::from_future(future)
    }

    /// Forces the evaluation of this lazy value and returns a reference to the result.
    ///
    /// This is equivalent to calling `.await` on a pinned reference, but is more explicit.
    ///
    /// The [Pin::static_ref] function may be useful if this is a static value.
    pub async fn get(self: Pin<&Self>) -> Pin<&T> {
        self.await
    }
}

enum Step<'a> {
    Start,
    Quick { guard: QuickInitGuard<'a> },
    Wait { guard: QueueWaiter<'a> },
    Run { head: QueueHead<'a> },
}

/// A helper struct for both of [Lazy]'s [IntoFuture]s
///
/// Note: the Lazy value may or may not be pinned, depending on what public struct wraps this one.
struct LazyFuture<'a, T, F> {
    lazy: &'a Lazy<T, F>,
    step: Step<'a>,
    // This is needed to guarantee Inner's refcount never overflows
    _pin: PhantomPinned,
}

impl<'a, T, F> LazyFuture<'a, T, F>
where
    F: Future<Output = T>,
{
    fn poll(&mut self, cx: &mut task::Context<'_>) -> task::Poll<&'a T> {
        struct QuickReadyGuard<'a, T, F> {
            this: &'a Lazy<T, F>,
            value: ManuallyDrop<T>,
            guard: QuickInitGuard<'a>,
        }

        // Prevent double-drop in case of panic in ManuallyDrop::drop
        impl<T, F> Drop for QuickReadyGuard<'_, T, F> {
            fn drop(&mut self) {
                // Safety: the union is currently empty and must be filled with a ready value
                unsafe {
                    let value = ManuallyDrop::take(&mut self.value);
                    (*self.this.value.get()).ready = ManuallyDrop::new(value);
                }
                self.guard.ready = true;
            }
        }

        struct ReadyGuard<'a, T, F> {
            this: &'a Lazy<T, F>,
            value: ManuallyDrop<T>,
            // head is a field here to ensure it is dropped after our Drop
            head: QueueHead<'a>,
        }

        // Prevent double-drop in case of panic in ManuallyDrop::drop
        impl<T, F> Drop for ReadyGuard<'_, T, F> {
            fn drop(&mut self) {
                // Safety: the union is currently empty and must be filled with a ready value
                unsafe {
                    let value = ManuallyDrop::take(&mut self.value);
                    (*self.this.value.get()).ready = ManuallyDrop::new(value);
                }
                self.head.guard.inner.set_ready();
            }
        }

        loop {
            match mem::replace(&mut self.step, Step::Start) {
                Step::Start => {
                    let state = self.lazy.inner.state.load(Ordering::Acquire);

                    if state & READY_BIT == 0 {
                        self.step = match self.lazy.inner.initialize(state == NEW) {
                            Err(guard) => Step::Quick { guard },
                            Ok(guard) => Step::Wait { guard },
                        };
                        continue;
                    }

                    // Safety: we just saw READY_BIT set
                    return task::Poll::Ready(unsafe { &(*self.lazy.value.get()).ready });
                }
                Step::Quick { guard } => {
                    // Safety: the union is in the running state and is pinned like self
                    let init =
                        unsafe { Pin::new_unchecked(&mut *(*self.lazy.value.get()).running) };
                    let value = match init.poll(cx) {
                        task::Poll::Pending => {
                            self.step = Step::Quick { guard };
                            return task::Poll::Pending;
                        }
                        task::Poll::Ready(value) => ManuallyDrop::new(value),
                    };
                    // Safety: the guard acts like QueueHead even if there is contention.
                    // This transitions the union to ready and updates state to reflect that.
                    unsafe {
                        let guard = QuickReadyGuard { this: &self.lazy, value, guard };
                        ManuallyDrop::drop(&mut (*self.lazy.value.get()).running);
                        drop(guard);
                    }

                    // Safety: just initialized
                    return task::Poll::Ready(unsafe { &(*self.lazy.value.get()).ready });
                }
                Step::Wait { mut guard } => match Pin::new(&mut guard).poll(cx) {
                    task::Poll::Pending => {
                        self.step = Step::Wait { guard };
                        return task::Poll::Pending;
                    }
                    task::Poll::Ready(None) => {
                        // Safety: getting None from QueueWaiter means it is ready
                        return task::Poll::Ready(unsafe { &(*self.lazy.value.get()).ready });
                    }
                    task::Poll::Ready(Some(head)) => {
                        self.step = Step::Run { head };
                        continue;
                    }
                },
                Step::Run { head } => {
                    // Safety: the union is in the running state and is pinned like self
                    let init =
                        unsafe { Pin::new_unchecked(&mut *(*self.lazy.value.get()).running) };
                    // We hold the QueueHead, so we know that nobody else has successfully run an init
                    // poll and that nobody else can start until it is dropped.  On error, panic, or
                    // drop of this Future, the head will be passed to another waiter.
                    let value = match init.poll(cx) {
                        task::Poll::Pending => {
                            self.step = Step::Run { head };
                            return task::Poll::Pending;
                        }
                        task::Poll::Ready(value) => ManuallyDrop::new(value),
                    };

                    // Safety: We still hold the head, so nobody else can write to value
                    // This transitions the union to ready and updates state to reflect that.
                    unsafe {
                        let head = ReadyGuard { this: &self.lazy, value, head };
                        ManuallyDrop::drop(&mut (*self.lazy.value.get()).running);

                        // mark the cell ready before giving up the head
                        drop(head);
                    }
                    // drop of QueueHead notifies other Futures
                    // drop of QueueRef (might) free the Queue

                    // Safety: just initialized
                    return task::Poll::Ready(unsafe { &(*self.lazy.value.get()).ready });
                }
            }
        }
    }
}

/// A helper struct for [Lazy]'s [IntoFuture]
pub struct LazyFuturePin<'a, T, F>(LazyFuture<'a, T, F>);

impl<'a, T, F> IntoFuture for Pin<&'a Lazy<T, F>>
where
    F: Future<Output = T>,
{
    type Output = Pin<&'a T>;
    type IntoFuture = LazyFuturePin<'a, T, F>;
    fn into_future(self) -> Self::IntoFuture {
        // Safety: this is Pin::deref, but with a lifetime of 'a
        let lazy = unsafe { Pin::into_inner_unchecked(self) };
        LazyFuturePin(LazyFuture { lazy, step: Step::Start, _pin: PhantomPinned })
    }
}

impl<'a, T, F> Future for LazyFuturePin<'a, T, F>
where
    F: Future<Output = T>,
{
    type Output = Pin<&'a T>;
    fn poll(self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> task::Poll<Pin<&'a T>> {
        // Safety: we don't move anything that needs to be pinned.
        let inner = unsafe { &mut Pin::into_inner_unchecked(self).0 };
        // Safety: because the original Lazy was pinned, the T it produces is also pinned
        inner.poll(cx).map(|p| unsafe { Pin::new_unchecked(p) })
    }
}

impl<T, F> Lazy<T, F>
where
    F: Future<Output = T> + Unpin,
{
    /// Forces the evaluation of this lazy value and returns a reference to the result.
    ///
    /// This is equivalent to calling `.await` on a reference, but may be clearer to call
    /// explicitly.
    ///
    /// Unlike [Self::get], this does not require pinning the object.
    pub async fn get_unpin(&self) -> &T {
        self.await
    }
}

/// A helper struct for [Lazy]'s [IntoFuture]
pub struct LazyFutureUnpin<'a, T, F>(LazyFuture<'a, T, F>);

impl<'a, T, F> IntoFuture for &'a Lazy<T, F>
where
    F: Future<Output = T> + Unpin,
{
    type Output = &'a T;
    type IntoFuture = LazyFutureUnpin<'a, T, F>;
    fn into_future(self) -> Self::IntoFuture {
        LazyFutureUnpin(LazyFuture { lazy: self, step: Step::Start, _pin: PhantomPinned })
    }
}

impl<'a, T, F> Future for LazyFutureUnpin<'a, T, F>
where
    F: Future<Output = T> + Unpin,
{
    type Output = &'a T;
    fn poll(self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> task::Poll<&'a T> {
        // Safety: we don't move anything that needs to be pinned.
        unsafe { Pin::into_inner_unchecked(self) }.0.poll(cx)
    }
}

impl<T, F> Lazy<T, F> {
    /// Creates a new lazy value with the given initializing future.
    ///
    /// This is equivalent to [Self::new] but with no type bound.
    pub const fn from_future(future: F) -> Self {
        Self {
            value: UnsafeCell::new(LazyState { running: ManuallyDrop::new(future) }),
            inner: Inner::new(),
        }
    }

    /// Creates an already-initialized lazy value.
    pub const fn with_value(value: T) -> Self {
        Self {
            value: UnsafeCell::new(LazyState { ready: ManuallyDrop::new(value) }),
            inner: Inner::new_ready(),
        }
    }

    /// Gets the value without blocking or starting the initialization.
    pub fn try_get(&self) -> Option<&T> {
        let state = self.inner.state.load(Ordering::Acquire);

        if state & READY_BIT == 0 {
            None
        } else {
            // Safety: just checked ready
            unsafe { Some(&(*self.value.get()).ready) }
        }
    }

    /// Gets the value without blocking or starting the initialization.
    ///
    /// This requires mutable access to self, so rust's aliasing rules prevent any concurrent
    /// access and allow violating the usual rules for accessing this cell.
    pub fn try_get_mut(self: Pin<&mut Self>) -> Option<Pin<&mut T>> {
        // Safety: unpinning for access
        let this = unsafe { self.get_unchecked_mut() };
        let state = *this.inner.state.get_mut();
        if state & READY_BIT == 0 {
            None
        } else {
            // Safety: just checked ready, and pinned as a projection
            unsafe { Some(Pin::new_unchecked(&mut this.value.get_mut().ready)) }
        }
    }

    /// Gets the value without blocking or starting the initialization.
    ///
    /// This requires mutable access to self, so rust's aliasing rules prevent any concurrent
    /// access and allow violating the usual rules for accessing this cell.
    pub fn try_get_mut_unpin(&mut self) -> Option<&mut T> {
        let state = *self.inner.state.get_mut();
        if state & READY_BIT == 0 {
            None
        } else {
            // Safety: just checked ready
            unsafe { Some(&mut self.value.get_mut().ready) }
        }
    }

    /// Takes ownership of the value if it was set.
    ///
    /// Similar to the try_get functions, this returns None if the future has not yet completed,
    /// even if the value would be available without blocking.
    pub fn into_inner(self) -> Option<T> {
        self.into_parts().ok()
    }

    /// Takes ownership of the value or the initializing future.
    pub fn into_parts(mut self) -> Result<T, F> {
        let state = *self.inner.state.get_mut();

        // Safety: we can take ownership of the contents of self.value as long as we avoid dropping
        // it when self goes out of scope.  The value EMPTY_STATE (!0) is used as a sentinel to
        // indicate that the union is empty - it's impossible for state to be set to that value
        // normally by the same logic that prevents refcount overflow.
        //
        // Note: it is not safe to do this in a &mut self method because none of the get()
        // functions handle EMPTY_STATE; that's not relevant here as we took ownership of self.
        // A working "Lazy::take(&mut self)" function would also need to create a new initializing
        // future, and at that point it's best done by just using mem::replace with a new Lazy.
        unsafe {
            *self.inner.state.get_mut() = EMPTY_STATE;
            if state & READY_BIT == 0 {
                Err(ptr::read(&*self.value.get_mut().running))
            } else {
                Ok(ptr::read(&*self.value.get_mut().ready))
            }
        }
    }

    /// Takes ownership of the value from a pinned object.
    ///
    /// This is equivalent to `mem::replace(self, replacement).into_inner()` but does not require
    /// that `F` be `Unpin` like that expression would.
    pub fn replace_and_take(self: Pin<&mut Self>, replacement: Self) -> Option<T>
    where
        T: Unpin,
    {
        // Safety: this reads fields and then open-codes Pin::set
        let this = unsafe { self.get_unchecked_mut() };
        let state = *this.inner.state.get_mut();
        let value = if state & READY_BIT == 0 {
            None
        } else {
            *this.inner.state.get_mut() = EMPTY_STATE;
            Some(unsafe { ptr::read(&*this.value.get_mut().ready) })
        };
        *this = replacement;
        value
    }
}

impl<T, F> Drop for Lazy<T, F> {
    fn drop(&mut self) {
        let state = *self.inner.state.get_mut();
        // Safety: the state always reflects the variant of the union that we must drop
        unsafe {
            if state == EMPTY_STATE {
                // do nothing (see into_inner and the _empty variant)
            } else if state & READY_BIT == 0 {
                ManuallyDrop::drop(&mut self.value.get_mut().running);
            } else {
                ManuallyDrop::drop(&mut self.value.get_mut().ready);
            }
        }
    }
}

impl<T: fmt::Debug, F> fmt::Debug for Lazy<T, F> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let value = self.try_get();
        fmt.debug_struct("Lazy").field("value", &value).field("inner", &self.inner).finish()
    }
}

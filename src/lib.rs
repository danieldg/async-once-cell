//! A collection of lazy initialized values that are created by `Future`s.
//!
//! [OnceCell]'s API should be familiar to anyone who has used the
//! [`once_cell`](https://crates.io/crates/once_cell) crate or the proposed `std::lazy` module.  It
//! provides an async version of a cell that can only be initialized once, permitting tasks to wait
//! on the initialization if it is already running instead of racing multiple initialization tasks.
//!
//! Unlike threads, tasks can be cancelled at any point where they block.  [OnceCell] deals with
//! this by allowing another initializer to run if the task currently initializing the cell is
//! dropped.  This also allows for fallible initialization using [OnceCell::get_or_try_init], and
//! for the initializing `Future` to contain borrows or use references to thread-local data.
//!
//! [Lazy] takes the opposite approach: it wraps a single `Future` which is cooperatively run to
//! completion by any polling task.  This requires that the initialization function be independent
//! of the calling context, but will never restart an initializing function just because the
//! surrounding task was cancelled.
//!
//! # Overhead
//!
//! Both cells use two `usize`s to store state and do not retain any allocations after
//! initialization is complete.  [OnceCell] and [Lazy] only allocate if there is contention.

use std::{
    cell::UnsafeCell,
    convert::Infallible,
    future::Future,
    panic::{RefUnwindSafe, UnwindSafe},
    pin::Pin,
    ptr,
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering},
    sync::Mutex,
    task,
};

/// Types that do not rely on pinning during initialization.
///
/// This module is only built if the `unpin` crate feature is enabled.
///
/// This module contains [OnceFuture](unpin::OnceFuture) and its wrappers [Lazy](unpin::Lazy) and
/// [ConstLazy](unpin::ConstLazy), which provide lazy initialization without requiring the
/// resulting structure be pinned.
///
/// This is the API exposed by the 0.3 version of this crate for `Lazy`.
#[cfg(feature = "unpin")]
pub mod unpin;

/// A cell which can be written to only once.
///
/// This allows initialization using an async closure that borrows from its environment.
///
/// ```
/// # async fn run() {
/// use std::rc::Rc;
/// use std::sync::Arc;
/// use async_once_cell::OnceCell;
///
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
/// ```
#[derive(Debug)]
pub struct OnceCell<T> {
    value: UnsafeCell<Option<T>>,
    inner: Inner,
}

// Safety: our UnsafeCell should be treated like an RwLock<T>
unsafe impl<T: Sync + Send> Sync for OnceCell<T> {}
unsafe impl<T: Send> Send for OnceCell<T> {}
impl<T> Unpin for OnceCell<T> {}
impl<T: RefUnwindSafe + UnwindSafe> RefUnwindSafe for OnceCell<T> {}
impl<T: UnwindSafe> UnwindSafe for OnceCell<T> {}

/// Monomorphic portion of the state
#[derive(Debug)]
struct Inner {
    state: AtomicUsize,
    queue: AtomicPtr<Queue>,
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

/// This is somewhat like Arc<Queue>, but holds the refcount in Inner instead of Queue so it can be
/// freed once the cell's initialization is complete.
struct QueueRef<'a> {
    inner: &'a Inner,
    queue: *const Queue,
}
// Safety: the queue is a reference (only the lack of a valid lifetime requires it to be a pointer)
unsafe impl<'a> Sync for QueueRef<'a> {}
unsafe impl<'a> Send for QueueRef<'a> {}

#[derive(Debug)]
struct QuickInitGuard<'a>(&'a Inner);

/// A Future that waits for acquisition of a QueueHead
struct QueueWaiter<'a> {
    guard: Option<QueueRef<'a>>,
}

/// A guard for the actual initialization of the OnceCell
struct QueueHead<'a> {
    guard: QueueRef<'a>,
}

const NEW: usize = 0x0;
const QINIT_BIT: usize = 1 + (usize::MAX >> 2);
const READY_BIT: usize = 1 + (usize::MAX >> 1);

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
                return Err(QuickInitGuard(self));
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
                        Box::from_raw(new_queue);
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

        debug_assert_eq!(prev_state & READY_BIT, 0, "Invalid state: somoene else set READY_BIT");
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
                    Box::from_raw(queue);
                }
            }
        }
    }
}

impl<'a> Drop for QuickInitGuard<'a> {
    fn drop(&mut self) {
        let prev_state = self.0.state.load(Ordering::Relaxed);
        if prev_state == QINIT_BIT | READY_BIT || prev_state == QINIT_BIT {
            let target = prev_state & !QINIT_BIT;
            // Try to finish the fast path of initialization if possible.
            if self
                .0
                .state
                .compare_exchange(prev_state, target, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                // If init succeeded, the Release in set_ready already ordered the value.  If init
                // failed, we made no writes that need to be ordered and there are no waiters to
                // wake, so we can leave the state at NEW.

                if target == READY_BIT {
                    // It's possible (though unlikely) that someone created the queue but abandoned
                    // their QueueRef before we finished our poll, resulting in us not observing
                    // them.  No wakes are needed in this case because there are no waiting tasks,
                    // but we should still clean up the allocation.
                    let queue = self.0.queue.swap(ptr::null_mut(), Ordering::Relaxed);
                    if !queue.is_null() {
                        // Synchronize with both the fetch_sub that lowered the refcount and the
                        // queue initialization.
                        std::sync::atomic::fence(Ordering::Acquire);
                        // Safety: we observed no active QueueRefs, and queue is only used by
                        // guard-holders.  Due to the swap, we are the only one who is freeing this
                        // particular queue.
                        unsafe {
                            Box::from_raw(queue);
                        }
                    }
                }
                return;
            }
        }

        // Slow path: get a guard, create the QueueHead we should have been holding, then drop it
        // so that the tasks are woken as intended.  This is needed regardless of if we succeeded
        // or not - either waiters need to run init themselves, or they need to read the value we
        // set.
        //
        // The guard is guaranteed to have been created with no QueueHead available because
        // QINIT_BIT is still set.
        let waiter = self.0.initialize(false).expect("Got a QuickInitGuard in slow init");
        let guard = waiter.guard.expect("No guard available even without polling");
        if guard.queue.is_null() {
            // The queue was already freed by someone else before we got our QueueRef (this must
            // have happend between the load of prev_state and initialize, because otherwise we
            // would have taken the fast path).  This implies that all other tasks have noticed
            // READY_BIT and do not need waking, so there is nothing left for us to do except
            // release our reference.
            drop(guard);
        } else {
            // Safety: the guard holds a place on the waiter list and we just checked that the
            // queue is non-null.  It will remain valid until guard is dropped.
            let queue = unsafe { &*guard.queue };
            let mut lock = queue.wakers.lock().unwrap();

            // Ensure that nobody else can grab the QueueHead between when we release QINIT_BIT and
            // when our QueueHead is dropped.
            lock.get_or_insert_with(Vec::new);
            // Allow someone else to take the head position once we drop it.  Ordering is handled
            // by the Mutex.
            self.0.state.fetch_and(!QINIT_BIT, Ordering::Relaxed);
            drop(lock);

            // Safety: we just took the head position, and we were the QuickInitGuard
            drop(QueueHead { guard })
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let queue = *self.queue.get_mut();
        if !queue.is_null() {
            // Safety: nobody else could have a reference
            unsafe {
                Box::from_raw(queue);
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
        let mut lock = queue.wakers.lock().unwrap();

        // Another task might have called set_ready() and dropped its QueueHead between our
        // optimistic lock-free check and our lock acquisition.  Don't return a QueueHead unless we
        // know for sure that we are allowed to initialize.
        let state = guard.inner.state.load(Ordering::Acquire);
        if state & READY_BIT != 0 {
            return task::Poll::Ready(None);
        }

        match lock.as_mut() {
            None if state & QINIT_BIT == 0 => {
                // take the head position and start a waker queue
                *lock = Some(Vec::new());
                drop(lock);

                // Safety: we know that nobody else has a QuickInitGuard because we are holding a
                // QueueRef that prevents state from being 0 (which is required to create a
                // new QuickInitGuard), and we just checked that one wasn't created before we
                // created our QueueRef.
                task::Poll::Ready(Some(QueueHead { guard: self.guard.take().unwrap() }))
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
    }
}

impl<'a> Drop for QueueHead<'a> {
    fn drop(&mut self) {
        // Safety: if queue is not null, then it is valid as long as the guard is alive
        if let Some(queue) = unsafe { self.guard.queue.as_ref() } {
            // Take the waker queue so the next QueueWaiter can make a new one
            let wakers = queue
                .wakers
                .lock()
                .expect("Lock poisoned")
                .take()
                .expect("QueueHead dropped without a waker list");
            for waker in wakers {
                waker.wake();
            }
        }
    }
}

impl<T> OnceCell<T> {
    /// Creates a new empty cell.
    pub const fn new() -> Self {
        Self { value: UnsafeCell::new(None), inner: Inner::new() }
    }

    /// Creates a new cell with the given contents.
    pub const fn new_with(value: Option<T>) -> Self {
        let inner = match value {
            Some(_) => Inner::new_ready(),
            None => Inner::new(),
        };
        Self { value: UnsafeCell::new(value), inner }
    }

    /// Gets the contents of the cell, initializing it with `init` if the cell was empty.
    ///
    /// Many tasks may call `get_or_init` concurrently with different initializing futures, but
    /// it is guaranteed that only one future will be executed as long as the resuting future is
    /// polled to completion.
    ///
    /// If `init` panics, the panic is propagated to the caller, and the cell remains uninitialized.
    ///
    /// If the Future returned by this function is dropped prior to completion, the cell remains
    /// uninitialized (and another init futures may be selected for polling).
    ///
    /// It is an error to reentrantly initialize the cell from `init`.  The current implementation
    /// deadlocks, but will recover if the offending task is dropped.
    pub async fn get_or_init(&self, init: impl Future<Output = T>) -> &T {
        match self.get_or_try_init(async move { Ok::<T, Infallible>(init.await) }).await {
            Ok(t) => t,
            Err(e) => match e {},
        }
    }

    /// Gets the contents of the cell, initializing it with `init` if the cell was empty.   If the
    /// cell was empty and `init` failed, an error is returned.
    ///
    /// If `init` panics, the panic is propagated to the caller, and the cell remains
    /// uninitialized.
    ///
    /// If the Future returned by this function is dropped prior to completion, the cell remains
    /// uninitialized.
    ///
    /// It is an error to reentrantly initialize the cell from `init`.  The current implementation
    /// deadlocks, but will recover if the offending task is dropped.
    pub async fn get_or_try_init<E>(
        &self,
        init: impl Future<Output = Result<T, E>>,
    ) -> Result<&T, E> {
        let state = self.inner.state.load(Ordering::Acquire);

        if state & READY_BIT == 0 {
            self.init_slow(state == NEW, init).await?;
        }

        // Safety: initialized on all paths
        Ok(unsafe { (&*self.value.get()).as_ref().unwrap() })
    }

    #[cold]
    async fn init_slow<E>(
        &self,
        try_quick: bool,
        init: impl Future<Output = Result<T, E>>,
    ) -> Result<(), E> {
        match self.inner.initialize(try_quick) {
            Err(guard) => {
                // Try to proceed assuming no contention.
                let value = init.await?;
                // Safety: the guard acts like QueueHead even if there is contention.
                unsafe {
                    *self.value.get() = Some(value);
                }
                self.inner.set_ready();
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
                        *self.value.get() = Some(value);
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
            unsafe { (&*self.value.get()).as_ref() }
        }
    }

    /// Gets a mutable reference to the underlying value.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.value.get_mut().as_mut()
    }

    /// Takes the value out of this `OnceCell`, moving it back to an uninitialized state.
    pub fn take(&mut self) -> Option<T> {
        self.value.get_mut().take()
    }

    /// Consumes the OnceCell, returning the wrapped value. Returns None if the cell was empty.
    pub fn into_inner(self) -> Option<T> {
        self.value.into_inner()
    }
}

#[derive(Debug)]
enum LazyState<T, F> {
    Running(F),
    Ready(T),
}

/// A value which is computed on demand by running a future.
///
/// Unlike [OnceCell], if a task is cancelled, the initializing future's execution will be
/// continued by other (concurrent or future) callers of [Lazy::get].
///
/// ```
/// # async fn run() {
/// use std::sync::Arc;
/// use async_once_cell::Lazy;
///
/// struct Data {
///     id: u32,
/// }
///
/// let shared = Arc::pin(Lazy::new(async move {
///     Data { id: 4 }
/// }));
///
/// assert_eq!(shared.as_ref().get().await.id, 4);
/// # }
/// ```
#[derive(Debug)]
pub struct Lazy<T, F> {
    value: UnsafeCell<LazyState<T, F>>,
    inner: Inner,
}

// Safety: our UnsafeCell should be treated like an RwLock<(T, F)>
unsafe impl<T: Sync + Send, F: Sync + Send> Sync for Lazy<T, F> {}
unsafe impl<T: Send, F: Send> Send for Lazy<T, F> {}
impl<T: Unpin, F: Unpin> Unpin for Lazy<T, F> {}
impl<T: RefUnwindSafe + UnwindSafe, F: RefUnwindSafe + UnwindSafe> RefUnwindSafe for Lazy<T, F> {}
impl<T: UnwindSafe, F: UnwindSafe> UnwindSafe for Lazy<T, F> {}

impl<T, F> Lazy<T, F>
where
    F: Future<Output = T>,
{
    /// Creates a new lazy value with the given initializing future.
    pub fn new(future: F) -> Self {
        Self::from_future(future)
    }

    /// Forces the evaluation of this lazy value and returns a reference to the result.
    ///
    /// The [Pin::static_ref] function may be useful if this is a static value.
    pub async fn get(self: Pin<&Self>) -> Pin<&T> {
        let state = self.inner.state.load(Ordering::Acquire);

        if state & READY_BIT == 0 {
            self.init_slow(state == NEW).await;
        }

        // Safety: initialized on all paths, and pinned like self
        unsafe {
            match &*self.value.get() {
                LazyState::Ready(v) => Pin::new_unchecked(v),
                _ => unreachable!(),
            }
        }
    }

    #[cold]
    async fn init_slow(self: Pin<&Self>, try_quick: bool) {
        match self.inner.initialize(try_quick) {
            Err(guard) => {
                let init = unsafe {
                    match &mut *self.value.get() {
                        LazyState::Running(f) => Pin::new_unchecked(f),
                        _ => unreachable!(),
                    }
                };
                let value = init.await;
                // Safety: the guard acts like QueueHead even if there is contention.
                // This overwrites the pinned future, dropping it in place
                unsafe {
                    *self.value.get() = LazyState::Ready(value);
                }
                self.inner.set_ready();
                drop(guard);
            }
            Ok(guard) => {
                if let Some(init_lock) = guard.await {
                    let init = unsafe {
                        match &mut *self.value.get() {
                            LazyState::Running(f) => Pin::new_unchecked(f),
                            _ => unreachable!(),
                        }
                    };
                    // We hold the QueueHead, so we know that nobody else has successfully run an init
                    // poll and that nobody else can start until it is dropped.  On error, panic, or
                    // drop of this Future, the head will be passed to another waiter.
                    let value = init.await;

                    // Safety: We still hold the head, so nobody else can write to value
                    // This overwrites the pinned future, dropping it in place
                    unsafe {
                        *self.value.get() = LazyState::Ready(value);
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
    }
}

impl<T, F> Lazy<T, F> {
    /// Creates a new lazy value with the given initializing future.
    ///
    /// This is equivalent to [Self::new] but with no type bound.
    pub const fn from_future(future: F) -> Self {
        Self { value: UnsafeCell::new(LazyState::Running(future)), inner: Inner::new() }
    }

    /// Creates an already-initialized lazy value.
    pub const fn with_value(value: T) -> Self {
        Self { value: UnsafeCell::new(LazyState::Ready(value)), inner: Inner::new_ready() }
    }

    /// Gets the value without blocking or starting the initialization.
    pub fn try_get(&self) -> Option<&T> {
        let state = self.inner.state.load(Ordering::Acquire);

        if state & READY_BIT == 0 {
            None
        } else {
            match unsafe { &*self.value.get() } {
                LazyState::Ready(v) => Some(v),
                _ => unreachable!(),
            }
        }
    }

    /// Gets the value without blocking or starting the initialization.
    ///
    /// This requires mutable access to self, so rust's aliasing rules prevent any concurrent
    /// access and allow violating the usual rules for accessing this cell.
    pub fn try_get_mut(self: Pin<&mut Self>) -> Option<Pin<&mut T>> {
        unsafe {
            match self.get_unchecked_mut().value.get_mut() {
                LazyState::Ready(v) => Some(Pin::new_unchecked(v)),
                _ => None,
            }
        }
    }

    /// Gets the value without blocking or starting the initialization.
    ///
    /// This requires mutable access to self, so rust's aliasing rules prevent any concurrent
    /// access and allow violating the usual rules for accessing this cell.
    pub fn try_get_mut_unpin(&mut self) -> Option<&mut T> {
        match self.value.get_mut() {
            LazyState::Ready(v) => Some(v),
            _ => None,
        }
    }

    /// Gets the value if it was set.
    pub fn into_inner(self) -> Option<T> {
        match self.value.into_inner() {
            LazyState::Ready(v) => Some(v),
            _ => None,
        }
    }
}

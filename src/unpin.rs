use std::{
    cell::UnsafeCell,
    convert::Infallible,
    future::Future,
    mem,
    panic::{RefUnwindSafe, UnwindSafe},
    pin::Pin,
    ptr,
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering},
    sync::{Arc, Mutex},
    task,
};

use super::{NEW, READY_BIT};

/// A Future which is executed exactly once, producing an output accessible without locking.
///
/// This is primarily used as a building block for [Lazy] and [ConstLazy], but can also be used on
/// its own similar to [OnceCell](crate::OnceCell).
///
/// ```
/// # async fn run() {
/// use std::sync::Arc;
/// use async_once_cell::unpin::OnceFuture;
///
/// let shared = Arc::new(OnceFuture::new());
/// let value : &i32 = shared.get_or_init_with(|| async {
///     4
/// }).await;
/// assert_eq!(value, &4);
/// # }
/// ```
#[derive(Debug)]
pub struct OnceFuture<T, F = Pin<Box<dyn Future<Output = T> + Send>>, I = Infallible> {
    value: UnsafeCell<LazyState<T, I>>,
    inner: LazyInner<F>,
}

// Safety: acts like RwLock<T> + Mutex<(I,F)>.
unsafe impl<T: Sync + Send, F: Send, I: Send> Sync for OnceFuture<T, F, I> {}
unsafe impl<T: Send, F: Send, I: Send> Send for OnceFuture<T, F, I> {}

// We pin F inside the allocated LazyWaker; this object can be moved freely
impl<T, F, I> Unpin for OnceFuture<T, F, I> {}

// It is possible to get T and I with &mut self, and &T with &self
impl<T: RefUnwindSafe + UnwindSafe, F, I: RefUnwindSafe> RefUnwindSafe for OnceFuture<T, F, I> {}
impl<T: UnwindSafe, F, I: UnwindSafe> UnwindSafe for OnceFuture<T, F, I> {}

enum LazyState<T, I> {
    New(I),
    Running,
    Ready(T),
}

#[derive(Debug)]
struct LazyInner<F> {
    state: AtomicUsize,
    queue: AtomicPtr<LazyWaker<F>>,
}

/// Contents of the Arc held by LazyInner and by any Waker given to the future.  This value is
/// pinned in the Arc.
struct LazyWaker<F> {
    future: UnsafeCell<Option<F>>,
    wakers: Mutex<(WakerState, Vec<task::Waker>)>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum WakerState {
    Unlocked,
    /// A task is currently polling the future or will soon start polling it
    LockedWithoutWake,
    /// The future returned Pending and has not seen a wakeup
    Pending,
    /// A task is currently polling the future but a wake has already been sent
    LockedWoken,
}

// Safety: acts like Mutex<F>
unsafe impl<F: Send> Send for LazyWaker<F> {}
unsafe impl<F: Send> Sync for LazyWaker<F> {}

/// A lock guard given to exactly one poller of a LazyWaker at a time.
struct LazyHead<'a, F> {
    // Note: this structure is passed to mem::forget during normal use; do not add Drop fields.
    waker: &'a Arc<LazyWaker<F>>,
}

impl<F> LazyInner<F> {
    fn initialize(&self) -> Option<Arc<LazyWaker<F>>> {
        // Increment the queue's reference count.  This ensures that queue won't be freed until we exit.
        let prev_state = self.state.fetch_add(1, Ordering::Acquire);

        // Note: unlike Arc, refcount overflow is impossible.  The only way to increment the
        // refcount is by calling poll on the Future returned by get_or_try_init, which is !Unpin.
        // The poll call requires a Pinned pointer to this Future, and the contract of Pin requires
        // Drop to be called on any !Unpin value that was pinned before the memory is reused.
        // Because the Drop impl of QueueRef decrements the refcount, an overflow would require
        // more than (usize::MAX / 4) QueueRef objects in memory, which is impossible as these
        // objects take up more than 4 bytes.

        let mut queue = self.queue.load(Ordering::Acquire);
        if queue.is_null() && prev_state & READY_BIT == 0 {
            let waker: LazyWaker<F> = LazyWaker {
                future: UnsafeCell::new(None),
                wakers: Mutex::new((WakerState::Unlocked, Vec::new())),
            };

            // Race with other callers of initialize to create the queue
            let new_queue = Arc::into_raw(Arc::new(waker)) as *mut _;

            match self.queue.compare_exchange(
                ptr::null_mut(),
                new_queue,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_null) => {
                    // Normal case: it was actually set.  The Release part of AcqRel orders this
                    // with all Acquires on the queue.
                    queue = new_queue;
                }
                Err(actual) => {
                    // we lost the race, but we have the (non-null) value now.
                    queue = actual;
                    // Safety: we just allocated it, and nobody else has seen it
                    unsafe {
                        Arc::from_raw(new_queue as *const _);
                    }
                }
            }
        }
        let rv = if queue.is_null() {
            None
        } else {
            // Safety: the queue won't be freed due to the refcount raise at the start of the
            // function, and if queue is nonnull it has at least one strong ref.
            unsafe {
                Arc::increment_strong_count(queue as *const _);
                Some(Arc::from_raw(queue as *const _))
            }
        };

        let prev_state = self.state.fetch_sub(1, Ordering::AcqRel);
        if prev_state & READY_BIT == 0 {
            // Normal case: not ready, this is the queue for this cell.
            debug_assert!(rv.is_some());
            rv
        } else {
            // We prevented the our reference to the queue from being freed when it's elgible for
            // freeing.  If we were the last one holding that reference, free it.
            if prev_state == READY_BIT + 1 {
                let queue = self.queue.swap(ptr::null_mut(), Ordering::Acquire);
                if !queue.is_null() {
                    // Safety: no other callers of initialize were present and any future ones will
                    // also observe READY_BIT.  This is the only function that uses this reference,
                    // so if we got a nonnull queue we are the only user of this reference.
                    unsafe {
                        Arc::decrement_strong_count(queue as *const _);
                    }
                }
            }
            // We checked READY_BIT and it's ready
            None
        }
    }

    fn set_ready(&self) {
        // This Release pairs with the Acquire any time we check READY_BIT, and ensures that the
        // writes to the cell's value are visible to the cell's readers.
        let prev_state = self.state.fetch_or(READY_BIT, Ordering::Release);

        debug_assert_eq!(prev_state & READY_BIT, 0, "Invalid state: somoene else set READY_BIT");

        // If nobody was in initialize() (normal case), then we kill our reference to the LazyWaker
        // Arc here.  Otherwise, that function will handle the cleanup.
        if prev_state == NEW {
            let queue = self.queue.swap(ptr::null_mut(), Ordering::Acquire);
            if !queue.is_null() {
                unsafe {
                    Arc::decrement_strong_count(queue as *const _);
                }
            }
        }
    }
}

impl<F> Drop for LazyInner<F> {
    fn drop(&mut self) {
        let queue = *self.queue.get_mut();
        if !queue.is_null() {
            // Safety: the only user of this reference is initialize, and we know it is not running
            // because it uses a borrow of this object.
            unsafe {
                Arc::decrement_strong_count(queue);
            }
        }
    }
}

impl<F> LazyWaker<F> {
    /// Return a LazyHead if the caller was the first task to arrive and the cell is still empty.
    /// Otherwise, return None if the cell is already populated and Pending otherwise.
    fn poll_head<'a>(
        self: &'a Arc<Self>,
        cx: &mut task::Context<'_>,
        inner: &LazyInner<F>,
    ) -> task::Poll<Option<LazyHead<'a, F>>> {
        let mut lock = self.wakers.lock().unwrap();

        // Don't give out the head if the cell is ready
        let state = inner.state.load(Ordering::Acquire);
        if state & READY_BIT != 0 {
            return task::Poll::Ready(None);
        }

        let wakers = &mut lock.1;
        let my_waker = cx.waker();
        for waker in wakers.iter() {
            if waker.will_wake(my_waker) {
                return task::Poll::Pending;
            }
        }
        wakers.push(my_waker.clone());

        match lock.0 {
            WakerState::Unlocked => {
                // Safety: this state change means we are the only LazyHead present
                lock.0 = WakerState::LockedWithoutWake;
                task::Poll::Ready(Some(LazyHead { waker: self }))
            }
            _ => {
                // In all other cases, someone will wake us: the owner of LazyHead if locked or the
                // Waker if the task was pending.
                task::Poll::Pending
            }
        }
    }
}

impl<F> task::Wake for LazyWaker<F> {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref()
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let mut lock = self.wakers.lock().unwrap();
        match lock.0 {
            WakerState::LockedWithoutWake => {
                // Postposne propagating the wakes until the poll is complete
                lock.0 = WakerState::LockedWoken;
                return;
            }
            WakerState::LockedWoken => return,
            WakerState::Pending => {
                lock.0 = WakerState::Unlocked;
            }
            WakerState::Unlocked => {
                // Note: the waker list should be empty
            }
        }
        let wakers = mem::replace(&mut lock.1, Vec::new());
        // Avoid holding the lock while waking in case there is a recursive wake
        drop(lock);
        for waker in wakers {
            waker.wake();
        }
    }
}

impl<'a, F> LazyHead<'a, F> {
    fn poll_inner(self, init: impl FnOnce() -> F) -> task::Poll<(Self, F::Output)>
    where
        F: Future + Send + 'static,
    {
        let ptr = self.waker.future.get();
        // Safety: only one task can acquire a LazyHead object, so we are safe to modify the shared
        // state.  The value of ptr is inside an Arc that is never exposed outside this module (and
        // we never call get_mut on the Arc), so the contents follow the rules of Pin even if the
        // Arc was not created using Arc::pin.
        let fut = unsafe { Pin::new_unchecked((*ptr).get_or_insert_with(init)) };
        let shared_waker = task::Waker::from(Arc::clone(self.waker));
        let mut ctx = task::Context::from_waker(&shared_waker);
        match fut.poll(&mut ctx) {
            task::Poll::Pending => {
                // The inner future is pending, so LazyHead should not send out wakes until or
                // unless the shared waker has been used.
                let mut lock = self.waker.wakers.lock().unwrap();
                match lock.0 {
                    WakerState::LockedWithoutWake => {
                        lock.0 = WakerState::Pending;
                        drop(lock);
                    }
                    WakerState::LockedWoken => {
                        // There was a wake while we held the lock.  Send wakes to all tasks.
                        lock.0 = WakerState::Unlocked;
                        let wakers = mem::replace(&mut lock.1, Vec::new());
                        drop(lock);
                        for waker in wakers {
                            waker.wake();
                        }
                    }
                    WakerState::Pending | WakerState::Unlocked => {
                        unreachable!();
                    }
                }
                // we just did the drop implementation, don't do it again.
                mem::forget(self);
                task::Poll::Pending
            }
            task::Poll::Ready(value) => {
                // Drop the pinned Future now that it has completed.  Safety: we still hold the lock.
                unsafe {
                    *ptr = None;
                }
                task::Poll::Ready((self, value))
            }
        }
    }
}

impl<'a, F> Drop for LazyHead<'a, F> {
    fn drop(&mut self) {
        // Note: this is only called if the poll_inner was Ready or in case of panic.  In either
        // case, we should transition to an Unlocked state and wake all waiting tasks.  If the
        // future was ready, they will all be able to pick up the value; if it paniced, the next
        // task in line will retry the poll (which will just panic again if the future was
        // generated by an async block).
        let mut lock = self.waker.wakers.lock().unwrap();
        match lock.0 {
            WakerState::LockedWoken | WakerState::LockedWithoutWake => {
                lock.0 = WakerState::Unlocked;
            }
            WakerState::Unlocked | WakerState::Pending => {
                unreachable!();
            }
        }
        let wakers = mem::replace(&mut lock.1, Vec::new());
        drop(lock);
        for waker in wakers {
            waker.wake();
        }
    }
}

impl<T, F, I> OnceFuture<T, F, I> {
    /// Creates a new OnceFuture with an initializing value
    pub const fn with_init(init: I) -> Self {
        OnceFuture {
            value: UnsafeCell::new(LazyState::New(init)),
            inner: LazyInner {
                state: AtomicUsize::new(NEW),
                queue: AtomicPtr::new(ptr::null_mut()),
            },
        }
    }

    /// Creates a new OnceFuture without an initializing value
    ///
    /// The resulting Future must be produced by the closure passed to [Self::get_or_init_with].
    /// This function is identical to [Self::new] but is more likely to need type hints.
    pub const fn with_no_init() -> Self {
        OnceFuture {
            value: UnsafeCell::new(LazyState::Running),
            inner: LazyInner {
                state: AtomicUsize::new(NEW),
                queue: AtomicPtr::new(ptr::null_mut()),
            },
        }
    }

    /// Creates a new OnceFuture that is immediately ready
    pub const fn with_value(value: T) -> Self {
        OnceFuture {
            value: UnsafeCell::new(LazyState::Ready(value)),
            inner: LazyInner {
                state: AtomicUsize::new(READY_BIT),
                queue: AtomicPtr::new(ptr::null_mut()),
            },
        }
    }

    /// Gets the value without blocking or starting the initialization.
    pub fn get(&self) -> Option<&T> {
        let state = self.inner.state.load(Ordering::Acquire);
        if state & READY_BIT == 0 {
            None
        } else {
            // Safety: READY_BIT is set
            unsafe {
                match &*self.value.get() {
                    LazyState::Ready(v) => Some(v),
                    _ => unreachable!(),
                }
            }
        }
    }

    /// Get mutable access to the initializer or final value.
    ///
    /// This requires mutable access to self, so rust's aliasing rules prevent any concurrent
    /// access and allow violating the usual rules for accessing this cell.
    pub fn get_mut(&mut self) -> (Option<&mut I>, Option<&mut T>) {
        match self.value.get_mut() {
            LazyState::New(i) => (Some(i), None),
            LazyState::Running => (None, None),
            LazyState::Ready(v) => (None, Some(v)),
        }
    }

    /// Gets the initializer or final value
    pub fn into_inner(self) -> (Option<I>, Option<T>) {
        match self.value.into_inner() {
            LazyState::New(i) => (Some(i), None),
            LazyState::Running => (None, None),
            LazyState::Ready(v) => (None, Some(v)),
        }
    }
}

impl<T, F> OnceFuture<T, F> {
    /// Creates a new OnceFuture without an initializing value
    ///
    /// The resulting Future must be produced by the closure passed to get_or_init_with
    pub const fn new() -> Self {
        Self::with_no_init()
    }
}

impl<F> OnceFuture<F::Output, F>
where
    F: Future + Send + 'static,
{
    /// Creates a new OnceFuture directly from a Future.
    ///
    /// The `gen_future` or `into_future` closures will never be called.
    pub fn from_future(future: F) -> Self {
        let rv = Self::new();
        let waker = rv.inner.initialize().unwrap();
        // Safe because we currently have exclusive ownership
        unsafe {
            *waker.future.get() = Some(future);
        }
        rv
    }
}

impl<T, F, I> OnceFuture<T, F, I>
where
    F: Future<Output = T> + Send + 'static,
{
    /// Create and run the future until it produces a result, then return a reference to that
    /// result.
    ///
    /// This is a convenience wrapper around [OnceFuture::get_or_populate_with] for use when the
    /// initializer value is not used or not present.
    pub async fn get_or_init_with(&self, gen_future: impl FnOnce() -> F) -> &T {
        self.get_or_populate_with(move |_| gen_future()).await
    }

    /// Create and run the future until it produces a result, then return a reference to that
    /// result.
    ///
    /// Only one `into_future` closure will be called per `OnceFuture` instance, and only if the
    /// future was not already set by `from_future`.
    pub async fn get_or_populate_with(&self, into_future: impl FnOnce(Option<I>) -> F) -> &T {
        struct Get<'a, T, F, I, P>(&'a OnceFuture<T, F, I>, Option<P>);

        impl<'a, T, F, I, P> Unpin for Get<'a, T, F, I, P> {}
        impl<'a, T, F, I, P> Future for Get<'a, T, F, I, P>
        where
            F: Future<Output = T> + Send + 'static,
            P: FnOnce(Option<I>) -> F,
        {
            type Output = &'a T;
            fn poll(mut self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> task::Poll<&'a T> {
                self.0.poll_populate(cx, |i| (self.1.take().unwrap())(i))
            }
        }
        Get(self, Some(into_future)).await
    }

    /// Create and run the future until it produces a result, then return a reference to that
    /// result.
    ///
    /// Only one `into_future` closure will be called per `OnceFuture` instance, and only if the
    /// future was not already set by `from_future`.
    pub fn poll_populate(
        &self,
        cx: &mut task::Context<'_>,
        into_future: impl FnOnce(Option<I>) -> F,
    ) -> task::Poll<&T> {
        let state = self.inner.state.load(Ordering::Acquire);
        if state & READY_BIT == 0 {
            match self.init_slow(cx, into_future) {
                task::Poll::Pending => return task::Poll::Pending,
                task::Poll::Ready(()) => {}
            }
        }
        // Safety: just initialized
        unsafe {
            match &*self.value.get() {
                LazyState::Ready(v) => task::Poll::Ready(v),
                _ => unreachable!(),
            }
        }
    }

    /// Do the actual init work.  If this returns Ready, the initialization succeeded.
    #[cold]
    fn init_slow(
        &self,
        cx: &mut task::Context<'_>,
        into_future: impl FnOnce(Option<I>) -> F,
    ) -> task::Poll<()> {
        let waker = self.inner.initialize();
        let waker = match waker {
            Some(waker) => waker,
            None => return task::Poll::Ready(()),
        };

        match waker.poll_head(cx, &self.inner) {
            task::Poll::Ready(Some(init_lock)) => {
                // Safety: init_lock ensures we have exclusive access
                let value = mem::replace(unsafe { &mut *self.value.get() }, LazyState::Running);
                let init = match value {
                    LazyState::New(init) => Some(init),
                    LazyState::Running => None,
                    LazyState::Ready(_) => unreachable!(),
                };

                match init_lock.poll_inner(move || into_future(init)) {
                    task::Poll::Ready((lock, value)) => {
                        // Safety: we still hold the lock
                        unsafe {
                            *self.value.get() = LazyState::Ready(value);
                        }
                        self.inner.set_ready();
                        drop(lock);
                    }
                    task::Poll::Pending => return task::Poll::Pending,
                }
            }
            task::Poll::Ready(None) => return task::Poll::Ready(()),
            task::Poll::Pending => return task::Poll::Pending,
        }
        task::Poll::Ready(())
    }
}

/// A value which is initialized on the first access.
///
/// See [ConstLazy] if you need to initialize in a const context.
///
/// ```
/// # async fn run() {
/// use std::sync::Arc;
/// use async_once_cell::unpin::Lazy;
///
/// let shared = Arc::new(Lazy::new(async {
///     4
/// }));
///
/// let value : &i32 = shared.get().await;
/// assert_eq!(value, &4);
/// # }
/// ```
///
/// You can also call `await` on a reference:
///
/// ```
/// # async fn run() {
/// use async_once_cell::unpin::Lazy;
/// struct Foo {
///     value: Lazy<i32>,
/// }
///
/// let foo = Foo {
///     value : Lazy::new(Box::pin(async { 4 })),
/// };
///
/// assert_eq!((&foo.value).await, &4);
/// # }
/// ```
#[derive(Debug)]
pub struct Lazy<T, F = Pin<Box<dyn Future<Output = T> + Send>>> {
    once: OnceFuture<T, F>,
}

impl<T, F> Lazy<T, F>
where
    F: Future<Output = T> + Send + 'static,
{
    /// Creates a new lazy value with the given initializing future.
    pub fn new(future: F) -> Self {
        Lazy { once: OnceFuture::from_future(future) }
    }

    /// Forces the evaluation of this lazy value and returns a reference to the result.
    ///
    /// This is equivalent to the `Future` impl on `&Lazy`, but is explicit and may be simpler to
    /// call.  This will panic if the initializing closure panics or has panicked.
    pub async fn get(&self) -> &T {
        self.await
    }
}

impl<T, F> Lazy<T, F> {
    /// Creates an already-initialized lazy value.
    pub const fn with_value(value: T) -> Self {
        Self { once: OnceFuture::with_value(value) }
    }

    /// Gets the value without blocking or starting the initialization.
    pub fn try_get(&self) -> Option<&T> {
        self.once.get()
    }

    /// Gets the value without blocking or starting the initialization.
    ///
    /// This requires mutable access to self, so rust's aliasing rules prevent any concurrent
    /// access and allow violating the usual rules for accessing this cell.
    pub fn try_get_mut(&mut self) -> Option<&mut T> {
        self.once.get_mut().1
    }

    /// Gets the value if it was set.
    pub fn into_value(self) -> Option<T> {
        // It would be confusing to only sometimes return the future, and it's rarely useful.
        self.once.into_inner().1
    }
}

impl<'a, T, F> Future for &'a Lazy<T, F>
where
    F: Future<Output = T> + Send + 'static,
{
    type Output = &'a T;
    fn poll(self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> task::Poll<&'a T> {
        // The init closure is unreachable because we always start with the Future set.
        self.once.poll_populate(cx, |_| unreachable!())
    }
}

/// A value which is initialized on the first access.
///
/// Note: This structure may be larger in size than [Lazy], but it does not allocate on the heap
/// until it is first polled, so is suitable for initializing statics.
#[derive(Debug)]
pub struct ConstLazy<T, F> {
    once: OnceFuture<T, F, F>,
}

impl<T, F> ConstLazy<T, F> {
    /// Creates a new lazy value with the given initializing future.
    pub const fn new(future: F) -> Self {
        ConstLazy { once: OnceFuture::with_init(future) }
    }

    /// Creates an already-initialized lazy value.
    pub const fn with_value(value: T) -> Self {
        Self { once: OnceFuture::with_value(value) }
    }

    /// Gets the value without blocking or starting the initialization.
    pub fn try_get(&self) -> Option<&T> {
        self.once.get()
    }

    /// Gets the value without blocking or starting the initialization.
    ///
    /// This requires mutable access to self, so rust's aliasing rules prevent any concurrent
    /// access and allow violating the usual rules for accessing this cell.
    pub fn try_get_mut(&mut self) -> Option<&mut T> {
        self.once.get_mut().1
    }

    /// Gets the value if it was set.
    pub fn into_value(self) -> Option<T> {
        // It would be confusing to only sometimes return the future, and it's rarely useful.
        self.once.into_inner().1
    }
}

impl<T, F> ConstLazy<T, F>
where
    F: Future<Output = T> + Send + 'static,
{
    /// Forces the evaluation of this lazy value and returns a reference to the result.
    ///
    /// This is equivalent to the `Future` impl on `&ConstLazy`, but is explicit and may be simpler
    /// to call.  This will panic if the initializing closure panics or has panicked.
    pub async fn get(&self) -> &T {
        self.await
    }
}

impl<'a, T, F> Future for &'a ConstLazy<T, F>
where
    F: Future<Output = T> + Send + 'static,
{
    type Output = &'a T;
    fn poll(self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> task::Poll<&'a T> {
        // The init closure always has an initialization value
        self.once.poll_populate(cx, |i| i.unwrap_or_else(|| unreachable!()))
    }
}

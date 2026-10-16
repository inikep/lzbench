#![doc = include_str!("../README.md")]
#![no_std]
#![deny(missing_docs, unsafe_op_in_unsafe_fn)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
pub mod io;

use core::fmt::{self, Debug, Formatter};
use core::mem::MaybeUninit;
use core::ptr::{self, NonNull};
use core::slice;

/// A borrowed buffer of initially uninitialized elements, which is incrementally filled.
///
/// This type makes it safer to work with `MaybeUninit` buffers, such as to read into a buffer
/// without having to initialize it first. It tracks the region of elements that have been filled
/// and whether the unfilled region was initialized.
///
/// In summary, the contents of the buffer can be visualized as:
///
/// ```text
/// [                capacity                ]
/// [ filled | unfilled (may be initialized) ]
/// ```
///
/// A `BorrowedBuf` is created around some existing elements (or capacity for elements) via a unique
/// reference (`&mut`). The `BorrowedBuf` can be configured (e.g., using [`clear`](Self::clear) or
/// [`set_init`](Self::set_init)), but cannot be directly written. To write into the buffer, use
/// [`unfilled`](Self::unfilled) to create a [`BorrowedCursor`]. The cursor has write-only access to
/// the unfilled portion of the buffer (you can think of it as a write-only iterator).
///
/// The lifetime `'data` is a bound on the lifetime of the underlying elements.
///
/// The type is most commonly used to manage bytes, but can manage any type of elements.
///
/// # Examples
///
/// ```
/// use borrowed_buf::BorrowedBuf;
/// use core::mem::MaybeUninit;
///
/// let mut storage = [MaybeUninit::<u8>::uninit(); 8];
/// let mut buf = BorrowedBuf::from(&mut storage[..]);
///
/// buf.unfilled().append(b"abc");
/// assert_eq!(buf.filled(), b"abc");
///
/// buf.clear();
/// assert_eq!(buf.len(), 0);
/// ```
pub struct BorrowedBuf<'data, T> {
    /// The buffer's underlying elements.
    buf: &'data mut [MaybeUninit<T>],
    /// The number of elements of `self.buf` that are known to be filled.
    filled: usize,
    /// Whether the entire unfilled part of `self.buf` has explicitly been initialized.
    init: bool,
}

impl<T> Debug for BorrowedBuf<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        BorrowedBufDebug {
            init: self.init,
            filled: self.filled,
            capacity: self.capacity(),
        }
        .fmt(f)
    }
}

struct BorrowedBufDebug {
    init: bool,
    filled: usize,
    capacity: usize,
}

impl Debug for BorrowedBufDebug {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BorrowedBuf")
            .field("init", &self.init)
            .field("filled", &self.filled)
            .field("capacity", &self.capacity)
            .finish()
    }
}

/// Creates a new `BorrowedBuf` from a fully initialized slice.
impl<'data, T: Copy> From<&'data mut [T]> for BorrowedBuf<'data, T> {
    #[inline]
    fn from(slice: &'data mut [T]) -> BorrowedBuf<'data, T> {
        BorrowedBuf {
            // SAFETY: `T` and `MaybeUninit<T>` have the same layout, and no initialized element is
            // ever uninitialized as per `BorrowedBuf`'s invariant.
            buf: unsafe { &mut *(slice as *mut [T] as *mut [MaybeUninit<T>]) },
            filled: 0,
            init: true,
        }
    }
}

/// Creates a new `BorrowedBuf` from an uninitialized buffer.
impl<'data, T: Copy> From<&'data mut [MaybeUninit<T>]> for BorrowedBuf<'data, T> {
    #[inline]
    fn from(buf: &'data mut [MaybeUninit<T>]) -> BorrowedBuf<'data, T> {
        BorrowedBuf {
            buf,
            filled: 0,
            init: false,
        }
    }
}

/// Creates a new `BorrowedBuf` from a cursor.
///
/// Use [`BorrowedCursor::with_unfilled_buf`] instead for a safer alternative.
impl<'data, T: Copy> From<BorrowedCursor<'data, T>> for BorrowedBuf<'data, T> {
    #[inline]
    fn from(buf: BorrowedCursor<'data, T>) -> BorrowedBuf<'data, T> {
        let filled = buf.filled();
        let init = buf.is_buf_init();
        let len = buf.buf_len();
        BorrowedBuf {
            // SAFETY: no initialized element is ever uninitialized as per `BorrowedBuf`'s
            // invariant, and the cursor holds the unique access to those elements for `'data`.
            buf: unsafe { slice::from_raw_parts_mut(buf.buf.as_ptr().add(filled), len - filled) },
            filled: 0,
            init,
        }
    }
}

impl<'data, T> BorrowedBuf<'data, T> {
    /// Returns the total capacity of the buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Returns the length of the filled part of the buffer.
    #[inline]
    #[allow(clippy::len_without_is_empty)] // Mirrors `core::io::BorrowedBuf`.
    pub fn len(&self) -> usize {
        self.filled
    }

    /// Returns `true` if the buffer is initialized.
    #[inline]
    pub fn is_init(&self) -> bool {
        self.init
    }
}

impl<'data, T: Copy> BorrowedBuf<'data, T> {
    /// Returns a shared reference to the filled portion of the buffer.
    #[inline]
    pub fn filled(&self) -> &[T] {
        // SAFETY: We only slice the filled part of the buffer, which is always valid.
        unsafe { assume_init_ref(self.buf.get_unchecked(..self.filled)) }
    }

    /// Returns a mutable reference to the filled portion of the buffer.
    #[inline]
    pub fn filled_mut(&mut self) -> &mut [T] {
        // SAFETY: We only slice the filled part of the buffer, which is always valid.
        unsafe { assume_init_mut(self.buf.get_unchecked_mut(..self.filled)) }
    }

    /// Returns a shared reference to the filled portion of the buffer with its original lifetime.
    #[inline]
    pub fn into_filled(self) -> &'data [T] {
        // SAFETY: We only slice the filled part of the buffer, which is always valid.
        unsafe { assume_init_ref(self.buf.get_unchecked(..self.filled)) }
    }

    /// Returns a mutable reference to the filled portion of the buffer with its original lifetime.
    ///
    /// ```
    /// use borrowed_buf::BorrowedBuf;
    /// use core::mem::MaybeUninit;
    ///
    /// fn fill(storage: &mut [MaybeUninit<u8>]) -> &mut [u8] {
    ///     let mut buf = BorrowedBuf::from(storage);
    ///     buf.unfilled().append(b"done");
    ///     buf.into_filled_mut()
    /// }
    ///
    /// let mut storage = [MaybeUninit::uninit(); 8];
    /// assert_eq!(fill(&mut storage), b"done");
    /// ```
    #[inline]
    pub fn into_filled_mut(self) -> &'data mut [T] {
        // SAFETY: We only slice the filled part of the buffer, which is always valid.
        unsafe { assume_init_mut(self.buf.get_unchecked_mut(..self.filled)) }
    }

    /// Returns a cursor over the unfilled part of the buffer.
    #[inline]
    pub fn unfilled<'this>(&'this mut self) -> BorrowedCursor<'this, T> {
        let borrowed_buf = NonNull::from(&mut *self);
        BorrowedCursor {
            buf: NonNull::from(&mut *self.buf).cast(),
            borrowed_buf,
        }
    }

    /// Clears the buffer, resetting the filled region to empty.
    ///
    /// The contents of the buffer are not modified.
    #[inline]
    pub fn clear(&mut self) -> &mut Self {
        self.filled = 0;
        self
    }

    /// Shortens the filled region to `len` elements.
    ///
    /// If `len` is greater than or equal to [`len`](Self::len), this has no effect. The contents
    /// of the buffer are not modified, and the elements cut off stay initialized, so
    /// [`is_init`](Self::is_init) is unchanged.
    ///
    /// This is not part of `core::io::BorrowedBuf`.
    ///
    /// ```
    /// use borrowed_buf::BorrowedBuf;
    /// use core::mem::MaybeUninit;
    ///
    /// let mut storage = [MaybeUninit::uninit(); 8];
    /// let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    /// buf.unfilled().append(b"abcdef");
    /// buf.truncate(2);
    /// assert_eq!(buf.filled(), b"ab");
    /// buf.unfilled().append(b"!");
    /// assert_eq!(buf.filled(), b"ab!");
    /// ```
    #[inline]
    pub fn truncate(&mut self, len: usize) -> &mut Self {
        if len < self.filled {
            self.filled = len;
        }
        self
    }

    /// Asserts that the unfilled part of the buffer is initialized.
    ///
    /// # Safety
    ///
    /// All the elements of the buffer must be initialized.
    ///
    /// ```
    /// use borrowed_buf::BorrowedBuf;
    /// use core::mem::MaybeUninit;
    ///
    /// let mut storage = [MaybeUninit::new(0u8); 4];
    /// let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    /// assert!(!buf.is_init());
    /// // SAFETY: every element of `storage` was initialized above.
    /// unsafe { buf.set_init() };
    /// buf.unfilled().advance_checked(4);
    /// assert_eq!(buf.filled(), &[0; 4]);
    /// ```
    #[inline]
    pub unsafe fn set_init(&mut self) -> &mut Self {
        self.init = true;
        self
    }
}

/// A writeable view of the unfilled portion of a [`BorrowedBuf`].
///
/// The unfilled portion may be uninitialized; see [`BorrowedBuf`] for details.
///
/// Data can be written directly to the cursor by using [`append`](BorrowedCursor::append) or
/// indirectly by getting a slice of part or all of the cursor and writing into the slice. In the
/// indirect case, the caller must call [`advance`](BorrowedCursor::advance) after writing to inform
/// the cursor how many elements have been written.
///
/// Once elements are written to the cursor, they become part of the filled portion of the
/// underlying `BorrowedBuf` and can no longer be accessed or re-written by the cursor. In other
/// words, the cursor tracks the unfilled part of the underlying `BorrowedBuf`.
///
/// The lifetime `'a` is a bound on the lifetime of the underlying buffer (which means it is a bound
/// on the elements in that buffer by transitivity).
///
/// # Examples
///
/// ```
/// use borrowed_buf::{BorrowedBuf, BorrowedCursor};
/// use core::mem::MaybeUninit;
///
/// // A producer that only sees the free space.
/// fn produce(mut cursor: BorrowedCursor<'_, u32>) {
///     let n = cursor.capacity().min(3);
///     // SAFETY: only initialized values are written.
///     let out = unsafe { cursor.as_mut() };
///     for (i, slot) in out[..n].iter_mut().enumerate() {
///         slot.write(i as u32 * 10);
///     }
///     // SAFETY: the first `n` elements of the cursor were just written.
///     unsafe { cursor.advance(n) };
/// }
///
/// let mut storage = [MaybeUninit::uninit(); 4];
/// let mut buf = BorrowedBuf::from(&mut storage[..]);
/// buf.unfilled().append(&[7]);
/// produce(buf.unfilled());
/// assert_eq!(buf.filled(), &[7, 0, 10, 20]);
/// ```
pub struct BorrowedCursor<'a, T> {
    /// The start of the elements of the buffer this cursor was created from.
    /// Safety invariant: this points to the start of the *whole* buffer of `*borrowed_buf` and is
    /// valid for reads and writes of `(*borrowed_buf).buf.len()` elements, so that
    /// `(*borrowed_buf).filled` indexes into it.
    buf: NonNull<MaybeUninit<T>>,
    /// The buffer this cursor was created from.
    /// Safety invariants:
    /// 1. `(*borrowed_buf).buf` is *never* accessed by the owner of the pointee while the `buf`
    ///    field above is alive, because there is a `&mut` of the pointee while the cursor is alive.
    /// 2. We promise to only access the `filled` and `init` fields and the metadata of the `buf`
    ///    field through the `borrowed_buf` pointer, never triggering any retag of `buf`'s pointer,
    ///    as the `buf` field above holds a reborrow of it and reaching the parent again would be a
    ///    foreign access for that reborrow. This includes not making a reference to the whole
    ///    pointee out of `borrowed_buf`, but only accessing those fields directly through pointer
    ///    manipulation.
    borrowed_buf: NonNull<BorrowedBuf<'a, T>>,
}

// SAFETY: A `BorrowedCursor<'a, T>` is a unique borrow of a `BorrowedBuf<'a, T>`, which is a
// `&'a mut [MaybeUninit<T>]` and two `Copy` fields. The `buf` raw pointer is used like
// `&mut [MaybeUninit<T>]` so `T: Send` -> `Send` and `T: Sync` -> `Sync`, and the
// `borrowed_buf` only touches two `Copy` fields, without depending on `T`.
unsafe impl<T: Send> Send for BorrowedCursor<'_, T> {}
// SAFETY: See the `Send` impl above.
unsafe impl<T: Sync> Sync for BorrowedCursor<'_, T> {}

impl<T> Debug for BorrowedCursor<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let buf = BorrowedBufDebug {
            init: self.is_buf_init(),
            filled: self.filled(),
            capacity: self.buf_len(),
        };

        f.debug_struct("BorrowedCursor").field("buf", &buf).finish()
    }
}

// Helpers to access underlying buffer state.
impl<'a, T> BorrowedCursor<'a, T> {
    #[inline]
    fn buf_mut(&mut self) -> &mut [MaybeUninit<T>] {
        let len = self.buf_len();
        // SAFETY: `buf` points to `len` elements that this cursor borrows exclusively.
        unsafe { slice::from_raw_parts_mut(self.buf.as_ptr(), len) }
    }

    #[inline]
    fn buf_len(&self) -> usize {
        // SAFETY: We read just the metadata of `buf` and avoid retagging the reference.
        unsafe {
            let borrowed_buf = self.borrowed_buf.as_ptr();
            let buf_ptr: *const &'a mut [MaybeUninit<T>] = &raw const (*borrowed_buf).buf;
            // Same layout:
            // https://doc.rust-lang.org/reference/type-layout.html#r-layout.pointer.intro
            let buf_ptr: *const *const [MaybeUninit<T>] = buf_ptr.cast();
            let buf: *const [MaybeUninit<T>] = *buf_ptr;
            buf.len()
        }
    }

    #[inline]
    fn unfilled_slice(&mut self) -> &mut [MaybeUninit<T>] {
        let filled = self.filled();
        // SAFETY: always in bounds.
        unsafe { self.buf_mut().get_unchecked_mut(filled..) }
    }

    #[inline]
    fn filled(&self) -> usize {
        // SAFETY: We access just `filled` and avoid foreign read on `buf`.
        unsafe { (*self.borrowed_buf.as_ptr()).filled }
    }

    #[inline]
    fn is_buf_init(&self) -> bool {
        // SAFETY: We access just `init` and avoid foreign read on `buf`.
        unsafe { (*self.borrowed_buf.as_ptr()).init }
    }

    /// # Safety
    ///
    /// In case of `true` all the elements of the cursor must be initialized.
    #[inline]
    unsafe fn set_buf_init(&mut self, init: bool) {
        // SAFETY: We access just `init` and avoid foreign read on `buf`.
        unsafe {
            (*self.borrowed_buf.as_ptr()).init = init;
        }
    }

    /// # Safety
    ///
    /// The next `n` elements of the cursor must be initialized.
    #[inline]
    unsafe fn add_filled(&mut self, n: usize) {
        // SAFETY: We access just `filled` and avoid foreign read on `buf`.
        unsafe {
            (*self.borrowed_buf.as_ptr()).filled += n;
        }
    }
}

impl<'a, T: Copy> BorrowedCursor<'a, T> {
    /// Reborrows this cursor by cloning it with a smaller lifetime.
    ///
    /// Since a cursor maintains unique access to its underlying buffer, the borrowed cursor is
    /// not accessible while the new cursor exists.
    #[inline]
    pub fn reborrow<'this>(&'this mut self) -> BorrowedCursor<'this, T> {
        BorrowedCursor {
            buf: self.buf,
            borrowed_buf: self.borrowed_buf,
        }
    }

    /// Returns the available space in the cursor.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf_len() - self.filled()
    }

    /// Returns the number of elements written to the `BorrowedBuf` this cursor was created from.
    ///
    /// In particular, the count returned is shared by all reborrows of the cursor.
    #[inline]
    pub fn written(&self) -> usize {
        self.filled()
    }

    /// Returns `true` if the buffer is initialized.
    #[inline]
    pub fn is_init(&self) -> bool {
        self.is_buf_init()
    }

    /// Set the buffer as fully initialized.
    ///
    /// # Safety
    ///
    /// All the elements of the cursor must be initialized.
    ///
    /// ```
    /// use borrowed_buf::BorrowedBuf;
    /// use core::mem::MaybeUninit;
    ///
    /// let mut storage = [MaybeUninit::<u8>::uninit(); 4];
    /// let mut buf = BorrowedBuf::from(&mut storage[..]);
    /// let mut cursor = buf.unfilled();
    /// // SAFETY: only initialized values are written.
    /// unsafe { cursor.as_mut() }.fill(MaybeUninit::new(b'x'));
    /// // SAFETY: every element of the cursor was just written.
    /// unsafe { cursor.set_init() };
    /// cursor.advance_checked(4);
    /// assert_eq!(buf.filled(), b"xxxx");
    /// ```
    #[inline]
    pub unsafe fn set_init(&mut self) {
        // SAFETY: the caller guarantees that all the elements of the cursor are initialized.
        unsafe { self.set_buf_init(true) }
    }

    /// Returns a mutable reference to the whole cursor.
    ///
    /// # Safety
    ///
    /// The caller must not uninitialize any elements of the cursor if it is initialized.
    #[inline]
    pub unsafe fn as_mut(&mut self) -> &mut [MaybeUninit<T>] {
        self.unfilled_slice()
    }

    /// Advances the cursor by asserting that `n` elements have been filled.
    ///
    /// After advancing, the `n` elements are no longer accessible via the cursor and can only be
    /// accessed via the underlying buffer. I.e., the buffer's filled portion grows by `n` elements
    /// and its unfilled portion (and the capacity of this cursor) shrinks by `n` elements.
    ///
    /// If less than `n` elements initialized (by the cursor's point of view), `set_init` should be
    /// called first.
    ///
    /// # Panics
    ///
    /// Panics if there are less than `n` elements initialized.
    #[inline]
    #[track_caller]
    pub fn advance_checked(&mut self, n: usize) -> &mut Self {
        // The subtraction cannot underflow by invariant of this type.
        let init_unfilled = if self.is_buf_init() {
            self.buf_len() - self.filled()
        } else {
            0
        };
        assert!(
            n <= init_unfilled,
            "advanced past the initialized part of the buffer"
        );

        // SAFETY: the next `n` elements are initialized, as asserted above.
        unsafe { self.advance(n) };
        self
    }

    /// Advances the cursor by asserting that `n` elements have been filled.
    ///
    /// After advancing, the `n` elements are no longer accessible via the cursor and can only be
    /// accessed via the underlying buffer. I.e., the buffer's filled portion grows by `n` elements
    /// and its unfilled portion (and the capacity of this cursor) shrinks by `n` elements.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `n` elements of the cursor have been initialized.
    #[inline]
    pub unsafe fn advance(&mut self, n: usize) -> &mut Self {
        // SAFETY: the caller guarantees that the first `n` elements of the cursor are initialized.
        unsafe { self.add_filled(n) };
        self
    }

    /// Append elements to the cursor, advancing position within its buffer.
    ///
    /// # Panics
    ///
    /// Panics if `self.capacity()` is less than `buf.len()`.
    ///
    /// ```
    /// use borrowed_buf::BorrowedBuf;
    /// use core::mem::MaybeUninit;
    ///
    /// let mut storage = [MaybeUninit::uninit(); 4];
    /// let mut buf = BorrowedBuf::from(&mut storage[..]);
    /// buf.unfilled().append(&[1u64, 2]);
    /// buf.unfilled().append(&[3]);
    /// assert_eq!(buf.filled(), &[1, 2, 3]);
    /// ```
    #[inline]
    #[track_caller]
    pub fn append(&mut self, buf: &[T]) {
        assert!(
            self.capacity() >= buf.len(),
            "appended past the end of the buffer"
        );

        // SAFETY: we do not de-initialize any of the elements of the slice, the destination has
        // room for `buf.len()` elements as asserted above, and `buf` can't overlap it because the
        // cursor borrows the destination exclusively.
        unsafe {
            let dst = self.as_mut().as_mut_ptr().cast::<T>();
            ptr::copy_nonoverlapping(buf.as_ptr(), dst, buf.len());
        }

        // SAFETY: these elements have just been initialized.
        unsafe { self.advance(buf.len()) };
    }

    /// Appends `n` copies of `value` to the cursor, advancing position within its buffer.
    ///
    /// Like [`append`](Self::append), this writes each element once and leaves the rest of the
    /// cursor untouched. It does not need a source slice of `n` elements.
    ///
    /// This is not part of `core::io::BorrowedCursor`.
    ///
    /// # Panics
    ///
    /// Panics if `self.capacity()` is less than `n`.
    ///
    /// ```
    /// use borrowed_buf::BorrowedBuf;
    /// use core::mem::MaybeUninit;
    ///
    /// let mut storage = [MaybeUninit::uninit(); 8];
    /// let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    /// let mut cursor = buf.unfilled();
    /// cursor.append(b"ab");
    /// cursor.append_fill(3, b'-');
    /// assert_eq!(buf.filled(), b"ab---");
    /// assert!(!buf.is_init());
    /// ```
    #[inline]
    #[track_caller]
    pub fn append_fill(&mut self, n: usize, value: T) {
        assert!(self.capacity() >= n, "appended past the end of the buffer");

        // Writing initialized values never de-initializes an element.
        self.unfilled_slice()[..n].fill(MaybeUninit::new(value));

        // SAFETY: these elements have just been initialized.
        unsafe { self.advance(n) };
    }

    /// Runs the given closure with a `BorrowedBuf` containing the unfilled part
    /// of the cursor.
    ///
    /// This enables inspecting what was written to the cursor.
    ///
    /// # Panics
    ///
    /// Panics if the `BorrowedBuf` given to the closure is replaced by another
    /// one.
    ///
    /// ```
    /// use borrowed_buf::BorrowedBuf;
    /// use core::mem::MaybeUninit;
    ///
    /// let mut storage = [MaybeUninit::uninit(); 8];
    /// let mut buf = BorrowedBuf::from(&mut storage[..]);
    /// let mut cursor = buf.unfilled();
    /// cursor.append(b"ab");
    /// let fresh = cursor.with_unfilled_buf(|sub| {
    ///     sub.unfilled().append(b"cd");
    ///     sub.filled().to_vec()
    /// });
    /// assert_eq!(fresh, b"cd");
    /// assert_eq!(buf.filled(), b"abcd");
    /// ```
    pub fn with_unfilled_buf<R>(&mut self, f: impl FnOnce(&mut BorrowedBuf<'_, T>) -> R) -> R {
        let mut buf = BorrowedBuf::from(self.reborrow());
        let prev_ptr = buf.buf as *const _;
        let res = f(&mut buf);

        // Check that the caller didn't replace the `BorrowedBuf`.
        // This is necessary for the safety of the code below: if the check wasn't
        // there, one could mark some elements as initialized even though they aren't.
        assert!(
            ptr::eq(prev_ptr, buf.buf),
            "the `BorrowedBuf` given to the closure was replaced"
        );

        let filled = buf.filled;
        let init = buf.init;

        // Update `init` and `filled` fields with what was written to the buffer.
        // `self.buf.filled` was the starting length of the `BorrowedBuf`.
        //
        // SAFETY: These elements were initialized/filled in the `BorrowedBuf`, and therefore they
        // are initialized/filled in the cursor too, because the buffer wasn't replaced.
        unsafe {
            self.set_buf_init(init);
            self.advance(filled);
        }

        res
    }
}

impl<'a, T: Default + Copy> BorrowedCursor<'a, T> {
    /// Initializes all elements in the cursor with their default value and
    /// returns them.
    ///
    /// ```
    /// use borrowed_buf::BorrowedBuf;
    /// use core::mem::MaybeUninit;
    ///
    /// let mut storage = [MaybeUninit::<f32>::uninit(); 4];
    /// let mut buf = BorrowedBuf::from(&mut storage[..]);
    /// let mut cursor = buf.unfilled();
    ///
    /// let out = cursor.ensure_init();
    /// assert_eq!(out, &[0.0; 4]);
    /// out[0] = 1.5;
    /// cursor.advance_checked(1);
    /// assert_eq!(buf.filled(), &[1.5]);
    /// ```
    #[inline]
    pub fn ensure_init(&mut self) -> &mut [T] {
        if !self.is_buf_init() {
            self.unfilled_slice().fill(MaybeUninit::new(T::default()));
            // SAFETY: buf is now initialized.
            unsafe { self.set_buf_init(true) };
        }

        // SAFETY: these elements have just been initialized if they weren't before.
        unsafe { assume_init_mut(self.unfilled_slice()) }
    }
}

/// Stable equivalent of `<[MaybeUninit<T>]>::assume_init_ref`.
///
/// # Safety
///
/// Every element of `s` must be initialized.
#[inline(always)]
const unsafe fn assume_init_ref<T>(s: &[MaybeUninit<T>]) -> &[T] {
    // SAFETY: the caller guarantees `s` is initialized; the layouts are identical.
    unsafe { &*(s as *const [MaybeUninit<T>] as *const [T]) }
}

/// Stable equivalent of `<[MaybeUninit<T>]>::assume_init_mut`.
///
/// # Safety
///
/// Every element of `s` must be initialized.
#[inline(always)]
const unsafe fn assume_init_mut<T>(s: &mut [MaybeUninit<T>]) -> &mut [T] {
    // SAFETY: the caller guarantees `s` is initialized; the layouts are identical.
    unsafe { &mut *(s as *mut [MaybeUninit<T>] as *mut [T]) }
}

/// Compile-fail checks for the soundness-relevant type properties.
///
/// A cursor must be invariant over `T`, or a short-lived value could be written into a buffer
/// of longer-lived ones:
///
/// ```compile_fail
/// use borrowed_buf::BorrowedCursor;
/// fn shorten<'a, 'b>(c: BorrowedCursor<'a, &'static str>) -> BorrowedCursor<'a, &'b str> {
///     c
/// }
/// ```
///
/// The buffer can't be accessed while a cursor over it is alive:
///
/// ```compile_fail,E0499
/// use borrowed_buf::BorrowedBuf;
/// use core::mem::MaybeUninit;
/// let mut storage = [MaybeUninit::<u8>::uninit(); 4];
/// let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
/// let mut cursor = buf.unfilled();
/// buf.clear();
/// cursor.append(b"a");
/// ```
///
/// A buffer built from a cursor can't outlive the cursor's borrow:
///
/// ```compile_fail,E0499
/// use borrowed_buf::BorrowedBuf;
/// use core::mem::MaybeUninit;
/// let mut storage = [MaybeUninit::<u8>::uninit(); 4];
/// let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
/// let sub = BorrowedBuf::from(buf.unfilled());
/// buf.clear();
/// drop(sub);
/// ```
///
/// A cursor is not `Send` when its elements are not:
///
/// ```compile_fail,E0277
/// use borrowed_buf::BorrowedBuf;
/// use core::mem::MaybeUninit;
/// fn assert_send<T: Send>(_: T) {}
/// let mut storage = [MaybeUninit::<*const u8>::uninit(); 1];
/// let mut buf = BorrowedBuf::<*const u8>::from(&mut storage[..]);
/// assert_send(buf.unfilled());
/// ```
#[cfg(doctest)]
struct CompileFail;

use borrowed_buf::{BorrowedBuf, BorrowedCursor};
use core::fmt::Debug;
use core::mem::MaybeUninit;

fn uninit<const N: usize>() -> [MaybeUninit<u8>; N] {
    [MaybeUninit::uninit(); N]
}

#[test]
fn from_uninit_slice_is_empty_and_uninit() {
    let mut storage = uninit::<8>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    assert_eq!(buf.capacity(), 8);
    assert_eq!(buf.len(), 0);
    assert!(!buf.is_init());
    assert_eq!(buf.filled(), b"");
    let cursor = buf.unfilled();
    assert_eq!(cursor.capacity(), 8);
    assert_eq!(cursor.written(), 0);
    assert!(!cursor.is_init());
}

#[test]
fn from_init_slice_is_fully_initialized() {
    let mut storage = [7u8; 5];
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    assert!(buf.is_init());
    let mut cursor = buf.unfilled();
    assert!(cursor.is_init());
    // Already initialized: nothing is overwritten.
    assert_eq!(cursor.ensure_init(), &[7; 5]);
    cursor.advance_checked(3);
    assert_eq!(buf.filled(), &[7; 3]);
}

#[test]
fn from_init_slice_writes_through_to_storage() {
    let mut storage = [0u8; 4];
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    buf.unfilled().append(b"ab");
    buf.filled_mut()[1] = b'B';
    assert_eq!(storage, *b"aB\0\0");
}

#[test]
fn zero_capacity() {
    let empty: &mut [MaybeUninit<u8>] = &mut [];
    let mut buf = BorrowedBuf::<u8>::from(empty);
    let mut cursor = buf.unfilled();
    assert_eq!(cursor.capacity(), 0);
    assert!(cursor.ensure_init().is_empty());
    // SAFETY: an empty cursor is trivially initialized.
    assert!(unsafe { cursor.as_mut() }.is_empty());
    cursor.append(&[]);
    cursor.advance_checked(0);
    // SAFETY: advancing by zero is always valid.
    unsafe { cursor.advance(0) };
    assert_eq!(cursor.with_unfilled_buf(|b| b.capacity()), 0);
    assert_eq!(buf.into_filled(), b"");
}

#[test]
fn append_fills_without_initializing_the_rest() {
    let mut storage = uninit::<8>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    buf.unfilled().append(b"abc");
    buf.unfilled().append(b"de");
    assert_eq!(buf.filled(), b"abcde");
    assert!(!buf.is_init());
    buf.filled_mut()[0] = b'A';
    assert_eq!(buf.into_filled(), b"Abcde");
}

#[test]
fn append_up_to_capacity() {
    let mut storage = uninit::<4>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.append(b"wxyz");
    assert_eq!(cursor.capacity(), 0);
    cursor.append(b"");
    assert_eq!(buf.filled(), b"wxyz");
}

#[test]
#[should_panic = "appended past the end"]
fn append_overflow_panics() {
    let mut storage = uninit::<4>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    buf.unfilled().append(b"12345");
}

#[test]
#[should_panic = "appended past the end"]
fn append_overflow_after_fill_panics() {
    let mut storage = uninit::<4>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.append(b"123");
    cursor.append(b"45");
}

#[test]
#[should_panic = "advanced past the initialized part"]
fn advance_checked_on_uninit_panics() {
    let mut storage = uninit::<4>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    buf.unfilled().advance_checked(1);
}

#[test]
#[should_panic = "advanced past the initialized part"]
fn advance_checked_past_capacity_panics() {
    let mut storage = [0u8; 4];
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.advance_checked(3);
    cursor.advance_checked(2);
}

#[test]
fn advance_checked_on_init_buffer() {
    let mut storage = *b"0123";
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.advance_checked(1).advance_checked(2);
    assert_eq!(cursor.written(), 3);
    assert_eq!(cursor.capacity(), 1);
    cursor.advance_checked(1);
    assert_eq!(buf.filled(), b"0123");
}

#[test]
fn advance_checked_after_append_on_init_buffer() {
    let mut storage = [b'.'; 6];
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.append(b"ab");
    // The rest of the cursor is still initialized.
    assert!(cursor.is_init());
    cursor.advance_checked(4);
    assert_eq!(buf.filled(), b"ab....");
}

#[test]
fn ensure_init_zeroes_uninit_buffer_once() {
    let mut storage = uninit::<6>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    buf.unfilled().append(b"xyz");

    let mut cursor = buf.unfilled();
    assert_eq!(cursor.ensure_init(), &[0; 3]);
    assert!(cursor.is_init());
    cursor.ensure_init().copy_from_slice(b"123");
    // A second call keeps what was written.
    assert_eq!(cursor.ensure_init(), b"123");
    cursor.advance_checked(2);
    assert_eq!(buf.filled(), b"xyz12");
    assert!(buf.is_init());

    // `clear` keeps the buffer initialized: the formerly filled part is initialized too.
    buf.clear();
    assert!(buf.is_init());
    assert_eq!(buf.unfilled().ensure_init(), b"xyz123");
}

#[test]
fn buf_set_init_on_initialized_storage() {
    let mut storage = [MaybeUninit::new(5u8); 4];
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    assert!(!buf.is_init());
    // SAFETY: every element of `storage` was initialized above.
    let buf = unsafe { buf.set_init() };
    assert!(buf.is_init());
    assert_eq!(buf.unfilled().ensure_init(), &[5; 4]);
    buf.unfilled().advance_checked(4);
    assert_eq!(buf.filled(), &[5; 4]);
}

#[test]
fn buf_set_init_after_partial_fill() {
    let mut storage = uninit::<4>();
    storage[2].write(b'c');
    storage[3].write(b'd');
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    buf.unfilled().append(b"ab");
    // SAFETY: the first two elements were filled and the last two written up front.
    unsafe { buf.set_init() };
    buf.unfilled().advance_checked(2);
    assert_eq!(buf.filled(), b"abcd");
}

#[test]
fn buf_set_init_is_idempotent_and_chains() {
    let mut storage = [1u8; 3];
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    // SAFETY: built from an initialized slice.
    unsafe { buf.set_init().set_init() }.clear();
    assert!(buf.is_init());
}

#[test]
fn cursor_set_init_after_writing_through_as_mut() {
    let mut storage = uninit::<6>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    buf.unfilled().append(b"ab");

    let mut cursor = buf.unfilled();
    // SAFETY: only initialized values are written.
    for (i, slot) in unsafe { cursor.as_mut() }.iter_mut().enumerate() {
        slot.write(b'0' + i as u8);
    }
    // SAFETY: every element of the cursor was just written.
    unsafe { cursor.set_init() };
    assert!(cursor.is_init());
    assert_eq!(cursor.ensure_init(), b"0123");
    cursor.advance_checked(3);
    assert_eq!(buf.filled(), b"ab012");
    assert!(buf.is_init());
}

#[test]
fn cursor_set_init_on_full_buffer() {
    let mut storage = uninit::<2>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.append(b"ab");
    // SAFETY: an empty cursor is trivially initialized.
    unsafe { cursor.set_init() };
    assert!(buf.is_init());
    assert_eq!(buf.filled(), b"ab");
}

#[test]
fn as_mut_then_advance() {
    let mut storage = uninit::<4>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    // SAFETY: only initialized bytes are written.
    let raw = unsafe { cursor.as_mut() };
    assert_eq!(raw.len(), 4);
    raw[0].write(1);
    raw[1].write(2);
    // SAFETY: the first 2 elements of the cursor were just written.
    unsafe { cursor.advance(2) };
    assert_eq!(cursor.written(), 2);
    assert_eq!(cursor.capacity(), 2);
    // `as_mut` now starts after the filled part.
    // SAFETY: only initialized bytes are written.
    let raw = unsafe { cursor.as_mut() };
    assert_eq!(raw.len(), 2);
    raw[0].write(3);
    // SAFETY: the first element of the cursor was just written.
    unsafe { cursor.advance(1) };
    assert_eq!(buf.filled(), &[1, 2, 3]);
    assert!(!buf.is_init());
}

#[test]
fn as_mut_overwrites_initialized_elements() {
    let mut storage = *b"....";
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.append(b"a");
    // SAFETY: only initialized bytes are written, so the cursor stays initialized.
    let raw = unsafe { cursor.as_mut() };
    raw[1].write(b'!');
    assert!(cursor.is_init());
    assert_eq!(cursor.ensure_init(), b".!.");
    cursor.advance_checked(3);
    assert_eq!(buf.filled(), b"a.!.");
}

#[test]
fn advance_chains_and_reaches_capacity() {
    let mut storage = uninit::<3>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    // SAFETY: only initialized bytes are written.
    unsafe { cursor.as_mut() }.fill(MaybeUninit::new(9));
    // SAFETY: all 3 elements were just written.
    unsafe { cursor.advance(1).advance(2) };
    assert_eq!(cursor.capacity(), 0);
    assert_eq!(buf.into_filled_mut(), &mut [9; 3]);
}

#[test]
fn reborrow_shares_written() {
    let mut storage = uninit::<8>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut outer = buf.unfilled();
    outer.append(b"a");
    {
        let mut inner = outer.reborrow();
        // `written` counts everything filled in the `BorrowedBuf`, like `core::io`.
        assert_eq!(inner.written(), 1);
        inner.append(b"bc");
        assert_eq!(inner.written(), 3);
        let mut innermost = inner.reborrow();
        innermost.append(b"d");
    }
    assert_eq!(outer.written(), 4);
    assert_eq!(outer.capacity(), 4);
    assert_eq!(buf.filled(), b"abcd");
}

#[test]
fn written_counts_from_start_of_buffer() {
    let mut storage = uninit::<8>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    buf.unfilled().append(b"abc");
    let cursor = buf.unfilled();
    assert_eq!(cursor.written(), 3);
    assert_eq!(cursor.capacity(), 5);
}

#[test]
fn clear_resets_filled_only() {
    let mut storage = uninit::<4>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    buf.unfilled().append(b"ab");
    assert_eq!(buf.clear().len(), 0);
    assert!(!buf.is_init());
    assert_eq!(buf.unfilled().capacity(), 4);
    buf.unfilled().append(b"xyzw");
    assert_eq!(buf.filled(), b"xyzw");
}

#[test]
fn into_filled_outlives_the_buf() {
    let mut storage = uninit::<4>();
    let filled: &[u8] = {
        let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
        buf.unfilled().append(b"ok");
        buf.into_filled()
    };
    assert_eq!(filled, b"ok");
}

#[test]
fn into_filled_mut_outlives_the_buf() {
    let mut storage = uninit::<4>();
    let filled: &mut [u8] = {
        let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
        buf.unfilled().append(b"ok");
        buf.into_filled_mut()
    };
    filled[0] = b'O';
    assert_eq!(filled, b"Ok");
}

#[test]
fn buf_from_cursor_covers_unfilled_part() {
    let mut storage = uninit::<6>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    buf.unfilled().append(b"ab");
    {
        let mut sub = BorrowedBuf::from(buf.unfilled());
        assert_eq!(sub.capacity(), 4);
        assert_eq!(sub.len(), 0);
        assert!(!sub.is_init());
        sub.unfilled().append(b"cd");
        assert_eq!(sub.into_filled(), b"cd");
    }
    // Filling through the sub-buffer doesn't advance the parent buffer.
    assert_eq!(buf.filled(), b"ab");
    // SAFETY: `cd` was written to the next 2 elements through the sub-buffer.
    unsafe { buf.unfilled().advance(2) };
    assert_eq!(buf.filled(), b"abcd");
}

#[test]
fn buf_from_cursor_inherits_init() {
    let mut storage = *b"abcdef";
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    buf.unfilled().advance_checked(2);
    let mut sub = BorrowedBuf::from(buf.unfilled());
    assert!(sub.is_init());
    assert_eq!(sub.unfilled().ensure_init(), b"cdef");
}

#[test]
fn with_unfilled_buf_propagates_filled_and_init() {
    let mut storage = uninit::<8>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.append(b"ab");

    let seen = cursor.with_unfilled_buf(|sub| {
        assert_eq!((sub.capacity(), sub.len(), sub.is_init()), (6, 0, false));
        sub.unfilled().append(b"cd");
        assert_eq!(sub.filled(), b"cd");
        sub.len()
    });
    assert_eq!(seen, 2);
    assert_eq!(cursor.written(), 4);
    assert!(!cursor.is_init());

    cursor.with_unfilled_buf(|sub| {
        let mut c = sub.unfilled();
        c.ensure_init()[0] = b'e';
        c.advance_checked(1);
    });
    assert!(cursor.is_init());
    cursor.advance_checked(3);
    assert_eq!(buf.filled(), b"abcde\0\0\0");
    assert!(buf.is_init());
}

#[test]
fn with_unfilled_buf_unsafe_ops_inside() {
    let mut storage = uninit::<5>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.with_unfilled_buf(|sub| {
        let mut c = sub.unfilled();
        // SAFETY: only initialized bytes are written.
        unsafe { c.as_mut() }.fill(MaybeUninit::new(b'z'));
        // SAFETY: every element of the cursor was just written.
        unsafe { c.set_init() };
        // SAFETY: the first 2 elements are initialized.
        unsafe { c.advance(2) };
    });
    assert_eq!(cursor.written(), 2);
    assert!(cursor.is_init());
    cursor.advance_checked(3);
    assert_eq!(buf.filled(), b"zzzzz");
}

#[test]
fn with_unfilled_buf_nested() {
    let mut storage = uninit::<6>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    buf.unfilled().with_unfilled_buf(|outer| {
        outer.unfilled().append(b"a");
        outer.unfilled().with_unfilled_buf(|inner| {
            inner.unfilled().append(b"bc");
            assert_eq!(inner.filled(), b"bc");
        });
        assert_eq!(outer.filled(), b"abc");
        outer.unfilled().append(b"d");
    });
    assert_eq!(buf.filled(), b"abcd");
}

#[test]
fn with_unfilled_buf_after_clear_inside_restarts_sub_buffer() {
    let mut storage = uninit::<4>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.with_unfilled_buf(|sub| {
        sub.unfilled().append(b"xy");
        sub.clear();
        sub.unfilled().append(b"z");
    });
    assert_eq!(buf.filled(), b"z");
}

#[test]
#[should_panic = "was replaced"]
fn with_unfilled_buf_rejects_replaced_buffer() {
    let mut storage = uninit::<4>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.with_unfilled_buf(|sub| {
        // Without the check this would mark the original buffer as initialized.
        let empty: &mut [u8] = &mut [];
        *sub = BorrowedBuf::from(empty);
        // SAFETY: the empty buffer is trivially initialized.
        unsafe { sub.set_init() };
    });
}

#[test]
fn with_unfilled_buf_panic_leaves_cursor_usable() {
    let mut storage = uninit::<4>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cursor.with_unfilled_buf(|sub| {
            sub.unfilled().append(b"lost");
            panic!("boom");
        })
    }));
    assert!(result.is_err());
    // The progress made inside the closure is not recorded.
    assert_eq!(cursor.written(), 0);
    cursor.append(b"ok");
    assert_eq!(buf.filled(), b"ok");
}

#[test]
fn cursor_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>(_: &T) {}
    let mut storage = uninit::<1>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    assert_send_sync(&buf);
    assert_send_sync(&buf.unfilled());
}

#[test]
fn cursor_moves_to_another_thread() {
    let mut storage = uninit::<4>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let cursor = buf.unfilled();
    std::thread::scope(|s| {
        s.spawn(move || {
            let mut cursor = cursor;
            cursor.append(b"thr");
        });
    });
    assert_eq!(buf.filled(), b"thr");
}

#[test]
fn cursor_lifetime_is_covariant() {
    fn shorten<'short, 'long: 'short>(c: BorrowedCursor<'long, u8>) -> BorrowedCursor<'short, u8> {
        c
    }
    let mut storage = uninit::<2>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    shorten(buf.unfilled()).append(b"v");
    assert_eq!(buf.filled(), b"v");
}

#[test]
fn debug_output() {
    let mut storage = uninit::<4>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    buf.unfilled().append(b"a");
    assert_eq!(
        format!("{buf:?}"),
        "BorrowedBuf { init: false, filled: 1, capacity: 4 }"
    );
    assert_eq!(
        format!("{:?}", buf.unfilled()),
        "BorrowedCursor { buf: BorrowedBuf { init: false, filled: 1, capacity: 4 } }"
    );
}

/// Applies a random sequence of operations and checks the result against a plain `Vec` model.
/// Under Miri this catches any read of uninitialized memory or aliasing violation.
#[test]
fn random_ops_match_model() {
    random_ops(|b| b);
    random_ops(|b| u64::from(b) << 40 | 0xff);
    random_ops(|b| (f32::from(b), [b; 3]));
    random_ops(|_| ());
}

struct Rng(u64);

impl Rng {
    fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % bound as u64) as usize
    }
}

/// What the buffer should contain: the elements known to be initialized, how many are filled and
/// whether the unfilled part is flagged as initialized.
struct Model<T> {
    data: Vec<Option<T>>,
    filled: usize,
    init: bool,
}

impl<T: Copy + Default + PartialEq + Debug> Model<T> {
    fn write(&mut self, at: usize, values: &[T]) {
        for (slot, v) in self.data[at..].iter_mut().zip(values) {
            *slot = Some(*v);
        }
    }

    fn check(&self, buf: &mut BorrowedBuf<'_, T>) {
        let filled: Vec<T> = self.data[..self.filled]
            .iter()
            .map(|v| v.unwrap())
            .collect();
        assert_eq!(buf.len(), self.filled);
        assert_eq!(buf.is_init(), self.init);
        assert_eq!(buf.filled(), &filled[..]);
        if self.init {
            let tail: Vec<T> = self.data[self.filled..]
                .iter()
                .map(|v| v.unwrap())
                .collect();
            assert_eq!(buf.unfilled().ensure_init(), &tail[..]);
        }
    }

    /// Runs one random operation on `cursor`, whose unfilled part starts at `self.filled` and ends
    /// at `end` in model coordinates.
    fn step(&mut self, rng: &mut Rng, mut cursor: BorrowedCursor<'_, T>, elem: &impl Fn(u8) -> T) {
        let room = cursor.capacity();
        let end = self.filled + room;
        assert_eq!(end, self.data.len());
        match rng.below(8) {
            0 => {
                let data: Vec<T> = (0..rng.below(room + 1))
                    .map(|_| elem(rng.below(256) as u8))
                    .collect();
                cursor.append(&data);
                self.write(self.filled, &data);
                self.filled += data.len();
            }
            1 => {
                let n = if self.init { rng.below(room + 1) } else { 0 };
                cursor.advance_checked(n);
                self.filled += n;
            }
            2 => {
                let tail = cursor.ensure_init();
                if self.init {
                    let expected: Vec<T> = self.data[self.filled..]
                        .iter()
                        .map(|v| v.unwrap())
                        .collect();
                    assert_eq!(tail, &expected[..]);
                } else {
                    assert!(tail.iter().all(|v| *v == T::default()));
                    for slot in &mut self.data[self.filled..] {
                        *slot = Some(T::default());
                    }
                    self.init = true;
                }
            }
            3 => {
                // Write a prefix in place, then advance over part of it.
                let n = rng.below(room + 1);
                let value = elem(rng.below(256) as u8);
                // SAFETY: only initialized values are written.
                for slot in unsafe { cursor.as_mut() }.iter_mut().take(n) {
                    slot.write(value);
                }
                let k = rng.below(n + 1);
                // SAFETY: the first `n >= k` elements of the cursor were just written.
                unsafe { cursor.advance(k) };
                let at = self.filled;
                self.write(at, &vec![value; n]);
                self.filled += k;
            }
            4 => {
                // Write the whole cursor in place and flag it as initialized.
                let value = elem(rng.below(256) as u8);
                // SAFETY: only initialized values are written.
                unsafe { cursor.as_mut() }.fill(MaybeUninit::new(value));
                // SAFETY: every element of the cursor was just written.
                unsafe { cursor.set_init() };
                let at = self.filled;
                self.write(at, &vec![value; room]);
                self.init = true;
            }
            5 => {
                let n = rng.below(room + 1);
                let data = seq(n, elem);
                let mut inner = cursor.reborrow();
                let before = inner.written();
                inner.append(&data);
                assert_eq!(inner.written(), before + n);
                self.write(self.filled, &data);
                self.filled += n;
            }
            6 => {
                // Run nested operations on a sub-buffer, modelled as a sub-model.
                let start = self.filled;
                let mut sub = Model {
                    data: self.data[start..].to_vec(),
                    filled: 0,
                    init: self.init,
                };
                cursor.with_unfilled_buf(|b| {
                    for _ in 0..rng.below(4) {
                        if rng.below(5) == 0 {
                            b.clear();
                            sub.filled = 0;
                        } else {
                            sub.step(rng, b.unfilled(), elem);
                        }
                    }
                    sub.check(b);
                });
                self.data[start..].copy_from_slice(&sub.data);
                self.filled += sub.filled;
                self.init = sub.init;
            }
            _ => {
                // Hand the cursor's region to a standalone sub-buffer.
                let start = self.filled;
                let mut b = BorrowedBuf::from(cursor);
                assert_eq!(b.capacity(), room);
                assert_eq!(b.is_init(), self.init);
                let n = rng.below(room + 1);
                let data = seq(n, elem);
                b.unfilled().append(&data);
                assert_eq!(b.into_filled(), &data[..]);
                // The parent isn't advanced, but the elements are now initialized.
                self.write(start, &data);
            }
        }
    }
}

fn random_ops<T: Copy + Default + PartialEq + Debug>(elem: impl Fn(u8) -> T) {
    const CAP: usize = 24;
    let rounds = if cfg!(miri) { 6 } else { 300 };
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);

    for _ in 0..rounds {
        let mut storage = [MaybeUninit::<T>::uninit(); CAP];
        let mut buf = BorrowedBuf::<T>::from(&mut storage[..]);
        let mut model = Model {
            data: vec![None; CAP],
            filled: 0,
            init: false,
        };

        for _ in 0..24 {
            match rng.below(10) {
                0 => {
                    buf.clear();
                    model.filled = 0;
                }
                1 if model.data.iter().all(Option::is_some) => {
                    // SAFETY: the model says every element is initialized.
                    unsafe { buf.set_init() };
                    model.init = true;
                }
                _ => model.step(&mut rng, buf.unfilled(), &elem),
            }
            model.check(&mut buf);
        }
    }
}

fn seq<T>(n: usize, elem: &impl Fn(u8) -> T) -> Vec<T> {
    (0..n as u8).map(elem).collect()
}

#[test]
fn append_fill_writes_n_copies_without_initializing_the_rest() {
    let mut storage = uninit::<8>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.append(b"ab");
    cursor.append_fill(3, 0);
    assert_eq!(cursor.written(), 5);
    assert_eq!(cursor.capacity(), 3);
    assert!(!cursor.is_init());
    cursor.append_fill(0, b'?');
    cursor.append(b"c");
    assert_eq!(buf.filled(), b"ab\0\0\0c");
    assert!(!buf.is_init());
}

#[test]
fn append_fill_up_to_capacity() {
    let mut storage = uninit::<4>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    buf.unfilled().append_fill(4, b'z');
    assert_eq!(buf.filled(), b"zzzz");
}

#[test]
fn append_fill_keeps_init_buffer_init() {
    let mut storage = *b"......";
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.append_fill(2, b'x');
    assert!(cursor.is_init());
    cursor.advance_checked(4);
    assert_eq!(buf.filled(), b"xx....");
}

#[test]
#[should_panic = "appended past the end"]
fn append_fill_overflow_panics() {
    let mut storage = uninit::<4>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.append(b"ab");
    cursor.append_fill(3, 0);
}

#[test]
fn truncate_shortens_filled_only() {
    let mut storage = uninit::<6>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    buf.unfilled().append(b"abcd");
    // Truncating to the current length or beyond is a no-op.
    assert_eq!(buf.truncate(4).len(), 4);
    assert_eq!(buf.truncate(10).len(), 4);
    assert_eq!(buf.truncate(1).filled(), b"a");
    assert!(!buf.is_init());
    assert_eq!(buf.unfilled().capacity(), 5);
    buf.unfilled().append(b"BCDEF");
    assert_eq!(buf.filled(), b"aBCDEF");
    assert_eq!(buf.truncate(0).len(), 0);
}

#[test]
fn truncate_keeps_init_buffer_init() {
    let mut storage = uninit::<4>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    buf.unfilled().append(b"ab");
    buf.unfilled().ensure_init();
    buf.truncate(1);
    assert!(buf.is_init());
    // The cut-off element is visible again through the initialized cursor.
    assert_eq!(buf.unfilled().ensure_init(), b"b\0\0");
}

#[test]
fn truncate_inside_with_unfilled_buf() {
    let mut storage = uninit::<8>();
    let mut buf = BorrowedBuf::<u8>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.append(b"ab");
    cursor.with_unfilled_buf(|sub| {
        sub.unfilled().append(b"cdef");
        // Only the sub-buffer's own filled part can be truncated.
        sub.truncate(1);
    });
    assert_eq!(cursor.written(), 3);
    assert_eq!(buf.filled(), b"abc");
}

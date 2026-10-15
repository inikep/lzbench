use borrowed_buf::BorrowedBuf;
use core::mem::MaybeUninit;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Pixel {
    r: u8,
    g: u8,
    b: u8,
    a: f32,
}

#[test]
fn u32_elements() {
    let mut storage = [MaybeUninit::<u32>::uninit(); 4];
    let mut buf = BorrowedBuf::<u32>::from(&mut storage[..]);
    buf.unfilled().append(&[u32::MAX, 1]);
    let mut cursor = buf.unfilled();
    assert_eq!(cursor.capacity(), 2);
    assert_eq!(cursor.ensure_init(), &[0, 0]);
    cursor.ensure_init()[0] = 7;
    cursor.advance_checked(1);
    assert_eq!(buf.filled(), &[u32::MAX, 1, 7]);
    assert!(buf.is_init());
}

#[test]
fn f32_ensure_init_uses_default() {
    let mut storage = [MaybeUninit::<f32>::uninit(); 3];
    let mut buf = BorrowedBuf::<f32>::from(&mut storage[..]);
    assert_eq!(buf.unfilled().ensure_init(), &[0.0; 3]);
}

#[test]
fn struct_elements() {
    let px = Pixel {
        r: 1,
        g: 2,
        b: 3,
        a: 0.5,
    };
    let mut storage = [MaybeUninit::<Pixel>::uninit(); 3];
    let mut buf = BorrowedBuf::<Pixel>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.append(&[px]);
    // SAFETY: only initialized values are written.
    let raw = unsafe { cursor.as_mut() };
    raw[0].write(Pixel { a: 1.0, ..px });
    // SAFETY: the first element of the cursor was just written.
    unsafe { cursor.advance(1) };
    assert_eq!(cursor.written(), 2);
    assert_eq!(buf.filled(), &[px, Pixel { a: 1.0, ..px }]);
}

#[test]
fn struct_elements_set_init() {
    let px = Pixel::default();
    let mut storage = [MaybeUninit::<Pixel>::uninit(); 2];
    let mut buf = BorrowedBuf::<Pixel>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    // SAFETY: only initialized values are written.
    unsafe { cursor.as_mut() }.fill(MaybeUninit::new(px));
    // SAFETY: every element of the cursor was just written.
    unsafe { cursor.set_init() };
    cursor.ensure_init()[1].g = 9;
    cursor.advance_checked(2);
    assert_eq!(buf.filled(), &[px, Pixel { g: 9, ..px }]);
}

#[test]
fn from_init_generic_slice() {
    let mut storage = [Pixel::default(); 2];
    let mut buf = BorrowedBuf::from(&mut storage[..]);
    buf.unfilled().ensure_init()[1].g = 9;
    buf.unfilled().advance_checked(2);
    assert_eq!(buf.filled()[1].g, 9);
}

#[test]
fn references_as_elements() {
    let words = [String::from("a"), String::from("b")];
    let mut storage = [MaybeUninit::<&str>::uninit(); 3];
    let mut buf = BorrowedBuf::<&str>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.append(&[words[0].as_str()]);
    cursor.with_unfilled_buf(|sub| sub.unfilled().append(&[words[1].as_str()]));
    assert_eq!(buf.into_filled(), &["a", "b"]);
}

#[test]
fn zero_sized_elements() {
    let mut storage = [MaybeUninit::<()>::uninit(); 5];
    let mut buf = BorrowedBuf::<()>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    cursor.append(&[(), ()]);
    assert_eq!(cursor.ensure_init().len(), 3);
    cursor.advance_checked(3);
    assert_eq!(buf.len(), 5);
    assert_eq!(buf.into_filled(), &[(); 5]);
}

#[test]
fn zero_sized_elements_unsafe_paths() {
    let mut storage = [MaybeUninit::<()>::uninit(); 4];
    let mut buf = BorrowedBuf::<()>::from(&mut storage[..]);
    let mut cursor = buf.unfilled();
    // SAFETY: zero-sized values are always initialized.
    unsafe { cursor.advance(1) };
    // SAFETY: as above.
    assert_eq!(unsafe { cursor.as_mut() }.len(), 3);
    // SAFETY: as above.
    unsafe { cursor.set_init() };
    cursor.with_unfilled_buf(|sub| {
        sub.unfilled().advance_checked(2);
    });
    let sub = BorrowedBuf::from(cursor);
    assert_eq!((sub.capacity(), sub.is_init()), (1, true));
    assert_eq!(buf.into_filled_mut().len(), 3);
}

#[test]
#[should_panic = "appended past the end"]
fn generic_append_overflow_panics() {
    let mut storage = [MaybeUninit::<u64>::uninit(); 1];
    let mut buf = BorrowedBuf::<u64>::from(&mut storage[..]);
    buf.unfilled().append(&[1, 2]);
}

#[test]
fn append_fill_struct_elements() {
    let px = Pixel {
        r: 1,
        g: 2,
        b: 3,
        a: 0.5,
    };
    let mut storage = [MaybeUninit::<Pixel>::uninit(); 4];
    let mut buf = BorrowedBuf::<Pixel>::from(&mut storage[..]);
    buf.unfilled().append_fill(3, px);
    assert_eq!(buf.filled(), &[px; 3]);
    buf.truncate(1);
    buf.unfilled().append_fill(1, Pixel::default());
    assert_eq!(buf.into_filled(), &[px, Pixel::default()]);
}

#[test]
fn append_fill_zero_sized_elements() {
    let mut storage = [MaybeUninit::<()>::uninit(); 5];
    let mut buf = BorrowedBuf::<()>::from(&mut storage[..]);
    buf.unfilled().append_fill(5, ());
    assert_eq!(buf.len(), 5);
    buf.truncate(2);
    assert_eq!(buf.unfilled().capacity(), 3);
}

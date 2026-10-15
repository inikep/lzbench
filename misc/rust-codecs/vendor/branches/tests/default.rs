use branches::{assume, likely, unlikely};
use core::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn test_likely_identity() {
    for &b in &[true, false] {
        assert_eq!(likely(b), b);
    }
}

#[test]
fn test_unlikely_identity() {
    for &b in &[true, false] {
        assert_eq!(unlikely(b), b);
    }
}

#[test]
fn test_likely_unlikely_mixed() {
    let data = [true, false, true, true, false];
    let mut c1 = 0usize;
    let mut c2 = 0usize;
    for &b in &data {
        if likely(b) {
            c1 += 1;
        }
        if unlikely(!b) {
            c2 += 1;
        }
    }
    assert_eq!(c1, 3);
    assert_eq!(c2, 2);
}

#[test]
fn test_assume_true_no_effect() {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    for i in 0..100 {
        unsafe {
            assume(i < 100);
        }
        COUNTER.fetch_add(1, Ordering::Relaxed);
    }
    assert_eq!(COUNTER.load(Ordering::Relaxed), 100);
}

// `assume` is const on rustc 1.57+; every toolchain that runs the test suite
// qualifies.
const fn const_len_like(n: usize) -> usize {
    unsafe { assume(n <= 8) };
    n
}

#[test]
fn test_assume_in_const_fn() {
    const N: usize = const_len_like(4);
    assert_eq!(N, 4);
    assert_eq!(const_len_like(8), 8);
}

#[test]
fn test_assume_in_loop_logic_preserved() {
    let mut sum = 0usize;
    for i in 1..=32 {
        unsafe {
            assume(i <= 32);
        }
        sum += i;
    }
    assert_eq!(sum, (32 * 33) / 2);
}

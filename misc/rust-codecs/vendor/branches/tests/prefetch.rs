#![cfg(feature = "prefetch")]
use branches::*;

#[test]
#[cfg(feature = "prefetch")]
fn test_prefetch_combined_usage_smoke() {
    let mut arr: Vec<u8> = Vec::with_capacity(1_000_000);
    for i in 0..1_000_000 {
        let value = match i % 4 {
            0 => 10,
            1 => 20,
            2 => 30,
            _ => 40,
        };
        arr.push(value);
    }
    let mut acc = 0u32;
    for &v in &arr {
        unsafe {
            assume(v <= 40);
        }
        if likely(v % 20 != 0) {
            acc += v as u32;
        } else if unlikely(v == 20) {
            acc += 1;
        }
        prefetch_read_data::<_, 0>(&arr);
    }
    assert_eq!(acc, 10250000);
}

#[test]
#[cfg(feature = "prefetch")]
fn test_prefetch_read_variants() {
    let buf = [0u8; 64];
    let ptr = buf.as_ptr();
    prefetch_read_data::<_, 0>(ptr);
    prefetch_read_data::<_, 1>(ptr);
    prefetch_read_data::<_, 2>(ptr);
    prefetch_read_data::<_, 3>(ptr);
    assert_eq!(buf.iter().sum::<u8>(), 0);
}

#[test]
#[cfg(feature = "prefetch")]
fn test_prefetch_write_variants() {
    let buf = [1u8; 32];
    let ptr = buf.as_ptr();
    prefetch_write_data::<_, 0>(ptr);
    prefetch_write_data::<_, 1>(ptr);
    prefetch_write_data::<_, 2>(ptr);
    prefetch_write_data::<_, 3>(ptr);
    assert_eq!(buf.iter().sum::<u8>(), 32);
}

// Out-of-range localities are valid input: they must clamp to non-temporal
// instead of crashing codegen (nightly previously ICEd on values outside 0..=3).
#[test]
#[cfg(feature = "prefetch")]
fn test_prefetch_out_of_range_locality() {
    let buf = [0u8; 8];
    let ptr = buf.as_ptr();
    prefetch_read_data::<_, { -1 }>(ptr);
    prefetch_read_data::<_, 4>(ptr);
    prefetch_read_data::<_, { i32::MAX }>(ptr);
    prefetch_write_data::<_, { -1 }>(ptr);
    prefetch_write_data::<_, 4>(ptr);
    prefetch_write_data::<_, { i32::MAX }>(ptr);
    assert_eq!(buf.iter().sum::<u8>(), 0);
}

#[test]
#[cfg(feature = "prefetch")]
fn test_prefetch_multiple_addresses() {
    let mut big = [0u8; 256];
    for (i, b) in big.iter_mut().enumerate() {
        *b = (i & 0xFF) as u8;
    }
    for chunk in big.chunks(16) {
        prefetch_read_data::<_, 0>(chunk.as_ptr());
        prefetch_write_data::<_, 1>(chunk.as_ptr());
    }
    let checksum: u32 = big.iter().map(|&b| b as u32).sum();
    assert_eq!(checksum, (0u32..256).sum());
}

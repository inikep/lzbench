
pub mod ans;
pub mod ans_fast;
pub mod bwt;
pub mod bwt_big;
pub mod text_detect;
pub mod match_maker;
pub mod lz_opt;
pub mod lz_full;
pub mod own_lz;
pub mod own_lz_fast;
pub mod zrw;
pub mod inhibitor;
pub mod trufold;
pub mod zpaq_fixed;
pub mod bwt_ans;
pub const VERSION: &str = "2.5.0";
pub const LOCK_DICKENS_REF: u32 = 243675;
pub fn version() -> &'static str { VERSION }
/// Accept a candidate blob that is strictly smaller than the input.
/// When `verify` is true (CLI / unit tests), fully decode and compare so a
/// buggy encoder cannot ship a bad frame. The lzbench FFI path passes
/// `verify = false`: round-trip checking there would add up to three full
/// decompressions to every reported compression timing.
fn accept_candidate(raw: &[u8], blob: Vec<u8>, verify: bool) -> Option<Vec<u8>> {
    if blob.len() >= raw.len() {
        return None;
    }
    if !verify {
        return Some(blob);
    }
    match pulsar_decode(&blob) {
        Ok(back) if back == raw => Some(blob),
        _ => None,
    }
}

fn pulsar_encode_inner(data: &[u8], verify: bool) -> Option<Vec<u8>> {
    let mut cand: Vec<Vec<u8>> = Vec::new();
    // OZL2 never won Silesia/Calgary vs BW; skip the slow matcher on large files.
    if data.len() < 256 * 1024 {
        if let Some(e) = own_lz::own_lz_encode(data) {
            if let Some(ok) = accept_candidate(data, e, verify) {
                cand.push(ok);
            }
        }
    }
    // PZ22 is a residual wrapper; skip on large files (never won vs OZL2 on Silesia/Calgary).
    if data.len() < 64 * 1024 {
        if let Some(ok) = accept_candidate(data, zpaq_fixed::compress_fixed(data), verify) {
            cand.push(ok);
        }
    }
    if data.len() >= 256 {
        if let Some(ok) = accept_candidate(data, bwt_ans::compress(data), verify) {
            cand.push(ok);
        }
    }
    cand.into_iter().min_by_key(|v| v.len())
}

/// Encode with candidate verification (safe for CLI / tests).
pub fn pulsar_encode(data: &[u8]) -> Option<Vec<u8>> {
    pulsar_encode_inner(data, true)
}

/// Encode without the verify round-trip; used by the lzbench FFI so reported
/// compression speed is encoder-only.
pub fn pulsar_encode_unchecked(data: &[u8]) -> Option<Vec<u8>> {
    pulsar_encode_inner(data, false)
}

pub fn pulsar_decode(data: &[u8]) -> Result<Vec<u8>, &'static str> {
    if data.len() >= 4 && &data[..4] == zpaq_fixed::MAGIC {
        return zpaq_fixed::decompress_fixed(data);
    }
    if data.len() >= 4 && (&data[..4] == bwt_ans::MAGIC || &data[..4] == bwt_ans::MAGIC22) {
        return bwt_ans::decompress(data);
    }
    if data.len() >= 4 && &data[..4] == own_lz::OZL2_MAGIC {
        return match own_lz::own_lz_decode(data) {
            Ok(v) => Ok(v),
            Err(_) => own_lz_fast::own_lz_decode_fast(data),
        };
    }
    own_lz::own_lz_decode(data).or_else(|_| own_lz_fast::own_lz_decode_fast(data))
}
use std::os::raw::{c_int, c_uchar};
#[no_mangle]
pub extern "C" fn pulsar_encode_c(input_ptr: *const c_uchar, input_len: usize, out_ptr: *mut *mut c_uchar, out_len: *mut usize) -> c_int {
    if input_ptr.is_null() || out_ptr.is_null() || out_len.is_null() { return -1; }
    let input = unsafe { std::slice::from_raw_parts(input_ptr, input_len) };
    match pulsar_encode(input) {
        Some(enc) => { let len=enc.len(); let boxed=enc.into_boxed_slice(); let ptr=Box::into_raw(boxed) as *mut c_uchar; unsafe { *out_ptr=ptr; *out_len=len; } 0 },
        None => 1,
    }
}
#[no_mangle]
pub extern "C" fn pulsar_decode_c(input_ptr: *const c_uchar, input_len: usize, out_ptr: *mut *mut c_uchar, out_len: *mut usize) -> c_int {
    if input_ptr.is_null() || out_ptr.is_null() || out_len.is_null() { return -1; }
    let input = unsafe { std::slice::from_raw_parts(input_ptr, input_len) };
    match pulsar_decode(input) {
        Ok(dec) => { let len=dec.len(); let boxed=dec.into_boxed_slice(); let ptr=Box::into_raw(boxed) as *mut c_uchar; unsafe { *out_ptr=ptr; *out_len=len; } 0 },
        Err(_) => -2,
    }
}
#[no_mangle]
pub extern "C" fn pulsar_free(ptr: *mut c_uchar, len: usize) { if ptr.is_null() { return; } unsafe { let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr, len)); } }

/// Frame flags for the lzbench FFI path: a 1-byte flag followed by the payload.
/// 0x01 = pulsar-coded, 0x00 = stored (raw copy, used when the input doesn't
/// compress). The stored fallback guarantees the compressor never returns 0
/// on success (lzbench treats any return <= 0 as a compression error).
const PULSAR_FRAME_STORED: u8 = 0x00;
const PULSAR_FRAME_CODED: u8 = 0x01;

/// Compress into caller buffer. Returns total bytes written (flag + payload),
/// -1 on error. Any Rust panic is caught and reported as -1: a panic must
/// never cross the FFI boundary and abort the host process.
#[no_mangle]
pub unsafe extern "C" fn pulsar_compress(
    in_ptr: *const c_uchar,
    in_len: usize,
    out_ptr: *mut c_uchar,
    out_len: usize,
) -> isize {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if in_ptr.is_null() || out_ptr.is_null() || out_len < 1 {
            return -1isize;
        }
        let input = std::slice::from_raw_parts(in_ptr, in_len);
        // Unchecked: lzbench timings must not include the verify round-trips.
        match pulsar_encode_unchecked(input) {
            Some(enc) if enc.len() + 1 <= out_len => {
                *out_ptr = PULSAR_FRAME_CODED;
                std::ptr::copy_nonoverlapping(enc.as_ptr(), out_ptr.add(1), enc.len());
                (enc.len() + 1) as isize
            }
            _ => {
                // Stored fallback: nothing compressed smaller (or the coded
                // frame wouldn't fit); store the raw input instead of
                // returning 0, which lzbench reports as a compression error.
                if input.len() + 1 > out_len {
                    return -1;
                }
                *out_ptr = PULSAR_FRAME_STORED;
                if !input.is_empty() {
                    std::ptr::copy_nonoverlapping(in_ptr, out_ptr.add(1), input.len());
                }
                (input.len() + 1) as isize
            }
        }
    }));
    r.unwrap_or(-1)
}

/// Decompress into caller buffer. Expects the 1-byte frame flag written by
/// pulsar_compress. Returns decoded length, -1 on error. Panics are caught
/// and reported as -1 (never cross the FFI boundary).
#[no_mangle]
pub unsafe extern "C" fn pulsar_decompress(
    in_ptr: *const c_uchar,
    in_len: usize,
    out_ptr: *mut c_uchar,
    out_len: usize,
) -> isize {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if in_ptr.is_null() || out_ptr.is_null() || in_len < 1 {
            return -1isize;
        }
        let flag = *in_ptr;
        let payload = std::slice::from_raw_parts(in_ptr.add(1), in_len - 1);
        match flag {
            PULSAR_FRAME_STORED => {
                if payload.len() > out_len {
                    return -1;
                }
                if !payload.is_empty() {
                    std::ptr::copy_nonoverlapping(payload.as_ptr(), out_ptr, payload.len());
                }
                payload.len() as isize
            }
            PULSAR_FRAME_CODED => match pulsar_decode(payload) {
                Ok(dec) => {
                    if dec.len() > out_len {
                        return -1;
                    }
                    if !dec.is_empty() {
                        std::ptr::copy_nonoverlapping(dec.as_ptr(), out_ptr, dec.len());
                    }
                    dec.len() as isize
                }
                Err(_) => -1,
            },
            _ => -1,
        }
    }));
    r.unwrap_or(-1)
}
#[no_mangle]
pub extern "C" fn pulsar_version() -> *const c_uchar { VERSION.as_ptr() }
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_obj1() { let data = b"obj1 test data repeat ".repeat(1000); let enc = pulsar_encode(&data).expect("compress"); let dec = pulsar_decode(&enc).unwrap(); assert_eq!(dec, data); }
    #[test]
    fn roundtrip_64k() { let data: Vec<u8> = (0..65536).map(|i| (i % 251) as u8).collect(); if let Some(enc) = pulsar_encode(&data) { let dec = pulsar_decode(&enc).unwrap(); assert_eq!(dec, data); } }
    #[test]
    fn bwt_roundtrip() { let data = b"banana_bandana_mississippi".to_vec(); let (l,p) = crate::bwt::bwt_encode(&data); let back = crate::bwt::bwt_decode(&l,p); assert_eq!(back, data); }
    #[test]
    fn match_maker_finds() { let data = b"abcabcabcabc".repeat(100); let tokens = crate::match_maker::compress_aware(&data, data.len() as u64); assert!(tokens.len() < data.len()); }

    fn rt(src: &[u8]) {
        if let Some(enc) = pulsar_encode(src) {
            let back = pulsar_decode(&enc).expect("decode");
            assert_eq!(back, src);
            assert!(enc.len() < src.len() || src.is_empty());
        } else {
            // incompressible is allowed; PZ22 must still store without expanding past header
            let framed = zpaq_fixed::compress_fixed(src);
            let back = zpaq_fixed::decompress_fixed(&framed).expect("pz22");
            assert_eq!(back, src);
            assert!(framed.len() <= src.len() + zpaq_fixed::HEADER);
        }
    }

    #[test]
    fn zeros() { rt(&vec![0u8; 4096]); }

    #[test]
    fn repeats() { rt(&vec![0xA5; 8000]); }

    #[test]
    fn incrementing() { rt(&(0..4000).map(|i| i as u8).collect::<Vec<_>>()); }

    #[test]
    fn text_phrase() { rt(&b"the quick brown fox jumps over the lazy dog. ".repeat(80)); }

    /// FFI round-trip through pulsar_compress/pulsar_decompress, covering the
    /// stored fallback (incompressible input must not return 0).
    fn ffi_rt(src: &[u8]) {
        let mut comp = vec![0u8; src.len() + 16];
        let mut decomp = vec![0u8; src.len() + 16];
        let cn = unsafe {
            pulsar_compress(src.as_ptr(), src.len(), comp.as_mut_ptr(), comp.len())
        };
        assert!(cn > 0, "compress must return > 0, got {}", cn);
        // Incompressible input takes the stored path: 1 flag byte + raw copy.
        if crate::pulsar_encode(src).is_none() {
            assert_eq!(comp[0], PULSAR_FRAME_STORED);
            assert_eq!(cn as usize, src.len() + 1);
        }
        let dn = unsafe {
            pulsar_decompress(comp.as_ptr(), cn as usize, decomp.as_mut_ptr(), decomp.len())
        };
        assert_eq!(dn as usize, src.len(), "decompress length mismatch");
        assert_eq!(&decomp[..src.len()], src, "round-trip mismatch");
    }

    #[test]
    fn ffi_roundtrip_compressible() {
        ffi_rt(&b"the quick brown fox jumps over the lazy dog. ".repeat(200));
    }

    #[test]
    fn ffi_roundtrip_incompressible() {
        // Deterministic incompressible input (xorshift64*).
        let mut x: u64 = 0x243F6A8885A308D3;
        let src: Vec<u8> = (0..1000).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 33) as u8 }).collect();
        assert!(crate::pulsar_encode(&src).is_none(), "test input should be incompressible");
        ffi_rt(&src);
    }

    #[test]
    fn ffi_roundtrip_tiny() { ffi_rt(b"hello"); }

    #[test]
    fn ffi_bad_flag_rejected() {
        let mut out = vec![0u8; 16];
        let bad = [0x42u8, 1, 2, 3];
        let n = unsafe { pulsar_decompress(bad.as_ptr(), bad.len(), out.as_mut_ptr(), out.len()) };
        assert_eq!(n, -1);
        let n = unsafe { pulsar_decompress(bad.as_ptr(), 0, out.as_mut_ptr(), out.len()) };
        assert_eq!(n, -1);
    }
}

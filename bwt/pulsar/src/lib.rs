
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
fn verified(raw: &[u8], blob: Vec<u8>) -> Option<Vec<u8>> {
    if blob.len() >= raw.len() { return None; }
    match pulsar_decode(&blob) {
        Ok(back) if back == raw => Some(blob),
        _ => None,
    }
}

pub fn pulsar_encode(data: &[u8]) -> Option<Vec<u8>> {
    let mut cand: Vec<Vec<u8>> = Vec::new();
    // OZL2 never won Silesia/Calgary vs BW; skip the slow matcher on large files.
    if data.len() < 256 * 1024 {
        if let Some(e) = own_lz::own_lz_encode(data) {
            if let Some(ok) = verified(data, e) { cand.push(ok); }
        }
    }
    // PZ22 is a residual wrapper; skip on large files (never won vs OZL2 on Silesia/Calgary).
    if data.len() < 64 * 1024 {
        if let Some(ok) = verified(data, zpaq_fixed::compress_fixed(data)) {
            cand.push(ok);
        }
    }
    if data.len() >= 256 {
        if let Some(ok) = verified(data, bwt_ans::compress(data)) {
            cand.push(ok);
        }
    }
    cand.into_iter().min_by_key(|v| v.len())
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

/// Compress into caller buffer. Returns encoded length, 0 if incompressible, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn pulsar_compress(
    in_ptr: *const c_uchar,
    in_len: usize,
    out_ptr: *mut c_uchar,
    out_len: usize,
) -> isize {
    if in_ptr.is_null() || out_ptr.is_null() {
        return -1;
    }
    let input = std::slice::from_raw_parts(in_ptr, in_len);
    match pulsar_encode(input) {
        Some(enc) => {
            if enc.len() > out_len {
                return -1;
            }
            std::ptr::copy_nonoverlapping(enc.as_ptr(), out_ptr, enc.len());
            enc.len() as isize
        }
        None => 0,
    }
}

/// Decompress into caller buffer. Returns decoded length, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn pulsar_decompress(
    in_ptr: *const c_uchar,
    in_len: usize,
    out_ptr: *mut c_uchar,
    out_len: usize,
) -> isize {
    if in_ptr.is_null() || out_ptr.is_null() {
        return -1;
    }
    let input = std::slice::from_raw_parts(in_ptr, in_len);
    match pulsar_decode(input) {
        Ok(dec) => {
            if dec.len() > out_len {
                return -1;
            }
            std::ptr::copy_nonoverlapping(dec.as_ptr(), out_ptr, dec.len());
            dec.len() as isize
        }
        Err(_) => -1,
    }
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
}

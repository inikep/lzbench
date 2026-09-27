//! PZ22 emit-gated path on top of Drive match_maker.

use crate::match_maker;
use crate::trufold::TruFold;
use crate::zrw::{take_run, RunClass};

pub const MAGIC: &[u8; 4] = b"PZ22";
pub const VERSION_BYTE: u8 = 2;
pub const HEADER: usize = 14;
pub const BLOCK: usize = 16 * 1024;

pub fn compress_fixed(src: &[u8]) -> Vec<u8> {
    if src.is_empty() {
        let mut out = Vec::from(*MAGIC);
        out.push(VERSION_BYTE);
        out.push(0);
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(BLOCK as u32).to_le_bytes());
        return out;
    }
    let tokens = match_maker::compress_aware(src, src.len() as u64);
    let packed = match_maker::encode_tokens(&tokens);
    let fold = TruFold::new();
    let mut resid = Vec::with_capacity(packed.len());
    let mut prefix = Vec::with_capacity(packed.len());
    for &b in &packed {
        resid.push(fold.residual(&prefix, prefix.len(), b) as u8);
        prefix.push(b);
    }
    let mut framed = Vec::from(*MAGIC);
    framed.push(VERSION_BYTE);
    framed.push(1);
    framed.extend_from_slice(&(src.len() as u32).to_le_bytes());
    framed.extend_from_slice(&(BLOCK as u32).to_le_bytes());
    if resid.len() + 8 < src.len() {
        framed.push(0x01);
        framed.extend_from_slice(&(resid.len() as u32).to_le_bytes());
        framed.extend_from_slice(&resid);
    } else {
        framed.push(0x00);
        framed.extend_from_slice(&(src.len() as u32).to_le_bytes());
        framed.extend_from_slice(src);
    }
    if framed.len() > src.len() + HEADER {
        let mut raw = Vec::from(*MAGIC);
        raw.push(VERSION_BYTE);
        raw.push(0);
        raw.extend_from_slice(&(src.len() as u32).to_le_bytes());
        raw.extend_from_slice(&(BLOCK as u32).to_le_bytes());
        raw.extend_from_slice(src);
        return raw;
    }
    let _ = take_run;
    let _ = RunClass::Normal;
    framed
}

pub fn decompress_fixed(src: &[u8]) -> Result<Vec<u8>, &'static str> {
    if src.len() < HEADER || &src[0..4] != MAGIC {
        return Err("pz22 magic");
    }
    if src[4] != VERSION_BYTE {
        return Err("pz22 ver");
    }
    let mode = src[5];
    let raw_len = u32::from_le_bytes(src[6..10].try_into().unwrap()) as usize;
    if mode == 0 {
        if src.len() != HEADER + raw_len {
            return Err("pz22 raw len");
        }
        return Ok(src[HEADER..].to_vec());
    }
    if src.len() < HEADER + 5 {
        return Err("pz22 short");
    }
    let flag = src[HEADER];
    let n = u32::from_le_bytes(src[HEADER + 1..HEADER + 5].try_into().unwrap()) as usize;
    let body = &src[HEADER + 5..];
    if body.len() < n {
        return Err("pz22 body");
    }
    if flag == 0x00 {
        return Ok(body[..n].to_vec());
    }
    let resid = &body[..n];
    let fold = TruFold::new();
    let mut packed = Vec::with_capacity(resid.len());
    for &r in resid {
        packed.push(fold.reconstruct_one(&packed, r as i8));
    }
    match_maker::decode_tokens(&packed, raw_len)
}

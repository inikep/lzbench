
use crate::match_maker;
use crate::lz_opt;
use crate::ans;
use crate::lz_full;
pub const OZL2_MAGIC: &[u8;4]=b"OZL2";
pub fn own_lz_encode(data: &[u8]) -> Option<Vec<u8>> {
    if data.len()<1024 { return None; }
    let tokens = match_maker::compress_aware(data, data.len() as u64);
    if tokens.is_empty() { return None; }
    let lz_bytes = lz_full::mm_tokens_to_bytes(&tokens);
    if lz_bytes.len()*2 >= data.len()*3 { return None; }
    let split = lz_opt::encode_dlz2(&lz_bytes, data.len()).ok()?;
    let mixed = ans::rans_encode(&lz_bytes);
    let best_payload = if split.len() < mixed.len() { split } else {
        let mut out=Vec::new();
        out.extend_from_slice(b"DICT");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&mixed);
        out
    };
    let mut out=Vec::with_capacity(8+best_payload.len());
    out.extend_from_slice(OZL2_MAGIC);
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&best_payload);
    if out.len() >= data.len() { return None; }
    Some(out)
}
pub fn own_lz_decode(buf: &[u8]) -> Result<Vec<u8>, &'static str> {
    if buf.len()<8 { return Err("ozl: short"); }
    if &buf[..4]!=OZL2_MAGIC { return Err("ozl: magic"); }
    let orig=u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let payload=&buf[8..];
    let lz_bytes = if payload.starts_with(b"DLZ2") {
        let (tokens, _orig) = lz_opt::decode_dlz2(payload)?;
        tokens
    } else if payload.starts_with(b"DICT") {
        if payload.len()<8 { return Err("dict short"); }
        let rans_blob=&payload[8..];
        ans::rans_decode(rans_blob)?
    } else {
        ans::rans_decode(payload)?
    };
    let mm_tokens = crate::lz_full::bytes_to_mm_tokens(&lz_bytes);
    let mut out=Vec::with_capacity(orig);
    for t in mm_tokens {
        match t {
            match_maker::Token::Lit(b) => out.push(b),
            match_maker::Token::Match{dist, len} => {
                if dist==0 || dist>out.len() { return Err("dist"); }
                if dist>=len { let src=out.len()-dist; out.extend_from_within(src..src+len); }
                else { let mut rem=len; while rem>0 { let src=out.len()-dist; let chunk=rem.min(dist); out.extend_from_within(src..src+chunk); rem-=chunk; } }
            }
        }
        if out.len()>=orig { break; }
    }
    if out.len()!=orig { return Err("len"); }
    Ok(out)
}

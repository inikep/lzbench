
use crate::ans_fast::{rans_decode_fast, fast_lz_copy};
use crate::own_lz::OZL2_MAGIC;
// lz_opt decode lives in own_lz; this path rebuilds DLZ2 streams directly.
use crate::ans;
pub fn own_lz_decode_fast(buf: &[u8]) -> Result<Vec<u8>, &'static str> {
    if buf.len()<8 { return Err("ozl: short"); }
    if &buf[..4]!=OZL2_MAGIC { return crate::own_lz::own_lz_decode(buf); }
    let orig=u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let payload=&buf[8..];
    let lz_bytes = if payload.starts_with(b"DLZ2") {
        if payload.len()<8 { return Err("dlz2 short"); }
        let mut pos=8usize;
        let take_blob = |buf: &[u8], pos: &mut usize| -> Result<Vec<u8>, &'static str> {
            if *pos+4>buf.len() { return Err("len"); }
            let n=u32::from_le_bytes(buf[*pos..*pos+4].try_into().unwrap()) as usize;
            *pos+=4;
            if *pos+n>buf.len() { return Err("blob"); }
            let v=buf[*pos..*pos+n].to_vec();
            *pos+=n;
            Ok(v)
        };
        let flags_blob=take_blob(payload, &mut pos)?;
        let lits_blob=take_blob(payload, &mut pos)?;
        let lens_blob=take_blob(payload, &mut pos)?;
        let dists_blob=take_blob(payload, &mut pos)?;
        let flags = rans_decode_fast(&flags_blob).or_else(|_| ans::rans_decode(&flags_blob))?;
        let lits = rans_decode_fast(&lits_blob).or_else(|_| ans::rans_decode(&lits_blob))?;
        let lens = rans_decode_fast(&lens_blob).or_else(|_| ans::rans_decode(&lens_blob))?;
        let dists = rans_decode_fast(&dists_blob).or_else(|_| ans::rans_decode(&dists_blob))?;
        let mut out=Vec::new();
        let mut li=0; let mut ni=0; let mut di=0;
        for &f in &flags {
            if f==0 { if li>=lits.len() { return Err("lit underrun"); } out.push(0x00); out.push(lits[li]); li+=1; }
            else { if ni>=lens.len() || di+3>dists.len() { return Err("match underrun"); } let off = dists[di] as usize | ((dists[di+1] as usize)<<8) | ((dists[di+2] as usize)<<16); di+=3; let len=lens[ni]; ni+=1; if off<=0xffff { out.push(0x01); out.extend_from_slice(&(off as u16).to_le_bytes()); out.push(len); } else { out.push(0x02); out.push((off & 0xFF) as u8); out.push(((off>>8)&0xFF) as u8); out.push(((off>>16)&0xFF) as u8); out.push(len); } }
        }
        out
    } else if payload.starts_with(b"DICT") {
        let rans_blob=&payload[8..];
        rans_decode_fast(rans_blob).or_else(|_| ans::rans_decode(rans_blob))?
    } else {
        rans_decode_fast(payload).or_else(|_| ans::rans_decode(payload))?
    };
    let mut out=Vec::with_capacity(orig);
    let mut i=0;
    while i<lz_bytes.len() && out.len()<orig {
        match lz_bytes[i] {
            0x00 => { if i+1>=lz_bytes.len() { break; } out.push(lz_bytes[i+1]); i+=2; }
            0x01 => { if i+3>=lz_bytes.len() { break; } let dist=u16::from_le_bytes([lz_bytes[i+1], lz_bytes[i+2]]) as usize; let len=lz_bytes[i+3] as usize; fast_lz_copy(&mut out, dist, len)?; i+=4; }
            0x02 => { if i+4>=lz_bytes.len() { break; } let dist=lz_bytes[i+1] as usize | ((lz_bytes[i+2] as usize)<<8) | ((lz_bytes[i+3] as usize)<<16); let len=lz_bytes[i+4] as usize; fast_lz_copy(&mut out, dist, len)?; i+=5; }
            _ => break,
        }
    }
    if out.len()!=orig { return Err("len"); }
    Ok(out)
}


use crate::ans;
pub const MAGIC2: &[u8; 4] = b"DLZ2";
fn split_tokens(tokens: &[u8]) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>), &'static str> {
    let mut flags = Vec::new(); let mut lits = Vec::new(); let mut lens = Vec::new(); let mut dists = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            0x00 => { if i+1 >= tokens.len() { return Err("dlz2: short lit"); } flags.push(0); lits.push(tokens[i+1]); i+=2; }
            0x01 => { if i+4 > tokens.len() { return Err("dlz2: short m16"); } flags.push(1); let off = u16::from_le_bytes([tokens[i+1], tokens[i+2]]) as u32; lens.push(tokens[i+3]); dists.extend_from_slice(&off.to_le_bytes()[..3]); i+=4; }
            0x02 => { if i+5 > tokens.len() { return Err("dlz2: short m24"); } flags.push(1); let off = tokens[i+1] as u32 | ((tokens[i+2] as u32)<<8) | ((tokens[i+3] as u32)<<16); lens.push(tokens[i+4]); dists.extend_from_slice(&off.to_le_bytes()[..3]); i+=5; }
            _ => return Err("dlz2: bad token"),
        }
    }
    Ok((flags, lits, lens, dists))
}
fn join_tokens(flags: &[u8], lits: &[u8], lens: &[u8], dists: &[u8]) -> Result<Vec<u8>, &'static str> {
    let mut out = Vec::new(); let mut li=0; let mut ni=0; let mut di=0;
    for &f in flags {
        if f==0 { if li>=lits.len() { return Err("lit underrun"); } out.push(0x00); out.push(lits[li]); li+=1; }
        else { if ni>=lens.len() || di+3>dists.len() { return Err("match underrun"); } let off = dists[di] as usize | ((dists[di+1] as usize)<<8) | ((dists[di+2] as usize)<<16); di+=3; let len=lens[ni]; ni+=1; if off<=0xffff { out.push(0x01); out.extend_from_slice(&(off as u16).to_le_bytes()); out.push(len); } else { out.push(0x02); out.push((off & 0xFF) as u8); out.push(((off>>8)&0xFF) as u8); out.push(((off>>16)&0xFF) as u8); out.push(len); } }
    }
    Ok(out)
}
fn put_blob(out: &mut Vec<u8>, blob: &[u8]) { out.extend_from_slice(&(blob.len() as u32).to_le_bytes()); out.extend_from_slice(blob); }
fn take_blob(buf: &[u8], pos: &mut usize) -> Result<Vec<u8>, &'static str> { if *pos+4>buf.len() { return Err("short len"); } let n=u32::from_le_bytes(buf[*pos..*pos+4].try_into().unwrap()) as usize; *pos+=4; if *pos+n>buf.len() { return Err("short blob"); } let v=buf[*pos..*pos+n].to_vec(); *pos+=n; Ok(v) }
pub fn encode_dlz2(tokens: &[u8], orig_len: usize) -> Result<Vec<u8>, &'static str> { let (flags, lits, lens, dists) = split_tokens(tokens)?; let mut out=Vec::new(); out.extend_from_slice(MAGIC2); out.extend_from_slice(&(orig_len as u32).to_le_bytes()); put_blob(&mut out, &ans::rans_encode(&flags)); put_blob(&mut out, &ans::rans_encode(&lits)); put_blob(&mut out, &ans::rans_encode(&lens)); put_blob(&mut out, &ans::rans_encode(&dists)); Ok(out) }
pub fn decode_dlz2(buf: &[u8]) -> Result<(Vec<u8>, usize), &'static str> { if buf.len()<8 || &buf[..4]!=MAGIC2 { return Err("magic"); } let orig_len=u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize; let mut pos=8; let flags = ans::rans_decode(&take_blob(buf, &mut pos)?)?; let lits = ans::rans_decode(&take_blob(buf, &mut pos)?)?; let lens = ans::rans_decode(&take_blob(buf, &mut pos)?)?; let dists = ans::rans_decode(&take_blob(buf, &mut pos)?)?; let tokens = join_tokens(&flags, &lits, &lens, &dists)?; Ok((tokens, orig_len)) }

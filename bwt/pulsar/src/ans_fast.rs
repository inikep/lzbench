
use crate::ans::{SCALE, SCALE_BITS, RANS_L, MAGIC};
fn build_sym_table(freq: &[u32; 256], start: &[u32; 256]) -> Vec<u8> {
    let mut table = vec![0u8; SCALE as usize];
    for s in 0..256 { let f=freq[s] as usize; if f==0 { continue; } let st=start[s] as usize; for i in 0..f { table[st+i]=s as u8; } }
    table
}
pub fn rans_decode_fast(buf: &[u8]) -> Result<Vec<u8>, &'static str> {
    if buf.len()<11 { return Err("ans: truncated"); }
    if &buf[..4]!=MAGIC { return crate::ans::rans_decode(buf); }
    let orig_len=u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let scale_bits=buf[8] as u32;
    if scale_bits!=SCALE_BITS { return Err("ans: scale"); }
    let n_used=u16::from_le_bytes(buf[9..11].try_into().unwrap()) as usize;
    let mut pos=11usize;
    let mut freq=[0u32;256];
    for _ in 0..n_used { if pos+3>buf.len() { return Err("ans: freq table"); } let s=buf[pos] as usize; let f=u16::from_le_bytes(buf[pos+1..pos+3].try_into().unwrap()) as u32; freq[s]=f; pos+=3; }
    if pos+4>buf.len() { return Err("ans: stream len"); }
    let slen=u32::from_le_bytes(buf[pos..pos+4].try_into().unwrap()) as usize;
    pos+=4;
    if pos+slen>buf.len() { return Err("ans: stream"); }
    let stream=&buf[pos..pos+slen];
    if slen<8 { return Err("ans: no state"); }
    let mut state=u64::from_le_bytes(stream[slen-8..].try_into().unwrap());
    let mut cursor=slen-8;
    let mut start=[0u32;256]; let mut run=0u32;
    for s in 0..256 { start[s]=run; run+=freq[s]; }
    if run!=SCALE { return Err("ans: freq sum"); }
    let sym_table=build_sym_table(&freq, &start);
    let mask=(SCALE-1) as u64;
    let mut out=vec![0u8; orig_len];
    for i in 0..orig_len {
        let slot=(state & mask) as usize;
        let sym=sym_table[slot];
        out[i]=sym;
        let f=freq[sym as usize] as u64;
        let c=start[sym as usize] as u64;
        state=f*(state>>SCALE_BITS)+(state & mask)-c;
        while state < RANS_L { if cursor==0 { return Err("ans: underrun"); } cursor-=1; state=(state<<8)|stream[cursor] as u64; }
    }
    Ok(out)
}
pub fn fast_lz_copy(out: &mut Vec<u8>, dist: usize, len: usize) -> Result<(), &'static str> {
    if dist==0 || dist>out.len() { return Err("dist"); }
    if dist>=len { let src=out.len()-dist; out.extend_from_within(src..src+len); }
    else {
        let mut rem=len;
        while rem>0 { let src=out.len()-dist; let chunk=rem.min(dist); out.extend_from_within(src..src+chunk); rem-=chunk; }
    }
    Ok(())
}

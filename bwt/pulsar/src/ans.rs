
pub const MAGIC: &[u8; 4] = b"AN0\0";
pub const RANS_L: u64 = 1 << 23;
pub const SCALE_BITS: u32 = 12;
pub const SCALE: u32 = 1 << SCALE_BITS;
pub struct RansEnc { pub state: u64 }
impl RansEnc {
    pub fn new() -> Self { Self { state: RANS_L } }
    pub fn encode(&mut self, freq: u32, cum: u32, scale: u32, out: &mut Vec<u8>) {
        let freq = freq.max(1);
        let x_max = ((RANS_L / scale as u64) << 8) * freq as u64;
        while self.state >= x_max { out.push((self.state & 0xff) as u8); self.state >>= 8; }
        self.state = (self.state / freq as u64) * scale as u64 + (self.state % freq as u64) + cum as u64;
    }
    pub fn flush(&self, out: &mut Vec<u8>) { out.extend_from_slice(&self.state.to_le_bytes()); }
}
fn normalize(counts: &[u32; 256]) -> ([u32; 256], [u32; 256]) {
    let total: u32 = counts.iter().sum();
    let mut freq = [0u32; 256];
    if total == 0 { freq[0]=SCALE; return (freq, [0u32;256]); }
    let mut assigned = 0u32;
    for (s,&c) in counts.iter().enumerate() {
        if c==0 { continue; }
        let f = ((c as u64 * SCALE as u64) / total as u64).max(1) as u32;
        freq[s]=f; assigned+=f;
    }
    if assigned!=SCALE {
        let mut big=0usize;
        for s in 0..256 { if freq[s]>freq[big] { big=s; } }
        if assigned < SCALE { freq[big]+=SCALE-assigned; }
        else {
            let extra=assigned-SCALE;
            if freq[big]>extra+1 { freq[big]-=extra; }
            else {
                let mut left=extra;
                for s in 0..256 { if left==0 { break; } if freq[s]>1 { let take=(freq[s]-1).min(left); freq[s]-=take; left-=take; } }
            }
        }
    }
    let mut start=[0u32;256]; let mut run=0u32;
    for s in 0..256 { start[s]=run; run+=freq[s]; }
    (freq,start)
}
pub fn rans_encode(data: &[u8]) -> Vec<u8> {
    let mut counts=[0u32;256];
    for &b in data { counts[b as usize]+=1; }
    let (freq,start)=normalize(&counts);
    let mut enc=RansEnc::new();
    let mut stream=Vec::with_capacity(data.len()/2+16);
    for &b in data.iter().rev() { enc.encode(freq[b as usize], start[b as usize], SCALE, &mut stream); }
    enc.flush(&mut stream);
    let used: Vec<(u8,u16)> = (0..256).filter(|&s| freq[s]>0).map(|s| (s as u8, freq[s] as u16)).collect();
    let mut out=Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.push(SCALE_BITS as u8);
    out.extend_from_slice(&(used.len() as u16).to_le_bytes());
    for (s,f) in used { out.push(s); out.extend_from_slice(&f.to_le_bytes()); }
    out.extend_from_slice(&(stream.len() as u32).to_le_bytes());
    out.extend_from_slice(&stream);
    out
}
fn find_sym_slow(slot: u32, freq: &[u32;256], start: &[u32;256]) -> u8 {
    for s in 0..256 { if freq[s]==0 { continue; } if slot>=start[s] && slot<start[s]+freq[s] { return s as u8; } } 0
}
pub fn rans_decode(buf: &[u8]) -> Result<Vec<u8>, &'static str> {
    if buf.len()<4+4+1+2+4+8 { return Err("ans: truncated"); }
    if &buf[..4]!=MAGIC { return Err("ans: bad magic"); }
    let orig_len=u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let scale_bits=buf[8] as u32;
    if scale_bits!=SCALE_BITS { return Err("ans: scale"); }
    let n_used=u16::from_le_bytes(buf[9..11].try_into().unwrap()) as usize;
    let mut pos=11usize;
    let mut freq=[0u32;256];
    for _ in 0..n_used {
        if pos+3>buf.len() { return Err("ans: freq table"); }
        let s=buf[pos] as usize;
        let f=u16::from_le_bytes(buf[pos+1..pos+3].try_into().unwrap()) as u32;
        freq[s]=f; pos+=3;
    }
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
    let mask=(SCALE-1) as u64;
    let mut out=vec![0u8; orig_len];
    for i in 0..orig_len {
        let slot=(state & mask) as u32;
        let s=find_sym_slow(slot, &freq, &start);
        out[i]=s;
        let f=freq[s as usize] as u64;
        let c=start[s as usize] as u64;
        state=f*(state>>SCALE_BITS)+(state & mask)-c;
        while state < RANS_L {
            if cursor==0 { return Err("ans: underrun"); }
            cursor-=1;
            state=(state<<8)|stream[cursor] as u64;
        }
    }
    Ok(out)
}

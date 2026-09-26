//! BW23: RLE-1 + BWT + MTF + Wheeler RLE-0 + 4-ctx rANS.
//! Contexts: RUNA / RUNB / small-MTF / rest. Adaptive block size.
//! BW22 (order-0) still decodes.

use crate::bwt;

pub const MAGIC: &[u8; 4] = b"BW23";
pub const MAGIC22: &[u8; 4] = b"BW22";
pub const BLOCK: usize = 900 * 1024;
const NCTX: usize = 4;

fn block_size_for(n: usize) -> usize {
    if n >= 8 * 1024 * 1024 {
        2 * 1024 * 1024
    } else {
        BLOCK
    }
}

fn tok_ctx(t: u16) -> usize {
    match t {
        0 => 0,
        1 => 1,
        2..=16 => 2,
        _ => 3,
    }
}

const RANS_L: u64 = 1 << 23;
const SCALE_BITS: u32 = 12; // nominal, same as crate::ans
const SCALE: u32 = 1 << SCALE_BITS;
const NSYM: usize = 257; // RUNA=0, RUNB=1, MTF 1..=255 -> 2..=256

fn mtf_encode(data: &[u8]) -> Vec<u8> {
    let mut list: Vec<u8> = (0..=255).collect();
    let mut out = Vec::with_capacity(data.len());
    for &b in data {
        let mut i = 0usize;
        while list[i] != b { i += 1; }
        out.push(i as u8);
        if i != 0 {
            let v = list.remove(i);
            list.insert(0, v);
        }
    }
    out
}

fn mtf_decode(data: &[u8]) -> Vec<u8> {
    let mut list: Vec<u8> = (0..=255).collect();
    let mut out = Vec::with_capacity(data.len());
    for &idx in data {
        let i = idx as usize;
        let v = list[i];
        out.push(v);
        if i != 0 { list.remove(i); list.insert(0, v); }
    }
    out
}

fn rle1_encode(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len());
    let mut i = 0;
    while i < src.len() {
        let c = src[i];
        let mut n = 1usize;
        while i + n < src.len() && src[i + n] == c && n < 259 { n += 1; }
        let emit = n.min(4);
        for _ in 0..emit { out.push(c); }
        if n >= 4 { out.push((n - 4) as u8); }
        i += n;
    }
    out
}

fn rle1_decode(src: &[u8]) -> Result<Vec<u8>, &'static str> {
    let mut out = Vec::with_capacity(src.len());
    let mut i = 0;
    let mut run_sym: Option<u8> = None;
    let mut run_len = 0usize;
    while i < src.len() {
        if run_len == 4 {
            if i >= src.len() { return Err("rle1 truncated count"); }
            out.extend(std::iter::repeat(run_sym.unwrap()).take(src[i] as usize));
            i += 1; run_sym = None; run_len = 0; continue;
        }
        let c = src[i]; out.push(c); i += 1;
        if run_sym == Some(c) { run_len += 1; } else { run_sym = Some(c); run_len = 1; }
    }
    if run_len == 4 { return Err("rle1 missing count"); }
    Ok(out)
}

fn wheeler_encode(mtf: &[u8]) -> Vec<u16> {
    let mut out = Vec::with_capacity(mtf.len());
    let mut i = 0;
    while i < mtf.len() {
        if mtf[i] == 0 {
            let mut run = 0usize;
            while i < mtf.len() && mtf[i] == 0 { run += 1; i += 1; }
            let mut r = run;
            while r > 0 {
                out.push(if r & 1 == 1 { 0 } else { 1 });
                r = (r - 1) >> 1;
            }
        } else { out.push(mtf[i] as u16 + 1); i += 1; }
    }
    out
}

fn wheeler_decode(tok: &[u16], expect: usize) -> Result<Vec<u8>, &'static str> {
    let mut out = Vec::with_capacity(expect);
    let mut i = 0;
    while i < tok.len() {
        if tok[i] <= 1 {
            let mut n = 0usize; let mut p = 1usize;
            while i < tok.len() && tok[i] <= 1 {
                n += if tok[i] == 0 { p } else { p * 2 };
                if p > expect.saturating_mul(2) + 4 { return Err("wheeler overflow"); }
                p <<= 1; i += 1;
            }
            out.extend(std::iter::repeat(0u8).take(n));
        } else {
            let v = tok[i] - 1;
            if v > 255 { return Err("wheeler sym"); }
            out.push(v as u8); i += 1;
        }
    }
    Ok(out)
}

fn normalize(counts: &[u32; NSYM]) -> ([u32; NSYM], [u32; NSYM]) {
    let total: u32 = counts.iter().sum();
    let mut freq = [0u32; NSYM];
    if total == 0 { freq[0] = SCALE; return (freq, [0u32; NSYM]); }
    let mut assigned = 0u32;
    for (s, &c) in counts.iter().enumerate() {
        if c == 0 { continue; }
        let f = ((c as u64 * SCALE as u64) / total as u64).max(1) as u32;
        freq[s] = f; assigned += f;
    }
    if assigned != SCALE {
        let mut big = 0usize;
        for s in 0..NSYM { if freq[s] > freq[big] { big = s; } }
        if assigned < SCALE { freq[big] += SCALE - assigned; }
        else {
            let extra = assigned - SCALE;
            if freq[big] > extra + 1 { freq[big] -= extra; }
            else {
                let mut left = extra;
                for s in 0..NSYM {
                    if left == 0 { break; }
                    if freq[s] > 1 {
                        let take = (freq[s] - 1).min(left);
                        freq[s] -= take; left -= take;
                    }
                }
            }
        }
    }
    let mut start = [0u32; NSYM];
    let mut run = 0u32;
    for s in 0..NSYM { start[s] = run; run += freq[s]; }
    (freq, start)
}

struct RansEnc { state: u64 }
impl RansEnc {
    fn new() -> Self { Self { state: RANS_L } }
    fn encode(&mut self, freq: u32, cum: u32, scale: u32, out: &mut Vec<u8>) {
        let freq = freq.max(1);
        let x_max = ((RANS_L / scale as u64) << 8) * freq as u64;
        while self.state >= x_max { out.push((self.state & 0xff) as u8); self.state >>= 8; }
        self.state = (self.state / freq as u64) * scale as u64 + (self.state % freq as u64) + cum as u64;
    }
    fn flush(&self, out: &mut Vec<u8>) { out.extend_from_slice(&self.state.to_le_bytes()); }
}

#[allow(dead_code)]
fn rans0_encode(data: &[u16]) -> Vec<u8> {
    let mut counts = [0u32; NSYM];
    for &b in data {
        let s = b as usize;
        if s < NSYM { counts[s] += 1; }
    }
    let (freq, start) = normalize(&counts);
    let mut enc = RansEnc::new();
    let mut stream = Vec::with_capacity(data.len() / 2 + 16);
    for &b in data.iter().rev() {
        let s = b as usize;
        enc.encode(freq[s], start[s], SCALE, &mut stream);
    }
    enc.flush(&mut stream);
    let used: Vec<(u16, u16)> = (0..NSYM).filter(|&s| freq[s] > 0).map(|s| (s as u16, freq[s] as u16)).collect();
    let mut out = Vec::new();
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.push(SCALE_BITS as u8);
    out.extend_from_slice(&(used.len() as u16).to_le_bytes());
    for (s, f) in used {
        out.extend_from_slice(&s.to_le_bytes());
        out.extend_from_slice(&f.to_le_bytes());
    }
    out.extend_from_slice(&(stream.len() as u32).to_le_bytes());
    out.extend_from_slice(&stream);
    out
}

fn find_sym(slot: u32, freq: &[u32; NSYM], start: &[u32; NSYM]) -> u16 {
    for s in 0..NSYM {
        if freq[s] == 0 { continue; }
        if slot >= start[s] && slot < start[s] + freq[s] { return s as u16; }
    }
    0
}

fn rans0_decode(buf: &[u8]) -> Result<Vec<u16>, &'static str> {
    if buf.len() < 4 + 1 + 2 + 4 + 8 { return Err("ans0 short"); }
    let orig_len = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    if buf[4] as u32 != SCALE_BITS { return Err("ans0 scale"); }
    let nused = u16::from_le_bytes(buf[5..7].try_into().unwrap()) as usize;
    let mut pos = 7usize;
    let mut freq = [0u32; NSYM];
    for _ in 0..nused {
        if pos + 4 > buf.len() { return Err("ans0 tbl"); }
        let s = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap()) as usize;
        let f = u16::from_le_bytes(buf[pos + 2..pos + 4].try_into().unwrap()) as u32;
        if s >= NSYM { return Err("ans0 sym"); }
        freq[s] = f; pos += 4;
    }
    let mut start = [0u32; NSYM];
    let mut run = 0u32;
    for s in 0..NSYM { start[s] = run; run += freq[s]; }
    if run != SCALE { return Err("ans0 sum"); }
    if pos + 4 > buf.len() { return Err("ans0 slen"); }
    let slen = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    if pos + slen > buf.len() { return Err("ans0 stream"); }
    let stream = &buf[pos..pos + slen];
    if slen < 8 { return Err("ans0 state"); }
    let mut state = u64::from_le_bytes(stream[slen - 8..].try_into().unwrap());
    let mut cursor = slen - 8;
    let mask = (SCALE - 1) as u64;
    let mut out = vec![0u16; orig_len];
    for i in 0..orig_len {
        let slot = (state & mask) as u32;
        let s = find_sym(slot, &freq, &start);
        out[i] = s;
        let f = freq[s as usize] as u64;
        let cum = start[s as usize] as u64;
        state = f * (state >> SCALE_BITS) + (state & mask) - cum;
        while state < RANS_L {
            if cursor == 0 { return Err("ans0 underrun"); }
            cursor -= 1;
            state = (state << 8) | stream[cursor] as u64;
        }
    }
    Ok(out)
}

fn rans4_encode(data: &[u16]) -> Vec<u8> {
    let mut counts = [[0u32; NSYM]; NCTX];
    let mut ctxs = vec![0usize; data.len()];
    let mut ctx = 0usize;
    for (i, &b) in data.iter().enumerate() {
        ctxs[i] = ctx;
        let s = b as usize;
        if s < NSYM {
            counts[ctx][s] += 1;
        }
        ctx = tok_ctx(b);
    }
    let mut tables = [( [0u32; NSYM], [0u32; NSYM] ); NCTX];
    for c in 0..NCTX {
        tables[c] = normalize(&counts[c]);
    }
    let mut enc = RansEnc::new();
    let mut stream = Vec::with_capacity(data.len() / 2 + 16);
    for i in (0..data.len()).rev() {
        let s = data[i] as usize;
        let c = ctxs[i];
        enc.encode(tables[c].0[s], tables[c].1[s], SCALE, &mut stream);
    }
    enc.flush(&mut stream);
    let mut out = Vec::new();
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.push(SCALE_BITS as u8);
    out.push(NCTX as u8);
    for c in 0..NCTX {
        let freq = &tables[c].0;
        let used: Vec<(u16, u16)> = (0..NSYM)
            .filter(|&s| freq[s] > 0)
            .map(|s| (s as u16, freq[s] as u16))
            .collect();
        out.extend_from_slice(&(used.len() as u16).to_le_bytes());
        for (s, f) in used {
            out.extend_from_slice(&s.to_le_bytes());
            out.extend_from_slice(&f.to_le_bytes());
        }
    }
    out.extend_from_slice(&(stream.len() as u32).to_le_bytes());
    out.extend_from_slice(&stream);
    out
}

fn rans4_decode(buf: &[u8]) -> Result<Vec<u16>, &'static str> {
    if buf.len() < 4 + 1 + 1 + 4 + 8 {
        return Err("ans4 short");
    }
    let orig_len = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    if buf[4] as u32 != SCALE_BITS {
        return Err("ans4 scale");
    }
    if buf[5] as usize != NCTX {
        return Err("ans4 nctx");
    }
    let mut pos = 6usize;
    let mut freq = [[0u32; NSYM]; NCTX];
    let mut start = [[0u32; NSYM]; NCTX];
    for c in 0..NCTX {
        if pos + 2 > buf.len() {
            return Err("ans4 tbl");
        }
        let nused = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        for _ in 0..nused {
            if pos + 4 > buf.len() {
                return Err("ans4 row");
            }
            let s = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap()) as usize;
            let f = u16::from_le_bytes(buf[pos + 2..pos + 4].try_into().unwrap()) as u32;
            if s >= NSYM {
                return Err("ans4 sym");
            }
            freq[c][s] = f;
            pos += 4;
        }
        let mut run = 0u32;
        for s in 0..NSYM {
            start[c][s] = run;
            run += freq[c][s];
        }
        if run != SCALE {
            return Err("ans4 sum");
        }
    }
    if pos + 4 > buf.len() {
        return Err("ans4 slen");
    }
    let slen = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    if pos + slen > buf.len() {
        return Err("ans4 stream");
    }
    let stream = &buf[pos..pos + slen];
    if slen < 8 {
        return Err("ans4 state");
    }
    let mut state = u64::from_le_bytes(stream[slen - 8..].try_into().unwrap());
    let mut cursor = slen - 8;
    let mask = (SCALE - 1) as u64;
    let mut out = vec![0u16; orig_len];
    let mut ctx = 0usize;
    for i in 0..orig_len {
        let slot = (state & mask) as u32;
        let s = find_sym(slot, &freq[ctx], &start[ctx]);
        out[i] = s;
        let f = freq[ctx][s as usize] as u64;
        let cum = start[ctx][s as usize] as u64;
        state = f * (state >> SCALE_BITS) + (state & mask) - cum;
        while state < RANS_L {
            if cursor == 0 {
                return Err("ans4 underrun");
            }
            cursor -= 1;
            state = (state << 8) | stream[cursor] as u64;
        }
        ctx = tok_ctx(s);
    }
    Ok(out)
}

fn encode_block(block: &[u8]) -> Vec<u8> {
    let pre = rle1_encode(block);
    let (l, primary) = bwt::bwt_encode(&pre);
    let mtf = mtf_encode(&l);
    let wh = wheeler_encode(&mtf);
    let coded = rans4_encode(&wh);
    let mut out = Vec::with_capacity(16 + coded.len());
    out.extend_from_slice(&(block.len() as u32).to_le_bytes());
    out.extend_from_slice(&(pre.len() as u32).to_le_bytes());
    out.extend_from_slice(&(primary as u32).to_le_bytes());
    out.extend_from_slice(&(coded.len() as u32).to_le_bytes());
    out.extend_from_slice(&coded);
    out
}

fn decode_block(buf: &[u8], pos: &mut usize, use4: bool) -> Result<Vec<u8>, &'static str> {
    if *pos + 16 > buf.len() { return Err("bw22 blk hdr"); }
    let raw_len = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap()) as usize;
    let pre_len = u32::from_le_bytes(buf[*pos + 4..*pos + 8].try_into().unwrap()) as usize;
    let primary = u32::from_le_bytes(buf[*pos + 8..*pos + 12].try_into().unwrap()) as usize;
    let clen = u32::from_le_bytes(buf[*pos + 12..*pos + 16].try_into().unwrap()) as usize;
    *pos += 16;
    if *pos + clen > buf.len() { return Err("bw22 blk body"); }
    let coded = &buf[*pos..*pos + clen];
    *pos += clen;
    let wh = if use4 {
        rans4_decode(coded)?
    } else {
        rans0_decode(coded)?
    };
    let mtf = wheeler_decode(&wh, pre_len)?;
    if mtf.len() != pre_len { return Err("bw22 mtf len"); }
    let l = mtf_decode(&mtf);
    let pre = bwt::bwt_decode(&l, primary);
    if pre.len() != pre_len { return Err("bw22 bwt len"); }
    let out = rle1_decode(&pre)?;
    if out.len() != raw_len { return Err("bw22 raw len"); }
    Ok(out)
}

pub fn compress(src: &[u8]) -> Vec<u8> {
    let blk = block_size_for(src.len());
    let mut out = Vec::from(*MAGIC);
    out.extend_from_slice(&(src.len() as u32).to_le_bytes());
    out.extend_from_slice(&(blk as u32).to_le_bytes());
    if src.is_empty() { return out; }
    for chunk in src.chunks(blk) {
        out.extend_from_slice(&encode_block(chunk));
    }
    out
}

pub fn decompress(src: &[u8]) -> Result<Vec<u8>, &'static str> {
    if src.len() < 12 {
        return Err("bw magic");
    }
    if &src[..4] != MAGIC && &src[..4] != MAGIC22 {
        return Err("bw22 magic");
    }
    let raw_len = u32::from_le_bytes(src[4..8].try_into().unwrap()) as usize;
    let use4 = &src[..4] == MAGIC;
    let mut pos = 12usize;
    let mut out = Vec::with_capacity(raw_len);
    while out.len() < raw_len {
        out.extend_from_slice(&decode_block(src, &mut pos, use4)?);
    }
    if out.len() != raw_len { return Err("bw22 total"); }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rt(src: &[u8]) { let enc = compress(src); assert_eq!(decompress(&enc).expect("dec"), src); }
    #[test] fn banana() { rt(b"banana_bandana_mississippi"); }
    #[test] fn zeros() { rt(&vec![0u8; 2000]); }
    #[test] fn text() { rt(&b"the quick brown fox jumps over the lazy dog. ".repeat(40)); }
    #[test] fn periodic() { rt(&b"abababab".repeat(80)); }
    #[test] fn rle1_loop() {
        let mut d = vec![7u8; 300]; d.extend_from_slice(&[1,2,3,3,3,3,3,9]);
        assert_eq!(rle1_decode(&rle1_encode(&d)).unwrap(), d);
    }
    #[test] fn wheeler_loop() {
        let d: Vec<u8> = (0..800).map(|i| if i % 5 == 0 { 0 } else { (i % 9) as u8 }).collect();
        assert_eq!(wheeler_decode(&wheeler_encode(&d), d.len()).unwrap(), d);
    }
    #[test] fn rans0_loop() {
        let d: Vec<u16> = (0..4000).map(|i| if i % 3 == 0 { 0 } else { (i % 17) as u16 }).collect();
        assert_eq!(rans0_decode(&rans0_encode(&d)).unwrap(), d);
    }
    #[test] fn rans4_loop() {
        let d: Vec<u16> = (0..4000).map(|i| if i % 3 == 0 { 0 } else { (i % 17) as u16 }).collect();
        assert_eq!(rans4_decode(&rans4_encode(&d)).unwrap(), d);
    }
}

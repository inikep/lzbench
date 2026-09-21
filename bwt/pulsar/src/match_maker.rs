//! 1.19.1 — zstd-style aware match finder.
//! Row bucket + lazy/lazy2 + text/entropy gates.
//! Not in pick_best until it beats obj1 10,322 / osdb 27,445.

use crate::text_detect::{is_text, text_ratio};

const MAX_MATCH: usize = 131_072;
const ROW_LOG_MAX: usize = 6;
const WINDOW_LOG: usize = 21;
const HASH_PRIME: u32 = 0x1E35_A7BD;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Fast,
    DFast,
    Greedy,
    Lazy,
    Lazy2,
    BTLazy2,
    BTOPT,
}

#[derive(Clone, Copy, Debug)]
pub struct Match {
    pub dist: usize,
    pub len: usize,
}

#[derive(Clone, Copy)]
pub struct Params {
    pub hash_log: usize,
    pub chain_log: usize,
    pub search_log: usize,
    pub min_match: usize,
    pub strategy: Strategy,
    pub row_log: usize,
}

impl Params {
    pub fn aware(data: &[u8], file_len: u64) -> Self {
        let ratio = text_ratio(data);
        let entropy = estimate_entropy(data);
        let is_txt = is_text(data);
        let mut base_hash = if file_len < 128 * 1024 {
            (file_len.max(16).next_power_of_two().trailing_zeros() as usize).saturating_sub(4)
        } else {
            17
        };
        base_hash = base_hash.clamp(10, 20);

        if is_txt && ratio > 0.90 {
            Self {
                hash_log: base_hash.max(17),
                chain_log: 13,
                search_log: 6,
                min_match: 3,
                strategy: Strategy::Lazy2,
                row_log: 4,
            }
        } else if is_txt {
            Self {
                hash_log: base_hash.max(16),
                chain_log: 12,
                search_log: 5,
                min_match: 4,
                strategy: Strategy::Lazy,
                row_log: 5,
            }
        } else if entropy < 6.5 {
            Self {
                hash_log: 19,
                chain_log: 16,
                search_log: 7,
                min_match: 4,
                strategy: Strategy::BTLazy2,
                row_log: 6,
            }
        } else if entropy > 7.5 {
            Self {
                hash_log: 15,
                chain_log: 7,
                search_log: 1,
                min_match: 6,
                strategy: Strategy::Fast,
                row_log: 4,
            }
        } else {
            Self {
                hash_log: 18,
                chain_log: 10,
                search_log: 4,
                min_match: 4,
                strategy: Strategy::Greedy,
                row_log: 4,
            }
        }
    }
}

fn estimate_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let sample = data.len().min(64 * 1024);
    let mut cnt = [0usize; 256];
    for &b in &data[..sample] {
        cnt[b as usize] += 1;
    }
    let total = sample as f64;
    let mut h = 0.0;
    for &c in &cnt {
        if c > 0 {
            let p = c as f64 / total;
            h -= p * p.log2();
        }
    }
    h
}

pub struct MatchMaker {
    params: Params,
    hash_table: Vec<u32>,
    chain_table: Vec<u32>,
    row_table: Vec<u32>,
    row_size: usize,
    n_rows: usize,
    window_mask: usize,
}

impl MatchMaker {
    pub fn new(params: Params, window_size: usize) -> Self {
        let win = window_size.max(256).next_power_of_two();
        let hash_log = params.hash_log.clamp(8, 20);
        let row_log = params.row_log.min(ROW_LOG_MAX).min(hash_log);
        let hash_size = 1 << hash_log;
        let row_size = 1 << row_log;
        let n_rows = hash_size / row_size;
        Self {
            params: Params {
                hash_log,
                row_log,
                ..params
            },
            hash_table: vec![u32::MAX; hash_size],
            chain_table: vec![u32::MAX; win],
            row_table: vec![u32::MAX; n_rows * row_size],
            row_size,
            n_rows,
            window_mask: win - 1,
        }
    }

    #[inline]
    fn hash_at(&self, data: &[u8], pos: usize) -> usize {
        let n = self.params.min_match.max(3);
        if pos + n > data.len() {
            return 0;
        }
        let v = if n >= 4 {
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
        } else {
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], 0])
        };
        let shift = 32 - self.params.hash_log as u32;
        (v.wrapping_mul(HASH_PRIME) >> shift) as usize
    }

    #[inline]
    fn count_match(&self, data: &[u8], p1: usize, p2: usize) -> usize {
        if p1 >= p2 || p2 >= data.len() {
            return 0;
        }
        let max = (data.len() - p2).min(p2 - p1).min(MAX_MATCH);
        let mut len = 0;
        while len + 8 <= max && data[p1 + len..p1 + len + 8] == data[p2 + len..p2 + len + 8] {
            len += 8;
        }
        while len < max && data[p1 + len] == data[p2 + len] {
            len += 1;
        }
        len
    }

    pub fn find_matches(&mut self, data: &[u8], pos: usize, out: &mut Vec<Match>) {
        out.clear();
        if pos + self.params.min_match > data.len() {
            return;
        }
        let h = self.hash_at(data, pos);
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        let mut searched = 0usize;
        let search_lim = 1 << self.params.search_log.min(12);

        let row = h / self.row_size;
        if row < self.n_rows {
            let base = row * self.row_size;
            for k in 0..self.row_size {
                let candidate = self.row_table[base + k];
                if candidate == u32::MAX {
                    continue;
                }
                let c = candidate as usize;
                if pos <= c || pos - c > (1 << WINDOW_LOG) {
                    continue;
                }
                let len = self.count_match(data, c, pos);
                if len >= self.params.min_match && len > best_len {
                    best_len = len;
                    best_dist = pos - c;
                }
                searched += 1;
                if searched >= search_lim {
                    break;
                }
            }
        }

        let deep = matches!(
            self.params.strategy,
            Strategy::Lazy | Strategy::Lazy2 | Strategy::BTLazy2 | Strategy::BTOPT | Strategy::Greedy
        );
        if deep && searched < search_lim {
            let chain_lim = 1 << self.params.chain_log.min(16);
            let mut chain_pos = self.hash_table[h];
            let mut walked = 0usize;
            while chain_pos != u32::MAX && walked < chain_lim && searched < search_lim {
                let c = chain_pos as usize;
                if pos > c && pos - c <= (1 << WINDOW_LOG) {
                    let len = self.count_match(data, c, pos);
                    if len >= self.params.min_match && len > best_len {
                        best_len = len;
                        best_dist = pos - c;
                        if best_len >= MAX_MATCH {
                            break;
                        }
                    }
                }
                chain_pos = self.chain_table[c & self.window_mask];
                walked += 1;
                searched += 1;
            }
        }

        if best_len >= self.params.min_match {
            out.push(Match {
                dist: best_dist,
                len: best_len,
            });
        }
    }

    pub fn insert(&mut self, data: &[u8], pos: usize) {
        if pos + 3 > data.len() {
            return;
        }
        let h = self.hash_at(data, pos);
        let prev = self.hash_table[h];
        self.chain_table[pos & self.window_mask] = prev;
        self.hash_table[h] = pos as u32;
        let row = h / self.row_size;
        if row < self.n_rows {
            let slot = h & (self.row_size - 1);
            self.row_table[row * self.row_size + slot] = pos as u32;
        }
    }

    pub fn lazy_parse(&mut self, data: &[u8]) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut pos = 0;
        let mut matches = Vec::new();
        let mut next_matches = Vec::new();
        while pos < data.len() {
            self.find_matches(data, pos, &mut matches);
            if matches.is_empty() {
                tokens.push(Token::Lit(data[pos]));
                self.insert(data, pos);
                pos += 1;
                continue;
            }
            let best = matches[0];
            let lazy = matches!(self.params.strategy, Strategy::Lazy | Strategy::Lazy2 | Strategy::BTLazy2);
            if lazy && pos + 1 < data.len() {
                self.find_matches(data, pos + 1, &mut next_matches);
                if !next_matches.is_empty() && next_matches[0].len > best.len + 2 {
                    tokens.push(Token::Lit(data[pos]));
                    self.insert(data, pos);
                    pos += 1;
                    continue;
                }
            }
            if self.params.strategy == Strategy::Lazy2 && pos + 2 < data.len() {
                let mut m2 = Vec::new();
                self.find_matches(data, pos + 2, &mut m2);
                if !m2.is_empty() && m2[0].len > best.len + 1 {
                    tokens.push(Token::Lit(data[pos]));
                    self.insert(data, pos);
                    pos += 1;
                    continue;
                }
            }
            tokens.push(Token::Match {
                dist: best.dist,
                len: best.len,
            });
            for i in 0..best.len {
                if pos + i < data.len() {
                    self.insert(data, pos + i);
                }
            }
            pos += best.len;
        }
        tokens
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Token {
    Lit(u8),
    Match { dist: usize, len: usize },
}

pub fn encode_tokens(tokens: &[Token]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() + 8);
    out.extend_from_slice(b"MM19");
    out.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for t in tokens {
        match *t {
            Token::Lit(b) => {
                out.push(0);
                out.push(b);
            }
            Token::Match { dist, len } => {
                out.push(1);
                out.extend_from_slice(&(len as u32).to_le_bytes());
                out.extend_from_slice(&(dist as u32).to_le_bytes());
            }
        }
    }
    out
}

pub fn decode_tokens(buf: &[u8], orig_len: usize) -> Result<Vec<u8>, &'static str> {
    if buf.len() < 8 || &buf[..4] != b"MM19" {
        return Err("mm19 magic");
    }
    let n = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let mut i = 8;
    let mut out = Vec::with_capacity(orig_len);
    for _ in 0..n {
        if i >= buf.len() {
            return Err("mm19 eof");
        }
        match buf[i] {
            0 => {
                i += 1;
                if i >= buf.len() {
                    return Err("mm19 lit");
                }
                out.push(buf[i]);
                i += 1;
            }
            1 => {
                i += 1;
                if i + 8 > buf.len() {
                    return Err("mm19 match");
                }
                let len = u32::from_le_bytes(buf[i..i + 4].try_into().unwrap()) as usize;
                let dist = u32::from_le_bytes(buf[i + 4..i + 8].try_into().unwrap()) as usize;
                i += 8;
                if dist == 0 || dist > out.len() {
                    return Err("mm19 dist");
                }
                for _ in 0..len {
                    let b = out[out.len() - dist];
                    out.push(b);
                }
            }
            _ => return Err("mm19 tag"),
        }
    }
    Ok(out)
}

pub fn compress_aware(data: &[u8], file_len: u64) -> Vec<Token> {
    let params = Params::aware(data, file_len);
    let window = (1usize << WINDOW_LOG).min(data.len().next_power_of_two().max(256));
    let mut mm = MatchMaker::new(params, window);
    mm.lazy_parse(data)
}

pub fn lz_compress(data: &[u8]) -> Vec<u8> {
    let tokens = compress_aware(data, data.len() as u64);
    encode_tokens(&tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aware_text_chooses_lazy2() {
        let data = b"the quick brown fox jumps over the lazy dog ".repeat(1000);
        let p = Params::aware(&data, 1_000_000);
        assert_eq!(p.min_match, 3);
        assert_eq!(p.strategy, Strategy::Lazy2);
    }

    #[test]
    fn aware_binary_random_chooses_fast() {
        let data: Vec<u8> = (0..=255).cycle().take(10_000).collect();
        let p = Params::aware(&data, 1_000_000);
        assert!(p.min_match >= 4);
        assert_eq!(p.strategy, Strategy::Fast);
    }

    #[test]
    fn row_finder_finds() {
        let data = b"abcabcabcabc".repeat(100);
        let p = Params::aware(&data, 10_000);
        let mut mm = MatchMaker::new(p, 1 << 16);
        let tokens = mm.lazy_parse(&data);
        assert!(tokens.len() < data.len());
        let enc = encode_tokens(&tokens);
        let back = decode_tokens(&enc, data.len()).expect("rt");
        assert_eq!(back, data.as_slice());
    }

    #[test]
    fn lazy_parse_repeat() {
        let data = b"aaaaabaaaaaab".repeat(100);
        let p = Params::aware(&data, 1000);
        let mut mm = MatchMaker::new(p, 1 << 16);
        let tokens = mm.lazy_parse(&data);
        let enc = encode_tokens(&tokens);
        let back = decode_tokens(&enc, data.len()).expect("rt");
        assert_eq!(back, data.as_slice());
    }

    #[test]
    fn obj1_token_size_probe() {
        let path = "/home/workdir/artifacts/corpora/obj1";
        if let Ok(data) = std::fs::read(path) {
            let tokens = compress_aware(&data, data.len() as u64);
            let enc = encode_tokens(&tokens);
            eprintln!("obj1 raw={} tokens={} mm19={}", data.len(), tokens.len(), enc.len());
            assert!(enc.len() > 0);
            let back = decode_tokens(&enc, data.len()).expect("obj1");
            assert_eq!(back, data);
        }
        let osdb = "/home/workdir/artifacts/corpora/osdb";
        if let Ok(data) = std::fs::read(osdb) {
            let slice = &data[..data.len().min(64_000)];
            let tokens = compress_aware(slice, slice.len() as u64);
            let enc = encode_tokens(&tokens);
            eprintln!("osdb64 raw={} tokens={} mm19={}", slice.len(), tokens.len(), enc.len());
            let back = decode_tokens(&enc, slice.len()).expect("osdb");
            assert_eq!(back, slice);
        }
    }
}

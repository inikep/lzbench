/// Suffix array of S + virtual sentinel (smaller than any byte).
pub fn build_sa(s: &[u8]) -> Vec<usize> {
    let n = s.len();
    let m = n + 1;
    let mut sa: Vec<usize> = (0..m).collect();
    let mut rank = vec![0i32; m];
    let mut tmp = vec![0i32; m];
    for i in 0..n {
        rank[i] = s[i] as i32 + 1;
    }
    rank[n] = 0;
    let mut k = 1usize;
    loop {
        sa.sort_unstable_by(|&a, &b| {
            let ra = rank[a];
            let rb = rank[b];
            if ra != rb {
                return ra.cmp(&rb);
            }
            let ra2 = if a + k < m { rank[a + k] } else { -1 };
            let rb2 = if b + k < m { rank[b + k] } else { -1 };
            ra2.cmp(&rb2)
        });
        tmp[sa[0]] = 0;
        let mut classes = 0i32;
        for i in 1..m {
            let a = sa[i - 1];
            let b = sa[i];
            let pa = (rank[a], if a + k < m { rank[a + k] } else { -1 });
            let pb = (rank[b], if b + k < m { rank[b + k] } else { -1 });
            if pa != pb {
                classes += 1;
            }
            tmp[b] = classes;
        }
        rank.copy_from_slice(&tmp);
        if classes == m as i32 - 1 || k >= m {
            break;
        }
        k *= 2;
    }
    sa
}

/// Returns (L column of S only, index of sentinel in the full n+1 BWT).
pub fn bwt_encode(data: &[u8]) -> (Vec<u8>, usize) {
    let n = data.len();
    if n == 0 {
        return (vec![], 0);
    }
    let sa = build_sa(data);
    let mut l = Vec::with_capacity(n);
    let mut primary = 0usize;
    for (i, &sai) in sa.iter().enumerate() {
        if sai == 0 {
            primary = i;
        } else {
            l.push(data[sai - 1]);
        }
    }
    (l, primary)
}

pub fn bwt_decode(l_column: &[u8], primary: usize) -> Vec<u8> {
    let n = l_column.len();
    if n == 0 {
        return vec![];
    }
    let m = n + 1;
    if primary > n {
        return vec![];
    }
    let mut lfull = vec![0u16; m];
    let mut j = 0usize;
    for i in 0..m {
        if i == primary {
            lfull[i] = 0;
        } else {
            lfull[i] = l_column[j] as u16 + 1;
            j += 1;
        }
    }
    let mut count = [0usize; 257];
    for &sym in &lfull {
        count[sym as usize] += 1;
    }
    let mut start = [0usize; 257];
    let mut sum = 0usize;
    for i in 0..257 {
        start[i] = sum;
        sum += count[i];
    }
    let mut occ = start;
    let mut lf = vec![0usize; m];
    for i in 0..m {
        let b = lfull[i] as usize;
        lf[i] = occ[b];
        occ[b] += 1;
    }
    let mut res_full = vec![0u16; m];
    let mut idx = primary;
    for i in (0..m).rev() {
        res_full[i] = lfull[idx];
        idx = lf[idx];
    }
    res_full
        .into_iter()
        .filter(|&x| x != 0)
        .map(|x| (x - 1) as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banana() {
        let s = b"banana_bandana_mississippi";
        let (l, p) = bwt_encode(s);
        assert_eq!(bwt_decode(&l, p), s);
    }

    #[test]
    fn periodic() {
        let s = b"abababababab";
        let (l, p) = bwt_encode(s);
        assert_eq!(bwt_decode(&l, p), s);
    }

    #[test]
    fn phrase() {
        let s = b"the quick brown fox jumps over the lazy dog. ".repeat(8);
        let (l, p) = bwt_encode(&s);
        assert_eq!(bwt_decode(&l, p), s);
    }

    #[test]
    fn zeros() {
        let s = vec![0u8; 64];
        let (l, p) = bwt_encode(&s);
        assert_eq!(bwt_decode(&l, p), s);
    }
}

#[cfg(test)]
mod sa_check {
    use super::*;
    fn naive_sa(s: &[u8]) -> Vec<usize> {
        let n = s.len();
        let m = n + 1;
        let mut sa: Vec<usize> = (0..m).collect();
        sa.sort_by(|&a, &b| {
            let mut i = a; let mut j = b;
            loop {
                let ra = if i == n { -1 } else if i > n { -2 } else { s[i] as i32 };
                let rb = if j == n { -1 } else if j > n { -2 } else { s[j] as i32 };
                if ra != rb { return ra.cmp(&rb); }
                if i == n || j == n { return ra.cmp(&rb); }
                i += 1; j += 1;
                if i > n && j > n { return std::cmp::Ordering::Equal; }
            }
        });
        sa
    }
    #[test]
    fn sa_matches_naive_on_phrase() {
        let s = b"the quick brown fox jumps over the lazy dog. ".repeat(6);
        let fast = build_sa(&s);
        let naive = naive_sa(&s);
        assert_eq!(fast, naive);
    }
    #[test]
    fn sa_matches_naive_periodic() {
        let s = b"ababababababababXababab";
        assert_eq!(build_sa(s), naive_sa(s));
    }
}

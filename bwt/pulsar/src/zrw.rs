//! ZRW — Zero-Run Walker. Classify short runs before entropy coding.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunClass {
    Repeat(u8),
    Inc,
    Dec,
    D2,
    Rice { k: u8 },
    Normal,
}

pub fn classify_run(buf: &[u8]) -> RunClass {
    if buf.len() < 3 {
        return RunClass::Normal;
    }
    if buf.windows(2).all(|w| w[0] == w[1]) {
        return RunClass::Repeat(buf[0]);
    }
    let diffs: Vec<i16> = buf
        .windows(2)
        .map(|w| w[1] as i16 - w[0] as i16)
        .collect();
    if diffs.iter().all(|&d| d == 1) {
        return RunClass::Inc;
    }
    if diffs.iter().all(|&d| d == -1) {
        return RunClass::Dec;
    }
    if diffs.len() >= 2 {
        let d2_ok = diffs.windows(2).all(|w| (w[1] - w[0]).abs() <= 1);
        if d2_ok {
            return RunClass::D2;
        }
    }
    let k = estimate_rice_k(buf);
    if k < 4 {
        RunClass::Rice { k }
    } else {
        RunClass::Normal
    }
}

pub fn estimate_rice_k(buf: &[u8]) -> u8 {
    if buf.len() < 2 {
        return 4;
    }
    let mut acc: u64 = 0;
    for w in buf.windows(2) {
        acc += (w[1] as i16 - w[0] as i16).unsigned_abs() as u64;
    }
    let mean = acc / (buf.len() as u64 - 1);
    match mean {
        0..=1 => 0,
        2..=3 => 1,
        4..=7 => 2,
        8..=15 => 3,
        _ => 4,
    }
}

/// Longest prefix of `buf` that stays in one productive run class.
pub fn take_run(buf: &[u8]) -> (RunClass, usize) {
    if buf.len() < 4 {
        return (RunClass::Normal, 0);
    }
    // Repeat
    let b0 = buf[0];
    let mut n = 1;
    while n < buf.len() && buf[n] == b0 {
        n += 1;
    }
    if n >= 4 {
        return (RunClass::Repeat(b0), n);
    }
    // Inc / Dec
    let mut inc = 1;
    while inc < buf.len() && buf[inc] == buf[inc - 1].wrapping_add(1) {
        inc += 1;
    }
    if inc >= 4 {
        return (RunClass::Inc, inc);
    }
    let mut dec = 1;
    while dec < buf.len() && buf[dec] == buf[dec - 1].wrapping_sub(1) {
        dec += 1;
    }
    if dec >= 4 {
        return (RunClass::Dec, dec);
    }
    (RunClass::Normal, 0)
}

pub fn emit_run(class: RunClass, len: usize, seed: u8, dst: &mut Vec<u8>) {
    match class {
        RunClass::Repeat(b) => {
            dst.push(0xF0);
            dst.push(b);
            dst.extend_from_slice(&(len as u32).to_le_bytes());
        }
        RunClass::Inc => {
            dst.push(0xF1);
            dst.push(seed);
            dst.extend_from_slice(&(len as u32).to_le_bytes());
        }
        RunClass::Dec => {
            dst.push(0xF2);
            dst.push(seed);
            dst.extend_from_slice(&(len as u32).to_le_bytes());
        }
        RunClass::D2 => {
            dst.push(0xF3);
            dst.push(seed);
            dst.extend_from_slice(&(len as u32).to_le_bytes());
        }
        RunClass::Rice { k } => {
            dst.push(0xF4);
            dst.push(k);
            dst.push(seed);
            dst.extend_from_slice(&(len as u32).to_le_bytes());
        }
        RunClass::Normal => dst.push(0xFF),
    }
}

pub fn expand_run(class: RunClass, len: usize, seed: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    match class {
        RunClass::Repeat(b) => out.resize(len, b),
        RunClass::Inc => {
            let mut b = seed;
            for _ in 0..len {
                out.push(b);
                b = b.wrapping_add(1);
            }
        }
        RunClass::Dec => {
            let mut b = seed;
            for _ in 0..len {
                out.push(b);
                b = b.wrapping_sub(1);
            }
        }
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_repeat_inc_dec() {
        assert_eq!(classify_run(&[7, 7, 7, 7, 7]), RunClass::Repeat(7));
        assert_eq!(classify_run(&[0, 1, 2, 3, 4]), RunClass::Inc);
        assert_eq!(classify_run(&[9, 8, 7, 6, 5]), RunClass::Dec);
    }
}

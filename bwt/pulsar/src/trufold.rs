//! TruFold — log-spaced predictor. Residuals grown, not stored as a table.

pub struct TruFold {
    log_table: [usize; 64],
    depth: u8,
}

impl Default for TruFold {
    fn default() -> Self {
        Self::new()
    }
}

impl TruFold {
    pub fn new() -> Self {
        let mut log_table = [0usize; 64];
        for i in 0..64 {
            log_table[i] = (1usize << (i / 8).min(16)) + (i * 3) % 7;
        }
        Self {
            log_table,
            depth: 6,
        }
    }

    #[inline]
    pub fn fold_predict(&self, ctx: &[u8], pos: usize) -> u8 {
        if pos == 0 || ctx.is_empty() {
            return 0;
        }
        let mut acc: i32 = 0;
        let mut w: i32 = 16;
        let depth = self.depth as usize;
        for i in 0..depth {
            let tap = self.log_table[i];
            if pos >= tap && tap > 0 {
                let a = ctx[pos - tap] as i32;
                let half = (tap / 2).max(1);
                let b = if pos >= half {
                    ctx[pos - half] as i32
                } else {
                    a
                };
                acc += (a - b) * w;
                w = w * 3 / 4;
            }
        }
        let pred = (acc >> 4).clamp(-128, 127) + (ctx[pos - 1] as i32);
        pred.clamp(0, 255) as u8
    }

    pub fn residual(&self, ctx: &[u8], pos: usize, actual: u8) -> i8 {
        let p = self.fold_predict(ctx, pos) as i16;
        (actual as i16 - p) as i8
    }

    pub fn reconstruct_one(&self, ctx: &[u8], residual: i8) -> u8 {
        let p = self.fold_predict(ctx, ctx.len()) as i16;
        (p + residual as i16) as u8
    }
}

/// Pack i8 residuals as raw bytes (identity). Gate still applies at block level.
pub fn tru8_pack(residuals: &[i8]) -> Vec<u8> {
    residuals.iter().map(|r| *r as u8).collect()
}

pub fn tru8_unpack(buf: &[u8]) -> Vec<i8> {
    buf.iter().map(|b| *b as i8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_then_residual_rebuilds() {
        let fold = TruFold::new();
        let ctx = b"aaaaaaaaaa".to_vec();
        let actual = b'a';
        let r = fold.residual(&ctx, ctx.len(), actual);
        let got = fold.reconstruct_one(&ctx, r);
        assert_eq!(got, actual);
    }
}

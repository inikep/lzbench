//! InhibitorBus + Mirrored Cost. c -= de, error = |E+C|.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Inhibit {
    Allow,
    Throttle(f32),
    Block,
    SuggestAlt(u8),
    AllowZero,
}

pub struct InhibitorBus {
    pub inhibits: [Inhibit; 8],
    pub mirrored: [i32; 8],
}

impl Default for InhibitorBus {
    fn default() -> Self {
        Self::new()
    }
}

impl InhibitorBus {
    pub fn new() -> Self {
        Self {
            inhibits: [Inhibit::Allow; 8],
            mirrored: [0; 8],
        }
    }

    pub fn apply(&mut self, model_id: usize, de: i32, c: &mut i32, err: &mut i32) -> Inhibit {
        *c -= de;
        *err = (*err + *c).abs();
        let id = model_id.min(7);
        match self.inhibits[id] {
            Inhibit::Allow => Inhibit::Allow,
            Inhibit::Throttle(k) => {
                *c = (*c as f32 * k) as i32;
                Inhibit::Throttle(k)
            }
            Inhibit::Block => {
                *c = i32::MAX / 2;
                Inhibit::Block
            }
            Inhibit::SuggestAlt(ctx) => {
                self.mirrored[id] = *c;
                Inhibit::SuggestAlt(ctx)
            }
            Inhibit::AllowZero => Inhibit::AllowZero,
        }
    }

    pub fn update(&mut self, id: usize, e: i32) {
        let id = id.min(7);
        self.mirrored[id] = (self.mirrored[id] * 7 + e) / 8;
    }

    /// Mix four order probabilities (0..4096) with mirrored-cost weights.
    pub fn mix_o0o1o2o4(&mut self, p0: u32, p1: u32, p2: u32, p4: u32) -> u32 {
        let mut w = [16i32, 24, 20, 12];
        for (i, wi) in w.iter_mut().enumerate() {
            let mut c = *wi;
            let mut err = 0;
            let de = self.mirrored[i] / 32;
            self.apply(i, de, &mut c, &mut err);
            *wi = c.clamp(1, 64);
        }
        let num = w[0] as u32 * p0 + w[1] as u32 * p1 + w[2] as u32 * p2 + w[3] as u32 * p4;
        let den = (w[0] + w[1] + w[2] + w[3]) as u32;
        (num / den).clamp(1, 4095)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirrored_cost_subtracts_de() {
        let mut bus = InhibitorBus::new();
        let mut c = 100;
        let mut err = 0;
        bus.apply(0, 10, &mut c, &mut err);
        assert_eq!(c, 90);
        assert_eq!(err, 90);
    }
}

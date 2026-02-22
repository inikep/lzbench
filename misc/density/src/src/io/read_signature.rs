#[derive(Debug)]
pub struct ReadSignature {
    pub(crate) value: u64,
}

impl ReadSignature {
    pub fn new(value: u64) -> Self {
        ReadSignature { value }
    }

    #[inline(always)]
    pub fn read_bits(&mut self, mask: u64, n: u8) -> u64 {
        let value = self.value & mask;
        self.value >>= n;
        value
    }
}
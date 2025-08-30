#[derive(Debug)]
pub struct WriteSignature {
    pub(crate) pos: usize,
    pub(crate) value: u64,
    pub(crate) shift: u8,
}

impl WriteSignature {
    pub fn new() -> Self {
        WriteSignature { pos: 0, value: 0, shift: 0 }
    }

    #[inline(always)]
    pub fn push_bits(&mut self, mask: u64, n: u8) {
        self.value |= mask << self.shift;
        self.shift += n;
    }

    #[inline(always)]
    pub fn init(&mut self, pos: usize) {
        self.pos = pos;
        self.value = 0;
        self.shift = 0;
    }
}
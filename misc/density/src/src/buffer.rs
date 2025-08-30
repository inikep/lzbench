pub struct Buffer<const N: usize> {
    pub buffer: [u8; N],
    pub index: usize,
}

impl<const N: usize> Buffer<N> {
    pub fn new() -> Self {
        Buffer {
            buffer: [0; N],
            index: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, bytes: &[u8]) {
        let future_index = self.index + bytes.len();
        self.buffer[self.index..future_index].copy_from_slice(bytes);
        self.index = future_index;
    }

    #[inline(always)]
    pub fn remaining_space(&self) -> usize {
        N - self.index
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.index = 0;
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.index == 0
    }
}
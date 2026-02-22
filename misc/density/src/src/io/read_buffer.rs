pub struct ReadBuffer<'a> {
    pub buffer: &'a [u8],
    pub index: usize,
}

impl<'a> ReadBuffer<'a> {
    pub fn new(buffer: &'a [u8]) -> Self {
        ReadBuffer {
            buffer,
            index: 0,
        }
    }

    #[inline(always)]
    pub fn remaining(&self) -> usize {
        self.buffer.len() - self.index
    }

    #[inline(always)]
    pub fn read(&mut self, n: usize) -> &[u8] {
        let end_index = self.index + n;
        let array = &self.buffer[self.index..end_index];
        self.index = end_index;
        array
    }
    #[inline(always)]
    pub fn rewind(&mut self, n: usize) {
        self.index -= n;
    }

    #[inline(always)]
    pub fn read_u64_le(&mut self) -> u64 {
        u64::from_le_bytes(<&[u8] as TryInto<[u8; size_of::<u64>()]>>::try_into(self.read(size_of::<u64>())).unwrap())
    }

    #[inline(always)]
    pub fn read_u32_le(&mut self) -> u32 {
        u32::from_le_bytes(<&[u8] as TryInto<[u8; size_of::<u32>()]>>::try_into(self.read(size_of::<u32>())).unwrap())
    }

    #[inline(always)]
    pub fn read_u16_le(&mut self) -> u16 {
        u16::from_le_bytes(<&[u8] as TryInto<[u8; size_of::<u16>()]>>::try_into(self.read(size_of::<u16>())).unwrap())
    }
}
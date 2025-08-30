use crate::io::write_signature::WriteSignature;

pub struct WriteBuffer<'a> {
    pub buffer: &'a mut [u8],
    pub index: usize,
}

impl<'a> WriteBuffer<'a> {
    pub fn new(buffer: &'a mut [u8]) -> Self {
        WriteBuffer {
            buffer,
            index: 0,
        }
    }

    #[inline(always)]
    pub fn write_at(&mut self, index: usize, bytes: &[u8]) -> usize {
        let future_index = index + bytes.len();
        self.buffer[index..future_index].copy_from_slice(bytes);
        future_index
    }

    #[inline(always)]
    pub fn ink(&mut self, signature: &mut WriteSignature) {
        self.write_at(signature.pos, &signature.value.to_le_bytes());
    }

    #[inline(always)]
    pub fn push(&mut self, bytes: &[u8]) {
        self.index = self.write_at(self.index, bytes);
    }

    #[inline(always)]
    pub fn skip(&mut self, length: usize) {
        self.index += length;
    }

    #[inline(always)]
    pub fn rewind(&mut self, length: usize) {
        self.index -= length;
    }
}
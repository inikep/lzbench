use crate::io::write_buffer::WriteBuffer;
use crate::io::write_signature::WriteSignature;

pub trait QuadEncoder {
    fn encode_quad(&mut self, quad: u32, out_buffer: &mut WriteBuffer, signature: &mut WriteSignature);
}
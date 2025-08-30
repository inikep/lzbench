use crate::codec::decoder::Decoder;
use crate::codec::protection_state::ProtectionState;
use crate::codec::quad_encoder::QuadEncoder;
use crate::errors::decode_error::DecodeError;
use crate::errors::encode_error::EncodeError;
use crate::io::read_buffer::ReadBuffer;
use crate::io::read_signature::ReadSignature;
use crate::io::write_buffer::WriteBuffer;
use crate::io::write_signature::WriteSignature;
use crate::{BYTE_SIZE_U128, BYTE_SIZE_U32};

pub trait Codec: QuadEncoder + Decoder {
    fn block_size() -> usize;
    fn decode_unit_size() -> usize;
    fn signature_significant_bytes() -> usize;
    fn clear_state(&mut self);

    fn safe_encode_buffer_size(size: usize) -> usize {
        let blocks = size / Self::block_size();
        size + blocks * Self::signature_significant_bytes() + if size % Self::block_size() > 0 { Self::signature_significant_bytes() } else { 0 }
    }

    #[inline(always)]
    fn write_signature(out_buffer: &mut WriteBuffer, signature: &mut WriteSignature) {
        out_buffer.ink(signature);
    }

    #[inline(always)]
    fn read_signature(in_buffer: &mut ReadBuffer) -> ReadSignature {
        ReadSignature::new(in_buffer.read_u64_le())
    }

    #[inline(always)]
    fn encode_block(&mut self, block: &[u8], out_buffer: &mut WriteBuffer, signature: &mut WriteSignature, protection_state: &mut ProtectionState) {
        if protection_state.revert_to_copy() {
            out_buffer.push(block);
            protection_state.decay();
        } else {
            let mark = out_buffer.index;
            signature.init(out_buffer.index);
            out_buffer.skip(Self::signature_significant_bytes());
            for sub_block in block.chunks(BYTE_SIZE_U128) {
                match <&[u8] as TryInto<[u8; BYTE_SIZE_U128]>>::try_into(sub_block) {
                    Ok(array) => {
                        let value_u128 = u128::from_le_bytes(array);
                        self.encode_quad((value_u128 & 0xffffffff) as u32, out_buffer, signature);
                        self.encode_quad(((value_u128 >> 32) & 0xffffffff) as u32, out_buffer, signature);
                        self.encode_quad(((value_u128 >> 64) & 0xffffffff) as u32, out_buffer, signature);
                        self.encode_quad((value_u128 >> 96) as u32, out_buffer, signature);
                    }
                    Err(_error) => {
                        // Less than 16 bytes left
                        for bytes in sub_block.chunks(BYTE_SIZE_U32) {
                            match <&[u8] as TryInto<[u8; BYTE_SIZE_U32]>>::try_into(bytes) {
                                Ok(array) => {
                                    self.encode_quad(u32::from_le_bytes(array), out_buffer, signature);
                                }
                                Err(_error) => {
                                    // Implicit signature plain flag (0x0)
                                    out_buffer.push(bytes);
                                }
                            }
                        }
                    }
                }
            }
            Self::write_signature(out_buffer, signature);
            protection_state.update(out_buffer.index - mark >= Self::block_size());
        }
    }

    fn encode(&mut self, input: &[u8], output: &mut [u8]) -> Result<usize, EncodeError> {
        let mut out_buffer = WriteBuffer::new(output);
        let mut signature = WriteSignature::new();
        let mut protection_state = ProtectionState::new();
        for block in input.chunks(Self::block_size()) {
            self.encode_block(block, &mut out_buffer, &mut signature, &mut protection_state);
        }
        Ok(out_buffer.index)
    }

    fn decode(&mut self, input: &[u8], output: &mut [u8]) -> Result<usize, DecodeError> {
        let mut in_buffer = ReadBuffer::new(input);
        let mut out_buffer = WriteBuffer::new(output);
        let mut protection_state = ProtectionState::new();
        let iterations = Self::block_size() / Self::decode_unit_size();

        while in_buffer.remaining() >= Self::signature_significant_bytes() + Self::block_size() {
            if protection_state.revert_to_copy() {
                out_buffer.push(in_buffer.read(Self::block_size()));
                protection_state.decay();
            } else {
                let mark = in_buffer.index;
                let mut signature = Self::read_signature(&mut in_buffer);
                for _ in 0..iterations {
                    self.decode_unit(&mut in_buffer, &mut signature, &mut out_buffer);
                }
                protection_state.update(in_buffer.index - mark >= Self::block_size());
            }
        }

        'end: while in_buffer.remaining() > 0 {
            if protection_state.revert_to_copy() {
                if in_buffer.remaining() > Self::block_size() {
                    out_buffer.push(in_buffer.read(Self::block_size()));
                } else {
                    out_buffer.push(in_buffer.read(in_buffer.remaining()));
                    break 'end;
                }
                protection_state.decay();
            } else {
                let mark = in_buffer.index;
                let mut signature = Self::read_signature(&mut in_buffer);
                for _ in 0..iterations {
                    if in_buffer.remaining() >= Self::decode_unit_size() {
                        self.decode_unit(&mut in_buffer, &mut signature, &mut out_buffer);
                    } else {
                        if self.decode_partial_unit(&mut in_buffer, &mut signature, &mut out_buffer) { break 'end; }
                    }
                }
                protection_state.update(in_buffer.index - mark >= Self::block_size());
            }
        }

        Ok(out_buffer.index)
    }
}
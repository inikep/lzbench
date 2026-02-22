use crate::algorithms::PLAIN_FLAG;
use crate::codec::codec::Codec;
use crate::codec::decoder::Decoder;
use crate::codec::quad_encoder::QuadEncoder;
use crate::errors::decode_error::DecodeError;
use crate::errors::encode_error::EncodeError;
use crate::io::read_buffer::ReadBuffer;
use crate::io::read_signature::ReadSignature;
use crate::io::write_buffer::WriteBuffer;
use crate::io::write_signature::WriteSignature;
use crate::{BIT_SIZE_U16, BIT_SIZE_U32, BYTE_SIZE_U32};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub(crate) const CHAMELEON_HASH_BITS: usize = BIT_SIZE_U16;
pub(crate) const CHAMELEON_HASH_MULTIPLIER: u32 = 0x9D6EF916;

pub(crate) const FLAG_SIZE_BITS: u8 = 1;
pub(crate) const MAP_FLAG: u64 = 0x1;

pub(crate) const PLAIN_PLAIN_FLAGS: u64 = (PLAIN_FLAG << 1) | PLAIN_FLAG;
pub(crate) const MAP_PLAIN_FLAGS: u64 = (PLAIN_FLAG << 1) | MAP_FLAG;
pub(crate) const PLAIN_MAP_FLAGS: u64 = (MAP_FLAG << 1) | PLAIN_FLAG;
// pub(crate) const _MAP_MAP_FLAGS: u64 = (MAP_FLAG << 1) | MAP_FLAG;

pub(crate) const DECODE_TWIN_FLAG_MASK: u64 = 0x3;
pub(crate) const DECODE_TWIN_FLAG_MASK_BITS: u8 = 2;
pub(crate) const DECODE_FLAG_MASK: u64 = 0x1;
pub(crate) const DECODE_FLAG_MASK_BITS: u8 = 1;

pub struct State {
    pub(crate) chunk_map: Vec<u32>,
}

pub struct Chameleon {
    pub state: State,
}

impl Chameleon {
    pub fn new() -> Self {
        Chameleon {
            state: State { chunk_map: vec![0; 1 << CHAMELEON_HASH_BITS] },
        }
    }

    pub fn encode(input: &[u8], output: &mut [u8]) -> Result<usize, EncodeError> {
        let mut chameleon = Chameleon::new();
        chameleon.encode(input, output)
    }

    pub fn decode(input: &[u8], output: &mut [u8]) -> Result<usize, DecodeError> {
        let mut chameleon = Chameleon::new();
        chameleon.decode(input, output)
    }

    #[inline(always)]
    fn decode_plain(&mut self, in_buffer: &mut ReadBuffer) -> u32 {
        let quad = in_buffer.read_u32_le();
        let hash = (quad.wrapping_mul(CHAMELEON_HASH_MULTIPLIER) >> (BIT_SIZE_U32 - CHAMELEON_HASH_BITS)) as u16;
        self.state.chunk_map[hash as usize] = quad;
        quad
    }

    #[inline(always)]
    fn decode_map(&mut self, in_buffer: &mut ReadBuffer) -> u32 {
        let hash = in_buffer.read_u16_le();
        let quad = self.state.chunk_map[hash as usize];
        quad
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn chameleon_encode(input: *const u8, input_size: usize, output: *mut u8, output_size: usize) -> usize {
        unsafe { Self::encode(from_raw_parts(input, input_size), from_raw_parts_mut(output, output_size)).unwrap_or(0) }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn chameleon_decode(input: *const u8, input_size: usize, output: *mut u8, output_size: usize) -> usize {
        unsafe { Self::decode(from_raw_parts(input, input_size), from_raw_parts_mut(output, output_size)).unwrap_or(0) }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn chameleon_safe_encode_buffer_size(size: usize) -> usize {
        Self::safe_encode_buffer_size(size)
    }
}

impl QuadEncoder for Chameleon {
    #[inline(always)]
    fn encode_quad(&mut self, quad: u32, out_buffer: &mut WriteBuffer, signature: &mut WriteSignature) {
        let hash_u16 = (quad.wrapping_mul(CHAMELEON_HASH_MULTIPLIER) >> (BIT_SIZE_U32 - CHAMELEON_HASH_BITS)) as u16;
        let dictionary_value = &mut self.state.chunk_map[hash_u16 as usize];
        if *dictionary_value != quad {
            signature.push_bits(PLAIN_FLAG, FLAG_SIZE_BITS);
            out_buffer.push(&quad.to_le_bytes());

            *dictionary_value = quad;
        } else {
            signature.push_bits(MAP_FLAG, FLAG_SIZE_BITS);
            out_buffer.push(&hash_u16.to_le_bytes());
        }
    }
}

impl Decoder for Chameleon {
    #[inline(always)]
    fn decode_unit(&mut self, in_buffer: &mut ReadBuffer, signature: &mut ReadSignature, out_buffer: &mut WriteBuffer) {
        let (quad_a, quad_b) = match signature.read_bits(DECODE_TWIN_FLAG_MASK, DECODE_TWIN_FLAG_MASK_BITS) {
            PLAIN_PLAIN_FLAGS => { (self.decode_plain(in_buffer), self.decode_plain(in_buffer)) }
            MAP_PLAIN_FLAGS => { (self.decode_map(in_buffer), self.decode_plain(in_buffer)) }
            PLAIN_MAP_FLAGS => { (self.decode_plain(in_buffer), self.decode_map(in_buffer)) }
            _ => { (self.decode_map(in_buffer), self.decode_map(in_buffer)) }
        };
        out_buffer.push(&quad_a.to_le_bytes());
        out_buffer.push(&quad_b.to_le_bytes());
    }

    #[inline(always)]
    fn decode_partial_unit(&mut self, in_buffer: &mut ReadBuffer, signature: &mut ReadSignature, out_buffer: &mut WriteBuffer) -> bool {
        for _ in 0..Self::decode_unit_size() / BYTE_SIZE_U32 {
            let quad = match signature.read_bits(DECODE_FLAG_MASK, DECODE_FLAG_MASK_BITS) {
                PLAIN_FLAG => {
                    match in_buffer.remaining() {
                        0 => { return true; }
                        1..=3 => {
                            out_buffer.push(in_buffer.read(in_buffer.remaining()));
                            return true;
                        }
                        _ => { self.decode_plain(in_buffer) }
                    }
                }
                _ => { self.decode_map(in_buffer) }
            };
            out_buffer.push(&quad.to_le_bytes());
        }
        false
    }
}

impl Codec for Chameleon {
    #[inline(always)]
    fn block_size() -> usize { BYTE_SIZE_U32 * (Self::signature_significant_bytes() << 3) }

    #[inline(always)]
    fn decode_unit_size() -> usize { 8 }

    #[inline(always)]
    fn signature_significant_bytes() -> usize { 8 }

    fn clear_state(&mut self) {
        self.state.chunk_map.fill(0);
    }
}

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
use crate::{BIT_SIZE_U16, BIT_SIZE_U32, BYTE_SIZE_U16, BYTE_SIZE_U32};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub(crate) const LION_HASH_BITS: usize = BIT_SIZE_U16;
pub(crate) const LION_HASH_MULTIPLIER: u32 = 0x9D6EF916;


pub(crate) const FLAG_SIZE_BITS: u8 = 3;
pub(crate) const PREDICTED_A_FLAG: u64 = 0x1;
pub(crate) const PREDICTED_B_FLAG: u64 = 0x2;
pub(crate) const PREDICTED_C_FLAG: u64 = 0x3;
pub(crate) const PREDICTED_D_FLAG: u64 = 0x4;
pub(crate) const PREDICTED_E_FLAG: u64 = 0x5;
pub(crate) const MAP_A_FLAG: u64 = 0x6;
pub(crate) const MAP_B_FLAG: u64 = 0x7;
pub(crate) const DECODE_FLAG_MASK: u64 = 0x7;
pub(crate) const DECODE_FLAG_MASK_BITS: u8 = 3;

pub struct State {
    pub(crate) last_hash: u16,
    pub(crate) chunk_map: Vec<ChunkData>,
    pub(crate) prediction_map: Vec<PredictionData>,
}

#[derive(Copy, Clone)]
pub struct ChunkData {
    pub(crate) chunk_a: u32,
    pub(crate) chunk_b: u32,
}

#[derive(Copy, Clone)]
pub struct PredictionData {
    pub(crate) next_a: u32,
    pub(crate) next_b: u32,
    pub(crate) next_c: u32,
    pub(crate) next_d: u32,
    pub(crate) next_e: u32,
}

#[inline(always)]
fn shift_predictions(prediction_data: &mut PredictionData, quad: u32) {
    prediction_data.next_e = prediction_data.next_d;
    prediction_data.next_d = prediction_data.next_c;
    prediction_data.next_c = prediction_data.next_b;
    prediction_data.next_b = prediction_data.next_a;
    prediction_data.next_a = quad;
}

pub struct Lion {
    pub state: State,
}

impl Lion {
    pub fn new() -> Self {
        Lion {
            state: State {
                last_hash: 0,
                chunk_map: vec![ChunkData { chunk_a: 0, chunk_b: 0 }; 1 << LION_HASH_BITS],
                prediction_map: vec![PredictionData { next_a: 0, next_b: 0, next_c: 0, next_d: 0, next_e: 0 }; 1 << LION_HASH_BITS],
            },
        }
    }

    pub fn encode(input: &[u8], output: &mut [u8]) -> Result<usize, EncodeError> {
        let mut lion = Lion::new();
        lion.encode(input, output)
    }

    pub fn decode(input: &[u8], output: &mut [u8]) -> Result<usize, DecodeError> {
        let mut lion = Lion::new();
        lion.decode(input, output)
    }

    #[inline(always)]
    fn decode_plain(&mut self, in_buffer: &mut ReadBuffer) -> (u16, u32) {
        let quad = in_buffer.read_u32_le();
        let hash = (quad.wrapping_mul(LION_HASH_MULTIPLIER) >> (BIT_SIZE_U32 - LION_HASH_BITS)) as u16;

        let chunk_data = &mut self.state.chunk_map[hash as usize];
        chunk_data.chunk_b = chunk_data.chunk_a;
        chunk_data.chunk_a = quad;

        shift_predictions(&mut self.state.prediction_map[self.state.last_hash as usize], quad);

        (hash, quad)
    }

    #[inline(always)]
    fn decode_map_a(&mut self, in_buffer: &mut ReadBuffer) -> (u16, u32) {
        let hash = in_buffer.read_u16_le();

        let chunk_data = &mut self.state.chunk_map[hash as usize];
        let quad = chunk_data.chunk_a;

        shift_predictions(&mut self.state.prediction_map[self.state.last_hash as usize], quad);

        (hash, quad)
    }

    #[inline(always)]
    fn decode_map_b(&mut self, in_buffer: &mut ReadBuffer) -> (u16, u32) {
        let hash = in_buffer.read_u16_le();

        let chunk_data = &mut self.state.chunk_map[hash as usize];
        let quad = chunk_data.chunk_b;

        chunk_data.chunk_b = chunk_data.chunk_a;
        chunk_data.chunk_a = quad;

        shift_predictions(&mut self.state.prediction_map[self.state.last_hash as usize], quad);

        (hash, quad)
    }

    #[inline(always)]
    fn decode_predicted_a(&mut self) -> (u16, u32) {
        let prediction_data = &mut self.state.prediction_map[self.state.last_hash as usize];
        let quad = prediction_data.next_a;
        let hash = (quad.wrapping_mul(LION_HASH_MULTIPLIER) >> (BIT_SIZE_U32 - LION_HASH_BITS)) as u16;

        (hash, quad)
    }

    #[inline(always)]
    fn decode_predicted_b(&mut self) -> (u16, u32) {
        let prediction_data = &mut self.state.prediction_map[self.state.last_hash as usize];
        let quad = prediction_data.next_b;
        let hash = (quad.wrapping_mul(LION_HASH_MULTIPLIER) >> (BIT_SIZE_U32 - LION_HASH_BITS)) as u16;

        prediction_data.next_b = prediction_data.next_a;
        prediction_data.next_a = quad;

        (hash, quad)
    }

    #[inline(always)]
    fn decode_predicted_c(&mut self) -> (u16, u32) {
        let prediction_data = &mut self.state.prediction_map[self.state.last_hash as usize];
        let quad = prediction_data.next_c;
        let hash = (quad.wrapping_mul(LION_HASH_MULTIPLIER) >> (BIT_SIZE_U32 - LION_HASH_BITS)) as u16;

        prediction_data.next_c = prediction_data.next_b;
        prediction_data.next_b = prediction_data.next_a;
        prediction_data.next_a = quad;

        (hash, quad)
    }

    #[inline(always)]
    fn decode_predicted_d(&mut self) -> (u16, u32) {
        let prediction_data = &mut self.state.prediction_map[self.state.last_hash as usize];
        let quad = prediction_data.next_d;
        let hash = (quad.wrapping_mul(LION_HASH_MULTIPLIER) >> (BIT_SIZE_U32 - LION_HASH_BITS)) as u16;

        prediction_data.next_d = prediction_data.next_c;
        prediction_data.next_c = prediction_data.next_b;
        prediction_data.next_b = prediction_data.next_a;
        prediction_data.next_a = quad;

        (hash, quad)
    }

    #[inline(always)]
    fn decode_predicted_e(&mut self) -> (u16, u32) {
        let prediction_data = &mut self.state.prediction_map[self.state.last_hash as usize];
        let quad = prediction_data.next_e;
        let hash = (quad.wrapping_mul(LION_HASH_MULTIPLIER) >> (BIT_SIZE_U32 - LION_HASH_BITS)) as u16;

        prediction_data.next_e = prediction_data.next_d;
        prediction_data.next_d = prediction_data.next_c;
        prediction_data.next_c = prediction_data.next_b;
        prediction_data.next_b = prediction_data.next_a;
        prediction_data.next_a = quad;

        (hash, quad)
    }

    #[inline(always)]
    fn update_last_hash(&mut self, hash_u16: u16) {
        self.state.last_hash = hash_u16;
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lion_encode(input: *const u8, input_size: usize, output: *mut u8, output_size: usize) -> usize {
        unsafe { Self::encode(from_raw_parts(input, input_size), from_raw_parts_mut(output, output_size)).unwrap_or(0) }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lion_decode(input: *const u8, input_size: usize, output: *mut u8, output_size: usize) -> usize {
        unsafe { Self::decode(from_raw_parts(input, input_size), from_raw_parts_mut(output, output_size)).unwrap_or(0) }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn lion_safe_encode_buffer_size(size: usize) -> usize {
        Self::safe_encode_buffer_size(size)
    }
}

impl QuadEncoder for Lion {
    #[inline(always)]
    fn encode_quad(&mut self, quad: u32, out_buffer: &mut WriteBuffer, signature: &mut WriteSignature) {
        let hash_u16 = (quad.wrapping_mul(LION_HASH_MULTIPLIER) >> (BIT_SIZE_U32 - LION_HASH_BITS)) as u16;
        let prediction_data = &mut self.state.prediction_map[self.state.last_hash as usize];
        let chunk_data = &mut self.state.chunk_map[hash_u16 as usize];

        if prediction_data.next_a != quad {
            if prediction_data.next_b != quad {
                if prediction_data.next_c != quad {
                    if prediction_data.next_d != quad {
                        if prediction_data.next_e != quad {
                            if chunk_data.chunk_a != quad {
                                if chunk_data.chunk_b != quad {
                                    signature.push_bits(PLAIN_FLAG, FLAG_SIZE_BITS); // Plain
                                    out_buffer.push(&quad.to_le_bytes());
                                } else {
                                    signature.push_bits(MAP_B_FLAG, FLAG_SIZE_BITS); // Map B
                                    out_buffer.push(&hash_u16.to_le_bytes());
                                }
                                chunk_data.chunk_b = chunk_data.chunk_a;
                                chunk_data.chunk_a = quad;

                                shift_predictions(prediction_data, quad);
                            } else {
                                signature.push_bits(MAP_A_FLAG, FLAG_SIZE_BITS); // Map A
                                out_buffer.push(&hash_u16.to_le_bytes());

                                shift_predictions(prediction_data, quad);
                            }
                        } else {
                            signature.push_bits(PREDICTED_E_FLAG, FLAG_SIZE_BITS); // Predicted E

                            shift_predictions(prediction_data, quad);
                        }
                    } else {
                        signature.push_bits(PREDICTED_D_FLAG, FLAG_SIZE_BITS); // Predicted D

                        prediction_data.next_d = prediction_data.next_c;
                        prediction_data.next_c = prediction_data.next_b;
                        prediction_data.next_b = prediction_data.next_a;
                        prediction_data.next_a = quad;
                    }
                } else {
                    signature.push_bits(PREDICTED_C_FLAG, FLAG_SIZE_BITS); // Predicted C

                    prediction_data.next_c = prediction_data.next_b;
                    prediction_data.next_b = prediction_data.next_a;
                    prediction_data.next_a = quad;
                }
            } else {
                signature.push_bits(PREDICTED_B_FLAG, FLAG_SIZE_BITS); // Predicted B

                prediction_data.next_b = prediction_data.next_a;
                prediction_data.next_a = quad;
            }
        } else {
            signature.push_bits(PREDICTED_A_FLAG, FLAG_SIZE_BITS); // Predicted A
        }

        self.update_last_hash(hash_u16);
    }
}

impl Decoder for Lion {
    #[inline(always)]
    fn decode_unit(&mut self, in_buffer: &mut ReadBuffer, signature: &mut ReadSignature, out_buffer: &mut WriteBuffer) {
        let (hash, quad) = match signature.read_bits(DECODE_FLAG_MASK, DECODE_FLAG_MASK_BITS) {
            PREDICTED_B_FLAG => { self.decode_predicted_b() }
            PREDICTED_C_FLAG => { self.decode_predicted_c() }
            PREDICTED_D_FLAG => { self.decode_predicted_d() }
            PREDICTED_E_FLAG => { self.decode_predicted_e() }
            MAP_A_FLAG => { self.decode_map_a(in_buffer) }
            MAP_B_FLAG => { self.decode_map_b(in_buffer) }
            PLAIN_FLAG => { self.decode_plain(in_buffer) }
            _ => { self.decode_predicted_a() }
        };
        self.state.last_hash = hash;
        out_buffer.push(&quad.to_le_bytes());
    }

    #[inline(always)]
    fn decode_partial_unit(&mut self, in_buffer: &mut ReadBuffer, signature: &mut ReadSignature, out_buffer: &mut WriteBuffer) -> bool {
        let (hash, quad) = match signature.read_bits(DECODE_FLAG_MASK, DECODE_FLAG_MASK_BITS) {
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
            PREDICTED_B_FLAG => { self.decode_predicted_b() }
            PREDICTED_C_FLAG => { self.decode_predicted_c() }
            PREDICTED_D_FLAG => { self.decode_predicted_d() }
            PREDICTED_E_FLAG => { self.decode_predicted_e() }
            MAP_A_FLAG => { self.decode_map_a(in_buffer) }
            MAP_B_FLAG => { self.decode_map_b(in_buffer) }
            _ => { self.decode_predicted_a() }
        };
        self.state.last_hash = hash;
        out_buffer.push(&quad.to_le_bytes());
        false
    }
}

impl Codec for Lion {
    #[inline(always)]
    fn block_size() -> usize { BYTE_SIZE_U32 * (Self::signature_significant_bytes() << 3) / FLAG_SIZE_BITS as usize }

    #[inline(always)]
    fn decode_unit_size() -> usize { 4 }

    #[inline(always)]
    fn signature_significant_bytes() -> usize { 6 }

    fn clear_state(&mut self) {
        self.state.last_hash = 0;
        self.state.chunk_map.fill(ChunkData { chunk_a: 0, chunk_b: 0 });
        self.state.prediction_map.fill(PredictionData { next_a: 0, next_b: 0, next_c: 0, next_d: 0, next_e: 0 });
    }

    #[inline(always)]
    fn write_signature(out_buffer: &mut WriteBuffer, signature: &mut WriteSignature) {
        out_buffer.write_at(signature.pos, &signature.value.to_le_bytes()[0..Self::signature_significant_bytes()]);
    }

    #[inline(always)]
    fn read_signature(in_buffer: &mut ReadBuffer) -> ReadSignature {
        match in_buffer.remaining() {
            0..=7 => {
                let bytes = [in_buffer.read(Self::signature_significant_bytes()), &[0, 0]].concat();
                ReadSignature::new(u64::from_le_bytes(<&[u8] as TryInto<[u8; size_of::<u64>()]>>::try_into(&bytes).unwrap()))
            }
            _ => {
                let bytes = in_buffer.read_u64_le();
                in_buffer.rewind(BYTE_SIZE_U16);
                ReadSignature::new(bytes & 0x0000ffffffffffff_u64)
            }
        }
    }
}
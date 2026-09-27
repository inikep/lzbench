//! Small headers are parsed transactionally in the bit reservoir.
//!
//! A header needs at most 32 bits. Peeking retains a partial field without
//! accepting any byte beyond it; dropping happens only after validation.

use super::super::{DecodeError, InvalidDataKind, WindowLimit};
use super::bits::{Bits, Input};
use crate::Window;

#[derive(Debug, Clone, Copy)]
pub(super) enum MetaBlock {
    End,
    Compressed { length: u64, last: bool },
    Uncompressed { length: u64 },
    Metadata { length: u64, last: bool },
}

pub(super) fn window(
    bits: &mut Bits,
    input: &mut Input<'_>,
    limit: WindowLimit,
) -> Result<Option<Window>, DecodeError> {
    let Some(first) = bits.peek(1, input)? else {
        return Ok(None);
    };
    let (declared, width, large) = if first == 0 {
        (16, 1, false)
    } else {
        let Some(short) = bits.peek(4, input)? else {
            return Ok(None);
        };
        if short >> 1 != 0 {
            (17 + (short >> 1) as u8, 4u32, false)
        } else {
            let Some(long) = bits.peek(7, input)? else {
                return Ok(None);
            };
            match long >> 4 {
                0 => (17, 7, false),
                1 => {
                    let Some(marker) = bits.peek(8, input)? else {
                        return Ok(None);
                    };
                    if marker != 0x11 {
                        return Err(InvalidDataKind::Header.into());
                    }
                    let Some(header) = bits.peek(14, input)? else {
                        return Ok(None);
                    };
                    let declared = (header >> 8) as u8;
                    if !(10..=62).contains(&declared) {
                        return Err(InvalidDataKind::Header.into());
                    }
                    (declared, 14, true)
                }
                other => (8 + other as u8, 7, false),
            }
        }
    };
    if large && !limit.allows_large() {
        return Err(DecodeError::LargeWindowDisabled);
    }
    if declared > limit.max_bits() {
        return Err(DecodeError::WindowLimitExceeded {
            declared,
            allowed: limit.max_bits(),
        });
    }
    let window = match if large {
        Window::large(declared)
    } else {
        Window::standard(declared)
    } {
        Ok(window) => window,
        Err(_) => return Err(InvalidDataKind::Header.into()),
    };
    bits.drop(width);
    Ok(Some(window))
}

pub(super) fn metablock(
    bits: &mut Bits,
    input: &mut Input<'_>,
) -> Result<Option<MetaBlock>, DecodeError> {
    let Some(first) = bits.peek(1, input)? else {
        return Ok(None);
    };
    let last = first != 0;
    let prefix = if last {
        let Some(flags) = bits.peek(2, input)? else {
            return Ok(None);
        };
        if flags & 2 != 0 {
            bits.drop(2);
            return Ok(Some(MetaBlock::End));
        }
        2
    } else {
        1
    };
    let Some(header) = bits.peek(prefix + 2, input)? else {
        return Ok(None);
    };
    let nibbles = (header >> prefix) as u8 + 4;
    if nibbles == 7 {
        return metadata(bits, input, prefix + 2, last);
    }
    let width = prefix + 2 + u32::from(nibbles) * 4;
    let Some(header) = bits.peek(width + u32::from(!last), input)? else {
        return Ok(None);
    };
    let encoded_length = (header >> (prefix + 2)) & ((1 << (nibbles * 4)) - 1);
    if nibbles > 4 && encoded_length >> ((nibbles - 1) * 4) == 0 {
        return Err(InvalidDataKind::MetaBlock.into());
    }
    let length = encoded_length + 1;
    let raw = !last && header >> width != 0;
    bits.drop(width + u32::from(!last));
    if raw {
        bits.align()?;
        Ok(Some(MetaBlock::Uncompressed { length }))
    } else {
        Ok(Some(MetaBlock::Compressed { length, last }))
    }
}

fn metadata(
    bits: &mut Bits,
    input: &mut Input<'_>,
    prefix: u32,
    last: bool,
) -> Result<Option<MetaBlock>, DecodeError> {
    let Some(header) = bits.peek(prefix + 3, input)? else {
        return Ok(None);
    };
    if (header >> prefix) & 1 != 0 {
        return Err(InvalidDataKind::MetaBlock.into());
    }
    let bytes = (header >> (prefix + 1)) as u8;
    let width = prefix + 3 + u32::from(bytes) * 8;
    let Some(header) = bits.peek(width, input)? else {
        return Ok(None);
    };
    let encoded_length = header >> (prefix + 3);
    if bytes > 1 && encoded_length >> ((bytes - 1) * 8) == 0 {
        return Err(InvalidDataKind::MetaBlock.into());
    }
    let length = if bytes == 0 { 0 } else { encoded_length + 1 };
    bits.drop(width);
    bits.align()?;
    Ok(Some(MetaBlock::Metadata { length, last }))
}

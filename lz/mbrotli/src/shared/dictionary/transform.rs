//! Borrowed word transformations. No transformed dictionary is materialized.

pub(crate) const SCRATCH_BYTES: usize = 2 * 255 + 31;
#[cfg(feature = "decompression")]
const TRIPLES: &[u8; 363] = include_bytes!("builtin_transforms.bin");
#[cfg(feature = "decompression")]
const STRINGS: &[u8; 217] = include_bytes!("builtin_prefix_suffix.bin");

/// Validated operation inputs. Prefix/suffix and word bytes remain borrowed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Transform<'a> {
    pub(crate) prefix: &'a [u8],
    pub(crate) operation: u8,
    pub(crate) suffix: &'a [u8],
    #[cfg(feature = "experimental")]
    pub(crate) parameter: u16,
}

#[cfg(feature = "decompression")]
impl Transform<'static> {
    pub(crate) fn builtin(index: usize) -> Option<Self> {
        let triple = TRIPLES.get(index.checked_mul(3)?..)?.first_chunk::<3>()?;
        Some(Self {
            prefix: stringlet(triple[0]),
            operation: triple[1],
            suffix: stringlet(triple[2]),
            #[cfg(feature = "experimental")]
            parameter: 0,
        })
    }
}

/// Byte offset of each length-prefixed stringlet in [`STRINGS`], precomputed so
/// a lookup is O(1) rather than a linear scan from the start on every transform.
#[cfg(feature = "decompression")]
const STRINGLET_OFFSETS: [u16; 64] = {
    let mut offsets = [0u16; 64];
    let mut start = 0usize;
    let mut index = 0usize;
    while index < offsets.len() && start < STRINGS.len() {
        offsets[index] = start as u16;
        start += STRINGS[start] as usize + 1;
        index += 1;
    }
    offsets
};

#[cfg(feature = "decompression")]
fn stringlet(id: u8) -> &'static [u8] {
    let start = usize::from(STRINGLET_OFFSETS[usize::from(id)]);
    &STRINGS[start + 1..start + 1 + usize::from(STRINGS[start])]
}

impl Transform<'_> {
    /// The grammar bounds words by 31 and stringlets by 255 bytes. Two guard
    /// bytes within the unused suffix area preserve the format's casing behavior on incomplete byte sequences.
    pub(crate) fn apply(self, word: &[u8], scratch: &mut [u8; SCRATCH_BYTES]) -> usize {
        let body = match self.operation {
            0..=9 => &word[..word.len().saturating_sub(usize::from(self.operation))],
            12..=20 => &word[usize::from(self.operation - 11).min(word.len())..],
            _ => word,
        };
        let start = self.prefix.len();
        let end = start + body.len();
        scratch[..start].copy_from_slice(self.prefix);
        scratch[start..end].copy_from_slice(body);
        if self.operation == 10 || self.operation == 11 {
            ferment(&mut scratch[start..], body.len(), self.operation == 11);
        }
        #[cfg(feature = "experimental")]
        if self.operation == 21 || self.operation == 22 {
            shift(
                &mut scratch[start..end],
                self.parameter,
                self.operation == 22,
            );
        }
        scratch[end..end + self.suffix.len()].copy_from_slice(self.suffix);
        end + self.suffix.len()
    }
}

fn ferment(bytes: &mut [u8], length: usize, all: bool) {
    let mut position = 0;
    while position < length {
        let byte = bytes[position];
        let step = if byte < 0xc0 {
            if byte.is_ascii_lowercase() {
                bytes[position] ^= 32;
            }
            1
        } else if byte < 0xe0 {
            bytes[position + 1] ^= 32;
            2
        } else {
            bytes[position + 2] ^= 5;
            3
        };
        position += step;
        if !all {
            break;
        }
    }
}

#[cfg(feature = "experimental")]
fn shift(mut bytes: &mut [u8], parameter: u16, all: bool) {
    let addend = u32::from(parameter) + if parameter >= 0x8000 { 0xff0000 } else { 0 };
    while let Some(&first) = bytes.first() {
        let (length, mask) = match first {
            0..=0x7f => (1, 0x7f),
            0xc0..=0xdf => (2, 0x1f),
            0xe0..=0xef => (3, 0x0f),
            0xf0..=0xf7 => (4, 7),
            _ => {
                if !all {
                    break;
                }
                bytes = &mut bytes[1..];
                continue;
            }
        };
        if bytes.len() < length {
            break;
        }
        let mut scalar = u32::from(first & mask);
        for &byte in &bytes[1..length] {
            scalar = (scalar << 6) | u32::from(byte & 63);
        }
        scalar += addend;
        for byte in bytes[1..length].iter_mut().rev() {
            *byte = (*byte & 0xc0) | (scalar as u8 & 63);
            scalar >>= 6;
        }
        bytes[0] = (first & !mask) | (scalar as u8 & mask);
        if !all {
            break;
        }
        bytes = &mut bytes[length..];
    }
}

//! Borrowed word transformations. No transformed dictionary is materialized.

pub(crate) const SCRATCH_BYTES: usize = 2 * 255 + 31;
#[cfg(feature = "decompression")]
const TRIPLES: &[u8; 363] = include_bytes!("builtin_transforms.bin");
#[cfg(feature = "decompression")]
const STRINGS: &[u8; 217] = include_bytes!("builtin_prefix_suffix.bin");

/// Validated operation inputs. Prefix/suffix and word bytes remain borrowed.
///
/// The serialized-dictionary paths apply transforms through this; the
/// built-in path uses [`apply_builtin`], which tests compare against it.
#[cfg(any(test, feature = "experimental"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Transform<'a> {
    pub(crate) prefix: &'a [u8],
    pub(crate) operation: u8,
    pub(crate) suffix: &'a [u8],
    #[cfg(feature = "experimental")]
    pub(crate) parameter: u16,
}

#[cfg(all(feature = "decompression", any(test, feature = "experimental")))]
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

#[cfg(all(feature = "decompression", any(test, feature = "experimental")))]
fn stringlet(id: u8) -> &'static [u8] {
    let start = usize::from(STRINGLET_OFFSETS[usize::from(id)]);
    &STRINGS[start + 1..start + 1 + usize::from(STRINGS[start])]
}

/// [`STRINGS`] followed by eight zero bytes, so every stringlet (at most
/// eight bytes) can be copied as one fixed eight-byte move.
#[cfg(feature = "decompression")]
const PADDED: [u8; 217 + 9] = {
    let mut padded = [0u8; 217 + 9];
    let mut index = 0;
    while index < STRINGS.len() {
        padded[index] = STRINGS[index];
        index += 1;
    }
    padded
};

/// The exact-length copy for the last words of the table, kept out of line
/// so the fixed 32-byte move stays a register copy.
#[cfg(feature = "decompression")]
#[cold]
#[inline(never)]
fn copy_exact(target: &mut [u8], source: &[u8]) {
    target.copy_from_slice(source);
}

/// Applies built-in transform `index` to the built-in word of `length`
/// bytes at `offset` with fixed-size over-copies. Same result as
/// `Transform::builtin(index)?.apply(..)`.
#[cfg(feature = "decompression")]
#[inline(always)]
pub(crate) fn apply_builtin(
    index: usize,
    words: &[u8],
    offset: usize,
    length: usize,
    scratch: &mut [u8; SCRATCH_BYTES],
) -> Option<usize> {
    let triple = TRIPLES.get(index.checked_mul(3)?..)?.first_chunk::<3>()?;
    let operation = triple[1];
    let prefix = usize::from(STRINGLET_OFFSETS[usize::from(triple[0] & 63)]);
    let suffix = usize::from(STRINGLET_OFFSETS[usize::from(triple[2] & 63)]);
    let prefix_len = usize::from(*PADDED.get(prefix)?);
    let suffix_len = usize::from(*PADDED.get(suffix)?);
    let (skip, keep) = match operation {
        0..=9 => (0, length.saturating_sub(usize::from(operation))),
        12..=20 => {
            let skip = usize::from(operation - 11).min(length);
            (skip, length - skip)
        }
        _ => (0, length),
    };
    scratch
        .first_chunk_mut::<8>()?
        .copy_from_slice(PADDED.get(prefix + 1..)?.first_chunk::<8>()?);
    let start = prefix_len;
    let end = start + keep;
    let source = offset + skip;
    match (
        words.get(source..).and_then(|w| w.first_chunk::<32>()),
        scratch
            .get_mut(start..)
            .and_then(|t| t.first_chunk_mut::<32>()),
    ) {
        (Some(word), Some(target)) => *target = *word,
        _ => copy_exact(
            scratch.get_mut(start..end)?,
            words.get(source..source + keep)?,
        ),
    }
    if operation == 10 || operation == 11 {
        ferment(&mut scratch[start..], keep, operation == 11);
    }
    scratch
        .get_mut(end..)?
        .first_chunk_mut::<8>()?
        .copy_from_slice(PADDED.get(suffix + 1..)?.first_chunk::<8>()?);
    Some(end + suffix_len)
}

#[cfg(any(test, feature = "experimental"))]
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

#[cfg(any(test, feature = "experimental", feature = "decompression"))]
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

#[cfg(all(test, feature = "decompression"))]
mod tests {
    use super::*;
    use crate::shared::dictionary::{
        BUILTIN_OFFSETS_BY_LENGTH, BUILTIN_SIZE_BITS_BY_LENGTH, BUILTIN_WORDS,
    };

    #[test]
    fn fixed_size_copies_match_the_reference_transform_for_every_builtin() {
        let transforms = TRIPLES.len() / 3;
        assert_eq!(transforms, 121);
        for length in 4..=24 {
            let count = 1usize << BUILTIN_SIZE_BITS_BY_LENGTH[length];
            for index in [0, 1, count / 2, count - 2, count - 1] {
                let offset = BUILTIN_OFFSETS_BY_LENGTH[length] as usize + index * length;
                let word = &BUILTIN_WORDS[offset..offset + length];
                for transform in 0..transforms {
                    let mut expected = [0xAAu8; SCRATCH_BYTES];
                    let Some(reference) = Transform::builtin(transform) else {
                        panic!("built-in transform {transform} exists");
                    };
                    let expected_len = reference.apply(word, &mut expected);
                    let mut actual = [0x55u8; SCRATCH_BYTES];
                    let actual_len =
                        apply_builtin(transform, BUILTIN_WORDS, offset, length, &mut actual);
                    assert_eq!(actual_len, Some(expected_len), "transform {transform}");
                    assert_eq!(
                        actual[..expected_len],
                        expected[..expected_len],
                        "transform {transform}, length {length}, word {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn an_unknown_builtin_transform_is_refused() {
        let mut scratch = [0u8; SCRATCH_BYTES];
        assert_eq!(apply_builtin(121, BUILTIN_WORDS, 0, 4, &mut scratch), None);
        assert_eq!(
            apply_builtin(usize::MAX, BUILTIN_WORDS, 0, 4, &mut scratch),
            None
        );
    }
}

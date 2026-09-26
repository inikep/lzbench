//! Validated, owned serialized dictionary data for decoding only.
//!
//! Offsets refer into an immutable owned source buffer. The parser creates no
//! search indexes and no expanded words. Every heap request shares one budget.

use super::dictionary::{
    BUILTIN_OFFSETS_BY_LENGTH, BUILTIN_SIZE_BITS_BY_LENGTH, BUILTIN_WORDS,
    transform::{SCRATCH_BYTES, Transform},
};
use crate::{DecodeError, InvalidDataKind, dictionary::DecodeDictionaryError};
use alloc::vec::Vec;
use core::ops::Range;

type Error = DecodeDictionaryError;

#[derive(Debug, Default)]
struct WordLayout {
    bits: [u8; 32],
    offsets: [usize; 32],
}

#[derive(Debug)]
struct TransformLayout {
    strings: usize,
    offsets: [u16; 256],
    triples: usize,
    count: usize,
    params: Option<usize>,
}

#[derive(Debug, Default)]
pub(crate) struct Description {
    source: Vec<u8>,
    words: Vec<WordLayout>,
    transforms: Vec<TransformLayout>,
    combinations: Vec<(u8, u8)>,
    contexts: Option<[u8; 64]>,
    retained: usize,
}

struct Cursor<'a> {
    source: &'a [u8],
    position: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8], Error> {
        let end = self
            .position
            .checked_add(length)
            .ok_or(Error::SizeOverflow)?;
        let bytes = self
            .source
            .get(self.position..end)
            .ok_or(Error::InvalidSerializedDictionary)?;
        self.position = end;
        Ok(bytes)
    }
    fn byte(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }
    fn count(&mut self) -> Result<usize, Error> {
        let count = usize::from(self.byte()?);
        if count > 64 {
            return Err(Error::InvalidSerializedDictionary);
        }
        Ok(count)
    }
    fn varint(&mut self) -> Result<u64, Error> {
        let mut value = 0;
        for shift in (0..63).step_by(7) {
            let byte = self.byte()?;
            value |= u64::from(byte & 127) << shift;
            if byte < 128 {
                return Ok(value);
            }
        }
        Err(Error::InvalidSerializedDictionary)
    }
}

fn reserve<T>(
    vector: &mut Vec<T>,
    count: usize,
    live: &mut usize,
    limit: Option<usize>,
) -> Result<(), Error> {
    let bytes = count
        .checked_mul(size_of::<T>())
        .ok_or(Error::SizeOverflow)?;
    let next = live.checked_add(bytes).ok_or(Error::SizeOverflow)?;
    if let Some(limit) = limit
        && next > limit
    {
        return Err(Error::MemoryLimitExceeded { limit });
    }
    vector
        .try_reserve_exact(count)
        .map_err(|_| Error::AllocationFailed)?;
    *live = live
        .checked_add(vector.capacity() * size_of::<T>())
        .ok_or(Error::SizeOverflow)?;
    Ok(())
}

impl Description {
    pub(crate) const fn retained_bytes(&self) -> usize {
        self.retained
    }
    pub(crate) fn custom(&self) -> bool {
        !self.words.is_empty() || !self.transforms.is_empty()
    }

    pub(crate) fn parse(
        source: &[u8],
        existing_bytes: usize,
        limit: Option<usize>,
    ) -> Result<(Self, Range<usize>), Error> {
        let mut cursor = Cursor {
            source,
            position: 0,
        };
        if cursor.take(2)? != [0x91, 0] {
            return Err(Error::InvalidSerializedDictionary);
        }
        let prefix_len = cursor.varint()?;
        if prefix_len > (1u64 << 62) - 16 {
            return Err(Error::InvalidSerializedDictionary);
        }
        let prefix_start = cursor.position;
        cursor.take(usize::try_from(prefix_len).map_err(|_| Error::SizeOverflow)?)?;
        let prefix = prefix_start..cursor.position;
        let mut result = Self::default();
        let mut live = existing_bytes;
        let word_count = cursor.count()?;
        reserve(&mut result.words, word_count, &mut live, limit)?;
        for _ in 0..word_count {
            result.words.push(WordLayout::parse(&mut cursor)?);
        }
        let transform_count = cursor.count()?;
        reserve(&mut result.transforms, transform_count, &mut live, limit)?;
        for _ in 0..transform_count {
            result.transforms.push(TransformLayout::parse(&mut cursor)?);
        }
        if word_count != 0 || transform_count != 0 {
            let count = cursor.count()?;
            if count == 0 {
                return Err(Error::InvalidSerializedDictionary);
            }
            reserve(&mut result.combinations, count, &mut live, limit)?;
            for _ in 0..count {
                let words = cursor.byte()?;
                let transforms = cursor.byte()?;
                if usize::from(words) > word_count || usize::from(transforms) > transform_count {
                    return Err(Error::InvalidSerializedDictionary);
                }
                result.combinations.push((words, transforms));
            }
            match cursor.byte()? {
                0 => {}
                1 => {
                    let mut contexts = [0; 64];
                    contexts.copy_from_slice(cursor.take(64)?);
                    if contexts.iter().any(|&index| usize::from(index) >= count) {
                        return Err(Error::InvalidSerializedDictionary);
                    }
                    result.contexts = Some(contexts);
                }
                _ => return Err(Error::InvalidSerializedDictionary),
            }
            // C accepts a complete description followed by a suffix; the suffix
            // is not part of the effective dictionary and need not be retained.
            reserve(&mut result.source, cursor.position, &mut live, limit)?;
            result.source.extend_from_slice(&source[..cursor.position]);
        }
        result.retained = live - existing_bytes;
        Ok((result, prefix))
    }

    pub(crate) fn resolve(
        &self,
        mut address: u64,
        length: usize,
        context: usize,
        scratch: &mut [u8; SCRATCH_BYTES],
    ) -> Result<usize, DecodeError> {
        if !(4..=31).contains(&length) {
            return Err(InvalidDataKind::DictionaryReference.into());
        }
        let preferred = self
            .contexts
            .as_ref()
            .map_or(0, |map| usize::from(map[context]));
        let order = core::iter::once(preferred)
            .chain((0..self.combinations.len()).filter(|&index| index != preferred));
        for combination in order {
            let (word_id, transform_id) = self.combinations[combination];
            let words = self.words.get(usize::from(word_id));
            let transforms = self.transforms.get(usize::from(transform_id));
            let bits = words.map_or(BUILTIN_SIZE_BITS_BY_LENGTH[length], |words| {
                words.bits[length]
            });
            let transform_count = transforms.map_or(121, |transforms| transforms.count);
            let word_count = if bits == 0 { 0 } else { 1u64 << bits };
            let span = word_count * transform_count as u64;
            if address >= span {
                address -= span;
                continue;
            }
            let index = (address & (word_count - 1)) as usize;
            let offset = words.map_or(BUILTIN_OFFSETS_BY_LENGTH[length] as usize, |words| {
                words.offsets[length]
            }) + index * length;
            let source = if words.is_some() {
                &self.source
            } else {
                BUILTIN_WORDS.as_slice()
            };
            let word = &source[offset..offset + length];
            let index = (address >> bits) as usize;
            let transform = if let Some(transforms) = transforms {
                transforms.view(&self.source, index)
            } else {
                Transform::builtin(index).ok_or(DecodeError::InvalidData {
                    kind: InvalidDataKind::DictionaryReference,
                })?
            };
            return Ok(transform.apply(word, scratch));
        }
        Err(InvalidDataKind::DictionaryReference.into())
    }
}

impl WordLayout {
    fn parse(cursor: &mut Cursor<'_>) -> Result<Self, Error> {
        let mut result = Self::default();
        result.bits[4..].copy_from_slice(cursor.take(28)?);
        for length in 4..32 {
            let bits = result.bits[length];
            if bits > 15 {
                return Err(Error::InvalidSerializedDictionary);
            }
            result.offsets[length] = cursor.position;
            if bits != 0 {
                cursor.take(length << bits)?;
            }
        }
        Ok(result)
    }
}

impl TransformLayout {
    fn parse(cursor: &mut Cursor<'_>) -> Result<Self, Error> {
        let bytes = cursor.take(2)?;
        let size = usize::from(u16::from_le_bytes([bytes[0], bytes[1]]));
        let strings = cursor.position;
        let source = cursor.take(size)?;
        let mut offsets = [0u16; 256];
        let mut position = 0usize;
        let mut count = 0usize;
        loop {
            let &length = source
                .get(position)
                .ok_or(Error::InvalidSerializedDictionary)?;
            if count == offsets.len() {
                return Err(Error::InvalidSerializedDictionary);
            }
            offsets[count] = position as u16;
            count += 1;
            position += 1;
            if length == 0 {
                if position != source.len() {
                    return Err(Error::InvalidSerializedDictionary);
                }
                break;
            }
            position += usize::from(length);
        }
        let transform_count = usize::from(cursor.byte()?);
        let triples = cursor.position;
        let triples_bytes = cursor.take(transform_count * 3)?;
        let mut shifts = false;
        for triple in triples_bytes.as_chunks::<3>().0 {
            if usize::from(triple[0]) >= count || usize::from(triple[2]) >= count || triple[1] > 22
            {
                return Err(Error::InvalidSerializedDictionary);
            }
            shifts |= triple[1] >= 21;
        }
        let params = if shifts {
            let start = cursor.position;
            let parameters = cursor.take(transform_count * 2)?;
            for (triple, pair) in triples_bytes
                .as_chunks::<3>()
                .0
                .iter()
                .zip(parameters.as_chunks::<2>().0)
            {
                if triple[1] < 21 && *pair != [0, 0] {
                    return Err(Error::InvalidSerializedDictionary);
                }
            }
            Some(start)
        } else {
            None
        };
        Ok(Self {
            strings,
            offsets,
            triples,
            count: transform_count,
            params,
        })
    }

    fn view<'a>(&self, source: &'a [u8], index: usize) -> Transform<'a> {
        let triple = &source[self.triples + index * 3..][..3];
        let string = |id: u8| {
            let start = self.strings + usize::from(self.offsets[usize::from(id)]);
            &source[start + 1..start + 1 + usize::from(source[start])]
        };
        let parameter = self.params.map_or(0, |start| {
            u16::from_le_bytes([source[start + index * 2], source[start + index * 2 + 1]])
        });
        Transform {
            prefix: string(triple[0]),
            operation: triple[1],
            suffix: string(triple[2]),
            parameter,
        }
    }
}

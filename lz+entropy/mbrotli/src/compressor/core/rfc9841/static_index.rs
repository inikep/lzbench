//! Immutable custom dictionary search, in RFC combination/address order.

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::serialized::{ListRef, SerializedDictionaryData};
use super::transform::{TransformList, TransformScratch};
use super::words::WordList;
use crate::compressor::core::hq::h10::BackwardMatch;
use crate::compressor::shared::SharedBrotliError;
use crate::shared::constants::HASH_MUL32;
use crate::shared::score::{SearchResult, backward_reference_score};

#[derive(Debug)]
struct Entry {
    head: u32,
    start: usize,
    length: u16,
    code: u8,
    address: u32,
    extended: bool,
}

/// One word/transform pairing, including both reference search shapes.
#[derive(Debug)]
pub(crate) struct StaticCombination {
    words: WordList,
    transforms: TransformList,
    shallow: Box<[(u16, u8)]>,
    entries: Vec<Entry>,
    bytes: Vec<u8>,
}

/// Prepared tables own no stream state and allocate nothing during a query.
#[derive(Debug)]
pub(crate) struct StaticIndex {
    combinations: Vec<StaticCombination>,
    contexts: Option<[u8; 64]>,
    pub(crate) source_bytes: usize,
}

fn head(bytes: &[u8]) -> u32 {
    bytes
        .first_chunk::<4>()
        .map_or(0, |v| u32::from_le_bytes(*v))
}

pub(crate) fn previous(data: &[u8], position: usize, mask: usize, back: usize) -> u8 {
    data.get(position.wrapping_sub(back) & mask)
        .copied()
        .unwrap_or(0)
}

fn check(bytes: usize, limit: u64) -> Result<(), SharedBrotliError> {
    if bytes as u64 > limit {
        return Err(SharedBrotliError::SharedContextTooLarge {
            bytes: bytes as u64,
            limit,
        });
    }
    Ok(())
}

impl StaticIndex {
    /// Reads the effective prepared representation without consulting indexes or
    /// rebuilding the caller's original attachment recipe.
    #[cfg(feature = "decompression")]
    pub(crate) fn decode_word(
        &self,
        mut address: u64,
        length: usize,
        context: usize,
        scratch: &mut [u8; crate::shared::dictionary::transform::SCRATCH_BYTES],
    ) -> Result<usize, crate::DecodeError> {
        use crate::shared::dictionary::transform::Transform;
        use crate::{DecodeError, InvalidDataKind};
        if !(4..=31).contains(&length) {
            return Err(InvalidDataKind::DictionaryReference.into());
        }
        let preferred = self
            .contexts
            .as_ref()
            .map_or(0, |map| usize::from(map[context]));
        let order = core::iter::once(preferred)
            .chain((0..self.combinations.len()).filter(|&index| index != preferred));
        for index in order {
            let combination = &self.combinations[index];
            let count = combination.words.word_count(length) as u64;
            let span = count * combination.transforms.len() as u64;
            if address >= span {
                address -= span;
                continue;
            }
            let word = combination.words.word(length, (address % count) as usize);
            let index = (address / count) as usize;
            let (prefix, operation, suffix) =
                combination
                    .transforms
                    .transform(index)
                    .ok_or(DecodeError::InvalidData {
                        kind: InvalidDataKind::DictionaryReference,
                    })?;
            let transform = Transform {
                prefix: combination.transforms.stringlet(usize::from(prefix)),
                operation,
                suffix: combination.transforms.stringlet(usize::from(suffix)),
                parameter: combination.transforms.parameter(index),
            };
            return Ok(transform.apply(word, scratch));
        }
        Err(InvalidDataKind::DictionaryReference.into())
    }

    /// Checks the full transformed index budget before allocating its tables.
    pub(crate) fn prepare(
        data: &SerializedDictionaryData,
        limit: u64,
        max_transformed_bytes: u64,
        max_entries: u64,
    ) -> Result<Self, SharedBrotliError> {
        // Vec allocations cannot exceed isize::MAX, even when callers raise
        // the u64 budget. This also bounds every per-entry counting addition.
        let limit = limit.min(isize::MAX as u64);
        check(TransformList::BUILTIN_HEAP_BYTES, limit)?;
        let builtin_words = WordList::builtin();
        let builtin_transforms = TransformList::builtin();
        let mut total = data.combinations().len() * size_of::<StaticCombination>()
            + TransformList::BUILTIN_HEAP_BYTES;
        let mut counts = [(0usize, 0usize); 64];
        let mut transformed_bytes = 0usize;
        let mut entries_count = 0usize;
        let mut scratch = TransformScratch::default();
        // Two passes: the first visits transforms without retaining results.
        // Thus even an adversarial expansion is refused before allocation.
        for (combination_index, combination) in data.combinations().iter().enumerate() {
            let words = match combination.words {
                ListRef::Builtin => &builtin_words,
                ListRef::Custom(i) => &data.word_lists()[usize::from(i)],
            };
            let transforms = match combination.transforms {
                ListRef::Builtin => &builtin_transforms,
                ListRef::Custom(i) => &data.transform_lists()[usize::from(i)],
            };
            total = total.saturating_add(
                32768 * size_of::<(u16, u8)>()
                    + words.data().len()
                    + transforms.wire_len()
                    + transforms.stringlet_count() * size_of::<u16>(),
            );
            check(total, limit)?;
            for length in 4..=31 {
                for index in 0..words.word_count(length) {
                    for transform in 0..transforms.len() {
                        let output =
                            transforms.apply(transform, words.word(length, index), &mut scratch);
                        if output.len() >= 4 {
                            counts[combination_index].0 += 1;
                            counts[combination_index].1 += output.len();
                            entries_count += 1;
                            transformed_bytes += output.len();
                            check(transformed_bytes, max_transformed_bytes)?;
                            check(entries_count, max_entries)?;
                            total = total.saturating_add(size_of::<Entry>() + output.len());
                            check(total, limit)?;
                        }
                    }
                }
            }
        }
        let mut combinations = Vec::with_capacity(data.combinations().len());
        for (combination_index, combination) in data.combinations().iter().enumerate() {
            let words = match combination.words {
                ListRef::Builtin => builtin_words.clone(),
                ListRef::Custom(i) => data.word_lists()[usize::from(i)].clone(),
            };
            let transforms = match combination.transforms {
                ListRef::Builtin => builtin_transforms.clone(),
                ListRef::Custom(i) => data.transform_lists()[usize::from(i)].clone(),
            };
            let mut entries = Vec::with_capacity(counts[combination_index].0);
            let mut bytes = Vec::with_capacity(counts[combination_index].1);
            let mut shallow = vec![(0, 0); 32768].into_boxed_slice();
            for length in (4..=31).rev() {
                for index in (0..words.word_count(length)).rev() {
                    let word = words.word(length, index);
                    let bucket = ((head(word).wrapping_mul(HASH_MUL32) >> 18) as usize) * 2
                        + usize::from(length < 8);
                    shallow[bucket] = (index as u16, length as u8);
                    for transform in 0..transforms.len() {
                        let output = transforms.apply(transform, word, &mut scratch);
                        if output.len() < 4 {
                            continue;
                        }
                        entries.push(Entry {
                            head: head(output),
                            start: bytes.len(),
                            length: output.len() as u16,
                            code: length as u8,
                            address: (index + (transform << words.size_bits(length))) as u32,
                            extended: !(0..=9).any(|cut| {
                                transforms.cutoff(cut) == Some(transform)
                                    && (0..=cut).all(|n| {
                                        transforms
                                            .cutoff(n)
                                            .is_some_and(|id| id >= n * 4 && id < n * 4 + 64)
                                    })
                            }),
                        });
                        bytes.extend_from_slice(output);
                    }
                }
            }
            entries.sort_unstable_by_key(|entry| {
                (entry.head, entry.length, entry.address, entry.code)
            });
            combinations.push(StaticCombination {
                words,
                transforms,
                shallow,
                entries,
                bytes,
            });
        }
        Ok(Self {
            combinations,
            contexts: data.context_map().copied(),
            source_bytes: data
                .word_lists()
                .iter()
                .map(|w| w.data().len())
                .sum::<usize>()
                + data
                    .transform_lists()
                    .iter()
                    .map(TransformList::wire_len)
                    .sum::<usize>(),
        })
    }

    pub(crate) fn combination(&self, context: usize) -> &StaticCombination {
        &self.combinations[self
            .contexts
            .as_ref()
            .map_or(0, |map| usize::from(map[context]))]
    }

    pub(crate) fn allocated_size(&self) -> usize {
        self.combinations.capacity() * size_of::<StaticCombination>()
            + self
                .combinations
                .iter()
                .map(|c| {
                    c.words.data().len()
                        + c.transforms.wire_len()
                        + c.transforms.stringlet_count() * size_of::<u16>()
                        + size_of_val(&*c.shallow)
                        + c.entries.capacity() * size_of::<Entry>()
                        + c.bytes.capacity()
                })
                .sum::<usize>()
    }

    /// Adds the smallest address at each transformed length across all combinations.
    pub(crate) fn find_all(
        &self,
        context: usize,
        input: &[u8],
        base: usize,
        max_distance: usize,
        matches: &mut Vec<BackwardMatch>,
    ) {
        let first = self
            .contexts
            .as_ref()
            .map_or(0, |map| usize::from(map[context]));
        let mut offsets = [0usize; 32];
        let mut found = [None::<(usize, usize)>; 542];
        let min_length = matches
            .last()
            .map_or(4, |m| m.length().saturating_add(1).max(4));
        for index in
            ::core::iter::once(first).chain((0..self.combinations.len()).filter(|&i| i != first))
        {
            let combination = &self.combinations[index];
            let key = head(input);
            let start = combination.entries.partition_point(|e| e.head < key);
            for entry in combination.entries[start..]
                .iter()
                .take_while(|e| e.head == key)
            {
                let length = usize::from(entry.length);
                let code = usize::from(entry.code);
                if length < min_length
                    || !input.starts_with(&combination.bytes[entry.start..entry.start + length])
                {
                    continue;
                }
                let Some(distance) = base
                    .checked_add(1)
                    .and_then(|v| v.checked_add(offsets[code]))
                    .and_then(|v| v.checked_add(entry.address as usize))
                else {
                    continue;
                };
                if distance > max_distance {
                    continue;
                }
                let candidate = (distance, code);
                if found[length].is_none_or(|old| candidate < old) {
                    found[length] = Some(candidate);
                }
            }
            for (length, offset) in offsets.iter_mut().enumerate().skip(4) {
                *offset += combination.words.word_count(length) * combination.transforms.len();
            }
        }
        for (length, entry) in found.into_iter().enumerate() {
            if let Some((distance, code)) = entry {
                matches.push(BackwardMatch::dictionary(distance, length, code));
            }
        }
    }
}

impl StaticCombination {
    /// Searches transforms not representable by the reference's shallow cutoff table.
    pub(crate) fn probe_extended(
        &self,
        input: &[u8],
        max_length: usize,
        base: usize,
        max_distance: usize,
        out: &mut SearchResult,
    ) -> bool {
        let key = head(input);
        let start = self.entries.partition_point(|e| e.head < key);
        let mut found = false;
        for entry in self.entries[start..].iter().take_while(|e| e.head == key) {
            let length = usize::from(entry.length);
            if !entry.extended
                || length > max_length
                || !input.starts_with(&self.bytes[entry.start..entry.start + length])
            {
                continue;
            }
            let Some(distance) = base.checked_add(1 + entry.address as usize) else {
                continue;
            };
            if distance > max_distance {
                continue;
            }
            let score = backward_reference_score(length, distance);
            if score > out.score {
                *out = SearchResult {
                    len: length,
                    distance,
                    score,
                    len_code_delta: i32::from(entry.code) - length as i32,
                };
                found = true;
            }
        }
        found
    }

    /// The pinned greedy encoder probes only identity/omit-last transforms.
    pub(crate) fn probe(
        &self,
        input: &[u8],
        max_length: usize,
        base: usize,
        max_distance: usize,
        bucket_offset: usize,
        out: &mut SearchResult,
    ) -> bool {
        let bucket = ((head(input).wrapping_mul(HASH_MUL32) >> 18) as usize) * 2 + bucket_offset;
        let (index, length) = self.shallow[bucket];
        let length = usize::from(length);
        if length == 0 || length > max_length {
            return false;
        }
        let word = self.words.word(length, usize::from(index));
        let matched = word.iter().zip(input).take_while(|(a, b)| a == b).count();
        if matched == 0 {
            return false;
        }
        let cut = length - matched;
        for n in 0..=cut {
            if self
                .transforms
                .cutoff(n)
                .is_none_or(|id| id < n * 4 || id >= n * 4 + 64)
            {
                return false;
            }
        }
        let Some(transform) = self.transforms.cutoff(cut) else {
            return false;
        };
        let Some(distance) =
            base.checked_add(1 + usize::from(index) + (transform << self.words.size_bits(length)))
        else {
            return false;
        };
        if distance > max_distance {
            return false;
        }
        let score = backward_reference_score(matched, distance);
        if score < out.score {
            return false;
        }
        *out = SearchResult {
            len: matched,
            distance,
            score,
            len_code_delta: length as i32 - matched as i32,
        };
        true
    }
}

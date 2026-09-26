//! Word transformations, built in and custom.
//!
//! A static dictionary reference names a word and a transformation of it. RFC
//! 7932 fixes one list of a hundred and twenty-one transformations; [RFC 9841
//! section 3.1.1] lets a shared dictionary supply its own instead, built from
//! the same twenty-three operations over caller-supplied prefixes and suffixes.
//!
//! Ports `BrotliTransformDictionaryWord`, `ToUpperCase`, `Shift` and
//! `ComputeCutoffTransforms` from `c/common/transform.c` and
//! `c/common/shared_dictionary.c` of the pinned reference (`google/brotli`
//! v1.2.0, commit `028fb5a`). The built-in prefix, suffix and transform tables
//! beside this file are extracted from `kPrefixSuffix` and `kTransformsData` of
//! that same file; Google distributes them under the MIT licence, see
//! `brotli-ffi/vendor/brotli/LICENSE`.
//!
//! The reference stores its tables as pointers into either static storage or
//! the caller's serialized bytes, so it never copies them. [`TransformList`]
//! keeps that property with [`Cow`]: the built-in list borrows the statics
//! below and costs nothing to construct, and a parsed list owns the bytes it
//! was decoded from.
//!
//! [RFC 9841 section 3.1.1]:
//!     https://www.rfc-editor.org/rfc/rfc9841.html#section-3.1.1

use alloc::boxed::Box;
use alloc::vec::Vec;

use alloc::borrow::Cow;

use thiserror::Error;

/// Operations a transform may apply (`BROTLI_NUM_TRANSFORM_TYPES`).
pub(crate) const NUM_TRANSFORM_TYPES: u8 = 23;

/// Transform id of `OmitLast9`, the largest cut a cutoff table records.
pub(crate) const MAX_CUT_OFF: usize = 9;

/// Transform id of `ShiftFirst`.
const SHIFT_FIRST: u8 = 21;

/// Transform id of `ShiftAll`.
const SHIFT_ALL: u8 = 22;

/// Longest prefix or suffix a stringlet may hold.
pub(crate) const MAX_STRINGLET_BYTES: usize = 255;

/// Stringlets one transform list may hold, terminator included.
///
/// [RFC 9841 section 5] puts `NUM_PREFIX_SUFFIX` in the range one to
/// two hundred and fifty-six, the last of which is always the zero-length
/// terminator.
///
/// [RFC 9841 section 5]: https://www.rfc-editor.org/rfc/rfc9841.html#section-5
pub(crate) const MAX_STRINGLETS: usize = 256;

/// Longest word a static dictionary may hold
/// (`SHARED_BROTLI_MAX_DICTIONARY_WORD_LENGTH`).
pub(crate) const MAX_WORD_LENGTH: usize = 31;

/// Longest output any transform can produce, prefix and suffix included.
///
/// Also the size of a [`TransformScratch`], which is deliberately larger than
/// the longest output: `ToUpperCase` may write one byte past the end of the
/// word it is given, which the reference relies on and the suffix then
/// overwrites. Sizing the scratch for the worst case rather than the exact one
/// is what lets the port keep that behaviour without an out-of-bounds write.
pub(crate) const MAX_TRANSFORMED_WORD_BYTES: usize = 2 * MAX_STRINGLET_BYTES + MAX_WORD_LENGTH;

/// Prefix and suffix stringlets of the RFC 7932 transform list (`kPrefixSuffix`).
static BUILTIN_PREFIX_SUFFIX: &[u8; 217] =
    include_bytes!("../../../shared/dictionary/builtin_prefix_suffix.bin");

/// Prefix, operation and suffix of each RFC 7932 transform (`kTransformsData`).
static BUILTIN_TRIPLES: &[u8; 363] =
    include_bytes!("../../../shared/dictionary/builtin_transforms.bin");

/// Stringlets the RFC 7932 prefix and suffix block holds (`kPrefixSuffixMap`).
const BUILTIN_STRINGLET_COUNT: usize = 50;

/// Offset of each RFC 7932 stringlet's length byte, derived at compile time.
const BUILTIN_STRINGLETS: [u16; BUILTIN_STRINGLET_COUNT] = builtin_stringlets();

/// Cutoff transform per cut for the RFC 7932 list, derived at compile time.
const BUILTIN_CUTOFF: [Option<u16>; MAX_CUT_OFF + 1] = builtin_cutoff();

/// Walks the built-in stringlet block the way [`index_stringlets`] does.
const fn builtin_stringlets() -> [u16; BUILTIN_STRINGLET_COUNT] {
    let mut offsets = [0u16; BUILTIN_STRINGLET_COUNT];
    let mut offset = 0usize;
    let mut index = 0usize;
    while index < BUILTIN_STRINGLET_COUNT {
        offsets[index] = offset as u16;
        offset += 1 + BUILTIN_PREFIX_SUFFIX[offset] as usize;
        index += 1;
    }
    offsets
}

/// Finds the lowest bare cut of `n` trailing bytes, as `compute_cutoffs` does.
const fn builtin_cutoff() -> [Option<u16>; MAX_CUT_OFF + 1] {
    let mut cutoff = [None; MAX_CUT_OFF + 1];
    let mut index = 0usize;
    while index * 3 < BUILTIN_TRIPLES.len() {
        let prefix = BUILTIN_TRIPLES[index * 3] as usize;
        let operation = BUILTIN_TRIPLES[index * 3 + 1] as usize;
        let suffix = BUILTIN_TRIPLES[index * 3 + 2] as usize;
        if operation <= MAX_CUT_OFF
            && cutoff[operation].is_none()
            && BUILTIN_PREFIX_SUFFIX[BUILTIN_STRINGLETS[prefix] as usize] == 0
            && BUILTIN_PREFIX_SUFFIX[BUILTIN_STRINGLETS[suffix] as usize] == 0
        {
            cutoff[operation] = Some(index as u16);
        }
        index += 1;
    }
    cutoff
}

/// Why a transform list could not be built from its parts.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum TransformListError {
    /// The prefix and suffix block was empty, so it cannot hold a terminator.
    #[error("a transform list's prefix and suffix data must hold at least a terminator")]
    EmptyStringlets,
    /// A stringlet claimed more bytes than the block holds.
    #[error("a stringlet of {length} bytes overruns the prefix and suffix data")]
    StringletOverrun {
        /// The length the stringlet claimed.
        length: usize,
    },
    /// The zero-length terminator was absent, or was not the final byte.
    #[error("the zero-length stringlet must be the last byte of the prefix and suffix data")]
    MisplacedTerminator,
    /// More stringlets than the format allows.
    #[error("a transform list holds at most {MAX_STRINGLETS} stringlets")]
    TooManyStringlets,
    /// More transforms than a one-byte count can express.
    #[error("a transform list holds at most 255 transforms, not {count}")]
    TooManyTransforms {
        /// How many transforms were offered.
        count: usize,
    },
    /// A transform named a stringlet that does not exist.
    #[error("transform {index} refers to stringlet {stringlet}, past the {count} defined")]
    UndefinedStringlet {
        /// Which transform named it.
        index: usize,
        /// The stringlet it named.
        stringlet: usize,
        /// How many stringlets exist.
        count: usize,
    },
    /// A transform named an operation the RFC does not define.
    #[error("transform {index} uses operation {operation}, past the {NUM_TRANSFORM_TYPES} defined")]
    UndefinedOperation {
        /// Which transform named it.
        index: usize,
        /// The operation it named.
        operation: u8,
    },
    /// The parameter block was neither absent nor two bytes per transform.
    #[error("a parameter block holds two bytes per transform: expected {expected}, found {found}")]
    ParameterLength {
        /// How many bytes were expected.
        expected: usize,
        /// How many were offered.
        found: usize,
    },
    /// A transform that does not shift was given a non-zero parameter.
    #[error("transform {index} does not shift, so its parameter must be zero, not {parameter}")]
    UnusedParameter {
        /// Which transform carried it.
        index: usize,
        /// The parameter it carried.
        parameter: u16,
    },
}

/// A reusable buffer one transformed word is written into.
///
/// Held by the caller across every candidate so a search allocates nothing per
/// word, which is what [section 37.4 of the redesign specification] requires.
/// Larger than the longest transform output on purpose; see
/// [`MAX_TRANSFORMED_WORD_BYTES`].
///
/// [section 37.4 of the redesign specification]: crate::dictionary
#[derive(Clone)]
pub(crate) struct TransformScratch {
    /// Working bytes, only the returned prefix of which is ever meaningful.
    bytes: [u8; MAX_TRANSFORMED_WORD_BYTES],
}

impl Default for TransformScratch {
    /// Returns a zeroed scratch buffer.
    fn default() -> Self {
        Self {
            bytes: [0; MAX_TRANSFORMED_WORD_BYTES],
        }
    }
}

impl ::core::fmt::Debug for TransformScratch {
    /// Prints the buffer's size rather than its contents, which are scratch.
    fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        formatter
            .debug_struct("TransformScratch")
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

/// One list of word transformations, built in or custom.
///
/// Immutable once built. The wire layout it mirrors is
/// [RFC 9841 section 5]'s transform list: a block of length-prefixed
/// stringlets, a triple of prefix, operation and suffix per transform, and a
/// parameter block present only when some transform shifts.
///
/// [RFC 9841 section 5]: https://www.rfc-editor.org/rfc/rfc9841.html#section-5
#[derive(Debug, Clone)]
pub(crate) struct TransformList {
    /// The stringlet block, exactly as it appears on the wire.
    prefix_suffix: Cow<'static, [u8]>,
    /// Offset of each stringlet's length byte inside `prefix_suffix`.
    stringlets: Box<[u16]>,
    /// Prefix id, operation and suffix id, three bytes per transform.
    triples: Cow<'static, [u8]>,
    /// Two bytes per transform, little-endian, or empty when none shifts.
    params: Cow<'static, [u8]>,
    /// Lowest transform that is a bare cut of `n` trailing bytes, per `n`.
    cutoff: [Option<u16>; MAX_CUT_OFF + 1],
}

impl TransformList {
    /// Heap scratch used when constructing the borrowed built-in transform list.
    pub(crate) const BUILTIN_HEAP_BYTES: usize = size_of_val(&BUILTIN_STRINGLETS);

    /// Returns the RFC 7932 transform list every ordinary stream uses.
    ///
    /// Borrows its tables from static storage, so this allocates only the
    /// stringlet offsets — fifty entries — and never the tables themselves.
    /// The offsets and the cutoff table are computed at compile time from the
    /// same bytes [`TransformList::from_parts`] would walk, so the built-in
    /// list needs no fallible step and no unreachable fallback; a test below
    /// re-derives both through `from_parts` and checks they agree.
    pub(crate) fn builtin() -> Self {
        Self {
            prefix_suffix: Cow::Borrowed(&BUILTIN_PREFIX_SUFFIX[..]),
            stringlets: Box::new(BUILTIN_STRINGLETS),
            triples: Cow::Borrowed(&BUILTIN_TRIPLES[..]),
            params: Cow::Borrowed(&[]),
            cutoff: BUILTIN_CUTOFF,
        }
    }

    /// Validates the three wire blocks and indexes the stringlets.
    ///
    /// `params` is either empty, meaning no transform shifts, or two bytes per
    /// transform in little-endian order.
    ///
    /// # Errors
    ///
    /// Returns the [`TransformListError`] naming the first rule broken. Every
    /// check the reference's `ParsePrefixSuffixTable` and `ParseTransformsList`
    /// make is made here, in the same order.
    pub(crate) fn from_parts(
        prefix_suffix: Cow<'static, [u8]>,
        triples: Cow<'static, [u8]>,
        params: Cow<'static, [u8]>,
    ) -> Result<Self, TransformListError> {
        let stringlets = index_stringlets(&prefix_suffix)?;
        let count = triples.len() / 3;
        if !triples.len().is_multiple_of(3) || count > 255 {
            return Err(TransformListError::TooManyTransforms { count });
        }
        let mut shifts = false;
        for index in 0..count {
            let (prefix_id, operation, suffix_id) = triple(&triples, index);
            for stringlet in [prefix_id, suffix_id] {
                if usize::from(stringlet) >= stringlets.len() {
                    return Err(TransformListError::UndefinedStringlet {
                        index,
                        stringlet: usize::from(stringlet),
                        count: stringlets.len(),
                    });
                }
            }
            if operation >= NUM_TRANSFORM_TYPES {
                return Err(TransformListError::UndefinedOperation { index, operation });
            }
            shifts |= operation == SHIFT_FIRST || operation == SHIFT_ALL;
        }
        // The wire carries a parameter block if and only if some transform
        // shifts, so a list that does not shift must not hold one and a list
        // that does must hold exactly two bytes per transform.
        let expected = if shifts { count * 2 } else { 0 };
        if params.len() != expected {
            return Err(TransformListError::ParameterLength {
                expected,
                found: params.len(),
            });
        }
        for index in 0..count {
            let (_, operation, _) = triple(&triples, index);
            let parameter = parameter_at(&params, index);
            if operation != SHIFT_FIRST && operation != SHIFT_ALL && parameter != 0 {
                return Err(TransformListError::UnusedParameter { index, parameter });
            }
        }
        let mut list = Self {
            prefix_suffix,
            stringlets,
            triples,
            params,
            cutoff: [None; MAX_CUT_OFF + 1],
        };
        list.compute_cutoffs();
        Ok(list)
    }

    /// Records the lowest bare "omit the last `n` bytes" transform per `n`.
    ///
    /// Mirrors `ComputeCutoffTransforms`. A transform qualifies only when both
    /// its prefix and its suffix are empty, because the encoder emits a cutoff
    /// match by shortening the copy and naming this transform, with nothing
    /// added on either side.
    fn compute_cutoffs(&mut self) {
        let mut cutoff = [None; MAX_CUT_OFF + 1];
        for index in 0..self.len() {
            let (prefix_id, operation, suffix_id) = triple(&self.triples, index);
            let Some(slot) = cutoff.get_mut(usize::from(operation)) else {
                continue;
            };
            if slot.is_some()
                || !self.stringlet(usize::from(prefix_id)).is_empty()
                || !self.stringlet(usize::from(suffix_id)).is_empty()
            {
                continue;
            }
            // A validated list holds at most 255 transforms, so this fits.
            *slot = u16::try_from(index).ok();
        }
        self.cutoff = cutoff;
    }

    /// Returns how many transforms the list defines.
    pub(crate) fn len(&self) -> usize {
        self.triples.len() / 3
    }

    /// Returns how many stringlets the list defines, terminator included.
    pub(crate) const fn stringlet_count(&self) -> usize {
        self.stringlets.len()
    }

    /// Returns one stringlet's bytes, or an empty slice for an unknown id.
    pub(crate) fn stringlet(&self, id: usize) -> &[u8] {
        let Some(&offset) = self.stringlets.get(id) else {
            return &[];
        };
        let start = usize::from(offset);
        let Some(&length) = self.prefix_suffix.get(start) else {
            return &[];
        };
        let body = start + 1;
        self.prefix_suffix
            .get(body..body + usize::from(length))
            .unwrap_or_default()
    }

    /// Returns the prefix id, operation and suffix id of one transform.
    ///
    /// `None` past the end of the list. The reference reads past its own
    /// transform array here and would report whatever followed it; refusing to
    /// is what keeps an out-of-range index from silently naming transform zero's
    /// prefix and suffix.
    pub(crate) fn transform(&self, index: usize) -> Option<(u8, u8, u8)> {
        if index >= self.len() {
            return None;
        }
        Some(triple(&self.triples, index))
    }

    /// Returns one transform's shift parameter, zero when it does not shift.
    pub(crate) fn parameter(&self, index: usize) -> u16 {
        parameter_at(&self.params, index)
    }

    /// Returns whether the list carries a parameter block on the wire.
    #[cfg(test)]
    pub(crate) fn has_params(&self) -> bool {
        !self.params.is_empty()
    }

    /// Returns the transform that cuts `n` bytes off the end and adds nothing.
    ///
    /// `None` when the list defines no such transform, which is what makes a
    /// cutoff match unavailable at that length.
    pub(crate) fn cutoff(&self, cut: usize) -> Option<usize> {
        self.cutoff.get(cut).copied().flatten().map(usize::from)
    }

    /// Applies one transform to `word`, writing into `scratch`.
    ///
    /// Returns the transformed bytes, which borrow the scratch buffer until the
    /// next call. Mirrors `BrotliTransformDictionaryWord` exactly, including
    /// the reference's habit of letting `ToUpperCase` write one byte past the
    /// word when the word ends mid-sequence: those bytes are always either
    /// overwritten by the suffix or left outside the returned slice, so the
    /// output is identical and nothing is written out of bounds.
    ///
    /// A `word` longer than [`MAX_WORD_LENGTH`] is truncated to it, which no
    /// caller can reach: a word list holds no longer word. An `index` past the
    /// end of the list copies the word unchanged.
    pub(crate) fn apply<'s>(
        &self,
        index: usize,
        word: &[u8],
        scratch: &'s mut TransformScratch,
    ) -> &'s [u8] {
        let word = &word[..word.len().min(MAX_WORD_LENGTH)];
        let Some((prefix_id, operation, suffix_id)) = self.transform(index) else {
            scratch.bytes[..word.len()].copy_from_slice(word);
            return &scratch.bytes[..word.len()];
        };
        let prefix = self.stringlet(usize::from(prefix_id));
        let suffix = self.stringlet(usize::from(suffix_id));

        let transform = crate::shared::dictionary::transform::Transform {
            prefix,
            operation,
            suffix,
            parameter: self.parameter(index),
        };
        let end = transform.apply(word, &mut scratch.bytes);
        &scratch.bytes[..end]
    }

    /// Returns how many bytes [`TransformList::serialize`] will append.
    pub(crate) fn wire_len(&self) -> usize {
        2 + self.prefix_suffix.len() + 1 + self.triples.len() + self.params.len()
    }

    /// Appends the list to `out` in the RFC 9841 transform list layout.
    ///
    /// Canonical: the stringlet block and the triples are written exactly as
    /// they are held, and the parameter block is written if and only if some
    /// transform shifts, which is the only encoding the RFC permits.
    pub(crate) fn serialize(&self, out: &mut Vec<u8>) {
        // A well-formed stringlet block is at most 256 * 256 bytes, which fits
        // the `u16` the wire uses; `from_parts` is what guarantees it.
        let length = u16::try_from(self.prefix_suffix.len()).unwrap_or(u16::MAX);
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(&self.prefix_suffix);
        out.push(u8::try_from(self.len()).unwrap_or(u8::MAX));
        out.extend_from_slice(&self.triples);
        out.extend_from_slice(&self.params);
    }
}

/// Returns the prefix id, operation and suffix id stored at `index`.
///
/// Reads out of range as the identity transform, which keeps every caller free
/// of a bounds check the validated length has already made impossible.
fn triple(triples: &[u8], index: usize) -> (u8, u8, u8) {
    match triples.get(index * 3..).and_then(<[u8]>::first_chunk::<3>) {
        Some(&[prefix, operation, suffix]) => (prefix, operation, suffix),
        None => (0, 0, 0),
    }
}

/// Returns the little-endian parameter stored at `index`, or zero.
fn parameter_at(params: &[u8], index: usize) -> u16 {
    match params.get(index * 2..).and_then(<[u8]>::first_chunk::<2>) {
        Some(chunk) => u16::from_le_bytes(*chunk),
        None => 0,
    }
}

/// Walks a stringlet block and returns the offset of every length byte.
///
/// Mirrors `ParsePrefixSuffixTable`: the block must end with a zero-length
/// stringlet and that terminator must be its very last byte.
fn index_stringlets(block: &[u8]) -> Result<Box<[u16]>, TransformListError> {
    if block.is_empty() {
        return Err(TransformListError::EmptyStringlets);
    }
    let mut offsets = Vec::new();
    let mut offset = 0usize;
    loop {
        let Some(&length) = block.get(offset) else {
            return Err(TransformListError::MisplacedTerminator);
        };
        // The block is at most `u16::MAX` bytes long, which `from_parts` and
        // the wire's own two-byte length field both guarantee.
        let Ok(position) = u16::try_from(offset) else {
            return Err(TransformListError::StringletOverrun { length: offset });
        };
        offsets.push(position);
        offset += 1;
        if length == 0 {
            return if offset == block.len() {
                Ok(offsets.into_boxed_slice())
            } else {
                Err(TransformListError::MisplacedTerminator)
            };
        }
        if offsets.len() >= MAX_STRINGLETS {
            return Err(TransformListError::TooManyStringlets);
        }
        offset += usize::from(length);
        if offset >= block.len() {
            return Err(TransformListError::StringletOverrun {
                length: usize::from(length),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    /// Cutoff transform ids of the RFC 7932 list, from `ComputeCutoffTransforms`.
    const BUILTIN_CUTOFFS: [i32; MAX_CUT_OFF + 1] = [0, 12, 27, 23, 42, 63, 56, 48, 59, 64];

    /// Packed cutoff table the built-in dictionary search already carries.
    const PACKED_CUTOFFS: u64 = 0x071B_520A_DA2D_3200;

    /// Applies one transform through a fresh scratch buffer.
    fn apply(list: &TransformList, index: usize, word: &[u8]) -> Vec<u8> {
        let mut scratch = TransformScratch::default();
        list.apply(index, word, &mut scratch).to_vec()
    }

    /// Builds a one-transform list from raw parts.
    fn one(prefix: &[u8], operation: u8, suffix: &[u8], parameter: u16) -> TransformList {
        let mut block = Vec::new();
        let mut ids = [0u8; 2];
        for (slot, string) in ids.iter_mut().zip([prefix, suffix]) {
            if string.is_empty() {
                continue;
            }
            *slot = u8::try_from(block.iter().filter(|_| false).count()).unwrap_or(0);
            *slot = count_stringlets(&block);
            block.push(u8::try_from(string.len()).expect("short enough"));
            block.extend_from_slice(string);
        }
        let terminator = count_stringlets(&block);
        for slot in &mut ids {
            if *slot == 0 && block.is_empty() {
                *slot = terminator;
            }
        }
        let (prefix_id, suffix_id) = (
            if prefix.is_empty() {
                terminator
            } else {
                ids[0]
            },
            if suffix.is_empty() {
                terminator
            } else {
                ids[1]
            },
        );
        block.push(0);
        let params = if operation == SHIFT_FIRST || operation == SHIFT_ALL {
            parameter.to_le_bytes().to_vec()
        } else {
            Vec::new()
        };
        TransformList::from_parts(
            Cow::Owned(block),
            Cow::Owned(vec![prefix_id, operation, suffix_id]),
            Cow::Owned(params),
        )
        .expect("the parts are well formed")
    }

    /// Counts the stringlets a partially built block already holds.
    fn count_stringlets(block: &[u8]) -> u8 {
        let mut offset = 0usize;
        let mut count = 0u8;
        while let Some(&length) = block.get(offset) {
            count += 1;
            offset += 1 + usize::from(length);
        }
        count
    }

    #[test]
    fn the_builtin_list_is_the_reference_one() {
        let list = TransformList::builtin();

        assert_eq!(list.len(), 121);
        assert_eq!(list.stringlet_count(), BUILTIN_STRINGLET_COUNT);
        assert!(!list.has_params());
    }

    #[test]
    fn the_builtin_tables_are_what_parsing_them_would_produce() {
        // `builtin` skips the validating constructor, so the compile-time
        // stringlet offsets and cutoff table are checked against the ones the
        // parser derives from the same bytes.
        let parsed = TransformList::from_parts(
            Cow::Borrowed(&BUILTIN_PREFIX_SUFFIX[..]),
            Cow::Borrowed(&BUILTIN_TRIPLES[..]),
            Cow::Borrowed(&[]),
        )
        .expect("the reference's own tables are well formed");
        let builtin = TransformList::builtin();

        // Called rather than read, so the const evaluation and the runtime one
        // are both exercised and both compared against the parser's.
        assert_eq!(parsed.stringlets.as_ref(), builtin_stringlets());
        assert_eq!(parsed.cutoff, builtin_cutoff());
        assert_eq!(builtin.stringlets.as_ref(), builtin_stringlets());
        assert_eq!(builtin.cutoff, builtin_cutoff());
        for index in 0..parsed.len() {
            assert_eq!(parsed.transform(index), builtin.transform(index));
        }
    }

    #[test]
    fn the_builtin_cutoffs_match_the_packed_table() {
        let list = TransformList::builtin();

        for (cut, &expected) in BUILTIN_CUTOFFS.iter().enumerate() {
            let packed = (cut << 2) + ((PACKED_CUTOFFS >> (cut * 6)) & 0x3F) as usize;
            assert_eq!(packed as i32, expected, "cut {cut}");
            assert_eq!(list.cutoff(cut), Some(expected as usize), "cut {cut}");
        }
    }

    #[test]
    fn the_identity_transform_copies_the_word() {
        let list = TransformList::builtin();

        assert_eq!(apply(&list, 0, b"word"), b"word");
    }

    #[test]
    fn a_prefix_and_suffix_surround_the_word() {
        let list = one(b"<<", 0, b">>", 0);

        assert_eq!(apply(&list, 0, b"tag"), b"<<tag>>");
    }

    #[test]
    fn omitting_the_last_bytes_shortens_the_word() {
        for cut in 1..=9u8 {
            let list = one(b"", cut, b"", 0);
            let expected = &b"abcdefghijkl"[..12 - usize::from(cut)];

            assert_eq!(apply(&list, 0, b"abcdefghijkl"), expected, "cut {cut}");
        }
    }

    #[test]
    fn omitting_more_than_the_word_holds_leaves_nothing() {
        let list = one(b"[", 9, b"]", 0);

        assert_eq!(apply(&list, 0, b"abcd"), b"[]");
    }

    #[test]
    fn omitting_the_first_bytes_shortens_the_word() {
        for skip in 1..=9u8 {
            let list = one(b"", 11 + skip, b"", 0);
            let expected = &b"abcdefghijkl"[usize::from(skip)..];

            assert_eq!(apply(&list, 0, b"abcdefghijkl"), expected, "skip {skip}");
        }
    }

    #[test]
    fn fermenting_uppercases_ascii() {
        let first = one(b"", 10, b"", 0);
        let all = one(b"", 11, b"", 0);

        assert_eq!(apply(&first, 0, b"word here"), b"Word here");
        assert_eq!(apply(&all, 0, b"word here"), b"WORD HERE");
    }

    #[test]
    fn fermenting_leaves_a_non_letter_alone() {
        let all = one(b"", 11, b"", 0);

        assert_eq!(apply(&all, 0, b"1234"), b"1234");
    }

    #[test]
    fn fermenting_a_two_byte_rune_flips_the_reference_bit() {
        // The format's casing model is an exclusive-or, not a Unicode mapping:
        // the second byte of a two-byte sequence has bit five flipped.
        let all = one(b"", 11, b"", 0);
        let word = [0xC3, 0xA9, 0xC3, 0xA9];

        assert_eq!(apply(&all, 0, &word), vec![0xC3, 0x89, 0xC3, 0x89]);
    }

    #[test]
    fn shifting_moves_an_ascii_scalar() {
        let first = one(b"", SHIFT_FIRST, b"", 1);
        let all = one(b"", SHIFT_ALL, b"", 1);

        assert_eq!(apply(&first, 0, b"abcd"), b"bbcd");
        assert_eq!(apply(&all, 0, b"abcd"), b"bcde");
    }

    #[test]
    fn shifting_wraps_inside_seven_bits() {
        let all = one(b"", SHIFT_ALL, b"", 1);

        assert_eq!(
            apply(&all, 0, &[0x7F, 0x41, 0x41, 0x41]),
            vec![0x00, 0x42, 0x42, 0x42]
        );
    }

    #[test]
    fn a_negative_shift_comes_from_the_sign_extension() {
        // 0xFFFF sign-extends to -1, so every scalar moves back one.
        let all = one(b"", SHIFT_ALL, b"", 0xFFFF);

        assert_eq!(apply(&all, 0, b"bcde"), b"abcd");
    }

    #[test]
    fn shifting_a_multibyte_rune_keeps_its_encoding() {
        let all = one(b"", SHIFT_ALL, b"", 1);
        let shifted = apply(&all, 0, &[0xC3, 0xA9, 0x61, 0x62]);

        // U+00E9 becomes U+00EA, still a two-byte sequence, and the ASCII
        // bytes after it each move on by one.
        assert_eq!(shifted, vec![0xC3, 0xAA, 0x62, 0x63]);
    }

    #[test]
    fn shifting_a_truncated_rune_stops_at_the_word_end() {
        // A three-byte lead with only two bytes left is left untouched and ends
        // the walk, which is what the RFC means by marking the rest transformed.
        let all = one(b"", SHIFT_ALL, b"", 1);

        assert_eq!(apply(&all, 0, &[0xE0, 0x80]), vec![0xE0, 0x80]);
    }

    #[test]
    fn an_empty_prefix_and_suffix_share_the_terminator() {
        let list = one(b"", 0, b"", 0);

        assert_eq!(list.stringlet_count(), 1);
        assert_eq!(list.stringlet(0), b"");
    }

    #[test]
    fn a_block_without_a_terminator_is_refused() {
        let outcome = TransformList::from_parts(
            Cow::Owned(vec![1, b'a']),
            Cow::Owned(vec![]),
            Cow::Owned(vec![]),
        );

        assert_eq!(
            outcome.unwrap_err(),
            TransformListError::StringletOverrun { length: 1 }
        );
    }

    #[test]
    fn a_terminator_before_the_end_is_refused() {
        let outcome = TransformList::from_parts(
            Cow::Owned(vec![0, 0]),
            Cow::Owned(vec![]),
            Cow::Owned(vec![]),
        );

        assert_eq!(
            outcome.unwrap_err(),
            TransformListError::MisplacedTerminator
        );
    }

    #[test]
    fn an_empty_block_is_refused() {
        let outcome =
            TransformList::from_parts(Cow::Owned(vec![]), Cow::Owned(vec![]), Cow::Owned(vec![]));

        assert_eq!(outcome.unwrap_err(), TransformListError::EmptyStringlets);
    }

    #[test]
    fn a_stringlet_running_past_the_block_is_refused() {
        let outcome = TransformList::from_parts(
            Cow::Owned(vec![9, b'a', b'b', 0]),
            Cow::Owned(vec![]),
            Cow::Owned(vec![]),
        );

        assert_eq!(
            outcome.unwrap_err(),
            TransformListError::StringletOverrun { length: 9 }
        );
    }

    #[test]
    fn a_transform_naming_a_missing_stringlet_is_refused() {
        let outcome = TransformList::from_parts(
            Cow::Owned(vec![0]),
            Cow::Owned(vec![4, 0, 0]),
            Cow::Owned(vec![]),
        );

        assert_eq!(
            outcome.unwrap_err(),
            TransformListError::UndefinedStringlet {
                index: 0,
                stringlet: 4,
                count: 1,
            }
        );
    }

    #[test]
    fn an_undefined_operation_is_refused() {
        let outcome = TransformList::from_parts(
            Cow::Owned(vec![0]),
            Cow::Owned(vec![0, NUM_TRANSFORM_TYPES, 0]),
            Cow::Owned(vec![]),
        );

        assert_eq!(
            outcome.unwrap_err(),
            TransformListError::UndefinedOperation {
                index: 0,
                operation: NUM_TRANSFORM_TYPES,
            }
        );
    }

    #[test]
    fn a_shift_without_parameters_is_refused() {
        let outcome = TransformList::from_parts(
            Cow::Owned(vec![0]),
            Cow::Owned(vec![0, SHIFT_ALL, 0]),
            Cow::Owned(vec![]),
        );

        assert_eq!(
            outcome.unwrap_err(),
            TransformListError::ParameterLength {
                expected: 2,
                found: 0,
            }
        );
    }

    #[test]
    fn parameters_without_a_shift_are_refused() {
        let outcome = TransformList::from_parts(
            Cow::Owned(vec![0]),
            Cow::Owned(vec![0, 0, 0]),
            Cow::Owned(vec![0, 0]),
        );

        assert_eq!(
            outcome.unwrap_err(),
            TransformListError::ParameterLength {
                expected: 0,
                found: 2,
            }
        );
    }

    #[test]
    fn a_non_zero_parameter_on_a_non_shift_is_refused() {
        let outcome = TransformList::from_parts(
            Cow::Owned(vec![0]),
            Cow::Owned(vec![0, 0, 0, 0, SHIFT_ALL, 0]),
            Cow::Owned(vec![7, 0, 1, 0]),
        );

        assert_eq!(
            outcome.unwrap_err(),
            TransformListError::UnusedParameter {
                index: 0,
                parameter: 7,
            }
        );
    }

    #[test]
    fn a_ragged_triple_block_is_refused() {
        let outcome = TransformList::from_parts(
            Cow::Owned(vec![0]),
            Cow::Owned(vec![0, 0]),
            Cow::Owned(vec![]),
        );

        assert_eq!(
            outcome.unwrap_err(),
            TransformListError::TooManyTransforms { count: 0 }
        );
    }

    #[test]
    fn serializing_a_list_round_trips_through_its_own_layout() {
        let list = one(b"pre", SHIFT_ALL, b"post", 0x1234);
        let mut bytes = Vec::new();
        list.serialize(&mut bytes);

        assert_eq!(bytes.len(), list.wire_len());
        // Two length bytes, the block, the count, the triple, the parameters.
        assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]) as usize, 10);
        assert_eq!(bytes[12], 1);
        assert_eq!(&bytes[16..], &[0x34, 0x12]);
    }

    #[test]
    fn a_cutoff_needs_an_empty_prefix_and_suffix() {
        let with_prefix = one(b"x", 1, b"", 0);
        let bare = one(b"", 1, b"", 0);

        assert_eq!(with_prefix.cutoff(1), None);
        assert_eq!(bare.cutoff(1), Some(0));
    }

    #[test]
    fn the_lowest_matching_cutoff_wins() {
        let list = TransformList::from_parts(
            Cow::Owned(vec![0]),
            Cow::Owned(vec![0, 2, 0, 0, 2, 0]),
            Cow::Owned(vec![]),
        )
        .expect("well formed");

        assert_eq!(list.cutoff(2), Some(0));
    }

    #[test]
    fn a_word_longer_than_the_format_allows_is_truncated() {
        let list = one(b"", 0, b"", 0);
        let word = [b'z'; MAX_WORD_LENGTH + 5];

        assert_eq!(apply(&list, 0, &word).len(), MAX_WORD_LENGTH);
    }

    #[test]
    fn an_index_past_the_end_is_the_identity() {
        let list = one(b"<", 11, b">", 0);

        assert_eq!(apply(&list, 9, b"word"), b"word");
        assert_eq!(list.transform(9), None);
        assert_eq!(list.parameter(9), 0);
    }

    #[test]
    fn the_longest_possible_transform_fits_the_scratch_buffer() {
        let long = vec![b'p'; MAX_STRINGLET_BYTES];
        let list = one(&long, 0, &long, 0);
        let word = [b'w'; MAX_WORD_LENGTH];

        assert_eq!(apply(&list, 0, &word).len(), MAX_TRANSFORMED_WORD_BYTES);
    }

    #[test]
    fn the_scratch_buffer_reports_its_size_rather_than_its_bytes() {
        let scratch = TransformScratch::default();

        assert!(format!("{scratch:?}").contains(&MAX_TRANSFORMED_WORD_BYTES.to_string()));
    }
}

//! Every static-dictionary word that matches at one position.
//!
//! Ports `BrotliFindAllStaticDictionaryMatchesFor` from `c/enc/static_dict.c`,
//! together with the word and bucket tables of `c/enc/static_dict_lut.c`, from
//! the pinned reference (`google/brotli` v1.2.0, commit `028fb5a`). The tables
//! are Google's, distributed under the MIT licence; see
//! `brotli-ffi/vendor/brotli/LICENSE`.
//!
//! The greedy qualities ask the dictionary for one best word. Qualities ten and
//! eleven cannot: their dynamic program prices a copy at every length, so it
//! needs the cheapest word *per length*. That is what this returns — a table
//! indexed by match length, each entry the smallest `(distance << 5) | code`
//! seen at that length, which is the reference's way of preferring the nearest
//! word among equals.
//!
//! The transform cases below are spelled out rather than derived. RFC 7932
//! fixes a hundred and twenty-one transforms, and the reference recognises a
//! chosen subset of them by literal comparison against the bytes that follow
//! the word; reproducing that subset exactly is the only way to reproduce its
//! output.

use super::{
    BUILTIN_OFFSETS_BY_LENGTH, BUILTIN_SIZE_BITS_BY_LENGTH, BUILTIN_WORDS, CUTOFF_TRANSFORMS,
    MAX_STATIC_DICTIONARY_MATCH_LEN, common_prefix_len,
};
use crate::shared::constants::HASH_MUL32;

/// Value marking "no word of this length matched" (`kInvalidMatch`).
pub(crate) const INVALID_MATCH: u32 = 0x0FFF_FFFF;

/// Number of buckets in the word lookup (`..._LUT_NUM_BUCKETS`).
const NUM_BUCKETS: usize = 32_768;

/// Number of entries in the word lookup (`..._LUT_NUM_ITEMS`).
const NUM_ITEMS: usize = 31_705;

/// Bucket heads, little-endian (`kStaticDictionaryBuckets`).
///
/// A bucket of zero is empty; otherwise it indexes [`WORD_ITEMS`].
static BUCKETS: &[u8; 2 * NUM_BUCKETS] = include_bytes!("lut_buckets.bin");

/// Word entries, four bytes each (`kStaticDictionaryWords`).
///
/// Byte zero is the word length with the high bit set on the last entry of a
/// bucket, byte one the transform id, bytes two and three the word index.
static WORD_ITEMS: &[u8; 4 * NUM_ITEMS] = include_bytes!("lut_words.bin");

/// Transform id of `BROTLI_TRANSFORM_UPPERCASE_FIRST`.
const TRANSFORM_UPPERCASE_FIRST: u8 = 10;

/// One entry of the word lookup (`DictWord`).
#[derive(Copy, Clone, Debug)]
struct DictWord {
    /// Length of the word.
    len: usize,
    /// Transform applied before matching: identity, uppercase-first or
    /// uppercase-all.
    transform: u8,
    /// Index of the word among those of its length.
    idx: usize,
    /// Whether this is the last entry of its bucket.
    last: bool,
}

/// Reads entry `index` of the word lookup.
#[inline]
fn item(index: usize) -> Option<DictWord> {
    let chunk = WORD_ITEMS
        .get(4 * index..)
        .and_then(<[u8]>::first_chunk::<4>)?;
    Some(DictWord {
        len: usize::from(chunk[0] & 0x1F),
        transform: chunk[1],
        idx: usize::from(u16::from_le_bytes([chunk[2], chunk[3]])),
        last: chunk[0] & 0x80 != 0,
    })
}

/// Returns the bucket the four bytes at the start of `data` hash into.
///
/// Mirrors `Hash15`; unlike the shallow probe's `Hash14`, this keeps fifteen
/// bits.
#[inline]
fn hash15(data: &[u8]) -> usize {
    let word = match data.first_chunk::<4>() {
        Some(chunk) => u32::from_le_bytes(*chunk),
        None => 0,
    };
    (word.wrapping_mul(HASH_MUL32) >> (32 - 15)) as usize
}

/// Returns the head entry of `data`'s bucket, or `None` when it is empty.
#[inline]
fn bucket_head(data: &[u8]) -> Option<usize> {
    let chunk = BUCKETS
        .get(2 * hash15(data)..)
        .and_then(<[u8]>::first_chunk::<2>)?;
    let offset = usize::from(u16::from_le_bytes(*chunk));
    (offset != 0).then_some(offset)
}

/// Returns the bytes of word `idx` among those of length `len`.
#[inline]
fn word(len: usize, idx: usize) -> Option<&'static [u8]> {
    let offset = *BUILTIN_OFFSETS_BY_LENGTH.get(len)? as usize + len * idx;
    BUILTIN_WORDS.get(offset..offset.checked_add(len)?)
}

/// Returns how many words of length `len` the dictionary holds.
#[inline]
fn words_of_length(len: usize) -> usize {
    match BUILTIN_SIZE_BITS_BY_LENGTH.get(len) {
        Some(&bits) => 1usize << bits,
        None => 0,
    }
}

/// Records that a word of `len` bytes matched, if it beats what is there.
///
/// Mirrors `AddMatch`. The packed value puts the distance above the length
/// code, so taking the minimum prefers the nearest word and, among equally near
/// ones, the smaller code.
#[inline]
fn add_match(distance: usize, len: usize, len_code: usize, matches: &mut [u32]) {
    let packed = ((distance as u32) << 5) + len_code as u32;
    if let Some(slot) = matches.get_mut(len) {
        *slot = (*slot).min(packed);
    }
}

/// Returns how many leading bytes of word `(len, id)` agree with `data`.
///
/// Mirrors `DictMatchLength`.
#[inline]
fn dict_match_length(data: &[u8], id: usize, len: usize, maxlen: usize) -> usize {
    match word(len, id) {
        Some(word) => common_prefix_len(word, data, len.min(maxlen)),
        None => 0,
    }
}

/// Returns whether the transformed word `w` matches the start of `data`.
///
/// Mirrors `IsMatch`. The lookup table only ever carries ASCII-uppercasable
/// words for the two case transforms, so uppercasing is the plain bit flip.
fn is_match(w: DictWord, data: &[u8], max_length: usize) -> bool {
    if w.len > max_length {
        return false;
    }
    let Some(dict) = word(w.len, w.idx) else {
        return false;
    };
    match w.transform {
        0 => common_prefix_len(dict, data, w.len) == w.len,
        TRANSFORM_UPPERCASE_FIRST => {
            let (Some(&first), Some(&target)) = (dict.first(), data.first()) else {
                return false;
            };
            first.is_ascii_lowercase()
                && (first ^ 32) == target
                && common_prefix_len(&dict[1..], &data[1..], w.len - 1) == w.len - 1
        }
        _ => dict.iter().zip(data).take(w.len).all(|(&left, &right)| {
            if left.is_ascii_lowercase() {
                (left ^ 32) == right
            } else {
                left == right
            }
        }),
    }
}

/// Fills `matches` with the best dictionary word at every length.
///
/// `matches` must be [`MAX_STATIC_DICTIONARY_MATCH_LEN`] + 1 long and filled
/// with [`INVALID_MATCH`]. Returns whether anything matched at all.
///
/// Mirrors `BrotliFindAllStaticDictionaryMatchesFor`. Shared and compound
/// dictionaries are not reachable through this crate's public API, so the
/// second-dictionary merge of `BrotliFindAllStaticDictionaryMatches` has no
/// counterpart here.
pub(crate) fn find_all(
    data: &[u8],
    min_length: usize,
    max_length: usize,
    matches: &mut [u32; MAX_STATIC_DICTIONARY_MATCH_LEN + 1],
) -> bool {
    let mut found = false;
    found |= plain_and_uppercase(data, min_length, max_length, matches);
    found |= space_or_dot_prefix(data, max_length, matches);
    found |= two_byte_prefix(data, max_length, matches);
    found |= long_prefix(data, max_length, matches);
    found
}

/// Words matching at the start of `data`, with their suffix transforms.
///
/// Covers the identity, omit-last and uppercase families.
fn plain_and_uppercase(
    data: &[u8],
    min_length: usize,
    max_length: usize,
    matches: &mut [u32],
) -> bool {
    let mut found = false;
    let Some(mut offset) = bucket_head(data) else {
        return false;
    };
    loop {
        let Some(w) = item(offset) else { return found };
        offset += 1;
        let l = w.len;
        let n = words_of_length(l);
        let id = w.idx;
        let end = w.last;

        if w.transform == 0 {
            let matchlen = dict_match_length(data, id, l, max_length);

            // "" + IDENTITY + ""
            if matchlen == l {
                add_match(id, l, l, matches);
                found = true;
            }
            // "" + OMIT_LAST_1 + "" and "" + OMIT_LAST_1 + "ing "
            if matchlen + 1 >= l {
                add_match(id + 12 * n, l - 1, l, matches);
                if l + 2 < max_length
                    && data.get(l - 1) == Some(&b'i')
                    && data.get(l) == Some(&b'n')
                    && data.get(l + 1) == Some(&b'g')
                    && data.get(l + 2) == Some(&b' ')
                {
                    add_match(id + 49 * n, l + 3, l, matches);
                }
                found = true;
            }
            // "" + OMIT_LAST_# + "" for # from two to nine
            let mut minlen = min_length;
            if l > 9 {
                minlen = minlen.max(l - 9);
            }
            let maxlen = matchlen.min(l.saturating_sub(2));
            for len in minlen..=maxlen {
                let cut = l - len;
                let transform_id = (cut << 2) + ((CUTOFF_TRANSFORMS >> (cut * 6)) & 0x3F) as usize;
                add_match(id + transform_id * n, len, l, matches);
                found = true;
            }

            if matchlen < l || l + 6 >= max_length {
                if end {
                    return found;
                }
                continue;
            }
            // "" + IDENTITY + <suffix>
            let s = data.get(l..).unwrap_or_default();
            identity_suffix(s, id, l, n, max_length, matches);
        } else {
            // UPPERCASE_FIRST or UPPERCASE_ALL.
            let is_all_caps = w.transform != TRANSFORM_UPPERCASE_FIRST;
            if !is_match(w, data, max_length) {
                if end {
                    return found;
                }
                continue;
            }
            // "" + kUppercase{First,All} + ""
            add_match(id + if is_all_caps { 44 } else { 9 } * n, l, l, matches);
            found = true;
            if l + 1 >= max_length {
                if end {
                    return found;
                }
                continue;
            }
            // "" + kUppercase{First,All} + <suffix>
            let s = data.get(l..).unwrap_or_default();
            uppercase_suffix(s, id, l, n, is_all_caps, matches);
        }

        if end {
            return found;
        }
    }
}

/// The `"" + IDENTITY + <suffix>` transform family.
#[expect(
    clippy::collapsible_match,
    reason = "the nesting mirrors the reference's decision tree byte for byte; \
              folding a test into its arm's pattern would break the \
              correspondence this port is audited against"
)]
fn identity_suffix(
    s: &[u8],
    id: usize,
    l: usize,
    n: usize,
    max_length: usize,
    matches: &mut [u32],
) {
    let at = |i: usize| s.get(i).copied().unwrap_or(0);
    match at(0) {
        b' ' => {
            add_match(id + n, l + 1, l, matches);
            match at(1) {
                b'a' => match at(2) {
                    b' ' => add_match(id + 28 * n, l + 3, l, matches),
                    b's' => {
                        if at(3) == b' ' {
                            add_match(id + 46 * n, l + 4, l, matches);
                        }
                    }
                    b't' => {
                        if at(3) == b' ' {
                            add_match(id + 60 * n, l + 4, l, matches);
                        }
                    }
                    b'n' => {
                        if at(3) == b'd' && at(4) == b' ' {
                            add_match(id + 10 * n, l + 5, l, matches);
                        }
                    }
                    _ => {}
                },
                b'b' => {
                    if at(2) == b'y' && at(3) == b' ' {
                        add_match(id + 38 * n, l + 4, l, matches);
                    }
                }
                b'i' => match at(2) {
                    b'n' => {
                        if at(3) == b' ' {
                            add_match(id + 16 * n, l + 4, l, matches);
                        }
                    }
                    b's' => {
                        if at(3) == b' ' {
                            add_match(id + 47 * n, l + 4, l, matches);
                        }
                    }
                    _ => {}
                },
                b'f' => match at(2) {
                    b'o' => {
                        if at(3) == b'r' && at(4) == b' ' {
                            add_match(id + 25 * n, l + 5, l, matches);
                        }
                    }
                    b'r' => {
                        if at(3) == b'o' && at(4) == b'm' && at(5) == b' ' {
                            add_match(id + 37 * n, l + 6, l, matches);
                        }
                    }
                    _ => {}
                },
                b'o' => match at(2) {
                    b'f' => {
                        if at(3) == b' ' {
                            add_match(id + 8 * n, l + 4, l, matches);
                        }
                    }
                    b'n' => {
                        if at(3) == b' ' {
                            add_match(id + 45 * n, l + 4, l, matches);
                        }
                    }
                    _ => {}
                },
                b'n' => {
                    if at(2) == b'o' && at(3) == b't' && at(4) == b' ' {
                        add_match(id + 80 * n, l + 5, l, matches);
                    }
                }
                b't' => match at(2) {
                    b'h' => match at(3) {
                        b'e' => {
                            if at(4) == b' ' {
                                add_match(id + 5 * n, l + 5, l, matches);
                            }
                        }
                        b'a' => {
                            if at(4) == b't' && at(5) == b' ' {
                                add_match(id + 29 * n, l + 6, l, matches);
                            }
                        }
                        _ => {}
                    },
                    b'o' => {
                        if at(3) == b' ' {
                            add_match(id + 17 * n, l + 4, l, matches);
                        }
                    }
                    _ => {}
                },
                b'w' => {
                    if at(2) == b'i' && at(3) == b't' && at(4) == b'h' && at(5) == b' ' {
                        add_match(id + 35 * n, l + 6, l, matches);
                    }
                }
                _ => {}
            }
        }
        b'"' => {
            add_match(id + 19 * n, l + 1, l, matches);
            if at(1) == b'>' {
                add_match(id + 21 * n, l + 2, l, matches);
            }
        }
        b'.' => {
            add_match(id + 20 * n, l + 1, l, matches);
            if at(1) == b' ' {
                add_match(id + 31 * n, l + 2, l, matches);
                if at(2) == b'T' && at(3) == b'h' {
                    if at(4) == b'e' {
                        if at(5) == b' ' {
                            add_match(id + 43 * n, l + 6, l, matches);
                        }
                    } else if at(4) == b'i' && at(5) == b's' && at(6) == b' ' {
                        add_match(id + 75 * n, l + 7, l, matches);
                    }
                }
            }
        }
        b',' => {
            add_match(id + 76 * n, l + 1, l, matches);
            if at(1) == b' ' {
                add_match(id + 14 * n, l + 2, l, matches);
            }
        }
        b'\n' => {
            add_match(id + 22 * n, l + 1, l, matches);
            if at(1) == b'\t' {
                add_match(id + 50 * n, l + 2, l, matches);
            }
        }
        b']' => add_match(id + 24 * n, l + 1, l, matches),
        b'\'' => add_match(id + 36 * n, l + 1, l, matches),
        b':' => add_match(id + 51 * n, l + 1, l, matches),
        b'(' => add_match(id + 57 * n, l + 1, l, matches),
        b'=' => match at(1) {
            b'"' => add_match(id + 70 * n, l + 2, l, matches),
            b'\'' => add_match(id + 86 * n, l + 2, l, matches),
            _ => {}
        },
        b'a' => {
            if at(1) == b'l' && at(2) == b' ' {
                add_match(id + 84 * n, l + 3, l, matches);
            }
        }
        b'e' => match at(1) {
            b'd' => {
                if at(2) == b' ' {
                    add_match(id + 53 * n, l + 3, l, matches);
                }
            }
            b'r' => {
                if at(2) == b' ' {
                    add_match(id + 82 * n, l + 3, l, matches);
                }
            }
            b's' => {
                if at(2) == b't' && at(3) == b' ' {
                    add_match(id + 95 * n, l + 4, l, matches);
                }
            }
            _ => {}
        },
        b'f' => {
            if at(1) == b'u' && at(2) == b'l' && at(3) == b' ' {
                add_match(id + 90 * n, l + 4, l, matches);
            }
        }
        b'i' => match at(1) {
            b'v' => {
                if at(2) == b'e' && at(3) == b' ' {
                    add_match(id + 92 * n, l + 4, l, matches);
                }
            }
            b'z' => {
                if at(2) == b'e' && at(3) == b' ' {
                    add_match(id + 100 * n, l + 4, l, matches);
                }
            }
            _ => {}
        },
        b'l' => match at(1) {
            b'e' => {
                if at(2) == b's' && at(3) == b's' && at(4) == b' ' {
                    add_match(id + 93 * n, l + 5, l, matches);
                }
            }
            b'y' => {
                if at(2) == b' ' {
                    add_match(id + 61 * n, l + 3, l, matches);
                }
            }
            _ => {}
        },
        b'o' => {
            if at(1) == b'u' && at(2) == b's' && at(3) == b' ' {
                add_match(id + 106 * n, l + 4, l, matches);
            }
        }
        _ => {}
    }
    let _ = max_length;
}

/// The `"" + kUppercase{First,All} + <suffix>` transform family.
fn uppercase_suffix(
    s: &[u8],
    id: usize,
    l: usize,
    n: usize,
    is_all_caps: bool,
    matches: &mut [u32],
) {
    let at = |i: usize| s.get(i).copied().unwrap_or(0);
    let pick = |caps: usize, plain: usize| if is_all_caps { caps } else { plain };
    match at(0) {
        b' ' => add_match(id + pick(68, 4) * n, l + 1, l, matches),
        b'"' => {
            add_match(id + pick(87, 66) * n, l + 1, l, matches);
            if at(1) == b'>' {
                add_match(id + pick(97, 69) * n, l + 2, l, matches);
            }
        }
        b'.' => {
            add_match(id + pick(101, 79) * n, l + 1, l, matches);
            if at(1) == b' ' {
                add_match(id + pick(114, 88) * n, l + 2, l, matches);
            }
        }
        b',' => {
            add_match(id + pick(112, 99) * n, l + 1, l, matches);
            if at(1) == b' ' {
                add_match(id + pick(107, 58) * n, l + 2, l, matches);
            }
        }
        b'\'' => add_match(id + pick(94, 74) * n, l + 1, l, matches),
        b'(' => add_match(id + pick(113, 78) * n, l + 1, l, matches),
        b'=' => match at(1) {
            b'"' => add_match(id + pick(105, 104) * n, l + 2, l, matches),
            b'\'' => add_match(id + pick(116, 108) * n, l + 2, l, matches),
            _ => {}
        },
        _ => {}
    }
}

/// Words matching after a leading `" "` or `"."`.
fn space_or_dot_prefix(data: &[u8], max_length: usize, matches: &mut [u32]) -> bool {
    if max_length < 5 {
        return false;
    }
    let first = data.first().copied().unwrap_or(0);
    if first != b' ' && first != b'.' {
        return false;
    }
    let is_space = first == b' ';
    let tail = data.get(1..).unwrap_or_default();
    let Some(mut offset) = bucket_head(tail) else {
        return false;
    };
    let mut found = false;
    loop {
        let Some(w) = item(offset) else { return found };
        offset += 1;
        let l = w.len;
        let n = words_of_length(l);
        let id = w.idx;
        let end = w.last;

        if w.transform == 0 {
            if !is_match(w, tail, max_length - 1) {
                if end {
                    return found;
                }
                continue;
            }
            // " " / "." + IDENTITY + ""
            add_match(id + if is_space { 6 } else { 32 } * n, l + 1, l, matches);
            found = true;
            if l + 2 >= max_length {
                if end {
                    return found;
                }
                continue;
            }
            // " " / "." + IDENTITY + <suffix>
            let s = data.get(l + 1..).unwrap_or_default();
            let at = |i: usize| s.get(i).copied().unwrap_or(0);
            match at(0) {
                b' ' => add_match(id + if is_space { 2 } else { 77 } * n, l + 2, l, matches),
                b'(' => add_match(id + if is_space { 89 } else { 67 } * n, l + 2, l, matches),
                b',' if is_space => {
                    add_match(id + 103 * n, l + 2, l, matches);
                    if at(1) == b' ' {
                        add_match(id + 33 * n, l + 3, l, matches);
                    }
                }
                b'.' if is_space => {
                    add_match(id + 71 * n, l + 2, l, matches);
                    if at(1) == b' ' {
                        add_match(id + 52 * n, l + 3, l, matches);
                    }
                }
                b'=' if is_space => match at(1) {
                    b'"' => add_match(id + 81 * n, l + 3, l, matches),
                    b'\'' => add_match(id + 98 * n, l + 3, l, matches),
                    _ => {}
                },
                _ => {}
            }
        } else if is_space {
            let is_all_caps = w.transform != TRANSFORM_UPPERCASE_FIRST;
            if !is_match(w, tail, max_length - 1) {
                if end {
                    return found;
                }
                continue;
            }
            // " " + kUppercase{First,All} + ""
            add_match(
                id + if is_all_caps { 85 } else { 30 } * n,
                l + 1,
                l,
                matches,
            );
            found = true;
            if l + 2 >= max_length {
                if end {
                    return found;
                }
                continue;
            }
            // " " + kUppercase{First,All} + <suffix>
            let s = data.get(l + 1..).unwrap_or_default();
            let at = |i: usize| s.get(i).copied().unwrap_or(0);
            let pick = |caps: usize, plain: usize| if is_all_caps { caps } else { plain };
            match at(0) {
                b' ' => add_match(id + pick(83, 15) * n, l + 2, l, matches),
                b',' => {
                    if !is_all_caps {
                        add_match(id + 109 * n, l + 2, l, matches);
                    }
                    if at(1) == b' ' {
                        add_match(id + pick(111, 65) * n, l + 3, l, matches);
                    }
                }
                b'.' => {
                    add_match(id + pick(115, 96) * n, l + 2, l, matches);
                    if at(1) == b' ' {
                        add_match(id + pick(117, 91) * n, l + 3, l, matches);
                    }
                }
                b'=' => match at(1) {
                    b'"' => add_match(id + pick(110, 118) * n, l + 3, l, matches),
                    b'\'' => add_match(id + pick(119, 120) * n, l + 3, l, matches),
                    _ => {}
                },
                _ => {}
            }
        }

        if end {
            return found;
        }
    }
}

/// Words matching after a leading `"e "`, `"s "`, `", "` or `"\xC2\xA0"`.
fn two_byte_prefix(data: &[u8], max_length: usize, matches: &mut [u32]) -> bool {
    if max_length < 6 {
        return false;
    }
    let at = |i: usize| data.get(i).copied().unwrap_or(0);
    let ascii = at(1) == b' ' && matches!(at(0), b'e' | b's' | b',');
    let nbsp = at(0) == 0xC2 && at(1) == 0xA0;
    if !ascii && !nbsp {
        return false;
    }
    let tail = data.get(2..).unwrap_or_default();
    let Some(mut offset) = bucket_head(tail) else {
        return false;
    };
    let mut found = false;
    loop {
        let Some(w) = item(offset) else { return found };
        offset += 1;
        let l = w.len;
        let n = words_of_length(l);
        let end = w.last;

        if w.transform == 0 && is_match(w, tail, max_length - 2) {
            if at(0) == 0xC2 {
                add_match(w.idx + 102 * n, l + 2, l, matches);
                found = true;
            } else if l + 2 < max_length && at(l + 2) == b' ' {
                let transform = match at(0) {
                    b'e' => 18,
                    b's' => 7,
                    _ => 13,
                };
                add_match(w.idx + transform * n, l + 3, l, matches);
                found = true;
            }
        }

        if end {
            return found;
        }
    }
}

/// Words matching after a leading `" the "` or `".com/"`.
fn long_prefix(data: &[u8], max_length: usize, matches: &mut [u32]) -> bool {
    if max_length < 9 {
        return false;
    }
    let at = |i: usize| data.get(i).copied().unwrap_or(0);
    let the = at(0) == b' ' && at(1) == b't' && at(2) == b'h' && at(3) == b'e' && at(4) == b' ';
    let com = at(0) == b'.' && at(1) == b'c' && at(2) == b'o' && at(3) == b'm' && at(4) == b'/';
    if !the && !com {
        return false;
    }
    let tail = data.get(5..).unwrap_or_default();
    let Some(mut offset) = bucket_head(tail) else {
        return false;
    };
    let mut found = false;
    loop {
        let Some(w) = item(offset) else { return found };
        offset += 1;
        let l = w.len;
        let n = words_of_length(l);
        let id = w.idx;
        let end = w.last;

        if w.transform == 0 && is_match(w, tail, max_length - 5) {
            add_match(id + if the { 41 } else { 72 } * n, l + 5, l, matches);
            found = true;
            if l + 5 < max_length && the {
                let s = data.get(l + 5..).unwrap_or_default();
                let s_at = |i: usize| s.get(i).copied().unwrap_or(0);
                if l + 8 < max_length
                    && s_at(0) == b' '
                    && s_at(1) == b'o'
                    && s_at(2) == b'f'
                    && s_at(3) == b' '
                {
                    add_match(id + 62 * n, l + 9, l, matches);
                    if l + 12 < max_length
                        && s_at(4) == b't'
                        && s_at(5) == b'h'
                        && s_at(6) == b'e'
                        && s_at(7) == b' '
                    {
                        add_match(id + 73 * n, l + 13, l, matches);
                    }
                }
            }
        }

        if end {
            return found;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Runs the search over `data`, returning the per-length match table.
    fn find(data: &[u8]) -> [u32; MAX_STATIC_DICTIONARY_MATCH_LEN + 1] {
        let mut matches = [INVALID_MATCH; MAX_STATIC_DICTIONARY_MATCH_LEN + 1];
        find_all(data, 4, data.len(), &mut matches);
        matches
    }

    /// Runs the C search over `data`, returning its per-length match table.
    ///
    /// Reaches `BrotliFindAllStaticDictionaryMatches` — which has no public
    /// header — through this workspace's shim.
    fn c_find(data: &[u8], min_length: usize) -> [u32; MAX_STATIC_DICTIONARY_MATCH_LEN + 1] {
        let mut matches = [INVALID_MATCH; MAX_STATIC_DICTIONARY_MATCH_LEN + 1];
        // SAFETY: `data` is readable for its own length, which is what is
        // passed as `max_length`, and `matches` is exactly the length the
        // reference documents.
        unsafe {
            google_brotli_ffi::mbrotli_shim_find_all_static_dictionary_matches(
                data.as_ptr(),
                min_length,
                data.len(),
                matches.as_mut_ptr(),
            );
        }
        matches
    }

    /// Compares both searches over `data` at several minimum lengths.
    fn assert_matches_c(name: &str, data: &[u8]) {
        for min_length in [4usize, 5, 8, 12, 20] {
            let expected = c_find(data, min_length);
            let mut actual = [INVALID_MATCH; MAX_STATIC_DICTIONARY_MATCH_LEN + 1];
            find_all(data, min_length, data.len(), &mut actual);
            assert_eq!(
                actual,
                expected,
                "case {name}, min_length {min_length}, data {:?}",
                &data[..data.len().min(24)]
            );
        }
    }

    /// Deterministic xorshift generator, so the fixtures are reproducible.
    struct Rng(u64);

    impl Rng {
        fn next_u8(&mut self) -> u8 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 24) as u8
        }
    }

    #[test]
    fn every_word_and_transform_agrees_with_the_c_search() {
        // Walking the dictionary's own words is what reaches the transform
        // cases: each word bare, then with every prefix and suffix the
        // reference recognises, and in both of its case transforms.
        let suffixes: [&[u8]; 24] = [
            b"", b" ", b" the ", b" of ", b" and ", b" in ", b" to ", b" a ", b" that ", b" with ",
            b" from ", b" by ", b" on ", b" as ", b" is ", b" not ", b"\"", b"\">", b". ",
            b". The ", b". This ", b", ", b"=\"", b"='",
        ];
        let words: [&[u8]; 12] = [
            b"time",
            b"download",
            b"government",
            b"information",
            b"description",
            b"background",
            b"the",
            b"quick",
            b"world",
            b"under",
            b"years",
            b"where",
        ];
        let prefixes: [&[u8]; 8] = [b"", b" ", b".", b"e ", b"s ", b", ", b" the ", b".com/"];
        for word in words {
            for suffix in suffixes {
                for prefix in prefixes {
                    let mut data = Vec::new();
                    data.extend_from_slice(prefix);
                    data.extend_from_slice(word);
                    data.extend_from_slice(suffix);
                    // Padding, so every transform has bytes to look at.
                    data.extend_from_slice(b"ZZZZZZZZZZZZZZZZ");
                    assert_matches_c("word", &data);

                    let mut upper_first = data.clone();
                    if let Some(byte) = upper_first.get_mut(prefix.len()) {
                        byte.make_ascii_uppercase();
                    }
                    assert_matches_c("upper-first", &upper_first);

                    let mut upper_all = data.clone();
                    for byte in &mut upper_all[prefix.len()..prefix.len() + word.len()] {
                        byte.make_ascii_uppercase();
                    }
                    assert_matches_c("upper-all", &upper_all);
                }
            }
        }
    }

    #[test]
    fn structural_bytes_agree_with_the_c_search() {
        let cases: [(&str, Vec<u8>); 7] = [
            ("ascending", (0..64u8).collect()),
            ("descending", (0..64u8).rev().collect()),
            ("zeros", vec![0u8; 64]),
            ("ones", vec![0xFFu8; 64]),
            ("high-bytes", (0..64u32).map(|i| 0x80 | (i as u8)).collect()),
            (
                "alternating",
                (0..64u32)
                    .map(|i| if i % 2 == 0 { 0 } else { 0xFF })
                    .collect(),
            ),
            (
                "utf8",
                "ærlig ånd øver ζωή 日本語のテキスト".as_bytes().to_vec(),
            ),
        ];
        for (name, data) in cases {
            assert_matches_c(name, &data);
        }
    }

    #[test]
    fn random_bytes_agree_with_the_c_search() {
        let mut rng = Rng(0x5D1C_7104_0BAD_C0DE);
        for case in 0..3000u32 {
            // A mix of entropies: uniform bytes almost never match, while a
            // small ASCII alphabet hits the dictionary constantly.
            let data: Vec<u8> = (0..48)
                .map(|_| match case % 3 {
                    0 => rng.next_u8(),
                    1 => b'a' + (rng.next_u8() % 26),
                    _ => match rng.next_u8() % 8 {
                        0 => b' ',
                        1 => b'.',
                        2 => b'e',
                        3 => b's',
                        4 => b',',
                        5 => b't',
                        6 => b'h',
                        _ => b'o',
                    },
                })
                .collect();
            assert_matches_c("random", &data);
        }
    }

    #[test]
    fn every_position_of_a_text_corpus_agrees_with_the_c_search() {
        // Real text reaches the prefixed families the synthetic cases only
        // exercise one at a time.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/brotli-ffi/vendor/brotli/tests/testdata/alice29.txt"
        );
        let Ok(corpus) = std::fs::read(path) else {
            // The vendored submodule is not checked out.
            return;
        };
        let window = 48usize;
        let mut position = 0usize;
        while position + window <= corpus.len().min(60_000) {
            assert_matches_c("alice29", &corpus[position..position + window]);
            position += 5;
        }
    }

    /// Returns the lengths at which anything matched.
    fn lengths(matches: &[u32; MAX_STATIC_DICTIONARY_MATCH_LEN + 1]) -> Vec<usize> {
        matches
            .iter()
            .enumerate()
            .filter(|&(_, &packed)| packed < INVALID_MATCH)
            .map(|(length, _)| length)
            .collect()
    }

    #[test]
    fn the_lookup_tables_have_the_reference_shape() {
        assert_eq!(BUCKETS.len(), 2 * NUM_BUCKETS);
        assert_eq!(WORD_ITEMS.len(), 4 * NUM_ITEMS);
        // Entry zero is the sentinel an empty bucket points at.
        let head = item(0).expect("entry zero");
        assert_eq!((head.len, head.transform, head.idx), (0, 0, 0));
    }

    #[test]
    fn every_word_entry_points_inside_the_dictionary() {
        for index in 1..NUM_ITEMS {
            let w = item(index).expect("entry in range");
            assert!(
                (4..=24).contains(&w.len),
                "entry {index} has length {}",
                w.len
            );
            assert!(
                w.idx < words_of_length(w.len),
                "entry {index} points past the words of its length"
            );
            assert!(word(w.len, w.idx).is_some(), "entry {index} has no bytes");
        }
    }

    #[test]
    fn every_bucket_head_is_in_range() {
        for bucket in 0..NUM_BUCKETS {
            let chunk = [BUCKETS[2 * bucket], BUCKETS[2 * bucket + 1]];
            let head = usize::from(u16::from_le_bytes(chunk));
            assert!(head < NUM_ITEMS, "bucket {bucket} points at {head}");
        }
    }

    #[test]
    fn a_plain_dictionary_word_matches_at_its_own_length() {
        // "time" is the first four-letter word of the dictionary.
        let matches = find(b"time and again");
        assert!(
            matches[4] < INVALID_MATCH,
            "no four-byte match: {:?}",
            lengths(&matches)
        );
        // Word index zero, identity transform, so the code is the length.
        assert_eq!(matches[4] & 31, 4);
    }

    #[test]
    fn a_suffix_transform_extends_the_match_past_the_word() {
        // "time" followed by a space is the "" + IDENTITY + " " transform, so a
        // five-byte match exists that the bare word cannot give.
        let matches = find(b"time to go");
        assert!(matches[4] < INVALID_MATCH);
        assert!(
            matches[5] < INVALID_MATCH,
            "the trailing-space transform was missed: {:?}",
            lengths(&matches)
        );
    }

    #[test]
    fn an_uppercase_first_word_is_found() {
        // The dictionary holds "time" in lower case only; the uppercase-first
        // transform is what makes "Time" match.
        let matches = find(b"Time flies");
        assert!(
            matches[4] < INVALID_MATCH,
            "the uppercase transform was missed"
        );
    }

    #[test]
    fn an_omit_last_transform_shortens_a_longer_word() {
        // "download" with its last byte omitted is a seven-byte match, which
        // only the OMIT_LAST_1 transform can produce.
        let mut table = [INVALID_MATCH; MAX_STATIC_DICTIONARY_MATCH_LEN + 1];
        find_all(b"downloa!!!!!!!!", 4, 15, &mut table);
        assert!(
            table[7] < INVALID_MATCH,
            "omit-last-1 was missed: {:?}",
            lengths(&table)
        );
    }

    #[test]
    fn a_space_prefixed_word_is_found() {
        let matches = find(b" the quick brown");
        assert!(
            lengths(&matches).iter().any(|&len| len >= 4),
            "no prefixed match: {:?}",
            lengths(&matches)
        );
    }

    #[test]
    fn a_binary_run_matches_the_words_the_reference_finds() {
        // The dictionary holds binary sequences as well as words, and an
        // ascending byte run is one of them. These are the exact matches
        // `BrotliFindAllStaticDictionaryMatches` reports for this input: one
        // eight-byte word plus its omit-last cuts, all with length code eight.
        let matches = find(&[0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09]);
        assert_eq!(lengths(&matches), vec![4, 5, 6, 7, 8]);
        let expected = [
            (4usize, 44014u32),
            (5, 24558),
            (6, 28654),
            (7, 13294),
            (8, 1006),
        ];
        for (length, distance) in expected {
            assert_eq!(matches[length] >> 5, distance, "length {length}");
            assert_eq!(matches[length] & 31, 8, "length {length}");
        }
    }

    #[test]
    fn nothing_matches_bytes_no_word_starts_with() {
        let matches = find(&[0xF7, 0xE3, 0xD1, 0xC5, 0xB9, 0xAD, 0x9B, 0x8F, 0x83, 0xF1]);
        assert!(lengths(&matches).is_empty(), "{:?}", lengths(&matches));
    }

    #[test]
    fn a_short_input_reaches_no_prefix_family() {
        // Each prefixed family has its own minimum length; below all of them
        // only the plain search runs, and it cannot match four bytes of one.
        let mut matches = [INVALID_MATCH; MAX_STATIC_DICTIONARY_MATCH_LEN + 1];
        assert!(!space_or_dot_prefix(b" the", 4, &mut matches));
        assert!(!two_byte_prefix(b"e the", 5, &mut matches));
        assert!(!long_prefix(b" the tim", 8, &mut matches));
    }

    #[test]
    fn the_nearest_word_wins_at_a_given_length() {
        // `add_match` keeps the minimum packed value, so a second word at the
        // same length only replaces the first when it is nearer.
        let mut matches = [INVALID_MATCH; MAX_STATIC_DICTIONARY_MATCH_LEN + 1];
        add_match(500, 6, 6, &mut matches);
        add_match(100, 6, 6, &mut matches);
        assert_eq!(matches[6] >> 5, 100);
        add_match(900, 6, 6, &mut matches);
        assert_eq!(matches[6] >> 5, 100);
    }
}

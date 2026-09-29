//! What a shared context holds, and how a match is found in it.
//!
//! The context is split in two:
//!
//! - [`SharedDictionaryData`] owns the caller's dictionary bytes;
//! - [`PreparedDictionaryIndexes`] owns the hash indexes built over them.
//!
//! Both are immutable once [`SharedContextInner::new`] returns, which is why
//! both hold `Box<[_]>` rather than `Vec<_>`: the spare-capacity word is dead
//! weight once a collection has stopped growing, and dropping it also drops
//! `push` and `truncate` from the type, so "semantically immutable after
//! preparation" is enforced rather than merely documented. The one collection
//! that does grow — the builder's attachment list — is a `Vec`.
//!
//! There is no third, mutable part yet: nothing a context owns today carries
//! LZ77 history, a distance cache, pending commands or an input position, so
//! there is no stream-semantic state for a session to reset and none for one
//! call to inherit from the last. The reusable encoder workspace that will
//! need a generation counter and an RAII idle guard arrives with the match
//! finders that use it; until then a context is simply read.
//!
//! There is no `Arc`, no lock, no atomic and no interior mutability anywhere
//! below this comment. One context is owned by one caller, and the exclusive
//! borrow the compression entry points take is what makes one context back at
//! most one active session.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::compressor::shared::SharedBrotliError;

use super::prefix::{MAX_PREFIX_DICTIONARIES, MAX_PREFIX_SEGMENT_BYTES, PrefixSources};
#[cfg(any(test, feature = "diagnostics"))]
use super::prepared::HASH_INPUT_BYTES;
use super::prepared::PreparedPrefix;

/// The caller's dictionary bytes, in attachment order.
///
/// Immutable for the whole life of the context: the only way to change what a
/// context contains is to build another one.
#[derive(Debug, Default)]
pub(crate) struct SharedDictionaryData {
    /// The LZ77 prefix, and the addressing that makes its segments one
    /// logical byte sequence.
    prefix: PrefixSources,
}

impl SharedDictionaryData {
    /// Returns the logical LZ77 prefix.
    pub(crate) const fn prefix(&self) -> &PrefixSources {
        &self.prefix
    }

    /// Returns the number of dictionary bytes the caller handed over.
    ///
    /// Today every attached byte is prefix; a serialized dictionary's word and
    /// transform tables will be counted here too once they are parsed.
    pub(crate) fn source_size(&self) -> usize {
        self.prefix.total_len() as usize
    }

    /// Returns the bytes the source buffers occupy on the heap.
    fn allocated_size(&self) -> usize {
        let segments = self.prefix.segment_count();
        self.source_size() + segments * size_of::<Box<[u8]>>() + segments * size_of::<u64>()
    }
}

/// One prepared hash index per attachment, in attachment order.
///
/// Built once, read many times, never mutated. Keeping the indexes beside the
/// dictionary bytes rather than inside them is what will let a session borrow
/// the two halves separately once there is a mutable third.
#[derive(Debug, Default)]
pub(crate) struct PreparedDictionaryIndexes {
    /// The index for attachment `i`, at position `i`.
    prefixes: Box<[PreparedPrefix]>,
}

impl PreparedDictionaryIndexes {
    /// Returns the index built over one attachment, if there is one.
    ///
    /// The indexes are in attachment order, so index `i` belongs to the `i`-th
    /// segment of [`SharedDictionaryData::prefix`].
    pub(crate) fn prefix(&self, index: usize) -> Option<&PreparedPrefix> {
        self.prefixes.get(index)
    }

    /// Returns the bytes the indexes occupy.
    fn allocated_size(&self) -> usize {
        self.prefixes
            .iter()
            .map(PreparedPrefix::allocated_size)
            .sum()
    }
}

/// Where an attached dictionary matched, and for how long.
///
/// Only the longest-match diagnostic produces one; the match finders read a
/// `SearchResult` instead.
#[cfg(any(test, feature = "diagnostics"))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct PrefixMatch {
    /// Logical address of the first matching byte in the concatenated prefix.
    pub(crate) offset: u64,
    /// How many bytes matched from there.
    pub(crate) length: usize,
}

/// The whole private state of one caller-owned shared context.
#[derive(Debug, Default)]
pub(crate) struct SharedContextInner {
    #[cfg(feature = "experimental")]
    pub(crate) static_index: Option<super::static_index::StaticIndex>,
    /// The caller's dictionary bytes.
    dictionaries: SharedDictionaryData,
    /// The indexes prepared over them.
    prepared: PreparedDictionaryIndexes,
}

impl SharedContextInner {
    /// Prepares a context from the attachments, in attachment order.
    ///
    /// All-or-nothing: every count, size and allocation limit is checked
    /// before the first index is built, so a failure leaves no half-built
    /// context and frees every temporary it took to find the failure.
    ///
    /// # Errors
    ///
    /// - [`SharedBrotliError::TooManyPrefixDictionaries`] past fifteen
    ///   attachments;
    /// - [`SharedBrotliError::DictionaryTooLarge`] when one attachment, the
    ///   logical prefix, or the total source exceeds what the index can
    ///   address or what the limits allow;
    /// - [`SharedBrotliError::SharedContextTooLarge`] when the prepared
    ///   indexes would exceed the allocation limit.
    pub(crate) fn new(
        attachments: Vec<Box<[u8]>>,
        limits: &Budget,
    ) -> Result<Self, SharedBrotliError> {
        let limit = limits.max_attachments.min(MAX_PREFIX_DICTIONARIES);
        if attachments.len() > limit {
            return Err(SharedBrotliError::TooManyPrefixDictionaries {
                attached: attachments.len(),
                limit,
            });
        }

        let mut total = 0u64;
        for attachment in &attachments {
            let length = attachment.len() as u64;
            if length > MAX_PREFIX_SEGMENT_BYTES {
                return Err(SharedBrotliError::DictionaryTooLarge {
                    bytes: length,
                    limit: MAX_PREFIX_SEGMENT_BYTES,
                });
            }
            total += length;
        }
        if total > limits.max_prefix_bytes {
            return Err(SharedBrotliError::DictionaryTooLarge {
                bytes: total,
                limit: limits.max_prefix_bytes,
            });
        }
        if total > limits.max_total_source_bytes {
            return Err(SharedBrotliError::DictionaryTooLarge {
                bytes: total,
                limit: limits.max_total_source_bytes,
            });
        }

        // Refuse before building rather than after: the estimate below is an
        // upper bound on what the indexes will really take, so a context that
        // passes here can only come out smaller than the limit allowed.
        let estimate = estimate_allocation(&attachments).unwrap_or(u64::MAX);
        if estimate > limits.max_allocated_bytes {
            return Err(SharedBrotliError::SharedContextTooLarge {
                bytes: estimate,
                limit: limits.max_allocated_bytes,
            });
        }

        let prefixes: Vec<PreparedPrefix> = attachments
            .iter()
            .map(|attachment| PreparedPrefix::new(attachment))
            .collect();

        Ok(Self {
            #[cfg(feature = "experimental")]
            static_index: None,
            dictionaries: SharedDictionaryData {
                prefix: PrefixSources::new(attachments),
            },
            prepared: PreparedDictionaryIndexes {
                prefixes: prefixes.into_boxed_slice(),
            },
        })
    }

    /// Returns the caller's dictionary bytes.
    pub(crate) const fn dictionaries(&self) -> &SharedDictionaryData {
        &self.dictionaries
    }

    /// Returns the prepared index of the `index`-th attachment.
    ///
    /// `None` for an attachment too short to hold a single hashable position,
    /// which has an empty index rather than none at all.
    pub(crate) fn prepared_prefix(&self, index: usize) -> Option<&PreparedPrefix> {
        self.prepared.prefix(index)
    }

    /// Returns whether the context addresses any prefix bytes at all.
    ///
    /// An empty context is not a special case anywhere else: it is what makes
    /// a shared call produce exactly the bytes the ordinary call would.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.dictionaries.prefix().is_empty()
    }

    /// Returns the bytes this context owns, sources and indexes together.
    pub(crate) fn allocated_size(&self) -> usize {
        let bytes =
            size_of::<Self>() + self.dictionaries.allocated_size() + self.prepared.allocated_size();
        #[cfg(feature = "experimental")]
        let bytes = bytes
            + self
                .static_index
                .as_ref()
                .map_or(0, super::static_index::StaticIndex::allocated_size);
        bytes
    }

    /// Returns the longest match the attached dictionaries offer for `input`.
    ///
    /// The candidate order is the reference's: attachments are searched oldest
    /// first, each attachment's
    /// bucket chain newest first, and a candidate replaces the incumbent only
    /// when it is strictly longer — so of two equally long matches the one in
    /// the older attachment, and within an attachment the one at the newer
    /// position, wins. A match may run off the end of the attachment it
    /// started in and continue into the next, which is the virtual
    /// concatenation RFC 9841 defines.
    ///
    /// `None` when nothing matched, or when `input` is shorter than the eight
    /// bytes a bucket key is computed from: the prepared index holds no entry
    /// that could be probed with fewer.
    ///
    /// No encoder takes this path — a match finder calls
    /// [`SharedContextInner::find_match`] instead — so it is compiled only for
    /// the `diagnostics` feature that exposes it and for this module's tests.
    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn longest_prefix_match(&self, input: &[u8]) -> Option<PrefixMatch> {
        let head = u64::from_le_bytes(*input.first_chunk::<HASH_INPUT_BYTES>()?);
        let sources = self.dictionaries.prefix();
        let mut best: Option<PrefixMatch> = None;
        for attachment in 0..sources.segment_count() {
            let Some(index) = self.prepared.prefix(attachment) else {
                continue;
            };
            let base = sources.segment_start(attachment);
            for candidate in index.candidates(head) {
                let offset = base + u64::from(candidate);
                let length = sources.match_length(offset, &[], input, input.len());
                if length > best.map_or(0, |found: PrefixMatch| found.length) {
                    best = Some(PrefixMatch { offset, length });
                }
            }
        }
        best
    }
}

/// The resource limits a context is built under, in the form checks need.
///
/// A private mirror of the public limits: the public type is a small `Copy`
/// value with named accessors, and this is the flat set of numbers the
/// construction path compares against, so no public accessor is called inside
/// a loop.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Budget {
    /// Largest total of every attached source byte.
    pub(crate) max_total_source_bytes: u64,
    /// Largest logical LZ77 prefix.
    pub(crate) max_prefix_bytes: u64,
    /// Largest peak allocation preparing the context may reach.
    pub(crate) max_allocated_bytes: u64,
    /// Most attachments the context may hold, never above the format's fifteen.
    pub(crate) max_attachments: usize,
}

/// Returns an upper bound on the peak memory preparing this context will use.
///
/// Deliberately loose, and deliberately computed before anything is built: the
/// real figure needs the item counts, which need the build. Every term is an
/// over-estimate of the corresponding table, so a context that fits this fits
/// the limit it was checked against — and because it bounds the peak, it also
/// bounds the finished context.
fn estimate_allocation(attachments: &[Box<[u8]>]) -> Option<u64> {
    // What one attachment costs beyond its own bytes: the fat pointer holding
    // it and the cumulative offset that addresses it.
    const PER_ATTACHMENT: u64 = (size_of::<Box<[u8]>>() + size_of::<u64>()) as u64;

    let mut total = size_of::<SharedContextInner>() as u64;
    for attachment in attachments {
        let length = attachment.len() as u64;
        total = total.checked_add(length)?.checked_add(PER_ATTACHMENT)?;
        total = total.checked_add(PreparedPrefix::allocation_bound(attachment.len())?)?;
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    const GENEROUS: Budget = Budget {
        max_total_source_bytes: u64::MAX,
        max_prefix_bytes: u64::MAX,
        max_allocated_bytes: u64::MAX,
        max_attachments: MAX_PREFIX_DICTIONARIES,
    };

    fn attach(segments: &[&[u8]]) -> Vec<Box<[u8]>> {
        segments
            .iter()
            .map(|segment| segment.to_vec().into_boxed_slice())
            .collect()
    }

    fn context(segments: &[&[u8]]) -> SharedContextInner {
        SharedContextInner::new(attach(segments), &GENEROUS).expect("prepared")
    }

    fn items(context: &SharedContextInner, attachment: usize) -> usize {
        context
            .prepared
            .prefix(attachment)
            .map_or(0, PreparedPrefix::item_count)
    }

    fn search(context: &SharedContextInner, input: &[u8]) -> Option<PrefixMatch> {
        context.longest_prefix_match(input)
    }

    #[test]
    fn an_empty_context_owns_nothing_but_itself() {
        let context = SharedContextInner::default();
        assert!(context.is_empty());
        assert_eq!(context.dictionaries().source_size(), 0);
        assert_eq!(context.dictionaries().prefix().segment_count(), 0);
        assert!(context.prepared.prefix(0).is_none());
        assert_eq!(context.allocated_size(), size_of::<SharedContextInner>());
        assert_eq!(search(&context, b"anything at all"), None);
    }

    #[test]
    fn attachments_keep_their_order_and_their_indexes() {
        let first: &[u8] = b"the quick brown fox";
        let second: &[u8] = b"jumps over the lazy dog";
        let context = context(&[first, second]);
        assert!(!context.is_empty());
        assert_eq!(
            context.dictionaries().source_size(),
            first.len() + second.len()
        );
        assert_eq!(context.dictionaries().prefix().segment(0), first);
        assert_eq!(context.dictionaries().prefix().segment(1), second);
        assert_eq!(items(&context, 0), first.len() - 8 + 1);
        assert_eq!(items(&context, 1), second.len() - 8 + 1);
        assert!(context.prepared.prefix(2).is_none());
        assert!(context.allocated_size() > context.dictionaries().source_size());
    }

    #[test]
    fn too_many_attachments_are_refused_before_anything_is_built() {
        let segments = vec![b"payload".as_slice(); MAX_PREFIX_DICTIONARIES + 1];
        assert!(matches!(
            SharedContextInner::new(attach(&segments), &GENEROUS),
            Err(SharedBrotliError::TooManyPrefixDictionaries {
                attached: 16,
                limit: 15
            })
        ));
        // Exactly at the limit still prepares.
        let segments = vec![b"payload".as_slice(); MAX_PREFIX_DICTIONARIES];
        assert!(SharedContextInner::new(attach(&segments), &GENEROUS).is_ok());
    }

    #[test]
    fn a_prefix_past_its_limit_is_refused() {
        let limits = Budget {
            max_prefix_bytes: 8,
            ..GENEROUS
        };
        assert!(matches!(
            SharedContextInner::new(attach(&[b"nine byte".as_slice()]), &limits),
            Err(SharedBrotliError::DictionaryTooLarge { bytes: 9, limit: 8 })
        ));
        let limits = Budget {
            max_total_source_bytes: 8,
            ..GENEROUS
        };
        assert!(matches!(
            SharedContextInner::new(attach(&[b"four".as_slice(), b"five!".as_slice()]), &limits),
            Err(SharedBrotliError::DictionaryTooLarge { bytes: 9, limit: 8 })
        ));
    }

    #[test]
    fn an_allocation_past_its_limit_is_refused() {
        let limits = Budget {
            max_allocated_bytes: 1024,
            ..GENEROUS
        };
        // Eight bytes are enough to force a real index, which is far past a
        // kibibyte on its own.
        assert!(matches!(
            SharedContextInner::new(attach(&[b"eight!!!".as_slice()]), &limits),
            Err(SharedBrotliError::SharedContextTooLarge { .. })
        ));
        // A source too short to hash builds no index and still fits.
        assert!(SharedContextInner::new(attach(&[b"seven!!".as_slice()]), &limits).is_ok());
    }

    #[test]
    fn the_estimate_is_never_smaller_than_the_context_it_predicts() {
        let cases: Vec<Vec<Box<[u8]>>> = vec![
            Vec::new(),
            attach(&[b"short".as_slice()]),
            vec![b"a".to_vec().into_boxed_slice(); MAX_PREFIX_DICTIONARIES],
            vec![
                b"the quick brown fox jumps over the lazy dog"
                    .to_vec()
                    .into_boxed_slice(),
                vec![b'z'; 40_000].into_boxed_slice(),
            ],
        ];
        for attachments in cases {
            let estimate = estimate_allocation(&attachments).expect("no overflow");
            let context = SharedContextInner::new(attachments, &GENEROUS).expect("built");
            assert!(
                estimate >= context.allocated_size() as u64,
                "estimate {estimate} under {}",
                context.allocated_size()
            );
        }
    }

    #[test]
    fn a_search_needs_eight_bytes_to_probe_with() {
        let context = context(&[b"the quick brown fox".as_slice()]);
        for len in 0..8usize {
            assert_eq!(search(&context, &b"the quick"[..len]), None);
        }
        assert_eq!(
            search(&context, b"the quic"),
            Some(PrefixMatch {
                offset: 0,
                length: 8
            })
        );
    }

    #[test]
    fn a_search_finds_the_longest_match_and_where_it_is() {
        let context = context(&[b"the quick brown fox".as_slice()]);
        assert_eq!(
            search(&context, b"quick brown foxes"),
            Some(PrefixMatch {
                offset: 4,
                length: 15
            })
        );
        assert_eq!(search(&context, b"nothing here at all"), None);
    }

    #[test]
    fn a_search_crosses_the_seam_between_attachments() {
        // The candidate's own eight hashed bytes have to fit inside the
        // attachment that indexed them; the match it grows into does not.
        let context = context(&[
            b"the quick brown fox jum".as_slice(),
            b"ps over the lazy dog".as_slice(),
        ]);
        assert_eq!(
            search(&context, b"brown fox jumps ov"),
            Some(PrefixMatch {
                offset: 10,
                length: 18
            })
        );
    }

    #[test]
    fn a_search_prefers_the_longest_match_over_the_nearest() {
        // "brown fox" appears in both attachments; only the second continues
        // into the bytes the input carries on with, so the longer one wins
        // even though the shorter one is nearer the stream.
        let context = context(&[
            b"a brown fox jumped once".as_slice(),
            b"a brown fox jumps twice".as_slice(),
        ]);
        let found = search(&context, b"a brown fox jumps twice more").expect("a match");
        assert_eq!(found.offset, 23);
        assert_eq!(found.length, 23);
    }
}

//! The error every RFC 9841 feature reports through, inside the crate.
//!
//! This module is private. RFC 9841's caller-facing surface is the
//! [`dictionary`](super::dictionary) module and the Large Window arm of
//! [`Window`](super::Window); what lives here is the low-level error the
//! encoders and the prepared-dictionary builder raise, which the public
//! [`DictionaryError`](super::dictionary::DictionaryError) and
//! [`EncodeError`](super::EncodeError) are built from.

use thiserror::Error;

/// Error raised by the RFC 9841 machinery below the public API.
#[derive(Error, Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum SharedBrotliError {
    /// The requested quality has no Large Window implementation.
    ///
    /// Qualities zero, one and two write distances through a static entropy
    /// model built for the RFC 7932 alphabet, so they cannot carry the wider
    /// one. The public API refuses this combination when a `Compressor` is
    /// built, so an encoder only ever raises it as an internal invariant.
    #[error("Quality level {quality} does not implement large window Brotli")]
    UnsupportedLargeWindow {
        /// The numeric quality that was asked for.
        quality: usize,
    },
    /// More than fifteen prefix dictionaries were attached to one dictionary.
    ///
    /// RFC 9841 gives a distance no way to say which of a sixteenth
    /// dictionary's bytes it meant, so the limit is the format's.
    #[error("A prepared dictionary holds at most {limit} attachments, not {attached}")]
    TooManyPrefixDictionaries {
        /// How many attachments the builder was given.
        attached: usize,
        /// How many it may hold.
        limit: usize,
    },
    /// One attachment, or the whole logical prefix, is larger than allowed.
    #[error("A shared dictionary of {bytes} bytes exceeds the limit of {limit}")]
    DictionaryTooLarge {
        /// How many bytes were offered.
        bytes: u64,
        /// How many were allowed.
        limit: u64,
    },
    /// Preparing the dictionary would allocate more than the limit allows.
    ///
    /// Reported from an upper bound computed before anything is built, so the
    /// allocation the limit refused is never actually made.
    #[error("Preparing a dictionary would allocate {bytes} bytes, past the limit of {limit}")]
    SharedContextTooLarge {
        /// The predicted allocation.
        bytes: u64,
        /// How many bytes were allowed.
        limit: u64,
    },
}

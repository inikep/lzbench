//! Safe Rust Zstandard ([RFC 8878][rfc]) codecs.
//!
//! The encoder is a port of the reference implementation, zstd 1.5.7, and
//! writes the same bytes it does; the decoder reads every frame RFC 8878
//! defines, with or without a dictionary.
//!
//! # Byte identity
//!
//! For the same input and parameters, single-threaded compression produces
//! exactly the bytes of zstd 1.5.7: one-shot compression matches
//! `ZSTD_compress2`, and an encoder session matches `ZSTD_compressStream2`
//! driven with the same input chunks, output capacities and operations. The
//! contract covers compression levels -131072 to 22, the explicit
//! compression parameters (window, hash, chain and search logs, minimum
//! match, target length, strategy), long distance matching, the checksum and
//! content-size flags, declared input sizes, size hints, and raw-content and
//! Zstandard-format dictionaries ([`dictionary`]), which match the reference
//! after `ZSTD_CCtx_loadDictionary_advanced`.
//!
//! Parallel compression writes valid frames that decode to the input but are
//! not those of the reference's multithreaded mode. The magicless format,
//! `targetCBlockSize` and external sequence producers are not supported.
//!
//! # Choose your API
//!
//! Compression and decompression expose the same core I/O shapes. Pick the shape that
//! matches how your application already moves bytes. [`ParallelCompressor`] is a separate
//! execution strategy for compression: it splits one frame into independent segments and
//! hands them to your executor as tasks, either for a whole input at once or for input
//! that arrives in chunks. Each codec requires its Cargo feature; reader/writer adapters
//! and parallel compression are unavailable with `no_std`.
//!
//! | I/O shape | Compression | Decompression |
//! | --- | --- | --- |
//! | Return a new `Vec<u8>` | [`Compressor::compress`][compressor-compress] | [`Decompressor::decompress`][decompressor-decompress] |
//! | Append to an existing Vec | [`compress_into`][compressor-compress-into] | [`decompress_into`][decompressor-decompress-into] |
//! | Write into a caller-owned slice | [`compress_to_slice`][compressor-compress-to-slice], [`compress_to_uninit`][compressor-compress-to-uninit] | [`decompress_to_slice`][decompressor-decompress-to-slice] |
//! | Pull output through `std::io::Read` | [`Compressor::reader`][compressor-reader] | [`Decompressor::reader`][decompressor-reader] |
//! | Push input through `std::io::Write` | [`Compressor::writer`][compressor-writer] | [`Decompressor::writer`][decompressor-writer] |
//! | Drive input/output incrementally | [`start`][compressor-start] → [`EncoderSession`] | [`start`][decompressor-start] → [`DecoderSession`] |
//! | Same, with the session owning the codec | [`into_session`][compressor-into-session] → [`EncoderSessionOwned`] | [`into_session`][decompressor-into-session] → [`DecoderSessionOwned`] |
//! | Split one known input across your threads | [`prepare_slice`][parallel-prepare-slice] / [`prepare_source`][parallel-prepare-source] → [`ScopedParallelBatch`] | — |
//! | Stream chunked input across your threads | [`start`][parallel-start] → [`ParallelEncoderSession`] + [`ParallelTaskQueue`] | — |
//! | Same, with the session owning the compressor | [`into_session`][parallel-into-session] → [`ParallelEncoderSessionOwned`] | — |
//! | Push or pull through `std::io` across your threads | [`writer`][parallel-writer], [`reader`][parallel-reader] | — |
//!
//! Each compression and decompression shape has a `_with_dictionary` form
//! that takes a dictionary from [`dictionary`].
//!
//! [`EncoderConfig`] selects the level and frame flags, and
//! [`CompressionParams`] overrides individual parameters exactly as
//! `ZSTD_CCtx_setParameter` would. [`DecodeLimits`] bound the window, input
//! and output, so untrusted frames cannot make the decoder allocate without
//! limit.
//!
//! ```
//! # #[cfg(all(feature = "compression", feature = "decompression"))]
//! # {
//! use mzstd::{Compressor, Decompressor, EncoderConfig, Level};
//!
//! let mut encoder = Compressor::new(EncoderConfig::default().with_level(Level::try_from(19)?))?;
//! let payload = "the quick brown fox ".repeat(500);
//! let compressed = encoder.compress(payload.as_bytes())?;
//! assert!(compressed.len() < payload.len() / 50);
//! assert_eq!(Decompressor::default().decompress(&compressed)?, payload.as_bytes());
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Structured, seekable containers
//!
//! With the `framed` feature, the [`framing` module][framing-module] holds ordered
//! resources, metadata, padding, a central directory and a footer in one
//! container built only from Zstandard frames: visible resources are
//! regular frames that any Zstandard decoder extracts, everything else lives
//! in skippable frames, and a final seek table in zstd's
//! `contrib/seekable_format` layout lets seekable decoders map offsets to
//! frames. The [framed compressor][framed-compressor] writes containers;
//! the [framed decompressor][framed-decompressor] validates and decodes them
//! as events or as owned resources, and with `std` reads single resources
//! through `Read + Seek`. The API mirrors mbrotli's framing API.
//!
//! ```
//! # #[cfg(all(feature = "framed", feature = "compression", feature = "decompression"))]
//! # {
//! use mzstd::framing::{FramedCompressor, FramedDecompressor, FramedItem, FramedResource};
//!
//! let items = [
//!     FramedItem::Resource(FramedResource::from(&b"first"[..])),
//!     FramedItem::Resource(FramedResource::from(&b"second"[..])),
//! ];
//! let container = FramedCompressor::new(Default::default())?.compress(items.as_slice().into())?;
//! let output = FramedDecompressor::new(Default::default())?.decompress(&container)?;
//! assert_eq!(output.resources[1].data, b"second");
//! // Any Zstandard decoder sees the visible resources, concatenated.
//! assert_eq!(mzstd::Decompressor::default().decompress(&container)?, b"firstsecond");
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Select only the codecs you need
//!
//! The default feature set is `std`, `compression`, and `decompression`. Disable default
//! features to select one codec, alongside `std` or `no_std` (which needs `alloc`):
//!
//! ```toml
//! mzstd = { version = "0.2", default-features = false, features = ["std", "decompression"] }
//! ```
//!
//! A disabled codec has no module and no re-exports. The `std::io` adapters
//! and parallel compression need `std`. `framed` adds the container APIs for
//! whichever codecs are enabled; its `std::io` adapters and seek reader need
//! `std`.
//!
//! [rfc]: https://www.rfc-editor.org/rfc/rfc8878
// Resolve available APIs locally; unavailable APIs link to feature selection.
#![cfg_attr(
    all(
        feature = "framed",
        any(feature = "compression", feature = "decompression")
    ),
    doc = "[framing-module]: crate::framing"
)]
#![cfg_attr(
    not(all(
        feature = "framed",
        any(feature = "compression", feature = "decompression")
    )),
    doc = "[framing-module]: #select-only-the-codecs-you-need"
)]
#![cfg_attr(
    all(feature = "framed", feature = "compression"),
    doc = "[framed-compressor]: crate::framing::FramedCompressor"
)]
#![cfg_attr(
    not(all(feature = "framed", feature = "compression")),
    doc = "[framed-compressor]: #select-only-the-codecs-you-need"
)]
#![cfg_attr(
    all(feature = "framed", feature = "decompression"),
    doc = "[framed-decompressor]: crate::framing::FramedDecompressor"
)]
#![cfg_attr(
    not(all(feature = "framed", feature = "decompression")),
    doc = "[framed-decompressor]: #select-only-the-codecs-you-need"
)]
#![cfg_attr(
    feature = "compression",
    doc = r#"
[compressor-compress]: crate::Compressor::compress
[compressor-compress-into]: crate::Compressor::compress_into
[compressor-compress-to-slice]: crate::Compressor::compress_to_slice
[compressor-compress-to-uninit]: crate::Compressor::compress_to_uninit
[compressor-start]: crate::Compressor::start
[`EncoderSession`]: crate::EncoderSession
[compressor-into-session]: crate::Compressor::into_session
[`EncoderSessionOwned`]: crate::EncoderSessionOwned
[`EncoderConfig`]: crate::EncoderConfig
[`CompressionParams`]: crate::CompressionParams
"#
)]
#![cfg_attr(
    not(feature = "compression"),
    doc = r#"
[compressor-compress]: #select-only-the-codecs-you-need
[compressor-compress-into]: #select-only-the-codecs-you-need
[compressor-compress-to-slice]: #select-only-the-codecs-you-need
[compressor-compress-to-uninit]: #select-only-the-codecs-you-need
[compressor-start]: #select-only-the-codecs-you-need
[`EncoderSession`]: #select-only-the-codecs-you-need
[compressor-into-session]: #select-only-the-codecs-you-need
[`EncoderSessionOwned`]: #select-only-the-codecs-you-need
[`EncoderConfig`]: #select-only-the-codecs-you-need
[`CompressionParams`]: #select-only-the-codecs-you-need
"#
)]
#![cfg_attr(
    feature = "decompression",
    doc = r#"
[decompressor-decompress]: crate::Decompressor::decompress
[decompressor-decompress-into]: crate::Decompressor::decompress_into
[decompressor-decompress-to-slice]: crate::Decompressor::decompress_to_slice
[decompressor-start]: crate::Decompressor::start
[`DecoderSession`]: crate::DecoderSession
[decompressor-into-session]: crate::Decompressor::into_session
[`DecoderSessionOwned`]: crate::DecoderSessionOwned
[`DecodeLimits`]: crate::DecodeLimits
"#
)]
#![cfg_attr(
    not(feature = "decompression"),
    doc = r#"
[decompressor-decompress]: #select-only-the-codecs-you-need
[decompressor-decompress-into]: #select-only-the-codecs-you-need
[decompressor-decompress-to-slice]: #select-only-the-codecs-you-need
[decompressor-start]: #select-only-the-codecs-you-need
[`DecoderSession`]: #select-only-the-codecs-you-need
[decompressor-into-session]: #select-only-the-codecs-you-need
[`DecoderSessionOwned`]: #select-only-the-codecs-you-need
[`DecodeLimits`]: #select-only-the-codecs-you-need
"#
)]
#![cfg_attr(
    all(feature = "compression", not(feature = "no_std")),
    doc = r#"
[compressor-reader]: crate::Compressor::reader
[compressor-writer]: crate::Compressor::writer
[`ParallelCompressor`]: crate::compressor::parallel::ParallelCompressor
[parallel-prepare-slice]: crate::compressor::parallel::ParallelCompressor::prepare_slice
[parallel-prepare-source]: crate::compressor::parallel::ParallelCompressor::prepare_source
[`ScopedParallelBatch`]: crate::compressor::parallel::ScopedParallelBatch
[parallel-start]: crate::compressor::parallel::ParallelCompressor::start
[`ParallelEncoderSession`]: crate::compressor::parallel::ParallelEncoderSession
[`ParallelTaskQueue`]: crate::compressor::parallel::ParallelTaskQueue
[parallel-into-session]: crate::compressor::parallel::ParallelCompressor::into_session
[`ParallelEncoderSessionOwned`]: crate::compressor::parallel::ParallelEncoderSessionOwned
[parallel-writer]: crate::compressor::parallel::ParallelCompressor::writer
[parallel-reader]: crate::compressor::parallel::ParallelCompressor::reader
"#
)]
#![cfg_attr(
    not(all(feature = "compression", not(feature = "no_std"))),
    doc = r#"
[compressor-reader]: #select-only-the-codecs-you-need
[compressor-writer]: #select-only-the-codecs-you-need
[`ParallelCompressor`]: #select-only-the-codecs-you-need
[parallel-prepare-slice]: #select-only-the-codecs-you-need
[parallel-prepare-source]: #select-only-the-codecs-you-need
[`ScopedParallelBatch`]: #select-only-the-codecs-you-need
[parallel-start]: #select-only-the-codecs-you-need
[`ParallelEncoderSession`]: #select-only-the-codecs-you-need
[`ParallelTaskQueue`]: #select-only-the-codecs-you-need
[parallel-into-session]: #select-only-the-codecs-you-need
[`ParallelEncoderSessionOwned`]: #select-only-the-codecs-you-need
[parallel-writer]: #select-only-the-codecs-you-need
[parallel-reader]: #select-only-the-codecs-you-need
"#
)]
#![cfg_attr(
    all(feature = "decompression", not(feature = "no_std")),
    doc = r#"
[decompressor-reader]: crate::Decompressor::reader
[decompressor-writer]: crate::Decompressor::writer
"#
)]
#![cfg_attr(
    not(all(feature = "decompression", not(feature = "no_std"))),
    doc = r#"
[decompressor-reader]: #select-only-the-codecs-you-need
[decompressor-writer]: #select-only-the-codecs-you-need
"#
)]
#![cfg_attr(feature = "no_std", no_std)]
#![cfg_attr(not(test), forbid(unsafe_code))]
#![deny(missing_docs)]
#![deny(missing_debug_implementations)]
#![deny(rustdoc::broken_intra_doc_links)]

#[cfg(any(feature = "compression", feature = "decompression"))]
extern crate alloc;

#[cfg(any(feature = "compression", feature = "decompression"))]
mod backend;
#[cfg(any(feature = "compression", feature = "decompression"))]
pub mod dictionary;
#[cfg(any(feature = "compression", feature = "decompression"))]
mod retention;
#[cfg(any(feature = "compression", feature = "decompression"))]
mod shared;
#[cfg(any(feature = "compression", feature = "decompression"))]
pub use backend::Backend;
#[cfg(any(feature = "compression", feature = "decompression"))]
pub use retention::RetentionPolicy;

#[cfg(all(
    any(feature = "compression", feature = "decompression"),
    not(feature = "no_std")
))]
mod finish_error;
#[cfg(all(
    any(feature = "compression", feature = "decompression"),
    not(feature = "no_std")
))]
pub mod io;

#[cfg(feature = "compression")]
pub mod compressor;
#[cfg(feature = "compression")]
pub use compressor::{
    CompressionParams, Compressor, CompressorBuilder, ConfigError, EncodeError, EncoderConfig,
    EncoderSession, EncoderSessionOwned, EncoderStatus, InputSize, LdmParams, Level,
    LongDistanceMatching, Operation, Parameter, Progress, SizeOverflow, Strategy, StreamConfig,
};

#[cfg(feature = "decompression")]
pub mod decompressor;

#[cfg(all(
    feature = "framed",
    any(feature = "compression", feature = "decompression")
))]
pub mod framing;
#[cfg(all(feature = "framed", feature = "compression"))]
pub use compressor::framing::{
    FramedCompressor, FramedCompressorBuilder, FramedEncodeConfig, FramedEncodeError,
    FramedEncodeFailure, FramedEncodeLocation, FramedEncodeOperation, FramedEncodeProgress,
    FramedEncodeStreamConfig, FramedEncoderSession, FramedEncoderSessionOwned, FramedEncoderStatus,
    FramedResourceSession,
};
#[cfg(all(feature = "framed", feature = "decompression"))]
pub use decompressor::framing::{
    FramedDecodeConfig, FramedDecodeError, FramedDecodeFailure, FramedDecodeLimits,
    FramedDecodeStreamConfig, FramedDecoderSession, FramedDecoderSessionOwned, FramedDecompressor,
    FramedOutput,
};

#[cfg(feature = "decompression")]
pub use decompressor::{
    DecodeConfigError, DecodeError, DecodeFailure, DecodeLimits, DecodeOperation, DecodeProgress,
    DecoderConfig, DecoderSession, DecoderSessionOwned, DecoderStatus, Decompressor,
    DecompressorBuilder, FrameMode, InvalidDataKind,
};

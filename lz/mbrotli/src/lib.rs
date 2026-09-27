//! Safe Rust Brotli codecs, independently selected with Cargo features.
//!
//! `compression` and `decompression` are enabled by default. Disable default
//! features and select either codec alongside `std` or `no_std`. For example:
//!
//! ```toml
//! mbrotli = { version = "0.5.2", default-features = false, features = ["std", "decompression"] }
//! ```
//!
//! A disabled codec has no module, API re-exports, dictionaries or I/O adapters.
//! Shared `Backend` and `RetentionPolicy` types exist when either codec is enabled.
//! With neither codec enabled, the crate exposes no codec API.
//!
//! # Choose your API
//!
//! Compression and decompression expose the same core I/O shapes. Pick the shape that
//! matches how your application already moves bytes; parallel compression is a separate
//! execution strategy, not another streaming API. Each codec requires its Cargo
//! feature; reader/writer adapters and parallel compression are unavailable with `no_std`.
//!
//! | I/O shape | Compression | Decompression |
//! | --- | --- | --- |
//! | Return a new `Vec<u8>` | [`Compressor::compress`][compressor-compress] | [`Decompressor::decompress`][decompressor-decompress] |
//! | Append to an existing Vec | [`compress_into`][compressor-compress-into] | [`decompress_into`][decompressor-decompress-into] |
//! | Write into a caller-owned slice | [`compress_to_slice`][compressor-compress-to-slice] | [`decompress_to_slice`][decompressor-decompress-to-slice] |
//! | Pull output through `std::io::Read` | [`Compressor::reader`][compressor-reader] | [`Decompressor::reader`][decompressor-reader] |
//! | Push input through `std::io::Write` | [`Compressor::writer`][compressor-writer] | [`Decompressor::writer`][decompressor-writer] |
//! | Drive input/output incrementally | [`start`][compressor-start] → [`EncoderSession`][encoder-session] | [`start`][decompressor-start] → [`DecoderSession`][decoder-session] |
//! | Same, with the session owning the codec | [`into_session`][compressor-into-session] → [`EncoderSessionOwned`][encoder-session-owned] | [`into_session`][decompressor-into-session] → [`DecoderSessionOwned`][decoder-session-owned] |
//!
//! ## Reuse memory between payloads
//!
//! [`Compressor`][compressor] and [`Decompressor`][decompressor] own reusable working state. Keep the codec and your
//! output buffer alive across operations when allocation reuse matters.
//!
//! <details>
//! <summary>Example: reuse the compressor and output buffer</summary>
//!
//! ```rust
//! # #[cfg(feature = "compression")]
//! # mod example {
//! use mbrotli::{Compressor, EncoderConfig, Quality};
//!
//! pub fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut encoder = Compressor::new(
//!         EncoderConfig::default().with_quality(Quality::Q5),
//!     )?;
//!     let mut output = Vec::new();
//!
//!     for input in [b"first payload".as_slice(), b"second payload".as_slice()] {
//!         output.clear(); // Keep the allocation; compress_into appends.
//!         let written = encoder.compress_into(input, &mut output)?;
//!         assert_eq!(written, 0..output.len());
//!     }
//!     Ok(())
//! }
//! # }
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! #     #[cfg(feature = "compression")]
//! #     example::main()?;
//! #     Ok(())
//! # }
//! ```
//!
//! </details>
//!
//! # `Read` and `Write` streaming
//!
//! Both codecs provide synchronous adapters for the standard Rust I/O traits. The two
//! adapter shapes are complementary:
//!
//! - `reader(...)` wraps an input `Read` and exposes transformed bytes through `Read`.
//! - `writer(...)` wraps an output `Write` and accepts source bytes through `Write`.
//!
//! That means you can plug Brotli into an existing pull-based or push-based pipeline
//! without first collecting the whole payload in memory.
//!
//! ## Compression I/O
//!
//! [`Compressor::reader`][compressor-reader] consumes **uncompressed** bytes from a `Read` source and yields
//! **compressed** bytes. [`Compressor::writer`][compressor-writer] accepts **uncompressed** bytes and writes
//! **compressed** bytes to its sink. Encoder writers must be explicitly finished;
//! dropping one abandons the stream.
//!
//! <details>
//! <summary>Compression with both <code>Read</code> and <code>Write</code></summary>
//!
//! ```rust
//! # #[cfg(all(feature = "compression", not(feature = "no_std")))]
//! # mod example {
//! use mbrotli::{Compressor, EncoderConfig, InputSize, Quality};
//! use std::io::{Read, Write};
//!
//! pub fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let payload = b"streamed payload";
//!     // Pull model: read compressed bytes from an uncompressed source.
//!     let mut encoder = Compressor::new(
//!         EncoderConfig::default().with_quality(Quality::Q5),
//!     )?;
//!     let stream = InputSize::Exact(payload.len() as u64).into();
//!     let mut reader = encoder.reader(&payload[..], stream)?;
//!     let mut compressed_from_reader = Vec::new();
//!     reader.read_to_end(&mut compressed_from_reader)?;
//!
//!     // Push model: write uncompressed bytes into a compressed sink.
//!     let mut encoder = Compressor::new(
//!         EncoderConfig::default().with_quality(Quality::Q5),
//!     )?;
//!     let stream = InputSize::Exact(payload.len() as u64).into();
//!     let mut writer = encoder.writer(Vec::new(), stream)?;
//!     writer.write_all(payload)?;
//!     let compressed_from_writer = writer
//!         .finish()
//!         .map_err(mbrotli::io::FinishError::into_error)?;
//!
//!     assert_eq!(compressed_from_reader, compressed_from_writer);
//!     Ok(())
//! }
//! # }
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! #     #[cfg(all(feature = "compression", not(feature = "no_std")))]
//! #     example::main()?;
//! #     Ok(())
//! # }
//! ```
//!
//! </details>
//!
//! [`flush()`][encoder-flush] makes accepted input decodable without ending the encoder stream, and flush
//! boundaries can affect compressed bytes. Use [`finish()`][encoder-finish] when the stream is complete.
//!
//! ## Decompression I/O
//!
//! [`Decompressor::reader`][decompressor-reader] consumes a **compressed** `Read` source and yields the
//! **decompressed** payload. [`Decompressor::writer`][decompressor-writer] accepts **compressed** bytes and
//! writes the **decompressed** payload to its sink.
//!
//! <details>
//! <summary>Decompression with both <code>Read</code> and <code>Write</code></summary>
//!
//! ```rust
//! # #[cfg(all(feature = "decompression", not(feature = "no_std")))]
//! # mod example {
//! use mbrotli::{DecodeStreamConfig, DecoderConfig, Decompressor};
//! use std::io::{Read, Write};
//!
//! fn decode_with_reader(compressed: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
//!     let mut decoder = Decompressor::new(DecoderConfig::default())?;
//!     let mut reader = decoder.reader(compressed, DecodeStreamConfig::default())?;
//!     let mut output = Vec::new();
//!     reader.read_to_end(&mut output)?;
//!     Ok(output)
//! }
//!
//! fn decode_with_writer(compressed: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
//!     let mut decoder = Decompressor::new(DecoderConfig::default())?;
//!     let mut writer = decoder.writer(Vec::new(), DecodeStreamConfig::default())?;
//!     writer.write_all(compressed)?;
//!     Ok(writer
//!         .finish()
//!         .map_err(mbrotli::io::FinishError::into_error)?)
//! }
//!
//! # pub fn main() -> Result<(), Box<dyn std::error::Error>> {
//! #     assert!(decode_with_reader(&[0x3b])?.is_empty());
//! #     assert!(decode_with_writer(&[0x3b])?.is_empty());
//! #     Ok(())
//! # }
//! # }
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! #     #[cfg(all(feature = "decompression", not(feature = "no_std")))]
//! #     example::main()?;
//! #     Ok(())
//! # }
//! ```
//!
//! </details>
//!
//! The decoder adapters use bounded internal buffering. Reader read-ahead can be recovered
//! with [`into_parts()`][decoder-into-parts], while decoder-writer finalization reports truncated or invalid
//! input instead of silently accepting an incomplete stream.
//!
//! # Parallel compression
//!
//! Parallel compression is independent of the `Read`/`Write` adapters. It changes how
//! one compression job is scheduled: mbrotli splits the input into segments, exposes
//! work items to the caller, and assembles the completed segments back into one Brotli
//! stream. The library does not create or own a thread pool.
//!
//! <details>
//! <summary>Example: parallel compression with scoped threads</summary>
//!
//! ```rust
//! # #[cfg(all(feature = "compression", not(feature = "no_std")))]
//! # mod example {
//! use mbrotli::compressor::parallel::{
//!     BatchConfig, ParallelCompressor, ParallelConfig, TaskCount,
//! };
//! use mbrotli::{EncoderConfig, Quality};
//!
//! pub fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let input = vec![b'a'; 8 << 20];
//!     let mut encoder = ParallelCompressor::new(
//!         EncoderConfig::default().with_quality(Quality::Q5),
//!         ParallelConfig::default(),
//!     )?;
//!     let mut batch = encoder.prepare_slice(
//!         &input,
//!         BatchConfig::auto(TaskCount::try_from(2)?),
//!     )?;
//!     let tasks = batch.take_tasks()?;
//!
//!     std::thread::scope(|scope| {
//!         for task in tasks {
//!             scope.spawn(move || task.run());
//!         }
//!     });
//!
//!     let mut output = Vec::new();
//!     let result = batch.finish_into(&mut output)?;
//!     assert_eq!(result.stats.effective_tasks, 2);
//!     assert!(output.len() < input.len());
//!     println!("{} -> {} bytes", input.len(), output.len());
//!     Ok(())
//! }
//! # }
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! #     #[cfg(all(feature = "compression", not(feature = "no_std")))]
//! #     example::main()?;
//! #     Ok(())
//! # }
//! ```
//!
//! </details>
//!
//! For fixed segment settings, parallel output is deterministic across task counts, but
//! it can differ in bytes and size from serial compression. The example stages compressed
//! segments in memory; see the [parallel guide][parallel] for budgets, disk staging,
//! file input, and other executors.
//!
//! # Native decompression
//!
//! [`Decompressor`][decompressor] provides reusable Vec/slice APIs, incremental sessions, and synchronous
//! reader/writer adapters. Vec appends are rolled back on failure. The decoder is
//! currently scalar Rust; SIMD acceleration applies to the encoder.
//!
//! **Configure limits for untrusted input.** Numeric budgets are unlimited by default.
//! This example accepts standard windows and sets explicit input, output, and workspace
//! budgets; choose limits appropriate for your application.
//!
//! ```rust
//! # #[cfg(feature = "decompression")]
//! # mod example {
//! use mbrotli::{DecodeLimits, DecoderConfig, Decompressor, WindowLimit};
//!
//! fn decode_payload(input: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
//!     let limits = DecodeLimits::default()
//!         .with_max_input_bytes(Some(1 << 20))
//!         .with_max_output_bytes(Some(8 << 20))
//!         .with_max_workspace_bytes(Some(32 << 20));
//!     let config = DecoderConfig::default()
//!         .with_window_limit(WindowLimit::standard(24)?)
//!         .with_limits(limits);
//!     let mut decoder = Decompressor::new(config)?;
//!
//!     Ok(decoder.decompress(input)?)
//! }
//!
//! # pub fn main() -> Result<(), Box<dyn std::error::Error>> {
//! #     assert!(decode_payload(&[0x3b])?.is_empty());
//! #     Ok(())
//! # }
//! # }
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! #     #[cfg(feature = "decompression")]
//! #     example::main()?;
//! #     Ok(())
//! # }
//! ```
//!
//! The workspace budget excludes caller-owned output and borrowed dictionaries; it is
//! not a total process-memory limit. Retain the decoder across calls when reuse matters.
//! See [decoder configuration and semantics][decoder] and [compatibility evidence][decoder-checks].
//!
//! # Structured framing
//!
//! With `compression,experimental`, [`framing::FramedCompressor`][framed-compressor]
//! owns reusable raw and container storage. Borrowed input preserves resource/metadata order.
//! Native sessions and one-shot operations support alloc; framed Read/Write
//! adapters require std APIs.
//!
//! ```
//! # #[cfg(all(feature = "compression", feature = "experimental"))]
//! # {
//! use mbrotli::framing::{FramedCompressor, FramedInput, FramedItem, FramedResource};
//! let items = [FramedItem::Resource(FramedResource::from(&b"hello"[..]))];
//! let mut encoder = FramedCompressor::new(Default::default())?;
//! let bytes = encoder.compress(FramedInput::from(items.as_slice()))?;
//! assert_eq!(&bytes[..4], &[0x91, 10, 66, 82]);
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! With `decompression,experimental`, [`FramedDecompressor`][framed-decompressor]
//! decodes a container into [`FramedOutput`][framed-output]: resources in wire order
//! with their metadata, global metadata, and the validated container layout.
//! Keep the decoder across calls to reuse its storage.
//! Compression support is optional; the decoder can be built on its own.
//!
//! To decode a container in `bytes`:
//!
//! ```
//! # #[cfg(all(feature = "decompression", feature = "experimental"))]
//! # {
//! # let bytes = *b"\x91\x0aBR\x00\x08\x02\x00\x00hello";
//! use mbrotli::framing::FramedDecompressor;
//! let mut decoder = FramedDecompressor::new(Default::default())?;
//! let output = decoder.decompress(&bytes)?;
//! assert_eq!(output.resources.len(), 1);
//! assert_eq!(output.resources[0].data, b"hello");
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! For incremental input, [`decoder.start(stream_config)`][framed-decompressor-start]
//! exposes resource and metadata events with payload fragments. With std APIs,
//! [`decoder.framed_reader(source, stream_config)`][framed-decompressor-reader]
//! wraps a `BufRead` source and exposes [`next_event()`][framed-reader-next-event].
//! This event reader preserves resource boundaries; it does not flatten the container
//! into one `Read` stream.
//!
//! [`FramedDecodeConfig`][framed-decode-config] defaults to
//! [`InputMode::FramedOnly`][framed-only]; [`InputMode::Auto`][framed-auto] also
//! accepts a raw Brotli member. [`FramedDecodeLimits`][framed-decode-limits] configures
//! resource, input, output, metadata and workspace budgets. External dictionary
//! references use an explicit [`DictionaryResolver`][dictionary-resolver]; resource
//! checksums are recorded without verification.
//! The owner, one-shot APIs and native sessions also work with `no_std` and `alloc`;
//! [`FramedReader`][framed-reader] requires std APIs. See [framed decoder mechanics][framed-decoder]
//! for validation, dictionaries and streaming semantics.
//!
//! # Select only the codecs you need
//!
//! The default feature set is `std`, `compression`, and `decompression`. Disable default
//! features to select one codec or use `no_std` with `alloc`. For an alloc-backed decoder:
//!
//! ```toml
//! [dependencies]
//! mbrotli = { version = "0.5.2", default-features = false, features = ["no_std", "decompression"] }
//! ```
//!
//! Add `"compression"` for both codecs, or use `"std"` instead of `"no_std"` for standard
//! I/O support. `no_std` requires a global allocator; it excludes I/O adapters, parallel
//! compression, framed I/O adapters, and profiling, and uses compile-time SIMD selection.
//! Cargo features are additive: another dependency can re-enable a codec or `std`.
//! Leave `std` and `hotpath*` disabled throughout the dependency graph for a std-free build.
//!
//! [parallel]: https://github.com/Mnwa/mbrotli/blob/master/docs/parallel.md
//! [decoder]: https://github.com/Mnwa/mbrotli/blob/master/architecture/decompressor.md
//! [framed-decoder]: https://github.com/Mnwa/mbrotli/blob/master/architecture/framed-decoder.md
//! [decoder-checks]: https://github.com/Mnwa/mbrotli/blob/master/architecture/decompressor-compatibility.md
// Resolve available APIs locally; unavailable APIs link to feature selection.
#![cfg_attr(
    all(feature = "compression", feature = "experimental"),
    doc = "[framed-compressor]: crate::framing::FramedCompressor"
)]
#![cfg_attr(
    not(all(feature = "compression", feature = "experimental")),
    doc = "[framed-compressor]: #select-only-the-codecs-you-need"
)]
#![cfg_attr(
    all(feature = "decompression", feature = "experimental"),
    doc = r#"
[framed-decompressor]: crate::framing::FramedDecompressor
[framed-output]: crate::framing::FramedOutput
[framed-decompressor-start]: crate::framing::FramedDecompressor::start
[framed-decode-config]: crate::framing::FramedDecodeConfig
[framed-only]: crate::framing::InputMode::FramedOnly
[framed-auto]: crate::framing::InputMode::Auto
[framed-decode-limits]: crate::framing::FramedDecodeLimits
[dictionary-resolver]: crate::framing::DictionaryResolver
"#
)]
#![cfg_attr(
    not(all(feature = "decompression", feature = "experimental")),
    doc = r#"
[framed-decompressor]: #select-only-the-codecs-you-need
[framed-output]: #select-only-the-codecs-you-need
[framed-decompressor-start]: #select-only-the-codecs-you-need
[framed-decode-config]: #select-only-the-codecs-you-need
[framed-only]: #select-only-the-codecs-you-need
[framed-auto]: #select-only-the-codecs-you-need
[framed-decode-limits]: #select-only-the-codecs-you-need
[dictionary-resolver]: #select-only-the-codecs-you-need
"#
)]
#![cfg_attr(
    all(
        feature = "decompression",
        feature = "experimental",
        not(feature = "no_std")
    ),
    doc = r#"
[framed-decompressor-reader]: crate::framing::FramedDecompressor::framed_reader
[framed-reader-next-event]: crate::framing::FramedReader::next_event
[framed-reader]: crate::framing::FramedReader
"#
)]
#![cfg_attr(
    not(all(
        feature = "decompression",
        feature = "experimental",
        not(feature = "no_std")
    )),
    doc = r#"
[framed-decompressor-reader]: #select-only-the-codecs-you-need
[framed-reader-next-event]: #select-only-the-codecs-you-need
[framed-reader]: #select-only-the-codecs-you-need
"#
)]
#![cfg_attr(
    feature = "compression",
    doc = r#"
[compressor]: crate::Compressor
[compressor-compress]: crate::Compressor::compress
[compressor-compress-into]: crate::Compressor::compress_into
[compressor-compress-to-slice]: crate::Compressor::compress_to_slice
[compressor-start]: crate::Compressor::start
[encoder-session]: crate::EncoderSession
[compressor-into-session]: crate::Compressor::into_session
[encoder-session-owned]: crate::EncoderSessionOwned
"#
)]
#![cfg_attr(
    not(feature = "compression"),
    doc = r#"
[compressor]: #select-only-the-codecs-you-need
[compressor-compress]: #select-only-the-codecs-you-need
[compressor-compress-into]: #select-only-the-codecs-you-need
[compressor-compress-to-slice]: #select-only-the-codecs-you-need
[compressor-start]: #select-only-the-codecs-you-need
[encoder-session]: #select-only-the-codecs-you-need
[compressor-into-session]: #select-only-the-codecs-you-need
[encoder-session-owned]: #select-only-the-codecs-you-need
"#
)]
#![cfg_attr(
    feature = "decompression",
    doc = r#"
[decompressor]: crate::Decompressor
[decompressor-decompress]: crate::Decompressor::decompress
[decompressor-decompress-into]: crate::Decompressor::decompress_into
[decompressor-decompress-to-slice]: crate::Decompressor::decompress_to_slice
[decompressor-start]: crate::Decompressor::start
[decoder-session]: crate::DecoderSession
[decompressor-into-session]: crate::Decompressor::into_session
[decoder-session-owned]: crate::DecoderSessionOwned
"#
)]
#![cfg_attr(
    not(feature = "decompression"),
    doc = r#"
[decompressor]: #select-only-the-codecs-you-need
[decompressor-decompress]: #select-only-the-codecs-you-need
[decompressor-decompress-into]: #select-only-the-codecs-you-need
[decompressor-decompress-to-slice]: #select-only-the-codecs-you-need
[decompressor-start]: #select-only-the-codecs-you-need
[decoder-session]: #select-only-the-codecs-you-need
[decompressor-into-session]: #select-only-the-codecs-you-need
[decoder-session-owned]: #select-only-the-codecs-you-need
"#
)]
#![cfg_attr(
    all(feature = "compression", not(feature = "no_std")),
    doc = r#"
[compressor-reader]: crate::Compressor::reader
[compressor-writer]: crate::Compressor::writer
[encoder-flush]: crate::io::EncoderWriter#method.flush
[encoder-finish]: crate::io::EncoderWriter::finish
"#
)]
#![cfg_attr(
    not(all(feature = "compression", not(feature = "no_std"))),
    doc = r#"
[compressor-reader]: #select-only-the-codecs-you-need
[compressor-writer]: #select-only-the-codecs-you-need
[encoder-flush]: #select-only-the-codecs-you-need
[encoder-finish]: #select-only-the-codecs-you-need
"#
)]
#![cfg_attr(
    all(feature = "decompression", not(feature = "no_std")),
    doc = r#"
[decompressor-reader]: crate::Decompressor::reader
[decompressor-writer]: crate::Decompressor::writer
[decoder-into-parts]: crate::io::DecoderReader::into_parts
"#
)]
#![cfg_attr(
    not(all(feature = "decompression", not(feature = "no_std"))),
    doc = r#"
[decompressor-reader]: #select-only-the-codecs-you-need
[decompressor-writer]: #select-only-the-codecs-you-need
[decoder-into-parts]: #select-only-the-codecs-you-need
"#
)]
#![cfg_attr(feature = "compression", doc = include_str!("compressor.md"))]
// The port is safe Rust by construction: the bit writer, the match scans and
// the SIMD kernels all shed their bounds checks through `as_chunks`,
// `first_chunk` and const-generic widths rather than through raw pointers.
// `forbid` rather than `deny`, so no module can opt back in.
//
// The differential unit tests inside `core::hq` and `core::rfc9841` call
// Google's C encoder through `google-brotli-ffi` to compare a stage against
// its reference, which is unavoidably `unsafe`. Those live behind `cfg(test)`
// and reach nothing that ships, so the ban is on everything but the test
// build rather than weakened to a `deny` the shipped code could opt out of.
#![cfg_attr(not(test), forbid(unsafe_code))]
#![cfg_attr(feature = "no_std", no_std)]
#![cfg_attr(
    feature = "no_std",
    doc = "
Std-only modules are unavailable in this mode:

```compile_fail
use mbrotli::io::EncoderWriter;
```

```compile_fail
use mbrotli::compressor::parallel::ParallelCompressor;
```

```compile_fail
use mbrotli::framing::FramedWriter;
```"
)]
#![deny(missing_docs)]
#![deny(missing_debug_implementations)]
#![deny(rustdoc::broken_intra_doc_links)]

#[cfg(any(feature = "compression", feature = "decompression"))]
#[cfg_attr(any(feature = "compression", test), macro_use)]
extern crate alloc;

#[cfg(test)]
extern crate std;

#[cfg(any(feature = "compression", feature = "decompression"))]
mod backend;
#[cfg(any(feature = "compression", feature = "decompression"))]
mod retention;
#[cfg(any(feature = "compression", feature = "decompression"))]
mod shared;
#[cfg(any(feature = "compression", feature = "decompression"))]
mod window;
#[cfg(any(feature = "compression", feature = "decompression"))]
pub use backend::Backend;
#[cfg(any(feature = "compression", feature = "decompression"))]
pub use retention::RetentionPolicy;
#[cfg(any(feature = "compression", feature = "decompression"))]
pub use window::{ConfigError, Window, WindowEncoding};

#[cfg(feature = "compression")]
pub mod compressor;
#[cfg(feature = "decompression")]
pub mod decompressor;
#[cfg(feature = "decompression")]
pub use decompressor::{
    DecodeFailure, DecodeOperation, DecodeProgress, DecoderSession, DecoderSessionOwned,
    DecoderStatus, Decompressor, DecompressorBuilder,
};

#[cfg(feature = "decompression")]
pub use decompressor::{
    DecodeConfigError, DecodeError, DecodeLimits, DecodeStreamConfig, DecoderConfig,
    InvalidDataKind, MemberMode, OutputSize, WindowLimit,
};

#[cfg(any(feature = "compression", feature = "decompression"))]
pub mod dictionary;
#[cfg(all(
    feature = "experimental",
    any(feature = "compression", feature = "decompression")
))]
pub mod framing;
#[cfg(all(feature = "experimental", feature = "decompression"))]
pub use decompressor::framing::{
    FramedDecodeConfig, FramedDecodeError, FramedDecodeFailure, FramedDecodeLimits,
    FramedDecodeStreamConfig, FramedDecoderSession, FramedDecoderSessionOwned, FramedDecompressor,
    FramedOutput,
};
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
pub use compressor::{
    BlockBits, BlockSize, CompressionMode, Compressor, CompressorBuilder, DistanceParams,
    EncodeError, EncoderConfig, EncoderSession, EncoderSessionOwned, EncoderStatus, InputSize,
    LiteralContextMode, Operation, Progress, Quality, SizeOverflow, StreamConfig,
};

#[cfg(all(feature = "experimental", feature = "compression"))]
pub use compressor::framing::{
    FramedCompressor, FramedCompressorBuilder, FramedEncodeConfig, FramedEncodeError,
    FramedEncodeFailure, FramedEncodeLocation, FramedEncodeOperation, FramedEncodeProgress,
    FramedEncodeStreamConfig, FramedEncoderSession, FramedEncoderSessionOwned, FramedEncoderStatus,
    FramedResourceSession,
};

//! Independently startable, byte-aligned, non-final parts (RFC 7932 §11.3).

use alloc::vec::Vec;

use super::driver::Encoder;
use super::fast::commands::store_meta_block_header;
use super::rfc9841::window::ResolvedWindow;
use crate::compressor::{Backend, BrotliCompressError, BrotliResult, EncoderConfig};
use crate::shared::bits::BitWriter;

/// A sealed proof: only the encoder can produce an aligned non-final part.
pub(crate) struct AlignedFragment<'a> {
    pub(crate) bytes: &'a [u8],
}

/// Retained, exclusively owned codec state, with a preselected fragment policy.
pub(crate) struct FragmentEncoder {
    /// An independent encoder: its fragments decode without the ones before.
    encoder: Option<Encoder<true>>,
    backend: Backend,
    config: EncoderConfig,
    bytes: Vec<u8>,
}

impl FragmentEncoder {
    pub(crate) const fn new(config: EncoderConfig, backend: Backend) -> Self {
        Self {
            encoder: None,
            backend,
            config,
            bytes: Vec::new(),
        }
    }

    /// Whether parts at `config`'s quality use history before them: only
    /// the greedy encoder can take it.
    pub(crate) fn takes_history(config: EncoderConfig) -> bool {
        (2..=9).contains(&config.quality().get())
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.bytes.capacity()
            + self
                .encoder
                .as_ref()
                .map_or(0, Encoder::<true>::retained_bytes)
    }

    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    /// Encodes `src` as one part. `history` is the input right before it,
    /// which the decoder holds by the time it reaches this part; the greedy
    /// qualities (2-9) let copies reach back into it, the others ignore it.
    pub(crate) fn encode(
        &mut self,
        history: &[u8],
        src: &[u8],
        first: bool,
    ) -> BrotliResult<AlignedFragment<'_>> {
        if src.is_empty() {
            return Err(BrotliCompressError::BufferOverflow);
        }
        let history = if Self::takes_history(self.config) {
            history
        } else {
            &[]
        };
        // The match finder is sized for every position it will store.
        let size_hint = history.len() + src.len();
        let params = self.config.lower(Some(size_hint));
        if !self
            .encoder
            .as_mut()
            .is_some_and(|e| e.reset_for(&params, size_hint))
        {
            self.encoder = Some(Encoder::new(self.backend, &params, size_hint)?);
        }
        let encoder = self
            .encoder
            .as_mut()
            .ok_or(BrotliCompressError::BufferOverflow)?;
        if let Encoder::Greedy(e) = encoder {
            e.prepend_history(history);
        }
        let prefix = &src[..src.len().min(2)];
        match encoder {
            Encoder::Fast(e) => e.begin_fragment(prefix)?,
            Encoder::Greedy(e) => e.begin_fragment(prefix)?,
            Encoder::Hq(e) => e.begin_fragment(prefix)?,
        }
        let mut header = [0u8; 32];
        let mut w = BitWriter::new(&mut header, 0);
        if first {
            let window = ResolvedWindow::new(&params);
            let window = if self.config.quality().get() <= 1 {
                window.at_least(18)
            } else {
                window
            };
            let (bits, count) = window.header();
            w.write(count, u64::from(bits));
        }
        store_meta_block_header(prefix.len(), true, &mut w);
        w.align();
        w.write_bytes(prefix);
        let len = w.position() / 8;
        self.bytes.clear();
        self.bytes.extend_from_slice(&header[..len]);
        let body = &src[prefix.len()..];
        let limit = encoder.block_size_limit();
        for (i, block) in body.chunks(limit).enumerate() {
            let last = (i + 1) * limit >= body.len();
            let bytes = if last {
                encoder.flush_block(block, None)?
            } else {
                encoder.encode_block_with(block, false, None)?
            };
            self.bytes.extend_from_slice(bytes);
        }
        let aligned = match encoder {
            Encoder::Fast(e) => e.fragment_aligned(),
            Encoder::Greedy(e) => e.fragment_aligned(),
            Encoder::Hq(e) => e.fragment_aligned(),
        };
        if !aligned {
            return Err(BrotliCompressError::BufferOverflow);
        }
        Ok(AlignedFragment { bytes: &self.bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::{Compressor, Quality};

    /// Decodes a complete stream with Google's decoder.
    fn c_decode(stream: &[u8], len: usize) -> Vec<u8> {
        let mut decoded = vec![0; len];
        let mut size = decoded.len();
        // SAFETY: both buffers are live, non-overlapping, and have the
        // lengths passed to C; the output length pointer is writable.
        let status = unsafe {
            google_brotli_ffi::BrotliDecoderDecompress(
                stream.len(),
                stream.as_ptr(),
                &mut size,
                decoded.as_mut_ptr(),
            )
        };
        assert_eq!(status, google_brotli_ffi::BROTLI_DECODER_RESULT_SUCCESS);
        decoded.truncate(size);
        decoded
    }

    /// `len` bytes that repeat every `period` bytes after a pseudo-random start.
    fn periodic(period: usize, len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed;
        let mut bytes: Vec<u8> = (0..period)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (state >> 24) as u8
            })
            .collect();
        for i in period..len {
            bytes.push(bytes[i - period]);
        }
        bytes
    }

    #[test]
    fn independent_parts_at_greedy_qualities_compress_like_a_serial_stream() {
        // Lines of consecutive numbers repeat at one distance after another,
        // which only the distance cache's short codes make cheap.
        let input: Vec<u8> = (1..=20_000)
            .flat_map(|n: u32| format!("{n}\n").into_bytes())
            .collect();
        for q in 2..=9 {
            for backend in Backend::available() {
                let config = EncoderConfig::default().with_quality(Quality::try_from(q).unwrap());
                let serial = Compressor::new(config).unwrap().compress(&input).unwrap();
                let mut encoder = FragmentEncoder::new(config, backend);
                let mut stream = encoder.encode(&[], &input, true).unwrap().bytes.to_vec();
                stream.push(3);
                assert_eq!(c_decode(&stream, input.len()), input, "q{q} {backend:?}");
                assert!(
                    stream.len() <= serial.len() * 21 / 20,
                    "q{q} {backend:?}: fragment {} bytes, serial {}",
                    stream.len(),
                    serial.len()
                );
            }
        }
    }

    #[test]
    fn independent_parts_never_inherit_the_decoder_distance_cache() {
        // The first part leaves distances 5, 6, 9 and 13 in the decoder's
        // cache; the second repeats at 4, 11, 15 and 16, the cache a fresh
        // stream starts with. A short code for those would decode against
        // the first part's distances instead.
        let first: Vec<u8> = [5, 6, 9, 13]
            .into_iter()
            .flat_map(|period| periodic(period, 2048, period as u32))
            .collect();
        let second: Vec<u8> = [4, 11, 15, 16]
            .into_iter()
            .flat_map(|period| periodic(period, 2048, 100 + period as u32))
            .collect();
        for q in 0..=11 {
            for backend in Backend::available() {
                let config = EncoderConfig::default().with_quality(Quality::try_from(q).unwrap());
                let mut encoder = FragmentEncoder::new(config, backend);
                let mut stream = encoder.encode(&[], &first, true).unwrap().bytes.to_vec();
                stream.extend_from_slice(encoder.encode(&[], &second, false).unwrap().bytes);
                stream.push(3);
                let mut input = first.clone();
                input.extend_from_slice(&second);
                assert_eq!(c_decode(&stream, input.len()), input, "q{q} {backend:?}");
            }
        }
    }

    #[test]
    fn independent_parts_round_trip_at_every_quality_and_backend() {
        for q in 0..=11 {
            for backend in Backend::available() {
                let config = EncoderConfig::default().with_quality(Quality::try_from(q).unwrap());
                let mut encoder = FragmentEncoder::new(config, backend);
                let mut input = Vec::new();
                let mut output = Vec::new();
                for (i, part) in [
                    b"a".to_vec(),
                    b"bc".to_vec(),
                    b"a repetitive dictionary compression testing abcabcabc ".repeat(300),
                    vec![0; 10000],
                ]
                .iter()
                .enumerate()
                {
                    input.extend_from_slice(part);
                    output.extend_from_slice(encoder.encode(&[], part, i == 0).unwrap().bytes);
                }
                output.push(3);
                let mut decoded = vec![0; input.len()];
                let mut size = decoded.len();
                // SAFETY: both buffers are live, non-overlapping, and have the
                // lengths passed to C; the output length pointer is writable.
                let status = unsafe {
                    google_brotli_ffi::BrotliDecoderDecompress(
                        output.len(),
                        output.as_ptr(),
                        &mut size,
                        decoded.as_mut_ptr(),
                    )
                };
                assert_eq!(
                    status,
                    google_brotli_ffi::BROTLI_DECODER_RESULT_SUCCESS,
                    "q{q}"
                );
                assert_eq!(size, input.len());
                assert_eq!(decoded, input, "q{q} {backend:?}");
                assert!(encoder.retained_bytes() > 0);
                assert!(encoder.encode(&[], &[], false).is_err());
            }
        }
    }

    /// A stream of `parts`, each encoded with the input before it as history.
    fn stream_with_history(encoder: &mut FragmentEncoder, parts: &[&[u8]]) -> (Vec<u8>, Vec<u8>) {
        let mut input = Vec::new();
        let mut stream = Vec::new();
        for (i, part) in parts.iter().enumerate() {
            stream.extend_from_slice(encoder.encode(&input, part, i == 0).unwrap().bytes);
            input.extend_from_slice(part);
        }
        stream.push(3);
        (stream, input)
    }

    #[test]
    fn independent_parts_copy_from_the_history_before_them() {
        let block = periodic(4096, 4096, 7);
        let repeated = block.repeat(4);
        for q in 0..=11 {
            for backend in Backend::available() {
                let config = EncoderConfig::default().with_quality(Quality::try_from(q).unwrap());
                let mut encoder = FragmentEncoder::new(config, backend);
                // One- and two-byte histories leave nothing whole to hash.
                for history in [&block[..1], &block[..2], &block[..]] {
                    let (stream, input) = stream_with_history(&mut encoder, &[history, &repeated]);
                    assert_eq!(c_decode(&stream, input.len()), input, "q{q} {backend:?}");
                }
                let first = encoder.encode(&[], &block, true).unwrap().bytes.len();
                let second = encoder
                    .encode(&block, &repeated, false)
                    .unwrap()
                    .bytes
                    .len();
                if FragmentEncoder::takes_history(config) {
                    // Copies of the history, not the block spelled out again.
                    assert!(second < first / 4, "q{q} {backend:?}: {second} of {first}");
                }
            }
        }
    }

    #[test]
    fn history_beyond_the_backward_limit_is_dropped() {
        // A 64 KiB window reaches 65,520 bytes back. The second part repeats
        // the start of a 100,000-byte history, which lies further away.
        let history = periodic(100_000, 100_000, 3);
        let mut part = history[..5000].to_vec();
        part.extend_from_slice(&history[95_000..]);
        for q in 2..=9 {
            for backend in Backend::available() {
                let config = EncoderConfig::default()
                    .with_quality(Quality::try_from(q).unwrap())
                    .with_window(crate::Window::standard(16).unwrap());
                let mut encoder = FragmentEncoder::new(config, backend);
                let (stream, input) = stream_with_history(&mut encoder, &[&history, &part]);
                assert_eq!(c_decode(&stream, input.len()), input, "q{q} {backend:?}");
            }
        }
    }
}

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
    encoder: Option<Encoder>,
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

    pub(crate) fn retained_bytes(&self) -> usize {
        self.bytes.capacity() + self.encoder.as_ref().map_or(0, Encoder::retained_bytes)
    }

    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(crate) fn encode(&mut self, src: &[u8], first: bool) -> BrotliResult<AlignedFragment<'_>> {
        if src.is_empty() {
            return Err(BrotliCompressError::BufferOverflow);
        }
        let params = self.config.lower(Some(src.len()));
        if !self
            .encoder
            .as_mut()
            .is_some_and(|e| e.reset_for(&params, src.len()))
        {
            let mut encoder = Encoder::new(self.backend.0, &params, src.len())?;
            match &mut encoder {
                Encoder::Fast(e) => e.select_fragment_kernels(self.backend.0),
                Encoder::Greedy(e) => e.select_fragment_kernels(self.backend.0),
                Encoder::Hq(e) => e.select_fragment_kernels(self.backend.0),
            }
            self.encoder = Some(encoder);
        }
        let encoder = self
            .encoder
            .as_mut()
            .ok_or(BrotliCompressError::BufferOverflow)?;
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
    use crate::compressor::Quality;

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
                    output.extend_from_slice(encoder.encode(part, i == 0).unwrap().bytes);
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
                assert!(encoder.encode(&[], false).is_err());
            }
        }
    }
}

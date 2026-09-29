//! Resolving a stream's window, from the parameters to the header bits.
//!
//! Ports `EncodeWindowBits` and the window half of `SanitizeParams` from
//! `c/enc/encode.c` and `c/enc/quality.h` of the pinned reference
//! (`google/brotli` v1.2.0, commit `028fb5a`), extended with the RFC 9841
//! large-window header.
//!
//! Every encoder family resolves its window through this one module, so the
//! header, the largest backward distance and the amount of history actually
//! kept are always derived from the same decision.

use crate::compressor::CompressParams;

/// Largest window the encoder keeps history for (`BROTLI_LARGE_MAX_WBITS`).
///
/// RFC 9841 allows a declared window of up to 62 bits, but no encoder needs to
/// remember that much to emit a valid stream: a shorter history only means
/// shorter distances, which every decoder for the declared window accepts. The
/// reference C encoder stops at the same 30 bits, which keeps the two
/// comparable wherever both implement the feature.
pub(crate) const MAX_ENCODER_WINDOW_BITS: usize = 30;

/// The window a stream was resolved to.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedWindow {
    /// Window size written to the stream header, in bits.
    declared_bits: usize,
    /// Whether the RFC 9841 large-window header is used.
    large: bool,
}

impl ResolvedWindow {
    /// Resolves the window `params` asks for.
    ///
    /// A large window is only ever selected by `WindowBits::large`; it is never
    /// inferred from the size, the quality, the input or the target. No range
    /// check is needed here: `WindowBits` can only be built by a constructor
    /// that already made one.
    pub(crate) const fn new(params: &CompressParams) -> Self {
        Self {
            declared_bits: params.lgwin.bits() as usize,
            large: params.lgwin.is_large(),
        }
    }

    /// Returns the same window with at least `bits` advertised.
    ///
    /// The fast qualities always claim eighteen bits even when they cut their
    /// input shorter, which is what the reference encoder does.
    pub(crate) const fn at_least(self, bits: usize) -> Self {
        if self.declared_bits >= bits {
            self
        } else {
            Self {
                declared_bits: bits,
                large: self.large,
            }
        }
    }

    /// Returns whether this stream uses the RFC 9841 large-window syntax.
    pub(crate) const fn is_large(self) -> bool {
        self.large
    }

    /// Returns the window the encoder actually keeps history for, in bits.
    ///
    /// This is the declared window for every ordinary stream, and for a large
    /// window it is capped at [`MAX_ENCODER_WINDOW_BITS`] so that a stream may
    /// declare a 62-bit window without the encoder sizing anything to it.
    pub(crate) const fn encoder_bits(self) -> usize {
        if self.declared_bits > MAX_ENCODER_WINDOW_BITS {
            MAX_ENCODER_WINDOW_BITS
        } else {
            self.declared_bits
        }
    }

    /// Encodes the stream header.
    ///
    /// Returns the bits and how many of them there are, in the reference's
    /// `last_bytes` / `last_bytes_bits` form. An ordinary window uses the
    /// RFC 7932 encoding of one, four or seven bits; a large window uses the
    /// fourteen-bit RFC 9841 form: the eight-bit marker `0b00010001` followed
    /// by six bits of window size.
    pub(crate) const fn header(self) -> (u16, u32) {
        let bits = self.declared_bits;
        if self.large {
            return (((bits as u16 & 0x3F) << 8) | 0x11, 14);
        }
        if bits == 16 {
            (0, 1)
        } else if bits == 17 {
            (1, 7)
        } else if bits > 17 {
            ((((bits - 17) << 1) | 0x01) as u16, 4)
        } else {
            ((((bits - 8) << 4) | 0x01) as u16, 7)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::core::fast::constants::WINDOW_BITS_FAST;
    use crate::compressor::{QualityLevel, WindowBits};

    fn resolve(lgwin: WindowBits) -> ResolvedWindow {
        ResolvedWindow::new(&CompressParams::new(QualityLevel::Q5, lgwin))
    }

    fn standard(bits: u8) -> ResolvedWindow {
        resolve(WindowBits::standard(bits).expect("a legal ordinary window"))
    }

    fn large(bits: u8) -> ResolvedWindow {
        resolve(WindowBits::large(bits).expect("a legal large window"))
    }

    #[test]
    fn an_ordinary_window_matches_the_reference_encoding() {
        assert_eq!(standard(16).header(), (0, 1));
        assert_eq!(standard(17).header(), (1, 7));
        assert_eq!(standard(18).header(), (3, 4));
        assert_eq!(standard(22).header(), (11, 4));
        assert_eq!(standard(24).header(), (15, 4));
        assert_eq!(standard(10).header(), (0x21, 7));
    }

    #[test]
    fn a_large_window_is_a_marker_and_six_bits() {
        for bits in 10u8..=62 {
            let (value, width) = large(bits).header();
            assert_eq!(width, 14, "{bits} bits");
            // Low eight bits are the marker, high six are the window.
            assert_eq!(value & 0xFF, 0x11, "{bits} bits");
            assert_eq!(value >> 8, u16::from(bits) & 0x3F, "{bits} bits");
        }
    }

    #[test]
    fn a_large_window_is_chosen_only_when_it_is_requested() {
        assert!(!standard(22).is_large());
        assert!(large(22).is_large());
        // The same numeric width still switches syntax when asked for.
        assert_ne!(standard(22).header(), large(22).header());
    }

    #[test]
    fn retained_history_stops_at_thirty_bits() {
        for bits in 10u8..=62 {
            let window = large(bits);
            let declared = usize::from(bits);
            assert_eq!(window.encoder_bits(), declared.min(MAX_ENCODER_WINDOW_BITS));
            // Never more history than the header promises, so every distance
            // the encoder can emit is one the decoder accepts.
            assert!(window.encoder_bits() <= declared);
        }
        for bits in 10u8..=24 {
            assert_eq!(standard(bits).encoder_bits(), usize::from(bits));
        }
    }

    #[test]
    fn raising_the_floor_keeps_the_syntax() {
        assert_eq!(standard(10).at_least(WINDOW_BITS_FAST).header(), (3, 4));
        assert_eq!(standard(22).at_least(WINDOW_BITS_FAST).header(), (11, 4));
        let raised = large(12).at_least(WINDOW_BITS_FAST);
        assert!(raised.is_large());
        assert_eq!(raised.header(), ((18u16 << 8) | 0x11, 14));
        assert_eq!(raised.encoder_bits(), 18);
    }
}

//! Opaque, host-validated execution backends; implementation tokens stay private.

use alloc::vec::Vec;

use fearless_simd::Level;

/// A supported execution backend: the SIMD instruction level the codecs'
/// kernels run with.
///
/// The default is the best level the host supports, detected once when a
/// codec is built and never inside a compression or decompression loop.
/// Every backend produces identical output; enumerate [`Self::available`]
/// for reproducible measurements and differential tests. Unsupported
/// backends cannot be constructed through this API.
///
/// # Examples
///
/// ```
/// use mzstd::Backend;
/// # #[cfg(all(feature = "compression", feature = "decompression"))]
/// # {
/// use mzstd::{Compressor, Decompressor, EncoderConfig};
///
/// let frame = Compressor::default().compress(b"payload payload payload")?;
/// for backend in Backend::available() {
///     let mut encoder = Compressor::builder(EncoderConfig::default())
///         .with_backend(backend)
///         .build()?;
///     assert_eq!(encoder.compress(b"payload payload payload")?, frame);
///     let mut decoder = Decompressor::builder(Default::default())
///         .with_backend(backend)
///         .build();
///     assert_eq!(decoder.decompress(&frame)?, b"payload payload payload");
/// }
/// # }
/// assert!(Backend::available().contains(&Backend::default()));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Copy, Clone)]
pub struct Backend(pub(crate) Level);

impl Backend {
    /// Portable scalar implementation, without explicit SIMD kernels.
    #[cfg(test)]
    pub(crate) const SCALAR: Self = Self(Level::fallback());

    /// Returns every distinct supported backend, from lower to higher SIMD
    /// levels.
    ///
    /// Scalar fallback is included only when the host requires it; the
    /// crate's own unit tests additionally include it everywhere.
    /// Detection happens here, never inside a codec loop. With `no_std`, only
    /// backends enabled by compile-time target features are available.
    #[must_use]
    pub fn available() -> Vec<Self> {
        let detected = Self::default().0;
        let mut backends = Vec::new();
        #[cfg(test)]
        backends.push(Self::SCALAR);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if let Some(token) = detected.as_sse2() {
                backends.push(Self(Level::Sse2(token)));
            }
            if let Some(token) = detected.as_sse4_2() {
                backends.push(Self(Level::Sse4_2(token)));
            }
            if let Some(token) = detected.as_avx2() {
                backends.push(Self(Level::Avx2(token)));
            }
            if let Some(token) = detected.as_avx512() {
                backends.push(Self(Level::Avx512(token)));
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(token) = detected.as_neon() {
            backends.push(Self(Level::Neon(token)));
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        if let Some(token) = detected.as_wasm_simd128() {
            backends.push(Self(Level::WasmSimd128(token)));
        }
        if !backends.contains(&Self(detected)) {
            backends.push(Self(detected));
        }
        backends
    }

    /// Stable diagnostic name, available without allocating or formatting.
    ///
    /// # Examples
    ///
    /// ```
    /// let name = mzstd::Backend::default().name();
    /// assert!(!name.is_empty());
    /// ```
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self.0 {
            #[cfg(any(
                test,
                not(any(
                    all(target_arch = "aarch64", target_feature = "neon"),
                    all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        target_feature = "sse2",
                        target_feature = "fxsr"
                    ),
                    all(target_arch = "wasm32", target_feature = "simd128")
                ))
            ))]
            Level::Fallback(_) => "fallback",
            #[cfg(target_arch = "aarch64")]
            Level::Neon(_) => "neon",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Sse2(_) => "sse2",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Sse4_2(_) => "sse4.2",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Avx2(_) => "avx2",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Avx512(_) => "avx512",
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Level::WasmSimd128(_) => "wasm-simd128",
            _ => "native",
        }
    }
}

impl Backend {
    /// Whether the level has a native byte shuffle (SSSE3 and up, NEON,
    /// WASM SIMD); without one, shuffles are emulated lane by lane.
    #[cfg(feature = "decompression")]
    pub(crate) fn has_fast_shuffle(self) -> bool {
        !matches!(self.name(), "fallback" | "sse2")
    }

    /// The level the block decoder's loops compile for: AVX-512 runs them
    /// with AVX2's features. The loops are scalar apart from 16- and
    /// 32-byte copies, so AVX-512 adds nothing to them, while its cheap
    /// gathers lead the compiler to vectorize the four Huffman streams'
    /// independent lookups across lanes, which lengthens every stream's
    /// dependency chain.
    #[cfg(feature = "decompression")]
    pub(crate) fn block_decode_level(self) -> Level {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Level::Avx512(_) = self.0
            && let Some(avx2) = self.0.as_avx2()
        {
            return Level::Avx2(avx2);
        }
        self.0
    }
}

impl Default for Backend {
    /// The best level the host supports; with `no_std`, the compile-time
    /// baseline.
    fn default() -> Self {
        #[cfg(not(feature = "no_std"))]
        let level = Level::try_detect().unwrap_or_else(Level::baseline);
        #[cfg(feature = "no_std")]
        let level = Level::baseline();
        Self(level)
    }
}

impl ::core::fmt::Debug for Backend {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        f.write_str(self.name())
    }
}

impl ::core::fmt::Display for Backend {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        f.write_str(self.name())
    }
}

impl PartialEq for Backend {
    fn eq(&self, other: &Self) -> bool {
        ::core::mem::discriminant(&self.0) == ::core::mem::discriminant(&other.0)
    }
}

impl Eq for Backend {}

#[cfg(test)]
mod tests {
    use super::Backend;

    #[test]
    fn backend_formats_as_its_stable_name() {
        for backend in Backend::available() {
            assert_eq!(alloc::format!("{backend}"), backend.name());
            assert_eq!(alloc::format!("{backend:?}"), backend.name());
        }
    }

    #[cfg(feature = "no_std")]
    #[test]
    fn no_std_selects_the_compile_time_baseline() {
        assert_eq!(
            Backend::default(),
            Backend(fearless_simd::Level::baseline())
        );
    }

    #[test]
    fn scalar_is_included_once_in_the_internal_backend_matrix() {
        let backends = Backend::available();
        assert_eq!(backends[0], Backend::SCALAR);
        assert_eq!(Backend::SCALAR.name(), "fallback");
        assert!(backends.contains(&Backend::default()));
        for (index, backend) in backends.iter().enumerate() {
            assert!(!backends[..index].contains(backend));
        }
    }
}

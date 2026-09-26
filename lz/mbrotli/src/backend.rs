//! Opaque, host-validated execution backends; implementation tokens stay private.

use alloc::vec::Vec;

use fearless_simd::Level;

/// A supported execution backend, independent of the SIMD implementation crate.
///
/// Use the default for normal compression, or enumerate [`Self::available`] for
/// reproducible measurements and differential tests. Unsupported backends cannot
/// be constructed through this API.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "compression")]
/// # {
/// use mbrotli::{Backend, Compressor, EncoderConfig};
/// let mut compressor = Compressor::builder(EncoderConfig::default())
///     .with_backend(Backend::default()).build()?;
/// assert!(!compressor.compress(b"payload")?.is_empty());
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Copy, Clone)]
pub struct Backend(pub(crate) Level);

impl Backend {
    /// Portable scalar implementation, without explicit SIMD kernels.
    #[cfg(test)]
    pub(crate) const SCALAR: Self = Self(Level::fallback());

    /// Returns every distinct supported backend, from lower to higher SIMD levels.
    ///
    /// Scalar fallback is included only when the host requires it. Internal unit
    /// tests additionally include the independent scalar implementation.
    ///
    /// Detection occurs here, never inside a compression loop. With `no_std`,
    /// only backends supported by compile-time target features are available.
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

impl Default for Backend {
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
        core::mem::discriminant(&self.0) == core::mem::discriminant(&other.0)
    }
}
impl Eq for Backend {}

#[cfg(test)]
mod tests {
    use super::Backend;
    #[cfg(feature = "compression")]
    use crate::{Compressor, EncoderConfig, Quality, Window};
    #[cfg(feature = "compression")]
    use alloc::vec::Vec;

    #[test]
    fn backend_formats_as_its_stable_name() {
        for backend in Backend::available() {
            assert_eq!(format!("{backend}"), backend.name());
            assert_eq!(format!("{backend:?}"), backend.name());
        }
    }

    #[cfg(feature = "no_std")]
    #[test]
    fn no_std_selects_the_compile_time_baseline_even_with_std_dependencies() {
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

    #[cfg(feature = "compression")]
    #[test]
    fn every_host_backend_matches_scalar_across_qualities_windows_and_boundaries() {
        let backends = Backend::available();
        let payload: Vec<u8> = (0..4096).map(|index| (index % 251) as u8).collect();
        for quality in 0..=11 {
            for bits in [10, 22] {
                let config = EncoderConfig::default()
                    .with_quality(Quality::try_from(quality).expect("quality"))
                    .with_window(Window::standard(bits).expect("window"));
                let mut scalar = Compressor::builder(config)
                    .with_backend(Backend::SCALAR)
                    .build()
                    .expect("scalar encoder");
                for backend in &backends {
                    let mut encoder = Compressor::builder(config)
                        .with_backend(*backend)
                        .build()
                        .expect("host encoder");
                    for len in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 4096] {
                        let input = &payload[..len];
                        assert_eq!(
                            encoder.compress(input).expect("host compression"),
                            scalar.compress(input).expect("scalar compression"),
                            "{backend}, quality {quality}, window {bits}, length {len}"
                        );
                    }
                }
            }
        }
    }
}

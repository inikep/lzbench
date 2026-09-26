//! Private implementation of the compressor.
//!
//! Nothing in this tree is part of the public API: the modules here own the
//! algorithms, the bitstream layout and the SIMD dispatch, while
//! [`crate::compressor`] owns the ergonomic surface built on top of them.

pub(crate) mod bound;
pub(crate) mod dispatch;
pub(crate) mod driver;
pub(crate) mod fast;
#[cfg(not(feature = "no_std"))]
pub(crate) mod fragment;
pub(crate) mod greedy;
pub(crate) mod hq;
pub(crate) mod rfc9841;
pub(crate) mod session;
mod stream;

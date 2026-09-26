//! The codecs export their C functions with `#[no_mangle]`; referencing each
//! crate here is enough for them to end up in liblzbench_rust.

#[cfg(feature = "density")]
pub use density_rs;
#[cfg(feature = "mbrotli")]
pub use mbrotli_ffi;

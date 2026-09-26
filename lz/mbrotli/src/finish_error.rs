use std::io::Error;

/// A finalisation that failed, with the adapter that can retry it.
///
/// `EncoderWriter::finish` or `DecoderWriter::finish` consumes the adapter, which would strand the
/// stream if a recoverable sink failure destroyed it. Instead the adapter comes
/// back here: retry with the adapter's `try_finish`, or drop it to abandon the
/// stream. [`Self::into_error`] keeps only the error and drops the adapter.
/// Decoder format, resource-limit, and truncated-input failures are terminal;
/// retaining the adapter only makes sink delivery failures retryable.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "compression")]
/// # {
/// use mbrotli::{Compressor, EncoderConfig, Quality};
/// use std::io::{ErrorKind, Write};
///
/// /// A sink that refuses the first write and accepts everything after.
/// struct Stubborn { written: Vec<u8>, refused: bool }
///
/// impl Write for Stubborn {
///     fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
///         if !self.refused {
///             self.refused = true;
///             return Err(std::io::Error::new(ErrorKind::WouldBlock, "not yet"));
///         }
///         self.written.extend_from_slice(buf);
///         Ok(buf.len())
///     }
///     fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
/// }
///
/// let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q1))?;
/// let mut sink = encoder.writer(Stubborn { written: Vec::new(), refused: false }, Default::default())?;
/// sink.write_all(b"payload payload")?;
///
/// // The first finish fails, and hands the adapter back.
/// let mut sink = match sink.finish() {
///     Ok(_) => unreachable!("the sink refuses its first write"),
///     Err(failure) => {
///         assert_eq!(failure.error().kind(), ErrorKind::WouldBlock);
///         failure.into_inner()
///     }
/// };
///
/// // The retry completes the very same stream.
/// let inner = sink.finish().map_err(mbrotli::io::FinishError::into_error)?;
/// assert!(!inner.written.is_empty());
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct FinishError<T> {
    /// What went wrong.
    error: Error,
    /// The adapter, so the caller can try again.
    writer: T,
}

impl<T> FinishError<T> {
    /// Preserves an adapter after failed finalization, for either codec direction.
    pub(crate) const fn from_parts(error: Error, writer: T) -> Self {
        Self { error, writer }
    }

    /// Returns the failure that stopped the stream from being terminated.
    #[must_use]
    pub const fn error(&self) -> &Error {
        &self.error
    }

    /// Takes the failure, dropping the adapter and abandoning the stream.
    #[must_use]
    pub fn into_error(self) -> Error {
        self.error
    }

    /// Takes the adapter back, so the finalisation can be retried.
    #[must_use]
    pub fn into_inner(self) -> T {
        self.writer
    }

    /// Splits the failure from the adapter.
    #[must_use]
    pub fn into_parts(self) -> (Error, T) {
        (self.error, self.writer)
    }
}

impl<T> std::fmt::Debug for FinishError<T> {
    /// Reports the failure; the adapter has no `Debug` bound to print with.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FinishError")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl<T> std::fmt::Display for FinishError<T> {
    /// Reports the failure that stopped the stream from being terminated.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the compressed stream could not be finished: {}",
            self.error
        )
    }
}

impl<T> std::error::Error for FinishError<T> {
    /// Returns the failure this wraps.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl<T> From<FinishError<T>> for Error {
    /// Takes the failure, dropping the adapter.
    fn from(value: FinishError<T>) -> Self {
        value.error
    }
}

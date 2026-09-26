use super::{core::driver::Driver, *};
use std::io::{self, Read};

/// Lazy encoded-byte reader over borrowed, complete resource slices.
#[derive(Debug)]
pub struct FramedEncoderReader<'c, 'input> {
    driver: Driver<'c, 'input>,
    deferred: Option<FramedEncodeError>,
    failed: bool,
}
impl FramedCompressor {
    /// Starts a lazy reader without encoding the input or performing I/O.
    /// # Errors
    /// Rejects forgotten sessions, unrepresentable aggregate lengths, or header allocation.
    /// # Examples
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::Read;
    /// let mut encoder = FramedCompressor::new(Default::default())?;
    /// let items = [FramedItem::Resource(FramedResource::from(&b"hello"[..]))];
    /// let mut bytes = Vec::new();
    /// encoder.framed_reader(items.as_slice().into(), Default::default())?
    ///     .read_to_end(&mut bytes)?;
    /// assert_eq!(bytes, encoder.compress(items.as_slice().into())?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn framed_reader<'c, 'input>(
        &'c mut self,
        input: FramedInput<'input>,
        stream: FramedEncodeStreamConfig,
    ) -> Result<FramedEncoderReader<'c, 'input>, FramedEncodeError> {
        Driver::aggregate(input)?;
        Ok(FramedEncoderReader {
            driver: Driver::new(self.start(stream)?, input),
            deferred: None,
            failed: false,
        })
    }
}
impl<'input> FramedEncoderReader<'_, 'input> {
    /// Cancels and returns the original description, without encoding or I/O.
    pub fn into_inner(self) -> FramedInput<'input> {
        self.driver.input
    }
}
impl Read for FramedEncoderReader<'_, '_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        if let Some(error) = self.deferred.take() {
            return Err(error.into());
        }
        if self.failed {
            return Err(FramedEncodeError::InvalidState.into());
        }
        match self.driver.process(output) {
            Ok(p) => Ok(p.produced),
            Err(e) => {
                self.failed = true;
                if e.produced != 0 {
                    self.deferred = Some(e.error);
                    Ok(e.produced)
                } else {
                    Err(e.error.into())
                }
            }
        }
    }
}

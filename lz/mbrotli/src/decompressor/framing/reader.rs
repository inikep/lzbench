use super::{core::Tag, *};
use crate::DecodeOperation;
use std::io::{self, BufRead};

/// Source failure or terminal decoder failure, preserving the source chain.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FramedReadError {
    /// Source error; WouldBlock does not end the decoder session.
    #[error("framed source failed: {0}")]
    Io(#[from] io::Error),
    /// Terminal codec failure with exact per-call progress.
    #[error("{0}")]
    Decode(#[from] FramedDecodeFailure),
}
/// Lending event reader. The buffered source retains all unaccepted suffix bytes.
#[derive(Debug)]
pub struct FramedReader<'d, 'dict, R> {
    session: FramedDecoderSession<'d, 'dict>,
    source: R,
    output: [u8; 8192],
    partial: Option<ProducedFragment>,
    eof: bool,
}
impl FramedDecompressor {
    /// Creates an event reader; wrap an ordinary `Read` in `BufReader` first.
    ///
    /// # Examples
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::BufRead;
    /// let mut decoder = FramedDecompressor::new(
    ///     FramedDecodeConfig::default().with_input_mode(InputMode::Auto))?;
    /// let input = [0x3b, b'T']; // An empty raw member and a protocol suffix.
    /// let mut reader = decoder.framed_reader(&input[..], Default::default())?;
    /// while reader.next_event()?.is_some() {}
    /// assert!(reader.is_finished());
    /// assert_eq!(reader.into_inner().fill_buf()?, b"T");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    /// # Errors
    /// Returns session-start policy and lifecycle errors.
    pub fn framed_reader<R: BufRead>(
        &mut self,
        reader: R,
        stream: FramedDecodeStreamConfig,
    ) -> Result<FramedReader<'_, 'static, R>, FramedDecodeError> {
        Ok(FramedReader {
            session: self.start(stream)?,
            source: reader,
            output: [0; 8192],
            partial: None,
            eof: false,
        })
    }
    /// Creates an event reader with a session-borrowed dictionary resolver.
    /// # Errors
    /// Returns session-start policy and lifecycle errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// struct Dictionaries;
    /// impl DictionaryResolver for Dictionaries {
    ///     fn resolve(&self, request: ExternalDictionaryRequest) -> Option<&[u8]> {
    ///         (request.id == DictionaryId([7; 32])
    ///             && request.kind == ExternalDictionaryKind::Prefix).then_some(b"prefix")
    ///     }
    /// }
    /// let dictionaries = Dictionaries;
    /// // Footerless full resource, Shared Brotli codec, one external prefix reference.
    /// let mut input = vec![0x91, 10, 66, 82, 0, 40, 2, 3, 0, 1, 2, 3];
    /// input.extend([7; 32]);
    /// input.extend([0, 0x3b]);
    /// let mut reader = decoder.framed_reader_with_dictionaries(
    ///     &dictionaries, input.as_slice(), Default::default())?;
    /// while reader.next_event()?.is_some() {}
    /// assert!(reader.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn framed_reader_with_dictionaries<'d, 'dict, R: BufRead>(
        &'d mut self,
        dictionaries: impl Into<DictionaryResolverRef<'dict>>,
        reader: R,
        stream: FramedDecodeStreamConfig,
    ) -> Result<FramedReader<'d, 'dict, R>, FramedDecodeError> {
        Ok(FramedReader {
            session: self.start_with_dictionaries(dictionaries, stream)?,
            source: reader,
            output: [0; 8192],
            partial: None,
            eof: false,
        })
    }
}
impl<R: BufRead> FramedReader<'_, '_, R> {
    /// Returns one borrowed event, or `None` only after complete validation.
    /// # Errors
    /// Returns source or codec errors. A failing call's payload remains available
    /// through `partial_output` until the next mutating call.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let input = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// let mut reader = decoder.framed_reader(&input[..], Default::default())?;
    /// let mut payload = Vec::new();
    /// while let Some(event) = reader.next_event()? {
    ///     if let FramedEvent::ResourceData(fragment) = event {
    ///         payload.extend_from_slice(fragment.bytes);
    ///     }
    /// }
    /// assert_eq!(payload, b"abc");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    // The by-value failure contract preserves progress even when allocation fails;
    // boxing this 128-byte record would require an allocation on the error path.
    #[expect(
        clippy::result_large_err,
        reason = "allocation-independent progress is part of the public contract"
    )]
    pub fn next_event(&mut self) -> Result<Option<FramedEvent<'_>>, FramedReadError> {
        self.partial = None;
        loop {
            if self.session.is_finished() {
                return Ok(None);
            }
            let operation = if self.eof {
                DecodeOperation::Finish
            } else {
                DecodeOperation::Process
            };
            // Pending semantic events and buffered codec output never require a
            // new transport read. In particular, a raw member can finish while
            // the enclosing connection remains open.
            let mut result = self.session.step(&[], &mut self.output, operation);
            if matches!(result, Ok((_, _, Tag::Input))) {
                let input = loop {
                    match self.source.fill_buf() {
                        Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                        result => break result?,
                    }
                };
                if input.is_empty() {
                    self.eof = true;
                }
                let operation = if self.eof {
                    DecodeOperation::Finish
                } else {
                    DecodeOperation::Process
                };
                result = self.session.step(input, &mut self.output, operation);
            }
            let (consumed, produced, tag) = match result {
                Ok(progress) => progress,
                Err(failure) => {
                    self.source.consume(failure.consumed);
                    self.partial = failure.last_output.clone();
                    return Err(failure.into());
                }
            };
            self.source.consume(consumed);
            match tag {
                Tag::Input | Tag::Output => continue,
                Tag::Finished => return Ok(None),
                _ => {
                    return self
                        .session
                        .owner
                        .engine
                        .event(tag, &self.output[..produced])
                        .ok_or(FramedReadError::Decode(FramedDecodeFailure {
                            error: FramedDecodeError::InvalidState,
                            consumed,
                            produced,
                            last_output: None,
                        }))
                        .map(Some);
                }
            }
        }
    }
    /// Payload produced by the last failing call, with resource-local identity.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let input = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// let mut reader = decoder.framed_reader(&input[..], Default::default())?;
    /// assert!(reader.partial_output().is_none());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn partial_output(&self) -> Option<ResourceData<'_>> {
        self.partial.as_ref().map(|f| ResourceData {
            position: f.position,
            bytes: &self.output[f.range.clone()],
        })
    }
    /// Borrows the buffered source.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let input = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// let mut reader = decoder.framed_reader(&input[..], Default::default())?;
    /// assert_eq!(*reader.get_ref(), input.as_slice());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn get_ref(&self) -> &R {
        &self.source
    }
    /// Borrows the source to repair an I/O failure. Do not consume or replace bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let input = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// let mut reader = decoder.framed_reader(std::io::Cursor::new(input), Default::default())?;
    /// assert_eq!(reader.get_mut().position(), 0); // Inspect without consuming source bytes.
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.source
    }
    /// Returns the buffered source and its unaccepted suffix without performing I/O.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let input = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// let mut reader = decoder.framed_reader(&input[..], Default::default())?;
    /// while reader.next_event()?.is_some() {}
    /// assert!(reader.into_inner().is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn into_inner(self) -> R {
        self.source
    }
    /// Whether whole-object validation succeeded.
    ///
    /// # Examples
    ///
    /// ```
    /// use mbrotli::framing::*;
    /// let mut decoder = FramedDecompressor::new(Default::default())?;
    /// let input = [0x91, 10, 66, 82, 0, 6, 2, 0, 0, b'a', b'b', b'c'];
    /// let mut reader = decoder.framed_reader(&input[..], Default::default())?;
    /// assert!(!reader.is_finished());
    /// while reader.next_event()?.is_some() {}
    /// assert!(reader.is_finished());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn is_finished(&self) -> bool {
        self.session.is_finished()
    }
}

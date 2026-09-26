use super::*;
use crate::dictionary::PreparedDictionary;
use crate::{Operation, StreamConfig};
use alloc::boxed::Box;
use std::io::{self, Write};

/// Container writer owning its sink and borrowing one reusable framed encoder.
#[derive(Debug)]
pub struct FramedWriter<'c, W> {
    session: FramedEncoderSession<'c>,
    transport: Transport<W>,
}
/// Fixed inline transport storage; no framing allocation is moved outside its budget.
#[derive(Debug)]
struct Transport<W> {
    sink: W,
    bytes: [u8; 8192],
    cursor: usize,
    length: usize,
    deferred: Option<FramedEncodeError>,
}
impl<W: Write> Transport<W> {
    fn drain(&mut self) -> Result<(), FramedEncodeError> {
        while self.cursor < self.length {
            match self.sink.write(&self.bytes[self.cursor..self.length]) {
                Ok(0) => return Err(io::Error::from(io::ErrorKind::WriteZero).into()),
                Ok(n) if n <= self.length - self.cursor => self.cursor += n,
                Ok(_) => return Err(io::Error::other("sink reported an oversized write").into()),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        self.cursor = 0;
        self.length = 0;
        if let Some(e) = self.deferred.take() {
            return Err(e);
        }
        Ok(())
    }
}
impl FramedCompressor {
    /// Starts a container writer without I/O, using this owner's framing policy.
    /// # Errors
    /// Rejects forgotten sessions or header allocation failure.
    /// # Examples
    /// ```
    /// use mbrotli::framing::*;
    /// use std::io::Write;
    /// let mut owner = FramedCompressor::new(Default::default())?;
    /// let mut writer = owner.framed_writer(Vec::new(), Default::default())?;
    /// let mut resource = writer.resource(Default::default(), Default::default())?;
    /// resource.write_all(b"streaming payload")?;
    /// resource.try_finish()?;
    /// drop(resource);
    /// let bytes = writer.finish().map_err(|e| e.error)?;
    /// assert_eq!(&bytes[..4], &[0x91, 10, 66, 82]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn framed_writer<W: Write>(
        &mut self,
        writer: W,
        stream: FramedEncodeStreamConfig,
    ) -> Result<FramedWriter<'_, W>, FramedEncodeError> {
        Ok(FramedWriter {
            session: self.start(stream)?,
            transport: Transport {
                sink: writer,
                bytes: [0; 8192],
                cursor: 0,
                length: 0,
                deferred: None,
            },
        })
    }
}
impl<W: Write> FramedWriter<'_, W> {
    fn drain(&mut self, operation: FramedEncodeOperation) -> Result<(), FramedEncodeError> {
        loop {
            self.transport.drain()?;
            let p = match self.session.process(&mut self.transport.bytes, operation) {
                Ok(p) => p,
                Err(e) => {
                    self.transport.length = e.produced;
                    self.transport.deferred = Some(e.error);
                    self.transport.drain()?;
                    return Err(FramedEncodeError::InvalidState);
                }
            };
            self.transport.length = p.produced;
            self.transport.drain()?;
            if p.status != FramedEncoderStatus::NeedsOutput {
                return Ok(());
            }
        }
    }
    /// Starts a compressed payload resource after draining earlier output.
    /// # Errors
    /// Reports transport failure, invalid settings/order or exhausted budgets.
    pub fn resource(
        &mut self,
        options: ResourceOptions,
        stream: StreamConfig,
    ) -> Result<ResourceWriter<'_, 'static, W>, FramedEncodeError> {
        self.drain(FramedEncodeOperation::Process)?;
        Ok(ResourceWriter {
            session: self.session.resource(options, stream)?,
            transport: &mut self.transport,
        })
    }
    /// Starts a resource with a borrowed dictionary and explicit wire references.
    /// # Errors
    /// As `resource`, plus invalid references or unsupported dictionary settings.
    pub fn resource_with_dictionary<'s, 'dict>(
        &'s mut self,
        options: ResourceOptions,
        stream: StreamConfig,
        dictionary: &'dict PreparedDictionary,
        references: &[DictionaryReference],
    ) -> Result<ResourceWriter<'s, 'dict, W>, FramedEncodeError> {
        self.drain(FramedEncodeOperation::Process)?;
        Ok(ResourceWriter {
            session: self
                .session
                .resource_with_dictionary(options, stream, dictionary, references)?,
            transport: &mut self.transport,
        })
    }
    /// Starts a verbatim payload resource after draining earlier output.
    /// # Errors
    /// Reports transport failure, invalid ordering or exhausted budgets.
    pub fn uncompressed_resource(
        &mut self,
        options: ResourceOptions,
    ) -> Result<ResourceWriter<'_, 'static, W>, FramedEncodeError> {
        self.drain(FramedEncodeOperation::Process)?;
        Ok(ResourceWriter {
            session: self.session.uncompressed_resource(options)?,
            transport: &mut self.transport,
        })
    }
    /// Queues uncompressed metadata after delivering earlier output.
    /// # Errors
    /// Reports transport, ordering, field validation or budget errors.
    pub fn metadata(
        &mut self,
        kind: MetadataKind,
        fields: &[MetadataField<'_>],
    ) -> Result<(), FramedEncodeError> {
        self.metadata_with_options(kind, fields, Default::default())
    }
    /// Queues metadata with independent original and repeated encodings.
    /// # Errors
    /// Invalid commands remain uncommitted; earlier pending output can be retried.
    pub fn metadata_with_options(
        &mut self,
        kind: MetadataKind,
        fields: &[MetadataField<'_>],
        options: MetadataOptions<'_>,
    ) -> Result<(), FramedEncodeError> {
        self.drain(FramedEncodeOperation::Process)?;
        self.session.metadata_with_options(kind, fields, options)
    }
    /// Selects repeated fields before the first metadata command.
    /// # Errors
    /// Reports invalid codes/order, disabled repetition, transport or budget errors.
    pub fn repeat_metadata_fields(&mut self, codes: &[[u8; 2]]) -> Result<(), FramedEncodeError> {
        self.drain(FramedEncodeOperation::Process)?;
        self.session.repeat_metadata_fields(codes)
    }
    /// Queues a padding chunk after delivering previous output.
    /// # Errors
    /// Reports ordering, budget or transport errors.
    pub fn padding(&mut self, bytes: usize) -> Result<(), FramedEncodeError> {
        self.drain(FramedEncodeOperation::Process)?;
        self.session.padding(bytes)
    }
    /// Delivers pending container bytes and flushes the sink.
    /// # Errors
    /// Retains the cursor on transport errors so callers can repair and retry.
    pub fn flush(&mut self) -> Result<(), FramedEncodeError> {
        self.drain(FramedEncodeOperation::Process)?;
        self.transport.sink.flush()?;
        Ok(())
    }
    /// Generates the suffix once, delivers it, and flushes the sink.
    /// # Errors
    /// Transport failures preserve all progress for retry; codec errors are terminal.
    pub fn try_finish(&mut self) -> Result<(), FramedEncodeError> {
        self.drain(FramedEncodeOperation::Finish)?;
        self.transport.sink.flush()?;
        Ok(())
    }
    /// Finalizes and returns the sink, retaining the whole writer on failure.
    /// # Errors
    /// As `try_finish`; recover the writer from the owning error to retry.
    pub fn finish(mut self) -> Result<W, Box<FramingFinishError<Self>>> {
        match self.try_finish() {
            Ok(()) => Ok(self.transport.sink),
            Err(error) => Err(Box::new(FramingFinishError {
                writer: self,
                error,
            })),
        }
    }
    /// Borrows the sink for inspection.
    pub const fn get_ref(&self) -> &W {
        &self.transport.sink
    }
    /// Repairs the sink. Inserting/removing bytes through this borrow invalidates offsets.
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.transport.sink
    }
    /// Container-relative queued offset for subsequent internal references.
    pub const fn next_chunk_offset(&self) -> u64 {
        self.session.next_chunk_offset()
    }
    /// Cancels without I/O and returns the sink; this does not finalize output.
    pub fn into_inner(self) -> W {
        self.transport.sink
    }
}
/// Borrowing Write adapter for one resource. Drop never writes or finalizes.
#[derive(Debug)]
pub struct ResourceWriter<'s, 'dict, W> {
    session: FramedResourceSession<'s, 'dict>,
    transport: &'s mut Transport<W>,
}
impl<W: Write> ResourceWriter<'_, '_, W> {
    fn complete(&mut self, operation: Operation) -> Result<(), FramedEncodeError> {
        loop {
            self.transport.drain()?;
            let p = match self
                .session
                .process(&[], &mut self.transport.bytes, operation)
            {
                Ok(p) => p,
                Err(e) => {
                    self.transport.length = e.produced;
                    self.transport.deferred = Some(e.error);
                    self.transport.drain()?;
                    return Err(FramedEncodeError::InvalidState);
                }
            };
            self.transport.length = p.produced;
            self.transport.drain()?;
            if p.status != FramedEncoderStatus::NeedsOutput {
                return Ok(());
            }
        }
    }
    /// Completes and delivers the resource, retrying transport without re-encoding.
    /// # Errors
    /// Codec failures are terminal; sink errors preserve the unwritten suffix.
    pub fn try_finish(&mut self) -> Result<(), FramedEncodeError> {
        self.complete(Operation::Finish)
    }
    /// Repairs the sink; writing container bytes through this borrow is forbidden.
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.transport.sink
    }
}
impl<W: Write> Write for ResourceWriter<'_, '_, W> {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        if input.is_empty() {
            return Ok(0);
        }
        self.transport.drain().map_err(io::Error::from)?;
        if self.session.is_finished() {
            return Err(FramedEncodeError::InvalidState.into());
        }
        loop {
            match self
                .session
                .process(input, &mut self.transport.bytes, Operation::Process)
            {
                Ok(p) => {
                    self.transport.length = p.produced;
                    if p.consumed != 0 {
                        return Ok(p.consumed);
                    }
                    self.transport.drain().map_err(io::Error::from)?;
                }
                Err(e) => {
                    self.transport.length = e.produced;
                    if e.consumed != 0 {
                        self.transport.deferred = Some(e.error);
                        return Ok(e.consumed);
                    }
                    return Err(e.error.into());
                }
            }
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        self.complete(Operation::Flush).map_err(io::Error::from)?;
        self.transport.sink.flush()
    }
}

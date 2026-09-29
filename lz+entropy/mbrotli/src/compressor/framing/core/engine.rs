//! Owner-resident lifecycle, command admission and precise progress.
use super::{Container, resource::Resource};
use crate::compressor::framing::*;
use crate::dictionary::PreparedDictionary;
use crate::{Compressor, InputSize, Operation, StreamConfig, WindowEncoding};
use FramedEncodeError as E;

#[derive(Debug)]
pub(in crate::compressor::framing) struct Engine {
    pub(super) container: Container,
    pub(super) resource: Resource,
    stream: FramedEncodeStreamConfig,
    pub(in crate::compressor::framing) total_in: u64,
    pub(in crate::compressor::framing) total_out: u64,
    failed: bool,
    finishing: bool,
}
impl Engine {
    pub(in crate::compressor::framing) fn new(config: FramingConfig) -> Self {
        Self {
            container: Container::new(config),
            resource: Resource::default(),
            stream: Default::default(),
            total_in: 0,
            total_out: 0,
            failed: false,
            finishing: false,
        }
    }
    pub(in crate::compressor::framing) fn reconfigure(&mut self, config: FramingConfig) {
        if config.chunk_bytes < self.container.config.chunk_bytes
            || self.retained_bytes() > config.max_buffer_bytes
        {
            *self = Self::new(config);
        } else {
            self.container.config = config;
        }
    }
    pub(in crate::compressor::framing) fn release_resource(&mut self, compressor: &mut Compressor) {
        self.resource.clear(compressor);
    }
    pub(in crate::compressor::framing) fn retained_bytes(&self) -> usize {
        self.container.retained_bytes() + self.resource.retained_bytes()
    }
    pub(in crate::compressor::framing) fn clear(&mut self, compressor: &mut Compressor) {
        self.resource.clear(compressor);
        self.container.reset();
        self.total_in = 0;
        self.total_out = 0;
        self.failed = false;
        self.finishing = false;
    }
    /// Makes every later call and command report `InvalidState`.
    pub(in crate::compressor::framing) const fn poison(&mut self) {
        self.failed = true;
    }
    pub(in crate::compressor::framing) fn start(
        &mut self,
        stream: FramedEncodeStreamConfig,
    ) -> Result<(), E> {
        self.stream = stream;
        self.container.start()
    }
    pub(in crate::compressor::framing) const fn offset(&self) -> u64 {
        self.container.offset()
    }
    pub(in crate::compressor::framing) const fn resources(&self) -> u64 {
        self.container.resources
    }
    pub(in crate::compressor::framing) const fn finished(&self) -> bool {
        self.container.is_finished()
    }
    pub(in crate::compressor::framing) const fn resource_finished(&self) -> bool {
        self.resource.finished
    }
    pub(in crate::compressor::framing) const fn resource_in(&self) -> u64 {
        self.resource.total_in
    }
    pub(in crate::compressor::framing) const fn resource_out(&self) -> u64 {
        self.resource.total_out
    }
    pub(in crate::compressor::framing) fn location(&self) -> FramedEncodeLocation {
        FramedEncodeLocation {
            resource_index: self.container.active.then_some(self.container.resources),
            item_index: None,
            resource_input_offset: self.container.active.then_some(self.resource.total_in),
            wire_offset: self.total_out,
        }
    }
    pub(in crate::compressor::framing) fn failure(
        &self,
        error: E,
        consumed: usize,
        produced: usize,
    ) -> FramedEncodeFailure {
        FramedEncodeFailure {
            error,
            consumed,
            produced,
            location: self.location(),
        }
    }
    fn command(&self) -> Result<(), E> {
        if self.failed || self.finishing {
            return Err(E::InvalidState);
        }
        if self.container.active {
            return Err(E::AbandonedResource);
        }
        self.container.drain()
    }
    pub(in crate::compressor::framing) fn metadata(
        &mut self,
        compressor: &mut Compressor,
        kind: MetadataKind,
        fields: &[MetadataField<'_>],
        options: MetadataOptions<'_>,
    ) -> Result<(), E> {
        self.command()?;
        self.container.metadata(compressor, kind, fields, options)
    }
    pub(in crate::compressor::framing) fn repeat(&mut self, codes: &[[u8; 2]]) -> Result<(), E> {
        self.command()?;
        self.container.repeat_metadata_fields(codes)
    }
    pub(in crate::compressor::framing) fn padding(&mut self, bytes: usize) -> Result<(), E> {
        self.command()?;
        self.container.padding(bytes)
    }
    pub(in crate::compressor::framing) fn begin_resource(
        &mut self,
        compressor: &mut Compressor,
        options: ResourceOptions,
        stream: StreamConfig,
        encoding: ResourceEncoding<'_>,
    ) -> Result<(), E> {
        self.command()?;
        let references = match encoding {
            ResourceEncoding::Uncompressed => alloc::vec::Vec::new(),
            ResourceEncoding::Brotli => {
                if compressor.config().window().encoding() == WindowEncoding::Large {
                    super::copy(&[0])?
                } else {
                    alloc::vec::Vec::new()
                }
            }
            ResourceEncoding::Shared { references, .. } => self.container.references(references)?,
        };
        self.resource.begin(
            &mut self.container,
            compressor,
            options,
            stream,
            encoding,
            references,
        )
    }
    pub(in crate::compressor::framing) fn process(
        &mut self,
        output: &mut [u8],
        operation: FramedEncodeOperation,
    ) -> Result<FramedEncodeProgress, FramedEncodeFailure> {
        let mut produced = 0;
        let result = (|| {
            if self.failed {
                return Err(E::InvalidState);
            }
            if self.container.active {
                return Err(E::AbandonedResource);
            }
            if self.finished() {
                return Ok(FramedEncoderStatus::Finished);
            }
            if self.finishing && operation != FramedEncodeOperation::Finish {
                return Err(E::InvalidState);
            }
            if operation == FramedEncodeOperation::Finish {
                if let InputSize::Exact(expected) = self.stream.input_size()
                    && expected != self.total_in
                {
                    return Err(E::InputSizeMismatch {
                        scope: "container",
                        expected,
                        actual: self.total_in,
                    });
                }
                self.finishing = true;
            }
            loop {
                let n = self.container.output(&mut output[produced..]);
                produced += n;
                self.total_out += n as u64;
                if self.container.has_pending() {
                    return Ok(FramedEncoderStatus::NeedsOutput);
                }
                if self.finished() {
                    return Ok(FramedEncoderStatus::Finished);
                }
                if operation == FramedEncodeOperation::Process {
                    return Ok(FramedEncoderStatus::NeedsInput);
                }
                self.container.finish()?;
            }
        })();
        match result {
            Ok(status) => Ok(FramedEncodeProgress {
                consumed: 0,
                produced,
                status,
            }),
            Err(error) => {
                self.failed = true;
                Err(self.failure(error, 0, produced))
            }
        }
    }
    pub(in crate::compressor::framing) fn process_resource(
        &mut self,
        compressor: &mut Compressor,
        dictionary: Option<&PreparedDictionary>,
        input: &[u8],
        output: &mut [u8],
        operation: Operation,
    ) -> Result<FramedEncodeProgress, FramedEncodeFailure> {
        if self.failed {
            return Err(self.failure(E::InvalidState, 0, 0));
        }
        let mut consumed = 0;
        let mut produced = 0;
        let result = self.resource.process(
            super::resource::Call {
                core: &mut self.container,
                compressor,
                dictionary,
                aggregate: self.stream.input_size(),
                total_in: &mut self.total_in,
                consumed: &mut consumed,
                produced: &mut produced,
            },
            input,
            output,
            operation,
        );
        self.total_out += produced as u64;
        match result {
            Ok(status) => Ok(FramedEncodeProgress {
                consumed,
                produced,
                status,
            }),
            Err(error) => {
                self.failed = true;
                Err(self.failure(error, consumed, produced))
            }
        }
    }
}

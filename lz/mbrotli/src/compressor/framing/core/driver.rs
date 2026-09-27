//! Shared lazy borrowed-input driver for all one-shot destinations and Read.
use crate::compressor::framing::*;
use crate::{InputSize, Operation};
use FramedEncodeError as E;

#[derive(Debug)]
pub(in crate::compressor::framing) struct Driver<'c, 'input> {
    session: FramedEncoderSession<'c>,
    pub(in crate::compressor::framing) input: FramedInput<'input>,
    item: usize,
    resource_offset: usize,
    resource_active: bool,
    selection_done: bool,
}
impl<'c, 'input> Driver<'c, 'input> {
    pub(in crate::compressor::framing) fn aggregate(
        input: FramedInput<'_>,
    ) -> Result<InputSize, E> {
        let size = input.items.iter().try_fold(0usize, |sum, item| {
            sum.checked_add(if let FramedItem::Resource(resource) = item {
                resource.data.len()
            } else {
                0
            })
            .ok_or(E::Overflow)
        })?;
        Ok(InputSize::Exact(size as u64))
    }
    pub(in crate::compressor::framing) const fn new(
        session: FramedEncoderSession<'c>,
        input: FramedInput<'input>,
    ) -> Self {
        Self {
            session,
            input,
            item: 0,
            resource_offset: 0,
            resource_active: false,
            selection_done: false,
        }
    }
    pub(in crate::compressor::framing) fn failure(
        &self,
        error: E,
        consumed: usize,
        produced: usize,
    ) -> FramedEncodeFailure {
        let mut e = self.session.owner.engine.failure(error, consumed, produced);
        e.location.item_index = (self.item < self.input.items.len()).then_some(self.item);
        if matches!(
            self.input.items.get(self.item),
            Some(FramedItem::Resource(_))
        ) {
            e.location.resource_index = Some(self.session.resources_encoded());
            e.location.resource_input_offset = Some(self.resource_offset as u64);
        }
        e
    }
    pub(in crate::compressor::framing) fn process(
        &mut self,
        output: &mut [u8],
    ) -> Result<FramedEncodeProgress, FramedEncodeFailure> {
        let mut consumed = 0;
        let mut produced = 0;
        let result = (|| {
            loop {
                if self.resource_active {
                    let FramedItem::Resource(resource) = self.input.items[self.item] else {
                        return Err(E::InvalidState);
                    };
                    let dictionary = match resource.encoding {
                        ResourceEncoding::Shared { dictionary, .. } => Some(dictionary),
                        _ => None,
                    };
                    let owner = &mut self.session.owner;
                    let p = match owner.engine.process_resource(
                        &mut owner.raw,
                        dictionary,
                        &resource.data[self.resource_offset..],
                        &mut output[produced..],
                        Operation::Finish,
                    ) {
                        Ok(p) => p,
                        Err(e) => {
                            consumed += e.consumed;
                            produced += e.produced;
                            self.resource_offset += e.consumed;
                            return Err(e.error);
                        }
                    };
                    consumed += p.consumed;
                    produced += p.produced;
                    self.resource_offset += p.consumed;
                    if p.status == FramedEncoderStatus::Finished {
                        self.resource_active = false;
                        self.item += 1;
                        self.resource_offset = 0;
                    } else {
                        return Ok(p.status);
                    }
                }
                let finish = self.item == self.input.items.len() && self.selection_done;
                let p = match self.session.process(
                    &mut output[produced..],
                    if finish {
                        FramedEncodeOperation::Finish
                    } else {
                        FramedEncodeOperation::Process
                    },
                ) {
                    Ok(p) => p,
                    Err(e) => {
                        produced += e.produced;
                        return Err(e.error);
                    }
                };
                produced += p.produced;
                if p.status != FramedEncoderStatus::NeedsInput {
                    return Ok(p.status);
                }
                if !self.selection_done {
                    if let Some(codes) = self.input.repeat_metadata_fields {
                        self.session.repeat_metadata_fields(codes)?;
                    }
                    self.selection_done = true;
                }
                if self.item == self.input.items.len() {
                    continue;
                }
                match self.input.items[self.item] {
                    FramedItem::Resource(resource) => {
                        if let InputSize::Exact(expected) = resource.stream.input_size()
                            && expected != resource.data.len() as u64
                        {
                            return Err(E::InputSizeMismatch {
                                scope: "resource",
                                expected,
                                actual: resource.data.len() as u64,
                            });
                        }
                        let owner = &mut self.session.owner;
                        owner.engine.begin_resource(
                            &mut owner.raw,
                            resource.options,
                            resource.stream,
                            resource.encoding,
                        )?;
                        self.resource_active = true;
                    }
                    FramedItem::Metadata {
                        kind,
                        fields,
                        options,
                    } => {
                        self.session.metadata_with_options(kind, fields, options)?;
                        self.item += 1;
                    }
                    FramedItem::Padding { bytes } => {
                        self.session.padding(bytes)?;
                        self.item += 1;
                    }
                }
            }
        })();
        result
            .map(|status| FramedEncodeProgress {
                consumed,
                produced,
                status,
            })
            .map_err(|error| self.failure(error, consumed, produced))
    }
}

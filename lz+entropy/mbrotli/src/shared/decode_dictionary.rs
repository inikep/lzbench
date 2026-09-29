//! Owned decode-only prefix storage; no encoder search indexes are constructed.

use crate::dictionary::{DecodeDictionaryError, DecodeDictionaryLimits, DictionaryAttachment};
use alloc::vec::Vec;

#[derive(Debug, Default)]
pub(crate) struct Prefixes {
    #[cfg(feature = "experimental")]
    pub(crate) description: Option<super::decode_serialized::Description>,
    segments: [Vec<u8>; 15],
    count: usize,
    total: u64,
    retained: usize,
}

impl Prefixes {
    pub(crate) fn build(
        attachments: &[DictionaryAttachment<'_>],
        limits: DecodeDictionaryLimits,
    ) -> Result<Self, DecodeDictionaryError> {
        let mut result = Self::default();
        let mut source_bytes = 0u64;
        for attachment in attachments {
            #[cfg(not(feature = "experimental"))]
            let DictionaryAttachment::Raw(source) = attachment;
            #[cfg(feature = "experimental")]
            let source = match attachment {
                DictionaryAttachment::Raw(source) => source,
                DictionaryAttachment::Serialized(source) => source,
            };
            source_bytes = source_bytes
                .checked_add(source.len() as u64)
                .ok_or(DecodeDictionaryError::SizeOverflow)?;
            if let Some(limit) = limits.max_source_bytes
                && source_bytes > limit
            {
                return Err(DecodeDictionaryError::SourceLimitExceeded { limit });
            }
            match attachment {
                DictionaryAttachment::Raw(source) => result.attach(source, limits)?,
                #[cfg(feature = "experimental")]
                DictionaryAttachment::Serialized(source) => {
                    result.attach_serialized(source, limits)?
                }
            }
        }
        Ok(result)
    }

    #[cfg(feature = "experimental")]
    fn attach_serialized(
        &mut self,
        source: &[u8],
        limits: DecodeDictionaryLimits,
    ) -> Result<(), DecodeDictionaryError> {
        let (description, prefix) = super::decode_serialized::Description::parse(
            source,
            self.retained,
            limits.max_owned_bytes,
        )?;
        if description.custom() && self.description.is_some() {
            return Err(DecodeDictionaryError::ConflictingStaticDictionaries);
        }
        let new_bytes = description.retained_bytes();
        // Until the replacement commits, both the old and the new description
        // are live. Account for that peak when copying an embedded prefix.
        self.retained = self
            .retained
            .checked_add(new_bytes)
            .ok_or(DecodeDictionaryError::SizeOverflow)?;
        if !prefix.is_empty() {
            self.attach(&source[prefix], limits)?;
        }
        if let Some(previous) = self.description.take() {
            self.retained -= previous.retained_bytes();
        }
        self.description = description.custom().then_some(description);
        Ok(())
    }

    fn attach(
        &mut self,
        source: &[u8],
        limits: DecodeDictionaryLimits,
    ) -> Result<(), DecodeDictionaryError> {
        if self.count == self.segments.len() {
            return Err(DecodeDictionaryError::TooManyAttachments);
        }
        let next = self
            .retained
            .checked_add(source.len())
            .ok_or(DecodeDictionaryError::SizeOverflow)?;
        if let Some(limit) = limits.max_owned_bytes
            && next > limit
        {
            return Err(DecodeDictionaryError::MemoryLimitExceeded { limit });
        }
        self.total = self
            .total
            .checked_add(source.len() as u64)
            .ok_or(DecodeDictionaryError::SizeOverflow)?;
        let segment = &mut self.segments[self.count];
        segment
            .try_reserve_exact(source.len())
            .map_err(|_| DecodeDictionaryError::AllocationFailed)?;
        segment.extend_from_slice(source);
        self.retained += segment.capacity();
        self.count += 1;
        Ok(())
    }

    pub(crate) const fn count(&self) -> usize {
        self.count
    }
    pub(crate) const fn retained_bytes(&self) -> usize {
        self.retained
    }
    pub(crate) const fn total(&self) -> u64 {
        self.total
    }

    /// Contiguous bytes from `offset` to the end of the segment holding it.
    /// Empty past the end of the prefix, so a caller advances segment by
    /// segment without a separate bounds test.
    pub(crate) fn run_from(&self, mut offset: u64) -> &[u8] {
        for segment in &self.segments[..self.count] {
            if offset < segment.len() as u64 {
                return &segment[offset as usize..];
            }
            offset -= segment.len() as u64;
        }
        &[]
    }
}

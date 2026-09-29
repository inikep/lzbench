use super::super::DecodeError;
use alloc::vec::Vec;

#[derive(Debug, Default)]
pub(super) struct Memory {
    pub(super) live: usize,
    pub(super) limit: Option<usize>,
}

impl Memory {
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(super) fn resize<T: Default + Clone>(
        &mut self,
        buffer: &mut Vec<T>,
        length: usize,
    ) -> Result<(), DecodeError> {
        self.reserve(buffer, length)?;
        buffer.resize(length, T::default());
        Ok(())
    }

    /// Ensures capacity for `length` elements under the workspace policy
    /// without touching the length, so a caller that appends in many small
    /// steps accounts once and then uses plain vector growth.
    pub(super) fn reserve<T>(
        &mut self,
        buffer: &mut Vec<T>,
        length: usize,
    ) -> Result<(), DecodeError> {
        if length > buffer.capacity() {
            // Grow geometrically from a non-empty buffer so a workspace that
            // fills up in steps reallocates a logarithmic number of times.
            let target = length.max(buffer.capacity().saturating_mul(2));
            // An allocator may implement realloc as allocate/copy/free. The
            // old buffer is still live when the replacement is requested.
            let replacement = target
                .checked_mul(size_of::<T>())
                .ok_or(DecodeError::SizeOverflow)?;
            let next = self
                .live
                .checked_add(replacement)
                .ok_or(DecodeError::SizeOverflow)?;
            if let Some(limit) = self.limit
                && next > limit
            {
                return Err(DecodeError::MemoryLimitExceeded { limit });
            }
            let before = buffer.capacity();
            buffer
                .try_reserve_exact(target - buffer.len())
                .map_err(|_| DecodeError::AllocationFailed)?;
            self.live += (buffer.capacity() - before) * size_of::<T>();
        }
        Ok(())
    }
}

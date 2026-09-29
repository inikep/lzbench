//! Demand-driven bit input over a 64-bit reservoir.
//!
//! The slow path accepts one byte at a time and never accepts a byte beyond
//! the field that needs it, so limits and progress stay byte-exact. The fast
//! path loads whole words while at least eight acceptable bytes remain. Output
//! pauses and member boundaries return current-call whole bytes with
//! [`Bits::unread`]; input pauses retain incomplete fields for the next call.

use super::super::{DecodeError, InvalidDataKind};

/// Widest field either path reads in one step.
pub(super) const MAX_PEEK: u32 = 56;

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Bits {
    value: u64,
    count: u32,
}

pub(crate) struct Input<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) consumed: usize,
    pub(crate) total_before: u64,
    pub(crate) limit: Option<u64>,
    /// End of the prefix this call may accept without per-byte limit checks,
    /// fixed at construction because the other fields never change mid-call.
    fast_end: usize,
}

impl<'a> Input<'a> {
    pub(crate) fn new(bytes: &'a [u8], total_before: u64, limit: Option<u64>) -> Self {
        let budget = limit
            .map_or(u64::MAX, |limit| limit.saturating_sub(total_before))
            .min(u64::MAX - total_before);
        let fast_end =
            usize::try_from(budget).map_or(bytes.len(), |budget| budget.min(bytes.len()));
        Self {
            bytes,
            consumed: 0,
            total_before,
            limit,
            fast_end,
        }
    }

    /// End of the prefix this call may accept without per-byte limit checks.
    #[inline(always)]
    pub(super) const fn fast_end(&self) -> usize {
        self.fast_end
    }
}

/// Low `count` bits set, for `count` below 64.
#[inline(always)]
pub(super) const fn mask(count: u32) -> u64 {
    debug_assert!(count < 64);
    (1u64 << count) - 1
}

impl Bits {
    /// Buffered bits.
    #[inline(always)]
    pub(super) const fn count(&self) -> u32 {
        self.count
    }

    /// Buffered bits, least significant first. Bits above `count` are zero
    /// or, after a whole-word refill, the input bytes that follow.
    #[inline(always)]
    pub(super) const fn value(&self) -> u64 {
        self.value
    }

    /// Accepts one byte, or reports exhausted input.
    pub(super) fn load_byte(&mut self, input: &mut Input<'_>) -> Result<bool, DecodeError> {
        let Some(&byte) = input.bytes.get(input.consumed) else {
            return Ok(false);
        };
        let next = input
            .total_before
            .checked_add(input.consumed as u64)
            .and_then(|value| value.checked_add(1))
            .ok_or(DecodeError::SizeOverflow)?;
        if let Some(limit) = input.limit
            && next > limit
        {
            return Err(DecodeError::InputLimitExceeded { limit });
        }
        self.value |= u64::from(byte) << self.count;
        self.count += 8;
        input.consumed += 1;
        Ok(true)
    }

    #[inline]
    pub(super) fn peek(
        &mut self,
        count: u32,
        input: &mut Input<'_>,
    ) -> Result<Option<u64>, DecodeError> {
        debug_assert!(count <= MAX_PEEK);
        while self.count < count {
            if !self.load_byte(input)? {
                return Ok(None);
            }
        }
        Ok(Some(self.value & mask(count)))
    }

    /// Loads whole bytes until at least 56 bits are buffered. Returns false,
    /// leaving the reservoir unchanged, when fewer than eight acceptable bytes
    /// remain before the input's fast end and at most 56 bits are buffered.
    #[inline(always)]
    pub(super) fn refill(&mut self, input: &mut Input<'_>) -> bool {
        let Input {
            bytes,
            consumed,
            fast_end,
            ..
        } = input;
        self.refill_from(&bytes[..*fast_end], consumed)
    }

    /// [`Self::refill`] over the acceptable prefix of the input, with the
    /// cursor held by the caller so a hot loop keeps it in a register.
    #[inline(always)]
    pub(super) fn refill_from(&mut self, fast: &[u8], consumed: &mut usize) -> bool {
        let Some(chunk) = fast
            .get(*consumed..)
            .and_then(|rest| rest.first_chunk::<8>())
        else {
            return self.count > MAX_PEEK;
        };
        debug_assert!(self.count < 64);
        // The whole word goes in unmasked: the bits past the bytes it accepts
        // are the input bytes that follow, which a later load ORs in again
        // unchanged. Leaving them keeps the mask off the decoding chain.
        self.value |= u64::from_le_bytes(*chunk) << self.count;
        let bytes = (63 - self.count) >> 3;
        self.count += bytes * 8;
        *consumed += bytes as usize;
        true
    }

    /// Clears the bits above `count`, which a whole-word refill leaves
    /// holding input bytes it did not accept, before the reservoir outlives
    /// the call whose input they came from.
    pub(super) const fn settle(&mut self) {
        self.value &= mask(self.count);
    }

    /// Returns whole buffered bytes accepted during this call to the input.
    /// A partial field carried over from an input pause can contain older
    /// whole bytes; those remain buffered because the caller already advanced.
    pub(super) fn unread(&mut self, input: &mut Input<'_>) {
        let whole = (self.count / 8).min(input.consumed.min(8) as u32);
        input.consumed -= whole as usize;
        self.count -= whole * 8;
        if self.count < 64 {
            self.value &= mask(self.count);
        }
    }

    #[inline(always)]
    pub(super) const fn drop(&mut self, count: u32) {
        self.value >>= count;
        self.count -= count;
    }

    /// Reads `count` already buffered bits.
    #[inline(always)]
    pub(super) const fn take(&mut self, count: u32) -> u64 {
        debug_assert!(count <= self.count);
        let value = self.value & mask(count);
        self.drop(count);
        value
    }

    #[inline]
    pub(super) fn read(
        &mut self,
        count: u32,
        input: &mut Input<'_>,
    ) -> Result<Option<u64>, DecodeError> {
        let value = self.peek(count, input)?;
        if value.is_some() {
            self.drop(count);
        }
        Ok(value)
    }

    pub(super) fn align(&mut self) -> Result<(), DecodeError> {
        let padding = self.count % 8;
        if self.value & mask(padding) != 0 {
            return Err(InvalidDataKind::Padding.into());
        }
        self.drop(padding);
        Ok(())
    }

    pub(super) fn uint8(&mut self, input: &mut Input<'_>) -> Result<Option<usize>, DecodeError> {
        let Some(first) = self.peek(1, input)? else {
            return Ok(None);
        };
        if first == 0 {
            self.drop(1);
            return Ok(Some(0));
        }
        let Some(head) = self.peek(4, input)? else {
            return Ok(None);
        };
        let n = (head >> 1) as u32;
        if n == 0 {
            self.drop(4);
            return Ok(Some(1));
        }
        let Some(all) = self.peek(4 + n, input)? else {
            return Ok(None);
        };
        self.drop(4 + n);
        Ok(Some((1usize << n) + (all >> 4) as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn probe(bytes: &[u8]) -> Input<'_> {
        Input::new(bytes, 0, None)
    }
    #[test]
    fn cumulative_input_overflow_is_reported_before_accepting_a_byte() {
        let mut input = Input::new(&[0], u64::MAX, None);
        assert!(matches!(
            Bits::default().read(1, &mut input),
            Err(DecodeError::SizeOverflow)
        ));
        assert_eq!(input.consumed, 0);
        assert_eq!(input.fast_end(), 0);
    }
    #[test]
    fn partial_fields_retain_bytes_but_never_read_past_the_requested_field() {
        let mut bits = Bits::default();
        let mut first = probe(&[0x01]);
        assert_eq!(bits.read(12, &mut first).unwrap(), None);
        let mut second = Input::new(&[0x02, 0xaa], 1, None);
        assert_eq!(bits.read(12, &mut second).unwrap(), Some(0x201));
        assert_eq!(second.consumed, 1);
        bits.align().unwrap();
        assert_eq!(bits.read(8, &mut second).unwrap(), Some(0xaa));
    }

    #[test]
    fn unread_keeps_bytes_accepted_by_an_earlier_call() {
        let mut bits = Bits {
            value: u64::MAX,
            count: 64,
        };
        let mut input = probe(&[]);
        bits.unread(&mut input);
        assert_eq!(bits.count(), 64);
        assert_eq!(bits.value(), u64::MAX);
        input.consumed = 2;
        bits.unread(&mut input);
        assert_eq!(input.consumed, 0);
        assert_eq!(bits.count(), 48);
        assert_eq!(bits.value(), mask(48));
    }
    #[test]
    fn unmasked_refill_agrees_with_byte_loads_and_settles_to_accepted_bits() {
        let bytes: alloc::vec::Vec<u8> = (1..=16).collect();
        let mut word = Bits::default();
        let mut input = probe(&bytes);
        assert!(word.refill(&mut input));
        assert_eq!((word.count(), input.consumed), (56, 7));
        // The eighth byte rides along above the accepted bits.
        assert_eq!(word.value(), u64::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8]));
        let mut exact = Bits::default();
        let mut reference = probe(&bytes);
        for _ in 0..7 {
            assert!(exact.load_byte(&mut reference).unwrap());
        }
        assert_eq!(word.value() & mask(56), exact.value());
        // Loading that byte again ORs in the bits already there.
        word.drop(12);
        exact.drop(12);
        assert!(word.load_byte(&mut input).unwrap());
        assert!(exact.load_byte(&mut reference).unwrap());
        assert_eq!((word.count(), word.value()), (exact.count(), exact.value()));
        // A settled reservoir holds only accepted bits.
        let mut ahead = Bits::default();
        let mut input = probe(&bytes);
        assert!(ahead.refill(&mut input));
        ahead.drop(4);
        ahead.settle();
        assert_eq!(
            ahead.value(),
            u64::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 0]) >> 4
        );
        // Without a whole word, the answer depends only on what is buffered.
        let mut short = probe(&bytes[..7]);
        assert!(!Bits::default().refill(&mut short));
        let mut full = Bits {
            value: 0,
            count: 57,
        };
        assert!(full.refill(&mut short));
        assert_eq!((full.count(), short.consumed), (57, 0));
    }

    #[test]
    fn refill_loads_whole_words_and_unread_returns_unused_bytes() {
        let bytes: alloc::vec::Vec<u8> = (1..=20).collect();
        let mut bits = Bits::default();
        let mut input = probe(&bytes);
        assert!(bits.read(3, &mut input).unwrap().is_some());
        assert!(bits.refill(&mut input));
        assert_eq!((bits.count(), input.consumed), (61, 8));
        assert!(bits.refill(&mut input));
        assert_eq!(bits.count(), 61);
        assert_eq!(bits.take(13), (0x03_02_01u64 >> 3) & mask(13));
        bits.unread(&mut input);
        assert_eq!(bits.count(), 0);
        assert_eq!(input.consumed, 2);
        assert_eq!(bits.read(8, &mut input).unwrap(), Some(3));
        // Fewer than eight acceptable bytes: the reservoir is left alone.
        let mut limited = Input::new(&bytes, 0, Some(9));
        limited.consumed = 2;
        assert!(!bits.refill(&mut limited));
        assert_eq!(bits.count(), 0);
        // A slice shorter than the fast end cannot supply a whole word either.
        input.consumed = bytes.len() - 3;
        assert!(!bits.refill(&mut input));
        assert_eq!(bits.count(), 0);
    }
    #[test]
    fn fast_end_respects_limits_and_slice_length() {
        let bytes = [0u8; 10];
        assert_eq!(probe(&bytes).fast_end(), 10);
        assert_eq!(Input::new(&bytes, 0, Some(7)).fast_end(), 7);
        assert_eq!(Input::new(&bytes, 9, Some(7)).fast_end(), 0);
        assert_eq!(Input::new(&bytes, u64::MAX - 3, None).fast_end(), 3);
        assert_eq!(mask(0), 0);
        assert_eq!(mask(56), (1 << 56) - 1);
    }
    #[test]
    fn full_reservoir_refill_and_uint8_decode() {
        let bytes = [0xff; 16];
        let mut bits = Bits::default();
        let mut input = probe(&bytes);
        assert!(bits.refill(&mut input));
        assert_eq!((bits.count(), input.consumed), (56, 7));
        bits.drop(56 - 7);
        assert!(bits.refill(&mut input));
        assert_eq!((bits.count(), input.consumed), (63, 14));
        assert_eq!(
            Bits::default().uint8(&mut probe(&[0b0001_0011])).unwrap(),
            Some(3)
        );
        assert_eq!(
            Bits::default().uint8(&mut probe(&[0b0000_0001])).unwrap(),
            Some(1)
        );
        assert_eq!(Bits::default().uint8(&mut probe(&[0])).unwrap(), Some(0));
    }
}

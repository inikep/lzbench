use super::super::DecodeError;
use super::{
    bits::{Bits, Input},
    huffman::{Builder, Huffman},
    memory::Memory,
};
use crate::shared::format::PREFIX_CODE_RANGES;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Stage {
    TypeCount,
    TypeTree,
    LengthTree,
    LengthSymbol,
    LengthExtra { symbol: usize },
    Ready,
}

/// Independent block-switch state for one of literals, commands or distances.
/// The two previous types are initialized to the format's virtual types 1 and 0.
#[derive(Debug)]
pub(super) struct Block {
    pub(super) count: usize,
    pub(super) current: usize,
    pub(super) remaining: u64,
    previous: usize,
    stage: Stage,
    types: Huffman,
    lengths: Huffman,
}

impl Default for Block {
    fn default() -> Self {
        Self {
            count: 1,
            current: 0,
            previous: 1,
            remaining: u64::MAX,
            stage: Stage::TypeCount,
            types: Huffman::default(),
            lengths: Huffman::default(),
        }
    }
}

impl Block {
    /// Starts a meta-block: one type, no switches, and no pending header.
    pub(super) const fn reset(&mut self) {
        self.count = 1;
        self.current = 0;
        self.previous = 1;
        self.remaining = u64::MAX;
        self.stage = Stage::TypeCount;
    }

    pub(super) fn header(
        &mut self,
        bits: &mut Bits,
        input: &mut Input<'_>,
        builder: &mut Builder,
        memory: &mut Memory,
    ) -> Result<bool, DecodeError> {
        loop {
            match self.stage {
                Stage::TypeCount => {
                    let Some(count) = bits.uint8(input)? else {
                        return Ok(false);
                    };
                    self.count = count + 1;
                    if count == 0 {
                        self.stage = Stage::Ready;
                        return Ok(true);
                    }
                    self.stage = Stage::TypeTree;
                }
                Stage::TypeTree => {
                    if !builder.read(self.count + 2, bits, input)? {
                        return Ok(false);
                    }
                    builder.build(self.count + 2, memory, &mut self.types)?;
                    self.stage = Stage::LengthTree;
                }
                Stage::LengthTree => {
                    if !builder.read(26, bits, input)? {
                        return Ok(false);
                    }
                    builder.build(26, memory, &mut self.lengths)?;
                    self.stage = Stage::LengthSymbol;
                }
                Stage::LengthSymbol | Stage::LengthExtra { .. } | Stage::Ready => {
                    return self.length(bits, input);
                }
            }
        }
    }

    fn length(&mut self, bits: &mut Bits, input: &mut Input<'_>) -> Result<bool, DecodeError> {
        if self.stage == Stage::LengthSymbol {
            let Some(symbol) = self.lengths.codes().decode_refilling(bits, input)? else {
                return Ok(false);
            };
            self.stage = Stage::LengthExtra { symbol };
        }
        if let Stage::LengthExtra { symbol } = self.stage {
            let (base, width) = PREFIX_CODE_RANGES[symbol];
            let Some(extra) = bits.read(width, input)? else {
                return Ok(false);
            };
            self.remaining = u64::from(base) + extra;
            self.stage = Stage::Ready;
        }
        Ok(true)
    }

    fn switch_to(&mut self, symbol: usize) {
        let next = match symbol {
            0 => self.previous,
            1 => self.current + 1,
            _ => symbol - 2,
        } % self.count;
        self.previous = self.current;
        self.current = next;
    }

    /// Byte-exact block switch when the current block is exhausted.
    pub(super) fn prepare(
        &mut self,
        bits: &mut Bits,
        input: &mut Input<'_>,
    ) -> Result<bool, DecodeError> {
        if self.remaining != 0 {
            return Ok(true);
        }
        if self.stage == Stage::Ready {
            let Some(symbol) = self.types.codes().decode_refilling(bits, input)? else {
                return Ok(false);
            };
            self.switch_to(symbol);
            self.stage = Stage::LengthSymbol;
        }
        self.length(bits, input)
    }

    /// Block switch with at least 54 buffered bits: two codes and a
    /// 24-bit extra field.
    #[inline(always)]
    pub(super) fn switch_fast(&mut self, bits: &mut Bits) {
        debug_assert!(self.remaining == 0 && self.stage == Stage::Ready);
        let symbol = self.types.codes().decode_fast(bits);
        self.switch_to(symbol);
        let symbol = self.lengths.codes().decode_fast(bits);
        let (base, width) = PREFIX_CODE_RANGES[symbol];
        self.remaining = u64::from(base) + bits.take(width);
    }

    /// Whether the fast switch may run: no partially parsed switch is pending.
    pub(super) fn switch_ready(&self) -> bool {
        self.stage == Stage::Ready
    }

    pub(super) const fn advance(&mut self) {
        self.remaining -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures;
    use super::*;

    fn switch_fixture(symbol: u64) -> alloc::vec::Vec<u8> {
        fixtures::fields(&[
            (4, 1), // two block types
            (2, 1),
            (2, 0),
            (2, symbol), // single type-switch symbol
            (2, 1),
            (2, 0),
            (5, 0), // single length symbol, base 1 + 2 bits
            (2, 0),
            (2, 0),
            (2, 0),
        ])
    }

    #[test]
    fn encoded_switches_use_previous_incremented_and_explicit_types() {
        for symbol in [0, 1, 3] {
            let bytes = switch_fixture(symbol);
            let mut bits = Bits::default();
            let mut input = fixtures::input(&bytes);
            let mut block = Block::default();
            let mut memory = Memory::default();
            assert!(
                block
                    .header(&mut bits, &mut input, &mut Builder::default(), &mut memory)
                    .unwrap()
            );
            assert_eq!((block.count, block.current, block.remaining), (2, 0, 1));
            block.advance();
            assert!(block.prepare(&mut bits, &mut input).unwrap());
            assert_eq!((block.current, block.remaining), (1, 1));
            block.advance();
            assert!(block.prepare(&mut bits, &mut input).unwrap());
            assert_eq!(block.current, usize::from(symbol == 3));
            block.reset();
            assert_eq!((block.count, block.remaining), (1, u64::MAX));
        }
    }

    #[test]
    fn fast_switch_matches_the_byte_exact_switch() {
        for symbol in [0, 1, 3] {
            let mut bytes = switch_fixture(symbol);
            bytes.extend([0u8; 8]);
            let mut slow = Block::default();
            let mut fast = Block::default();
            let mut memory = Memory::default();
            for (block, use_fast) in [(&mut slow, false), (&mut fast, true)] {
                let mut bits = Bits::default();
                let mut input = fixtures::input(&bytes);
                assert!(
                    block
                        .header(&mut bits, &mut input, &mut Builder::default(), &mut memory)
                        .unwrap()
                );
                block.advance();
                bits.unread(&mut input);
                assert!(bits.refill(&mut input));
                if use_fast {
                    assert!(block.switch_ready());
                    block.switch_fast(&mut bits);
                } else {
                    assert!(block.prepare(&mut bits, &mut input).unwrap());
                }
            }
            assert_eq!(
                (slow.current, slow.previous, slow.remaining),
                (fast.current, fast.previous, fast.remaining)
            );
        }
    }
}

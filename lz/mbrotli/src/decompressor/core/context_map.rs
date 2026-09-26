use super::super::{DecodeError, InvalidDataKind};
use super::{
    bits::{Bits, Input},
    huffman::{Builder, Huffman},
    memory::Memory,
};
use alloc::vec::Vec;

/// Each phase commits only complete fields; pending repeats retain their width.
#[derive(Debug, Default)]
enum Stage {
    #[default]
    TreeCount,
    RunPrefix,
    Tree,
    Entries,
    Repeat {
        width: u32,
    },
    Transform,
    Complete,
}

#[derive(Debug, Default)]
pub(super) struct ContextMap {
    pub(super) values: Vec<u8>,
    pub(super) trees: usize,
    stage: Stage,
    run: u8,
    index: usize,
    tree: Huffman,
    /// One bit per literal block type whose 64 contexts share a tree.
    trivial: [u64; 4],
}

impl ContextMap {
    pub(super) fn reset(&mut self) {
        self.stage = Stage::TreeCount;
        self.values.clear();
        self.trivial = [0; 4];
    }

    /// Marks each literal block type whose 64 contexts select one tree, so
    /// literal decoding can skip the context computation for it.
    pub(super) fn detect_trivial(&mut self) {
        self.trivial = [0; 4];
        for (block, contexts) in self.values.as_chunks::<64>().0.iter().enumerate() {
            let sample = contexts[0];
            let error = contexts
                .iter()
                .fold(0u8, |error, &value| error | (value ^ sample));
            if error == 0 && block < 256 {
                self.trivial[block >> 6] |= 1 << (block & 63);
            }
        }
    }

    /// Whether literal block type `block` decodes every context with one tree.
    #[inline(always)]
    pub(super) const fn trivial(&self, block: usize) -> bool {
        (self.trivial[(block >> 6) & 3] >> (block & 63)) & 1 != 0
    }

    pub(super) fn read(
        &mut self,
        size: usize,
        bits: &mut Bits,
        input: &mut Input<'_>,
        builder: &mut Builder,
        memory: &mut Memory,
    ) -> Result<bool, DecodeError> {
        loop {
            match self.stage {
                Stage::TreeCount => {
                    let Some(n) = bits.uint8(input)? else {
                        return Ok(false);
                    };
                    self.trees = n + 1;
                    self.index = 0;
                    memory.resize(&mut self.values, size)?;
                    self.values.fill(0);
                    if n == 0 {
                        self.stage = Stage::Complete;
                        return Ok(true);
                    }
                    self.stage = Stage::RunPrefix;
                }
                Stage::RunPrefix => {
                    let Some(flag) = bits.peek(1, input)? else {
                        return Ok(false);
                    };
                    self.run = if flag == 0 {
                        bits.drop(1);
                        0
                    } else {
                        let Some(n) = bits.read(5, input)? else {
                            return Ok(false);
                        };
                        (n >> 1) as u8 + 1
                    };
                    self.stage = Stage::Tree;
                }
                Stage::Tree => {
                    let alphabet = self.trees + usize::from(self.run);
                    if !builder.read(alphabet, bits, input)? {
                        return Ok(false);
                    }
                    builder.build(alphabet, memory, &mut self.tree)?;
                    self.stage = Stage::Entries;
                }
                Stage::Entries => {
                    if self.index == size {
                        self.stage = Stage::Transform;
                        continue;
                    }
                    let Some(symbol) = self.tree.codes().decode_refilling(bits, input)? else {
                        return Ok(false);
                    };
                    if symbol != 0 && symbol <= usize::from(self.run) {
                        self.stage = Stage::Repeat {
                            width: symbol as u32,
                        };
                        continue;
                    }
                    let value = if symbol == 0 {
                        0
                    } else {
                        symbol - usize::from(self.run)
                    };
                    if value >= self.trees {
                        return Err(InvalidDataKind::ContextMap.into());
                    }
                    self.values[self.index] = value as u8;
                    self.index += 1;
                }
                Stage::Repeat { width } => {
                    let Some(extra) = bits.read(width, input)? else {
                        return Ok(false);
                    };
                    let end = self.index + (1 << width) + extra as usize;
                    if end > size {
                        return Err(InvalidDataKind::ContextMap.into());
                    }
                    self.values[self.index..end].fill(0);
                    self.index = end;
                    self.stage = Stage::Entries;
                }
                Stage::Transform => {
                    let Some(mtf) = bits.read(1, input)? else {
                        return Ok(false);
                    };
                    if mtf != 0 {
                        let mut symbols = [0u8; 256];
                        for (i, value) in symbols.iter_mut().enumerate() {
                            *value = i as u8;
                        }
                        for value in &mut self.values {
                            let index = usize::from(*value);
                            *value = symbols[index];
                            symbols[..=index].rotate_right(1);
                        }
                    }
                    self.stage = Stage::Complete;
                    return Ok(true);
                }
                Stage::Complete => return Ok(true),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures;
    use super::*;

    #[test]
    fn encoded_zero_runs_and_move_to_front_have_observable_results() {
        let cases = [
            (
                fixtures::fields(&[
                    (4, 1),
                    (5, 1),
                    (2, 1),
                    (2, 0),
                    (2, 1),
                    (1, 0),
                    (1, 0),
                    (1, 1),
                ]),
                [0, 0, 0, 0],
            ),
            (
                fixtures::fields(&[
                    (4, 1),
                    (1, 0),
                    (2, 1),
                    (2, 1),
                    (1, 0),
                    (1, 1),
                    (4, 0b0101),
                    (1, 1),
                ]),
                [1, 1, 0, 0],
            ),
        ];
        for (bytes, expected) in cases {
            let mut map = ContextMap::default();
            assert!(
                map.read(
                    4,
                    &mut Bits::default(),
                    &mut fixtures::input(&bytes),
                    &mut Builder::default(),
                    &mut Memory::default()
                )
                .unwrap()
            );
            assert_eq!(map.trees, 2);
            assert_eq!(map.values, expected);
        }
    }

    #[test]
    fn trivial_literal_blocks_are_those_whose_contexts_share_one_tree() {
        let mut map = ContextMap {
            values: alloc::vec![0; 64 * 3],
            trees: 6,
            ..ContextMap::default()
        };
        map.values[64] = 1;
        map.values[128..].fill(5);
        map.detect_trivial();
        assert!(map.trivial(0));
        assert!(!map.trivial(1));
        assert!(map.trivial(2));
        assert!(!map.trivial(3));
        map.reset();
        assert!(!map.trivial(0));
    }
}

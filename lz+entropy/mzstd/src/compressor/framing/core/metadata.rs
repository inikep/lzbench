//! Metadata serialization and its self-contained Zstandard frames.

use alloc::vec::Vec;

use super::number;
use crate::compressor::Compressor;
use crate::compressor::framing::{FramingError, MetadataEncoding, MetadataField};
use crate::framing::core::varint_len;

/// Serializes `fields`, or only those whose code `codes` selects.
pub(super) fn serialize(
    fields: &[MetadataField<'_>],
    codes: Option<&[[u8; 2]]>,
    capacity: usize,
) -> Result<Vec<u8>, FramingError> {
    let mut content = super::bytes(capacity)?;
    for field in fields {
        if codes.is_none_or(|codes| codes.contains(&field.code)) {
            content.extend_from_slice(&field.code);
            number(field.value.len() as u64, &mut content)?;
            content.extend_from_slice(field.value);
        }
    }
    Ok(content)
}

/// Builds the chunk header of metadata `kind` and its content, compressed
/// into one Zstandard frame unless `encoding` stores it.
pub(super) fn encode(
    compressor: &mut Compressor,
    kind: u8,
    content: Vec<u8>,
    encoding: MetadataEncoding<'_>,
    references: Vec<u8>,
) -> Result<(Vec<u8>, Vec<u8>), FramingError> {
    // Include the repeated-kind byte before commit.
    let capacity = 3 + varint_len(content.len() as u64) + references.len();
    let mut header = super::bytes(capacity)?;
    header.push(kind);
    let dictionary = match encoding {
        MetadataEncoding::Uncompressed => {
            header.push(0);
            return Ok((header, content));
        }
        MetadataEncoding::Zstd => {
            header.push(2);
            None
        }
        MetadataEncoding::Shared { dictionary, .. } => {
            header.push(3);
            Some(dictionary)
        }
    };
    number(content.len() as u64, &mut header)?;
    header.extend_from_slice(&references);
    let bound =
        Compressor::max_compressed_size(content.len()).map_err(|_| FramingError::Overflow)?;
    let mut encoded = super::bytes(bound)?;
    match dictionary {
        Some(dictionary) => {
            compressor.compress_with_dictionary_into(dictionary, &content, &mut encoded)?
        }
        None => compressor.compress_into(&content, &mut encoded)?,
    };
    Ok((header, encoded))
}

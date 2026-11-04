//! Chunk header and metadata parsers over borrowed bytes.

use super::super::*;
use crate::framing::core::{
    CHUNK_MAGIC, FLAG_CHECKSUM, FLAG_DETACHED, FRAME_HEADER, ID_LEN, ID_TYPE, MAX_REFERENCES,
    MAX_VARINT_LEN, data_flags,
};
use crate::framing::{DictionaryId, DictionaryReference, MetadataKind};
use FramedDecodeError as E;
use alloc::vec::Vec;

/// Reads a varint at `cursor`; `None` when the bytes end first. Nonminimal
/// encodings are accepted, a ninth continuation byte is not.
pub(super) fn number(bytes: &[u8], cursor: &mut usize) -> Result<Option<u64>, E> {
    let mut value = 0;
    for shift in (0..7 * MAX_VARINT_LEN).step_by(7) {
        let Some(&byte) = bytes.get(*cursor) else {
            return Ok(None);
        };
        *cursor += 1;
        value |= u64::from(byte & 127) << shift;
        if byte < 128 {
            return Ok(Some(value));
        }
    }
    Err(E::InvalidChunk)
}

/// Reads a little-endian `u32` at `at`.
pub(super) fn le32(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)
        .and_then(|b| b.try_into().ok())
        .map(u32::from_le_bytes)
}

/// Parses a chunk header frame from its first byte: the skippable frame
/// header, then the chunk header, which must lie inside the frame. Returns
/// `None` until `bytes` holds the whole header, and an error as soon as
/// they cannot start a valid one.
pub(super) fn header(
    bytes: &[u8],
    offset: u64,
    reserve: impl FnOnce(usize) -> Result<(), E>,
) -> Result<Option<ChunkInfo>, E> {
    let Some(magic) = le32(bytes, 0) else {
        return Ok(None);
    };
    if magic != CHUNK_MAGIC {
        return Err(E::InvalidChunk);
    }
    let Some(size) = le32(bytes, 4) else {
        return Ok(None);
    };
    let end = FRAME_HEADER + size as usize;
    let mut p = FRAME_HEADER;
    macro_rules! byte {
        () => {{
            if p >= end {
                return Err(E::InvalidChunk);
            }
            let Some(&b) = bytes.get(p) else {
                return Ok(None);
            };
            p += 1;
            b
        }};
    }
    macro_rules! var {
        () => {{
            let Some(n) = number(bytes, &mut p)? else {
                if p >= end {
                    return Err(E::InvalidChunk);
                }
                return Ok(None);
            };
            if p > end {
                return Err(E::InvalidChunk);
            }
            n
        }};
    }
    macro_rules! hash {
        () => {{
            if byte!() != ID_TYPE {
                return Err(E::InvalidChunk);
            }
            let mut id = [0; ID_LEN - 1];
            for b in &mut id {
                *b = byte!();
            }
            DictionaryId(id)
        }};
    }
    let tag = if size == 0 { 0 } else { byte!() };
    let kind = match tag {
        0 => ChunkType::Padding,
        1 => ChunkType::Metadata,
        2 => ChunkType::Data,
        3 => ChunkType::FirstPartial,
        4 => ChunkType::MiddlePartial,
        5 => ChunkType::LastPartial,
        6 => ChunkType::FooterMetadata,
        7 => ChunkType::GlobalMetadata,
        8 => ChunkType::RepeatMetadata,
        9 => ChunkType::CentralDirectory,
        10 => ChunkType::Footer,
        _ => return Err(E::InvalidChunk),
    };
    let mut codec = None;
    let mut declared_size = None;
    let mut references = [None; MAX_REFERENCES];
    let mut count = 0;
    if (1..=8).contains(&tag) {
        codec = Some(match byte!() {
            0 => Codec::Uncompressed,
            2 => Codec::Zstd,
            3 => Codec::SharedZstd,
            _ => return Err(E::InvalidChunk),
        });
        if codec != Some(Codec::Uncompressed) {
            declared_size = Some(var!());
        }
        if codec == Some(Codec::SharedZstd) {
            count = byte!() as usize;
            if count == 0 || count > MAX_REFERENCES {
                return Err(E::InvalidDictionaryReference);
            }
            let mut serialized = 0;
            for slot in &mut references[..count] {
                let flags = byte!();
                *slot = Some(match flags {
                    0 => DictionaryReference::PrefixResource(var!()),
                    1 => DictionaryReference::PrefixChunk(var!()),
                    4 => DictionaryReference::SerializedResource(var!()),
                    2 => DictionaryReference::PrefixId(hash!()),
                    6 => DictionaryReference::SerializedId(hash!()),
                    _ => return Err(E::InvalidDictionaryReference),
                });
                serialized += usize::from(flags & 4 != 0);
            }
            // A frame has one dictionary: one serialized, or only prefixes.
            if serialized != 0 && count != 1 {
                return Err(E::InvalidDictionaryReference);
            }
        }
    }
    let mut flags = 0;
    let mut checksum = None;
    let mut body_length = None;
    if (2..=5).contains(&tag) {
        flags = byte!();
        if flags & !data_flags(tag) != 0 {
            return Err(E::InvalidChunk);
        }
        if flags & FLAG_CHECKSUM != 0 {
            checksum = Some(hash!());
        }
        if flags & FLAG_DETACHED != 0 {
            if codec == Some(Codec::Uncompressed) {
                return Err(E::InvalidChunk);
            }
            body_length = Some(var!());
        }
    }
    let repeated_kind = if tag == 8 {
        Some(match byte!() {
            1 => MetadataKind::Resource,
            6 => MetadataKind::Footer,
            _ => return Err(E::InvalidMetadata),
        })
    } else {
        None
    };
    if p != bytes.len() {
        return Err(E::InvalidChunk);
    }
    // A detached chunk keeps its content in the body frame only.
    if body_length.is_some() && p != end {
        return Err(E::InvalidChunk);
    }
    let length = (end as u64)
        .checked_add(body_length.unwrap_or(0))
        .ok_or(E::SizeOverflow)?;
    // Header storage is reserved by the caller before parsing. This bounded
    // temporary is transferred into the persistent record on success.
    reserve(count * size_of::<DictionaryReference>())?;
    let mut dictionaries = Vec::new();
    dictionaries
        .try_reserve_exact(count)
        .map_err(|_| E::AllocationFailed)?;
    dictionaries.extend(references[..count].iter().flatten().copied());
    Ok(Some(ChunkInfo {
        offset: ChunkOffset(offset),
        length,
        kind,
        codec,
        declared_size,
        dictionaries,
        header_bytes: Vec::new(),
        flags,
        checksum,
        repeated_kind,
        body_length,
    }))
}

type FieldIndex = Vec<([u8; 2], ::core::ops::Range<usize>)>;

/// Indexes serialized metadata fields, validating codes and reserved values.
pub(super) fn metadata(bytes: &[u8], kind: MetadataKind) -> Result<FieldIndex, E> {
    let mut fields = Vec::new();
    fields
        .try_reserve_exact(bytes.len() / 3)
        .map_err(|_| E::AllocationFailed)?;
    let mut p = 0;
    let mut name = false;
    let mut time = false;
    while p < bytes.len() {
        let code: [u8; 2] = bytes
            .get(p..p.checked_add(2).ok_or(E::SizeOverflow)?)
            .ok_or(E::InvalidMetadata)?
            .try_into()
            .map_err(|_| E::InvalidMetadata)?;
        p += 2;
        let length = number(bytes, &mut p)
            .map_err(|_| E::InvalidMetadata)?
            .ok_or(E::InvalidMetadata)?;
        let end = p
            .checked_add(usize::try_from(length).map_err(|_| E::SizeOverflow)?)
            .ok_or(E::SizeOverflow)?;
        let value = bytes.get(p..end).ok_or(E::InvalidMetadata)?;
        match &code {
            b"id" if kind == MetadataKind::Resource && !name => {
                ::core::str::from_utf8(value).map_err(|_| E::InvalidMetadata)?;
                name = true;
            }
            b"mt" if kind == MetadataKind::Resource && !time && value.len() == 8 => {
                time = true;
            }
            _ if code.iter().all(u8::is_ascii_uppercase) => {}
            _ => return Err(E::InvalidMetadata),
        }
        fields.push((code, p..end));
        p = end;
    }
    Ok(fields)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// A chunk header frame of `fields`, sized for `inline` content bytes.
    fn frame(fields: &[u8], inline: usize) -> Vec<u8> {
        let mut bytes = CHUNK_MAGIC.to_le_bytes().to_vec();
        bytes.extend_from_slice(&((fields.len() + inline) as u32).to_le_bytes());
        bytes.extend_from_slice(fields);
        bytes
    }

    #[test]
    fn varints_accept_nonminimal_but_reject_a_tenth_byte() {
        let mut cursor = 0;
        assert_eq!(number(&[0x80, 0], &mut cursor).unwrap(), Some(0));
        assert_eq!(cursor, 2);
        assert!(number(&[0x80; 10], &mut 0).is_err());
        assert_eq!(number(&[0xff; 8], &mut 0).unwrap(), None);
    }

    #[test]
    fn header_cannot_read_fields_outside_its_frame() {
        let budget = |_| Ok(());
        // A data chunk whose codec byte would lie past its one-byte frame.
        assert!(header(&frame(&[2], 0), 13, budget).is_err());
        assert!(header(&frame(&[2], 1), 13, budget).unwrap().is_none());
        assert!(header(&frame(&[9], 0), 13, budget).unwrap().is_some());
        assert!(header(&frame(&[11], 0), 13, budget).is_err());
        assert!(header(&[0x5a, 0x2a, 0x4d, 0x18], 13, budget).is_err());
        assert!(header(&[0x5b, 0x2a], 13, budget).unwrap().is_none());
        // An empty frame is padding.
        let padding = header(&frame(&[], 0), 13, budget).unwrap().unwrap();
        assert_eq!(padding.kind, ChunkType::Padding);
        assert_eq!(padding.length, 8);
    }

    #[test]
    fn detached_bodies_need_a_compressed_codec_and_no_inline_content() {
        let budget = |_| Ok(());
        let detached = header(&frame(&[2, 2, 5, 4, 9], 0), 13, budget)
            .unwrap()
            .unwrap();
        assert_eq!(detached.body_length, Some(9));
        assert_eq!(detached.length, 8 + 5 + 9);
        assert!(header(&frame(&[2, 0, 4, 9], 0), 13, budget).is_err());
        assert!(header(&frame(&[2, 2, 5, 4, 9], 1), 13, budget).is_err());
        // Flags a middle chunk may not carry.
        assert!(header(&frame(&[4, 2, 5, 1], 0), 13, budget).is_err());
    }

    #[test]
    fn references_name_one_serialized_dictionary_or_only_prefixes() {
        let budget = |_| Ok(());
        let mut serialized = vec![2, 3, 1, 1, 6, ID_TYPE];
        serialized.extend_from_slice(&[7; 32]);
        serialized.push(0);
        assert!(
            header(&frame(&serialized, 1), 13, budget)
                .unwrap()
                .is_some()
        );
        let mut mixed = vec![2, 3, 1, 2, 6, ID_TYPE];
        mixed.extend_from_slice(&[7; 32]);
        mixed.extend_from_slice(&[0, 13, 0]);
        assert!(matches!(
            header(&frame(&mixed, 1), 13, budget),
            Err(E::InvalidDictionaryReference)
        ));
        assert!(matches!(
            header(&frame(&[2, 3, 1, 0, 0], 1), 13, budget),
            Err(E::InvalidDictionaryReference)
        ));
        assert!(matches!(
            header(&frame(&[2, 3, 1, 1, 9, 13, 0], 1), 13, budget),
            Err(E::InvalidDictionaryReference)
        ));
    }

    #[test]
    fn metadata_index_preserves_duplicate_custom_values() {
        let fields = metadata(b"AA\x01xAA\x01y", MetadataKind::Global).unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[1].1, 7..8);
        assert!(metadata(b"id\x00", MetadataKind::Global).is_err());
        assert!(metadata(b"A", MetadataKind::Global).is_err());
    }
}

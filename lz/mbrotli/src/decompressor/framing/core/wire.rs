use super::super::*;
use crate::framing::{DictionaryId, DictionaryReference, MetadataKind};
use FramedDecodeError as E;
use alloc::vec::Vec;

pub(super) fn number(bytes: &[u8], cursor: &mut usize) -> Result<Option<u64>, E> {
    let mut value = 0;
    for shift in (0..63).step_by(7) {
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

pub(super) fn header(
    bytes: &[u8],
    offset: u64,
    reserve: impl FnOnce(usize) -> Result<(), E>,
) -> Result<Option<ChunkInfo>, E> {
    let mut p = 0;
    let Some(length) = number(bytes, &mut p)? else {
        return Ok(None);
    };
    let length = length.checked_add(p as u64).ok_or(E::SizeOverflow)?;
    macro_rules! byte {
        () => {{
            if p as u64 >= length {
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
                if p as u64 >= length {
                    return Err(E::InvalidChunk);
                }
                return Ok(None);
            };
            if p as u64 > length {
                return Err(E::InvalidChunk);
            }
            n
        }};
    }
    macro_rules! hash {
        () => {{
            if byte!() != 3 {
                return Err(E::InvalidChunk);
            }
            let mut id = [0; 32];
            for b in &mut id {
                *b = byte!();
            }
            DictionaryId(id)
        }};
    }
    let tag = if length == p as u64 { 0 } else { byte!() };
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
    let mut references = [None; 255];
    let mut count = 0;
    if (1..=8).contains(&tag) {
        codec = Some(match byte!() {
            0 => Codec::Uncompressed,
            1 => Codec::KeepDecoder,
            2 => Codec::Brotli,
            3 => Codec::SharedBrotli,
            _ => return Err(E::InvalidChunk),
        });
        if codec != Some(Codec::Uncompressed) {
            declared_size = Some(var!());
        }
        if codec == Some(Codec::SharedBrotli) {
            count = byte!() as usize;
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
            }
        }
    }
    let mut flags = 0;
    let mut checksum = None;
    if (2..=5).contains(&tag) {
        flags = byte!();
        let mask = match tag {
            3 => 1,
            4 => 0,
            5 => 2,
            _ => 3,
        };
        if flags & !mask != 0 {
            return Err(E::InvalidChunk);
        }
        if flags & 2 != 0 {
            checksum = Some(hash!());
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
    }))
}

type FieldIndex = Vec<([u8; 2], ::core::ops::Range<usize>)>;

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
    #[test]
    fn varints_accept_nonminimal_but_reject_ninth_continuation_byte() {
        let mut cursor = 0;
        assert_eq!(number(&[0x80, 0], &mut cursor).unwrap(), Some(0));
        assert_eq!(cursor, 2);
        assert!(number(&[0x80; 9], &mut 0).is_err());
        assert_eq!(number(&[0xff; 8], &mut 0).unwrap(), None);
    }
    #[test]
    fn header_cannot_read_fields_outside_its_declared_extent() {
        let budget = |_| Ok(());
        assert!(header(&[1, 2], 5, budget).is_err());
        assert!(header(&[4, 2, 0, 4], 5, budget).is_err());
        assert!(header(&[4, 2, 9], 5, budget).is_err());
        assert!(header(&[0], 5, budget).unwrap().is_some());
    }
    #[test]
    fn metadata_index_preserves_duplicate_custom_values() {
        let fields = metadata(b"AA\x01xAA\x01y", MetadataKind::Global).unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[1].1, 7..8);
        assert!(metadata(b"id\x00", MetadataKind::Global).is_err());
    }
}

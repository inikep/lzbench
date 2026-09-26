//! Resolves the built-in word address before applying its borrowed transform.

use super::super::{DecodeError, InvalidDataKind};
use crate::dictionary::DictionaryRef;
use crate::shared::dictionary::{
    BUILTIN_OFFSETS_BY_LENGTH, BUILTIN_SIZE_BITS_BY_LENGTH, BUILTIN_WORDS,
    transform::{SCRATCH_BYTES, Transform},
};

pub(super) fn resolve(
    dictionary: Option<DictionaryRef<'_>>,
    address: u64,
    length: usize,
    context: usize,
    scratch: &mut [u8; SCRATCH_BYTES],
) -> Result<usize, DecodeError> {
    #[cfg(feature = "experimental")]
    if let Some(result) =
        dictionary.and_then(|dictionary| dictionary.custom_word(address, length, context, scratch))
    {
        return result;
    }
    #[cfg(not(feature = "experimental"))]
    let _ = (dictionary, context);
    if !(4..=24).contains(&length) {
        return Err(InvalidDataKind::DictionaryReference.into());
    }
    let shift = BUILTIN_SIZE_BITS_BY_LENGTH[length];
    let transform = usize::try_from(address >> shift)
        .ok()
        .and_then(Transform::builtin)
        .ok_or(DecodeError::InvalidData {
            kind: InvalidDataKind::DictionaryReference,
        })?;
    let index = (address & ((1 << shift) - 1)) as usize;
    let offset = BUILTIN_OFFSETS_BY_LENGTH[length] as usize + index * length;
    Ok(transform.apply(&BUILTIN_WORDS[offset..offset + length], scratch))
}

//! Primitives shared by every RFC 9841 feature.
//!
//! [RFC 9841] adds three separable things to RFC 7932: Large Window Brotli,
//! shared dictionaries, and a framing container. The first is implemented; the
//! second exists as far as the context, its indexes and its search, plus the
//! serialized dictionary format behind the `experimental` feature; the third is
//! written behind that same feature. What they all sit on
//! lives here, so the dictionary format and the container can never disagree
//! with each other about how a value is encoded.
//!
//! - [`window`] resolves a stream's window and writes its header.
//! - [`varint`] is the base-128 integer both the dictionary and the container
//!   are built out of (behind the `experimental` feature).
//! - [`words`] and [`transform`] are the two halves of a custom static
//!   dictionary (behind the `experimental` feature).
//! - [`serialized`] parses and writes the serialized dictionary stream
//!   (behind the `experimental` feature).
//! - [`prefix`] turns the attached dictionaries into one logical byte
//!   sequence and maps backward distances onto it.
//! - [`prepared`] is the hash index built over one attachment.
//! - [`context`] owns the two together and searches them.
//! - [`search`] is the match finders' view of an attached context.
//!
//! [RFC 9841]: https://www.rfc-editor.org/rfc/rfc9841.html

pub(crate) mod context;
pub(crate) mod prefix;
pub(crate) mod prepared;
pub(crate) mod search;
#[cfg(feature = "experimental")]
pub(crate) mod serialized;
#[cfg(feature = "experimental")]
pub(crate) mod static_index;
#[cfg(feature = "experimental")]
pub(crate) mod transform;
#[cfg(feature = "experimental")]
pub(crate) mod varint;
pub(crate) mod window;
#[cfg(feature = "experimental")]
pub(crate) mod words;

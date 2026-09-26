// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(
    missing_docs,
    reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
)]
use crate::{Simd, SimdBase, SimdFloat, SimdInt, seal::Seal};
use core::error::Error;
use core::fmt::{Binary, Debug, Display, LowerExp, UpperExp};
use core::iter::{Product, Sum};
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use core::str::FromStr;

/// A value that carries a SIMD token.
///
/// Implemented by all SIMD tokens, vectors, and masks, and by references to these
/// types. User-defined wrappers can implement this trait by returning the token
/// of a contained value.
///
/// The [`#[simd]`](https://docs.rs/fearless_simd_macros/latest/fearless_simd_macros/attr.simd.html)
/// attribute extracts a token from its first non-receiver parameter using this
/// trait.
///
/// ```
/// use fearless_simd::{ExtractToken, Simd, f32x4};
///
/// // Four consecutive audio samples, processed together.
/// struct AudioSamples<S: Simd>(f32x4<S>);
///
/// impl<S: Simd> ExtractToken for AudioSamples<S> {
///     type S = S;
///
///     #[inline]
///     fn token(&self) -> S {
///         self.0.token()
///     }
/// }
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not carry a SIMD token",
    note = "If you are using #[simd], the first non-receiver parameter must carry a SIMD token. See the #[simd] documentation for more information."
)]
pub trait ExtractToken {
    /// The SIMD implementation associated with this value.
    type S: Simd;

    /// Get the SIMD token associated with this value.
    fn token(&self) -> Self::S;
}

impl<T: ExtractToken + ?Sized> ExtractToken for &T {
    type S = T::S;

    #[inline]
    fn token(&self) -> Self::S {
        T::token(*self)
    }
}

impl<T: ExtractToken + ?Sized> ExtractToken for &mut T {
    type S = T::S;

    #[inline]
    fn token(&self) -> Self::S {
        T::token(&**self)
    }
}

/// Element-wise selection between two SIMD vectors using `self`.
pub trait Select<T: Seal>: Seal {
    /// For each logical lane of this mask, select the first operand if the lane is true, and select the second
    /// operand if the lane is false.
    ///
    /// Masks may be converted to and from signed integer lane arrays for compatibility with older APIs. For those
    /// conversions, false is encoded as all zeroes (integer value 0) and true is encoded as all ones (integer value -1).
    /// If a mask is constructed from any other integer bit pattern, the result of this operation is unspecified.
    fn select(self, if_true: T, if_false: T) -> T;
}

/// Conversion of SIMD vectors to and from same-width vectors of `u8` lanes.
///
/// [`Bytes::bitcast`] uses this byte representation to reinterpret any two
/// non-mask SIMD vectors with the same total width and SIMD token. This is a
/// bitwise reinterpretation: it does not perform numeric conversion.
pub trait Bytes: Sized + Seal {
    /// The same-width SIMD vector of `u8` lanes used as the byte representation.
    ///
    /// This type is its own byte representation.
    type Bytes: Bytes<Bytes = Self::Bytes>;

    /// Reinterpret this vector as a same-width vector of `u8` lanes.
    fn to_bytes(self) -> Self::Bytes;

    /// Reinterpret a same-width vector of `u8` lanes as this vector type.
    fn from_bytes(value: Self::Bytes) -> Self;

    #[doc(alias = "reinterpret")]
    #[doc(alias = "transmute")]
    #[doc(alias = "to_bits")]
    #[doc(alias = "from_bits")]
    /// Bitcast directly to another SIMD vector with the same byte representation.
    /// This is effectively a safe [transmute](core::mem::transmute) for SIMD types.
    ///
    /// This works in code generic over a [`Simd`] implementation,
    /// including between native-width vectors with different lane types:
    ///
    /// ```
    /// # use fearless_simd::prelude::*;
    /// fn i8s_as_f64s<S: Simd>(value: S::i8s) -> S::f64s {
    ///     value.bitcast()
    /// }
    /// ```
    #[inline(always)]
    fn bitcast<U: Bytes<Bytes = Self::Bytes>>(self) -> U {
        U::from_bytes(self.to_bytes())
    }
}

pub(crate) mod seal {
    #[expect(
        unnameable_types,
        reason = "This is a sealed trait, so being unnameable is the entire point"
    )]
    pub trait Seal {}
}

impl Seal for f32 {}
impl Seal for f64 {}
impl Seal for u8 {}
impl Seal for i8 {}
impl Seal for u16 {}
impl Seal for i16 {}
impl Seal for u32 {}
impl Seal for i32 {}
impl Seal for u64 {}
impl Seal for i64 {}

/// Value conversion, adding a SIMD blessing.
///
/// Analogous to [`From`], but takes a SIMD token, which is used to bless
/// the new value. Most such conversions are safe transmutes, but this
/// trait also supports splats, and implementations can use the SIMD token
/// to use an efficient splat intrinsic.
///
/// The [`SimdInto`] trait is also provided for convenience.
pub trait SimdFrom<T, S: Simd> {
    fn simd_from(simd: S, value: T) -> Self;
}

/// Value conversion, adding a SIMD blessing.
///
/// This trait is syntactic sugar for [`SimdFrom`] and exists only to allow
/// `impl SimdInto` syntax in signatures, which would otherwise require
/// cumbersome `where` clauses in terms of `SimdFrom`.
///
/// Avoid implementing this trait directly, prefer implementing [`SimdFrom`].
pub trait SimdInto<T, S> {
    fn simd_into(self, simd: S) -> T;
}

impl<F, T: SimdFrom<F, S>, S: Simd> SimdInto<T, S> for F {
    fn simd_into(self, simd: S) -> T {
        SimdFrom::simd_from(simd, self)
    }
}

impl<T, S: Simd> SimdFrom<T, S> for T {
    fn simd_from(_simd: S, value: T) -> Self {
        value
    }
}

/// Types that can be used as elements in SIMD vectors.
pub trait SimdElement:
    Copy
    + Clone
    + Seal
    + Default
    + Debug
    + Display
    + FromStr
    + LowerExp
    + UpperExp
    + PartialOrd
    + PartialEq
    + From<bool>
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Div<Self, Output = Self>
    + DivAssign<Self>
    + Rem<Self, Output = Self>
    + RemAssign<Self>
    + Sum<Self>
    + Product<Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> Div<&'a Self, Output = Self>
    + for<'a> DivAssign<&'a Self>
    + for<'a> Rem<&'a Self, Output = Self>
    + for<'a> RemAssign<&'a Self>
    + for<'a> Sum<&'a Self>
    + for<'a> Product<&'a Self>
{
    /// The associated mask lane type. This will be a signed integer of the same size as this type.
    type Mask: SimdElement<Mask = Self::Mask>;

    /// The size of an element in bits.
    const BITS: usize = size_of::<Self>() * u8::BITS as usize;
}

impl SimdElement for f32 {
    type Mask = i32;
}

impl SimdElement for f64 {
    type Mask = i64;
}

impl SimdElement for u8 {
    type Mask = i8;
}

impl SimdElement for i8 {
    type Mask = Self;
}

impl SimdElement for u16 {
    type Mask = i16;
}

impl SimdElement for i16 {
    type Mask = Self;
}

impl SimdElement for u32 {
    type Mask = i32;
}

impl SimdElement for i32 {
    type Mask = Self;
}

impl SimdElement for u64 {
    type Mask = i64;
}

impl SimdElement for i64 {
    type Mask = Self;
}

/// Types that can be used as elements in integer SIMD vectors.
///
/// [`Self::Native`] selects the native-width vector for this scalar type and a
/// SIMD backend, so an integer-generic caller can use [`crate::dispatch!`]:
///
/// ```
/// use fearless_simd::{dispatch, prelude::*, Level};
///
/// #[inline(always)]
/// fn count_ones_simd<S: Simd, T: SimdIntElement>(simd: S, value: T) -> T {
///     T::Native::<S>::splat(simd, value).count_ones()[0]
/// }
///
/// fn count_ones<T: SimdIntElement>(level: Level, value: T) -> T {
///     dispatch!(level, simd => count_ones_simd(simd, value))
/// }
///
/// let level = Level::new();
/// assert_eq!(count_ones(level, 7_u8), 3);
/// assert_eq!(count_ones(level, 7_i64), 3);
/// ```
pub trait SimdIntElement:
    SimdElement
    + Eq
    + Ord
    + Binary
    + Not<Output = Self>
    + Shl<usize, Output = Self>
    + ShlAssign<usize>
    + Shr<usize, Output = Self>
    + ShrAssign<usize>
    + BitAnd<Self, Output = Self>
    + BitAndAssign<Self>
    + BitOr<Self, Output = Self>
    + BitOrAssign<Self>
    + BitXor<Self, Output = Self>
    + BitXorAssign<Self>
    + TryFrom<u8, Error: Copy + Error + Eq>
    + TryFrom<u16, Error: Copy + Error + Eq>
    + TryFrom<u32, Error: Copy + Error + Eq>
    + TryFrom<u64, Error: Copy + Error + Eq>
    + TryFrom<u128, Error: Copy + Error + Eq>
    + TryFrom<usize, Error: Copy + Error + Eq>
    + TryFrom<i8, Error: Copy + Error + Eq>
    + TryFrom<i16, Error: Copy + Error + Eq>
    + TryFrom<i32, Error: Copy + Error + Eq>
    + TryFrom<i64, Error: Copy + Error + Eq>
    + TryFrom<i128, Error: Copy + Error + Eq>
    + TryFrom<isize, Error: Copy + Error + Eq>
    + for<'a> Shl<&'a usize, Output = Self>
    + for<'a> ShlAssign<&'a usize>
    + for<'a> Shr<&'a usize, Output = Self>
    + for<'a> ShrAssign<&'a usize>
    + for<'a> BitAnd<&'a Self, Output = Self>
    + for<'a> BitAndAssign<&'a Self>
    + for<'a> BitOr<&'a Self, Output = Self>
    + for<'a> BitOrAssign<&'a Self>
    + for<'a> BitXor<&'a Self, Output = Self>
    + for<'a> BitXorAssign<&'a Self>
{
    /// The native-width integer vector for this scalar type and backend `S`.
    ///
    /// Its lane count depends on both the scalar type and the backend, and is
    /// available as `T::Native::<S>::LEN` with [`SimdBase`] in scope.
    type Native<S: Simd>: SimdInt<S, Element = Self>;
}

impl SimdIntElement for u8 {
    type Native<S: Simd> = S::u8s;
}
impl SimdIntElement for u16 {
    type Native<S: Simd> = S::u16s;
}
impl SimdIntElement for u32 {
    type Native<S: Simd> = S::u32s;
}
impl SimdIntElement for u64 {
    type Native<S: Simd> = S::u64s;
}
impl SimdIntElement for i8 {
    type Native<S: Simd> = S::i8s;
}
impl SimdIntElement for i16 {
    type Native<S: Simd> = S::i16s;
}
impl SimdIntElement for i32 {
    type Native<S: Simd> = S::i32s;
}
impl SimdIntElement for i64 {
    type Native<S: Simd> = S::i64s;
}

/// Types that can be used as elements in float SIMD vectors.
///
/// The scalar conversion bounds are limited to types that every floating-point
/// element can represent losslessly, including f16 for forward-compatibility.
///
/// [`Self::Native`] selects the native-width vector for this scalar type and a
/// SIMD backend, so a float-generic caller can use [`crate::dispatch!`]:
///
/// ```
/// use fearless_simd::{dispatch, prelude::*, Level};
///
/// #[inline(always)]
/// fn sqrt_simd<S: Simd, T: SimdFloatElement>(simd: S, value: T) -> T {
///     T::Native::<S>::splat(simd, value).sqrt()[0]
/// }
///
/// fn sqrt<T: SimdFloatElement>(level: Level, value: T) -> T {
///     dispatch!(level, simd => sqrt_simd(simd, value))
/// }
///
/// let level = Level::new();
/// assert_eq!(sqrt(level, 4.0_f32), 2.0);
/// assert_eq!(sqrt(level, 4.0_f64), 2.0);
/// ```
pub trait SimdFloatElement: SimdElement + Neg<Output = Self> + From<i8> + From<u8> {
    /// The native-width floating-point vector for this scalar type and backend `S`.
    ///
    /// Its lane count depends on both the scalar type and the backend, and is
    /// available as `T::Native::<S>::LEN` with [`SimdBase`] in scope.
    type Native<S: Simd>: SimdFloat<S, Element = Self>;
}

impl SimdFloatElement for f32 {
    type Native<S: Simd> = S::f32s;
}
impl SimdFloatElement for f64 {
    type Native<S: Simd> = S::f64s;
}

/// Construction of integer vectors from floats by truncation
pub trait SimdCvtTruncate<T: Seal>: Seal {
    fn truncate_from(x: T) -> Self;
    fn truncate_from_precise(x: T) -> Self;
}

/// Construction of floating point vectors from integers
pub trait SimdCvtFloat<T: Seal>: Seal {
    fn float_from(x: T) -> Self;
}

/// Interleaved loads and stores for SIMD vectors.
///
/// This trait is currently implemented only for numeric 128-bit vector types.
/// These operations load/store up to 512 bits in one go, and loading into wider vectors
/// usually degrades performance by causing more register pressure.
///
/// If processing in wider vectors is desirable, combine the 128-bit vectors into larger ones
/// and process them together as a single vector. This avoids issues with register pressure.
///
/// # Example
///
/// This generic color transform swaps the red and blue channels of RGBA pixels:
///
/// ```
/// use fearless_simd::prelude::*;
///
/// fn swap_red_blue<S: Simd, V: SimdInterleaved<S>>(
///     simd: S,
///     pixels: &mut [V::Element],
/// ) {
///     let mut chunks = pixels.chunks_exact_mut(V::LEN * 4);
///     for chunk in &mut chunks {
///         let [red, green, blue, alpha] = V::load_four_interleaved(simd, chunk);
///         V::store_four_interleaved([blue, green, red, alpha], chunk);
///     }
///
///     for pixel in chunks.into_remainder().chunks_exact_mut(4) {
///         pixel.swap(0, 2);
///     }
/// }
/// ```
pub trait SimdInterleaved<S: Simd>: SimdBase<S> {
    /// Load four 128-bit vectors from a slice with 4-way interleaving.
    ///
    /// This is useful e.g. in image processing to turn interleaved RGBA pixels into vectors of each color component.
    ///
    /// For example, with 32-bit lanes, memory laid out as
    /// `[r0, g0, b0, a0, r1, g1, b1, a1, r2, g2, b2, a2, r3, g3, b3, a3]` loads as
    /// `[[r0, r1, r2, r3], [g0, g1, g2, g3], [b0, b1, b2, b3], [a0, a1, a2, a3]]`.
    ///
    /// # Panics
    ///
    /// Panics unless `src.len()` is exactly `Self::LEN * 4`.
    fn load_four_interleaved(simd: S, src: &[Self::Element]) -> [Self; 4];

    /// Store four vectors into a scalar slice with four-way interleaving.
    ///
    /// This is the inverse of [`load_four_interleaved`](Self::load_four_interleaved).
    ///
    /// This is useful e.g. in image processing to turn vectors of each color component into interleaved RGBA pixels.
    /// For example, with 32-bit lanes, vectors containing
    /// `[[r0, r1, r2, r3], [g0, g1, g2, g3], [b0, b1, b2, b3], [a0, a1, a2, a3]]` get stored as
    /// `[r0, g0, b0, a0, r1, g1, b1, a1, r2, g2, b2, a2, r3, g3, b3, a3]`.
    ///
    /// # Panics
    ///
    /// Panics unless `dest.len()` is exactly `Self::LEN * 4`.
    fn store_four_interleaved(vectors: [Self; 4], dest: &mut [Self::Element]);
}

/// Concatenation of two SIMD vectors.
///
/// This is implemented on all vectors 256 bits and lower, producing vectors of up to 512 bits.
pub trait SimdCombine<S: Simd>: SimdBase<S> + Seal {
    type Combined: SimdBase<S, Element = Self::Element, Block = Self::Block>
        + SimdSplit<S, Split = Self>;

    /// Concatenate two vectors into a new one that's twice as long.
    fn combine(self, rhs: impl SimdInto<Self, S>) -> Self::Combined;
}

/// Splitting of one SIMD vector into two.
///
/// This is implemented on all vectors 256 bits and higher, producing vectors of down to 128 bits.
pub trait SimdSplit<S: Simd>: SimdBase<S> + Seal {
    type Split: SimdBase<S, Element = Self::Element, Block = Self::Block>
        + SimdCombine<S, Combined = Self>;

    /// Split this vector into left and right halves.
    fn split(self) -> (Self::Split, Self::Split);
}

/// Widening conversion of a numeric SIMD vector.
///
/// Integer lanes are sign-extended or zero-extended according to their type. Finite floating-point
/// lanes are converted losslessly from `f32` to `f64`; infinities and NaNs remain infinities and
/// NaNs. The result is returned as two vectors with the same bit width as the input: the first
/// contains the widened lower lanes and the second contains the widened upper lanes.
///
/// ```
/// use fearless_simd::{f32x4, f64x2, prelude::*, u8x16, u16x8};
///
/// fn fixed<S: Simd>(value: u8x16<S>) -> (u16x8<S>, u16x8<S>) {
///     value.widen()
/// }
///
/// fn native<S: Simd>(value: S::u8s) -> (S::u16s, S::u16s) {
///     value.widen()
/// }
///
/// fn fixed_float<S: Simd>(value: f32x4<S>) -> (f64x2<S>, f64x2<S>) {
///     value.widen()
/// }
///
/// fn native_float<S: Simd>(value: S::f32s) -> (S::f64s, S::f64s) {
///     value.widen()
/// }
/// ```
pub trait SimdWiden<S: Simd>: SimdBase<S> + Seal {
    /// The same-width vector type with lanes twice as wide.
    type Widened: SimdNarrow<S, Narrowed = Self>;

    /// Widen every lane, returning the lower and upper halves in that order.
    fn widen(self) -> (Self::Widened, Self::Widened);
}

/// Narrowing conversion of two numeric SIMD vectors.
///
/// Both inputs have the same bit width as the result. The first input supplies the lower result
/// lanes and the second input supplies the upper result lanes. Integer narrowing either retains
/// the low destination-width bits or saturates, depending on the method. Floating-point narrowing
/// converts `f64` to `f32` using Rust's `as` semantics; for floats,
/// [`saturating_narrow`](Self::saturating_narrow) and
/// [`relaxed_narrow`](Self::relaxed_narrow) are identical to [`narrow`](Self::narrow).
///
/// ```
/// use fearless_simd::{f32x4, f64x2, prelude::*, i16x8, i8x16};
///
/// fn fixed<S: Simd>(low: i16x8<S>, high: i16x8<S>) -> i8x16<S> {
///     low.narrow(high)
/// }
///
/// fn native<S: Simd>(low: S::i16s, high: S::i16s) -> S::i8s {
///     low.saturating_narrow(high)
/// }
///
/// fn fixed_float<S: Simd>(low: f64x2<S>, high: f64x2<S>) -> f32x4<S> {
///     low.narrow(high)
/// }
///
/// fn native_float<S: Simd>(low: S::f64s, high: S::f64s) -> S::f32s {
///     low.saturating_narrow(high)
/// }
/// ```
pub trait SimdNarrow<S: Simd>: SimdBase<S> + Seal {
    /// The same-width vector type with lanes half as wide.
    type Narrowed: SimdWiden<S, Widened = Self>;

    /// Narrow every lane.
    ///
    /// This conversion behaves identically to the `as` operator:
    ///  - Integers are truncated.
    ///  - Floating-point values follow IEEE 754 narrowing behavior in round-to-even mode:
    ///    they are rounded to the nearest representable `f32`, with ties resolved to even; overflow produces signed infinity.
    ///
    /// # Example
    ///
    /// ```
    /// use fearless_simd::{dispatch, Level, i64x2, i32x4, prelude::*};
    ///
    /// let level = Level::new();
    /// dispatch!(level, simd => {
    ///     let low = i64x2::simd_from(simd, [1, -1]);
    ///     let high = i64x2::simd_from(simd, [i64::MAX - 5, i64::MIN + 5]);
    ///     let narrowed: i32x4<_> = low.narrow(high);
    ///     assert_eq!(*narrowed, [1, -1, -6, 5]);
    /// });
    /// ```
    fn narrow(self, high: Self) -> Self::Narrowed;

    /// Narrow with saturation for integers. Floats behave identically to [`narrow`](Self::narrow).
    ///
    /// Integer values that overflow the narrowed type become the closest representable value for the narrowed type.
    /// For example, `1234u16` becomes `u8::MAX` after narrowing, and `-1234i16` becomes `i8::MIN`.
    ///
    /// # Example
    ///
    /// ```
    /// use fearless_simd::{dispatch, Level, i64x2, i32x4, prelude::*};
    ///
    /// let level = Level::new();
    /// dispatch!(level, simd => {
    ///     let low = i64x2::simd_from(simd, [1, -1]);
    ///     let high = i64x2::simd_from(simd, [i64::MAX - 5, i64::MIN + 5]);
    ///     let narrowed: i32x4<_> = low.saturating_narrow(high);
    ///     assert_eq!(*narrowed, [1, -1, i32::MAX, i32::MIN]);
    /// });
    fn saturating_narrow(self, high: Self) -> Self::Narrowed;

    /// Narrow using the cheapest operation for the active SIMD backend, assuming no overflow.
    ///
    /// This is useful when you're sure the result fits into the destination type,
    /// so the distinction between [`narrow`](Self::narrow) and [`saturating_narrow`](Self::saturating_narrow)
    /// doesn't matter.
    ///
    /// This method will panic in debug mode if any of the inputs do not fit into the narrower type.
    /// This operation remains memory-safe and never causes undefined behavior,
    /// but will produce arbitrary values on overflow in release mode.
    ///
    /// Floats behave identically to [`narrow`](Self::narrow), with no additional precondition.
    ///
    /// # Example
    ///
    /// ```
    /// use fearless_simd::{dispatch, Level, i64x2, i32x4, prelude::*};
    ///
    /// let level = Level::new();
    /// dispatch!(level, simd => {
    ///     let low = i64x2::simd_from(simd, [1, -1]);
    ///     let high = i64x2::simd_from(simd, [5, -5]);
    ///     let narrowed: i32x4<_> = low.relaxed_narrow(high);
    ///     assert_eq!(*narrowed, [1, -1, 5, -5]);
    /// });
    fn relaxed_narrow(self, high: Self) -> Self::Narrowed;
}

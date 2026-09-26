//! Host-validated tokens are selected once and retained with the encoder.
//!
//! Dynamic calls stop at this boundary. Each selected implementation enters a
//! feature-enabled function once and passes its concrete token to inner loops.

use alloc::boxed::Box;
use alloc::vec::Vec;

use fearless_simd::{Level, Simd, dispatch};

use super::fast::{FastCore, encode_fragment};
use super::greedy::backward_references::{ReferenceState, create_backward_references};
use super::greedy::hashers::{MatchFinder, with_matcher};
use super::greedy::params::GreedyParams;
use super::hq::block_splitter::{BlockCosts, assign_blocks};
use super::hq::h10::BinaryTreeMatcher;
use super::hq::params::{HqParams, HqQuality};
use super::hq::zopfli::{
    ZopfliState, ZopfliWorkspace, create_hq_zopfli_backward_references,
    create_zopfli_backward_references,
};
use super::rfc9841::context::SharedContextInner;
use crate::shared::bits::BitWriter;
use crate::shared::command::{Command, CommandExtension, extend_last_command};
use crate::shared::ringbuffer::{BlockSpan, Window};

/// Borrowed greedy state for one monomorphized scan.
pub(crate) struct GreedyInput<'a> {
    pub(crate) matcher: &'a mut MatchFinder,
    pub(crate) params: &'a GreedyParams,
    pub(crate) window: Window<'a>,
    pub(crate) span: BlockSpan,
    pub(crate) attached: Option<&'a SharedContextInner>,
    pub(crate) references: &'a mut ReferenceState,
    pub(crate) commands: &'a mut Vec<Command>,
}

/// Borrowed high-quality state for one monomorphized search.
pub(crate) struct HqInput<'a> {
    pub(crate) matcher: &'a mut BinaryTreeMatcher,
    pub(crate) params: &'a HqParams,
    pub(crate) window: Window<'a>,
    pub(crate) span: BlockSpan,
    pub(crate) attached: Option<&'a SharedContextInner>,
    pub(crate) references: &'a mut ZopfliState,
    pub(crate) workspace: &'a mut ZopfliWorkspace,
    pub(crate) commands: &'a mut Vec<Command>,
}

/// Type-erased outer boundary; no feature detection is performed by its calls.
pub(crate) trait Kernels: Send + Sync {
    fn fast(
        &self,
        core: &mut FastCore,
        input: &[u8],
        is_last: bool,
        table: &mut [i32],
        writer: &mut BitWriter<'_>,
    );
    fn extend(&self, input: CommandExtension<'_>);
    fn fast_append(
        &self,
        core: &mut FastCore,
        input: &[u8],
        is_last: bool,
        table: &mut [i32],
        writer: &mut BitWriter<'_, Vec<u8>>,
    );
    fn greedy(&self, input: GreedyInput<'_>);
    fn hq(&self, input: HqInput<'_>);
    fn assign_blocks(&self, input: BlockCosts<'_>);
    fn stitch(
        &self,
        matcher: &mut BinaryTreeMatcher,
        input_size: usize,
        position: usize,
        window: Window<'_>,
    );
}

/// The boxed proof tokens are zero-sized for all currently supported backends.
///
/// `S` is the level the fragment, copy-extension and high-quality kernels
/// run at; `G` the level the greedy loops run at, see [`boxed`].
struct Selected<S, G, const INDEPENDENT: bool> {
    simd: S,
    greedy: G,
}

/// Boxes the kernels for the token `simd`, choosing the greedy loops' level.
///
/// The greedy loops are scalar but for one byte-equality mask per search,
/// which SSE2 already provides. Compiling them for every x86 level would
/// multiply their code — a quarter of the crate — without a faster
/// instruction to show for it, so every x86 token hands them SSE2. The
/// scalar backend keeps them scalar, as the unfiltered oracle; other
/// architectures keep their own level.
fn boxed<S: Simd, const INDEPENDENT: bool>(simd: S) -> Box<dyn Kernels> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if let Some(sse2) = simd.level().as_sse2() {
            return Box::new(Selected::<S, fearless_simd::Sse2, INDEPENDENT> {
                simd,
                greedy: sse2,
            });
        }
    }
    Box::new(Selected::<S, S, INDEPENDENT> { simd, greedy: simd })
}

/// Resolves the backend once, when a retained encoder is constructed.
pub(crate) fn select(level: Level) -> Box<dyn Kernels> {
    dispatch!(level, simd => boxed::<_, false>(simd))
}

/// Selects isolated fragment kernels once per worker allocation.
#[cfg(any(test, not(feature = "no_std")))]
pub(crate) fn select_independent(level: Level) -> Box<dyn Kernels> {
    dispatch!(level, simd => boxed::<_, true>(simd))
}

impl<S: Simd, G: Simd, const INDEPENDENT: bool> Kernels for Selected<S, G, INDEPENDENT> {
    fn fast(
        &self,
        core: &mut FastCore,
        input: &[u8],
        is_last: bool,
        table: &mut [i32],
        writer: &mut BitWriter<'_>,
    ) {
        self.simd.vectorize(
            #[inline(always)]
            || encode_fragment::<_, INDEPENDENT>(self.simd, core, input, is_last, table, writer),
        );
    }

    fn extend(&self, input: CommandExtension<'_>) {
        self.simd.vectorize(
            #[inline(always)]
            || extend_last_command(self.simd, input),
        );
    }

    fn fast_append(
        &self,
        core: &mut FastCore,
        input: &[u8],
        is_last: bool,
        table: &mut [i32],
        writer: &mut BitWriter<'_, Vec<u8>>,
    ) {
        self.simd.vectorize(
            #[inline(always)]
            || encode_fragment::<_, INDEPENDENT>(self.simd, core, input, is_last, table, writer),
        );
    }

    fn greedy(&self, input: GreedyInput<'_>) {
        let GreedyInput {
            matcher,
            params,
            window,
            span,
            attached,
            references,
            commands,
        } = input;
        self.greedy.vectorize(
            #[inline(always)]
            || match attached {
                None => with_matcher!(matcher, |finder| create_backward_references::<
                    _,
                    _,
                    false,
                    INDEPENDENT,
                >(
                    self.greedy,
                    finder,
                    params,
                    window,
                    span,
                    None,
                    references,
                    commands
                )),
                Some(_) => {
                    with_matcher!(matcher, |finder| create_backward_references::<
                        _,
                        _,
                        true,
                        INDEPENDENT,
                    >(
                        self.greedy,
                        finder,
                        params,
                        window,
                        span,
                        attached,
                        references,
                        commands
                    ))
                }
            },
        );
    }

    fn assign_blocks(&self, input: BlockCosts<'_>) {
        assign_blocks(self.simd, input);
    }

    fn hq(&self, input: HqInput<'_>) {
        let HqInput {
            matcher,
            params,
            window,
            span,
            attached,
            references,
            workspace,
            commands,
        } = input;
        self.simd.vectorize(
            #[inline(always)]
            || match params.quality {
                HqQuality::Q10 => create_zopfli_backward_references::<_, INDEPENDENT>(
                    self.simd,
                    span.bytes as usize,
                    span.position as usize,
                    window.data,
                    window.mask,
                    params,
                    attached,
                    matcher,
                    workspace,
                    references,
                    commands,
                ),
                HqQuality::Q11 => create_hq_zopfli_backward_references::<_, INDEPENDENT>(
                    self.simd,
                    span.bytes as usize,
                    span.position as usize,
                    window.data,
                    window.mask,
                    params,
                    attached,
                    matcher,
                    workspace,
                    references,
                    commands,
                ),
            },
        );
    }

    fn stitch(
        &self,
        matcher: &mut BinaryTreeMatcher,
        input_size: usize,
        position: usize,
        window: Window<'_>,
    ) {
        self.simd.vectorize(
            #[inline(always)]
            || {
                matcher.stitch_to_previous_block(
                    self.simd,
                    input_size,
                    position,
                    window.data,
                    window.mask,
                )
            },
        );
    }
}

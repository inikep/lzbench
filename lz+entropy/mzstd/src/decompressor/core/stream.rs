//! Frame sequencing: the resumable streaming state machine and the
//! one-shot driver over complete input.
//!
//! Both walk the same grammar: frames back to back, each either a
//! skippable frame or a Zstandard frame of blocks and an optional checksum.
//! The streaming [`Stream`] accepts input in any pieces: it collects each
//! header and each whole block content in a scratch buffer before acting on
//! it, regenerates blocks into its [`History`], and delivers output from
//! there in any pieces. [`Stream::decode_all`] parses straight from a
//! complete slice and regenerates into the caller's vector, which serves as
//! the history.
//!
//! Every entry point takes the operation's dictionary, if any, as a
//! [`DictView`]: the decoder keeps no reference to it between calls, and
//! each frame starts from it.

use ::core::mem::MaybeUninit;
use alloc::vec::Vec;

use borrowed_buf::BorrowedBuf;

use super::block::{BlockHeader, BlockKind, BlockOutput};
use super::copy::WILDCOPY_OVERLENGTH;
use super::dictionary::DictView;
use super::frame::{FrameDecoder, FrameHeader, MAGIC_SIZE, header_size};
use super::history::{History, Sink};
use crate::Backend;
use crate::decompressor::{
    DecodeError, DecodeFailure, DecodeOperation, DecodeProgress, DecoderConfig, DecoderStatus,
    FrameMode, InvalidDataKind,
};
use crate::shared::format::{
    BLOCK_HEADER_SIZE, CHECKSUM_SIZE, MAGIC_NUMBER, MAGIC_SKIPPABLE_MASK, MAGIC_SKIPPABLE_START,
    SKIPPABLE_HEADER_SIZE,
};

/// Size of a skippable frame's size field.
const SKIPPABLE_SIZE_FIELD: usize = SKIPPABLE_HEADER_SIZE - MAGIC_SIZE;
/// Largest skippable frame payload: the reference rejects sizes whose
/// header-inclusive total overflows 32 bits.
const SKIPPABLE_PAYLOAD_MAX: u32 = u32::MAX - SKIPPABLE_HEADER_SIZE as u32;

/// What the streaming decoder expects next.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State {
    /// A frame magic number, or the end of the input between frames.
    Magic,
    /// A skippable frame's size field.
    SkippableSize,
    /// This many skippable payload bytes.
    Skipping(u64),
    /// A frame header, from its descriptor byte.
    FrameHeader,
    /// A block header.
    BlockHeader,
    /// The content of this block.
    BlockContent(BlockHeader),
    /// A frame's content checksum.
    Checksum,
    /// A single-frame operation is complete.
    Done,
}

/// Moves bytes from `input` into `scratch` until it holds `need`, returning
/// whether it does.
fn fill(scratch: &mut Vec<u8>, input: &mut &[u8], need: usize) -> bool {
    let take = need.saturating_sub(scratch.len()).min(input.len());
    let (head, tail) = input.split_at(take);
    scratch.extend_from_slice(head);
    *input = tail;
    scratch.len() >= need
}

/// The first `N` bytes of `bytes`, which the caller has filled.
fn array<const N: usize>(bytes: &[u8]) -> Result<[u8; N], DecodeError> {
    bytes
        .first_chunk::<N>()
        .copied()
        .ok_or(DecodeError::InvalidState)
}

/// Whether `magic` opens a skippable frame.
const fn is_skippable(magic: u32) -> bool {
    magic & MAGIC_SKIPPABLE_MASK == MAGIC_SKIPPABLE_START
}

/// A cursor over complete one-shot input, possibly cut short by the input
/// budget.
struct SliceReader<'a> {
    data: &'a [u8],
    position: usize,
    /// The budget, when it cut the input short.
    limit: Option<u64>,
}

impl<'a> SliceReader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], DecodeError> {
        let bytes = self
            .data
            .get(self.position..)
            .and_then(|rest| rest.get(..count))
            .ok_or(match self.limit {
                Some(limit) => DecodeError::InputLimitExceeded { limit },
                None => DecodeError::UnexpectedEndOfInput,
            })?;
        self.position += count;
        Ok(bytes)
    }

    fn take_array<const N: usize>(&mut self) -> Result<[u8; N], DecodeError> {
        array(self.take(N)?)
    }

    fn is_empty(&self) -> bool {
        self.position >= self.data.len()
    }
}

/// Streaming decoder state, owned by a `Decompressor` and reused between
/// operations.
#[derive(Debug)]
pub(crate) struct Stream {
    config: DecoderConfig,
    state: State,
    scratch: Vec<u8>,
    history: History,
    frame: FrameDecoder,
    total_in: u64,
    total_out: u64,
    decoded: u64,
    frames: u64,
    complete_frames: u64,
    final_end: Option<u64>,
    finished: bool,
    failed: bool,
    /// The SIMD level every kernel runs with.
    backend: Backend,
}

impl Stream {
    /// An idle decoder for `config` on `backend`; nothing is allocated yet.
    pub(crate) fn new(config: DecoderConfig, backend: Backend) -> Self {
        Self {
            config,
            state: State::Magic,
            scratch: Vec::new(),
            history: History::default(),
            frame: FrameDecoder::default(),
            total_in: 0,
            total_out: 0,
            decoded: 0,
            frames: 0,
            complete_frames: 0,
            final_end: None,
            finished: false,
            failed: false,
            backend,
        }
    }

    /// The SIMD level the decoder runs with.
    pub(crate) const fn backend(&self) -> Backend {
        self.backend
    }

    /// Starts a new operation under `config`, keeping allocations.
    pub(crate) fn reset(&mut self, config: DecoderConfig) {
        self.config = config;
        self.state = State::Magic;
        self.scratch.clear();
        self.history.reset();
        self.total_in = 0;
        self.total_out = 0;
        self.decoded = 0;
        self.frames = 0;
        self.complete_frames = 0;
        self.final_end = None;
        self.finished = false;
        self.failed = false;
    }

    /// Drops every buffer, keeping the operation's state and counters; only
    /// valid between operations or once one has finished or failed.
    pub(crate) fn release(&mut self) {
        self.scratch = Vec::new();
        self.history.release();
        self.frame.release();
    }

    /// Heap bytes held for reuse.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.scratch.capacity() + self.history.retained_bytes() + self.frame.retained_bytes()
    }

    /// Whether the operation completed.
    pub(crate) const fn is_finished(&self) -> bool {
        self.finished
    }

    /// Compressed bytes accepted so far.
    pub(crate) const fn total_in(&self) -> u64 {
        self.total_in
    }

    /// Decompressed bytes delivered so far.
    pub(crate) const fn total_out(&self) -> u64 {
        self.total_out
    }

    /// Zstandard frames completed so far.
    pub(crate) const fn frames_decoded(&self) -> u64 {
        self.frames
    }

    /// The window of the most recent frame header.
    pub(crate) fn window_size(&self) -> Option<u64> {
        self.frame.header().map(|header| header.window_size)
    }

    /// Runs one call of the incremental contract; every call of one
    /// operation passes the same `dictionary`.
    pub(crate) fn process(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        operation: DecodeOperation,
        dictionary: Option<DictView<'_>>,
    ) -> Result<DecodeProgress, DecodeFailure> {
        self.process_into(input, output, operation, dictionary)
    }

    /// [`Self::process`] into memory that may be uninitialized. Calls of
    /// the two may alternate on one operation; either writes exactly
    /// `output[..produced]`, also when it fails.
    pub(crate) fn process_uninit(
        &mut self,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
        operation: DecodeOperation,
        dictionary: Option<DictView<'_>>,
    ) -> Result<DecodeProgress, DecodeFailure> {
        self.process_into(input, output, operation, dictionary)
    }

    /// The body of [`Self::process`] and [`Self::process_uninit`], one
    /// instantiation per destination type.
    fn process_into<O: Sink + ?Sized>(
        &mut self,
        input: &[u8],
        output: &mut O,
        operation: DecodeOperation,
        dictionary: Option<DictView<'_>>,
    ) -> Result<DecodeProgress, DecodeFailure> {
        let failure = |error, consumed, produced| DecodeFailure {
            error,
            consumed,
            produced,
        };
        if self.failed {
            return Err(failure(DecodeError::InvalidState, 0, 0));
        }
        if self.finished {
            return Ok(DecodeProgress {
                consumed: 0,
                produced: 0,
                status: DecoderStatus::Finished,
            });
        }
        let finish = operation == DecodeOperation::Finish;
        let end = self.total_in.saturating_add(input.len() as u64);
        let consistent = match (finish, self.final_end) {
            (false, None) => true,
            (false, Some(_)) => false,
            (true, None) => {
                self.final_end = Some(end);
                true
            }
            (true, Some(expected)) => expected == end,
        };
        if !consistent {
            self.failed = true;
            return Err(failure(DecodeError::InvalidState, 0, 0));
        }
        let limit = self.config.limits().max_input_bytes();
        let (offered, limited) = match limit {
            Some(limit) => {
                let room = limit.saturating_sub(self.total_in);
                if input.len() as u64 > room {
                    (&input[..room as usize], true)
                } else {
                    (input, false)
                }
            }
            None => (input, false),
        };
        let mut remaining = offered;
        let mut produced = 0;
        let result = self.run(&mut remaining, output, &mut produced, finish, dictionary);
        let consumed = offered.len() - remaining.len();
        self.total_in += consumed as u64;
        self.total_out += produced as u64;
        let result = result.and_then(|status| match (status, limit) {
            (DecoderStatus::NeedsInput, Some(limit)) if limited => {
                Err(DecodeError::InputLimitExceeded { limit })
            }
            (DecoderStatus::NeedsInput, _) if finish => Err(DecodeError::UnexpectedEndOfInput),
            (status, _) => Ok(status),
        });
        match result {
            Ok(status) => Ok(DecodeProgress {
                consumed,
                produced,
                status,
            }),
            Err(error) => {
                self.failed = true;
                Err(failure(error, consumed, produced))
            }
        }
    }

    /// Advances the state machine until it needs input or output space, or
    /// the operation completes.
    fn run<O: Sink + ?Sized>(
        &mut self,
        input: &mut &[u8],
        output: &mut O,
        produced: &mut usize,
        finish: bool,
        dictionary: Option<DictView<'_>>,
    ) -> Result<DecoderStatus, DecodeError> {
        loop {
            if self.history.pending() > 0 {
                *produced += self.history.deliver(output, *produced);
                if self.history.pending() > 0 {
                    return Ok(DecoderStatus::NeedsOutput);
                }
            }
            match self.state {
                State::Done => {
                    self.finished = true;
                    return Ok(DecoderStatus::Finished);
                }
                State::Magic => {
                    if self.scratch.is_empty() && input.is_empty() {
                        if finish
                            && self.complete_frames > 0
                            && self.config.frame_mode() == FrameMode::Concatenated
                        {
                            self.finished = true;
                            return Ok(DecoderStatus::Finished);
                        }
                        return Ok(DecoderStatus::NeedsInput);
                    }
                    if !fill(&mut self.scratch, input, MAGIC_SIZE) {
                        return Ok(DecoderStatus::NeedsInput);
                    }
                    let magic = u32::from_le_bytes(array(&self.scratch)?);
                    self.scratch.clear();
                    self.state = if magic == MAGIC_NUMBER {
                        State::FrameHeader
                    } else if is_skippable(magic) {
                        State::SkippableSize
                    } else {
                        return Err(InvalidDataKind::Magic.into());
                    };
                }
                State::SkippableSize => {
                    if !fill(&mut self.scratch, input, SKIPPABLE_SIZE_FIELD) {
                        return Ok(DecoderStatus::NeedsInput);
                    }
                    let size = u32::from_le_bytes(array(&self.scratch)?);
                    self.scratch.clear();
                    if size > SKIPPABLE_PAYLOAD_MAX {
                        return Err(InvalidDataKind::SkippableFrame.into());
                    }
                    self.state = State::Skipping(u64::from(size));
                }
                State::Skipping(rest) => {
                    let take = rest.min(input.len() as u64);
                    *input = &input[take as usize..];
                    if take < rest {
                        self.state = State::Skipping(rest - take);
                        return Ok(DecoderStatus::NeedsInput);
                    }
                    self.complete_frames += 1;
                    self.state = self.after_frame();
                }
                State::FrameHeader => {
                    if !fill(&mut self.scratch, input, 1) {
                        return Ok(DecoderStatus::NeedsInput);
                    }
                    let size = header_size(self.scratch[0]);
                    if !fill(&mut self.scratch, input, size) {
                        return Ok(DecoderStatus::NeedsInput);
                    }
                    let header = FrameHeader::parse(&self.scratch)?;
                    self.scratch.clear();
                    self.frame
                        .begin(header, &self.config.limits(), dictionary)?;
                    self.history.reset();
                    self.state = State::BlockHeader;
                }
                State::BlockHeader => {
                    if !fill(&mut self.scratch, input, BLOCK_HEADER_SIZE) {
                        return Ok(DecoderStatus::NeedsInput);
                    }
                    let block = BlockHeader::parse(array(&self.scratch)?, self.frame.block_max())?;
                    self.scratch.clear();
                    self.state = State::BlockContent(block);
                }
                State::BlockContent(block) => {
                    if !fill(&mut self.scratch, input, block.content_size()) {
                        return Ok(DecoderStatus::NeedsInput);
                    }
                    let window = self.window_size().ok_or(DecodeError::InvalidState)?;
                    let (mut buffer, start) =
                        self.history.prepare(window, self.frame.block_max())?;
                    let end = self.frame.decode_block(
                        self.backend,
                        &block,
                        &self.scratch,
                        &mut buffer,
                        0..start,
                        dictionary,
                    )?;
                    self.history.commit(end);
                    self.decoded += (end - start) as u64;
                    if let Some(limit) = self.config.limits().max_output_bytes()
                        && self.decoded > limit
                    {
                        return Err(DecodeError::OutputLimitExceeded { limit });
                    }
                    self.scratch.clear();
                    self.state = if !block.last {
                        State::BlockHeader
                    } else if self.frame.has_checksum() {
                        State::Checksum
                    } else {
                        self.end_frame()
                    };
                }
                State::Checksum => {
                    if !fill(&mut self.scratch, input, CHECKSUM_SIZE) {
                        return Ok(DecoderStatus::NeedsInput);
                    }
                    self.frame.verify_checksum(array(&self.scratch)?)?;
                    self.scratch.clear();
                    self.state = self.end_frame();
                }
            }
        }
    }

    /// Counts a completed Zstandard frame and picks what follows it.
    fn end_frame(&mut self) -> State {
        self.frames += 1;
        self.complete_frames += 1;
        self.after_frame()
    }

    /// What follows a completed frame of either kind: a single-frame
    /// operation ends after a skippable frame too, as a
    /// `ZSTD_decompressStream` member does.
    const fn after_frame(&self) -> State {
        match self.config.frame_mode() {
            FrameMode::Single => State::Done,
            FrameMode::Concatenated => State::Magic,
        }
    }

    /// Decodes complete input, appending the output to `dst` and using it
    /// as the history. On error `dst` may hold a partial result past its
    /// original length.
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(crate) fn decode_all(
        &mut self,
        src: &[u8],
        dst: &mut Vec<u8>,
        dictionary: Option<DictView<'_>>,
    ) -> Result<(), DecodeError> {
        let start = dst.len();
        // Frames that declare a size are cut at their end, as by
        // `decode_to_slice`, so no slack past the output is zeroed. A frame
        // that overruns its size then fails differently, so any error
        // reruns the input uncut, which defines the results.
        if self.decode_all_with(src, dst, true, dictionary).is_ok() {
            return Ok(());
        }
        dst.truncate(start);
        self.reset(self.config);
        self.decode_all_with(src, dst, false, dictionary)
    }

    /// One pass of [`Self::decode_all`]; `cut` stops each block at its
    /// frame's declared end.
    fn decode_all_with(
        &mut self,
        src: &[u8],
        dst: &mut Vec<u8>,
        cut: bool,
        dictionary: Option<DictView<'_>>,
    ) -> Result<(), DecodeError> {
        let mut out = VecOutput {
            buffer: dst,
            filled: 0,
            frame_end: None,
            cut,
            limit: 0,
        };
        out.filled = out.buffer.len();
        let result = self.walk(src, &mut out, dictionary);
        let filled = out.filled;
        dst.truncate(filled);
        result
    }

    /// Decodes complete input straight into `dst`, returning the output's
    /// length, when every Zstandard frame declares a content size that
    /// fits. `None` asks the caller to take the incremental path, which
    /// also reports any error: this path stops at the first one, and may
    /// have written anywhere before the end of the frame it failed in.
    ///
    /// Each frame's output is cut to its declared size, so chunked copies
    /// never touch bytes past the output.
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(crate) fn decode_to_slice(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        dictionary: Option<DictView<'_>>,
    ) -> Option<usize> {
        let mut out = SliceOutput {
            buffer: dst,
            filled: 0,
            frame_end: 0,
        };
        self.walk(src, &mut out, dictionary).ok()?;
        Some(out.filled)
    }

    /// [`Self::decode_to_slice`] into memory that may be uninitialized.
    ///
    /// The decoder reads its output back as history, so `dst` is zeroed
    /// one block ahead of it, never past the current frame's declared end:
    /// on success no byte after the returned length has been written.
    #[cfg_attr(all(feature = "hotpath", not(feature = "no_std")), hotpath::measure)]
    pub(crate) fn decode_to_uninit(
        &mut self,
        src: &[u8],
        dst: &mut [MaybeUninit<u8>],
        dictionary: Option<DictView<'_>>,
    ) -> Option<usize> {
        let mut out = UninitOutput {
            buffer: BorrowedBuf::from(dst),
            filled: 0,
            frame_end: 0,
            limit: 0,
        };
        self.walk(src, &mut out, dictionary).ok()?;
        Some(out.filled)
    }

    /// The one-shot frame grammar over complete input, writing through
    /// `out`; every frame starts from `dictionary`.
    fn walk<O: Output>(
        &mut self,
        src: &[u8],
        out: &mut O,
        dictionary: Option<DictView<'_>>,
    ) -> Result<(), DecodeError> {
        let limits = self.config.limits();
        let base = out.filled();
        let (data, limit) = match limits.max_input_bytes() {
            Some(limit) if src.len() as u64 > limit => (&src[..limit as usize], Some(limit)),
            _ => (src, None),
        };
        let mut reader = SliceReader {
            data,
            position: 0,
            limit,
        };
        let mut complete_frames = 0u64;
        loop {
            if reader.is_empty() {
                if let Some(limit) = limit {
                    return Err(DecodeError::InputLimitExceeded { limit });
                }
                if complete_frames > 0 && self.config.frame_mode() == FrameMode::Concatenated {
                    return Ok(());
                }
                return Err(DecodeError::UnexpectedEndOfInput);
            }
            let magic = u32::from_le_bytes(reader.take_array()?);
            if is_skippable(magic) {
                let size = u32::from_le_bytes(reader.take_array()?);
                if size > SKIPPABLE_PAYLOAD_MAX {
                    return Err(InvalidDataKind::SkippableFrame.into());
                }
                reader.take(size as usize)?;
                complete_frames += 1;
                if self.config.frame_mode() == FrameMode::Single {
                    return single_end(&reader, src);
                }
                continue;
            }
            if magic != MAGIC_NUMBER {
                return Err(InvalidDataKind::Magic.into());
            }
            let descriptor = reader.data.get(reader.position).copied();
            let descriptor = descriptor.ok_or(match limit {
                Some(limit) => DecodeError::InputLimitExceeded { limit },
                None => DecodeError::UnexpectedEndOfInput,
            })?;
            let header = FrameHeader::parse(reader.take(header_size(descriptor))?)?;
            self.frame.begin(header, &limits, dictionary)?;
            let frame_start = out.filled();
            out.begin_frame(header.content_size, reader.data.len() - reader.position)?;
            loop {
                let block = BlockHeader::parse(reader.take_array()?, self.frame.block_max())?;
                let content = reader.take(block.content_size())?;
                let start = out.filled();
                let end = if let Some(placed) = out.place(&block, content)? {
                    self.frame.account_block(&block, placed)?;
                    start + placed.len()
                } else {
                    let buffer = out.reserve(self.frame.block_max())?;
                    self.frame.decode_block(
                        self.backend,
                        &block,
                        content,
                        buffer,
                        frame_start..start,
                        dictionary,
                    )?
                };
                out.commit(end);
                if let Some(limit) = limits.max_output_bytes()
                    && (end - base) as u64 > limit
                {
                    return Err(DecodeError::OutputLimitExceeded { limit });
                }
                if block.last {
                    break;
                }
            }
            if self.frame.has_checksum() {
                self.frame.verify_checksum(reader.take_array()?)?;
            }
            complete_frames += 1;
            if self.config.frame_mode() == FrameMode::Single {
                return single_end(&reader, src);
            }
        }
    }
}

/// The end of a single-frame one-shot operation whose frame ended at
/// `reader`'s position in `src`: anything after it is trailing data.
const fn single_end(reader: &SliceReader<'_>, src: &[u8]) -> Result<(), DecodeError> {
    if reader.position < src.len() {
        return Err(DecodeError::TrailingData {
            offset: reader.position as u64,
        });
    }
    Ok(())
}

/// Where the one-shot driver writes: a buffer whose prefix up to
/// [`Output::filled`] is output, the current frame's included.
trait Output {
    /// Bytes of output so far.
    fn filled(&self) -> usize;

    /// Starts a frame that declares `content_size`, if it does.
    /// `unread` input bytes follow the header.
    fn begin_frame(&mut self, content_size: Option<u64>, unread: usize) -> Result<(), DecodeError>;

    /// The buffer, which may grow to hold a block of `block_max` bytes
    /// after the output where the output can grow.
    fn reserve(&mut self, block_max: usize) -> Result<&mut dyn BlockOutput, DecodeError>;

    /// Records the end of the block just decoded.
    fn commit(&mut self, end: usize);

    /// Writes a raw or RLE block straight after the output and returns
    /// its bytes, or `None` to have the block decoded into
    /// [`Self::reserve`]'s buffer.
    fn place(
        &mut self,
        _block: &BlockHeader,
        _content: &[u8],
    ) -> Result<Option<&[u8]>, DecodeError> {
        Ok(None)
    }
}

/// Output bytes per unread input byte that [`VecOutput`] reserves for a
/// declared frame before seeing any output.
const PLAUSIBLE_RATIO: usize = 32;

/// Bytes a growing output zeroes at a time ahead of a block's writes:
/// small enough to stay in the first-level cache until the copies
/// overwrite them, large enough that each extension costs little.
const ZERO_CHUNK: usize = 4 * 1024;

/// A caller's vector, grown with initialized slack for chunked copies.
///
/// Its length is the end of the zeroed prefix, which runs ahead of the
/// output; each compressed block extends it a chunk at a time
/// ([`BlockOutput::grow`]), up to [`Self::limit`].
struct VecOutput<'a> {
    buffer: &'a mut Vec<u8>,
    filled: usize,
    /// Where the current frame ends, when it declares a size.
    frame_end: Option<usize>,
    /// Whether blocks are cut at [`Self::frame_end`].
    cut: bool,
    /// How far the block in progress may extend the buffer.
    limit: usize,
}

impl BlockOutput for VecOutput<'_> {
    fn buffer(&mut self) -> &mut [u8] {
        self.buffer
    }

    fn grow(&mut self, end: usize) -> bool {
        let len = self.buffer.len();
        let target = end.max(len + ZERO_CHUNK).min(self.limit);
        if target <= len {
            return false;
        }
        // `reserve` made room for the limit, so this never reallocates.
        self.buffer.resize(target, 0);
        true
    }

    fn append(&mut self, at: usize, literals: &[u8], offset: usize, match_length: usize) -> bool {
        let lit_end = at + literals.len();
        if at > self.buffer.len() || lit_end + match_length > self.limit {
            return false;
        }
        let start = match lit_end.checked_sub(offset) {
            _ if match_length == 0 => 0,
            Some(start) if offset > 0 => start,
            _ => return false,
        };
        // Bytes zeroed past `at` are written again by the append.
        self.buffer.truncate(at);
        self.buffer.extend_from_slice(literals);
        // The copied source doubles while it is shorter than the match, so
        // an overlapping match repeats its period.
        let mut copied = 0;
        while copied < match_length {
            let step = (match_length - copied).min(offset + copied);
            self.buffer.extend_from_within(start..start + step);
            copied += step;
        }
        true
    }
}

impl Output for VecOutput<'_> {
    fn filled(&self) -> usize {
        self.filled
    }

    fn begin_frame(&mut self, content_size: Option<u64>, unread: usize) -> Result<(), DecodeError> {
        self.frame_end = content_size
            .and_then(|size| usize::try_from(size).ok())
            .and_then(|size| self.filled.checked_add(size));
        // A declared size is reserved up front only as far as the input
        // could plausibly regenerate it, 32 bytes per unread byte: an
        // allocation then always needs input in proportion, and further
        // growth follows the output actually produced.
        if let Some(end) = self.frame_end {
            let plausible = self
                .filled
                .saturating_add(unread.saturating_mul(PLAUSIBLE_RATIO));
            reserve_output(self.buffer, end.min(plausible), Some(end))?;
        }
        Ok(())
    }

    /// Reserves the block's room without zeroing it: the block zeroes
    /// what it reaches, a chunk at a time.
    fn reserve(&mut self, block_max: usize) -> Result<&mut dyn BlockOutput, DecodeError> {
        let slack = block_max + WILDCOPY_OVERLENGTH;
        let wanted = self.filled + slack;
        if self.cut
            && let Some(end) = self.frame_end
        {
            let len = wanted.min(end);
            reserve_output(self.buffer, len, Some(end))?;
            // Slack zeroed for an earlier, uncut block may reach past the
            // frame's end; the buffer is cut back to it.
            self.buffer.truncate(len);
            self.limit = len;
            return Ok(self);
        }
        let bound = self.frame_end.and_then(|end| end.checked_add(slack));
        reserve_output(self.buffer, wanted, bound)?;
        self.limit = wanted;
        Ok(self)
    }

    fn commit(&mut self, end: usize) {
        self.filled = end;
    }

    /// Appends raw and RLE blocks in one pass: zeroing room for them first
    /// would write every byte twice.
    fn place(&mut self, block: &BlockHeader, content: &[u8]) -> Result<Option<&[u8]>, DecodeError> {
        let len = match block.kind {
            BlockKind::Raw => content.len(),
            BlockKind::Rle => block.size,
            BlockKind::Compressed => return Ok(None),
        };
        let start = self.filled;
        self.buffer.truncate(start);
        reserve_output(self.buffer, start + len, self.frame_end)?;
        if block.kind == BlockKind::Rle {
            let &byte = content.first().ok_or(InvalidDataKind::BlockSize)?;
            self.buffer.resize(start + len, byte);
        } else {
            self.buffer.extend_from_slice(content);
        }
        Ok(Some(&self.buffer[start..]))
    }
}

/// Makes room for `len` bytes in `buffer`, without initializing them.
///
/// Capacity doubles, as `Vec` would grow it, so allocation stays within
/// twice the output requested so far. When the frame's last request
/// `bound` is known and lies within that factor, capacity jumps straight
/// to it: the frame then needs no further reallocation, copy, or capacity
/// past its end.
fn reserve_output(
    buffer: &mut Vec<u8>,
    len: usize,
    bound: Option<usize>,
) -> Result<(), DecodeError> {
    if buffer.capacity() < len {
        let doubled = buffer.capacity().saturating_mul(2).max(len);
        let capacity = match bound {
            Some(bound) if bound >= len && bound <= len.saturating_mul(2) => bound,
            Some(bound) if bound >= len => doubled.min(bound),
            _ => doubled,
        };
        buffer
            .try_reserve_exact(capacity - buffer.len())
            .map_err(|_| DecodeError::AllocationFailed)?;
    }
    Ok(())
}

/// A caller's slice, cut at each frame's declared end.
struct SliceOutput<'a> {
    buffer: &'a mut [u8],
    filled: usize,
    frame_end: usize,
}

impl BlockOutput for SliceOutput<'_> {
    fn buffer(&mut self) -> &mut [u8] {
        &mut self.buffer[..self.frame_end]
    }

    fn grow(&mut self, _end: usize) -> bool {
        false
    }

    fn grows(&self) -> bool {
        false
    }
}

impl Output for SliceOutput<'_> {
    fn filled(&self) -> usize {
        self.filled
    }

    fn begin_frame(
        &mut self,
        content_size: Option<u64>,
        _unread: usize,
    ) -> Result<(), DecodeError> {
        let room = (self.buffer.len() - self.filled) as u64;
        match content_size {
            Some(size) if size <= room => {
                self.frame_end = self.filled + size as usize;
                Ok(())
            }
            _ => Err(DecodeError::OutputTooSmall {
                written: self.filled,
            }),
        }
    }

    fn reserve(&mut self, _block_max: usize) -> Result<&mut dyn BlockOutput, DecodeError> {
        Ok(self)
    }

    fn commit(&mut self, end: usize) {
        self.filled = end;
    }
}

/// A caller's uninitialized slice, cut at each frame's declared end as
/// [`SliceOutput`] is. The buffer's filled part is the zeroed-ahead prefix
/// the decoder may read and write; it grows a chunk at a time, never past
/// the block in progress, so each byte is zeroed at most once and only
/// inside a frame's output.
struct UninitOutput<'a> {
    buffer: BorrowedBuf<'a, u8>,
    /// Bytes of output, never more than `buffer.len()`.
    filled: usize,
    frame_end: usize,
    /// How far the block in progress may extend the zeroed prefix.
    limit: usize,
}

impl BlockOutput for UninitOutput<'_> {
    fn buffer(&mut self) -> &mut [u8] {
        self.buffer.filled_mut()
    }

    fn grow(&mut self, end: usize) -> bool {
        let len = self.buffer.len();
        let target = end.max(len + ZERO_CHUNK).min(self.limit);
        if target <= len {
            return false;
        }
        self.zero_to(target);
        true
    }
}

impl UninitOutput<'_> {
    /// Extends the zeroed prefix to at least `end`.
    fn zero_to(&mut self, end: usize) {
        if let Some(grow) = end.checked_sub(self.buffer.len()).filter(|&grow| grow > 0) {
            self.buffer.unfilled().append_fill(grow, 0);
        }
    }
}

impl Output for UninitOutput<'_> {
    fn filled(&self) -> usize {
        self.filled
    }

    fn begin_frame(
        &mut self,
        content_size: Option<u64>,
        _unread: usize,
    ) -> Result<(), DecodeError> {
        let room = (self.buffer.capacity() - self.filled) as u64;
        match content_size {
            Some(size) if size <= room => {
                self.frame_end = self.filled + size as usize;
                Ok(())
            }
            _ => Err(DecodeError::OutputTooSmall {
                written: self.filled,
            }),
        }
    }

    /// Bounds the block's room; the block zeroes what it reaches, a chunk
    /// at a time.
    fn reserve(&mut self, block_max: usize) -> Result<&mut dyn BlockOutput, DecodeError> {
        self.limit = self
            .filled
            .saturating_add(block_max + WILDCOPY_OVERLENGTH)
            .min(self.frame_end);
        Ok(self)
    }

    fn commit(&mut self, end: usize) {
        self.filled = end;
    }

    /// Writes raw and RLE blocks in one pass: zeroing room for them first
    /// would write every byte twice.
    fn place(&mut self, block: &BlockHeader, content: &[u8]) -> Result<Option<&[u8]>, DecodeError> {
        let len = match block.kind {
            BlockKind::Raw => content.len(),
            BlockKind::Rle => block.size,
            BlockKind::Compressed => return Ok(None),
        };
        let start = self.filled;
        let end = start + len;
        if end > self.frame_end {
            return Err(DecodeError::OutputTooSmall { written: start });
        }
        // Bytes zeroed ahead for an earlier block are overwritten in place;
        // the rest is appended.
        let zeroed = self.buffer.len().clamp(start, end);
        if block.kind == BlockKind::Rle {
            let &byte = content.first().ok_or(InvalidDataKind::BlockSize)?;
            self.buffer.filled_mut()[start..zeroed].fill(byte);
            self.buffer.unfilled().append_fill(end - zeroed, byte);
        } else {
            let (head, tail) = content.split_at(zeroed - start);
            self.buffer.filled_mut()[start..zeroed].copy_from_slice(head);
            self.buffer.unfilled().append(tail);
        }
        Ok(Some(&self.buffer.filled()[start..end]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompressor::DecodeLimits;

    /// `hello` as one raw block in a single-segment frame with a checksum.
    const HELLO: [u8; 18] = [
        0x28, 0xb5, 0x2f, 0xfd, 0x24, 5, 0x29, 0, 0, b'h', b'e', b'l', b'l', b'o', 0x0b, 0x2f,
        0x77, 0xcc,
    ];
    /// An empty skippable frame.
    const SKIP: [u8; 9] = [0x5a, 0x2a, 0x4d, 0x18, 1, 0, 0, 0, 0xee];

    fn checksum_of_hello() -> [u8; 4] {
        let mut hasher = crate::shared::xxhash::Xxh64::new(0);
        hasher.update(b"hello");
        (hasher.digest() as u32).to_le_bytes()
    }

    fn hello() -> Vec<u8> {
        let mut frame = HELLO.to_vec();
        frame[14..].copy_from_slice(&checksum_of_hello());
        frame
    }

    fn stream(config: DecoderConfig) -> Stream {
        let mut stream = Stream::new(DecoderConfig::DEFAULT, Backend::default());
        stream.reset(config);
        stream
    }

    #[test]
    fn a_frame_streams_byte_by_byte() {
        let frame = hello();
        let mut stream = stream(DecoderConfig::DEFAULT);
        let mut decoded = Vec::new();
        for byte in &frame {
            let mut out = [0u8; 1];
            let progress = stream
                .process(
                    ::core::slice::from_ref(byte),
                    &mut out,
                    DecodeOperation::Process,
                    None,
                )
                .expect("valid");
            assert_eq!(progress.consumed, 1);
            decoded.extend_from_slice(&out[..progress.produced]);
            if progress.status == DecoderStatus::NeedsOutput {
                loop {
                    let progress = stream
                        .process(&[], &mut out, DecodeOperation::Process, None)
                        .expect("valid");
                    decoded.extend_from_slice(&out[..progress.produced]);
                    if progress.status != DecoderStatus::NeedsOutput {
                        break;
                    }
                }
            }
        }
        let progress = stream
            .process(&[], &mut [], DecodeOperation::Finish, None)
            .expect("valid");
        assert_eq!(progress.status, DecoderStatus::Finished);
        assert_eq!(decoded, b"hello");
        assert_eq!(stream.total_in(), frame.len() as u64);
        assert_eq!(stream.total_out(), 5);
        assert_eq!(stream.frames_decoded(), 1);
        assert_eq!(stream.window_size(), Some(5));
        assert!(stream.is_finished());
        // A finished operation reports no further progress.
        let again = stream
            .process(&[1, 2], &mut [0; 4], DecodeOperation::Finish, None)
            .expect("finished");
        assert_eq!((again.consumed, again.produced), (0, 0));
    }

    #[test]
    fn single_mode_stops_after_the_first_frame() {
        let mut input = hello();
        input.extend_from_slice(b"tail");
        let mut stream = stream(FrameMode::Single.into());
        let mut out = [0u8; 16];
        let progress = stream
            .process(&input, &mut out, DecodeOperation::Process, None)
            .expect("valid");
        assert_eq!(progress.status, DecoderStatus::Finished);
        assert_eq!(progress.consumed, input.len() - 4);
        assert_eq!(&out[..progress.produced], b"hello");
        assert_eq!(stream.frames_decoded(), 1);
    }

    #[test]
    fn single_mode_stops_after_a_skippable_frame() {
        let mut input = SKIP.to_vec();
        input.extend_from_slice(&hello());
        let mut stream = stream(FrameMode::Single.into());
        let mut out = [0u8; 16];
        let progress = stream
            .process(&input, &mut out, DecodeOperation::Process, None)
            .expect("valid");
        assert_eq!(progress.status, DecoderStatus::Finished);
        assert_eq!((progress.consumed, progress.produced), (SKIP.len(), 0));
        assert_eq!(stream.frames_decoded(), 0);
        assert!(matches!(
            stream.decode_all(&input, &mut Vec::new(), None),
            Err(DecodeError::TrailingData { offset: 9 })
        ));
    }

    #[test]
    fn finish_contract_violations_poison_the_session() {
        let frame = hello();
        let mut stream = stream(DecoderConfig::DEFAULT);
        stream
            .process(&frame[..4], &mut [], DecodeOperation::Finish, None)
            .expect_err("truncated");
        let mut stream = self::stream(DecoderConfig::DEFAULT);
        stream
            .process(&frame[..4], &mut [], DecodeOperation::Process, None)
            .expect("accepted");
        let progress = stream
            .process(&frame[4..], &mut [0; 2], DecodeOperation::Finish, None)
            .expect("partial");
        assert_eq!(progress.status, DecoderStatus::NeedsOutput);
        let rest = &frame[4 + progress.consumed..];
        let failure = stream
            .process(&rest[1..], &mut [0; 8], DecodeOperation::Finish, None)
            .expect_err("changed suffix");
        assert!(matches!(failure.error, DecodeError::InvalidState));
        let failure = stream
            .process(rest, &mut [0; 8], DecodeOperation::Finish, None)
            .expect_err("failed");
        assert!(matches!(failure.error, DecodeError::InvalidState));

        let mut stream = self::stream(DecoderConfig::DEFAULT);
        stream
            .process(&frame[..4], &mut [], DecodeOperation::Finish, None)
            .expect_err("truncated");
        let mut stream = self::stream(DecoderConfig::DEFAULT);
        stream
            .process(&frame, &mut [0; 2], DecodeOperation::Finish, None)
            .expect("partial");
        let failure = stream
            .process(&[], &mut [0; 8], DecodeOperation::Process, None)
            .expect_err("process after finish");
        assert!(matches!(failure.error, DecodeError::InvalidState));
    }

    #[test]
    fn limits_stop_streaming_operations() {
        let frame = hello();
        let input_limit = DecodeLimits::default().with_max_input_bytes(Some(10));
        let mut stream = stream(input_limit.into());
        let failure = stream
            .process(&frame, &mut [0; 8], DecodeOperation::Process, None)
            .expect_err("input budget");
        assert!(matches!(
            failure.error,
            DecodeError::InputLimitExceeded { limit: 10 }
        ));
        assert_eq!(failure.consumed, 10);

        let output_limit = DecodeLimits::default().with_max_output_bytes(Some(4));
        let mut stream = self::stream(output_limit.into());
        let failure = stream
            .process(&frame, &mut [0; 8], DecodeOperation::Finish, None)
            .expect_err("output budget");
        assert!(matches!(
            failure.error,
            DecodeError::OutputLimitExceeded { limit: 4 }
        ));
        assert_eq!(failure.produced, 0);
    }

    #[test]
    fn streaming_rejects_bad_magic_and_oversized_skippable_frames() {
        let mut stream = stream(DecoderConfig::DEFAULT);
        let failure = stream
            .process(b"nope", &mut [], DecodeOperation::Process, None)
            .expect_err("bad magic");
        assert!(matches!(
            failure.error,
            DecodeError::InvalidData {
                kind: InvalidDataKind::Magic
            }
        ));
        let mut stream = self::stream(DecoderConfig::DEFAULT);
        let failure = stream
            .process(
                &[0x50, 0x2a, 0x4d, 0x18, 0xf8, 0xff, 0xff, 0xff],
                &mut [],
                DecodeOperation::Process,
                None,
            )
            .expect_err("oversized");
        assert!(matches!(
            failure.error,
            DecodeError::InvalidData {
                kind: InvalidDataKind::SkippableFrame
            }
        ));
        // A skippable frame alone is a complete concatenated operation.
        let mut stream = self::stream(DecoderConfig::DEFAULT);
        let progress = stream
            .process(&SKIP[..5], &mut [], DecodeOperation::Process, None)
            .expect("partial");
        assert_eq!(progress.status, DecoderStatus::NeedsInput);
        let progress = stream
            .process(&SKIP[5..], &mut [], DecodeOperation::Process, None)
            .expect("partial");
        assert_eq!(progress.status, DecoderStatus::NeedsInput);
        let progress = stream
            .process(&[], &mut [], DecodeOperation::Finish, None)
            .expect("complete");
        assert_eq!(progress.status, DecoderStatus::Finished);
        assert_eq!(stream.frames_decoded(), 0);
    }

    #[test]
    fn decode_all_handles_frames_skippable_frames_and_trailing_bytes() {
        let mut stream = stream(DecoderConfig::DEFAULT);
        let mut input = hello();
        input.extend_from_slice(&SKIP);
        input.extend_from_slice(&hello());
        let mut out = b">".to_vec();
        stream.decode_all(&input, &mut out, None).expect("valid");
        assert_eq!(out, b">hellohello");

        let mut single = self::stream(FrameMode::Single.into());
        let mut out = Vec::new();
        assert!(matches!(
            single.decode_all(&input, &mut out, None),
            Err(DecodeError::TrailingData { offset: 18 })
        ));
        let mut out = Vec::new();
        single.decode_all(&hello(), &mut out, None).expect("valid");
        assert_eq!(out, b"hello");

        let mut out = Vec::new();
        assert!(matches!(
            stream.decode_all(&[], &mut out, None),
            Err(DecodeError::UnexpectedEndOfInput)
        ));
        assert!(matches!(
            stream.decode_all(&HELLO[..4], &mut out, None),
            Err(DecodeError::UnexpectedEndOfInput)
        ));
        assert!(matches!(
            stream.decode_all(
                &[0x50, 0x2a, 0x4d, 0x18, 0xf8, 0xff, 0xff, 0xff],
                &mut out,
                None
            ),
            Err(DecodeError::InvalidData {
                kind: InvalidDataKind::SkippableFrame
            })
        ));
        assert!(matches!(
            stream.decode_all(b"nope", &mut out, None),
            Err(DecodeError::InvalidData {
                kind: InvalidDataKind::Magic
            })
        ));
        let mut bad = hello();
        bad[17] ^= 1;
        assert!(matches!(
            stream.decode_all(&bad, &mut out, None),
            Err(DecodeError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn output_growth_doubles_and_stops_at_a_frames_bound() {
        let grow_output = |buffer: &mut Vec<u8>, len, bound| {
            reserve_output(buffer, len, bound)?;
            if buffer.len() < len {
                buffer.resize(len, 0);
            }
            Ok::<(), DecodeError>(())
        };
        // Without a bound, capacity doubles.
        let mut buffer = Vec::new();
        grow_output(&mut buffer, 100, None).expect("grows");
        assert_eq!((buffer.len(), buffer.capacity()), (100, 100));
        grow_output(&mut buffer, 150, None).expect("grows");
        assert_eq!((buffer.len(), buffer.capacity()), (150, 200));
        // A bound within twice the request is taken at once.
        grow_output(&mut buffer, 250, Some(480)).expect("grows");
        assert_eq!((buffer.len(), buffer.capacity()), (250, 480));
        // Requests within the capacity, or already met, allocate nothing.
        grow_output(&mut buffer, 480, Some(480)).expect("fits");
        grow_output(&mut buffer, 10, None).expect("met");
        assert_eq!((buffer.len(), buffer.capacity()), (480, 480));
        // A distant bound only caps the doubling; one below the request is
        // ignored.
        grow_output(&mut buffer, 500, Some(10_000)).expect("grows");
        assert_eq!(buffer.capacity(), 960);
        grow_output(&mut buffer, 1000, Some(5)).expect("grows");
        assert_eq!((buffer.len(), buffer.capacity()), (1000, 1920));
        assert!(buffer.iter().all(|&byte| byte == 0));
        assert!(matches!(
            grow_output(&mut buffer, usize::MAX, None),
            Err(DecodeError::AllocationFailed)
        ));
    }

    #[test]
    fn vec_output_zeroes_in_chunks_and_appends_long_sequences() {
        let mut dst = b"ab".to_vec();
        let mut out = VecOutput {
            buffer: &mut dst,
            filled: 2,
            frame_end: None,
            cut: true,
            limit: 0,
        };
        // Room for a block is reserved, not zeroed.
        let room = out.reserve(10).expect("room");
        assert_eq!(room.buffer(), b"ab");
        // A chunk at a time, capped at the block and the overlength.
        assert!(room.grow(3));
        assert_eq!(room.buffer().len(), 2 + 10 + WILDCOPY_OVERLENGTH);
        assert!(!room.grow(50));
        // Appending cuts the zeroed bytes; an overlapping match repeats
        // its period.
        assert!(room.append(2, b"cd", 2, 7));
        assert_eq!(room.buffer(), b"abcdcdcdcdc");
        // A match before the buffer or with offset zero, a sequence past
        // the room, or a start past the buffer is refused.
        assert!(!room.append(11, b"", 12, 1));
        assert!(!room.append(11, b"", 0, 1));
        assert!(!room.append(11, b"x", 1, 40));
        assert!(!room.append(12, b"x", 1, 1));
        // Without a match the offset does not matter.
        assert!(room.append(11, b"xyz", 0, 0));
        assert_eq!(room.buffer(), b"abcdcdcdcdcxyz");
        assert!(dst.capacity() >= 2 + 10 + WILDCOPY_OVERLENGTH);
    }

    #[test]
    fn decode_all_cuts_declared_frames_and_reruns_failures_uncut() {
        // A declared frame leaves no slack and no spare capacity.
        let mut out = Vec::new();
        stream(DecoderConfig::DEFAULT)
            .decode_all(&hello(), &mut out, None)
            .expect("valid");
        assert_eq!((out.as_slice(), out.capacity()), (&b"hello"[..], 5));
        // An overrun is caught uncut, with the reference's error.
        let mut overrun = alloc::vec![0x28, 0xb5, 0x2f, 0xfd, 0x40, 0x00, 0x00, 0x00];
        overrun.extend_from_slice(&((300u32 << 3) | 1).to_le_bytes()[..3]);
        overrun.extend_from_slice(&[9; 300]);
        let mut out = b">".to_vec();
        assert!(matches!(
            stream(DecoderConfig::DEFAULT).decode_all(&overrun, &mut out, None),
            Err(DecodeError::ContentSizeMismatch {
                declared: 256,
                decoded: 300
            })
        ));
        assert_eq!(out, b">");
    }

    #[test]
    fn decode_all_enforces_budgets() {
        let frame = hello();
        let mut out = Vec::new();
        for cut in [3, 4, 10, 17] {
            let limits = DecodeLimits::default().with_max_input_bytes(Some(cut));
            let mut stream = stream(limits.into());
            assert!(
                matches!(
                    stream.decode_all(&frame, &mut out, None),
                    Err(DecodeError::InputLimitExceeded { .. })
                ),
                "cut {cut}"
            );
        }
        let mut doubled = frame.clone();
        doubled.extend_from_slice(&frame);
        let limits = DecodeLimits::default().with_max_input_bytes(Some(18));
        assert!(matches!(
            stream(limits.into()).decode_all(&doubled, &mut out, None),
            Err(DecodeError::InputLimitExceeded { limit: 18 })
        ));
        let limits = DecodeLimits::default().with_max_output_bytes(Some(4));
        assert!(matches!(
            stream(limits.into()).decode_all(&frame, &mut out, None),
            Err(DecodeError::OutputLimitExceeded { limit: 4 })
        ));
    }

    /// Reads back memory a test initialized in full.
    fn bytes(output: &[MaybeUninit<u8>]) -> Vec<u8> {
        // SAFETY: each test initializes every element before reading.
        output
            .iter()
            .map(|slot| unsafe { slot.assume_init() })
            .collect()
    }

    fn block(kind: BlockKind, size: usize) -> BlockHeader {
        BlockHeader {
            last: false,
            kind,
            size,
        }
    }

    #[test]
    fn uninit_output_zeroes_on_demand_within_the_block_and_frame() {
        let mut storage = [MaybeUninit::new(0xAAu8); 64];
        let mut out = UninitOutput {
            buffer: BorrowedBuf::from(&mut storage[..]),
            filled: 0,
            frame_end: 0,
            limit: 0,
        };
        assert!(matches!(
            out.begin_frame(None, 0),
            Err(DecodeError::OutputTooSmall { written: 0 })
        ));
        assert!(matches!(
            out.begin_frame(Some(65), 0),
            Err(DecodeError::OutputTooSmall { written: 0 })
        ));
        out.begin_frame(Some(60), 0).expect("fits");
        let room = out.reserve(1).expect("room");
        // Nothing is zeroed until the block asks; then a chunk at a time,
        // capped at the block's maximum and the overlength.
        assert!(room.buffer().is_empty());
        assert!(room.grow(2));
        assert_eq!(room.buffer().len(), 1 + WILDCOPY_OVERLENGTH);
        assert!(!room.grow(40));
        out.commit(1);
        // The frame's end caps the zeroing however large a block may be.
        let room = out.reserve(1 << 17).expect("room");
        assert!(room.grow(2));
        assert_eq!(room.buffer().len(), 60);
        assert_eq!(out.filled(), 1);
        assert_eq!(bytes(&storage[..61]), [&[0u8; 60][..], &[0xAA]].concat());
    }

    #[test]
    fn uninit_output_places_raw_and_rle_blocks_over_and_past_the_zeroed_prefix() {
        let mut storage = [MaybeUninit::new(0xAAu8); 64];
        let mut out = UninitOutput {
            buffer: BorrowedBuf::from(&mut storage[..]),
            filled: 0,
            frame_end: 0,
            limit: 0,
        };
        out.begin_frame(Some(40), 0).expect("fits");
        assert!(
            out.place(&block(BlockKind::Compressed, 3), b"abc")
                .expect("deferred")
                .is_none()
        );
        // Zero the copies' overlength ahead, then place a raw block across
        // its end.
        let room = out.reserve(0).expect("room");
        assert!(room.grow(0));
        assert!(room.buffer().iter().all(|&byte| byte == 0));
        out.commit(0);
        let zeroed = out.buffer.len();
        assert_eq!(zeroed, WILDCOPY_OVERLENGTH);
        assert!(matches!(
            out.place(&block(BlockKind::Rle, 1), b""),
            Err(DecodeError::InvalidData { .. })
        ));
        let raw = alloc::vec![b'r'; zeroed + 3];
        let placed = out
            .place(&block(BlockKind::Raw, raw.len()), &raw)
            .expect("fits");
        assert_eq!(placed, Some(&raw[..]));
        out.commit(raw.len());
        // An RLE block appended entirely past the zeroed prefix.
        let placed = out.place(&block(BlockKind::Rle, 5), b"z").expect("fits");
        assert_eq!(placed, Some(&b"zzzzz"[..]));
        let end = raw.len() + 5;
        out.commit(end);
        assert!(matches!(
            out.place(&block(BlockKind::Rle, 40), b"z"),
            Err(DecodeError::OutputTooSmall { .. })
        ));
        let expected = [&raw[..], b"zzzzz", &[0xAA; 64][end..]].concat();
        assert_eq!(bytes(&storage), expected);
    }

    #[test]
    fn decode_to_uninit_writes_nothing_past_the_output() {
        let frame = hello();
        let mut storage = [MaybeUninit::new(0xAAu8); 8];
        let mut stream = stream(DecoderConfig::DEFAULT);
        assert_eq!(stream.decode_to_uninit(&frame, &mut storage, None), Some(5));
        assert_eq!(bytes(&storage), b"hello\xAA\xAA\xAA");
        // A frame that does not fit is left to the incremental path.
        assert_eq!(
            stream.decode_to_uninit(&frame, &mut storage[..4], None),
            None
        );
    }

    #[test]
    fn process_uninit_delivers_exactly_the_produced_prefix() {
        let frame = hello();
        let mut stream = stream(DecoderConfig::DEFAULT);
        let mut storage = [MaybeUninit::new(0xAAu8); 3];
        let progress = stream
            .process_uninit(&frame, &mut storage, DecodeOperation::Finish, None)
            .expect("decodes");
        assert_eq!(progress.produced, 3);
        assert_eq!(progress.status, DecoderStatus::NeedsOutput);
        assert_eq!(bytes(&storage), b"hel");
        let mut storage = [MaybeUninit::new(0xAAu8); 4];
        let progress = stream
            .process_uninit(
                &frame[progress.consumed..],
                &mut storage,
                DecodeOperation::Finish,
                None,
            )
            .expect("decodes");
        assert_eq!(progress.produced, 2);
        assert_eq!(progress.status, DecoderStatus::Finished);
        assert_eq!(bytes(&storage), b"lo\xAA\xAA");
    }
}

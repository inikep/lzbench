//! Buffered streaming (`ZSTD_compressStream2` and
//! `ZSTD_compressStream_generic` in buffered mode).
//!
//! Input is gathered into a ring-like buffer of `window + block` bytes and
//! compressed a block at a time; output that does not fit the caller's
//! buffer waits in an internal one. Two details make the bytes depend on the
//! call sequence, and both are ported as is: a `Flush` ends the current block
//! early, and an `End` whose remaining input fits a `compressBound` worth of
//! output while the buffer is empty is compressed straight from the caller's
//! memory, which the window then sees as a new segment.
//!
//! The caller's output is a [`Destination`]: initialized bytes, or memory that
//! the engine zeroes just ahead of each block it writes there directly.

use alloc::vec::Vec;

use super::cctx::{Engine, RequestedParams, compress_bound};
use super::cdict::CDict;
use super::error::CoreError;
use super::output::{Destination, Output, SliceOutput};
use super::window::{Addr, Segment, Segments};

/// What a streaming call asks for (`ZSTD_EndDirective`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum EndOp {
    Continue,
    Flush,
    End,
}

/// `ZSTD_cStreamStage`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Stage {
    Init,
    Load,
    Flush,
}

/// Streaming buffers and their positions.
#[derive(Debug)]
pub(crate) struct Stream {
    stage: Stage,
    /// Declared size plus one; `0` when unknown.
    pledged_src_size_plus_one: u64,
    in_buff: Vec<u8>,
    in_to_compress: usize,
    in_buff_pos: usize,
    in_buff_target: usize,
    out_buff: Vec<u8>,
    out_buff_content_size: usize,
    out_buff_flushed_size: usize,
    frame_ended: bool,
    /// Identity of the next caller input segment.
    next_input_id: u64,
}

/// What one [`Stream::compress`] call did.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct StreamStep {
    /// Input bytes consumed.
    pub(crate) consumed: usize,
    /// Output bytes written.
    pub(crate) produced: usize,
    /// Bytes still waiting in the internal output buffer.
    pub(crate) remaining: usize,
}

impl Stream {
    /// A stream waiting for its first call.
    pub(crate) const fn new() -> Self {
        Self {
            stage: Stage::Init,
            pledged_src_size_plus_one: 0,
            in_buff: Vec::new(),
            in_to_compress: 0,
            in_buff_pos: 0,
            in_buff_target: 0,
            out_buff: Vec::new(),
            out_buff_content_size: 0,
            out_buff_flushed_size: 0,
            frame_ended: false,
            next_input_id: 1 << 32,
        }
    }

    /// Declares the size of the next frame (`ZSTD_CCtx_setPledgedSrcSize`).
    ///
    /// # Errors
    ///
    /// [`CoreError::StageWrong`] once the frame has started.
    pub(crate) fn set_pledged_src_size(&mut self, size: u64) -> Result<(), CoreError> {
        if self.stage != Stage::Init {
            return Err(CoreError::StageWrong);
        }
        self.pledged_src_size_plus_one = size.wrapping_add(1);
        Ok(())
    }

    /// Abandons the current frame (`ZSTD_CCtx_reset(ZSTD_reset_session_only)`).
    pub(crate) fn reset_session(&mut self) {
        self.stage = Stage::Init;
        self.pledged_src_size_plus_one = 0;
    }

    /// Whether no frame is in progress.
    pub(crate) fn is_idle(&self) -> bool {
        self.stage == Stage::Init
    }

    /// Bytes retained by the stream buffers.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.in_buff.capacity() + self.out_buff.capacity()
    }

    /// Releases the stream buffers.
    pub(crate) fn release(&mut self) {
        self.in_buff = Vec::new();
        self.out_buff = Vec::new();
    }

    /// `ZSTD_CCtx_init_compressStream2` in buffered mode.
    fn init(
        &mut self,
        engine: &mut Engine,
        requested: &RequestedParams,
        dict: Option<&CDict>,
        end_op: EndOp,
        in_size: usize,
    ) -> Result<(), CoreError> {
        if end_op == EndOp::End {
            self.pledged_src_size_plus_one = in_size as u64 + 1;
        }
        let pledged = self.pledged_src_size_plus_one.wrapping_sub(1);
        super::begin_frame(engine, requested, pledged, &Segments::input(&[], 0, dict))?;
        let window_log = engine.applied.cparams.window_log;
        let window_size = (1u64 << window_log).min(pledged).max(1) as usize;
        let block_size = engine.block_size_max;
        let in_size = window_size + block_size;
        let out_size = compress_bound(block_size).unwrap_or(0) + 1;
        if self.in_buff.len() < in_size {
            self.in_buff.resize(in_size, 0);
        }
        self.in_buff.truncate(in_size);
        if self.out_buff.len() < out_size {
            self.out_buff.resize(out_size, 0);
        }
        self.out_buff.truncate(out_size);
        self.in_to_compress = 0;
        self.in_buff_pos = 0;
        self.in_buff_target =
            engine.block_size_max + usize::from(engine.block_size_max as u64 == pledged);
        self.out_buff_content_size = 0;
        self.out_buff_flushed_size = 0;
        self.stage = Stage::Load;
        self.frame_ended = false;
        Ok(())
    }

    /// One `ZSTD_compressStream2` call: consumes from `input`, writes into
    /// `output`, and reports how much of each it used and how many bytes
    /// still wait in the internal output buffer.
    ///
    /// # Errors
    ///
    /// Any failure of the block compressor, and
    /// [`CoreError::SrcSizeWrong`] when the input contradicts a declared size.
    pub(crate) fn compress<D: Destination + ?Sized>(
        &mut self,
        engine: &mut Engine,
        requested: &RequestedParams,
        dict: Option<&CDict>,
        output: &mut D,
        input: &[u8],
        end_op: EndOp,
    ) -> Result<StreamStep, CoreError> {
        if self.stage == Stage::Init {
            let started = self.init(engine, requested, dict, end_op, input.len());
            if started.is_err() {
                self.reset_session();
            }
            started?;
        }
        let mut consumed = 0;
        let mut produced = 0;
        let result = self.generic(
            engine,
            dict,
            output,
            &mut produced,
            input,
            &mut consumed,
            end_op,
        );
        if result.is_err() {
            self.reset_session();
        }
        result?;
        Ok(StreamStep {
            consumed,
            produced,
            remaining: self.out_buff_content_size - self.out_buff_flushed_size,
        })
    }

    /// `ZSTD_compressStream_generic` in buffered mode.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors ZSTD_compressStream_generic's state and buffers"
    )]
    fn generic<D: Destination + ?Sized>(
        &mut self,
        engine: &mut Engine,
        dict: Option<&CDict>,
        output: &mut D,
        out_pos: &mut usize,
        input: &[u8],
        in_pos: &mut usize,
        flush_mode: EndOp,
    ) -> Result<(), CoreError> {
        let iend = input.len();
        let oend = output.capacity();
        let mut ip = *in_pos;
        let mut op = *out_pos;
        let input_id = self.next_input_id;
        self.next_input_id += 1;

        let result = loop {
            match self.stage {
                Stage::Init => break Err(CoreError::StageWrong),
                Stage::Load => {
                    if flush_mode == EndOp::End
                        && compress_bound(iend - ip).is_some_and(|bound| oend - op >= bound)
                        && self.in_buff_pos == 0
                    {
                        // Shortcut: compress straight from the caller's input.
                        let segments = Segments {
                            buffer: &self.in_buff,
                            ..Segments::input(input, input_id, dict)
                        };
                        let src = Addr::new(Segment::Input(input_id), ip as i64);
                        let c_size = engine.compress_end_to(
                            &segments,
                            &mut output.tail(op),
                            src,
                            iend - ip,
                        )?;
                        ip = iend;
                        op += c_size;
                        self.frame_ended = true;
                        self.reset_session();
                        break Ok(());
                    }
                    let to_load = self.in_buff_target - self.in_buff_pos;
                    let loaded = to_load.min(iend - ip);
                    self.in_buff[self.in_buff_pos..self.in_buff_pos + loaded]
                        .copy_from_slice(&input[ip..ip + loaded]);
                    self.in_buff_pos += loaded;
                    ip += loaded;
                    if flush_mode == EndOp::Continue && self.in_buff_pos < self.in_buff_target {
                        break Ok(());
                    }
                    if flush_mode == EndOp::Flush && self.in_buff_pos == self.in_to_compress {
                        break Ok(());
                    }

                    let i_size = self.in_buff_pos - self.in_to_compress;
                    let o_size = oend - op;
                    let direct = compress_bound(i_size).is_some_and(|bound| o_size >= bound);
                    let last_block = flush_mode == EndOp::End && ip == iend;
                    let segments = Segments {
                        buffer: &self.in_buff,
                        ..Segments::input(&[], 0, dict)
                    };
                    let src = Addr::new(Segment::Buffer, self.in_to_compress as i64);
                    let c_size = if direct {
                        compress_block(
                            engine,
                            &segments,
                            &mut output.tail(op),
                            src,
                            i_size,
                            last_block,
                        )?
                    } else {
                        compress_block(
                            engine,
                            &segments,
                            &mut SliceOutput(&mut self.out_buff),
                            src,
                            i_size,
                            last_block,
                        )?
                    };
                    self.frame_ended = last_block;
                    self.in_buff_target = self.in_buff_pos + engine.block_size_max;
                    if self.in_buff_target > self.in_buff.len() {
                        self.in_buff_pos = 0;
                        self.in_buff_target = engine.block_size_max;
                    }
                    self.in_to_compress = self.in_buff_pos;
                    if direct {
                        op += c_size;
                        if self.frame_ended {
                            self.reset_session();
                            break Ok(());
                        }
                        continue;
                    }
                    self.out_buff_content_size = c_size;
                    self.out_buff_flushed_size = 0;
                    self.stage = Stage::Flush;
                }
                Stage::Flush => {
                    let to_flush = self.out_buff_content_size - self.out_buff_flushed_size;
                    let flushed = to_flush.min(oend - op);
                    output.put(
                        op,
                        &self.out_buff
                            [self.out_buff_flushed_size..self.out_buff_flushed_size + flushed],
                    );
                    op += flushed;
                    self.out_buff_flushed_size += flushed;
                    if to_flush != flushed {
                        break Ok(());
                    }
                    self.out_buff_content_size = 0;
                    self.out_buff_flushed_size = 0;
                    if self.frame_ended {
                        self.reset_session();
                        break Ok(());
                    }
                    self.stage = Stage::Load;
                }
            }
        };
        *in_pos = ip;
        *out_pos = op;
        result
    }
}

/// Compresses the gathered input as the frame's last chunk or as one more.
fn compress_block<O: Output>(
    engine: &mut Engine,
    segments: &Segments<'_>,
    dst: &mut O,
    src: Addr,
    len: usize,
    last_block: bool,
) -> Result<usize, CoreError> {
    if last_block {
        engine.compress_end_to(segments, dst, src, len)
    } else {
        engine.compress_continue_to(segments, dst, src, len, false)
    }
}

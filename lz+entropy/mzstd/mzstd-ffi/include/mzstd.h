/*
 * mzstd C ABI: one-shot and incremental Zstandard compression and
 * decompression.
 *
 * Link against libmzstd_ffi (static or shared) built from the mzstd-ffi
 * crate. The one-shot functions are thread-safe and keep no state between
 * calls. A session handle (mzstd_encoder, mzstd_decoder) keeps its codec
 * between calls; it may move between threads but must serve one call at a
 * time.
 */
#ifndef MZSTD_H
#define MZSTD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    /* Success; *output_len holds the number of bytes written. */
    MZSTD_OK = 0,
    /* Null output_len or handle, null buffer with a non-zero length, a
     * length above PTRDIFF_MAX, overlapping arguments, or a level, window
     * log or operation out of range. */
    MZSTD_INVALID_PARAMETER = 1,
    /* The result does not fit in the output buffer. */
    MZSTD_OUTPUT_TOO_SMALL = 2,
    /* Corrupt or truncated input, a checksum or content size mismatch, a
     * frame that needs a dictionary, allocation failure, or any other
     * failure. */
    MZSTD_ERROR = 3
} mzstd_result;

/*
 * Compresses input into output as one Zstandard frame.
 *
 * level: -131072..=22; 0 selects the default, 3. Unlike ZSTD_compress, a
 * level out of range is rejected rather than clamped.
 * On entry *output_len is the capacity of output; on return it is the number
 * of bytes written on MZSTD_OK and 0 otherwise (a null or misaligned
 * output_len is left untouched). mzstd_compress_bound(input_len) bytes are
 * always enough. input may be NULL when input_len is 0.
 *
 * The frame is byte-identical to ZSTD_compress(level) from zstd 1.5.7, and
 * a smaller output fails exactly when ZSTD_compress fails: at least 18 bytes
 * (the largest frame header) and some slack past the frame are needed.
 */
mzstd_result mzstd_compress(const uint8_t *input, size_t input_len,
                            uint8_t *output, size_t *output_len,
                            int level);

/*
 * Decompresses every frame in input into output, skipping skippable frames.
 *
 * On entry *output_len is the capacity of output; on return it is the number
 * of bytes written on MZSTD_OK and 0 otherwise. An empty input decodes to
 * nothing. Windows up to 2 GiB are accepted. output may be uninitialized;
 * on MZSTD_OK no byte after the result is written.
 */
mzstd_result mzstd_decompress(const uint8_t *input, size_t input_len,
                              uint8_t *output, size_t *output_len);

/*
 * The largest frame mzstd_compress writes for input_len bytes, equal to
 * ZSTD_compressBound; 0 when it does not fit in a size_t.
 */
size_t mzstd_compress_bound(size_t input_len);

/* ---- Sessions --------------------------------------------------------- */

/* What a session needs after a call that returned MZSTD_OK. */
typedef enum {
    /* The call did all it could with its input; pass more, or the next
     * operation. */
    MZSTD_NEEDS_INPUT = 0,
    /* Output is waiting; call again with room for it. */
    MZSTD_NEEDS_OUTPUT = 1,
    /* The frame (encoder) or operation (decoder) is complete and delivered.
     * Later calls report this again until the handle is reset. */
    MZSTD_FINISHED = 2
} mzstd_status;

/* ZSTD_EndDirective. */
typedef enum {
    /* Take input; compress whole blocks as they fill. */
    MZSTD_CONTINUE = 0,
    /* Make everything taken so far decodable; ends the current block. */
    MZSTD_FLUSH = 1,
    /* Compress everything left and end the frame. */
    MZSTD_END = 2
} mzstd_operation;

typedef enum {
    /* More input may follow; an empty input does not end it. */
    MZSTD_DECODE_PROCESS = 0,
    /* This is all the remaining input; use it on every later call of the
     * operation, with exactly the input not yet consumed. */
    MZSTD_DECODE_FINISH = 1
} mzstd_decode_operation;

/* The pledged_size that leaves a frame's size undeclared;
 * ZSTD_CONTENTSIZE_UNKNOWN. */
#define MZSTD_CONTENT_SIZE_UNKNOWN UINT64_MAX

typedef struct mzstd_encoder mzstd_encoder;
typedef struct mzstd_decoder mzstd_decoder;

/*
 * Every *_process call follows the same rules. On entry *input_len is the
 * number of bytes offered at input and *output_len the capacity of output;
 * on MZSTD_OK they are the bytes consumed and produced and *status is set.
 * After any other result both lengths are 0 and *status is untouched (a
 * null or misaligned input_len, output_len or status leaves all three
 * untouched). MZSTD_INVALID_PARAMETER leaves the session as it was;
 * MZSTD_ERROR abandons its frame or operation until the next reset. output
 * may be uninitialized.
 */

/*
 * Creates an encoder at level (-131072..=22, 0 for 3) and starts its first
 * frame, of undeclared size. Frames record their content size when known
 * and carry no checksum. *encoder is the handle on MZSTD_OK and NULL
 * otherwise.
 */
mzstd_result mzstd_encoder_create(int level, mzstd_encoder **encoder);

/*
 * Abandons the current frame and starts a new one of pledged_size bytes
 * (ZSTD_CCtx_setPledgedSrcSize), or MZSTD_CONTENT_SIZE_UNKNOWN. Clears an
 * earlier MZSTD_ERROR.
 */
mzstd_result mzstd_encoder_reset(mzstd_encoder *encoder,
                                 uint64_t pledged_size);

/*
 * One ZSTD_compressStream2 step: the same inputs, output capacities and
 * operations give byte-identical frames to zstd 1.5.7. Repeat MZSTD_FLUSH
 * or MZSTD_END, with the unconsumed input, while it reports
 * MZSTD_NEEDS_OUTPUT. Treat only the produced bytes of output as written;
 * scratch bytes past them may change too. Input that contradicts the
 * pledged size is MZSTD_ERROR.
 */
mzstd_result mzstd_encoder_process(mzstd_encoder *encoder,
                                   const uint8_t *input, size_t *input_len,
                                   uint8_t *output, size_t *output_len,
                                   mzstd_operation operation,
                                   mzstd_status *status);

/* Releases an encoder; NULL is ignored. */
void mzstd_encoder_free(mzstd_encoder *encoder);

/*
 * Creates a decoder of concatenated frames whose windows are at most
 * 2^window_log_max bytes (10..=31, 0 for 27: ZSTD_d_windowLogMax).
 * *decoder is the handle on MZSTD_OK and NULL otherwise.
 */
mzstd_result mzstd_decoder_create(int window_log_max,
                                  mzstd_decoder **decoder);

/* Abandons the current operation and starts a new one. Clears an earlier
 * MZSTD_ERROR. */
mzstd_result mzstd_decoder_reset(mzstd_decoder *decoder);

/*
 * One ZSTD_decompressStream step. Skippable frames are skipped. A frame
 * boundary reports MZSTD_NEEDS_INPUT until MZSTD_DECODE_FINISH confirms the
 * input ends there; input that ends inside a frame is MZSTD_ERROR. No byte
 * of output after the produced ones is written.
 */
mzstd_result mzstd_decoder_process(mzstd_decoder *decoder,
                                   const uint8_t *input, size_t *input_len,
                                   uint8_t *output, size_t *output_len,
                                   mzstd_decode_operation operation,
                                   mzstd_status *status);

/* Releases a decoder; NULL is ignored. */
void mzstd_decoder_free(mzstd_decoder *decoder);

#ifdef __cplusplus
}
#endif

#endif /* MZSTD_H */

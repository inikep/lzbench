/*
 * mbrotli C ABI: one-shot and streaming Brotli compression and decompression.
 *
 * Link against libmbrotli_ffi (static or shared) built from the mbrotli-ffi
 * crate. Distinct sessions may run concurrently; each handle requires
 * exclusive access. One-shot functions keep no state between calls.
 */
#ifndef MBROTLI_H
#define MBROTLI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    /* Success; *output_len holds the number of bytes written. */
    MBROTLI_OK = 0,
    /* Null output_len, null buffer with a non-zero length, a length above
     * PTRDIFF_MAX, overlapping arguments, or quality/lgwin out of range. */
    MBROTLI_INVALID_PARAMETER = 1,
    /* The result does not fit in the output buffer. */
    MBROTLI_OUTPUT_TOO_SMALL = 2,
    /* Corrupt, truncated or trailing input, allocation failure, or any other
     * failure. */
    MBROTLI_ERROR = 3,
    /* Streaming: more input needed, or encoder flush fully delivered. */
    MBROTLI_NEEDS_INPUT = 4,
    /* Streaming: provide output space and retry with unconsumed input. */
    MBROTLI_NEEDS_OUTPUT = 5,
    /* Streaming: stream complete, all output delivered. */
    MBROTLI_FINISHED = 6
} mbrotli_result;

/*
 * Compresses input into output as one complete Brotli stream.
 *
 * quality: 0..=11 (11 is densest). lgwin: window bits, 10..=24 (22 default).
 * On entry *output_len is the capacity of output; on return it is the number
 * of bytes written on MBROTLI_OK and 0 otherwise (a null or misaligned
 * output_len is left untouched). mbrotli_compress_bound(input_len) bytes are
 * always enough. input may be NULL when input_len is 0.
 *
 * The output is byte-identical to BrotliEncoderCompress(quality, lgwin,
 * BROTLI_MODE_GENERIC, ...), including its fallback to an uncompressed stream
 * when compression would exceed mbrotli_compress_bound(input_len).
 *
 * Bytes of output past the ones reported are never written.
 */
mbrotli_result mbrotli_compress(
    const uint8_t *input,
    size_t input_len,
    uint8_t *output,
    size_t *output_len,
    int quality,
    int lgwin
);

/*
 * Decompresses exactly one complete Brotli stream from input into output.
 *
 * On entry *output_len is the capacity of output; on return it is the number
 * of bytes written on MBROTLI_OK and 0 otherwise. On MBROTLI_OUTPUT_TOO_SMALL
 * output holds as much of the data as fits; decoding stops there, so a stream
 * corrupt only past that point is reported this way too, and MBROTLI_ERROR
 * with a larger buffer. Bytes after the stream are an error.
 *
 * output serves as the decoder's history, so no separate window is copied
 * out of. Bytes of output past the ones reported may be overwritten only when
 * its capacity is at most 1 KiB, which is zeroed first; larger capacities are
 * written only up to the result.
 */
mbrotli_result mbrotli_decompress(
    const uint8_t *input,
    size_t input_len,
    uint8_t *output,
    size_t *output_len
);

/*
 * Largest output mbrotli_compress can produce for input_len bytes at any
 * quality and window, or 0 if that bound does not fit in size_t. Equal to
 * BrotliEncoderMaxCompressedSize(input_len).
 */
size_t mbrotli_compress_bound(size_t input_len);

/* Session operations are accepted as int so unknown values can be rejected. */
typedef enum {
    MBROTLI_PROCESS = 0,
    MBROTLI_FLUSH = 1,
    MBROTLI_FINISH = 2
} mbrotli_operation;

/* ABI-stable opaque types: only pointers cross the C boundary. Their size,
 * alignment and fields are private implementation details, not part of this
 * header's ABI. Do not define these structs, allocate/copy/dereference handle
 * objects, or pass them by value. create/destroy in the same library instance
 * own their entire lifetime; internal Rust session layout may change freely. */
typedef struct mbrotli_encoder_session mbrotli_encoder_session;
typedef struct mbrotli_decoder_session mbrotli_decoder_session;

/* Create owning sessions. Returns NULL on invalid configuration or failure.
 * Encoder: quality 0..11, lgwin 10..24, initially unknown input size.
 * Decoder: default single-member mode, accepting RFC 9841 large windows.
 * Destroy each returned handle exactly once. NULL destroy is a no-op.
 * Never pass a freed handle or a handle of the wrong kind. */
mbrotli_encoder_session *mbrotli_encoder_session_create(int quality, int lgwin);
mbrotli_decoder_session *mbrotli_decoder_session_create(void);
void mbrotli_encoder_session_destroy(mbrotli_encoder_session *session);
void mbrotli_decoder_session_destroy(mbrotli_decoder_session *session);

/* Abandon the current stream and start an independent one, reusing retained
 * workspace and configuration. Returns OK, INVALID_PARAMETER for NULL or
 * misalignment, or ERROR on failure. May recover a failed session. */
mbrotli_result mbrotli_encoder_session_reinit(mbrotli_encoder_session *session);
mbrotli_result mbrotli_decoder_session_reinit(mbrotli_decoder_session *session);

/* Process into output which NEED NOT BE INITIALIZED.
 * On entry: *input_len = available bytes, *output_len = output capacity.
 * On return: counts are bytes consumed/produced by THIS call. Advance input
 * by the consumed count. Buffers are borrowed only for the duration of a call.
 * NULL buffers are allowed only with zero length. Count pointers must be
 * non-NULL, aligned and readable/writable. Input must be readable, output
 * writable; handle, buffers and count slots must be pairwise disjoint.
 * No concurrent access to handle/counts/output or mutation of input is allowed.
 *
 * Returns NEEDS_INPUT, NEEDS_OUTPUT, FINISHED, INVALID_PARAMETER or ERROR.
 * NEEDS_OUTPUT requires more output space even if all input was consumed.
 * Once FINISH is offered, repeat FINISH with exactly the unconsumed suffix
 * (same bytes); use the input-free finish only after all input was accepted.
 * Invalid arguments leave counts and session unchanged. On a caught panic
 * both counts are zero and output unspecified. ERROR requires reinit or destroy.
 * Unknown operation values, NULL/misaligned handles/counts, impossible lengths
 * and overlapping arguments are rejected. Other invalid pointers are UB.
 *
 * Encoder accepts PROCESS/FLUSH/FINISH. Repeat FLUSH until NEEDS_INPUT, then
 * input may resume. On ERROR both counts are zero and output unspecified.
 * Successful calls initialize exactly output[0..*output_len]; no later bytes
 * are written. Streaming does not use the one-shot bound or fallback rules.
 *
 * Decoder accepts PROCESS/FINISH. FINISH declares EOF (detects truncation).
 * FINISHED leaves any trailing bytes unconsumed. On success AND codec ERROR,
 * counts are exact and only output[0..*output_len] has been initialized.
 * No bytes past that prefix are written. */
mbrotli_result mbrotli_encoder_session_process(
    mbrotli_encoder_session *session, const uint8_t *input, size_t *input_len,
    uint8_t *output, size_t *output_len, int operation);
mbrotli_result mbrotli_decoder_session_process(
    mbrotli_decoder_session *session, const uint8_t *input, size_t *input_len,
    uint8_t *output, size_t *output_len, int operation);

/* Input-free shortcuts, with the same uninitialized-output/count contracts.
 * Repeat while NEEDS_OUTPUT is returned. Encoder flush uses FLUSH; decoder
 * flush uses PROCESS (drains accepted data without EOF). Both finish calls
 * use FINISH. Decoder flush after a pending FINISH is an invalid state. */
mbrotli_result mbrotli_encoder_session_flush(
    mbrotli_encoder_session *session, uint8_t *output, size_t *output_len);
mbrotli_result mbrotli_encoder_session_finish(
    mbrotli_encoder_session *session, uint8_t *output, size_t *output_len);
mbrotli_result mbrotli_decoder_session_flush(
    mbrotli_decoder_session *session, uint8_t *output, size_t *output_len);
mbrotli_result mbrotli_decoder_session_finish(
    mbrotli_decoder_session *session, uint8_t *output, size_t *output_len);

#ifdef __cplusplus
}
#endif

#endif /* MBROTLI_H */

#ifndef LZBENCH_PULSAR_H
#define LZBENCH_PULSAR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Frame format: 1-byte flag (0x01 = pulsar-coded, 0x00 = stored/raw copy)
 * followed by the payload. A stored frame is emitted when the input doesn't
 * compress, so this never returns 0 on success (lzbench treats <= 0 as a
 * compression error). Returns total bytes written, -1 on error. Panics are
 * caught inside and reported as -1, never crossing the FFI boundary. */
intptr_t pulsar_compress(const uint8_t *in, size_t in_len, uint8_t *out, size_t out_len);
/* Expects the frame format written by pulsar_compress.
 * Returns decoded length, -1 on error. */
intptr_t pulsar_decompress(const uint8_t *in, size_t in_len, uint8_t *out, size_t out_len);

#ifdef __cplusplus
}
#endif

#endif

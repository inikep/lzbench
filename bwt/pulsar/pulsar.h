#ifndef LZBENCH_PULSAR_H
#define LZBENCH_PULSAR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns encoded length, 0 if incompressible, -1 on error. */
intptr_t pulsar_compress(const uint8_t *in, size_t in_len, uint8_t *out, size_t out_len);
/* Returns decoded length, -1 on error. */
intptr_t pulsar_decompress(const uint8_t *in, size_t in_len, uint8_t *out, size_t out_len);

#ifdef __cplusplus
}
#endif

#endif

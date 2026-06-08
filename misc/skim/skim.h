#include <stddef.h>
#include <stdint.h>

typedef struct skim_encoder skim_encoder_t;
typedef struct skim_decoder skim_decoder_t;

skim_encoder_t *skim_encoder_create(void);

void skim_encoder_destroy(skim_encoder_t *encoder_ptr);

size_t skim_encoder_output_buffer_bound(size_t len);

size_t skim_encoder_compress(skim_encoder_t *encoder_ptr,
                             const uint8_t *input_ptr, size_t input_len,
                             uint8_t *output_ptr);

void skim_encoder_reset(skim_encoder_t *encoder_ptr);

skim_decoder_t *skim_decoder_create(void);

void skim_decoder_destroy(skim_decoder_t *decoder_ptr);

size_t skim_decoder_exact_output_length(const uint8_t *input_ptr,
                                        size_t input_len);

size_t skim_decoder_decompress(skim_decoder_t *, const uint8_t *input_ptr,
                               size_t input_len, uint8_t *output_ptr,
                               size_t output_len);

void skim_decoder_reset(skim_decoder_t *decoder_ptr);

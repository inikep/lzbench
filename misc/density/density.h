#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

uintptr_t chameleon_encode(const uint8_t *input,
                           uintptr_t input_size,
                           uint8_t *output,
                           uintptr_t output_size);

uintptr_t chameleon_decode(const uint8_t *input,
                           uintptr_t input_size,
                           uint8_t *output,
                           uintptr_t output_size);

uintptr_t chameleon_safe_encode_buffer_size(uintptr_t size);

uintptr_t cheetah_encode(const uint8_t *input,
                         uintptr_t input_size,
                         uint8_t *output,
                         uintptr_t output_size);

uintptr_t cheetah_decode(const uint8_t *input,
                         uintptr_t input_size,
                         uint8_t *output,
                         uintptr_t output_size);

uintptr_t cheetah_safe_encode_buffer_size(uintptr_t size);

uintptr_t lion_encode(const uint8_t *input,
                      uintptr_t input_size,
                      uint8_t *output,
                      uintptr_t output_size);

uintptr_t lion_decode(const uint8_t *input,
                      uintptr_t input_size,
                      uint8_t *output,
                      uintptr_t output_size);

uintptr_t lion_safe_encode_buffer_size(uintptr_t size);
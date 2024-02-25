#include <stdlib.h>
#include "common.h"

static const unsigned char common_characters[] = {
    0x20, 0x00, 0x30, 0x65, 0x69, 0x3e, 0x74, 0x6f,
    0x3c, 0x61, 0x6e, 0x73, 0xa, 0x72, 0x2f, 0x2e
};


static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}


void tamp_initialize_dictionary(unsigned char *buffer, size_t size){
    uint32_t seed = 3758097560;
    uint32_t randbuf = 0;
    for(size_t i=0; i < size; i++){
        if( TAMP_UNLIKELY((i & 0x7) == 0) )
            randbuf = xorshift32(&seed);
        buffer[i] = common_characters[randbuf & 0x0F];
        randbuf >>= 4;
    }
}


int8_t tamp_compute_min_pattern_size(uint8_t window, uint8_t literal) {
    return 2 + (window > (10 + ((literal - 5) << 1)));
}

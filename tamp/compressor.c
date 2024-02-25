#include "compressor.h"
#include "common.h"
#include <stdlib.h>
#include <stdbool.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define BUILD_BUG_ON(condition) ((void)sizeof(char[1 - 2*!!(condition)]))

#define MAX_PATTERN_SIZE (compressor->min_pattern_size + 13)
#define WINDOW_SIZE (1 << compressor->conf_window)
// 0xF because sizeof(TampCompressor.input) == 16;
#define input_add(offset) (\
            (compressor->input_pos + offset) & 0xF \
        )
#define read_input(offset) ( \
        compressor->input[input_add(offset)] \
        )
#define IS_LITERAL_FLAG (1 << compressor->conf_literal)

#define FLUSH_CODE (0xAB)

// encodes [min_pattern_bytes, min_pattern_bytes + 13] pattern lengths
static const unsigned char huffman_codes[] = {
    0x0, 0x3, 0x8, 0xb, 0x14, 0x24, 0x26, 0x2b, 0x4b, 0x54, 0x94, 0x95, 0xaa, 0x27
};
// These bit lengths pre-add the 1 bit for the 0-value is_literal flag.
static const uint8_t huffman_bits[] = {
    0x2, 0x3, 0x5, 0x5, 0x6, 0x7, 0x7, 0x7, 0x8, 0x8, 0x9, 0x9, 0x9, 0x7
};

static inline void write_to_bit_buffer(TampCompressor *compressor, uint32_t bits, uint8_t n_bits){
    compressor->bit_buffer_pos += n_bits;
    compressor->bit_buffer |= bits << (32 - compressor->bit_buffer_pos);
}

/**
 * @brief Partially flush the internal bit buffer.
 *
 * Up to 7 bits may remain in the internal bit buffer.
 */
static inline tamp_res partial_flush(TampCompressor *compressor, unsigned char *output, size_t output_size, size_t *output_written_size) {
    for (
        *output_written_size = output_size;
        compressor->bit_buffer_pos >= 8 && output_size;
        output_size--, compressor->bit_buffer_pos -= 8, compressor->bit_buffer <<= 8
        )
        *output++ = compressor->bit_buffer >> 24;
    *output_written_size -= output_size;
    return (compressor->bit_buffer_pos >= 8) ? TAMP_OUTPUT_FULL : TAMP_OK;
}

/**
 * @brief Find the best match for the current input buffer.
 *
 * @param[in,out] compressor TampCompressor object to perform search on.
 * @param[out] match_index  If match_size is 0, this value is undefined.
 * @param[out] match_size Size of best found match.
 */
static inline void find_best_match(
        TampCompressor *compressor,
        uint16_t *match_index,
        uint8_t *match_size
        ){
    *match_size = 0;

    if(TAMP_UNLIKELY(compressor->input_size < compressor->min_pattern_size))
        return;

    const uint16_t first_second = (read_input(0) << 8) | read_input(1);
    const uint16_t window_size_minus_1 = WINDOW_SIZE - 1;
    const uint8_t max_pattern_size = MIN(compressor->input_size, MAX_PATTERN_SIZE);
    uint16_t window_rolling_2_byte = compressor->window[0];
    unsigned char c;

    for(uint16_t window_index=0; window_index < window_size_minus_1; window_index++){
        window_rolling_2_byte <<= 8;
        window_rolling_2_byte |= compressor->window[window_index + 1];
        if(TAMP_LIKELY(window_rolling_2_byte != first_second)){
            continue;
        }

        for(uint8_t input_offset=2; ; input_offset++){
            if(TAMP_UNLIKELY(input_offset > *match_size)){
                *match_size = input_offset;
                *match_index = window_index;
                if(TAMP_UNLIKELY(*match_size == max_pattern_size))
                    return;
            }

            if(TAMP_UNLIKELY(window_index + input_offset > window_size_minus_1))
                return;

            c = read_input(input_offset);
            if(TAMP_LIKELY(compressor->window[window_index + input_offset] != c))
                break;
        }
    }
}

tamp_res tamp_compressor_init(TampCompressor *compressor, const TampConf *conf, unsigned char *window){
    const TampConf conf_default = {.window=10, .literal=8, .use_custom_dictionary=false};
    if(!conf)
        conf = &conf_default;
    if( conf->window < 8 || conf->window > 15)
        return TAMP_INVALID_CONF;
    if( conf->literal < 5 || conf->literal > 8)
        return TAMP_INVALID_CONF;

    for(uint8_t i=0; i < sizeof(TampCompressor); i++)  // Zero-out the struct
        ((unsigned char *)compressor)[i] = 0;

    compressor->conf_literal = conf->literal;
    compressor->conf_window = conf->window;
    compressor->conf_use_custom_dictionary = conf->use_custom_dictionary;

    compressor->window = window;
    compressor->min_pattern_size = tamp_compute_min_pattern_size(conf->window, conf->literal);

    if(!compressor->conf_use_custom_dictionary)
        tamp_initialize_dictionary(window, (1 << conf->window));

    // Write header to bit buffer
    write_to_bit_buffer(compressor, conf->window - 8, 3);
    write_to_bit_buffer(compressor, conf->literal - 5, 2);
    write_to_bit_buffer(compressor, conf->use_custom_dictionary, 1);
    write_to_bit_buffer(compressor, 0, 1);  // Reserved
    write_to_bit_buffer(compressor, 0, 1);  // No more header bytes

    return TAMP_OK;
}


tamp_res tamp_compressor_compress_poll(TampCompressor *compressor, unsigned char *output, size_t output_size, size_t *output_written_size){
    tamp_res res;
    const uint16_t window_mask = (1 << compressor->conf_window) - 1;
    size_t output_written_size_proxy;

    if(!output_written_size)
        output_written_size = &output_written_size_proxy;
    *output_written_size = 0;

    if(TAMP_UNLIKELY(compressor->input_size == 0))
        return TAMP_OK;

    {
        // Make sure there's enough room in the bit buffer.
        size_t flush_bytes_written;
        res = partial_flush(compressor, output, output_size, &flush_bytes_written);
        (*output_written_size) += flush_bytes_written;
        if(TAMP_UNLIKELY(res != TAMP_OK))
            return res;
        output_size -= flush_bytes_written;
        output += flush_bytes_written;
    }

    if(TAMP_UNLIKELY(output_size == 0))
        return TAMP_OUTPUT_FULL;

    uint8_t match_size = 0;
    uint16_t match_index = 0;
    find_best_match(compressor, &match_index, &match_size);

    if(TAMP_UNLIKELY(match_size < compressor->min_pattern_size)){
        // Write LITERAL
        match_size = 1;
        unsigned char c = read_input(0);
        if(TAMP_UNLIKELY(c >> compressor->conf_literal)){
            return TAMP_EXCESS_BITS;
        }
        write_to_bit_buffer(compressor, IS_LITERAL_FLAG | c, compressor->conf_literal + 1);
    }
    else{
        // Write TOKEN
        uint8_t huffman_index = match_size - compressor->min_pattern_size;
        write_to_bit_buffer(compressor, huffman_codes[huffman_index], huffman_bits[huffman_index]);
        write_to_bit_buffer(compressor, match_index, compressor->conf_window);
    }
    // Populate Window
    for(uint8_t i=0; i < match_size; i++){
        compressor->window[compressor->window_pos] = read_input(0);
        compressor->window_pos = (compressor->window_pos + 1) & window_mask;
        compressor->input_pos = input_add(1);
    }
    compressor->input_size -= match_size;

    return TAMP_OK;
}


void tamp_compressor_sink(
        TampCompressor *compressor,
        const unsigned char *input,
        size_t input_size,
        size_t *consumed_size
        ){
    size_t consumed_size_proxy;
    if(TAMP_LIKELY(consumed_size))
        *consumed_size = 0;
    else
        consumed_size = &consumed_size_proxy;

    for(size_t i=0; i < input_size; i++){
        if(TAMP_UNLIKELY(compressor->input_size == sizeof(compressor->input)))
            break;
        compressor->input[input_add(compressor->input_size)] = input[i];
        compressor->input_size += 1;
        (*consumed_size)++;
    }
}

tamp_res tamp_compressor_compress(
        TampCompressor *compressor,
        unsigned char *output,
        size_t output_size,
        size_t *output_written_size,
        const unsigned char *input,
        size_t input_size,
        size_t *input_consumed_size
        ){
    tamp_res res;
    size_t input_consumed_size_proxy, output_written_size_proxy;

    if(TAMP_LIKELY(output_written_size))
        *output_written_size = 0;
    else
        output_written_size = &output_written_size_proxy;

    if(TAMP_LIKELY(input_consumed_size))
        *input_consumed_size = 0;
    else
        input_consumed_size = &input_consumed_size_proxy;

    while(input_size > 0 && output_size > 0){
        {
            // Sink Data into input buffer.
            size_t consumed;
            tamp_compressor_sink(compressor, input, input_size, &consumed);
            input += consumed;
            input_size -= consumed;
            (*input_consumed_size) += consumed;
        }
        if(TAMP_LIKELY(compressor->input_size == sizeof(compressor->input))){
            // Input buffer is full and ready to start compressing.
            size_t chunk_output_written_size;
            res = tamp_compressor_compress_poll(compressor, output, output_size, &chunk_output_written_size);
            output += chunk_output_written_size;
            output_size -= chunk_output_written_size;
            (*output_written_size) += chunk_output_written_size;
            if(TAMP_UNLIKELY(res != TAMP_OK))
                return res;
        }
    }
    return TAMP_OK;
}

tamp_res tamp_compressor_flush(
        TampCompressor *compressor,
        unsigned char *output,
        size_t output_size,
        size_t *output_written_size,
        bool write_token
        ){
    tamp_res res;
    size_t chunk_output_written_size;
    size_t output_written_size_proxy;

    if(!output_written_size)
        output_written_size = &output_written_size_proxy;
    *output_written_size = 0;

    while(compressor->input_size){
        // Compress the remainder of the input buffer.
        res = tamp_compressor_compress_poll(compressor, output, output_size, &chunk_output_written_size);
        (*output_written_size) += chunk_output_written_size;
        if(TAMP_UNLIKELY(res != TAMP_OK))
            return res;
        output_size -= chunk_output_written_size;
        output += chunk_output_written_size;
    }

    // Perform partial flush to see if we need a FLUSH token, and to subsequently
    // make room for the FLUSH token.
    res = partial_flush(compressor, output, output_size, &chunk_output_written_size);
    output_size -= chunk_output_written_size;
    (*output_written_size) += chunk_output_written_size;
    output += chunk_output_written_size;
    if(TAMP_UNLIKELY(res != TAMP_OK))
        return res;

    // Check if there's enough output buffer space
    if (compressor->bit_buffer_pos){
        if (output_size == 0){
            return TAMP_OUTPUT_FULL;
        }
        if(write_token){
            if(output_size < 2)
                return TAMP_OUTPUT_FULL;
            write_to_bit_buffer(compressor, FLUSH_CODE, 9);
        }
    }

    // Flush the remainder of the output bit-buffer
    while(compressor->bit_buffer_pos){
        *output = compressor->bit_buffer >> 24;
        output++;
        compressor->bit_buffer <<= 8;
        compressor->bit_buffer_pos -= MIN(compressor->bit_buffer_pos, 8);
        output_size--;
        (*output_written_size)++;
    }

    return TAMP_OK;
}

tamp_res tamp_compressor_compress_and_flush(
        TampCompressor *compressor,
        unsigned char *output,
        size_t output_size,
        size_t *output_written_size,
        const unsigned char *input,
        size_t input_size,
        size_t *input_consumed_size,
        bool write_token
        ){
    tamp_res res;
    size_t flush_size;
    size_t output_written_size_proxy;

    if(!output_written_size)
        output_written_size = &output_written_size_proxy;

    res = tamp_compressor_compress(
            compressor,
            output,
            output_size,
            output_written_size,
            input,
            input_size,
            input_consumed_size
            );
    if(TAMP_UNLIKELY(res != TAMP_OK))
        return res;

    res = tamp_compressor_flush(
            compressor,
            output + *output_written_size,
            output_size - *output_written_size,
            &flush_size,
            write_token
            );

    (*output_written_size) += flush_size;

    if(TAMP_UNLIKELY(res != TAMP_OK))
        return res;

    return TAMP_OK;
}

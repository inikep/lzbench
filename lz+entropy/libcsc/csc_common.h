#ifndef _CSC_COMMON_H_
#define _CSC_COMMON_H_

#include <stdint.h>

//#ifdef _7Z_TYPES_
#include <Types.h>
//#else
//#endif

#define CSC_PROP_SIZE (4 + 3 + 3)

#define DECODE_ERROR (-96)
#define WRITE_ERROR (-97)
#define READ_ERROR (-98)

const size_t CSC_WRITE_ABORT = (size_t)-1;

typedef struct _CSCProps {
    // LZ77 dictionary size, 32 KB to 1GB, real size is 8KB less
    size_t dict_size;

    // compressed stream block size, leaving it as default is good
    uint32_t csc_blocksize; // must be < 16M

    // uncompressed block size, it determines how much data 
    // can be tried to process in one comp/decom call back.
    // default 1 * MB
    uint32_t raw_blocksize; // must be < 16M

    /******* Below only works while compression *****/
    // hash table match finder hash bits, advice value 17 - 24
    // depending on dict_size
    uint8_t hash_bits;

    // hash table match finder candidates num. Work well with lz_mode = 0 or 1.
    uint8_t hash_width;

    // binary tree entrance hash bits, advice value 17 - 24
    // depending on dict_size, 0 to disable bt match finder
    uint8_t bt_hash_bits;

    // Binary tree match finder working range, 
    // must be less than dict_size, 0 to disable bt match finder
    uint32_t bt_size;

    // max steps binary tree match finder will try 
    uint32_t bt_cyc;

    // default 32 should be good enough, does not benefit much with higher value
    uint8_t good_len;

    // lz engine mode:
    // 0 fastest, 1 efficient, 2 advanced ( 3+ times than 2)
    // for 0 and 1, better to turn bt match finder off
    // and 2 with bt match finder on
    uint8_t lz_mode;

    // 3 kinds of filters supported so far
    uint8_t DLTFilter;
    uint8_t TXTFilter;
    uint8_t EXEFilter;
} CSCProps;

#endif


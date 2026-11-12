/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Lempel Ziv Prediction                                     */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

   Copyright (c) 2009-2026 Ilya Grebnov <ilya.grebnov@gmail.com>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Please see the file LICENSE for full copyright information and file AUTHORS
for full list of contributors.

See also the bsc and libbsc web site:
  http://libbsc.com/ for more information.

--*/

#include <stdlib.h>
#include <memory.h>
#include <string.h>

#include "lzp.h"

#include "../adler32/adler32.h"
#include "../platform/platform.h"
#include "../libbsc.h"

#define LIBBSC_LZP_MATCH_FLAG 	0xf2

static INLINE int bsc_lzp_num_blocks(int n)
{
    struct entry { int blocks; int threshold; };

    static const struct entry break_points[] =
    {
        { 128, 128 * 128 * 65536 },
        {  96,  96 *  96 * 65536 },
        {  64,  64 *  64 * 65536 },
        {  48,  48 *  48 * 65536 },
        {  32,  32 *  32 * 65536 },
        {  24,  24 *  24 * 65536 },
        {  16,  16 *  16 * 65536 },
        {  12,  12 *  12 * 65536 },
        {   8,   8 *   8 * 65536 },
        {   6,   6 *   6 * 65536 },
        {   4,   4 *   4 * 65536 },
        {   2,   2 *   2 * 65536 },
    };

    for (int i = 0; i < sizeof(break_points) / sizeof(break_points[0]); i += 1)
    {
        if (n >= break_points[i].threshold) { return break_points[i].blocks; }
    }

    return 1;
}

#if !defined(LIBBSC_NO_UNALIGNED_ACCESS) && (defined(LIBBSC_x86_64) || defined(LIBBSC_AArch64))

template<class T> int bsc_lzp_encode_small(const unsigned char * RESTRICT input, const unsigned char * inputEnd, unsigned char * RESTRICT output, unsigned char * outputEnd, int * RESTRICT lookup, int mask)
{
    const unsigned char *   inputStart      = input;
    const unsigned char *   inputMinLenEnd  = inputEnd - sizeof(T) - 32;

    const unsigned char *   outputStart     = output;
    const unsigned char *   outputEOB       = outputEnd - 8;

    for (int i = 0; i < 4; ++i) { *output++ = *input++; }

    {
        while ((input < inputMinLenEnd) && (output < outputEOB))
        {
            unsigned long long next8 = *(unsigned long long *)(input - 4); *(unsigned int *)(output) = (unsigned int)(next8 >> 32); next8 = bsc_byteswap_uint64(next8);

            int value;
            {
                unsigned long long mix = (((next8 >> 12) ^ next8) >> 3) ^ next8;

                const unsigned int index0 = (mix >> (4 * 8)) & mask; value = lookup[index0]; lookup[index0] = (int)(input - inputStart + 0); 
                if ((*(T *)(input + 0) == *(T *)(inputStart + value)) && (value > 0)) goto LIBBSC_LZP_GOOD_MATCH_FOUND1;
                if (((unsigned char)(next8 >> 3 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND1;

                const unsigned int index1 = (mix >> (3 * 8)) & mask; value = lookup[index1]; lookup[index1] = (int)(input - inputStart + 1); 
                if ((*(T *)(input + 1) == *(T *)(inputStart + value)) && (value > 0)) goto LIBBSC_LZP_GOOD_MATCH_FOUND2;
                if (((unsigned char)(next8 >> 2 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND2;

                const unsigned int index2 = (mix >> (2 * 8)) & mask; value = lookup[index2]; lookup[index2] = (int)(input - inputStart + 2); 
                if ((*(T *)(input + 2) == *(T *)(inputStart + value)) && (value > 0)) goto LIBBSC_LZP_GOOD_MATCH_FOUND3;
                if (((unsigned char)(next8 >> 1 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND3;

                const unsigned int index3 = (mix >> (1 * 8)) & mask; value = lookup[index3]; lookup[index3] = (int)(input - inputStart + 3); 
                if ((*(T *)(input + 3) == *(T *)(inputStart + value)) && (value > 0)) goto LIBBSC_LZP_GOOD_MATCH_FOUND4;
                if (((unsigned char)(next8 >> 0 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND4;

                input += 4; output += 4;

                continue;
            }

LIBBSC_LZP_GOOD_MATCH_FOUND4:
            input += 1; output += 1;
LIBBSC_LZP_GOOD_MATCH_FOUND3:
            input += 1; output += 1;
LIBBSC_LZP_GOOD_MATCH_FOUND2:
            input += 1; output += 1;
LIBBSC_LZP_GOOD_MATCH_FOUND1:

            {
                const unsigned char * RESTRICT reference = inputStart + value;

                long long len = sizeof(T);

                for (; input + len < inputMinLenEnd; len += sizeof(unsigned long long))
                {
                    unsigned long long m;
                    if ((m = (*(unsigned long long *)(input + len)) ^ *(unsigned long long *)(reference + len)) != 0) 
                    {
                        len += bsc_bit_scan_forward64(m) / 8; break;
                    }
                }

                input += len; len -= sizeof(T);

                *output++ = LIBBSC_LZP_MATCH_FLAG; while (len >= 254) { len -= 254; *output++ = 254; if (output >= outputEOB) break; } *output++ = (unsigned char)(len); 
            
                continue;
            }

LIBBSC_LZP_BAD_MATCH_FOUND4:
            input += 4; output += 4; *output++ = 255; continue;
LIBBSC_LZP_BAD_MATCH_FOUND3:
            input += 3; output += 3; *output++ = 255; continue;
LIBBSC_LZP_BAD_MATCH_FOUND2:
            input += 2; output += 2; *output++ = 255; continue;
LIBBSC_LZP_BAD_MATCH_FOUND1:
            input += 1; output += 1; *output++ = 255; continue;
        }
    }
    
    {
        unsigned int context = input[-1] | (input[-2] << 8) | (input[-3] << 16) | (input[-4] << 24);

        while ((input < inputEnd) && (output < outputEOB))
        {
            unsigned int index = ((context >> 15) ^ context ^ (context >> 3)) & mask;
            int value = lookup[index]; lookup[index] = (int)(input - inputStart);

            unsigned char next = *output++ = *input++; context = (context << 8) | next;
            if (next == LIBBSC_LZP_MATCH_FLAG && value > 0) *output++ = 255;
        }
    }

    return (output >= outputEOB) ? LIBBSC_NOT_COMPRESSIBLE : (int)(output - outputStart);
}

template<class T> int bsc_lzp_encode_small2x(const unsigned char * RESTRICT input, const unsigned char * inputEnd, unsigned char * RESTRICT output, unsigned char * outputEnd, int * RESTRICT lookup, int mask)
{
    const unsigned char *   inputStart      = input;
    const unsigned char *   inputMinLenEnd  = inputEnd - sizeof(T) - sizeof(T) - 32;

    const unsigned char *   outputStart     = output;
    const unsigned char *   outputEOB       = outputEnd - 8;

    for (int i = 0; i < 4; ++i) { *output++ = *input++; }

    {
        while ((input < inputMinLenEnd) && (output < outputEOB))
        {
            unsigned long long next8 = *(unsigned long long *)(input - 4); *(unsigned int *)(output) = (unsigned int)(next8 >> 32); next8 = bsc_byteswap_uint64(next8);

            int value;
            {
                unsigned long long mix = (((next8 >> 12) ^ next8) >> 3) ^ next8;

                const unsigned int index0 = (mix >> (4 * 8)) & mask; value = lookup[index0]; lookup[index0] = (int)(input - inputStart + 0); 
                if ((*(T *)(input + sizeof(T) + 0) == *(T *)(inputStart + value + sizeof(T))) && (value > 0) && (*(T *)(input + 0) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND1;
                if (((unsigned char)(next8 >> 3 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND1;

                const unsigned int index1 = (mix >> (3 * 8)) & mask; value = lookup[index1]; lookup[index1] = (int)(input - inputStart + 1); 
                if ((*(T *)(input + sizeof(T) + 1) == *(T *)(inputStart + value + sizeof(T))) && (value > 0) && (*(T *)(input + 1) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND2;
                if (((unsigned char)(next8 >> 2 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND2;

                const unsigned int index2 = (mix >> (2 * 8)) & mask; value = lookup[index2]; lookup[index2] = (int)(input - inputStart + 2); 
                if ((*(T *)(input + sizeof(T) + 2) == *(T *)(inputStart + value + sizeof(T))) && (value > 0) && (*(T *)(input + 2) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND3;
                if (((unsigned char)(next8 >> 1 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND3;

                const unsigned int index3 = (mix >> (1 * 8)) & mask; value = lookup[index3]; lookup[index3] = (int)(input - inputStart + 3); 
                if ((*(T *)(input + sizeof(T) + 3) == *(T *)(inputStart + value + sizeof(T))) && (value > 0) && (*(T *)(input + 3) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND4;
                if (((unsigned char)(next8 >> 0 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND4;

                input += 4; output += 4;

                continue;
            }

LIBBSC_LZP_GOOD_MATCH_FOUND4:
            input += 1; output += 1;
LIBBSC_LZP_GOOD_MATCH_FOUND3:
            input += 1; output += 1;
LIBBSC_LZP_GOOD_MATCH_FOUND2:
            input += 1; output += 1;
LIBBSC_LZP_GOOD_MATCH_FOUND1:

            {
                const unsigned char * RESTRICT reference = inputStart + value;

                long long len = sizeof(T) + sizeof(T);

                for (; input + len < inputMinLenEnd; len += sizeof(unsigned long long))
                {
                    unsigned long long m;
                    if ((m = (*(unsigned long long *)(input + len)) ^ *(unsigned long long *)(reference + len)) != 0) 
                    {
                        len += bsc_bit_scan_forward64(m) / 8; break;
                    }
                }

                input += len; len -= sizeof(T) + sizeof(T);

                *output++ = LIBBSC_LZP_MATCH_FLAG; while (len >= 254) { len -= 254; *output++ = 254; if (output >= outputEOB) break; } *output++ = (unsigned char)(len); 
            
                continue;
            }

LIBBSC_LZP_BAD_MATCH_FOUND4:
            input += 4; output += 4; *output++ = 255; continue;
LIBBSC_LZP_BAD_MATCH_FOUND3:
            input += 3; output += 3; *output++ = 255; continue;
LIBBSC_LZP_BAD_MATCH_FOUND2:
            input += 2; output += 2; *output++ = 255; continue;
LIBBSC_LZP_BAD_MATCH_FOUND1:
            input += 1; output += 1; *output++ = 255; continue;
        }
    }
    
    {
        unsigned int context = input[-1] | (input[-2] << 8) | (input[-3] << 16) | (input[-4] << 24);

        while ((input < inputEnd) && (output < outputEOB))
        {
            unsigned int index = ((context >> 15) ^ context ^ (context >> 3)) & mask;
            int value = lookup[index]; lookup[index] = (int)(input - inputStart);

            unsigned char next = *output++ = *input++; context = (context << 8) | next;
            if (next == LIBBSC_LZP_MATCH_FLAG && value > 0) *output++ = 255;
        }
    }

    return (output >= outputEOB) ? LIBBSC_NOT_COMPRESSIBLE : (int)(output - outputStart);
}

template<class T> int bsc_lzp_encode_medium(const unsigned char * RESTRICT input, const unsigned char * inputEnd, unsigned char * RESTRICT output, unsigned char * outputEnd, int * RESTRICT lookup, int mask, int minLen)
{
    const unsigned char *   inputStart      = input;
    const unsigned char *   inputMinLenEnd  = inputEnd - sizeof(T) - sizeof(T) - 32;

    const unsigned char *   outputStart     = output;
    const unsigned char *   outputEOB       = outputEnd - 8;

    for (int i = 0; i < 4; ++i) { *output++ = *input++; }

    {
        while ((input < inputMinLenEnd) && (output < outputEOB))
        {
            unsigned long long next8 = *(unsigned long long *)(input - 4); *(unsigned int *)(output) = (unsigned int)(next8 >> 32); next8 = bsc_byteswap_uint64(next8);

            int value;
            {
                unsigned long long mix = (((next8 >> 12) ^ next8) >> 3) ^ next8;

                const unsigned int index0 = (mix >> (4 * 8)) & mask; value = lookup[index0]; lookup[index0] = (int)(input - inputStart + 0); 
                if ((*(T *)(input + minLen - sizeof(T) + 0) == *(T *)(inputStart + value + minLen - sizeof(T))) && (value > 0) && (*(T *)(input + 0) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND1;
                if (((unsigned char)(next8 >> 3 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND1;

                const unsigned int index1 = (mix >> (3 * 8)) & mask; value = lookup[index1]; lookup[index1] = (int)(input - inputStart + 1); 
                if ((*(T *)(input + minLen - sizeof(T) + 1) == *(T *)(inputStart + value + minLen - sizeof(T))) && (value > 0) && (*(T *)(input + 1) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND2;
                if (((unsigned char)(next8 >> 2 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND2;

                const unsigned int index2 = (mix >> (2 * 8)) & mask; value = lookup[index2]; lookup[index2] = (int)(input - inputStart + 2); 
                if ((*(T *)(input + minLen - sizeof(T) + 2) == *(T *)(inputStart + value + minLen - sizeof(T))) && (value > 0) && (*(T *)(input + 2) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND3;
                if (((unsigned char)(next8 >> 1 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND3;

                const unsigned int index3 = (mix >> (1 * 8)) & mask; value = lookup[index3]; lookup[index3] = (int)(input - inputStart + 3); 
                if ((*(T *)(input + minLen - sizeof(T) + 3) == *(T *)(inputStart + value + minLen - sizeof(T))) && (value > 0) && (*(T *)(input + 3) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND4;
                if (((unsigned char)(next8 >> 0 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND4;

                input += 4; output += 4;

                continue;
            }

LIBBSC_LZP_GOOD_MATCH_FOUND4:
            input += 1; output += 1;
LIBBSC_LZP_GOOD_MATCH_FOUND3:
            input += 1; output += 1;
LIBBSC_LZP_GOOD_MATCH_FOUND2:
            input += 1; output += 1;
LIBBSC_LZP_GOOD_MATCH_FOUND1:

            {
                const unsigned char * RESTRICT reference = inputStart + value;

                long long len = minLen;

                for (; input + len < inputMinLenEnd; len += sizeof(unsigned long long))
                {
                    unsigned long long m;
                    if ((m = (*(unsigned long long *)(input + len)) ^ *(unsigned long long *)(reference + len)) != 0) 
                    {
                        len += bsc_bit_scan_forward64(m) / 8; break;
                    }
                }

                input += len; len -= minLen;

                *output++ = LIBBSC_LZP_MATCH_FLAG; while (len >= 254) { len -= 254; *output++ = 254; if (output >= outputEOB) break; } *output++ = (unsigned char)(len); 
            
                continue;
            }

LIBBSC_LZP_BAD_MATCH_FOUND4:
            input += 4; output += 4; *output++ = 255; continue;
LIBBSC_LZP_BAD_MATCH_FOUND3:
            input += 3; output += 3; *output++ = 255; continue;
LIBBSC_LZP_BAD_MATCH_FOUND2:
            input += 2; output += 2; *output++ = 255; continue;
LIBBSC_LZP_BAD_MATCH_FOUND1:
            input += 1; output += 1; *output++ = 255; continue;
        }
    }
    
    {
        unsigned int context = input[-1] | (input[-2] << 8) | (input[-3] << 16) | (input[-4] << 24);

        while ((input < inputEnd) && (output < outputEOB))
        {
            unsigned int index = ((context >> 15) ^ context ^ (context >> 3)) & mask;
            int value = lookup[index]; lookup[index] = (int)(input - inputStart);

            unsigned char next = *output++ = *input++; context = (context << 8) | next;
            if (next == LIBBSC_LZP_MATCH_FLAG && value > 0) *output++ = 255;
        }
    }

    return (output >= outputEOB) ? LIBBSC_NOT_COMPRESSIBLE : (int)(output - outputStart);
}

template<class T> int bsc_lzp_encode_large(const unsigned char * RESTRICT input, const unsigned char * inputEnd, unsigned char * RESTRICT output, unsigned char * outputEnd, int * RESTRICT lookup, int mask, int minLen)
{
    const unsigned char * inputStart     = input;
    const unsigned char * outputStart    = output;
    const unsigned char * outputEOB      = outputEnd - 8;
    const unsigned char * inputMinLenEnd = inputEnd - minLen - 32;

    const unsigned char * heuristic      = input;
    unsigned long long    heuristic_v    = (input < inputMinLenEnd) ? *(unsigned long long *)heuristic : 0;

    for (int i = 0; i < 4; ++i) { *output++ = *input++; }

    {
        while ((input < inputMinLenEnd) && (output < outputEOB))
        {
            unsigned long long next8 = *(unsigned long long *)(input - 4); *(unsigned int *)(output) = (unsigned int)(next8 >> 32); next8 = bsc_byteswap_uint64(next8);

            int value;
            {
                unsigned long long mix = (((next8 >> 12) ^ next8) >> 3) ^ next8;

                const unsigned int index0 = (mix >> (4 * 8)) & mask; value = lookup[index0]; lookup[index0] = (int)(input - inputStart + 0);
                if ((*(T *)(input + minLen - sizeof(T) + 0) == *(T *)(inputStart + value + minLen - sizeof(T))) && (value > 0) && (*(T *)(input + 0) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND1;
                if (((unsigned char)(next8 >> 3 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND1;

                const unsigned int index1 = (mix >> (3 * 8)) & mask; value = lookup[index1]; lookup[index1] = (int)(input - inputStart + 1);
                if ((*(T *)(input + minLen - sizeof(T) + 1) == *(T *)(inputStart + value + minLen - sizeof(T))) && (value > 0) && (*(T *)(input + 1) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND2;
                if (((unsigned char)(next8 >> 2 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND2;

                const unsigned int index2 = (mix >> (2 * 8)) & mask; value = lookup[index2]; lookup[index2] = (int)(input - inputStart + 2);
                if ((*(T *)(input + minLen - sizeof(T) + 2) == *(T *)(inputStart + value + minLen - sizeof(T))) && (value > 0) && (*(T *)(input + 2) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND3;
                if (((unsigned char)(next8 >> 1 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND3;

                const unsigned int index3 = (mix >> (1 * 8)) & mask; value = lookup[index3]; lookup[index3] = (int)(input - inputStart + 3);
                if ((*(T *)(input + minLen - sizeof(T) + 3) == *(T *)(inputStart + value + minLen - sizeof(T))) && (value > 0) && (*(T *)(input + 3) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND4;
                if (((unsigned char)(next8 >> 0 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND4;

                input += 4; output += 4;

                continue;
            }

LIBBSC_LZP_GOOD_MATCH_FOUND4:
            input += 1; output += 1;
LIBBSC_LZP_GOOD_MATCH_FOUND3:
            input += 1; output += 1;
LIBBSC_LZP_GOOD_MATCH_FOUND2:
            input += 1; output += 1;
LIBBSC_LZP_GOOD_MATCH_FOUND1:

            {
                const unsigned char * RESTRICT reference = inputStart + value;

                if (((heuristic - input) > 0) && (heuristic_v != *(unsigned long long *)(reference + (heuristic - input))))
                {
                    goto LIBBSC_LZP_MATCH_NOT_FOUND;
                }

                long long len = sizeof(T);

                for (; input + len < inputMinLenEnd; len += sizeof(unsigned long long))
                {
                    unsigned long long m;
                    if ((m = (*(unsigned long long *)(input + len)) ^ *(unsigned long long *)(reference + len)) != 0) 
                    {
                        len += bsc_bit_scan_forward64(m) / 8; break;
                    }
                }

                if (len < minLen) { heuristic = input + len; heuristic_v = *(unsigned long long *)heuristic; goto LIBBSC_LZP_MATCH_NOT_FOUND; }

                input += len; len -= minLen;

                *output++ = LIBBSC_LZP_MATCH_FLAG; while (len >= 254) { len -= 254; *output++ = 254; if (output >= outputEOB) break; } *output++ = (unsigned char)(len); 
            
                continue;
            }

LIBBSC_LZP_MATCH_NOT_FOUND:
            input += 1; if (*output++ == LIBBSC_LZP_MATCH_FLAG) { *output++ = 255; } continue;
LIBBSC_LZP_BAD_MATCH_FOUND4:
            input += 4; output += 4; *output++ = 255; continue;
LIBBSC_LZP_BAD_MATCH_FOUND3:
            input += 3; output += 3; *output++ = 255; continue;
LIBBSC_LZP_BAD_MATCH_FOUND2:
            input += 2; output += 2; *output++ = 255; continue;
LIBBSC_LZP_BAD_MATCH_FOUND1:
            input += 1; output += 1; *output++ = 255; continue;
        }        
    }
    
    {
        unsigned int context = input[-1] | (input[-2] << 8) | (input[-3] << 16) | (input[-4] << 24);

        while ((input < inputEnd) && (output < outputEOB))
        {
            unsigned int index = ((context >> 15) ^ context ^ (context >> 3)) & mask;
            int value = lookup[index]; lookup[index] = (int)(input - inputStart);

            unsigned char next = *output++ = *input++; context = (context << 8) | next;
            if (next == LIBBSC_LZP_MATCH_FLAG && value > 0) *output++ = 255;
        }
    }

    return (output >= outputEOB) ? LIBBSC_NOT_COMPRESSIBLE : (int)(output - outputStart);
}

int bsc_lzp_encode_large_fast_path(const unsigned char * RESTRICT input, const unsigned char * inputEnd, unsigned char * RESTRICT output, unsigned char * outputEnd, int * RESTRICT lookup)
{
    typedef unsigned long long T;

    const int mask      = (int)(1 << LIBBSC_DEFAULT_LZPHASHSIZE) - 1;
    const int minLen    = LIBBSC_DEFAULT_LZPMINLEN;

    const unsigned char * inputStart     = input;
    const unsigned char * outputStart    = output;
    const unsigned char * outputEOB      = outputEnd - 8;
    const unsigned char * inputMinLenEnd = inputEnd - minLen - 32;

    const unsigned char * heuristic      = input;
    unsigned long long    heuristic_v    = (input < inputMinLenEnd) ? *(unsigned long long *)heuristic : 0;

    for (int i = 0; i < 4; ++i) { *output++ = *input++; }

    {
        while ((input < inputMinLenEnd) && (output < outputEOB))
        {
            unsigned long long next8 = *(unsigned long long *)(input - 4); *(unsigned int *)(output) = (unsigned int)(next8 >> 32); next8 = bsc_byteswap_uint64(next8);

            int value;
            {
                unsigned long long mix = (((next8 >> 12) ^ next8) >> 3) ^ next8;

                const unsigned int index0 = (mix >> (4 * 8)) & mask; value = lookup[index0]; lookup[index0] = (int)(input - inputStart + 0);
                if ((*(T *)(input + minLen - sizeof(T) + 0) == *(T *)(inputStart + value + minLen - sizeof(T))) && (value > 0) && (*(T *)(input + 0) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND1;
                if (((unsigned char)(next8 >> 3 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND1;

                const unsigned int index1 = (mix >> (3 * 8)) & mask; value = lookup[index1]; lookup[index1] = (int)(input - inputStart + 1);
                if ((*(T *)(input + minLen - sizeof(T) + 1) == *(T *)(inputStart + value + minLen - sizeof(T))) && (value > 0) && (*(T *)(input + 1) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND2;
                if (((unsigned char)(next8 >> 2 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND2;

                const unsigned int index2 = (mix >> (2 * 8)) & mask; value = lookup[index2]; lookup[index2] = (int)(input - inputStart + 2);
                if ((*(T *)(input + minLen - sizeof(T) + 2) == *(T *)(inputStart + value + minLen - sizeof(T))) && (value > 0) && (*(T *)(input + 2) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND3;
                if (((unsigned char)(next8 >> 1 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND3;

                const unsigned int index3 = (mix >> (1 * 8)) & mask; value = lookup[index3]; lookup[index3] = (int)(input - inputStart + 3);
                if ((*(T *)(input + minLen - sizeof(T) + 3) == *(T *)(inputStart + value + minLen - sizeof(T))) && (value > 0) && (*(T *)(input + 3) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND4;
                if (((unsigned char)(next8 >> 0 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_BAD_MATCH_FOUND4;

                input += 4; output += 4;

                continue;
            }

LIBBSC_LZP_GOOD_MATCH_FOUND4:
            input += 1; output += 1;
LIBBSC_LZP_GOOD_MATCH_FOUND3:
            input += 1; output += 1;
LIBBSC_LZP_GOOD_MATCH_FOUND2:
            input += 1; output += 1;
LIBBSC_LZP_GOOD_MATCH_FOUND1:

            {
                const unsigned char * RESTRICT reference = inputStart + value;

                if (((heuristic - input) > 0) && (heuristic_v != *(unsigned long long *)(reference + (heuristic - input))))
                {
                    goto LIBBSC_LZP_MATCH_NOT_FOUND;
                }

                long long len = sizeof(T);

                for (; input + len < inputMinLenEnd; len += sizeof(unsigned long long))
                {
                    unsigned long long m;
                    if ((m = (*(unsigned long long *)(input + len)) ^ *(unsigned long long *)(reference + len)) != 0) 
                    {
                        len += bsc_bit_scan_forward64(m) / 8; break;
                    }
                }

                if (len < minLen) { heuristic = input + len; heuristic_v = *(unsigned long long *)heuristic; goto LIBBSC_LZP_MATCH_NOT_FOUND; }

                input += len; len -= minLen;

                *output++ = LIBBSC_LZP_MATCH_FLAG; while (len >= 254) { len -= 254; *output++ = 254; if (output >= outputEOB) break; } *output++ = (unsigned char)(len); 
            
                continue;
            }

LIBBSC_LZP_MATCH_NOT_FOUND:
            input += 1; if (*output++ == LIBBSC_LZP_MATCH_FLAG) { *output++ = 255; } continue;
LIBBSC_LZP_BAD_MATCH_FOUND4:
            input += 4; output += 4; *output++ = 255; continue;
LIBBSC_LZP_BAD_MATCH_FOUND3:
            input += 3; output += 3; *output++ = 255; continue;
LIBBSC_LZP_BAD_MATCH_FOUND2:
            input += 2; output += 2; *output++ = 255; continue;
LIBBSC_LZP_BAD_MATCH_FOUND1:
            input += 1; output += 1; *output++ = 255; continue;
        }        
    }
    
    {
        unsigned int context = input[-1] | (input[-2] << 8) | (input[-3] << 16) | (input[-4] << 24);

        while ((input < inputEnd) && (output < outputEOB))
        {
            unsigned int index = ((context >> 15) ^ context ^ (context >> 3)) & mask;
            int value = lookup[index]; lookup[index] = (int)(input - inputStart);

            unsigned char next = *output++ = *input++; context = (context << 8) | next;
            if (next == LIBBSC_LZP_MATCH_FLAG && value > 0) *output++ = 255;
        }
    }

    return (output >= outputEOB) ? LIBBSC_NOT_COMPRESSIBLE : (int)(output - outputStart);
}

#endif

int bsc_lzp_encode_generic(const unsigned char * RESTRICT input, const unsigned char * inputEnd, unsigned char * RESTRICT output, unsigned char * outputEnd, int * RESTRICT lookup, int mask, int minLen)
{
    const unsigned char *   inputStart  = input;
    const unsigned char *   outputStart = output;
    const unsigned char *   outputEOB   = outputEnd - 8;

    const unsigned char * heuristic      = input;
    const unsigned char * inputMinLenEnd = inputEnd - minLen - 32;

    for (int i = 0; i < 4; ++i) { *output++ = *input++; }

    {
        unsigned int context = input[-1] | (input[-2] << 8) | (input[-3] << 16) | (input[-4] << 24);

        while ((input < inputMinLenEnd) && (output < outputEOB))
        {
            unsigned int index = ((context >> 15) ^ context ^ (context >> 3)) & mask;
            int value = lookup[index]; lookup[index] = (int)(input - inputStart);
            if (value > 0)
            {
                const unsigned char * RESTRICT reference = inputStart + value;
#ifndef LIBBSC_NO_UNALIGNED_ACCESS
                if ((*(unsigned int *)(input + minLen - 4) == *(unsigned int *)(reference + minLen - 4)) && (*(unsigned int *)(input) == *(unsigned int *)(reference)))
#else
                if ((memcmp(input + minLen - 4, reference + minLen - 4, sizeof(unsigned int)) == 0) && (memcmp(input, reference, sizeof(unsigned int)) == 0))
#endif
                {
#ifndef LIBBSC_NO_UNALIGNED_ACCESS
                    if ((heuristic > input) && (*(unsigned int *)heuristic != *(unsigned int *)(reference + (heuristic - input))))
#else
                    if ((heuristic > input) && (memcmp(heuristic, reference + (heuristic - input), sizeof(unsigned int)) != 0))
#endif
                    {
                        goto LIBBSC_LZP_MATCH_NOT_FOUND;
                    }

                    int len = 4;
                    for (; input + len < inputMinLenEnd; len += sizeof(unsigned int))
                    {
#ifndef LIBBSC_NO_UNALIGNED_ACCESS
                        if (*(unsigned int *)(input + len) != *(unsigned int *)(reference + len)) break;
#else
                        if (memcmp(input + len, reference + len, sizeof(unsigned int)) != 0) break;
#endif
                    }

                    if (len < minLen) { heuristic = input + len; goto LIBBSC_LZP_MATCH_NOT_FOUND; }

#ifndef LIBBSC_NO_UNALIGNED_ACCESS
                    len += sizeof(unsigned short) * (*(unsigned short *)(input + len) == *(unsigned short *)(reference + len));
                    len += sizeof(unsigned char ) * (*(unsigned char  *)(input + len) == *(unsigned char  *)(reference + len));
#else
                    len += input[len] == reference[len];
                    len += input[len] == reference[len];
                    len += input[len] == reference[len];
#endif

                    input += len; context = input[-1] | (input[-2] << 8) | (input[-3] << 16) | (input[-4] << 24);

                    *output++ = LIBBSC_LZP_MATCH_FLAG;

                    len -= minLen; while (len >= 254) { len -= 254; *output++ = 254; if (output >= outputEOB) break; }

                    *output++ = (unsigned char)(len);
                }
                else
                {

LIBBSC_LZP_MATCH_NOT_FOUND:
                    unsigned char next = *output++ = *input++; context = (context << 8) | next;
                    if (next == LIBBSC_LZP_MATCH_FLAG) *output++ = 255;
                }
            }
            else
            {
                context = (context << 8) | (*output++ = *input++);
            }
        }
    }
    
    {
        unsigned int context = input[-1] | (input[-2] << 8) | (input[-3] << 16) | (input[-4] << 24);

        while ((input < inputEnd) && (output < outputEOB))
        {
            unsigned int index = ((context >> 15) ^ context ^ (context >> 3)) & mask;
            int value = lookup[index]; lookup[index] = (int)(input - inputStart);

            unsigned char next = *output++ = *input++; context = (context << 8) | next;
            if (next == LIBBSC_LZP_MATCH_FLAG && value > 0) *output++ = 255;
        }
    }

    return (output >= outputEOB) ? LIBBSC_NOT_COMPRESSIBLE : (int)(output - outputStart);
}

int bsc_lzp_encode_block(const unsigned char * input, const unsigned char * inputEnd, unsigned char * output, unsigned char * outputEnd, int * table, int hashSize, int minLen)
{
    if (inputEnd - input - minLen < 32)
    {
        return LIBBSC_NOT_COMPRESSIBLE;
    }

    int result = LIBBSC_NOT_ENOUGH_MEMORY;
    if (int * lookup = table != NULL ? table : (int *)bsc_zero_malloc(((size_t)1 << hashSize) * sizeof(int)))
    {
#if !defined(LIBBSC_NO_UNALIGNED_ACCESS) && (defined(LIBBSC_x86_64) || defined(LIBBSC_AArch64))
        if (hashSize <= 17)
        {
            result = (minLen == 1 * (int)sizeof(unsigned int      ) && result == LIBBSC_NOT_ENOUGH_MEMORY) ? bsc_lzp_encode_small  <unsigned int      >(input, inputEnd, output, outputEnd, lookup, (int)(1 << hashSize) - 1) : result;
            result = (minLen == 1 * (int)sizeof(unsigned long long) && result == LIBBSC_NOT_ENOUGH_MEMORY) ? bsc_lzp_encode_small  <unsigned long long>(input, inputEnd, output, outputEnd, lookup, (int)(1 << hashSize) - 1) : result;
            result = (minLen == 2 * (int)sizeof(unsigned long long) && result == LIBBSC_NOT_ENOUGH_MEMORY) ? bsc_lzp_encode_small2x<unsigned long long>(input, inputEnd, output, outputEnd, lookup, (int)(1 << hashSize) - 1) : result;
            result = (minLen <= 2 * (int)sizeof(unsigned int      ) && result == LIBBSC_NOT_ENOUGH_MEMORY) ? bsc_lzp_encode_medium <unsigned int      >(input, inputEnd, output, outputEnd, lookup, (int)(1 << hashSize) - 1, minLen) : result;
            result = (minLen <= 2 * (int)sizeof(unsigned long long) && result == LIBBSC_NOT_ENOUGH_MEMORY) ? bsc_lzp_encode_medium <unsigned long long>(input, inputEnd, output, outputEnd, lookup, (int)(1 << hashSize) - 1, minLen) : result;

            if (result == LIBBSC_NOT_ENOUGH_MEMORY)
            {
                result = hashSize == LIBBSC_DEFAULT_LZPHASHSIZE && minLen == LIBBSC_DEFAULT_LZPMINLEN
                    ? bsc_lzp_encode_large_fast_path(input, inputEnd, output, outputEnd, lookup)
                    : bsc_lzp_encode_large<unsigned long long>(input, inputEnd, output, outputEnd, lookup, (int)(1 << hashSize) - 1, minLen);
            }
        }
#endif

        result = result == LIBBSC_NOT_ENOUGH_MEMORY ? bsc_lzp_encode_generic(input, inputEnd, output, outputEnd, lookup, (int)(1 << hashSize) - 1, minLen) : result;

        if (table == NULL) bsc_free(lookup);
    }

    return result;
}

int bsc_lzp_decode_block(const unsigned char * RESTRICT input, const unsigned char * inputEnd, unsigned char * RESTRICT output, int * table, int hashSize, int minLen)
{
    if (inputEnd - input < 4)
    {
        return LIBBSC_UNEXPECTED_EOB;
    }

    if (int * RESTRICT lookup = table != NULL ? table : (int *)bsc_zero_malloc(((size_t)1 << hashSize) * sizeof(int)))
    {
        unsigned int            mask        = (int)(1 << hashSize) - 1;
        const unsigned char *   outputStart = output;

        for (int i = 0; i < 4; ++i) { *output++ = *input++; }

#if !defined(LIBBSC_NO_UNALIGNED_ACCESS) && (defined(LIBBSC_x86_64) || defined(LIBBSC_AArch64))
        if (hashSize <= 17)
        {
            unsigned int prev4 = *(unsigned int *)(output - 4);

            while (input < inputEnd - 8)
            {
#if LIBBSC_CPU_FEATURE >= LIBBSC_CPU_FEATURE_SSE2
                if (input < inputEnd - 16)
                {
                    __m128i next16 = _mm_loadu_si128((const __m128i *)input);
                    if (_mm_movemask_epi8(_mm_cmpeq_epi8(next16, _mm_set1_epi8((char)LIBBSC_LZP_MATCH_FLAG))) == 0)
                    {
                        _mm_storeu_si128((__m128i *)output, next16);

                        unsigned long long lo8      = *(unsigned long long *)(input + 0);
                        unsigned long long hi8      = *(unsigned long long *)(input + 8);

                        unsigned long long next80   = bsc_byteswap_uint64((lo8 << 32) | prev4);
                        unsigned long long next81   = bsc_byteswap_uint64(lo8);
                        unsigned long long next82   = bsc_byteswap_uint64((hi8 << 32) | (lo8 >> 32));
                        unsigned long long next83   = bsc_byteswap_uint64(hi8);

                        unsigned long long mix0     = (((next80 >> 12) ^ next80) >> 3) ^ next80;
                        unsigned long long mix1     = (((next81 >> 12) ^ next81) >> 3) ^ next81;
                        unsigned long long mix2     = (((next82 >> 12) ^ next82) >> 3) ^ next82;
                        unsigned long long mix3     = (((next83 >> 12) ^ next83) >> 3) ^ next83;

                        lookup[(mix0 >> (4 * 8)) & mask] = (int)(output - outputStart +  0);
                        lookup[(mix0 >> (3 * 8)) & mask] = (int)(output - outputStart +  1);
                        lookup[(mix0 >> (2 * 8)) & mask] = (int)(output - outputStart +  2);
                        lookup[(mix0 >> (1 * 8)) & mask] = (int)(output - outputStart +  3);
                        lookup[(mix1 >> (4 * 8)) & mask] = (int)(output - outputStart +  4);
                        lookup[(mix1 >> (3 * 8)) & mask] = (int)(output - outputStart +  5);
                        lookup[(mix1 >> (2 * 8)) & mask] = (int)(output - outputStart +  6);
                        lookup[(mix1 >> (1 * 8)) & mask] = (int)(output - outputStart +  7);
                        lookup[(mix2 >> (4 * 8)) & mask] = (int)(output - outputStart +  8);
                        lookup[(mix2 >> (3 * 8)) & mask] = (int)(output - outputStart +  9);
                        lookup[(mix2 >> (2 * 8)) & mask] = (int)(output - outputStart + 10);
                        lookup[(mix2 >> (1 * 8)) & mask] = (int)(output - outputStart + 11);
                        lookup[(mix3 >> (4 * 8)) & mask] = (int)(output - outputStart + 12);
                        lookup[(mix3 >> (3 * 8)) & mask] = (int)(output - outputStart + 13);
                        lookup[(mix3 >> (2 * 8)) & mask] = (int)(output - outputStart + 14);
                        lookup[(mix3 >> (1 * 8)) & mask] = (int)(output - outputStart + 15);

                        prev4 = (unsigned int)(hi8 >> 32); input += 16; output += 16;

                        continue;
                    }
                }
#elif LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_A64
                {
                    unsigned long long next8    = *(unsigned long long *)(input);
                    unsigned long long flags    = next8 ^ (LIBBSC_LZP_MATCH_FLAG * 0x0101010101010101ull);

                    if (((flags - 0x0101010101010101ull) & ~flags & 0x8080808080808080ull) == 0)
                    {
                        *(unsigned long long *)(output) = next8;

                        unsigned long long next80   = bsc_byteswap_uint64((next8 << 32) | prev4);
                        unsigned long long next81   = bsc_byteswap_uint64(next8);

                        unsigned long long mix0     = (((next80 >> 12) ^ next80) >> 3) ^ next80;
                        unsigned long long mix1     = (((next81 >> 12) ^ next81) >> 3) ^ next81;

                        lookup[(mix0 >> (4 * 8)) & mask] = (int)(output - outputStart + 0);
                        lookup[(mix0 >> (3 * 8)) & mask] = (int)(output - outputStart + 1);
                        lookup[(mix0 >> (2 * 8)) & mask] = (int)(output - outputStart + 2);
                        lookup[(mix0 >> (1 * 8)) & mask] = (int)(output - outputStart + 3);
                        lookup[(mix1 >> (4 * 8)) & mask] = (int)(output - outputStart + 4);
                        lookup[(mix1 >> (3 * 8)) & mask] = (int)(output - outputStart + 5);
                        lookup[(mix1 >> (2 * 8)) & mask] = (int)(output - outputStart + 6);
                        lookup[(mix1 >> (1 * 8)) & mask] = (int)(output - outputStart + 7);

                        prev4 = (unsigned int)(next8 >> 32); input += 8; output += 8;

                        continue;
                    }
                }
#endif

                unsigned int next4          = *(unsigned int *)(output) = *(unsigned int *)(input);
                unsigned long long next8    = bsc_byteswap_uint64(((unsigned long long)next4 << 32) | prev4);

                int value;
                {
                    unsigned long long mix = (((next8 >> 12) ^ next8) >> 3) ^ next8;

                    const unsigned int flags = next4 ^ (LIBBSC_LZP_MATCH_FLAG * 0x01010101u);
                    if (((flags - 0x01010101u) & ~flags & 0x80808080u) == 0)
                    {
                        lookup[(mix >> (4 * 8)) & mask] = (int)(output - outputStart + 0);
                        lookup[(mix >> (3 * 8)) & mask] = (int)(output - outputStart + 1);
                        lookup[(mix >> (2 * 8)) & mask] = (int)(output - outputStart + 2);
                        lookup[(mix >> (1 * 8)) & mask] = (int)(output - outputStart + 3);

                        prev4 = next4; input += 4; output += 4;

                        continue;
                    }

                    const unsigned int index0 = (mix >> (4 * 8)) & mask; value = lookup[index0]; lookup[index0] = (int)(output - outputStart + 0);
                    if (((unsigned char)(next8 >> 3 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_MATCH_FOUND1;

                    const unsigned int index1 = (mix >> (3 * 8)) & mask; value = lookup[index1]; lookup[index1] = (int)(output - outputStart + 1);
                    if (((unsigned char)(next8 >> 2 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_MATCH_FOUND2;

                    const unsigned int index2 = (mix >> (2 * 8)) & mask; value = lookup[index2]; lookup[index2] = (int)(output - outputStart + 2);
                    if (((unsigned char)(next8 >> 1 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_MATCH_FOUND3;

                    const unsigned int index3 = (mix >> (1 * 8)) & mask; value = lookup[index3]; lookup[index3] = (int)(output - outputStart + 3);
                    if (((unsigned char)(next8 >> 0 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_MATCH_FOUND4;

                    prev4 = next4; input += 4; output += 4;

                    continue;
                }

LIBBSC_LZP_MATCH_FOUND4:
                input += 1; output += 1;
LIBBSC_LZP_MATCH_FOUND3:
                input += 1; output += 1;
LIBBSC_LZP_MATCH_FOUND2:
                input += 1; output += 1;
LIBBSC_LZP_MATCH_FOUND1:
                input += 1;

                if (*input != 255)
                {
                    int len = minLen; while (true) { len += *input; if (*input++ != 254) break; }

                    const unsigned char * reference = outputStart + value;
                          unsigned char * outputEnd = output + len;

                    if (output - reference >= 8 && len >= 8)
                    {
                        while (output < outputEnd - 8) { memcpy(output, reference, 8); output += 8; reference += 8; }
                        memcpy(outputEnd - 8, outputEnd - 8 - (output - reference), 8); output = outputEnd;
                    }
                    else
                    {
                        while (output < outputEnd) { *output++ = *reference++; }
                    }

                    prev4 = *(unsigned int *)(output - 4);
                }
                else
                {
                    input++; output++; prev4 = *(unsigned int *)(output - 4); 
                }
            }
        }
#endif

        {
            unsigned int context = output[-1] | (output[-2] << 8) | (output[-3] << 16) | (output[-4] << 24);

            while (input < inputEnd)
            {
#if !defined(LIBBSC_NO_UNALIGNED_ACCESS)
#if LIBBSC_CPU_FEATURE >= LIBBSC_CPU_FEATURE_SSE2
                if (input < inputEnd - 16)
                {
                    __m128i next16 = _mm_loadu_si128((const __m128i *)input);
                    if (_mm_movemask_epi8(_mm_cmpeq_epi8(next16, _mm_set1_epi8((char)LIBBSC_LZP_MATCH_FLAG))) == 0)
                    {
                        _mm_storeu_si128((__m128i *)output, next16);

                        unsigned long long next81   = bsc_byteswap_uint64(*(unsigned long long *)(input + 0));
                        unsigned long long next80   = ((unsigned long long)context << 32) | (next81 >> 32);
                        unsigned long long next82   = bsc_byteswap_uint64(*(unsigned long long *)(input + 4));
                        unsigned long long next83   = bsc_byteswap_uint64(*(unsigned long long *)(input + 8));

                        const unsigned int context0  = (unsigned int)(next80 >> (4 * 8)); lookup[((context0 >> 15) ^ context0 ^ (context0 >> 3)) & mask] = (int)(output - outputStart +  0);
                        const unsigned int context1  = (unsigned int)(next80 >> (3 * 8)); lookup[((context1 >> 15) ^ context1 ^ (context1 >> 3)) & mask] = (int)(output - outputStart +  1);
                        const unsigned int context2  = (unsigned int)(next80 >> (2 * 8)); lookup[((context2 >> 15) ^ context2 ^ (context2 >> 3)) & mask] = (int)(output - outputStart +  2);
                        const unsigned int context3  = (unsigned int)(next80 >> (1 * 8)); lookup[((context3 >> 15) ^ context3 ^ (context3 >> 3)) & mask] = (int)(output - outputStart +  3);
                        const unsigned int context4  = (unsigned int)(next81 >> (4 * 8)); lookup[((context4 >> 15) ^ context4 ^ (context4 >> 3)) & mask] = (int)(output - outputStart +  4);
                        const unsigned int context5  = (unsigned int)(next81 >> (3 * 8)); lookup[((context5 >> 15) ^ context5 ^ (context5 >> 3)) & mask] = (int)(output - outputStart +  5);
                        const unsigned int context6  = (unsigned int)(next81 >> (2 * 8)); lookup[((context6 >> 15) ^ context6 ^ (context6 >> 3)) & mask] = (int)(output - outputStart +  6);
                        const unsigned int context7  = (unsigned int)(next81 >> (1 * 8)); lookup[((context7 >> 15) ^ context7 ^ (context7 >> 3)) & mask] = (int)(output - outputStart +  7);
                        const unsigned int context8  = (unsigned int)(next82 >> (4 * 8)); lookup[((context8 >> 15) ^ context8 ^ (context8 >> 3)) & mask] = (int)(output - outputStart +  8);
                        const unsigned int context9  = (unsigned int)(next82 >> (3 * 8)); lookup[((context9 >> 15) ^ context9 ^ (context9 >> 3)) & mask] = (int)(output - outputStart +  9);
                        const unsigned int context10 = (unsigned int)(next82 >> (2 * 8)); lookup[((context10 >> 15) ^ context10 ^ (context10 >> 3)) & mask] = (int)(output - outputStart + 10);
                        const unsigned int context11 = (unsigned int)(next82 >> (1 * 8)); lookup[((context11 >> 15) ^ context11 ^ (context11 >> 3)) & mask] = (int)(output - outputStart + 11);
                        const unsigned int context12 = (unsigned int)(next83 >> (4 * 8)); lookup[((context12 >> 15) ^ context12 ^ (context12 >> 3)) & mask] = (int)(output - outputStart + 12);
                        const unsigned int context13 = (unsigned int)(next83 >> (3 * 8)); lookup[((context13 >> 15) ^ context13 ^ (context13 >> 3)) & mask] = (int)(output - outputStart + 13);
                        const unsigned int context14 = (unsigned int)(next83 >> (2 * 8)); lookup[((context14 >> 15) ^ context14 ^ (context14 >> 3)) & mask] = (int)(output - outputStart + 14);
                        const unsigned int context15 = (unsigned int)(next83 >> (1 * 8)); lookup[((context15 >> 15) ^ context15 ^ (context15 >> 3)) & mask] = (int)(output - outputStart + 15);

                        context = (unsigned int)next83; input += 16; output += 16;

                        continue;
                    }
                }
#elif LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_A64
                if (input < inputEnd - 8)
                {
                    unsigned long long next8    = *(unsigned long long *)(input);
                    unsigned long long flags    = next8 ^ (LIBBSC_LZP_MATCH_FLAG * 0x0101010101010101ull);

                    if (((flags - 0x0101010101010101ull) & ~flags & 0x8080808080808080ull) == 0)
                    {
                        *(unsigned long long *)(output) = next8;

                        unsigned long long next81   = bsc_byteswap_uint64(next8);
                        unsigned long long next80   = ((unsigned long long)context << 32) | (next81 >> 32);

                        const unsigned int context0  = (unsigned int)(next80 >> (4 * 8)); lookup[((context0 >> 15) ^ context0 ^ (context0 >> 3)) & mask] = (int)(output - outputStart +  0);
                        const unsigned int context1  = (unsigned int)(next80 >> (3 * 8)); lookup[((context1 >> 15) ^ context1 ^ (context1 >> 3)) & mask] = (int)(output - outputStart +  1);
                        const unsigned int context2  = (unsigned int)(next80 >> (2 * 8)); lookup[((context2 >> 15) ^ context2 ^ (context2 >> 3)) & mask] = (int)(output - outputStart +  2);
                        const unsigned int context3  = (unsigned int)(next80 >> (1 * 8)); lookup[((context3 >> 15) ^ context3 ^ (context3 >> 3)) & mask] = (int)(output - outputStart +  3);
                        const unsigned int context4  = (unsigned int)(next81 >> (4 * 8)); lookup[((context4 >> 15) ^ context4 ^ (context4 >> 3)) & mask] = (int)(output - outputStart +  4);
                        const unsigned int context5  = (unsigned int)(next81 >> (3 * 8)); lookup[((context5 >> 15) ^ context5 ^ (context5 >> 3)) & mask] = (int)(output - outputStart +  5);
                        const unsigned int context6  = (unsigned int)(next81 >> (2 * 8)); lookup[((context6 >> 15) ^ context6 ^ (context6 >> 3)) & mask] = (int)(output - outputStart +  6);
                        const unsigned int context7  = (unsigned int)(next81 >> (1 * 8)); lookup[((context7 >> 15) ^ context7 ^ (context7 >> 3)) & mask] = (int)(output - outputStart +  7);

                        context = (unsigned int)next81; input += 8; output += 8;

                        continue;
                    }
                }
#endif
#endif

                unsigned int index = ((context >> 15) ^ context ^ (context >> 3)) & mask;
                int value = lookup[index]; lookup[index] = (int)(output - outputStart);
                if (*input == LIBBSC_LZP_MATCH_FLAG && value > 0)
                {
                    input++;
                    if (*input != 255)
                    {
                        int len = minLen; while (true) { len += *input; if (*input++ != 254) break; }

                        const unsigned char * reference = outputStart + value;
                              unsigned char * outputEnd = output + len;

                        while (output < outputEnd) *output++ = *reference++;

                        context = output[-1] | (output[-2] << 8) | (output[-3] << 16) | (output[-4] << 24);
                    }
                    else
                    {
                        input++; context = (context << 8) | (*output++ = LIBBSC_LZP_MATCH_FLAG);
                    }
                }
                else
                {
                    context = (context << 8) | (*output++ = *input++);
                }
            }
        }

        if (table == NULL) bsc_free(lookup);

        return (int)(output - outputStart);
    }

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_lzp_compress_serial(const unsigned char * input, unsigned char * output, int n, int hashSize, int minLen)
{
    if (bsc_lzp_num_blocks(n) == 1)
    {
        int result = bsc_lzp_encode_block(input, input + n, output + 1, output + n - 1, NULL, hashSize, minLen);
        if (result >= LIBBSC_NO_ERROR) result = (output[0] = 1, result + 1);

        return result;
    }

    int nBlocks   = bsc_lzp_num_blocks(n);
    int chunkSize = n / nBlocks;
    int outputPtr = 1 + 8 * nBlocks;

    size_t  tableSize = ((size_t)1 << hashSize) * sizeof(int);
    int *   table     = tableSize <= (size_t)chunkSize ? (int *)bsc_zero_malloc(tableSize) : NULL;

    output[0] = nBlocks;
    for (int blockId = 0; blockId < nBlocks; ++blockId)
    {
        int inputStart  = blockId * chunkSize;
        int inputSize   = blockId != nBlocks - 1 ? chunkSize : n - inputStart;
        int outputSize  = inputSize; if (outputSize > n - outputPtr) outputSize = n - outputPtr;

        if (table != NULL && blockId > 0) memset(table, 0, tableSize);

        int result = bsc_lzp_encode_block(input + inputStart, input + inputStart + inputSize, output + outputPtr, output + outputPtr + outputSize, table, hashSize, minLen);
        if (result < LIBBSC_NO_ERROR)
        {
            if (outputPtr + inputSize >= n) { outputPtr = LIBBSC_NOT_COMPRESSIBLE; break; }
            result = inputSize; memcpy(output + outputPtr, input + inputStart, inputSize);
        }

        memcpy(output + 1 + 8 * blockId + 0, &inputSize, sizeof(int));
        memcpy(output + 1 + 8 * blockId + 4, &result, sizeof(int));

        outputPtr += result;
    }

    if (table != NULL) bsc_free(table);

    return outputPtr;
}

#ifdef LIBBSC_OPENMP

int bsc_lzp_compress_parallel(const unsigned char * input, unsigned char * output, int n, int hashSize, int minLen)
{
    int nBlocks   = bsc_lzp_num_blocks(n);
    int chunkSize = n / nBlocks;

    int numThreads = omp_get_max_threads() / omp_get_num_threads();
    if (numThreads > nBlocks) numThreads = nBlocks;
    if (numThreads < 1) numThreads = 1;

    if (numThreads == 1 && input != output)
    {
        return bsc_lzp_compress_serial(input, output, n, hashSize, minLen);
    }

    if (unsigned char * buffer = (unsigned char *)bsc_malloc(n * sizeof(unsigned char)))
    {
        int compressionResult[ALPHABET_SIZE];

        #pragma omp parallel for schedule(dynamic, 1) num_threads(numThreads)
        for (int blockId = 0; blockId < nBlocks; ++blockId)
        {
            int blockStart   = blockId * chunkSize;
            int blockSize    = blockId != nBlocks - 1 ? chunkSize : n - blockStart;

            compressionResult[blockId] = bsc_lzp_encode_block(input + blockStart, input + blockStart + blockSize, buffer + blockStart, buffer + blockStart + blockSize, NULL, hashSize, minLen);
            if (compressionResult[blockId] < LIBBSC_NO_ERROR)
            {
                compressionResult[blockId] = blockSize; memcpy(buffer + blockStart, input + blockStart, blockSize);
            }
        }

        int result = 1 + 8 * nBlocks;
        for (int blockId = 0; blockId < nBlocks; ++blockId) { result += compressionResult[blockId]; }

        if (result < n)
        {
            output[0] = nBlocks;

            #pragma omp parallel for schedule(dynamic, 1) num_threads(numThreads)
            for (int blockId = 0; blockId < nBlocks; ++blockId)
            {
                int blockStart   = blockId * chunkSize;
                int blockSize    = blockId != nBlocks - 1 ? chunkSize : n - blockStart;
                int outputPtr    = 1 + 8 * nBlocks;

                for (int p = 0; p < blockId; ++p) { outputPtr += compressionResult[p]; }

                memcpy(output + 1 + 8 * blockId + 0, &blockSize, sizeof(int));
                memcpy(output + 1 + 8 * blockId + 4, &compressionResult[blockId], sizeof(int));
                memcpy(output + outputPtr, buffer + blockStart, compressionResult[blockId]);
            }
        }
        else
        {
            result = LIBBSC_NOT_COMPRESSIBLE;
        }

        bsc_free(buffer);

        return result;
    }
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

#endif

int bsc_lzp_compress(const unsigned char * input, unsigned char * output, int n, int hashSize, int minLen, int features)
{

#ifdef LIBBSC_OPENMP

    if ((bsc_lzp_num_blocks(n) != 1) && (features & LIBBSC_FEATURE_MULTITHREADING))
    {
        return bsc_lzp_compress_parallel(input, output, n, hashSize, minLen);
    }

#endif

    if (input == output)
    {
        if (unsigned char * buffer = (unsigned char *)bsc_malloc(n * sizeof(unsigned char)))
        {
            int result = bsc_lzp_compress_serial(input, buffer, n, hashSize, minLen);
            if (result >= LIBBSC_NO_ERROR) memcpy(output, buffer, result);

            bsc_free(buffer);

            return result;
        }
        return LIBBSC_NOT_ENOUGH_MEMORY;
    }

    return bsc_lzp_compress_serial(input, output, n, hashSize, minLen);
}

int bsc_lzp_decompress(const unsigned char * input, unsigned char * output, int n, int hashSize, int minLen, unsigned int * adler32, int features)
{
    int nBlocks = input[0];

    if (nBlocks == 1)
    {
        int result = bsc_lzp_decode_block(input + 1, input + n, output, NULL, hashSize, minLen);
        if (adler32 != NULL && result >= LIBBSC_NO_ERROR) *adler32 = bsc_adler32(output, result, features);

        return result;
    }

    int             decompressionResult[ALPHABET_SIZE];
    unsigned int    decompressionChecksum[ALPHABET_SIZE];

    int pendingBlocks = nBlocks;

#ifdef LIBBSC_OPENMP

    int numThreads = omp_get_max_threads() / omp_get_num_threads();
    if (numThreads > nBlocks) numThreads = nBlocks;

    if ((numThreads > 1) && (features & LIBBSC_FEATURE_MULTITHREADING))
    {
        #pragma omp parallel for schedule(dynamic, 1) num_threads(numThreads)
        for (int blockId = 0; blockId < nBlocks; ++blockId)
        {
            int inputPtr  = 0; int inputSize;
            int outputPtr = 0; int outputSize;

            inputPtr += 1 + 8 * nBlocks;

            for (int p = 0; p < blockId; ++p)
            {
                memcpy(&inputSize , input + 1 + 8 * p + 4, sizeof(int));
                memcpy(&outputSize, input + 1 + 8 * p + 0, sizeof(int));

                inputPtr += inputSize; outputPtr += outputSize;
            }

            memcpy(&inputSize , input + 1 + 8 * blockId + 4, sizeof(int));
            memcpy(&outputSize, input + 1 + 8 * blockId + 0, sizeof(int));

            if (inputSize != outputSize)
            {
                decompressionResult[blockId] = bsc_lzp_decode_block(input + inputPtr, input + inputPtr + inputSize, output + outputPtr, NULL, hashSize, minLen);
            }
            else
            {
                decompressionResult[blockId] = inputSize; memcpy(output + outputPtr, input + inputPtr, inputSize);
            }

            if (decompressionResult[blockId] >= LIBBSC_NO_ERROR && decompressionResult[blockId] != outputSize) decompressionResult[blockId] = LIBBSC_DATA_CORRUPT;
            if (adler32 != NULL && decompressionResult[blockId] >= LIBBSC_NO_ERROR) decompressionChecksum[blockId] = bsc_adler32(output + outputPtr, decompressionResult[blockId], features);
        }

        pendingBlocks = 0;
        for (int blockId = 0; blockId < nBlocks; ++blockId)
        {
            if (decompressionResult[blockId] == LIBBSC_NOT_ENOUGH_MEMORY) pendingBlocks++;
        }
    }

#endif

    if (pendingBlocks > 0)
    {
        int chunkSize; memcpy(&chunkSize, input + 1, sizeof(int));

        size_t  tableSize = ((size_t)1 << hashSize) * sizeof(int);
        int *   table     = tableSize <= (size_t)chunkSize ? (int *)bsc_zero_malloc(tableSize) : NULL;
        bool    tableUsed = false;

        for (int blockId = 0; blockId < nBlocks; ++blockId)
        {
            if (pendingBlocks < nBlocks && decompressionResult[blockId] != LIBBSC_NOT_ENOUGH_MEMORY) continue;

            int inputPtr  = 0; int inputSize;
            int outputPtr = 0; int outputSize;

            inputPtr += 1 + 8 * nBlocks;

            for (int p = 0; p < blockId; ++p)
            {
                memcpy(&inputSize , input + 1 + 8 * p + 4, sizeof(int));
                memcpy(&outputSize, input + 1 + 8 * p + 0, sizeof(int));

                inputPtr += inputSize; outputPtr += outputSize;
            }

            memcpy(&inputSize , input + 1 + 8 * blockId + 4, sizeof(int));
            memcpy(&outputSize, input + 1 + 8 * blockId + 0, sizeof(int));

            if (inputSize != outputSize)
            {
                if (table != NULL && tableUsed) memset(table, 0, tableSize);

                decompressionResult[blockId] = bsc_lzp_decode_block(input + inputPtr, input + inputPtr + inputSize, output + outputPtr, table, hashSize, minLen); tableUsed = true;
            }
            else
            {
                decompressionResult[blockId] = inputSize; memcpy(output + outputPtr, input + inputPtr, inputSize);
            }

            if (decompressionResult[blockId] >= LIBBSC_NO_ERROR && decompressionResult[blockId] != outputSize) decompressionResult[blockId] = LIBBSC_DATA_CORRUPT;
            if (adler32 != NULL && decompressionResult[blockId] >= LIBBSC_NO_ERROR) decompressionChecksum[blockId] = bsc_adler32(output + outputPtr, decompressionResult[blockId], features);
        }

        if (table != NULL) bsc_free(table);
    }

    int dataSize = 0, result = LIBBSC_NO_ERROR;
    for (int blockId = 0; blockId < nBlocks; ++blockId)
    {
        if (decompressionResult[blockId] < LIBBSC_NO_ERROR) result = decompressionResult[blockId];
        dataSize += decompressionResult[blockId];
    }

    if (adler32 != NULL && result == LIBBSC_NO_ERROR)
    {
        *adler32 = 1;
        for (int blockId = 0; blockId < nBlocks; ++blockId) { *adler32 = bsc_adler32_combine(*adler32, decompressionChecksum[blockId], decompressionResult[blockId]); }
    }

    return (result == LIBBSC_NO_ERROR) ? dataSize : result;
}

/*-----------------------------------------------------------*/
/* End                                               lzp.cpp */
/*-----------------------------------------------------------*/

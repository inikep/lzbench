/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Lempel Ziv Prediction                                     */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

   Copyright (c) 2009-2021 Ilya Grebnov <ilya.grebnov@gmail.com>

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

#include "../platform/platform.h"
#include "../libbsc.h"

#define LIBBSC_LZP_MATCH_FLAG 	0xf2

static INLINE int bsc_lzp_num_blocks(int n)
{
    if (n <       256 * 1024)   return 1;
    if (n <  4 * 1024 * 1024)   return 2;
    if (n < 16 * 1024 * 1024)   return 4;

    return 8;
}

#if defined(LIBBSC_ALLOW_UNALIGNED_ACCESS) && (defined(__x86_64__) || defined(__aarch64__))

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
                const unsigned int index0 = (((next8 >> (4 * 8)) >> 15) ^ (next8 >> (4 * 8)) ^ ((next8 >> (4 * 8)) >> 3)) & mask; value = lookup[index0]; lookup[index0] = (int)(input - inputStart + 0); 
                if (value > 0 && (*(T *)(input + 0) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND1;
                if (value > 0 && ((unsigned char)(next8 >> 3 * 8) == LIBBSC_LZP_MATCH_FLAG)) goto LIBBSC_LZP_BAD_MATCH_FOUND1;

                const unsigned int index1 = (((next8 >> (3 * 8)) >> 15) ^ (next8 >> (3 * 8)) ^ ((next8 >> (3 * 8)) >> 3)) & mask; value = lookup[index1]; lookup[index1] = (int)(input - inputStart + 1); 
                if (value > 0 && (*(T *)(input + 1) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND2;
                if (value > 0 && ((unsigned char)(next8 >> 2 * 8) == LIBBSC_LZP_MATCH_FLAG)) goto LIBBSC_LZP_BAD_MATCH_FOUND2;

                const unsigned int index2 = (((next8 >> (2 * 8)) >> 15) ^ (next8 >> (2 * 8)) ^ ((next8 >> (2 * 8)) >> 3)) & mask; value = lookup[index2]; lookup[index2] = (int)(input - inputStart + 2); 
                if (value > 0 && (*(T *)(input + 2) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND3;
                if (value > 0 && ((unsigned char)(next8 >> 1 * 8) == LIBBSC_LZP_MATCH_FLAG)) goto LIBBSC_LZP_BAD_MATCH_FOUND3;

                const unsigned int index3 = (((next8 >> (1 * 8)) >> 15) ^ (next8 >> (1 * 8)) ^ ((next8 >> (1 * 8)) >> 3)) & mask; value = lookup[index3]; lookup[index3] = (int)(input - inputStart + 3); 
                if (value > 0 && (*(T *)(input + 3) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND4;
                if (value > 0 && ((unsigned char)(next8 >> 0 * 8) == LIBBSC_LZP_MATCH_FLAG)) goto LIBBSC_LZP_BAD_MATCH_FOUND4;

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
                const unsigned int index0 = (((next8 >> (4 * 8)) >> 15) ^ (next8 >> (4 * 8)) ^ ((next8 >> (4 * 8)) >> 3)) & mask; value = lookup[index0]; lookup[index0] = (int)(input - inputStart + 0); 
                if (value > 0 && (*(T *)(input + sizeof(T) + 0) == *(T *)(inputStart + value + sizeof(T))) && (*(T *)(input + 0) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND1;
                if (value > 0 && ((unsigned char)(next8 >> 3 * 8) == LIBBSC_LZP_MATCH_FLAG)) goto LIBBSC_LZP_BAD_MATCH_FOUND1;

                const unsigned int index1 = (((next8 >> (3 * 8)) >> 15) ^ (next8 >> (3 * 8)) ^ ((next8 >> (3 * 8)) >> 3)) & mask; value = lookup[index1]; lookup[index1] = (int)(input - inputStart + 1); 
                if (value > 0 && (*(T *)(input + sizeof(T) + 1) == *(T *)(inputStart + value + sizeof(T))) && (*(T *)(input + 1) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND2;
                if (value > 0 && ((unsigned char)(next8 >> 2 * 8) == LIBBSC_LZP_MATCH_FLAG)) goto LIBBSC_LZP_BAD_MATCH_FOUND2;

                const unsigned int index2 = (((next8 >> (2 * 8)) >> 15) ^ (next8 >> (2 * 8)) ^ ((next8 >> (2 * 8)) >> 3)) & mask; value = lookup[index2]; lookup[index2] = (int)(input - inputStart + 2); 
                if (value > 0 && (*(T *)(input + sizeof(T) + 2) == *(T *)(inputStart + value + sizeof(T))) && (*(T *)(input + 2) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND3;
                if (value > 0 && ((unsigned char)(next8 >> 1 * 8) == LIBBSC_LZP_MATCH_FLAG)) goto LIBBSC_LZP_BAD_MATCH_FOUND3;

                const unsigned int index3 = (((next8 >> (1 * 8)) >> 15) ^ (next8 >> (1 * 8)) ^ ((next8 >> (1 * 8)) >> 3)) & mask; value = lookup[index3]; lookup[index3] = (int)(input - inputStart + 3); 
                if (value > 0 && (*(T *)(input + sizeof(T) + 3) == *(T *)(inputStart + value + sizeof(T))) && (*(T *)(input + 3) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND4;
                if (value > 0 && ((unsigned char)(next8 >> 0 * 8) == LIBBSC_LZP_MATCH_FLAG)) goto LIBBSC_LZP_BAD_MATCH_FOUND4;

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
                const unsigned int index0 = (((next8 >> (4 * 8)) >> 15) ^ (next8 >> (4 * 8)) ^ ((next8 >> (4 * 8)) >> 3)) & mask; value = lookup[index0]; lookup[index0] = (int)(input - inputStart + 0); 
                if (value > 0 && (*(T *)(input + minLen - sizeof(T) + 0) == *(T *)(inputStart + value + minLen - sizeof(T))) && (*(T *)(input + 0) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND1;
                if (value > 0 && ((unsigned char)(next8 >> 3 * 8) == LIBBSC_LZP_MATCH_FLAG)) goto LIBBSC_LZP_BAD_MATCH_FOUND1;

                const unsigned int index1 = (((next8 >> (3 * 8)) >> 15) ^ (next8 >> (3 * 8)) ^ ((next8 >> (3 * 8)) >> 3)) & mask; value = lookup[index1]; lookup[index1] = (int)(input - inputStart + 1); 
                if (value > 0 && (*(T *)(input + minLen - sizeof(T) + 1) == *(T *)(inputStart + value + minLen - sizeof(T))) && (*(T *)(input + 1) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND2;
                if (value > 0 && ((unsigned char)(next8 >> 2 * 8) == LIBBSC_LZP_MATCH_FLAG)) goto LIBBSC_LZP_BAD_MATCH_FOUND2;

                const unsigned int index2 = (((next8 >> (2 * 8)) >> 15) ^ (next8 >> (2 * 8)) ^ ((next8 >> (2 * 8)) >> 3)) & mask; value = lookup[index2]; lookup[index2] = (int)(input - inputStart + 2); 
                if (value > 0 && (*(T *)(input + minLen - sizeof(T) + 2) == *(T *)(inputStart + value + minLen - sizeof(T))) && (*(T *)(input + 2) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND3;
                if (value > 0 && ((unsigned char)(next8 >> 1 * 8) == LIBBSC_LZP_MATCH_FLAG)) goto LIBBSC_LZP_BAD_MATCH_FOUND3;

                const unsigned int index3 = (((next8 >> (1 * 8)) >> 15) ^ (next8 >> (1 * 8)) ^ ((next8 >> (1 * 8)) >> 3)) & mask; value = lookup[index3]; lookup[index3] = (int)(input - inputStart + 3); 
                if (value > 0 && (*(T *)(input + minLen - sizeof(T) + 3) == *(T *)(inputStart + value + minLen - sizeof(T))) && (*(T *)(input + 3) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND4;
                if (value > 0 && ((unsigned char)(next8 >> 0 * 8) == LIBBSC_LZP_MATCH_FLAG)) goto LIBBSC_LZP_BAD_MATCH_FOUND4;

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
    const unsigned char *   inputStart  = input;
    const unsigned char *   outputStart = output;
    const unsigned char *   outputEOB   = outputEnd - 8;

    const unsigned char * heuristic      = input;
    const unsigned char * inputMinLenEnd = inputEnd - minLen - 32;

    for (int i = 0; i < 4; ++i) { *output++ = *input++; }

    {
        while ((input < inputMinLenEnd) && (output < outputEOB))
        {
            unsigned long long next8 = *(unsigned long long *)(input - 4); *(unsigned int *)(output) = (unsigned int)(next8 >> 32); next8 = bsc_byteswap_uint64(next8);

            int value;
            {
                const unsigned int index0 = (((next8 >> (4 * 8)) >> 15) ^ (next8 >> (4 * 8)) ^ ((next8 >> (4 * 8)) >> 3)) & mask; value = lookup[index0]; lookup[index0] = (int)(input - inputStart + 0); 
                if (value > 0 && input > heuristic && (*(T *)(input + minLen - sizeof(T) + 0) == *(T *)(inputStart + value + minLen - sizeof(T))) && (*(T *)(input + 0) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND1;
                if (value > 0 && ((unsigned char)(next8 >> 3 * 8) == LIBBSC_LZP_MATCH_FLAG)) goto LIBBSC_LZP_BAD_MATCH_FOUND1;

                const unsigned int index1 = (((next8 >> (3 * 8)) >> 15) ^ (next8 >> (3 * 8)) ^ ((next8 >> (3 * 8)) >> 3)) & mask; value = lookup[index1]; lookup[index1] = (int)(input - inputStart + 1); 
                if (value > 0 && input > heuristic && (*(T *)(input + minLen - sizeof(T) + 1) == *(T *)(inputStart + value + minLen - sizeof(T))) && (*(T *)(input + 1) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND2;
                if (value > 0 && ((unsigned char)(next8 >> 2 * 8) == LIBBSC_LZP_MATCH_FLAG)) goto LIBBSC_LZP_BAD_MATCH_FOUND2;

                const unsigned int index2 = (((next8 >> (2 * 8)) >> 15) ^ (next8 >> (2 * 8)) ^ ((next8 >> (2 * 8)) >> 3)) & mask; value = lookup[index2]; lookup[index2] = (int)(input - inputStart + 2); 
                if (value > 0 && input > heuristic && (*(T *)(input + minLen - sizeof(T) + 2) == *(T *)(inputStart + value + minLen - sizeof(T))) && (*(T *)(input + 2) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND3;
                if (value > 0 && ((unsigned char)(next8 >> 1 * 8) == LIBBSC_LZP_MATCH_FLAG)) goto LIBBSC_LZP_BAD_MATCH_FOUND3;

                const unsigned int index3 = (((next8 >> (1 * 8)) >> 15) ^ (next8 >> (1 * 8)) ^ ((next8 >> (1 * 8)) >> 3)) & mask; value = lookup[index3]; lookup[index3] = (int)(input - inputStart + 3); 
                if (value > 0 && input > heuristic && (*(T *)(input + minLen - sizeof(T) + 3) == *(T *)(inputStart + value + minLen - sizeof(T))) && (*(T *)(input + 3) == *(T *)(inputStart + value))) goto LIBBSC_LZP_GOOD_MATCH_FOUND4;
                if (value > 0 && ((unsigned char)(next8 >> 0 * 8) == LIBBSC_LZP_MATCH_FLAG)) goto LIBBSC_LZP_BAD_MATCH_FOUND4;

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

                if (len < minLen) { heuristic = input + len; goto LIBBSC_LZP_MATCH_NOT_FOUND; }

                input += len; len -= minLen;

                *output++ = LIBBSC_LZP_MATCH_FLAG; while (len >= 254) { len -= 254; *output++ = 254; if (output >= outputEOB) break; } *output++ = (unsigned char)(len); 
            
                continue;
            }

LIBBSC_LZP_MATCH_NOT_FOUND:
            if ((*output++ = *input++) == LIBBSC_LZP_MATCH_FLAG) { *output++ = 255; }

            continue;

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
#if defined(LIBBSC_ALLOW_UNALIGNED_ACCESS)
                if ((*(unsigned int *)(input + minLen - 4) == *(unsigned int *)(reference + minLen - 4)) && (*(unsigned int *)(input) == *(unsigned int *)(reference)))
#else
                if ((memcmp(input + minLen - 4, reference + minLen - 4, sizeof(unsigned int)) == 0) && (memcmp(input, reference, sizeof(unsigned int)) == 0))
#endif
                {
                    if ((heuristic > input) && (*(unsigned int *)heuristic != *(unsigned int *)(reference + (heuristic - input))))
                    {
                        goto LIBBSC_LZP_MATCH_NOT_FOUND;
                    }

                    int len = 4;
                    for (; input + len < inputMinLenEnd; len += sizeof(unsigned int))
                    {
                        if (*(unsigned int *)(input + len) != *(unsigned int *)(reference + len)) break;
                    }

                    if (len < minLen)
                    {
                        if (heuristic < input + len) heuristic = input + len;
                        goto LIBBSC_LZP_MATCH_NOT_FOUND;
                    }

#if defined(LIBBSC_ALLOW_UNALIGNED_ACCESS)
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

int bsc_lzp_encode_block(const unsigned char * input, const unsigned char * inputEnd, unsigned char * output, unsigned char * outputEnd, int hashSize, int minLen)
{
    if (inputEnd - input - minLen < 32)
    {
        return LIBBSC_NOT_COMPRESSIBLE;
    }

    int result = LIBBSC_NOT_ENOUGH_MEMORY;
    if (int * lookup = (int *)bsc_zero_malloc((int)(1 << hashSize) * sizeof(int)))
    {
#if defined(LIBBSC_ALLOW_UNALIGNED_ACCESS) && (defined(__x86_64__) || defined(__aarch64__))
        result = (minLen == 1 * (int)sizeof(unsigned int      ) && result == LIBBSC_NOT_ENOUGH_MEMORY) ? bsc_lzp_encode_small  <unsigned int      >(input, inputEnd, output, outputEnd, lookup, (int)(1 << hashSize) - 1) : result;
        result = (minLen == 1 * (int)sizeof(unsigned long long) && result == LIBBSC_NOT_ENOUGH_MEMORY) ? bsc_lzp_encode_small  <unsigned long long>(input, inputEnd, output, outputEnd, lookup, (int)(1 << hashSize) - 1) : result;
        result = (minLen == 2 * (int)sizeof(unsigned long long) && result == LIBBSC_NOT_ENOUGH_MEMORY) ? bsc_lzp_encode_small2x<unsigned long long>(input, inputEnd, output, outputEnd, lookup, (int)(1 << hashSize) - 1) : result;
        result = (minLen <= 2 * (int)sizeof(unsigned int      ) && result == LIBBSC_NOT_ENOUGH_MEMORY) ? bsc_lzp_encode_medium <unsigned int      >(input, inputEnd, output, outputEnd, lookup, (int)(1 << hashSize) - 1, minLen) : result;
        result = (minLen <= 2 * (int)sizeof(unsigned long long) && result == LIBBSC_NOT_ENOUGH_MEMORY) ? bsc_lzp_encode_medium <unsigned long long>(input, inputEnd, output, outputEnd, lookup, (int)(1 << hashSize) - 1, minLen) : result;
        
        result = result == LIBBSC_NOT_ENOUGH_MEMORY ? bsc_lzp_encode_large<unsigned long long>(input, inputEnd, output, outputEnd, lookup, (int)(1 << hashSize) - 1, minLen) : result;
#endif

        result = result == LIBBSC_NOT_ENOUGH_MEMORY ? bsc_lzp_encode_generic(input, inputEnd, output, outputEnd, lookup, (int)(1 << hashSize) - 1, minLen) : result;

        bsc_free(lookup);
    }

    return result;
}

int bsc_lzp_decode_block(const unsigned char * RESTRICT input, const unsigned char * inputEnd, unsigned char * RESTRICT output, int hashSize, int minLen)
{
    if (inputEnd - input < 4)
    {
        return LIBBSC_UNEXPECTED_EOB;
    }

    if (int * RESTRICT lookup = (int *)bsc_zero_malloc((int)(1 << hashSize) * sizeof(int)))
    {
        unsigned int            mask        = (int)(1 << hashSize) - 1;
        const unsigned char *   outputStart = output;

        for (int i = 0; i < 4; ++i) { *output++ = *input++; }

#if defined(LIBBSC_ALLOW_UNALIGNED_ACCESS) && (defined(__x86_64__) || defined(__aarch64__))
        if (hashSize <= 17)
        {
            unsigned int prev4 = *(unsigned int *)(output - 4);

            while (input < inputEnd - 8)
            {
                unsigned int next4          = *(unsigned int *)(output) = *(unsigned int *)(input);
                unsigned long long next8    = bsc_byteswap_uint64(((unsigned long long)next4 << 32) | prev4);

                int value;
                {
                    const unsigned int index0 = (((next8 >> (4 * 8)) >> 15) ^ (next8 >> (4 * 8)) ^ ((next8 >> (4 * 8)) >> 3)) & mask;
                    value = lookup[index0]; lookup[index0] = (int)(output - outputStart + 0); if (((unsigned char)(next8 >> 3 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_MATCH_FOUND1;

                    const unsigned int index1 = (((next8 >> (3 * 8)) >> 15) ^ (next8 >> (3 * 8)) ^ ((next8 >> (3 * 8)) >> 3)) & mask;
                    value = lookup[index1]; lookup[index1] = (int)(output - outputStart + 1); if (((unsigned char)(next8 >> 2 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_MATCH_FOUND2;

                    const unsigned int index2 = (((next8 >> (2 * 8)) >> 15) ^ (next8 >> (2 * 8)) ^ ((next8 >> (2 * 8)) >> 3)) & mask;
                    value = lookup[index2]; lookup[index2] = (int)(output - outputStart + 2); if (((unsigned char)(next8 >> 1 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_MATCH_FOUND3;

                    const unsigned int index3 = (((next8 >> (1 * 8)) >> 15) ^ (next8 >> (1 * 8)) ^ ((next8 >> (1 * 8)) >> 3)) & mask;
                    value = lookup[index3]; lookup[index3] = (int)(output - outputStart + 3); if (((unsigned char)(next8 >> 0 * 8) == LIBBSC_LZP_MATCH_FLAG) && (value > 0)) goto LIBBSC_LZP_MATCH_FOUND4;

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

                    while (output < outputEnd) { *output++ = *reference++; }

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

        bsc_free(lookup);

        return (int)(output - outputStart);
    }

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_lzp_compress_serial(const unsigned char * input, unsigned char * output, int n, int hashSize, int minLen)
{
    if (bsc_lzp_num_blocks(n) == 1)
    {
        int result = bsc_lzp_encode_block(input, input + n, output + 1, output + n - 1, hashSize, minLen);
        if (result >= LIBBSC_NO_ERROR) result = (output[0] = 1, result + 1);

        return result;
    }

    int nBlocks   = bsc_lzp_num_blocks(n);
    int chunkSize = n / nBlocks;
    int outputPtr = 1 + 8 * nBlocks;

    output[0] = nBlocks;
    for (int blockId = 0; blockId < nBlocks; ++blockId)
    {
        int inputStart  = blockId * chunkSize;
        int inputSize   = blockId != nBlocks - 1 ? chunkSize : n - inputStart;
        int outputSize  = inputSize; if (outputSize > n - outputPtr) outputSize = n - outputPtr;

        int result = bsc_lzp_encode_block(input + inputStart, input + inputStart + inputSize, output + outputPtr, output + outputPtr + outputSize, hashSize, minLen);
        if (result < LIBBSC_NO_ERROR)
        {
            if (outputPtr + inputSize >= n) return LIBBSC_NOT_COMPRESSIBLE;
            result = inputSize; memcpy(output + outputPtr, input + inputStart, inputSize);
        }
#if defined(LIBBSC_ALLOW_UNALIGNED_ACCESS)
        *(int *)(output + 1 + 8 * blockId + 0) = inputSize;
        *(int *)(output + 1 + 8 * blockId + 4) = result;
#else
        memcpy(output + 1 + 8 * blockId + 0, &inputSize, sizeof(int));
        memcpy(output + 1 + 8 * blockId + 4, &result, sizeof(int));
#endif

        outputPtr += result;
    }

    return outputPtr;
}

#ifdef LIBBSC_OPENMP

int bsc_lzp_compress_parallel(const unsigned char * input, unsigned char * output, int n, int hashSize, int minLen)
{
    if (unsigned char * buffer = (unsigned char *)bsc_malloc(n * sizeof(unsigned char)))
    {
        int compressionResult[ALPHABET_SIZE];

        int nBlocks   = bsc_lzp_num_blocks(n);
        int result    = LIBBSC_NO_ERROR;
        int chunkSize = n / nBlocks;

        int numThreads = omp_get_max_threads();
        if (numThreads > nBlocks) numThreads = nBlocks;

        output[0] = nBlocks;
        #pragma omp parallel num_threads(numThreads) if(numThreads > 1)
        {
            if (omp_get_num_threads() == 1)
            {
                result = bsc_lzp_compress_serial(input, output, n, hashSize, minLen);
            }
            else
            {
                #pragma omp for schedule(dynamic)
                for (int blockId = 0; blockId < nBlocks; ++blockId)
                {
                    int blockStart   = blockId * chunkSize;
                    int blockSize    = blockId != nBlocks - 1 ? chunkSize : n - blockStart;

                    compressionResult[blockId] = bsc_lzp_encode_block(input + blockStart, input + blockStart + blockSize, buffer + blockStart, buffer + blockStart + blockSize, hashSize, minLen);
                    if (compressionResult[blockId] < LIBBSC_NO_ERROR) compressionResult[blockId] = blockSize;

                    *(int *)(output + 1 + 8 * blockId + 0) = blockSize;
                    *(int *)(output + 1 + 8 * blockId + 4) = compressionResult[blockId];
                }

                #pragma omp single
                {
                    result = 1 + 8 * nBlocks;
                    for (int blockId = 0; blockId < nBlocks; ++blockId)
                    {
                        result += compressionResult[blockId];
                    }

                    if (result >= n) result = LIBBSC_NOT_COMPRESSIBLE;
                }

                if (result >= LIBBSC_NO_ERROR)
                {
                    #pragma omp for schedule(dynamic)
                    for (int blockId = 0; blockId < nBlocks; ++blockId)
                    {
                        int blockStart   = blockId * chunkSize;
                        int blockSize    = blockId != nBlocks - 1 ? chunkSize : n - blockStart;

                        int outputPtr = 1 + 8 * nBlocks;
                        for (int p = 0; p < blockId; ++p) outputPtr += compressionResult[p];

                        if (compressionResult[blockId] != blockSize)
                        {
                            memcpy(output + outputPtr, buffer + blockStart, compressionResult[blockId]);
                        }
                        else
                        {
                            memcpy(output + outputPtr, input + blockStart, compressionResult[blockId]);
                        }
                    }
                }
            }
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

    return bsc_lzp_compress_serial(input, output, n, hashSize, minLen);
}

int bsc_lzp_decompress(const unsigned char * input, unsigned char * output, int n, int hashSize, int minLen, int features)
{
    int nBlocks = input[0];

    if (nBlocks == 1)
    {
        return bsc_lzp_decode_block(input + 1, input + n, output, hashSize, minLen);
    }

    int decompressionResult[ALPHABET_SIZE];

#ifdef LIBBSC_OPENMP

    if (features & LIBBSC_FEATURE_MULTITHREADING)
    {
        #pragma omp parallel for schedule(dynamic)
        for (int blockId = 0; blockId < nBlocks; ++blockId)
        {
            int inputPtr = 0;  for (int p = 0; p < blockId; ++p) inputPtr  += *(int *)(input + 1 + 8 * p + 4);
            int outputPtr = 0; for (int p = 0; p < blockId; ++p) outputPtr += *(int *)(input + 1 + 8 * p + 0);

            inputPtr += 1 + 8 * nBlocks;

            int inputSize  = *(int *)(input + 1 + 8 * blockId + 4);
            int outputSize = *(int *)(input + 1 + 8 * blockId + 0);

            if (inputSize != outputSize)
            {
                decompressionResult[blockId] = bsc_lzp_decode_block(input + inputPtr, input + inputPtr + inputSize, output + outputPtr, hashSize, minLen);
            }
            else
            {
                decompressionResult[blockId] = inputSize; memcpy(output + outputPtr, input + inputPtr, inputSize);
            }
        }
    }
    else

#endif

    {
        for (int blockId = 0; blockId < nBlocks; ++blockId)
        {
            int inputPtr = 0;  for (int p = 0; p < blockId; ++p) inputPtr  += *(int *)(input + 1 + 8 * p + 4);
            int outputPtr = 0; for (int p = 0; p < blockId; ++p) outputPtr += *(int *)(input + 1 + 8 * p + 0);

            inputPtr += 1 + 8 * nBlocks;

            int inputSize  = *(int *)(input + 1 + 8 * blockId + 4);
            int outputSize = *(int *)(input + 1 + 8 * blockId + 0);

            if (inputSize != outputSize)
            {
                decompressionResult[blockId] = bsc_lzp_decode_block(input + inputPtr, input + inputPtr + inputSize, output + outputPtr, hashSize, minLen);
            }
            else
            {
                decompressionResult[blockId] = inputSize; memcpy(output + outputPtr, input + inputPtr, inputSize);
            }
        }
    }

    int dataSize = 0, result = LIBBSC_NO_ERROR;
    for (int blockId = 0; blockId < nBlocks; ++blockId)
    {
        if (decompressionResult[blockId] < LIBBSC_NO_ERROR) result = decompressionResult[blockId];
        dataSize += decompressionResult[blockId];
    }

    return (result == LIBBSC_NO_ERROR) ? dataSize : result;
}

/*-----------------------------------------------------------*/
/* End                                               lzp.cpp */
/*-----------------------------------------------------------*/

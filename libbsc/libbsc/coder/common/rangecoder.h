/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Range coder                                               */
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

#ifndef _LIBBSC_CODER_RANGECODER_H
#define _LIBBSC_CODER_RANGECODER_H

#include "../../platform/platform.h"

class RangeCoder
{

private:

    union ari
    {
        struct u
        {
            unsigned int low32;
            unsigned int carry;
        } u;
        unsigned long long low;
    } ari;

    unsigned int ari_code;
    unsigned int ari_ffnum;
    unsigned int ari_cache;
    unsigned int ari_range;

    const unsigned short * RESTRICT ari_input;
          unsigned short * RESTRICT ari_output;
          unsigned short * RESTRICT ari_outputEOB;
          unsigned short * RESTRICT ari_outputStart;

    INLINE void OutputShort(unsigned short s)
    {
#if defined(LIBBSC_ALLOW_UNALIGNED_ACCESS)
        *ari_output++ = s;
#else
        memcpy(ari_output++, &s, sizeof(unsigned short));
#endif
    };

    INLINE unsigned short InputShort()
    {
#if defined(LIBBSC_ALLOW_UNALIGNED_ACCESS)
        return *ari_input++;
#else
        unsigned short ret;
        memcpy(&ret, ari_input++, sizeof(unsigned short));
        return ret;
#endif
    };

    NOINLINE unsigned int ShiftLow()
    {
        if (ari.u.low32 < 0xffff0000U || ari.u.carry)
        {
            OutputShort(ari_cache + ari.u.carry);
            if (ari_ffnum)
            {
                unsigned short s = ari.u.carry - 1;
                do { OutputShort(s); } while (--ari_ffnum);
            }
            ari_cache = ari.u.low32 >> 16; ari.u.carry = 0;
        } else ari_ffnum++;
        ari.u.low32 <<= 16;

        return ari_range << 16;
    }

public:

    INLINE bool CheckEOB()
    {
        return ari_output >= ari_outputEOB;
    }

    INLINE void InitEncoder(unsigned char * output, int outputSize)
    {
        ari_outputStart = (unsigned short *)output;
        ari_output      = (unsigned short *)output;
        ari_outputEOB   = (unsigned short *)(output + outputSize - 16);
        ari.low         = 0;
        ari_ffnum       = 0;
        ari_cache       = 0;
        ari_range       = 0xffffffff;
    };

    INLINE int FinishEncoder()
    {
        if (ari_range < 0x10000)
        {
            ShiftLow();
        }

        ShiftLow(); ShiftLow(); ShiftLow();
        return (int)(ari_output - ari_outputStart) * sizeof(ari_output[0]);
    }

    template <int P = 12> INLINE void EncodeBit0(int probability)
    {
        if (ari_range < 0x10000)
        {
            ari_range = ShiftLow();
        }

        ari_range = (ari_range >> P) * probability;
    }

    template <int P = 12> INLINE void EncodeBit1(int probability)
    {
        if (ari_range < 0x10000)
        {
            ari_range = ShiftLow();
        }

        unsigned int range = (ari_range >> P) * probability;
        ari.low += range; ari_range -= range;
    }

    template <int P = 12> INLINE void EncodeBit(unsigned int bit, int probability)
    {
        if (ari_range < 0x10000)
        {
            ari_range = ShiftLow();
        }

        unsigned int range = (ari_range >> P) * probability;

        ari.low   = ari.low + ((~bit + 1u) & range);
        ari_range = range   + ((~bit + 1u) & (ari_range - range - range));
    }

    INLINE void EncodeBit(unsigned int bit)
    {
        if (bit) EncodeBit1(2048); else EncodeBit0(2048);
    };

    INLINE void EncodeByte(unsigned int byte)
    {
        for (int bit = 7; bit >= 0; --bit)
        {
            EncodeBit(byte & (1 << bit));
        }
    };

    INLINE void EncodeWord(unsigned int word)
    {
        for (int bit = 31; bit >= 0; --bit)
        {
            EncodeBit(word & (1 << bit));
        }
    };

    INLINE void InitDecoder(const unsigned char * input)
    {
        ari_input = (unsigned short *)input;
        ari_code  = 0;
        ari_range = 0xffffffff;
        ari_code  = (ari_code << 16) | InputShort();
        ari_code  = (ari_code << 16) | InputShort();
        ari_code  = (ari_code << 16) | InputShort();
    };

    template <int P = 12> INLINE int PeakBit(int probability)
    {
        if (ari_range < 0x10000)
        {
            ari_range <<= 16; ari_code = (ari_code << 16) | InputShort();
        }

        return ari_code >= (ari_range >> P) * probability;
    }

    template <int P = 12> INLINE int DecodeBit(int probability)
    {
        if (ari_range < 0x10000)
        {
            ari_range <<= 16; ari_code = (ari_code << 16) | InputShort();
        }

        unsigned int range = (ari_range >> P) * probability;
        int bit = ari_code >= range;

        ari_range = bit ? ari_range - range : range;
        ari_code  = bit ? ari_code  - range : ari_code;

        return bit;
    }

    template <int P = 12> INLINE void DecodeBit0(int probability)
    {
        ari_range = (ari_range >> P) * probability;
    }

    template <int P = 12> INLINE void DecodeBit1(int probability)
    {
        unsigned int range = (ari_range >> P) * probability;
        ari_code -= range; ari_range -= range;
    }

    INLINE unsigned int DecodeBit()
    {
        return DecodeBit(2048);
    }

    INLINE unsigned int DecodeByte()
    {
        unsigned int byte = 0;
        for (int bit = 7; bit >= 0; --bit)
        {
            byte += byte + DecodeBit();
        }
        return byte;
    }

    INLINE unsigned int DecodeWord()
    {
        unsigned int word = 0;
        for (int bit = 31; bit >= 0; --bit)
        {
            word += word + DecodeBit();
        }
        return word;
    }
};

#endif

/*-----------------------------------------------------------*/
/* End                                          rangecoder.h */
/*-----------------------------------------------------------*/

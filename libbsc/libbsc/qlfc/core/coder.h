/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Binary arithmetic coder                                   */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

Copyright (c) 2009-2011 Ilya Grebnov <ilya.grebnov@gmail.com>

See file AUTHORS for a full list of contributors.

The bsc and libbsc is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The bsc and libbsc is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the bsc and libbsc. If not, see http://www.gnu.org/licenses/.

Please see the files COPYING and COPYING.LIB for full copyright information.

See also the bsc and libbsc web site:
  http://libbsc.com/ for more information.

--*/

#ifndef _LIBBSC_QLFC_CODER_H
#define _LIBBSC_QLFC_CODER_H

#include "../../platform/platform.h"

class BinaryCoder
{

private:

    long long    ari_low;
    unsigned int ari_code;
    unsigned int ari_ffnum;
    unsigned int ari_cache;
    unsigned int ari_range;

    const unsigned char * input;

    unsigned char * output;
    unsigned char * outputEOB;
    unsigned char * outputStart;

    INLINE void ShiftLow()
    {
        if ((ari_low ^ 0xff000000) > 0xffffff)
        {
            OutputByte((unsigned char)(ari_cache + (ari_low >> 32)));
            int c = (int)(0xff + (ari_low >> 32));
            while (ari_ffnum) OutputByte(c), ari_ffnum--;
            ari_cache = (unsigned int)(ari_low) >> 24;
        } else ari_ffnum++;
        ari_low = (unsigned int)ari_low << 8;
    }

    INLINE void OutputByte(unsigned char b)
    {
        *output++ = b;
    };

    INLINE unsigned char InputByte()
    {
        return *input++;
    };

public:

    INLINE bool CheckEOB()
    {
        return output >= outputEOB;
    }

    INLINE void InitEncoder(unsigned char * output, int n)
    {
        this->outputStart = output;
        this->output      = output;
        this->outputEOB   = output + n - 16;

        ari_low   = 0;
        ari_ffnum = 0;
        ari_cache = 0;
        ari_range = 0xffffffff;
    };

    INLINE int FinishEncoder()
    {
        ari_low += (ari_range >> 1);

        ShiftLow();
        ShiftLow();
        ShiftLow();
        ShiftLow();
        ShiftLow();

        return (int)(output - outputStart);
    }

    INLINE void EncodeBit0(int probability)
    {
        ari_range = (ari_range >> 12) * probability;
        while (ari_range < 0x1000000)
        {
            ShiftLow();
            ari_range <<= 8;
        }
    }

    INLINE void EncodeBit1(int probability)
    {
        unsigned int range = (ari_range >> 12) * probability;
        ari_low += range; ari_range -= range;
        while (ari_range < 0x1000000)
        {
            ShiftLow();
            ari_range <<= 8;
        }
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
        this->input = input;

        ari_code  = 0;
        ari_ffnum = 0;
        ari_cache = 0;
        ari_range = 0xffffffff;

        ari_code = (ari_code << 8) | InputByte();
        ari_code = (ari_code << 8) | InputByte();
        ari_code = (ari_code << 8) | InputByte();
        ari_code = (ari_code << 8) | InputByte();
        ari_code = (ari_code << 8) | InputByte();
    };

    INLINE int DecodeBit(int probability)
    {
        unsigned int range = (ari_range >> 12) * probability;
        if (ari_code >= range)
        {
            ari_code -= range; ari_range -= range;
            while (ari_range < 0x1000000)
            {
                ari_code = (ari_code << 8) | InputByte();
                ari_range <<= 8;
            }
            return 1;
        }
        ari_range = range;
        while (ari_range < 0x1000000)
        {
            ari_code = (ari_code << 8) | InputByte();
            ari_range <<= 8;
        }
        return 0;
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
/* End                                               coder.h */
/*-----------------------------------------------------------*/

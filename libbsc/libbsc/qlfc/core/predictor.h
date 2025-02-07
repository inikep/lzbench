/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Probability counter and logistic mixer                    */
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

#ifndef _LIBBSC_QLFC_PREDICTOR_H
#define _LIBBSC_QLFC_PREDICTOR_H

#include "../../platform/platform.h"

#include "tables.h"

struct ProbabilityCounter
{

public:

    static INLINE void UpdateBit0(short & probability, const int threshold, const int adaptationRate)
    {
        probability = probability + (((4096 - threshold - probability) * adaptationRate) >> 12);
    };

    static INLINE void UpdateBit1(short & probability, const int threshold, const int adaptationRate)
    {
        probability = probability - (((probability - threshold) * adaptationRate) >> 12);
    };

};

struct ProbabilityMixer
{

private:

    short stretchedProbability0;
    short stretchedProbability1;
    short stretchedProbability2;
    int   mixedProbability;
    int   index;

    short probabilityMap[17];

    int weight0;
    int weight1;
    int weight2;

public:

    INLINE void Init()
    {
        weight0 = weight1 = 2048 << 5; weight2 = 0;
        for (int p = 0; p < 17; ++p)
        {
            probabilityMap[p] = bsc_squash((p - 8) * 256);
        }
    }

    INLINE int Mixup(const int probability0, const int probability1, const int probability2)
    {
        stretchedProbability0 = bsc_stretch(probability0);
        stretchedProbability1 = bsc_stretch(probability1);
        stretchedProbability2 = bsc_stretch(probability2);

        short stretchedProbability = (stretchedProbability0 * weight0 + stretchedProbability1 * weight1 + stretchedProbability2 * weight2) >> 17;

        if (stretchedProbability < -2047) stretchedProbability = -2047;
        if (stretchedProbability >  2047) stretchedProbability =  2047;

        index                       = (stretchedProbability + 2048) >> 8;
        const int weight            = stretchedProbability & 255;
        const int probability       = bsc_squash(stretchedProbability);
        const int mappedProbability = probabilityMap[index] + (((probabilityMap[index + 1] - probabilityMap[index]) * weight) >> 8);

        return mixedProbability = (3 * probability + mappedProbability) >> 2;
    };

    INLINE int MixupAndUpdateBit0(const int probability0,  const int probability1,  const int probability2,
                                  const int learningRate0, const int learningRate1, const int learningRate2,
                                  const int threshold,     const int adaptationRate
    )
    {
        const short stretchedProbability0 = bsc_stretch(probability0);
        const short stretchedProbability1 = bsc_stretch(probability1);
        const short stretchedProbability2 = bsc_stretch(probability2);

        short stretchedProbability = (stretchedProbability0 * weight0 + stretchedProbability1 * weight1 + stretchedProbability2 * weight2) >> 17;

        if (stretchedProbability < -2047) stretchedProbability = -2047;
        if (stretchedProbability >  2047) stretchedProbability =  2047;

        const int weight            = stretchedProbability & 255;
        const int index             = (stretchedProbability + 2048) >> 8;
        const int probability       = bsc_squash(stretchedProbability);
        const int mappedProbability = probabilityMap[index] + (((probabilityMap[index + 1] - probabilityMap[index]) * weight) >> 8);
        const int mixedProbability  = (3 * probability + mappedProbability) >> 2;

        ProbabilityCounter::UpdateBit0(probabilityMap[index], threshold, adaptationRate);
        ProbabilityCounter::UpdateBit0(probabilityMap[index + 1], threshold, adaptationRate);

        const int eps = mixedProbability - 4095;

        weight0 -= (learningRate0 * eps * stretchedProbability0) >> 16;
        weight1 -= (learningRate1 * eps * stretchedProbability1) >> 16;
        weight2 -= (learningRate2 * eps * stretchedProbability2) >> 16;

        return mixedProbability;
    };

    INLINE int MixupAndUpdateBit1(const int probability0,  const int probability1,  const int probability2,
                                  const int learningRate0, const int learningRate1, const int learningRate2,
                                  const int threshold,     const int adaptationRate
    )
    {
        const short stretchedProbability0 = bsc_stretch(probability0);
        const short stretchedProbability1 = bsc_stretch(probability1);
        const short stretchedProbability2 = bsc_stretch(probability2);

        short stretchedProbability = (stretchedProbability0 * weight0 + stretchedProbability1 * weight1 + stretchedProbability2 * weight2) >> 17;

        if (stretchedProbability < -2047) stretchedProbability = -2047;
        if (stretchedProbability >  2047) stretchedProbability =  2047;

        const int weight            = stretchedProbability & 255;
        const int index             = (stretchedProbability + 2048) >> 8;
        const int probability       = bsc_squash(stretchedProbability);
        const int mappedProbability = probabilityMap[index] + (((probabilityMap[index + 1] - probabilityMap[index]) * weight) >> 8);
        const int mixedProbability  = (3 * probability + mappedProbability) >> 2;

        ProbabilityCounter::UpdateBit1(probabilityMap[index], threshold, adaptationRate);
        ProbabilityCounter::UpdateBit1(probabilityMap[index + 1], threshold, adaptationRate);

        const int eps = mixedProbability - 1;

        weight0 -= (learningRate0 * eps * stretchedProbability0) >> 16;
        weight1 -= (learningRate1 * eps * stretchedProbability1) >> 16;
        weight2 -= (learningRate2 * eps * stretchedProbability2) >> 16;

        return mixedProbability;
    };

    INLINE void UpdateBit0(const int learningRate0, const int learningRate1, const int learningRate2,
                           const int threshold,     const int adaptationRate
    )
    {
        ProbabilityCounter::UpdateBit0(probabilityMap[index], threshold, adaptationRate);
        ProbabilityCounter::UpdateBit0(probabilityMap[index + 1], threshold, adaptationRate);

        const int eps = mixedProbability - 4095;

        weight0 -= (learningRate0 * eps * stretchedProbability0) >> 16;
        weight1 -= (learningRate1 * eps * stretchedProbability1) >> 16;
        weight2 -= (learningRate2 * eps * stretchedProbability2) >> 16;
    };

    INLINE void UpdateBit1(const int learningRate0, const int learningRate1, const int learningRate2,
                           const int threshold,     const int adaptationRate
    )
    {
        ProbabilityCounter::UpdateBit1(probabilityMap[index], threshold, adaptationRate);
        ProbabilityCounter::UpdateBit1(probabilityMap[index + 1], threshold, adaptationRate);

        const int eps = mixedProbability - 1;

        weight0 -= (learningRate0 * eps * stretchedProbability0) >> 16;
        weight1 -= (learningRate1 * eps * stretchedProbability1) >> 16;
        weight2 -= (learningRate2 * eps * stretchedProbability2) >> 16;
    };

};


#endif

/*-----------------------------------------------------------*/
/* End                                           predictor.h */
/*-----------------------------------------------------------*/

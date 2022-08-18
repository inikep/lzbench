/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Probability counter and logistic mixer                    */
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

#ifndef _LIBBSC_CODER_PREDICTOR_H
#define _LIBBSC_CODER_PREDICTOR_H

#include "../../platform/platform.h"

#include "tables.h"

struct ProbabilityCounter
{

public:

    static INLINE void UpdateBit(unsigned int bit, short & probability, const int threshold0, const int adaptationRate0, const int threshold1, const int adaptationRate1)
    {
        int delta0 = probability * adaptationRate0 - ((4096 - threshold0) * adaptationRate0 - 4095);
        int delta1 = probability * adaptationRate1 - (threshold1 * adaptationRate1);
        
        probability = probability - ((bit ? delta1 : delta0) >> 12);
    }

    static INLINE void UpdateBit0(short & probability, const int threshold, const int adaptationRate)
    {
        probability = probability + (((4096 - threshold - probability) * adaptationRate) >> 12);
    };

    static INLINE void UpdateBit1(short & probability, const int threshold, const int adaptationRate)
    {
        probability = probability - (((probability - threshold) * adaptationRate) >> 12);
    };

    template <int R> static INLINE void UpdateBit(unsigned int bit, short & probability, const int threshold0, const int threshold1)
    {
        probability = probability - ((probability - (bit ? threshold1 : threshold0)) >> R);
    }

    template <int R> static INLINE void UpdateBit(short & probability, const int threshold)
    {
        probability = probability - ((probability - threshold) >> R);
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

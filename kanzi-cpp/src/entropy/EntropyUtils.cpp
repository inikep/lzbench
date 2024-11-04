/*
Copyright 2011-2024 Frederic Langlet
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <algorithm>
#include <deque>
#include <sstream>
#include "EntropyUtils.hpp"

using namespace kanzi;
using namespace std;

class FreqSortData {
public:
    uint* _freq;
    uint _symbol;

    FreqSortData(uint* freq, int symbol) :
         _freq(freq)
       , _symbol(symbol)
    {
    }
};

struct FreqDataComparator {
    bool operator()(FreqSortData const& fd1, FreqSortData const& fd2) const
    {
        // Decreasing frequency then decreasing symbol
        int r;
        return  ((r = *fd1._freq - *fd2._freq) == 0) ? fd1._symbol > fd2._symbol: r > 0;
    }
};

// alphabet must be sorted in increasing order
// length = alphabet array length up to 256
int EntropyUtils::encodeAlphabet(OutputBitStream& obs, uint alphabet[], int length, int count)
{
    // Alphabet length must be a power of 2
    if ((length & (length - 1)) != 0)
        return -1;

    if ((length > 256) || (count > length))
        return -1;

    if (count == 0) {
        obs.writeBit(FULL_ALPHABET);
        obs.writeBit(ALPHABET_0);
    }
    else if (count == 256) {
        obs.writeBit(FULL_ALPHABET);
        obs.writeBit(ALPHABET_256);
    }
    else {
        // Partial alphabet
        obs.writeBit(PARTIAL_ALPHABET);
        byte masks[32] = { byte(0) };

        for (int i = 0; i < count; i++)
            masks[alphabet[i] >> 3] |= byte(1 << (alphabet[i] & 7));

        const int lastMask = alphabet[count - 1] >> 3;
        obs.writeBits(lastMask, 5);

        for (int i = 0; i <= lastMask; i++)
            obs.writeBits(uint64(masks[i]), 8);
    }

    return count;
}

int EntropyUtils::decodeAlphabet(InputBitStream& ibs, uint alphabet[]) THROW
{
    // Read encoding mode from bitstream
    if (ibs.readBit() == FULL_ALPHABET) {
        const int alphabetSize = (ibs.readBit() == ALPHABET_256) ? 256 : 0;

        // Full alphabet
        for (int i = 0; i < alphabetSize; i++)
            alphabet[i] = i;

        return alphabetSize;
    }

    // Partial alphabet
    const int lastMask = int(ibs.readBits(5));
    int count = 0;

    // Decode presence flags
    for (int i = 0; i <= lastMask; i++) {
        const byte mask = byte(ibs.readBits(8));

        for (int j = 0; j < 8; j++) {
            if ((mask & byte(1 << j)) != byte(0)) {
                alphabet[count++] = (i << 3) + j;
            }
        }
    }

    return count;
}


// Returns the size of the alphabet
// length is the length of the alphabet array
// 'totalFreq 'is the sum of frequencies.
// 'scale' is the target new total of frequencies
// The alphabet and freqs parameters are updated
int EntropyUtils::normalizeFrequencies(uint freqs[], uint alphabet[], int length, uint totalFreq, uint scale) THROW
{
    if (length > 256) {
        stringstream ss;
        ss << "Invalid alphabet size parameter: " << scale << " (must be less than or equal to 256)";
        throw invalid_argument(ss.str());
    }

    if ((scale < 256) || (scale > 65536)) {
        stringstream ss;
        ss << "Invalid scale parameter: " << scale << " (must be in [256..65536])";
        throw invalid_argument(ss.str());
    }

    if ((length == 0) || (totalFreq == 0))
        return 0;

    // Number of present symbols
    int alphabetSize = 0;

    // shortcut
    if (totalFreq == scale) {
        for (int i = 0; i < 256; i++) {
            if (freqs[i] != 0)
                alphabet[alphabetSize++] = i;
        }

        return alphabetSize;
    }

    uint sumScaledFreq = 0;
    uint sumFreq = 0;
    uint freqMax = 0;
    int idxMax = -1;

    // Scale frequencies by stretching distribution over complete range
    for (int i = 0; (i < length) && (sumFreq < totalFreq); i++) {
        alphabet[i] = 0;
        const uint f = freqs[i];

        if (f == 0)
            continue;

        if (f > freqMax) {
            freqMax = f;
            idxMax = i;
        }

        sumFreq += f;
        int64 sf = int64(f) * int64(scale);
        uint scaledFreq;

        if (sf <= int64(totalFreq)) {
            // Quantum of frequency
            scaledFreq = 1;
        }
        else {
            // Find best frequency rounding value
            scaledFreq = uint(sf / int64(totalFreq));
            const int64 prod = int64(scaledFreq) * int64(totalFreq);
            const int64 errCeiling = prod + int64(totalFreq) - sf;
            const int64 errFloor = sf - prod;

            if (errCeiling < errFloor)
                scaledFreq++;
        }

        alphabet[alphabetSize++] = i;
        sumScaledFreq += scaledFreq;
        freqs[i] = scaledFreq;
    }

    if (alphabetSize == 0)
        return 0;

    if (alphabetSize == 1) {
        freqs[alphabet[0]] = scale;
        return 1;
    }

    if (sumScaledFreq == scale)
        return alphabetSize;

    const int delta = int(sumScaledFreq - scale);

    if (abs(delta) * 100 < int(freqs[idxMax]) * 5) {
        // Fast path: just adjust the max frequency 
        freqs[idxMax] -= delta;
        return alphabetSize;
    }
   
    // Slow path: spread error across frequencies
    const int inc = (sumScaledFreq > scale) ? -1 : 1;
    deque<FreqSortData> queue;

    // Create sorted queue of present symbols
    for (int i = 0; i < alphabetSize; i++) {
        if (int(freqs[alphabet[i]]) == -inc)
            continue;
        
        if (alphabetSize * freqs[alphabet[i]] >= scale)
            queue.push_front(FreqSortData(&freqs[alphabet[i]], alphabet[i]));
        else
            queue.push_back(FreqSortData(&freqs[alphabet[i]], alphabet[i]));
    }

    if (queue.empty()) {
        freqs[idxMax] -= delta;
        return alphabetSize;
    }

    sort(queue.begin(), queue.end(), FreqDataComparator());

    while (sumScaledFreq != scale) {
        // Remove next symbol
#if __cplusplus >= 201103L
        FreqSortData fsd = move(queue.front());
#else
        FreqSortData fsd = queue.front();
#endif
        queue.pop_front();

        // Do not zero out any frequency
        if (int(*fsd._freq) == -inc) {
            continue;
        }
           
        // Distort frequency and re-enqueue
        *fsd._freq += inc;
        sumScaledFreq += inc;
        queue.push_back(fsd);
    }

    return alphabetSize;
}

int EntropyUtils::writeVarInt(OutputBitStream& obs, uint32 value)
{
    uint32 res = 0;

    while (value >= 128) {
        obs.writeBits(0x80 | (value & 0x7F), 8);
        value >>= 7;
        res++;
    }

    obs.writeBits(value, 8);
    return res;
}

uint32 EntropyUtils::readVarInt(InputBitStream& ibs)
{
    uint32 value = uint32(ibs.readBits(8));
    uint32 res = value & 0x7F;
    int shift = 7;

    while ((value >= 128) && (shift <= 28)) {
        value = uint32(ibs.readBits(8));
        res |= ((value & 0x7F) << shift);
        shift += 7;
    }

    return res;
}

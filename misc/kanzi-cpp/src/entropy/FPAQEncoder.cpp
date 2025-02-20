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
#include <stdexcept>
#include "FPAQEncoder.hpp"
#include "EntropyUtils.hpp"

using namespace kanzi;
using namespace std;

FPAQEncoder::FPAQEncoder(OutputBitStream& bitstream)
    : _bitstream(bitstream)
    , _sba(new byte[0], 0)
{
    reset();
}

FPAQEncoder::~FPAQEncoder()
{
    _dispose();
    delete[] _sba._array;
}

bool FPAQEncoder::reset()
{
    _low = 0;
    _high = TOP;
    _disposed = false;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 256; j++)
            _probs[i][j] = PSCALE >> 1;
    }

    return true;
}

int FPAQEncoder::encode(const byte block[], uint blkptr, uint count)
{
    if (count >= MAX_BLOCK_SIZE)
        throw invalid_argument("Invalid block size parameter (max is 1<<30)");

    uint startChunk = blkptr;
    const uint end = blkptr + count;

    // Split block into chunks, encode chunk and write bit array to bitstream
    while (startChunk < end) {
        const uint chunkSize = min(DEFAULT_CHUNK_SIZE, end - startChunk);

        if (_sba._length < int(chunkSize + (chunkSize >> 3))) {
            delete[] _sba._array;
            _sba._length = chunkSize + (chunkSize >> 3);
            _sba._array = new byte[_sba._length];
        }

        _sba._index = 0;
        const uint endChunk = startChunk + chunkSize;
        int ctx = 0;

        for (uint i = startChunk; i < endChunk; i++) {
            const int val = int(block[i]);
            const int bits = val + 256;
            encodeBit(val & 0x80, _probs[ctx][1]);
            encodeBit(val & 0x40, _probs[ctx][bits >> 7]);
            encodeBit(val & 0x20, _probs[ctx][bits >> 6]);
            encodeBit(val & 0x10, _probs[ctx][bits >> 5]);
            encodeBit(val & 0x08, _probs[ctx][bits >> 4]);
            encodeBit(val & 0x04, _probs[ctx][bits >> 3]);
            encodeBit(val & 0x02, _probs[ctx][bits >> 2]);
            encodeBit(val & 0x01, _probs[ctx][bits >> 1]);
            ctx = val >> 6;
        }

        EntropyUtils::writeVarInt(_bitstream, uint32(_sba._index));
        _bitstream.writeBits(&_sba._array[0], 8 * _sba._index);
        startChunk += chunkSize;

        if (startChunk < end)
            _bitstream.writeBits(_low | MASK_0_24, 56);
    }

    return count;
}

void FPAQEncoder::_dispose()
{
    if (_disposed == true)
        return;

    _disposed = true;
    _bitstream.writeBits(_low | MASK_0_24, 56);
}

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
#include "FPAQDecoder.hpp"
#include "EntropyUtils.hpp"

using namespace kanzi;
using namespace std;

FPAQDecoder::FPAQDecoder(InputBitStream& bitstream) THROW
    : _low(0)
    , _high(TOP)
    , _current(0)
    , _bitstream(bitstream)
    , _sba(new byte[0], 0)
{
    reset();
}

FPAQDecoder::~FPAQDecoder()
{
    _dispose();
    delete[] _sba._array;
}

bool FPAQDecoder::reset()
{
    _low = 0;
    _high = TOP;
    _current = 0;
    _ctx = 1;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 256; j++)
            _probs[i][j] = PSCALE >> 1;
    }

    _p = _probs[0];
    return true;
}

int FPAQDecoder::decode(byte block[], uint blkptr, uint count)
{
    if (count >= MAX_BLOCK_SIZE)
        throw invalid_argument("Invalid block size parameter (max is 1<<30)");

    uint startChunk = blkptr;
    const uint end = blkptr + count;

    // Split block into chunks, read bit array from bitstream and decode chunk
    while (startChunk < end) {
        const uint chunkSize = min(DEFAULT_CHUNK_SIZE, end - startChunk);
        const int szBytes = int(EntropyUtils::readVarInt(_bitstream));
        _current = _bitstream.readBits(56);

        if (szBytes == 0)
           break;

        if (_sba._length < szBytes) {
            delete[] _sba._array;
            _sba._length = szBytes + (szBytes >> 3);
            _sba._array = new byte[_sba._length];
        }

        _bitstream.readBits(&_sba._array[0], 8 * szBytes);
        _sba._index = 0;
        const uint endChunk = startChunk + chunkSize;
        _p = _probs[0];

        for (uint i = startChunk; i < endChunk; i++) {
            _ctx = 1;
            decodeBit(int(_p[_ctx]));
            decodeBit(int(_p[_ctx]));
            decodeBit(int(_p[_ctx]));
            decodeBit(int(_p[_ctx]));
            decodeBit(int(_p[_ctx]));
            decodeBit(int(_p[_ctx]));
            decodeBit(int(_p[_ctx]));
            decodeBit(int(_p[_ctx]));
            block[i] = byte(_ctx);
            _p = _probs[(_ctx & 0xFF) >> 6];
        }

        startChunk = endChunk;
    }

    return count;
}



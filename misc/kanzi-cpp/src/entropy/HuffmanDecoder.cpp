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
#include <sstream>
#include "HuffmanDecoder.hpp"
#include "EntropyUtils.hpp"
#include "ExpGolombDecoder.hpp"
#include "../BitStreamException.hpp"
#include "../Memory.hpp"

using namespace kanzi;
using namespace std;

// The chunk size indicates how many bytes are encoded (per block) before
// resetting the frequency stats.
HuffmanDecoder::HuffmanDecoder(InputBitStream& bitstream, int chunkSize) : _bitstream(bitstream)
{
    if (chunkSize < 1024)
        throw invalid_argument("Huffman codec: The chunk size must be at least 1024");

    if (chunkSize > HuffmanCommon::MAX_CHUNK_SIZE) {
        stringstream ss;
        ss << "Huffman codec: The chunk size must be at most " << HuffmanCommon::MAX_CHUNK_SIZE;
        throw invalid_argument(ss.str());
    }

    _chunkSize = chunkSize;
    _buffer = new byte[0];
    _bufferSize = 0;
    reset();
}

bool HuffmanDecoder::reset()
{
    // Default lengths & canonical codes
    for (uint16 i = 0; i < 256; i++) {
        _codes[i] = i;
        _sizes[i] = 8;
    }

    memset(_alphabet, 0, sizeof(_alphabet));
    memset(_table, 0, sizeof(_table));
    return true;
}

int HuffmanDecoder::readLengths()
{
    const int count = EntropyUtils::decodeAlphabet(_bitstream, _alphabet);

    if (count == 0)
        return 0;

    ExpGolombDecoder egdec(_bitstream, true);
    int8 curSize = 2;

    // Read lengths from bitstream
    for (int i = 0; i < count; i++) {
        const uint s = _alphabet[i];

        if (s > 255) {
            stringstream ss;
            ss << "Invalid bitstream: incorrect Huffman symbol " << s;
            throw BitStreamException(ss.str(), BitStreamException::INVALID_STREAM);
        }

        _codes[s] = 0;
        curSize += int8(egdec.decodeByte());

        if ((curSize <= 0) || (curSize > HuffmanCommon::MAX_SYMBOL_SIZE)) {
            stringstream ss;
            ss << "Invalid bitstream: incorrect size " << int(curSize);
            ss << " for Huffman symbol " << s;
            throw BitStreamException(ss.str(), BitStreamException::INVALID_STREAM);
        }

        _sizes[s] = uint16(curSize);
    }

    // Create canonical codes
    if (HuffmanCommon::generateCanonicalCodes(_sizes, _codes, _alphabet, count) < 0) {
        stringstream ss;
        ss << "Could not generate Huffman codes: max code length (";
        ss << HuffmanCommon::MAX_SYMBOL_SIZE;
        ss << " bits) exceeded";
        throw BitStreamException(ss.str(), BitStreamException::INVALID_STREAM);
    }

    return count;
}

// max(CodeLen) must be <= MAX_SYMBOL_SIZE
void HuffmanDecoder::buildDecodingTable(int count)
{
    memset(_table, 0, sizeof(_table));
    int length = 0;

    for (int i = 0; i < count; i++) {
        const uint16 s = uint16(_alphabet[i]);

        if (_sizes[s] > length)
            length = _sizes[s];

        // code -> size, symbol
        const uint16 val = (s << 8) | _sizes[s];

        // All DECODING_BATCH_SIZE bit values read from the bit stream and
        // starting with the same prefix point to symbol s
        uint idx = uint(_codes[s]) << (DECODING_BATCH_SIZE - length);
        const uint end = idx + (1 << (DECODING_BATCH_SIZE - length));

        while (idx < end)
            _table[idx++] = val;
    }
}

int HuffmanDecoder::decode(byte block[], uint blkptr, uint count)
{
    if (count == 0)
        return 0;

    uint startChunk = blkptr;
    const uint end = blkptr + count;

    while (startChunk < end) {
        const uint endChunk = min(startChunk + _chunkSize, end);

        // For each chunk, read code lengths, rebuild codes, rebuild decoding table
        const int alphabetSize = readLengths();

        if (alphabetSize <= 0)
            return startChunk - blkptr;

        if (alphabetSize == 1) {
            // Shortcut for chunks with only one symbol
            memset(&block[startChunk], _alphabet[0], size_t(endChunk - startChunk));
            startChunk = endChunk;
            continue;
        }

        buildDecodingTable(alphabetSize);

        // Read number of streams. Only 1 steam supported for now
        if (_bitstream.readBits(2) != 0)
            return -1;

        // Read chunk size
        const int szBits = EntropyUtils::readVarInt(_bitstream);

        if (szBits < 0)
            return -1;

        // Read compressed data from bitstream
        if (szBits != 0) {
            const int sz = (szBits + 7) >> 3;
            const uint minLenBuf = uint(max(sz + (sz >> 3), 1024));

            if (_bufferSize < minLenBuf) {
                delete[] _buffer;
                _bufferSize = minLenBuf;
                _buffer = new byte[_bufferSize];
            }

            _bitstream.readBits(&_buffer[0], szBits);

            uint64 state = 0; // holds bits read from bitstream
            uint8 bits = 0; // number of available bits in state
            int idx = 0;
            uint n = startChunk;

            while (idx < sz - 8) {
                const uint8 shift = (56 - bits) & -8;
                state = (state << shift) | (uint64(BigEndian::readLong64(&_buffer[idx])) >> 1 >> (63 - shift)); // handle shift = 0
                idx += (shift >> 3);
                uint8 bs = bits + shift - DECODING_BATCH_SIZE;
                const uint16 val0 = _table[(state >> bs) & TABLE_MASK];
                bs -= uint8(val0);
                const uint16 val1 = _table[(state >> bs) & TABLE_MASK];
                bs -= uint8(val1);
                const uint16 val2 = _table[(state >> bs) & TABLE_MASK];
                bs -= uint8(val2);
                const uint16 val3 = _table[(state >> bs) & TABLE_MASK];
                bs -= uint8(val3);
                bits = bs + DECODING_BATCH_SIZE;
                block[n + 0] = byte(val0 >> 8);
                block[n + 1] = byte(val1 >> 8);
                block[n + 2] = byte(val2 >> 8);
                block[n + 3] = byte(val3 >> 8);
                n += 4;
            }

            // Last bytes
            uint nbBits = idx * 8;

            while (n < endChunk) {
                while ((bits < HuffmanCommon::MAX_SYMBOL_SIZE) && (idx < sz)) {
                    state = (state << 8) | uint64(_buffer[idx] & byte(0xFF));
                    idx++;
                    nbBits = (idx == sz) ? szBits : nbBits + 8;

                    // 'bits' may overshoot when idx == sz due to padding state bits
                    // It is necessary to compute proper _table indexes
                    // and has no consequence (except bits != 0 at end of chunk)
                    bits += 8;
                }

                uint16 val;

                if (bits >= DECODING_BATCH_SIZE)
                    val = _table[(state >> (bits - DECODING_BATCH_SIZE)) & TABLE_MASK];
                else
                    val = _table[(state << (DECODING_BATCH_SIZE - bits)) & TABLE_MASK];

                bits -= uint8(val);
                block[n++] = byte(val >> 8);
            }
        }

        startChunk = endChunk;
    }

    return count;
}


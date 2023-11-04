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

#include <sstream>
#include <algorithm>
#include <vector>
#include "HuffmanEncoder.hpp"
#include "EntropyUtils.hpp"
#include "ExpGolombEncoder.hpp"
#include "../Global.hpp"
#include "../Memory.hpp"

using namespace kanzi;
using namespace std;

// The chunk size indicates how many bytes are encoded (per block) before
// resetting the frequency stats. 0 means that frequencies calculated at the
// beginning of the block apply to the whole block.
HuffmanEncoder::HuffmanEncoder(OutputBitStream& bitstream, int chunkSize) THROW : _bitstream(bitstream)
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

bool HuffmanEncoder::reset()
{
    for (int i = 0; i < 256; i++)
        _codes[i] = i;

    return true;
}

// Rebuild Huffman codes
int HuffmanEncoder::updateFrequencies(uint freqs[]) THROW
{
    int count = 0;
    uint16 sizes[256] = { 0 };
    uint alphabet[256] = { 0 };

    for (int i = 0; i < 256; i++) {
        _codes[i] = 0;

        if (freqs[i] > 0)
            alphabet[count++] = i;
    }

    EntropyUtils::encodeAlphabet(_bitstream, alphabet, 256, count);

    if (count == 0)
        return 0;

    if (count == 1) {
        _codes[alphabet[0]] = 1 << 24;
        sizes[alphabet[0]] = 1;
    }
    else {
        int retries = 0;
        uint ranks[256]; // sorted ranks

        while (true) {
            for (int i = 0; i < count; i++)
                ranks[i] = (freqs[alphabet[i]] << 8) | alphabet[i];

            const uint maxCodeLen = computeCodeLengths(sizes, ranks, count);

            if (maxCodeLen == 0) {
                throw invalid_argument("Could not generate Huffman codes: invalid code length 0");
            }

            if (maxCodeLen <= HuffmanCommon::MAX_SYMBOL_SIZE) {
                // Usual case
                HuffmanCommon::generateCanonicalCodes(sizes, _codes, ranks, count);
                break;
            }

            // Sometimes, codes exceed the budget for the max code length => scale down
            // and normalize frequencies (boost the smallest freqs) and try once more.
            if (retries > 2) {
                stringstream ss;
                ss << "Could not generate Huffman codes: max code length (";
                ss << HuffmanCommon::MAX_SYMBOL_SIZE;
                ss << " bits) exceeded";
                throw invalid_argument(ss.str());
            }

            retries++;
            uint alpha[256] = { 0 };
            uint f[256];
            uint totalFreq = 0;

            for (int i = 0; i < count; i++) {
                f[i] = freqs[alphabet[i]];
                totalFreq += f[i];
            }

            // Normalize to a smaller scale
            EntropyUtils::normalizeFrequencies(f, alpha, count, totalFreq,
                HuffmanCommon::MAX_CHUNK_SIZE >> (retries + 1));

            for (int i = 0; i < count; i++)
                freqs[alphabet[i]] = f[i];
        }
    }

    // Transmit code lengths only, freqs and codes do not matter
    ExpGolombEncoder egenc(_bitstream, true);
    uint16 prevSize = 2;

    // Pack size and code (size <= MAX_SYMBOL_SIZE bits)
    // Unary encode the code length differences
    for (int i = 0; i < count; i++) {
        const int s = alphabet[i];
        _codes[s] |= (sizes[s] << 24);
        egenc.encodeByte(byte(sizes[s] - prevSize));
        prevSize = sizes[s];
    }

    return count;
}

uint HuffmanEncoder::computeCodeLengths(uint16 sizes[], uint ranks[], int count)
{
    // Sort ranks by increasing freqs (first key) and increasing value (second key)
    vector<uint> v(ranks, ranks + count);
    sort(v.begin(), v.end());
    uint freqs[256] = { 0 };

    for (int i = 0; i < count; i++) {
        freqs[i] = v[i] >> 8;
        ranks[i] = v[i] & 0xFF;

        if (freqs[i] == 0)
            return 0;
    }

    // See [In-Place Calculation of Minimum-Redundancy Codes]
    // by Alistair Moffat & Jyrki Katajainen
    computeInPlaceSizesPhase1(freqs, count);
    computeInPlaceSizesPhase2(freqs, count);
    uint maxCodeLen = 0;

    for (int i = 0; i < count; i++) {
        const uint codeLen = freqs[i];

        if (maxCodeLen < codeLen) {
            maxCodeLen = codeLen;

            if (maxCodeLen > HuffmanCommon::MAX_SYMBOL_SIZE)
                break;
        }

        sizes[ranks[i]] = uint16(codeLen);
    }

    return maxCodeLen;
}

void HuffmanEncoder::computeInPlaceSizesPhase1(uint data[], int n)
{
    for (int s = 0, r = 0, t = 0; t < n - 1; t++) {
        uint sum = 0;

        for (int i = 0; i < 2; i++) {
            if ((s >= n) || ((r < t) && (data[r] < data[s]))) {
                sum += data[r];
                data[r] = t;
                r++;
                continue;
            }

            sum += data[s];

            if (s > t)
                data[s] = 0;

            s++;
        }

        data[t] = sum;
    }
}

void HuffmanEncoder::computeInPlaceSizesPhase2(uint data[], int n)
{
    uint topLevel = n - 2; //root
    uint depth = 1;
    uint i = n;
    uint totalNodesAtLevel = 2;

    while (i > 0) {
        uint k = topLevel;

        while ((k != 0) && (data[k - 1] >= topLevel))
            k--;

        const int internalNodesAtLevel = topLevel - k;
        const int leavesAtLevel = totalNodesAtLevel - internalNodesAtLevel;

        for (int j = 0; j < leavesAtLevel; j++)
            data[--i] = depth;

        totalNodesAtLevel = internalNodesAtLevel << 1;
        topLevel = k;
        depth++;
    }
}

// Dynamically compute the frequencies for every chunk of data in the block
int HuffmanEncoder::encode(const byte block[], uint blkptr, uint count)
{
    if (count == 0)
        return 0;

    const uint end = blkptr + count;
    uint startChunk = blkptr;
    uint sz = uint(_chunkSize);
    const uint minLenBuf = max(min(sz + (sz >> 3), 2 * count), uint(65536));

    if (_bufferSize < minLenBuf) {
        delete[] _buffer;
        _bufferSize = minLenBuf;
        _buffer = new byte[_bufferSize];
    }

    while (startChunk < end) {
        // Update frequencies and rebuild Huffman codes
        const uint endChunk = min(startChunk + _chunkSize, end);
        uint freqs[256] = { 0 };
        Global::computeHistogram(&block[startChunk], endChunk - startChunk, freqs);

        if (updateFrequencies(freqs) <= 1) {
           // Skip chunk if only one symbol
           startChunk = endChunk;
           continue;
        }

        const uint endChunk4 = ((endChunk - startChunk) & -4) + startChunk;
        int idx = 0;
        uint64 state = 0;
        int bits = 0; // accumulated bits

        // Encode chunk
        for (uint i = startChunk; i < endChunk4; i += 4) {
            uint code;
            code = _codes[int(block[i])];
            const uint codeLen0 = code >> 24;
            state = (state << codeLen0) | uint64(code & 0xFFFFFF);
            code = _codes[int(block[i + 1])];
            const uint codeLen1 = code >> 24;
            state = (state << codeLen1) | uint64(code & 0xFFFFFF);
            code = _codes[int(block[i + 2])];
            const uint codeLen2 = code >> 24;
            state = (state << codeLen2) | uint64(code & 0xFFFFFF);
            code = _codes[int(block[i + 3])];
            const uint codeLen3 = code >> 24;
            state = (state << codeLen3) | uint64(code & 0xFFFFFF);
            bits += (codeLen0 + codeLen1 + codeLen2 + codeLen3);
            const uint8 shift = bits & -8;
            BigEndian::writeLong64(&_buffer[idx], state << (64 - bits));
            bits -= shift;
            idx += (shift >> 3);
        }

        for (uint i = endChunk4; i < endChunk; i++) {
            const uint code = _codes[int(block[i])];
            const uint codeLen = code >> 24;
            state = (state << codeLen) | uint64(code & 0xFFFFFF);
            bits += codeLen;
        }

        const uint nbBits = (idx * 8) + bits;

        while (bits >= 8) {
            bits -= 8;
            _buffer[idx++] = byte(state >> bits);
        }

        if (bits > 0)
            _buffer[idx++] = byte(state << (8 - bits));

        // Write number of streams (0->1, 1->4, 2->8, 3->32)
        _bitstream.writeBits(uint64(0), 2);

        // Write chunk size in bits
        EntropyUtils::writeVarInt(_bitstream, nbBits);

        // Write compressed data to bitstream
        _bitstream.writeBits(&_buffer[0], nbBits);

        startChunk = endChunk;
    }

    return count;
}

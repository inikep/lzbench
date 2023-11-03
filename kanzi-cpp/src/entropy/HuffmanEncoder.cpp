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
#include <vector>
#include <sstream>

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
HuffmanEncoder::HuffmanEncoder(OutputBitStream& bitstream, int chunkSize) : _bitstream(bitstream)
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
    for (uint16 i = 0; i < 256; i++)
        _codes[i] = i;

    return true;
}

// Rebuild Huffman codes
int HuffmanEncoder::updateFrequencies(uint freqs[])
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
        _codes[alphabet[0]] = 1 << 12;
        sizes[alphabet[0]] = 1;
    }
    else {
        uint ranks[256]; // sorted ranks

        for (int i = 0; i < count; i++)
            ranks[i] = (freqs[alphabet[i]] << 8) | alphabet[i];

        uint maxCodeLen = computeCodeLengths(sizes, ranks, count);

        if (maxCodeLen == 0) {
            throw invalid_argument("Could not generate Huffman codes: invalid code length 0");
        }

        if (maxCodeLen > HuffmanCommon::MAX_SYMBOL_SIZE) {
            maxCodeLen = limitCodeLengths(alphabet, freqs, sizes, ranks, count);

            if (maxCodeLen == 0) {
                throw invalid_argument("Could not generate Huffman codes: invalid code length 0");
            }

            if (maxCodeLen > HuffmanCommon::MAX_SYMBOL_SIZE) {
               stringstream ss;
               ss << "Could not generate Huffman codes: max code length (";
               ss << HuffmanCommon::MAX_SYMBOL_SIZE << " bits) exceeded";
               throw length_error(ss.str());
            }
        }

        HuffmanCommon::generateCanonicalCodes(sizes, _codes, ranks, count);
    }

    // Transmit code lengths only, freqs and codes do not matter
    ExpGolombEncoder egenc(_bitstream, true);
    uint16 prevSize = 2;

    // Pack size and code (size <= MAX_SYMBOL_SIZE bits)
    // Unary encode the code length differences
    for (int i = 0; i < count; i++) {
        const int s = alphabet[i];
        _codes[s] |= uint16(sizes[s] << 12);
        egenc.encodeByte(byte(sizes[s] - prevSize));
        prevSize = sizes[s];
    }

    return count;
}


uint HuffmanEncoder::limitCodeLengths(const uint alphabet[], uint freqs[], uint16 sizes[], uint ranks[], int count) const
{
   int n = 0;
   int debt = 0;

   // Fold over-the-limit sizes, skip at-the-limit sizes => incur bit debt
   while (sizes[ranks[n]] >= HuffmanCommon::MAX_SYMBOL_SIZE) {
       debt += (sizes[ranks[n]] - HuffmanCommon::MAX_SYMBOL_SIZE);
       sizes[ranks[n]] = HuffmanCommon::MAX_SYMBOL_SIZE;
       n++;
   }

   // Check (up to) 6 levels; one vector per size delta
   vector<int> v[6];

   while (n < count) {
       const int idx = HuffmanCommon::MAX_SYMBOL_SIZE - 1 - sizes[ranks[n]];

       if ((idx > 5) || (debt < (1 << idx)))
          break;

       v[idx].push_back(n);
       n++;
   }

   int idx = 5;

   // Repay bit debt in a "semi optimized" way
   while ((debt > 0) && (idx >= 0)) {
      if ((v[idx].empty() == true) || (debt < (1 << idx))) {
         idx--;
         continue;
      }

      sizes[ranks[v[idx][0]]]++;
      debt -= (1 << idx);
      v[idx].erase(v[idx].begin());
   }

   idx = 0;

   // Adjust if necessary
   while ((debt > 0) && (idx < 6)) {
      if (v[idx].empty() == true) {
         idx++;
         continue;
      }

      sizes[ranks[v[idx][0]]]++;
      debt -= (1 << idx);
      v[idx].erase(v[idx].begin());
   }

   if (debt > 0) {
       // Fallback to slow (more accurate) path if fast path failed to repay the debt
       uint alpha[256] = { 0 };
       uint f[256];
       uint totalFreq = 0;

       for (int i = 0; i < count; i++) {
           f[i] = freqs[alphabet[i]];
           totalFreq += f[i];
       }

       // Renormalize to a smaller scale
       EntropyUtils::normalizeFrequencies(f, alpha, count, totalFreq, HuffmanCommon::MAX_CHUNK_SIZE >> 3);

       for (int i = 0; i < count; i++) {
           freqs[alphabet[i]] = f[i];
           ranks[i] = (f[i] << 8) | alphabet[i];
       }

       return computeCodeLengths(sizes, ranks, count);
   }

   return HuffmanCommon::MAX_SYMBOL_SIZE;
}


// Called only when more than 1 symbol
uint HuffmanEncoder::computeCodeLengths(uint16 sizes[], uint ranks[], int count) const
{
    // Sort ranks by increasing freqs (first key) and increasing value (second key)
    vector<uint> v(ranks, ranks + count);
    sort(v.begin(), v.end());
    uint freqs[256] = { 0 };

    for (int i = 0; i < count; i++) {
        ranks[i] = v[i] & 0xFF;
        freqs[i] = v[i] >> 8;

        if (freqs[i] == 0)
            return 0;
    }

    // See [In-Place Calculation of Minimum-Redundancy Codes]
    // by Alistair Moffat & Jyrki Katajainen
    computeInPlaceSizesPhase1(freqs, count);
    const uint maxCodeLen = computeInPlaceSizesPhase2(freqs, count);

    for (int i = 0; i < count; i++)
        sizes[ranks[i]] = uint16(freqs[i]);

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

// n must be at least 2
// return max symbol length
uint HuffmanEncoder::computeInPlaceSizesPhase2(uint data[], int n)
{
    if (n < 2)
       return 0;

    uint topLevel = n - 2; //root
    uint depth = 1;
    uint totalNodesAtLevel = 2;

    while (n > 0) {
        uint k = topLevel;

        while ((k != 0) && (data[k - 1] >= topLevel))
            k--;

        const int internalNodesAtLevel = topLevel - k;
        const int leavesAtLevel = totalNodesAtLevel - internalNodesAtLevel;

        for (int j = 0; j < leavesAtLevel; j++)
            data[--n] = depth;

        totalNodesAtLevel = internalNodesAtLevel << 1;
        topLevel = k;
        depth++;
    }

    return depth - 1;
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
        const uint endChunk = min(startChunk + sz, end);
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
        int bits = 0; // number of accumulated bits

        // Encode chunk
        for (uint i = startChunk; i < endChunk4; i += 4) {
            const uint16 code0 = _codes[int(block[i])];
            const uint16 codeLen0 = code0 >> 12;
            const uint16 code1 = _codes[int(block[i + 1])];
            const uint16 codeLen1 = code1 >> 12;
            const uint16 code2 = _codes[int(block[i + 2])];
            const uint16 codeLen2 = code2 >> 12;
            const uint16 code3 = _codes[int(block[i + 3])];
            const uint16 codeLen3 = code3 >> 12;
            state = (state << codeLen0) | uint64(code0 & 0x0FFF);
            state = (state << codeLen1) | uint64(code1 & 0x0FFF);
            state = (state << codeLen2) | uint64(code2 & 0x0FFF);
            state = (state << codeLen3) | uint64(code3 & 0x0FFF);
            bits += (codeLen0 + codeLen1 + codeLen2 + codeLen3);
            BigEndian::writeLong64(&_buffer[idx], state << (64 - bits));
            idx += (bits >> 3);
            bits &= 7;
        }

        for (uint i = endChunk4; i < endChunk; i++) {
            const uint16 code = _codes[int(block[i])];
            const uint16 codeLen = code >> 12;
            state = (state << codeLen) | uint64(code & 0x0FFF);
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


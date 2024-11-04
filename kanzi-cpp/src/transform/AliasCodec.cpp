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

#include "AliasCodec.hpp"
#include "../Global.hpp"
#include "../Memory.hpp"

using namespace kanzi;
using namespace std;


AliasCodec::AliasCodec(Context& ctx) : 
          _pCtx(&ctx) 
{
   _order = _pCtx->getInt("alias", 1);
}


bool AliasCodec::forward(SliceArray<byte>& input, SliceArray<byte>& output, int count)
{
    if (count == 0)
        return true;

    if (count < MIN_BLOCK_SIZE)
        return false;

    if (!SliceArray<byte>::isValid(input))
        throw invalid_argument("Alias codec: Invalid input block");

    if (!SliceArray<byte>::isValid(output))
        throw invalid_argument("Alias codec: Invalid output block");

    if (output._length < getMaxEncodedLength(count))
        return false;

    Global::DataType dt = Global::UNDEFINED;

    if (_pCtx != nullptr) {
        dt = (Global::DataType) _pCtx->getInt("dataType", Global::UNDEFINED);

        if ((dt == Global::MULTIMEDIA) || (dt == Global::UTF8))
            return false;

        if ((dt == Global::EXE) || (dt == Global::BIN))
            return false;
    }

    byte* dst = &output._array[output._index];
    byte* src = &input._array[input._index];

    // Find missing 1-byte symbols
    uint freqs0[256] = { 0 };
    Global::computeHistogram(&src[0], count, freqs0, true);
    int n0 = 0;
    int absent[256] = { 0 };

    for (int i = 0; i < 256; i++) {
        if (freqs0[i] == 0)
            absent[n0++] = i;
    }

    if (n0 < 16) {
        if (_pCtx != nullptr) {
            if (dt == Global::UNDEFINED) {
                dt = Global::detectSimpleType(count, freqs0);

                if (dt != Global::UNDEFINED)
                    _pCtx->putInt("dataType", dt);
            }
        }

        return false;
    }

    int srcIdx, dstIdx;
 
    if (n0 >= 240) {
        // Small alphabet => pack bits
        dst[0] = byte(n0);
        dstIdx = 1;

        if (n0 == 255) {
            // One symbol
            dst[1] = src[0];
            LittleEndian::writeInt32(&dst[2], count);
            dstIdx = 6;
            srcIdx = count;
        }
        else {
            srcIdx = 0;
            byte map8[256] = { byte(0) };

            for (int i = 0, j = 0; i < 256; i++) {
                if (freqs0[i] != 0) {
                    dst[dstIdx++] = byte(i);
                    map8[i] = byte(j++);
                }
            }

            if (n0 >= 252) {
                // 4 symbols or less
                const int c3 = count & 3;
                dst[dstIdx++] = byte(c3);
                memcpy(&dst[dstIdx], &src[srcIdx], c3);
                srcIdx += c3;
                dstIdx += c3;

                while (srcIdx < count) {
                    dst[dstIdx++] = (map8[int(src[srcIdx + 0])] << 6) | (map8[int(src[srcIdx + 1])] << 4) |
                        (map8[int(src[srcIdx + 2])] << 2) | map8[int(src[srcIdx + 3])];
                    srcIdx += 4;
                }
            }
            else {
                // 16 symbols or less
                dst[dstIdx++] = byte(count & 1);

                if ((count & 1) != 0)
                    dst[dstIdx++] = src[srcIdx++];

                while (srcIdx < count) {
                    dst[dstIdx++] = (map8[int(src[srcIdx])] << 4) | map8[int(src[srcIdx + 1])];
                    srcIdx += 2;
                }
            }
        }
    }
    else {
        if (_order == 0)
            return false;

        // Digram encoding
        vector<sdAlias> v;
        
        {
            // Find missing 2-byte symbols
            uint* freqs1 = new uint[65536];
            memset(freqs1, 0, 65536 * sizeof(uint));
            Global::computeHistogram(&src[0], count, freqs1, false);
            int n1 = 0;

            for (uint32 i = 0; i < 65536; i++) {
                if (freqs1[i] == 0)
                    continue;

#if __cplusplus >= 201103L
                v.emplace_back(i, freqs1[i]);
#else
                sdAlias a(i, freqs1[i]);
                v.push_back(a);
#endif

                n1++;
            }

            delete[] freqs1;

            if (n1 < n0) {
                // Fewer distinct 2-byte symbols than 1-byte symbols
                n0 = n1;

                if (n0 < 16) 
                    return false;
            }   

            // Sort by decreasing order 1 frequencies
            sort(v.begin(), v.end());  
        }

        int16 map16[65536];
 
        // Build map symbol -> alias
        for (int i = 0; i < 65536; i++)
            map16[i] = 0x100 | int16(i >> 8);

        int savings = 0;
        dst[0] = byte(n0);
        dst[1] = byte(0);
        srcIdx = 0;
        dstIdx = 2;

        // Header: emit map data
        for (int i = 0; i < n0; i++) {
            const sdAlias& sd = v[i];
            savings += sd.freq; // ignore factor 2
            const int idx = sd.val;
            map16[idx] = int16(absent[i]) | 0x200;
            dst[dstIdx] = byte(idx >> 8);
            dst[dstIdx + 1] = byte(idx);
            dst[dstIdx + 2] = byte(absent[i]);
            dstIdx += 3;
         }

        // Worth it?
        if (savings * 20 < count)
            return false;

        v.clear();
        const int srcEnd = count - 1;

        // Emit aliased data
        while (srcIdx < srcEnd) {
            const int16 alias = map16[(int(src[srcIdx]) << 8) | int(src[srcIdx + 1])];
            dst[dstIdx++] = byte(alias);
            srcIdx += (alias >> 8);
        }

        if (srcIdx != count) {
            dst[1] = byte(1);
            dst[dstIdx++] = src[srcIdx++];
        }
    }

    input._index += srcIdx;
    output._index += dstIdx;
    return dstIdx < count;
}


bool AliasCodec::inverse(SliceArray<byte>& input, SliceArray<byte>& output, int count)
{
    if (count == 0)
        return true;

    if (!SliceArray<byte>::isValid(input))
        throw invalid_argument("Alias codec: Invalid input block");

    if (!SliceArray<byte>::isValid(output))
        throw invalid_argument("Alias codec: Invalid output block");

    byte* dst = &output._array[output._index];
    byte* src = &input._array[input._index];
    int n = int(src[0]);

    if (n < 16)
        return false;

    int srcIdx;
    int dstIdx = 0;

    if (n >= 240) {
        n = 256 - n; 
        srcIdx = 1;

        if (n == 1) {
            // One symbol
            const byte val = src[1];
            const int oSize = LittleEndian::readInt32(&src[2]);

            if (output._index + oSize > output._length)
                return false;

            memset(&dst[0], int(val), oSize);
            srcIdx = count;
            dstIdx = oSize;
        }
        else {
            // Rebuild map alias -> symbol
            byte idx2symb[16] = { byte(0) };

            for (int i = 0; i < n; i++)
                idx2symb[i] = src[srcIdx++];

            const int adjust = int(src[srcIdx++]);

            if ((adjust < 0) || (adjust >= 4))
                return false;

            if (n <= 4) {
                // 4 symbols or less 
                int32 decodeMap[256] = { 0 };

                for (int i = 0; i < 256; i++) {
                    int32 val;
                    val  = int32(idx2symb[(i >> 0) & 0x03]);
                    val <<= 8;
                    val |= int32(idx2symb[(i >> 2) & 0x03]);
                    val <<= 8;
                    val |= int32(idx2symb[(i >> 4) & 0x03]);
                    val <<= 8;
                    val |= int32(idx2symb[(i >> 6) & 0x03]);
                    decodeMap[i] = val;
                }

                memcpy(&dst[dstIdx], &src[srcIdx], adjust);
                srcIdx += adjust;
                dstIdx += adjust;

                while (srcIdx < count) {
                    LittleEndian::writeInt32(&dst[dstIdx], decodeMap[int(src[srcIdx++])]);
                    dstIdx += 4;
                }
            }
            else
            {
                // 16 symbols or less
                int16 decodeMap[256] = { 0 };

                for (int i = 0; i < 256; i++) {
                    int16 val = int16(idx2symb[i& 0x0F]);
                    val <<= 8;
                    val |= int16(idx2symb[i >> 4]);
                    decodeMap[i] = val;
                }

                if (adjust != 0)
                    dst[dstIdx++] = src[srcIdx++];

                while (srcIdx < count) {
                    LittleEndian::writeInt16(&dst[dstIdx], decodeMap[int(src[srcIdx++])]);
                    dstIdx += 2;
                }
            }
        }
    }
    else {
        // Rebuild map alias -> symbol
        int map16[256] = { 0 };
        const int srcEnd = count - int(src[1]);
        srcIdx = 2;

        for (int i = 0; i < 256; i++)
            map16[i] = 0x10000 | i;

        for (int i = 0; i < n; i++) {
            map16[int(src[srcIdx + 2])] = 0x20000 | int(src[srcIdx]) | (int(src[srcIdx + 1]) << 8);
            srcIdx += 3;
        }

        while (srcIdx < srcEnd) {
            const int val = map16[int(src[srcIdx++])];
            dst[dstIdx] = byte(val);
            dst[dstIdx + 1] = byte(val >> 8);
            dstIdx += (val >> 16);
        }

        if (src[1] != byte(0))
            dst[dstIdx++] = src[srcIdx++];
    }

    input._index += srcIdx;
    output._index += dstIdx;
    return true;
}

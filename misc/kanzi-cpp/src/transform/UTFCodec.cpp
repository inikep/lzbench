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
#include <cstring>
#include <vector>
#include "UTFCodec.hpp"
#include "../Global.hpp"
#include "../types.hpp"

using namespace kanzi;
using namespace std;

const int UTFCodec::SIZES[16] = { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4 };


bool UTFCodec::forward(SliceArray<byte>& input, SliceArray<byte>& output, int count)
{
    if (count == 0)
        return true;

    if (count < MIN_BLOCK_SIZE)
        return false;

    if (!SliceArray<byte>::isValid(input))
        throw invalid_argument("UTFCodec: Invalid input block");

    if (!SliceArray<byte>::isValid(output))
        throw invalid_argument("UTFCodec: Invalid output block");

    if (output._length - output._index < getMaxEncodedLength(count))
        return false;

    const byte* src = &input._array[input._index];
    byte* dst = &output._array[output._index];
    bool mustValidate = true;

    if (_pCtx != nullptr) {
        Global::DataType dt = (Global::DataType)_pCtx->getInt("dataType", Global::UNDEFINED);

        if ((dt != Global::UNDEFINED) && (dt != Global::UTF8))
            return false;

        mustValidate = dt != Global::UTF8;
    }

    int start = 0;

    // First (possibly) invalid symbols (due to block truncation)
    while ((start < 4) && (SIZES[uint8(src[start]) >> 4] == 0))
        start++;

    if ((mustValidate == true) && (validate(&src[start], count - start - 4)) == false)
        return false;

    // 1-3 bit size + (7 or 11 or 16 or 21) bit payload
    // 3 MSBs indicate symbol size (limit map size to 22 bits)
    // 000 -> 7 bits
    // 001 -> 11 bits
    // 010 -> 16 bits
    // 1xx -> 21 bits
    uint32* aliasMap = new uint32[1 << 22];
    memset(aliasMap, 0, size_t(1 << 22) * sizeof(uint32));
    vector<sdUTF> v;
    int n = 0;
    bool res = true;

    for (int i = start; i < count - 4; ) {
        uint32 val;
        const int s = pack(&src[i], val);

        if (s == 0) {
            res = false;
            break;
        }

        if (aliasMap[val] == 0) {
#if __cplusplus >= 201103L
            v.emplace_back(val, 0);
#else
            sdUTF u(val, 0);
            v.push_back(u);
#endif

            if (++n >= 32768) {
                res = false;
                break;
            }
        }

        aliasMap[val]++;
        i += s;
    }

    const int dstEnd = count - (count / 10);

    if ((res == false) || (n == 0) || ((3 * n + 6) >= dstEnd)) {
        delete[] aliasMap;
        return false;
    }

    for (int i = 0; i < n; i++)
        v[i].freq = aliasMap[v[i].val];

    // Sort ranks by decreasing frequencies;
    sort(v.begin(), v.end());

    int dstIdx = 2;

    // Emit map length then map data
    dst[dstIdx++] = byte(n >> 8);
    dst[dstIdx++] = byte(n);

    int estimate = dstIdx + 6;

    for (int i = 0; i < n; i++) {
        estimate += int((i < 128) ? v[i].freq : 2 * v[i].freq);
        const uint32 s = v[i].val;
        aliasMap[s] = (i < 128) ? i : 0x10080 | ((i << 1) & 0xFF00) | (i & 0x7F);
        dst[dstIdx] = byte(s >> 16);
        dst[dstIdx + 1] = byte(s >> 8);
        dst[dstIdx + 2] = byte(s);
        dstIdx += 3;
    }

    if (estimate >= dstEnd) {
        // Not worth it
        delete[] aliasMap;
        return false;
    }

    // Emit first (possibly invalid) symbols (due to block truncation)
    for (int i = 0; i < start; i++)
        dst[dstIdx++] = src[i];

    v.clear();
    int srcIdx = start;

    // Emit aliases
    while (srcIdx < count - 4) {
        uint32 val;
        srcIdx += pack(&src[srcIdx], val);
        const uint32 alias = aliasMap[val];
        dst[dstIdx++] = byte(alias);
        dst[dstIdx] = byte(alias >> 8);
        dstIdx += (alias >> 16);
    }

    dst[0] = byte(start);
    dst[1] = byte(srcIdx - (count - 4));

    // Emit last (possibly invalid) symbols (due to block truncation)
    while (srcIdx < count)
        dst[dstIdx++] = src[srcIdx++];

    delete[] aliasMap;
    input._index += srcIdx;
    output._index += dstIdx;
    return dstIdx < dstEnd;
}

bool UTFCodec::inverse(SliceArray<byte>& input, SliceArray<byte>& output, int count)
{
    if (count == 0)
        return true;

    if (count < 4)
        return false;

    if (!SliceArray<byte>::isValid(input))
        throw invalid_argument("UTFCodec: Invalid input block");

    if (!SliceArray<byte>::isValid(output))
        throw invalid_argument("UTFCodec: Invalid output block");

    const byte* src = &input._array[input._index];
    byte* dst = &output._array[output._index];
    const int start = int(src[0]);
    const int adjust = int(src[1]); // adjust end of regular processing
    const int n = (int(src[2]) << 8) + int(src[3]);

    // Protect against invalid map size value
    if ((n >= 32768) || (3 * n >= count))
        return false;

#pragma pack(1)
    struct symb {
        uint32 val;
        uint8 len;
    };

    symb m[32768];
    int srcIdx = 4;

    // Build inverse mapping
    for (int i = 0; i < n; i++) {
        int s = (uint32(src[srcIdx]) << 16) | (uint32(src[srcIdx + 1]) << 8) | uint32(src[srcIdx + 2]);
        const int sl = unpack(s, (byte*)&m[i].val);

        if (sl == 0) {
            return false;
        }

        m[i].len = uint8(sl);
        srcIdx += 3;
    }

    int dstIdx = 0;
    const int srcEnd = count - 4 + adjust;

    for (int i = 0; i < start; i++)
        dst[dstIdx++] = src[srcIdx++];

    // Emit data
    while (srcIdx < srcEnd) {
        uint alias = uint(src[srcIdx++]);

        if (alias >= 128)
            alias = (uint(src[srcIdx++]) << 7) + (alias & 0x7F);

        const symb& s = m[alias];
        memcpy(&dst[dstIdx], &s.val, 4);
        dstIdx += s.len;
    }

    for (int i = srcEnd; i < count; i++)
        dst[dstIdx++] = src[srcIdx++];

    input._index += srcIdx;
    output._index += dstIdx;
    return srcIdx == count;
}


bool UTFCodec::validate(const byte block[], int count)
{
    uint freqs0[256] = { 0 };
    uint* freqs1 = new uint[65536];
    memset(&freqs1[0], 0, 65536 * sizeof(uint));
    uint f0[256] = { 0 };
    uint f1[256] = { 0 };
    uint f3[256] = { 0 };
    uint f2[256] = { 0 };
    uint8 prv = 0;
    const uint8* data = reinterpret_cast<const uint8*>(&block[0]);
    const int count4 = count & -4;
    bool res = true;

    // Unroll loop
    for (int i = 0; i < count4; i += 4) {
        const uint8 cur0 = data[i];
        const uint8 cur1 = data[i + 1];
        const uint8 cur2 = data[i + 2];
        const uint8 cur3 = data[i + 3];
        f0[cur0]++;
        f1[cur1]++;
        f2[cur2]++;
        f3[cur3]++;
        freqs1[(prv * 256) + cur0]++;
        freqs1[(cur0 * 256) + cur1]++;
        freqs1[(cur1 * 256) + cur2]++;
        freqs1[(cur2 * 256) + cur3]++;
        prv = cur3;
    }

    for (int i = count4; i < count; i++) {
        freqs0[data[i]]++;
        freqs1[(prv * 256) + data[i]]++;
        prv = data[i];
    }

    for (int i = 0; i < 256; i++) {
        freqs0[i] += (f0[i] + f1[i] + f2[i] + f3[i]);
    }

    int sum = 0;

    // Check UTF-8
    // See Unicode 14 Standard - UTF-8 Table 3.7
    // U+0000..U+007F          00..7F
    // U+0080..U+07FF          C2..DF 80..BF
    // U+0800..U+0FFF          E0 A0..BF 80..BF
    // U+1000..U+CFFF          E1..EC 80..BF 80..BF
    // U+D000..U+D7FF          ED 80..9F 80..BF 80..BF
    // U+E000..U+FFFF          EE..EF 80..BF 80..BF
    // U+10000..U+3FFFF        F0 90..BF 80..BF 80..BF
    // U+40000..U+FFFFF        F1..F3 80..BF 80..BF 80..BF
    // U+100000..U+10FFFF      F4 80..8F 80..BF 80..BF
    if ((freqs0[0xC0] > 0) || (freqs0[0xC1] > 0)) {
        res = false;
        goto end;
    }

    for (int i = 0xF5; i <= 0xFF; i++) {
        if (freqs0[i] > 0) {
            res = false;
            goto end;
        }
    }

    for (int i = 0; i < 256; i++) {
        // Exclude < 0xE0A0 || > 0xE0BF
        if (((i < 0xA0) || (i > 0xBF)) && (freqs1[(0xE0 * 256) + i] > 0)) {
            res = false;
            goto end;
        }

        // Exclude < 0xED80 || > 0xEDE9F
        if (((i < 0x80) || (i > 0x9F)) && (freqs1[(0xED * 256) + i] > 0)) {
            res = false;
            goto end;
        }

        // Exclude < 0xF090 || > 0xF0BF
        if (((i < 0x90) || (i > 0xBF)) && (freqs1[(0xF0 * 256) + i] > 0)) {
            res = false;
            goto end;
        }

        // Exclude < 0xF480 || > 0xF4BF
        if (((i < 0x80) || (i > 0xBF)) && (freqs1[(0xF4 * 256) + i] > 0)) {
            res = false;
            goto end;
        }

        // Count non-primary bytes
        if ((i >= 0x80) && (i <= 0xBF))
            sum += freqs0[i];
    }

end:
    delete[] freqs1;

    // Ad-hoc threshold
    return (res == true) && (sum >= (count / 4));
}

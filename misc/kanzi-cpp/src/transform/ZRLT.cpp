/*
Copyright 2011-2025 Frederic Langlet
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

#include <cstring>
#include <stddef.h>
#include "../Global.hpp"
#include "ZRLT.hpp"

using namespace kanzi;
using namespace std;

bool ZRLT::forward(SliceArray<byte>& input, SliceArray<byte>& output, int length)
{
    if (length == 0)
        return true;

    if (!SliceArray<byte>::isValid(input))
        throw invalid_argument("ZRLT: Invalid input block");

    if (!SliceArray<byte>::isValid(output))
        throw invalid_argument("ZRLT: Invalid output block");

    if (output._length - output._index < getMaxEncodedLength(length))
        return false;

    const byte* src = &input._array[input._index];
    byte* dst = &output._array[output._index];
    uint srcIdx = 0;
    uint dstIdx = 0;
    const uint srcEnd = length;
    const uint dstEnd = length; // do not expand
    bool res = true;
    byte zeros[4] = { byte(0) };

    while (srcIdx < srcEnd) {
        if (src[srcIdx] == byte(0)) {
            uint runLength = 1;

            while ((srcIdx + runLength + 4 < srcEnd) && (memcmp(&src[srcIdx + runLength], &zeros[0], 4) == 0))
                runLength += 4;

            while ((srcIdx + runLength < srcEnd) && src[srcIdx + runLength] == byte(0))
                runLength++;

            srcIdx += runLength;

            // Encode length
            runLength++;
            int log = Global::_log2(uint32(runLength));

            if (dstIdx >= dstEnd - log) {
                res = false;
                break;
            }

            // Write every bit as a byte except the most significant one
            while (log > 0) {
                log--;
                dst[dstIdx++] = byte((runLength >> log) & 1);
            }

            continue;
        }

        const int val = int(src[srcIdx]);

        if (val >= 0xFE) {
           if (dstIdx >= dstEnd - 1) {
                res = false;
                break;
            }

            dst[dstIdx] = byte(0xFF);
            dstIdx++;
            dst[dstIdx] = byte(val - 0xFE);
        }
        else {
           if (dstIdx >= dstEnd) {
                res = false;
                break;
            }

            dst[dstIdx] = byte(val + 1);
        }

        srcIdx++;
        dstIdx++;
    }

    input._index += srcIdx;
    output._index += dstIdx;
    return res && (srcIdx == srcEnd);
}

bool ZRLT::inverse(SliceArray<byte>& input, SliceArray<byte>& output, int length)
{
    if (length == 0)
        return true;

    if (!SliceArray<byte>::isValid(input))
        throw invalid_argument("ZRLT: Invalid input block");

    if (!SliceArray<byte>::isValid(output))
        throw invalid_argument("ZRLT: Invalid output block");

    const byte* src = &input._array[input._index];
    byte* dst = &output._array[output._index];
    uint srcIdx = 0;
    uint dstIdx = 0;
    const uint srcEnd = length;
    const uint dstEnd = output._length;
    uint runLength = 0;

    while (true) {
        uint val = uint(src[srcIdx]);

        if (val <= 1) {
            // Generate the run length bit by bit (but force MSB)
            runLength = 1;

            do {
                runLength += (runLength + val);
                srcIdx++;

                if (srcIdx >= srcEnd)
                    goto End;

                val = uint(src[srcIdx]);
            }
            while (val <= 1);

            runLength--;

            if (runLength > 0) {
                if (runLength >= dstEnd - dstIdx)
                    goto End;

                memset(&dst[dstIdx], 0, size_t(runLength));
                dstIdx += runLength;
                runLength = 0;
                continue;
            }
        }

        // Regular data processing
        if (val == 0xFF) {
            srcIdx++;

            if (srcIdx >= srcEnd)
                goto End;

            dst[dstIdx] = byte(0xFE + int(src[srcIdx]));
        }
        else {
            dst[dstIdx] = byte(val - 1);
        }

        srcIdx++;
        dstIdx++;

        if ((srcIdx >= srcEnd) || (dstIdx >= dstEnd))
            break;
    }

End:
    if (runLength > 0) {
        runLength--;

        // If runLength is not 1, add trailing 0s
        if (runLength > dstEnd - dstIdx)
            return false;

        if (runLength > 0) {
            memset(&dst[dstIdx], 0, size_t(runLength));
            dstIdx += runLength;
        }
    }

    input._index += srcIdx;
    output._index += dstIdx;
    return srcIdx == srcEnd;
}

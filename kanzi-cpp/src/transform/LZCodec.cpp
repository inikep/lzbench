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

#include "LZCodec.hpp"
#include "../Memory.hpp"
#include "../util.hpp" // Visual Studio min/max
#include "TransformFactory.hpp"

using namespace kanzi;
using namespace std;

LZCodec::LZCodec() THROW
{
    _delegate = new LZXCodec<false>();
}

LZCodec::LZCodec(Context& ctx) THROW
{
    const int lzType = ctx.getInt("lz", TransformFactory<byte>::LZ_TYPE);

    if (lzType == TransformFactory<byte>::LZP_TYPE) {
        _delegate = (Transform<byte>*)new LZPCodec(ctx);
    }
    else if (lzType == TransformFactory<byte>::LZX_TYPE) {
        _delegate = (Transform<byte>*)new LZXCodec<true>(ctx);
    }
    else {
        _delegate = (Transform<byte>*)new LZXCodec<false>(ctx);
    }
}

bool LZCodec::forward(SliceArray<byte>& input, SliceArray<byte>& output, int count) THROW
{
    if (count == 0)
        return true;

    if (input._array == output._array)
        return false;

    return _delegate->forward(input, output, count);
}

bool LZCodec::inverse(SliceArray<byte>& input, SliceArray<byte>& output, int count) THROW
{
    if (count == 0)
        return true;

    if (input._array == output._array)
        return false;

    return _delegate->inverse(input, output, count);
}

template <bool T>
bool LZXCodec<T>::forward(SliceArray<byte>& input, SliceArray<byte>& output, int count)
{
    if (count == 0)
        return true;

    if (!SliceArray<byte>::isValid(input))
        throw invalid_argument("LZ codec: Invalid input block");

    if (!SliceArray<byte>::isValid(output))
        throw invalid_argument("LZ codec: Invalid output block");

    if (output._length < getMaxEncodedLength(count))
        return false;

    // If too small, skip
    if (count < MIN_BLOCK_LENGTH)
        return false;

    if (_hashSize == 0) {
        _hashSize = (T == true) ? 1 << HASH_LOG2 : 1 << HASH_LOG1;
        delete[] _hashes;
        _hashes = new int32[_hashSize];
    }

    if (_bufferSize < max(count / 5, 256)) {
        _bufferSize = max(count / 5, 256);
        delete[] _mLenBuf;
        _mLenBuf = new byte[_bufferSize];
        delete[] _mBuf;
        _mBuf = new byte[_bufferSize];
        delete[] _tkBuf;
        _tkBuf = new byte[_bufferSize];
    }

    memset(_hashes, 0, sizeof(int32) * _hashSize);
    const int srcEnd = count - 16 - 1;
    byte* dst = &output._array[output._index];
    byte* src = &input._array[input._index];
    const int maxDist = (srcEnd < 4 * MAX_DISTANCE1) ? MAX_DISTANCE1 : MAX_DISTANCE2;
    dst[12] = (maxDist == MAX_DISTANCE1) ? byte(0) : byte(1);
    int mm = MIN_MATCH4;

    if (_pCtx != nullptr) {
        Global::DataType dt = (Global::DataType)_pCtx->getInt("dataType", Global::UNDEFINED);

        if (dt == Global::DNA) {
            // Longer min match for DNA input
            mm = MIN_MATCH9;
            dst[12] |= byte(2);
        }
        else if (dt == Global::SMALL_ALPHABET) {
            return false;
        }
    }

    const int minMatch = mm;
    const int dThreshold = (maxDist == MAX_DISTANCE1) ? 1 << 8 : 1 << 16;
    int srcIdx = 0;
    int dstIdx = 13;
    int anchor = 0;
    int mIdx = 0;
    int mLenIdx = 0;
    int tkIdx = 0;
    int repd[] = { count, count };
    int repIdx = 0;
    int srcInc = 0;

    while (srcIdx < srcEnd) {
        const int minRef = max(srcIdx - maxDist, 0);
        int bestLen = 0;
        int ref = srcIdx + 1 - repd[repIdx];

        if ((ref > minRef) && (memcmp(&src[srcIdx + 1], &src[ref], 4) == 0)) {
            // Check repd first
            bestLen = findMatch(src, srcIdx + 1, ref, min(srcEnd - srcIdx - 1, MAX_MATCH));

            if (bestLen < minMatch) {
                ref = srcIdx + 1 - repd[1 - repIdx];

                if ((ref > minRef) && (memcmp(&src[srcIdx + 1], &src[ref], 4) == 0)) {
                    bestLen = findMatch(src, srcIdx + 1, ref, min(srcEnd - srcIdx - 1, MAX_MATCH));
                }
            }
        }

        if (bestLen < minMatch) {
            // Check match at position in hash table
            const int32 h0 = hash(&src[srcIdx]);
            prefetchWrite(&_hashes[h0]);
            ref = _hashes[h0];
            _hashes[h0] = srcIdx;

            if ((ref > minRef) && (memcmp(&src[srcIdx], &src[ref], 4) == 0)) {
                bestLen = findMatch(src, srcIdx, ref, min(srcEnd - srcIdx, MAX_MATCH));
            }

            // No good match ?
            if (bestLen < minMatch) {
                srcIdx++;
                srcIdx += (srcInc >> 6);
                srcInc++;
                repIdx = 0;
                continue;
            }

            if ((ref != srcIdx - repd[0]) && (ref != srcIdx - repd[1])) {
                // Check if better match at next position
                const int32 h1 = hash(&src[srcIdx + 1]);
                const int ref1 = _hashes[h1];
                _hashes[h1] = srcIdx + 1;

                if ((ref1 > minRef + 1) && (memcmp(&src[srcIdx + 1], &src[ref1], 4) == 0)) {
                    const int bestLen1 = findMatch(src, srcIdx + 1, ref1, min(srcEnd - srcIdx - 1, MAX_MATCH));

                    // Select best match
                    if ((bestLen1 > bestLen) || ((bestLen1 == bestLen) && (ref1 > ref))) {
                        if ((src[srcIdx] == src[ref1 - 1]) && (bestLen1 < MAX_MATCH)) {
                            ref = ref1 - 1;
                            bestLen = bestLen1 + 1;
                        }
                        else {
                            ref = ref1;
                            bestLen = bestLen1;
                            srcIdx++;
                        }
                    }
                }
            }
        }
        else {
            const int32 h0 = hash(&src[srcIdx]);
            _hashes[h0] = srcIdx;

            if ((src[srcIdx] == src[ref - 1]) && (bestLen < MAX_MATCH)) {
                bestLen++;
                ref--;
            }
            else {
                srcIdx++;
                const int32 h1 = hash(&src[srcIdx]);
                _hashes[h1] = srcIdx;
            }
        }

        // Emit match
        srcInc = 0;

        // Token: 3 bits litLen + 1 bit flag + 4 bits mLen (LLLFMMMM)
        // LLL  : <= 7  --> LLL == literal length (if 7, remainder encoded outside of token)
        // MMMM : <= 14 --> MMMM == match length (if 14, remainder encoded outside of token)
        //        == 15 if dist == repd0 or repd1 && matchLen fully encoded outside of token
        // F    : if MMMM == 15, flag = 0 if dist == repd0 and 1 if dist == repd1
        //        else flag = 1 if dist >= dThreshold and 0 otherwise
        const int dist = srcIdx - ref;
        const int mLen = bestLen - minMatch;
        const int litLen = srcIdx - anchor;
        int token;

        if (dist == repd[0]) {
            token = 0x0F;
            mLenIdx += emitLength(&_mLenBuf[mLenIdx], mLen);
        }
        else if (dist == repd[1]) {
            token = 0x1F;
            mLenIdx += emitLength(&_mLenBuf[mLenIdx], mLen);
        }
        else {
            // Emit distance (since not repeat)
            if (maxDist == MAX_DISTANCE2) {
                if (dist >= 65536)
                    _mBuf[mIdx++] = byte(dist >> 16);

                _mBuf[mIdx++] = byte(dist >> 8);
            }
            else {
                if (dist >= 256)
                    _mBuf[mIdx++] = byte(dist >> 8);
            }

            _mBuf[mIdx++] = byte(dist);
            
            // Emit match length
            if (mLen >= 14) {
                if (mLen == 14) {
                    // Avoid the penalty of one extra byte to encode match length
                    token = (dist >= dThreshold) ? 0x1D : 0x0D;
                    bestLen--;
                }
                else {
                    token = (dist >= dThreshold) ? 0x1E : 0x0E;
                    mLenIdx += emitLength(&_mLenBuf[mLenIdx], mLen - 14);
                }
            }
            else {
                token = (dist >= dThreshold) ? 0x10 | mLen : mLen;
            }
        }

        repd[1] = repd[0];
        repd[0] = dist;
        repIdx = 1;

        // Emit token
        // Literals to process ?
        if (litLen == 0) {
            _tkBuf[tkIdx++] = byte(token);
        }
        else {
            // Emit literal length
            if (litLen >= 7) {
                if (litLen >= (1 << 24))
                    return false;

                _tkBuf[tkIdx++] = byte((7 << 5) | token);
                dstIdx += emitLength(&dst[dstIdx], litLen - 7);
            }
            else {
                _tkBuf[tkIdx++] = byte((litLen << 5) | token);
            }

            // Emit literals
            emitLiterals(&src[anchor], &dst[dstIdx], litLen);
            dstIdx += litLen;
        }

        if (mIdx >= _bufferSize - 8) {
            // Expand match buffer
            byte* mBuf = new byte[(_bufferSize * 3) / 2];
            memcpy(&mBuf[0], &_mBuf[0], _bufferSize);
            delete[] _mBuf;
            _mBuf = mBuf;

            if (mLenIdx >= _bufferSize - 4) {
                byte* mLenBuf = new byte[(_bufferSize * 3) / 2];
                memcpy(&mLenBuf[0], &_mLenBuf[0], _bufferSize);
                delete[] _mLenBuf;
                _mLenBuf = mLenBuf;
            }

            _bufferSize = (_bufferSize * 3) / 2;
        }

        // Fill _hashes and update positions
        anchor = srcIdx + bestLen;

        while (++srcIdx < anchor) {
            const int32 h = hash(&src[srcIdx]);
            _hashes[h] = srcIdx;
        }

        prefetchRead(&src[srcIdx + 64]);
    }

    // Emit last literals
    const int litLen = count - anchor;

    if (dstIdx + litLen + tkIdx + mIdx >= output._index + count)
        return false;

    if (litLen >= 7) {
        _tkBuf[tkIdx++] = byte(7 << 5);
        dstIdx += emitLength(&dst[dstIdx], litLen - 7);
    }
    else {
        _tkBuf[tkIdx++] = byte(litLen << 5);
    }

    memcpy(&dst[dstIdx], &src[anchor], litLen);
    dstIdx += litLen;

    // Emit buffers: literals + tokens + matches
    LittleEndian::writeInt32(&dst[0], dstIdx);
    LittleEndian::writeInt32(&dst[4], tkIdx);
    LittleEndian::writeInt32(&dst[8], mIdx);
    memcpy(&dst[dstIdx], &_tkBuf[0], tkIdx);
    dstIdx += tkIdx;
    memcpy(&dst[dstIdx], &_mBuf[0], mIdx);
    dstIdx += mIdx;
    memcpy(&dst[dstIdx], &_mLenBuf[0], mLenIdx);
    dstIdx += mLenIdx;
    input._index += count;
    output._index += dstIdx;
    return true;
}

template <bool T>
bool LZXCodec<T>::inverse(SliceArray<byte>& input, SliceArray<byte>& output, int count)
{
    if (count == 0)
        return true;

    if (count < 13)
        return false;

    if (!SliceArray<byte>::isValid(input))
        throw invalid_argument("LZ codec: Invalid input block");

    if (!SliceArray<byte>::isValid(output))
        throw invalid_argument("LZ codec: Invalid output block");

    const int dstEnd = output._length;
    byte* dst = &output._array[output._index];
    byte* src = &input._array[input._index];

    int tkIdx = LittleEndian::readInt32(&src[0]);
    int mIdx = LittleEndian::readInt32(&src[4]);
    int mLenIdx = LittleEndian::readInt32(&src[8]);

    if ((tkIdx < 0) || (mIdx < 0) || (mLenIdx < 0))
        return false;

    mIdx += tkIdx;
    mLenIdx += mIdx;

    if ((tkIdx > count) || (mIdx > count) || (mLenIdx > count))
        return false;

    const int srcEnd = tkIdx - 13;
    const int mFlag = int(src[12]) & 1;
    const int maxDist = (mFlag == 0) ? MAX_DISTANCE1 : MAX_DISTANCE2;
    const int minMatch = ((int(src[12]) & 2) == 0) ? MIN_MATCH4 : MIN_MATCH9;
    bool res = true;
    int srcIdx = 13;
    int dstIdx = 0;
    int repd0 = 0;
    int repd1 = 0;

    while (true) {
        const int token = int(src[tkIdx++]);

        if (token >= 32) {
            // Get literal length
            const int litLen = (token >= 0xE0) ? 7 + readLength(src, srcIdx) : token >> 5;

            // Emit literals
            const byte* s = &src[srcIdx];
            byte* d = &dst[dstIdx];
            srcIdx += litLen;
            dstIdx += litLen;

            if (srcIdx >= srcEnd) {
                memcpy(d, s, litLen);
                break;
            }

            emitLiterals(s, d, litLen);
        }

        // Get match length and distance
        int mLen = token & 0x0F;
        int dist;

        if (mLen == 15) {
            // Repetition distance, read mLen fully outside of token
            mLen = minMatch + readLength(src, mLenIdx);
            dist = ((token & 0x10) == 0) ? repd0 : repd1;
        }
        else {
            // Read mLen remainder (if any) outside of token
            mLen = (mLen == 14) ? 14 + minMatch + readLength(src, mLenIdx) : mLen + minMatch;
            dist = int(src[mIdx++]);

            if (mFlag != 0)
                dist = (dist << 8) | int(src[mIdx++]);

            //if ((token & 0x10) != 0) {
            //    dist = (dist << 8) | int(src[mIdx++]);
            //}
            const int t = (token >> 4) & 1;
            dist = (dist << (8 * t)) | (-t & int(src[mIdx]));
            mIdx += t;
        }

        repd1 = repd0;
        repd0 = dist;
        const int mEnd = dstIdx + mLen;
        int ref = dstIdx - dist;

        // Sanity check
        if ((ref < 0) || (dist > maxDist) || (mEnd > dstEnd)) {
            res = false;
            goto exit;
        }

        // Copy match
        if (dist >= 16) {
            do {
                // No overlap
                memcpy(&dst[dstIdx], &dst[ref], 16);
                ref += 16;
                dstIdx += 16;
            } while (dstIdx < mEnd);
        }
        else if (dist >= 4) {
            do {
                // No overlap
                memcpy(&dst[dstIdx], &dst[ref], 4);
                ref += 4;
                dstIdx += 4;
            } while (dstIdx < mEnd);
        }
        else {
            for (int i = 0; i < mLen; i++)
                dst[dstIdx + i] = dst[ref + i];
        }

        dstIdx = mEnd;
    }

exit:
    output._index += dstIdx;
    input._index += mIdx;
    return res && (srcIdx == srcEnd + 13);
}

bool LZPCodec::forward(SliceArray<byte>& input, SliceArray<byte>& output, int count)
{
    if (count == 0)
        return true;

    if (count < 4)
        return false;

    if (!SliceArray<byte>::isValid(input))
        throw invalid_argument("LZP codec: Invalid input block");

    if (!SliceArray<byte>::isValid(output))
        throw invalid_argument("LZP codec: Invalid output block");

    if (output._length < getMaxEncodedLength(count))
        return false;

    // If too small, skip
    if (count < MIN_BLOCK_LENGTH)
        return false;

    byte* dst = &output._array[output._index];
    byte* src = &input._array[input._index];
    const int srcEnd = count;
    const int dstEnd = output._length - 4;

    if (_hashSize == 0) {
        _hashSize = 1 << HASH_LOG;
        delete[] _hashes;
        _hashes = new int32[_hashSize];
    }

    memset(_hashes, 0, sizeof(int32) * _hashSize);
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
    int32 ctx = LittleEndian::readInt32(&src[0]);
    int srcIdx = 4;
    int dstIdx = 4;

    while ((srcIdx < srcEnd - MIN_MATCH) && (dstIdx < dstEnd)) {
        prefetchRead(&src[srcIdx + MIN_MATCH]);
        const uint32 h = (HASH_SEED * ctx) >> HASH_SHIFT;
        const int32 ref = _hashes[h];
        _hashes[h] = srcIdx;
        int bestLen = 0;

        // Find a match
        if ((ref != 0) && (memcmp(&src[ref + MIN_MATCH - 8], &src[srcIdx + MIN_MATCH - 8], 8) == 0))
            bestLen = findMatch(src, srcIdx, ref, srcEnd - srcIdx);

        // No good match ?
        if (bestLen < MIN_MATCH) {
            const int val = int(src[srcIdx]);
            ctx = (ctx << 8) | val;
            dst[dstIdx++] = src[srcIdx++];

            if ((ref != 0) && (val == MATCH_FLAG))
                dst[dstIdx++] = byte(0xFF);

            continue;
        }

        srcIdx += bestLen;
        prefetchRead(&src[srcIdx - 4]);
        ctx = LittleEndian::readInt32(&src[srcIdx - 4]);
        dst[dstIdx++] = byte(MATCH_FLAG);
        bestLen -= MIN_MATCH;

        // Emit match length
        while (bestLen >= 254) {
            bestLen -= 254;
            dst[dstIdx++] = byte(0xFE);

            if (dstIdx >= dstEnd)
                break;
        }

        dst[dstIdx++] = byte(bestLen);
    }

    while ((srcIdx < srcEnd) && (dstIdx < dstEnd)) {
        const uint32 h = (HASH_SEED * ctx) >> HASH_SHIFT;
        const int ref = _hashes[h];
        _hashes[h] = srcIdx;
        const int val = int32(src[srcIdx]);
        ctx = (ctx << 8) | val;
        dst[dstIdx++] = src[srcIdx++];

        if ((ref != 0) && (val == MATCH_FLAG) && (dstIdx < dstEnd))
            dst[dstIdx++] = byte(0xFF);
    }

    input._index += srcIdx;
    output._index += dstIdx;
    return (srcIdx == count) && (dstIdx < (count - (count >> 6)));
}

bool LZPCodec::inverse(SliceArray<byte>& input, SliceArray<byte>& output, int count)
{
    if (count == 0)
        return true;

    if (!SliceArray<byte>::isValid(input))
        throw invalid_argument("LZP codec: Invalid input block");

    if (!SliceArray<byte>::isValid(output))
        throw invalid_argument("LZP codec: Invalid output block");

    if (count < 4)
        return false;

    const int srcEnd = count;
    byte* dst = &output._array[output._index];
    byte* src = &input._array[input._index];

    if (_hashSize == 0) {
        _hashSize = 1 << HASH_LOG;
        delete[] _hashes;
        _hashes = new int32[_hashSize];
    }

    memset(_hashes, 0, sizeof(int32) * _hashSize);
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
    int32 ctx = LittleEndian::readInt32(&dst[0]);
    int srcIdx = 4;
    int dstIdx = 4;

    while (srcIdx < srcEnd) {
        const int32 h = (HASH_SEED * ctx) >> HASH_SHIFT;
        int ref = _hashes[h];
        _hashes[h] = dstIdx;

        if ((ref == 0) || (src[srcIdx] != byte(MATCH_FLAG))) {
            ctx = (ctx << 8) | int32(src[srcIdx]);
            dst[dstIdx++] = src[srcIdx++];
            continue;
        }

        srcIdx++;

        if (src[srcIdx] == byte(0xFF)) {
            ctx = (ctx << 8) | int32(MATCH_FLAG);
            dst[dstIdx++] = byte(MATCH_FLAG);
            srcIdx++;
            continue;
        }

        int mLen = MIN_MATCH;

        while ((srcIdx < srcEnd) && (src[srcIdx] == byte(0xFE))) {
            srcIdx++;
            mLen += 254;
        }

        if (srcIdx >= srcEnd)
            return false;

        mLen += int(src[srcIdx++]);
        const int mEnd = dstIdx + mLen;

        if (dstIdx >= ref + 8) {
            do {
                // No overlap
                memcpy(&dst[dstIdx], &dst[ref], 8);
                ref += 8;
                dstIdx += 8;
            } while (dstIdx < mEnd);
        }
        else {
            for (int i = 0; i < mLen; i++)
                dst[dstIdx + i] = dst[ref + i];
        }

        dstIdx = mEnd;
        ctx = LittleEndian::readInt32(&dst[dstIdx - 4]);
    }

    input._index += srcIdx;
    output._index += dstIdx;
    return srcIdx == srcEnd;
}

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

#pragma once
#ifndef _LZCodec_
#define _LZCodec_

#include "../Context.hpp"
#include "../Global.hpp"
#include "../Transform.hpp"
#include "../Memory.hpp"

namespace kanzi {

    class LZCodec FINAL : public Transform<byte> {

    public:
        LZCodec() THROW;

        LZCodec(Context& ctx) THROW;

        virtual ~LZCodec() { delete _delegate; }

        bool forward(SliceArray<byte>& src, SliceArray<byte>& dst, int length) THROW;

        bool inverse(SliceArray<byte>& src, SliceArray<byte>& dst, int length) THROW;

        // Required encoding output buffer size
        int getMaxEncodedLength(int srcLen) const
        {
            return _delegate->getMaxEncodedLength(srcLen);
        }

    private:
        Transform<byte>* _delegate;
    };

    // Simple byte oriented LZ77 implementation.
    template <bool T>
    class LZXCodec FINAL : public Transform<byte> {
    public:
        LZXCodec()
        {
            _hashes = new int32[0];
            _hashSize = 0;
            _tkBuf = new byte[0];
            _mLenBuf = new byte[0];
            _mBuf = new byte[0];
            _bufferSize = 0;
            _pCtx = nullptr;
        }

        LZXCodec(Context& ctx)
        {
            _hashes = new int32[0];
            _hashSize = 0;
            _tkBuf = new byte[0];
            _mLenBuf = new byte[0];
            _mBuf = new byte[0];
            _bufferSize = 0;
            _pCtx = &ctx;
        }

        virtual ~LZXCodec()
        {
            _bufferSize = 0;
            _hashSize = 0;
            delete[] _hashes;
            delete[] _mLenBuf;
            delete[] _mBuf;
            delete[] _tkBuf;
        }

        bool forward(SliceArray<byte>& src, SliceArray<byte>& dst, int length) THROW;

        bool inverse(SliceArray<byte>& src, SliceArray<byte>& dst, int length) THROW;

        // Required encoding output buffer size
        int getMaxEncodedLength(int srcLen) const
        {
            return (srcLen <= 1024) ? srcLen + 16 : srcLen + (srcLen / 64);
        }

    private:
        static const uint HASH_SEED = 0x1E35A7BD;
        static const uint HASH_LOG1 = 17;
        static const uint HASH_SHIFT1 = 40 - HASH_LOG1;
        static const uint HASH_MASK1 = (1 << HASH_LOG1) - 1;
        static const uint HASH_LOG2 = 21;
        static const uint HASH_SHIFT2 = 48 - HASH_LOG2;
        static const uint HASH_MASK2 = (1 << HASH_LOG2) - 1;
        static const int MAX_DISTANCE1 = (1 << 16) - 2;
        static const int MAX_DISTANCE2 = (1 << 24) - 2;
        static const int MIN_MATCH4 = 4;
        static const int MIN_MATCH9 = 9;
        static const int MAX_MATCH = 65535 + 254 + 15 + MIN_MATCH4;
        static const int MIN_BLOCK_LENGTH = 24;
        static const int MIN_MATCH_MIN_DIST = 1 << 16;

        int32* _hashes;
        int _hashSize;
        byte* _mLenBuf;
        byte* _mBuf;
        byte* _tkBuf;
        int _bufferSize;
        Context* _pCtx;

        static int emitLength(byte block[], int len);

        static void emitLiterals(const byte src[], byte dst[], int len);

        static int findMatch(const byte block[], const int pos, const int ref, const int maxMatch);

        static int readLength(const byte block[], int& pos);

        static int32 hash(const byte* p);
    };

    class LZPCodec FINAL : public Transform<byte> {
    public:
        LZPCodec()
        {
            _hashes = new int32[0];
            _hashSize = 0;
        }

        LZPCodec(Context&)
        {
            _hashes = new int32[0];
            _hashSize = 0;
        }

        virtual ~LZPCodec()
        {
            delete[] _hashes;
        }

        bool forward(SliceArray<byte>& src, SliceArray<byte>& dst, int length) THROW;

        bool inverse(SliceArray<byte>& src, SliceArray<byte>& dst, int length) THROW;

        // Required encoding output buffer size
        int getMaxEncodedLength(int srcLen) const
        {
            return (srcLen <= 1024) ? srcLen + 16 : srcLen + (srcLen / 64);
        }

    private:
        static const uint HASH_SEED = 0x7FEB352D;
        static const uint HASH_LOG = 16;
        static const uint HASH_SHIFT = 32 - HASH_LOG;
        static const int MIN_MATCH = 64;
        static const int MIN_BLOCK_LENGTH = 128;
        static const int MATCH_FLAG = 0xFC;

        int32* _hashes;
        int _hashSize;

        static int findMatch(const byte block[], const int pos, const int ref, const int maxMatch);
    };

    template <bool T>
    inline void LZXCodec<T>::emitLiterals(const byte src[], byte dst[], int len)
    {
        for (int i = 0; i < len; i += 8)
            memcpy(&dst[i], &src[i], 8);
    }

    template <bool T>
    inline int32 LZXCodec<T>::hash(const byte* p)
    {
        return (T == true) ? ((LittleEndian::readLong64(p) * HASH_SEED) >> HASH_SHIFT2) & HASH_MASK2 :
            ((LittleEndian::readLong64(p) * HASH_SEED) >> HASH_SHIFT1) & HASH_MASK1;
    }

    template <bool T>
    inline int LZXCodec<T>::emitLength(byte block[], int length)
    {
        if (length < 254) {
            block[0] = byte(length);
            return 1;
        }

        if (length < 65536 + 254) {
            length = (length - 254) | 0x00FE0000;
            kanzi::BigEndian::writeInt32(&block[0], length << 8);
            return 3;
        }

        length = (length - 255) | 0xFF000000;
        kanzi::BigEndian::writeInt32(&block[0], length);
        return 4;
    }

    template <bool T>
    inline int LZXCodec<T>::readLength(const byte block[], int& pos)
    {
        int res = int(block[pos++]);

        if (res < 254)
            return res;

        if (res == 254) {
            res += ((kanzi::BigEndian::readInt16(&block[pos])) & 0xFFFF);
            pos += 2;
            return res;
        }

        res += ((kanzi::BigEndian::readInt32(&block[pos])) >> 8);
        pos += 3;
        return res;
    }


    template <bool T>
    inline int LZXCodec<T>::findMatch(const byte src[], const int srcIdx, const int ref, const int maxMatch)
    {
        int n = 0;

        while (n + 4 <= maxMatch) {
            const int32 diff = LittleEndian::readInt32(&src[srcIdx + n]) ^ LittleEndian::readInt32(&src[ref + n]);

            if (diff != 0) {
                n += (Global::trailingZeros(uint32(diff)) >> 3);
                break;
            }

            n += 4;
        }

        return n;
    }


    inline int LZPCodec::findMatch(const byte src[], const int srcIdx, const int ref, const int maxMatch)
    {
        int n = 0;

        while (n + 8 <= maxMatch) {
            const int64 diff = LittleEndian::readLong64(&src[srcIdx + n]) ^ LittleEndian::readLong64(&src[ref + n]);

            if (diff != 0) {
                n += (Global::trailingZeros(uint64(diff)) >> 3);
                break;
            }

            n += 8;
        }

        return n;
    }

}
#endif


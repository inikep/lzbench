/*
Copyright 2011-2026 Frederic Langlet
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

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include "../Error.hpp"
#include "../bitstream/DefaultOutputBitStream.hpp"
#include "../entropy/EntropyDecoderFactory.hpp"
#include "../io/CompressedInputStream.hpp"
#include "../io/IOException.hpp"
#include "../transform/TransformFactory.hpp"

using namespace std;
using namespace kanzi;

#define ASSERT_TRUE(cond, msg)                                           \
    do {                                                                 \
        if (!(cond)) {                                                   \
            cerr << "ASSERT FAILED: " << msg << " (" << __FILE__         \
                 << ":" << __LINE__ << ")" << endl;                      \
            return 1;                                                    \
        }                                                                \
    } while (0)

static uint32 computeHeaderChecksum(int bsVersion, uint64 checksumSize,
    short entropyType, uint64 transformType, int blockSize, int szMask, uint64 outputSize)
{
    uint32 seed = uint32((bsVersion >= 6 ? 0x01030507 : 1) * bsVersion);
    const uint32 hash = 0x1E35A7BD;
    uint32 checksum = hash * seed;

    if (bsVersion >= 6)
        checksum ^= hash * uint32(~checksumSize);

    checksum ^= hash * uint32(~entropyType);
    checksum ^= hash * uint32((~transformType) >> 32);
    checksum ^= hash * uint32(~transformType);
    checksum ^= hash * uint32(~blockSize);

    if (szMask != 0) {
        checksum ^= hash * uint32((~outputSize) >> 32);
        checksum ^= hash * uint32(~outputSize);
    }

    return (checksum >> 23) ^ (checksum >> 3);
}

static string buildHeader(int type, int bsVersion, uint64 checksumSize, short entropyType,
    uint64 transformType, int blockSize, int szMask, uint64 outputSize, bool validChecksum)
{
    stringbuf buffer;
    iostream io(&buffer);
    DefaultOutputBitStream obs(io, 16384);
    const int crcSize = bsVersion <= 5 ? 16 : 24;

    obs.writeBits(uint32(type), 32);
    obs.writeBits(uint32(bsVersion), 4);

    if (bsVersion >= 6)
        obs.writeBits(checksumSize, 2);
    else
        obs.writeBit(checksumSize == 0 ? 0 : 1);

    obs.writeBits(uint64(entropyType), 5);
    obs.writeBits(transformType, 48);
    obs.writeBits(uint64(blockSize >> 4), 28);
    obs.writeBits(uint64(szMask), 2);

    if (szMask != 0)
        obs.writeBits(outputSize, 16 * szMask);

    if (bsVersion >= 6)
        obs.writeBits(uint64(0), 15);

    uint32 checksum = computeHeaderChecksum(bsVersion, checksumSize,
        entropyType, transformType, blockSize, szMask, outputSize);

    if (validChecksum == false)
        checksum ^= 1;

    obs.writeBits(checksum, crcSize);
    obs.close();
    return buffer.str();
}

static int expectHeaderFailure(const string& name, const string& data,
    int expectedError, const string& expectedText)
{
    cout << "Test malformed header: " << name << endl;
    istringstream is(data);
    CompressedInputStream cis(is, 1);
    char dst[1];

    try {
        cis.read(dst, 1);
    }
    catch (const IOException& e) {
        ASSERT_TRUE(e.error() == expectedError, "Unexpected IOException error code");
        ASSERT_TRUE(string(e.what()).find(expectedText) != string::npos,
            "Unexpected IOException message");
        return 0;
    }
    catch (const exception& e) {
        cerr << "Unexpected exception: " << e.what() << endl;
        return 1;
    }

    cerr << "Expected an exception for malformed header: " << name << endl;
    return 1;
}

int main()
{
    const int version = 6;
    const int type = 0x4B414E5A;
    const int blockSize = 1024;
    const short entropy = EntropyDecoderFactory::ANS0_TYPE;
    const uint64 transform = uint64(TransformFactory<kanzi::byte>::LZ_TYPE) << 42;

    if (expectHeaderFailure("invalid type",
            buildHeader(type ^ 1, version, 0, entropy, transform, blockSize, 0, 0, true),
            Error::ERR_INVALID_FILE, "Invalid stream type") != 0) {
        return 1;
    }

    if (expectHeaderFailure("unsupported version",
            buildHeader(type, version + 1, 0, entropy, transform, blockSize, 0, 0, true),
            Error::ERR_STREAM_VERSION, "cannot read this version") != 0) {
        return 1;
    }

    if (expectHeaderFailure("invalid checksum size",
            buildHeader(type, version, 3, entropy, transform, blockSize, 0, 0, true),
            Error::ERR_INVALID_FILE, "incorrect block checksum size") != 0) {
        return 1;
    }

    if (expectHeaderFailure("unknown entropy type",
            buildHeader(type, version, 0, 31, transform, blockSize, 0, 0, true),
            Error::ERR_INVALID_CODEC, "unknown entropy type") != 0) {
        return 1;
    }

    if (expectHeaderFailure("unknown transform type",
            buildHeader(type, version, 0, entropy, uint64(63) << 42, blockSize, 0, 0, true),
            Error::ERR_INVALID_CODEC, "unknown transform type") != 0) {
        return 1;
    }

    if (expectHeaderFailure("invalid block size",
            buildHeader(type, version, 0, entropy, transform, blockSize - 16, 0, 0, true),
            Error::ERR_BLOCK_SIZE, "incorrect block size") != 0) {
        return 1;
    }

    if (expectHeaderFailure("header checksum mismatch",
            buildHeader(type, version, 0, entropy, transform, blockSize, 0, 0, false),
            Error::ERR_CRC_CHECK, "header checksum mismatch") != 0) {
        return 1;
    }

    cout << "All malformed stream tests passed." << endl;
    return 0;
}

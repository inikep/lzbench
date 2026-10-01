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

#include <algorithm>
#include <iostream>
#include <time.h>
#include <vector>
#include "../types.hpp"
#include "../util/strings.hpp"
#include "../entropy/HuffmanEncoder.hpp"
#include "../entropy/RangeEncoder.hpp"
#include "../entropy/ANSRangeEncoder.hpp"
#include "../entropy/BinaryEntropyEncoder.hpp"
#include "../entropy/ExpGolombEncoder.hpp"
#include "../entropy/FPAQEncoder.hpp"
#include "../entropy/EntropyUtils.hpp"
#include "../bitstream/DefaultOutputBitStream.hpp"
#include "../bitstream/DefaultInputBitStream.hpp"
#include "../bitstream/DebugOutputBitStream.hpp"
#include "../entropy/HuffmanDecoder.hpp"
#include "../entropy/RangeDecoder.hpp"
#include "../entropy/ANSRangeDecoder.hpp"
#include "../entropy/BinaryEntropyDecoder.hpp"
#include "../entropy/ExpGolombDecoder.hpp"
#include "../entropy/FPAQDecoder.hpp"
#include "../entropy/CMPredictor.hpp"
#include "../entropy/TPAQPredictor.hpp"

using namespace kanzi;
using namespace std;

static EntropyEncoder* getEncoder(string name, OutputBitStream& obs, Predictor* predictor);
static EntropyDecoder* getDecoder(string name, InputBitStream& ibs, Predictor* predictor);

static void copyBits(InputBitStream& ibs, OutputBitStream& obs, uint64 count)
{
    while (count >= 64) {
        obs.writeBits(ibs.readBits(64), 64);
        count -= 64;
    }

    if (count > 0)
        obs.writeBits(ibs.readBits(uint(count)), uint(count));
}

static string encodeEntropyPayload(const string& name, const kanzi::byte block[], uint size)
{
    stringbuf encoded;
    iostream ios(&encoded);
    DefaultOutputBitStream obs(ios, 16384);
    EntropyEncoder* ec = getEncoder(name, obs, NULL);

    if (ec == NULL)
        return "";

    const int res = ec->encode(block, 0, size);
    ec->dispose();
    delete ec;
    obs.close();
    return (res == int(size)) ? encoded.str() : "";
}

static string shrinkFPAQDeclaredSize(const string& data)
{
    istringstream is(data);
    DefaultInputBitStream ibs(is, 16384);
    stringbuf mutated;
    iostream ios(&mutated);
    DefaultOutputBitStream obs(ios, 16384);
    const uint32 sz = EntropyUtils::readVarInt(ibs);

    if (sz == 0)
        return "";

    EntropyUtils::writeVarInt(obs, sz - 1);
    copyBits(ibs, obs, uint64(data.size()) * 8 - ibs.read());
    obs.close();
    return mutated.str();
}

static string zeroFPAQDeclaredSize(const string& data)
{
    istringstream is(data);
    DefaultInputBitStream ibs(is, 16384);
    stringbuf mutated;
    iostream ios(&mutated);
    DefaultOutputBitStream obs(ios, 16384);
    const uint32 sz = EntropyUtils::readVarInt(ibs);

    if (sz == 0)
        return "";

    EntropyUtils::writeVarInt(obs, 0);
    copyBits(ibs, obs, uint64(data.size()) * 8 - ibs.read());
    obs.close();
    return mutated.str();
}

static string shrinkANSDeclaredSize(const string& data)
{
    istringstream is(data);
    DefaultInputBitStream ibs(is, 16384);
    stringbuf mutated;
    iostream ios(&mutated);
    DefaultOutputBitStream obs(ios, 16384);
    const uint logRange = uint(ibs.readBits(3));
    obs.writeBits(logRange, 3);
    uint alphabet[256];
    const int alphabetSize = EntropyUtils::decodeAlphabet(ibs, alphabet);

    if (EntropyUtils::encodeAlphabet(obs, alphabet, 256, alphabetSize) < 0)
        return "";

    if (alphabetSize > 1) {
        const int chkSize = (alphabetSize >= 64) ? 8 : 6;

        for (int i = 1; i < alphabetSize; i += chkSize) {
            const uint logMax = uint(ibs.readBits(4));
            obs.writeBits(logMax, 4);

            if (logMax == 0)
                continue;

            const int endj = min(i + chkSize, alphabetSize);

            for (int j = i; j < endj; j++)
                obs.writeBits(ibs.readBits(logMax), logMax);
        }
    }

    const uint32 sz = EntropyUtils::readVarInt(ibs);

    if (sz == 0)
        return "";

    EntropyUtils::writeVarInt(obs, sz - 1);
    copyBits(ibs, obs, uint64(data.size()) * 8 - ibs.read());
    obs.close();
    return mutated.str();
}

static string shrinkHuffmanDeclaredSize(const string& data)
{
    istringstream is(data);
    DefaultInputBitStream ibs(is, 16384);
    stringbuf mutated;
    iostream ios(&mutated);
    DefaultOutputBitStream obs(ios, 16384);
    uint alphabet[256];
    const int alphabetSize = EntropyUtils::decodeAlphabet(ibs, alphabet);

    if (EntropyUtils::encodeAlphabet(obs, alphabet, 256, alphabetSize) < 0)
        return "";

    if (alphabetSize <= 1)
        return "";

    ExpGolombDecoder egdec(ibs, true);
    ExpGolombEncoder egenc(obs, true);

    for (int i = 0; i < alphabetSize; i++) {
        const kanzi::byte delta = egdec.decodeByte();
        egenc.encodeByte(delta);
    }

    uint32 sz[4];
    int idx = -1;

    for (int i = 0; i < 4; i++) {
        sz[i] = EntropyUtils::readVarInt(ibs);

        if ((sz[i] > 0) && ((idx < 0) || (sz[i] > sz[idx])))
            idx = i;
    }

    if (idx < 0)
        return "";

    for (int i = 0; i < 4; i++)
        EntropyUtils::writeVarInt(obs, sz[i] - ((i == idx) ? 1 : 0));

    copyBits(ibs, obs, uint64(data.size()) * 8 - ibs.read());
    obs.close();
    return mutated.str();
}

static bool malformedEntropyIsRejected(const string& name, const string& data, uint size)
{
    istringstream is(data);
    DefaultInputBitStream ibs(is, 16384);
    EntropyDecoder* ed = getDecoder(name, ibs, NULL);

    if (ed == NULL)
        return false;

    vector<kanzi::byte> decoded(size, kanzi::byte(0));

    try {
        const int res = ed->decode(&decoded[0], 0, size);
        ed->dispose();
        delete ed;
        ibs.close();
        return res != int(size);
    }
    catch (const exception&) {
        delete ed;

        try {
            ibs.close();
        }
        catch (const exception&) {
        }

        return true;
    }
}

int testDeclaredPayloadConsumption()
{
    cout << endl
         << "=== Declared payload consumption test ===" << endl;
    const uint size = 4096;
    vector<kanzi::byte> values(size);

    for (uint i = 0; i < size; i++)
        values[i] = kanzi::byte(((i * 13) ^ (i >> 3) ^ ((i & 15) << 4)) & 0xFF);

    struct TestCase {
        const char* name;
        string (*mutator)(const string&);
    };

    TestCase tests[3] = {
        { "HUFFMAN", shrinkHuffmanDeclaredSize },
        { "ANS0",    shrinkANSDeclaredSize },
        { "FPAQ",    shrinkFPAQDeclaredSize }
    };

    for (int i = 0; i < 3; i++) {
        const string encoded = encodeEntropyPayload(tests[i].name, &values[0], size);

        if (encoded.empty()) {
            cout << "Could not encode test payload for " << tests[i].name << endl;
            return 1;
        }

        const string mutated = tests[i].mutator(encoded);

        if (mutated.empty()) {
            cout << "Could not mutate test payload for " << tests[i].name << endl;
            return 2;
        }

        if (malformedEntropyIsRejected(tests[i].name, mutated, size) == false) {
            cout << "Malformed payload accepted for " << tests[i].name << endl;
            return 3;
        }
    }

    cout << "Declared payload consumption test passed" << endl;
    return 0;
}

int testFPAQZeroDeclaredSize()
{
    cout << endl
         << "=== FPAQ zero declared size test ===" << endl;
    const uint size = 1 << 20;
    vector<kanzi::byte> values(size);

    for (uint i = 0; i < size; i++)
        values[i] = kanzi::byte((i * 17) & 0xFF);

    const string encoded = encodeEntropyPayload("FPAQ", &values[0], size);

    if (encoded.empty()) {
        cout << "Could not encode test payload for FPAQ" << endl;
        return 1;
    }

    const string mutated = zeroFPAQDeclaredSize(encoded);

    if (mutated.empty()) {
        cout << "Could not mutate test payload for FPAQ" << endl;
        return 2;
    }

    if (malformedEntropyIsRejected("FPAQ", mutated, size) == false) {
        cout << "Malformed zero-sized FPAQ payload accepted" << endl;
        return 3;
    }

    cout << "FPAQ zero declared size test passed" << endl;
    return 0;
}

int testANS1MissingContext()
{
    cout << endl << "=== ANS1 implicit context test ===" << endl;
    const uint size = 40;
    const uint alphabet[] = { 1 };

    // Exercise both freshly allocated tables and tables from an earlier chunk.
    for (int warm = 0; warm < 2; warm++) {
        stringbuf buffer;
        iostream ios(&buffer);
        DefaultOutputBitStream obs(ios, 16384);
        vector<kanzi::byte> previous(size, kanzi::byte(1));

        if (warm != 0) {
            ANSRangeEncoder encoder(obs, 1);
            encoder.encode(&previous[0], 0, size);
        }

        // Context 0 emits 1; empty context 1 implicitly emits 0.
        obs.writeBits(uint64(0), 3); // log range = 8
        EntropyUtils::encodeAlphabet(obs, alphabet, 256, 1);

        for (int i = 1; i < 256; i++)
            EntropyUtils::encodeAlphabet(obs, alphabet, 256, 0);

        EntropyUtils::writeVarInt(obs, 8);

        for (int i = 0; i < 4; i++)
            obs.writeBits(uint32(1 << 15), 32);

        // Each lane renormalizes once, on its first symbol.
        const kanzi::byte payload[8] = { kanzi::byte(0) };
        obs.writeBits(payload, 64);
        obs.close();

        istringstream is(buffer.str());
        DefaultInputBitStream ibs(is, 16384);
        ANSRangeDecoder decoder(ibs, 1);
        vector<kanzi::byte> decoded(size);

        if (warm != 0) {
            if ((decoder.decode(&decoded[0], 0, size) != int(size)) ||
                (decoded != previous))
                return 1;
        }

        if (decoder.decode(&decoded[0], 0, size) != int(size))
            return 1;

        for (uint i = 0; i < size; i++) {
            if (decoded[i] != kanzi::byte(1 - ((i % (size / 4)) & 1))) {
                cout << "Incorrect ANS1 implicit context output" << endl;
                return 1;
            }
        }
    }

    cout << "ANS1 implicit context test passed" << endl;
    return 0;
}

int testHuffmanFragmentedRoundTrip()
{
    cout << endl
         << "=== Huffman fragmented round-trip test ===" << endl;
    const uint size = 1649;
    vector<kanzi::byte> values(size);
    vector<kanzi::byte> decoded(size, kanzi::byte(0));
    uint32 state = 3454687338U;

    for (uint i = 0; i < size; i++) {
        state = (state * 1103515245U) + 12345U;
        const uint32 val0 = state >> 24;
        state = (state * 1103515245U) + 12345U;
        const uint32 val1 = state >> 24;
        values[i] = (val0 < 192) ? kanzi::byte(val1 & 15) : kanzi::byte(val1);
    }

    stringbuf buffer;
    iostream ios(&buffer);
    DefaultOutputBitStream obs(ios, 1 << 15);
    HuffmanEncoder encoder(obs);

    if (encoder.encode(&values[0], 0, size) != int(size)) {
        cout << "Encoding error in Huffman fragmented round-trip test" << endl;
        return 1;
    }

    encoder.dispose();
    obs.close();
    ios.rdbuf()->pubseekpos(0);
    DefaultInputBitStream ibs(ios, 1 << 15);
    HuffmanDecoder decoder(ibs);

    if (decoder.decode(&decoded[0], 0, size) != int(size)) {
        cout << "Decoding error in Huffman fragmented round-trip test" << endl;
        return 1;
    }

    decoder.dispose();
    ibs.close();

    if (memcmp(&values[0], &decoded[0], size) != 0) {
        cout << "Mismatch in Huffman fragmented round-trip test" << endl;
        return 1;
    }

    cout << "Huffman fragmented round-trip test passed" << endl;
    return 0;
}

class ConstantPredictor FINAL : public Predictor
{
public:
    explicit ConstantPredictor(int value) : _value(value) {}

    void update(int) {}

    int get() { return _value; }

private:
    int _value;
};

static Predictor* getPredictor(string type)
{
    if (type.compare("TPAQ") == 0)
        return new TPAQPredictor<false>();

    if (type.compare("TPAQX") == 0)
        return new TPAQPredictor<true>();

    if (type.compare("CM") == 0)
        return new CMPredictor();

    return nullptr;
}

static EntropyEncoder* getEncoder(string name, OutputBitStream& obs, Predictor* predictor)
{
    if (name.compare("HUFFMAN") == 0)
        return new HuffmanEncoder(obs);

    if (name.compare("ANS0") == 0)
        return new ANSRangeEncoder(obs, 0);

    if (name.compare("ANS1") == 0)
        return new ANSRangeEncoder(obs, 1);

    if (name.compare("RANGE") == 0)
        return new RangeEncoder(obs);

    if (name.compare("EXPGOLOMB") == 0)
        return new ExpGolombEncoder(obs);

    if (name.compare("FPAQ") == 0)
        return new FPAQEncoder(obs);

    if (predictor != nullptr) {
       if (name.compare("TPAQ") == 0)
           return new BinaryEntropyEncoder(obs, predictor, true);

       if (name.compare("CM") == 0)
           return new BinaryEntropyEncoder(obs, predictor, true);
    }

    cout << "No such entropy encoder: " << name << endl;
    return nullptr;
}

static EntropyDecoder* getDecoder(string name, InputBitStream& ibs, Predictor* predictor)
{
    if (name.compare("HUFFMAN") == 0)
        return new HuffmanDecoder(ibs);

    if (name.compare("ANS0") == 0)
        return new ANSRangeDecoder(ibs, 0);

    if (name.compare("ANS1") == 0)
        return new ANSRangeDecoder(ibs, 1);

    if (name.compare("RANGE") == 0)
        return new RangeDecoder(ibs);

    if (name.compare("FPAQ") == 0)
        return new FPAQDecoder(ibs);

    if (predictor != nullptr) {
        if (name.compare("TPAQ") == 0)
            return new BinaryEntropyDecoder(ibs, predictor, true);

        if (name.compare("CM") == 0)
            return new BinaryEntropyDecoder(ibs, predictor, true);
    }

    if (name.compare("EXPGOLOMB") == 0)
        return new ExpGolombDecoder(ibs);

    cout << "No such entropy decoder: " << name << endl;
    return nullptr;
}

static int testBinaryPayloadBoundary()
{
    cout << endl
         << "=== Binary entropy payload boundary test ===" << endl;
    const uint size = 1 << 16;
    vector<kanzi::byte> values(size, kanzi::byte(0));
    stringbuf buffer;
    iostream ios(&buffer);
    DefaultOutputBitStream obs(ios, 16384);
    BinaryEntropyEncoder encoder(obs, new ConstantPredictor(4095), true);

    if (encoder.encode(&values[0], 0, size) != int(size))
        return 1;

    encoder.dispose();
    obs.close();
    const string mutated = zeroFPAQDeclaredSize(buffer.str());

    if (mutated.empty())
        return 2;

    istringstream is(mutated);
    DefaultInputBitStream ibs(is, 16384);
    BinaryEntropyDecoder decoder(ibs, new ConstantPredictor(4095), true);
    vector<kanzi::byte> decoded(size, kanzi::byte(0));
    bool accepted = false;

    try {
        accepted = decoder.decode(&decoded[0], 0, size) == int(size);
    }
    catch (const exception&) {
        // Expected: the first four-byte refill crosses the declared boundary.
    }

    decoder.dispose();
    ibs.close();

    if (accepted) {
        cout << "Malformed binary entropy payload was accepted" << endl;
        return 3;
    }

    cout << "Binary entropy payload boundary test passed" << endl;
    return 0;
}

int testEntropyCodecCorrectness(const string& name)
{
    // Test behavior
    cout << "=== Correctness test for " << name << " ===" << endl;
    srand((uint)time(nullptr));
    int res = 0;

    for (int ii = 1; ii < 50; ii++) {
        cout << endl
             << endl
             << "Test " << ii << endl;
        kanzi::byte val[256];
        int size = 40;

        if (ii == 3) {
            kanzi::byte val2[] = { (kanzi::byte)0, (kanzi::byte)0, (kanzi::byte)32, (kanzi::byte)15, (kanzi::byte)-4, (kanzi::byte)16, (kanzi::byte)0, (kanzi::byte)16, (kanzi::byte)0, (kanzi::byte)7, (kanzi::byte)-1, (kanzi::byte)-4, (kanzi::byte)-32, (kanzi::byte)0, (kanzi::byte)31, (kanzi::byte)-1 };
            size = 16;
            memcpy(val, &val2[0], size);
        }
        else if (ii == 2) {
            kanzi::byte val2[] = { (kanzi::byte)0x3d, (kanzi::byte)0x4d, (kanzi::byte)0x54, (kanzi::byte)0x47, (kanzi::byte)0x5a, (kanzi::byte)0x36, (kanzi::byte)0x39, (kanzi::byte)0x26, (kanzi::byte)0x72, (kanzi::byte)0x6f, (kanzi::byte)0x6c, (kanzi::byte)0x65, (kanzi::byte)0x3d, (kanzi::byte)0x70, (kanzi::byte)0x72, (kanzi::byte)0x65 };
            size = 16;
            memcpy(val, &val2[0], size);
        }
        else if (ii == 1) {
            for (int i = 0; i < size; i++)
                val[i] = kanzi::byte(2); // all identical
        }
        else if (ii == 4) {
            for (int i = 0; i < size; i++)
                val[i] = kanzi::byte(2 + (i & 1)); // 2 symbols
        }
        else if (ii == 5) {
            size = 1;
            val[0] = kanzi::byte(42);
        }
        else if (ii == 6) {
            size = 2;
            val[0] = kanzi::byte(42);
            val[1] = kanzi::byte(42);
        }
        else if (ii == 7) {
            for (int i = 0; i < 44; i++)
                val[i] = kanzi::byte(i & 7);
        }
        else {
            size = 256;

            for (int i = 0; i < 256; i++)
                val[i] = kanzi::byte(64 + 4 * ii + (rand() % (8*ii + 1)));
        }

        kanzi::byte* values = &val[0];
        cout << "Original:" << endl;

        for (int i = 0; i < size; i++)
            cout << int(values[i]) << " ";

        cout << endl
             << endl
             << "Encoded:" << endl;
        stringbuf buffer;
        iostream ios(&buffer);
        DefaultOutputBitStream obs(ios);
        DebugOutputBitStream dbgobs(obs);
        dbgobs.showByte(true);

        EntropyEncoder* ec = getEncoder(name, dbgobs, getPredictor(name));

        if (ec == nullptr)
           return 1;

        ec->encode(values, 0, size);
        ec->dispose();
        delete ec;
        dbgobs.close();
        ios.rdbuf()->pubseekpos(0);

        DefaultInputBitStream ibs(ios);
        EntropyDecoder* ed = getDecoder(name, ibs, getPredictor(name));

        if (ed == nullptr)
           return 1;

        cout << endl
             << endl
             << "Decoded:" << endl;
        bool ok = true;
        kanzi::byte* values2 = new kanzi::byte[size];
        ed->decode(values2, 0, size);
        ed->dispose();
        delete ed;
        ibs.close();

        for (int j = 0; j < size; j++) {
            if (values[j] != values2[j])
                ok = false;

            cout << (int)values2[j] << " ";
        }

        cout << endl;
        cout << (ok ? "Identical" : "Different") << endl;
        delete[] values2;
        res = ok ? 0 : 2;
    }

    return res;
}

static int testExpGolombUnsignedRoundTrip()
{
    cout << "=== Correctness test for unsigned EXPGOLOMB ===" << endl;
    const uint size = 256;
    kanzi::byte values[size];

    for (uint i = 0; i < size; i++)
        values[i] = kanzi::byte(i);

    stringbuf buffer;
    iostream ios(&buffer);
    DefaultOutputBitStream obs(ios);
    ExpGolombEncoder encoder(obs, false);

    if (encoder.encode(values, 0, size) != int(size)) {
        cout << "Unsigned EXPGOLOMB encoding failed" << endl;
        return 1;
    }

    encoder.dispose();
    obs.close();
    ios.rdbuf()->pubseekpos(0);

    DefaultInputBitStream ibs(ios);
    ExpGolombDecoder decoder(ibs, false);
    kanzi::byte decoded[size];

    if (decoder.decode(decoded, 0, size) != int(size)) {
        cout << "Unsigned EXPGOLOMB decoding failed" << endl;
        return 1;
    }

    decoder.dispose();
    ibs.close();

    for (uint i = 0; i < size; i++) {
        if (values[i] != decoded[i]) {
            cout << "Unsigned EXPGOLOMB mismatch at index " << i << endl;
            return 1;
        }
    }

    return 0;
}

int testEntropyCodecSpeed(const string& name)
{
    // Test speed
    cout << endl
         << endl
         << "=== Speed test for " << name << " ===" << endl;
    int repeats[] = { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
    int size = 500000;
    int iter = 100;
    int res = 0;

    srand((uint)time(nullptr));
    kanzi::byte values1[500000];
    kanzi::byte values2[500000];

    for (int jj = 0; jj < 3; jj++) {
        cout << endl
             << "Test " << (jj + 1) << endl;
        double delta1 = 0, delta2 = 0;

        for (int ii = 0; ii < iter; ii++) {
            int idx = 0;
            memset(values1, 0x00, size);
            memset(values2, 0xAA, size);
            int n = 0;

            while (n < size) {
                int n0 = n;
                int len = max(min(repeats[idx], size - n), 1);
                idx = (idx + 1) & 0x0F;
                kanzi::byte b = (kanzi::byte)(rand() % 255);

                for (int j = n0; j < n0 + len; j++) {
                    values1[j] = b;
                    n++;
                }
            }

            // Encode
            stringbuf buffer;
            iostream ios(&buffer);
            DefaultOutputBitStream obs(ios, 16384);
            EntropyEncoder* ec = getEncoder(name, obs, getPredictor(name));

            if (ec == nullptr)
                 return 1;

            clock_t before1 = clock();

            if (ec->encode(values1, 0, size) < 0) {
                cout << "Encoding error" << endl;
                delete ec;
                return 1;
            }

            ec->dispose();
            clock_t after1 = clock();
            delta1 += (after1 - before1);
            delete ec;
            obs.close();

            // Decode
            ios.rdbuf()->pubseekpos(0);
            DefaultInputBitStream ibs(ios, 16384);
            EntropyDecoder* ed = getDecoder(name, ibs, getPredictor(name));

            if (ed == nullptr)
                 return 1;

            clock_t before2 = clock();

            if (ed->decode(values2, 0, size) < 0) {
                cout << "Decoding error" << endl;
                delete ed;
                return 1;
            }

            ed->dispose();
            clock_t after2 = clock();
            delta2 += (after2 - before2);
            delete ed;
            ibs.close();

            // Sanity check
            for (int i = 0; i < size; i++) {
                if (values1[i] != values2[i]) {
                    cout << "Error at index " << i << " (" << (int)values1[i] << "<->" << (int)values2[i] << ")" << endl;
                    res = 1;
                    break;
                }
            }
        }

        // KB = 1000, KiB = 1024
        double prod = double(iter) * double(size);
        double b2KiB = double(1) / double(1024);
        double d1_sec = delta1 / CLOCKS_PER_SEC;
        double d2_sec = delta2 / CLOCKS_PER_SEC;
        cout << "Encode [ms]        : " << (int)(d1_sec * 1000) << endl;
        cout << "Throughput [KiB/s] : " << (int)(prod * b2KiB / d1_sec) << endl;
        cout << "Decode [ms]        : " << (int)(d2_sec * 1000) << endl;
        cout << "Throughput [KiB/s] : " << (int)(prod * b2KiB / d2_sec) << endl;
    }

    return res;
}

int testBinaryEntropyBufferGrowth()
{
    cout << endl
         << "=== Buffer growth test for binary entropy codec ===" << endl;
    const int size = 1 << 20;
    vector<kanzi::byte> values(size, kanzi::byte(0));
    vector<kanzi::byte> decoded(size, kanzi::byte(0xAA));
    stringbuf buffer;
    iostream ios(&buffer);
    DefaultOutputBitStream obs(ios, 1 << 15);
    BinaryEntropyEncoder encoder(obs, new ConstantPredictor(4095), true);

    if (encoder.encode(&values[0], 0, uint(values.size())) != size) {
        cout << "Encoding error in buffer growth test" << endl;
        return 1;
    }

    encoder.dispose();
    obs.close();
    ios.rdbuf()->pubseekpos(0);
    DefaultInputBitStream ibs(ios, 1 << 15);
    BinaryEntropyDecoder decoder(ibs, new ConstantPredictor(4095), true);

    if (decoder.decode(&decoded[0], 0, uint(decoded.size())) != size) {
        cout << "Decoding error in buffer growth test" << endl;
        return 1;
    }

    decoder.dispose();
    ibs.close();

    if (memcmp(&values[0], &decoded[0], values.size()) != 0) {
        cout << "Mismatch in buffer growth test" << endl;
        return 1;
    }

    cout << "Buffer growth test passed" << endl;
    return 0;
}

#ifdef __GNUG__
int main(int argc, const char* argv[])
#else
int TestEntropyCodec_main(int argc, const char* argv[])
#endif
{
    int res = 0;

    try {
        res |= testExpGolombUnsignedRoundTrip();
        res |= testBinaryEntropyBufferGrowth();
        res |= testBinaryPayloadBoundary();
        res |= testDeclaredPayloadConsumption();
        res |= testFPAQZeroDeclaredSize();
        res |= testANS1MissingContext();
        res |= testHuffmanFragmentedRoundTrip();
        vector<string> codecs;
        bool doPerf = true;

        if (argc == 1) {
#if __cplusplus < 201103L
            string allCodecs[] = { "HUFFMAN", "ANS0", "ANS1", "RANGE", "EXPGOLOMB", "CM", "TPAQ" };
            const int count = int(sizeof(allCodecs) / sizeof(allCodecs[0]));

            for (int i = 0; i < count; i++)
                codecs.push_back(allCodecs[i]);
#else
            codecs = { "HUFFMAN", "ANS0", "ANS1", "RANGE", "EXPGOLOMB", "CM", "TPAQ" };
#endif
        }
        else {
            string str = argv[1];
            transform(str.begin(), str.end(), str.begin(), safeToUpper);

            if (str == "-TYPE=ALL") {
#if __cplusplus < 201103L
               string allCodecs[] = { "HUFFMAN", "ANS0", "ANS1", "RANGE", "EXPGOLOMB", "CM", "TPAQ" };
               const int count = int(sizeof(allCodecs) / sizeof(allCodecs[0]));

               for (int i = 0; i < count; i++)
                   codecs.push_back(allCodecs[i]);
#else
               codecs = { "HUFFMAN", "ANS0", "ANS1", "RANGE", "EXPGOLOMB", "CM", "TPAQ" };
#endif
            }
            else {
                codecs.push_back(str.substr(6));
            }

        if (argc > 2) {
                str = argv[2];
                transform(str.begin(), str.end(), str.begin(), safeToUpper);
                doPerf = str != "-NOPERF";
            }
        }

        for (vector<string>::iterator it = codecs.begin(); it != codecs.end(); ++it) {
            cout << endl
                 << endl
                 << "Test" << *it << endl;
            res |= testEntropyCodecCorrectness(*it);

        if (doPerf == true)
               res |= testEntropyCodecSpeed(*it);
        }
    }
    catch (const exception& e) {
        cout << e.what() << endl;
        res = 123;
    }

    cout << endl;
    cout << ((res == 0) ? "Success" : "Failure") << endl;
    return res;
}

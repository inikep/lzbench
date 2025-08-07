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

#include <algorithm>
#include <iostream>
#include <time.h>
#include <vector>
#include "../types.hpp"
#include "../entropy/HuffmanEncoder.hpp"
#include "../entropy/RangeEncoder.hpp"
#include "../entropy/ANSRangeEncoder.hpp"
#include "../entropy/BinaryEntropyEncoder.hpp"
#include "../entropy/ExpGolombEncoder.hpp"
#include "../entropy/FPAQEncoder.hpp"
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
        byte val[256];
        int size = 40;

        if (ii == 3) {
            byte val2[] = { (byte)0, (byte)0, (byte)32, (byte)15, (byte)-4, (byte)16, (byte)0, (byte)16, (byte)0, (byte)7, (byte)-1, (byte)-4, (byte)-32, (byte)0, (byte)31, (byte)-1 };
            size = 16;
            memcpy(val, &val2[0], size);
        }
        else if (ii == 2) {
            byte val2[] = { (byte)0x3d, (byte)0x4d, (byte)0x54, (byte)0x47, (byte)0x5a, (byte)0x36, (byte)0x39, (byte)0x26, (byte)0x72, (byte)0x6f, (byte)0x6c, (byte)0x65, (byte)0x3d, (byte)0x70, (byte)0x72, (byte)0x65 };
            size = 16;
            memcpy(val, &val2[0], size);
        }
        else if (ii == 1) {
            for (int i = 0; i < size; i++)
                val[i] = byte(2); // all identical
        }
        else if (ii == 4) {
            for (int i = 0; i < size; i++)
                val[i] = byte(2 + (i & 1)); // 2 symbols
        }
        else if (ii == 5) {
            size = 1;
            val[0] = byte(42);
        }
        else if (ii == 6) {
            size = 2;
            val[0] = byte(42);
            val[1] = byte(42);
        }
        else if (ii == 7) {
            for (int i = 0; i < 44; i++)
                val[i] = byte(i & 7);
        }
        else {
            size = 256;

            for (int i = 0; i < 256; i++)
                val[i] = byte(64 + 4 * ii + (rand() % (8*ii + 1)));
        }

        byte* values = &val[0];
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
        byte* values2 = new byte[size];
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
    byte values1[500000];
    byte values2[500000];

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
                byte b = (byte)(rand() % 255);

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

#ifdef __GNUG__
int main(int argc, const char* argv[])
#else
int TestEntropyCodec_main(int argc, const char* argv[])
#endif
{
    int res = 0;

    try {
        vector<string> codecs;
        bool doPerf = true;

        if (argc == 1) {
#if __cplusplus < 201103L
            string allCodecs[8] = { "HUFFMAN", "ANS0", "ANS1", "RANGE", "EXPGOLOMB", "CM", "TPAQ" };

            for (int i = 0; i < 8; i++)
                codecs.push_back(allCodecs[i]);
#else
            codecs = { "HUFFMAN", "ANS0", "ANS1", "RANGE", "EXPGOLOMB", "CM", "TPAQ" };
#endif
        }
        else {
            string str = argv[1];
            transform(str.begin(), str.end(), str.begin(), ::toupper);

            if (str == "-TYPE=ALL") {
#if __cplusplus < 201103L
               string allCodecs[] = { "HUFFMAN", "ANS0", "ANS1", "RANGE", "EXPGOLOMB", "CM", "TPAQ" };

               for (int i = 0; i < 8; i++)
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
                transform(str.begin(), str.end(), str.begin(), ::toupper);
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
    catch (exception& e) {
        cout << e.what() << endl;
        res = 123;
    }

    cout << endl;
    cout << ((res == 0) ? "Success" : "Failure") << endl;
    return res;
}

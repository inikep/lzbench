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
#include "../types.hpp"
#include "../transform/AliasCodec.hpp"
#include "../transform/FSDCodec.hpp"
#include "../transform/LZCodec.hpp"
#include "../transform/NullTransform.hpp"
#include "../transform/RLT.hpp"
#include "../transform/ROLZCodec.hpp"
#include "../transform/SBRT.hpp"
#include "../transform/SRT.hpp"
#include "../transform/TransformFactory.hpp"
#include "../transform/ZRLT.hpp"

using namespace std;
using namespace kanzi;

static Transform<byte>* getByteTransform(string name, Context& ctx)
{
    if (name.compare("SRT") == 0)
        return new SRT(ctx);

    if (name.compare("RLT") == 0)
        return new RLT(ctx);

    if (name.compare("ZRLT") == 0)
        return new ZRLT(ctx);

    if (name.compare("LZ") == 0)
        return new LZCodec(ctx);

    if (name.compare("LZX") == 0){
        ctx.putInt("lz", TransformFactory<byte>::LZX_TYPE);
        return new LZCodec(ctx);
    }

    if (name.compare("LZP") == 0){
        ctx.putInt("lz", TransformFactory<byte>::LZP_TYPE);
        return new LZCodec(ctx);
    }

    if (name.compare("ROLZ") == 0)
        return new ROLZCodec(ctx);

    if (name.compare("ROLZX") == 0)
        return new ROLZCodec(ctx);

    if (name.compare("RANK") == 0)
        return new SBRT(SBRT::MODE_RANK, ctx);

    if (name.compare("MTFT") == 0)
        return new SBRT(SBRT::MODE_MTF, ctx);

    if (name.compare("MM") == 0)
        return new FSDCodec(ctx);

    if (name.compare("NONE") == 0)
        return new NullTransform(ctx);

    if (name.compare("ALIAS") == 0)
        return new AliasCodec(ctx);

    cout << "No such byte transform: " << name << endl;
    return nullptr;
}

int testTransformsCorrectness(const string& name)
{
    srand((uint)time(nullptr));

    cout << endl
         << "Correctness for " << name << endl;
    int mod = (name == "ZRLT") ? 5 : 256;
    int res = 0;

    for (int ii = 0; ii < 51; ii++) {
        cout << endl
             << "Test " << ii << endl;
        int size; // Declare size, will be set in conditions
        byte values[1024 * 1024] = { byte(0xAA) };

        if (ii != 50) {
            size = 80000;
        }

        if (name == "ALIAS")
          mod = 15 + 12 * ii;

        if (ii == 0) {
            size = 32;
            byte arr[32] = {
                (byte)0, (byte)1, (byte)2, (byte)2, (byte)2, (byte)2, (byte)7, (byte)9,
                (byte)9, (byte)16, (byte)16, (byte)16, (byte)1, (byte)3, (byte)3, (byte)3,
                (byte)3, (byte)3, (byte)3, (byte)3, (byte)3, (byte)3, (byte)3, (byte)3,
                (byte)3, (byte)3, (byte)3, (byte)3, (byte)3, (byte)3, (byte)3, (byte)3
            };

            memcpy(values, &arr[0], size);
        }
        else if (ii < 10) {
            size = ii;
            memset(values, ii, size);
        }
        else if (ii == 10) {
            size = 255;
            memset(values, ii, size);
            values[127] = byte(255);
        }
        else if (ii == 11) {
            size = 80000;
            byte arr[80000];
            arr[0] = byte(1);

            for (int i = 1; i < 80000; i++)
                arr[i] = byte(8);

            memcpy(values, &arr[0], size);
        }
        else if (ii == 12) {
            size = 8;
            byte arr[8] = { (byte)0, (byte)0, (byte)1, (byte)1, (byte)2, (byte)2, (byte)3, (byte)3 };
            memcpy(values, &arr[0], size);
        }
        else if (ii == 13) {
            // For RLT
            size = 512;
            byte arr[512];

            for (int i = 0; i < 256; i++) {
                arr[2 * i] = byte(i);
                arr[2 * i + 1] = byte(i);
            }

            arr[1] = byte(255); // force RLT escape to be first symbol
            memcpy(values, &arr[0], size);
        }
        else if (ii == 14) {
            // Lots of zeros
            size = 1024;
            byte arr[1024] = { byte(0) };

            for (int i = 0; i < size; i++) {
                int val = rand() % 100;

                if (val >= 33)
                    val = 0;

                arr[i] = byte(val);
            }

            memcpy(values, &arr[0], size);
        }
        else if (ii == 15) {
            // Lots of zeros
            size = 2048;
            byte arr[2048] = { byte(0) };

            for (int i = 0; i < size; i++) {
                int val = rand() % 100;

                if (val >= 33)
                    val = 0;

                arr[i] = byte(val);
            }

            memcpy(values, &arr[0], size);
        }
        else if (ii == 16) {
            // Totally random
            size = 512;
            byte arr[512] = { byte(0) };

            // Leave zeros at the beginning for ZRLT to succeed
            for (int j = 20; j < 512; j++)
                arr[j] = byte(rand() % mod);

            memcpy(values, &arr[0], size);
        }
        else if (ii < 25) {
            size = 2048;
            byte arr[2048] = { byte(0) };
            const int step = max(ii - 5, 2);
            arr[60] = byte(rand() % mod);
            arr[61] = byte(rand() % mod);
            arr[62] = byte(rand() % mod);
            arr[63] = byte(rand() % mod);

            // Simulate interleaved channels for MM
            for (int j = 64; j + step < size; j += step) {
                for (int k = 0; k < step; k++)
                   arr[j + k] = arr[j + k - step];
            }

            memcpy(values, &arr[0], size);
        }
        else if (ii == 50) {
            cout << "Large random data" << endl;
            size = 1024 * 1024;
            byte* arr = new byte[size];

            for (int i = 0; i < size; i++)
                arr[i] = byte(rand() % 256);

            memcpy(values, arr, size);
            delete[] arr;
        }
        else {
            size = 1024;
            byte arr[1024] = { byte(0) };

            // Leave zeros at the beginning for ZRLT to succeed
            int idx = 20;

            while (idx < 1024) {
                int len = rand() % 120; // above LZP min match threshold

                if (len % 3 == 0)
                    len = 1;

                byte val = byte(rand() % mod);
                int end = (idx + len) < size ? idx + len : size;

                for (int j = idx; j < end; j++)
                    arr[j] = val;

                idx += len;
            }

            memcpy(values, &arr[0], size);
        }

        Context ctx;
        ctx.putInt("bsVersion", 6);
        ctx.putString("transform", name);
        Transform<byte>* ff = getByteTransform(name, ctx);

        if (ff == nullptr)
            return 1;

        Transform<byte>* fi = getByteTransform(name, ctx);

        if (fi == nullptr) {
            delete ff;
            return 1;
        }

        const int dstSize = ff->getMaxEncodedLength(size);
        byte* input = new byte[size];
        byte* output = new byte[dstSize];
        byte* reverse = new byte[size];

        SliceArray<byte> iba1(input, size, 0);
        SliceArray<byte> iba2(output, dstSize, 0);
        SliceArray<byte> iba3(reverse, size, 0);
        memset(output, 0xAA, dstSize);
        memset(reverse, 0xAA, size);
        int count;

        for (int i = 0; i < size; i++)
            input[i] = values[i];

        cout << endl
             << "Original: " << endl;

        if (ii == 11) {
            cout << "1 8 (" << (size - 1) << " times)";
        }
        else {
            if (size > 1024) {
                cout << "Large data block - not printing all values.";
            }
            else {
                for (int i = 0; i < size; i++)
                    cout << (int(input[i]) & 0xFF) << " ";
            }
        }

        if (ff->forward(iba1, iba2, size) == false) {
            if ((iba1._index != size) || (iba2._index >= iba1._index)) {
                cout << endl
                     << "No compression (ratio > 1.0), skip reverse" << endl;
                delete ff;
                delete fi;
                delete[] input;
                delete[] output;
                delete[] reverse;
                continue;
            }

            cout << endl
                 << "Encoding error" << endl;
            res = 1;
            ff = nullptr;
            goto End;
        }

        if (name != "MM") { // MM can expand
            if ((iba1._index != size) || (iba1._index < iba2._index)) {
                cout << endl
                     << "No compression (ratio > 1.0), skip reverse" << endl;
                delete ff;
                delete fi;
                delete[] input;
                delete[] output;
                delete[] reverse;
                continue;
            }
        }

        cout << endl;
        cout << "Coded: " << endl;

        if (iba2._index > 1024) {
            cout << "Large data block - not printing all values.";
        }
        else {
            for (int i = 0; i < iba2._index; i++)
                cout << (int(output[i]) & 0xFF) << " ";
        }

        cout << " (Compression ratio: " << (iba2._index * 100 / size) << "%)" << endl;
        count = iba2._index;
        iba1._index = 0;
        iba2._index = 0;
        iba3._index = 0;

        if (fi->inverse(iba2, iba3, count) == false) {
            cout << "Decoding error" << endl;
            res = 1;
            goto End;
        }

        cout << "Decoded: " << endl;

        if (ii == 11) {
            cout << "1 8 (" << (size - 1) << " times)";
        }
        else {
            if (size > 1024) {
                cout << "Large data block - not printing all values.";
            }
            else {
                for (int i = 0; i < size; i++)
                    cout << (int(reverse[i]) & 0xFF) << " ";
            }
        }

        cout << endl;

        for (int i = 0; i < size; i++) {
            if (input[i] != reverse[i]) {
                cout << "Different (index " << i << ": ";
                cout << (int(input[i]) & 0xFF) << " - " << (int(reverse[i]) & 0xFF);
                cout << ")" << endl;
                res = 1;
                goto End;
            }
        }

        cout << endl
             << "Identical" << endl
             << endl;

    End:
        if (ff != nullptr)
           delete ff;

        if (fi != nullptr)
           delete fi;

        delete[] input;
        delete[] output;
        delete[] reverse;
    }

    return res;
}

int testTransformsSpeed(const string& name)
{
    // Test speed
    srand((uint)time(nullptr));
    int iter = 50000;

    if ((name == "ROLZ") || (name == "SRT") || (name == "RANK") || (name == "MTFT"))
        iter = 4000;

    int size = 30000;
    int res = 0;

    cout << endl
         << endl
         << "Speed test for " << name << endl;
    cout << "Iterations: " << iter << endl;
    cout << endl;
    byte input[50000] = { byte(0) };
    byte output[50000] = { byte(0) };
    byte reverse[50000] = { byte(0) };
    Context ctx;
    Transform<byte>* f = getByteTransform(name, ctx);

    if (f == nullptr)
        return 1;

    SliceArray<byte> iba1(input, size, 0);
    SliceArray<byte> iba2(output, f->getMaxEncodedLength(size), 0);
    SliceArray<byte> iba3(reverse, size, 0);
    int mod = (name == "ZRLT") ? 5 : 256;
    delete f;

    for (int jj = 0; jj < 3; jj++) {
        // Generate random data with runs
        // Leave zeros at the beginning for ZRLT to succeed
        int n = iter / 20;

        if (name == "ALIAS")
            mod = 5 + 80 * jj;

        while (n < size) {
            byte val = byte(rand() % mod);
            input[n++] = val;
            int run = rand() % 256;
            run -= 220;

            while ((--run > 0) && (n < size))
                input[n++] = val;
        }

        clock_t before, after;
        double delta1 = 0;
        double delta2 = 0;

        for (int ii = 0; ii < iter; ii++) {
            Transform<byte>* ff = getByteTransform(name, ctx);
            iba1._index = 0;
            iba2._index = 0;
            before = clock();

            if (ff->forward(iba1, iba2, size) == false) {
                if ((iba1._index != size) || (iba2._index >= iba1._index)) {
                   cout << endl
                        << "No compression (ratio > 1.0), skip reverse" << endl;
                   continue;
                }

                cout << "Encoding error" << endl;
                delete ff;
                continue;
            }

            after = clock();
            delta1 += (after - before);
            delete ff;
        }

        int count = iba2._index;

        for (int ii = 0; ii < iter; ii++) {
            Transform<byte>* fi = getByteTransform(name, ctx);
            iba3._index = 0;
            iba2._index = 0;
            before = clock();

            if (fi->inverse(iba2, iba3, count) == false) {
                cout << "Decoding error" << endl;
                delete fi;
                return 1;
            }

            after = clock();
            delta2 += (after - before);
            delete fi;
        }

        int idx = -1;

        // Sanity check
        for (int i = 0; i < iba1._index; i++) {
            if (iba1._array[i] != iba3._array[i]) {
                idx = i;
                break;
            }
        }

        if (idx >= 0) {
            cout << "Failure at index " << idx << " (" << (int)iba1._array[idx];
            cout << "<->" << (int)iba3._array[idx] << ")" << endl;
            res = 1;
        }

        // MB = 1000 * 1000, MiB = 1024 * 1024
        double prod = double(iter) * double(size);
        double b2MiB = double(1) / double(1024 * 1024);
        double d1_sec = delta1 / CLOCKS_PER_SEC;
        double d2_sec = delta2 / CLOCKS_PER_SEC;
        cout << name << " encoding [ms]: " << (int)(d1_sec * 1000) << endl;
        cout << "Throughput [MiB/s]: " << (int)(prod * b2MiB / d1_sec) << endl;
        cout << name << " decoding [ms]: " << (int)(d2_sec * 1000) << endl;
        cout << "Throughput [MiB/s]: " << (int)(prod * b2MiB / d2_sec) << endl;
    }

    return res;
}

#ifdef __GNUG__
int main(int argc, const char* argv[])
#else
int TestTransforms_main(int argc, const char* argv[])
#endif
{
    int res = 0;

    try {
        vector<string> codecs;
        bool doPerf = true;

        if (argc == 1) {
#if __cplusplus < 201103L
            string allCodecs[13] = { "LZ", "LZX", "LZP", "ROLZ", "ROLZX", "RLT", "ZRLT", "RANK", "SRT", "NONE", "ALIAS", "MM", "MTFT" };

            for (int i = 0; i < 13; i++)
                codecs.push_back(allCodecs[i]);
#else
            codecs = { "LZ", "LZX", "LZP", "ROLZ", "ROLZX", "RLT", "ZRLT", "RANK", "SRT", "NONE", "ALIAS", "MM", "MTFT" };
#endif
        }
        else {
            string str = argv[1];
            transform(str.begin(), str.end(), str.begin(), ::toupper);

            if (str != "-TYPE=ALL") {
                codecs.push_back(str.substr(6));

                if (str.compare(0, 6, "-TYPE=") != 0) {
                     cout << "Missing transform type" << endl;
                     return 1;
                }
            } else {
#if __cplusplus < 201103L
                string allCodecs[13] = { "LZ", "LZX", "LZP", "ROLZ", "ROLZX", "RLT", "ZRLT", "RANK", "SRT", "NONE", "ALIAS", "MM", "MTFT" };

                for (int i = 0; i < 13; i++)
                    codecs.push_back(allCodecs[i]);
#else
                codecs = { "LZ", "LZX", "LZP", "ROLZ", "ROLZX", "RLT", "ZRLT", "RANK", "SRT", "NONE", "ALIAS", "MM", "MTFT" };
#endif
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
            res = testTransformsCorrectness(*it);

            if (res)
               break;

            if ((doPerf == true) && (*it != "LZP") && (*it != "MM")) { // skip codecs with no good data
               res = testTransformsSpeed(*it);

               if (res)
                  break;
            }
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


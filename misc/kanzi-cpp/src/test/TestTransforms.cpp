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
#include "../transform/AliasCodec.hpp"
#include "../transform/EXECodec.hpp"
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

static void writeInt16LE(kanzi::byte buf[], int value)
{
    buf[0] = kanzi::byte(value);
    buf[1] = kanzi::byte(value >> 8);
}

static void writeInt32LE(kanzi::byte buf[], int value)
{
    buf[0] = kanzi::byte(value);
    buf[1] = kanzi::byte(value >> 8);
    buf[2] = kanzi::byte(value >> 16);
    buf[3] = kanzi::byte(value >> 24);
}

static vector<kanzi::byte> createPEBlock(int arch)
{
    const int size = 8192;
    const int codeStart = 512;
    const int codeLen = 4096;
    const int posPE = 0x80;
    vector<kanzi::byte> data(size, kanzi::byte(0x90));
    data[0] = kanzi::byte('M');
    data[1] = kanzi::byte('Z');
    writeInt32LE(&data[60], posPE);
    data[posPE] = kanzi::byte('P');
    data[posPE + 1] = kanzi::byte('E');
    data[posPE + 2] = kanzi::byte(0);
    data[posPE + 3] = kanzi::byte(0);
    writeInt16LE(&data[posPE + 4], arch);
    writeInt32LE(&data[posPE + 28], codeLen);
    writeInt32LE(&data[posPE + 44], codeStart);
    return data;
}

static void setPECodeLength(vector<kanzi::byte>& data, int codeLen)
{
    writeInt32LE(&data[0x80 + 28], codeLen);
}

static vector<kanzi::byte> createELF64Block(int arch)
{
    const int size = 8192;
    const int codeStart = 512;
    const int codeLen = 4096;
    const int posSection = 0x100;
    vector<kanzi::byte> data(size, kanzi::byte(0));
    data[0] = kanzi::byte(0x7F);
    data[1] = kanzi::byte('E');
    data[2] = kanzi::byte('L');
    data[3] = kanzi::byte('F');
    data[4] = kanzi::byte(2);
    data[5] = kanzi::byte(1);
    writeInt16LE(&data[18], arch);
    writeInt16LE(&data[0x3A], 0x40);
    writeInt16LE(&data[0x3C], 1);
    writeInt32LE(&data[0x28], posSection);
    writeInt32LE(&data[posSection + 4], 1);
    writeInt32LE(&data[posSection + 0x18], codeStart);
    writeInt32LE(&data[posSection + 0x20], codeLen);
    return data;
}

static void fillX86Code(vector<kanzi::byte>& data, int codeStart, int codeLen)
{
    for (int i = codeStart; i + 5 <= codeStart + codeLen; i += 5) {
        data[i] = kanzi::byte(0xE8);
        data[i + 1] = kanzi::byte(0);
        data[i + 2] = kanzi::byte(0);
        data[i + 3] = kanzi::byte(0);
        data[i + 4] = kanzi::byte(0);
    }
}

static void fillARM64Code(vector<kanzi::byte>& data, int codeStart, int codeLen)
{
    for (int i = codeStart; i + 4 <= codeStart + codeLen; i += 4)
        writeInt32LE(&data[i], 0x14000000);
}

static void fillX86ExpandedCode(vector<kanzi::byte>& data, int codeStart, int codeLen)
{
    for (int i = codeStart; i + 8 <= codeStart + codeLen; i += 8) {
        const bool escaped = (((i - codeStart) >> 3) < 24);
        data[i] = kanzi::byte(0xE8);
        data[i + 1] = kanzi::byte(0);
        data[i + 2] = kanzi::byte(0);
        data[i + 3] = kanzi::byte(0);
        data[i + 4] = kanzi::byte(0);
        data[i + 5] = escaped ? kanzi::byte(0x9B) : kanzi::byte(0x90);
        data[i + 6] = kanzi::byte(0x90);
        data[i + 7] = kanzi::byte(0x90);
    }
}

static void addX86BoundaryJCC(vector<kanzi::byte>& data, int codeStart, int codeLen)
{
    const int idx = codeStart + codeLen - 5;
    data[idx] = kanzi::byte(0x0F);
    data[idx + 1] = kanzi::byte(0x85);
    data[idx + 2] = kanzi::byte(0);
    data[idx + 3] = kanzi::byte(0);
    data[idx + 4] = kanzi::byte(0);
    data[idx + 5] = kanzi::byte(0);
}

static vector<kanzi::byte> createX86BoundaryBlock()
{
    vector<kanzi::byte> data = createPEBlock(0x014C);
    const int codeStart = 512;
    const int codeLen = 85;
    setPECodeLength(data, codeLen);
    fillX86Code(data, codeStart, 16 * 5);
    addX86BoundaryJCC(data, codeStart, codeLen);
    return data;
}

static int testEXERoundTrip(const string& name, vector<kanzi::byte>& data)
{
    cout << endl
         << "Correctness for " << name << endl;
    Context ctx;
    EXECodec codec(ctx);
    vector<kanzi::byte> encoded(codec.getMaxEncodedLength(int(data.size())), kanzi::byte(0));
    vector<kanzi::byte> decoded(data.size(), kanzi::byte(0));
    SliceArray<kanzi::byte> input(&data[0], int(data.size()), 0);
    SliceArray<kanzi::byte> output(&encoded[0], int(encoded.size()), 0);
    SliceArray<kanzi::byte> reverse(&decoded[0], int(decoded.size()), 0);

    if (codec.forward(input, output, int(data.size())) == false) {
        cout << "Encoding error" << endl;
        return 1;
    }

    const int encodedSize = output._index;
    input._index = 0;
    output._index = 0;

    if (codec.inverse(output, reverse, encodedSize) == false) {
        cout << "Decoding error" << endl;
        return 1;
    }

    if ((reverse._index != int(data.size())) || (memcmp(&data[0], &decoded[0], data.size()) != 0)) {
        cout << "Round-trip mismatch" << endl;
        return 1;
    }

    vector<kanzi::byte> small(encodedSize - 10, kanzi::byte(0));
    SliceArray<kanzi::byte> tooSmall(&small[0], int(small.size()), 0);
    output._index = 0;

    if (codec.inverse(output, tooSmall, encodedSize) != false) {
        cout << "Undersized output buffer should fail" << endl;
        return 1;
    }

    cout << "Identical" << endl;
    return 0;
}

static int testEXECodec()
{
    vector<kanzi::byte> x86 = createPEBlock(0x014C);
    fillX86Code(x86, 512, 4096);

    if (testEXERoundTrip("EXE-X86", x86) != 0)
        return 1;

    vector<kanzi::byte> arm64 = createELF64Block(0x00B7);
    fillARM64Code(arm64, 512, 4096);

    if (testEXERoundTrip("EXE-ARM64", arm64) != 0)
        return 1;

    {
        cout << endl
             << "Correctness for EXE-X86-Expanded" << endl;
        Context ctx;
        EXECodec codec(ctx);
        vector<kanzi::byte> expanded = createPEBlock(0x014C);
        fillX86ExpandedCode(expanded, 512, 4096);
        vector<kanzi::byte> encoded(codec.getMaxEncodedLength(int(expanded.size())), kanzi::byte(0));
        vector<kanzi::byte> decoded(expanded.size(), kanzi::byte(0));
        SliceArray<kanzi::byte> input(&expanded[0], int(expanded.size()), 0);
        SliceArray<kanzi::byte> output(&encoded[0], int(encoded.size()), 0);
        SliceArray<kanzi::byte> reverse(&decoded[0], int(decoded.size()), 0);

        if (codec.forward(input, output, int(expanded.size())) == false) {
            cout << "Encoding error" << endl;
            return 1;
        }

        if (output._index <= int(expanded.size()) + 9) {
            cout << "Expected encoded block expansion beyond header" << endl;
            return 1;
        }

        const int encodedSize = output._index;
        output._index = 0;
        reverse._index = 0;

        if (codec.inverse(output, reverse, encodedSize) == false) {
            cout << "Decoding error" << endl;
            return 1;
        }

        if ((reverse._index != int(expanded.size())) ||
            (memcmp(&expanded[0], &decoded[0], expanded.size()) != 0)) {
            cout << "Round-trip mismatch" << endl;
            return 1;
        }

        cout << "Identical" << endl;
    }

    vector<kanzi::byte> boundary = createX86BoundaryBlock();

    if (testEXERoundTrip("EXE-X86-Boundary-JCC", boundary) != 0)
        return 1;

    {
        cout << endl
             << "Correctness for EXE-X86-Legacy-Boundary-JCC" << endl;
        Context ctx;
        EXECodec codec(ctx);
        vector<kanzi::byte> legacy = createX86BoundaryBlock();
        vector<kanzi::byte> encoded(codec.getMaxEncodedLength(int(legacy.size())), kanzi::byte(0));
        vector<kanzi::byte> decoded(legacy.size(), kanzi::byte(0));
        SliceArray<kanzi::byte> input(&legacy[0], int(legacy.size()), 0);
        SliceArray<kanzi::byte> output(&encoded[0], int(encoded.size()), 0);
        SliceArray<kanzi::byte> reverse(&decoded[0], int(decoded.size()), 0);

        if (codec.forward(input, output, int(legacy.size())) == false) {
            cout << "Encoding error" << endl;
            return 1;
        }

        const int encodedSize = output._index;
        const int codeEnd = LittleEndian::readInt32(&encoded[5]);

        if ((codeEnd >= encodedSize) || (encoded[codeEnd] != kanzi::byte(0x0F))) {
            cout << "Unexpected boundary layout" << endl;
            return 1;
        }

        writeInt32LE(&encoded[5], codeEnd + 1);
        output._index = 0;
        reverse._index = 0;

        if (codec.inverse(output, reverse, encodedSize) == false) {
            cout << "Decoding error" << endl;
            return 1;
        }

        if ((reverse._index != int(legacy.size())) ||
            (memcmp(&legacy[0], &decoded[0], legacy.size()) != 0)) {
            cout << "Round-trip mismatch" << endl;
            return 1;
        }

        cout << "Identical" << endl;
    }

    return 0;
}

static int testZRLTMalformed()
{
    cout << endl
         << "Malformed ZRLT" << endl;
    Context ctx;
    ZRLT codec(ctx);

    {
        kanzi::byte encoded[1] = { kanzi::byte(2) };
        kanzi::byte decoded[5];
        memset(decoded, 0x7E, sizeof(decoded));
        SliceArray<kanzi::byte> input(encoded, 1, 0);
        SliceArray<kanzi::byte> output(decoded, 4, 3);

        if (codec.inverse(input, output, 1) == false) {
            cout << "Valid offset decode failed" << endl;
            return 1;
        }

        if ((output._index != 4) || (decoded[3] != kanzi::byte(1)) ||
            (decoded[4] != kanzi::byte(0x7E))) {
            cout << "Valid offset decode corrupted output" << endl;
            return 1;
        }
    }

    {
        kanzi::byte encoded[2] = { kanzi::byte(2), kanzi::byte(2) };
        kanzi::byte decoded[5];
        memset(decoded, 0x7E, sizeof(decoded));
        SliceArray<kanzi::byte> input(encoded, 2, 0);
        SliceArray<kanzi::byte> output(decoded, 4, 3);

        if (codec.inverse(input, output, 2) != false) {
            cout << "Oversized offset decode should fail" << endl;
            return 1;
        }

        if (decoded[4] != kanzi::byte(0x7E)) {
            cout << "Oversized offset decode wrote past logical output" << endl;
            return 1;
        }
    }

    {
        kanzi::byte encoded[1] = { kanzi::byte(2) };
        kanzi::byte decoded[1] = { kanzi::byte(0) };
        SliceArray<kanzi::byte> input(encoded, 1, 0);
        SliceArray<kanzi::byte> output(decoded, 1, 0);

        if (codec.inverse(input, output, 2) != false) {
            cout << "Oversized encoded length should fail" << endl;
            return 1;
        }
    }

    {
        kanzi::byte encoded[1] = { kanzi::byte(0xFF) };
        kanzi::byte decoded[1] = { kanzi::byte(0x7E) };
        SliceArray<kanzi::byte> input(encoded, 1, 0);
        SliceArray<kanzi::byte> output(decoded, 1, 0);

        if (codec.inverse(input, output, 1) != false) {
            cout << "Truncated escape should fail" << endl;
            return 1;
        }

        if (decoded[0] != kanzi::byte(0x7E)) {
            cout << "Truncated escape wrote output" << endl;
            return 1;
        }
    }

    {
        kanzi::byte encoded[1] = { kanzi::byte(0) };
        kanzi::byte decoded[1] = { kanzi::byte(0x7E) };
        SliceArray<kanzi::byte> input(encoded, 1, 0);
        SliceArray<kanzi::byte> output(decoded, 0, 0);

        if (codec.inverse(input, output, 1) != false) {
            cout << "Oversized zero run should fail" << endl;
            return 1;
        }

        if (decoded[0] != kanzi::byte(0x7E)) {
            cout << "Oversized zero run wrote output" << endl;
            return 1;
        }
    }

    cout << "Malformed ZRLT tests passed" << endl;
    return 0;
}

static Transform<kanzi::byte>* getByteTransform(string name, Context& ctx)
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
        ctx.putInt("lz", TransformFactory<kanzi::byte>::LZX_TYPE);
        return new LZCodec(ctx);
    }

    if (name.compare("LZP") == 0){
        ctx.putInt("lz", TransformFactory<kanzi::byte>::LZP_TYPE);
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
        int size = 80000; // Declare size, will be updated in conditions
        kanzi::byte values[1024 * 1024] = { kanzi::byte(0xAA) };

        if (name == "ALIAS")
          mod = 15 + 12 * ii;

        if (ii == 0) {
            size = 32;
            kanzi::byte arr[32] = {
                (kanzi::byte)0, (kanzi::byte)1, (kanzi::byte)2, (kanzi::byte)2, (kanzi::byte)2, (kanzi::byte)2, (kanzi::byte)7, (kanzi::byte)9,
                (kanzi::byte)9, (kanzi::byte)16, (kanzi::byte)16, (kanzi::byte)16, (kanzi::byte)1, (kanzi::byte)3, (kanzi::byte)3, (kanzi::byte)3,
                (kanzi::byte)3, (kanzi::byte)3, (kanzi::byte)3, (kanzi::byte)3, (kanzi::byte)3, (kanzi::byte)3, (kanzi::byte)3, (kanzi::byte)3,
                (kanzi::byte)3, (kanzi::byte)3, (kanzi::byte)3, (kanzi::byte)3, (kanzi::byte)3, (kanzi::byte)3, (kanzi::byte)3, (kanzi::byte)3
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
            values[127] = kanzi::byte(255);
        }
        else if (ii == 11) {
            size = 80000;
            kanzi::byte arr[80000];
            arr[0] = kanzi::byte(1);

            for (int i = 1; i < 80000; i++)
                arr[i] = kanzi::byte(8);

            memcpy(values, &arr[0], size);
        }
        else if (ii == 12) {
            size = 8;
            kanzi::byte arr[8] = { (kanzi::byte)0, (kanzi::byte)0, (kanzi::byte)1, (kanzi::byte)1, (kanzi::byte)2, (kanzi::byte)2, (kanzi::byte)3, (kanzi::byte)3 };
            memcpy(values, &arr[0], size);
        }
        else if (ii == 13) {
            // For RLT
            size = 512;
            kanzi::byte arr[512];

            for (int i = 0; i < 256; i++) {
                arr[2 * i] = kanzi::byte(i);
                arr[2 * i + 1] = kanzi::byte(i);
            }

            arr[1] = kanzi::byte(255); // force RLT escape to be first symbol
            memcpy(values, &arr[0], size);
        }
        else if (ii == 14) {
            // Lots of zeros
            size = 1024;
            kanzi::byte arr[1024] = { kanzi::byte(0) };

            for (int i = 0; i < size; i++) {
                int val = rand() % 100;

                if (val >= 33)
                    val = 0;

                arr[i] = kanzi::byte(val);
            }

            memcpy(values, &arr[0], size);
        }
        else if (ii == 15) {
            // Lots of zeros
            size = 2048;
            kanzi::byte arr[2048] = { kanzi::byte(0) };

            for (int i = 0; i < size; i++) {
                int val = rand() % 100;

                if (val >= 33)
                    val = 0;

                arr[i] = kanzi::byte(val);
            }

            memcpy(values, &arr[0], size);
        }
        else if (ii == 16) {
            // Totally random
            size = 512;
            kanzi::byte arr[512] = { kanzi::byte(0) };

            // Leave zeros at the beginning for ZRLT to succeed
            for (int j = 20; j < 512; j++)
                arr[j] = kanzi::byte(rand() % mod);

            memcpy(values, &arr[0], size);
        }
        else if (ii < 25) {
            size = 2048;
            kanzi::byte arr[2048] = { kanzi::byte(0) };
            const int step = max(ii - 5, 2);
            arr[60] = kanzi::byte(rand() % mod);
            arr[61] = kanzi::byte(rand() % mod);
            arr[62] = kanzi::byte(rand() % mod);
            arr[63] = kanzi::byte(rand() % mod);

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
            kanzi::byte* arr = new kanzi::byte[size];

            for (int i = 0; i < size; i++)
                arr[i] = kanzi::byte(rand() % 256);

            memcpy(values, arr, size);
            delete[] arr;
        }
        else {
            size = 1024;
            kanzi::byte arr[1024] = { kanzi::byte(0) };

            // Leave zeros at the beginning for ZRLT to succeed
            int idx = 20;

            while (idx < 1024) {
                int len = rand() % 120; // above LZP min match threshold

                if (len % 3 == 0)
                    len = 1;

                kanzi::byte val = kanzi::byte(rand() % mod);
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
        Transform<kanzi::byte>* ff = getByteTransform(name, ctx);

        if (ff == nullptr)
            return 1;

        Transform<kanzi::byte>* fi = getByteTransform(name, ctx);

        if (fi == nullptr) {
            delete ff;
            return 1;
        }

        const int dstSize = ff->getMaxEncodedLength(size);
        kanzi::byte* input = new kanzi::byte[size];
        kanzi::byte* output = new kanzi::byte[dstSize];
        kanzi::byte* reverse = new kanzi::byte[size];

        SliceArray<kanzi::byte> iba1(input, size, 0);
        SliceArray<kanzi::byte> iba2(output, dstSize, 0);
        SliceArray<kanzi::byte> iba3(reverse, size, 0);
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
    kanzi::byte input[50000] = { kanzi::byte(0) };
    kanzi::byte output[50000] = { kanzi::byte(0) };
    kanzi::byte reverse[50000] = { kanzi::byte(0) };
    Context ctx;
    Transform<kanzi::byte>* f = getByteTransform(name, ctx);

    if (f == nullptr)
        return 1;

    SliceArray<kanzi::byte> iba1(input, size, 0);
    SliceArray<kanzi::byte> iba2(output, f->getMaxEncodedLength(size), 0);
    SliceArray<kanzi::byte> iba3(reverse, size, 0);
    int mod = (name == "ZRLT") ? 5 : 256;
    delete f;

    for (int jj = 0; jj < 3; jj++) {
        // Generate random data with runs
        // Leave zeros at the beginning for ZRLT to succeed
        int n = iter / 20;

        if (name == "ALIAS")
            mod = 5 + 80 * jj;

        while (n < size) {
            kanzi::byte val = kanzi::byte(rand() % mod);
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
            Transform<kanzi::byte>* ff = getByteTransform(name, ctx);
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
            Transform<kanzi::byte>* fi = getByteTransform(name, ctx);
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
        res = testEXECodec();

        if (res != 0)
            return res;

        res = testZRLTMalformed();

        if (res != 0)
            return res;

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
    catch (const exception& e) {
        cout << e.what() << endl;
        res = 123;
    }

    cout << endl;
    cout << ((res == 0) ? "Success" : "Failure") << endl;
    return res;
}

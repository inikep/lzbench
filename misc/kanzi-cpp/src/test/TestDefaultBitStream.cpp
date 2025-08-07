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
#include <cstdio>
#include <iostream>
#include <fstream>
#include <streambuf>
#include <time.h>
#include "../bitstream/DebugOutputBitStream.hpp"
#include "../bitstream/DefaultInputBitStream.hpp"
#include "../bitstream/DefaultOutputBitStream.hpp"
#include "../BitStreamException.hpp"
#include "../util.hpp"
#include "../io/IOException.hpp"

using namespace std;
using namespace kanzi;

int testBitStreamCorrectnessAligned1()
{
    // Test correctness (byte aligned)
    cout << "Correctness Test - write long - byte aligned" << endl;
    const int length = 100;
    int* values = new int[length];
    int res = 0;
    srand((uint)time(nullptr));
    cout << "\nInitial" << endl;

    // Check correctness of read() and written()
    for (int t = 1; t <= 32; t++) {
        stringbuf buffer;
        iostream ios(&buffer);
        DefaultOutputBitStream obs(ios, 16384);
        cout << endl;
        obs.writeBits(0x01234567L, t);
        cout << "Written (before close): " << obs.written() << endl;
        obs.close();
        cout << "Written (after close): " << obs.written() << endl;
        ios.rdbuf()->pubseekpos(0);
        DefaultInputBitStream ibs(ios, 16384);
        ibs.readBits(t);

        cout << ((ibs.read() == uint64(t)) ? "OK" : "KO") << endl;
        cout << "Read (before close): " << ibs.read() << endl;
        ibs.close();
        cout << "Read (after close): " << ibs.read() << endl;
    }

    for (int test = 1; test <= 10; test++) {
        stringbuf buffer;
        iostream ios(&buffer);
        DefaultOutputBitStream obs(ios, 16384);
        DebugOutputBitStream dbs(obs, cout);
        dbs.showByte(true);

        for (int i = 0; i < length; i++) {
            values[i] = rand();
            cout << values[i] << " ";

            if ((i % 20) == 19)
                cout << endl;
        }

        cout << endl
             << endl;

        for (int i = 0; i < length; i++) {
            dbs.writeBits(values[i], 32);
        }

        // Close first to force flush()
        dbs.close();
        ios.rdbuf()->pubseekpos(0);
        istringstream is;
        char* cvalues = new char[4 * length];

        for (int i = 0; i < length; i++) {
            cvalues[4 * i] = (values[i] >> 24) & 0xFF;
            cvalues[4 * i + 1] = (values[i] >> 16) & 0xFF;
            cvalues[4 * i + 2] = (values[i] >> 8) & 0xFF;
            cvalues[4 * i + 3] = values[i] & 0xFF;
        }

        is.read(cvalues, length);

        DefaultInputBitStream ibs(ios, 16384);
        cout << endl
             << endl
             << "Read:" << endl;
        bool ok = true;

        for (int i = 0; i < length; i++) {
            int x = (int)ibs.readBits(32);
            cout << x;
            cout << ((x == values[i]) ? " " : "* ");
            ok &= (x == values[i]);

            if ((i % 20) == 19)
                cout << endl;
        }

        delete[] cvalues;
        ibs.close();
        res = (ok == true) ? 0 : 1;
        cout << endl;
        cout << endl
             << "Bits written: " << dbs.written() << endl;
        cout << endl
             << "Bits read: " << ibs.read() << endl;
        cout << endl
             << "\n" << (ok ? "Success" : "Failure") << endl;
        cout << endl;
        cout << endl;
    }

    delete[] values;
    return res;
}

int testBitStreamCorrectnessMisaligned1()
{
    // Test correctness (not byte aligned)
    cout << "Correctness Test - write long - not byte aligned" << endl;
    const int length = 100;
    int* values = new int[length];
    int res = 0;
    srand((uint)time(nullptr));
    cout << "\nInitial" << endl;

    // Check correctness of read() and written()
    for (int t = 1; t <= 32; t++) {
        stringbuf buffer;
        iostream ios(&buffer);
        DefaultOutputBitStream obs(ios, 16384);
        cout << endl;
        obs.writeBit(1);
        obs.writeBits(0x01234567L, t);
        cout << "Written (before close): " << obs.written() << endl;
        obs.close();
        cout << "Written (after close): " << obs.written() << endl;
        ios.rdbuf()->pubseekpos(0);
        DefaultInputBitStream ibs(ios, 16384);
        ibs.readBit();
        ibs.readBits(t);

        cout << ((ibs.read() == uint64(t + 1)) ? "OK" : "KO") << endl;
        cout << "Read (before close): " << ibs.read() << endl;
        ibs.close();
        cout << "Read (after close): " << ibs.read() << endl;
    }

    for (int test = 1; test <= 10; test++) {
        stringbuf buffer;
        iostream ios(&buffer);
        DefaultOutputBitStream obs(ios, 16384);
        DebugOutputBitStream dbs(obs, cout);
        dbs.showByte(true);

        for (int i = 0; i < length; i++) {
            values[i] = rand();
            const int mask = (1 << (1 + (i % 30))) - 1;
            values[i] &= mask;
            cout << values[i] << " ";

            if ((i % 20) == 19)
                cout << endl;
        }

        cout << endl
             << endl;

        for (int i = 0; i < length; i++) {
            dbs.writeBits(values[i], 1 + (i % 30));
        }

        // Close first to force flush()
        dbs.close();

        if (test == 10) {
            try {
                cout << "\nTrying to write to closed stream" << endl;
                dbs.writeBit(1);
            } catch (BitStreamException& e) {
                cout << "\nException: " << e.what() << endl;
            }
        }

        ios.rdbuf()->pubseekpos(0);
        istringstream is;
        char* cvalues = new char[4 * length];

        for (int i = 0; i < length; i++) {
            cvalues[4 * i] = (values[i] >> 24) & 0xFF;
            cvalues[4 * i + 1] = (values[i] >> 16) & 0xFF;
            cvalues[4 * i + 2] = (values[i] >> 8) & 0xFF;
            cvalues[4 * i + 3] = (values[i] >> 0) & 0xFF;
        }

        is.read(cvalues, length);

        DefaultInputBitStream ibs(ios, 16384);
        cout << endl
             << endl
             << "Read:" << endl;
        bool ok = true;

        for (int i = 0; i < length; i++) {
            int x = (int)ibs.readBits((1 + (i % 30)));
            cout << x;
            cout << ((x == values[i]) ? " " : "* ");
            ok &= (x == values[i]);

            if ((i % 20) == 19)
                cout << endl;
        }

        delete[] cvalues;
        ibs.close();
        res = (ok == true) ? 0 : 1;
        cout << endl;
        cout << endl
             << "Bits written: " << dbs.written() << endl;
        cout << endl
             << "Bits read: " << ibs.read() << endl;
        cout << endl
             << "\n" << (ok ? "Success" : "Failure") << endl;
        cout << endl;
        cout << endl;

        if (test == 10) {
            try {
                cout << "\nTrying to read from closed stream" << endl;
                ibs.readBit();
            } catch (BitStreamException& e) {
                cout << "\nException: " << e.what() << endl;
            }
        }
    }

    delete[] values;
    return res;
}

int testBitStreamSpeed1(const string& fileName)
{
    // Test speed
    cout << "\nSpeed Test1" << endl;

    int values[] = { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3,
        31, 14, 41, 15, 59, 92, 26, 65, 53, 35, 58, 89, 97, 79, 93, 32 };

    int iter = 150;
    uint64 written = 0;
    uint64 read = 0;
    double delta1 = 0, delta2 = 0;
    int nn = 100000 * 32;

    for (int test = 1; test <= iter; test++) {
        ofstream os(fileName.c_str(), std::ofstream::binary);
        DefaultOutputBitStream obs(os, 1024 * 1024);
        clock_t before = clock();

        for (int i = 0; i < nn; i++) {
            obs.writeBits((uint64)values[i % 32], 1 + (i & 63));
        }

        // Close first to force flush()
        obs.close();
        os.close();
        clock_t after = clock();
        delta1 += (after - before);
        written += obs.written();

        ifstream is(fileName.c_str(), std::ifstream::binary);
        DefaultInputBitStream ibs(is, 1024 * 1024);
        before = clock();

        for (int i = 0; i < nn; i++) {
            ibs.readBits(1 + (i & 63));
        }

        ibs.close();
        is.close();
        after = clock();
        delta2 += (after - before);
        read += ibs.read();
    }

    // MB = 1000 * 1000, MiB = 1024 * 1024
    double d = 8.0 * 1024.0 * 1024.0;
    cout << written << " bits written (" << (written / 1024 / 1024 / 8) << " MB)" << endl;
    cout << read << " bits read (" << (read / 1024 / 1024 / 8) << " MB)" << endl;
    cout << endl;
    cout << "Write [ms]         : " << (int)(delta1 / CLOCKS_PER_SEC * 1000) << endl;
    cout << "Throughput [MiB/s] : " << (int)((double)written / d / (delta1 / CLOCKS_PER_SEC)) << endl;
    cout << "Read [ms]          : " << (int)(delta2 / CLOCKS_PER_SEC * 1000) << endl;
    cout << "Throughput [MiB/s] : " << (int)((double)read / d / (delta2 / CLOCKS_PER_SEC)) << endl;
    return 0;
}

int testBitStreamCorrectnessAligned2()
{
    // Test correctness (byte aligned)
    cout << "Correctness Test - write array - byte aligned" << endl;
    const int length = 100;
    byte* input = new byte[length];
    byte* output = new byte[length];
    int res = 0;
    srand((uint)time(nullptr));
    cout << "\nInitial" << endl;

    for (int test = 1; test <= 10; test++) {
        stringbuf buffer;
        iostream ios(&buffer);
        DefaultOutputBitStream obs(ios, 16384);
        DebugOutputBitStream dbs(obs, cout);
        dbs.showByte(true);

        for (int i = 0; i < length; i++) {
            input[i] = (byte) rand();
            cout << (int(input[i]) & 0xFF) << " ";

            if ((i % 20) == 19)
                cout << endl;
        }

        cout << endl
             << endl;

        uint count = 8 + test*(20+(test&1)) + (test&3);
        dbs.writeBits(input, count);
        cout << obs.written() << endl;

        // Close first to force flush()
        dbs.close();

        if (test == 10) {
            try {
                cout << "\nTrying to write to closed stream" << endl;
                dbs.writeBit(1);
            } catch (BitStreamException& e) {
                cout << "\nException: " << e.what() << endl;
            }
        }

        ios.rdbuf()->pubseekpos(0);
        istringstream is;
        char* cvalues = new char[length];

        for (int i = 0; i < length; i++) {
            cvalues[i] = char(input[i]) & 0xFF;
        }

        is.read(cvalues, length);

        DefaultInputBitStream ibs(ios, 16384);
        cout << endl
             << endl
             << "Read:" << endl;

        uint r = ibs.readBits(output, count);
        bool ok = r == count;

        if (ok == true) {
           for (uint i = 0; i < (r>>3); i++) {
               cout << (int(output[i]) & 0xFF);
               cout << ((output[i] == input[i]) ? " " : "* ");
               ok &= (output[i] == input[i]);

               if ((i % 20) == 19)
                   cout << endl;
           }
        }

        delete[] cvalues;
        ibs.close();
        res = (ok == true) ? 0 : 1;
        cout << endl;
        cout << endl
             << "Bits written: " << dbs.written() << endl;
        cout << endl
             << "Bits read: " << ibs.read() << endl;
        cout << endl
             << "\n" << (ok ? "Success" : "Failure") << endl;
        cout << endl;
        cout << endl;

        if (test == 10) {
            try {
                cout << "\nTrying to read from closed stream" << endl;
                ibs.readBit();
            } catch (BitStreamException& e) {
                cout << "\nException: " << e.what() << endl;
            }
        }
    }

    delete[] input;
    delete[] output;
    return res;
}

int testBitStreamCorrectnessMisaligned2()
{
    // Test correctness (not byte aligned)
    cout << "Correctness Test - write array - not byte aligned" << endl;
    const int length = 100;
    byte* input = new byte[length];
    byte* output = new byte[length];
    int res = 0;
    srand((uint)time(nullptr));
    cout << "\nInitial" << endl;

    for (int test = 1; test <= 10; test++) {
        stringbuf buffer;
        iostream ios(&buffer);
        DefaultOutputBitStream obs(ios, 16384);
        DebugOutputBitStream dbs(obs, cout);
        dbs.showByte(true);

        for (int i = 0; i < length; i++) {
            input[i] = (byte) rand();
            cout << (int(input[i]) & 0xFF) << " ";

            if ((i % 20) == 19)
                cout << endl;
        }

        cout << endl
             << endl;

        uint count = 8 + test*(20+(test&1)) + (test&3);
        dbs.writeBit(0);
        dbs.writeBits(&input[1], count);

        // Close first to force flush()
        dbs.close();
        ios.rdbuf()->pubseekpos(0);
        istringstream is;
        char* cvalues = new char[4 * length];

        for (int i = 0; i < length; i++) {
            cvalues[i] = char(input[i]) & 0xFF;
        }

        is.read(cvalues, length);

        DefaultInputBitStream ibs(ios, 16384);
        cout << endl
             << endl
             << "Read:" << endl;

        ibs.readBit();
        uint r = ibs.readBits(&output[1], count);
        bool ok = r == count;

        if (ok == true) {
           for (uint i = 1; i < 1 + (r>>3); i++) {
               cout << (int(output[i]) & 0xFF);
               cout << ((output[i] == input[i]) ? " " : "* ");
               ok &= (output[i] == input[i]);

               if ((i % 20) == 19)
                   cout << endl;
           }
        }

        delete[] cvalues;
        ibs.close();
        res = (ok == true) ? 0 : 1;
        cout << endl;
        cout << endl
             << "Bits written: " << dbs.written() << endl;
        cout << endl
             << "Bits read: " << ibs.read() << endl;
        cout << endl
             << "\n" << (ok ? "Success" : "Failure") << endl;
        cout << endl;
        cout << endl;
    }

    delete[] input;
    delete[] output;
    return res;
}

int testSeek(const string& name)
{
#if !defined(_MSC_VER) || _MSC_VER > 1500
    // Test correctness (not byte aligned)
    cout << endl << "Seek Test" << endl << endl;
    byte input[256];
    byte output[256];

    for (int i = 0; i < 256; i++)
       input[i] = byte(i);

    cout << "Test OutputBitStream" << endl;
    ofstream ofs(name.c_str(), ios_base::out | ios_base::binary);
    DefaultOutputBitStream obs(ofs);

    for (int i = 0; i < 128; i++)
       obs.writeBits(uint64(0xAA), 8);

    obs.seek(8 * 32);
    obs.writeBits(&input[10], 8 * 32);
    obs.seek(8 * 2);
    obs.writeBits(&input[100], 8 * 32);
    obs.close();
    ofs.close();
    cout << "Bits written: " << obs.written() << endl;
    remove(name.c_str());

    cout << endl;
    cout << "Test InputBitStream" << endl;

    for (int i = 0; i < 256; i++)
        input[i] = byte(i);

    ofstream ofs2(name.c_str(), ios_base::out | ios_base::binary);
    ofs2.write(reinterpret_cast<const char*>(input), 256);
    ofs2.close();
    ifstream ifs(name.c_str(), ios_base::in | ios_base::binary);
    DefaultInputBitStream ibs(ifs);
    memset(output, 0, 256);
    ibs.readBits(&output[0], 8 * 16);

    for (int i = 0; i < 16; i++) {
       if (output[i] != byte(i)) {
          cout << "Read failure" << endl;
          remove(name.c_str());
          return 1;
       }
    }

    // Positions in bytes
    int64 positions[5] = { 50, 0, 20, 33, 0 };
    memset(output, 0, 256);

    for (int i = 0; i < 5; i ++) {
       int64 pos = positions[i];
       cout << "Seek " << pos << endl;
       ibs.seek(8 * pos);

       if (ibs.tell() != 8 * pos) {
          cout << "Seek/tell mismatch" << endl;
          remove(name.c_str());
          return 2;
       }

       if (ibs.tell() != 8 * ifs.tellg()) {
          cout << "Seek/tell mismatch" << endl;
          remove(name.c_str());
          return 3;
       }

       cout << "Read bits at position " << pos << endl;
       ibs.readBits(&output[pos], 8 * 10);
       int64 r = ibs.readBits(8);

       if (r != pos + 10) {
          cout << "Incorrect number of read bits" << endl;
          remove(name.c_str());
          return 4;
       }

       for (int j = 0; j < 10; j++) {
          if (output[pos + j] != byte(pos + j)) {
             cout << "Read failure" << endl;
             remove(name.c_str());
             return 5;
          }
       }

       cout << "OK" << endl;
    }

    cout << "Bits read: " << ibs.read() << endl;
    remove(name.c_str());
    cout << endl << "Success" << endl;
#endif
    return 0;
}

int testBitStreamSpeed2(const string& fileName)
{
    // Test speed
    cout << "\nSpeed Test2" << endl;

    byte values[] = { (byte)3, (byte)1, (byte)4, (byte)1, (byte)5,(byte) 9, (byte)2, (byte)6,
        (byte)5, (byte)3, (byte)5, (byte)8, (byte)9, (byte)7, (byte)9, (byte)3,
        (byte)31, (byte)14, (byte)41, (byte)15, (byte)59, (byte)92, (byte)26, (byte)65,
        (byte)53, (byte)35, (byte)58, (byte)89, (byte)97, (byte)79, (byte)93, (byte)32 };

    int iter = 150;
    uint64 written = 0;
    uint64 read = 0;
    double delta1 = 0, delta2 = 0;
    byte* input = new byte[3250000*32];
    byte* output = new byte[3250000*32];

    for (int i = 0; i < 3250000; i++) {
        memcpy(&input[i*32], &values[0], 32);
    }

    for (int test = 1; test <= iter; test++) {
        ofstream os(fileName.c_str(), std::ofstream::binary);
        DefaultOutputBitStream obs(os, 1024 * 1024);
        clock_t before = clock();

        obs.writeBits(input, 3250000*32);

        // Close first to force flush()
        obs.close();
        os.close();
        clock_t after = clock();
        delta1 += (after - before);
        written += obs.written();

        ifstream is(fileName.c_str(), std::ifstream::binary);
        DefaultInputBitStream ibs(is, 1024 * 1024);
        before = clock();

        ibs.readBits(output, 3250000*32);

        ibs.close();
        is.close();
        after = clock();
        delta2 += (after - before);
        read += ibs.read();
    }

    // MiB = 1024 * 1024, MB = 1000 * 1000
    double d = 1024.0 * 8192.0;
    cout << written << " bits written (" << (written / 1024 / 1024 / 8) << " MiB)" << endl;
    cout << read << " bits read (" << (read / 1024 / 1024 / 8) << " MiB)" << endl;
    cout << endl;
    cout << "Write [ms]         : " << (int)(delta1 / CLOCKS_PER_SEC * 1000) << endl;
    cout << "Throughput [MiB/s] : " << (int)((double)written / d / (delta1 / CLOCKS_PER_SEC)) << endl;
    cout << "Read [ms]          : " << (int)(delta2 / CLOCKS_PER_SEC * 1000) << endl;
    cout << "Throughput [MiB/s] : " << (int)((double)read / d / (delta2 / CLOCKS_PER_SEC)) << endl;

    delete[] input;
    delete[] output;
    return 0;
}


#ifdef __GNUG__
int main(int argc, const char* argv[])
#else
int TestDefaultBitStream_main(int argc, const char* argv[])
#endif
{
    bool doPerf = true;

    if (argc <= 1) {
        cout << "Missing temp output file" << endl;
        exit(1);
    }

    if (argc > 2) {
        string str = argv[2];
        transform(str.begin(), str.end(), str.begin(), ::toupper);
        doPerf = str != "-NOPERF";
    }

    int res = 0;

    try {
       string fileName = argv[1];
       res |= testBitStreamCorrectnessAligned1();
       res |= testBitStreamCorrectnessAligned2();
       res |= testBitStreamCorrectnessMisaligned1();
       res |= testBitStreamCorrectnessMisaligned2();
       res |= testSeek(fileName);

       if (doPerf == true) {
          res |= testBitStreamSpeed1(fileName);
          res |= testBitStreamSpeed2(fileName);
       }
    } catch (kanzi::IOException& e) {
       cout << "Exception: " << e.what() << endl;
       res = 99;
    } catch (BitStreamException& e) {
       cout << "Exception: " << e.what() << endl;
       res = 99;
    }

    return res;
}


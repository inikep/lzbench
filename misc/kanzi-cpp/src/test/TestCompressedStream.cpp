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

#include <cstdio>
#include <fstream>
#include <iostream>
#include "../io/CompressedInputStream.hpp"
#include "../io/CompressedOutputStream.hpp"
#include "../io/IOException.hpp"

using namespace std;
using namespace kanzi;

uint64 compress1(byte block[], uint length)
{
    cout << "Test - Regular (RLT+TEXT&HUFFMAN)" << endl;
    uint blockSize = (length / (1 + (rand() & 3))) & -16;
    byte* buf = new byte[length];
    memcpy(&buf[0], &block[0], size_t(length));
    stringbuf buffer;
    iostream ios(&buffer);
    CompressedOutputStream* cos = new CompressedOutputStream(ios, 1, "HUFFMAN", "RLT+TEXT", blockSize);
    cos->write((const char*)block, length);
    cos->close();
    uint64 written = cos->getWritten();
    ios.seekg(0);
    memset(&block[0], 0, size_t(length));
    CompressedInputStream* cis = new CompressedInputStream(ios, 1);

    while (true) {
       cis->read((char*)block, length);

       if (cis->gcount() != length)
          break;
    }

    cis->close();
    uint64 read = cis->getRead();
    delete cos;
    delete cis;

    if (memcmp(&buf[0], &block[0], length) != 0)
       return 3;

    delete[] buf;
    return read ^ written;
}

uint64 compress2(byte block[], uint length)
{
    int jobs;
    srand((uint)time(nullptr));
    int checksum = 32 * min(rand() & 3, 2);
    uint blockSize = (length / (1 + (rand() & 3))) & -16;

#ifdef CONCURRENCY_ENABLED
    jobs = 1 + (rand() & 3);
    cout << "Test - " << jobs << " job(s) - ";
#else
    jobs = 1;
    cout << "Test - ";
#endif
    if (checksum == 0)
       cout << "no checksum";
    else
       cout << checksum << " bits checksum";

    cout << " (LZX&ANS0)" << endl;

    byte* buf = new byte[length];
    memcpy(&buf[0], &block[0], size_t(length));
    stringbuf buffer;
    iostream ios(&buffer);
    CompressedOutputStream* cos = new CompressedOutputStream(ios, jobs, "ANS0", "LZX", blockSize, checksum);
    cos->write((const char*)block, length);
    cos->close();
    uint64 written = cos->getWritten();
    ios.seekg(0);
    memset(&block[0], 0, size_t(length));
    CompressedInputStream* cis = new CompressedInputStream(ios, jobs);

    while (true) {
       cis->read((char*)block, length);

       if (cis->gcount() != length)
          break;
    }

    cis->close();
    uint64 read = cis->getRead();
    delete cos;
    delete cis;

    if (memcmp(&buf[0], &block[0], length) != 0)
       return 3;

    delete[] buf;
    return read ^ written;
}

uint64 compress3(byte block[], uint length)
{
    cout << "Test - incompressible data (LZP+RLT&FPAQ)" << endl;
    uint blockSize = (length / (1 + (rand() & 3))) & -16;
    byte* buf = new byte[length];
    memcpy(&buf[0], &block[0], size_t(length));
    stringbuf buffer;
    iostream ios(&buffer);
    CompressedOutputStream* cos = new CompressedOutputStream(ios, 1, "FPAQ", "LZP+ZRLT", blockSize, 32);
    cos->write((const char*)block, length);
    cos->close();
    uint64 written = cos->getWritten();
    ios.seekg(0);
    memset(&block[0], 0, size_t(length));
    CompressedInputStream* cis = new CompressedInputStream(ios, 1);

    while (true) {
       cis->read((char*)block, length);

       if (cis->gcount() != length)
          break;
    }

    cis->close();
    uint64 read = cis->getRead();
    delete cos;
    delete cis;

    if (memcmp(&buf[0], &block[0], length) != 0)
       return 3;

    delete[] buf;
    return read ^ written;
}

uint64 compress4(byte block[], uint length)
{
    CompressedOutputStream* cos = nullptr;
    uint64 res;

    try {
        cout << "Test - write after close (TEXT&HUFFMAN)" << endl;
        stringbuf buffer;
        iostream os(&buffer);
        cos = new CompressedOutputStream(os, 1, "HUFFMAN", "TEXT");
        cos->write((const char*)block, length);
        cos->close();
        cos->put(char(0));
        cout << "Failure: no exception thrown in write after close" << endl;
        res = 1;
    }
    catch (ios_base::failure& e) {
        cout << "OK, Exception " << e.what() << endl;
        res = 0;
    }

    delete cos;
    return res;
}

uint64 compress5(byte block[], uint length)
{
    CompressedOutputStream* cos = nullptr;
    CompressedInputStream* cis = nullptr;
    uint64 res;

    try {
        cout << "Test - read after close (TEXT&HUFFMAN)" << endl;
        stringbuf buffer;
        iostream ios(&buffer);
        cos = new CompressedOutputStream(ios, 1, "HUFFMAN", "TEXT");
        cos->write((const char*)block, length);
        cos->close();
        ios.seekg(0);
        cis = new CompressedInputStream(ios, 1);

        while (true) {
            cis->read((char*)block, length);

            if (cis->gcount() != length)
                break;
        }

        cis->close();
        cis->get();
        cout << "Failure: no exception thrown in read after close" << endl;
        res = 1;
    }
    catch (ios_base::failure& e) {
        cout << "OK, Exception " << e.what() << endl;
        res = 0;
    }

    delete cos;
    delete cis;
    return res;
}

int testCorrectness(int, const char*[])
{
    // Test correctness
    cout << "Correctness Test" << endl;
    const int length0 = 65536;
    byte* incompressible = new byte[length0 << 6];
    byte* values = new byte[length0 << 6];
    bool res = true;
    srand((uint)time(nullptr));

    for (int test = 1; test <= 50; test++) {
        const int length = length0 << (test % 7);

        for (int i = 0; i < length; i++) {
            incompressible[i] = byte(rand());
            values[i] = byte(rand() % (4 * test + 1));
        }

        cout << endl;
        cout << "Iteration " << test << " (size " << length << ")" << endl;
        uint64 cres;
        cres = compress1(values, length);
        cout << ((cres == 0) ? "Success" : "Failure") << endl;
        res &= (cres == 0);
        cres = compress2(values, length);
        cout << ((cres == 0) ? "Success" : "Failure") << endl;
        res &= (cres == 0);
        cres = compress3(incompressible, length);
        cout << ((cres == 0) ? "Success" : "Failure") << endl;
        res &= (cres == 0);

        if (test == 1) {
            cres = compress4(values, length);
            cout << ((cres == 0) ? "Success" : "Failure") << endl;
            res &= (cres == 0);
            cres = compress5(values, length);
            cout << ((cres == 0) ? "Success" : "Failure") << endl;
            res &= (cres == 0);
        }
    }

    delete[] incompressible;
    delete[] values;
    return (res == true) ? 0 : 1;
}

void testSeek(string name)
{
#if !defined(_MSC_VER) || _MSC_VER > 1500
    ifstream ifs;
    ifs.open(name.c_str(), ios::binary|ios::in);

    char buf[1024];
    int64 pos1 = 0;
    int64 pos2 = 0;
    CompressedInputStream cis(ifs, 1);
    cis.read((char*)buf, 100);

    for (int i = 0; i < 100; i++)
       cout << buf[i];

    cout << endl << endl;

    // Block positions for enwik8 compressed with L5 & version 2.4
    // To be updated.
    int64 positions[4] = { 17729391, 26695626, 8843019, 192 };

    for (int i = 0; i< 4; i++) {
        pos1 = positions[i]; 
        cis.seek(pos1);
        pos2 = cis.tell();
        cout << pos1 << " / " << pos2 << endl;
        cis.read(buf, 100);

        for (int j = 0; j < 100; j++)
           cout << buf[j];

        cout << endl << endl;
    }
    
    cis.close();
    ifs.close();
#endif
}

#ifdef __GNUG__
int main(int argc, const char* argv[])
#else
int TestCompressedStream_main(int argc, const char* argv[])
#endif
{
    try {
        //testSeek("/tmp/enwik8.knz");
        return testCorrectness(argc, argv);
    }
    catch (IOException& e) {
        cout << "Exception: " << e.what() << endl;
        return e.error();
    }
    catch (runtime_error& e) {
        cout << "Exception: " << e.what() << endl;
        return 255;
    }
}

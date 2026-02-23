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
#include <cstring>
#include <iostream>
#include <time.h>
#include "../types.hpp"
#include "../transform/BWT.hpp"
#include "../transform/BWTS.hpp"

using namespace std;
using namespace kanzi;

int testBWTCorrectness(bool isBWT)
{
    // Test behavior
    cout << endl
         << endl
         << (isBWT ? "BWT" : "BWTS") << " Correctness test" << endl;
    srand((uint)time(nullptr));
    int res = 0;
    kanzi::byte* pBuf = new kanzi::byte[8 * 1024 * 1024];
    kanzi::byte* buf1 = pBuf;

    for (int ii = 1; ii <= 20; ii++) {
        int size = 128;

        if (ii == 1) {
            string str("mississippi");
            const char* val2 = str.c_str();
            cout << val2 << endl;
            size = int(str.length());
            memcpy(buf1, &val2[0], size);
        }
        else if (ii == 2) {
            string str("3.14159265358979323846264338327950288419716939937510");
            const char* val2 = str.c_str();
            size = int(str.length());
            memcpy(buf1, &val2[0], size);
        }
        else if (ii == 3) {
            string str("SIX.MIXED.PIXIES.SIFT.SIXTY.PIXIE.DUST.BOXES");
            const char* val2 = str.c_str();
            size = int(str.length());
            memcpy(buf1, &val2[0], size);
        }
        else if (ii < 20) {
            for (int i = 0; i < size; i++)
                buf1[i] = kanzi::byte(65 + (rand() % (4 * ii)));
        }
        else {
            size = 8*1024*1024;

            for (int i = 0; i < size; i++)
                buf1[i] = kanzi::byte(i);
        }

        Transform<kanzi::byte>* tf;

        if (isBWT) {
            tf = new BWT();
        }
        else {
            tf = new BWTS();
        }

        kanzi::byte* input = &buf1[0];
        kanzi::byte* transform = new kanzi::byte[size];
        kanzi::byte* reverse = new kanzi::byte[size];
        cout << endl
             << "Test " << ii << endl;

        if (size < 512) {
             cout << "Input   : ";

             for (int i = 0; i < size; i++)
                cout << char(input[i]);
        }

        SliceArray<kanzi::byte> ia1(input, size, 0);
        SliceArray<kanzi::byte> ia2(transform, size, 0);
        tf->forward(ia1, ia2, size);

        if (size < 512) {
             cout << endl
                  << "Encoded : ";

             for (int i = 0; i < size; i++)
                cout << char(transform[i]);

             cout << "  ";
        }

        if (isBWT) {
            BWT* bwt = (BWT*) tf;
            int chunks = BWT::getBWTChunks(size);
            int* pi = new int[chunks];

            for (int i=0; i<chunks; i++) {
                pi[i] = bwt->getPrimaryIndex(i);
                cout << "(Primary index=" << pi[i] << ")" << endl;
            }

            delete tf;
            tf = new BWT();
            bwt = (BWT*) tf;

            for (int i=0; i<chunks; i++) {
                bwt->setPrimaryIndex(i, pi[i]);
            }

            delete[] pi;
        }
        else {
            delete tf;
            tf = new BWTS();
            cout << endl;
        }

        SliceArray<kanzi::byte> ia3(reverse, size, 0);
        ia2._index = 0;
        tf->inverse(ia2, ia3, size);

        int idx = -1;

        if (size < 512) {
            cout << "Reverse : ";

            for (int i = 0; i < size; i++)
                cout << char(reverse[i]);
        }

        for (int j = 0; j < size; j++) {
            if (input[j] != reverse[j]) {
                idx = j;
                res = 1;
                break;
            }
        }

        cout << endl;

        if (idx == -1)
            cout << "Identical" << endl;
        else
            cout << "Different at index " << idx << " (" << int(input[idx]) << "<->" << int(reverse[idx]) << ")" << endl;

        delete tf;
        delete[] transform;
        delete[] reverse;
    }

    delete[] pBuf;
    return res;
}

int testBWTSpeed(bool isBWT, int iter, bool isSmallSize)
{
    // Test speed
    int size = isSmallSize ? 256 * 1024 : 10 * 1024 * 1024;
    int res = 0;

    cout << endl
         << endl
         << (isBWT ? "BWT" : "BWTS") << " Speed test" << endl;
    cout << "Iterations: " << iter << endl;
    cout << "Transform size: " << size << endl;
    srand(uint(time(nullptr)));

    for (int jj = 0; jj < 3; jj++) {
        kanzi::byte* input = isSmallSize ? new kanzi::byte[256 * 1024] : new kanzi::byte[10 * 1024 * 1024];
        kanzi::byte* output = isSmallSize ? new kanzi::byte[256 * 1024] : new kanzi::byte[10 * 1024 * 1024];
        kanzi::byte* reverse = isSmallSize ? new kanzi::byte[256 * 1024] : new kanzi::byte[10 * 1024 * 1024];
        SliceArray<kanzi::byte> ia1(input, size, 0);
        SliceArray<kanzi::byte> ia2(output, size, 0);
        SliceArray<kanzi::byte> ia3(reverse, size, 0);
        double delta1 = 0, delta2 = 0;
        Transform<kanzi::byte>* tf = nullptr;
        Transform<kanzi::byte>* ti = nullptr;
        const int chunks = BWT::getBWTChunks(size);
        int pi[8];

        for (int ii = 0; ii < iter; ii++) {
            if (isBWT) {
                tf = new BWT();
            }
            else {
                tf = new BWTS();
            }

            for (int i = 0; i < size; i++) {
                input[i] = kanzi::byte(1 + (rand() % 255));
            }

            clock_t before1 = clock();
            ia1._index = 0;
            ia2._index = 0;
            tf->forward(ia1, ia2, size);
            clock_t after1 = clock();
            delta1 += (after1 - before1);

            if (isBWT) {
                BWT* bwt = (BWT*)tf;

                for (int i = 0; i < chunks; i++)
                    pi[i] = bwt->getPrimaryIndex(i);
            }

            delete tf;

            clock_t before2 = clock();
            ia2._index = 0;
            ia3._index = 0;

            if (isBWT) {
                ti = new BWT();
                BWT* bwt = (BWT*)ti;

                for (int i = 0; i < chunks; i++)
                    bwt->setPrimaryIndex(i, pi[i]);
            }
            else {
                ti = new BWTS();
            }

            ti->inverse(ia2, ia3, size);
            clock_t after2 = clock();
            delta2 += (after2 - before2);
            delete ti;

            // Sanity check
            for (int i = 0; i < size; i++) {
                if (input[i] != reverse[i]) {
                    cout << "Failure at index " << i << " (" << int(input[i]) << "<->" << int(reverse[i]) << ")" << endl;
                    res = 1;
                    break;
                }
            }
        }

        delete[] input;
        delete[] output;
        delete[] reverse;

        // KB = 1000, KiB = 1024
        double prod = double(iter) * double(size);
        double b2KiB = double(1) / double(1024);
        double d1_sec = double(delta1) / CLOCKS_PER_SEC;
        double d2_sec = double(delta2) / CLOCKS_PER_SEC;
        cout << "Forward transform [ms] : " << int(d1_sec * 1000) << endl;
        cout << "Throughput [KiB/s]     : " << int(prod * b2KiB / d1_sec) << endl;
        cout << "Reverse transform [ms] : " << int(d2_sec * 1000) << endl;
        cout << "Throughput [KiB/s]     : " << int(prod * b2KiB / d2_sec) << endl;
        cout << endl;
    }

    return res;
}

#ifdef __GNUG__
int main(int argc, const char* argv[])
#else
int TestBWT_main(int argc, const char* argv[])
#endif
{
    bool doPerf = true;

    if (argc > 1) {
        string str = argv[1];
        transform(str.begin(), str.end(), str.begin(), ::toupper);
        doPerf = str != "-NOPERF";
    }

    int res = 0;
    res |= testBWTCorrectness(true);
    res |= testBWTCorrectness(false);

    if (doPerf) {
       res |= testBWTSpeed(true, 200, true); // test MergeTPSI inverse
       res |= testBWTSpeed(true, 5, false); // test BiPSIv2 inverse
       res |= testBWTSpeed(false, 200, true);
    }

    return res;
}


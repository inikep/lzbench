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

#include <cstring>
#include <vector>
#include "BWT.hpp"
#include "../Global.hpp"
#include "../Memory.hpp"

#ifdef CONCURRENCY_ENABLED
#include <future>
#endif

using namespace kanzi;
using namespace std;

BWT::BWT(int jobs)
{
    _buffer = new uint[0];
    _sa = new int[0];
    _bufferSize = 0;
    _saSize = 0;

#ifdef CONCURRENCY_ENABLED
    _pool = nullptr;

    if (jobs < 1)
        throw invalid_argument("The number of jobs must be at least 1");
#else
    if (jobs != 1)
        throw invalid_argument("The number of jobs is limited to 1 in this version");
#endif

    _jobs = jobs;
    memset(_primaryIndexes, 0, sizeof(int) * 8);
}


BWT::BWT(Context& ctx)
{
    _buffer = new uint[0];
    _sa = new int[0];
    _bufferSize = 0;
    _saSize = 0;
    int jobs = ctx.getInt("jobs", 1);

#ifdef CONCURRENCY_ENABLED
    _pool = ctx.getPool(); // can be null

    if (jobs < 1)
        throw invalid_argument("The number of jobs must be at least 1");
#else
    if (jobs != 1)
        throw invalid_argument("The number of jobs is limited to 1 in this version");
#endif

    _jobs = jobs;
    memset(_primaryIndexes, 0, sizeof(int) * 8);
}

bool BWT::setPrimaryIndex(int n, int primaryIndex)
{
    if ((primaryIndex < 0) || (n < 0) || (n >= 8))
        return false;

    _primaryIndexes[n] = primaryIndex;
    return true;
}

bool BWT::forward(SliceArray<byte>& input, SliceArray<byte>& output, int count)
{
    if (count == 0)
        return true;

    if (!SliceArray<byte>::isValid(input))
       throw invalid_argument("BWT: Invalid input block");

    if (!SliceArray<byte>::isValid(output))
        throw invalid_argument("BWT: Invalid output block");

    if (count > MAX_BLOCK_SIZE)
        return false;

    if (count == 1) {
        output._array[output._index++] = input._array[input._index++];
        return true;
    }

    byte* src = &input._array[input._index];
    byte* dst = &output._array[output._index];

    // Lazy dynamic memory allocation
    if (_saSize < count) {
         delete[] _sa;
         _saSize = count;
         _sa = new int[_saSize];
    }

    _saAlgo.computeBWT(src, dst, _sa, count, _primaryIndexes, getBWTChunks(count));
    input._index += count;
    output._index += count;
    return true;
}

bool BWT::inverse(SliceArray<byte>& input, SliceArray<byte>& output, int count)
{
    if (count == 0)
        return true;

    if (!SliceArray<byte>::isValid(input))
        throw invalid_argument("BWT: Invalid input block");

    if (!SliceArray<byte>::isValid(output))
        throw invalid_argument("BWT: Invalid output block");

    if (count == 1) {
        output._array[output._index++] = input._array[input._index++];
        return true;
    }

    // Find the fastest way to implement inverse based on block size
    if (count <= BLOCK_SIZE_THRESHOLD2)
        return inverseMergeTPSI(input, output, count);

    return inverseBiPSIv2(input, output, count);
}

// When count <= BLOCK_SIZE_THRESHOLD2, mergeTPSI algo
bool BWT::inverseMergeTPSI(SliceArray<byte>& input, SliceArray<byte>& output, int count)
{
    // Lazy dynamic memory allocation
    if (_bufferSize < count) {
        delete[] _buffer;
        _bufferSize = max(count, 256);
        _buffer = new uint[_bufferSize];
    }

    const int pIdx = getPrimaryIndex(0);

    if ((pIdx < 0) || (pIdx > count))
        return false;

    // Build array of packed index + value (assumes block size < 1<<24)
    uint buckets[256] = { 0 };
    Global::computeHistogram(&input._array[input._index], count, buckets);

    for (int i = 0, sum = 0; i < 256; i++) {
        const int tmp = buckets[i];
        buckets[i] = sum;
        sum += tmp;
    }

    const byte* src = &input._array[input._index];
    byte* dst = &output._array[output._index];
    memset(&_buffer[0], 0, size_t(_bufferSize) * sizeof(uint));

    for (int i = 0; i < pIdx; i++) {
        const uint8 val = uint8(src[i]);
        _buffer[buckets[val]] = ((i - 1) << 8) | val;
        buckets[val]++;
    }

    for (int i = pIdx; i < count; i++) {
        const uint8 val = uint8(src[i]);
        _buffer[buckets[val]] = (i << 8) | val;
        buckets[val]++;
    }

    if (count < BLOCK_SIZE_THRESHOLD1) {
        for (int i = 0, t = pIdx - 1; i < count; i++) {
            const uint ptr = _buffer[t];
            dst[i] = byte(ptr);
            t = ptr >> 8;
        }
    }
    else {
        const int ckSize = ((count & 7) == 0) ? count >> 3 : (count >> 3) + 1;
        int t0 = getPrimaryIndex(0) - 1;
        int t1 = getPrimaryIndex(1) - 1;
        int t2 = getPrimaryIndex(2) - 1;
        int t3 = getPrimaryIndex(3) - 1;
        int t4 = getPrimaryIndex(4) - 1;
        int t5 = getPrimaryIndex(5) - 1;
        int t6 = getPrimaryIndex(6) - 1;
        int t7 = getPrimaryIndex(7) - 1;
        byte* d0 = &dst[ckSize * 0];
        byte* d1 = &dst[ckSize * 1];
        byte* d2 = &dst[ckSize * 2];
        byte* d3 = &dst[ckSize * 3];
        byte* d4 = &dst[ckSize * 4];
        byte* d5 = &dst[ckSize * 5];
        byte* d6 = &dst[ckSize * 6];
        byte* d7 = &dst[ckSize * 7];
        int n = 0;
        int ptr;

        #define S(t, d) ptr = _buffer[t]; \
           d[n] = byte(ptr); \
           t = ptr >> 8; \
           prefetchRead(&_buffer[t])

        while (true) {
            S(t0, d0);
            S(t1, d1);
            S(t2, d2);
            S(t3, d3);
            S(t4, d4);
            S(t5, d5);
            S(t6, d6);
            S(t7, d7);
            n++;

            if (ptr < 0)
                break;
        }

        while (n < ckSize) {
            S(t0, d0);
            S(t1, d1);
            S(t2, d2);
            S(t3, d3);
            S(t4, d4);
            S(t5, d5);
            S(t6, d6);
            n++;
        }
    }

    input._index += count;
    output._index += count;
    return true;
}

// When count > BLOCK_SIZE_THRESHOLD2, biPSIv2 algo
bool BWT::inverseBiPSIv2(SliceArray<byte>& input, SliceArray<byte>& output, int count)
{
    // Lazy dynamic memory allocations
    if (_bufferSize < count + 1) {
        delete[] _buffer;
        _bufferSize = max(count + 1, 256);
        _buffer = new uint[_bufferSize];
    }

    const byte* src = &input._array[input._index];
    byte* dst = &output._array[output._index];
    const int pIdx = getPrimaryIndex(0);

    if ((pIdx < 0) || (pIdx > count))
        return false;

    uint* buckets = new uint[65536];
    memset(&_buffer[0], 0, _bufferSize * sizeof(uint));
    memset(&buckets[0], 0, 65536 * sizeof(uint));
    uint freqs[256] = { 0 };
    Global::computeHistogram(&input._array[input._index], count, freqs);

    for (int sum = 1, c = 0; c < 256; c++) {
        const int f = sum;
        sum += int(freqs[c]);
        freqs[c] = f;

        if (f != sum) {
            uint* ptr = &buckets[c << 8];
            const int hi = min(sum, pIdx);

            for (int i = f; i < hi; i++)
                ptr[int(src[i])]++;

            const int lo = max(f - 1, pIdx);

            for (int i = lo; i < sum - 1; i++)
                ptr[int(src[i])]++;
        }
    }

    const int lastc = int(src[0]);
    uint16* fastBits = new uint16[MASK_FASTBITS + 1];
    memset(&fastBits[0], 0, size_t(MASK_FASTBITS + 1) * sizeof(uint16));
    int shift = 0;

    while ((count >> shift) > MASK_FASTBITS)
        shift++;

    for (int v = 0, sum = 1, c = 0; c < 256; c++) {
        if (c == lastc)
            sum++;

        uint* ptr = &buckets[c];

        for (int d = 0; d < 256; d++) {
            const int s = sum;
            sum += ptr[d << 8];
            ptr[d << 8] = s;

            if (s == sum)
                continue;

            for (; v <= ((sum - 1) >> shift); v++)
                fastBits[v] = uint16((c << 8) | d);
        }
    }

    memset(&_buffer[0], 0, size_t(_bufferSize) * sizeof(uint));
    int n = 0;

    while (n < pIdx) {
        const int c = int(src[n]);
        const int p = freqs[c];

        if (p < pIdx)
            _buffer[buckets[(c << 8) | int(src[p])]++] = n;
        else if (p > pIdx)
            _buffer[buckets[(c << 8) | int(src[p - 1])]++] = n;

        freqs[c]++;
        n++;
    }

    while (n < count) {
        const int c = int(src[n]);
        const int p = freqs[c];
        freqs[c]++;
        n++;

        if (p < pIdx)
            _buffer[buckets[(c << 8) | int(src[p])]++] = n;
        else if (p > pIdx)
            _buffer[buckets[(c << 8) | int(src[p - 1])]++] = n;
    }

    for (int c = 0; c < 256; c++) {
        for (int d = 0; d < c; d++) {
            swap(buckets[(d << 8) | c],  buckets[(c << 8) | d]);
        }
    }

    const int chunks = getBWTChunks(count);

    // Build inverse
    const int st = count / chunks;
    const int ckSize = (chunks * st == count) ? st : st + 1;
    const int nbTasks = (_jobs < chunks) ? _jobs : chunks;

    if (nbTasks == 1) {
        InverseBiPSIv2Task<int> task(_buffer, buckets, fastBits, dst, _primaryIndexes,
            count, 0, ckSize, 0, chunks);
        task.run();
    }
    else {
#ifdef CONCURRENCY_ENABLED
        // Several chunks may be decoded concurrently (depending on the availability
        // of jobs per block).
        int jobsPerTask[64];
        Global::computeJobsPerTask(jobsPerTask, chunks, nbTasks);
        vector<future<int> > futures;
        vector<InverseBiPSIv2Task<int>*> tasks;

        // Create one task per job
        for (int j = 0, c = 0; j < nbTasks; j++) {
            // Each task decodes jobsPerTask[j] chunks
            InverseBiPSIv2Task<int>* task = new InverseBiPSIv2Task<int>(_buffer, buckets, fastBits, dst, _primaryIndexes,
                count, c * ckSize, ckSize, c, c + jobsPerTask[j]);
            tasks.push_back(task);

            if (_pool == nullptr)
               futures.push_back(async(launch::async, &InverseBiPSIv2Task<int>::run, task));
            else
               futures.push_back(_pool->schedule(&InverseBiPSIv2Task<int>::run, task));

            c += jobsPerTask[j];
        }

        // Wait for completion of all concurrent tasks
        for (int j = 0; j < nbTasks; j++)
            futures[j].get();

        // Cleanup
        for (InverseBiPSIv2Task<int>* task : tasks)
            delete task;
#else
        // nbTasks > 1 but concurrency is not enabled (should never happen)
        delete[] fastBits;
        delete[] buckets;
        throw invalid_argument("Error during BWT inverse: concurrency not supported");
#endif
    }

    dst[count - 1] = byte(lastc);
    delete[] fastBits;
    delete[] buckets;
    input._index += count;
    output._index += count;
    return true;
}

template <class T>
InverseBiPSIv2Task<T>::InverseBiPSIv2Task(uint* buf, uint* buckets, uint16* fastBits, byte* output,
                                          int* primaryIndexes, int total, int start, int ckSize, int firstChunk, int lastChunk)
                                          : _data(buf)
                                          , _buckets(buckets)
                                          , _fastBits(fastBits)
                                          , _primaryIndexes(primaryIndexes)
                                          , _dst(output)
                                          , _total(total)
                                          , _start(start)
                                          , _ckSize(ckSize)
                                          , _firstChunk(firstChunk)
                                          , _lastChunk(lastChunk)
{
}

template <class T>
T InverseBiPSIv2Task<T>::run()
{
    uint sh = 0;

    while ((_total >> sh) > BWT::MASK_FASTBITS)
        sh++;

    const uint shift = sh;
    int c = _firstChunk;
    byte* d0 = &_dst[0 * _ckSize];
    byte* d1 = &_dst[1 * _ckSize];
    byte* d2 = &_dst[2 * _ckSize];
    byte* d3 = &_dst[3 * _ckSize];

    if (_start + 4 * _ckSize < _total) {
        for (; c + 3 < _lastChunk; c += 4) {
            const int end = _start + _ckSize;
            uint p0 = _primaryIndexes[c];
            uint p1 = _primaryIndexes[c + 1];
            uint p2 = _primaryIndexes[c + 2];
            uint p3 = _primaryIndexes[c + 3];

            for (int i = _start + 1; i <= end; i += 2) {
                prefetchRead(&_data[p0]);
                prefetchRead(&_data[p1]);
                prefetchRead(&_data[p2]);
                prefetchRead(&_data[p3]);
                uint16 s0 = _fastBits[p0 >> shift];
                uint16 s1 = _fastBits[p1 >> shift];
                uint16 s2 = _fastBits[p2 >> shift];
                uint16 s3 = _fastBits[p3 >> shift];

                while (_buckets[s0] <= (const uint)p0)
                    s0++;

                while (_buckets[s1] <= (const uint)p1)
                    s1++;

                while (_buckets[s2] <= (const uint)p2)
                    s2++;

                while (_buckets[s3] <= (const uint)p3)
                    s3++;

                d0[i - 1] = byte(s0 >> 8);
                d0[i] = byte(s0);
                d1[i - 1] = byte(s1 >> 8);
                d1[i] = byte(s1);
                d2[i - 1] = byte(s2 >> 8);
                d2[i] = byte(s2);
                d3[i - 1] = byte(s3 >> 8);
                d3[i] = byte(s3);

                p0 = _data[p0];
                p1 = _data[p1];
                p2 = _data[p2];
                p3 = _data[p3];
            }

            _start = end + 3 * _ckSize;
        }
    }

    for (; c < _lastChunk; c++) {
        const int end = min(_start + _ckSize, _total - 1);
        uint p = _primaryIndexes[c];

        for (int i = _start + 1; i <= end; i += 2) {
            uint16 s = _fastBits[p >> shift];

            while (_buckets[s] <= (const uint)p)
                s++;

            _dst[i - 1] = byte(s >> 8);
            _dst[i] = byte(s);
            p = _data[p];
        }

        _start = end;
    }

    return T(0);
}

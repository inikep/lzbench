/*
 * Copyright (c) Przemyslaw Skibinski <inikep@gmail.com>
 * All rights reserved.
 *
 * This source code is dual-licensed under the GPLv2 and GPLv3 licenses.
 * For additional details, refer to the LICENSE file located in the root
 * directory of this source tree.
 */

#ifndef LZBENCH_THREADPOOL_H
#define LZBENCH_THREADPOOL_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>
#include <vector>

#include "lzbench.h"

struct CompressionTask {
    bool isCompression;
    size_t chunkNo;
    uint8_t* input;
    size_t inputSize;
    uint8_t* output;
    size_t maxOutputSize;
    compress_func codec_function;
    const codec_options_t* codec_options;
    std::vector<char*> *workmems;
};


class ThreadPool {
public:
    ThreadPool() {};
    ThreadPool(size_t numThreads, size_t numBlocks);
    ~ThreadPool();

    void enqueue(CompressionTask task);
    void waitForCompletion();
    void clear();

    std::vector<size_t> chunkSizes;  // Stores original chunk sizes
    std::vector<size_t> compSizes;   // Stores compressed sizes
    std::vector<size_t> comptasksDone;
    std::vector<size_t> decomptasksDone;
    size_t numThreads;

private:
    std::vector<std::thread> workers;
    std::queue<CompressionTask> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
    std::atomic<size_t> activeTasks;

    void workerThread(int threadNo);
};

#ifdef DISABLE_THREADING
ThreadPool::~ThreadPool() {};
#endif // #ifndef DISABLE_THREADING

#endif // #ifndef LZBENCH_THREADPOOL_H

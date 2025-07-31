/*
 * Copyright (c) Przemyslaw Skibinski <inikep@gmail.com>
 * All rights reserved.
 *
 * This source code is dual-licensed under the GPLv2 and GPLv3 licenses.
 * For additional details, refer to the LICENSE file located in the root
 * directory of this source tree.
 */

#include <cstring> // memcpy
#include "threadpool.h"

ThreadPool::ThreadPool(size_t numThreads, size_t numBlocks)
    : chunkSizes(numBlocks), compSizes(numBlocks), comptasksDone(numThreads), decomptasksDone(numThreads), numThreads(numThreads), stop(false), activeTasks(0) {

    //printf("ThreadPool::ThreadPool numThreads=%zu numBlocks=%zu\n", numThreads, numBlocks);
    for (size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back(&ThreadPool::workerThread, this, i);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers) {
        worker.join();
    }
}

// Add a new task to the queue
void ThreadPool::enqueue(CompressionTask task) {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        tasks.push(task);
    }
    activeTasks++;
    condition.notify_one();
}

// Wait for all tasks to finish
void ThreadPool::waitForCompletion() {
    while (activeTasks > 0) {
        std::this_thread::yield();
    }
}

void ThreadPool::clear() {
    comptasksDone.assign(numThreads, 0);
    decomptasksDone.assign(numThreads, 0);
}

// Worker thread function
void ThreadPool::workerThread(int threadNo) {
    codec_options_t workerCodecOptions;
    while (true) {
        CompressionTask task;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            condition.wait(lock, [this] { return !tasks.empty() || stop; });

            if (stop && tasks.empty()) return;

            task = tasks.front();
            tasks.pop();
        }

        if (task.isCompression) {
            //printf("COMP1 chunkNo=%zu threadNo=%d compress=%zu -maxOutputSize=%zu\n", task.chunkNo, threadNo, task.inputSize, task.maxOutputSize);
            memcpy(&workerCodecOptions, task.codec_options, sizeof(codec_options_t));
            workerCodecOptions.work_mem = (*task.workmems)[threadNo];
            compSizes[task.chunkNo] = task.codec_function((char*)task.input, task.inputSize, (char*)task.output, task.maxOutputSize, &workerCodecOptions);
            comptasksDone[threadNo]++;
            //printf("COMP2 chunkNo=%zu threadNo=%d compress=%zu -> %zu\n", task.chunkNo, threadNo, task.inputSize, compSizes[task.chunkNo]);

        } else {
            //printf("DECOMP1 chunkNo=%zu threadNo=%d decompress=%zu maxOutputSize=%zu\n", task.chunkNo, threadNo, task.inputSize, task.maxOutputSize);
            memcpy(&workerCodecOptions, task.codec_options, sizeof(codec_options_t));
            workerCodecOptions.work_mem = (*task.workmems)[threadNo];
            chunkSizes[task.chunkNo] = task.codec_function((char*)task.input, task.inputSize, (char*)task.output, task.maxOutputSize, &workerCodecOptions);
            decomptasksDone[threadNo]++;
            //printf("DECOMP2 chunkNo=%zu threadNo=%d decompress=%zu -> %zu\n", task.chunkNo, threadNo, task.inputSize, chunkSizes[task.chunkNo]);
        }

        activeTasks--;
    }
}

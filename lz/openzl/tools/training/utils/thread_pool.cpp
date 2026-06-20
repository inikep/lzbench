// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "tools/training/utils/thread_pool.h"
#include "openzl/cpp/Exception.hpp"
#include "tools/logger/Logger.h"

namespace openzl::training {
using namespace openzl::tools::logger;

ThreadPool::ThreadPool(size_t numThreads_) : numThreads(numThreads_)
{
    if (numThreads == 0) {
        throw Exception("Number of threads must not be 0");
    }
    Logger::log_c(
            VERBOSE1, "Creating thread pool with %zu threads", numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
        threads_.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queueMutex_);
                    condition_.wait(lock, [this]() {
                        return stop_ || !taskQueue_.empty();
                    });
                    if (stop_ && taskQueue_.empty()) {
                        return;
                    }
                    task = std::move(taskQueue_.front());
                    taskQueue_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool()
{
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (auto& thread : threads_) {
        thread.join();
    }
}
} // namespace openzl::training

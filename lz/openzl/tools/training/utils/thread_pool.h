// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <queue>
#include <thread>
#include <vector>

namespace openzl::training {
class ThreadPool {
   public:
    explicit ThreadPool(size_t numThreads_);

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool(ThreadPool&&)                 = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool& operator=(ThreadPool&&)      = delete;

    ~ThreadPool();

    /**
     * This function accepts a callable object and its arguments, packages them
     * into a task, and enqueues the task for execution by the thread pool. It
     * returns a `std::future` object that can be used to retrieve the result of
     * the task once it has been executed. This function is intended to be used
     * for running asynchronous tasks in parallel.
     *
     * @param func The callable object to be executed.
     * @param args The arguments to be passed to the callable object.
     * @return std::future<ReturnType> A future object that holds the result of
     * the task.
     */
    template <typename Func, typename... Args>
    auto run(Func&& func, Args&&... args)
            -> std::future<decltype(func(args...))>
    {
        using ReturnType = decltype(func(args...));
        auto task =
                std::make_shared<std::packaged_task<ReturnType()>>(std::bind(
                        std::forward<Func>(func), std::forward<Args>(args)...));
        std::future<ReturnType> result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            taskQueue_.emplace([task]() { (*task)(); });
        }
        condition_.notify_one();
        return result;
    }

    const size_t numThreads;

   private:
    std::mutex queueMutex_;
    std::queue<std::function<void()>> taskQueue_;
    std::vector<std::thread> threads_;
    std::condition_variable condition_;
    bool stop_ = false;
};
} // namespace openzl::training

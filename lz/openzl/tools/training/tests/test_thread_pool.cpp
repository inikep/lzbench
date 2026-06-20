// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include "openzl/cpp/Exception.hpp"
#include "tools/training/utils/thread_pool.h"

using openzl::training::ThreadPool;

namespace openzl {
class TestThreadPool : public ::testing::Test {
   public:
    void SetUp() override
    {
        pool_ = std::make_unique<ThreadPool>(8);
    }

   protected:
    std::unique_ptr<ThreadPool> pool_;
};

TEST_F(TestThreadPool, TestThreadPoolNoThreads)
{
    EXPECT_THROW(std::make_unique<ThreadPool>(0), Exception);
}

TEST_F(TestThreadPool, TestThreadPoolRunSingleTask)
{
    auto future = pool_->run([]() { return 42; });
    EXPECT_EQ(future.get(), 42);
}

TEST_F(TestThreadPool, TestThreadPoolRunMultipleTasks)
{
    auto future1 = pool_->run([]() { return 1; });
    auto future2 = pool_->run([]() { return 2; });
    auto future3 = pool_->run([]() { return 3; });
    EXPECT_EQ(future1.get(), 1);
    EXPECT_EQ(future2.get(), 2);
    EXPECT_EQ(future3.get(), 3);
}

TEST_F(TestThreadPool, TestThreadPoolTaskExecution)
{
    std::atomic<int> counter = 0;
    std::vector<std::future<void>> futures;
    futures.reserve(10);
    for (int i = 0; i < 10; ++i) {
        futures.push_back(pool_->run([&counter]() { counter++; }));
    }
    for (auto& future : futures) {
        future.get();
    }
    EXPECT_EQ(counter.load(), 10);
}
} // namespace openzl

// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <benchmark/benchmark.h>
#include <string>
#include <string_view>
#include <vector>

namespace zstrong::bench {

class BenchmarkConfig {
   public:
    void setUseShortList(bool shortList);
    bool useShortList() const;
    const std::vector<std::string>& getShortList() const;
    bool shouldRegister(std::string_view name) const;

    static BenchmarkConfig& instance();

    BenchmarkConfig(const BenchmarkConfig&)            = delete;
    BenchmarkConfig& operator=(const BenchmarkConfig&) = delete;

   private:
    bool shortList_ = false;

    BenchmarkConfig() {}
    ~BenchmarkConfig() {}

}; // struct singleton_t

template <typename F>
void RegisterBenchmark(const std::string& name, F&& func)
{
    if (BenchmarkConfig::instance().shouldRegister(name))
        benchmark::RegisterBenchmark(name.c_str(), std::forward<F>(func));
}

} // namespace zstrong::bench

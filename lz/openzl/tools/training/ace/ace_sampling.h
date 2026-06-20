// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <openzl/cpp/poly/Span.hpp>
#include <random>

namespace openzl {
namespace training {

/// https://en.wikipedia.org/wiki/Reservoir_sampling
template <typename T>
class ACEReservoirSampler {
   public:
    explicit ACEReservoirSampler(std::mt19937_64& rng) : rng_(rng) {}

    void reset()
    {
        chosen_ = nullptr;
        count_  = 0;
    }

    void update(T& value)
    {
        std::uniform_int_distribution<size_t> dist(0, count_);
        if (dist(rng_) == 0) {
            chosen_ = &value;
        }
        ++count_;
    }

    T* get() const
    {
        return chosen_;
    }

   private:
    std::mt19937_64& rng_;
    T* chosen_{ nullptr };
    size_t count_{ 0 };
};

template <typename T>
T randomChoice(std::mt19937_64& rng, poly::span<const T> choices)
{
    assert(!choices.empty());
    std::uniform_int_distribution<size_t> dist(0, choices.size() - 1);
    return choices[dist(rng)];
}
} // namespace training
} // namespace openzl

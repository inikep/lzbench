// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>

#include "openzl/cpp/Type.hpp"

namespace openzl {
namespace nodes {
struct InputMetadata {
    Type type;
    const char* name = "";
};

struct OutputMetadata {
    Type type;
    const char* name = "";
};

template <
        size_t kNumInputs,
        size_t kSingletonOutputs,
        size_t kVariableOutputs = 0>
struct NodeMetadata {
    std::array<InputMetadata, kNumInputs> inputs;
    std::array<OutputMetadata, kSingletonOutputs> singletonOutputs;
    std::array<OutputMetadata, kVariableOutputs> variableOutputs;
    bool lastInputIsVariable = false;
    const char* description  = "";
};
} // namespace nodes

namespace graphs {
struct InputMetadata {
    TypeMask typeMask;
    const char* name = "";
};

template <size_t kNumInputs>
struct GraphMetadata {
    std::array<InputMetadata, kNumInputs> inputs;
    bool lastInputIsVariable = false;
    const char* description  = "";
};

} // namespace graphs

} // namespace openzl

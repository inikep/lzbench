// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>
#include <vector>

#include "openzl/cpp/Type.hpp"
#include "openzl/cpp/poly/Optional.hpp"

namespace openzl {
struct SimpleCodecDescription {
    unsigned id;
    poly::optional<std::string> name;
    Type inputType;
    std::vector<Type> outputTypes;
};

struct VariableOutputCodecDescription {
    unsigned id;
    poly::optional<std::string> name;
    Type inputType;
    std::vector<Type> singletonOutputTypes;
    std::vector<Type> variableOutputTypes;

    static VariableOutputCodecDescription fromSimple(
            SimpleCodecDescription desc)
    {
        return VariableOutputCodecDescription{
            .id                   = desc.id,
            .name                 = std::move(desc.name),
            .inputType            = desc.inputType,
            .singletonOutputTypes = std::move(desc.outputTypes),
        };
    }
};

struct MultiInputCodecDescription {
    unsigned id;
    poly::optional<std::string> name;
    std::vector<Type> inputTypes;
    bool lastInputIsVariable{ false };
    std::vector<Type> singletonOutputTypes;
    std::vector<Type> variableOutputTypes;

    static MultiInputCodecDescription fromVariableOutput(
            VariableOutputCodecDescription desc)
    {
        return MultiInputCodecDescription{
            .id                   = desc.id,
            .name                 = std::move(desc.name),
            .inputTypes           = { desc.inputType },
            .singletonOutputTypes = std::move(desc.singletonOutputTypes),
            .variableOutputTypes  = std::move(desc.variableOutputTypes),
        };
    }
};

} // namespace openzl

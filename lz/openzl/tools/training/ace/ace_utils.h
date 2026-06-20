// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/cpp/Type.hpp"

namespace openzl {
namespace training {

/// Maps an input type to a mask of all types that can be converted to that
/// type.
TypeMask typeToMaskWithConversion(Type accepts);
/// @returns if @p inputType is compatible with a graph that accepts tyhe types
/// @p accepts
bool isCompatible(TypeMask accepts, Type inputType);
/// @returns if @p inputType is compatible with a node that accepts the type
/// @p accepts
bool isCompatible(Type accepts, Type inputType);

} // namespace training
} // namespace openzl

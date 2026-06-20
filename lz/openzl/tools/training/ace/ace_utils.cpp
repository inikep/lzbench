// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "tools/training/ace/ace_utils.h"

namespace openzl {
namespace training {

bool isCompatible(TypeMask accepts, Type inputType)
{
    return (TypeMask(int(inputType)) & accepts) != TypeMask::None;
}

TypeMask typeToMaskWithConversion(Type accepts)
{
    switch (accepts) {
        case Type::Serial:
            return TypeMask::Serial | TypeMask::Struct | TypeMask::Numeric;
        case Type::Struct:
            return TypeMask::Struct | TypeMask::Numeric;
        case Type::Numeric:
            return TypeMask::Numeric;
        case Type::String:
            return TypeMask::String;
        default:
            assert(false);
            return TypeMask::None;
    }
}

bool isCompatible(Type accepts, Type inputType)
{
    return isCompatible(typeToMaskWithConversion(accepts), inputType);
}

} // namespace training
} // namespace openzl

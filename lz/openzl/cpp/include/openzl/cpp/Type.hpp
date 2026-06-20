// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <vector>

#include "openzl/cpp/poly/Span.hpp"
#include "openzl/zl_data.h"

namespace openzl {
enum class Type {
    Serial  = ZL_Type_serial,
    Struct  = ZL_Type_struct,
    Numeric = ZL_Type_numeric,
    String  = ZL_Type_string,
};

enum class TypeMask {
    None    = 0,
    Serial  = ZL_Type_serial,
    Struct  = ZL_Type_struct,
    Numeric = ZL_Type_numeric,
    String  = ZL_Type_string,
    Any     = ZL_Type_any,
};

constexpr inline TypeMask operator|(TypeMask a, TypeMask b)
{
    return static_cast<TypeMask>(static_cast<int>(a) | static_cast<int>(b));
}
constexpr inline TypeMask& operator|=(TypeMask& a, TypeMask b)
{
    a = a | b;
    return a;
}

constexpr inline TypeMask operator&(TypeMask a, TypeMask b)
{
    return static_cast<TypeMask>(static_cast<int>(a) & static_cast<int>(b));
}
constexpr inline TypeMask& operator&=(TypeMask& a, TypeMask b)
{
    a = a & b;
    return a;
}

inline ZL_Type typeToCType(const Type type)
{
    return static_cast<ZL_Type>(static_cast<int>(type));
}

inline std::vector<ZL_Type> typesToCTypes(poly::span<const Type> types)
{
    std::vector<ZL_Type> cTypes;
    cTypes.reserve(types.size());
    for (auto type : types) {
        cTypes.push_back(static_cast<ZL_Type>(static_cast<int>(type)));
    }
    return cTypes;
}

inline ZL_Type typeMaskToCType(const TypeMask type)
{
    return static_cast<ZL_Type>(static_cast<int>(type));
}

inline std::vector<ZL_Type> typesMasksToCTypes(poly::span<const TypeMask> types)
{
    std::vector<ZL_Type> cTypes;
    cTypes.reserve(types.size());
    for (auto type : types) {
        cTypes.push_back(static_cast<ZL_Type>(static_cast<int>(type)));
    }
    return cTypes;
}
} // namespace openzl

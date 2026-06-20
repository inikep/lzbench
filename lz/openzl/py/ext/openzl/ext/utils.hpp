// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <string>
#include "openzl/openzl.hpp"

namespace openzl {
namespace py {

inline std::string typeName(Type type)
{
    switch (type) {
        case Type::Serial:
            return "Type.Serial";
        case Type::Struct:
            return "Type.Struct";
        case Type::Numeric:
            return "Type.Numeric";
        case Type::String:
            return "Type.String";
        default:
            throw Exception("Unknown type");
    }
}

inline std::string typeName(TypeMask mask)
{
    if (mask == TypeMask::Any) {
        return "TypeMask.Any";
    } else if (mask == TypeMask::None) {
        return "TypeMask.None";
    }

    std::string str;
    if ((mask & TypeMask::Serial) != TypeMask::None) {
        str += "TypeMask.Serial | ";
    }
    if ((mask & TypeMask::Struct) != TypeMask::None) {
        str += "TypeMask.Struct | ";
    }
    if ((mask & TypeMask::Numeric) != TypeMask::None) {
        str += "TypeMask.Numeric | ";
    }
    if ((mask & TypeMask::String) != TypeMask::None) {
        str += "TypeMask.String | ";
    }
    str.pop_back();
    str.pop_back();
    str.pop_back();
    return str;
}

template <typename TypeT>
std::string ioDocs(TypeT type, std::string name, bool isInput)
{
    if (name.empty()) {
        name = isInput ? "input" : "output";
    }
    return name + ": " + typeName(type);
}

template <typename IOMetas>
std::string ioDocs(const IOMetas& metas, bool isInput)
{
    std::string docs;
    for (const auto& [type, name] : metas) {
        docs += ioDocs(type, name, isInput);
        docs += "\n";
    }
    return docs;
}

} // namespace py
} // namespace openzl

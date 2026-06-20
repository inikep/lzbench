// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "custom_transforms/thrift/constants.h" // @manual
#include "openzl/zl_data.h"                     // @manual

#include <array>
#include <string>

#pragma once

namespace zstrong::thrift {

enum class TType : uint8_t {
    T_STOP   = 0,
    T_VOID   = 1,
    T_BOOL   = 2,
    T_BYTE   = 3,
    T_DOUBLE = 4,
    T_U16    = 5, /* unused by thrift */
    T_I16    = 6,
    T_U32    = 7, /* unused by thrift */
    T_I32    = 8,
    T_U64    = 9,
    T_I64    = 10,
    T_STRING = 11,
    T_STRUCT = 12,
    T_MAP    = 13,
    T_SET    = 14,
    T_LIST   = 15,
    T_UTF8   = 16,
    T_UTF16  = 17,
    T_STREAM = 18,
    T_FLOAT  = 19,
};

enum class CType : uint8_t {
    CT_STOP          = 0x00,
    CT_BOOLEAN_TRUE  = 0x01,
    CT_BOOLEAN_FALSE = 0x02,
    CT_BYTE          = 0x03,
    CT_I16           = 0x04,
    CT_I32           = 0x05,
    CT_I64           = 0x06,
    CT_DOUBLE        = 0x07,
    CT_BINARY        = 0x08,
    CT_LIST          = 0x09,
    CT_SET           = 0x0A,
    CT_MAP           = 0x0B,
    CT_STRUCT        = 0x0C,
    CT_FLOAT         = 0x0D,
    CT_VOID          = 0x0E,
};

const std::string& thriftTypeToString(TType type);

constexpr std::array<TType, 15> CTypeToTType = {
    TType::T_STOP,   // CT_STOP
    TType::T_BOOL,   // CT_BOOLEAN_TRUE
    TType::T_BOOL,   // CT_BOOLEAN_FALSE
    TType::T_BYTE,   // CT_BYTE
    TType::T_I16,    // CT_I16
    TType::T_I32,    // CT_I32
    TType::T_I64,    // CT_I64
    TType::T_DOUBLE, // CT_DOUBLE
    TType::T_STRING, // CT_BINARY
    TType::T_LIST,   // CT_LIST
    TType::T_SET,    // CT_SET
    TType::T_MAP,    // CT_MAP
    TType::T_STRUCT, // CT_STRUCT
    TType::T_FLOAT,  // CT_FLOAT
    TType::T_VOID,   // CT_VOID
};

constexpr std::array<CType, 20> TTypeToCType = {
    CType::CT_STOP,         // T_STOP
    CType::CT_VOID,         // unused
    CType::CT_BOOLEAN_TRUE, // T_BOOL
    CType::CT_BYTE,         // T_BYTE
    CType::CT_DOUBLE,       // T_DOUBLE
    CType::CT_VOID,         // unused
    CType::CT_I16,          // T_I16
    CType::CT_VOID,         // unused
    CType::CT_I32,          // T_I32
    CType::CT_VOID,         // unused
    CType::CT_I64,          // T_I64
    CType::CT_BINARY,       // T_STRING
    CType::CT_STRUCT,       // T_STRUCT
    CType::CT_MAP,          // T_MAP
    CType::CT_SET,          // T_SET
    CType::CT_LIST,         // T_LIST
    CType::CT_VOID,         // unused
    CType::CT_VOID,         // unused
    CType::CT_VOID,         // unused
    CType::CT_FLOAT,        // T_FLOAT
};

constexpr std::array<TType, (size_t)SingletonId::kNumSingletonIds>
        SingletonIdToTType = {
            TType::T_BYTE,   // kTypes
            TType::T_I16,    // kFieldDeltas
            TType::T_U32,    // kLengths
            TType::T_BOOL,   // kBool
            TType::T_BYTE,   // kInt8
            TType::T_I16,    // kInt16
            TType::T_I32,    // kInt32
            TType::T_I64,    // kInt64
            TType::T_FLOAT,  // kFloat32
            TType::T_DOUBLE, // kFloat64
            TType::T_BYTE,   // kBinary
            TType::T_BYTE,   // kConfig
        };

struct TTypeInfo {
    ZL_Type ztype;
    size_t width;
};

constexpr TTypeInfo getTypeInfo(TType ttype, int formatVersion)
{
    // Please delete this branch once the old format is deprecated.
    static_assert(ZL_MIN_FORMAT_VERSION < kMinFormatVersionStringVSF);
    if (ttype == TType::T_STRING
        && formatVersion < kMinFormatVersionStringVSF) {
        return { .ztype = ZL_Type_serial, .width = 1 };
    }

    switch (ttype) {
        case TType::T_BOOL:
        case TType::T_BYTE:
            return { .ztype = ZL_Type_serial, .width = 1 };
        case TType::T_I16:
        case TType::T_U16:
            return { .ztype = ZL_Type_numeric, .width = 2 };
        case TType::T_U32:
        case TType::T_I32:
        case TType::T_FLOAT:
            return { .ztype = ZL_Type_numeric, .width = 4 };
        case TType::T_U64:
        case TType::T_I64:
        case TType::T_DOUBLE:
            return { .ztype = ZL_Type_numeric, .width = 8 };
        case TType::T_STRING:
            return { .ztype = ZL_Type_string, .width = 1 };
        default:
            throw std::runtime_error{
                "typeInfo() is only defined for primitive TTypes!"
            };
    }
}

struct StringType {};

struct AnyType {};

} // namespace zstrong::thrift

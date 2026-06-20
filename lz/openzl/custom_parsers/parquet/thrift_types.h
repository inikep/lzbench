// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <stdexcept>

namespace zstrong {
namespace parquet {

enum class TType : uint8_t {
    T_STOP   = 0,
    T_VOID   = 1,
    T_BOOL   = 2,
    T_BYTE   = 3,
    T_DOUBLE = 4,
    T_I16    = 6,
    T_I32    = 8,
    T_U64    = 9,
    T_I64    = 10,
    T_STRING = 11,
    T_STRUCT = 12,
    T_MAP    = 13,
    T_SET    = 14,
    T_LIST   = 15,
    T_UUID   = 16,
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
    CT_STRUCT        = 0x0C
};

inline TType getTType(uint8_t ctype)
{
    switch ((CType)ctype) {
        case CType::CT_STOP:
            return TType::T_STOP;
        case CType::CT_BOOLEAN_FALSE:
        case CType::CT_BOOLEAN_TRUE:
            return TType::T_BOOL;
        case CType::CT_BYTE:
            return TType::T_BYTE;
        case CType::CT_I16:
            return TType::T_I16;
        case CType::CT_I32:
            return TType::T_I32;
        case CType::CT_I64:
            return TType::T_I64;
        case CType::CT_DOUBLE:
            return TType::T_DOUBLE;
        case CType::CT_BINARY:
            return TType::T_STRING;
        case CType::CT_LIST:
            return TType::T_LIST;
        case CType::CT_SET:
            return TType::T_SET;
        case CType::CT_MAP:
            return TType::T_MAP;
        case CType::CT_STRUCT:
            return TType::T_STRUCT;
        default:
            throw std::runtime_error("Invalid CType!");
    }
}
} // namespace parquet
} // namespace zstrong

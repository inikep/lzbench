// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stack>
#include <stdexcept>
#include <string>
#include "custom_parsers/parquet/thrift_types.h"
#include "openzl/shared/varint.h"
#include "openzl/zl_errors.h"

namespace zstrong {
namespace parquet {
/**
 * The maximum depth of the thrift structure. We don't need this to be very
 * large in order to handle parquet format/metadata.
 */
constexpr uint32_t kMaxDepth = 100;

class ThriftCompactReader {
   public:
    ThriftCompactReader(const char* src, size_t srcSize)
            : currPtr_(src), srcEnd_(src + srcSize)
    {
    }

    uint32_t readStructBegin();
    uint32_t readStructEnd();
    uint32_t readFieldBegin(TType& fieldType, int16_t& fieldId);
    uint32_t readMapBegin(TType& keyType, TType& valType, uint32_t& size);
    uint32_t readMapEnd();
    uint32_t readListBegin(TType& elemType, uint32_t& size);
    uint32_t readListEnd();
    uint32_t readSetBegin(TType& elemType, uint32_t& size);
    uint32_t readSetEnd();
    uint32_t readBool(bool& value);
    uint32_t readByte(int8_t& byte);
    uint32_t readI16(int16_t& i16);
    uint32_t readI32(int32_t& i32);
    uint32_t readI64(int64_t& i64);
    uint32_t readDouble(double& value);
    uint32_t readString(std::string& str);
    uint32_t readBinary(std::string& str);

    /**
     * Skips the current field of the given type.
     */
    uint32_t skip(TType fieldType);

    size_t getRemaining()
    {
        return srcEnd_ - currPtr_;
    }

   private:
    /**
     * The current read location and the end of the source buffer.
     */
    const char* currPtr_;
    const char* srcEnd_;

    /**
     * Used to keep track of the current bool value read from the field type.
     */
    bool hasBoolVal_ = false;
    bool boolVal_    = false;

    /**
     * Used to keep track of the last field for the current and previous
     * structs.
     */
    int16_t lastFieldId_ = 0;
    std::stack<int16_t> lastField_;

    template <typename T>
    uint32_t readVarInt(T& i)
    {
        static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
        auto oldCurr = currPtr_;
        // varint decode
        auto ptr = (const uint8_t*)currPtr_;
        auto res = ZL_varintDecode(&ptr, (const uint8_t*)srcEnd_);
        if (ZL_RES_isError(res)) {
            throw std::runtime_error("Remaining buffer too small!");
        }
        currPtr_ = (const char*)ptr;
        i        = (T)ZL_RES_value(res);
        return currPtr_ - oldCurr;
    }

    template <typename T>
    uint32_t readInt(T& i)
    {
        static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
        auto read = readVarInt(i);
        // zigzag decode
        i = (T)((i >> 1) ^ -(i & 0x1));
        return read;
    }

    void descend()
    {
        if (!--height_) {
            throw std::runtime_error("Exceeded max depth!");
        }
    }

    void ascend()
    {
        ++height_;
    }

   private:
    size_t height_ = kMaxDepth;
};

} // namespace parquet
} // namespace zstrong

// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "thrift_compact_reader.h"
#include <stdexcept>
#include "openzl/shared/mem.h"

namespace zstrong {
namespace parquet {

uint32_t ThriftCompactReader::readStructBegin()
{
    descend();
    lastField_.push(lastFieldId_);
    lastFieldId_ = 0;

    return 0;
}

uint32_t ThriftCompactReader::readStructEnd()
{
    ascend();
    lastFieldId_ = lastField_.top();
    lastField_.pop();

    return 0;
}

uint32_t ThriftCompactReader::readFieldBegin(TType& fieldType, int16_t& fieldId)
{
    uint32_t read = 0;
    int8_t byte   = 0;

    read += readByte(byte);

    int8_t type = byte & 0x0F;

    if (type == (int8_t)CType::CT_STOP) {
        fieldType = TType::T_STOP;
        fieldId   = 0;
        return read;
    }

    // mask off the 4 MSB of the type header. it could contain a field id delta.
    auto modifier = (int16_t)(((uint8_t)byte & 0xF0) >> 4);
    if (modifier == 0) {
        read += readInt(fieldId);
    } else {
        fieldId = (int16_t)(lastFieldId_ + modifier);
    }
    fieldType = getTType(type);

    if (type == (uint8_t)CType::CT_BOOLEAN_TRUE
        || type == (uint8_t)CType::CT_BOOLEAN_FALSE) {
        hasBoolVal_ = true;
        boolVal_    = type == (uint8_t)CType::CT_BOOLEAN_TRUE;
    }

    lastFieldId_ = fieldId;
    return read;
}

/**
 * Read a map header off the wire. If the size is zero, skip reading the key
 * and value type.
 */
uint32_t ThriftCompactReader::readMapBegin(
        TType& keyType,
        TType& valType,
        uint32_t& size)
{
    descend();
    uint32_t read   = 0;
    int8_t kvType   = 0;
    int32_t mapSize = 0;

    read += readVarInt(mapSize);
    if (mapSize < 0) {
        throw std::runtime_error("Negative size!");
    }

    if (mapSize != 0) {
        read += readByte(kvType);
    }

    keyType = getTType(static_cast<int8_t>(static_cast<uint8_t>(kvType) >> 4));
    valType = getTType(static_cast<int8_t>(static_cast<uint8_t>(kvType) & 0xf));
    size    = (uint32_t)mapSize;

    // TODO: maybe check size available
    return read;
};

uint32_t ThriftCompactReader::readMapEnd()
{
    ascend();
    return 0;
};

/**
 * Read a list header off the wire. If the list size is 0-14, the size will
 * be packed with the element type. Otherwise, the 4 MSB
 * of the element type header will be 0xF, and a varint will follow with the
 * true size.
 */
uint32_t ThriftCompactReader::readListBegin(TType& elemType, uint32_t& size)
{
    descend();
    int8_t sizeType  = 0;
    uint32_t read    = 0;
    int32_t listSize = 0;
    read += readByte(sizeType);

    listSize = (uint8_t)sizeType >> 4;
    if (listSize == 15) {
        read += readVarInt(listSize);
    }

    if (listSize < 0) {
        throw std::runtime_error("Negative size!");
    }

    elemType = getTType(static_cast<int8_t>(sizeType & 0x0f));
    size     = (uint32_t)listSize;

    // TODO: maybe check size available
    return read;
};

uint32_t ThriftCompactReader::readListEnd()
{
    ascend();
    return 0;
};

uint32_t ThriftCompactReader::readSetBegin(TType& elemType, uint32_t& size)
{
    return readListBegin(elemType, size);
};

uint32_t ThriftCompactReader::readSetEnd()
{
    return readListEnd();
};

uint32_t ThriftCompactReader::readBool(bool& val)
{
    if (hasBoolVal_) {
        val         = boolVal_;
        hasBoolVal_ = false;
        return 0;
    } else {
        int8_t i8 = 0;
        readByte(i8);
        val = (i8 == (int8_t)CType::CT_BOOLEAN_TRUE);
        return 1;
    }
}

uint32_t ThriftCompactReader::readByte(int8_t& byte)
{
    if (getRemaining() < sizeof(int8_t)) {
        throw std::runtime_error("Remaining buffer too small!");
    }
    memcpy(&byte, currPtr_, sizeof(int8_t));
    currPtr_++;
    return sizeof(int8_t);
}

uint32_t ThriftCompactReader::readI16(int16_t& i16)
{
    return readInt(i16);
}

uint32_t ThriftCompactReader::readI32(int32_t& i32)
{
    return readInt(i32);
}

uint32_t ThriftCompactReader::readI64(int64_t& i64)
{
    return readInt(i64);
}

uint32_t ThriftCompactReader::readDouble(double& value)
{
    static_assert(sizeof(double) == sizeof(uint64_t));
    if (getRemaining() < sizeof(double)) {
        throw std::runtime_error("Remaining buffer too small!");
    }
    value = (double)ZL_readLE64(currPtr_);
    currPtr_ += sizeof(uint64_t);
    return sizeof(uint64_t);
}

uint32_t ThriftCompactReader::readBinary(std::string& str)
{
    int32_t read  = 0;
    uint32_t size = 0;

    read += readVarInt(size);

    // handle empty string
    if (size == 0) {
        str = "";
        return read;
    }
    if (size < 0) {
        throw std::runtime_error("Negative size!");
    }
    if (getRemaining() < size) {
        throw std::runtime_error("Remaining buffer too small!");
    }

    // TODO: limit string size / use heap?
    str.resize(size);
    memcpy(str.data(), currPtr_, size);
    currPtr_ += size;

    return read + (uint32_t)size;
}

uint32_t ThriftCompactReader::readString(std::string& str)
{
    return readBinary(str);
}

uint32_t ThriftCompactReader::skip(TType type)
{
    switch (type) {
        case TType::T_BYTE: {
            int8_t bytev = 0;
            return readByte(bytev);
        }
        case TType::T_BOOL: {
            bool b = false;
            return readBool(b);
        }
        case TType::T_I16: {
            int16_t i16 = 0;
            return readI16(i16);
        }
        case TType::T_I32: {
            int32_t i32 = 0;
            return readI32(i32);
        }
        case TType::T_I64: {
            int64_t i64 = 0;
            return readI64(i64);
        }
        case TType::T_DOUBLE: {
            double d = 0;
            return readDouble(d);
        }
        case TType::T_STRING: {
            std::string str;
            return readBinary(str);
        }
        case TType::T_STRUCT: {
            uint32_t result = 0;
            int16_t fid     = 0;
            TType ftype{};
            result += readStructBegin();
            while (true) {
                result += readFieldBegin(ftype, fid);
                if (ftype == TType::T_STOP) {
                    break;
                }
                result += skip(ftype);
            }
            result += readStructEnd();
            return result;
        }
        case TType::T_MAP: {
            uint32_t result = 0;
            TType keyType{};
            TType valType{};
            uint32_t size = 0;
            result += readMapBegin(keyType, valType, size);
            for (uint32_t i = 0; i < size; i++) {
                result += skip(keyType);
                result += skip(valType);
            }
            result += readMapEnd();
            return result;
        }
        case TType::T_SET: {
            uint32_t result = 0;
            TType elemType{};
            uint32_t size = 0;
            result += readSetBegin(elemType, size);
            for (uint32_t i = 0; i < size; i++) {
                result += skip(elemType);
            }
            result += readSetEnd();
            return result;
        }
        case TType::T_LIST: {
            uint32_t result = 0;
            TType elemType{};
            uint32_t size = 0;
            result += readListBegin(elemType, size);
            for (uint32_t i = 0; i < size; i++) {
                result += skip(elemType);
            }
            result += readListEnd();
            return result;
        }
        case TType::T_STOP:
        case TType::T_VOID:
        case TType::T_UUID:
        case TType::T_U64:
        default:
            throw std::runtime_error("Unimplemented!");
    }
}

} // namespace parquet
} // namespace zstrong

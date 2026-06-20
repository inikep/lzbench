// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "custom_transforms/thrift/thrift_types.h" // @manual

namespace zstrong::thrift {

const std::string& thriftTypeToString(TType type)
{
    switch (type) {
        case TType::T_STOP: {
            static const std::string name = "T_STOP";
            return name;
        }
        case TType::T_VOID: {
            static const std::string name = "T_VOID";
            return name;
        }
        case TType::T_BOOL: {
            static const std::string name = "T_BOOL";
            return name;
        }
        case TType::T_BYTE: {
            static const std::string name = "T_BYTE";
            return name;
        }
        case TType::T_DOUBLE: {
            static const std::string name = "T_DOUBLE";
            return name;
        }
        case TType::T_U16: {
            static const std::string name = "T_U16";
            return name;
        }
        case TType::T_I16: {
            static const std::string name = "T_I16";
            return name;
        }
        case TType::T_U32: {
            static const std::string name = "T_U32";
            return name;
        }
        case TType::T_I32: {
            static const std::string name = "T_I32";
            return name;
        }
        case TType::T_U64: {
            static const std::string name = "T_U64";
            return name;
        }
        case TType::T_I64: {
            static const std::string name = "T_I64";
            return name;
        }
        case TType::T_STRING: {
            static const std::string name = "T_STRING";
            return name;
        }
        case TType::T_STRUCT: {
            static const std::string name = "T_STRUCT";
            return name;
        }
        case TType::T_MAP: {
            static const std::string name = "T_MAP";
            return name;
        }
        case TType::T_SET: {
            static const std::string name = "T_SET";
            return name;
        }
        case TType::T_LIST: {
            static const std::string name = "T_LIST";
            return name;
        }
        case TType::T_UTF8: {
            static const std::string name = "T_UTF8";
            return name;
        }
        case TType::T_UTF16: {
            static const std::string name = "T_UTF16";
            return name;
        }
        case TType::T_STREAM: {
            static const std::string name = "T_STREAM";
            return name;
        }
        case TType::T_FLOAT: {
            static const std::string name = "T_FLOAT";
            return name;
        }
    }
    {
        static const std::string name = "T_UNKNOWN";
        return name;
    }
}

} // namespace zstrong::thrift

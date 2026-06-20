// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_conversion.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/Exception.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
namespace detail {
constexpr NodeMetadata<1, 1>
conversionMetadata(Type src, Type dst, const char* description)
{
    return { .inputs           = { InputMetadata{ .type = src } },
             .singletonOutputs = { OutputMetadata{ .type = dst,
                                                   .name = "converted" } },
             .description      = description };
}

template <typename ConvertT>
class ConvertNode : public Node {
   public:
    ConvertNode() = default;

    NodeID baseNode() const override
    {
        return ConvertT::node;
    }

    GraphID operator()(Compressor& compressor, GraphID converted) const
    {
        return buildGraph(compressor, { &converted, 1 });
    }

    ~ConvertNode() override = default;
};
} // namespace detail

class ConvertStructToSerial
        : public detail::ConvertNode<ConvertStructToSerial> {
   public:
    static constexpr NodeID node = ZL_NODE_CONVERT_STRUCT_TO_SERIAL;

    static constexpr NodeMetadata<1, 1> metadata = detail::conversionMetadata(
            Type::Struct,
            Type::Serial,
            "Convert struct to serial");
};

class ConvertSerialToStruct
        : public detail::ConvertNode<ConvertSerialToStruct> {
   public:
    static constexpr NodeID node = ZL_NODE_CONVERT_SERIAL_TO_STRUCT;

    static constexpr NodeMetadata<1, 1> metadata = detail::conversionMetadata(
            Type::Serial,
            Type::Struct,
            "Convert a serial input to a struct output with the given struct size");

    explicit ConvertSerialToStruct(int structSizeBytes)
            : structSizeBytes_(structSizeBytes)
    {
    }

    poly::optional<NodeParameters> parameters() const override
    {
        if (structSizeBytes_ <= 0) {
            throw Exception(
                    "Bad struct size: " + std::to_string(structSizeBytes_));
        }
        LocalParams params;
        params.addIntParam(ZL_trlip_tokenSize, structSizeBytes_);
        return NodeParameters{ .localParams = std::move(params) };
    }

   private:
    int structSizeBytes_;
};

class ConvertNumToSerialLE : public detail::ConvertNode<ConvertNumToSerialLE> {
   public:
    static constexpr NodeID node = ZL_NODE_CONVERT_NUM_TO_SERIAL;

    static constexpr NodeMetadata<1, 1> metadata = detail::conversionMetadata(
            Type::Numeric,
            Type::Serial,
            "Convert numeric to serial in little-endian format");
};

class ConvertSerialToNum8 : public detail::ConvertNode<ConvertSerialToNum8> {
   public:
    static constexpr NodeID node = ZL_NODE_CONVERT_SERIAL_TO_NUM8;

    static constexpr NodeMetadata<1, 1> metadata = detail::conversionMetadata(
            Type::Serial,
            Type::Numeric,
            "Convert serial input of 8-bit data to numeric output");
};

class ConvertSerialToNumLE16
        : public detail::ConvertNode<ConvertSerialToNumLE16> {
   public:
    static constexpr NodeID node = ZL_NODE_CONVERT_SERIAL_TO_NUM_LE16;

    static constexpr NodeMetadata<1, 1> metadata = detail::conversionMetadata(
            Type::Serial,
            Type::Numeric,
            "Convert serial input of little-endian 16-bit data to numeric output");
};

class ConvertSerialToNumLE32
        : public detail::ConvertNode<ConvertSerialToNumLE32> {
   public:
    static constexpr NodeID node = ZL_NODE_CONVERT_SERIAL_TO_NUM_LE32;

    static constexpr NodeMetadata<1, 1> metadata = detail::conversionMetadata(
            Type::Serial,
            Type::Numeric,
            "Convert serial input of little-endian 32-bit data to numeric output");
};

class ConvertSerialToNumLE64
        : public detail::ConvertNode<ConvertSerialToNumLE64> {
   public:
    static constexpr NodeID node = ZL_NODE_CONVERT_SERIAL_TO_NUM_LE64;

    static constexpr NodeMetadata<1, 1> metadata = detail::conversionMetadata(
            Type::Serial,
            Type::Numeric,
            "Convert serial input of little-endian 64-bit data to numeric output");
};

class ConvertSerialToNumLE : public Node {
   public:
    explicit ConvertSerialToNumLE(int intSizeBytes)
            : baseNode_(getNode(intSizeBytes))
    {
    }

    NodeID baseNode() const override
    {
        return baseNode_;
    }

    GraphID operator()(Compressor& compressor, GraphID converted)
    {
        return buildGraph(compressor, { &converted, 1 });
    }

    ~ConvertSerialToNumLE() override = default;

   private:
    static NodeID getNode(int intSizeBytes)
    {
        switch (intSizeBytes) {
            case 1:
                return ConvertSerialToNum8::node;
            case 2:
                return ConvertSerialToNumLE16::node;
            case 4:
                return ConvertSerialToNumLE32::node;
            case 8:
                return ConvertSerialToNumLE64::node;
            default:
                throw Exception(
                        "Bad int size: " + std::to_string(intSizeBytes));
        }
    }
    NodeID baseNode_;
};

struct ConvertSerialToNumBE16
        : public detail::ConvertNode<ConvertSerialToNumBE16> {
   public:
    static constexpr NodeID node = ZL_NODE_CONVERT_SERIAL_TO_NUM_BE16;

    static constexpr NodeMetadata<1, 1> metadata = detail::conversionMetadata(
            Type::Serial,
            Type::Numeric,
            "Convert serial input of big-endian 16-bit data to numeric output");
};

class ConvertSerialToNumBE32
        : public detail::ConvertNode<ConvertSerialToNumBE32> {
   public:
    static constexpr NodeID node = ZL_NODE_CONVERT_SERIAL_TO_NUM_BE32;

    static constexpr NodeMetadata<1, 1> metadata = detail::conversionMetadata(
            Type::Serial,
            Type::Numeric,
            "Convert serial input of big-endian 32-bit data to numeric output");
};

class ConvertSerialToNumBE64
        : public detail::ConvertNode<ConvertSerialToNumBE64> {
   public:
    static constexpr NodeID node = ZL_NODE_CONVERT_SERIAL_TO_NUM_BE64;

    static constexpr NodeMetadata<1, 1> metadata = detail::conversionMetadata(
            Type::Serial,
            Type::Numeric,
            "Convert serial input of big-endian 64-bit data to numeric output");
};

class ConvertSerialToNumBE : public Node {
   public:
    explicit ConvertSerialToNumBE(int intSizeBytes)
            : baseNode_(getNode(intSizeBytes))
    {
    }

    NodeID baseNode() const override
    {
        return baseNode_;
    }

    GraphID operator()(Compressor& compressor, GraphID converted)
    {
        return buildGraph(compressor, { &converted, 1 });
    }

   private:
    static NodeID getNode(int intSizeBytes)
    {
        switch (intSizeBytes) {
            case 1:
                return ConvertSerialToNum8::node;
            case 2:
                return ConvertSerialToNumBE16::node;
            case 4:
                return ConvertSerialToNumBE32::node;
            case 8:
                return ConvertSerialToNumBE64::node;
            default:
                throw Exception(
                        "Bad int size: " + std::to_string(intSizeBytes));
        }
    }

    NodeID baseNode_;
};

class ConvertNumToStructLE : public detail::ConvertNode<ConvertNumToStructLE> {
   public:
    static constexpr NodeID node = ZL_NODE_CONVERT_NUM_TO_STRUCT_LE;

    static constexpr NodeMetadata<1, 1> metadata = detail::conversionMetadata(
            Type::Numeric,
            Type::Struct,
            "Convert numeric input to a little-endian fixed-size struct output");
};

class ConvertStructToNumLE : public detail::ConvertNode<ConvertStructToNumLE> {
   public:
    static constexpr NodeID node = ZL_NODE_CONVERT_STRUCT_TO_NUM_LE;

    static constexpr NodeMetadata<1, 1> metadata = detail::conversionMetadata(
            Type::Struct,
            Type::Numeric,
            "Convert little-endian fixed-size struct input to numeric output");
};

class ConvertStructToNumBE : public detail::ConvertNode<ConvertStructToNumBE> {
   public:
    static constexpr NodeID node = ZL_NODE_CONVERT_STRUCT_TO_NUM_BE;

    static constexpr NodeMetadata<1, 1> metadata = detail::conversionMetadata(
            Type::Struct,
            Type::Numeric,
            "Convert big-endian fixed-size struct input to numeric output");
};

class ConvertSerialToString : public Node {
   public:
    static constexpr NodeMetadata<1, 1> metadata = detail::conversionMetadata(
            Type::Serial,
            Type::String,
            "Convert a serial input to a string output by telling OpenZL the string lengths");

    /// @warning @p stringLens must out-live the usage of this class.
    explicit ConvertSerialToString(poly::span<const uint32_t> stringLens)
            : stringLens_(stringLens)
    {
    }

    NodeID baseNode() const override
    {
        throw Exception("ConvertSerialToString: Only run() is supported!");
    }

    virtual Edge::RunNodeResult run(Edge& edge) const override
    {
        auto edges = unwrap(ZL_Edge_runConvertSerialToStringNode(
                edge.get(), stringLens_.data(), stringLens_.size()));
        return Edge::convert(edges);
    }

    ~ConvertSerialToString() override = default;

   private:
    poly::span<const uint32_t> stringLens_;
};

class SeparateStringComponents : public Node {
   public:
    static constexpr NodeID node = ZL_NODE_SEPARATE_STRING_COMPONENTS;

    static constexpr NodeMetadata<1, 2> metadata = {
        .inputs = { InputMetadata{ .type = Type::String, .name = "strings"}},
        .singletonOutputs = {
            OutputMetadata{ .type = Type::Serial, .name = "string content"},
            OutputMetadata{ .type = Type::Numeric, .name = "32-bit string lengths"},
        },
        .description = "Separate a string input into its content and lengths streams",
    };

    SeparateStringComponents() = default;

    NodeID baseNode() const override
    {
        return node;
    }

    GraphID operator()(Compressor& compressor, GraphID content, GraphID lengths)
            const
    {
        return buildGraph(
                compressor, std::initializer_list<GraphID>{ content, lengths });
    }

    ~SeparateStringComponents() override = default;
};

} // namespace nodes
} // namespace openzl

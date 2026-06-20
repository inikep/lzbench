// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openzl/openzl.hpp"
#include "openzl/zl_compressor.h"
#include "openzl/zl_ctransform.h"
#include "openzl/zl_data.h"
#include "openzl/zl_decompress.h"
#include "openzl/zl_dtransform.h"
#include "openzl/zl_opaque_types.h"
#include "openzl/zl_selector.h"

namespace zstrong {
using TypedRef    = openzl::Input;
using TypedBuffer = openzl::Output;

/**
 * Unwraps the report and throws an exception if it contains an error.
 *
 * Note: prefer ContextObject::unwrap if possible, since only those methods
 * include a stack trace in the error message.
 */
using openzl::unwrap;

class CCtx : public openzl::CCtx {
   public:
    const ZL_CCtx& operator*() const
    {
        return *get();
    }
    ZL_CCtx& operator*()
    {
        return *get();
    }
};

class DCtx : public openzl::DCtx {
   public:
    const ZL_DCtx& operator*() const
    {
        return *get();
    }
    ZL_DCtx& operator*()
    {
        return *get();
    }
};

class CGraph : public openzl::Compressor {
   public:
    std::string compress(
            std::string_view data,
            std::optional<std::unordered_map<ZL_CParam, int>> const&
                    globalParams = std::nullopt);
    std::string compressMulti(
            std::vector<std::string_view> const& data,
            std::optional<std::unordered_map<ZL_CParam, int>> const&
                    globalParams = std::nullopt);
    std::string decompress(std::string_view data);
    std::vector<std::string> decompressMulti(std::string_view data);

    const ZL_Compressor& operator*() const
    {
        return *get();
    }
    ZL_Compressor& operator*()
    {
        return *get();
    }
};

/// C++ wrapper for local parameters that owns the storage
class LocalParams {
   public:
    LocalParams() {}
    explicit LocalParams(ZL_LocalParams params) : params_(params) {}

    void push_back(ZL_IntParam param)
    {
        params_.addIntParam(param);
    }

    void push_back(ZL_CopyParam param)
    {
        params_.addCopyParam(param);
    }

    void push_back(ZL_RefParam param)
    {
        params_.addRefParam(param);
    }

    ZL_LocalParams const& operator*() const
    {
        return *params_;
    }

    ZL_LocalParams const* operator->() const
    {
        return params_.get();
    }

    std::span<ZL_CopyParam const> genericParams() const
    {
        return params_.getCopyParams();
    }

    std::span<ZL_RefParam const> refParams() const
    {
        return params_.getRefParams();
    }

    std::span<ZL_IntParam const> intParams() const
    {
        return params_.getIntParams();
    }

   private:
    openzl::LocalParams params_;
};

/// C++ wrapper for typed graph desc that owns the storage
class MIGraphDesc {
   public:
    explicit MIGraphDesc(
            ZL_MIGraphDesc desc,
            std::vector<ZL_Type> inStreamTypes,
            std::vector<ZL_Type> outStreamTypes)
            : desc_(desc),
              inTypeStorage_(std::move(inStreamTypes)),
              outTypeStorage_(std::move(outStreamTypes))
    {
        desc_.inputTypes = inTypeStorage_.data();
        assert(inTypeStorage_.size() == desc.nbInputs);
        desc_.soTypes = outTypeStorage_.data();
        desc_.voTypes = outTypeStorage_.data() + desc.nbSOs;
        assert(outTypeStorage_.size() == desc.nbSOs + desc.nbVOs);
    }

    MIGraphDesc(MIGraphDesc const&) = delete;
    MIGraphDesc(MIGraphDesc&&)      = default;

    MIGraphDesc& operator=(MIGraphDesc const&) = delete;
    MIGraphDesc& operator=(MIGraphDesc&&)      = default;

    ZL_MIGraphDesc const& operator*() const
    {
        return desc_;
    }

    ZL_MIGraphDesc const* operator->() const
    {
        return &desc_;
    }

   private:
    ZL_MIGraphDesc desc_;
    std::vector<ZL_Type> inTypeStorage_;
    std::vector<ZL_Type> outTypeStorage_;
};

/// Base class for Zstrong transforms.
class Transform {
   public:
    virtual ZL_GraphID registerTransform(
            ZL_Compressor& cgraph,
            std::span<ZL_GraphID const> successors,
            ZL_LocalParams const& params) const = 0;

    virtual void registerTransform(ZL_DCtx& dctx) const = 0;

    virtual size_t nbInputs() const = 0;

    virtual size_t nbSuccessors() const = 0;

    virtual ZL_Type inputType(size_t idx) const = 0;

    virtual ZL_Type outputType(size_t idx) const = 0;

    virtual std::string description() const
    {
        return "";
    }

    virtual std::string successorName(size_t idx) const
    {
        return "successor" + std::to_string(idx);
    }

    virtual size_t nbVariableSuccessors() const
    {
        return 0;
    }

    size_t nbFixedSuccessors() const
    {
        return nbSuccessors() - nbVariableSuccessors();
    }

    virtual ~Transform() {}
};

/// Base class for Zstrong graphs
class Graph {
   public:
    virtual ZL_GraphID registerGraph(ZL_Compressor& cgraph) const = 0;

    virtual void registerGraph(ZL_DCtx& dctx) const = 0;

    virtual ZL_Type inputType() const = 0;

    virtual std::string description() const
    {
        return "";
    }

    virtual ~Graph() {}
};

/// Base class for Zstrong selectors
class Selector {
   public:
    virtual ZL_GraphID registerSelector(
            ZL_Compressor& cgraph,
            std::span<ZL_GraphID const> successors,
            ZL_LocalParams const& localParams) const = 0;

    virtual std::optional<size_t> expectedNbSuccessors() const
    {
        return std::nullopt;
    }

    virtual ZL_Type inputType() const = 0;

    virtual std::string description() const
    {
        return "";
    }

    virtual ~Selector() {}
};

/// Helper class for Zstrong custom selectors that allow you to implement a
/// selector as a C++ virtual function and get access to `this`.
class CustomSelector : public Selector {
   public:
    virtual ZL_GraphID select(
            ZL_Selector const* selCtx,
            ZL_Input const* input,
            std::span<ZL_GraphID const> successors) const = 0;

    ZL_GraphID registerSelector(
            ZL_Compressor& cgraph,
            std::span<ZL_GraphID const> successors,
            ZL_LocalParams const& localParams = {}) const override;
};

/// Helper function for registereing Selector instances which are owned
/// by the graph using smart pointers (note the definition of ownership here
/// is a bit murky as this can be shared ownership).
ZL_GraphID registerOwnedSelector(
        ZL_Compressor& cgraph,
        std::shared_ptr<CustomSelector const> selector,
        std::span<ZL_GraphID const> successors,
        const ZL_LocalParams& originalLocalParams = {},
        std::string_view name                     = "");

/// Helper class for Zstrong custom transforms that allow you to implement a
/// transform as C++ virtual functions and get access to `this`.
class CustomTransform : public Transform {
   public:
    explicit CustomTransform(ZL_IDType transformID) : transformID_(transformID)
    {
    }

    ZL_Report encode(ZL_Encoder* eictx, ZL_Input const* input);

    virtual ZL_Report encode(
            ZL_Encoder* eictx,
            ZL_Input const* inputs[],
            size_t nbInputs) const = 0;

    virtual ZL_Report decode(ZL_Decoder* dictx, ZL_Input const* inputs[]) const;

    virtual ZL_Report decode(
            ZL_Decoder* dictx,
            ZL_Input const* fixedInputs[],
            size_t nbFixed,
            ZL_Input const* voInputs[],
            size_t nbVO) const;

    ZL_GraphID registerTransform(
            ZL_Compressor& cgraph,
            std::span<ZL_GraphID const> successors,
            ZL_LocalParams const& localParams) const override;

    void registerTransform(ZL_DCtx& graph) const override;

   protected:
    MIGraphDesc graphDesc() const;

   private:
    ZL_IDType transformID_;
};

struct ParamInfo {
    int key;
    std::string name;
    std::string docs;
};

class ParameterizedTransform : public Transform {
   public:
    /// @returns The required int params for the transform.
    virtual std::vector<ParamInfo> const& intParams() const = 0;

    /// @returns The required generic params for the transform.
    virtual std::vector<ParamInfo> const& genericParams() const = 0;
};

using TransformMap =
        std::unordered_map<std::string, std::unique_ptr<Transform>>;
using ParameterizedTransformMap = std::
        unordered_map<std::string, std::unique_ptr<ParameterizedTransform>>;
using GraphMap    = std::unordered_map<std::string, std::unique_ptr<Graph>>;
using SelectorMap = std::unordered_map<std::string, std::unique_ptr<Selector>>;

/// @returns A map from name -> standard transform
ParameterizedTransformMap const& getStandardTransforms();
/// @returns A map from name -> standard graph
GraphMap const& getStandardGraphs();
/// @returns A map from name -> standard selector
SelectorMap const& getStandardSelectors();

// Compress from a Graph
std::string compress(
        std::string_view data,
        Graph const& graph,
        std::optional<std::unordered_map<ZL_CParam, int>> const& globalParams =
                std::nullopt);
std::string compressMulti(
        const std::vector<std::string_view>& data,
        Graph const& graph,
        std::optional<std::unordered_map<ZL_CParam, int>> const& globalParams =
                std::nullopt);

// Compress from a GraphID
std::string compress(
        std::string_view data,
        ZL_GraphID const graphID,
        std::optional<std::unordered_map<ZL_CParam, int>> const& globalParams =
                std::nullopt);

// Compress from a CCtx
std::string compress(
        CCtx& cctx,
        std::string_view data,
        CGraph const& cgraph,
        std::optional<std::unordered_map<ZL_CParam, int>> const& globalParams =
                std::nullopt);
std::string compressMulti(
        CCtx& cctx,
        std::vector<std::string_view> data,
        CGraph const& cgraph,
        std::optional<std::unordered_map<ZL_CParam, int>> const& globalParams =
                std::nullopt);
void compress(
        CCtx& cctx,
        std::string* out,
        std::string_view data,
        CGraph const& graph,
        std::optional<std::unordered_map<ZL_CParam, int>> const& globalParams =
                std::nullopt);
void compressMulti(
        CCtx& cctx,
        std::string* out,
        std::vector<std::string_view> data,
        CGraph const& graph,
        std::optional<std::unordered_map<ZL_CParam, int>> const& globalParams =
                std::nullopt);

std::string decompress(std::string_view data);
std::string decompress(std::string_view data, Graph const& graph);
std::string decompress(
        DCtx& dctx,
        std::string_view data,
        std::optional<size_t> maxDstSize = std::nullopt);
std::vector<std::string> decompressMulti(std::string_view data);
std::vector<std::string> decompressMulti(
        std::string_view data,
        Graph const& graph);
std::vector<std::string> decompressMulti(
        DCtx& dctx,
        std::string_view data,
        std::optional<size_t> maxDstSize = std::nullopt);

size_t getHeaderSize(std::string_view data);

/// @returns decompression speed for each compressed sample, speed in Mbps
std::vector<double> measureDecompressionSpeeds(
        const std::vector<std::string_view>& compressed);
/// @returns decompression speed for each compressed sample with a given
/// custom graph, speed in Mbps
std::vector<double> measureDecompressionSpeeds(
        const std::vector<std::string_view>& compressed,
        Graph const& graph);
/// @returns decompression speed for a compressed sample, speed in Mbps
double measureDecompressionSpeed(std::string_view compressed);
/// @returns decompression speed for a compressed sample with a given custom
/// graph, speed in Mbps
double measureDecompressionSpeed(
        std::string_view compressed,
        Graph const& graph);

} // namespace zstrong

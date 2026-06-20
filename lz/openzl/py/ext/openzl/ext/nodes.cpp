// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/ext/nodes.hpp"

#include <any>
#include <memory>

#include "openzl/ext.hpp"
#include "openzl/ext/graphs.hpp"
#include "openzl/ext/utils.hpp"
#include "openzl/openzl.hpp"

namespace openzl {
namespace py {
namespace {

template <typename, typename = void>
struct HasMetadata : std::false_type {};

template <typename T>
using Void = void;

template <typename T>
struct HasMetadata<T, Void<decltype(T::metadata)>> : std::true_type {};

template <typename T>
std::string nodeDocstring()
{
    if constexpr (HasMetadata<T>::value) {
        const auto& meta = T::metadata;
        std::string docs = meta.description;
        docs += "\n\nInputs:\n";
        docs += ioDocs(meta.inputs, true);
        if (meta.lastInputIsVariable) {
            docs += "\t...";
        }
        if (!meta.singletonOutputs.empty()) {
            docs += "\n\nSingleton Outputs:\n";
            docs += ioDocs(meta.singletonOutputs, false);
        }
        if (!meta.variableOutputs.empty()) {
            docs += "\n\nVariable Outputs:\n";
            docs += ioDocs(meta.variableOutputs, false);
        }
        return docs;
    } else {
        return "";
    }
}

class PyNodeTrampoline : public PyNodeBase {
   public:
    NB_TRAMPOLINE(PyNodeBase, 6);

    std::vector<nb::ref<PyEdge>> run(nb::ref<PyEdge> edge) const override
    {
        NB_OVERRIDE(run, edge);
    }

    std::vector<nb::ref<PyEdge>> runMultiInput(
            std::vector<nb::ref<PyEdge>> edges) const override
    {
        NB_OVERRIDE_NAME("run_multi_input", runMultiInput, edges);
    }

    GraphID buildGraph(
            nb::ref<PyCompressor> compressor,
            const std::vector<GraphID>& successors) const override
    {
        NB_OVERRIDE_NAME("build_graph", buildGraph, compressor, successors);
    }

    NodeID parameterize(nb::ref<PyCompressor> compressor) const override
    {
        NB_OVERRIDE(parameterize, compressor);
    }

    NodeID baseNode() const override
    {
        NB_OVERRIDE_PURE_NAME("base_node", baseNode);
    }

    poly::optional<NodeParameters> parameters() const override
    {
        NB_OVERRIDE(parameters);
    }
};

template <typename NodeT>
class PyNode : public PyNodeBase {
   public:
    template <typename... Args>
    explicit PyNode(Args&&... args) : node_(std::forward<Args>(args)...)
    {
    }

    PyNode(const PyNode&) = default;
    PyNode(PyNode&&)      = default;

    PyNode& operator=(const PyNode&) = default;
    PyNode& operator=(PyNode&&)      = default;

    std::vector<nb::ref<PyEdge>> run(nb::ref<PyEdge> edge) const override
    {
        return PyEdge::convert(node_.run(*edge->get()));
    }

    std::vector<nb::ref<PyEdge>> runMultiInput(
            std::vector<nb::ref<PyEdge>> inputs) const override
    {
        auto inEdges = PyEdge::convert(inputs);
        return PyEdge::convert(node_.runMultiInput(inEdges));
    }

    GraphID buildGraph(
            nb::ref<PyCompressor> compressor,
            const std::vector<GraphID>& successors) const override
    {
        return node_.buildGraph(*compressor, successors);
    }

    NodeID parameterize(nb::ref<PyCompressor> compressor) const override
    {
        return node_.parameterize(*compressor);
    }

    NodeID baseNode() const override
    {
        return node_.baseNode();
    }

    poly::optional<NodeParameters> parameters() const override
    {
        return node_.parameters();
    }

    template <typename... Successors>
    GraphID call(
            nb::ref<PyCompressor> compressor,
            const Successors&... successors)
    {
        return buildGraph(
                compressor,
                std::initializer_list<GraphID>{
                        buildSuccessor(compressor, successors)... });
    }

    /// Preserve the lifetime of @p val
    template <typename T>
    void stash(T val)
    {
        stash_.emplace_back(std::move(val));
    }

    ~PyNode() override = default;

   private:
    GraphID buildSuccessor(
            nb::ref<PyCompressor>& compressor,
            const std::variant<GraphID, nb::ref<PyGraphBase>>& successor) const
    {
        if (const GraphID* ptr = std::get_if<GraphID>(&successor)) {
            return buildSuccessor(compressor, *ptr);
        } else if (
                const nb::ref<PyGraphBase>* ptr =
                        std::get_if<nb::ref<PyGraphBase>>(&successor)) {
            return buildSuccessor(compressor, *ptr);
        } else {
            throw Exception("Uknown graph type");
        }
    }

    GraphID buildSuccessor(nb::ref<PyCompressor>&, GraphID successor) const
    {
        return successor;
    }

    GraphID buildSuccessor(
            nb::ref<PyCompressor>& compressor,
            const nb::ref<PyGraphBase>& successor) const
    {
        return successor->parameterize(compressor);
    }

    NodeT node_;
    std::vector<std::any> stash_;
};

void registerNodeBaseClass(nb::module_& n)
{
    nb::class_<PyNodeBase, PyNodeTrampoline>(
            n,
            "Node",
            nb::intrusive_ptr<PyNodeBase>(
                    [](PyNodeBase* o, PyObject* po) noexcept {
                        o->set_self_py(po);
                    }))
            .def(nb::init<>())
            .def("base_node", &PyNodeBase::baseNode)
            .def("parameters", &PyNodeBase::parameters)
            .def("run", &PyNodeBase::run, nb::arg("edge"))
            .def("run_multi_input",
                 &PyNodeBase::runMultiInput,
                 nb::arg("edges"))
            .def("build_graph",
                 &PyNodeBase::buildGraph,
                 nb::arg("compressor"),
                 nb::arg("successors"))
            .def("parameterize",
                 &PyNodeBase::parameterize,
                 nb::arg("compressor"));
}

template <typename>
using ToGraph = std::variant<GraphID, nb::ref<PyGraphBase>>;

template <typename NodeT, typename... SuccessorNames>
nb::class_<PyNode<NodeT>, PyNodeBase>
registerNode(nb::module_& n, const char* name, SuccessorNames... successorNames)
{
    using N   = PyNode<NodeT>;
    auto docs = nodeDocstring<NodeT>();
    // TODO(terrelln): Expose parameters
    return nb::class_<N, PyNodeBase>(n, name, docs.c_str())
            .def("__call__",
                 &PyNode<NodeT>::template call<ToGraph<SuccessorNames>...>,
                 nb::arg("compressor"),
                 nb::arg(successorNames)...)
            .def("run", &N::run, nb::arg("edge"))
            .def("run_multi_input", &N::runMultiInput, nb::arg("edges"))
            .def("build_graph",
                 &N::buildGraph,
                 nb::arg("compressor"),
                 nb::arg("successors"))
            .def("parameterize", &N::parameterize, nb::arg("compressor"))
            .def_prop_ro("base_node", &N::baseNode);
}

template <typename NodeT, typename... SuccessorNames>
nb::class_<PyNode<NodeT>, PyNodeBase> registerSimpleNode(
        nb::module_& n,
        const char* name,
        SuccessorNames... successorNames)
{
    return registerNode<NodeT>(n, name, successorNames...).def(nb::init<>());
}

void registerBitunpackNode(nb::module_& n)
{
    registerNode<nodes::Bitunpack>(n, "Bitunpack", "successor")
            .def(nb::init<int>(), nb::arg("num_bits"));
}

void registerConcatNodes(nb::module_& n)
{
    registerSimpleNode<nodes::ConcatSerial>(
            n, "ConcatSerial", "lengths", "concatenated");
    registerSimpleNode<nodes::ConcatStruct>(
            n, "ConcatStruct", "lengths", "concatenated");
    registerSimpleNode<nodes::ConcatNumeric>(
            n, "ConcatNumeric", "lengths", "concatenated");
    registerSimpleNode<nodes::ConcatString>(
            n, "ConcatString", "lengths", "concatenated");
    registerNode<nodes::Concat>(n, "Concat", "lengths", "concatenated")
            .def(nb::init<Type>(), nb::arg("type"));
}

void registerConversionNodes(nb::module_& n)
{
    registerSimpleNode<nodes::ConvertStructToSerial>(
            n, "ConvertStructToSerial", "successor");
    registerNode<nodes::ConvertSerialToStruct>(
            n, "ConvertSerialToStruct", "successor")
            .def(nb::init<int>(), nb::arg("struct_size_bytes"));
    registerSimpleNode<nodes::ConvertNumToSerialLE>(
            n, "ConvertNumToSerialLE", "successor");
    registerSimpleNode<nodes::ConvertSerialToNum8>(
            n, "ConvertSerialToNum8", "successor");
    registerSimpleNode<nodes::ConvertSerialToNumLE16>(
            n, "ConvertSerialToNumLE16", "successor");
    registerSimpleNode<nodes::ConvertSerialToNumLE32>(
            n, "ConvertSerialToNumLE32", "successor");
    registerSimpleNode<nodes::ConvertSerialToNumLE64>(
            n, "ConvertSerialToNumLE64", "successor");
    registerSimpleNode<nodes::ConvertSerialToNumBE16>(
            n, "ConvertSerialToNumBE16", "successor");
    registerSimpleNode<nodes::ConvertSerialToNumBE32>(
            n, "ConvertSerialToNumBE32", "successor");
    registerSimpleNode<nodes::ConvertSerialToNumBE64>(
            n, "ConvertSerialToNumBE64", "successor");
    registerNode<nodes::ConvertSerialToNumLE>(
            n, "ConvertSerialToNumLE", "successor")
            .def(nb::init<int>(), nb::arg("int_size_bytes"));
    registerNode<nodes::ConvertSerialToNumBE>(
            n, "ConvertSerialToNumBE", "successor")
            .def(nb::init<int>(), nb::arg("int_size_bytes"));
    registerSimpleNode<nodes::ConvertNumToStructLE>(
            n, "ConvertNumToStructLE", "successor");
    registerSimpleNode<nodes::ConvertStructToNumLE>(
            n, "ConvertStructToNumLE", "successor");
    registerSimpleNode<nodes::ConvertStructToNumBE>(
            n, "ConvertStructToNumBE", "successor");
    registerNode<nodes::ConvertSerialToString>(
            n, "ConvertSerialToString", "successor")
            .def(
                    "__init__",
                    [](PyNode<nodes::ConvertSerialToString>* obj,
                       std::vector<uint32_t> stringLens) {
                        new (obj) PyNode<nodes::ConvertSerialToString>(
                                stringLens);
                        obj->stash(std::move(stringLens));
                    },
                    nb::arg("string_lens"));
    registerSimpleNode<nodes::SeparateStringComponents>(
            n, "SeparateStringComponents", "content", "lengths");
}

void registerDedupNode(nb::module_& n)
{
    registerSimpleNode<nodes::DedupNumeric>(n, "DedupNumeric", "successor");
}

void registerDeltaIntNode(nb::module_& n)
{
    registerSimpleNode<nodes::DeltaInt>(n, "DeltaInt", "successor");
}

void registerDispatchSerialNode(nb::module_& n)
{
    registerNode<nodes::DispatchSerial>(
            n, "DispatchSerial", "tags", "sizes", "dispatched")
            .def(
                    "__init__",
                    [](PyNode<nodes::DispatchSerial>* obj,
                       std::vector<uint32_t> segmentTags,
                       std::vector<size_t> segmentSizes,
                       unsigned numTags) {
                        new (obj) PyNode<nodes::DispatchSerial>(
                                nodes::DispatchSerial::Instructions{
                                        .segmentTags  = segmentTags,
                                        .segmentSizes = segmentSizes,
                                        .numTags      = numTags,
                                });
                        obj->stash(std::move(segmentTags));
                        obj->stash(std::move(segmentSizes));
                    },
                    nb::kw_only(),
                    nb::arg("segment_tags"),
                    nb::arg("segment_sizes"),
                    nb::arg("num_tags"));
}

void registerDispatchStringNode(nb::module_& n)
{
    registerNode<nodes::DispatchString>(
            n, "DispatchString", "tags", "dispatched")
            .def(
                    "__init__",
                    [](PyNode<nodes::DispatchString>* obj,
                       std::vector<uint16_t> tags,
                       unsigned numTags) {
                        new (obj) PyNode<nodes::DispatchString>(tags, numTags);
                        obj->stash(std::move(tags));
                    },
                    nb::kw_only(),
                    nb::arg("tags"),
                    nb::arg("num_tags"));
}

void registerDivideByNode(nb::module_& n)
{
    registerNode<nodes::DivideBy>(n, "DivideBy", "successor")
            .def(nb::init<poly::optional<uint64_t>>(),
                 nb::kw_only(),
                 nb::arg("divisor") = poly::nullopt);
}

void registerFieldLzNode(nb::module_& n)
{
    registerNode<nodes::FieldLz>(
            n,
            "FieldLz",
            "literals",
            "tokens",
            "offsets",
            "extra_literal_lengths",
            "extra_match_lengths")
            .def(nb::init<poly::optional<int>>(),
                 nb::arg("compression_level") = poly::nullopt);
}

void registerFloatDeconstructNodes(nb::module_& n)
{
    registerSimpleNode<nodes::Float32Deconstruct>(
            n, "Float32Deconstruct", "sign_frac", "exponent");
    registerSimpleNode<nodes::BFloat16Deconstruct>(
            n, "BFloat16Deconstruct", "sign_frac", "exponent");
    registerSimpleNode<nodes::Float16Deconstruct>(
            n, "Float16Deconstruct", "sign_frac", "exponent");
}

void registerMergeSortedNode(nb::module_& n)
{
    registerSimpleNode<nodes::MergeSorted>(
            n, "MergeSorted", "bitset", "sorted");
}

void registerParseIntNode(nb::module_& n)
{
    registerSimpleNode<nodes::ParseInt>(n, "ParseInt", "successor");
}

void registerPrefixNode(nb::module_& n)
{
    registerSimpleNode<nodes::Prefix>(n, "Prefix", "successor");
}

void registerQuantizeNodes(nb::module_& n)
{
    registerSimpleNode<nodes::QuantizeOffsets>(
            n, "QuantizeOffsets", "codes", "extra_bits");
    registerSimpleNode<nodes::QuantizeLengths>(
            n, "QuantizeLengths", "codes", "extra_bits");
}

void registerRangePackNode(nb::module_& n)
{
    registerSimpleNode<nodes::RangePack>(n, "RangePack", "successor");
}

template <typename SplitT>
void registerSplitNode(nb::module_& n, const char* name)
{
    registerNode<SplitT>(n, name, "successor")
            .def(
                    "__init__",
                    [](PyNode<SplitT>* obj, std::vector<size_t> segmentSizes) {
                        new (obj) PyNode<SplitT>(segmentSizes);
                        obj->stash(std::move(segmentSizes));
                    },
                    nb::kw_only(),
                    nb::arg("segment_sizes"));
}

void registerSplitNodes(nb::module_& n)
{
    registerSplitNode<nodes::SplitSerial>(n, "SplitSerial");
    registerSplitNode<nodes::SplitStruct>(n, "SplitStruct");
    registerSplitNode<nodes::SplitNumeric>(n, "SplitNumeric");
    registerSplitNode<nodes::SplitString>(n, "SplitString");
    registerSplitNode<nodes::Split>(n, "Split");
}

void registerSplitByStructNodes(nb::module_& n)
{
    // TODO(terrelln): Hook up once C++ supports it
}

void registerTokenizeNodes(nb::module_& n)
{
    registerSimpleNode<nodes::TokenizeStruct>(
            n, "TokenizeStruct", "alphabet", "indices");
    registerNode<nodes::TokenizeNumeric>(
            n, "TokenizeNumeric", "alphabet", "indices")
            .def(nb::init<bool>(), nb::kw_only(), nb::arg("sort") = false);
    registerNode<nodes::TokenizeString>(
            n, "TokenizeString", "alphabet", "indices")
            .def(nb::init<bool>(), nb::kw_only(), nb::arg("sort") = false);
    registerNode<nodes::Tokenize>(n, "Tokenize", "alphabet", "indices")
            .def(nb::init<Type, bool>(),
                 nb::kw_only(),
                 nb::arg("type"),
                 nb::arg("sort") = false);
}

void registerTransposeNode(nb::module_& n)
{
    registerSimpleNode<nodes::TransposeSplit>(n, "TransposeSplit", "successor");
}

void registerZigzagNode(nb::module_& n)
{
    registerSimpleNode<nodes::Zigzag>(n, "Zigzag", "successor");
}
} // namespace

void registerNodesModule(nb::module_& m)
{
    auto n = m.def_submodule("nodes");
    registerNodeBaseClass(n);
    registerBitunpackNode(n);
    registerConcatNodes(n);
    registerConversionNodes(n);
    registerDedupNode(n);
    registerDeltaIntNode(n);
    registerDispatchSerialNode(n);
    registerDispatchStringNode(n);
    registerDivideByNode(n);
    registerFieldLzNode(n);
    registerFloatDeconstructNodes(n);
    registerMergeSortedNode(n);
    registerParseIntNode(n);
    registerPrefixNode(n);
    registerQuantizeNodes(n);
    registerRangePackNode(n);
    registerSplitNodes(n);
    registerTokenizeNodes(n);
    registerTransposeNode(n);
    registerZigzagNode(n);
}
} // namespace py
} // namespace openzl

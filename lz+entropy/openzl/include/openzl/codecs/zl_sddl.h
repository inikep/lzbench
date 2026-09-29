// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_SDDL_H
#define OPENZL_CODECS_SDDL_H

#include "openzl/zl_errors.h"
#include "openzl/zl_opaque_types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * Builds a Simple Data Description Language graph with the provided
 * (pre-compiled) @p description and @p successor graph.
 *
 * See the SDDL page in the documentation for a complete description of this
 * component.
 *
 * ### Graph Topology
 *
 * ``` mermaid
 * flowchart TD
 *     subgraph SDDL Graph
 *         Desc([Description]);
 *         Input([Input]);
 *         Conv@{ shape: procs, label: "Type Conversions"};
 *         Engine[SDDL Engine];
 *         Inst([Instructions]);
 *         Disp[/Dispatch Transform\];
 *         Succ[Successor Graph];
 *
 *         Desc --> Engine;
 *         Input --> Engine;
 *         Engine --> Inst;
 *         Inst -->|Dispatch Instructions| Disp;
 *         Input --> Disp;
 *         Inst -->|Type Information| Conv;
 *         Disp ==>|Many Streams| Conv;
 *         Conv ==>|Many Streams| Succ;
 *     end
 *
 *     OuterInput[ZL_Input] --> Input;
 *     OuterParam[ZL_LocalCopyParam] --> Desc;
 * ```
 *
 * This graph takes a single serial input and applies the @p description to it,
 * using that description to decompose the input into fields which are mapped
 * to one or more output streams. These streams, as well as two control streams
 * are all sent to a single invocation of the @p successor graph. @p successor
 * must therefore be a multi-input graph able to accept any number of numeric
 * and serial streams (at least).
 *
 * (The control streams are: a numeric stream containing the stream indices
 * into which each field has been placed and a numeric stream containing the
 * size of each field. See also the documentation for `dispatchN_byTag` and
 * particularly, @ref ZL_Edge_runDispatchNode, which is the underlying
 * component that this graph uses to actually decompose the input, for more
 * information about the dispatch operation. These streams respectively are the
 * first and second stream passed into the successor graph, and the streams
 * into which the input has been dispatched follow, in order.)
 *
 * The streams on which the @p successor is invoked are also tagged with int
 * metadata, with key 0 set to their index. (For the moment. Future work may
 * allow for more robust/stable tagging.) This makes this graph compatible with
 * the generic clustering graph (see @ref ZL_Clustering_registerGraph), and the
 * `sddl` profile in the demo CLI, for example, is set up that way, with the
 * SDDL graph succeeded by the generic clusterer.
 *
 * ### Data Description
 *
 * This graph requires a @p description of the input format that it is intended
 * to parse and dispatch. SDDL has both a human-writeable description language
 * and a binary, compiled representation of that language. This component only
 * accepts descriptions in the binary format.
 *
 * Use @ref openzl::sddl::Compiler::compile to do that translation.
 *
 * Note that the OpenZL demo CLI can also compile SDDL descriptions, as part of
 * using the `sddl` profile.
 */
ZL_RESULT_OF(ZL_GraphID)
ZL_Compressor_buildSDDLGraph(
        ZL_Compressor* compressor,
        const void* description,
        size_t descriptionSize,
        ZL_GraphID successor);

/**
 * Graph ID for the base SDDL component.
 *
 * Note that this graph is non-functional on its own! You must parameterize it
 * with (1) a description and (2) a successor.
 *
 * Prefer using @ref ZL_Compressor_buildSDDLGraph to do that.
 */
#define ZL_GRAPH_SDDL                                       \
    (ZL_GraphID)                                            \
    {                                                       \
        ZL_StandardGraphID_simple_data_description_language \
    }

/**
 * Parameter ID at which the SDDL graph expects to receive the description it
 * will execute.
 */
#define ZL_SDDL_DESCRIPTION_PID 522

#if defined(__cplusplus)
}
#endif

#endif // OPENZL_CODECS_SDDL_H

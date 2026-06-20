// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_ENCODER_REGISTRY_H
#define ZSTRONG_TRANSFORMS_ENCODER_REGISTRY_H

#include "openzl/compress/cnode.h"          // CNode
#include "openzl/compress/compress_types.h" // NodeType_e
#include "openzl/compress/private_nodes.h"  // ZL_PrivateStandardNodeID_end
#include "openzl/shared/portability.h"
#include "openzl/zl_ctransform.h" // ZL_TypedEncoderDesc
#include "openzl/zl_selector.h"   // ZL_SelectorDesc

ZL_BEGIN_C_DECLS

#define STANDARD_ENCODERS_NB ZL_PrivateStandardNodeID_end
extern const CNode ER_standardNodes[STANDARD_ENCODERS_NB];

/**
 * @returns The number of valid node IDs
 */
size_t ER_getNbStandardNodes(void);

/**
 * Fills @nodes with all the valid node IDs
 *
 * @pre: @nodesSize must be at least @ER_getNbNodes().
 */
void ER_getAllStandardNodeIDs(ZL_NodeID* nodes, size_t nodesSize);

typedef ZL_Report (*ER_StandardNodesCallback)(
        void* opaque,
        ZL_NodeID node,
        const CNode* cnode);

/**
 * Calls @p cb on every standard node, and short-circuits if it returns an
 * error.
 */
ZL_Report ER_forEachStandardNode(ER_StandardNodesCallback cb, void* opaque);

typedef struct {
    ZL_VOEncoderFn transform_f;
} ER_VOadaptor_desc;

ZL_Report
ER_VOadaptor(ZL_Encoder* eictx, const ZL_Input* ins[], size_t nbInputs);

ZL_END_C_DECLS

#endif

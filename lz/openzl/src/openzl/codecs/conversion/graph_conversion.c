// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/conversion/graph_conversion.h"

ZL_NodeID ZL_Node_convertSerialToNumLE(size_t bitWidth)
{
    switch (bitWidth) {
        case 8:
            return ZL_NODE_CONVERT_SERIAL_TO_NUM_LE8;
        case 16:
            return ZL_NODE_CONVERT_SERIAL_TO_NUM_LE16;
        case 32:
            return ZL_NODE_CONVERT_SERIAL_TO_NUM_LE32;
        case 64:
            return ZL_NODE_CONVERT_SERIAL_TO_NUM_LE64;
        default:
            return (ZL_NodeID){ ZL_StandardNodeID_illegal };
    }
}

ZL_NodeID ZL_Node_convertSerialToNumBE(size_t bitWidth)
{
    switch (bitWidth) {
        case 8:
            return ZL_NODE_CONVERT_SERIAL_TO_NUM_BE8;
        case 16:
            return ZL_NODE_CONVERT_SERIAL_TO_NUM_BE16;
        case 32:
            return ZL_NODE_CONVERT_SERIAL_TO_NUM_BE32;
        case 64:
            return ZL_NODE_CONVERT_SERIAL_TO_NUM_BE64;
        default:
            return (ZL_NodeID){ ZL_StandardNodeID_illegal };
    }
}

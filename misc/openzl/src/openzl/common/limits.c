// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "openzl/common/limits.h"

size_t ZL_runtimeInputLimit(unsigned formatVersion)
{
    if (formatVersion <= 14)
        return 1;
    return 2048;
}

size_t ZL_runtimeNodeInputLimit(unsigned formatVersion)
{
    if (formatVersion <= 15)
        return 1;
    return ZL_runtimeInputLimit(formatVersion);
}

size_t ZL_runtimeNodeLimit(unsigned formatVersion)
{
    if (formatVersion < 9) {
        // Format versions < 9 claimed to support 1024 nodes, but in actuality
        // the frame header could only encode up to 255.
        return 256;
    }
    if (formatVersion < 20) {
        return 10000;
    }
    return 20000;
}

size_t ZL_runtimeStreamLimit(unsigned formatVersion)
{
    if (formatVersion < 9) {
        // Format versions < 9 claimed to support 1024 streams, but in actuality
        // the frame header could only encode up to 255.
        return 256;
    }
    if (formatVersion < 16) {
        return 10000;
    }
    // Should be at least slightly than the transform output stream limit
    // to make testing the transform output stream limit easier.
    return 110000;
}

size_t ZL_transformOutStreamsLimit(unsigned formatVersion)
{
    if (formatVersion < 9) {
        return 32;
    }
    if (formatVersion < 16) {
        return 1024;
    }
    return 100000;
}

/*
 * Copyright (c) 2025-2026, Bertrand Lebonnois
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef ZXC_CONSTANTS_H
#define ZXC_CONSTANTS_H

/*
 * ============================================================================
 * ZXC Compression Library - Public Constants
 * ============================================================================
 */

/* ===============================================================
 * ZXC Versioning
 * ===============================================================
 */

#define ZXC_VERSION_MAJOR 0
#define ZXC_VERSION_MINOR 5
#define ZXC_VERSION_PATCH 1

#define ZXC_STR_HELPER(x) #x
#define ZXC_STR(x) ZXC_STR_HELPER(x)

#define ZXC_LIB_VERSION_STR    \
    ZXC_STR(ZXC_VERSION_MAJOR) \
    "." ZXC_STR(ZXC_VERSION_MINOR) "." ZXC_STR(ZXC_VERSION_PATCH)

/* =============================================================
 * ZXC Compression Levels
 * =============================================================
 */

typedef enum {
    ZXC_LEVEL_FASTEST = 1,   // Fastest compression, best for real-time applications
    ZXC_LEVEL_FAST = 2,      // Fast compression, good for real-time applications
    ZXC_LEVEL_DEFAULT = 3,   // Recommended: ratio > LZ4, decode speed > LZ4
    ZXC_LEVEL_BALANCED = 4,  // Good ratio, good decode speed
    ZXC_LEVEL_COMPACT = 5    // High density. Best for storage/firmware/assets.
} zxc_compression_level_t;

#endif  // ZXC_CONSTANTS_H

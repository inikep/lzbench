/*
 * Copyright (c) 2025, Bertrand Lebonnois
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
#define ZXC_VERSION_MINOR 3
#define ZXC_VERSION_PATCH 0

#define ZXC_STR_HELPER(x) #x
#define ZXC_STR(x) ZXC_STR_HELPER(x)

#define ZXC_LIB_VERSION_STR    \
    ZXC_STR(ZXC_VERSION_MAJOR) \
    "." ZXC_STR(ZXC_VERSION_MINOR) "." ZXC_STR(ZXC_VERSION_PATCH)

/* =============================================================
 * ZXC Compression Levels
 * =============================================================
 */

#define ZXC_LEVEL_FAST (2)      // Fastest compression, best for real-time applications
#define ZXC_LEVEL_DEFAULT (3)   // Recommended: ratio > LZ4, decode speed > LZ4
#define ZXC_LEVEL_BALANCED (4)  // Good ratio, good decode speed
#define ZXC_LEVEL_COMPACT (5)   // High density. Best for storage/firmware/assets.

#endif  // ZXC_CONSTANTS_H
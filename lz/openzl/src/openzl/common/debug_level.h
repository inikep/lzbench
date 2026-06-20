// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * \file
 *
 * This file defines the global debug level.
 */

#ifndef ZSTRONG_COMMON_DEBUG_LEVEL_H
#define ZSTRONG_COMMON_DEBUG_LEVEL_H

#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/**
 * The ZL_DBG_LVL macro is a single lever that configures default values for
 * various finer-grained controls over what kinds of debug functionality are
 * compiled into the library and what the default states (active or dormant)
 * of those pieces are at runtime.
 *
 * The individual components can be configured directly, and are described
 * below.
 *
 * 0: Everything is disabled.
 * 1: Enable ZL_STATIC_ASSERT().
 * 2: Enable ZL_LOG().
 * 3: Enable ZL_ASSERT().
 * 4: Enable ZL_DLOG().
 */
#ifndef ZL_DBG_LVL
#    ifdef NDEBUG
/* NDEBUG is a conventional signal to discard debug constructs (specifically,
 * asserts). We take our cue from this signal, defaulting to disabling all of
 * our debug functions when it is present.
 */
#        define ZL_DBG_LVL 2
#    else
#        define ZL_DBG_LVL 4
#    endif
#endif

ZL_END_C_DECLS

#endif // ZSTRONG_COMMON_DEBUG_LEVEL_H

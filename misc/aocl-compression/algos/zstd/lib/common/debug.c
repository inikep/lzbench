/* ******************************************************************
 * debug
 * Part of FSE library
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Modifications Copyright (C) 2026, Advanced Micro Devices. All rights reserved.
 *
 * You can contact the author at :
 * - Source repository : https://github.com/Cyan4973/FiniteStateEntropy
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
****************************************************************** */


/*
 * This module only hosts one global variable
 * which can be used to dynamically influence the verbosity of traces,
 * such as DEBUGLOG and RAWLOG
 */

#ifdef AOCL_LLC_PREFIX
#include "../../../common/aoclPrefix.h"
#endif
#include "debug.h"
#if !defined(ZSTD_LINUX_KERNEL) || (DEBUGLEVEL>=2)
/* We only use this when DEBUGLEVEL>=2, but we get -Werror=pedantic errors if a
 * translation unit is empty. So remove this from Linux kernel builds, but
 * otherwise just leave it in.
 */
int g_debuglevel = DEBUGLEVEL;
#endif

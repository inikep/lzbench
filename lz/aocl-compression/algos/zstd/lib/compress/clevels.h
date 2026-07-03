/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

/**
 * Modifications Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ZSTD_CLEVELS_H
#define ZSTD_CLEVELS_H

#define ZSTD_STATIC_LINKING_ONLY  /* ZSTD_compressionParameters  */
#include "../zstd.h"

/*-=====  Pre-defined compression levels  =====-*/

#define ZSTD_MAX_CLEVEL     22

#ifdef __GNUC__
__attribute__((__unused__))
#endif

static const ZSTD_compressionParameters ZSTD_defaultCParameters[4][ZSTD_MAX_CLEVEL+1] = {
{   /* "default" - for any srcSize > 256 KB */
    /* W,  C,  H,  S,  L, TL, strat */
    { 19, 12, 13,  1,  6,  1, ZSTD_fast    },  /* base for negative levels */
    { 19, 13, 14,  1,  7,  0, ZSTD_fast    },  /* level  1 */
    { 20, 15, 16,  1,  6,  0, ZSTD_fast    },  /* level  2 */
    { 21, 16, 17,  1,  5,  0, ZSTD_dfast   },  /* level  3 */
    { 21, 18, 18,  1,  5,  0, ZSTD_dfast   },  /* level  4 */
    { 21, 18, 19,  3,  5,  2, ZSTD_greedy  },  /* level  5 */
    { 21, 18, 19,  3,  5,  4, ZSTD_lazy    },  /* level  6 */
    { 21, 19, 20,  4,  5,  8, ZSTD_lazy    },  /* level  7 */
    { 21, 19, 20,  4,  5, 16, ZSTD_lazy2   },  /* level  8 */
    { 22, 20, 21,  4,  5, 16, ZSTD_lazy2   },  /* level  9 */
    { 22, 21, 22,  5,  5, 16, ZSTD_lazy2   },  /* level 10 */
    { 22, 21, 22,  6,  5, 16, ZSTD_lazy2   },  /* level 11 */
    { 22, 22, 23,  6,  5, 32, ZSTD_lazy2   },  /* level 12 */
    { 22, 22, 22,  4,  5, 32, ZSTD_btlazy2 },  /* level 13 */
    { 22, 22, 23,  5,  5, 32, ZSTD_btlazy2 },  /* level 14 */
    { 22, 23, 23,  6,  5, 32, ZSTD_btlazy2 },  /* level 15 */
    { 22, 22, 22,  5,  5, 48, ZSTD_btopt   },  /* level 16 */
    { 23, 23, 22,  5,  4, 64, ZSTD_btopt   },  /* level 17 */
    { 23, 23, 22,  6,  3, 64, ZSTD_btultra },  /* level 18 */
    { 23, 24, 22,  7,  3,256, ZSTD_btultra2},  /* level 19 */
    { 25, 25, 23,  7,  3,256, ZSTD_btultra2},  /* level 20 */
    { 26, 26, 24,  7,  3,512, ZSTD_btultra2},  /* level 21 */
    { 27, 27, 25,  9,  3,999, ZSTD_btultra2},  /* level 22 */
},
{   /* for srcSize <= 256 KB */
    /* W,  C,  H,  S,  L,  T, strat */
    { 18, 12, 13,  1,  5,  1, ZSTD_fast    },  /* base for negative levels */
    { 18, 13, 14,  1,  6,  0, ZSTD_fast    },  /* level  1 */
    { 18, 14, 14,  1,  5,  0, ZSTD_dfast   },  /* level  2 */
    { 18, 16, 16,  1,  4,  0, ZSTD_dfast   },  /* level  3 */
    { 18, 16, 17,  3,  5,  2, ZSTD_greedy  },  /* level  4.*/
    { 18, 17, 18,  5,  5,  2, ZSTD_greedy  },  /* level  5.*/
    { 18, 18, 19,  3,  5,  4, ZSTD_lazy    },  /* level  6.*/
    { 18, 18, 19,  4,  4,  4, ZSTD_lazy    },  /* level  7 */
    { 18, 18, 19,  4,  4,  8, ZSTD_lazy2   },  /* level  8 */
    { 18, 18, 19,  5,  4,  8, ZSTD_lazy2   },  /* level  9 */
    { 18, 18, 19,  6,  4,  8, ZSTD_lazy2   },  /* level 10 */
    { 18, 18, 19,  5,  4, 12, ZSTD_btlazy2 },  /* level 11.*/
    { 18, 19, 19,  7,  4, 12, ZSTD_btlazy2 },  /* level 12.*/
    { 18, 18, 19,  4,  4, 16, ZSTD_btopt   },  /* level 13 */
    { 18, 18, 19,  4,  3, 32, ZSTD_btopt   },  /* level 14.*/
    { 18, 18, 19,  6,  3,128, ZSTD_btopt   },  /* level 15.*/
    { 18, 19, 19,  6,  3,128, ZSTD_btultra },  /* level 16.*/
    { 18, 19, 19,  8,  3,256, ZSTD_btultra },  /* level 17.*/
    { 18, 19, 19,  6,  3,128, ZSTD_btultra2},  /* level 18.*/
    { 18, 19, 19,  8,  3,256, ZSTD_btultra2},  /* level 19.*/
    { 18, 19, 19, 10,  3,512, ZSTD_btultra2},  /* level 20.*/
    { 18, 19, 19, 12,  3,512, ZSTD_btultra2},  /* level 21.*/
    { 18, 19, 19, 13,  3,999, ZSTD_btultra2},  /* level 22.*/
},
{   /* for srcSize <= 128 KB */
    /* W,  C,  H,  S,  L,  T, strat */
    { 17, 12, 12,  1,  5,  1, ZSTD_fast    },  /* base for negative levels */
    { 17, 12, 13,  1,  6,  0, ZSTD_fast    },  /* level  1 */
    { 17, 13, 15,  1,  5,  0, ZSTD_fast    },  /* level  2 */
    { 17, 15, 16,  2,  5,  0, ZSTD_dfast   },  /* level  3 */
    { 17, 17, 17,  2,  4,  0, ZSTD_dfast   },  /* level  4 */
    { 17, 16, 17,  3,  4,  2, ZSTD_greedy  },  /* level  5 */
    { 17, 16, 17,  3,  4,  4, ZSTD_lazy    },  /* level  6 */
    { 17, 16, 17,  3,  4,  8, ZSTD_lazy2   },  /* level  7 */
    { 17, 16, 17,  4,  4,  8, ZSTD_lazy2   },  /* level  8 */
    { 17, 16, 17,  5,  4,  8, ZSTD_lazy2   },  /* level  9 */
    { 17, 16, 17,  6,  4,  8, ZSTD_lazy2   },  /* level 10 */
    { 17, 17, 17,  5,  4,  8, ZSTD_btlazy2 },  /* level 11 */
    { 17, 18, 17,  7,  4, 12, ZSTD_btlazy2 },  /* level 12 */
    { 17, 18, 17,  3,  4, 12, ZSTD_btopt   },  /* level 13.*/
    { 17, 18, 17,  4,  3, 32, ZSTD_btopt   },  /* level 14.*/
    { 17, 18, 17,  6,  3,256, ZSTD_btopt   },  /* level 15.*/
    { 17, 18, 17,  6,  3,128, ZSTD_btultra },  /* level 16.*/
    { 17, 18, 17,  8,  3,256, ZSTD_btultra },  /* level 17.*/
    { 17, 18, 17, 10,  3,512, ZSTD_btultra },  /* level 18.*/
    { 17, 18, 17,  5,  3,256, ZSTD_btultra2},  /* level 19.*/
    { 17, 18, 17,  7,  3,512, ZSTD_btultra2},  /* level 20.*/
    { 17, 18, 17,  9,  3,512, ZSTD_btultra2},  /* level 21.*/
    { 17, 18, 17, 11,  3,999, ZSTD_btultra2},  /* level 22.*/
},
{   /* for srcSize <= 16 KB */
    /* W,  C,  H,  S,  L,  T, strat */
    { 14, 12, 13,  1,  5,  1, ZSTD_fast    },  /* base for negative levels */
    { 14, 14, 15,  1,  5,  0, ZSTD_fast    },  /* level  1 */
    { 14, 14, 15,  1,  4,  0, ZSTD_fast    },  /* level  2 */
    { 14, 14, 15,  2,  4,  0, ZSTD_dfast   },  /* level  3 */
    { 14, 14, 14,  4,  4,  2, ZSTD_greedy  },  /* level  4 */
    { 14, 14, 14,  3,  4,  4, ZSTD_lazy    },  /* level  5.*/
    { 14, 14, 14,  4,  4,  8, ZSTD_lazy2   },  /* level  6 */
    { 14, 14, 14,  6,  4,  8, ZSTD_lazy2   },  /* level  7 */
    { 14, 14, 14,  8,  4,  8, ZSTD_lazy2   },  /* level  8.*/
    { 14, 15, 14,  5,  4,  8, ZSTD_btlazy2 },  /* level  9.*/
    { 14, 15, 14,  9,  4,  8, ZSTD_btlazy2 },  /* level 10.*/
    { 14, 15, 14,  3,  4, 12, ZSTD_btopt   },  /* level 11.*/
    { 14, 15, 14,  4,  3, 24, ZSTD_btopt   },  /* level 12.*/
    { 14, 15, 14,  5,  3, 32, ZSTD_btultra },  /* level 13.*/
    { 14, 15, 15,  6,  3, 64, ZSTD_btultra },  /* level 14.*/
    { 14, 15, 15,  7,  3,256, ZSTD_btultra },  /* level 15.*/
    { 14, 15, 15,  5,  3, 48, ZSTD_btultra2},  /* level 16.*/
    { 14, 15, 15,  6,  3,128, ZSTD_btultra2},  /* level 17.*/
    { 14, 15, 15,  7,  3,256, ZSTD_btultra2},  /* level 18.*/
    { 14, 15, 15,  8,  3,256, ZSTD_btultra2},  /* level 19.*/
    { 14, 15, 15,  8,  3,512, ZSTD_btultra2},  /* level 20.*/
    { 14, 15, 15,  9,  3,512, ZSTD_btultra2},  /* level 21.*/
    { 14, 15, 15, 10,  3,999, ZSTD_btultra2},  /* level 22.*/
},
};

/* AOCL specific changes to the default parameters set by zstd
* 
* Commit 0235abbbba59705febbf0d6d04f65912e88bd50b : Compression ratio change with optimizations off fixed
*       AOCL_ZSTD_defaultCParameters created
*       > 256 KB
*           level 3 : W changed from 21 to 23
*           level 4 : W changed from 21 to 23
*  
* Commit 27a5c0394577b3fc1cb076593f316a56f4cd117e : Prefetch potential future match candidates in dfast
*       > 256 KB
*           level 3 : C changed from 16 to 15
*           level 4 : C changed from 18 to 16
*       <= 128 KB
*           level 3 : C changed from 15 to 14
*           level 4 : C changed from 17 to 15
*       <= 16 KB
*           level 3 : C changed from 14 to 13
* 
* Commit c3851db97f38428995610f6beb35f8b39851c5ed : Extend aocl optimizations to fast, lazy and greedy compression
*       > 256 KB
*           level 1 : W changed from 19 to 23
*           level 2 : W changed from 20 to 23
*/
static const ZSTD_compressionParameters AOCL_ZSTD_defaultCParameters[4][ZSTD_MAX_CLEVEL+1] = {
{   /* "default" - for any srcSize > 256 KB */
    /* W,  C,  H,  S,  L, TL, strat */
    { 19, 12, 13,  1,  6,  1, ZSTD_fast    },  /* base for negative levels */
    { 23, 13, 14,  1,  7,  0, ZSTD_fast    },  /* level  1 */
    { 23, 15, 16,  1,  6,  0, ZSTD_fast    },  /* level  2 */
    { 23, 15, 17,  1,  5,  0, ZSTD_dfast   },  /* level  3 */
    { 23, 16, 18,  1,  5,  0, ZSTD_dfast   },  /* level  4 */
    { 21, 18, 19,  3,  5,  2, ZSTD_greedy  },  /* level  5 */
#ifdef AOCL_COMPRESS_FAST
    /* Larger windows and settings from next levels to improve ratio.
     * Accounts for ratio drop from selective lazy evaluation */
    { 23, 19, 20,  4,  5,  8, ZSTD_lazy    },  /* level  6 */
    { 23, 19, 20,  4,  5, 16, ZSTD_lazy2   },  /* level  7 */
    { 23, 20, 21,  4,  5, 16, ZSTD_lazy2   },  /* level  8 */
    { 24, 21, 22,  5,  5, 16, ZSTD_lazy2   },  /* level  9 */
    { 24, 21, 22,  6,  5, 16, ZSTD_lazy2   },  /* level 10 */
    { 24, 22, 23,  6,  5, 32, ZSTD_lazy2   },  /* level 11 */
    { 24, 23, 24,  6,  5, 32, ZSTD_lazy2   },  /* level 12 */
#else
    { 21, 18, 19,  3,  5,  4, ZSTD_lazy    },  /* level  6 */
    { 21, 19, 20,  4,  5,  8, ZSTD_lazy    },  /* level  7 */
    { 21, 19, 20,  4,  5, 16, ZSTD_lazy2   },  /* level  8 */
    { 22, 20, 21,  4,  5, 16, ZSTD_lazy2   },  /* level  9 */
    { 22, 21, 22,  5,  5, 16, ZSTD_lazy2   },  /* level 10 */
    { 22, 21, 22,  6,  5, 16, ZSTD_lazy2   },  /* level 11 */
    { 22, 22, 23,  6,  5, 32, ZSTD_lazy2   },  /* level 12 */
#endif
    { 22, 22, 22,  4,  5, 32, ZSTD_btlazy2 },  /* level 13 */
    { 22, 22, 23,  5,  5, 32, ZSTD_btlazy2 },  /* level 14 */
    { 22, 23, 23,  6,  5, 32, ZSTD_btlazy2 },  /* level 15 */
    { 22, 22, 22,  5,  5, 48, ZSTD_btopt   },  /* level 16 */
    { 23, 23, 22,  5,  4, 64, ZSTD_btopt   },  /* level 17 */
    { 23, 23, 22,  6,  3, 64, ZSTD_btultra },  /* level 18 */
    { 23, 24, 22,  7,  3,256, ZSTD_btultra2},  /* level 19 */
    { 25, 25, 23,  7,  3,256, ZSTD_btultra2},  /* level 20 */
    { 26, 26, 24,  7,  3,512, ZSTD_btultra2},  /* level 21 */
    { 27, 27, 25,  9,  3,999, ZSTD_btultra2},  /* level 22 */
},
{   /* for srcSize <= 256 KB */
    /* W,  C,  H,  S,  L,  T, strat */
    { 18, 12, 13,  1,  5,  1, ZSTD_fast    },  /* base for negative levels */
    { 18, 13, 14,  1,  6,  0, ZSTD_fast    },  /* level  1 */
    { 18, 14, 14,  1,  5,  0, ZSTD_dfast   },  /* level  2 */
    { 18, 16, 16,  1,  4,  0, ZSTD_dfast   },  /* level  3 */
    { 18, 16, 17,  3,  5,  2, ZSTD_greedy  },  /* level  4.*/
    { 18, 17, 18,  5,  5,  2, ZSTD_greedy  },  /* level  5.*/
#ifdef AOCL_COMPRESS_FAST
    { 20, 18, 19,  4,  4,  4, ZSTD_lazy    },  /* level  6 */
    { 20, 18, 19,  4,  4,  8, ZSTD_lazy2   },  /* level  7 */
    { 20, 18, 19,  5,  4,  8, ZSTD_lazy2   },  /* level  8 */
    { 20, 18, 19,  6,  4,  8, ZSTD_lazy2   },  /* level  9 */
    { 20, 19, 19,  6,  5,  8, ZSTD_lazy2   },  /* level 10 */
#else
    { 18, 18, 19,  3,  5,  4, ZSTD_lazy    },  /* level  6.*/
    { 18, 18, 19,  4,  4,  4, ZSTD_lazy    },  /* level  7 */
    { 18, 18, 19,  4,  4,  8, ZSTD_lazy2   },  /* level  8 */
    { 18, 18, 19,  5,  4,  8, ZSTD_lazy2   },  /* level  9 */
    { 18, 18, 19,  6,  4,  8, ZSTD_lazy2   },  /* level 10 */
#endif
    { 18, 18, 19,  5,  4, 12, ZSTD_btlazy2 },  /* level 11.*/
    { 18, 19, 19,  7,  4, 12, ZSTD_btlazy2 },  /* level 12.*/
    { 18, 18, 19,  4,  4, 16, ZSTD_btopt   },  /* level 13 */
    { 18, 18, 19,  4,  3, 32, ZSTD_btopt   },  /* level 14.*/
    { 18, 18, 19,  6,  3,128, ZSTD_btopt   },  /* level 15.*/
    { 18, 19, 19,  6,  3,128, ZSTD_btultra },  /* level 16.*/
    { 18, 19, 19,  8,  3,256, ZSTD_btultra },  /* level 17.*/
    { 18, 19, 19,  6,  3,128, ZSTD_btultra2},  /* level 18.*/
    { 18, 19, 19,  8,  3,256, ZSTD_btultra2},  /* level 19.*/
    { 18, 19, 19, 10,  3,512, ZSTD_btultra2},  /* level 20.*/
    { 18, 19, 19, 12,  3,512, ZSTD_btultra2},  /* level 21.*/
    { 18, 19, 19, 13,  3,999, ZSTD_btultra2},  /* level 22.*/
},
{   /* for srcSize <= 128 KB */
    /* W,  C,  H,  S,  L,  T, strat */
    { 17, 12, 12,  1,  5,  1, ZSTD_fast    },  /* base for negative levels */
    { 17, 12, 13,  1,  6,  0, ZSTD_fast    },  /* level  1 */
    { 17, 13, 15,  1,  5,  0, ZSTD_fast    },  /* level  2 */
    { 17, 14, 16,  2,  5,  0, ZSTD_dfast   },  /* level  3 */
    { 17, 15, 17,  2,  4,  0, ZSTD_dfast   },  /* level  4 */
    { 17, 16, 17,  3,  4,  2, ZSTD_greedy  },  /* level  5 */
#ifdef AOCL_COMPRESS_FAST
    { 19, 16, 17,  3,  4,  8, ZSTD_lazy2   },  /* level  6 */
    { 19, 16, 17,  4,  4,  8, ZSTD_lazy2   },  /* level  7 */
    { 19, 16, 17,  5,  4,  8, ZSTD_lazy2   },  /* level  8 */
    { 19, 16, 17,  6,  4,  8, ZSTD_lazy2   },  /* level  9 */
    { 19, 17, 17,  6,  5,  8, ZSTD_lazy2   },  /* level  10 */
#else
    { 17, 16, 17,  3,  4,  4, ZSTD_lazy    },  /* level  6 */
    { 17, 16, 17,  3,  4,  8, ZSTD_lazy2   },  /* level  7 */
    { 17, 16, 17,  4,  4,  8, ZSTD_lazy2   },  /* level  8 */
    { 17, 16, 17,  5,  4,  8, ZSTD_lazy2   },  /* level  9 */
    { 17, 16, 17,  6,  4,  8, ZSTD_lazy2   },  /* level 10 */
#endif
    { 17, 17, 17,  5,  4,  8, ZSTD_btlazy2 },  /* level 11 */
    { 17, 18, 17,  7,  4, 12, ZSTD_btlazy2 },  /* level 12 */
    { 17, 18, 17,  3,  4, 12, ZSTD_btopt   },  /* level 13.*/
    { 17, 18, 17,  4,  3, 32, ZSTD_btopt   },  /* level 14.*/
    { 17, 18, 17,  6,  3,256, ZSTD_btopt   },  /* level 15.*/
    { 17, 18, 17,  6,  3,128, ZSTD_btultra },  /* level 16.*/
    { 17, 18, 17,  8,  3,256, ZSTD_btultra },  /* level 17.*/
    { 17, 18, 17, 10,  3,512, ZSTD_btultra },  /* level 18.*/
    { 17, 18, 17,  5,  3,256, ZSTD_btultra2},  /* level 19.*/
    { 17, 18, 17,  7,  3,512, ZSTD_btultra2},  /* level 20.*/
    { 17, 18, 17,  9,  3,512, ZSTD_btultra2},  /* level 21.*/
    { 17, 18, 17, 11,  3,999, ZSTD_btultra2},  /* level 22.*/
},
{   /* for srcSize <= 16 KB */
    /* W,  C,  H,  S,  L,  T, strat */
    { 14, 12, 13,  1,  5,  1, ZSTD_fast    },  /* base for negative levels */
    { 14, 14, 15,  1,  5,  0, ZSTD_fast    },  /* level  1 */
    { 14, 14, 15,  1,  4,  0, ZSTD_fast    },  /* level  2 */
    { 14, 13, 15,  2,  4,  0, ZSTD_dfast   },  /* level  3 */
    { 14, 14, 14,  4,  4,  2, ZSTD_greedy  },  /* level  4 */
    { 14, 14, 14,  3,  4,  4, ZSTD_lazy    },  /* level  5.*/
    { 14, 14, 14,  4,  4,  8, ZSTD_lazy2   },  /* level  6 */
    { 14, 14, 14,  6,  4,  8, ZSTD_lazy2   },  /* level  7 */
    { 14, 14, 14,  8,  4,  8, ZSTD_lazy2   },  /* level  8.*/
    { 14, 15, 14,  5,  4,  8, ZSTD_btlazy2 },  /* level  9.*/
    { 14, 15, 14,  9,  4,  8, ZSTD_btlazy2 },  /* level 10.*/
    { 14, 15, 14,  3,  4, 12, ZSTD_btopt   },  /* level 11.*/
    { 14, 15, 14,  4,  3, 24, ZSTD_btopt   },  /* level 12.*/
    { 14, 15, 14,  5,  3, 32, ZSTD_btultra },  /* level 13.*/
    { 14, 15, 15,  6,  3, 64, ZSTD_btultra },  /* level 14.*/
    { 14, 15, 15,  7,  3,256, ZSTD_btultra },  /* level 15.*/
    { 14, 15, 15,  5,  3, 48, ZSTD_btultra2},  /* level 16.*/
    { 14, 15, 15,  6,  3,128, ZSTD_btultra2},  /* level 17.*/
    { 14, 15, 15,  7,  3,256, ZSTD_btultra2},  /* level 18.*/
    { 14, 15, 15,  8,  3,256, ZSTD_btultra2},  /* level 19.*/
    { 14, 15, 15,  8,  3,512, ZSTD_btultra2},  /* level 20.*/
    { 14, 15, 15,  9,  3,512, ZSTD_btultra2},  /* level 21.*/
    { 14, 15, 15, 10,  3,999, ZSTD_btultra2},  /* level 22.*/
},
};

static const ZSTD_compressionParameters (*AOCL_ZSTD_defaultCParameters_used)[ZSTD_MAX_CLEVEL+1] = ZSTD_defaultCParameters;

#ifdef AOCL_COMPRESS_FAST
/* Lazy evaluation is performed only if the first match length 
 * is <= AOCL_ZSTD_compressFastLazyLimit */
#if AOCL_COMPRESS_FAST == 1
#define AOCL_COMPRESS_FAST_BASE_LAZY_LIMIT 5
#elif AOCL_COMPRESS_FAST >= 2
#define AOCL_COMPRESS_FAST_BASE_LAZY_LIMIT 4
#endif
static const unsigned AOCL_ZSTD_compressFastLazyLimit[ZSTD_MAX_CLEVEL + 1] = {
    AOCL_COMPRESS_FAST_BASE_LAZY_LIMIT,      /* base for negative levels */
    0,  /* level  1 */
    0,  /* level  2 */
    0,  /* level  3 */
    0,  /* level  4 */
    0,  /* level  5 */
    AOCL_COMPRESS_FAST_BASE_LAZY_LIMIT,      /* level  6 */
    AOCL_COMPRESS_FAST_BASE_LAZY_LIMIT + 1,  /* level  7 */
    AOCL_COMPRESS_FAST_BASE_LAZY_LIMIT + 2,  /* level  8.*/
    AOCL_COMPRESS_FAST_BASE_LAZY_LIMIT + 2,  /* level  9.*/
    AOCL_COMPRESS_FAST_BASE_LAZY_LIMIT + 2,  /* level 10.*/
    AOCL_COMPRESS_FAST_BASE_LAZY_LIMIT + 4,  /* level 11.*/
    AOCL_COMPRESS_FAST_BASE_LAZY_LIMIT + 4,  /* level 12.*/
    0,  /* level 13 */
    0,  /* level 14 */
    0,  /* level 15 */
    0,  /* level 16 */
    0,  /* level 17 */
    0,  /* level 18 */
    0,  /* level 19 */
    0,  /* level 20 */
    0,  /* level 21 */
    0,  /* level 22 */
};
#endif /* AOCL_COMPRESS_FAST */

#endif  /* ZSTD_CLEVELS_H */

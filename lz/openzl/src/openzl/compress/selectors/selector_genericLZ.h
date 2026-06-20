// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_COMPRESS_SELECTORS_SELECTOR_GENERICLZ_H
#define ZSTRONG_COMPRESS_SELECTORS_SELECTOR_GENERICLZ_H

#include "openzl/shared/portability.h"
#include "openzl/zl_selector.h" // ZL_SelectorDesc

ZL_BEGIN_C_DECLS

// .selector_f = SI_selector_genericLZ, .inStreamType = ZL_Type_serial,
ZL_GraphID SI_selector_genericLZ(
        const ZL_Selector* selCtx,
        const ZL_Input* inputStream,
        const ZL_GraphID* customGraphs,
        size_t nbCustomGraphs);

ZL_END_C_DECLS

#endif

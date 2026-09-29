// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_COMPRESS_SELECTORS_SELECTOR_ENTROPY_H
#define ZSTRONG_COMPRESS_SELECTORS_SELECTOR_ENTROPY_H

#include "openzl/shared/portability.h"
#include "openzl/zl_graph_api.h"

ZL_BEGIN_C_DECLS

ZL_GraphID EI_selector_entropy(ZL_Graph const* gctx, ZL_Edge const* input);

ZL_END_C_DECLS

#endif

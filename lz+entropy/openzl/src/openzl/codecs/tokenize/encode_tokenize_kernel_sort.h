// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_TOKENIZE_ENCODE_TOKENIZE_KERNEL_SORT_H
#define ZSTRONG_TRANSFORMS_TOKENIZE_ENCODE_TOKENIZE_KERNEL_SORT_H

#include "openzl/codecs/tokenize/encode_tokenize_kernel.h"

void pqdsortVsf(VSFKey* data, size_t nbElts);

#endif

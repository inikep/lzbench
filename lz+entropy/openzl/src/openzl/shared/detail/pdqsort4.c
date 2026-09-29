// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/shared/pdqsort.h"

#define T uint32_t
#include "openzl/shared/detail/pdqsort-inl.h"

void pdqsort4(uint32_t* data, size_t nbElts)
{
    pdqsort_branchless(data, data + nbElts);
}

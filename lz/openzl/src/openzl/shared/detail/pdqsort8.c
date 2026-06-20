// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/shared/pdqsort.h"

#define T uint64_t
#include "openzl/shared/detail/pdqsort-inl.h"

void pdqsort8(uint64_t* data, size_t nbElts)
{
    pdqsort_branchless(data, data + nbElts);
}

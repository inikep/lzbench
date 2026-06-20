// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/shared/pdqsort.h"

#define T uint16_t
#include "openzl/shared/detail/pdqsort-inl.h"

void pdqsort2(uint16_t* data, size_t nbElts)
{
    pdqsort_branchless(data, data + nbElts);
}

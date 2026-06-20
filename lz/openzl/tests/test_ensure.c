// Copyright (c) Meta Platforms, Inc. and affiliates.

// Example using ensure.h compile-time validation framework

#include "openzl/common/ensure.h"

// Test : validate that error condition has been checked
// must be compiled as a unit (does not form a valid program)
// In particular, far_function() link does not exist.

typedef struct {
    int value;
    int error;
} Report;
inline int isError(Report r)
{
    return r.error != 0;
}
static inline int getValidResult(Report r)
{
    ZL_ENSURE(r.error == 0);
    return r.value;
}

Report far_function(
        int i); // function body not known: return value can be anything

int test(int i)
{
    Report r = far_function(i);
#ifndef TEST_ENSURE_WILL_FAIL
    // Assuming ZS_ENABLE_ENSURE is set, and running on `gcc`
    // removing below line will trigger a compilation warning
    if (isError(r))
        return -1;
#endif
    return getValidResult(r);
}

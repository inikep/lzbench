// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include "openzl/common/vector.h"

namespace {

const size_t kDefaultVectorCapacity = 1024;

TEST(VectorTests, creation)
{
    VECTOR(int32_t) vec = {};
    ASSERT_EQ(VECTOR_SIZE(vec), 0u);
    ASSERT_EQ(VECTOR_CAPACITY(vec), 0u);
    ASSERT_EQ(VECTOR_MAX_CAPACITY(vec), 0u);
    memset((void*)&vec, 0xff, sizeof(vec));
    VECTOR_INIT(vec, 10);
    ASSERT_EQ(VECTOR_SIZE(vec), 0u);
    ASSERT_EQ(VECTOR_CAPACITY(vec), 0u);
    ASSERT_EQ(VECTOR_MAX_CAPACITY(vec), 10u);
    ASSERT_EQ(VECTOR_DATA(vec), nullptr);
    VECTOR_DESTROY(vec);
}

TEST(VectorTests, empty)
{
    VECTOR(int32_t) vec = VECTOR_EMPTY(kDefaultVectorCapacity);
    ASSERT_EQ(VECTOR_SIZE(vec), 0u);
    ASSERT_EQ(VECTOR_CAPACITY(vec), 0u);
    ASSERT_EQ(VECTOR_MAX_CAPACITY(vec), kDefaultVectorCapacity);
    ASSERT_EQ(VECTOR_DATA(vec), nullptr);
}

TEST(VectorTests, destruction)
{
    const int elem      = 0x1337;
    VECTOR(int32_t) vec = VECTOR_EMPTY(kDefaultVectorCapacity);
    ASSERT_TRUE(VECTOR_PUSHBACK(vec, elem));
    ASSERT_GE(VECTOR_CAPACITY(vec), 1u);
    VECTOR_DESTROY(vec);
    ASSERT_EQ(VECTOR_CAPACITY(vec), 0u);
    ASSERT_EQ(VECTOR_SIZE(vec), 0u);
    ASSERT_EQ(VECTOR_DATA(vec), nullptr);
    // Destroy again to make sure nothing breaks
    VECTOR_DESTROY(vec);
    ASSERT_EQ(VECTOR_CAPACITY(vec), 0u);
    ASSERT_EQ(VECTOR_SIZE(vec), 0u);
    ASSERT_EQ(VECTOR_DATA(vec), nullptr);
}

TEST(VectorTests, pushback)
{
    const int elem      = 0x1337;
    VECTOR(int32_t) vec = VECTOR_EMPTY(kDefaultVectorCapacity);
    ASSERT_EQ(VECTOR_SIZE(vec), 0u);
    for (size_t i = 1; i <= 100; i++) {
        ASSERT_TRUE(VECTOR_PUSHBACK(vec, elem));
        ASSERT_EQ(VECTOR_SIZE(vec), i);
        ASSERT_GE(VECTOR_CAPACITY(vec), i);
    }
    VECTOR_DESTROY(vec);
}

TEST(VectorTests, clear)
{
    const int elem      = 0x1337;
    VECTOR(int32_t) vec = VECTOR_EMPTY(kDefaultVectorCapacity);
    ASSERT_EQ(VECTOR_SIZE(vec), 0u);
    ASSERT_EQ(VECTOR_CAPACITY(vec), 0u);
    ASSERT_TRUE(VECTOR_PUSHBACK(vec, elem));
    ASSERT_EQ(VECTOR_SIZE(vec), 1u);
    ASSERT_GE(VECTOR_CAPACITY(vec), 1u);
    VECTOR_CLEAR(vec);
    ASSERT_EQ(VECTOR_SIZE(vec), 0u);
    ASSERT_GE(VECTOR_CAPACITY(vec), 1u);
    ASSERT_EQ(VECTOR_MAX_CAPACITY(vec), kDefaultVectorCapacity);
    VECTOR_DESTROY(vec);
}

TEST(VectorTests, reset)
{
    const int elem      = 0x1337;
    VECTOR(int32_t) vec = VECTOR_EMPTY(kDefaultVectorCapacity);
    ASSERT_EQ(VECTOR_SIZE(vec), 0u);
    ASSERT_EQ(VECTOR_CAPACITY(vec), 0u);
    ASSERT_TRUE(VECTOR_PUSHBACK(vec, elem));
    ASSERT_EQ(VECTOR_SIZE(vec), 1u);
    ASSERT_GE(VECTOR_CAPACITY(vec), 1u);
    VECTOR_CLEAR(vec);
    ASSERT_EQ(VECTOR_SIZE(vec), 0u);
    ASSERT_GE(VECTOR_CAPACITY(vec), 0u);
    ASSERT_EQ(VECTOR_MAX_CAPACITY(vec), kDefaultVectorCapacity);
    VECTOR_DESTROY(vec);
}

TEST(VectorTests, capacityAndReservation)
{
    const int elem      = 0x1337;
    VECTOR(int32_t) vec = VECTOR_EMPTY(1026);
    ASSERT_EQ(VECTOR_CAPACITY(vec), 0u);
    ASSERT_EQ(VECTOR_MAX_CAPACITY(vec), 1026u);
    ASSERT_EQ(VECTOR_RESERVE(vec, 1024), 1024u);
    ASSERT_EQ(VECTOR_CAPACITY(vec), 1024u);
    const int* originalDataPointer = VECTOR_DATA(vec);
    // Check that after reservation pointer stays stable and capacity doesn't
    // grow. Note that this doesn't promise that realloc didn't happen and
    // kept the same pointer, but it's at least some kind of sanity for
    // pointer stability.
    for (size_t i = 1; i <= 1024; i++) {
        ASSERT_TRUE(VECTOR_PUSHBACK(vec, elem));
        ASSERT_EQ(originalDataPointer, VECTOR_DATA(vec));
    }
    ASSERT_EQ(VECTOR_CAPACITY(vec), 1024u);
    // Check we grow until we hit max capacity
    ASSERT_TRUE(VECTOR_PUSHBACK(vec, elem));
    ASSERT_EQ(VECTOR_CAPACITY(vec), 1026u);
    ASSERT_TRUE(VECTOR_PUSHBACK(vec, elem));
    ASSERT_EQ(VECTOR_CAPACITY(vec), 1026u);
    ASSERT_FALSE(VECTOR_PUSHBACK(vec, elem));
    ASSERT_EQ(VECTOR_CAPACITY(vec), 1026u);
    VECTOR_DESTROY(vec);
}

TEST(VectorTests, smallReservationDoesNothing)
{
    VECTOR(int32_t) vec = VECTOR_EMPTY(2048);
    ASSERT_EQ(VECTOR_CAPACITY(vec), 0u);
    ASSERT_EQ(VECTOR_RESERVE(vec, 1024), 1024u);
    ASSERT_EQ(VECTOR_CAPACITY(vec), 1024u);
    ASSERT_EQ(VECTOR_RESERVE(vec, 10), 1024u);
    ASSERT_EQ(VECTOR_CAPACITY(vec), 1024u);
    ASSERT_EQ(VECTOR_RESERVE(vec, 2048), 2048u);
    ASSERT_EQ(VECTOR_CAPACITY(vec), 2048u);
    VECTOR_DESTROY(vec);
}

TEST(VectorTests, reserveExponentialGrowth)
{
    VECTOR(int32_t) vec = VECTOR_EMPTY(kDefaultVectorCapacity);
    ASSERT_EQ(VECTOR_CAPACITY(vec), 0u);
    // When we reserve within next capacity we get the next capacity
    ASSERT_GT(VECTOR_RESERVE(vec, 1), 1u);

    // When we reserve beyond the next capacity we get exact
    ASSERT_EQ(VECTOR_RESERVE(vec, 100), 100u);

    // When we reserve within next capacity we get the next capacity
    ASSERT_GT(VECTOR_RESERVE(vec, 101), 101u);

    // No growth if we are smaller
    size_t const capacity = VECTOR_CAPACITY(vec);
    ASSERT_EQ(VECTOR_RESERVE(vec, 102), capacity);
    VECTOR_DESTROY(vec);
}

TEST(VectorTests, resize)
{
    VECTOR(int32_t) vec = VECTOR_EMPTY(kDefaultVectorCapacity);
    ASSERT_EQ(VECTOR_SIZE(vec), 0u);
    ASSERT_EQ(VECTOR_RESIZE(vec, 1023), 1023u);
    ASSERT_EQ(VECTOR_SIZE(vec), 1023u);
    ASSERT_EQ(VECTOR_RESIZE(vec, 10), 10u);
    ASSERT_EQ(VECTOR_SIZE(vec), 10u);
    ASSERT_EQ(VECTOR_CAPACITY(vec), 1023u);
    VECTOR_DESTROY(vec);

    // Elements should be zeroed
    for (size_t i = 0; i < VECTOR_SIZE(vec); ++i) {
        ASSERT_EQ(VECTOR_AT(vec, i), 0);
    }
}

TEST(VectorTests, resizeExponentialGrowth)
{
    VECTOR(int32_t) vec = VECTOR_EMPTY(kDefaultVectorCapacity);

    ASSERT_EQ(VECTOR_SIZE(vec), 0u);
    ASSERT_EQ(VECTOR_RESIZE(vec, 100), 100u);
    ASSERT_EQ(VECTOR_CAPACITY(vec), 100u);
    ASSERT_EQ(VECTOR_SIZE(vec), 100u);
    ASSERT_EQ(VECTOR_RESIZE(vec, 101), 101u);

    // We should grow exponentially
    size_t const capacity = VECTOR_CAPACITY(vec);
    ASSERT_GT(VECTOR_CAPACITY(vec), 101u);
    ASSERT_EQ(VECTOR_RESIZE(vec, 102), 102u);
    ASSERT_EQ(VECTOR_CAPACITY(vec), capacity);

    VECTOR_DESTROY(vec);
}

TEST(VectorTests, at)
{
    VECTOR(size_t) vec = VECTOR_EMPTY(kDefaultVectorCapacity);
    for (size_t i = 0; i < 100; i++) {
        ASSERT_TRUE(VECTOR_PUSHBACK(vec, i));
    }
    ASSERT_EQ(VECTOR_RESIZE(vec, 200), 200u);
    for (size_t i = 100; i < 200; i++) {
        VECTOR_AT(vec, i) = i;
    }
    for (size_t i = 0; i < 200; i++) {
        ASSERT_EQ(VECTOR_AT(vec, i), i);
    }
    VECTOR_DESTROY(vec);
}

TEST(VectorTests, popback)
{
    size_t elem        = 0x1337;
    VECTOR(size_t) vec = VECTOR_EMPTY(kDefaultVectorCapacity);
    ASSERT_TRUE(VECTOR_PUSHBACK(vec, elem));
    ASSERT_EQ(VECTOR_SIZE(vec), 1u);
    VECTOR_POPBACK(vec);
    ASSERT_EQ(VECTOR_SIZE(vec), 0u);
    for (size_t i = 0; i < 100; i++) {
        ASSERT_TRUE(VECTOR_PUSHBACK(vec, elem));
    }
    for (size_t i = 100; i > 0; i--) {
        VECTOR_POPBACK(vec);
        ASSERT_EQ(VECTOR_SIZE(vec), i - 1);
    }
    VECTOR_DESTROY(vec);
}

TEST(VectorTests, evaluateOnce)
{
    size_t vecEvaluations   = 0;
    size_t sizetEvaluations = 0;
    size_t elem             = 0x1337;
    size_t count            = 0;
    VECTOR(size_t) vec      = VECTOR_EMPTY(kDefaultVectorCapacity);
    auto getVec             = [&vecEvaluations, &vec]() -> VECTOR(size_t) & {
        vecEvaluations++;
        return vec;
    };
    auto getSizet = [&sizetEvaluations](size_t& v) -> size_t& {
        sizetEvaluations++;
        return v;
    };
    VECTOR_INIT(getVec(), 1024);
    ASSERT_EQ(vecEvaluations, 1u);

    ASSERT_TRUE(VECTOR_PUSHBACK(getVec(), getSizet(elem)));
    ASSERT_EQ(vecEvaluations, 2u);
    ASSERT_EQ(sizetEvaluations, 1u);

    count = 100;
    ASSERT_EQ(VECTOR_RESERVE(getVec(), getSizet(count)), 100u);
    ASSERT_EQ(vecEvaluations, 3u);
    ASSERT_EQ(sizetEvaluations, 2u);

    count = 200;
    ASSERT_EQ(VECTOR_RESIZE(getVec(), getSizet(count)), 200u);
    ASSERT_EQ(vecEvaluations, 4u);
    ASSERT_EQ(sizetEvaluations, 3u);

    VECTOR_POPBACK(getVec());
    ASSERT_EQ(vecEvaluations, 5u);

    count                                = 1;
    VECTOR_AT(getVec(), getSizet(count)) = 1;
    ASSERT_EQ(vecEvaluations, 6u);
    ASSERT_EQ(sizetEvaluations, 4u);

    VECTOR_DESTROY(getVec());
    ASSERT_EQ(vecEvaluations, 7u);
}

TEST(VectorTests, createInArena)
{
    Arena* arena = ALLOC_HeapArena_create();

    VECTOR(int32_t) vec{};
    VECTOR_INIT_IN_ARENA(vec, arena, 100);

    for (int32_t i = 0; i < 100; ++i) {
        ASSERT_TRUE(VECTOR_PUSHBACK(vec, i));
    }

    // No destroy

    ALLOC_Arena_freeArena(arena);
}

} // namespace

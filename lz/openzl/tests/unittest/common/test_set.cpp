// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <random>
#include <unordered_set>

#include "openzl/common/set.h"

#include <gtest/gtest.h>

using namespace ::testing;

ZL_DECLARE_SET_TYPE(TestSet, int);

namespace {
uint32_t constexpr kDefaultMaxCapacity = 1000000;
}

TEST(SetTest, Empty)
{
    TestSet set = TestSet_create(kDefaultMaxCapacity);
    ASSERT_EQ(TestSet_size(&set), 0u);
    ASSERT_EQ(TestSet_capacity(&set), 0u);
    ASSERT_FALSE(TestSet_eraseVal(&set, 0));
    auto iter = TestSet_iter(&set);
    ASSERT_EQ(nullptr, TestSet_Iter_get(iter));
    ASSERT_EQ(nullptr, TestSet_Iter_next(&iter));
    TestSet_destroy(&set);
}

TEST(SetTest, Clear)
{
    TestSet emptySet = TestSet_create(kDefaultMaxCapacity);
    TestSet resetSet = TestSet_create(kDefaultMaxCapacity);

    auto insert = TestSet_insertVal(&resetSet, 0);
    ASSERT_TRUE(insert.inserted);
    ASSERT_EQ(TestSet_size(&resetSet), 1);
    TestSet_clear(&resetSet);
    ASSERT_EQ(TestSet_size(&resetSet), 0);
    ASSERT_NE(TestSet_capacity(&resetSet), 0);
    ASSERT_NE(memcmp(&emptySet, &resetSet, sizeof(resetSet)), 0);

    insert = TestSet_insertVal(&resetSet, 0);
    ASSERT_TRUE(insert.inserted);
    ASSERT_EQ(TestSet_size(&resetSet), 1);

    insert = TestSet_insertVal(&resetSet, 1);
    ASSERT_TRUE(insert.inserted);
    ASSERT_EQ(TestSet_size(&resetSet), 2);

    TestSet_destroy(&emptySet);
    TestSet_destroy(&resetSet);
}

TEST(SetTest, Reserve)
{
    for (int guaranteeNoAllocations = 0; guaranteeNoAllocations <= 1;
         ++guaranteeNoAllocations) {
        TestSet set = TestSet_create(kDefaultMaxCapacity);
        ASSERT_EQ(TestSet_size(&set), 0u);
        ASSERT_EQ(TestSet_capacity(&set), 0u);

        ASSERT_TRUE(TestSet_reserve(&set, 10, !!guaranteeNoAllocations));
        ASSERT_EQ(TestSet_capacity(&set), 10u);
        ASSERT_TRUE(TestSet_reserve(&set, 11, !!guaranteeNoAllocations));
        ASSERT_GT(TestSet_capacity(&set), 11u);

        ASSERT_TRUE(TestSet_reserve(&set, 10, !!guaranteeNoAllocations));
        ASSERT_GT(TestSet_capacity(&set), 11u);

        TestSet_destroy(&set);
    }
}

TEST(SetTest, Insert)
{
    TestSet set = TestSet_create(kDefaultMaxCapacity);

    ASSERT_EQ(TestSet_size(&set), 0u);
    ASSERT_EQ(nullptr, TestSet_findVal(&set, 0));
    {
        auto [ptr, inserted, badAlloc] = TestSet_insertVal(&set, 0);
        ASSERT_FALSE(badAlloc);
        ASSERT_TRUE(inserted);
        ASSERT_EQ(*ptr, 0);
    }
    ASSERT_EQ(0, *TestSet_findVal(&set, 0));
    ASSERT_EQ(TestSet_size(&set), 1u);

    {
        auto [ptr, inserted, badAlloc] = TestSet_insertVal(&set, 0);
        ASSERT_FALSE(badAlloc);
        ASSERT_FALSE(inserted);
        ASSERT_EQ(*ptr, 0);
    }
    ASSERT_EQ(0, *TestSet_findVal(&set, 0));
    ASSERT_EQ(TestSet_size(&set), 1u);

    {
        int key                        = 1;
        auto [ptr, inserted, badAlloc] = TestSet_insert(&set, &key);
        ASSERT_FALSE(badAlloc);
        ASSERT_TRUE(inserted);
        ASSERT_EQ(*ptr, 1);
    }
    ASSERT_EQ(1, *TestSet_findVal(&set, 1));
    ASSERT_EQ(TestSet_size(&set), 2u);

    TestSet_destroy(&set);
}

TEST(SetTest, Find)
{
    TestSet set = TestSet_create(kDefaultMaxCapacity);

    auto testFindNull = [&set](int key) {
        if (TestSet_findVal(&set, key) != nullptr) {
            return false;
        }
        if (TestSet_find(&set, &key) != nullptr) {
            return false;
        }
        if (TestSet_findMutVal(&set, key) != nullptr) {
            return false;
        }
        if (TestSet_findMut(&set, &key) != nullptr) {
            return false;
        }

        if (TestSet_contains(&set, &key)) {
            return false;
        }
        if (TestSet_containsVal(&set, key)) {
            return false;
        }
        return true;
    };

    auto testFind = [&set](int key) {
        if (TestSet_findVal(&set, key) == nullptr) {
            return false;
        }
        if (TestSet_find(&set, &key) == nullptr) {
            return false;
        }
        if (TestSet_findMutVal(&set, key) == nullptr) {
            return false;
        }
        if (TestSet_findMut(&set, &key) == nullptr) {
            return false;
        }

        if (!TestSet_contains(&set, &key)) {
            return false;
        }
        if (!TestSet_containsVal(&set, key)) {
            return false;
        }

        auto const* entry = TestSet_findVal(&set, key);
        if (*entry != key) {
            return false;
        }

        entry = TestSet_find(&set, &key);
        if (*entry != key) {
            return false;
        }

        entry = TestSet_findMutVal(&set, key);
        if (*entry != key) {
            return false;
        }

        entry = TestSet_findMut(&set, &key);
        if (*entry != key) {
            return false;
        }
        return true;
    };

    for (int i = 0; i < 100; ++i) {
        int const key = i * 10;

        ASSERT_TRUE(testFindNull(key));
        ASSERT_FALSE(TestSet_insertVal(&set, key).badAlloc);
        ASSERT_TRUE(testFind(key));
    }
    ASSERT_EQ(TestSet_size(&set), 100u);

    for (int i = 0; i < 100; ++i) {
        int const key = i * 10;
        ASSERT_TRUE(testFind(key));
        ASSERT_TRUE(testFindNull(key + 1));
    }

    for (int i = 0; i < 100; ++i) {
        int const key = i * 10;
        ASSERT_TRUE(testFind(key));
        ASSERT_TRUE(TestSet_eraseVal(&set, key));
        ASSERT_TRUE(testFindNull(key));
        ASSERT_TRUE(testFindNull(key + 1));
    }

    ASSERT_EQ(TestSet_size(&set), 0u);

    TestSet_destroy(&set);
}

TEST(SetTest, Erase)
{
    TestSet set = TestSet_create(kDefaultMaxCapacity);

    int key = 0;
    ASSERT_FALSE(TestSet_erase(&set, &key));
    ASSERT_FALSE(TestSet_eraseVal(&set, key));

    ASSERT_EQ(TestSet_size(&set), 0u);

    TestSet_insertVal(&set, 0);
    TestSet_insertVal(&set, 1);
    TestSet_insertVal(&set, 2);
    TestSet_insertVal(&set, 3);

    ASSERT_EQ(TestSet_size(&set), 4u);

    key = 0;
    ASSERT_TRUE(TestSet_erase(&set, &key));
    ASSERT_FALSE(TestSet_erase(&set, &key));

    ASSERT_EQ(TestSet_size(&set), 3u);
    ASSERT_FALSE(TestSet_containsVal(&set, 0));
    ASSERT_TRUE(TestSet_containsVal(&set, 1));
    ASSERT_TRUE(TestSet_containsVal(&set, 2));
    ASSERT_TRUE(TestSet_containsVal(&set, 3));

    ASSERT_TRUE(TestSet_eraseVal(&set, 2));
    ASSERT_FALSE(TestSet_eraseVal(&set, 2));

    ASSERT_EQ(TestSet_size(&set), 2u);
    ASSERT_FALSE(TestSet_containsVal(&set, 0));
    ASSERT_TRUE(TestSet_containsVal(&set, 1));
    ASSERT_FALSE(TestSet_containsVal(&set, 2));
    ASSERT_TRUE(TestSet_containsVal(&set, 3));

    TestSet_destroy(&set);
}

TEST(SetTest, MatchesstdSet)
{
    TestSet set = TestSet_create(kDefaultMaxCapacity);

    std::mt19937 gen(0xdeadbeef);
    std::uniform_int_distribution<int> keyDist(0, 1000);
    std::uniform_int_distribution<int> opDist(0, 2);

    std::unordered_set<int> stdSet;
    for (size_t i = 0; i < 100000; ++i) {
        auto const op  = opDist(gen);
        auto const key = keyDist(gen);
        if (op == 0) {
            auto it  = stdSet.find(key);
            auto ptr = TestSet_findVal(&set, key);
            if (it == stdSet.end()) {
                ASSERT_TRUE(ptr == nullptr);
            } else {
                ASSERT_FALSE(ptr == nullptr);
                ASSERT_EQ(*it, *ptr);
            }
        } else if (op == 1) {
            auto [it, inserted] = stdSet.insert(key);
            auto insert         = TestSet_insertVal(&set, key);
            ASSERT_FALSE(insert.badAlloc);
            ASSERT_EQ(insert.inserted, inserted);
            ASSERT_EQ(*it, *insert.ptr);
        } else {
            ASSERT_EQ(op, 2);
            auto const erased = stdSet.erase(key);
            ASSERT_EQ(erased, TestSet_eraseVal(&set, key));
        }
    }
    ASSERT_EQ(TestSet_size(&set), stdSet.size());
    int const* key;
    for (auto it = TestSet_iter(&set); (key = TestSet_Iter_next(&it));) {
        ASSERT_TRUE(stdSet.count(*key) == 1);
        stdSet.erase(*key);
    }
    ASSERT_TRUE(stdSet.empty());

    TestSet_destroy(&set);
}

TEST(SetTest, iter)
{
    TestSet set = TestSet_create(kDefaultMaxCapacity);
    std::unordered_set<int> expected;

    auto testIter = [&set, &expected]() {
        {
            auto remaining = expected;
            auto iter      = TestSet_iter(&set);
            for (;;) {
                auto* entry = TestSet_Iter_get(iter);
                if (entry != TestSet_Iter_next(&iter)) {
                    return false;
                }
                if (entry == nullptr) {
                    break;
                }
                auto it = remaining.find(*entry);
                if (it == remaining.end()) {
                    return false;
                }
                if (*it != *entry) {
                    return false;
                }
                remaining.erase(*entry);
            }
            if (!remaining.empty()) {
                return false;
            }
        }
        {
            auto remaining = expected;
            auto iter      = TestSet_iterMut(&set);
            for (;;) {
                auto* entry = TestSet_IterMut_get(iter);
                if (entry != TestSet_IterMut_next(&iter)) {
                    return false;
                }
                if (entry == nullptr) {
                    break;
                }
                auto it = remaining.find(*entry);
                if (it == remaining.end()) {
                    return false;
                }
                if (*it != *entry) {
                    return false;
                }
                remaining.erase(*entry);
            }
            if (!remaining.empty()) {
                return false;
            }
        }
        return true;
    };

    ASSERT_TRUE(testIter());

    ASSERT_TRUE(TestSet_reserve(&set, 10, false));

    ASSERT_TRUE(testIter());

    for (int i = 0; i < 100; ++i) {
        int const key = i * 7 % 100;
        ASSERT_FALSE(TestSet_insertVal(&set, key).badAlloc);
        ASSERT_TRUE(expected.emplace(key).second);
        ASSERT_TRUE(testIter());
    }

    for (int i = 0; i < 100; ++i) {
        int const key = i * 7 % 100;
        ASSERT_TRUE(TestSet_erase(&set, &key));
        expected.erase(key);
        ASSERT_TRUE(testIter());
    }

    TestSet_destroy(&set);
}

struct Key {
    int ignored;
    int key;
};

static size_t TestCustomSet_hash(Key const* key)
{
    return (size_t)key->key;
}

static bool TestCustomSet_eq(Key const* lhs, Key const* rhs)
{
    return lhs->key == rhs->key;
}

ZL_DECLARE_CUSTOM_SET_TYPE(TestCustomSet, Key);

TEST(SetTest, CustomSet)
{
    TestCustomSet set = TestCustomSet_create(kDefaultMaxCapacity);

    ASSERT_TRUE(TestCustomSet_insertVal(&set, { 0, 0 }).inserted);
    ASSERT_FALSE(TestCustomSet_insertVal(&set, { 1, 0 }).inserted);
    ASSERT_TRUE(TestCustomSet_insertVal(&set, { 0, 1 }).inserted);

    for (int key = 2; key < 100; ++key) {
        ASSERT_TRUE(TestCustomSet_insertVal(&set, { 0, key }).inserted);
        for (int ignored = 0; ignored < 100; ++ignored) {
            ASSERT_FALSE(
                    TestCustomSet_insertVal(&set, { ignored, key }).inserted);
        }
    }

    TestCustomSet_destroy(&set);
}

TEST(SetTest, SmallCapacityLimit)
{
    TestSet set = TestSet_create(10);
    ASSERT_TRUE(TestSet_reserve(&set, 10, /* guaranteeNoAllocations */ true));
    const size_t capacity      = TestSet_capacity(&set);
    const size_t chainCapacity = set.table_.chainCapacity;

    for (size_t offset = 0; offset < 100; ++offset) {
        for (size_t i = 0; i < 10; ++i) {
            ASSERT_EQ(TestSet_insertVal(&set, i + offset).inserted, true);
        }
        ASSERT_EQ(TestSet_capacity(&set), capacity);
        ASSERT_EQ(set.table_.chainCapacity, chainCapacity);
        TestSet_clear(&set);
    }
    ASSERT_EQ(TestSet_capacity(&set), capacity);
    ASSERT_EQ(set.table_.chainCapacity, chainCapacity);

    TestSet_destroy(&set);
}

TEST(SetTest, TinyCapacityLimit)
{
    TestSet set = TestSet_create(1);
    ASSERT_TRUE(TestSet_reserve(&set, 1, /* guaranteeNoAllocations */ true));
    const size_t capacity      = TestSet_capacity(&set);
    const size_t chainCapacity = set.table_.chainCapacity;

    ASSERT_EQ(TestSet_insertVal(&set, 0).inserted, true);

    ASSERT_EQ(TestSet_capacity(&set), capacity);
    ASSERT_EQ(set.table_.chainCapacity, chainCapacity);

    TestSet_destroy(&set);
}

TEST(SetTest, CreateInArena)
{
    Arena* arena = ALLOC_HeapArena_create();

    TestSet set = TestSet_createInArena(arena, 100);

    for (int i = 0; i < 100; ++i) {
        ASSERT_TRUE(TestSet_insertVal(&set, i).inserted);
    }

    // No destroy

    ALLOC_Arena_freeArena(arena);
}

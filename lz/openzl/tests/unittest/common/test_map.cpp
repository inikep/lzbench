// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <random>
#include <unordered_map>

#include "openzl/common/map.h"

#include <gtest/gtest.h>

using namespace ::testing;

ZL_DECLARE_MAP_TYPE(TestMap, int, int);

namespace {
uint32_t constexpr kDefaultMaxCapacity = 1000000;
}

TEST(MapTest, Empty)
{
    TestMap map = TestMap_create(kDefaultMaxCapacity);
    ASSERT_EQ(TestMap_size(&map), 0u);
    ASSERT_EQ(TestMap_capacity(&map), 0u);
    ASSERT_EQ(TestMap_maxCapacity(&map), kDefaultMaxCapacity);
    ASSERT_FALSE(TestMap_eraseVal(&map, 0u));
    auto iter = TestMap_iter(&map);
    ASSERT_EQ(nullptr, TestMap_Iter_get(iter));
    ASSERT_EQ(nullptr, TestMap_Iter_next(&iter));
    TestMap_destroy(&map);
}

TEST(MapTest, Clear)
{
    TestMap emptyMap = TestMap_create(kDefaultMaxCapacity);
    TestMap resetMap = TestMap_create(kDefaultMaxCapacity);

    auto insert = TestMap_insertVal(&resetMap, { 0, 0 });
    ASSERT_TRUE(insert.inserted);
    ASSERT_EQ(TestMap_size(&resetMap), 1);
    TestMap_clear(&resetMap);
    ASSERT_EQ(TestMap_size(&resetMap), 0);
    ASSERT_NE(TestMap_capacity(&resetMap), 0);
    ASSERT_NE(memcmp(&emptyMap, &resetMap, sizeof(resetMap)), 0);

    insert = TestMap_insertVal(&resetMap, { 0, 0 });
    ASSERT_TRUE(insert.inserted);
    ASSERT_EQ(TestMap_size(&resetMap), 1);

    insert = TestMap_insertVal(&resetMap, { 1, 1 });
    ASSERT_TRUE(insert.inserted);
    ASSERT_EQ(TestMap_size(&resetMap), 2);

    TestMap_destroy(&emptyMap);
    TestMap_destroy(&resetMap);
}

TEST(MapTest, Reserve)
{
    for (int guaranteeNoAllocations = 0; guaranteeNoAllocations <= 1;
         ++guaranteeNoAllocations) {
        TestMap map = TestMap_create(kDefaultMaxCapacity);
        ASSERT_EQ(TestMap_size(&map), 0u);
        ASSERT_EQ(TestMap_capacity(&map), 0u);

        ASSERT_TRUE(TestMap_reserve(&map, 10, !!guaranteeNoAllocations));
        ASSERT_EQ(TestMap_capacity(&map), 10u);
        ASSERT_TRUE(TestMap_reserve(&map, 11, !!guaranteeNoAllocations));
        ASSERT_GT(TestMap_capacity(&map), 11u);

        ASSERT_TRUE(TestMap_reserve(&map, 10, !!guaranteeNoAllocations));
        ASSERT_GT(TestMap_capacity(&map), 11u);

        TestMap_destroy(&map);
    }
}

TEST(MapTest, ReserveGuaranteeToAllocations)
{
    size_t constexpr kCapacity = 10;
    TestMap map                = TestMap_create(kDefaultMaxCapacity);
    ASSERT_TRUE(TestMap_reserve(&map, kCapacity, true));
    void const* tablePtr = map.table_.table;
    void const* chainPtr = map.table_.chain;

    std::mt19937 gen(0xdeadbeef);
    std::uniform_int_distribution<int> dist;

    std::unordered_map<int, int> stdMap;
    for (size_t i = 0; i < 10000; ++i) {
        if (stdMap.size() == kCapacity) {
            ASSERT_EQ(TestMap_size(&map), kCapacity);
            auto it = stdMap.begin();
            std::advance(it, (size_t)dist(gen) % kCapacity);
            int const key = it->first;

            ASSERT_TRUE(TestMap_findVal(&map, key) != nullptr);
            ASSERT_EQ(TestMap_findVal(&map, key)->key, it->first);
            ASSERT_EQ(TestMap_findVal(&map, key)->val, it->second);

            stdMap.erase(it);
            ASSERT_TRUE(TestMap_eraseVal(&map, key));
        }
        auto [it, inserted] = stdMap.emplace(dist(gen), dist(gen));
        auto insert = TestMap_insertVal(&map, { it->first, it->second });
        ASSERT_EQ(insert.badAlloc, false);
        ASSERT_EQ(insert.inserted, inserted);
        ASSERT_EQ(insert.ptr->key, it->first);
        ASSERT_EQ(insert.ptr->val, it->second);
    }

    ASSERT_EQ(tablePtr, map.table_.table);
    ASSERT_EQ(chainPtr, map.table_.chain);

    TestMap_destroy(&map);
}

TEST(MapTest, MaxCapacity)
{
    TestMap map = TestMap_create(10);
    ASSERT_EQ(TestMap_maxCapacity(&map), 10u);
    ASSERT_FALSE(TestMap_reserve(&map, 11, false));
    ASSERT_FALSE(TestMap_reserve(&map, 11, true));
    ASSERT_EQ(TestMap_size(&map), 0u);
    for (int i = 0; i < 10; ++i) {
        auto const insert = TestMap_insertVal(&map, { i, i });
        ASSERT_TRUE(insert.inserted);
        ASSERT_FALSE(insert.badAlloc);
    }
    ASSERT_EQ(TestMap_size(&map), 10u);
    {
        auto const insert = TestMap_insertVal(&map, { 0, 0 });
        ASSERT_FALSE(insert.inserted);
        // This is perhaps unexpected behavior, but we reserve before we check
        // for duplicates.
        ASSERT_TRUE(insert.badAlloc);
    }
    {
        auto const insert = TestMap_insertVal(&map, { 10, 10 });
        ASSERT_FALSE(insert.inserted);
        ASSERT_TRUE(insert.badAlloc);
    }
    ASSERT_EQ(TestMap_size(&map), 10u);
    TestMap_destroy(&map);

    map = TestMap_create(10);
    ASSERT_TRUE(TestMap_reserve(&map, 10, false));
    ASSERT_EQ(TestMap_size(&map), 0u);
    for (int i = 0; i < 10; ++i) {
        auto const insert = TestMap_insertVal(&map, { i, i });
        ASSERT_TRUE(insert.inserted);
        ASSERT_FALSE(insert.badAlloc);
    }
    ASSERT_EQ(TestMap_size(&map), 10u);
    {
        auto const insert = TestMap_insertVal(&map, { 0, 0 });
        ASSERT_FALSE(insert.inserted);
        // This is perhaps unexpected behavior, but we reserve before we check
        // for duplicates.
        ASSERT_TRUE(insert.badAlloc);
    }
    {
        auto const insert = TestMap_insertVal(&map, { 10, 10 });
        ASSERT_FALSE(insert.inserted);
        ASSERT_TRUE(insert.badAlloc);
    }
    ASSERT_EQ(TestMap_size(&map), 10u);
    TestMap_destroy(&map);
}

TEST(MapTest, Insert)
{
    TestMap map = TestMap_create(kDefaultMaxCapacity);

    ASSERT_EQ(TestMap_size(&map), 0u);
    ASSERT_EQ(nullptr, TestMap_findVal(&map, 0));
    {
        auto [ptr, inserted, badAlloc] = TestMap_insertVal(&map, { 0, 0 });
        ASSERT_FALSE(badAlloc);
        ASSERT_TRUE(inserted);
        ASSERT_EQ(ptr->key, 0);
        ASSERT_EQ(ptr->val, 0);
    }
    ASSERT_EQ(0, TestMap_findVal(&map, 0)->key);
    ASSERT_EQ(0, TestMap_findVal(&map, 0)->val);
    ASSERT_EQ(TestMap_size(&map), 1u);

    {
        auto [ptr, inserted, badAlloc] = TestMap_insertVal(&map, { 0, 1 });
        ASSERT_FALSE(badAlloc);
        ASSERT_FALSE(inserted);
        ASSERT_EQ(ptr->key, 0);
        ASSERT_EQ(ptr->val, 0);
    }
    ASSERT_EQ(0, TestMap_findVal(&map, 0)->key);
    ASSERT_EQ(0, TestMap_findVal(&map, 0)->val);
    ASSERT_EQ(TestMap_size(&map), 1u);

    {
        TestMap_Entry entry            = { 1, 2 };
        auto [ptr, inserted, badAlloc] = TestMap_insert(&map, &entry);
        ASSERT_FALSE(badAlloc);
        ASSERT_TRUE(inserted);
        ASSERT_EQ(ptr->key, 1);
        ASSERT_EQ(ptr->val, 2);
    }
    ASSERT_EQ(1, TestMap_findVal(&map, 1)->key);
    ASSERT_EQ(2, TestMap_findVal(&map, 1)->val);
    ASSERT_EQ(TestMap_size(&map), 2u);

    TestMap_destroy(&map);
}

TEST(MapTest, Find)
{
    TestMap map = TestMap_create(kDefaultMaxCapacity);

    auto testFindNull = [&map](int key) {
        if (TestMap_findVal(&map, key) != nullptr) {
            return false;
        }
        if (TestMap_find(&map, &key) != nullptr) {
            return false;
        }
        if (TestMap_findMutVal(&map, key) != nullptr) {
            return false;
        }
        if (TestMap_findMut(&map, &key) != nullptr) {
            return false;
        }

        if (TestMap_contains(&map, &key)) {
            return false;
        }
        if (TestMap_containsVal(&map, key)) {
            return false;
        }
        return true;
    };

    auto testFind = [&map](int key, int val) {
        if (TestMap_findVal(&map, key) == nullptr) {
            return false;
        }
        if (TestMap_find(&map, &key) == nullptr) {
            return false;
        }
        if (TestMap_findMutVal(&map, key) == nullptr) {
            return false;
        }
        if (TestMap_findMut(&map, &key) == nullptr) {
            return false;
        }

        if (!TestMap_contains(&map, &key)) {
            return false;
        }
        if (!TestMap_containsVal(&map, key)) {
            return false;
        }

        auto const* entry = TestMap_findVal(&map, key);
        if (entry->key != key || entry->val != val) {
            return false;
        }

        entry = TestMap_find(&map, &key);
        if (entry->key != key || entry->val != val) {
            return false;
        }

        entry = TestMap_findMutVal(&map, key);
        if (entry->key != key || entry->val != val) {
            return false;
        }

        entry = TestMap_findMut(&map, &key);
        if (entry->key != key || entry->val != val) {
            return false;
        }
        return true;
    };

    for (int i = 0; i < 100; ++i) {
        int const key = i * 10;
        int const val = i;

        ASSERT_TRUE(testFindNull(key));
        ASSERT_FALSE(TestMap_insertVal(&map, { key, val }).badAlloc);
        ASSERT_TRUE(testFind(key, val));
    }
    ASSERT_EQ(TestMap_size(&map), 100u);

    for (int i = 0; i < 100; ++i) {
        int const key = i * 10;
        int const val = i;
        ASSERT_TRUE(testFind(key, val));
        ASSERT_TRUE(testFindNull(key + 1));
    }

    for (int i = 0; i < 100; ++i) {
        int const key = i * 10;
        int const val = i;
        ASSERT_TRUE(testFind(key, val));
        ASSERT_TRUE(TestMap_eraseVal(&map, key));
        ASSERT_TRUE(testFindNull(key));
        ASSERT_TRUE(testFindNull(key + 1));
    }

    ASSERT_EQ(TestMap_size(&map), 0u);

    TestMap_destroy(&map);
}

TEST(MapTest, Erase)
{
    TestMap map = TestMap_create(kDefaultMaxCapacity);

    int key = 0;
    ASSERT_FALSE(TestMap_erase(&map, &key));
    ASSERT_FALSE(TestMap_eraseVal(&map, key));

    ASSERT_EQ(TestMap_size(&map), 0u);

    TestMap_insertVal(&map, { 0, 0 });
    TestMap_insertVal(&map, { 1, 1 });
    TestMap_insertVal(&map, { 2, 2 });
    TestMap_insertVal(&map, { 3, 3 });

    ASSERT_EQ(TestMap_size(&map), 4u);

    key = 0;
    ASSERT_TRUE(TestMap_erase(&map, &key));
    ASSERT_FALSE(TestMap_erase(&map, &key));

    ASSERT_EQ(TestMap_size(&map), 3u);
    ASSERT_FALSE(TestMap_containsVal(&map, 0));
    ASSERT_TRUE(TestMap_containsVal(&map, 1));
    ASSERT_TRUE(TestMap_containsVal(&map, 2));
    ASSERT_TRUE(TestMap_containsVal(&map, 3));

    ASSERT_TRUE(TestMap_eraseVal(&map, 2));
    ASSERT_FALSE(TestMap_eraseVal(&map, 2));

    ASSERT_EQ(TestMap_size(&map), 2u);
    ASSERT_FALSE(TestMap_containsVal(&map, 0));
    ASSERT_TRUE(TestMap_containsVal(&map, 1));
    ASSERT_FALSE(TestMap_containsVal(&map, 2));
    ASSERT_TRUE(TestMap_containsVal(&map, 3));

    TestMap_destroy(&map);
}

TEST(MapTest, iter)
{
    TestMap map = TestMap_create(kDefaultMaxCapacity);
    std::unordered_map<int, int> expected;

    auto testIter = [&map, &expected]() {
        {
            auto remaining = expected;
            auto iter      = TestMap_iter(&map);
            for (;;) {
                auto* entry = TestMap_Iter_get(iter);
                if (entry != TestMap_Iter_next(&iter)) {
                    return false;
                }
                if (entry == nullptr) {
                    break;
                }
                auto it = remaining.find(entry->key);
                if (it == remaining.end()) {
                    return false;
                }
                if (it->first != entry->key || it->second != entry->val) {
                    return false;
                }
                remaining.erase(entry->key);
            }
            if (!remaining.empty()) {
                return false;
            }
        }
        {
            auto remaining = expected;
            auto iter      = TestMap_iterMut(&map);
            for (;;) {
                auto* entry = TestMap_IterMut_get(iter);
                if (entry != TestMap_IterMut_next(&iter)) {
                    return false;
                }
                if (entry == nullptr) {
                    break;
                }
                auto it = remaining.find(entry->key);
                if (it == remaining.end()) {
                    return false;
                }
                if (it->first != entry->key || it->second != entry->val) {
                    return false;
                }
                remaining.erase(entry->key);
            }
            if (!remaining.empty()) {
                return false;
            }
        }
        return true;
    };

    ASSERT_TRUE(testIter());

    ASSERT_TRUE(TestMap_reserve(&map, 10, false));

    ASSERT_TRUE(testIter());

    for (int i = 0; i < 100; ++i) {
        int const key = i * 7 % 100;
        int const val = i;
        ASSERT_FALSE(TestMap_insertVal(&map, { key, val }).badAlloc);
        ASSERT_TRUE(expected.emplace(key, val).second);
        ASSERT_TRUE(testIter());
    }

    for (int i = 0; i < 100; ++i) {
        int const key = i * 7 % 100;
        ASSERT_TRUE(TestMap_erase(&map, &key));
        expected.erase(key);
        ASSERT_TRUE(testIter());
    }

    TestMap_destroy(&map);
}

struct Key {
    int ignored;
    int key;
};

static size_t TestCustomMap_hash(Key const* key)
{
    return (size_t)key->key;
}

static bool TestCustomMap_eq(Key const* lhs, Key const* rhs)
{
    return lhs->key == rhs->key;
}

ZL_DECLARE_CUSTOM_MAP_TYPE(TestCustomMap, Key, int);

TEST(MapTest, CustomMap)
{
    TestCustomMap map = TestCustomMap_create(kDefaultMaxCapacity);

    ASSERT_TRUE(TestCustomMap_insertVal(&map, { { 0, 0 }, 0 }).inserted);
    ASSERT_FALSE(TestCustomMap_insertVal(&map, { { 1, 0 }, 0 }).inserted);
    ASSERT_TRUE(TestCustomMap_insertVal(&map, { { 0, 1 }, 1 }).inserted);

    for (int key = 2; key < 100; ++key) {
        ASSERT_TRUE(
                TestCustomMap_insertVal(&map, { { 0, key }, key }).inserted);
        for (int ignored = 0; ignored < 100; ++ignored) {
            ASSERT_FALSE(TestCustomMap_insertVal(
                                 &map, { { ignored, key }, key + ignored })
                                 .inserted);
        }
    }

    TestCustomMap_destroy(&map);
}

TEST(MapTest, CreateInArena)
{
    Arena* arena = ALLOC_HeapArena_create();

    TestMap set = TestMap_createInArena(arena, 100);

    for (int i = 0; i < 100; ++i) {
        ASSERT_TRUE(TestMap_insertVal(&set, { i, i }).inserted);
    }

    // No destroy

    ALLOC_Arena_freeArena(arena);
}

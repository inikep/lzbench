// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_COMMON_MAP_H
#define ZSTRONG_COMMON_MAP_H

#include <limits.h>
#include <string.h>
#include "openzl/common/detail/table.h"
#include "openzl/common/set.h"

ZL_BEGIN_C_DECLS

/**
 * Declares a map named @p Map_ with key type @p Key_ and value type @p Val_.
 * Only one map with the name @p Map_ can be created in the same translation
 * unit. Generates default hash & equality functions. Uses XXH3 for hashing by
 * default.
 *
 * Maps are unordered data structures which map keys to values, and allow you to
 * lookup entries (key, value pairs) by the key. If you only need keys and not
 * values, you should use the set API in `set.h`.
 *
 * It provides operations like find(), contains(), insert(), erase(), and also
 * iteration over all the elements. The map will grow dynamically as needed, but
 * to avoid the overhead of rehashing you can call reserve() ahead of time.
 *
 * All functions operating on the @p Map_ begin with @p Map_. See `map_api.h`
 * for documentation.
 *
 * Example usage that builds a histogram of the input ints:
 *
 *   ZL_DECLARE_MAP_TYPE(IntCountMap, int, size_t);
 *   size_t histogram(int const* data, size_t size) {
 *       // Create the map
 *       IntCountMap map = IntCountMap_create();
 *
 *       // Optionally reserve, but we don't have a good estimate of the
 *       // cardinality, so we can just skip it.
 *
 *       for (size_t i = 0; i < size; ++i) {
 *           // Lookup the entry.
 *           IntCountMap_Entry* entry = IntCountMap_findMut(&map, data + i);
 *           if (entry == NULL) {
 *               // If it is not present, insert it with a count of 0.
 *               IntCountMap_Entry entry = { data[i], 0 };
 *               IntCountMap_Insert insert = IntCountMap_insert(&map, &entry);
 *               ZL_REQUIRE(!insert.badAlloc);
 *
 *               // Set entry to the newly inserted entry.
 *               entry = insert.ptr;
 *           }
 *           // Increment the count.
 *           ++entry->val;
 *       }
 *
 *       // use the histogram... E.g. iterate over the values.
 *       size_t totalCount = 0;
 *       IntCountMap_Iter iter = IntCountMap_iter(&map);
 *       IntCountMap_Entry const* entry;
 *       for ( ; entry = IntCountMap_Iter_next(&iter); ) {
 *           totalCount += entry->val;
 *       }
 *
 *       // The total count is equal to the input size.
 *       ZL_ASSERT_EQ(totalCount, size);
 *
 *       size_t const numUniqueInts = IntCountMap_size(&map);
 *
 *       // Destroy the map and free its memory.
 *       IntCountMap_destroy(&map);
 *
 *       return numUniqueInts;
 *   }
 */
#define ZL_DECLARE_MAP_TYPE(Map_, Key_, Val_)    \
    ZL_DECLARE_TABLE_DEFAULT_HASH_FN(Map_, Key_) \
    ZL_DECLARE_TABLE_DEFAULT_EQ_FN(Map_, Key_)   \
    _ZL_DECLARE_MAP_TYPE_IMPL(Map_, Key_, Val_)

/**
 * A specialization of ZL_DECLARE_MAP_TYPE() for maps with Key_ types that
 * already provide implementations of their hash and equality functions, which
 * just uses those functions. Expects these two functions to be present:
 *
 *     size_t ${Key_}_hash($Key_ const* key);
 *     bool ${Key_}_eq($Key_ const* lhs, $Key_ const* rhs);
 *
 * StringView and ZL_Name are examples of types that provide this.
 *
 * Note that this means Key_ must be a single, bare word and not a multi-word
 * or pointer type. In the case that your Key_ type is, either typedef it to a
 * suitable name or just use ZL_DECLARE_CUSTOM_MAP_TYPE() which has no such
 * requirement.
 */
#define ZL_DECLARE_PREDEF_MAP_TYPE(Map_, Key_, Val_) \
    ZL_DECLARE_TABLE_PREDEF_HASH_FN(Map_, Key_)      \
    ZL_DECLARE_TABLE_PREDEF_EQ_FN(Map_, Key_)        \
    _ZL_DECLARE_MAP_TYPE_IMPL(Map_, Key_, Val_)

/**
 * The same as ZL_DECLARE_MAP_TYPE() except uses custom hash & equality
 * functions. Requires these two functions to be present:
 *
 *     size_t ${Map_}_hash($Key_ const* key);
 *     bool ${Map_}_eq($Key_ const* lhs, $Key_ const* rhs);
 */
#define ZL_DECLARE_CUSTOM_MAP_TYPE(Map_, Key_, Val_) \
    ZL_DECLARE_TABLE_CUSTOM_HASH_FN(Map_, Key_)      \
    ZL_DECLARE_TABLE_CUSTOM_EQ_FN(Map_, Key_)        \
    _ZL_DECLARE_MAP_TYPE_IMPL(Map_, Key_, Val_)

/**
 * Common base implementation macro. Don't invoke directly.
 */
#define _ZL_DECLARE_MAP_TYPE_IMPL(Map_, Key_, Val_) \
    typedef struct {                                \
        Key_ key;                                   \
        Val_ val;                                   \
    } Map_##_Entry;                                 \
    typedef Key_ Map_##_Key;                        \
    ZL_DECLARE_TABLE_DEFAULT_POLICY(Map_);          \
    ZL_DECLARE_TABLE(Map_, Map_##_Entry, Map_##_Key, Detail_##Map_##_kPolicy)

ZL_END_C_DECLS

#endif

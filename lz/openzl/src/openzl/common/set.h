// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_COMMON_SET_H
#define ZSTRONG_COMMON_SET_H

#include <limits.h>
#include <string.h>
#include "openzl/common/assertion.h"
#include "openzl/common/detail/table.h"
#include "openzl/shared/bits.h"
#include "openzl/shared/overflow.h"
#include "openzl/shared/portability.h"
#include "openzl/shared/utils.h"
#include "openzl/shared/xxhash.h"

ZL_BEGIN_C_DECLS

/**
 * Declares a set named @p Set_ with key type @p Key_.
 * Only one set with the name @p Set_ can be created in the same translation
 * unit. Generates default hash & equality functions. Uses XXH3 for hashing by
 * default.
 *
 * Sets are unordered data structures which allow you to insert and lookup
 * elements by a key. If you need to also store a value for each key, you should
 * use the map API in `map.h`.
 *
 * It provides operations like find(), contains(), insert(), erase(), and also
 * iteration over all the elements. The set will grow dynamically as needed, but
 * to avoid the overhead of rehashing you can call reserve() ahead of time.
 *
 * All functions operating on the @p Set_ begin with @p Set_. See `set_api.h`
 * for documentation.
 *
 * Example usage that computes the exact cardinality of a list of ints:
 *
 *   ZL_DECLARE_SET_TYPE(UniqueIntSet, int);
 *   size_t cardinality(int const* data, size_t size) {
 *       // Create the set
 *       UniqueIntSet set = UniqueIntSet_create();
 *
 *       // Optionally reserve, but we don't have a good estimate of the
 *       // cardinality, so we can just skip it.
 *
 *       size_t numUniqueInts = 0;
 *
 *       for (size_t i = 0; i < size; ++i) {
 *           // Try to insert the int to our set.
 *           UniqueIntSet_Insert insert = UniqueIntSet_insertVal(&set, data[i]);
 *           ZL_REQUIRE(!insert.badAlloc);
 *
 *           // If we inserted the int, then we haven't seen it before.
 *           // Increment our counter.
 *           if (insert.inserted) {
 *               ++numUniqueInts;
 *           }
 *       }
 *
 *       // numUniqueInts is actually exactly the same as our set size.
 *       // So we didn't actually have to keep track of it, just for an example.
 *       ZL_ASSERT_EQ(UniqueIntSet_size(&set), numUniqueInts);
 *
 *       // Destroy the set and free its memory.
 *       UniqueIntSet_destroy(&set);
 *
 *      return numUniqueInts;
 *   }
 */
#define ZL_DECLARE_SET_TYPE(Set_, Key_)          \
    typedef Key_ Set_##_Entry;                   \
    typedef Key_ Set_##_Key;                     \
    ZL_DECLARE_TABLE_DEFAULT_HASH_FN(Set_, Key_) \
    ZL_DECLARE_TABLE_DEFAULT_EQ_FN(Set_, Key_)   \
    ZL_DECLARE_TABLE_DEFAULT_POLICY(Set_);       \
    ZL_DECLARE_TABLE(Set_, Set_##_Entry, Set_##_Key, Detail_##Set_##_kPolicy)

/**
 * The same as ZL_DECLARE_SET_TYPE() except uses custom hash & equality
 * functions. Requires these two functions to be present:
 *
 *     size_t ${Set_}_hash($Key_ const* key);
 *     bool ${Set_}_eq($Key_ const* lhs, $Key_ const* rhs);
 */
#define ZL_DECLARE_CUSTOM_SET_TYPE(Set_, Key_)  \
    typedef Key_ Set_##_Entry;                  \
    typedef Key_ Set_##_Key;                    \
    ZL_DECLARE_TABLE_CUSTOM_HASH_FN(Set_, Key_) \
    ZL_DECLARE_TABLE_CUSTOM_EQ_FN(Set_, Key_)   \
    ZL_DECLARE_TABLE_DEFAULT_POLICY(Set_);      \
    ZL_DECLARE_TABLE(Set_, Set_##_Entry, Set_##_Key, Detail_##Set_##_kPolicy)

ZL_END_C_DECLS

#endif

// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/common/set.h"

/**
 * This file provides an example set declaration, and explicitly declares all
 * the functions that ZL_DECLARE_SET_TYPE() generates. This gives us a place to
 * put documentation about how to use the API. But this file is only meant for
 * documentation purposes.
 *
 * All of these functions and structs begin with MySet_. For your set type, they
 * will have the name of your set as the prefix.
 */

#error "Example for API documentation only, not meant to be included"

ZL_DECLARE_SET_TYPE(MySet, int);

// typedef struct {
//     MySet_Key* ptr;   //< Pointer to the key.
//     bool inserted;    //< Whether insertion took place.
//     bool badAlloc;    //< Whether allocation failed.
// } MySet_Insert;

/**
 * Creates a new empty MySet.
 * The table must be destroeyd with @ref MySet_destroy().
 *
 * @param maxCapacity The maxinum number of entries that the
 * set can hold. Insertion will fail if the set attempts to
 * grow beyond this limit.
 * NOTE: insert() will return badAlloc if the set size is equal to the
 * maxCapacity, even if the key is already in the set.
 *
 * @returns A newly constructed empty table.
 */
ZL_TABLE_INLINE MySet MySet_create(uint32_t maxCapacity);

/**
 * Creates a new empty MySet in @p arena.
 * The table may be destroyed with @ref MySet_destroy(), or implicitly destroyed
 * by freeing all the memory owned by the @p arena.
 *
 * @param arena All allocations made by the set are in @p arena, so memory may
 * be managed by freeing all memory owned by the arena, rather than explicitly
 * destroying the set.
 * @param maxCapacity The maxinum number of entries that the
 * set can hold. Insertion will fail if the set attempts to
 * grow beyond this limit.
 * NOTE: insert() will return badAlloc if the set size is equal to the
 * maxCapacity, even if the key is already in the set.
 *
 * @returns A newly constructed empty table.
 */
ZL_TABLE_INLINE MySet MySet_createInArena(Arena* arena, uint32_t maxCapacity);

/**
 * Destroys the set and frees all its resources.
 */
ZL_TABLE_INLINE void MySet_destroy(MySet* set);

/**
 * Clears the map without releasing memory.
 * @post MyMap_size(map) == 0
 */
ZL_TABLE_INLINE void MySet_clear(MySet* set);

/**
 * @returns The number of keys in the set.
 */
ZL_TABLE_INLINE size_t MySet_size(MySet const* set);

/**
 * @returns The capacity of the set.
 */
ZL_TABLE_INLINE size_t MySet_capacity(MySet const* table);

/**
 * @returns The max capacity of the map.
 */
ZL_TABLE_INLINE size_t MySet_maxCapacity(MySet const* table);

/**
 * Reserves enough space in the set for @p capacity entries before growing.
 * WARNING: Invalidates pointers & iterators.
 *
 * @param capacity Reserve space for this many entries.
 * @param guaranteeNoAllocations Set to true if you want to guarantee that there
 * will be no allocations until the size of the set surpasses @p capacity. This
 * ensures that in the worst case scenario where all keys hash to the same
 * bucket we still won't allocate. This option ~doubles the memory usage, so
 * don't set it unless you need the guarantee that no set functions will
 * allocate.
 *
 * @returns false iff reservation failed due to a bad allocation.
 */
ZL_TABLE_INLINE bool
MySet_reserve(MySet* set, uint32_t capacity, bool guaranteeNoAllocations);

/**
 * Looks up @p key in the set and returns the entry if found, and NULL
 * otherwise.
 * WARNING: The pointer is invalidated by MySet_reserve(), MySet_insert(), and
 * MySet_erase().
 * NOTE: The key cannot be modified to change its hash or equality.
 *
 * @returns A pointer to the entry if present, else NULL.
 */
ZL_TABLE_INLINE MySet_Entry const* MySet_find(
        MySet const* set,
        MySet_Key const* key);

/// Same as MySet_find() except returns a mutable pointer.
ZL_TABLE_INLINE MySet_Entry* MySet_findMut(MySet* set, MySet_Key const* key);

/// Same as MySet_find() except takes the key by value.
ZL_TABLE_INLINE MySet_Entry const* MySet_findVal(
        MySet const* set,
        MySet_Key const key);

/// Same as MySet_findMut() except takes the key by value.
ZL_TABLE_INLINE MySet_Entry* MySet_findMutVal(MySet* set, MySet_Key const key);

/// @returns true iff the @p key is present in the set.
ZL_TABLE_INLINE bool MySet_contains(MySet const* set, MySet_Key const* key);

/// @returns true iff the @p key is present in the set.
ZL_TABLE_INLINE bool MySet_containsVal(MySet const* set, MySet_Key const key);

/**
 * Inserts @p key into the set if it isn't already present.
 * If the @p key is already present, it does not insert, and returns a pointer
 * to the entry.
 * WARNING: Invalidates pointers & iterators.
 * WARNING: The pointer is invalidated by MySet_reserve(), MySet_insert(), and
 * MySet_erase().
 *
 * @returns a struct containing the pointer, whether the insertion took place,
 * and whether an allocation failed. `ptr` will only be `NULL` if `badAlloc` is
 * true.
 */
ZL_TABLE_INLINE MySet_Insert MySet_insert(MySet* set, MySet_Key const* key);

/// Same as MySet_insert() except takes @p key by value.
ZL_TABLE_INLINE MySet_Insert MySet_insertVal(MySet* set, MySet_Key const key);

/**
 * Erases @p key from the set if it is present.
 * WARNING: Invalidates pointers & iterators.
 *
 * @returns true iff the key was present.
 */
ZL_TABLE_INLINE bool MySet_erase(MySet* set, MySet_Key const* key);

/// Same as MySet_erase() expect takes @p key by value.
ZL_TABLE_INLINE bool MySet_eraseVal(MySet* set, MySet_Key const key);

/**
 * @returns an iterator into the set that can be used to iterate through
 * all the elements.
 * WARNING: The iterator is invalidated by MySet_reserve(), MySet_insert(), and
 * MySet_erase().
 *
 * E.g. it can be used like this.
 *
 *     MySet_Iter iter = MySet_iter(set);
 *     for (MySet_Entry const* entry; entry = MySet_Iter_next(&iter); ) {
 *         use(entry);
 *     }
 */
ZL_TABLE_INLINE MySet_Iter MySet_iter(MySet const* set);

/**
 * @returns The entry pointed to by @p iter and advances it to the next element.
 * Returns NULL once the iterator reaches the end of the set.
 */
ZL_TABLE_INLINE MySet_Entry const* MySet_Iter_next(MySet_Iter* iter);

/**
 * @returns The entry pointed to by @p iter or NULL if it's reached the end of
 * the set.
 */
ZL_TABLE_INLINE MySet_Entry const* MySet_Iter_get(MySet_Iter iter);

/**
 * @returns an iterator into the set that can be used to iterate
 * through all the elements and provide mutable pointers to the keys.
 * WARNING: The iterator is invalidated by MySet_reserve(), MySet_insert(), and
 * MySet_erase().
 */
ZL_TABLE_INLINE MySet_IterMut MySet_iterMut(MySet* set);

/**
 * @returns The entry pointed to by @p iter and advances it to the next element.
 * Returns NULL once the iterator reaches the end of the set.
 */
ZL_TABLE_INLINE MySet_Entry* MySet_IterMut_next(MySet_IterMut* iter);

/**
 * @returns The entry pointed to by @p iter or NULL if it's reached the end of
 * the set.
 */
ZL_TABLE_INLINE MySet_Entry* MySet_IterMut_get(MySet_IterMut iter);

/**
 * Converts a mutable iterator into an immutable iterator.
 *
 * @returns An immutable copy of iter.
 */
ZL_TABLE_INLINE MySet_Iter MySet_IterMut_const(MySet_IterMut iter);

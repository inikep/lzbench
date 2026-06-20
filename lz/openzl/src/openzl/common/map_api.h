// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/common/map.h"

/**
 * This file provides an example map declaration, and explicitly declares all
 * the functions that ZL_DECLARE_MAP_TYPE() generates. This gives us a place to
 * put documentation about how to use the API. But this file is only meant for
 * documentation purposes.
 *
 * All of these functions and structs begin with MyMap_. For your map type, they
 * will have the name of your map as the prefix.
 */

#error "Example for API documentation only, not meant to be included"

ZL_DECLARE_MAP_TYPE(MyMap, int, int);

// typedef struct {
//     MyMap_Entry* ptr; //< Pointer to the entry.
//     bool inserted;    //< Whether insertion took place.
//     bool badAlloc;    //< Whether allocation failed.
// } MyMap_Insert;

/**
 * Creates a new empty MyMap.
 * The table must be destroeyd with @ref MyMap_destroy().
 *
 * @param maxCapacity The maxinum number of entries that the
 * map can hold. Insertion will fail if the map attempts to
 * grow beyond this limit.
 * NOTE: insert() will return badAlloc if the map size is equal to the
 * maxCapacity, even if the key is already in the map.
 *
 * @returns A newly constructed empty table.
 */
ZL_TABLE_INLINE MyMap MyMap_create(uint32_t maxCapacity);

/**
 * Creates a new empty MyMap in @p arena.
 * The table may be destroyed with @ref MyMap_destroy(), or implicitly destroyed
 * by freeing all the memory owned by the @p arena.
 *
 * @param arena All allocations made by the map are in @p arena, so memory may
 * be managed by freeing all memory owned by the arena, rather than explicitly
 * destroying the map.
 * @param maxCapacity The maxinum number of entries that the
 * map can hold. Insertion will fail if the map attempts to
 * grow beyond this limit.
 * NOTE: insert() will return badAlloc if the map size is equal to the
 * maxCapacity, even if the key is already in the map.
 *
 * @returns A newly constructed empty table.
 */
ZL_TABLE_INLINE MyMap MyMap_createInArena(Arena* arena, uint32_t maxCapacity);

/**
 * Destroys the map and frees all its resources.
 */
ZL_TABLE_INLINE void MyMap_destroy(MyMap* map);

/**
 * Clears the map without releasing memory.
 * @post MyMap_size(map) == 0
 */
ZL_TABLE_INLINE void MyMap_clear(MyMap* map);

/**
 * @returns The number of entries in the map.
 */
ZL_TABLE_INLINE size_t MyMap_size(MyMap const* map);

/**
 * @returns The capacity of the map.
 */
ZL_TABLE_INLINE size_t MyMap_capacity(MyMap const* table);

/**
 * @returns The max capacity of the map.
 */
ZL_TABLE_INLINE size_t MyMap_maxCapacity(MyMap const* table);

/**
 * Reserves enough space in the map for @p capacity entries before growing.
 * WARNING: Invalidates pointers & iterators.
 *
 * @param capacity Reserve space for this many entries.
 * @param guaranteeNoAllocations Set to true if you want to guarantee that there
 * will be no allocations until the size of the map surpasses @p capacity. This
 * ensures that in the worst case scenario where all keys hash to the same
 * bucket we still won't allocate. This option ~doubles the memory usage, so
 * don't set it unless you need the guarantee that no map functions will
 * allocate.
 *
 * @returns false iff reservation failed due to a bad allocation.
 */
ZL_TABLE_INLINE bool
MyMap_reserve(MyMap* map, uint32_t capacity, bool guaranteeNoAllocations);

/**
 * Looks up @p key in the map and returns the entry if found, and NULL
 * otherwise.
 * WARNING: The pointer is invalidated by MyMap_reserve(), MyMap_insert(), and
 * MyMap_erase().
 * NOTE: The entry key cannot be modified to change its hash or equality.
 *
 * @returns A pointer to the entry if present, else NULL.
 */
ZL_TABLE_INLINE MyMap_Entry const* MyMap_find(
        MyMap const* map,
        MyMap_Key const* key);

/// Same as MyMap_find() except returns a mutable pointer.
ZL_TABLE_INLINE MyMap_Entry* MyMap_findMut(MyMap* map, MyMap_Key const* key);

/// Same as MyMap_find() except takes the key by value.
ZL_TABLE_INLINE MyMap_Entry const* MyMap_findVal(
        MyMap const* map,
        MyMap_Key const key);

/// Same as MyMap_findMut() except takes the key by value.
ZL_TABLE_INLINE MyMap_Entry* MyMap_findMutVal(MyMap* map, MyMap_Key const key);

/// @returns true iff the @p key is present in the map.
ZL_TABLE_INLINE bool MyMap_contains(MyMap const* map, MyMap_Key const* key);

/// @returns true iff the @p key is present in the map.
ZL_TABLE_INLINE bool MyMap_containsVal(MyMap const* map, MyMap_Key const key);

/**
 * Inserts @p key into the map if it isn't already present.
 * If the @p key is already present, it does not insert, and returns a pointer
 * to the entry.
 * WARNING: Invalidates pointers & iterators.
 * WARNING: The pointer is invalidated by MyMap_reserve(), MyMap_insert(), and
 * MyMap_erase().
 *
 * @returns a struct containing the pointer, whether the insertion took place,
 * and whether an allocation failed. `ptr` will only be `NULL` if `badAlloc` is
 * true.
 */
ZL_TABLE_INLINE MyMap_Insert MyMap_insert(MyMap* map, MyMap_Entry const* entry);

/// Same as MyMap_insert() except takes @p entry by value.
ZL_TABLE_INLINE MyMap_Insert
MyMap_insertVal(MyMap* map, MyMap_Entry const entry);

/**
 * Erases @p key from the map if it is present.
 * WARNING: Invalidates pointers & iterators.
 *
 * @returns true iff the key was present.
 */
ZL_TABLE_INLINE bool MyMap_erase(MyMap* map, MyMap_Key const* key);

/// Same as MayMap_erase() except takes @p key by value.
ZL_TABLE_INLINE bool MyMap_eraseVal(MyMap* map, MyMap_Key const key);

/**
 * @returns an iterator into the map that can be used to iterate through
 * all the elements and provide const pointers to the entries.
 * WARNING: The iterator is invalidated by MyMap_reserve(), MyMap_insert(), and
 * MyMap_erase().
 *
 * E.g. it can be used like this.
 *
 *     MyMap_Iter iter = MyMap_iter(map);
 *     for (MyMap_Entry const* entry; entry = MyMap_Iter_next(&iter); ) {
 *         use(entry);
 *     }
 */
ZL_TABLE_INLINE MyMap_Iter MyMap_iter(MyMap const* map);

/**
 * @returns The entry pointed to by @p iter and advances it to the next element.
 * Returns NULL once the iterator reaches the end of the map.
 */
ZL_TABLE_INLINE MyMap_Entry const* MyMap_Iter_next(MyMap_Iter* iter);

/**
 * @returns The entry pointed to by @p iter or NULL if it's reached the end of
 * the map.
 */
ZL_TABLE_INLINE MyMap_Entry const* MyMap_Iter_get(MyMap_Iter iter);

/**
 * @returns an iterator into the map that can be used to iterate
 * through all the elements and provide mutable pointers to the entries.
 * WARNING: The iterator is invalidated by MyMap_reserve(), MyMap_insert(), and
 * MyMap_erase().
 */
ZL_TABLE_INLINE MyMap_IterMut MyMap_iterMut(MyMap* map);

/**
 * @returns The entry pointed to by @p iter and advances it to the next element.
 * Returns NULL once the iterator reaches the end of the map.
 */
ZL_TABLE_INLINE MyMap_Entry* MyMap_IterMut_next(MyMap_IterMut* iter);

/**
 * @returns The entry pointed to by @p iter or NULL if it's reached the end of
 * the map.
 */
ZL_TABLE_INLINE MyMap_Entry* MyMap_IterMut_get(MyMap_IterMut iter);

/**
 * Converts a mutable iterator into an immutable iterator.
 *
 * @returns An immutable copy of iter.
 */
ZL_TABLE_INLINE MyMap_Iter MyMap_IterMut_const(MyMap_IterMut iter);

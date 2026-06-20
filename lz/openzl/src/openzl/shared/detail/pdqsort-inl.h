/*
    This source was modified to work with Zstrong.
*/

/*
    pdqsort.h - Pattern-defeating quicksort.

    Copyright (c) 2021 Orson Peters

    This software is provided 'as-is', without any express or implied warranty.
   In no event will the authors be held liable for any damages arising from the
   use of this software.

    Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software in a
   product, an acknowledgment in the product documentation would be appreciated
   but is not required.

    2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.

    3. This notice may not be removed or altered from any source distribution.
*/

#ifndef PDQSORT_H
#define PDQSORT_H

#include "openzl/shared/bits.h"
#include "openzl/shared/portability.h"
#include "openzl/shared/utils.h"

// We'd use `uint64_t` as the default to avoid linter issues
// But T should be defined by the user as the desired type.
#ifndef T
#    define T uint64_t
#endif

#ifndef COMP
#    define COMP(a, b) (*(a) < *(b))
#endif

#define INSERTION_SORT_THRESHOLD ((size_t)24)
#define NINTHER_THRESHOLD ((size_t)128)
#define BLOCK_SIZE ((size_t)64)
#define CACHELINE_SIZE ((size_t)64)

ZL_INLINE void swap(T* a, T* b)
{
    T tmp = *a;
    *a    = *b;
    *b    = tmp;
}

// Function to heapify a subtree rooted with node i which is an index in data[]
ZL_INLINE void heapify(T* data, size_t nbElts, size_t current)
{
    while (true) {
        size_t left  = 2 * current + 1; // left = 2*i + 1
        size_t right = 2 * current + 2; // right = 2*i + 2
        // Use ternary operator instead of if-else
        size_t largest = (left < nbElts && COMP(data + current, data + left))
                ? left
                : current;
        largest        = (right < nbElts && COMP(data + largest, data + right))
                       ? right
                       : largest;

        if (largest == current) {
            break; // if the root is the largest then break the loop
        }
        swap(&data[current], &data[largest]);
        current = largest; // update the root to be the largest and continue the
                           // loop
    }
}

// Sorts [begin, end) using heap sort with the given comparison
// function.
ZL_INLINE void heapSort(T* begin, T* end)
{
    if (end <= begin)
        return;
    size_t const nbElts = (size_t)(end - begin);
    if (nbElts <= 1)
        return;
    // Build heap (rearrange array)
    for (size_t i = nbElts / 2; i-- > 0;)
        heapify(begin, nbElts, (size_t)i);
    // One by one extract an element from heap
    for (size_t i = nbElts; i-- > 0;) {
        // Move current root to end
        swap(begin, begin + i);
        // call max heapify on the reduced heap
        heapify(begin, (size_t)i, 0);
    }
}

// Sorts [begin, end) using insertion sort with the given comparison
// function.
ZL_INLINE void insertion_sort(T* begin, T* end)
{
    if (begin == end)
        return;

    for (T* cur = begin + 1; cur != end; ++cur) {
        T* sift   = cur;
        T* sift_1 = cur - 1;

        // Compare first so we can avoid 2 moves for an element already
        // positioned correctly.
        if (COMP(sift, sift_1)) {
            T tmp = *sift;

            do {
                *sift-- = *sift_1;
            } while (sift != begin && COMP(&tmp, --sift_1));

            *sift = tmp;
        }
    }
}

// Sorts [begin, end) using insertion sort with the given comparison
// function. Assumes
// *(begin - 1) is an element smaller than or equal to any element in
// [begin, end).
ZL_INLINE void unguarded_insertion_sort(T* begin, T* end)
{
    if (begin == end)
        return;

    for (T* cur = begin + 1; cur != end; ++cur) {
        T* sift   = cur;
        T* sift_1 = cur - 1;

        // Compare first so we can avoid 2 moves for an element already
        // positioned correctly.
        if (COMP(sift, sift_1)) {
            T tmp = *sift;

            do {
                *sift-- = *sift_1;
            } while (COMP(&tmp, --sift_1));

            *sift = tmp;
        }
    }
}

// Attempts to use insertion sort on [begin, end). Will return false if more
// than partial_insertion_sort_limit elements were moved, and abort sorting.
// Otherwise it will successfully sort and return true.
ZL_INLINE bool partial_insertion_sort(T* begin, T* end)
{
    const size_t kPartialInsertionSortLimit = 8;

    if (begin == end)
        return true;

    size_t limit = 0;
    for (T* cur = begin + 1; cur != end; ++cur) {
        T* sift   = cur;
        T* sift_1 = cur - 1;

        // Compare first so we can avoid 2 moves for an element already
        // positioned correctly.
        if (COMP(sift, sift_1)) {
            T tmp = *sift;

            do {
                *sift-- = *sift_1;
            } while (sift != begin && COMP(&tmp, --sift_1));

            *sift = tmp;
            limit += (size_t)(cur - sift);
        }

        if (limit > kPartialInsertionSortLimit)
            return false;
    }

    return true;
}

ZL_INLINE void sort2(T* a, T* b)
{
    if (COMP(b, a)) {
        swap(a, b);
    }
}

// Sorts the elements *a, *b and *c using comparison function COMP.
ZL_INLINE void sort3(T* a, T* b, T* c)
{
    sort2(a, b);
    sort2(b, c);
    sort2(a, b);
}

ZL_INLINE void* align_cacheline(void* p)
{
    uintptr_t ip = (uintptr_t)p;
    ip           = (ip + CACHELINE_SIZE - 1) & ~(CACHELINE_SIZE - 1);
    return (void*)ip;
}

ZL_INLINE void swap_offsets(
        T* first,
        T* last,
        unsigned char* offsets_l,
        unsigned char* offsets_r,
        size_t num,
        bool use_swaps)
{
    if (use_swaps) {
        // This case is needed for the descending distribution, where we
        // need to have proper swapping for pdqsort to remain O(n).
        for (size_t i = 0; i < num; ++i) {
            swap(first + offsets_l[i], last - offsets_r[i]);
        }
    } else if (num > 0) {
        T* l  = first + offsets_l[0];
        T* r  = last - offsets_r[0];
        T tmp = *l;
        *l    = *r;
        for (size_t i = 1; i < num; ++i) {
            l  = first + offsets_l[i];
            *r = *l;
            r  = last - offsets_r[i];
            *l = *r;
        }
        *r = tmp;
    }
}

// Partitions [begin, end) around pivot *begin using comparison function
// COMP. Elements equal to the pivot are put in the right-hand partition.
// Returns the position of the pivot after partitioning and whether the
// passed sequence already was correctly partitioned. Assumes the pivot is a
// median of at least 3 elements and that [begin, end) is at least
// INSERTION_SORT_THRESHOLD long. Uses branchless partitioning.
ZL_INLINE T*
partition_right_branchless(T* begin, T* end, bool* alreadyPartitionedOut)
{
    // Move pivot into local for speed.
    T pivot  = *begin;
    T* first = begin;
    T* last  = end;

    // Find the first element greater than or equal than the pivot (the
    // median of 3 guarantees this exists).
    while (COMP(++first, &pivot))
        ;

    // Find the first element strictly smaller than the pivot. We have to
    // guard this search if there was no element before *first.
    if (first - 1 == begin)
        while (first < last && !COMP(--last, &pivot))
            ;
    else
        while (!COMP(--last, &pivot))
            ;

    // If the first pair of elements that should be swapped to partition are
    // the same element, the passed in sequence already was correctly
    // partitioned.
    bool already_partitioned = first >= last;
    if (!already_partitioned) {
        swap(first, last);
        ++first;

        // The following branchless partitioning is derived from
        // "BlockQuicksort: How Branch Mispredictions don’t affect
        // Quicksort" by Stefan Edelkamp and Armin Weiss, but heavily
        // micro-optimized.
        unsigned char offsets_l_storage[BLOCK_SIZE + CACHELINE_SIZE];
        unsigned char offsets_r_storage[BLOCK_SIZE + CACHELINE_SIZE];
        unsigned char* offsets_l = align_cacheline(offsets_l_storage);
        unsigned char* offsets_r = align_cacheline(offsets_r_storage);

        T* offsets_l_base = first;
        T* offsets_r_base = last;
        size_t num_l, num_r, start_l, start_r;
        num_l = num_r = start_l = start_r = 0;

        while (first < last) {
            // Fill up offset blocks with elements that are on the wrong
            // side. First we determine how much elements are considered for
            // each offset block.
            size_t num_unknown = (size_t)(last - first);
            size_t left_split  = num_l == 0
                     ? (num_r == 0 ? num_unknown / 2 : num_unknown)
                     : 0;
            size_t right_split = num_r == 0 ? (num_unknown - left_split) : 0;

            // Fill the offset blocks.
            if (left_split >= BLOCK_SIZE) {
                for (size_t i = 0; i < BLOCK_SIZE;) {
                    offsets_l[num_l] = (unsigned char)i++;
                    num_l += !COMP(first, &pivot);
                    ++first;
                    offsets_l[num_l] = (unsigned char)i++;
                    num_l += !COMP(first, &pivot);
                    ++first;
                    offsets_l[num_l] = (unsigned char)i++;
                    num_l += !COMP(first, &pivot);
                    ++first;
                    offsets_l[num_l] = (unsigned char)i++;
                    num_l += !COMP(first, &pivot);
                    ++first;
                    offsets_l[num_l] = (unsigned char)i++;
                    num_l += !COMP(first, &pivot);
                    ++first;
                    offsets_l[num_l] = (unsigned char)i++;
                    num_l += !COMP(first, &pivot);
                    ++first;
                    offsets_l[num_l] = (unsigned char)i++;
                    num_l += !COMP(first, &pivot);
                    ++first;
                    offsets_l[num_l] = (unsigned char)i++;
                    num_l += !COMP(first, &pivot);
                    ++first;
                }
            } else {
                for (size_t i = 0; i < left_split;) {
                    offsets_l[num_l] = (unsigned char)i++;
                    num_l += !COMP(first, &pivot);
                    ++first;
                }
            }

            if (right_split >= BLOCK_SIZE) {
                for (size_t i = 0; i < BLOCK_SIZE;) {
                    offsets_r[num_r] = (unsigned char)++i;
                    num_r += (size_t)COMP(--last, &pivot);
                    offsets_r[num_r] = (unsigned char)++i;
                    num_r += (size_t)COMP(--last, &pivot);
                    offsets_r[num_r] = (unsigned char)++i;
                    num_r += (size_t)COMP(--last, &pivot);
                    offsets_r[num_r] = (unsigned char)++i;
                    num_r += (size_t)COMP(--last, &pivot);
                    offsets_r[num_r] = (unsigned char)++i;
                    num_r += (size_t)COMP(--last, &pivot);
                    offsets_r[num_r] = (unsigned char)++i;
                    num_r += (size_t)COMP(--last, &pivot);
                    offsets_r[num_r] = (unsigned char)++i;
                    num_r += (size_t)COMP(--last, &pivot);
                    offsets_r[num_r] = (unsigned char)++i;
                    num_r += (size_t)COMP(--last, &pivot);
                }
            } else {
                for (size_t i = 0; i < right_split;) {
                    offsets_r[num_r] = (unsigned char)++i;
                    num_r += (size_t)COMP(--last, &pivot);
                }
            }

            // Swap elements and update block sizes and first/last
            // boundaries.
            size_t num = ZL_MIN(num_l, num_r);
            swap_offsets(
                    offsets_l_base,
                    offsets_r_base,
                    offsets_l + start_l,
                    offsets_r + start_r,
                    num,
                    num_l == num_r);
            num_l -= num;
            num_r -= num;
            start_l += num;
            start_r += num;

            if (num_l == 0) {
                start_l        = 0;
                offsets_l_base = first;
            }

            if (num_r == 0) {
                start_r        = 0;
                offsets_r_base = last;
            }
        }

        // We have now fully identified [first, last)'s proper position.
        // Swap the last elements.
        if (num_l) {
            offsets_l += start_l;
            while (num_l--)
                swap(offsets_l_base + offsets_l[num_l], --last);
            first = last;
        }
        if (num_r) {
            offsets_r += start_r;
            while (num_r--)
                swap(offsets_r_base - offsets_r[num_r], first), ++first;
            last = first;
        }
    }

    // Put the pivot in the right place.
    T* pivot_pos = first - 1;
    *begin       = *pivot_pos;
    *pivot_pos   = pivot;

    *alreadyPartitionedOut = already_partitioned;

    return pivot_pos;
}

// Partitions [begin, end) around pivot *begin using comparison function
// COMP. Elements equal to the pivot are put in the right-hand partition.
// Returns the position of the pivot after partitioning and whether the
// passed sequence already was correctly partitioned. Assumes the pivot is a
// median of at least 3 elements and that [begin, end) is at least
// INSERTION_SORT_THRESHOLD long.
ZL_INLINE T* partition_right(T* begin, T* end, bool* already_partitioned)
{
    // Move pivot into local for speed.
    T pivot = *begin;

    T* first = begin;
    T* last  = end;

    // Find the first element greater than or equal than the pivot (the
    // median of 3 guarantees this exists).
    while (COMP(++first, &pivot))
        ;

    // Find the first element strictly smaller than the pivot. We have to
    // guard this search if there was no element before *first.
    if (first - 1 == begin)
        while (first < last && !COMP(--last, &pivot))
            ;
    else
        while (!COMP(--last, &pivot))
            ;

    // If the first pair of elements that should be swapped to partition are
    // the same element, the passed in sequence already was correctly
    // partitioned.
    *already_partitioned = first >= last;

    // Keep swapping pairs of elements that are on the wrong side of the
    // pivot. Previously swapped pairs guard the searches, which is why the
    // first iteration is special-cased above.
    while (first < last) {
        swap(first, last);
        while (COMP(++first, &pivot))
            ;
        while (!COMP(--last, &pivot))
            ;
    }

    // Put the pivot in the right place.
    T* pivot_pos = first - 1;
    *begin       = *pivot_pos;
    *pivot_pos   = pivot;

    return pivot_pos;
}

// Similar function to the one above, except elements equal to the pivot are
// put to the left of the pivot and it doesn't check or return if the passed
// sequence already was partitioned. Since this is rarely used (the many
// equal case), and in that case pdqsort already has O(n) performance, no
// block quicksort is applied here for simplicity.
ZL_INLINE T* partition_left(T* begin, T* end)
{
    T pivot  = *begin;
    T* first = begin;
    T* last  = end;

    while (COMP(&pivot, --last))
        ;

    if (last + 1 == end)
        while (first < last && !COMP(&pivot, ++first))
            ;
    else
        while (!COMP(&pivot, ++first))
            ;

    while (first < last) {
        swap(first, last);
        while (COMP(&pivot, --last))
            ;
        while (!COMP(&pivot, ++first))
            ;
    }

    T* pivot_pos = last;
    *begin       = *pivot_pos;
    *pivot_pos   = pivot;

    return pivot_pos;
}

ZL_INLINE void pdqsort_loop(
        const bool branchless,
        T* begin,
        T* end,
        int bad_allowed,
        bool leftmost)
{
    // Use a while loop for tail recursion elimination.
    while (true) {
        size_t size = (size_t)(end - begin);

        // Insertion sort is faster for small arrays.
        if (size < INSERTION_SORT_THRESHOLD) {
            if (leftmost)
                insertion_sort(begin, end);
            else
                unguarded_insertion_sort(begin, end);
            return;
        }

        // Choose pivot as median of 3 or pseudomedian of 9.
        size_t s2 = size / 2;
        if (size > NINTHER_THRESHOLD) {
            sort3(begin, begin + s2, end - 1);
            sort3(begin + 1, begin + (s2 - 1), end - 2);
            sort3(begin + 2, begin + (s2 + 1), end - 3);
            sort3(begin + (s2 - 1), begin + s2, begin + (s2 + 1));
            swap(begin, begin + s2);
        } else
            sort3(begin + s2, begin, end - 1);

        // If *(begin - 1) is the end of the right partition of a previous
        // partition operation there is no element in [begin, end) that is
        // smaller than *(begin - 1). Then if our pivot compares equal to
        // *(begin - 1) we change strategy, putting equal elements in the
        // left partition, greater elements in the right partition. We do
        // not have to recurse on the left partition, since it's sorted (all
        // equal).
        if (!leftmost && !COMP((begin - 1), begin)) {
            begin = partition_left(begin, end) + 1;
            continue;
        }

        // Partition and get results.
        bool already_partitioned;
        T* pivot_pos = branchless
                ? partition_right_branchless(begin, end, &already_partitioned)
                : partition_right(begin, end, &already_partitioned);

        // Check for a highly unbalanced partition.
        size_t l_size          = (size_t)(pivot_pos - begin);
        size_t r_size          = (size_t)(end - (pivot_pos + 1));
        bool highly_unbalanced = l_size < size / 8 || r_size < size / 8;

        // If we got a highly unbalanced partition we shuffle elements to
        // break many patterns.
        if (highly_unbalanced) {
            // If we had too many bad partitions, switch to heapsort to
            // guarantee O(n log n).
            if (--bad_allowed == 0) {
                heapSort(begin, end);
                return;
            }

            if (l_size >= INSERTION_SORT_THRESHOLD) {
                swap(begin, begin + l_size / 4);
                swap(pivot_pos - 1, pivot_pos - l_size / 4);

                if (l_size > NINTHER_THRESHOLD) {
                    swap(begin + 1, begin + (l_size / 4 + 1));
                    swap(begin + 2, begin + (l_size / 4 + 2));
                    swap(pivot_pos - 2, pivot_pos - (l_size / 4 + 1));
                    swap(pivot_pos - 3, pivot_pos - (l_size / 4 + 2));
                }
            }

            if (r_size >= INSERTION_SORT_THRESHOLD) {
                swap(pivot_pos + 1, pivot_pos + (1 + r_size / 4));
                swap(end - 1, end - r_size / 4);

                if (r_size > NINTHER_THRESHOLD) {
                    swap(pivot_pos + 2, pivot_pos + (2 + r_size / 4));
                    swap(pivot_pos + 3, pivot_pos + (3 + r_size / 4));
                    swap(end - 2, end - (1 + r_size / 4));
                    swap(end - 3, end - (2 + r_size / 4));
                }
            }
        } else {
            // If we were decently balanced and we tried to sort an already
            // partitioned sequence try to use insertion sort.
            if (already_partitioned && partial_insertion_sort(begin, pivot_pos)
                && partial_insertion_sort(pivot_pos + 1, end))
                return;
        }

        // Sort the left partition first using recursion and do tail
        // recursion elimination for the right-hand partition.
        pdqsort_loop(branchless, begin, pivot_pos, bad_allowed, leftmost);
        begin    = pivot_pos + 1;
        leftmost = false;
    }
}

ZL_INLINE void pdqsort_branch(T* begin, T* end)
{
    if (begin == end)
        return;
    pdqsort_loop(false, begin, end, ZL_nextPow2((size_t)(end - begin)), true);
}

ZL_INLINE void pdqsort_branchless(T* begin, T* end)
{
    if (begin == end)
        return;
    pdqsort_loop(true, begin, end, ZL_nextPow2((size_t)(end - begin)), true);
}

#endif

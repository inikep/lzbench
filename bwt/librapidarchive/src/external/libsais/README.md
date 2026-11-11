# libsais

The libsais library provides fast (see [Benchmarks](#benchmarks) below) linear-time construction of suffix array (SA), generalized suffix array (GSA), longest common prefix (LCP) array, permuted LCP (PLCP) array, Burrows-Wheeler transform (BWT) and inverse BWT, based on the induced sorting algorithm described in the following papers (with optional OpenMP support for multi-threaded parallel construction):
* Ge Nong, Sen Zhang, Wai Hong Chan *Two Efficient Algorithms for Linear Suffix Array Construction*, 2009
* Juha Karkkainen, Giovanni Manzini, Simon J. Puglisi *Permuted Longest-Common-Prefix Array*, 2009
* Nataliya Timoshevskaya, Wu-chun Feng *SAIS-OPT: On the characterization and optimization of the SA-IS algorithm for suffix array construction*, 2014
* Jing Yi Xie, Ge Nong, Bin Lao, Wentao Xu *Scalable Suffix Sorting on a Multicore Machine*, 2020

Copyright (c) 2021-2026 Ilya Grebnov <ilya.grebnov@gmail.com>

>The libsais is inspired by [libdivsufsort](https://github.com/y-256/libdivsufsort), [sais](https://sites.google.com/site/yuta256/sais) libraries by Yuta Mori and [msufsort](https://github.com/michaelmaniscalco/msufsort) by Michael Maniscalco.

## libcubwt
If you are looking for even faster construction times, you can try [libcubwt](https://github.com/IlyaGrebnov/libcubwt) a library for GPU-based suffix array, inverse suffix array and Burrows-Wheeler transform construction.

## Introduction
The libsais provides simple C99 API to construct suffix array and Burrows-Wheeler transformed string from a given string over constant-size alphabet. The algorithm runs in a linear time using typically only ~16KB of extra memory (with 2n bytes as absolute worst-case; where n is the length of the string). OpenMP acceleration uses 200KB of addition memory per thread.

> * The libsais works with compilers from GNU, Microsoft and Intel, but I recommend Clang for best performance.
> * The libsais is sensitive to fast memory and software prefetching and might not be suitable for some workloads. Please benchmark yourself.

## License
The libsais is released under the [Apache License Version 2.0 with LLVM Exceptions](LICENSE "Apache license with LLVM Exceptions")

## Multi-threading
The libsais is memory-bound, so performance scales primarily with memory bandwidth and concurrency, not raw compute. The optimal number of threads for suffix array construction depends on CPU and memory architecture, as different systems (DDR4 vs DDR5, Intel vs AMD, single- vs multi-CCD) saturate memory at different points with maximum throughput generally following the number of memory channels rather than total core count. The x86-64 dual-channel systems typically saturate near 8 threads, but in practice may show good scaling at 6, 12 or even 16 threads.

> * An OMP_STACKSIZE below 32K is not supported.
> * Inverse BWT uses the requested number of threads, as suffix array construction does; if the operating system cannot create them (e.g. under a per-user process limit), the OpenMP runtime ends the program.
> * In libsais64 and libsais16x64, the number of threads must be from 0 to INT_MAX; a larger number returns -1.
> * If an input symbol frequency table (freq) is given to inverse BWT, it must match the symbol counts of the BWT string, as single-threaded calls use it without checking.
> * With MinGW (libgomp), a very large number of threads can overflow the main thread's stack, and long runs of nested calls with many threads can exhaust the process's handles, as libgomp leaks one for each blocked thread of a nested team.

## Changes
_For the full changelog, see [CHANGES](CHANGES)._

* November 9, 2026 (2.12.2)
  * Improved performance of suffix array and Burrows-Wheeler transform construction and of the inverse Burrows-Wheeler transform, both single- and multi-threaded.
  * Fixed a hang at process exit with MinGW (libgomp) after a multi-threaded inverse BWT: inverse BWT now uses the requested number of threads in every parallel region, as suffix array construction does.
  * libsais64 and libsais16x64 now return -1 for a thread count above INT_MAX.
  * Fixed out-of-bounds writes in libsais64 and libsais16x64 with thread counts of 2^32 or more in 32-bit builds.
  * Worked around a GCC issue that made AVX-512 builds fault under AddressSanitizer.
  * Resolved compiler warnings under -Wstrict-aliasing=1 and -Wconversion.
* October 21, 2026 (2.12.1)
  * Fixed compilation errors and warnings with MSVC for ARM64 and ARM64EC targets.
  * Fixed incorrect results on big-endian AArch64 targets, and of libsais64 and libsais16x64 on all big-endian targets.
  * Fixed byte-order detection on big-endian BSD targets.
  * Big-endian targets now always use the portable (non-SIMD) code.
* October 18, 2026 (2.12.0)
  * Changed license to Apache License 2.0 with LLVM Exceptions.
  * Support for cyclic Burrows-Wheeler transform and inverse BWT construction (e.g. as used by bzip2).
  * Significantly improved performance of inverse BWT, both single- and multi-threaded.
  * Improved performance of suffix array and Burrows-Wheeler transform construction.
  * Fixed crashes and incorrect results when OpenMP provides fewer threads than requested (e.g. under OMP_THREAD_LIMIT).
  * Fixed out-of-bounds memory access in inverse BWT for inputs that are not a valid BWT.
  * Fixed handling of invalid alphabet sizes, of alphabets larger than 2^31 symbols (libsais64, libsais16x64) and of very large thread counts in 32-bit builds.
  * Resolved compiler warnings with GCC 16.
* September 1, 2025 (2.10.4)
  * Tuned prefetch distance for improved throughput.
* August 12, 2025 (2.10.3)
  * No functional changes, added CMake install and export package rules.
* June 10, 2025 (2.10.2)
  * Improved performance of suffix array and burrows wheeler transform construction on degenerate inputs.
* May 11, 2025 (2.10.1)
  * No functional changes, slightly improved performance.
* April 12, 2025 (2.10.0)
  * Improved performance, with noticeable gains on ARM architecture.
  * Fixed compiler warnings and addressed undefined behavior.
* March 19, 2025 (2.9.1)
  * No functional changes, resolved compiler warnings & undefined behavior.
* March 16, 2025 (2.9.0)
  * Support for generalized suffix array (GSA) construction.
  * Support for longest common prefix array (LCP) construction for generalized suffix array (GSA).

## Versions of the libsais
* [libsais.c](src/libsais.c) (and corresponding [libsais.h](include/libsais.h)) is for suffix array, GSA, PLCP, LCP, forward BWT and reverse BWT construction over 8-bit inputs smaller than 2GB (2147483648 bytes).
  * [libsais64.c](src/libsais64.c) (and corresponding [libsais64.h](include/libsais64.h)) is optional extension of the library for inputs larger or equlas to 2GB (2147483648 bytes).
  * This versions of the library could also be used to construct suffix array of an integer array (with a caveat that input array must be mutable).
* [libsais16.c](src/libsais16.c) + [libsais16x64.c](src/libsais16x64.c) (and corresponding [libsais16.h](include/libsais16.h) + [libsais16x64.h](include/libsais16x64.h)) is independent version of the library for 16-bit inputs.
  * This version of the library could also be used to construct suffix array and BWT of a set of strings by adding a unique end-of-string symbol to each string and then computing the result for the concatenated string.

## Examples of APIs (see [libsais.h](include/libsais.h), [libsais16.h](include/libsais16.h), [libsais16x64.h](include/libsais16x64.h) and [libsais64.h](include/libsais64.h) for complete APIs list)
```c
    /**
    * Constructs the suffix array of a given string.
    * @param T [0..n-1] The input string.
    * @param SA [0..n-1+fs] The output array of suffixes.
    * @param n The length of the given string.
    * @param fs The extra space available at the end of SA array (0 should be enough for most cases).
    * @param freq [0..255] The output symbol frequency table (can be NULL).
    * @return 0 if no error occurred, -1 or -2 otherwise.
    */
    int32_t libsais(const uint8_t * T, int32_t * SA, int32_t n, int32_t fs, int32_t * freq);

    /**
    * Constructs the suffix array of a given integer array.
    * Note, during construction input array will be modified, but restored at the end if no errors occurred.
    * @param T [0..n-1] The input integer array.
    * @param SA [0..n-1+fs] The output array of suffixes.
    * @param n The length of the integer array.
    * @param k The alphabet size of the input integer array.
    * @param fs Extra space available at the end of SA array (can be 0, but 4k or better 6k is recommended for optimal performance).
    * @return 0 if no error occurred, -1 or -2 otherwise.
    */
    int32_t libsais_int(int32_t * T, int32_t * SA, int32_t n, int32_t k, int32_t fs);

    /**
    * Constructs the burrows-wheeler transformed string (BWT) of a given string.
    * @param T [0..n-1] The input string.
    * @param U [0..n-1] The output string (can be T).
    * @param A [0..n-1+fs] The temporary array.
    * @param n The length of the given string.
    * @param fs The extra space available at the end of A array (0 should be enough for most cases).
    * @param freq [0..255] The output symbol frequency table (can be NULL).
    * @return The primary index if no error occurred, -1 or -2 otherwise.
    */
    int32_t libsais_bwt(const uint8_t * T, uint8_t * U, int32_t * A, int32_t n, int32_t fs, int32_t * freq);

    /**
    * Constructs the original string from a given burrows-wheeler transformed string (BWT) with primary index.
    * @param T [0..n-1] The input string.
    * @param U [0..n-1] The output string (can be T).
    * @param A [0..n] The temporary array (NOTE, temporary array must be n + 1 size).
    * @param n The length of the given string.
    * @param freq [0..255] The input symbol frequency table (can be NULL).
    * @param i The primary index.
    * @return 0 if no error occurred, -1 or -2 otherwise.
    */
    int32_t libsais_unbwt(const uint8_t * T, uint8_t * U, int32_t * A, int32_t n, const int32_t * freq, int32_t i);

    /**
    * Constructs the permuted longest common prefix array (PLCP) of a given string and a suffix array.
    * @param T [0..n-1] The input string.
    * @param SA [0..n-1] The input suffix array.
    * @param PLCP [0..n-1] The output permuted longest common prefix array.
    * @param n The length of the string and the suffix array.
    * @return 0 if no error occurred, -1 otherwise.
    */
    int32_t libsais_plcp(const uint8_t * T, const int32_t * SA, int32_t * PLCP, int32_t n);
```

## Example installation using [CPM](https://github.com/cpm-cmake/CPM.cmake)
```cmake
CPMAddPackage(
  NAME libsais
  GITHUB_REPOSITORY IlyaGrebnov/libsais
  GIT_TAG v2.12.2
  OPTIONS
    "LIBSAIS_USE_OPENMP OFF"
    "LIBSAIS_BUILD_SHARED_LIB OFF"
)

target_link_libraries(<your target> libsais)
```

# Algorithm description
The libsais uses the SA-IS (Suffix Array Induced Sorting) algorithm to construct both the suffix array and the Burrows-Wheeler transform through recursive decomposition and induced sorting:
* Initially, the algorithm classifies each position in a string as either an S-type or an L-type, based on whether the suffix starting at that position is lexicographically smaller or larger than the suffix at the adjacent right position. Positions identified as S-type, which have an adjacent left L-type position, are further categorized as LMS-type (Leftmost S-type) positions. Next, the algorithm splits the input string into LMS substrings, which start at an LMS-type position and extend up to the next adjacent LMS-type position. These LMS substrings are then lexicographically sorted through induced sorting and subsequently replaced in the input string with their corresponding sorted ranks, thus forming a new, compacted string. This compacted string reduces the problem size, enabling the algorithm to perform a recursive decomposition in which it is reapplied to construct the suffix array for the compacted string. And at the end of the recursive call, the suffix array for the input string is constructed from the suffix array of the compacted string using another round of induced sorting.
* The induced sorting is a core mechanic of the SA-IS algorithm and is employed twice during each recursive call: initially before the recursive call to establish the order of LMS substrings, and subsequently after the recursive call to finalize the order of the suffixes of the string. This process involves two sequential scans: a left-to-right scan that determines the order of L-type positions based on the LMS-type positions, followed by a right-to-left scan that establishes the order of S-type positions based on L-type positions. These scans efficiently extend the ordering from LMS-type positions to all positions in the string.

The SA-IS algorithm is quite elegant, yet implementing it efficiently presents multiple challenges. The primary challenge is that the SA-IS algorithm exhibits random memory access patterns, which can significantly decrease efficiency due to cache misses. Another significant challenge is that the SA-IS algorithm is not a lightweight construction algorithm; it requires additional memory to support positions classification, induced sorting, compacted string representations, and recursive decomposition. To circumvent this, the libsais implements careful optimizations that are worth highlighting:
* The libsais is meticulously designed from the ground up to leverage the capabilities of modern microprocessors, aiming to minimize various stalls and enhance throughput through instruction-level parallelism. The library employs sophisticated techniques such as manual loop unrolling, software prefetching, and branch elimination to achieve this goal. Moreover, it strives to minimize the number of passes over the data by combining multiple operations into a single function. A prime example of these techniques could be observed in the initialization phase of the SA-IS algorithm. In this phase, the entire logic required to classify positions, count symbols into various buckets, and segment the string into LMS substrings is executed through a single, completely branch-less loop:
```c
        for (i = m - 1, j = omp_block_start + prefetch_distance + 3; i >= j; i -= 4)
        {
            libsais_prefetchr(&T[i - 2 * prefetch_distance]);

            libsais_prefetchw(&buckets[BUCKETS_INDEX4(T[i - prefetch_distance - 0], 0)]);
            libsais_prefetchw(&buckets[BUCKETS_INDEX4(T[i - prefetch_distance - 1], 0)]);
            libsais_prefetchw(&buckets[BUCKETS_INDEX4(T[i - prefetch_distance - 2], 0)]);
            libsais_prefetchw(&buckets[BUCKETS_INDEX4(T[i - prefetch_distance - 3], 0)]);

            c1 = T[i - 0]; f1 = (fast_uint_t)(c1 > (c0 - (fast_sint_t)(f0))); SA[m] = (sa_sint_t)(i + 1); m -= (f1 & ~f0);
            buckets[BUCKETS_INDEX4((fast_uint_t)c0, f0 + f0 + f1)]++;

            c0 = T[i - 1]; f0 = (fast_uint_t)(c0 > (c1 - (fast_sint_t)(f1))); SA[m] = (sa_sint_t)(i - 0); m -= (f0 & ~f1);
            buckets[BUCKETS_INDEX4((fast_uint_t)c1, f1 + f1 + f0)]++;

            c1 = T[i - 2]; f1 = (fast_uint_t)(c1 > (c0 - (fast_sint_t)(f0))); SA[m] = (sa_sint_t)(i - 1); m -= (f1 & ~f0);
            buckets[BUCKETS_INDEX4((fast_uint_t)c0, f0 + f0 + f1)]++;

            c0 = T[i - 3]; f0 = (fast_uint_t)(c0 > (c1 - (fast_sint_t)(f1))); SA[m] = (sa_sint_t)(i - 2); m -= (f0 & ~f1);
            buckets[BUCKETS_INDEX4((fast_uint_t)c1, f1 + f1 + f0)]++;
        }
```
* To sort LMS substrings lexicographically and compute their ranks, the libsais algorithm begins by gathering LMS-type positions as they appear in the string, placing them at the end of the suffix array. The library then employs two passes of induced sorting, which concludes with these same LMS-type positions ordered lexicographically at the beginning of the suffix array. Once all LMS-type positions are sorted, the ranks of the LMS substrings are computed by inspecting each pair of adjacent positions to determine if the corresponding LMS substrings are identical. If they are the same, they receive the same rank; otherwise, the rank is incremented by one. 
  * The first challenge of induced sorting is that, during passes over the suffix array, we need to examine each value to determine if it represents a valid position or an empty space, whether this position is not the beginning of the string (and thus could induce another position), and if the induced position is going to be of the necessary type (for example, during a left-to-right scan, we are only inducing L-type positions). This process can cause branch mispredictions and corresponding microprocessor stalls. To address this challenge, libsais employs following techniques. Firstly, the library uses two pointers per induction bucket, each pointing to different sections of the suffix array depending on the type of positions these positions will be inducing next. This approach allows for the separation of LS-type (meaning S-type, which induces L-type; this is the same as LMS-type) and LL-type positions needed for the left-to-right scan from SL-type and SS-type positions needed for the right-to-left scan. Secondly, by understanding the distribution of symbols based on their position types and the types they induce (i.e., SS, SL, LS, LL), we can pre-calculate pointers for each bucket, leaving no empty spaces. And thirdly, by removing the first LMS position and all positions left of it from the initial gathering and distribution, we eliminate the need to check whether a position is not the beginning of the string. These techniques not only result in a completely branch-less loop for each induction sorting pass but also eliminate redundant scanning and the final gathering of LMS-type positions at the beginning of the suffix array.
  * The second challenge arises after induced sorting when we need to compute the ranks of LMS substrings. To accomplish this, we must first calculate and store the lengths of LMS substrings and then inspect each pair of adjacent LMS-type positions to determine if the corresponding LMS substrings are identical. This comparison starts with their lengths, and if they are the same, proceeds to compare the substrings themselves. Such operations exhibits random memory access patterns, which can significantly decrease efficiency due to cache misses. However, libsais avoids this inefficient logic by incorporating the ranking of LMS substrings as part of the induced sorting process itself. The library achieves this by marking the most significant bit (MSB) of positions in the suffix array that start new ranking groups. Each time a position is processed during induced sorting, the library checks the MSB and increments the current rank if the beginning of a new ranking group is encountered. Additionally, for each pointer in an induction bucket, the rank of the previous induced position is maintained. Whenever another position is induced, this previous rank is used to determine whether to mark the newly induced position as the beginning of a new rank group. All the logic to update the ranks and mark the beginnings of new ranking groups is implemented using bit manipulation and is completely branch-less.
```c
    for (i = omp_block_start, j = omp_block_start + omp_block_size - 2 * prefetch_distance - 1; i < j; i += 2)
    {
        libsais_prefetchr(&SA[i + 3 * prefetch_distance]);

        libsais_prefetchr(&T[SA[i + 2 * prefetch_distance + 0] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i + 2 * prefetch_distance + 0] & SAINT_MAX] - 2);
        libsais_prefetchr(&T[SA[i + 2 * prefetch_distance + 1] & SAINT_MAX] - 1);
        libsais_prefetchr(&T[SA[i + 2 * prefetch_distance + 1] & SAINT_MAX] - 2);

        sa_sint_t p0 = SA[i + prefetch_distance + 0] & SAINT_MAX; sa_sint_t v0 = BUCKETS_INDEX4(T[p0 - (p0 > 0)], 0); libsais_prefetchw(&buckets[v0]);
        sa_sint_t p1 = SA[i + prefetch_distance + 1] & SAINT_MAX; sa_sint_t v1 = BUCKETS_INDEX4(T[p1 - (p1 > 0)], 0); libsais_prefetchw(&buckets[v1]);

        sa_sint_t p2 = SA[i + 0]; d += (p2 < 0); p2 &= SAINT_MAX; sa_sint_t v2 = BUCKETS_INDEX4(T[p2 - 1], T[p2 - 2] >= T[p2 - 1]);
        SA[buckets[v2]++] = (p2 - 1) | (sa_sint_t)((sa_uint_t)(buckets[2 + v2] != d) << (SAINT_BIT - 1)); buckets[2 + v2] = d;

        sa_sint_t p3 = SA[i + 1]; d += (p3 < 0); p3 &= SAINT_MAX; sa_sint_t v3 = BUCKETS_INDEX4(T[p3 - 1], T[p3 - 2] >= T[p3 - 1]);
        SA[buckets[v3]++] = (p3 - 1) | (sa_sint_t)((sa_uint_t)(buckets[2 + v3] != d) << (SAINT_BIT - 1)); buckets[2 + v3] = d;
    }
```
* In the SA-IS algorithm, after induced sorting, the ranks of LMS substrings are computed in suffix order. These ranks then need to be scattered to reorder them in string order before being gathered again to form the compacted string for recursion. At this point, some LMS substrings may be unique, meaning they don't share their rank with any other LMS substring. Being unique, these substrings are essentially already sorted, and their position relative to other LMS substrings is already determined. However, these unique LMS substrings may still be necessary for sorting other, non-unique LMS substrings during recursion-unless a unique LMS substring is immediately followed by another unique LMS substring in the string. In such cases, the rank of any subsequent unique LMS substrings becomes redundant in the compacted string, as it will not be utilized. Leveraging this insight, libsais employs a strategy to further reduce the size of the compacted string by omitting such redundant LMS substring ranks. This process involves a few steps. First, unique LMS substrings are identified by looking ahead while scanning LMS-positions in the suffix array during the ranking and scattering phase. When scattering LMS substring ranks to form the compacted string, the most significant bit (MSB) of the rank is used to mark that this rank is unique. Next, as the library scans the ranks in string order and detects tandems of unique ranks using the MSB, it then  recalculates the MSB for ranks which are redundant, thus markign them for removal from the compacted string. Subsequently, the libsais rescans the LMS-positions in suffix order to recompute the ranks, now focusing only on the ranks of the remaining LMS substrings. The library also uses MSB of first symbol of LMS substrings to mark that LMS substring is removed from the compacted string. Finally, the library builds the compacted string based on the newly recalculated ranks for the remaining LMS substrings, while also saving the final positions for the removed LMS substrings before proceeding with recursion. This reduction process not only further decreases the size of the compacted string but also reduces the alphabet size of the reduced string and creates additional free space in the suffix array, which can be utilized during recursion.
* The SA-IS algorithm, while robust for suffix array construction, is not considered lightweight due to its need for additional memory for tasks such as position classification, induced sorting, the creation of compacted string representations, and recursive decomposition. To mitigate this, libsais optimizes memory usage by not storing position classifications and striving to reuse the memory space allocated for the suffix array for induced sorting, compacted string representations, and recursive decomposition processes. Since position classifications are not stored, the library recalculates them as needed, typically involving checks of adjacent symbols for a given position. Although this approach may seem straightforward, it introduces the challenge of random memory access. Nevertheless, libsais manages these accesses in a manner that either avoids unnecessary memory fetches or minimizes cache penalties. In situations where avoiding cache penalties is unfeasible, the library leverages the most significant bit (MSB) bits for computations, as branch mispredictions on modern microprocessors generally incur lower penalties than cache misses. Memory reuse for the suffix array, despite appearing straightforward, also presents hidden challenges related to implementation complexity. In certain cases, the available space in the suffix array may not suffice for the most optimal algorithm implementation mentioned above. Although such instances are rare, the library aims to deliver optimal performance without additional memory allocation by resorting to a less efficient variant of induced sorting. To accommodate various scenarios, libsais includes four distinct implementations tailored to different breakpoints based on alphabet size (denoted by 'k'): 6k, 4k, 2k, and 1k, with each implementation optimized to ensure performance efficiency. Extensive efforts have been dedicated to refining these implementations, including significant time invested in using various sanitizers to confirm the correctness of the algorithms. Ultimately, while there are specific inputs under which libsais might require additional memory-most of which tend to be synthetic tests designed specifically to challenge the SA-IS algorithm-such instances are relatively rare. In these exceptional cases, the library is designed to allocate only the minimum necessary amount of memory while still delivering the best possible performance.
* The libsais library, initially was developed for constructing suffix arrays, but has broadened its scope to include the calculation of the longest common prefix (LCP) and both the forward and inverse Burrows-Wheeler Transform (BWT) with considerable efforts has been dedicated to refining these algorithms to ensure they deliver maximum performance and maintain the correctness. An illustrative example is the forward BWT, which performance is nearly identical to that of its suffix array construction which is achieved by integrating a modified version of the induced sorting implementation within the final stage of the SA-IS algorithm. Rather than inducing suffix positions at this stage, the library induces the Burrows-Wheeler Transform directly. This approach also supports in-place transformation, maintaining a memory usage of 5n, making it an sutable for data compression applications. Similarly, the inverse BWT is fine-tuned to operate in-place, adhering to the same memory efficiency of 5n with an additional optimization of a bi-gram LF-mapping technique, which allows for the decoding of two symbols simultaneously effectively reduces the number of cache misses during the inversion of the Burrows-Wheeler Transform.
 
# Benchmarks

Full list of benchmarks are moved to own [Benchmarks.md](Benchmarks.md) file.

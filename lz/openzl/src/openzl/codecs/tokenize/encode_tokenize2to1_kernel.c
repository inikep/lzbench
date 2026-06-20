// Copyright (c) Meta Platforms, Inc. and affiliates.

/* Note : using stdc assert() presumes that disabling is controlled by defining
 * NDEBUG this may have to be changed if there is some other form of centralized
 * debug control.
 * Note that this is a transform's kernel,
 * which implies a goal of minimal dependency.
 */
#include <assert.h>
#include <stdint.h>
#include <string.h> // memset

#include "openzl/codecs/tokenize/encode_tokenize2to1_kernel.h"
#include "openzl/shared/bits.h"

static void detectPresents(
        uint8_t present[TOK2_CARDINALITY_MAX],
        const uint16_t* srcSymbols,
        size_t nbSymbols)
{
    assert(present != NULL);
    assert(srcSymbols != NULL);

    memset(present, 0, TOK2_CARDINALITY_MAX);
    for (size_t n = 0; n < nbSymbols; n++) {
        present[srcSymbols[n]] = 1;
    }
}

size_t TOK2_numSort_cardinality(
        uint8_t present[TOK2_CARDINALITY_MAX],
        const uint16_t* srcSymbols,
        size_t nbSymbols)
{
    detectPresents(present, srcSymbols, nbSymbols);

    /* note : if nbSymbols is small,
     * there might be more optimized strategies available.
     */
    size_t cardinality = 0;
    for (size_t n = 0; n < TOK2_CARDINALITY_MAX; n++) {
        cardinality += present[n];
    }

    return cardinality;
}

/* writeAlphabet:
 * take as input an array of 64KB bytes with 0/1 values,
 * signalling if a given 2-bytes symbol is present (1) or not (0).
 * will **overwrite** `present`, assigning an index to any present symbols,
 * starting from 0, and preserving order.
 * @return : alphabetSize
 *
 * CONDITION : this function only works properly if alphabetSize <= 256
 */
static size_t writeAlphabet(
        uint16_t* dstAlphabet,
        size_t alphabetCapacity,
        uint8_t present[TOK2_CARDINALITY_MAX])
{
    (void)alphabetCapacity;
    assert(dstAlphabet != NULL);
    assert(present != NULL);
    size_t index = 0;
    for (int ai = 0; ai < TOK2_CARDINALITY_MAX; ai++) {
        if (present[ai]) {
            present[ai] = (uint8_t)index;
            assert(index < alphabetCapacity);
            dstAlphabet[index] = (uint16_t)ai;
            index++;
        }
    }
    assert(index <= 256);
    return index;
}

/* writeIndexes:
 * convert uint16_t srcSymbols
 * into uint8_t dstIndex
 * following translation map provided by @indexes
 * CONDITIONS : all buffers must exist and be valid
 *              dstIndex capacity must be >= nbSymbols
 * Will write exactly nbSymbols bytes into dstIndex
 */
static void writeIndexes(
        uint8_t* dstIndex,
        const uint16_t* srcSymbols,
        size_t nbSymbols,
        const uint8_t indexes[TOK2_CARDINALITY_MAX])
{
    assert(dstIndex != NULL);
    assert(srcSymbols != NULL);
    assert(indexes != NULL);
    for (size_t n = 0; n < nbSymbols; n++) {
        dstIndex[n] = indexes[srcSymbols[n]];
    }
}

void TOK2_numSort_encode_into1(
        uint8_t* dstIndex,
        size_t indexCapacity,
        uint16_t* dstAlphabet,
        size_t alphabetCapacity,
        const uint16_t* srcSymbols,
        size_t nbSymbols,
        uint8_t* present)
{
    assert(dstAlphabet != NULL);
    size_t const alphabetSize =
            writeAlphabet(dstAlphabet, alphabetCapacity, present);
    assert(alphabetSize <= 256);
    (void)alphabetSize;

    assert(dstIndex != NULL);
    assert(indexCapacity >= nbSymbols);
    (void)indexCapacity;
    writeIndexes(dstIndex, srcSymbols, nbSymbols, present);
}

size_t ZS_tokenize2to1_encode_wksp(
        void* workspace,
        uint8_t* dstIndex,
        size_t indexCapacity,
        uint16_t* dstAlphabet,
        size_t alphabetCapacity,
        const uint16_t* srcSymbols,
        size_t nbSymbols)
{
    (void)indexCapacity;
    assert(workspace != NULL);
    uint8_t* const present = workspace;

    assert(srcSymbols != NULL);
    detectPresents(present, srcSymbols, nbSymbols);

    assert(dstAlphabet != NULL);
    assert(alphabetCapacity >= 256);
    size_t const alphabetSize =
            writeAlphabet(dstAlphabet, alphabetCapacity, present);

    assert(dstIndex != NULL);
    assert(indexCapacity >= nbSymbols);
    writeIndexes(dstIndex, srcSymbols, nbSymbols, present);

    return alphabetSize;
}

size_t ZS_tokenize2to1_encode(
        uint8_t* dstIndex,
        size_t indexCapacity,
        uint16_t* dstAlphabet,
        size_t alphabetCapacity,
        const uint16_t* srcSymbols,
        size_t nbSymbols)
{
    /* note : using stack space to detect present symbols.
     *        it's a big tax on stack but seems preferable to using heap space,
     *        it's preferable to avoid dealing with malloc() inside transforms.
     */
    uint8_t present[TOK2_CARDINALITY_MAX];

    return ZS_tokenize2to1_encode_wksp(
            present,
            dstIndex,
            indexCapacity,
            dstAlphabet,
            alphabetCapacity,
            srcSymbols,
            nbSymbols);
}

/* ===   Fixed-size-fields variant   === */

// Note : may need to separate this part of the kernel
// so that the other part (numeric) remains without dependency.
#include "openzl/shared/mem.h" // ZL_readLE16
size_t TOK2_fsf_cardinality(
        uint8_t present[TOK2_CARDINALITY_MAX],
        const void* src2BSymbols,
        size_t nbSymbols)
{
    assert(src2BSymbols != NULL);

    /* When conditions are right, prefer numeric variant
     * which requires less optimization efforts from the compiler
     * but generates exactly the same result */
    if (MEM_IS_ALIGNED(src2BSymbols, uint16_t) && ZL_isLittleEndian()) {
        return TOK2_numSort_cardinality(present, src2BSymbols, nbSymbols);
    }

    assert(present != NULL);
    memset(present, 0, TOK2_CARDINALITY_MAX);
    for (size_t n = 0; n < nbSymbols; n++) {
        present[ZL_readLE16((const char*)src2BSymbols + 2 * n)] = 1;
    }

    /* note : if nbSymbols is small,
     * there might be more optimized strategies available.
     */
    size_t cardinality = 0;
    for (size_t n = 0; n < TOK2_CARDINALITY_MAX; n++) {
        cardinality += present[n];
    }

    return cardinality;
}

static size_t writeAlphabet_fsf2(
        void* dstAlphabet2B,
        size_t alphabet2BCapacity,
        uint8_t present[TOK2_CARDINALITY_MAX])
{
    (void)alphabet2BCapacity;
    assert(dstAlphabet2B != NULL);
    assert(present != NULL);
    size_t index = 0;
    for (int ai = 0; ai < TOK2_CARDINALITY_MAX; ai++) {
        if (present[ai]) {
            present[ai] = (uint8_t)index;
            assert(index < alphabet2BCapacity);
            ZL_writeLE16((char*)dstAlphabet2B + 2 * index, (uint16_t)ai);
            index++;
        }
    }
    assert(index <= 256);
    return index;
}

static void writeIndexes_fsf2(
        uint8_t* dstIndex,
        const void* src2BSymbols,
        size_t nbSymbols,
        const uint8_t indexes[TOK2_CARDINALITY_MAX])
{
    assert(dstIndex != NULL);
    assert(src2BSymbols != NULL);
    assert(indexes != NULL);
    for (size_t n = 0; n < nbSymbols; n++) {
        dstIndex[n] = indexes[ZL_readLE16((const char*)src2BSymbols + 2 * n)];
    }
}

void TOK2_fsf_encode_into1(
        uint8_t* dstIndex,
        size_t indexCapacity,
        void* dst2BAlphabet,
        size_t alphabetCapacity,
        const void* src2BSymbols,
        size_t nbSymbols,
        uint8_t* present)
{
    assert(dst2BAlphabet != NULL);
    assert(src2BSymbols != NULL);
    /* When conditions are right, prefer numeric variant
     * which requires less optimization efforts from the compiler
     * but generates exactly the same result */
    if (MEM_IS_ALIGNED(dst2BAlphabet, uint16_t)
        && MEM_IS_ALIGNED(src2BSymbols, uint16_t) && ZL_isLittleEndian()) {
        TOK2_numSort_encode_into1(
                dstIndex,
                indexCapacity,
                dst2BAlphabet,
                alphabetCapacity,
                src2BSymbols,
                nbSymbols,
                present);
        return;
    }
    size_t const alphabetSize =
            writeAlphabet_fsf2(dst2BAlphabet, alphabetCapacity, present);
    assert(alphabetSize <= 256);
    (void)alphabetSize;

    assert(dstIndex != NULL);
    assert(indexCapacity >= nbSymbols);
    (void)indexCapacity;
    writeIndexes_fsf2(dstIndex, src2BSymbols, nbSymbols, present);
}

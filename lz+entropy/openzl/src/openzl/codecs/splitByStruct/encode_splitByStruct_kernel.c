// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <assert.h>
#include <string.h> // memcpy

#include "openzl/codecs/splitByStruct/encode_splitByStruct_kernel.h"

// Function dedicated to structures with all member sizes <= 8
static void ZS_dafss_nosmsgt8(
        void* restrict dstBuffers[],
        size_t nbStructMembers,
        const void* src,
        size_t nbStructs,
        const size_t* structMemberSizes);

// General Function for structures with any member size
static void ZS_dafss_anysms(
        void* restrict dstBuffers[],
        size_t nbStructMembers,
        const void* src,
        size_t nbStructs,
        const size_t* structMemberSizes);

/* ZS_dispatchArrayFixedSizeStruct():
 *
 * See interface descrition in *.h.
 *
 * It's a simple though efficient kernel,
 * which loops through the structure definition
 * to copy each member into its own target buffer.
 *
 * It features 2 variants (automatically selected),
 * a generic one, for structures featuring members of any size,
 * which runs at ~1.4 GB/s on my laptop,
 * and a specialized one, when all members have a size <= 8,
 * which is fairly common (ex: sao, vrs),
 * in which case, speed jumps to 5.4 GB/s.
 *
 * A JIT compiler could do better,
 * by taking full advantage of the fixed structure of fixed size
 * to produce a more compact code with less branches,
 * resulting in faster speed (~9 GB/s for sao ingestion on my laptop).
 * But it's a much more complex undertaking.
 *
 * Another strategy not attempted here is inlining,
 * hoping that the structure definition could be determined at compile time
 * and result in an automatic simplification of the loop.
 * At this stage, it's unclear if this is a reasonable expectation,
 * as the rest of the code can act as an obfuscator,
 * making it impossible for the compiler to keep track of structure definition.
 *
 * The kernel relies on cpu's branch predictor to correctly detect the
 * instruction path. This is probably fine up to a certain (?) amount of
 * elements, as the loop becomes bigger with larger structs, thus making branch
 * history more complex. This issue is simplified in the nosmsgt8 variant, which
 * only has the loop branches to care about, thus expanding branch history by a
 * large factor, but could still become a problem for very large structures with
 * many members.
 *
 * Another potential topic of concern for large structures could be the TLB :
 * as the loop writes into many output buffers, each tracked by a pointer,
 * which is fine as long as the nb of buffers doesn't overflow the TLB.
 * Otherwise, it costs a TLB miss every time, increasing latency.
 * The size of the TLB varies with each cpu architecture, so no single limit.
 * On Skylake, it's 64 (16x4) entries, leaving some room for reasonable
 * structures.
 *
 * If TLB thrashing is a concern, a potential strategy could be prefetching,
 * but it's not guaranteed to make a positive impact,
 * as the amount of prefetching per work unit is very high in my opinion.
 *
 * Another strategy is to fill one @dstBuffer at a time,
 * scanning the entire input and extracting target member at each scan.
 * It requires as many scans are structure members,
 * which becomes more costly as this number increases,
 * so it doesn't seem promising for performance.
 * But it can also be multi-threaded.
 * And rather than doing one member at a time,
 * it could extract a "batch" of members per scan.
 *
 * So, a potential advanced strategy for structures with a _lot_ of members
 * could be to cut the job into several "batches",
 * in charge of a subset of members suitable for TLB capacity,
 * and run them in parallel in different threads.
 */

void ZS_dispatchArrayFixedSizeStruct(
        void* restrict dstBuffers[],
        size_t nbStructMembers,
        const void* src,
        size_t srcSize,
        const size_t* structMemberSizes)
{
    // Special case : empty input ==> no dispatch
    // support src==NULL, and empty structs.
    if (srcSize == 0) {
        return;
    }
    assert(src != NULL);

    // Special case: only one member, effectively a copy-paste
    // Dealing with it here free the caller from special-casing this situation
    // on its own side
    if (nbStructMembers == 1) {
        memcpy(dstBuffers[0], src, srcSize);
        return;
    }

    int smsgt8        = 0;
    size_t structSize = 0;

    for (size_t n = 0; n < nbStructMembers; n++) {
        smsgt8 |= (structMemberSizes[n] > 8);
        structSize += structMemberSizes[n];
    }
    // Since srcSize > 0, only non-empty struct are valid at this stage
    assert(structSize > 0);

    size_t const nbStructs = srcSize / structSize;
    assert(nbStructs * structSize == srcSize); /* exact multiple */

    if (!smsgt8) {
        ZS_dafss_nosmsgt8(
                dstBuffers, nbStructMembers, src, nbStructs, structMemberSizes);
        return;
    }

    ZS_dafss_anysms(
            dstBuffers, nbStructMembers, src, nbStructs, structMemberSizes);
}

static void ZS_dafss_nosmsgt8(
        void* restrict dstBuffers[],
        size_t nbStructMembers,
        const void* src,
        size_t nbStructs,
        const size_t* structMemberSizes)
{
    size_t smallestSms = (size_t)(-1);

    for (size_t n = 0; n < nbStructMembers; n++) {
        if (structMemberSizes[n] < smallestSms) {
            smallestSms = structMemberSizes[n];
        }
    }

    // Only non-empty structs are allowed at this stage
    // All fixed-size members are presumed to have a size > 0
    assert(smallestSms > 0);
    // Condition to enter this function : all members must have a size <= 8
    assert(smallestSms <= 8);

    size_t const nbSafeRounds_precalc[8] = { 8, 4, 3, 2, 2, 2, 2, 1 };
    size_t const nbSafeRounds = nbSafeRounds_precalc[smallestSms - 1];

    if (nbStructs > nbSafeRounds) {
        for (size_t n = 0; n < (nbStructs - nbSafeRounds); n++) {
            for (size_t bufid = 0; bufid < nbStructMembers; bufid++) {
                size_t const sms = structMemberSizes[bufid];
                memcpy(dstBuffers[bufid], src, 8);
                dstBuffers[bufid] = (char*)(dstBuffers[bufid]) + sms;
                src               = (const char*)src + sms;
            }
        }
    }

    /* last rounds use exact field size, to control buffer overflow */
    size_t const nbLastRounds =
            (nbStructs < nbSafeRounds) ? nbStructs : nbSafeRounds;
    ZS_dafss_anysms(
            dstBuffers, nbStructMembers, src, nbLastRounds, structMemberSizes);
}

static void ZS_dafss_anysms(
        void* restrict dstBuffers[],
        size_t nbStructMembers,
        const void* src,
        size_t nbStructs,
        const size_t* structMemberSizes)
{
    for (size_t n = 0; n < nbStructs; n++) {
        for (size_t bufid = 0; bufid < nbStructMembers; bufid++) {
            size_t const sms = structMemberSizes[bufid];
            memcpy(dstBuffers[bufid], src, sms);
            dstBuffers[bufid] = (char*)(dstBuffers[bufid]) + sms;
            src               = (const char*)src + sms;
        }
    }
}

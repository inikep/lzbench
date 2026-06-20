// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "benchmark/unitBench/scenarios/misc/sao.h"

// Bulk ingestion of arrays of fixed size structures
#include "openzl/codecs/splitByStruct/encode_splitByStruct_kernel.h"
size_t saoIngest_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    // Pretends dispatching an input as if it was a sao file
    (void)customPayload;

    size_t const structSize           = 28;
    size_t const structMemberSizes[6] = { 8, 8, 2, 2, 4, 4 };
    size_t const nbStructs            = srcSize / structSize;
    void* dstBuffers[6]               = {
        dst,
        (char*)dst + nbStructs * 8,
        (char*)dst + nbStructs * 16,
        (char*)dst + nbStructs * 18,
        (char*)dst + nbStructs * 20,
        (char*)dst + nbStructs * 24,
    };
    assert(dstCapacity >= srcSize);
    (void)dstCapacity;

    ZS_dispatchArrayFixedSizeStruct(
            dstBuffers, 6, src, srcSize, structMemberSizes);

    return srcSize;
}

size_t saoIngestCompiled_wrapper(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload)
{
    (void)customPayload;
    // Pretends dispatching an input as if it was a sao file

    size_t const structSize           = 28;
    size_t const structMemberSizes[6] = { 8, 8, 2, 2, 4, 4 };
    size_t const nbStructs            = srcSize / structSize;
    void* dstBuffers[6]               = {
        dst,
        (char*)dst + nbStructs * 8,
        (char*)dst + nbStructs * 16,
        (char*)dst + nbStructs * 18,
        (char*)dst + nbStructs * 20,
        (char*)dst + nbStructs * 24,
    };
    assert(dstCapacity >= srcSize);
    (void)dstCapacity;

    for (size_t n = 0; n < nbStructs; n++) {
        {
            size_t const sms = structMemberSizes[0];
            memcpy(dstBuffers[0], src, sms);
            dstBuffers[0] = (char*)(dstBuffers[0]) + sms;
            src           = (const char*)src + sms;
        }
        {
            size_t const sms = structMemberSizes[1];
            memcpy(dstBuffers[1], src, sms);
            dstBuffers[1] = (char*)(dstBuffers[1]) + sms;
            src           = (const char*)src + sms;
        }
        {
            size_t const sms = structMemberSizes[2];
            memcpy(dstBuffers[2], src, sms);
            dstBuffers[2] = (char*)(dstBuffers[2]) + sms;
            src           = (const char*)src + sms;
        }
        {
            size_t const sms = structMemberSizes[3];
            memcpy(dstBuffers[3], src, sms);
            dstBuffers[3] = (char*)(dstBuffers[3]) + sms;
            src           = (const char*)src + sms;
        }
        {
            size_t const sms = structMemberSizes[4];
            memcpy(dstBuffers[4], src, sms);
            dstBuffers[4] = (char*)(dstBuffers[4]) + sms;
            src           = (const char*)src + sms;
        }
        {
            size_t const sms = structMemberSizes[5];
            memcpy(dstBuffers[5], src, sms);
            dstBuffers[5] = (char*)(dstBuffers[5]) + sms;
            src           = (const char*)src + sms;
        }
    }

    return srcSize;
}

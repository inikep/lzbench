// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_BENCHMARK_UNITBENCH_BENCH_ENTRY_H
#define ZSTRONG_BENCHMARK_UNITBENCH_BENCH_ENTRY_H

#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_decompress.h"

// prototype for the function to benchmark
typedef size_t (*BMK_benchFn_t)(
        const void* src,
        size_t srcSize,
        void* dst,
        size_t dstCapacity,
        void* customPayload);

/* ==================================================
 * Payload
 * ================================================== */
typedef struct {
    const char* name;
    ZL_GraphFn graphF;
    ZL_CCtx* cctx;
    ZL_Compressor* cgraph;
    ZL_DCtx* dctx;
    int intParam;
} BenchPayload;
// Preparation function
// Can modify input, so that it corresponds to scenario's expectation
typedef size_t (
        *BMK_prepFn_t)(void* src, size_t srcSize, const BenchPayload* bp);

typedef size_t (*BMK_initFn_t)(void* initPayload);

typedef size_t (*BMK_outSize_f)(const void* src, size_t srcSize);

/* BMK_runTime_t: valid result return type */

typedef struct {
    double nanoSecPerRun; /* time per iteration (over all blocks) */
    size_t sumOfReturn;   /* sum of return values */
} BMK_runTime_t;
typedef void (*BMK_display_f)(
        const char* srcname,
        const char* fname,
        BMK_runTime_t rt,
        size_t srcSize);

/**
 * Each scenario is described within a single structure, defined here.
 * Many of its fields are optional.
 * The structure is declared in-place within the @p scenarioList array
 */
typedef struct {
    /// Required: name of the scenario
    const char* name;

    /// Required (for custom scenarios only): the function to benchmark
    BMK_benchFn_t func;

    /// Required (for standard scenarios only): Graph creation function,
    /// look into zs2_compressor.h for its signature.
    /// Setting .graphF to != NULL will imply .func,
    /// and trigger a round-trip scenario.
    /// Either .graphF or .func must be != NULL for the
    /// scenario to be valid!
    ZL_GraphFn graphF;

    /// Optional: modify input buffer for benchmark.
    /// This is uncommon;
    /// it may be needed to massage or verify input
    /// so that it corresponds to the scenario's expectations
    BMK_prepFn_t prep;

    /// Optional: this function is run only once, at the beginning of the
    /// benchmark
    BMK_initFn_t init;

    /// Optional: tells how much memory must be
    /// allocated for dstCapacity (the output of .func). If left blank,
    /// unitBench will use ZL_compressBound() by default.
    BMK_outSize_f outSize;

    /// Optional: custom result display function
    BMK_display_f display;
} Bench_Entry;

#endif // ZSTRONG_BENCHMARK_UNITBENCH_BENCH_ENTRY_H

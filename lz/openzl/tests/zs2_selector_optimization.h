// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_ZS_SELECTOR_OPTIMIZATION_H
#define ZSTRONG_ZS_SELECTOR_OPTIMIZATION_H

#include "openzl/zl_compressor.h"
#include "openzl/zl_selector.h"

#if defined(__cplusplus)
extern "C" {
#endif

// State

void ZS2_SelectorOpt_setEnabled(int enabled);

int ZS2_SelectorOpt_isEnabled(void);

typedef struct ZS2_SelectorOptState_s ZS2_SelectorOptState;
struct ZS2_SelectorOptState_s {
    ZL_GraphID* possibleGraphs;
    size_t nbPossibleGraphs;
    size_t idx;
    ZL_GraphID selected;
    int done;
};

ZS2_SelectorOptState ZS2_SelectorOptState_init(void);

void ZS2_SelectorOptState_next(
        ZS2_SelectorOptState* state,
        const ZL_GraphID* possibleGraphs,
        size_t nbPossibleGraphs);

void ZS2_SelectorOptState_destroy(ZS2_SelectorOptState* state);

// Shim

ZL_GraphID ZS2_selector_opt_shim_generic(
        ZS2_SelectorOptState* state,
        ZL_SerialSelectorFn selector,
        const void* src,
        size_t srcSize,
        const ZL_GraphID* customGraphs,
        size_t nbCustomGraphs);

// Results

typedef struct {
    ZL_GraphID graphid;
    size_t srcSize;
    size_t size;
    double durationNs;
} ZS2_SelectorOptResult;

typedef struct {
    ZS2_SelectorOptResult* results;
    size_t nbResults;
} ZS2_SelectorOptResults;

ZS2_SelectorOptResults ZS2_SelectorOptResults_init(void);

void ZS2_SelectorOptResults_addResult(
        ZS2_SelectorOptResults* results,
        ZS2_SelectorOptResult result);

void ZS2_SelectorOptResults_print(const ZS2_SelectorOptResults* results);

size_t ZS2_SelectorOptResults_lastSize(const ZS2_SelectorOptResults* results);

void ZS2_SelectorOptResults_destroy(ZS2_SelectorOptResults* results);

// Results Aggregation

typedef struct {
    size_t in_size_sum;  // sum of input sizes (should be == to other choices)
    size_t out_size_sum; // sum of output sizes using this choice

    // The total number of bytes saved by using this graph when it's best over
    // the next best. I.e., the value of having this choice assuming perfect
    // selection.
    size_t improvement_sum;

    size_t avail_count; // how many times was this available (should be always?)
    size_t best_count;  // how many times was this best (or tied for best)
    size_t best_exc_count; // how many times was this uniquely best
    size_t sel_count;      // how many times was this selected
    size_t selbest_count;  // how many times was this best & selected

    size_t avail_size;    // for how many bytes was this available (should be
                          // always?)
    size_t best_size;     // for how many bytes was this best (or tied for best)
    size_t best_exc_size; // for how many bytes was this uniquely best
    size_t sel_size;      // for how many bytes was this selected
    size_t selbest_size;  // for how many bytes was this best & selected
} ZS2_SelectorOptAggrChoiceResult;

typedef struct {
    // Aggregations for if the selector had always chosen a particular graph.
    ZS2_SelectorOptAggrChoiceResult* graph_results;
    size_t nb_graphs;

    // A synthetic aggregation for if the selector had always chosen the best
    // graph.
    ZS2_SelectorOptAggrChoiceResult best_result;

    // An aggregation based on the selections actually made by the selector.
    ZS2_SelectorOptAggrChoiceResult selected_result;
} ZS2_SelectorOptAggrResults;

ZS2_SelectorOptAggrChoiceResult ZS2_SelectorOptAggrChoiceResult_init(void);

ZS2_SelectorOptAggrResults ZS2_SelectorOptAggrResults_init(void);

ZS2_SelectorOptAggrChoiceResult* ZS2_SelectorOptAggrResults_getChoiceResult(
        ZS2_SelectorOptAggrResults* aggr,
        ZL_GraphID graphid);

void ZS2_SelectorOptAggrResults_addResult(
        ZS2_SelectorOptAggrResults* aggr,
        const ZS2_SelectorOptResults* result);

void ZS2_SelectorOptAggrResults_print(const ZS2_SelectorOptAggrResults* aggr);

void ZS2_SelectorOptAggrResults_destroy(ZS2_SelectorOptAggrResults* aggr);

// Runner

ZS2_SelectorOptResults ZS2_selector_opt_run(
        ZS2_SelectorOptState* state,
        void* dst,
        size_t dstCapacity,
        void const* src,
        size_t srcSize,
        ZL_GraphFn graph);

ZS2_SelectorOptResults ZS2_selector_opt_run_cgraph(
        ZS2_SelectorOptState* state,
        void* dst,
        size_t dstCapacity,
        void const* src,
        size_t srcSize,
        ZL_Compressor const* cgraph);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // ZSTRONG_ZS_SELECTOR_OPTIMIZATION_H

// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "tests/zs2_selector_optimization.h"

#include <stdlib.h>
#include <string.h>

#include "openzl/common/assertion.h"

#include "tools/time/timefn.h"

int ZS2_SelectorOpt_enabled = 1;

void ZS2_SelectorOpt_setEnabled(int enabled)
{
    ZS2_SelectorOpt_enabled = !!enabled;
}

int ZS2_SelectorOpt_isEnabled(void)
{
    return ZS2_SelectorOpt_enabled;
}

ZL_GraphID ZS2_selector_opt_shim_generic(
        ZS2_SelectorOptState* state,
        ZL_SerialSelectorFn selector,
        const void* src,
        size_t srcSize,
        const ZL_GraphID* customGraphs,
        size_t nbCustomGraphs)
{
    if (ZS2_SelectorOpt_isEnabled()) {
        ZS2_SelectorOptState_next(state, customGraphs, nbCustomGraphs);
        if (state->idx < state->nbPossibleGraphs) {
            state->selected = state->possibleGraphs[state->idx];
        } else {
            state->done = 1;
            state->selected =
                    selector(src, srcSize, customGraphs, nbCustomGraphs);
        }
        return state->selected;
    } else {
        return selector(src, srcSize, customGraphs, nbCustomGraphs);
    }
}

ZS2_SelectorOptState ZS2_SelectorOptState_init(void)
{
    ZS2_SelectorOptState state;
    state.possibleGraphs   = NULL;
    state.nbPossibleGraphs = 0;
    state.idx              = (size_t)-1;
    state.done             = 0;
    return state;
}

void ZS2_SelectorOptState_next(
        ZS2_SelectorOptState* state,
        const ZL_GraphID* possibleGraphs,
        size_t nbPossibleGraphs)
{
    if (state->possibleGraphs == NULL) {
        state->possibleGraphs   = malloc(nbPossibleGraphs * sizeof(ZL_GraphID));
        state->nbPossibleGraphs = nbPossibleGraphs;
        ZL_REQUIRE_NN(state->possibleGraphs);
        memcpy(state->possibleGraphs,
               possibleGraphs,
               nbPossibleGraphs * sizeof(ZL_GraphID));
    } else {
        ZL_REQUIRE_EQ(state->nbPossibleGraphs, nbPossibleGraphs);
        for (size_t i = 0; i < nbPossibleGraphs; i++) {
            ZL_REQUIRE_EQ(state->possibleGraphs[i].gid, possibleGraphs[i].gid);
        }
    }
    state->idx++;
    ZL_LOG(V,
           "Selector Optimization Iter %lu of %lu",
           state->idx,
           state->nbPossibleGraphs + 1);
}

void ZS2_SelectorOptState_destroy(ZS2_SelectorOptState* state)
{
    ZL_REQUIRE_NN(state);
    free(state->possibleGraphs);
}

ZS2_SelectorOptResults ZS2_SelectorOptResults_init(void)
{
    ZS2_SelectorOptResults results;
    results.results   = NULL;
    results.nbResults = 0;
    return results;
}

void ZS2_SelectorOptResults_addResult(
        ZS2_SelectorOptResults* results,
        ZS2_SelectorOptResult result)
{
    results->results =
            realloc(results->results,
                    sizeof(ZS2_SelectorOptResult) * ++(results->nbResults));
    ZL_ASSERT_NN(results->results);
    results->results[results->nbResults - 1] = result;
}

void ZS2_SelectorOptResults_print(const ZS2_SelectorOptResults* results)
{
    ZL_LOG(ALWAYS, "Selector Benchmark Results:");
    ZL_RLOG(ALWAYS, "Choice   : Sel?Best?Graph Size\n");
    const ZS2_SelectorOptResult* selected =
            &results->results[results->nbResults - 1];
    size_t bestSize = results->results[0].size;
    for (size_t i = 1; i < results->nbResults - 1; i++) {
        const ZS2_SelectorOptResult* result = &results->results[i];
        if (result->size < bestSize) {
            bestSize = result->size;
        }
    }

    for (size_t i = 0; i < results->nbResults - 1; i++) {
        const ZS2_SelectorOptResult* result = &results->results[i];
        ZL_RLOG(ALWAYS,
                "Choice %2lu: %3s %4s %5u %7lu\n",
                i + 1,
                selected->graphid.gid == result->graphid.gid ? "Sel" : "",
                bestSize == result->size ? "Best" : "",
                result->graphid.gid,
                result->size);
    }
}

void ZS2_SelectorOptResults_destroy(ZS2_SelectorOptResults* results)
{
    free(results->results);
    results->results   = NULL;
    results->nbResults = 0;
}

size_t ZS2_SelectorOptResults_lastSize(const ZS2_SelectorOptResults* results)
{
    ZL_ASSERT_GE(results->nbResults, 1);
    return results->results[results->nbResults - 1].size;
}

ZS2_SelectorOptAggrChoiceResult ZS2_SelectorOptAggrChoiceResult_init(void)
{
    return (ZS2_SelectorOptAggrChoiceResult){
        .in_size_sum     = 0,
        .out_size_sum    = 0,
        .improvement_sum = 0,
        .avail_count     = 0,
        .best_count      = 0,
        .sel_count       = 0,
        .selbest_count   = 0,
    };
}

ZS2_SelectorOptAggrResults ZS2_SelectorOptAggrResults_init(void)
{
    ZS2_SelectorOptAggrResults results;
    results.graph_results   = NULL;
    results.nb_graphs       = 0;
    results.best_result     = ZS2_SelectorOptAggrChoiceResult_init();
    results.selected_result = ZS2_SelectorOptAggrChoiceResult_init();
    return results;
}

ZS2_SelectorOptAggrChoiceResult* ZS2_SelectorOptAggrResults_getChoiceResult(
        ZS2_SelectorOptAggrResults* aggr,
        ZL_GraphID graphid)
{
    if (aggr->nb_graphs <= graphid.gid) {
        aggr->graph_results = realloc(
                aggr->graph_results,
                sizeof(ZS2_SelectorOptAggrChoiceResult) * (graphid.gid + 1));
        for (size_t i = aggr->nb_graphs; i <= graphid.gid; i++) {
            aggr->graph_results[i] = ZS2_SelectorOptAggrChoiceResult_init();
        }
        aggr->nb_graphs = graphid.gid + 1;
    }
    return &aggr->graph_results[graphid.gid];
}

void ZS2_SelectorOptAggrResults_addResult(
        ZS2_SelectorOptAggrResults* aggr,
        const ZS2_SelectorOptResults* result)
{
    const ZS2_SelectorOptResult* selected =
            &result->results[result->nbResults - 1];
    size_t bestSize = result->results[0].size;
    size_t nbBest   = 1;
    for (size_t i = 1; i < result->nbResults - 1; i++) {
        const ZS2_SelectorOptResult* graph_result = &result->results[i];
        if (graph_result->size < bestSize) {
            bestSize = graph_result->size;
            nbBest   = 1;
        } else if (graph_result->size == bestSize) {
            nbBest++;
        }
    }

    const size_t srcSize = selected->srcSize;
    for (size_t i = 0; i < result->nbResults - 1; i++) {
        const ZS2_SelectorOptResult* graph_result = &result->results[i];
        ZL_ASSERT_EQ(srcSize, graph_result->srcSize);
    }

    aggr->best_result.in_size_sum += srcSize;
    aggr->best_result.out_size_sum += bestSize;
    aggr->best_result.avail_count++;
    aggr->best_result.best_count++;
    aggr->best_result.best_exc_count += (nbBest == 1);
    aggr->best_result.avail_size += srcSize;
    aggr->best_result.best_size += srcSize;
    aggr->best_result.best_exc_size += (nbBest == 1) ? srcSize : 0;

    aggr->selected_result.in_size_sum += srcSize;
    aggr->selected_result.out_size_sum += selected->size;
    aggr->selected_result.avail_count++;
    aggr->selected_result.sel_count++;
    aggr->selected_result.avail_size += srcSize;
    aggr->selected_result.sel_size += srcSize;

    if (selected->size == bestSize) {
        aggr->selected_result.best_count++;
        aggr->selected_result.best_exc_count += (nbBest == 1);
        aggr->selected_result.selbest_count++;
        aggr->selected_result.best_size += srcSize;
        aggr->selected_result.best_exc_size += (nbBest == 1) ? srcSize : 0;
        aggr->selected_result.selbest_size += srcSize;
        aggr->best_result.sel_count++;
        aggr->best_result.selbest_count++;
        aggr->best_result.sel_size += srcSize;
        aggr->best_result.selbest_size += srcSize;
    }

    for (size_t i = 0; i < result->nbResults - 1; i++) {
        const ZS2_SelectorOptResult* graph_result = &result->results[i];
        ZS2_SelectorOptAggrChoiceResult* graph_aggr =
                ZS2_SelectorOptAggrResults_getChoiceResult(
                        aggr, graph_result->graphid);

        graph_aggr->in_size_sum += graph_result->srcSize;
        graph_aggr->out_size_sum += graph_result->size;

        graph_aggr->avail_count++;
        graph_aggr->avail_size += srcSize;

        if (selected->graphid.gid == graph_result->graphid.gid) {
            graph_aggr->sel_count++;
            graph_aggr->sel_size += srcSize;
        }

        if (graph_result->size == bestSize) {
            graph_aggr->best_count++;
            graph_aggr->best_exc_count += (nbBest == 1);
            graph_aggr->best_size += srcSize;
            graph_aggr->best_exc_size += (nbBest == 1) ? srcSize : 0;

            if (selected->graphid.gid == graph_result->graphid.gid) {
                graph_aggr->selbest_count++;
                graph_aggr->selbest_size += srcSize;
            }

            size_t secondBestSize = (size_t)-1;
            for (size_t j = 0; j < result->nbResults - 1; j++) {
                const ZS2_SelectorOptResult* second_result =
                        &result->results[j];
                if (second_result->graphid.gid != graph_result->graphid.gid) {
                    if (second_result->size < secondBestSize) {
                        secondBestSize = second_result->size;
                    }
                }
            }

            graph_aggr->improvement_sum += secondBestSize - bestSize;
        }
    }
}

void ZS2_SelectorOptAggrResults_print(const ZS2_SelectorOptAggrResults* aggr)
{
    (void)aggr;
    ZL_LOG(ALWAYS, "Selector Benchmark Aggregate Results:");

    if (!ZS2_SelectorOpt_isEnabled()) {
        ZL_LOG(ALWAYS,
               "Note: Selector Optimization is Disabled. Limited results available.");
    }

    ZL_RLOG(ALWAYS,
            "%5s %11s -> %11s | "
            // "%5s %11s %7s | "
            "%5s %11s %7s %7s | "
            "%5s %11s %7s | "
            "%5s %11s %7s | "
            "%9s\n",
            "Graph",
            "Input B",
            "Output B",
            // "Avail",
            // "",
            // "",
            "#Best",
            "Best B",
            "Best %",
            "Exc %",
            "#Sel",
            "Sel B",
            "Sel %",
            "#S&B",
            "S&B B",
            "S&B %",
            "Value B");

    for (size_t gid = 0; gid < aggr->nb_graphs; gid++) {
        const ZS2_SelectorOptAggrChoiceResult* graph_aggr =
                &aggr->graph_results[gid];
        if (graph_aggr->avail_count == 0) {
            continue;
        }

        ZL_RLOG(ALWAYS,
                "%-4lu: %11lu -> %11lu | "
                // "%5lu %11lu %6.2lf%% | "
                "%5lu %11lu %6.2lf%% %6.2lf%% | "
                "%5lu %11lu %6.2lf%% | "
                "%5lu %11lu %6.2lf%% | "
                "%9lu\n",
                gid,
                graph_aggr->in_size_sum,
                graph_aggr->out_size_sum,
                // graph_aggr->avail_count,
                // graph_aggr->avail_size,
                // 100 * ((double)graph_aggr->avail_size) /
                // (double)graph_aggr->in_size_sum,
                graph_aggr->best_count,
                graph_aggr->best_size,
                100 * ((double)graph_aggr->best_size)
                        / (double)graph_aggr->in_size_sum,
                100 * ((double)graph_aggr->best_exc_size)
                        / (double)graph_aggr->in_size_sum,
                graph_aggr->sel_count,
                graph_aggr->sel_size,
                100 * ((double)graph_aggr->sel_size)
                        / (double)graph_aggr->in_size_sum,
                graph_aggr->selbest_count,
                graph_aggr->selbest_size,
                100 * ((double)graph_aggr->selbest_size)
                        / (double)graph_aggr->in_size_sum,
                graph_aggr->improvement_sum);
    }

    ZL_RLOG(ALWAYS,
            "%-4s: %11lu -> %11lu | "
            // "%5lu %11lu %6.2lf%% | "
            "%5lu %11lu %6.2lf%% %6.2lf%% | "
            "%5lu %11lu %6.2lf%% | "
            "%5lu %11lu %6.2lf%% |\n",
            "Best",
            aggr->best_result.in_size_sum,
            aggr->best_result.out_size_sum,
            // aggr->best_result.avail_count,
            // aggr->best_result.avail_size,
            // 100 * ((double)aggr->best_result.avail_size) /
            // (double)aggr->best_result.in_size_sum,
            aggr->best_result.best_count,
            aggr->best_result.best_size,
            100 * ((double)aggr->best_result.best_size)
                    / (double)aggr->best_result.in_size_sum,
            100 * ((double)aggr->best_result.best_exc_size)
                    / (double)aggr->best_result.in_size_sum,
            aggr->best_result.sel_count,
            aggr->best_result.sel_size,
            100 * ((double)aggr->best_result.sel_size)
                    / (double)aggr->best_result.in_size_sum,
            aggr->best_result.selbest_count,
            aggr->best_result.selbest_size,
            100 * ((double)aggr->best_result.selbest_size)
                    / (double)aggr->best_result.in_size_sum);

    ZL_RLOG(ALWAYS,
            "%-4s: %11lu -> %11lu | "
            // "%5lu %11lu %6.2lf%% | "
            "%5lu %11lu %6.2lf%% %6.2lf%% | "
            "%5lu %11lu %6.2lf%% | "
            "%5lu %11lu %6.2lf%% |\n",
            "Sel",
            aggr->selected_result.in_size_sum,
            aggr->selected_result.out_size_sum,
            // aggr->selected_result.avail_count,
            // aggr->selected_result.avail_size,
            // 100 * ((double)aggr->selected_result.avail_size) /
            // (double)aggr->selected_result.in_size_sum,
            aggr->selected_result.best_count,
            aggr->selected_result.best_size,
            100 * ((double)aggr->selected_result.best_size)
                    / (double)aggr->selected_result.in_size_sum,
            100 * ((double)aggr->selected_result.best_exc_size)
                    / (double)aggr->selected_result.in_size_sum,
            aggr->selected_result.sel_count,
            aggr->selected_result.sel_size,
            100 * ((double)aggr->selected_result.sel_size)
                    / (double)aggr->selected_result.in_size_sum,
            aggr->selected_result.selbest_count,
            aggr->selected_result.selbest_size,
            100 * ((double)aggr->selected_result.selbest_size)
                    / (double)aggr->selected_result.in_size_sum);

    ZL_RLOG(ALWAYS,
            "Improvement of %lu B (%6.2lf%%) is possible through better selection.\n",
            aggr->selected_result.out_size_sum - aggr->best_result.out_size_sum,
            100
                    * (double)(aggr->selected_result.out_size_sum
                               - aggr->best_result.out_size_sum)
                    / (double)aggr->selected_result.out_size_sum);
}

void ZS2_SelectorOptAggrResults_destroy(ZS2_SelectorOptAggrResults* aggr)
{
    free(aggr->graph_results);
    aggr->graph_results = NULL;
    aggr->nb_graphs     = 0;
}

ZS2_SelectorOptResults ZS2_selector_opt_run(
        ZS2_SelectorOptState* state,
        void* dst,
        size_t dstCapacity,
        void const* src,
        size_t srcSize,
        ZL_GraphFn graph)
{
    ZL_Compressor* cgraph = ZL_Compressor_create();
    ZL_REQUIRE_NN(cgraph);
    ZL_GraphID const startingNode = graph(cgraph);
    ZL_REQUIRE(!ZL_isError(
            ZL_Compressor_selectStartingGraphID(cgraph, startingNode)));

    ZS2_SelectorOptResults const results = ZS2_selector_opt_run_cgraph(
            state, dst, dstCapacity, src, srcSize, cgraph);
    ZL_Compressor_free(cgraph);

    return results;
}

ZS2_SelectorOptResults ZS2_selector_opt_run_cgraph(
        ZS2_SelectorOptState* state,
        void* dst,
        size_t dstCapacity,
        void const* src,
        size_t srcSize,
        ZL_Compressor const* cgraph)
{
    *state = ZS2_SelectorOptState_init();

    ZS2_SelectorOptResults results = ZS2_SelectorOptResults_init();

    while (!state->done) {
        TIME_t start      = TIME_getTime();
        ZL_Report const r = ZL_compress_usingCompressor(
                dst, dstCapacity, src, srcSize, cgraph);
        double duration = (double)TIME_clockSpan_ns(start);
        ZS2_SelectorOptResults_addResult(
                &results,
                (ZS2_SelectorOptResult){ .graphid    = state->selected,
                                         .srcSize    = srcSize,
                                         .durationNs = duration,
                                         .size       = ZL_validResult(r) });
    }

    ZS2_SelectorOptState_destroy(state);

    return results;
}

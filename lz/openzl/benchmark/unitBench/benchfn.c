// Copyright (c) Meta Platforms, Inc. and affiliates.

/* *************************************
 *  Includes
 ***************************************/
#include <assert.h> /* assert */
#include <stdlib.h> /* malloc, free */
#include <string.h> /* memset */

#include "benchmark/unitBench/benchfn.h"
#include "tools/time/timefn.h"

/* *************************************
 *  Constants
 ***************************************/
#define TIMELOOP_MICROSEC SEC_TO_MICRO       /* 1 second */
#define TIMELOOP_NANOSEC (1 * 1000000000ULL) /* 1 second */

#define KB *(1 << 10)
#define MB *(1 << 20)
#define GB *(1U << 30)

/* *************************************
 *  Debug errors
 ***************************************/
#if defined(DEBUG) && (DEBUG >= 1)
#    include <stdio.h> /* fprintf */
#    define DISPLAY(...) fprintf(stderr, __VA_ARGS__)
#    define DEBUGOUTPUT(...)          \
        {                             \
            if (DEBUG)                \
                DISPLAY(__VA_ARGS__); \
        }
#else
#    define DEBUGOUTPUT(...)
#endif

/* error without displaying */
#define RETURN_QUIET_ERROR(retValue, ...)              \
    {                                                  \
        DEBUGOUTPUT("%s: %i: \n", __FILE__, __LINE__); \
        DEBUGOUTPUT("Error : ");                       \
        DEBUGOUTPUT(__VA_ARGS__);                      \
        DEBUGOUTPUT(" \n");                            \
        return retValue;                               \
    }

/* Abort execution if a condition is not met */
#define CONTROL(c)                           \
    {                                        \
        if (!(c)) {                          \
            DEBUGOUTPUT("error: %s \n", #c); \
            abort();                         \
        }                                    \
    }

/* *************************************
 *  Benchmarking an arbitrary function
 ***************************************/

int BMK_isSuccessful_runOutcome(BMK_runOutcome_t outcome)
{
    return outcome.error_tag_never_ever_use_directly == 0;
}

/* warning : this function will stop program execution if outcome is invalid !
 *           check outcome validity first, using BMK_isValid_runResult() */
BMK_runTime_t BMK_extract_runTime(BMK_runOutcome_t outcome)
{
    CONTROL(outcome.error_tag_never_ever_use_directly == 0);
    return outcome.internal_never_ever_use_directly;
}

size_t BMK_extract_errorResult(BMK_runOutcome_t outcome)
{
    CONTROL(outcome.error_tag_never_ever_use_directly != 0);
    return outcome.error_result_never_ever_use_directly;
}

static BMK_runOutcome_t BMK_runOutcome_error(size_t errorResult)
{
    BMK_runOutcome_t b;
    memset(&b, 0, sizeof(b));
    b.error_tag_never_ever_use_directly    = 1;
    b.error_result_never_ever_use_directly = errorResult;
    return b;
}

static BMK_runOutcome_t BMK_setValid_runTime(BMK_runTime_t runTime)
{
    BMK_runOutcome_t outcome;
    outcome.error_tag_never_ever_use_directly = 0;
    outcome.internal_never_ever_use_directly  = runTime;
    return outcome;
}

/* initFn will be measured once, benchFn will be measured `nbLoops` times */
/* initFn is optional, provide NULL if none */
/* benchFn must return a size_t value that errorFn can interpret */
/* takes # of blocks and list of size & stuff for each. */
/* can report result of benchFn for each block into blockResult. */
/* blockResult is optional, provide NULL if this information is not required */
/* note : time per loop can be reported as zero if run time < timer resolution
 */
BMK_runOutcome_t BMK_benchFunction(BMK_benchParams_t p, unsigned nbLoops)
{
    /* init */
    for (size_t i = 0; i < p.blockCount; i++) {
        memset(p.dstBuffers[i],
               0xE5,
               p.dstCapacities[i]); /* warm up and erase result buffer */
    }

    /* benchmark */
    size_t dstSize = 0;
    nbLoops += !nbLoops; /* minimum nbLoops is 1 */
    TIME_waitForNextTick();
    TIME_t clockStart = TIME_getTime();
    if (p.initFn != NULL)
        p.initFn(p.initPayload);
    for (unsigned loopNb = 0; loopNb < nbLoops; loopNb++) {
        for (unsigned blockNb = 0; blockNb < p.blockCount; blockNb++) {
            size_t const res = p.benchFn(
                    p.srcBuffers[blockNb],
                    p.srcSizes[blockNb],
                    p.dstBuffers[blockNb],
                    p.dstCapacities[blockNb],
                    p.benchPayload);
            if (loopNb == 0) {
                if (p.blockResults != NULL)
                    p.blockResults[blockNb] = res;
                if ((p.errorFn != NULL) && (p.errorFn(res))) {
                    RETURN_QUIET_ERROR(
                            BMK_runOutcome_error(res),
                            "Function benchmark failed on block %u (of size %zu) with error %zu",
                            blockNb,
                            p.srcSizes[blockNb],
                            res);
                }
                dstSize += res;
            }
        }
    } /* for (loopNb = 0; loopNb < nbLoops; loopNb++) */

    Duration_ns const totalTime = TIME_clockSpan_ns(clockStart);
    BMK_runTime_t const rt      = {
             .nanoSecPerRun = (double)totalTime / nbLoops,
             .sumOfReturn   = dstSize,
    };
    return BMK_setValid_runTime(rt);
}

/* ====  Benchmarking any function, providing intermediate results  ==== */

struct BMK_timedFnState_s {
    Duration_ns timeSpent;
    Duration_ns timeBudget;
    Duration_ns runBudget;
    BMK_runTime_t fastestRun;
    unsigned nbLoops;
    TIME_t coolTime;
}; /* typedef'd to BMK_timedFnState_t within bench.h */

BMK_timedFnState_t* BMK_createTimedFnState(unsigned total_ms, unsigned run_ms)
{
    BMK_timedFnState_t* const r = (BMK_timedFnState_t*)malloc(sizeof(*r));
    if (r == NULL)
        return NULL; /* malloc() error */
    BMK_resetTimedFnState(r, total_ms, run_ms);
    return r;
}

void BMK_freeTimedFnState(BMK_timedFnState_t* state)
{
    free(state);
}

BMK_timedFnState_t* BMK_initStatic_timedFnState(
        void* buffer,
        size_t size,
        unsigned total_ms,
        unsigned run_ms)
{
    typedef char check_size
            [2
                     * (sizeof(BMK_timedFnState_shell)
                        >= sizeof(struct BMK_timedFnState_s))
             - 1]; /* static assert : a compilation failure indicates that
                      BMK_timedFnState_shell is not large enough */
    typedef struct {
        check_size c;
        BMK_timedFnState_t tfs;
    } tfs_align; /* force tfs to be aligned at its next best position */
    size_t const tfs_alignment =
            offsetof(tfs_align, tfs); /* provides the minimal alignment
                                         restriction for BMK_timedFnState_t */
    BMK_timedFnState_t* const r = (BMK_timedFnState_t*)buffer;
    if (buffer == NULL)
        return NULL;
    if (size < sizeof(struct BMK_timedFnState_s))
        return NULL;
    if ((size_t)buffer % tfs_alignment)
        return NULL; /* buffer must be properly aligned */
    BMK_resetTimedFnState(r, total_ms, run_ms);
    return r;
}

void BMK_resetTimedFnState(
        BMK_timedFnState_t* timedFnState,
        unsigned total_ms,
        unsigned run_ms)
{
    if (!total_ms)
        total_ms = 1;
    if (!run_ms)
        run_ms = 1;
    if (run_ms > total_ms)
        run_ms = total_ms;
    timedFnState->timeSpent  = 0;
    timedFnState->timeBudget = (Duration_ns)total_ms * TIMELOOP_NANOSEC / 1000;
    timedFnState->runBudget  = (Duration_ns)run_ms * TIMELOOP_NANOSEC / 1000;
    timedFnState->fastestRun.nanoSecPerRun = (double)TIMELOOP_NANOSEC
            * 2000000000; /* hopefully large enough : must be larger than any
                             potential measurement */
    timedFnState->fastestRun.sumOfReturn = (size_t)(-1LL);
    timedFnState->nbLoops                = 1;
    timedFnState->coolTime               = TIME_getTime();
}

/* Tells if nb of seconds set in timedFnState for all runs is spent.
 * note : this function will return 1 if BMK_benchFunctionTimed() has actually
 * errored. */
int BMK_isCompleted_TimedFn(const BMK_timedFnState_t* timedFnState)
{
    return (timedFnState->timeSpent >= timedFnState->timeBudget);
}

#undef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define MINUSABLETIME (TIMELOOP_NANOSEC / 2) /* 0.5 seconds */

BMK_runOutcome_t BMK_benchTimedFn(BMK_timedFnState_t* cont, BMK_benchParams_t p)
{
    Duration_ns const runBudget  = cont->runBudget;
    Duration_ns const runTimeMin = { runBudget / 2 };
    int completed                = 0;
    BMK_runTime_t bestRunTime    = cont->fastestRun;

    while (!completed) {
        BMK_runOutcome_t const runResult = BMK_benchFunction(p, cont->nbLoops);

        if (!BMK_isSuccessful_runOutcome(runResult)) { /* error : move out */
            return runResult;
        }

        {
            BMK_runTime_t const newRunTime = BMK_extract_runTime(runResult);
            double const looduration_ns_ns =
                    newRunTime.nanoSecPerRun * cont->nbLoops;

            cont->timeSpent += (Duration_ns)looduration_ns_ns;

            /* estimate nbLoops for next run to last runBudget.ns */
            if (looduration_ns_ns > ((double)runBudget / 50)) {
                double const fastestRun_ns = MIN(
                        bestRunTime.nanoSecPerRun, newRunTime.nanoSecPerRun);
                cont->nbLoops =
                        (unsigned)((double)runBudget / fastestRun_ns) + 1;
            } else {
                /* previous run was too short : blindly increase workload by x
                 * multiplier */
                const unsigned multiplier = 10;
                assert(cont->nbLoops
                       < ((unsigned)-1) / multiplier); /* avoid overflow */
                cont->nbLoops *= multiplier;
            }

            if (looduration_ns_ns < (double)runTimeMin) {
                /* don't report results for which benchmark run time was too
                 * small : increased risks of rounding errors */
                assert(completed == 0);
                continue;
            } else {
                if (newRunTime.nanoSecPerRun < bestRunTime.nanoSecPerRun) {
                    bestRunTime = newRunTime;
                }
                completed = 1;
            }
        }
    } /* while (!completed) */

    return BMK_setValid_runTime(bestRunTime);
}

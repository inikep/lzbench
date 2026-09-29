// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZS_COMPRESS_MATCH_FINDER_H
#define ZS_COMPRESS_MATCH_FINDER_H

#include "openzl/codecs/common/window.h"
#include "openzl/codecs/rolz/encode_rolz_sequences.h"
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

typedef enum {
    ZS_MatchFinderStrategy_greedy,
    ZS_MatchFinderStrategy_lazy,
    ZS_MatchFinderStrategy_lazy2,
} ZS_MatchFinderStrategy_e;

/**
 * Cumulative list of ALL possible parameters.
 * Not all parameters are used by all match finders.
 */
typedef struct {
    bool rolzEnabled;
    unsigned rolzContextDepth; //< # of ROLZ context bytes (0 == disabled)
    unsigned rolzContextLog;   //< # of bits in the ROLZ context
    unsigned rolzRowLog;       //< Log # of entries in the ROLZ row context
    unsigned rolzMinLength;    //< Minimum match length for ROLZ
    unsigned rolzSearchLog;
    bool rolzPredictMatchLength; //< Should we use predicted match length?

    bool lzEnabled;
    unsigned lzHashLog;     //< LZ hash log
    unsigned lzChainLog;    //< LZ chain log
    unsigned lzMinLength;   //< Minimum match length for LZ
    unsigned lzSearchLog;   //< Log # of LZ searches to do
    unsigned lzSearchDelay; //< Search this many positions behind ROLZ for LZ
                            // matches
    unsigned lzTableLog;
    unsigned lzRowLog;
    bool lzLargeMatch;

    unsigned tableLog;
    unsigned rowLog;
    unsigned searchLog;
    unsigned minLength;

    unsigned fieldSize;
    unsigned fixedOffset;

    ZS_MatchFinderStrategy_e strategy;

    unsigned repMinLength; //< Minimum match length for repcodes
} ZS_MatchFinderParameters;

typedef struct {
    ZS_window const* window;
} ZS_matchFinderCtx;

typedef struct {
    char const* name;
    // Returns the amount of memory needed for a match finder with the given
    // parameters.
    // An increase in size needed means a memory ghrowth is needed.
    size_t (*ctx_sizeNeeded)(ZS_MatchFinderParameters const* params);
    // Creates a new instance of a match finder, returns NULL on error
    ZS_matchFinderCtx* (*ctx_create)(
            ZS_window const* window,
            ZS_MatchFinderParameters const* params);
    // Resets a match finder, allowing it to be used again
    void (*ctx_reset)(ZS_matchFinderCtx* ctx);
    // Resizes and resets a match finder, resizing would reuse exisitng
    // allocated memory if possible and would grow if necessary, but never
    // shrink.
    // Returns true on success, false otherwise.
    bool (*ctx_resize)(
            ZS_matchFinderCtx* ctx,
            ZS_window const* window,
            ZS_MatchFinderParameters const* params);
    // Destroys the match finder context and releases all memory associated with
    // it.
    void (*ctx_release)(ZS_matchFinderCtx* ctx);
    // Parses the input stream and generates sequences.
    void (*parse)(
            ZS_matchFinderCtx* ctx,
            ZS_RolzSeqStore* seqs,
            uint8_t const* src,
            size_t size);
} ZS_matchFinder;

extern const ZS_matchFinder ZS_lazyMatchFinder;
extern const ZS_matchFinder ZS_doubleFastLzMatchFinder;

ZL_END_C_DECLS

#endif // ZS_COMPRESS_MATCH_FINDER_H

/* deflate.c -- compress data using the deflation algorithm
 * Copyright (C) 1995-2026 Jean-loup Gailly and Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/*
 *  ALGORITHM
 *
 *      The "deflation" process depends on being able to identify portions
 *      of the input text which are identical to earlier input (within a
 *      sliding window trailing behind the input currently being processed).
 *
 *      The most straightforward technique turns out to be the fastest for
 *      most input files: try all possible matches and select the longest.
 *      The key feature of this algorithm is that insertions into the string
 *      dictionary are very simple and thus fast, and deletions are avoided
 *      completely. Insertions are performed at each input character, whereas
 *      string matches are performed only when the previous match ends. So it
 *      is preferable to spend more time in matches to allow very fast string
 *      insertions and avoid deletions. The matching algorithm for small
 *      strings is inspired from that of Rabin & Karp. A brute force approach
 *      is used to find longer strings when a small match has been found.
 *      A similar algorithm is used in comic (by Jan-Mark Wams) and freeze
 *      (by Leonid Broukhis).
 *         A previous version of this file used a more sophisticated algorithm
 *      (by Fiala and Greene) which is guaranteed to run in linear amortized
 *      time, but has a larger average cost, uses more memory and is patented.
 *      However the F&G algorithm may be faster for some highly redundant
 *      files if the parameter max_chain_length (described below) is too large.
 *
 *  ACKNOWLEDGEMENTS
 *
 *      The idea of lazy evaluation of matches is due to Jan-Mark Wams, and
 *      I found it in 'freeze' written by Leonid Broukhis.
 *      Thanks to many people for bug reports and testing.
 *
 *  REFERENCES
 *
 *      Deutsch, L.P.,"DEFLATE Compressed Data Format Specification".
 *      Available at https://datatracker.ietf.org/doc/html/rfc1951
 *
 *      A description of the Rabin and Karp algorithm is given in the book
 *         "Algorithms" by R. Sedgewick, Addison-Wesley, p252.
 *
 *      Fiala,E.R., and Greene,D.H.
 *         Data Compression with Finite Windows, Comm.ACM, 32,4 (1989) 490-595
 *
 */

/* @(#) $Id$ */

#include "deflate.h"
#include "zturbo.h"

const char deflate_copyright[] =
   " deflate 1.3.2 Copyright 1995-2026 Jean-loup Gailly and Mark Adler ";
/*
  If you use the zlib library in a product, an acknowledgment is welcome
  in the documentation of your product. If for some reason you cannot
  include such an acknowledgment, I would appreciate that you keep this
  copyright string in the executable of your product.
 */

typedef enum {
    need_more,      /* block not completed, need more input or more output */
    block_done,     /* block flush performed */
    finish_started, /* finish started, need only more output at next deflate */
    finish_done     /* finish done, accept no more input or output */
} block_state;

typedef block_state (*compress_func)(deflate_state *s, int flush);
/* Compression function. Returns the block state after the call. */

local block_state deflate_stored(deflate_state *s, int flush);
local block_state deflate_fast(deflate_state *s, int flush);
#ifndef FASTEST
local block_state deflate_slow(deflate_state *s, int flush);
#endif
#ifdef Z_TURBO_THREADS
local void mt_reset(deflate_state *s);
local void mt_free(z_streamp strm, deflate_state *s);
local void mt_catch_up(deflate_state *s, unsigned end);
local void mt_restart(deflate_state *s, Z_U8 at, int partial);
#endif
local block_state deflate_rle(deflate_state *s, int flush);
local block_state deflate_huff(deflate_state *s, int flush);
#ifdef Z_TURBO_OPT
local block_state deflate_opt(deflate_state *s, int flush);
local int opt_alloc(z_streamp strm, deflate_state *s);
local void opt_free(z_streamp strm, deflate_state *s);
local void opt_dictionary(deflate_state *s, const Bytef *dict, uInt len);
#  define MAX_LEVEL 12
#else
#  define MAX_LEVEL 9
#endif

#ifdef Z_TURBO_OPT
/* ===========================================================================
 * zlib-turbo: levels 10 to 12. A compressor of its own, whose output is not
 * zlib's (levels 0 to 9 keep zlib's): it parses the input OPT_SEG bytes at a
 * time, with the window before them, and writes them as one or more blocks.
 *
 *   - Matches: a binary tree of the window's strings (by a hash of four bytes)
 *     gives, at each position, the longest match within a search depth and
 *     the nearest match for each shorter length; a hash of three bytes adds
 *     the nearest three-byte match. They are kept for the whole segment.
 *   - Parse: the cheapest path through the segment for a given bit cost of
 *     every literal, length and distance. The first costs come from the
 *     statistics of a lazy parse, as zlib's; then each block is parsed
 *     again with the costs of its own Huffman codes from the parse before,
 *     several times, keeping the smallest block.
 *   - Blocks: the segment is split where the statistics change, by dynamic
 *     programming over candidate boundaries with an entropy estimate of each
 *     block's size; adjacent blocks merge when that is smaller. Each block is
 *     written dynamic, fixed or stored, whichever is smallest, with optimal
 *     length-limited Huffman codes (package-merge) and the shortest coding of
 *     the code lengths.
 *
 * The output depends only on the input and the flushes, not on how the input
 * is fed: segments end only at flushes and every OPT_SEG bytes. The tree and
 * the hash tables are kept from one segment to the next. They hold positions
 * plus abase, which grows as the window moves down in in[], so that entries
 * from before the window are seen as none without clearing anything; and a
 * position goes into the tree only once MAX_MATCH bytes follow it, so that its
 * place in the tree is right.
 */

#define OPT_SEG (1U << 18)      /* bytes parsed at a time */
#define OPT_CACHE 8             /* match cache entries per position */
#define OPT_GRID 256            /* most candidate block boundaries */
#define OPT_H4_BITS 16          /* hash of four bytes, for the tree */
#define OPT_H3_BITS 20          /* hash of three bytes: rarely two in a slot */
#define OPT_UNSEEN 15           /* bits of a symbol the codes do not have */
#define OPT_GMIN 512            /* fewest symbols between candidate boundaries */
#define OPT_TINY 64             /* segments shorter: one parse, no passes */
#define OPT_SLACK (OPT_SEG >> 2) /* room in in[] for the window to move up */
#define OPT_LL 286              /* literal/length codes */
#define OPT_BIG 0xffffffffU

/* parameters of the levels besides the search depth and nice length (in
   configuration_table): parses per block, rounds of block splitting,
   candidate boundaries */
local const struct opt_level_s {
    unsigned passes, rounds, grid;
} opt_level[3] = {
/* 10 */ {2, 1, 96},
/* 11 */ {4, 1, 160},
/* 12 */ {8, 2, 256}};

/* The codes of a block. */
typedef struct opt_tree_s {
    uch ll[OPT_LL + 2], dl[30], bl[19];
    ush llc[OPT_LL + 2], dlc[30], blc[19];
    int hlit, hdist, hclen, nrle;
    ush rle[OPT_LL + 30];       /* code length symbols, extra bits << 5 */
    ulg bits;                   /* the whole block, header bit to end code */
} opt_tree;

typedef struct opt_state_s {
    uchf *zbuf;                 /* zlib's pending_buf, back when done */
    ulg zbuf_size;
    uchf *out;                  /* pending_buf while the levels are used */
    Bytef *in;                  /* window, then the bytes to parse */
    unsigned len;               /* bytes in in[] */
    unsigned start;             /* where the bytes not yet parsed start */
    int fresh;                  /* in[] empty: take the window from s */
    unsigned *child, *h4, *h3;  /* binary tree, hash tables: positions plus
                                   abase, 0 for none */
    unsigned abase;             /* added to positions in in[] when kept */
    unsigned ins;               /* the first position not in the tree yet */
    unsigned *off;              /* cache[off[i]..off[i + 1]): matches at i */
    unsigned *cache;            /* matches: length << 16 | distance - 1 */
    unsigned *cost, *ch;        /* parse: cost to the end, choice */
    unsigned *sym, *sym2, *tmp; /* symbols: length << 16 | literal or dist-1 */
    unsigned *flog;             /* f * 16 log2(f) */
    unsigned *pf;               /* symbol counts at candidate boundaries */
    unsigned nflog;
    ulg bb;                     /* bit buffer */
    int bc;                     /* bits in bb */
    uch dslot[512];             /* distance code of dist - 1, as d_code() */
    uch lslot[259];             /* length code - 257 of a length */
    /* scratch */
    unsigned lc[259], dc[30], litc[256];
    unsigned fll[OPT_LL + 2], fd[30];
    uch pm[16][2 * (OPT_LL + 2)];
    ulg pw[2][2 * (OPT_LL + 2)];
    ush order[OPT_LL + 2];
    ulg rc[OPT_LL + 31];
    ush rch[OPT_LL + 31];
    ulg sb[OPT_GRID + 1];
    ush sf[OPT_GRID + 1];
    opt_tree t, t2;
} opt_state;
#endif /* Z_TURBO_OPT */

/* ===========================================================================
 * Local data
 */

#define NIL 0
/* Tail of hash chains */

#ifndef TOO_FAR
#  define TOO_FAR 4096
#endif
/* Matches of length 3 are discarded if their distance exceeds TOO_FAR */

/* Values for max_lazy_match, good_match and max_chain_length, depending on
 * the desired pack level (0..9). The values given below have been tuned to
 * exclude worst case performance for pathological files. Better values may be
 * found for specific files.
 */
typedef struct config_s {
   ush good_length; /* reduce lazy search above this match length */
   ush max_lazy;    /* do not perform lazy search above this match length */
   ush nice_length; /* quit search above this match length */
   ush max_chain;
   compress_func func;
} config;

#ifdef FASTEST
local const config configuration_table[2] = {
/*      good lazy nice chain */
/* 0 */ {0,    0,  0,    0, deflate_stored},  /* store only */
/* 1 */ {4,    4,  8,    4, deflate_fast}}; /* max speed, no lazy matches */
#else
local const config configuration_table[MAX_LEVEL + 1] = {
/*      good lazy nice chain */
/* 0 */ {0,    0,  0,    0, deflate_stored},  /* store only */
/* 1 */ {4,    4,  8,    4, deflate_fast}, /* max speed, no lazy matches */
/* 2 */ {4,    5, 16,    8, deflate_fast},
/* 3 */ {4,    6, 32,   32, deflate_fast},

/* 4 */ {4,    4, 16,   16, deflate_slow},  /* lazy matches */
/* 5 */ {8,   16, 32,   32, deflate_slow},
/* 6 */ {8,   16, 128, 128, deflate_slow},
/* 7 */ {8,   32, 128, 256, deflate_slow},
/* 8 */ {32, 128, 258, 1024, deflate_slow},
/* 9 */ {32, 258, 258, 4096, deflate_slow}, /* max compression */
#ifdef Z_TURBO_OPT
/* zlib-turbo: levels 10 to 12, nice length and search depth only */
/* 10 */ {0,    0,  96,   24, deflate_opt},
/* 11 */ {0,    0, 160,   64, deflate_opt},
/* 12 */ {0,    0, 258,  256, deflate_opt},
#endif
};
#endif

/* Note: the deflate() code requires max_lazy >= MIN_MATCH and max_chain >= 4
 * For deflate_fast() (levels <= 3) good is ignored and lazy has a different
 * meaning.
 */

/* rank Z_BLOCK between Z_NO_FLUSH and Z_PARTIAL_FLUSH */
#define RANK(f) (((f) * 2) - ((f) > 4 ? 9 : 0))

/* ===========================================================================
 * Update a hash value with the given input byte
 * IN  assertion: all calls to UPDATE_HASH are made with consecutive input
 *    characters, so that a running hash key can be computed from the previous
 *    key instead of complete recalculation each time.
 */
#define UPDATE_HASH(s,h,c) (h = (((h) << s->hash_shift) ^ (c)) & s->hash_mask)


/* ===========================================================================
 * Insert string str in the dictionary and set match_head to the previous head
 * of the hash chain (the most recent string with same hash key). Return
 * the previous length of the hash chain.
 * If this file is compiled with -DFASTEST, the compression level is forced
 * to 1, and no hash chains are maintained.
 * IN  assertion: all calls to INSERT_STRING are made with consecutive input
 *    characters and the first MIN_MATCH bytes of str are valid (except for
 *    the last MIN_MATCH-1 bytes of the input file).
 */
#ifdef FASTEST
#define INSERT_STRING(s, str, match_head) \
   (UPDATE_HASH(s, s->ins_h, s->window[(str) + (MIN_MATCH-1)]), \
    match_head = s->head[s->ins_h], \
    s->head[s->ins_h] = (Pos)(str))
#else
#define INSERT_STRING(s, str, match_head) \
   (UPDATE_HASH(s, s->ins_h, s->window[(str) + (MIN_MATCH-1)]), \
    match_head = s->prev[(str) & s->w_mask] = s->head[s->ins_h], \
    s->head[s->ins_h] = (Pos)(str))
#endif

/* ===========================================================================
 * Initialize the hash table (avoiding 64K overflow for 16 bit systems).
 * prev[] will be initialized on the fly.
 */
#define CLEAR_HASH(s) \
    do { \
        s->head[s->hash_size - 1] = NIL; \
        zmemzero(s->head, (unsigned)(s->hash_size - 1)*sizeof(*s->head)); \
        s->slid = 0; \
    } while (0)

/* ===========================================================================
 * Slide the hash table when sliding the window down (could be avoided with 32
 * bit values at the expense of memory usage). We slide even when level == 0 to
 * keep the hash table consistent if we switch back to level > 0 later.
 *
 * slide_table() subtracts wsize from the n entries at p, or sets them to NIL
 * if less than wsize. n is a multiple of eight, since hash_size and w_size are
 * powers of two no less than 256. The eight independent updates per step let
 * compilers use vector instructions (a saturating 16-bit subtraction). In
 * prev[], an entry not on any hash chain is garbage, but its value will never
 * be used.
 */
#if defined(__has_feature)
#  if __has_feature(memory_sanitizer)
     __attribute__((no_sanitize("memory")))
#  endif
#endif
local void slide_table(Posf *p, unsigned n, unsigned wsize) {
    Pos w = (Pos)wsize;
    int i;

    do {
        for (i = 0; i < 8; i++) {
            Pos m = p[i];
            p[i] = (Pos)(m >= w ? m - w : NIL);
        }
        p += 8;
        n -= 8;
    } while (n);
}

local void slide_hash(deflate_state *s) {
    slide_table(s->head, s->hash_size, s->w_size);
#ifndef FASTEST
    slide_table(s->prev, s->w_size, s->w_size);
#endif
    s->slid = 1;
}

/* ===========================================================================
 * Read a new buffer from the current input stream, update the adler32
 * and total number of bytes read.  All deflate() input goes through
 * this function so some applications may wish to modify it to avoid
 * allocating a large strm->next_in buffer and copying from it.
 * (See also flush_pending()).
 */
local unsigned read_buf(z_streamp strm, Bytef *buf, unsigned size) {
    unsigned len = strm->avail_in;

    if (len > size) len = size;
    if (len == 0) return 0;

    strm->avail_in  -= len;

    zmemcpy(buf, strm->next_in, len);
    if (strm->state->wrap == 1) {
        strm->adler = adler32(strm->adler, buf, len);
    }
#ifdef GZIP
    else if (strm->state->wrap == 2) {
        strm->adler = crc32(strm->adler, buf, len);
    }
#endif
    strm->next_in  += len;
    strm->total_in += len;

    return len;
}

/* ===========================================================================
 * Fill the window when the lookahead becomes insufficient.
 * Updates strstart and lookahead.
 *
 * IN assertion: lookahead < MIN_LOOKAHEAD
 * OUT assertions: strstart <= window_size-MIN_LOOKAHEAD
 *    At least one byte has been read, or avail_in == 0; reads are
 *    performed for at least two bytes (required for the zip translate_eol
 *    option -- not supported here).
 */
local void fill_window(deflate_state *s) {
    unsigned n;
    unsigned more;    /* Amount of free space at the end of the window. */
    uInt wsize = s->w_size;

    Assert(s->lookahead < MIN_LOOKAHEAD, "already enough lookahead");

    do {
        more = (unsigned)(s->window_size -(ulg)s->lookahead -(ulg)s->strstart);

        /* Deal with !@#$% 64K limit: */
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4127)
#endif
        if (sizeof(int) <= 2) {
#ifdef _MSC_VER
#pragma warning(pop)
#endif
            if (more == 0 && s->strstart == 0 && s->lookahead == 0) {
                more = wsize;

            } else if (more == (unsigned)(-1)) {
                /* Very unlikely, but possible on 16 bit machine if
                 * strstart == 0 && lookahead == 1 (input done a byte at time)
                 */
                more--;
            }
        }

        /* If the window is almost full and there is insufficient lookahead,
         * move the upper half to the lower one to make room in the upper half.
         */
        if (s->strstart >= wsize + MAX_DIST(s)) {

            zmemcpy(s->window, s->window + wsize, (unsigned)wsize - more);
            s->match_start -= wsize;
            s->strstart    -= wsize; /* we now have strstart >= MAX_DIST */
            s->block_start -= (long) wsize;
            if (s->insert > s->strstart)
                s->insert = s->strstart;
#ifdef Z_TURBO_THREADS
            if (s->mt_lag)
                s->mt_slides++;     /* done when the parse catches up */
            else
#endif
            {
                slide_hash(s);
            }
#ifdef Z_TURBO_THREADS
            s->mt_abs += wsize;
#endif
            more += wsize;
        }
        if (s->strm->avail_in == 0) break;

        /* If there was no sliding:
         *    strstart <= WSIZE+MAX_DIST-1 && lookahead <= MIN_LOOKAHEAD - 1 &&
         *    more == window_size - lookahead - strstart
         * => more >= window_size - (MIN_LOOKAHEAD-1 + WSIZE + MAX_DIST-1)
         * => more >= window_size - 2*WSIZE + 2
         * In the BIG_MEM or MMAP case (not yet supported),
         *   window_size == input_size + MIN_LOOKAHEAD  &&
         *   strstart + s->lookahead <= input_size => more >= MIN_LOOKAHEAD.
         * Otherwise, window_size == 2*WSIZE so more >= 2.
         * If there was sliding, more >= WSIZE. So in all cases, more >= 2.
         */
        Assert(more >= 2, "more < 2");

        n = read_buf(s->strm, s->window + s->strstart + s->lookahead, more);
        s->lookahead += n;

        /* Initialize the hash value now that we have some input: */
#ifdef Z_TURBO_THREADS
        if (s->insert && s->mt_lag)
            mt_catch_up(s, s->strstart - s->insert);
#endif
        if (s->lookahead + s->insert >= MIN_MATCH) {
            uInt str = s->strstart - s->insert;
            s->ins_h = s->window[str];
            UPDATE_HASH(s, s->ins_h, s->window[str + 1]);
#if MIN_MATCH != 3
            Call UPDATE_HASH() MIN_MATCH-3 more times
#endif
            while (s->insert) {
                UPDATE_HASH(s, s->ins_h, s->window[str + MIN_MATCH-1]);
#ifndef FASTEST
                s->prev[str & s->w_mask] = s->head[s->ins_h];
#endif
                s->head[s->ins_h] = (Pos)str;
                str++;
                s->insert--;
                if (s->lookahead + s->insert < MIN_MATCH)
                    break;
            }
        }
        /* If the whole input has less than MIN_MATCH bytes, ins_h is garbage,
         * but this is not important since only literal bytes will be emitted.
         */

    } while (s->lookahead < MIN_LOOKAHEAD && s->strm->avail_in != 0);

    /* If the WIN_INIT bytes after the end of the current data have never been
     * written, then zero those bytes in order to avoid memory check reports of
     * the use of uninitialized (or uninitialised as Julian writes) bytes by
     * the longest match routines.  Update the high water mark for the next
     * time through here.  WIN_INIT is set to MAX_MATCH since the longest match
     * routines allow scanning to strstart + MAX_MATCH, ignoring lookahead.
     */
    if (s->high_water < s->window_size) {
        ulg curr = s->strstart + (ulg)(s->lookahead);
        ulg init;

        if (s->high_water < curr) {
            /* Previous high water mark below current data -- zero WIN_INIT
             * bytes or up to end of window, whichever is less.
             */
            init = s->window_size - curr;
            if (init > WIN_INIT)
                init = WIN_INIT;
            zmemzero(s->window + curr, (unsigned)init);
            s->high_water = curr + init;
        }
        else if (s->high_water < (ulg)curr + WIN_INIT) {
            /* High water mark at or above current data, but below current data
             * plus WIN_INIT -- zero out to current data plus WIN_INIT, or up
             * to end of window, whichever is less.
             */
            init = (ulg)curr + WIN_INIT - s->high_water;
            if (init > s->window_size - s->high_water)
                init = s->window_size - s->high_water;
            zmemzero(s->window + s->high_water, (unsigned)init);
            s->high_water += init;
        }
    }

    Assert((ulg)s->strstart <= s->window_size - MIN_LOOKAHEAD,
           "not enough room for search");
}

/* ========================================================================= */
int ZEXPORT deflateInit_(z_streamp strm, int level, const char *version,
                         int stream_size) {
    return deflateInit2_(strm, level, Z_DEFLATED, MAX_WBITS, DEF_MEM_LEVEL,
                         Z_DEFAULT_STRATEGY, version, stream_size);
    /* To do: ignore strm->next_in if we use it as window */
}

/* ========================================================================= */
int ZEXPORT deflateInit2_(z_streamp strm, int level, int method,
                          int windowBits, int memLevel, int strategy,
                          const char *version, int stream_size) {
    deflate_state *s;
    int wrap = 1;
    static const char my_version[] = ZLIB_VERSION;

    if (version == Z_NULL || version[0] != my_version[0] ||
        stream_size != sizeof(z_stream)) {
        return Z_VERSION_ERROR;
    }
    if (strm == Z_NULL) return Z_STREAM_ERROR;

    strm->msg = Z_NULL;
    if (strm->zalloc == (alloc_func)0) {
#ifdef Z_SOLO
        return Z_STREAM_ERROR;
#else
        strm->zalloc = zcalloc;
        strm->opaque = (voidpf)0;
#endif
    }
    if (strm->zfree == (free_func)0)
#ifdef Z_SOLO
        return Z_STREAM_ERROR;
#else
        strm->zfree = zcfree;
#endif

#ifdef FASTEST
    if (level != 0) level = 1;
#else
    if (level == Z_DEFAULT_COMPRESSION) level = 6;
#endif

    if (windowBits < 0) { /* suppress zlib wrapper */
        wrap = 0;
        if (windowBits < -15)
            return Z_STREAM_ERROR;
        windowBits = -windowBits;
    }
#ifdef GZIP
    else if (windowBits > 15) {
        wrap = 2;       /* write gzip wrapper instead */
        windowBits -= 16;
    }
#endif
    if (memLevel < 1 || memLevel > MAX_MEM_LEVEL || method != Z_DEFLATED ||
        windowBits < 8 || windowBits > 15 || level < 0 ||
        level > MAX_LEVEL ||
        strategy < 0 || strategy > Z_FIXED || (windowBits == 8 && wrap != 1)) {
        return Z_STREAM_ERROR;
    }
    if (windowBits == 8) windowBits = 9;  /* until 256-byte window bug fixed */
    s = (deflate_state *) ZALLOC(strm, 1, sizeof(deflate_state));
    if (s == Z_NULL) return Z_MEM_ERROR;
    zmemzero(s, sizeof(deflate_state));
    strm->state = (struct internal_state FAR *)s;
    s->strm = strm;
    s->status = INIT_STATE;     /* to pass state test in deflateReset() */

    s->wrap = wrap;
    s->gzhead = Z_NULL;
    s->w_bits = (uInt)windowBits;
    s->w_size = 1 << s->w_bits;
    s->w_mask = s->w_size - 1;

    s->hash_bits = (uInt)memLevel + 7;
    s->hash_size = 1 << s->hash_bits;
    s->hash_mask = s->hash_size - 1;
    s->hash_shift =  ((s->hash_bits + MIN_MATCH-1) / MIN_MATCH);

    s->window = (Bytef *) ZALLOC(strm, s->w_size, 2*sizeof(Byte));
    s->prev   = (Posf *)  ZALLOC(strm, s->w_size, sizeof(Pos));
    s->head   = (Posf *)  ZALLOC(strm, s->hash_size, sizeof(Pos));

    s->high_water = 0;      /* nothing written to s->window yet */

    s->lit_bufsize = 1 << (memLevel + 6); /* 16K elements by default */

    /* We overlay pending_buf and sym_buf. This works since the average size
     * for length/distance pairs over any compressed block is assured to be 31
     * bits or less.
     *
     * Analysis: The longest fixed codes are a length code of 8 bits plus 5
     * extra bits, for lengths 131 to 257. The longest fixed distance codes are
     * 5 bits plus 13 extra bits, for distances 16385 to 32768. The longest
     * possible fixed-codes length/distance pair is then 31 bits total.
     *
     * sym_buf starts one-fourth of the way into pending_buf. So there are
     * three bytes in sym_buf for every four bytes in pending_buf. Each symbol
     * in sym_buf is three bytes -- two for the distance and one for the
     * literal/length. As each symbol is consumed, the pointer to the next
     * sym_buf value to read moves forward three bytes. From that symbol, up to
     * 31 bits are written to pending_buf. The closest the written pending_buf
     * bits gets to the next sym_buf symbol to read is just before the last
     * code is written. At that time, 31*(n - 2) bits have been written, just
     * after 24*(n - 2) bits have been consumed from sym_buf. sym_buf starts at
     * 8*n bits into pending_buf. (Note that the symbol buffer fills when n - 1
     * symbols are written.) The closest the writing gets to what is unread is
     * then n + 14 bits. Here n is lit_bufsize, which is 16384 by default, and
     * can range from 128 to 32768.
     *
     * Therefore, at a minimum, there are 142 bits of space between what is
     * written and what is read in the overlain buffers, so the symbols cannot
     * be overwritten by the compressed data. That space is actually 139 bits,
     * due to the three-bit fixed-code block header.
     *
     * That covers the case where either Z_FIXED is specified, forcing fixed
     * codes, or when the use of fixed codes is chosen, because that choice
     * results in a smaller compressed block than dynamic codes. That latter
     * condition then assures that the above analysis also covers all dynamic
     * blocks. A dynamic-code block will only be chosen to be emitted if it has
     * fewer bits than a fixed-code block would for the same set of symbols.
     * Therefore its average symbol length is assured to be less than 31. So
     * the compressed data for a dynamic block also cannot overwrite the
     * symbols from which it is being constructed.
     */

    s->pending_buf = (uchf *) ZALLOC(strm, s->lit_bufsize, LIT_BUFS);
    s->pending_buf_size = (ulg)s->lit_bufsize * 4;

    if (s->window == Z_NULL || s->prev == Z_NULL || s->head == Z_NULL ||
        s->pending_buf == Z_NULL) {
        s->status = FINISH_STATE;
        strm->msg = ERR_MSG(Z_MEM_ERROR);
        deflateEnd (strm);
        return Z_MEM_ERROR;
    }
#ifdef LIT_MEM
    s->d_buf = (ushf *)(s->pending_buf + (s->lit_bufsize << 1));
    s->l_buf = s->pending_buf + (s->lit_bufsize << 2);
    s->sym_end = s->lit_bufsize - 1;
#else
    s->sym_buf = s->pending_buf + s->lit_bufsize;
    s->sym_end = (s->lit_bufsize - 1) * 3;
#endif
    /* We avoid equality with lit_bufsize*3 because of wraparound at 64K
     * on 16 bit machines and because stored blocks are restricted to
     * 64K-1 bytes.
     */

    s->level = level;
    s->strategy = strategy;
    s->method = (Byte)method;
#ifdef Z_TURBO_OPT
    if (level > 9 && opt_alloc(strm, s) != Z_OK) {
        s->status = FINISH_STATE;
        strm->msg = ERR_MSG(Z_MEM_ERROR);
        deflateEnd (strm);
        return Z_MEM_ERROR;
    }
#endif

    return deflateReset(strm);
}

/* =========================================================================
 * Check for a valid deflate stream state. Return 0 if ok, 1 if not.
 */
local int deflateStateCheck(z_streamp strm) {
    deflate_state *s;
    if (strm == Z_NULL ||
        strm->zalloc == (alloc_func)0 || strm->zfree == (free_func)0)
        return 1;
    s = strm->state;
    if (s == Z_NULL || s->strm != strm || (s->status != INIT_STATE &&
#ifdef GZIP
                                           s->status != GZIP_STATE &&
#endif
                                           s->status != EXTRA_STATE &&
                                           s->status != NAME_STATE &&
                                           s->status != COMMENT_STATE &&
                                           s->status != HCRC_STATE &&
                                           s->status != BUSY_STATE &&
                                           s->status != FINISH_STATE))
        return 1;
    return 0;
}

/* ========================================================================= */
int ZEXPORT deflateSetDictionary(z_streamp strm, const Bytef *dictionary,
                                 uInt  dictLength) {
    deflate_state *s;
    uInt str, n;
    int wrap;
    unsigned avail;
    z_const unsigned char *next;

    if (deflateStateCheck(strm) || dictionary == Z_NULL)
        return Z_STREAM_ERROR;
    s = strm->state;
    wrap = s->wrap;
    if (wrap == 2 || (wrap == 1 && s->status != INIT_STATE) || s->lookahead)
        return Z_STREAM_ERROR;

    /* when using zlib wrappers, compute Adler-32 for provided dictionary */
    if (wrap == 1)
        strm->adler = adler32(strm->adler, dictionary, dictLength);
#ifdef Z_TURBO_OPT
    if (s->opt != Z_NULL && !s->opt->fresh) {
        opt_dictionary(s, dictionary, dictLength);
        return Z_OK;
    }
#endif
    s->wrap = 0;                    /* avoid computing Adler-32 in read_buf */

    /* if dictionary would fill window, just replace the history */
    if (dictLength >= s->w_size) {
        if (wrap == 0) {            /* already empty otherwise */
#ifdef Z_TURBO_THREADS
            Z_U8 at = s->mt_abs + s->strstart;
#endif
            CLEAR_HASH(s);
            s->strstart = 0;
            s->block_start = 0L;
            s->insert = 0;
#ifdef Z_TURBO_THREADS
            mt_restart(s, at, 0);   /* the window starts again */
#endif
        }
        dictionary += dictLength - s->w_size;  /* use the tail */
        dictLength = s->w_size;
    }

    /* insert dictionary into window and hash */
    avail = strm->avail_in;
    next = strm->next_in;
    strm->avail_in = dictLength;
    strm->next_in = (z_const Bytef *)dictionary;
    fill_window(s);
    while (s->lookahead >= MIN_MATCH) {
        str = s->strstart;
        n = s->lookahead - (MIN_MATCH-1);
        do {
            UPDATE_HASH(s, s->ins_h, s->window[str + MIN_MATCH-1]);
#ifndef FASTEST
            s->prev[str & s->w_mask] = s->head[s->ins_h];
#endif
            s->head[s->ins_h] = (Pos)str;
            str++;
        } while (--n);
        s->strstart = str;
        s->lookahead = MIN_MATCH-1;
        fill_window(s);
    }
    s->strstart += s->lookahead;
    s->block_start = (long)s->strstart;
    s->insert = s->lookahead;
    s->lookahead = 0;
    s->match_length = s->prev_length = MIN_MATCH-1;
    s->match_available = 0;
    strm->next_in = next;
    strm->avail_in = avail;
    s->wrap = wrap;
#ifdef Z_TURBO_THREADS
    mt_reset(s);
#endif
    return Z_OK;
}

/* ========================================================================= */
int ZEXPORT deflateGetDictionary(z_streamp strm, Bytef *dictionary,
                                 uInt *dictLength) {
    deflate_state *s;
    uInt len;

    if (deflateStateCheck(strm))
        return Z_STREAM_ERROR;
    s = strm->state;
#ifdef Z_TURBO_OPT
    if (s->opt != Z_NULL && !s->opt->fresh) {
        len = s->opt->len < s->w_size ? s->opt->len : s->w_size;
        if (dictionary != Z_NULL && len)
            zmemcpy(dictionary, s->opt->in + s->opt->len - len, len);
        if (dictLength != Z_NULL)
            *dictLength = len;
        return Z_OK;
    }
#endif
    len = s->strstart + s->lookahead;
    if (len > s->w_size)
        len = s->w_size;
    if (dictionary != Z_NULL && len)
        zmemcpy(dictionary, s->window + s->strstart + s->lookahead - len, len);
    if (dictLength != Z_NULL)
        *dictLength = len;
    return Z_OK;
}

/* ===========================================================================
 * Prepared dictionaries (zturbo.h). A prepared dictionary holds the state that
 * deflateSetDictionary() leaves on a stream just initialized or reset: the hash
 * table, the hash chains of the dictionary's strings, ins_h, and the number of
 * strings not yet inserted. Copying that state gives what indexing would give.
 */
struct z_dict_s {
    free_func zfree;        /* to free this */
    voidpf opaque;
    uInt w_bits;            /* windowBits and hash_bits of the stream */
    uInt hash_bits;
    uLong adler;            /* Adler-32 of the whole dictionary */
    uInt len;               /* length of the whole dictionary */
    uInt tail;              /* bytes in the window: the last w_size at most */
    uInt insert;            /* strings not yet inserted at the end */
    uInt ins_h;             /* hash after the last insertion */
    Bytef *bytes;           /* the whole dictionary */
    Posf *head;             /* the hash table, 1 << hash_bits entries */
    Posf *prev;             /* the chains, tail - insert entries */
};

/* Whether s is as deflateInit2() or deflateReset() left it, with nothing in
   its window or hash table. */
local int fresh_state(deflate_state *s) {
    return s->status == INIT_STATE && s->strstart == 0 && s->lookahead == 0 &&
           s->insert == 0 && s->block_start == 0;
}

/* ========================================================================= */
int ZEXPORT deflatePrepareDictionary(z_streamp strm, const Bytef *dictionary,
                                     uInt dictLength, z_dict **dict) {
    deflate_state *s;
    z_dict *d;
    ulg nhead, nprev, size;
    int ret;

    if (dict == Z_NULL)
        return Z_STREAM_ERROR;
    *dict = Z_NULL;
    if (deflateStateCheck(strm) || dictionary == Z_NULL)
        return Z_STREAM_ERROR;
    s = strm->state;
    if (!fresh_state(s))
        return Z_STREAM_ERROR;
    ret = deflateSetDictionary(strm, dictionary, dictLength);
    if (ret != Z_OK)
        return ret;

    /* one allocation: the header, the two tables, then the bytes */
    nhead = s->hash_size;
    nprev = s->strstart - s->insert;
    size = sizeof(z_dict) + (nhead + nprev) * sizeof(Pos) + dictLength;
    if (size < dictLength || size != (uInt)size)
        return Z_MEM_ERROR;
    d = (z_dict *)ZALLOC(strm, 1, (uInt)size);
    if (d == Z_NULL)
        return Z_MEM_ERROR;
    d->zfree = strm->zfree;
    d->opaque = strm->opaque;
    d->w_bits = s->w_bits;
    d->hash_bits = s->hash_bits;
    d->adler = adler32(1L, dictionary, dictLength);
    d->len = dictLength;
    d->tail = s->strstart;
    d->insert = s->insert;
    d->ins_h = s->ins_h;
    d->head = (Posf *)(d + 1);
    d->prev = d->head + nhead;
    d->bytes = (Bytef *)(d->prev + nprev);
    zmemcpy(d->head, s->head, (unsigned)(nhead * sizeof(Pos)));
#ifndef FASTEST
    zmemcpy(d->prev, s->prev, (unsigned)(nprev * sizeof(Pos)));
#endif
    zmemcpy(d->bytes, dictionary, dictLength);
    *dict = d;
    return Z_OK;
}

/* ========================================================================= */
int ZEXPORT deflateUseDictionary(z_streamp strm, const z_dict *dict) {
    deflate_state *s;
    unsigned avail;
    z_const unsigned char *next;

    if (deflateStateCheck(strm) || dict == Z_NULL)
        return Z_STREAM_ERROR;
    s = strm->state;
    if (s->wrap == 2 || !fresh_state(s) || dict->w_bits != s->w_bits ||
        dict->hash_bits != s->hash_bits)
        return deflateSetDictionary(strm, dict->bytes, dict->len);

    /* what deflateSetDictionary() does, with the indexing copied */
    if (s->wrap == 1)
        strm->adler = dict->adler;
    zmemcpy(s->window, dict->bytes + (dict->len - dict->tail), dict->tail);
    zmemcpy(s->head, dict->head, (unsigned)(s->hash_size * sizeof(Pos)));
#ifndef FASTEST
    zmemcpy(s->prev, dict->prev,
            (unsigned)((dict->tail - dict->insert) * sizeof(Pos)));
#endif
    s->ins_h = dict->ins_h;
    s->strstart = dict->tail;
    avail = strm->avail_in;             /* fill_window() with no input: */
    next = strm->next_in;               /* only the zeroing past the end */
    strm->avail_in = 0;
    fill_window(s);
    strm->next_in = next;
    strm->avail_in = avail;
    s->block_start = (long)s->strstart;
    s->insert = dict->insert;
    s->match_length = s->prev_length = MIN_MATCH-1;
    s->match_available = 0;
    strm->total_in += dict->tail;       /* as read_buf() counts it */
#ifdef Z_TURBO_THREADS
    mt_reset(s);
#endif
    return Z_OK;
}

/* ========================================================================= */
void ZEXPORT deflateFreeDictionary(z_dict *dict) {
    if (dict != Z_NULL)
        dict->zfree(dict->opaque, (voidpf)dict);
}

/* ========================================================================= */
int ZEXPORT deflateResetKeep(z_streamp strm) {
    deflate_state *s;

    if (deflateStateCheck(strm)) {
        return Z_STREAM_ERROR;
    }

    strm->total_in = strm->total_out = 0;
    strm->msg = Z_NULL; /* use zfree if we ever allocate msg dynamically */
    strm->data_type = Z_UNKNOWN;

    s = (deflate_state *)strm->state;
    s->pending = 0;
    s->pending_out = s->pending_buf;

    if (s->wrap < 0) {
        s->wrap = -s->wrap; /* was made negative by deflate(..., Z_FINISH); */
    }
    s->status =
#ifdef GZIP
        s->wrap == 2 ? GZIP_STATE :
#endif
        INIT_STATE;
    strm->adler =
#ifdef GZIP
        s->wrap == 2 ? crc32(0L, Z_NULL, 0) :
#endif
        adler32(0L, Z_NULL, 0);
    s->last_flush = -2;

    _tr_init(s);

    return Z_OK;
}

/* ===========================================================================
 * Initialize the "longest match" routines for a new zlib stream
 */
local void lm_init(deflate_state *s) {
    s->window_size = (ulg)2L*s->w_size;

    CLEAR_HASH(s);

    /* Set the default configuration parameters:
     */
    s->max_lazy_match   = configuration_table[s->level].max_lazy;
    s->good_match       = configuration_table[s->level].good_length;
    s->nice_match       = configuration_table[s->level].nice_length;
    s->max_chain_length = configuration_table[s->level].max_chain;

    s->strstart = 0;
    s->block_start = 0L;
    s->lookahead = 0;
    s->insert = 0;
    s->match_length = s->prev_length = MIN_MATCH-1;
    s->match_available = 0;
    s->ins_h = 0;
#ifdef Z_TURBO_THREADS
    s->mt_abs = 0;
    s->mt_exact = 1;
    s->mt_lag = 0;
    s->mt_slides = 0;
    s->mt_floor = 1;            /* window[0] is never matched */
    s->mt_from = 0;
    mt_reset(s);
#endif
#ifdef Z_TURBO_OPT
    if (s->opt != Z_NULL) {
        s->opt->abase += s->opt->len + 1;       /* opt_forget() */
        s->opt->len = s->opt->start = s->opt->ins = 0;
        s->opt->fresh = 1;
    }
#endif
}

/* ========================================================================= */
int ZEXPORT deflateReset(z_streamp strm) {
    int ret;

    ret = deflateResetKeep(strm);
    if (ret == Z_OK)
        lm_init(strm->state);
    return ret;
}

/* ========================================================================= */
int ZEXPORT deflateSetHeader(z_streamp strm, gz_headerp head) {
    if (deflateStateCheck(strm) || strm->state->wrap != 2)
        return Z_STREAM_ERROR;
    strm->state->gzhead = head;
    return Z_OK;
}

/* ========================================================================= */
int ZEXPORT deflatePending(z_streamp strm, unsigned *pending, int *bits) {
    if (deflateStateCheck(strm)) return Z_STREAM_ERROR;
    if (bits != Z_NULL)
        *bits = strm->state->bi_valid;
    if (pending != Z_NULL) {
        *pending = (unsigned)strm->state->pending;
        if (*pending != strm->state->pending) {
            *pending = (unsigned)-1;
            return Z_BUF_ERROR;
        }
    }
    return Z_OK;
}

/* ========================================================================= */
int ZEXPORT deflateUsed(z_streamp strm, int *bits) {
    if (deflateStateCheck(strm)) return Z_STREAM_ERROR;
    if (bits != Z_NULL)
        *bits = strm->state->bi_used;
    return Z_OK;
}

/* ========================================================================= */
int ZEXPORT deflatePrime(z_streamp strm, int bits, int value) {
    deflate_state *s;
    int put;

    if (deflateStateCheck(strm)) return Z_STREAM_ERROR;
    s = strm->state;
    /* zlib-turbo: where zlib's symbol buffer starts, also when threads use
       their own */
#ifdef LIT_MEM
    if (bits < 0 || bits > 16 ||
        s->pending_buf + (s->lit_bufsize << 1) <
            s->pending_out + ((Buf_size + 7) >> 3))
        return Z_BUF_ERROR;
#else
    if (bits < 0 || bits > 16 ||
        s->pending_buf + s->lit_bufsize <
            s->pending_out + ((Buf_size + 7) >> 3))
        return Z_BUF_ERROR;
#endif
    do {
        put = Buf_size - s->bi_valid;
        if (put > bits)
            put = bits;
        s->bi_buf |= (ush)((value & ((1 << put) - 1)) << s->bi_valid);
        s->bi_valid += put;
        _tr_flush_bits(s);
        value >>= put;
        bits -= put;
    } while (bits);
    return Z_OK;
}

/* ========================================================================= */
int ZEXPORT deflateParams(z_streamp strm, int level, int strategy) {
    deflate_state *s;
    compress_func func;
#if defined(Z_TURBO_THREADS) && !defined(FASTEST)
    int full;
#endif

    if (deflateStateCheck(strm)) return Z_STREAM_ERROR;
    s = strm->state;

#ifdef FASTEST
    if (level != 0) level = 1;
#else
    if (level == Z_DEFAULT_COMPRESSION) level = 6;
#endif
    if (level < 0 || level > MAX_LEVEL || strategy < 0 ||
        strategy > Z_FIXED) {
        return Z_STREAM_ERROR;
    }
#ifdef Z_TURBO_OPT
    if ((level > 9) != (s->level > 9)) {
        /* to or from levels 10 to 12: only before deflate() starts */
        if (s->last_flush != -2 || s->pending)
            return Z_STREAM_ERROR;
        if (level > 9) {
            if (opt_alloc(strm, s) != Z_OK) {
                opt_free(strm, s);
                return Z_MEM_ERROR;
            }
        }
        else
            opt_free(strm, s);
    }
#endif
    func = configuration_table[s->level].func;
#if defined(Z_TURBO_THREADS) && !defined(FASTEST)
    /* whether the stream so far inserted every position in the hash chains */
    full = s->level != 0 && func == deflate_slow &&
           s->strategy != Z_HUFFMAN_ONLY && s->strategy != Z_RLE;
#endif

    if ((strategy != s->strategy || func != configuration_table[level].func) &&
        s->last_flush != -2) {
        /* Flush the last buffer: */
        int err = deflate(strm, Z_BLOCK);
        if (err == Z_STREAM_ERROR)
            return err;
        if (strm->avail_in || (s->strstart - s->block_start) + s->lookahead)
            return Z_BUF_ERROR;
    }
    if (s->level != level) {
        if (s->level == 0 && s->matches != 0) {
            if (s->matches == 1)
                slide_hash(s);
            else
                CLEAR_HASH(s);
            s->matches = 0;
        }
        s->level = level;
        s->max_lazy_match   = configuration_table[level].max_lazy;
        s->good_match       = configuration_table[level].good_length;
        s->nice_match       = configuration_table[level].nice_length;
        s->max_chain_length = configuration_table[level].max_chain;
    }
    s->strategy = strategy;
#if defined(Z_TURBO_THREADS) && !defined(FASTEST)
    if (!full && s->last_flush != -2)
        mt_restart(s, s->mt_abs + s->strstart, 1);
#endif
    return Z_OK;
}

/* ========================================================================= */
int ZEXPORT deflateTune(z_streamp strm, int good_length, int max_lazy,
                        int nice_length, int max_chain) {
    deflate_state *s;

    if (deflateStateCheck(strm)) return Z_STREAM_ERROR;
    s = strm->state;
    s->good_match = (uInt)good_length;
    s->max_lazy_match = (uInt)max_lazy;
    s->nice_match = nice_length;
    s->max_chain_length = (uInt)max_chain;
    return Z_OK;
}

/* =========================================================================
 * For the default windowBits of 15 and memLevel of 8, this function returns a
 * close to exact, as well as small, upper bound on the compressed size. This
 * is an expansion of ~0.03%, plus a small constant.
 *
 * For any setting other than those defaults for windowBits and memLevel, one
 * of two worst case bounds is returned. This is at most an expansion of ~4% or
 * ~13%, plus a small constant.
 *
 * Both the 0.03% and 4% derive from the overhead of stored blocks. The first
 * one is for stored blocks of 16383 bytes (memLevel == 8), whereas the second
 * is for stored blocks of 127 bytes (the worst case memLevel == 1). The
 * expansion results from five bytes of header for each stored block.
 *
 * The larger expansion of 13% results from a window size less than or equal to
 * the symbols buffer size (windowBits <= memLevel + 7). In that case some of
 * the data being compressed may have slid out of the sliding window, impeding
 * a stored block from being emitted. Then the only choice is a fixed or
 * dynamic block, where a fixed block limits the maximum expansion to 9 bits
 * per 8-bit byte, plus 10 bits for every block. The smallest block size for
 * which this can occur is 255 (memLevel == 2).
 *
 * Shifts are used to approximate divisions, for speed.
 */
z_size_t ZEXPORT deflateBound_z(z_streamp strm, z_size_t sourceLen) {
    deflate_state *s;
    z_size_t fixedlen, storelen, wraplen, bound;

    /* upper bound for fixed blocks with 9-bit literals and length 255
       (memLevel == 2, which is the lowest that may not use stored blocks) --
       ~13% overhead plus a small constant */
    fixedlen = sourceLen + (sourceLen >> 3) + (sourceLen >> 8) +
               (sourceLen >> 9) + 4;
    if (fixedlen < sourceLen)
        fixedlen = (z_size_t)-1;

    /* upper bound for stored blocks with length 127 (memLevel == 1) --
       ~4% overhead plus a small constant */
    storelen = sourceLen + (sourceLen >> 5) + (sourceLen >> 7) +
               (sourceLen >> 11) + 7;
    if (storelen < sourceLen)
        storelen = (z_size_t)-1;

    /* if can't get parameters, return larger bound plus a wrapper */
    if (deflateStateCheck(strm)) {
        bound = fixedlen > storelen ? fixedlen : storelen;
        return bound + 18 < bound ? (z_size_t)-1 : bound + 18;
    }

    /* compute wrapper length */
    s = strm->state;
    switch (s->wrap < 0 ? -s->wrap : s->wrap) {
    case 0:                                 /* raw deflate */
        wraplen = 0;
        break;
    case 1:                                 /* zlib wrapper */
        wraplen = 6 + (s->strstart ? 4 : 0);
        break;
#ifdef GZIP
    case 2:                                 /* gzip wrapper */
        wraplen = 18;
        if (s->gzhead != Z_NULL) {          /* user-supplied gzip header */
            Bytef *str;
            if (s->gzhead->extra != Z_NULL)
                wraplen += 2 + s->gzhead->extra_len;
            str = s->gzhead->name;
            if (str != Z_NULL)
                do {
                    wraplen++;
                } while (*str++);
            str = s->gzhead->comment;
            if (str != Z_NULL)
                do {
                    wraplen++;
                } while (*str++);
            if (s->gzhead->hcrc)
                wraplen += 2;
        }
        break;
#endif
    default:                                /* for compiler happiness */
        wraplen = 18;
    }

    /* if not default parameters, return one of the conservative bounds */
    if (s->w_bits != 15 || s->hash_bits != 8 + 7) {
        bound = s->w_bits <= s->hash_bits && s->level ? fixedlen :
                                                        storelen;
        return bound + wraplen < bound ? (z_size_t)-1 : bound + wraplen;
    }

    /* default settings: return tight bound for that case -- ~0.03% overhead
       plus a small constant */
    bound = sourceLen + (sourceLen >> 12) + (sourceLen >> 14) +
            (sourceLen >> 25) + 13 - 6 + wraplen;
    return bound < sourceLen ? (z_size_t)-1 : bound;
}
uLong ZEXPORT deflateBound(z_streamp strm, uLong sourceLen) {
    z_size_t bound = deflateBound_z(strm, sourceLen);
    return (uLong)bound != bound ? (uLong)-1 : (uLong)bound;
}

/* =========================================================================
 * Put a short in the pending buffer. The 16-bit value is put in MSB order.
 * IN assertion: the stream state is correct and there is enough room in
 * pending_buf.
 */
local void putShortMSB(deflate_state *s, uInt b) {
    put_byte(s, (Byte)(b >> 8));
    put_byte(s, (Byte)(b & 0xff));
}

/* =========================================================================
 * Flush as much pending output as possible. All deflate() output, except for
 * some deflate_stored() output, goes through this function so some
 * applications may wish to modify it to avoid allocating a large
 * strm->next_out buffer and copying into it. (See also read_buf()).
 */
local void flush_pending(z_streamp strm) {
    unsigned len;
    deflate_state *s = strm->state;

    _tr_flush_bits(s);
    len = s->pending > strm->avail_out ? strm->avail_out :
                                         (unsigned)s->pending;
    if (len == 0) return;

    zmemcpy(strm->next_out, s->pending_out, len);
    strm->next_out  += len;
    s->pending_out  += len;
    strm->total_out += len;
    strm->avail_out -= len;
    s->pending      -= len;
    if (s->pending == 0) {
        s->pending_out = s->pending_buf;
    }
}

/* ===========================================================================
 * Update the header CRC with the bytes s->pending_buf[beg..s->pending - 1].
 */
#define HCRC_UPDATE(beg) \
    do { \
        if (s->gzhead->hcrc && s->pending > (beg)) \
            strm->adler = crc32_z(strm->adler, s->pending_buf + (beg), \
                                  s->pending - (beg)); \
    } while (0)

/* ========================================================================= */
int ZEXPORT deflate(z_streamp strm, int flush) {
    int old_flush; /* value of flush param for previous deflate call */
    deflate_state *s;

    if (deflateStateCheck(strm) || flush > Z_BLOCK || flush < 0) {
        return Z_STREAM_ERROR;
    }
    s = strm->state;

    if (strm->next_out == Z_NULL ||
        (strm->avail_in != 0 && strm->next_in == Z_NULL) ||
        (s->status == FINISH_STATE && flush != Z_FINISH)) {
        ERR_RETURN(strm, Z_STREAM_ERROR);
    }
    if (strm->avail_out == 0) ERR_RETURN(strm, Z_BUF_ERROR);

    old_flush = s->last_flush;
    s->last_flush = flush;

    /* Flush as much pending output as possible */
    if (s->pending != 0) {
        flush_pending(strm);
        if (strm->avail_out == 0) {
            /* Since avail_out is 0, deflate will be called again with
             * more output space, but possibly with both pending and
             * avail_in equal to zero. There won't be anything to do,
             * but this is not an error situation so make sure we
             * return OK instead of BUF_ERROR at next call of deflate:
             */
            s->last_flush = -1;
            return Z_OK;
        }

    /* Make sure there is something to do and avoid duplicate consecutive
     * flushes. For repeated and useless calls with Z_FINISH, we keep
     * returning Z_STREAM_END instead of Z_BUF_ERROR.
     */
    } else if (strm->avail_in == 0 && RANK(flush) <= RANK(old_flush) &&
               flush != Z_FINISH) {
        ERR_RETURN(strm, Z_BUF_ERROR);
    }

    /* User must not provide more input after the first FINISH: */
    if (s->status == FINISH_STATE && strm->avail_in != 0) {
        ERR_RETURN(strm, Z_BUF_ERROR);
    }

    /* Write the header */
    if (s->status == INIT_STATE && s->wrap == 0)
        s->status = BUSY_STATE;
    if (s->status == INIT_STATE) {
        /* zlib header */
        uInt header = (Z_DEFLATED + ((s->w_bits - 8) << 4)) << 8;
        uInt level_flags;

        if (s->strategy >= Z_HUFFMAN_ONLY || s->level < 2)
            level_flags = 0;
        else if (s->level < 6)
            level_flags = 1;
        else if (s->level == 6)
            level_flags = 2;
        else
            level_flags = 3;
        header |= (level_flags << 6);
        if (s->strstart != 0) header |= PRESET_DICT;
        header += 31 - (header % 31);

        putShortMSB(s, header);

        /* Save the adler32 of the preset dictionary: */
        if (s->strstart != 0) {
            putShortMSB(s, (uInt)(strm->adler >> 16));
            putShortMSB(s, (uInt)(strm->adler & 0xffff));
        }
        strm->adler = adler32(0L, Z_NULL, 0);
        s->status = BUSY_STATE;

        /* Compression must start with an empty pending buffer */
        flush_pending(strm);
        if (s->pending != 0) {
            s->last_flush = -1;
            return Z_OK;
        }
    }
#ifdef GZIP
    if (s->status == GZIP_STATE) {
        /* gzip header */
        strm->adler = crc32(0L, Z_NULL, 0);
        put_byte(s, 31);
        put_byte(s, 139);
        put_byte(s, 8);
        if (s->gzhead == Z_NULL) {
            put_byte(s, 0);
            put_byte(s, 0);
            put_byte(s, 0);
            put_byte(s, 0);
            put_byte(s, 0);
            put_byte(s, s->level == 9 ? 2 :
                     (s->strategy >= Z_HUFFMAN_ONLY || s->level < 2 ?
                      4 : 0));
            put_byte(s, OS_CODE);
            s->status = BUSY_STATE;

            /* Compression must start with an empty pending buffer */
            flush_pending(strm);
            if (s->pending != 0) {
                s->last_flush = -1;
                return Z_OK;
            }
        }
        else {
            put_byte(s, (s->gzhead->text ? 1 : 0) +
                     (s->gzhead->hcrc ? 2 : 0) +
                     (s->gzhead->extra == Z_NULL ? 0 : 4) +
                     (s->gzhead->name == Z_NULL ? 0 : 8) +
                     (s->gzhead->comment == Z_NULL ? 0 : 16)
                     );
            put_byte(s, (Byte)(s->gzhead->time & 0xff));
            put_byte(s, (Byte)((s->gzhead->time >> 8) & 0xff));
            put_byte(s, (Byte)((s->gzhead->time >> 16) & 0xff));
            put_byte(s, (Byte)((s->gzhead->time >> 24) & 0xff));
            put_byte(s, s->level == 9 ? 2 :
                     (s->strategy >= Z_HUFFMAN_ONLY || s->level < 2 ?
                      4 : 0));
            put_byte(s, s->gzhead->os & 0xff);
            if (s->gzhead->extra != Z_NULL) {
                put_byte(s, s->gzhead->extra_len & 0xff);
                put_byte(s, (s->gzhead->extra_len >> 8) & 0xff);
            }
            if (s->gzhead->hcrc)
                strm->adler = crc32_z(strm->adler, s->pending_buf,
                                      s->pending);
            s->gzindex = 0;
            s->status = EXTRA_STATE;
        }
    }
    if (s->status == EXTRA_STATE) {
        if (s->gzhead->extra != Z_NULL) {
            ulg beg = s->pending;   /* start of bytes to update crc */
            ulg left = (s->gzhead->extra_len & 0xffff) - s->gzindex;
            while (s->pending + left > s->pending_buf_size) {
                ulg copy = s->pending_buf_size - s->pending;
                zmemcpy(s->pending_buf + s->pending,
                        s->gzhead->extra + s->gzindex, copy);
                s->pending = s->pending_buf_size;
                HCRC_UPDATE(beg);
                s->gzindex += copy;
                flush_pending(strm);
                if (s->pending != 0) {
                    s->last_flush = -1;
                    return Z_OK;
                }
                beg = 0;
                left -= copy;
            }
            zmemcpy(s->pending_buf + s->pending,
                    s->gzhead->extra + s->gzindex, left);
            s->pending += left;
            HCRC_UPDATE(beg);
            s->gzindex = 0;
        }
        s->status = NAME_STATE;
    }
    if (s->status == NAME_STATE) {
        if (s->gzhead->name != Z_NULL) {
            ulg beg = s->pending;   /* start of bytes to update crc */
            int val;
            do {
                if (s->pending == s->pending_buf_size) {
                    HCRC_UPDATE(beg);
                    flush_pending(strm);
                    if (s->pending != 0) {
                        s->last_flush = -1;
                        return Z_OK;
                    }
                    beg = 0;
                }
                val = s->gzhead->name[s->gzindex++];
                put_byte(s, val);
            } while (val != 0);
            HCRC_UPDATE(beg);
            s->gzindex = 0;
        }
        s->status = COMMENT_STATE;
    }
    if (s->status == COMMENT_STATE) {
        if (s->gzhead->comment != Z_NULL) {
            ulg beg = s->pending;   /* start of bytes to update crc */
            int val;
            do {
                if (s->pending == s->pending_buf_size) {
                    HCRC_UPDATE(beg);
                    flush_pending(strm);
                    if (s->pending != 0) {
                        s->last_flush = -1;
                        return Z_OK;
                    }
                    beg = 0;
                }
                val = s->gzhead->comment[s->gzindex++];
                put_byte(s, val);
            } while (val != 0);
            HCRC_UPDATE(beg);
        }
        s->status = HCRC_STATE;
    }
    if (s->status == HCRC_STATE) {
        if (s->gzhead->hcrc) {
            if (s->pending + 2 > s->pending_buf_size) {
                flush_pending(strm);
                if (s->pending != 0) {
                    s->last_flush = -1;
                    return Z_OK;
                }
            }
            put_byte(s, (Byte)(strm->adler & 0xff));
            put_byte(s, (Byte)((strm->adler >> 8) & 0xff));
            strm->adler = crc32(0L, Z_NULL, 0);
        }
        s->status = BUSY_STATE;

        /* Compression must start with an empty pending buffer */
        flush_pending(strm);
        if (s->pending != 0) {
            s->last_flush = -1;
            return Z_OK;
        }
    }
#endif

    /* Start a new block or continue the current one.
     */
    if (strm->avail_in != 0 || s->lookahead != 0 ||
        (flush != Z_NO_FLUSH && s->status != FINISH_STATE)) {
        block_state bstate;

        bstate = s->level == 0 ? deflate_stored(s, flush) :
                 s->strategy == Z_HUFFMAN_ONLY ? deflate_huff(s, flush) :
                 s->strategy == Z_RLE ? deflate_rle(s, flush) :
                 (*(configuration_table[s->level].func))(s, flush);

        if (bstate == finish_started || bstate == finish_done) {
            s->status = FINISH_STATE;
        }
        if (bstate == need_more || bstate == finish_started) {
            if (strm->avail_out == 0) {
                s->last_flush = -1; /* avoid BUF_ERROR next call, see above */
            }
            return Z_OK;
            /* If flush != Z_NO_FLUSH && avail_out == 0, the next call
             * of deflate should use the same flush parameter to make sure
             * that the flush is complete. So we don't have to output an
             * empty block here, this will be done at next call. This also
             * ensures that for a very small output buffer, we emit at most
             * one empty block.
             */
        }
        if (bstate == block_done) {
            if (flush == Z_PARTIAL_FLUSH) {
                _tr_align(s);
            } else if (flush != Z_BLOCK) { /* FULL_FLUSH or SYNC_FLUSH */
                _tr_stored_block(s, (char*)0, 0L, 0);
                /* For a full flush, this empty block will be recognized
                 * as a special marker by inflate_sync().
                 */
                if (flush == Z_FULL_FLUSH) {
#ifdef Z_TURBO_THREADS
                    Z_U8 at = s->mt_abs + s->strstart;
#endif
                    CLEAR_HASH(s);             /* forget history */
                    if (s->lookahead == 0) {
                        s->strstart = 0;
                        s->block_start = 0L;
                        s->insert = 0;
                    }
#ifdef Z_TURBO_THREADS
                    mt_restart(s, at, 0);      /* history starts here */
#endif
                }
            }
            flush_pending(strm);
            if (strm->avail_out == 0) {
              s->last_flush = -1; /* avoid BUF_ERROR at next call, see above */
              return Z_OK;
            }
        }
    }

    if (flush != Z_FINISH) return Z_OK;
    if (s->wrap <= 0) return Z_STREAM_END;

    /* Write the trailer */
#ifdef GZIP
    if (s->wrap == 2) {
        put_byte(s, (Byte)(strm->adler & 0xff));
        put_byte(s, (Byte)((strm->adler >> 8) & 0xff));
        put_byte(s, (Byte)((strm->adler >> 16) & 0xff));
        put_byte(s, (Byte)((strm->adler >> 24) & 0xff));
        put_byte(s, (Byte)(strm->total_in & 0xff));
        put_byte(s, (Byte)((strm->total_in >> 8) & 0xff));
        put_byte(s, (Byte)((strm->total_in >> 16) & 0xff));
        put_byte(s, (Byte)((strm->total_in >> 24) & 0xff));
    }
    else
#endif
    {
        putShortMSB(s, (uInt)(strm->adler >> 16));
        putShortMSB(s, (uInt)(strm->adler & 0xffff));
    }
    flush_pending(strm);
    /* If avail_out is zero, the application will call deflate again
     * to flush the rest.
     */
    if (s->wrap > 0) s->wrap = -s->wrap; /* write the trailer only once! */
    return s->pending != 0 ? Z_OK : Z_STREAM_END;
}

/* ========================================================================= */
int ZEXPORT deflateEnd(z_streamp strm) {
    int status;

    if (deflateStateCheck(strm)) return Z_STREAM_ERROR;

    status = strm->state->status;

    /* Deallocate in reverse order of allocations: */
#ifdef Z_TURBO_THREADS
    mt_free(strm, strm->state);
#endif
#ifdef Z_TURBO_OPT
    opt_free(strm, strm->state);
#endif
    TRY_FREE(strm, strm->state->pending_buf);
    TRY_FREE(strm, strm->state->head);
    TRY_FREE(strm, strm->state->prev);
    TRY_FREE(strm, strm->state->window);

    ZFREE(strm, strm->state);
    strm->state = Z_NULL;

    return status == BUSY_STATE ? Z_DATA_ERROR : Z_OK;
}

/* =========================================================================
 * Copy the source state to the destination state.
 * To simplify the source, this is not supported for 16-bit MSDOS (which
 * doesn't have enough memory anyway to duplicate compression states).
 */
int ZEXPORT deflateCopy(z_streamp dest, z_streamp source) {
#ifdef MAXSEG_64K
    (void)dest;
    (void)source;
    return Z_STREAM_ERROR;
#else
    deflate_state *ds;
    deflate_state *ss;


    if (deflateStateCheck(source) || dest == Z_NULL) {
        return Z_STREAM_ERROR;
    }

    ss = source->state;

    zmemcpy(dest, source, sizeof(z_stream));

    ds = (deflate_state *) ZALLOC(dest, 1, sizeof(deflate_state));
    if (ds == Z_NULL) return Z_MEM_ERROR;
    zmemzero(ds, sizeof(deflate_state));
    dest->state = (struct internal_state FAR *) ds;
    zmemcpy(ds, ss, sizeof(deflate_state));
    ds->strm = dest;
#ifdef Z_TURBO_THREADS
    ds->mt = Z_NULL;
#endif
#ifdef Z_TURBO_OPT
    ds->opt = Z_NULL;
#endif

    ds->window = (Bytef *) ZALLOC(dest, ds->w_size, 2*sizeof(Byte));
    ds->prev   = (Posf *)  ZALLOC(dest, ds->w_size, sizeof(Pos));
    ds->head   = (Posf *)  ZALLOC(dest, ds->hash_size, sizeof(Pos));
    ds->pending_buf = (uchf *) ZALLOC(dest, ds->lit_bufsize, LIT_BUFS);

    if (ds->window == Z_NULL || ds->prev == Z_NULL || ds->head == Z_NULL ||
        ds->pending_buf == Z_NULL) {
        deflateEnd (dest);
        return Z_MEM_ERROR;
    }
    /* following zmemcpy's do not work for 16-bit MSDOS */
    zmemcpy(ds->window, ss->window, ss->high_water);
    zmemcpy(ds->prev, ss->prev,
            (ss->slid || ss->strstart - ss->insert > ds->w_size ? ds->w_size :
                ss->strstart - ss->insert) * sizeof(Pos));
    zmemcpy(ds->head, ss->head, ds->hash_size * sizeof(Pos));

    ds->pending_out = ds->pending_buf + (ss->pending_out - ss->pending_buf);
#ifdef Z_TURBO_OPT
    if (ss->opt == Z_NULL)
#endif
    zmemcpy(ds->pending_out, ss->pending_out, ss->pending);
#ifdef LIT_MEM
    ds->d_buf = (ushf *)(ds->pending_buf + (ds->lit_bufsize << 1));
    ds->l_buf = ds->pending_buf + (ds->lit_bufsize << 2);
    zmemcpy(ds->d_buf, ss->d_buf, ss->sym_next * sizeof(ush));
    zmemcpy(ds->l_buf, ss->l_buf, ss->sym_next);
#else
    ds->sym_buf = ds->pending_buf + ds->lit_bufsize;
    zmemcpy(ds->sym_buf, ss->sym_buf, ss->sym_next);
#endif
#ifdef Z_TURBO_OPT
    if (ss->opt != Z_NULL) {            /* its own pending_buf */
        if (opt_alloc(dest, ds) != Z_OK) {
            deflateEnd (dest);
            return Z_MEM_ERROR;
        }
        zmemcpy(ds->opt->in, ss->opt->in, ss->opt->len);
        zmemcpy(ds->opt->child, ss->opt->child,
                2 * ss->opt->len * sizeof(unsigned));
        zmemcpy(ds->opt->h4, ss->opt->h4, (1U << OPT_H4_BITS) *
                                          sizeof(unsigned));
        zmemcpy(ds->opt->h3, ss->opt->h3, (1U << OPT_H3_BITS) *
                                          sizeof(unsigned));
        ds->opt->abase = ss->opt->abase;
        ds->opt->ins = ss->opt->ins;
        ds->opt->len = ss->opt->len;
        ds->opt->start = ss->opt->start;
        ds->opt->fresh = ss->opt->fresh;
        ds->pending_out = ds->pending_buf +
                          (ss->pending_out - ss->pending_buf);
        zmemcpy(ds->pending_out, ss->pending_out, ss->pending);
    }
#endif

    ds->l_desc.dyn_tree = ds->dyn_ltree;
    ds->d_desc.dyn_tree = ds->dyn_dtree;
    ds->bl_desc.dyn_tree = ds->bl_tree;

    return Z_OK;
#endif /* MAXSEG_64K */
}

#ifndef FASTEST
/* ===========================================================================
 * Set match_start to the longest match starting at the given string and
 * return its length. Matches shorter or equal to prev_length are discarded,
 * in which case the result is equal to prev_length and match_start is
 * garbage.
 * IN assertions: cur_match is the head of the hash chain for the current
 *   string (strstart) and its distance is <= MAX_DIST, and prev_length >= 1
 * OUT assertion: the match length is not greater than s->lookahead.
 */
#if defined(Z_U8) && !defined(UNALIGNED_OK)
#  if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
#    include <intrin.h>
#  endif
/* The two bytes at p as an unsigned short, in memory order. */
local unsigned short load16(const Bytef *p) {
    unsigned short v;

    zmemcpy(&v, p, 2);
    return v;
}

/* Return the number of leading bytes on which a and b agree, at most
   MAX_MATCH - 2, comparing eight bytes at a time. */
local unsigned count_match(const Bytef *a, const Bytef *b) {
    unsigned n = 0;
    Z_U8 x, y;

    do {
        zmemcpy(&x, a + n, 8);
        zmemcpy(&y, b + n, 8);
        if (x != y) {
#  if defined(__GNUC__) && defined(__BYTE_ORDER__) && \
      __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            return n + ((unsigned)__builtin_ctzll(x ^ y) >> 3);
#  elif defined(__GNUC__) && defined(__BYTE_ORDER__) && \
        __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            return n + ((unsigned)__builtin_clzll(x ^ y) >> 3);
#  elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
            unsigned long i;
            _BitScanForward64(&i, x ^ y);
            return n + (unsigned)(i >> 3);
#  else
            while (a[n] == b[n])
                n++;
            return n;
#  endif
        }
        n += 8;
    } while (n < MAX_MATCH - 2);
    return n;
}
#endif

local uInt longest_match(deflate_state *s, IPos cur_match) {
    unsigned chain_length = s->max_chain_length;/* max hash chain length */
    Bytef *scan = s->window + s->strstart;      /* current string */
    Bytef *match;                               /* matched string */
    int len;                                    /* length of current match */
    int best_len = (int)s->prev_length;         /* best match length so far */
    int nice_match = s->nice_match;             /* stop if match long enough */
    IPos limit = s->strstart > (IPos)MAX_DIST(s) ?
        s->strstart - (IPos)MAX_DIST(s) : NIL;
    /* Stop when cur_match becomes <= limit. To simplify the code,
     * we prevent matches with the string of window index 0.
     */
    Posf *prev = s->prev;
    uInt wmask = s->w_mask;

#ifdef UNALIGNED_OK
    /* Compare two bytes at a time. Note: this is not always beneficial.
     * Try with and without -DUNALIGNED_OK to check.
     */
    Bytef *strend = s->window + s->strstart + MAX_MATCH - 1;
    ush scan_start = *(ushf*)scan;
    ush scan_end   = *(ushf*)(scan + best_len - 1);
#else
#  ifdef Z_U8
    unsigned short scan_start = load16(scan);
    unsigned short scan_end = load16(scan + best_len - 1);
#  else
    Bytef *strend = s->window + s->strstart + MAX_MATCH;
    Byte scan_end1  = scan[best_len - 1];
    Byte scan_end   = scan[best_len];
#  endif
#endif

    /* The code is optimized for HASH_BITS >= 8 and MAX_MATCH-2 multiple of 16.
     * It is easy to get rid of this optimization if necessary.
     */
    Assert(s->hash_bits >= 8 && MAX_MATCH == 258, "Code too clever");

    /* Do not waste too much time if we already have a good match: */
    if (s->prev_length >= s->good_match) {
        chain_length >>= 2;
    }
    /* Do not look for matches beyond the end of the input. This is necessary
     * to make deflate deterministic.
     */
    if ((uInt)nice_match > s->lookahead) nice_match = (int)s->lookahead;

    Assert((ulg)s->strstart <= s->window_size - MIN_LOOKAHEAD,
           "need lookahead");

    do {
        Assert(cur_match < s->strstart, "no future");
        match = s->window + cur_match;

        /* Skip to next match if the match length cannot increase
         * or if the match length is less than 2.  Note that the checks below
         * for insufficient lookahead only occur occasionally for performance
         * reasons.  Therefore uninitialized memory will be accessed, and
         * conditional jumps will be made that depend on those values.
         * However the length of the match is limited to the lookahead, so
         * the output of deflate is not affected by the uninitialized values.
         */
#if (defined(UNALIGNED_OK) && MAX_MATCH == 258)
        /* This code assumes sizeof(unsigned short) == 2. Do not use
         * UNALIGNED_OK if your compiler uses a different size.
         */
        if (*(ushf*)(match + best_len - 1) != scan_end ||
            *(ushf*)match != scan_start) continue;

        /* It is not necessary to compare scan[2] and match[2] since they are
         * always equal when the other bytes match, given that the hash keys
         * are equal and that HASH_BITS >= 8. Compare 2 bytes at a time at
         * strstart + 3, + 5, up to strstart + 257. We check for insufficient
         * lookahead only every 4th comparison; the 128th check will be made
         * at strstart + 257. If MAX_MATCH-2 is not a multiple of 8, it is
         * necessary to put more guard bytes at the end of the window, or
         * to check more often for insufficient lookahead.
         */
        Assert(scan[2] == match[2], "scan[2]?");
        scan++, match++;
        do {
        } while (*(ushf*)(scan += 2) == *(ushf*)(match += 2) &&
                 *(ushf*)(scan += 2) == *(ushf*)(match += 2) &&
                 *(ushf*)(scan += 2) == *(ushf*)(match += 2) &&
                 *(ushf*)(scan += 2) == *(ushf*)(match += 2) &&
                 scan < strend);
        /* The funny "do {}" generates better code on most compilers */

        /* Here, scan <= window + strstart + 257 */
        Assert(scan <= s->window + (unsigned)(s->window_size - 1),
               "wild scan");
        if (*scan == *match) scan++;

        len = (MAX_MATCH - 1) - (int)(strend - scan);
        scan = strend - (MAX_MATCH-1);

#else /* UNALIGNED_OK */

#  ifdef Z_U8
        /* Check bytes best_len - 1 and best_len, and 0 and 1, two at a time.
         * scan[2] and match[2] are equal when the other bytes match, given
         * that the hash keys are equal and that HASH_BITS >= 8. Then compare
         * up to strstart + 257, eight bytes at a time.
         */
        if (load16(match + best_len - 1) != scan_end ||
            load16(match) != scan_start) continue;
        len = 2 + (int)count_match(scan + 2, match + 2);
#  else
        if (match[best_len]     != scan_end  ||
            match[best_len - 1] != scan_end1 ||
            *match              != *scan     ||
            *++match            != scan[1])      continue;

        /* The check at best_len - 1 can be removed because it will be made
         * again later. (This heuristic is not always a win.)
         * It is not necessary to compare scan[2] and match[2] since they
         * are always equal when the other bytes match, given that
         * the hash keys are equal and that HASH_BITS >= 8.
         */
        scan += 2, match++;
        Assert(*scan == *match, "match[2]?");

        /* We check for insufficient lookahead only every 8th comparison;
         * the 256th check will be made at strstart + 258.
         */
        do {
        } while (*++scan == *++match && *++scan == *++match &&
                 *++scan == *++match && *++scan == *++match &&
                 *++scan == *++match && *++scan == *++match &&
                 *++scan == *++match && *++scan == *++match &&
                 scan < strend);

        Assert(scan <= s->window + (unsigned)(s->window_size - 1),
               "wild scan");

        len = MAX_MATCH - (int)(strend - scan);
        scan = strend - MAX_MATCH;
#  endif

#endif /* UNALIGNED_OK */

        if (len > best_len) {
            s->match_start = cur_match;
            best_len = len;
            if (len >= nice_match) break;
#ifdef UNALIGNED_OK
            scan_end = *(ushf*)(scan + best_len - 1);
#elif defined(Z_U8)
            scan_end = load16(scan + best_len - 1);
#else
            scan_end1  = scan[best_len - 1];
            scan_end   = scan[best_len];
#endif
        }
    } while ((cur_match = prev[cur_match & wmask]) > limit
             && --chain_length != 0);

    if ((uInt)best_len <= s->lookahead) return (uInt)best_len;
    return s->lookahead;
}

#else /* FASTEST */

/* ---------------------------------------------------------------------------
 * Optimized version for FASTEST only
 */
local uInt longest_match(deflate_state *s, IPos cur_match) {
    Bytef *scan = s->window + s->strstart;      /* current string */
    Bytef *match;                               /* matched string */
    int len;                                    /* length of current match */
    Bytef *strend = s->window + s->strstart + MAX_MATCH;

    /* The code is optimized for HASH_BITS >= 8 and MAX_MATCH-2 multiple of 16.
     * It is easy to get rid of this optimization if necessary.
     */
    Assert(s->hash_bits >= 8 && MAX_MATCH == 258, "Code too clever");

    Assert((ulg)s->strstart <= s->window_size - MIN_LOOKAHEAD,
           "need lookahead");

    Assert(cur_match < s->strstart, "no future");

    match = s->window + cur_match;

    /* Return failure if the match length is less than 2:
     */
    if (match[0] != scan[0] || match[1] != scan[1]) return MIN_MATCH-1;

    /* The check at best_len - 1 can be removed because it will be made
     * again later. (This heuristic is not always a win.)
     * It is not necessary to compare scan[2] and match[2] since they
     * are always equal when the other bytes match, given that
     * the hash keys are equal and that HASH_BITS >= 8.
     */
    scan += 2, match += 2;
    Assert(*scan == *match, "match[2]?");

    /* We check for insufficient lookahead only every 8th comparison;
     * the 256th check will be made at strstart + 258.
     */
    do {
    } while (*++scan == *++match && *++scan == *++match &&
             *++scan == *++match && *++scan == *++match &&
             *++scan == *++match && *++scan == *++match &&
             *++scan == *++match && *++scan == *++match &&
             scan < strend);

    Assert(scan <= s->window + (unsigned)(s->window_size - 1), "wild scan");

    len = MAX_MATCH - (int)(strend - scan);

    if (len < MIN_MATCH) return MIN_MATCH - 1;

    s->match_start = cur_match;
    return (uInt)len <= s->lookahead ? (uInt)len : s->lookahead;
}

#endif /* FASTEST */

#ifdef ZLIB_DEBUG

#define EQUAL 0
/* result of memcmp for equal strings */

/* ===========================================================================
 * Check that the match at match_start is indeed a match.
 */
local void check_match(deflate_state *s, IPos start, IPos match, int length) {
    /* check that the match is indeed a match */
    Bytef *back = s->window + (int)match, *here = s->window + start;
    IPos len = (IPos)length;
    if (match == (IPos)-1) {
        /* match starts one byte before the current window -- just compare the
           subsequent length-1 bytes */
        back++;
        here++;
        len--;
    }
    if (zmemcmp(back, here, len) != EQUAL) {
        fprintf(stderr, " start %u, match %d, length %d\n",
                start, (int)match, length);
        do {
            fprintf(stderr, "(%02x %02x)", *back++, *here++);
        } while (--len != 0);
        z_error("invalid match");
    }
    if (z_verbose > 1) {
        fprintf(stderr,"\\[%d,%d]", start - match, length);
        do { putc(s->window[start++], stderr); } while (--length != 0);
    }
}
#else
#  define check_match(s, start, match, length)
#endif /* ZLIB_DEBUG */

/* ===========================================================================
 * Flush the current block, with given end-of-file flag.
 * IN assertion: strstart is set to the end of the current match.
 */
#define FLUSH_BLOCK_ONLY(s, last) { \
   _tr_flush_block(s, (s->block_start >= 0L ? \
                   (charf *)&s->window[(unsigned)s->block_start] : \
                   (charf *)Z_NULL), \
                (ulg)((long)s->strstart - s->block_start), \
                (last)); \
   s->block_start = s->strstart; \
   flush_pending(s->strm); \
   Tracev((stderr,"[FLUSH]")); \
}

/* Same but force premature exit if necessary. */
#define FLUSH_BLOCK(s, last) { \
   FLUSH_BLOCK_ONLY(s, last); \
   if (s->strm->avail_out == 0) return (last) ? finish_started : need_more; \
}

/* Maximum stored block length in deflate format (not including header). */
#define MAX_STORED 65535

/* Minimum of a and b. */
#define MIN(a, b) ((a) > (b) ? (b) : (a))

/* ===========================================================================
 * Copy without compression as much as possible from the input stream, return
 * the current block state.
 *
 * In case deflateParams() is used to later switch to a non-zero compression
 * level, s->matches (otherwise unused when storing) keeps track of the number
 * of hash table slides to perform. If s->matches is 1, then one hash table
 * slide will be done when switching. If s->matches is 2, the maximum value
 * allowed here, then the hash table will be cleared, since two or more slides
 * is the same as a clear.
 *
 * deflate_stored() is written to minimize the number of times an input byte is
 * copied. It is most efficient with large input and output buffers, which
 * maximizes the opportunities to have a single copy from next_in to next_out.
 */
local block_state deflate_stored(deflate_state *s, int flush) {
    /* Smallest worthy block size when not flushing or finishing. By default
     * this is 32K. This can be as small as 507 bytes for memLevel == 1. For
     * large input and output buffers, the stored block size will be larger.
     */
    unsigned min_block = (unsigned)(MIN(s->pending_buf_size - 5, s->w_size));

    /* Copy as many min_block or larger stored blocks directly to next_out as
     * possible. If flushing, copy the remaining available input to next_out as
     * stored blocks, if there is enough space.
     */
    int last = 0;
    unsigned len, left, have;
    unsigned used = s->strm->avail_in;
#ifdef Z_TURBO_THREADS
    s->mt_exact = 0;            /* this moves the window its own way */
#endif
    do {
        /* Set len to the maximum size block that we can copy directly with the
         * available input data and output space. Set left to how much of that
         * would be copied from what's left in the window.
         */
        len = MAX_STORED;       /* maximum deflate stored block length */
        have = ((unsigned)s->bi_valid + 42) >> 3;   /* bytes in header */
        if (s->strm->avail_out < have)          /* need room for header */
            break;
            /* maximum stored block length that will fit in avail_out: */
        have = s->strm->avail_out - have;
        left = (unsigned)(s->strstart - s->block_start);    /* window bytes */
        if (len > (ulg)left + s->strm->avail_in)
            len = left + s->strm->avail_in;     /* limit len to the input */
        if (len > have)
            len = have;                         /* limit len to the output */

        /* If the stored block would be less than min_block in length, or if
         * unable to copy all of the available input when flushing, then try
         * copying to the window and the pending buffer instead. Also don't
         * write an empty block when flushing -- deflate() does that.
         */
        if (len < min_block && ((len == 0 && flush != Z_FINISH) ||
                                flush == Z_NO_FLUSH ||
                                len != left + s->strm->avail_in))
            break;

        /* Make a dummy stored block in pending to get the header bytes,
         * including any pending bits. This also updates the debugging counts.
         */
        last = flush == Z_FINISH && len == left + s->strm->avail_in ? 1 : 0;
        _tr_stored_block(s, (char *)0, 0L, last);

        /* Replace the lengths in the dummy stored block with len. */
        s->pending_buf[s->pending - 4] = (Bytef)len;
        s->pending_buf[s->pending - 3] = (Bytef)(len >> 8);
        s->pending_buf[s->pending - 2] = (Bytef)~len;
        s->pending_buf[s->pending - 1] = (Bytef)(~len >> 8);

        /* Write the stored block header bytes. */
        flush_pending(s->strm);

#ifdef ZLIB_DEBUG
        /* Update debugging counts for the data about to be copied. */
        s->compressed_len += len << 3;
        s->bits_sent += len << 3;
#endif

        /* Copy uncompressed bytes from the window to next_out. */
        if (left) {
            if (left > len)
                left = len;
            zmemcpy(s->strm->next_out, s->window + s->block_start, left);
            s->strm->next_out += left;
            s->strm->avail_out -= left;
            s->strm->total_out += left;
            s->block_start += left;
            len -= left;
        }

        /* Copy uncompressed bytes directly from next_in to next_out, updating
         * the check value.
         */
        if (len) {
            read_buf(s->strm, s->strm->next_out, len);
            s->strm->next_out += len;
            s->strm->avail_out -= len;
            s->strm->total_out += len;
        }
    } while (last == 0);

    /* Update the sliding window with the last s->w_size bytes of the copied
     * data, or append all of the copied data to the existing window if less
     * than s->w_size bytes were copied. Also update the number of bytes to
     * insert in the hash tables, in the event that deflateParams() switches to
     * a non-zero compression level.
     */
    used -= s->strm->avail_in;      /* number of input bytes directly copied */
    if (used) {
        /* If any input was used, then no unused input remains in the window,
         * therefore s->block_start == s->strstart.
         */
        if (used >= s->w_size) {    /* supplant the previous history */
            s->matches = 2;         /* clear hash */
            zmemcpy(s->window, s->strm->next_in - s->w_size, s->w_size);
            s->strstart = s->w_size;
            s->insert = s->strstart;
        }
        else {
            if (s->window_size - s->strstart <= used) {
                /* Slide the window down. */
                s->strstart -= s->w_size;
                zmemcpy(s->window, s->window + s->w_size, s->strstart);
                if (s->matches < 2)
                    s->matches++;   /* add a pending slide_hash() */
                if (s->insert > s->strstart)
                    s->insert = s->strstart;
            }
            zmemcpy(s->window + s->strstart, s->strm->next_in - used, used);
            s->strstart += used;
            s->insert += MIN(used, s->w_size - s->insert);
        }
        s->block_start = s->strstart;
    }
    if (s->high_water < s->strstart)
        s->high_water = s->strstart;

    /* If the last block was written to next_out, then done. */
    if (last) {
        s->bi_used = 8;
        return finish_done;
    }

    /* If flushing and all input has been consumed, then done. */
    if (flush != Z_NO_FLUSH && flush != Z_FINISH &&
        s->strm->avail_in == 0 && (long)s->strstart == s->block_start)
        return block_done;

    /* Fill the window with any remaining input. */
    have = (unsigned)(s->window_size - s->strstart);
    if (s->strm->avail_in > have && s->block_start >= (long)s->w_size) {
        /* Slide the window down. */
        s->block_start -= s->w_size;
        s->strstart -= s->w_size;
        zmemcpy(s->window, s->window + s->w_size, s->strstart);
        if (s->matches < 2)
            s->matches++;           /* add a pending slide_hash() */
        have += s->w_size;          /* more space now */
        if (s->insert > s->strstart)
            s->insert = s->strstart;
    }
    if (have > s->strm->avail_in)
        have = s->strm->avail_in;
    if (have) {
        read_buf(s->strm, s->window + s->strstart, have);
        s->strstart += have;
        s->insert += MIN(have, s->w_size - s->insert);
    }
    if (s->high_water < s->strstart)
        s->high_water = s->strstart;

    /* There was not enough avail_out to write a complete worthy or flushed
     * stored block to next_out. Write a stored block to pending instead, if we
     * have enough input for a worthy block, or if flushing and there is enough
     * room for the remaining input as a stored block in the pending buffer.
     */
    have = ((unsigned)s->bi_valid + 42) >> 3;   /* bytes in header */
        /* maximum stored block length that will fit in pending: */
    have = (unsigned)MIN(s->pending_buf_size - have, MAX_STORED);
    min_block = MIN(have, s->w_size);
    left = (unsigned)(s->strstart - s->block_start);
    if (left >= min_block ||
        ((left || flush == Z_FINISH) && flush != Z_NO_FLUSH &&
         s->strm->avail_in == 0 && left <= have)) {
        len = MIN(left, have);
        last = flush == Z_FINISH && s->strm->avail_in == 0 &&
               len == left ? 1 : 0;
        _tr_stored_block(s, (charf *)s->window + s->block_start, len, last);
        s->block_start += len;
        flush_pending(s->strm);
    }

    /* We've done all we can with the available input and output. */
    if (last)
        s->bi_used = 8;
    return last ? finish_started : need_more;
}

/* ===========================================================================
 * Compress as much as possible from the input stream, return the current
 * block state.
 * This function does not perform lazy evaluation of matches and inserts
 * new strings in the dictionary only for unmatched strings or for short
 * matches. It is used only for the fast compression options.
 */
local block_state deflate_fast(deflate_state *s, int flush) {
    IPos hash_head;       /* head of the hash chain */
    int bflush;           /* set if current block must be flushed */

    for (;;) {
        /* Make sure that we always have enough lookahead, except
         * at the end of the input file. We need MAX_MATCH bytes
         * for the next match, plus MIN_MATCH bytes to insert the
         * string following the next match.
         */
        if (s->lookahead < MIN_LOOKAHEAD) {
            fill_window(s);
            if (s->lookahead < MIN_LOOKAHEAD && flush == Z_NO_FLUSH) {
                return need_more;
            }
            if (s->lookahead == 0) break; /* flush the current block */
        }

        /* Insert the string window[strstart .. strstart + 2] in the
         * dictionary, and set hash_head to the head of the hash chain:
         */
        hash_head = NIL;
        if (s->lookahead >= MIN_MATCH) {
            INSERT_STRING(s, s->strstart, hash_head);
        }

        /* Find the longest match, discarding those <= prev_length.
         * At this point we have always match_length < MIN_MATCH
         */
        if (hash_head != NIL && s->strstart - hash_head <= MAX_DIST(s)) {
            /* To simplify the code, we prevent matches with the string
             * of window index 0 (in particular we have to avoid a match
             * of the string with itself at the start of the input file).
             */
            s->match_length = longest_match (s, hash_head);
            /* longest_match() sets match_start */
        }
        if (s->match_length >= MIN_MATCH) {
            check_match(s, s->strstart, s->match_start, (int)s->match_length);

            _tr_tally_dist(s, s->strstart - s->match_start,
                           s->match_length - MIN_MATCH, bflush);

            s->lookahead -= s->match_length;

            /* Insert new strings in the hash table only if the match length
             * is not too large. This saves time but degrades compression.
             */
#ifndef FASTEST
            if (s->match_length <= s->max_insert_length &&
                s->lookahead >= MIN_MATCH) {
                s->match_length--; /* string at strstart already in table */
                do {
                    s->strstart++;
                    INSERT_STRING(s, s->strstart, hash_head);
                    /* strstart never exceeds WSIZE-MAX_MATCH, so there are
                     * always MIN_MATCH bytes ahead.
                     */
                } while (--s->match_length != 0);
                s->strstart++;
            } else
#endif
            {
                s->strstart += s->match_length;
                s->match_length = 0;
                s->ins_h = s->window[s->strstart];
                UPDATE_HASH(s, s->ins_h, s->window[s->strstart + 1]);
#if MIN_MATCH != 3
                Call UPDATE_HASH() MIN_MATCH-3 more times
#endif
                /* If lookahead < MIN_MATCH, ins_h is garbage, but it does not
                 * matter since it will be recomputed at next deflate call.
                 */
            }
        } else {
            /* No match, output a literal byte */
            Tracevv((stderr,"%c", s->window[s->strstart]));
            _tr_tally_lit(s, s->window[s->strstart], bflush);
            s->lookahead--;
            s->strstart++;
        }
        if (bflush) FLUSH_BLOCK(s, 0);
    }
    s->insert = s->strstart < MIN_MATCH-1 ? s->strstart : MIN_MATCH-1;
    if (flush == Z_FINISH) {
        FLUSH_BLOCK(s, 1);
        return finish_done;
    }
    if (s->sym_next)
        FLUSH_BLOCK(s, 0);
    return block_done;
}

#ifndef FASTEST
/* ===========================================================================
 * zlib-turbo: parallel match search for deflate_slow() (levels 4 to 9), with
 * the same output as one thread. Built with -DZ_TURBO_THREADS (POSIX threads)
 * and enabled per stream with deflateSetThreads() in zturbo.h.
 *
 * Why the output can be the same: deflate_slow() inserts every position into
 * the hash chains, so the chain at a position holds all earlier positions
 * with the same hash, whatever matches were chosen before. longest_match()
 * then depends on the window and on two things the parse decides: prev_length,
 * which only filters out matches that are not longer, and whether prev_length
 * reached good_match, which cuts the chain to a quarter. So the longest match
 * at a position over the whole chain, or over its first quarter, is a fact of
 * the input, and a worker can find it ahead of the parse.
 *
 * To search the positions the parse will search, and with the chain length it
 * will use, each worker runs deflate_slow()'s parse over its chunk of input,
 * starting a little before it. The parse resynchronizes quickly, after which
 * the worker searches exactly what the real parse will ask for. The real parse,
 * unchanged, then takes each result whose chain length is the one it needs, and
 * searches itself when there is none (before resynchronization, and at
 * positions whose search could see the end of the available input).
 *
 * Positions are absolute: s->mt_abs is the stream position of window[0]. Each
 * chunk has its own copy of the bytes it needs (MAX_DIST and the warm-up before
 * it, MAX_MATCH after it) and its own hash chains, built from them. A chain cut
 * at the start of the copy only drops positions too far back to be matched.
 */
#ifdef Z_TURBO_THREADS

/* Threads: Windows' own (Vista or later), or POSIX threads elsewhere. */
local void *mt_worker(void *arg);
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0600
#    undef _WIN32_WINNT
#    define _WIN32_WINNT 0x0600
#  endif
#  undef Freq                   /* deflate.h's names, used by windows.h */
#  undef Code
#  undef Dad
#  undef Len
#  include <windows.h>
#  include <process.h>
#  define Freq fc.freq
#  define Code fc.code
#  define Dad  dl.dad
#  define Len  dl.len
typedef SRWLOCK mt_mutex;
typedef CONDITION_VARIABLE mt_cond;
typedef HANDLE mt_thread;
#  define mt_mutex_init(m) InitializeSRWLock(m)
#  define mt_mutex_destroy(m) ((void)(m))
#  define mt_lock(m) AcquireSRWLockExclusive(m)
#  define mt_unlock(m) ReleaseSRWLockExclusive(m)
#  define mt_cond_init(c) InitializeConditionVariable(c)
#  define mt_cond_destroy(c) ((void)(c))
#  define mt_wait(c, m) ((void)SleepConditionVariableSRW(c, m, INFINITE, 0))
#  define mt_signal(c) WakeConditionVariable(c)
#  define mt_broadcast(c) WakeAllConditionVariable(c)
local unsigned __stdcall mt_start(void *arg) {
    mt_worker(arg);
    return 0;
}
/* Start a thread running mt_worker(arg): 0 on success. */
local int mt_create(mt_thread *t, void *arg) {
    uintptr_t h = _beginthreadex(NULL, 0, mt_start, arg, 0, NULL);
    *t = (HANDLE)h;
    return h == 0;
}
local void mt_join(mt_thread t) {
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
}
#else
#  include <pthread.h>
typedef pthread_mutex_t mt_mutex;
typedef pthread_cond_t mt_cond;
typedef pthread_t mt_thread;
#  define mt_mutex_init(m) pthread_mutex_init(m, NULL)
#  define mt_mutex_destroy(m) pthread_mutex_destroy(m)
#  define mt_lock(m) pthread_mutex_lock(m)
#  define mt_unlock(m) pthread_mutex_unlock(m)
#  define mt_cond_init(c) pthread_cond_init(c, NULL)
#  define mt_cond_destroy(c) pthread_cond_destroy(c)
#  define mt_wait(c, m) pthread_cond_wait(c, m)
#  define mt_signal(c) pthread_cond_signal(c)
#  define mt_broadcast(c) pthread_cond_broadcast(c)
#  define mt_create(t, arg) pthread_create(t, NULL, mt_worker, arg)
#  define mt_join(t) pthread_join(t, NULL)
#endif

#define MT_CHUNK 262144U        /* positions searched per task */
#define MT_WARM 4096U           /* positions parsed before a chunk to resynchronize */
#define MT_FREE 0
#define MT_QUEUED 1
#define MT_RUNNING 2
#define MT_DONE 3
#define MT_COPY_NEED 0          /* a slot's input bytes: not copied yet, */
#define MT_COPY_BUSY 1          /* being copied, */
#define MT_COPY_DONE 2          /* or in its data */

/* A search at a position: len is the longest match length (2 if none) plus
   MT_FULL or MT_PART for the chain length used, or 0 if not searched. */
#define MT_FULL 0x1000
#define MT_PART 0x2000
typedef struct {
    ush len;
    ush dist;
} mt_res;

typedef struct {
    Z_U8 start;                 /* stream position of the first position */
    Z_U8 data0;                 /* stream position of data[0] */
    Z_U8 floor;                 /* the first position the chains can hold */
    unsigned hist;              /* bytes of data before start */
    unsigned n;                 /* positions searched */
    unsigned size;              /* bytes in data */
    int state;
    Bytef *data;
    const Bytef *src;           /* input bytes still to copy into data */
    unsigned src_at, src_len;   /* where in data, and how many */
    int copy;                   /* MT_COPY_* state of those bytes */
    ush *link;                  /* distance back to the previous position with
                                   the same hash, or 0 for none or too far */
    mt_res *res;                /* the searches at the n positions */
} mt_slot;

struct mt_arg_s;

struct mt_engine_s {
    int threads;
    unsigned nslots;
    mt_slot *slot;
    unsigned first;             /* oldest slot in use */
    unsigned used;              /* slots in use, from first on */
    int ready;                  /* the parse has seen slot[first] done */
    Z_U8 next;                  /* stream position where the next chunk starts */
    /* the parameters of the parse and search, as when the chunks were made */
    unsigned max_chain, good, lazy;
    int nice, strategy;
    unsigned hash_bits, hash_shift, hash_mask, max_dist;
    int **headtab;              /* one hash table per worker, and the parse */
    struct mt_arg_s *args;      /* worker arguments */
    mt_thread *tid;
    mt_mutex lock;
    mt_cond work;
    mt_cond done;
    int quit;
    int started;                /* worker threads running */
    /* block encoding on a worker, while the parse goes on (mt_flush_block) */
    deflate_state *es;          /* writes the block in flight */
    z_stream es_strm;           /* for es->strm->data_type */
    int enc;                    /* MT_FREE, MT_QUEUED, MT_RUNNING or MT_DONE */
    int in_flight;              /* enc is not MT_FREE (the parse's copy) */
    charf *enc_buf;             /* the block's input for a stored block */
    ulg enc_len;
    uchf *sym[2];               /* symbol buffers, filled and written in turn */
    Bytef *snap;                /* copy of the block's input */
};

/* The longest match at data index i, as longest_match() finds it when no
   earlier match is to be beaten, from the candidate at q on, among at most
   count candidates (0: no limit). Returns its length (2 if none) and sets
   *best to its index. */
local int mt_longest(struct mt_engine_s *e, mt_slot *sl, unsigned i, int q,
                     unsigned count, int *best) {
    Bytef *d = sl->data, *scan = d + i;
    ush *link = sl->link;
    unsigned short scan_start = load16(scan);
    unsigned short scan_end = load16(scan + 1);
    int len, best_len = MIN_MATCH - 1, nice = e->nice;
    unsigned seen = 0;
    Z_U8 p = sl->data0 + i;
    Z_U8 limit = p > e->max_dist ? p - e->max_dist : 0;

    *best = -1;
    for (;;) {
        Bytef *match = d + q;
        if (load16(match + best_len - 1) == scan_end &&
            load16(match) == scan_start) {
            len = 2 + (int)count_match(scan + 2, match + 2);
            if (len > best_len) {
                best_len = len;
                *best = q;
                if (len >= nice)
                    break;
                scan_end = load16(scan + best_len - 1);
            }
        }
        if (++seen == count || link[q] == 0)
            break;
        q -= link[q];
        if (sl->data0 + (unsigned)q <= limit)
            break;
    }
    return best_len;
}

/* Build the chains of slot sl, run deflate_slow()'s parse over it, and record
   the searches the parse makes in the chunk. */
local void mt_search(struct mt_engine_s *e, mt_slot *sl, int *headtab) {
    Bytef *d = sl->data;
    ush *link = sl->link;
    unsigned i, end = sl->hist + sl->n, nhash;
    unsigned shift = e->hash_shift, mask = e->hash_mask;
    unsigned full = e->max_chain, part = e->max_chain >> 2;
    unsigned match_length = MIN_MATCH - 1, match_available = 0;
    int match_start = 0;

    /* the hash chains of the copy */
    for (i = 0; i < (1U << e->hash_bits); i++)
        headtab[i] = -1;
    nhash = sl->size - (MIN_MATCH - 1);
    if (nhash > end)
        nhash = end;
    i = 0;
    if (sl->floor > sl->data0) {        /* positions no chain holds */
        unsigned k = sl->floor - sl->data0 < nhash ?
                     (unsigned)(sl->floor - sl->data0) : nhash;
        for (; i < k; i++)
            link[i] = 0;
    }
    for (; i < nhash; i++) {
        unsigned h = (((unsigned)d[i] << (2 * shift)) ^
                      ((unsigned)d[i + 1] << shift) ^ d[i + 2]) & mask;
        int q = headtab[h];
        link[i] = (ush)(q < 0 || i - (unsigned)q > e->max_dist ? 0 : i - (unsigned)q);
        headtab[h] = (int)i;
    }
    zmemzero(sl->res, sl->n * sizeof(mt_res));

    /* deflate_slow(), searching instead of inserting and tallying */
    i = sl->hist > MT_WARM ? sl->hist - MT_WARM : 0;
    while (i < end) {
        int q = link[i] ? (int)(i - link[i]) : -1;
        unsigned prev_length = match_length;

        match_length = MIN_MATCH - 1;
        if (q >= 0 && prev_length < e->lazy && i - (unsigned)q <= e->max_dist) {
            int part_chain = prev_length >= e->good, b;
            int len = mt_longest(e, sl, i, q, part_chain ? part : full, &b);

            if (i >= sl->hist) {
                mt_res *r = sl->res + (i - sl->hist);
                r->len = (ush)(len | (part_chain ? MT_PART : MT_FULL));
                r->dist = (ush)(len > MIN_MATCH - 1 ? i - (unsigned)b : 0);
            }
            if ((unsigned)len > prev_length) {
                match_length = (unsigned)len;
                match_start = b;
            }
            else
                match_length = prev_length;
            if (match_length <= 5 && (e->strategy == Z_FILTERED
#if TOO_FAR <= 32767
                || (match_length == MIN_MATCH &&
                    i - (unsigned)match_start > TOO_FAR)
#endif
                ))
                match_length = MIN_MATCH - 1;
        }
        if (prev_length >= MIN_MATCH && match_length <= prev_length) {
            i += prev_length - 1;       /* past the previous match */
            match_available = 0;
            match_length = MIN_MATCH - 1;
        }
        else {
            match_available = 1;
            i++;
        }
    }
    (void)match_available;
}

typedef struct mt_arg_s {
    struct mt_engine_s *e;
    int k;
} mt_arg;

local void mt_take_data(struct mt_engine_s *e, mt_slot *sl);

local void *mt_worker(void *arg) {
    struct mt_engine_s *e = ((mt_arg *)arg)->e;
    int *headtab = e->headtab[((mt_arg *)arg)->k];

    mt_lock(&e->lock);
    for (;;) {
        mt_slot *sl = NULL;
        unsigned j;
        if (e->enc == MT_QUEUED) {      /* a block to write: first */
            e->enc = MT_RUNNING;
            mt_unlock(&e->lock);
            _tr_flush_block(e->es, e->enc_buf, e->enc_len, 0);
            mt_lock(&e->lock);
            e->enc = MT_DONE;
            mt_broadcast(&e->done);
            continue;
        }
        for (j = 0; j < e->used; j++) {
            mt_slot *c = e->slot + (e->first + j) % e->nslots;
            if (c->state == MT_QUEUED) {
                sl = c;
                break;
            }
        }
        if (sl == NULL) {
            if (e->quit)
                break;
            mt_wait(&e->work, &e->lock);
            continue;
        }
        sl->state = MT_RUNNING;
        mt_take_data(e, sl);
        mt_unlock(&e->lock);
        mt_search(e, sl, headtab);
        mt_lock(&e->lock);
        sl->state = MT_DONE;
        mt_broadcast(&e->done);
    }
    mt_unlock(&e->lock);
    return NULL;
}

/* Wait for the slot to be free of workers, then free it. Called locked. */
local void mt_release(struct mt_engine_s *e, mt_slot *sl) {
    while (sl->state == MT_RUNNING)
        mt_wait(&e->done, &e->lock);
    sl->state = MT_FREE;
}

/* Drop all chunks, and start again at the current position. */
local void mt_reset(deflate_state *s) {
    struct mt_engine_s *e = s->mt;
    unsigned j;

    if (e == Z_NULL)
        return;
    mt_lock(&e->lock);
    for (j = 0; j < e->used; j++)
        mt_release(e, e->slot + (e->first + j) % e->nslots);
    e->first = e->used = 0;
    e->ready = 0;
    e->next = s->mt_abs + s->strstart;
    if (e->next < s->mt_from)
        e->next = s->mt_from;
    e->max_chain = s->max_chain_length;
    e->good = s->good_match;
    e->lazy = s->max_lazy_match;
    e->nice = s->nice_match;
    e->strategy = s->strategy;
    mt_unlock(&e->lock);
}

/* Set up slot sl's data, the bytes at stream positions [from, from + size):
   copy those in the window now, as the parse moves the window, and leave the
   rest, in the input not yet read into it, for the thread that searches the
   slot to copy. */
local void mt_copy(deflate_state *s, mt_slot *sl, Z_U8 from) {
    Z_U8 wend = s->mt_abs + s->strstart + s->lookahead;
    unsigned k = 0;

    if (from < wend) {
        k = (unsigned)(wend - from < sl->size ? wend - from : sl->size);
        zmemcpy(sl->data, s->window + (unsigned)(from - s->mt_abs), k);
    }
    sl->src_at = k;
    sl->src_len = sl->size - k;
    sl->src = sl->src_len ? s->strm->next_in + (unsigned)(from + k - wend) :
                            Z_NULL;
    sl->copy = sl->src_len ? MT_COPY_NEED : MT_COPY_DONE;
}

/* Make sure slot sl has its input bytes, copying them unless another thread
   is. Called and returns locked. */
local void mt_take_data(struct mt_engine_s *e, mt_slot *sl) {
    while (sl->copy != MT_COPY_DONE) {
        if (sl->copy == MT_COPY_NEED) {
            sl->copy = MT_COPY_BUSY;
            mt_unlock(&e->lock);
            zmemcpy(sl->data + sl->src_at, sl->src, sl->src_len);
            mt_lock(&e->lock);
            sl->copy = MT_COPY_DONE;
            mt_broadcast(&e->done);
        }
        else
            mt_wait(&e->done, &e->lock);
    }
}

/* Before deflate() returns, give every queued chunk its input bytes: the
   application may reuse its input buffer after that. */
local void mt_settle(deflate_state *s) {
    struct mt_engine_s *e = s->mt;
    unsigned j;

    mt_lock(&e->lock);
    for (j = 0; j < e->used; j++)
        mt_take_data(e, e->slot + (e->first + j) % e->nslots);
    mt_unlock(&e->lock);
}

/* Queue chunks while there are free slots and input for them. */
local void mt_fill(deflate_state *s) {
    struct mt_engine_s *e = s->mt;
    Z_U8 avail = s->mt_abs + s->strstart + s->lookahead + s->strm->avail_in;

    /* positions with MIN_LOOKAHEAD bytes after them in the input, at least
       4096 of them */
    while (e->used < e->nslots && avail >= e->next + MIN_LOOKAHEAD + 4096) {
        mt_slot *sl = e->slot + (e->first + e->used) % e->nslots;
        Z_U8 start = e->next, lo;
        unsigned n, back = e->max_dist + 8 + MT_WARM;

        n = avail - MIN_LOOKAHEAD - start < MT_CHUNK ?
            (unsigned)(avail - MIN_LOOKAHEAD - start) : MT_CHUNK;
        lo = start > back ? start - back : 0;
        if (lo < s->mt_abs)
            lo = s->mt_abs;
        /* the slot is not seen by the workers until used counts it */
        sl->start = start;
        sl->data0 = lo;
        sl->floor = s->mt_floor;
        sl->hist = (unsigned)(start - lo);
        sl->n = n;
        sl->size = sl->hist + n + MAX_MATCH;
        mt_copy(s, sl, lo);
        mt_lock(&e->lock);
        sl->state = MT_QUEUED;
        e->used++;
        mt_signal(&e->work);
        mt_unlock(&e->lock);
        e->next = start + n;
    }
}

/* Insert the positions skipped since s->mt_ins into the hash chains, up to
   window position end (exclusive), as deflate_slow() would have, and stop
   lagging. Positions more than MAX_DIST before end are left out: no search
   from end on can reach them, and a chain that ends early there ends where
   zlib's would stop. */
local void mt_catch_up(deflate_state *s, unsigned end) {
    unsigned from = s->mt_ins > s->mt_abs ? (unsigned)(s->mt_ins - s->mt_abs) : 0;
    unsigned str, stop = s->strstart + s->lookahead;
    uInt h = 0;

    /* the slides skipped: their order with the insertions skipped does not
       matter, as a slide leaves positions inserted after it unchanged. Two or
       more slides empty the tables of everything before the lag. */
    if (s->mt_slides == 1)
        slide_hash(s);
    else if (s->mt_slides > 1) {
        CLEAR_HASH(s);
        s->slid = 1;
    }
    s->mt_slides = 0;
    if (end > MAX_DIST(s) + 1 && from < end - MAX_DIST(s) - 1)
        from = end - MAX_DIST(s) - 1;
    if (stop >= MIN_MATCH - 1)
        stop -= MIN_MATCH - 1;          /* positions with three bytes */
    else
        stop = 0;
    if (stop > end)
        stop = end;
    for (str = from; str < stop; str++) {
        h = (((uInt)s->window[str] << (2 * s->hash_shift)) ^
             ((uInt)s->window[str + 1] << s->hash_shift) ^
             s->window[str + 2]) & s->hash_mask;
        s->prev[str & s->w_mask] = s->head[h];
        s->head[h] = (Pos)str;
    }
    if (end >= 1 && end + 1 < s->strstart + s->lookahead)
        s->ins_h = (((uInt)s->window[end - 1] << (2 * s->hash_shift)) ^
                    ((uInt)s->window[end] << s->hash_shift) ^
                    s->window[end + 1]) & s->hash_mask;
    (void)h;
    s->mt_ins = s->mt_abs + end;
    s->mt_lag = 0;
}

/* The hash chains start again at stream position at, whose window index is
   now strstart: with no history before it (a full flush, or a dictionary that
   replaced the window), or, if partial, with only some of the positions
   before it (after another compression function or strategy). The window
   positions are counted from here, whatever moved the window before, and the
   chunks start again. */
local void mt_restart(deflate_state *s, Z_U8 at, int partial) {
    s->mt_abs = at - s->strstart;
    s->mt_exact = 1;
    s->mt_lag = 0;
    s->mt_slides = 0;
    if (partial) {
        s->mt_floor = s->mt_abs + 1;
        s->mt_from = at + MAX_DIST(s) + 1;
    }
    else {
        s->mt_floor = at > s->mt_abs ? at : s->mt_abs + 1;
        s->mt_from = 0;
    }
    mt_reset(s);
}

/* The chunk that holds the current position, searched, or Z_NULL if there is
   none (positions before the chunks or too near the end of the available
   input). Frees the chunks behind, queues new ones. */
local mt_slot *mt_slot_at(deflate_state *s) {
    struct mt_engine_s *e = s->mt;
    Z_U8 p = s->mt_abs + s->strstart;
    mt_slot *sl;

    if (e->max_chain != s->max_chain_length || e->good != s->good_match ||
        e->lazy != s->max_lazy_match || e->nice != s->nice_match ||
        e->strategy != s->strategy)
        mt_reset(s);            /* deflateTune() or deflateParams() */

    /* free the chunks the parse has passed, and queue more */
    if (e->used && e->slot[e->first].start + e->slot[e->first].n <= p) {
        mt_lock(&e->lock);
        while (e->used &&
               e->slot[e->first].start + e->slot[e->first].n <= p) {
            mt_release(e, e->slot + e->first);
            e->first = (e->first + 1) % e->nslots;
            e->used--;
            e->ready = 0;
        }
        if (e->used == 0 && e->next < p)
            e->next = p;
        mt_unlock(&e->lock);
    }
    if (e->used < e->nslots)
        mt_fill(s);
    if (e->used == 0 || p < e->slot[e->first].start) {
        return Z_NULL;
    }
    sl = e->slot + e->first;
    if (!e->ready) {
        mt_lock(&e->lock);
        while (sl->state != MT_DONE) {
            if (sl->state == MT_QUEUED) {   /* search it here */
                sl->state = MT_RUNNING;
                mt_take_data(e, sl);
                mt_unlock(&e->lock);
                mt_search(e, sl, e->headtab[e->threads]);
                mt_lock(&e->lock);
                sl->state = MT_DONE;
                mt_broadcast(&e->done);
            }
            else
                mt_wait(&e->done, &e->lock);
        }
        mt_unlock(&e->lock);
        e->ready = 1;
    }
    return sl;
}

/* What INSERT_STRING() would give as hash_head for the current position, from
   the chunk's chains: the previous position with the same hash, or NIL if it
   is at or before window[0] (slid out or index 0) or too far (the chunk keeps
   no link longer than MAX_DIST, and deflate_slow() searches no further) */
local IPos mt_head(deflate_state *s, mt_slot *sl) {
    Z_U8 p = s->mt_abs + s->strstart;
    unsigned link = sl->link[sl->hist + (unsigned)(p - sl->start)];

    if (link == 0 || p - link <= s->mt_abs)
        return NIL;
    return (IPos)(p - link - s->mt_abs);
}

/* longest_match() for deflate_slow(), from the workers' search when it is the
   one needed, or else searched here after bringing the chains up to date */
local uInt mt_result(deflate_state *s, mt_slot *sl, IPos cur_match) {
    mt_res r = sl->res[(unsigned)(s->mt_abs + s->strstart - sl->start)];

    if ((r.len & (s->prev_length >= s->good_match ? MT_PART : MT_FULL)) == 0) {
        IPos h;
        if (s->mt_lag)
            mt_catch_up(s, s->strstart);
        INSERT_STRING(s, s->strstart, h);
        s->mt_ins = s->mt_abs + s->strstart + 1;
        Assert(h == cur_match, "mt_head differs from the chains");
        (void)h;
        return longest_match(s, cur_match);     /* not searched, or not so */
    }
    r.len &= 0x1ff;
    if (r.len > s->prev_length) {
        s->match_start = s->strstart - r.dist;
        return r.len;
    }
    return s->prev_length;
}

/* ===========================================================================
 * Block encoding while the parse goes on. When a block is full, the parse
 * hands its symbols and trees to e->es and goes on in the other symbol buffer,
 * while a worker writes the block with _tr_flush_block(). zlib's own symbol
 * buffer, which shares pending_buf, is not used meanwhile.
 *
 * The output and everything the application can see stay as in zlib: a block
 * is handed off only if the output space is sure to take all of it, so that
 * zlib would have gone on parsing too, and the parse takes the block's output
 * and bits back at the next block or before deflate_slow() returns. The block
 * written by the worker starts with an empty pending_buf, and gets a copy of
 * its input in case a stored block is shorter, since the window can slide.
 */

/* zlib's pending_buf, also while levels 10 to 12 use one of their own */
#ifdef Z_TURBO_OPT
#  define MT_ZBUF(s) ((s)->opt != Z_NULL ? (s)->opt->zbuf : (s)->pending_buf)
#else
#  define MT_ZBUF(s) ((s)->pending_buf)
#endif

/* The start of the symbol buffer: the distances first with LIT_MEM. */
#ifdef LIT_MEM
#  define MT_SYM(s) ((uchf *)(s)->d_buf)
#  define MT_SYM_ZLIB(s) (MT_ZBUF(s) + ((s)->lit_bufsize << 1))
#else
#  define MT_SYM(s) ((s)->sym_buf)
#  define MT_SYM_ZLIB(s) (MT_ZBUF(s) + (s)->lit_bufsize)
#endif

/* Use the symbol buffer at sym, of 3 * lit_bufsize bytes. */
local void mt_sym_at(deflate_state *s, uchf *sym) {
#ifdef LIT_MEM
    s->d_buf = (ushf *)sym;
    s->l_buf = sym + (s->lit_bufsize << 1);
#else
    s->sym_buf = sym;
#endif
}

/* Move the symbols so far to the buffer at sym, and use it. */
local void mt_sym_move(deflate_state *s, uchf *sym) {
#ifdef LIT_MEM
    zmemcpy(sym, s->d_buf, s->sym_next * sizeof(ush));
    zmemcpy(sym + (s->lit_bufsize << 1), s->l_buf, s->sym_next);
#else
    zmemcpy(sym, s->sym_buf, s->sym_next);
#endif
    mt_sym_at(s, sym);
}

/* Wait for the block in flight, if any, and take its output and bits. */
local void mt_encoded(deflate_state *s) {
    struct mt_engine_s *e = s->mt;
    deflate_state *es = e->es;

    if (!e->in_flight)
        return;
    mt_lock(&e->lock);
    if (e->enc == MT_QUEUED) {          /* no worker free: write it here */
        e->enc = MT_RUNNING;
        mt_unlock(&e->lock);
        _tr_flush_block(es, e->enc_buf, e->enc_len, 0);
        mt_lock(&e->lock);
    }
    else
        while (e->enc != MT_DONE)
            mt_wait(&e->done, &e->lock);
    e->enc = MT_FREE;
    mt_unlock(&e->lock);
    e->in_flight = 0;
    s->pending = es->pending;
    s->bi_buf = es->bi_buf;
    s->bi_valid = es->bi_valid;
    s->bi_used = es->bi_used;
    s->strm->data_type = es->strm->data_type;
#ifdef ZLIB_DEBUG
    s->compressed_len = es->compressed_len;
    s->bits_sent = es->bits_sent;
#endif
    flush_pending(s->strm);
}

/* FLUSH_BLOCK_ONLY() for deflate_slow() with threads: hand the block off. */
local void mt_flush_block(deflate_state *s, int last) {
    struct mt_engine_s *e = s->mt;
    deflate_state *es = e->es;
    ulg len = (ulg)((long)s->strstart - s->block_start), bound;

    mt_encoded(s);
    bound = _tr_block_bound(s);
    if (last || s->pending != 0 || s->strm->avail_out <= bound) {
        FLUSH_BLOCK_ONLY(s, last);      /* as zlib: may fill the output */
        return;
    }
    Assert(s->pending_out == s->pending_buf, "pending_out not reset");
    e->enc_buf = Z_NULL;                /* stored only if not over bound */
    if (s->block_start >= 0L && len + 4 <= bound) {
        zmemcpy(e->snap, s->window + (unsigned)s->block_start, (unsigned)len);
        e->enc_buf = (charf *)e->snap;
    }
    e->enc_len = len;
    es->level = s->level;
    es->strategy = s->strategy;
    es->strm->data_type = s->strm->data_type;
    es->pending_buf = s->pending_buf;
    es->pending_buf_size = s->pending_buf_size;
    es->pending = 0;
    es->lit_bufsize = s->lit_bufsize;
    es->bi_buf = s->bi_buf;
    es->bi_valid = s->bi_valid;
    es->bi_used = s->bi_used;
#ifdef ZLIB_DEBUG
    es->compressed_len = s->compressed_len;
    es->bits_sent = s->bits_sent;
#endif
    mt_sym_at(es, MT_SYM(s));
    _tr_hand_off(s, es);
    mt_sym_at(s, MT_SYM(s) == e->sym[0] ? e->sym[1] : e->sym[0]);
    s->block_start = s->strstart;
    mt_lock(&e->lock);
    e->enc = MT_QUEUED;
    mt_signal(&e->work);
    mt_unlock(&e->lock);
    e->in_flight = 1;
}

local void mt_free(z_streamp strm, deflate_state *s) {
    struct mt_engine_s *e = s->mt;
    unsigned j;
    int k;

    if (e == Z_NULL)
        return;
    if (e->started)
        mt_encoded(s);
    if (e->sym[0] != Z_NULL && MT_SYM(s) != MT_SYM_ZLIB(s))
        mt_sym_move(s, MT_SYM_ZLIB(s));
    if (e->started) {
        mt_lock(&e->lock);
        e->quit = 1;
        mt_broadcast(&e->work);
        mt_unlock(&e->lock);
        for (k = 0; k < e->threads; k++)
            mt_join(e->tid[k]);
        mt_mutex_destroy(&e->lock);
        mt_cond_destroy(&e->work);
        mt_cond_destroy(&e->done);
    }
    if (e->slot != Z_NULL)
        for (j = 0; j < e->nslots; j++) {
            TRY_FREE(strm, e->slot[j].data);
            TRY_FREE(strm, e->slot[j].link);
            TRY_FREE(strm, e->slot[j].res);
        }
    if (e->headtab != Z_NULL)
        for (k = 0; k <= e->threads; k++)
            TRY_FREE(strm, e->headtab[k]);
    TRY_FREE(strm, e->headtab);
    TRY_FREE(strm, e->es);
    TRY_FREE(strm, e->sym[0]);
    TRY_FREE(strm, e->sym[1]);
    TRY_FREE(strm, e->snap);
    TRY_FREE(strm, e->args);
    TRY_FREE(strm, e->slot);
    TRY_FREE(strm, e->tid);
    ZFREE(strm, e);
    s->mt = Z_NULL;
}

/* ========================================================================= */
int ZEXPORT deflateSetThreads(z_streamp strm, int threads) {
    deflate_state *s;
    struct mt_engine_s *e;
    unsigned j, size;
    int k;

    if (deflateStateCheck(strm) || threads < 0 || threads > 256)
        return Z_STREAM_ERROR;
    s = strm->state;
    mt_free(strm, s);
    if (threads <= 1)
        return Z_OK;
    e = (struct mt_engine_s *)ZALLOC(strm, 1, sizeof(struct mt_engine_s));
    if (e == Z_NULL)
        return Z_MEM_ERROR;
    zmemzero(e, sizeof(struct mt_engine_s));
    s->mt = e;
    e->threads = threads;
    e->nslots = 2 * (unsigned)threads + 2;
    e->hash_bits = s->hash_bits;
    e->hash_shift = s->hash_shift;
    e->hash_mask = s->hash_mask;
    e->max_dist = MAX_DIST(s);
    e->slot = (mt_slot *)ZALLOC(strm, e->nslots, sizeof(mt_slot));
    e->tid = (mt_thread *)ZALLOC(strm, threads, sizeof(mt_thread));
    e->headtab = (int **)ZALLOC(strm, threads + 1, sizeof(int *));
    e->args = (mt_arg *)ZALLOC(strm, threads, sizeof(mt_arg));
    if (e->slot == Z_NULL || e->tid == Z_NULL || e->headtab == Z_NULL ||
        e->args == Z_NULL) {
        mt_free(strm, s);
        return Z_MEM_ERROR;
    }
    zmemzero(e->slot, e->nslots * sizeof(mt_slot));
    zmemzero(e->headtab, (threads + 1) * sizeof(int *));
    size = e->max_dist + 8 + MT_WARM + MT_CHUNK + MAX_MATCH;
    for (j = 0; j < e->nslots; j++) {
        e->slot[j].data = (Bytef *)ZALLOC(strm, size, 1);
        e->slot[j].link = (ush *)ZALLOC(strm, size, sizeof(ush));
        e->slot[j].res = (mt_res *)ZALLOC(strm, MT_CHUNK, sizeof(mt_res));
        if (e->slot[j].data == Z_NULL || e->slot[j].link == Z_NULL ||
            e->slot[j].res == Z_NULL) {
            mt_free(strm, s);
            return Z_MEM_ERROR;
        }
    }
    for (k = 0; k <= threads; k++) {
        e->headtab[k] = (int *)ZALLOC(strm, 1U << e->hash_bits, sizeof(int));
        if (e->headtab[k] == Z_NULL) {
            mt_free(strm, s);
            return Z_MEM_ERROR;
        }
    }
    e->es = (deflate_state *)ZALLOC(strm, 1, sizeof(deflate_state));
    e->sym[0] = (uchf *)ZALLOC(strm, s->lit_bufsize, 3);
    e->sym[1] = (uchf *)ZALLOC(strm, s->lit_bufsize, 3);
    e->snap = (Bytef *)ZALLOC(strm, s->w_size, 2);
    if (e->es == Z_NULL || e->sym[0] == Z_NULL || e->sym[1] == Z_NULL ||
        e->snap == Z_NULL) {
        mt_free(strm, s);
        return Z_MEM_ERROR;
    }
    zmemzero(e->es, sizeof(deflate_state));
    e->es->strm = &e->es_strm;
    _tr_init(e->es);
    mt_sym_move(s, e->sym[0]);
    mt_mutex_init(&e->lock);
    mt_cond_init(&e->work);
    mt_cond_init(&e->done);
    e->started = 1;
    for (k = 0; k < threads; k++) {
        e->args[k].e = e;
        e->args[k].k = k;
        if (mt_create(&e->tid[k], e->args + k) != 0) {
            e->threads = k;     /* join those started */
            mt_free(strm, s);
            return Z_MEM_ERROR;
        }
    }
    mt_reset(s);
    return Z_OK;
}

/* deflate_slow()'s block flushes: on a worker with threads */
#  define SLOW_FLUSH_ONLY(s, last) { \
    if (s->mt != Z_NULL) \
        mt_flush_block(s, last); \
    else \
        FLUSH_BLOCK_ONLY(s, last); \
}
#  define SLOW_FLUSH(s, last) { \
    SLOW_FLUSH_ONLY(s, last); \
    if (s->strm->avail_out == 0) return (last) ? finish_started : need_more; \
}
#else
#  define SLOW_FLUSH_ONLY(s, last) FLUSH_BLOCK_ONLY(s, last)
#  define SLOW_FLUSH(s, last) FLUSH_BLOCK(s, last)
#endif /* Z_TURBO_THREADS */

/* ===========================================================================
 * Same as above, but achieves better compression. We use a lazy
 * evaluation for matches: a match is finally adopted only if there is
 * no better match at the next window position.
 */
local block_state deflate_slow_run(deflate_state *s, int flush) {
    IPos hash_head;          /* head of hash chain */
    int bflush;              /* set if current block must be flushed */
#ifdef Z_TURBO_THREADS
    mt_slot *sl;             /* the workers' chunk at this position */
    mt_slot *cur = Z_NULL;   /* the chunk last found by mt_slot_at() */
    Z_U8 cur_end = 0;        /* the stream position after it, 0 if none: the
                                chunks change only in mt_slot_at() */
#endif

    /* Process the input block. */
    for (;;) {
        /* Make sure that we always have enough lookahead, except
         * at the end of the input file. We need MAX_MATCH bytes
         * for the next match, plus MIN_MATCH bytes to insert the
         * string following the next match.
         */
        if (s->lookahead < MIN_LOOKAHEAD) {
            fill_window(s);
            if (s->lookahead < MIN_LOOKAHEAD && flush == Z_NO_FLUSH) {
                return need_more;
            }
            if (s->lookahead == 0) break; /* flush the current block */
        }

        /* Insert the string window[strstart .. strstart + 2] in the
         * dictionary, and set hash_head to the head of the hash chain:
         */
        hash_head = NIL;
#ifdef Z_TURBO_THREADS
        sl = Z_NULL;
        if (s->mt != Z_NULL && s->mt_exact && s->lookahead >= MIN_MATCH) {
            if (s->mt_abs + s->strstart < cur_end)
                sl = cur;               /* still in the same chunk */
            else {
                sl = cur = mt_slot_at(s);
                cur_end = sl != Z_NULL ? sl->start + sl->n : 0;
            }
        }
        if (sl != Z_NULL) {             /* the chunk's chains, no insertion */
            if (!s->mt_lag) {
                s->mt_lag = 1;
                s->mt_ins = s->mt_abs + s->strstart;
            }
            hash_head = mt_head(s, sl);
        }
        else {
            if (s->mt_lag)
                mt_catch_up(s, s->strstart);
#endif
        if (s->lookahead >= MIN_MATCH) {
            INSERT_STRING(s, s->strstart, hash_head);
        }
#ifdef Z_TURBO_THREADS
        }
#endif

        /* Find the longest match, discarding those <= prev_length.
         */
        s->prev_length = s->match_length, s->prev_match = s->match_start;
        s->match_length = MIN_MATCH-1;

        if (hash_head != NIL && s->prev_length < s->max_lazy_match &&
            s->strstart - hash_head <= MAX_DIST(s)) {
            /* To simplify the code, we prevent matches with the string
             * of window index 0 (in particular we have to avoid a match
             * of the string with itself at the start of the input file).
             */
#ifdef Z_TURBO_THREADS
            s->match_length = sl != Z_NULL ? mt_result(s, sl, hash_head) :
                                             longest_match(s, hash_head);
#else
            s->match_length = longest_match (s, hash_head);
#endif
            /* longest_match() sets match_start */

            if (s->match_length <= 5 && (s->strategy == Z_FILTERED
#if TOO_FAR <= 32767
                || (s->match_length == MIN_MATCH &&
                    s->strstart - s->match_start > TOO_FAR)
#endif
                )) {

                /* If prev_match is also MIN_MATCH, match_start is garbage
                 * but we will ignore the current match anyway.
                 */
                s->match_length = MIN_MATCH-1;
            }
        }
        /* If there was a match at the previous step and the current
         * match is not better, output the previous match:
         */
        if (s->prev_length >= MIN_MATCH && s->match_length <= s->prev_length) {
            uInt max_insert = s->strstart + s->lookahead - MIN_MATCH;
            /* Do not insert strings in hash table beyond this. */

            check_match(s, s->strstart - 1, s->prev_match, (int)s->prev_length);

            _tr_tally_dist(s, s->strstart - 1 - s->prev_match,
                           s->prev_length - MIN_MATCH, bflush);

            /* Insert in hash table all strings up to the end of the match.
             * strstart - 1 and strstart are already inserted. If there is not
             * enough lookahead, the last two strings are not inserted in
             * the hash table.
             */
            s->lookahead -= s->prev_length - 1;
            s->prev_length -= 2;
#ifdef Z_TURBO_THREADS
            if (s->mt_lag) {            /* no insertion: skip the match */
                s->strstart += s->prev_length;
                s->prev_length = 0;
            }
            else
#endif
            do {
                if (++s->strstart <= max_insert) {
                    INSERT_STRING(s, s->strstart, hash_head);
                }
            } while (--s->prev_length != 0);
            s->match_available = 0;
            s->match_length = MIN_MATCH-1;
            s->strstart++;

            if (bflush) SLOW_FLUSH(s, 0);

        } else if (s->match_available) {
            /* If there was no match at the previous position, output a
             * single literal. If there was a match but the current match
             * is longer, truncate the previous match to a single literal.
             */
            Tracevv((stderr,"%c", s->window[s->strstart - 1]));
            _tr_tally_lit(s, s->window[s->strstart - 1], bflush);
            if (bflush) {
                SLOW_FLUSH_ONLY(s, 0);
            }
            s->strstart++;
            s->lookahead--;
            if (s->strm->avail_out == 0) return need_more;
        } else {
            /* There is no previous match to compare with, wait for
             * the next step to decide.
             */
            s->match_available = 1;
            s->strstart++;
            s->lookahead--;
        }
    }
    Assert (flush != Z_NO_FLUSH, "no flush?");
    if (s->match_available) {
        Tracevv((stderr,"%c", s->window[s->strstart - 1]));
        _tr_tally_lit(s, s->window[s->strstart - 1], bflush);
        s->match_available = 0;
    }
    s->insert = s->strstart < MIN_MATCH-1 ? s->strstart : MIN_MATCH-1;
    if (flush == Z_FINISH) {
        SLOW_FLUSH(s, 1);
        return finish_done;
    }
    if (s->sym_next)
        SLOW_FLUSH(s, 0);
    return block_done;
}

/* deflate_slow(), leaving the hash chains as zlib does when it returns */
local block_state deflate_slow(deflate_state *s, int flush) {
    block_state bs;
    bs = deflate_slow_run(s, flush);
#ifdef Z_TURBO_THREADS
    if (s->mt_lag)
        mt_catch_up(s, s->strstart - s->insert);
    if (s->mt != Z_NULL) {
        mt_encoded(s);
        mt_settle(s);
    }
#endif
    return bs;
}
#endif /* FASTEST */

#ifndef Z_TURBO_THREADS
/* ========================================================================= */
int ZEXPORT deflateSetThreads(z_streamp strm, int threads) {
    if (deflateStateCheck(strm) || threads < 0)
        return Z_STREAM_ERROR;
    return threads <= 1 ? Z_OK : Z_STREAM_ERROR;
}
#endif

/* ===========================================================================
 * For Z_RLE, simply look for runs of bytes, generate matches only of distance
 * one.  Do not maintain a hash table.  (It will be regenerated if this run of
 * deflate switches away from Z_RLE.)
 */
local block_state deflate_rle(deflate_state *s, int flush) {
    int bflush;             /* set if current block must be flushed */
    uInt prev;              /* byte at distance one to match */
    Bytef *scan, *strend;   /* scan goes up to strend for length of run */

    for (;;) {
        /* Make sure that we always have enough lookahead, except
         * at the end of the input file. We need MAX_MATCH bytes
         * for the longest run, plus one for the unrolled loop.
         */
        if (s->lookahead <= MAX_MATCH) {
            fill_window(s);
            if (s->lookahead <= MAX_MATCH && flush == Z_NO_FLUSH) {
                return need_more;
            }
            if (s->lookahead == 0) break; /* flush the current block */
        }

        /* See how many times the previous byte repeats */
        s->match_length = 0;
        if (s->lookahead >= MIN_MATCH && s->strstart > 0) {
            scan = s->window + s->strstart - 1;
            prev = *scan;
            if (prev == *++scan && prev == *++scan && prev == *++scan) {
                strend = s->window + s->strstart + MAX_MATCH;
                do {
                } while (prev == *++scan && prev == *++scan &&
                         prev == *++scan && prev == *++scan &&
                         prev == *++scan && prev == *++scan &&
                         prev == *++scan && prev == *++scan &&
                         scan < strend);
                s->match_length = MAX_MATCH - (uInt)(strend - scan);
                if (s->match_length > s->lookahead)
                    s->match_length = s->lookahead;
            }
            Assert(scan <= s->window + (uInt)(s->window_size - 1),
                   "wild scan");
        }

        /* Emit match if have run of MIN_MATCH or longer, else emit literal */
        if (s->match_length >= MIN_MATCH) {
            check_match(s, s->strstart, s->strstart - 1, (int)s->match_length);

            _tr_tally_dist(s, 1, s->match_length - MIN_MATCH, bflush);

            s->lookahead -= s->match_length;
            s->strstart += s->match_length;
            s->match_length = 0;
        } else {
            /* No match, output a literal byte */
            Tracevv((stderr,"%c", s->window[s->strstart]));
            _tr_tally_lit(s, s->window[s->strstart], bflush);
            s->lookahead--;
            s->strstart++;
        }
        if (bflush) FLUSH_BLOCK(s, 0);
    }
    s->insert = 0;
    if (flush == Z_FINISH) {
        FLUSH_BLOCK(s, 1);
        return finish_done;
    }
    if (s->sym_next)
        FLUSH_BLOCK(s, 0);
    return block_done;
}

/* ===========================================================================
 * For Z_HUFFMAN_ONLY, do not look for matches.  Do not maintain a hash table.
 * (It will be regenerated if this run of deflate switches away from Huffman.)
 */
local block_state deflate_huff(deflate_state *s, int flush) {
    int bflush;             /* set if current block must be flushed */

    for (;;) {
        /* Make sure that we have a literal to write. */
        if (s->lookahead == 0) {
            fill_window(s);
            if (s->lookahead == 0) {
                if (flush == Z_NO_FLUSH)
                    return need_more;
                break;      /* flush the current block */
            }
        }

        /* Output a literal byte */
        s->match_length = 0;
        Tracevv((stderr,"%c", s->window[s->strstart]));
        _tr_tally_lit(s, s->window[s->strstart], bflush);
        s->lookahead--;
        s->strstart++;
        if (bflush) FLUSH_BLOCK(s, 0);
    }
    s->insert = 0;
    if (flush == Z_FINISH) {
        FLUSH_BLOCK(s, 1);
        return finish_done;
    }
    if (s->sym_next)
        FLUSH_BLOCK(s, 0);
    return block_done;
}

#ifdef Z_TURBO_OPT
local const uch opt_lext[29] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
    5, 5, 5, 5, 0};
local const ush opt_lbase[29] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59,
    67, 83, 99, 115, 131, 163, 195, 227, 258};
local const uch opt_dext[30] = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
    11, 11, 12, 12, 13, 13};
local const ush opt_dbase[30] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513,
    769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
local const uch opt_blorder[19] = {
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

#define OPT_DSLOT(o, d) ((d) < 256 ? (o)->dslot[d] : \
                         (o)->dslot[256 + ((d) >> 7)])

/* 16 log2(x) for x >= 1, to a sixteenth of a bit. */
local unsigned opt_log2(ulg x) {
    int n = 12, b;
    unsigned frac = 0;

    while (x >= 8192) {
        x >>= 1;
        n++;
    }
    while (x < 4096) {
        x <<= 1;
        n--;
    }
    for (b = 8; b; b >>= 1) {
        x = (x * x) >> 12;
        if (x >= 8192) {
            x >>= 1;
            frac |= (unsigned)b;
        }
    }
    return 16 * (unsigned)n + frac;
}

/* ===========================================================================
 * Optimal code lengths of at most limit bits for freq[0..n-1], by package-
 * merge. Symbols of zero frequency get no code. Needs two used symbols.
 */
local void opt_huff(opt_state *o, const unsigned *freq, int n, int limit,
                    uch *len) {
    int m = 0, i, j, k, t, np, cur, sel;
    ush *ord = o->order;
    ulg *prev = o->pw[0], *next = o->pw[1], *sw;

    for (i = 0; i < n; i++) {
        len[i] = 0;
        if (freq[i]) {                  /* insertion sort by frequency */
            for (j = m; j > 0 && freq[ord[j - 1]] > freq[i]; j--)
                ord[j] = ord[j - 1];
            ord[j] = (ush)i;
            m++;
        }
    }
    if (m < 2) {
        if (m == 1)
            len[ord[0]] = 1;
        return;
    }
    /* deepest list: the leaves */
    for (i = 0; i < m; i++) {
        prev[i] = freq[ord[i]];
        o->pm[limit - 1][i] = 1;
    }
    cur = m;
    for (t = limit - 2; t >= 0; t--) {
        np = cur >> 1;                  /* packages of the list below */
        i = j = k = 0;
        while (i < m || j < np) {
            if (j >= np || (i < m && (ulg)freq[ord[i]] <= prev[2 * j] +
                                                       prev[2 * j + 1])) {
                next[k] = freq[ord[i++]];
                o->pm[t][k++] = 1;
            }
            else {
                next[k] = prev[2 * j] + prev[2 * j + 1];
                j++;
                o->pm[t][k++] = 0;
            }
        }
        cur = k;
        sw = prev, prev = next, next = sw;
    }
    /* take 2m - 2 items from the top list, and as many as they package from
       the lists below; a leaf is one bit longer for each list it is taken in */
    sel = 2 * m - 2;
    for (t = 0; t < limit && sel > 0; t++) {
        int leaves = 0, packs = 0;
        for (k = 0; k < sel; k++) {
            if (o->pm[t][k])
                len[ord[leaves++]]++;
            else
                packs++;
        }
        sel = 2 * packs;
    }
}

/* Canonical codes, bit-reversed for writing, from the lengths. */
local void opt_codes(const uch *len, ush *code, int n) {
    ush count[16], next[16];
    unsigned c = 0, r, v;
    int i, b;

    for (i = 0; i < 16; i++)
        count[i] = 0;
    for (i = 0; i < n; i++)
        count[len[i]]++;
    count[0] = 0;
    for (b = 1; b < 16; b++) {
        c = (c + count[b - 1]) << 1;
        next[b] = (ush)c;
    }
    for (i = 0; i < n; i++)
        if (len[i]) {
            v = next[len[i]]++;
            for (r = 0, b = 0; b < len[i]; b++, v >>= 1)
                r = (r << 1) | (v & 1);
            code[i] = (ush)r;
        }
}

/* Make sure at least two symbols have a frequency, for complete codes. */
local void opt_two(unsigned *freq, int n) {
    int i, used = 0;

    for (i = 0; i < n; i++)
        used += freq[i] != 0;
    for (i = 0; used < 2 && i < n; i++)
        if (freq[i] == 0) {
            freq[i] = 1;
            used++;
        }
}

/* ===========================================================================
 * The header of a dynamic block for the lengths in t: the shortest run-length
 * coding of the code lengths for the code length code, which in turn is built
 * from that coding, a few times over. Returns the header's bits after the
 * block type.
 */
local ulg opt_header(opt_state *o, opt_tree *t) {
    uch seq[OPT_LL + 30], bl[19];
    unsigned f[19], blcost[19];
    ush run[OPT_LL + 31];
    int n, i, r, it, hclen, best_n = 0;
    ulg best = OPT_BIG, bits;
    ush rle[OPT_LL + 30];

    for (t->hlit = OPT_LL; t->hlit > 257 && t->ll[t->hlit - 1] == 0;)
        t->hlit--;
    for (t->hdist = 30; t->hdist > 1 && t->dl[t->hdist - 1] == 0;)
        t->hdist--;
    n = t->hlit + t->hdist;
    for (i = 0; i < t->hlit; i++)
        seq[i] = t->ll[i];
    for (i = 0; i < t->hdist; i++)
        seq[t->hlit + i] = t->dl[i];
    run[n] = 0;
    for (i = n - 1; i >= 0; i--)
        run[i] = (ush)(i + 1 < n && seq[i + 1] == seq[i] ? run[i + 1] + 1 : 1);
    for (i = 0; i < 19; i++)
        blcost[i] = 4 + (i >= 16 ? 2 : 0);  /* first guess */
    for (it = 0; it < 3; it++) {
        int k = 0;
        /* shortest coding for blcost, by dynamic programming from the end */
        o->rc[n] = 0;
        for (i = n - 1; i >= 0; i--) {
            ulg c = blcost[seq[i]] + o->rc[i + 1], d;
            ush h = 0;
            if (seq[i] == 0) {
                int top = run[i] < 138 ? run[i] : 138;
                for (r = 3; r <= 10 && r <= top; r++) {
                    d = blcost[17] + 3 + o->rc[i + r];
                    if (d < c)
                        c = d, h = (ush)r;
                }
                /* 18: only the longest runs, which leave the rest of the
                   zeros to a short code */
                for (r = top - 10 > 11 ? top - 10 : 11; r <= top; r++) {
                    d = blcost[18] + 7 + o->rc[i + r];
                    if (d < c)
                        c = d, h = (ush)r;
                }
            }
            if (i > 0 && seq[i - 1] == seq[i])
                for (r = 3; r <= 6 && r <= run[i]; r++) {
                    d = blcost[16] + 2 + o->rc[i + r];
                    if (d < c)
                        c = d, h = (ush)(r | 0x100);
                }
            o->rc[i] = c;
            o->rch[i] = h;
        }
        for (i = 0; i < 19; i++)
            f[i] = 0;
        for (i = 0; i < n;) {
            r = o->rch[i] & 0xff;
            if (r == 0) {
                rle[k++] = seq[i];
                f[seq[i]]++;
                i++;
            }
            else if (o->rch[i] & 0x100) {
                rle[k++] = (ush)(16 | (r - 3) << 5);
                f[16]++;
                i += r;
            }
            else if (r <= 10) {
                rle[k++] = (ush)(17 | (r - 3) << 5);
                f[17]++;
                i += r;
            }
            else {
                rle[k++] = (ush)(18 | (r - 11) << 5);
                f[18]++;
                i += r;
            }
        }
        opt_two(f, 19);
        opt_huff(o, f, 19, 7, bl);
        for (hclen = 19; hclen > 4 && bl[opt_blorder[hclen - 1]] == 0;)
            hclen--;
        bits = 5 + 5 + 4 + 3 * (ulg)hclen;
        for (i = 0; i < k; i++) {
            r = rle[i] & 31;
            bits += bl[r] + (r == 16 ? 2 : r == 17 ? 3 : r == 18 ? 7 : 0);
        }
        if (bits < best) {
            best = bits;
            best_n = k;
            t->hclen = hclen;
            zmemcpy(t->bl, bl, 19);
            zmemcpy(t->rle, rle, k * sizeof(ush));
        }
        for (i = 0; i < 19; i++)
            blcost[i] = (bl[i] ? bl[i] : 8) +
                        (i == 16 ? 2 : i == 17 ? 3 : i == 18 ? 7 : 0);
    }
    t->nrle = best_n;
    return best;
}

/* Symbol counts of syms[0..n-1] into o->fll and o->fd, with the end code. */
local void opt_count(opt_state *o, const unsigned *syms, unsigned n) {
    unsigned i, v;

    for (i = 0; i < OPT_LL; i++)
        o->fll[i] = 0;
    for (i = 0; i < 30; i++)
        o->fd[i] = 0;
    for (i = 0; i < n; i++) {
        v = syms[i];
        if (v >> 16) {
            o->fll[257 + o->lslot[v >> 16]]++;
            o->fd[OPT_DSLOT(o, v & 0xffff)]++;
        }
        else
            o->fll[v]++;
    }
    o->fll[256] = 1;
}

/* Build t for the counts in o->fll and o->fd: codes and the block's bits. */
local ulg opt_build(opt_state *o, opt_tree *t) {
    unsigned fl[OPT_LL], fd[30];
    ulg bits = 3;
    int i;

    zmemcpy(fl, o->fll, sizeof(fl));
    zmemcpy(fd, o->fd, sizeof(fd));
    opt_two(fl, OPT_LL);
    opt_two(fd, 30);
    opt_huff(o, fl, OPT_LL, 15, t->ll);
    opt_huff(o, fd, 30, 15, t->dl);
    for (i = 0; i < OPT_LL; i++)
        bits += (ulg)o->fll[i] *
                (t->ll[i] + (i > 256 ? opt_lext[i - 257] : 0));
    for (i = 0; i < 30; i++)
        bits += (ulg)o->fd[i] * (t->dl[i] + opt_dext[i]);
    bits += opt_header(o, t);
    t->bits = bits;
    return bits;
}

/* Bits of the counts in o->fll and o->fd as a fixed-code block. */
local ulg opt_fixed_bits(opt_state *o) {
    ulg bits = 3;
    int i;

    for (i = 0; i < OPT_LL; i++)
        bits += (ulg)o->fll[i] * ((i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 :
                                   8) + (i > 256 ? opt_lext[i - 257] : 0));
    for (i = 0; i < 30; i++)
        bits += (ulg)o->fd[i] * (5 + opt_dext[i]);
    return bits;
}

/* The parse's costs, 16ths of a bit, from the lengths in t. */
local void opt_costs_tree(opt_state *o, const opt_tree *t) {
    unsigned i;

    for (i = 0; i < 256; i++)
        o->litc[i] = 16 * (t->ll[i] ? t->ll[i] : OPT_UNSEEN);
    for (i = 3; i <= 258; i++) {
        unsigned c = 257 + o->lslot[i];
        o->lc[i] = 16 * ((t->ll[c] ? t->ll[c] : OPT_UNSEEN) +
                         opt_lext[c - 257]);
    }
    for (i = 0; i < 30; i++)
        o->dc[i] = 16 * ((t->dl[i] ? t->dl[i] : OPT_UNSEEN) + opt_dext[i]);
}

/* The parse's costs from the counts in o->fll and o->fd: -log2 of each
   symbol's frequency, as an entropy coder would spend. */
local void opt_costs_entropy(opt_state *o) {
    ulg tl = 0, td = 0;
    unsigned i, lt, ld, c;

    for (i = 0; i < OPT_LL; i++)
        tl += o->fll[i];
    for (i = 0; i < 30; i++)
        td += o->fd[i];
    lt = opt_log2(tl + 1);
    ld = opt_log2(td + 1);
#define OPT_ENT(f, lg) ((f) ? (lg) - opt_log2(f) + 0 : (lg) + 16)
    for (i = 0; i < 256; i++) {
        c = OPT_ENT(o->fll[i], lt);
        o->litc[i] = c < 16 ? 16 : c > 16 * OPT_UNSEEN ? 16 * OPT_UNSEEN : c;
    }
    for (i = 3; i <= 258; i++) {
        unsigned k = 257 + o->lslot[i];
        c = OPT_ENT(o->fll[k], lt);
        c = c < 16 ? 16 : c > 16 * OPT_UNSEEN ? 16 * OPT_UNSEEN : c;
        o->lc[i] = c + 16 * opt_lext[k - 257];
    }
    for (i = 0; i < 30; i++) {
        c = OPT_ENT(o->fd[i], ld);
        c = c < 16 ? 16 : c > 16 * OPT_UNSEEN ? 16 * OPT_UNSEEN : c;
        o->dc[i] = c + 16 * opt_dext[i];
    }
#undef OPT_ENT
}

/* The parse's costs of fixed codes. */
local void opt_costs_fixed(opt_state *o) {
    unsigned i, c;

    for (i = 0; i < 256; i++)
        o->litc[i] = 16 * (i < 144 ? 8 : 9);
    for (i = 3; i <= 258; i++) {
        c = 257 + o->lslot[i];
        o->lc[i] = 16 * ((c < 280 ? 7 : 8) + opt_lext[c - 257]);
    }
    for (i = 0; i < 30; i++)
        o->dc[i] = 16 * (5 + opt_dext[i]);
}

/* ===========================================================================
 * Matches.
 */
local unsigned opt_h4(const Bytef *p) {
    unsigned v = (unsigned)p[0] | (unsigned)p[1] << 8 | (unsigned)p[2] << 16 |
                 (unsigned)p[3] << 24;
    return ((v * 2654435761U) >> (32 - OPT_H4_BITS)) &
           ((1U << OPT_H4_BITS) - 1);
}

local unsigned opt_h3(const Bytef *p) {
    unsigned v = (unsigned)p[0] | (unsigned)p[1] << 8 | (unsigned)p[2] << 16;
    return ((v * 2654435761U) >> (32 - OPT_H3_BITS)) &
           ((1U << OPT_H3_BITS) - 1);
}

/* Find the matches at position p, and put them in out unless it is Z_NULL, by
   increasing length, each the nearest of its length found; insert p in the
   hash tables and the tree if ins. max_len is what the input allows (at least
   3). Returns the number of matches. */
local unsigned opt_find(opt_state *o, unsigned p, unsigned max_len,
                        unsigned nice, unsigned depth, unsigned max_dist,
                        unsigned *out, int ins) {
    Bytef *in = o->in, *m, *q = o->in + p;
    unsigned *child = o->child, *lt = Z_NULL, *gt = Z_NULL;
    unsigned h, cur, c3, lt_len = 0, gt_len = 0, len, best = 2, n = 0;
    unsigned pa = p + o->abase;         /* p as stored */
    unsigned cutoff = p > max_dist ? pa - max_dist : o->abase;

    if (nice > max_len)
        nice = max_len;
    h = opt_h3(q);                      /* the nearest three-byte match */
    c3 = o->h3[h];
    if (ins)
        o->h3[h] = pa;
    if (out != Z_NULL && c3 >= cutoff && c3 < pa) {
        m = in + (c3 - o->abase);
        if (m[0] == q[0] && m[1] == q[1] && m[2] == q[2]) {
            out[n++] = 3U << 16 | (pa - c3 - 1);
            best = 3;
        }
    }
    if (max_len < 4)
        return n;
    h = opt_h4(q);
    cur = o->h4[h];
    if (ins) {
        o->h4[h] = pa;
        lt = child + 2 * p;
        gt = lt + 1;
    }
    for (;;) {
        if (cur < cutoff || cur >= pa || depth-- == 0) {
            if (ins)
                *lt = *gt = 0;
            break;
        }
        m = in + (cur - o->abase);
        len = lt_len < gt_len ? lt_len : gt_len;
        if (m[len] == q[len]) {
            while (++len < max_len && m[len] == q[len])
                ;
            if (len > best) {
                best = len;
                if (out != Z_NULL) {
                    if (n == 1 && len >= 4 && pa - cur - 1 <= (out[0] & 0xffff))
                        n = 0;          /* nearer and longer: drop the 3 */
                    out[n++] = len << 16 | (pa - cur - 1);
                }
            }
            if (len >= nice) {
                if (ins) {
                    *lt = child[2 * (cur - o->abase)];
                    *gt = child[2 * (cur - o->abase) + 1];
                }
                break;
            }
        }
        if (m[len] < q[len]) {
            if (ins) {
                *lt = cur;
                lt = child + 2 * (cur - o->abase) + 1;
            }
            lt_len = len;
            cur = child[2 * (cur - o->abase) + 1];
        }
        else {
            if (ins) {
                *gt = cur;
                gt = child + 2 * (cur - o->abase);
            }
            gt_len = len;
            cur = child[2 * (cur - o->abase)];
        }
    }
    return n;
}

/* ===========================================================================
 * The cheapest parse of in[a..b) for the costs in o, into o->tmp. Returns the
 * number of symbols.
 */
local unsigned opt_parse(opt_state *o, unsigned a, unsigned b) {
    unsigned n = b - a, i, k, L, len, end, c, t, best, choice, dist;
    unsigned *cost = o->cost, *ch = o->ch, *e;
    const Bytef *in = o->in;

    cost[n] = 0;
    for (i = n; i-- > 0;) {
        best = cost[i + 1] + o->litc[in[a + i]];
        choice = 0;
        e = o->cache + o->off[a - o->start + i];
        end = o->off[a - o->start + i + 1] - o->off[a - o->start + i];
        L = 3;
        for (k = 0; k < end; k++) {
            len = e[k] >> 16;
            if (len > n - i)
                len = n - i;
            dist = e[k] & 0xffff;
            c = o->dc[OPT_DSLOT(o, dist)];
            for (; L <= len; L++) {
                t = c + o->lc[L] + cost[i + L];
                if (t < best) {
                    best = t;
                    choice = L << 16 | dist;
                }
            }
        }
        cost[i] = best;
        ch[i] = choice;
    }
    for (i = 0, k = 0; i < n; k++) {
        c = ch[i];
        if (c) {
            o->tmp[k] = c;
            i += c >> 16;
        }
        else
            o->tmp[k] = in[a + i++];
    }
    return k;
}

/* Bits of syms[0..n-1] as the best of a dynamic and a fixed block, building
   the dynamic codes in t. *fixed is set if fixed is better. */
local ulg opt_bits(opt_state *o, const unsigned *syms, unsigned n,
                   opt_tree *t, int *fixed) {
    ulg d, f;

    opt_count(o, syms, n);
    f = opt_fixed_bits(o);
    d = opt_build(o, t);
    *fixed = f <= d;
    return f <= d ? f : d;
}

/* Bits of a stored block of n bytes, aligning first (worst case). */
#define OPT_STORED_BITS(n) (((ulg)(n) / 65535 + 1) * (3 + 7 + 32) + \
                            8 * (ulg)(n))

/* ===========================================================================
 * Bit writing into pending_buf.
 */
local void opt_put(deflate_state *s, opt_state *o, unsigned v, int n) {
    o->bb |= (ulg)v << o->bc;
    o->bc += n;
    while (o->bc >= 8) {
        put_byte(s, (Byte)o->bb);
        o->bb >>= 8;
        o->bc -= 8;
    }
}

/* Write syms[0..n-1] as a block, with t's codes or fixed ones. */
local void opt_write(deflate_state *s, opt_state *o, const unsigned *syms,
                     unsigned n, opt_tree *t, int fixed, int last) {
    unsigned i, v, c, len, d;
    int k;

    if (fixed) {
        for (i = 0; i < 288; i++)
            t->ll[i] = (uch)(i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8);
        for (i = 0; i < 30; i++)
            t->dl[i] = 5;
        opt_codes(t->ll, t->llc, 288);
        opt_codes(t->dl, t->dlc, 30);
        opt_put(s, o, (unsigned)last | 2, 3);
    }
    else {
        opt_codes(t->ll, t->llc, OPT_LL);
        opt_codes(t->dl, t->dlc, 30);
        opt_codes(t->bl, t->blc, 19);
        opt_put(s, o, (unsigned)last | 4, 3);
        opt_put(s, o, (unsigned)t->hlit - 257, 5);
        opt_put(s, o, (unsigned)t->hdist - 1, 5);
        opt_put(s, o, (unsigned)t->hclen - 4, 4);
        for (k = 0; k < t->hclen; k++)
            opt_put(s, o, t->bl[opt_blorder[k]], 3);
        for (k = 0; k < t->nrle; k++) {
            c = t->rle[k] & 31;
            opt_put(s, o, t->blc[c], t->bl[c]);
            if (c >= 16)
                opt_put(s, o, t->rle[k] >> 5, c == 16 ? 2 : c == 17 ? 3 : 7);
        }
    }
    for (i = 0; i < n; i++) {
        v = syms[i];
        len = v >> 16;
        if (len) {
            c = o->lslot[len];
            opt_put(s, o, t->llc[257 + c], t->ll[257 + c]);
            if (opt_lext[c])
                opt_put(s, o, len - opt_lbase[c], opt_lext[c]);
            d = v & 0xffff;
            c = OPT_DSLOT(o, d);
            opt_put(s, o, t->dlc[c], t->dl[c]);
            if (opt_dext[c])
                opt_put(s, o, d + 1 - opt_dbase[c], opt_dext[c]);
        }
        else
            opt_put(s, o, t->llc[v], t->ll[v]);
    }
    opt_put(s, o, t->llc[256], t->ll[256]);
}

/* Write in[a..b) as stored blocks. */
local void opt_stored(deflate_state *s, opt_state *o, unsigned a, unsigned b,
                      int last) {
    unsigned n;

    do {
        n = b - a > 65535 ? 65535 : b - a;
        opt_put(s, o, (unsigned)(last && a + n == b), 3);
        s->bi_used = ((o->bc - 1) & 7) + 1;
        if (o->bc)
            opt_put(s, o, 0, 8 - o->bc);
        put_byte(s, (Byte)n);
        put_byte(s, (Byte)(n >> 8));
        put_byte(s, (Byte)~n);
        put_byte(s, (Byte)(~n >> 8));
        zmemcpy(s->pending_buf + s->pending, o->in + a, n);
        s->pending += n;
        a += n;
    } while (a < b);
}

/* ===========================================================================
 * Split the symbols sym[0..ns) of in[a..b) into blocks, by dynamic
 * programming over candidate boundaries with an entropy estimate of each
 * block. Puts the boundaries, as symbol indices, in bnd[0..nb], and returns
 * nb, the number of blocks.
 */
local unsigned opt_split(opt_state *o, const unsigned *sym, unsigned ns,
                         unsigned grid, unsigned *bnd) {
    unsigned g, np, i, j, k, v, *pf = o->pf, *ci, *cj;
    ulg *best = o->sb;
    ush *from = o->sf;
    ulg c, tl, td, el, ed, ex;
    unsigned usedl, usedd;

    g = (ns + grid - 1) / grid;
    if (g < OPT_GMIN)
        g = OPT_GMIN;
    np = (ns + g - 1) / g;              /* candidate blocks */
    if (np <= 1) {
        bnd[0] = 0;
        bnd[1] = ns;
        return 1;
    }
    /* counts before each candidate boundary: OPT_LL + 30 per boundary */
    for (k = 0; k < OPT_LL + 30; k++)
        pf[k] = 0;
    for (i = 0, j = 1; j <= np; j++) {
        unsigned *p = pf + (ulg)j * (OPT_LL + 30);
        zmemcpy(p, p - (OPT_LL + 30), (OPT_LL + 30) * sizeof(unsigned));
        for (; i < ns && i < j * g; i++) {
            v = sym[i];
            if (v >> 16) {
                p[257 + o->lslot[v >> 16]]++;
                p[OPT_LL + OPT_DSLOT(o, v & 0xffff)]++;
            }
            else
                p[v]++;
        }
    }
    best[0] = 0;
    for (j = 1; j <= np; j++) {
        best[j] = (ulg)-1;
        cj = pf + (ulg)j * (OPT_LL + 30);
        for (i = 0; i < j; i++) {
            ci = pf + (ulg)i * (OPT_LL + 30);
            tl = 1, td = 0, el = 0, ed = 0, ex = 0, usedl = 1, usedd = 0;
            for (k = 0; k < OPT_LL; k++) {
                v = cj[k] - ci[k];
                if (v) {
                    tl += v;
                    el += o->flog[v];
                    usedl++;
                    if (k > 256)
                        ex += (ulg)v * opt_lext[k - 257];
                }
            }
            for (k = 0; k < 30; k++) {
                v = cj[OPT_LL + k] - ci[OPT_LL + k];
                if (v) {
                    td += v;
                    ed += o->flog[v];
                    usedd++;
                    ex += (ulg)v * opt_dext[k];
                }
            }
            c = best[i] + o->flog[tl] - el + o->flog[td] - ed + 16 * ex +
                16 * (40 + 4 * (ulg)(usedl + usedd));
            if (c < best[j]) {
                best[j] = c;
                from[j] = (ush)i;
            }
        }
    }
    for (k = 0, j = np; j; j = from[j])
        k++;
    bnd[k] = ns;
    for (i = k, j = np; j; j = from[j])
        bnd[--i] = from[j] * g < ns ? from[j] * g : ns;
    bnd[0] = 0;
    return k;
}

/* ===========================================================================
 * Compress in[start..end) as one or more blocks into pending_buf.
 */
local unsigned opt_segment(deflate_state *s, opt_state *o, unsigned end,
                           int last) {
    const struct opt_level_s *lv = opt_level + (s->level - 10);
    unsigned start = o->start, n = end - start, p, q, i, k, ns, nb = 0, a, b;
    unsigned r, lim;
    unsigned max_dist = s->w_size, nice = s->nice_match, depth;
    unsigned bnd[OPT_GRID + 2], pos[OPT_GRID + 2];
    unsigned char kind[OPT_GRID + 2];   /* 0 dynamic, 1 fixed, 2 stored */
    ulg cst[OPT_GRID + 2], total, bits;
    unsigned room = OPT_SEG * OPT_CACHE;
    int fixed;
    unsigned *sym = o->sym, *sym2 = o->sym2, *sw;

    depth = s->max_chain_length;
    if (nice > MAX_MATCH)
        nice = MAX_MATCH;
    o->bb = s->bi_buf;
    o->bc = s->bi_valid;
    while (o->bc >= 8) {
        put_byte(s, (Byte)o->bb);
        o->bb >>= 8;
        o->bc -= 8;
    }
    if (n == 0) {                       /* only for the last block */
        opt_put(s, o, 3, 3);            /* fixed, last */
        opt_put(s, o, 0, 7);            /* end code */
        goto done;
    }

    /* the positions before start not in the tree yet, then the matches; a
       position goes into the tree once MAX_MATCH bytes follow it */
    lim = o->len > MAX_MATCH ? o->len - MAX_MATCH : 0;
    for (p = o->ins; p < start && p < lim; p++)
        opt_find(o, p, MAX_MATCH, nice, depth, max_dist, Z_NULL, 1);
    o->off[0] = 0;
    for (i = 0; i < n; i++) {
        if (o->off[i] + MAX_MATCH > room) {     /* cache full: stop here */
            n = i;
            end = start + n;
            last = 0;
            break;
        }
        p = start + i;
        q = end - p > MAX_MATCH ? MAX_MATCH : end - p;
        k = q >= 3 ? opt_find(o, p, q, nice, depth, max_dist,
                              o->cache + o->off[i], p < lim) : 0;
        o->off[i + 1] = o->off[i] + k;
        if (k && (o->cache[o->off[i] + k - 1] >> 16) >= nice) {
            r = o->cache[o->off[i] + k - 1] >> 16;  /* long: skip into it */
            while (--r && i + 1 < n) {
                i++;
                p++;
                if (p < lim)
                    opt_find(o, p, MAX_MATCH, nice, depth, max_dist, Z_NULL,
                             1);
                o->off[i + 1] = o->off[i];
            }
        }
    }

    p = start + n < lim ? start + n : lim;  /* all before p are in the tree */
    if (p > o->ins)
        o->ins = p;

    if (s->strategy == Z_FIXED) {       /* fixed codes only */
        opt_costs_fixed(o);
        ns = opt_parse(o, start, end);
        opt_count(o, o->tmp, ns);
        if (opt_fixed_bits(o) < OPT_STORED_BITS(n)) {
            opt_write(s, o, o->tmp, ns, &o->t, 1, last);
        }
        else
            opt_stored(s, o, start, end, last);
        goto done;
    }
    if (n < OPT_TINY) {                 /* a few bytes, as between flushes:
                                           one parse, for fixed codes */
        opt_costs_fixed(o);
        ns = opt_parse(o, start, end);
        bits = opt_bits(o, o->tmp, ns, &o->t, &fixed);
        if (OPT_STORED_BITS(n) < bits)
            opt_stored(s, o, start, end, last);
        else
            opt_write(s, o, o->tmp, ns, &o->t, fixed, last);
        goto done;
    }

    /* first costs: the statistics of a lazy parse, as zlib's */
    for (i = 0, ns = 0; i < n; ns++) {
        k = o->off[i + 1] - o->off[i];
        if (k) {
            r = o->cache[o->off[i] + k - 1];
            if ((r >> 16) > n - i)
                r = (n - i) << 16 | (r & 0xffff);
            if ((r >> 16) == 3 && (r & 0xffff) >= 4096)
                r = 0;                  /* zlib's TOO_FAR */
            if (r && i + 1 < n && o->off[i + 2] > o->off[i + 1] &&
                (o->cache[o->off[i + 2] - 1] >> 16) > (r >> 16))
                r = 0;                  /* longer at the next position */
            if ((r >> 16) >= 3) {
                sym[ns] = r;
                i += r >> 16;
                continue;
            }
        }
        sym[ns] = o->in[start + i++];
    }
    opt_count(o, sym, ns);
    opt_costs_entropy(o);
    ns = opt_parse(o, start, end);
    opt_count(o, o->tmp, ns);
    opt_costs_entropy(o);
    ns = opt_parse(o, start, end);
    zmemcpy(sym, o->tmp, ns * sizeof(unsigned));

    for (r = 0; r < lv->rounds; r++) {
        unsigned w = 0;
        nb = opt_split(o, sym, ns, lv->grid, bnd);
        /* positions of the boundaries */
        for (i = 0, k = 0, p = start; k <= nb; k++) {
            for (; i < bnd[k]; i++)
                p += sym[i] >> 16 ? sym[i] >> 16 : 1;
            pos[k] = p;
        }
        /* parse each block again with its own codes */
        for (k = 0; k < nb; k++) {
            unsigned bs = bnd[k], bn = bnd[k + 1] - bnd[k], pass, tn;
            ulg best;
            a = pos[k];
            b = pos[k + 1];
            best = opt_bits(o, sym + bs, bn, &o->t, &fixed);
            zmemcpy(sym2 + w, sym + bs, bn * sizeof(unsigned));
            for (pass = 0; pass < lv->passes; pass++) {
                if (fixed)
                    opt_costs_fixed(o);
                else
                    opt_costs_tree(o, &o->t);
                tn = opt_parse(o, a, b);
                bits = opt_bits(o, o->tmp, tn, &o->t, &fixed);
                if (bits >= best)
                    break;
                best = bits;
                bn = tn;
                zmemcpy(sym2 + w, o->tmp, tn * sizeof(unsigned));
            }
            bnd[k] = w;
            w += bn;
        }
        bnd[nb] = w;
        sw = sym, sym = sym2, sym2 = sw;
        ns = w;
    }
    o->sym = sym;
    o->sym2 = sym2;

    /* each block's kind, then merge adjacent blocks when that is smaller */
    for (k = 0; k < nb; k++) {
        cst[k] = opt_bits(o, sym + bnd[k], bnd[k + 1] - bnd[k], &o->t,
                          &fixed);
        kind[k] = (unsigned char)fixed;
        if (OPT_STORED_BITS(pos[k + 1] - pos[k]) < cst[k]) {
            cst[k] = OPT_STORED_BITS(pos[k + 1] - pos[k]);
            kind[k] = 2;
        }
    }
    total = nb ? cst[0] : 0;
    for (k = 1, i = 1; k < nb; k++) {
        if (kind[i - 1] != 2 && kind[k] != 2) {
            bits = opt_bits(o, sym + bnd[i - 1], bnd[k + 1] - bnd[i - 1],
                            &o->t, &fixed);
            if (bits <= cst[i - 1] + cst[k]) {
                total += bits - cst[i - 1];
                cst[i - 1] = bits;
                kind[i - 1] = (unsigned char)fixed;
                bnd[i] = bnd[k + 1];
                pos[i] = pos[k + 1];
                continue;
            }
        }
        bnd[i] = bnd[k];
        pos[i] = pos[k];
        cst[i] = cst[k];
        kind[i] = kind[k];
        total += cst[k];
        i++;
        bnd[i] = bnd[k + 1];
        pos[i] = pos[k + 1];
    }
    nb = i;
    if (total > OPT_STORED_BITS(n)) {   /* all stored is smaller */
        opt_stored(s, o, start, end, last);
        goto done;
    }
    for (k = 0; k < nb; k++) {
        if (kind[k] == 2) {             /* stored, with the stored after */
            for (i = k; i + 1 < nb && kind[i + 1] == 2; i++)
                ;
            opt_stored(s, o, pos[k], pos[i + 1], last && i + 1 == nb);
            k = i;
            continue;
        }
        opt_bits(o, sym + bnd[k], bnd[k + 1] - bnd[k], &o->t, &fixed);
        opt_write(s, o, sym + bnd[k], bnd[k + 1] - bnd[k], &o->t,
                  kind[k] == 1, last && k + 1 == nb);
    }
  done:
    if (last) {
        s->bi_used = ((o->bc - 1) & 7) + 1;
        if (o->bc)
            opt_put(s, o, 0, 8 - o->bc);
    }
    s->bi_buf = (ush)o->bb;
    s->bi_valid = o->bc;
    return end;
}

/* ===========================================================================
 * Memory of the levels: allocate on the way in, free on the way out.
 */
local int opt_alloc(z_streamp strm, deflate_state *s) {
    opt_state *o;
    unsigned i, d, c, win = s->w_size;

    o = (opt_state *)ZALLOC(strm, 1, sizeof(opt_state));
    if (o == Z_NULL)
        return Z_MEM_ERROR;
    zmemzero(o, sizeof(opt_state));
    s->opt = o;
    o->in = (Bytef *)ZALLOC(strm, win + OPT_SLACK + OPT_SEG + 8, 1);
    o->child = (unsigned *)ZALLOC(strm, 2 * (win + OPT_SLACK + OPT_SEG),
                                   sizeof(unsigned));
    o->h4 = (unsigned *)ZALLOC(strm, 1U << OPT_H4_BITS, sizeof(unsigned));
    o->h3 = (unsigned *)ZALLOC(strm, 1U << OPT_H3_BITS, sizeof(unsigned));
    o->off = (unsigned *)ZALLOC(strm, OPT_SEG + 1, sizeof(unsigned));
    o->cache = (unsigned *)ZALLOC(strm, OPT_SEG * OPT_CACHE + MAX_MATCH,
                                  sizeof(unsigned));
    o->cost = (unsigned *)ZALLOC(strm, OPT_SEG + 1, sizeof(unsigned));
    o->ch = (unsigned *)ZALLOC(strm, OPT_SEG + 1, sizeof(unsigned));
    o->sym = (unsigned *)ZALLOC(strm, OPT_SEG + 1, sizeof(unsigned));
    o->sym2 = (unsigned *)ZALLOC(strm, OPT_SEG + 1, sizeof(unsigned));
    o->tmp = (unsigned *)ZALLOC(strm, OPT_SEG + 1, sizeof(unsigned));
    o->nflog = OPT_SEG + 2;
    o->flog = (unsigned *)ZALLOC(strm, o->nflog, sizeof(unsigned));
    o->pf = (unsigned *)ZALLOC(strm, (OPT_GRID + 1) * (OPT_LL + 30),
                               sizeof(unsigned));
    o->out = (uchf *)ZALLOC(strm, OPT_SEG + (OPT_SEG >> 4) + 4096, 1);
    if (o->in == Z_NULL || o->child == Z_NULL || o->h4 == Z_NULL ||
        o->h3 == Z_NULL || o->off == Z_NULL || o->cache == Z_NULL ||
        o->cost == Z_NULL || o->ch == Z_NULL || o->sym == Z_NULL ||
        o->sym2 == Z_NULL || o->tmp == Z_NULL || o->flog == Z_NULL ||
        o->pf == Z_NULL || o->out == Z_NULL)
        return Z_MEM_ERROR;             /* opt_free() frees the rest */
    zmemzero(o->h4, (1U << OPT_H4_BITS) * sizeof(unsigned));
    zmemzero(o->h3, (1U << OPT_H3_BITS) * sizeof(unsigned));
    o->abase = 1;
    for (i = 0; i < o->nflog; i++)
        o->flog[i] = i ? i * opt_log2(i) : 0;
    for (d = 0, c = 0; c < 30; c++)     /* d_code() of zlib, for dist - 1 */
        for (i = 0; i < (1U << opt_dext[c]); i++, d++)
            if (d < 256)
                o->dslot[d] = (uch)c;
            else if ((d >> 7) < 256)
                o->dslot[256 + (d >> 7)] = (uch)c;
    for (c = 0; c < 29; c++)
        for (i = opt_lbase[c]; i < (c == 28 ? 259U : opt_lbase[c + 1]); i++)
            o->lslot[i] = (uch)c;
    o->lslot[258] = 28;
    o->zbuf = s->pending_buf;
    o->zbuf_size = s->pending_buf_size;
    s->pending_buf = o->out;
    s->pending_out = o->out;
    s->pending_buf_size = OPT_SEG + (OPT_SEG >> 4) + 4096;
    o->fresh = 1;
    return Z_OK;
}

local void opt_free(z_streamp strm, deflate_state *s) {
    opt_state *o = s->opt;

    if (o == Z_NULL)
        return;
    if (o->zbuf != Z_NULL) {
        s->pending_buf = o->zbuf;
        s->pending_out = o->zbuf;
        s->pending_buf_size = o->zbuf_size;
    }
    TRY_FREE(strm, o->out);
    TRY_FREE(strm, o->pf);
    TRY_FREE(strm, o->flog);
    TRY_FREE(strm, o->tmp);
    TRY_FREE(strm, o->sym2);
    TRY_FREE(strm, o->sym);
    TRY_FREE(strm, o->ch);
    TRY_FREE(strm, o->cost);
    TRY_FREE(strm, o->cache);
    TRY_FREE(strm, o->off);
    TRY_FREE(strm, o->h3);
    TRY_FREE(strm, o->h4);
    TRY_FREE(strm, o->child);
    TRY_FREE(strm, o->in);
    ZFREE(strm, o);
    s->opt = Z_NULL;
}

/* Keep the last w_size bytes before in[o->start] as the window, moving them
   down once the window has moved OPT_SLACK bytes up. */
local void opt_keep(deflate_state *s, opt_state *o) {
    unsigned from = o->start > s->w_size + OPT_SLACK ?
                    o->start - s->w_size : 0;

    unsigned i;

    if (from) {
        for (i = 0; i < o->len - from; i++)     /* overlapping: forward */
            o->in[i] = o->in[from + i];
        for (i = 0; i < 2 * (o->len - from); i++)
            o->child[i] = o->child[2 * from + i];
        o->len -= from;
        o->start -= from;
        o->ins -= from;
        o->abase += from;
    }
}

/* Forget the window: what the tables hold is then before it. */
local void opt_forget(opt_state *o) {
    o->abase += o->len + 1;
    o->len = o->start = o->ins = 0;
}

/* ===========================================================================
 * The compress function of levels 10 to 12.
 */
local block_state deflate_opt(deflate_state *s, int flush) {
    opt_state *o = s->opt;
    z_streamp strm = s->strm;
    unsigned cap, end;
    int last;

    if (o->fresh) {                     /* the window: a dictionary, if any */
        unsigned n = s->strstart + s->lookahead;
        if (n > s->w_size)
            n = s->w_size;
        zmemcpy(o->in, s->window + s->strstart + s->lookahead - n, n);
        o->len = o->start = n;
        o->ins = 0;
        s->strstart = 0;
        s->lookahead = 0;
        o->fresh = 0;
    }
    for (;;) {
        if (o->abase > 0x7fffffffU) {   /* positions as stored about to wrap */
            zmemzero(o->h4, (1U << OPT_H4_BITS) * sizeof(unsigned));
            zmemzero(o->h3, (1U << OPT_H3_BITS) * sizeof(unsigned));
            o->abase = 1;
            o->ins = 0;
        }
        cap = o->start + OPT_SEG;
        if (o->len < cap && strm->avail_in)
            o->len += read_buf(strm, o->in + o->len, cap - o->len);
        s->lookahead = o->len - o->start;
        if (o->len < cap && flush == Z_NO_FLUSH)
            return need_more;
        last = flush == Z_FINISH && strm->avail_in == 0;
        if (o->len == o->start && !last)
            break;
        end = opt_segment(s, o, o->len, last);
        if (end < o->len)
            last = 0;
        o->start = end;
        s->lookahead = o->len - o->start;
        opt_keep(s, o);
        flush_pending(strm);
        if (last)
            return strm->avail_out == 0 ? finish_started : finish_done;
        if (strm->avail_out == 0)
            return need_more;
        if (strm->avail_in == 0 && flush == Z_NO_FLUSH)
            return need_more;
    }
    if (flush == Z_FULL_FLUSH)          /* no matches before a full flush */
        opt_forget(o);
    return block_done;
}

/* deflateSetDictionary() after levels 10 to 12 have started: the dictionary
   goes after their window. */
local void opt_dictionary(deflate_state *s, const Bytef *dict, uInt len) {
    opt_state *o = s->opt;

    if (len >= s->w_size) {
        dict += len - s->w_size;
        len = s->w_size;
        opt_forget(o);
    }
    o->start = o->len;
    opt_keep(s, o);
    zmemcpy(o->in + o->len, dict, len);
    o->len += len;
    o->start = o->len;
    opt_keep(s, o);
}
#endif /* Z_TURBO_OPT */

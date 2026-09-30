// ACEAPEX lzbench integration — clean C API
#include "../../bench/codecs.h"
#include "aceapex.h"
#include "aceapex_api.cpp"
#include <stdlib.h>
#include <string.h>

struct AcepxState { int threads; int level; };

char* lzbench_aceapex_init(size_t insize, size_t level, size_t threads) {
    AcepxState* s = (AcepxState*)malloc(sizeof(AcepxState));
    if (!s) return nullptr;
    s->threads = threads > 0 ? (int)threads : 1;
    s->level   = level > 0 ? (int)level : 2;
    return (char*)s;
}

void lzbench_aceapex_deinit(char* workmem) { free(workmem); }

int64_t lzbench_aceapex_compress(char* inbuf, size_t insize,
                                  char* outbuf, size_t outsize,
                                  codec_options_t* opts) {
    AcepxState* s = (AcepxState*)opts->work_mem;
    if (!s) return -1;
    int thr = opts->threads > 0 ? opts->threads : s->threads;
    int64_t r = aceapex_compress(inbuf, insize, outbuf, outsize, s->level, thr);
    return r >= 0 ? r : -1;
}

int64_t lzbench_aceapex_decompress(char* inbuf, size_t insize,
                                    char* outbuf, size_t outsize,
                                    codec_options_t* opts) {
    AcepxState* s = (AcepxState*)opts->work_mem;
    // thr == 1 (lzbench -T: one codec copy per pool thread) decodes on the calling
    // thread and spawns nothing; -I# gives the codec its own budget.
    int thr = opts->threads > 0 ? opts->threads : (s ? s->threads : 1);
    int64_t r = aceapex_decompress_mt(inbuf, insize, outbuf, outsize, thr);
    return r >= 0 ? r : -1;
}

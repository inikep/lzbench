// lzbench glue for the ACEAPEX CUDA decoder (same .aet as the CPU codec).
#ifndef BENCH_REMOVE_ACEAPEX
#ifdef BENCH_HAS_CUDA
#include "bench/codecs.h"
#include "aceapex_cuda.h"
#include <cstdio>

char* lzbench_aceapex_cuda_init(size_t insize, size_t level, size_t threads)
{
    if(!aceapex_cg_available()){
        fprintf(stderr, "aceapex_cuda: no CUDA device available at runtime\n");
        return NULL;
    }
    // delegate to the CPU codec init: compression reuses lzbench_aceapex_compress,
    // which expects the workmem/levels prepared by lzbench_aceapex_init
    return lzbench_aceapex_init(insize, level, threads);
}

void lzbench_aceapex_cuda_deinit(char* workmem)
{
    aceapex_cg_release();
    lzbench_aceapex_deinit(workmem);
}

int64_t lzbench_aceapex_cuda_decompress(char *inbuf, size_t insize, char *outbuf,
                                        size_t outsize, codec_options_t *codec_options)
{
    (void)codec_options;
    aceapex_streams_t s;
    if(aceapex_decode_streams(inbuf, insize, &s) != 0) return 0;
    int64_t r = aceapex_cg_match_decode(&s, outbuf, outsize);
    aceapex_streams_free(&s);
    return r;
}
#endif // BENCH_HAS_CUDA
#endif // BENCH_REMOVE_ACEAPEX

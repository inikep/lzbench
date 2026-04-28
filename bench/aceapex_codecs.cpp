// ACEAPEX lzbench integration
// Add to bench/lz_codecs.cpp

#include "codecs.h"
#include <stdlib.h>
#include <string.h>
#include <vector>

// Forward declarations from aceapex_main.cpp
// We include it directly
#define ACEAPEX_NO_MAIN
#include "../lz/aceapex/aceapex_main.cpp"

struct AcepxState {
    int threads;
    int level;
};

char* lzbench_aceapex_init(size_t insize, size_t level, size_t threads) {
    AcepxState* s = (AcepxState*)malloc(sizeof(AcepxState));
    s->threads = threads > 0 ? (int)threads : 1;
    s->level   = level > 0 ? (int)level : 2;
    return (char*)s;
}

void lzbench_aceapex_deinit(char* workmem) {
    free(workmem);
}

int64_t lzbench_aceapex_compress(char* inbuf, size_t insize,
                                  char* outbuf, size_t outsize,
                                  codec_options_t* opts)
{
    AcepxState* s = (AcepxState*)opts->work_mem;
    if (!s) return -1;
    int thr = opts->threads > 0 ? opts->threads : s->threads;
    int lvl = s->level;

    std::vector<BlockOffsets> boffs;
    uint8_t *rl,*ro,*rn,*rc;
    size_t tl,to,tn,tc,nb;
    double et;

    if (!encode_file((const uint8_t*)inbuf,insize,thr,lvl,boffs,
                     rl,tl,ro,to,rn,tn,rc,tc,et,nb)) return -1;

    // Entropy encode
    size_t zls,zos,zns,zcs;
    uint8_t *zl,*zo,*zn,*zc;
    zl=lit_compress(rl,tl,zls);
    entropy_encode(rl,tl,ro,to,rn,tn,rc,tc,zl,zls,zo,zos,zn,zns,zc,zcs);
    free(rl);free(ro);free(rn);free(rc);

    // Build header
    AetHeader hdr;
    memcpy(hdr.magic,"ACEPX2\0\0",8);
    hdr.version=2; hdr.orig_size=insize;
    hdr.block_size=BLOCK_SIZE; hdr.num_blocks=nb;
    uint64_t hv=OUR_CHECKSUM(inbuf,insize);
    memcpy(hdr.xxhash,&hv,8);
    hdr.zlit_sz=zls;hdr.zoff_sz=zos;hdr.zlen_sz=zns;hdr.zcmd_sz=zcs;

    size_t total=sizeof(hdr)+nb*sizeof(BlockOffsets)+zls+zos+zns+zcs;
    if (total>outsize){free(zl);free(zo);free(zn);free(zc);return -1;}

    uint8_t* p=(uint8_t*)outbuf;
    memcpy(p,&hdr,sizeof(hdr)); p+=sizeof(hdr);
    memcpy(p,boffs.data(),nb*sizeof(BlockOffsets)); p+=nb*sizeof(BlockOffsets);
    memcpy(p,zl,zls);p+=zls;
    memcpy(p,zo,zos);p+=zos;
    memcpy(p,zn,zns);p+=zns;
    memcpy(p,zc,zcs);
    free(zl);free(zo);free(zn);free(zc);
    return (int64_t)total;
}

int64_t lzbench_aceapex_decompress(char* inbuf, size_t insize,
                                    char* outbuf, size_t outsize,
                                    codec_options_t* opts)
{
    const uint8_t* p=(const uint8_t*)inbuf;
    AetHeader hdr; memcpy(&hdr,p,sizeof(hdr));
    if (memcmp(hdr.magic,"ACEPX2\0\0",8)!=0) return -1;
    if (hdr.orig_size>outsize) return -1;
    p+=sizeof(hdr);
    std::vector<BlockOffsets> boffs(hdr.num_blocks);
    memcpy(boffs.data(),p,hdr.num_blocks*sizeof(BlockOffsets));
    p+=hdr.num_blocks*sizeof(BlockOffsets);
    uint8_t* zl=(uint8_t*)malloc(hdr.zlit_sz); memcpy(zl,p,hdr.zlit_sz); p+=hdr.zlit_sz;
    uint8_t* zo=(uint8_t*)malloc(hdr.zoff_sz); memcpy(zo,p,hdr.zoff_sz); p+=hdr.zoff_sz;
    uint8_t* zn=(uint8_t*)malloc(hdr.zlen_sz); memcpy(zn,p,hdr.zlen_sz); p+=hdr.zlen_sz;
    uint8_t* zc=(uint8_t*)malloc(hdr.zcmd_sz); memcpy(zc,p,hdr.zcmd_sz);
    size_t os=*(uint64_t*)zo,ns=*(uint64_t*)zn,cs=*(uint64_t*)zc;
    size_t ls=0; uint8_t* l=lit_decompress(zl,hdr.zlit_sz,ls);
    uint8_t* o=(uint8_t*)malloc(os); fse_chunked_decomp(zo,os,o);
    uint8_t* n=(uint8_t*)malloc(ns); fse_chunked_decomp(zn,ns,n);
    uint8_t* c=(uint8_t*)malloc(cs); fse_chunked_decomp(zc,cs,c);
    free(zl);free(zo);free(zn);free(zc);
    int nth=opts&&opts->threads>0?opts->threads:1;
    parallel_decode(l,o,n,c,boffs.data(),hdr.num_blocks,
                    (uint8_t*)outbuf,hdr.orig_size,hdr.block_size,nth);
    free(l);free(o);free(n);free(c);
    return (int64_t)hdr.orig_size;
}

// ACEAPEX Streaming variant — 23MB RAM, ratio 3.004x
static const size_t ACE_HISTORY = 1*1024*1024;
static const size_t ACE_CHUNK   = 4*1024*1024;

char* lzbench_aceapex_stream_init(size_t insize, size_t level, size_t threads) {
    AcepxState* s = (AcepxState*)malloc(sizeof(AcepxState));
    s->threads = 1; s->level = level > 0 ? (int)level : 2;
    return (char*)s;
}

int64_t lzbench_aceapex_stream_compress(char* inbuf, size_t insize,
                                         char* outbuf, size_t outsize,
                                         codec_options_t* opts)
{
    AcepxState* s = (AcepxState*)opts->work_mem;
    if (!s) return -1;
    int lvl = s->level;
    size_t num_chunks = (insize + ACE_CHUNK - 1) / ACE_CHUNK;

    // Setup output buffer writer
    uint8_t* op = (uint8_t*)outbuf;
    AetHeader hdr; memset(&hdr,0,sizeof(hdr));
    memcpy(hdr.magic,"ACEPX2\0\0",8);
    hdr.version=2; hdr.orig_size=insize;
    hdr.block_size=(uint32_t)ACE_CHUNK; hdr.num_blocks=(uint32_t)num_chunks;
    memcpy(op,&hdr,sizeof(hdr)); op+=sizeof(hdr);
    std::vector<BlockOffsets> boffs(num_chunks);
    memset(boffs.data(),0,num_chunks*sizeof(BlockOffsets));
    uint8_t* boffs_ptr=op; op+=num_chunks*sizeof(BlockOffsets);

    // Hash table
    uint32_t hm=(1u<<17)-1, cm=(1u<<20)-1;
    ThreadHashTable* ht=(ThreadHashTable*)calloc(1,sizeof(ThreadHashTable));
    ht->pos=(int32_t*)calloc(hm+1,sizeof(int32_t));
    ht->epoch=(uint32_t*)calloc(hm+1,sizeof(uint32_t));
    ht->chain=(int32_t*)malloc(((size_t)cm+1)*sizeof(int32_t));
    memset(ht->chain,-1,((size_t)cm+1)*sizeof(int32_t));
    ht->hash_mask=hm; ht->chain_mask=cm; ht->max_attempts=(lvl>=2)?32:4;

    uint8_t* window=(uint8_t*)malloc(ACE_HISTORY+ACE_CHUNK);
    size_t history_len=0, total_read=0, co=0;

    for(size_t ci=0;ci<num_chunks;ci++) {
        size_t to_read=std::min(ACE_CHUNK,insize-total_read);
        memcpy(window+history_len,(uint8_t*)inbuf+total_read,to_read);
        size_t wlen=history_len+to_read; total_read+=to_read;

        size_t buf_sz=to_read*2+65536;
        BlockResult res;
        res.lit_buf=(uint8_t*)malloc(buf_sz); res.off_buf=(uint8_t*)malloc(buf_sz);
        res.len_buf=(uint8_t*)malloc(buf_sz); res.cmd_buf=(uint8_t*)malloc(buf_sz);
        res.lit_size=res.off_size=res.len_size=res.cmd_size=res.overflow=0;
        compress_block(window,wlen,history_len,wlen,ht,&res);

        size_t zls,zos,zns,zcs;
        uint8_t* zl=lit_compress(res.lit_buf,res.lit_size,zls);
        uint8_t *zo,*zn,*zc;
        entropy_encode(res.lit_buf,res.lit_size,res.off_buf,res.off_size,
                       res.len_buf,res.len_size,res.cmd_buf,res.cmd_size,
                       zl,zls,zo,zos,zn,zns,zc,zcs);

        boffs[ci].lit_off=co; boffs[ci].lit_sz=zls;
        boffs[ci].off_off=co+zls; boffs[ci].off_sz=zos;
        boffs[ci].len_off=co+zls+zos; boffs[ci].len_sz=zns;
        boffs[ci].cmd_off=co+zls+zos+zns; boffs[ci].cmd_sz=zcs;
        co+=zls+zos+zns+zcs;

        if((op-(uint8_t*)outbuf+zls+zos+zns+zcs)>(ptrdiff_t)outsize){
            free(zl);free(zo);free(zn);free(zc);
            free(res.lit_buf);free(res.off_buf);free(res.len_buf);free(res.cmd_buf);
            free(window);free(ht->pos);free(ht->epoch);free(ht->chain);free(ht);
            return -1;
        }
        memcpy(op,zl,zls);op+=zls; memcpy(op,zo,zos);op+=zos;
        memcpy(op,zn,zns);op+=zns; memcpy(op,zc,zcs);op+=zcs;
        free(zl);free(zo);free(zn);free(zc);
        free(res.lit_buf);free(res.off_buf);free(res.len_buf);free(res.cmd_buf);

        size_t keep=std::min(ACE_HISTORY,wlen);
        memmove(window,window+wlen-keep,keep);
        history_len=keep; ht->cur_epoch++;
    }
    free(window);free(ht->pos);free(ht->epoch);free(ht->chain);free(ht);

    hdr.zlit_sz=co; hdr.zoff_sz=0; hdr.zlen_sz=0; hdr.zcmd_sz=0;
    memcpy(outbuf,&hdr,sizeof(hdr));
    memcpy(boffs_ptr,boffs.data(),num_chunks*sizeof(BlockOffsets));
    return (int64_t)(op-(uint8_t*)outbuf);
}

int64_t lzbench_aceapex_stream_decompress(char* inbuf, size_t insize,
                                           char* outbuf, size_t outsize,
                                           codec_options_t* opts)
{
    const uint8_t* p=(const uint8_t*)inbuf;
    AetHeader hdr; memcpy(&hdr,p,sizeof(hdr));
    if (memcmp(hdr.magic,"ACEPX2\0\0",8)!=0) return -1;
    if (hdr.orig_size>outsize) return -1;
    p+=sizeof(hdr);
    uint32_t nb=hdr.num_blocks;
    std::vector<BlockOffsets> boffs(nb);
    memcpy(boffs.data(),p,nb*sizeof(BlockOffsets)); p+=nb*sizeof(BlockOffsets);
    const uint8_t* data=p;
    size_t out_pos=0;
    for(uint32_t i=0;i<nb;i++) {
        uint8_t* zl=(uint8_t*)data+boffs[i].lit_off;
        uint8_t* zo=(uint8_t*)data+boffs[i].off_off;
        uint8_t* zn=(uint8_t*)data+boffs[i].len_off;
        uint8_t* zc=(uint8_t*)data+boffs[i].cmd_off;
        size_t os=*(uint64_t*)zo,ns=*(uint64_t*)zn,cs=*(uint64_t*)zc;
        size_t ls=0; uint8_t* l=lit_decompress(zl,boffs[i].lit_sz,ls);
        uint8_t* o=(uint8_t*)malloc(os); fse_chunked_decomp(zo,os,o);
        uint8_t* n=(uint8_t*)malloc(ns); fse_chunked_decomp(zn,ns,n);
        uint8_t* c=(uint8_t*)malloc(cs); fse_chunked_decomp(zc,cs,c);
        size_t bsize=std::min((size_t)hdr.block_size,hdr.orig_size-out_pos);
        decompress_streams((uint8_t*)outbuf+out_pos,bsize,l,ls,o,os,n,ns,c,cs);
        out_pos+=bsize; free(l);free(o);free(n);free(c);
    }
    return (int64_t)hdr.orig_size;
}

// ACEPX3 lzbench integration
struct V3Header3 { char magic[8]; uint32_t version,flags; uint64_t orig_size; uint32_t num_chunks,chunk_size; };
struct V3Chunk3  { uint32_t magic,flags; uint64_t raw_size,lit_size,off_size,len_size,cmd_size; };
static const size_t V3H=1*1024*1024, V3C=4*1024*1024;

char* lzbench_aceapex3_init(size_t insize, size_t level, size_t threads) {
    if(level==0) level=2;
    return lzbench_aceapex_init(insize, level, threads);
}

int64_t lzbench_aceapex3_compress(char* inbuf, size_t insize, char* outbuf, size_t outsize, codec_options_t* opts) {
    AcepxState* s=(AcepxState*)opts->work_mem; if(!s) return -1;
    uint32_t nc=(uint32_t)((insize+V3C-1)/V3C);
    uint8_t* op=(uint8_t*)outbuf;
    V3Header3 fh; memcpy(fh.magic,"ACEPX3\0\0",8);
    fh.version=3;fh.flags=0;fh.orig_size=insize;fh.num_chunks=nc;fh.chunk_size=(uint32_t)V3C;
    memcpy(op,&fh,sizeof(fh)); op+=sizeof(fh);
    uint32_t hm=(1u<<17)-1,cm=(1u<<20)-1;
    ThreadHashTable* ht=(ThreadHashTable*)calloc(1,sizeof(ThreadHashTable));
    ht->pos=(int32_t*)calloc(hm+1,sizeof(int32_t));
    ht->epoch=(uint32_t*)calloc(hm+1,sizeof(uint32_t));
    ht->chain=(int32_t*)malloc(((size_t)cm+1)*sizeof(int32_t));
    memset(ht->chain,-1,((size_t)cm+1)*sizeof(int32_t));
    ht->hash_mask=hm;ht->chain_mask=cm;ht->max_attempts=(s->level>=2)?32:4;
    uint8_t* window=(uint8_t*)malloc(V3H+V3C);
    size_t hl=0,tr=0;
    for(uint32_t ci=0;ci<nc;ci++){
        size_t rd=std::min(V3C,insize-tr);
        memcpy(window+hl,(uint8_t*)inbuf+tr,rd); size_t wl=hl+rd; tr+=rd;
        size_t bsz=rd*2+65536;
        BlockResult r; r.lit_buf=(uint8_t*)malloc(bsz);r.off_buf=(uint8_t*)malloc(bsz);
        r.len_buf=(uint8_t*)malloc(bsz);r.cmd_buf=(uint8_t*)malloc(bsz);
        r.lit_size=r.off_size=r.len_size=r.cmd_size=r.overflow=0;
        compress_block(window,wl,hl,wl,ht,&r);
        size_t zls,zos,zns,zcs; uint8_t*zl=lit_compress(r.lit_buf,r.lit_size,zls);
        uint8_t*zo,*zn,*zc;
        entropy_encode(r.lit_buf,r.lit_size,r.off_buf,r.off_size,r.len_buf,r.len_size,r.cmd_buf,r.cmd_size,zl,zls,zo,zos,zn,zns,zc,zcs);
        free(r.lit_buf);free(r.off_buf);free(r.len_buf);free(r.cmd_buf);
        V3Chunk3 ch; ch.magic=0x434B4E48;ch.flags=0;ch.raw_size=rd;
        ch.lit_size=zls;ch.off_size=zos;ch.len_size=zns;ch.cmd_size=zcs;
        size_t need=sizeof(ch)+zls+zos+zns+zcs;
        if((op-(uint8_t*)outbuf)+(ptrdiff_t)need>(ptrdiff_t)outsize){free(zl);free(zo);free(zn);free(zc);free(window);free(ht->pos);free(ht->epoch);free(ht->chain);free(ht);return -1;}
        memcpy(op,&ch,sizeof(ch));op+=sizeof(ch);
        memcpy(op,zl,zls);op+=zls;memcpy(op,zo,zos);op+=zos;
        memcpy(op,zn,zns);op+=zns;memcpy(op,zc,zcs);op+=zcs;
        free(zl);free(zo);free(zn);free(zc);
        size_t keep=std::min(V3H,wl); memmove(window,window+wl-keep,keep); hl=keep; ht->cur_epoch++;
    }
    free(window);free(ht->pos);free(ht->epoch);free(ht->chain);free(ht);
    return (int64_t)(op-(uint8_t*)outbuf);
}

struct V3DecJob {
    const uint8_t*zl,*zo,*zn,*zc;
    size_t zls,zos,zns,zcs;
    uint8_t* dst; size_t raw_size;
};
static void* v3_dec_worker(void* arg) {
    V3DecJob* j=(V3DecJob*)arg;
    size_t os=*(uint64_t*)j->zo,ns=*(uint64_t*)j->zn,cs=*(uint64_t*)j->zc;
    size_t ls=0; uint8_t*l=lit_decompress(j->zl,j->zls,ls);
    uint8_t*o=(uint8_t*)malloc(os); fse_chunked_decomp((uint8_t*)j->zo,os,o);
    uint8_t*n=(uint8_t*)malloc(ns); fse_chunked_decomp((uint8_t*)j->zn,ns,n);
    uint8_t*c=(uint8_t*)malloc(cs); fse_chunked_decomp((uint8_t*)j->zc,cs,c);
    decompress_streams(j->dst,j->raw_size,l,ls,o,os,n,ns,c,cs);
    free(l);free(o);free(n);free(c);
    return nullptr;
}
int64_t lzbench_aceapex3_decompress(char* inbuf, size_t insize, char* outbuf, size_t outsize, codec_options_t* opts) {
    const uint8_t* p=(const uint8_t*)inbuf;
    V3Header3 fh; memcpy(&fh,p,sizeof(fh)); p+=sizeof(fh);
    if(memcmp(fh.magic,"ACEPX3\0\0",8)!=0) return -1;
    if(fh.orig_size>outsize) return -1;
    int nth=opts->threads>0?opts->threads:8;
    std::vector<V3DecJob> jobs(fh.num_chunks);
    std::vector<size_t> offsets(fh.num_chunks);
    size_t out_pos=0;
    // Parse all chunk headers first
    for(uint32_t ci=0;ci<fh.num_chunks;ci++){
        V3Chunk3 ch; memcpy(&ch,p,sizeof(ch)); p+=sizeof(ch);
        if(ch.magic!=0x434B4E48) return -1;
        jobs[ci]={p,p+ch.lit_size,p+ch.lit_size+ch.off_size,
                  p+ch.lit_size+ch.off_size+ch.len_size,
                  ch.lit_size,ch.off_size,ch.len_size,ch.cmd_size,
                  (uint8_t*)outbuf+out_pos,ch.raw_size};
        p+=ch.lit_size+ch.off_size+ch.len_size+ch.cmd_size;
        out_pos+=ch.raw_size;
    }
    // Parallel decode
    std::vector<pthread_t> pts(nth);
    size_t ci=0;
    while(ci<fh.num_chunks) {
        size_t batch=std::min((size_t)nth,fh.num_chunks-ci);
        for(size_t t=0;t<batch;t++) pthread_create(&pts[t],nullptr,v3_dec_worker,&jobs[ci+t]);
        for(size_t t=0;t<batch;t++) pthread_join(pts[t],nullptr);
        ci+=batch;
    }
    return (int64_t)fh.orig_size;
}

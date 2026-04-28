#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <vector>
#include <pthread.h>
#include <algorithm>
#include <zstd.h>

static size_t LIT_compress(void* dst, size_t dstCap, const void* src, size_t srcSize) {
    return ZSTD_compress(dst, dstCap, src, srcSize, 1);
}
static size_t LIT_decompress(void* dst, size_t dstCap, const void* src, size_t srcSize) {
    return ZSTD_decompress(dst, dstCap, src, srcSize);
}
static size_t LIT_compressBound(size_t srcSize) {
    return ZSTD_compressBound(srcSize);
}
static unsigned LIT_isError(size_t code) {
    return ZSTD_isError(code);
}

struct FseJob { const uint8_t*in; size_t isz; uint8_t*out; size_t osz; };
static void* fse_worker(void* a){
    FseJob* j=(FseJob*)a;
    size_t r=LIT_compress(j->out,LIT_compressBound(j->isz),j->in,j->isz);
    if(LIT_isError(r)||r==0){ memcpy(j->out,j->in,j->isz); j->osz=j->isz|(uint64_t(1)<<62); }
    else j->osz=r;
    return nullptr;
}

static uint8_t* fse_comp(const uint8_t* src,size_t sz,size_t& out_sz,int nc=16){
    if(sz<(size_t)nc*65536){
        size_t cap=LIT_compressBound(sz)+16;
        uint8_t* buf=(uint8_t*)malloc(cap);
        *(uint64_t*)buf=sz;
        size_t csz=LIT_compress(buf+8,cap-8,src,sz);
        if(LIT_isError(csz)||csz==0){ *(uint64_t*)buf=sz|(uint64_t(1)<<63); memcpy(buf+8,src,sz); out_sz=sz+8; }
        else out_sz=csz+8;
        return buf;
    }
    size_t csz=(sz+nc-1)/nc;
    std::vector<FseJob> jobs(nc);
    std::vector<uint8_t*> bufs(nc);
    for(int i=0;i<nc;i++){
        size_t off=i*csz, isz=std::min(csz,sz-off);
        bufs[i]=(uint8_t*)malloc(LIT_compressBound(isz)+8);
        jobs[i]={src+off,isz,bufs[i],0};
    }
    std::vector<pthread_t> pts(nc);
    for(int i=0;i<nc;i++) pthread_create(&pts[i],nullptr,fse_worker,&jobs[i]);
    for(int i=0;i<nc;i++) pthread_join(pts[i],nullptr);
    size_t hdr=8+nc*16;
    size_t total=hdr;
    for(int i=0;i<nc;i++) total+=(jobs[i].osz&~(uint64_t(1)<<62));
    uint8_t* res=(uint8_t*)malloc(total);
    *(uint32_t*)res=(uint32_t)nc; *(uint32_t*)(res+4)=0;
    uint64_t* rsz=(uint64_t*)(res+8);
    uint64_t* csz2=(uint64_t*)(res+8+nc*8);
    uint8_t* p=res+hdr;
    for(int i=0;i<nc;i++){
        rsz[i]=jobs[i].isz;
        size_t actual=jobs[i].osz&~(uint64_t(1)<<62);
        csz2[i]=(jobs[i].osz>>62)&1 ? actual|(uint64_t(1)<<63) : actual;
        memcpy(p,bufs[i],actual); p+=actual; free(bufs[i]);
    }
    out_sz=total; return res;
}

static uint8_t* fse_decomp(const uint8_t* src,size_t sz,size_t& orig_sz){
    uint32_t nc=*(const uint32_t*)src;
    if(nc==0||nc>256){
        orig_sz=*(const uint64_t*)src&~(uint64_t(1)<<63);
        int raw=(*(const uint64_t*)src>>63)&1;
        uint8_t* out=(uint8_t*)malloc(orig_sz);
        if(raw) memcpy(out,src+8,orig_sz);
        else LIT_decompress(out,orig_sz,src+8,sz-8);
        return out;
    }
    const uint64_t* rsz=(const uint64_t*)(src+8);
    const uint64_t* csz2=(const uint64_t*)(src+8+nc*8);
    orig_sz=0; for(uint32_t i=0;i<nc;i++) orig_sz+=rsz[i];
    uint8_t* out=(uint8_t*)malloc(orig_sz);
    const uint8_t* p=src+8+nc*16;
    size_t off=0;
    for(uint32_t i=0;i<nc;i++){
        size_t raw_sz=rsz[i];
        uint64_t c=csz2[i];
        int is_raw=(c>>63)&1;
        size_t comp_sz=c&~(uint64_t(1)<<63);
        if(is_raw) memcpy(out+off,p,raw_sz);
        else LIT_decompress(out+off,raw_sz,p,comp_sz);
        off+=raw_sz; p+=comp_sz;
    }
    return out;
}

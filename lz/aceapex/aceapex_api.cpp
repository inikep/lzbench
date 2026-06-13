#define ACEAPEX_NO_MAIN
#include "ax_align.h"
#include "aceapex_main.cpp"
#include "aceapex.h"
#include <vector>

size_t aceapex_compress_bound(size_t src_size) {
    // Worst case: incompressible data + header overhead
    return src_size + src_size/8 + 1024;
}

int64_t aceapex_compress(
    const void* src, size_t src_size,
    void*       dst, size_t dst_capacity,
    int         level,
    int         threads)
{
    if (!src || !dst) return ACEAPEX_ERR_DATA;
    if (threads <= 0) threads = 8;
    if (level <= 0)   level   = 2;

    std::vector<BlockOffsets> boffs;
    uint8_t *rl,*ro,*rn,*rc;
    size_t tl,to,tn,tc,nb;
    if (!encode_file((const uint8_t*)src,src_size,threads,level,
                     boffs,rl,tl,ro,to,rn,tn,rc,tc,nb))
        return ACEAPEX_ERR_MEMORY;

    size_t zls,zos,zns,zcs;
    uint8_t *zl,*zo,*zn,*zc;
    zl=lit_compress(rl,tl,zls);
    entropy_encode(rl,tl,ro,to,rn,tn,rc,tc,
                   zl,zls,zo,zos,zn,zns,zc,zcs);
    free(rl);free(ro);free(rn);free(rc);

    AetHeader hdr;
    memcpy(hdr.magic,"ACEPX2\0\0",8);
    hdr.version=2; hdr.orig_size=src_size;
    hdr.block_size=BLOCK_SIZE; hdr.num_blocks=nb;
    uint64_t hv=OUR_CHECKSUM(src,src_size);
    memcpy(hdr.xxhash,&hv,8);
    hdr.zlit_sz=zls;hdr.zoff_sz=zos;
    hdr.zlen_sz=zns;hdr.zcmd_sz=zcs;

    size_t total=sizeof(hdr)+nb*sizeof(BlockOffsets)
                 +zls+zos+zns+zcs;
    if (total>dst_capacity) {
        free(zl);free(zo);free(zn);free(zc);
        return ACEAPEX_ERR_BUFFER;
    }

    uint8_t* p=(uint8_t*)dst;
    memcpy(p,&hdr,sizeof(hdr)); p+=sizeof(hdr);
    memcpy(p,boffs.data(),nb*sizeof(BlockOffsets));
    p+=nb*sizeof(BlockOffsets);
    memcpy(p,zl,zls);p+=zls;
    memcpy(p,zo,zos);p+=zos;
    memcpy(p,zn,zns);p+=zns;
    memcpy(p,zc,zcs);
    free(zl);free(zo);free(zn);free(zc);
    return (int64_t)total;
}

int64_t aceapex_decompress(
    const void* src, size_t src_size,
    void*       dst, size_t dst_capacity,
    int         threads)
{
    const uint8_t* p=(const uint8_t*)src;
    AetHeader hdr; memcpy(&hdr,p,sizeof(hdr));
    if (memcmp(hdr.magic,"ACEPX2\0\0",8)!=0) return ACEAPEX_ERR_DATA;
    if (hdr.orig_size>dst_capacity) return ACEAPEX_ERR_BUFFER;
    p+=sizeof(hdr);
    std::vector<BlockOffsets> boffs(hdr.num_blocks);
    memcpy(boffs.data(),p,hdr.num_blocks*sizeof(BlockOffsets));
    p+=hdr.num_blocks*sizeof(BlockOffsets);
    uint8_t* zl=(uint8_t*)malloc(hdr.zlit_sz);
    uint8_t* zo=(uint8_t*)malloc(hdr.zoff_sz);
    uint8_t* zn=(uint8_t*)malloc(hdr.zlen_sz);
    uint8_t* zc=(uint8_t*)malloc(hdr.zcmd_sz);
    if(!zl||!zo||!zn||!zc){free(zl);free(zo);free(zn);free(zc);return ACEAPEX_ERR_MEMORY;}
    memcpy(zl,p,hdr.zlit_sz); p+=hdr.zlit_sz;
    memcpy(zo,p,hdr.zoff_sz); p+=hdr.zoff_sz;
    memcpy(zn,p,hdr.zlen_sz); p+=hdr.zlen_sz;
    memcpy(zc,p,hdr.zcmd_sz);
    size_t os=AX_read64((zo)),ns=AX_read64((zn)),cs=AX_read64((zc));
    size_t ls=0; uint8_t* l=lit_decompress(zl,hdr.zlit_sz,ls);
    if(!l){free(zl);free(zo);free(zn);free(zc);return ACEAPEX_ERR_MEMORY;}
    uint8_t* o=(uint8_t*)malloc(os);
    uint8_t* n=(uint8_t*)malloc(ns);
    uint8_t* c=(uint8_t*)malloc(cs);
    if(!o||!n||!c){free(o);free(n);free(c);free(zl);free(zo);free(zn);free(zc);return ACEAPEX_ERR_MEMORY;}
    fse_chunked_decomp(zo,os,o); fse_chunked_decomp(zn,ns,n); fse_chunked_decomp(zc,cs,c);
    free(zl);free(zo);free(zn);free(zc);
    parallel_decode(l,o,n,c,boffs.data(),hdr.num_blocks,
                    (uint8_t*)dst,hdr.orig_size,hdr.block_size,threads);
    free(l);free(o);free(n);free(c);
    return (int64_t)hdr.orig_size;
}


// ---- stream-level decode export for the CUDA decoder (lz/aceapex/cuda) ----
extern "C" {
typedef struct {
    uint8_t *lit, *off, *len, *cmd;
    uint64_t lit_sz, off_sz, len_sz, cmd_sz;
    void    *boffs_vec;
    const void *boffs;
    size_t   num_blocks;
    uint32_t block_size;
    uint64_t orig_size;
} aceapex_streams_t;

int aceapex_decode_streams(const void* src, size_t src_size, aceapex_streams_t* out)
{
    if (!src || !out || src_size < sizeof(AetHeader)) return -1;
    const uint8_t* p=(const uint8_t*)src;
    AetHeader hdr; memcpy(&hdr,p,sizeof(hdr));
    if (memcmp(hdr.magic,"ACEPX2\0\0",8)!=0) return -1;
    p+=sizeof(hdr);
    std::vector<BlockOffsets>* bv=new std::vector<BlockOffsets>(hdr.num_blocks);
    memcpy(bv->data(),p,hdr.num_blocks*sizeof(BlockOffsets));
    p+=hdr.num_blocks*sizeof(BlockOffsets);
    uint8_t* zl=(uint8_t*)malloc(hdr.zlit_sz);
    uint8_t* zo=(uint8_t*)malloc(hdr.zoff_sz);
    uint8_t* zn=(uint8_t*)malloc(hdr.zlen_sz);
    uint8_t* zc=(uint8_t*)malloc(hdr.zcmd_sz);
    if(!zl||!zo||!zn||!zc){free(zl);free(zo);free(zn);free(zc);delete bv;return -2;}
    memcpy(zl,p,hdr.zlit_sz); p+=hdr.zlit_sz;
    memcpy(zo,p,hdr.zoff_sz); p+=hdr.zoff_sz;
    memcpy(zn,p,hdr.zlen_sz); p+=hdr.zlen_sz;
    memcpy(zc,p,hdr.zcmd_sz);
    size_t os=*(uint64_t*)zo, ns=*(uint64_t*)zn, cs=*(uint64_t*)zc;
    size_t ls=0; uint8_t* l=lit_decompress(zl,hdr.zlit_sz,ls);
    if(!l){free(zl);free(zo);free(zn);free(zc);delete bv;return -2;}
    uint8_t* o=(uint8_t*)malloc(os);
    uint8_t* n=(uint8_t*)malloc(ns);
    uint8_t* c=(uint8_t*)malloc(cs);
    if(!o||!n||!c){free(o);free(n);free(c);free(l);free(zl);free(zo);free(zn);free(zc);delete bv;return -2;}
    fse_chunked_decomp(zo,os,o); fse_chunked_decomp(zn,ns,n); fse_chunked_decomp(zc,cs,c);
    free(zl);free(zo);free(zn);free(zc);
    out->lit=l; out->off=o; out->len=n; out->cmd=c;
    out->lit_sz=ls; out->off_sz=os; out->len_sz=ns; out->cmd_sz=cs;
    out->boffs_vec=(void*)bv; out->boffs=(const void*)bv->data();
    out->num_blocks=hdr.num_blocks; out->block_size=hdr.block_size;
    out->orig_size=hdr.orig_size;
    return 0;
}

void aceapex_streams_free(aceapex_streams_t* s)
{
    if(!s) return;
    free(s->lit); free(s->off); free(s->len); free(s->cmd);
    delete (std::vector<BlockOffsets>*)s->boffs_vec;
    s->lit=s->off=s->len=s->cmd=nullptr; s->boffs_vec=nullptr; s->boffs=nullptr;
}
} // extern "C"

// =============================================================================
// lz/aceapex/cuda/aceapex_cuda.cu
// GPU LZ match decode for the ACEAPEX .aet format (lzbench codec variant).
// Dependency: CUDA Runtime only. Entropy is decoded by the existing in-tree
// CPU code (aceapex_decode_streams); this module executes the LZ match phase
// on the GPU (warp per block; blocks are independent by format design) and
// copies the result back to host. Kernel lineage: full_gpu_decode_v3
// (bit-perfect on enwik9 / silesia.tar / FASTQ NA12878; see
// github.com/yasha1971-coder/aceapex).
// =============================================================================
#ifdef BENCH_HAS_CUDA
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "aceapex_cuda.h"

#define ACK0(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"aceapex_cuda CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 0;} }while(0)

#pragma pack(push,1)
struct CgBlockOffsets {
    uint64_t lit_off, off_off, len_off, cmd_off;
    uint64_t lit_sz,  off_sz,  len_sz,  cmd_sz;
};
#pragma pack(pop)

__device__ __forceinline__ uint32_t d_read_varint(const uint8_t* buf, uint32_t& p, uint32_t limit){
    uint32_t val=0, shift=0;
    while(p<limit){ uint8_t b=buf[p++]; val|=(uint32_t)(b&0x7F)<<shift; if(!(b&0x80)) return val; shift+=7; }
    return val;
}

__global__ void cg_k_decode(const uint8_t* __restrict__ LIT, const uint8_t* __restrict__ OFF,
                            const uint8_t* __restrict__ LEN, const uint8_t* __restrict__ CMD,
                            const CgBlockOffsets* __restrict__ boffs, uint32_t num_blocks,
                            uint64_t orig_size, uint32_t block_size, uint8_t* __restrict__ out)
{
    uint32_t gw   = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    uint32_t lane = threadIdx.x & 31;
    if (gw >= num_blocks) return;
    CgBlockOffsets bo = boffs[gw];
    const uint8_t* lit = LIT + bo.lit_off;
    const uint8_t* off = OFF + bo.off_off;
    const uint8_t* len = LEN + bo.len_off;
    const uint8_t* cmd = CMD + bo.cmd_off;
    uint64_t base = (uint64_t)gw * block_size;
    uint64_t rem  = orig_size - base;
    uint32_t dst_size = (uint32_t)((rem < (uint64_t)block_size) ? rem : (uint64_t)block_size);
    uint8_t* dst = out + base;
    uint32_t lp=0, op=0, np=0, cp=0;
    uint32_t rep[4]={1,2,4,8};
    uint32_t out_pos=0;
    uint32_t cmd_sz=(uint32_t)bo.cmd_sz, lit_sz=(uint32_t)bo.lit_sz;
    uint32_t off_sz=(uint32_t)bo.off_sz, len_sz=(uint32_t)bo.len_sz;
    while (out_pos < dst_size) {
        uint32_t type=2, l=0, aux=0;
        if (lane==0) {
            while (cp < cmd_sz) {
                uint8_t c = cmd[cp++];
                if (c==0xFF){ rep[0]=1;rep[1]=2;rep[2]=4;rep[3]=8; continue; }
                if (c<0x80){
                    l=(uint32_t)c+1;
                    if (lp+l>lit_sz || out_pos+l>dst_size) { type=2; break; }
                    type=0; aux=lp; lp+=l;
                } else if ((c&0xC0)==0x80){
                    uint32_t ri=(c>>4)&3, lv=c&0x0F;
                    if (lv==0x0F) lv += d_read_varint(len,np,len_sz);
                    l=lv+6;
                    uint32_t dist=rep[ri];
                    if (ri>0){ for(int i=(int)ri;i>0;i--) rep[i]=rep[i-1]; rep[0]=dist; }
                    if (!dist || dist>out_pos || out_pos+l>dst_size) { type=2; break; }
                    type=1; aux=dist;
                } else {
                    uint32_t lv=(c==0xFE)? d_read_varint(len,np,len_sz) : (uint32_t)(c&0x3F);
                    l=lv+6;
                    uint32_t dist=d_read_varint(off,op,off_sz);
                    rep[3]=rep[2];rep[2]=rep[1];rep[1]=rep[0];rep[0]=dist;
                    if (!dist || dist>out_pos || out_pos+l>dst_size) { type=2; break; }
                    type=1; aux=dist;
                }
                break;
            }
        }
        type = __shfl_sync(0xffffffffu, type, 0);
        l    = __shfl_sync(0xffffffffu, l,    0);
        aux  = __shfl_sync(0xffffffffu, aux,  0);
        if (type==2) break;
        if (type==0) {
            for (uint32_t i=lane; i<l; i+=32) dst[out_pos+i] = lit[aux+i];
        } else {
            uint32_t src = out_pos - aux;
            if (aux >= l) { for (uint32_t i=lane; i<l; i+=32) dst[out_pos+i] = dst[src+i]; }
            else          { for (uint32_t i=lane; i<l; i+=32) dst[out_pos+i] = dst[src + (i % aux)]; }
        }
        __syncwarp();
        out_pos += l;
    }
}

// ---- cached, grow-only device context (lzbench loops the call) --------------
struct CgCtx {
    uint8_t *dLit=nullptr,*dOff=nullptr,*dLen=nullptr,*dCmd=nullptr,*dOut=nullptr;
    size_t cLit=0,cOff=0,cLen=0,cCmd=0,cOut=0;
    CgBlockOffsets* dBO=nullptr; size_t cBO=0;
};
static CgCtx g_cg;

static int cg_grow(void** p, size_t* cap, size_t need){
    if(need<=*cap) return 1;
    if(*p) cudaFree(*p);
    *p=nullptr; *cap=0;
    if(cudaMalloc(p,need)!=cudaSuccess){ fprintf(stderr,"aceapex_cuda: cudaMalloc %zu failed\n",(size_t)need); return 0; }
    *cap=need; return 1;
}

extern "C" int aceapex_cg_available(void){
    int n=0; if(cudaGetDeviceCount(&n)!=cudaSuccess) return 0;
    return n>0?1:0;
}

extern "C" void aceapex_cg_release(void){
    CgCtx&c=g_cg;
    if(c.dLit)cudaFree(c.dLit); if(c.dOff)cudaFree(c.dOff);
    if(c.dLen)cudaFree(c.dLen); if(c.dCmd)cudaFree(c.dCmd);
    if(c.dOut)cudaFree(c.dOut); if(c.dBO)cudaFree(c.dBO);
    c=CgCtx();
}

extern "C" int64_t aceapex_cg_match_decode(const aceapex_streams_t* s, void* dst, size_t dst_capacity)
{
    if(!s || !dst || s->orig_size>dst_capacity) return 0;
    static int timing=-1;
    if(timing<0){ const char* e=getenv("ACEAPEX_CUDA_TIMING"); timing=(e&&*e=='1')?1:0; }
    CgCtx& c=g_cg;
    if(!cg_grow((void**)&c.dLit,&c.cLit,s->lit_sz?s->lit_sz:1)) return 0;
    if(!cg_grow((void**)&c.dOff,&c.cOff,s->off_sz?s->off_sz:1)) return 0;
    if(!cg_grow((void**)&c.dLen,&c.cLen,s->len_sz?s->len_sz:1)) return 0;
    if(!cg_grow((void**)&c.dCmd,&c.cCmd,s->cmd_sz?s->cmd_sz:1)) return 0;
    if(!cg_grow((void**)&c.dOut,&c.cOut,s->orig_size)) return 0;
    if(!cg_grow((void**)&c.dBO ,&c.cBO ,s->num_blocks*sizeof(CgBlockOffsets))) return 0;

    cudaEvent_t e0,e1,e2,e3;
    if(timing==1){ cudaEventCreate(&e0);cudaEventCreate(&e1);cudaEventCreate(&e2);cudaEventCreate(&e3); cudaEventRecord(e0); }
    ACK0(cudaMemcpy(c.dLit,s->lit,s->lit_sz,cudaMemcpyHostToDevice));
    ACK0(cudaMemcpy(c.dOff,s->off,s->off_sz,cudaMemcpyHostToDevice));
    ACK0(cudaMemcpy(c.dLen,s->len,s->len_sz,cudaMemcpyHostToDevice));
    ACK0(cudaMemcpy(c.dCmd,s->cmd,s->cmd_sz,cudaMemcpyHostToDevice));
    ACK0(cudaMemcpy(c.dBO ,s->boffs,s->num_blocks*sizeof(CgBlockOffsets),cudaMemcpyHostToDevice));
    if(timing==1) cudaEventRecord(e1);
    const int TPB=128;
    uint32_t grid=(uint32_t)(((uint64_t)s->num_blocks*32 + TPB-1)/TPB);
    cg_k_decode<<<grid,TPB>>>(c.dLit,c.dOff,c.dLen,c.dCmd,c.dBO,(uint32_t)s->num_blocks,
                              s->orig_size,s->block_size,c.dOut);
    if(timing==1) cudaEventRecord(e2);
    ACK0(cudaMemcpy(dst,c.dOut,s->orig_size,cudaMemcpyDeviceToHost));
    if(timing==1){ cudaEventRecord(e3); cudaEventSynchronize(e3); }
    ACK0(cudaDeviceSynchronize());
    ACK0(cudaGetLastError());
    if(timing==1){
        float a=0,b=0,d=0,t=0;
        cudaEventElapsedTime(&a,e0,e1); cudaEventElapsedTime(&b,e1,e2);
        cudaEventElapsedTime(&d,e2,e3); cudaEventElapsedTime(&t,e0,e3);
        fprintf(stderr,"[aceapex_cuda] H2D %.2fms | LZ kernel %.2fms | D2H %.2fms | GPU total %.2fms"
                       " (kernel device-resident: %.1f GB/s)\n",
                a,b,d,t, s->orig_size/(b*1e-3)/1e9);
        timing=2;
    }
    return (int64_t)s->orig_size;
}
#endif // BENCH_HAS_CUDA

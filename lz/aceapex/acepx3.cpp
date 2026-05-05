#define ACEAPEX_NO_MAIN
#include "aceapex_main.cpp"

// ACEPX3 File Header
struct V3Header {
    char     magic[8];   // "ACEPX3\0\0"
    uint32_t version;    // 3
    uint32_t flags;      // 0
    uint64_t orig_size;
    uint32_t num_chunks;
    uint32_t chunk_size;
};

// ACEPX3 Chunk Header
struct V3Chunk {
    uint32_t magic;      // 0x434B4E48 'CHNK'
    uint32_t flags;
    uint64_t raw_size;
    uint64_t lit_size;
    uint64_t off_size;
    uint64_t len_size;
    uint64_t cmd_size;
};

static const size_t V3_HISTORY = 1*1024*1024;
static const size_t V3_CHUNK   = 4*1024*1024;

int acepx3_encode(const char* in_path, const char* out_path, int level, int threads) {
    FILE* fin = fopen(in_path,"rb");
    if (!fin) { fprintf(stderr,"Cannot open: %s\n",in_path); return 1; }
    fseek(fin,0,SEEK_END); uint64_t total_size=ftell(fin); fseek(fin,0,SEEK_SET);

    uint32_t num_chunks = (uint32_t)((total_size + V3_CHUNK - 1) / V3_CHUNK);
    fprintf(stderr,"[*] ACEPX3 encode: %s (%.2f MB) chunks=%u\n",
            in_path, total_size/1e6, num_chunks);

    FILE* fout = fopen(out_path,"wb");
    if (!fout) { fclose(fin); return 1; }

    // Write file header
    V3Header fhdr;
    memcpy(fhdr.magic,"ACEPX3\0\0",8);
    fhdr.version=3; fhdr.flags=0;
    fhdr.orig_size=total_size;
    fhdr.num_chunks=num_chunks;
    fhdr.chunk_size=(uint32_t)V3_CHUNK;
    fwrite(&fhdr,sizeof(fhdr),1,fout);

    // Setup hash table — persists across chunks
    uint32_t hm=(1u<<17)-1, cm=(1u<<20)-1;
    ThreadHashTable* ht=(ThreadHashTable*)calloc(1,sizeof(ThreadHashTable));
    ht->pos=(int32_t*)calloc(hm+1,sizeof(int32_t));
    ht->epoch=(uint32_t*)calloc(hm+1,sizeof(uint32_t));
    ht->chain=(int32_t*)malloc(((size_t)cm+1)*sizeof(int32_t));
    memset(ht->chain,-1,((size_t)cm+1)*sizeof(int32_t));
    ht->hash_mask=hm; ht->chain_mask=cm;
    ht->max_attempts=(level>=2)?32:4;

    uint8_t* window=(uint8_t*)malloc(V3_HISTORY+V3_CHUNK);
    size_t history_len=0, total_read=0;
    size_t total_compressed=0;

    for(uint32_t ci=0;ci<num_chunks;ci++) {
        size_t to_read=std::min(V3_CHUNK,size_t(total_size-total_read));
        fread(window+history_len,1,to_read,fin);
        size_t wlen=history_len+to_read; total_read+=to_read;

        // Compress chunk
        size_t buf_sz=to_read*2+65536;
        BlockResult res;
        res.lit_buf=(uint8_t*)malloc(buf_sz); res.off_buf=(uint8_t*)malloc(buf_sz);
        res.len_buf=(uint8_t*)malloc(buf_sz); res.cmd_buf=(uint8_t*)malloc(buf_sz);
        res.lit_size=res.off_size=res.len_size=res.cmd_size=res.overflow=0;
        compress_block(window,wlen,history_len,wlen,ht,&res);

        // Entropy encode
        size_t zls,zos,zns,zcs;
        uint8_t* zl=lit_compress(res.lit_buf,res.lit_size,zls);
        uint8_t *zo,*zn,*zc;
        entropy_encode(res.lit_buf,res.lit_size,res.off_buf,res.off_size,
                       res.len_buf,res.len_size,res.cmd_buf,res.cmd_size,
                       zl,zls,zo,zos,zn,zns,zc,zcs);
        free(res.lit_buf);free(res.off_buf);free(res.len_buf);free(res.cmd_buf);

        // Write chunk header + data
        V3Chunk chdr;
        chdr.magic=0x434B4E48; chdr.flags=0;
        chdr.raw_size=to_read;
        chdr.lit_size=zls; chdr.off_size=zos;
        chdr.len_size=zns; chdr.cmd_size=zcs;
        fwrite(&chdr,sizeof(chdr),1,fout);
        fwrite(zl,1,zls,fout); fwrite(zo,1,zos,fout);
        fwrite(zn,1,zns,fout); fwrite(zc,1,zcs,fout);
        free(zl);free(zo);free(zn);free(zc);

        total_compressed+=sizeof(chdr)+zls+zos+zns+zcs;

        // Shift window
        size_t keep=std::min(V3_HISTORY,wlen);
        memmove(window,window+wlen-keep,keep);
        history_len=keep; ht->cur_epoch++;
    }
    fclose(fin); fclose(fout);
    free(window);free(ht->pos);free(ht->epoch);free(ht->chain);free(ht);

    fprintf(stderr,"  Ratio: %.5fx\n",(double)total_size/total_compressed);
    return 0;
}

struct V3DecChunk {
    uint8_t *l,*o,*n,*c;
    size_t ls,os,ns,cs,raw_size;
    uint8_t* dst;
};

static void* v3_dec_worker(void* arg) {
    V3DecChunk* ch=(V3DecChunk*)arg;
    decompress_streams(ch->dst,ch->raw_size,
        ch->l,ch->ls,ch->o,ch->os,
        ch->n,ch->ns,ch->c,ch->cs);
    return nullptr;
}

int acepx3_decode(const char* in_path, const char* out_path, int threads=8) {
    FILE* fin=fopen(in_path,"rb");
    if (!fin) { fprintf(stderr,"Cannot open\n"); return 1; }

    V3Header fhdr; fread(&fhdr,sizeof(fhdr),1,fin);
    if (memcmp(fhdr.magic,"ACEPX3\0\0",8)!=0) {
        fprintf(stderr,"Bad magic\n"); fclose(fin); return 1;
    }
    uint32_t nb=fhdr.num_chunks;

    // Read all chunks into memory
    std::vector<V3DecChunk> chunks(nb);
    uint8_t* dst_buf=(uint8_t*)malloc(fhdr.orig_size+65536);
    size_t out_pos=0;

    for(uint32_t ci=0;ci<nb;ci++) {
        V3Chunk chdr; fread(&chdr,sizeof(chdr),1,fin);
        uint8_t* zl=(uint8_t*)malloc(chdr.lit_size);
        uint8_t* zo=(uint8_t*)malloc(chdr.off_size);
        uint8_t* zn=(uint8_t*)malloc(chdr.len_size);
        uint8_t* zc=(uint8_t*)malloc(chdr.cmd_size);
        fread(zl,1,chdr.lit_size,fin);
        fread(zo,1,chdr.off_size,fin);
        fread(zn,1,chdr.len_size,fin);
        fread(zc,1,chdr.cmd_size,fin);
        size_t os=*(uint64_t*)zo,ns=*(uint64_t*)zn,cs=*(uint64_t*)zc;
        size_t ls=0; uint8_t* l=lit_decompress(zl,chdr.lit_size,ls);
        uint8_t* o=(uint8_t*)malloc(os); fse_chunked_decomp(zo,os,o);
        uint8_t* n=(uint8_t*)malloc(ns); fse_chunked_decomp(zn,ns,n);
        uint8_t* c=(uint8_t*)malloc(cs); fse_chunked_decomp(zc,cs,c);
        free(zl);free(zo);free(zn);free(zc);
        chunks[ci]={l,o,n,c,ls,os,ns,cs,chdr.raw_size,dst_buf+out_pos};
        out_pos+=chdr.raw_size;
    }
    fclose(fin);

    // Parallel decode
    int nt=std::min((int)nb,threads);
    std::vector<pthread_t> pts(nt);
    std::atomic<int> next(0);
    struct PArg { std::vector<V3DecChunk>*ch; std::atomic<int>*nx; int nb; };
    PArg pa={&chunks,&next,(int)nb};
    for(int t=0;t<nt;t++) {
        pthread_create(&pts[t],nullptr,[](void*a)->void*{
            PArg*p=(PArg*)a;
            while(true){
                int i=p->nx->fetch_add(1);
                if(i>=(int)p->nb) break;
                v3_dec_worker(&(*p->ch)[i]);
            }
            return nullptr;
        },&pa);
    }
    for(int t=0;t<nt;t++) pthread_join(pts[t],nullptr);

    // Write output
    FILE* fout=fopen(out_path,"wb");
    fwrite(dst_buf,1,out_pos,fout);
    fclose(fout);

    for(auto& ch:chunks){free(ch.l);free(ch.o);free(ch.n);free(ch.c);}
    free(dst_buf);
    return 0;
}

int main(int argc, char** argv) {
    const char* cmd=argc>1?argv[1]:"c";
    const char* in=nullptr; const char* out=nullptr;
    int level=2, threads=8;
    for(int i=2;i<argc;i++) {
        if (!strcmp(argv[i],"--in")&&i+1<argc) in=argv[++i];
        else if (!strcmp(argv[i],"--out")&&i+1<argc) out=argv[++i];
        else if (!strcmp(argv[i],"--threads")&&i+1<argc) threads=atoi(argv[++i]);
    }
    if (!in||!out) { fprintf(stderr,"Usage: %s c|d --in <in> --out <out>\n",argv[0]); return 1; }
    if (!strcmp(cmd,"d")) return acepx3_decode(in,out,threads);
    return acepx3_encode(in,out,level,threads);
}

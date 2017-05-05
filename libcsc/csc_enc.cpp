#include <csc_enc.h>
#include <csc_memio.h>
#include <csc_typedef.h>
#include <csc_encoder_main.h>
#include <csc_default_alloc.h>
#include <stdlib.h>

struct CSCEncInstance
{
    CSCEncoder *encoder;
    MemIO *io;
    ISzAlloc *alloc;
    uint32_t raw_blocksize;
};

void CSCEncProps_Init(CSCProps *p, uint32_t dict_size, int level)
{
    dict_size += 10 * KB; // a little more, real size is 8KB smaller than set number
    if (dict_size < 32 * KB) dict_size = 32 * KB;
    if (dict_size > 1024 * MB) dict_size = 1024 * MB;
    p->dict_size = dict_size; 
    if (level < 1) level = 1;
    if (level > 5) level = 5;
    p->DLTFilter = 1;
    p->TXTFilter = 1;
    p->EXEFilter = 1;
    p->csc_blocksize = 64 * KB;
    p->raw_blocksize = 2 * MB;

    uint32_t hbits = 20;
    if (dict_size < MB)
        hbits = 19;
    else if (dict_size <= 4 * MB) 
        hbits = 20;
    else if (dict_size <= 16 * MB) 
        hbits = 21;
    else if (dict_size <= 64 * MB) 
        hbits = 22;
    else if (dict_size <= 256 * MB) 
        hbits = 23;
    else
        hbits = 24;
    while(((uint32_t)1 << hbits) > dict_size) hbits--;

    if (dict_size <= 16 * MB) 
        p->bt_size = dict_size;
    else if (dict_size <= 64 * MB) 
        p->bt_size = (dict_size - 16 * MB) / 2 + 16 * MB;
    else if (dict_size <= 256 * MB) 
        p->bt_size = (dict_size - 64 * MB) / 4 + 40 * MB;
    else
        p->bt_size = (dict_size - 256 * MB) / 8 + 88 * MB;

    p->good_len = 32;
    p->hash_bits = hbits;
    p->bt_hash_bits = hbits + 1;
    switch (level) {
        case 1:
            p->hash_width = 1;
            p->lz_mode = 2;
            p->bt_size = 0;
            p->hash_bits ++;
            break;
        case 2:
            p->hash_width = 8;
            p->lz_mode = 2;
            p->bt_size = 0;
            p->good_len = 24;
            p->hash_bits --;
            break;
        case 3:
            p->hash_width = 2;
            p->lz_mode = 3;
            p->bt_size = 0;
            p->good_len = 16;
            p->hash_bits ++;
            break;
        case 4:
            p->hash_width = 8;
            p->lz_mode = 3;
            p->bt_size = 0;
            p->good_len = 24;
            p->hash_bits --;
            break;
        case 5:
            p->lz_mode = 3;
            p->good_len = 48;
            p->bt_cyc = 32;
            //p->bt_size = p->dict_size;
            p->hash_width = 0;
            break;
    }

    if (p->bt_size == p->dict_size) {
        p->hash_width = 0;
    }
}

uint64_t CSCEnc_EstMemUsage(const CSCProps *p)
{
    uint64_t ret = 0;
    ret += p->dict_size;
    ret += p->csc_blocksize * 2;
    if (p->bt_size) 
        ret += ((1 << p->bt_hash_bits) + 2 * p->bt_size) * sizeof(uint32_t);
    if (p->hash_width)
        ret += (p->hash_width * (1 << p->hash_bits)) * sizeof(uint32_t);
    ret += 80 * KB *sizeof(uint32_t);
    ret += 256 * 256 * sizeof(uint32_t) * 2;
    ret += 2 * MB;
    return ret;
}

CSCEncHandle CSCEnc_Create(const CSCProps *props, 
        ISeqOutStream *outstream,
        ISzAlloc *alloc)
{
    if (alloc == NULL) {
        alloc = default_alloc;
    }
    CSCEncInstance *csc = (CSCEncInstance *)alloc->Alloc(alloc, sizeof(CSCEncInstance));

    csc->io = (MemIO *)alloc->Alloc(alloc, sizeof(MemIO));
    csc->io->Init(outstream, props->csc_blocksize, alloc);
    csc->raw_blocksize = props->raw_blocksize;
    csc->encoder = (CSCEncoder *)alloc->Alloc(alloc, sizeof(CSCEncoder));
    csc->alloc = alloc;
    if (csc->encoder->Init(props, csc->io, alloc) < 0) {
        CSCEnc_Destroy((void *)csc);
        return NULL;
    } else
        return (void*)csc;
}

void CSCEnc_Destroy(CSCEncHandle p)
{
    CSCEncInstance *csc = (CSCEncInstance *)p;
    csc->encoder->Destroy();
    ISzAlloc *alloc = csc->alloc;
    alloc->Free(alloc, csc->encoder);
    alloc->Free(alloc, csc->io);
    alloc->Free(alloc, csc);
}

void CSCEnc_WriteProperties(const CSCProps *props, uint8_t *s, int full)
{
    (void)full;
    s[0] = ((props->dict_size >> 24) & 0xff);
    s[1] = ((props->dict_size >> 16) & 0xff);
    s[2] = ((props->dict_size >> 8) & 0xff);
    s[3] = ((props->dict_size) & 0xff);
    s[4] = ((props->csc_blocksize >> 16) & 0xff);
    s[5] = ((props->csc_blocksize >> 8) & 0xff);
    s[6] = ((props->csc_blocksize) & 0xff);
    s[7] = ((props->raw_blocksize >> 16) & 0xff);
    s[8] = ((props->raw_blocksize >> 8) & 0xff);
    s[9] = ((props->raw_blocksize) & 0xff);
}

int CSCEnc_Encode(CSCEncHandle p, 
        ISeqInStream *is,
        ICompressProgress *progress)
{
    int ret = 0;
    CSCEncInstance *csc = (CSCEncInstance *)p;
    uint8_t *buf = (uint8_t *)csc->alloc->Alloc(csc->alloc, csc->raw_blocksize);
    uint64_t insize = 0;

    for(;;) {
        size_t size = csc->raw_blocksize;
        ret = is->Read(is, buf, &size);
        if (ret >= 0 && size) {
            insize += size;
            ret = 0;
            try {
                csc->encoder->Compress(buf, size);
            } catch (int errcode) {
                ret = errcode;
            }
            if (progress)
                progress->Progress(progress, insize, csc->encoder->GetCompressedSize());
        } else if (ret < 0) {
            ret = READ_ERROR;
        }

        if (ret < 0 || size == 0)
            break;
    }
    csc->alloc->Free(csc->alloc, buf);
    return ret;
}

int CSCEnc_Encode_Flush(CSCEncHandle p)
{
    CSCEncInstance *csc = (CSCEncInstance *)p;
    try {
        csc->encoder->WriteEOF();
        csc->encoder->Flush();
    } catch (int errcode) {
        return errcode;
    }
    return 0;
}


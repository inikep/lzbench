#ifndef _CSC_ENC_H_
#define _CSC_ENC_H_
#include <csc_common.h>

EXTERN_C_BEGIN


// set default parameters with dict_size and level
// further parameter changes must be after this call
// level should be 1 to 4
void CSCEncProps_Init(CSCProps *p, uint32_t dict_size = 64000000, int level = 2);

void CSCEnc_WriteProperties(const CSCProps *props, uint8_t *stream, int full);

uint64_t CSCEnc_EstMemUsage(const CSCProps *props);

typedef void * CSCEncHandle;

// alloc can be NULL, so default malloc/free will be used
CSCEncHandle CSCEnc_Create(const CSCProps *props,
                           ISeqOutStream *outstream,
                           ISzAlloc *alloc);

void CSCEnc_Destroy(CSCEncHandle p);

int CSCEnc_Encode(CSCEncHandle p, 
        ISeqInStream *instream,
        ICompressProgress *progress);

int CSCEnc_Encode_Flush(CSCEncHandle p);

/*
int CSCEnc_SimpleEncode(uint8_t *dest,
        size_t *destLen,
        const uint8_t *src,
        size_t srcLen,
        ICompressProgress *progress,
        ISzAlloc *alloc);
*/

EXTERN_C_END

#endif


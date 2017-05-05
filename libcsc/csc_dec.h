#ifndef _CSC_DEC_H_
#define _CSC_DEC_H_
#include <csc_common.h>

EXTERN_C_BEGIN

// DecProps usually needs to be read from existing data
void CSCDec_ReadProperties(CSCProps *props, uint8_t *stream);

typedef void * CSCDecHandle;

// alloc can be NULL, so default malloc/free will be used
CSCDecHandle CSCDec_Create(const CSCProps *props,
                           ISeqInStream *instream,
                           ISzAlloc *alloc);

void CSCDec_Destroy(CSCDecHandle p);

int CSCDec_Decode(CSCDecHandle p, 
        ISeqOutStream *outstream,
        ICompressProgress *progress);


/*
int CSCDec_SimpleDecode(uint8_t *dest,
        size_t *destLen,
        const uint8_t *src,
        size_t srcLen,
        ICompressProgress *progress,
        ISzAlloc *alloc);
*/

EXTERN_C_END

#endif


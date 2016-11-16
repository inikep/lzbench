#define LZ5_LENGTH_SIZE_LZ4(len) ((len >= (1<<16)+RUN_MASK_LZ4) ? 5 : ((len >= 254+RUN_MASK_LZ4) ? 3 : ((len >= RUN_MASK_LZ4) ? 1 : 0)))

FORCE_INLINE int LZ5_encodeSequence_LZ4 (
    LZ5_stream_t* ctx,
    const BYTE** ip,
    const BYTE** anchor,
    size_t matchLength,
    const BYTE* const match)
{
    size_t length = (size_t)(*ip - *anchor);
    BYTE* token = (ctx->flagsPtr)++;
    (void) ctx;

    COMPLOG_CODEWORDS_LZ4("literal : %u  --  match : %u  --  offset : %u\n", (U32)(*ip - *anchor), (U32)matchLength, (U32)(*ip-match));
  
    /* Encode Literal length */
 //   if (ctx->literalsPtr > ctx->literalsEnd - length - LZ5_LENGTH_SIZE_LZ4(length) - 2 - WILDCOPYLENGTH) { LZ5_LOG_COMPRESS_LZ4("encodeSequence overflow1\n"); return 1; }   /* Check output limit */
    if (length >= RUN_MASK_LZ4) 
    {   size_t len; 
        *token = RUN_MASK_LZ4; 
        len = length - RUN_MASK_LZ4;
        if (len >= (1<<16)) { *(ctx->literalsPtr) = 255;  MEM_writeLE24(ctx->literalsPtr+1, (U32)(len));  ctx->literalsPtr += 4; }
        else if (len >= 254) { *(ctx->literalsPtr) = 254;  MEM_writeLE16(ctx->literalsPtr+1, (U16)(len));  ctx->literalsPtr += 3; }
        else *(ctx->literalsPtr)++ = (BYTE)len;
    }
    else *token = (BYTE)length;

    /* Copy Literals */
    LZ5_wildCopy(ctx->literalsPtr, *anchor, (ctx->literalsPtr) + length);
    ctx->literalsPtr += length;

    /* Encode Offset */
//    if (match > *ip) printf("match > *ip\n"), exit(1);
//    if ((U32)(*ip-match) >= (1<<16)) printf("off=%d\n", (U32)(*ip-match)), exit(1);
    MEM_writeLE16(ctx->literalsPtr, (U16)(*ip-match));
    ctx->literalsPtr+=2;

    /* Encode MatchLength */
    length = matchLength - MINMATCH;
  //  if (ctx->literalsPtr > ctx->literalsEnd - 5 /*LZ5_LENGTH_SIZE_LZ4(length)*/) { LZ5_LOG_COMPRESS_LZ4("encodeSequence overflow2\n"); return 1; }   /* Check output limit */
    if (length >= ML_MASK_LZ4) {
        *token += (BYTE)(ML_MASK_LZ4<<RUN_BITS_LZ4);
        length -= ML_MASK_LZ4;
        if (length >= (1<<16)) { *(ctx->literalsPtr) = 255;  MEM_writeLE24(ctx->literalsPtr+1, (U32)(length));  ctx->literalsPtr += 4; }
        else if (length >= 254) { *(ctx->literalsPtr) = 254;  MEM_writeLE16(ctx->literalsPtr+1, (U16)(length));  ctx->literalsPtr += 3; }
        else *(ctx->literalsPtr)++ = (BYTE)length;
    }
    else *token += (BYTE)(length<<RUN_BITS_LZ4);

    /* Prepare next loop */
    *ip += matchLength;
    *anchor = *ip;

    return 0;
}

FORCE_INLINE int LZ5_encodeLastLiterals_LZ4 (
    LZ5_stream_t* ctx,
    const BYTE** ip,
    const BYTE** anchor)
{
    size_t length = (int)(*ip - *anchor);
//    BYTE* token = ctx->flagsPtr++;

    (void)ctx;

#if 0
    LZ5_LOG_COMPRESS_LZ4("LZ5_encodeLastLiterals_LZ4 length=%d LZ5_LENGTH_SIZE_LZ4(length)=%d oend-op=%d\n", (int)length, LZ5_LENGTH_SIZE_LZ4(length), (int)(ctx->literalsEnd-ctx->literalsPtr));
  //  if (ctx->literalsPtr > ctx->literalsEnd - length - LZ5_LENGTH_SIZE_LZ4(length)) { LZ5_LOG_COMPRESS_LZ4("LastLiterals overflow\n"); return 1; } /* Check output buffer overflow */
    if (length >= RUN_MASK_LZ4) 
    {   size_t len; 
        *token = RUN_MASK_LZ4;
        len = length - RUN_MASK_LZ4;
        if (len >= (1<<16)) { *ctx->literalsPtr = 255;  MEM_writeLE24(ctx->literalsPtr+1, (U32)(len));  ctx->literalsPtr += 4; }
        else if (len >= 254) { *ctx->literalsPtr = 254;  MEM_writeLE16(ctx->literalsPtr+1, (U16)(len));  ctx->literalsPtr += 3; }
        else *ctx->literalsPtr++ = (BYTE)len;
    }
    else *token = (BYTE)length;
#endif

    memcpy(ctx->literalsPtr, *anchor, length);
    ctx->literalsPtr += length;
    return 0;
}


FORCE_INLINE size_t LZ5_get_price_LZ4(LZ5_stream_t* const ctx, size_t litLength, U32 offset, size_t matchLength)
{
    size_t price = 8 + (matchLength==1); // (ctx->literalsPtr)++;
    (void)ctx;
    (void)offset;

    /* Encode Literal length */
    if (litLength >= RUN_MASK_LZ4) 
    {
        size_t len = litLength - RUN_MASK_LZ4;
        if (len >= (1<<16)) price += 32;
        else if (len >= 254) price += 24;
        else price += 8;
    }

    price += 8*litLength;  /* Copy Literals */

    /* Encode Offset */
    if (offset) {
        price += 16; // ctx->literalsPtr+=2;
     
        /* Encode MatchLength */
        if (matchLength < MINMATCH) return 1<<16;//LZ5_MAX_PRICE; // error
        matchLength -= MINMATCH;
        if (matchLength >= ML_MASK_LZ4) {
            matchLength -= ML_MASK_LZ4;
            if (matchLength >= (1<<16)) price += 32;
            else if (matchLength >= 254) price += 24;
            else price += 8;
        }
    }

    return price;
}


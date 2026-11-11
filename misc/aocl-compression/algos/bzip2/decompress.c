
/*-------------------------------------------------------------*/
/*--- Decompression machinery                               ---*/
/*---                                          decompress.c ---*/
/*-------------------------------------------------------------*/

/* ------------------------------------------------------------------
   This file is part of bzip2/libbzip2, a program and library for
   lossless, block-sorting data compression.

   bzip2/libbzip2 version 1.0.8 of 13 July 2019
   Copyright (C) 1996-2019 Julian Seward <jseward@acm.org>
   Modifications Copyright (C) 2023-2026, Advanced Micro Devices. All rights reserved.

   Please read the WARNING, DISCLAIMER and PATENTS sections in the 
   README file.

   This program is released under the terms of the license contained
   in the file LICENSE.
   ------------------------------------------------------------------ */


#include "bzlib_private.h"


/*---------------------------------------------------*/
static
void makeMaps_d ( DState* s )
{
   Int32 i;
   s->nInUse = 0;
   for (i = 0; i < 256; i++)
      if (s->inUse[i]) {
         s->seqToUnseq[s->nInUse] = i;
         s->nInUse++;
      }
}


/*---------------------------------------------------*/
#define RETURN(rrr)                               \
   { retVal = rrr; goto save_state_and_return; };

#define GET_BITS(lll,vvv,nnn)                     \
   case lll: s->state = lll;                      \
   while (True) {                                 \
      if (s->bsLive >= nnn) {                     \
         UInt32 v;                                \
         v = (s->bsBuff >>                        \
             (s->bsLive-nnn)) & ((1 << nnn)-1);   \
         s->bsLive -= nnn;                        \
         vvv = v;                                 \
         break;                                   \
      }                                           \
      if (s->strm->avail_in == 0) RETURN(BZ_OK);  \
      s->bsBuff                                   \
         = (s->bsBuff << 8) |                     \
           ((UInt32)                              \
              (*((UChar*)(s->strm->next_in))));   \
      s->bsLive += 8;                             \
      s->strm->next_in++;                         \
      s->strm->avail_in--;                        \
      s->strm->total_in_lo32++;                   \
      if (s->strm->total_in_lo32 == 0)            \
         s->strm->total_in_hi32++;                \
   }

#define GET_UCHAR(lll,uuu)                        \
   GET_BITS(lll,uuu,8)

#define GET_BIT(lll,uuu)                          \
   GET_BITS(lll,uuu,1)

/*---------------------------------------------------*/
#define GET_MTF_VAL(label1,label2,lval)           \
{                                                 \
   if (groupPos == 0) {                           \
      groupNo++;                                  \
      if (groupNo >= nSelectors)                  \
         RETURN(BZ_DATA_ERROR);                   \
      groupPos = BZ_G_SIZE;                       \
      gSel = s->selector[groupNo];                \
      gMinlen = s->minLens[gSel];                 \
      gLimit = &(s->limit[gSel][0]);              \
      gPerm = &(s->perm[gSel][0]);                \
      gBase = &(s->base[gSel][0]);                \
   }                                              \
   groupPos--;                                    \
   zn = gMinlen;                                  \
   GET_BITS(label1, zvec, zn);                    \
   while (1) {                                    \
      if (zn > 20 /* the longest code */)         \
         RETURN(BZ_DATA_ERROR);                   \
      if (zvec <= gLimit[zn]) break;              \
      zn++;                                       \
      GET_BIT(label2, zj);                        \
      zvec = (zvec << 1) | zj;                    \
   };                                             \
   if (zvec - gBase[zn] < 0                       \
       || zvec - gBase[zn] >= BZ_MAX_ALPHA_SIZE)  \
      RETURN(BZ_DATA_ERROR);                      \
   lval = gPerm[zvec - gBase[zn]];                \
}


/*---------------------------------------------------*/
Int32 BZ2_decompress ( DState* s )
{
   UChar      uc;
   Int32      retVal;
   Int32      minLen, maxLen;
   bz_stream* strm = s->strm;

   /* stuff that needs to be saved/restored */
   Int32  i;
   Int32  j;
   Int32  t;
   Int32  alphaSize;
   Int32  nGroups;
   Int32  nSelectors;
   Int32  EOB;
   Int32  groupNo;
   Int32  groupPos;
   Int32  nextSym;
   Int32  nblockMAX;
   Int32  nblock;
   Int32  es;
   Int32  N;
   Int32  curr;
   Int32  zt;
   Int32  zn; 
   Int32  zvec;
   Int32  zj;
   Int32  gSel;
   Int32  gMinlen;
   Int32* gLimit;
   Int32* gBase;
   Int32* gPerm;

   if (s->state == BZ_X_MAGIC_1) {
      /*initialise the save area*/
      s->save_i           = 0;
      s->save_j           = 0;
      s->save_t           = 0;
      s->save_alphaSize   = 0;
      s->save_nGroups     = 0;
      s->save_nSelectors  = 0;
      s->save_EOB         = 0;
      s->save_groupNo     = 0;
      s->save_groupPos    = 0;
      s->save_nextSym     = 0;
      s->save_nblockMAX   = 0;
      s->save_nblock      = 0;
      s->save_es          = 0;
      s->save_N           = 0;
      s->save_curr        = 0;
      s->save_zt          = 0;
      s->save_zn          = 0;
      s->save_zvec        = 0;
      s->save_zj          = 0;
      s->save_gSel        = 0;
      s->save_gMinlen     = 0;
      s->save_gLimit      = NULL;
      s->save_gBase       = NULL;
      s->save_gPerm       = NULL;
   }

   /*restore from the save area*/
   i           = s->save_i;
   j           = s->save_j;
   t           = s->save_t;
   alphaSize   = s->save_alphaSize;
   nGroups     = s->save_nGroups;
   nSelectors  = s->save_nSelectors;
   EOB         = s->save_EOB;
   groupNo     = s->save_groupNo;
   groupPos    = s->save_groupPos;
   nextSym     = s->save_nextSym;
   nblockMAX   = s->save_nblockMAX;
   nblock      = s->save_nblock;
   es          = s->save_es;
   N           = s->save_N;
   curr        = s->save_curr;
   zt          = s->save_zt;
   zn          = s->save_zn; 
   zvec        = s->save_zvec;
   zj          = s->save_zj;
   gSel        = s->save_gSel;
   gMinlen     = s->save_gMinlen;
   gLimit      = s->save_gLimit;
   gBase       = s->save_gBase;
   gPerm       = s->save_gPerm;

   retVal = BZ_OK;

   switch (s->state) {

      GET_UCHAR(BZ_X_MAGIC_1, uc);
      if (uc != BZ_HDR_B) RETURN(BZ_DATA_ERROR_MAGIC);

      GET_UCHAR(BZ_X_MAGIC_2, uc);
      if (uc != BZ_HDR_Z) RETURN(BZ_DATA_ERROR_MAGIC);

      GET_UCHAR(BZ_X_MAGIC_3, uc)
      if (uc != BZ_HDR_h) RETURN(BZ_DATA_ERROR_MAGIC);

      GET_BITS(BZ_X_MAGIC_4, s->blockSize100k, 8)
      if (s->blockSize100k < (BZ_HDR_0 + 1) || 
          s->blockSize100k > (BZ_HDR_0 + 9)) RETURN(BZ_DATA_ERROR_MAGIC);
      s->blockSize100k -= BZ_HDR_0;

      if (s->smallDecompress) {
         s->ll16 = BZALLOC( s->blockSize100k * 100000 * sizeof(UInt16) );
         s->ll4  = BZALLOC( 
                      ((1 + s->blockSize100k * 100000) >> 1) * sizeof(UChar) 
                   );
         if (s->ll16 == NULL || s->ll4 == NULL) RETURN(BZ_MEM_ERROR);
      } else {
         s->tt  = BZALLOC( s->blockSize100k * 100000 * sizeof(Int32) );
         if (s->tt == NULL) RETURN(BZ_MEM_ERROR);
      }

      GET_UCHAR(BZ_X_BLKHDR_1, uc);

      if (uc == 0x17) goto endhdr_2;
      if (uc != 0x31) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_BLKHDR_2, uc);
      if (uc != 0x41) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_BLKHDR_3, uc);
      if (uc != 0x59) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_BLKHDR_4, uc);
      if (uc != 0x26) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_BLKHDR_5, uc);
      if (uc != 0x53) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_BLKHDR_6, uc);
      if (uc != 0x59) RETURN(BZ_DATA_ERROR);

      s->currBlockNo++;
      if (s->verbosity >= 2)
         VPrintf1 ( "\n    [%d: huff+mtf ", s->currBlockNo );
 
      s->storedBlockCRC = 0;
      GET_UCHAR(BZ_X_BCRC_1, uc);
      s->storedBlockCRC = (s->storedBlockCRC << 8) | ((UInt32)uc);
      GET_UCHAR(BZ_X_BCRC_2, uc);
      s->storedBlockCRC = (s->storedBlockCRC << 8) | ((UInt32)uc);
      GET_UCHAR(BZ_X_BCRC_3, uc);
      s->storedBlockCRC = (s->storedBlockCRC << 8) | ((UInt32)uc);
      GET_UCHAR(BZ_X_BCRC_4, uc);
      s->storedBlockCRC = (s->storedBlockCRC << 8) | ((UInt32)uc);

      GET_BITS(BZ_X_RANDBIT, s->blockRandomised, 1);

      s->origPtr = 0;
      GET_UCHAR(BZ_X_ORIGPTR_1, uc);
      s->origPtr = (s->origPtr << 8) | ((Int32)uc);
      GET_UCHAR(BZ_X_ORIGPTR_2, uc);
      s->origPtr = (s->origPtr << 8) | ((Int32)uc);
      GET_UCHAR(BZ_X_ORIGPTR_3, uc);
      s->origPtr = (s->origPtr << 8) | ((Int32)uc);

      if (s->origPtr < 0)
         RETURN(BZ_DATA_ERROR);
      if (s->origPtr > 10 + 100000*s->blockSize100k) 
         RETURN(BZ_DATA_ERROR);

      /*--- Receive the mapping table ---*/
      for (i = 0; i < 16; i++) {
         GET_BIT(BZ_X_MAPPING_1, uc);
         if (uc == 1) 
            s->inUse16[i] = True; else 
            s->inUse16[i] = False;
      }

      for (i = 0; i < 256; i++) s->inUse[i] = False;

      for (i = 0; i < 16; i++)
         if (s->inUse16[i])
            for (j = 0; j < 16; j++) {
               GET_BIT(BZ_X_MAPPING_2, uc);
               if (uc == 1) s->inUse[i * 16 + j] = True;
            }
      makeMaps_d ( s );
      if (s->nInUse == 0) RETURN(BZ_DATA_ERROR);
      alphaSize = s->nInUse+2;

      /*--- Now the selectors ---*/
      GET_BITS(BZ_X_SELECTOR_1, nGroups, 3);
      if (nGroups < 2 || nGroups > BZ_N_GROUPS) RETURN(BZ_DATA_ERROR);
      GET_BITS(BZ_X_SELECTOR_2, nSelectors, 15);
      if (nSelectors < 1) RETURN(BZ_DATA_ERROR);
      for (i = 0; i < nSelectors; i++) {
         j = 0;
         while (True) {
            GET_BIT(BZ_X_SELECTOR_3, uc);
            if (uc == 0) break;
            j++;
            if (j >= nGroups) RETURN(BZ_DATA_ERROR);
         }
         /* Having more than BZ_MAX_SELECTORS doesn't make much sense
            since they will never be used, but some implementations might
            "round up" the number of selectors, so just ignore those. */
         if (i < BZ_MAX_SELECTORS)
           s->selectorMtf[i] = j;
      }
      if (nSelectors > BZ_MAX_SELECTORS)
        nSelectors = BZ_MAX_SELECTORS;

      /*--- Undo the MTF values for the selectors. ---*/
      {
         UChar pos[BZ_N_GROUPS], tmp, v;
         for (v = 0; v < nGroups; v++) pos[v] = v;
   
         for (i = 0; i < nSelectors; i++) {
            v = s->selectorMtf[i];
            tmp = pos[v];
            while (v > 0) { pos[v] = pos[v-1]; v--; }
            pos[0] = tmp;
            s->selector[i] = tmp;
         }
      }

      /*--- Now the coding tables ---*/
      for (t = 0; t < nGroups; t++) {
         GET_BITS(BZ_X_CODING_1, curr, 5);
         for (i = 0; i < alphaSize; i++) {
            while (True) {
               if (curr < 1 || curr > 20) RETURN(BZ_DATA_ERROR);
               GET_BIT(BZ_X_CODING_2, uc);
               if (uc == 0) break;
               GET_BIT(BZ_X_CODING_3, uc);
               if (uc == 0) curr++; else curr--;
            }
            s->len[t][i] = curr;
         }
      }

      /*--- Create the Huffman decoding tables ---*/
      for (t = 0; t < nGroups; t++) {
         minLen = 32;
         maxLen = 0;
         for (i = 0; i < alphaSize; i++) {
            if (s->len[t][i] > maxLen) maxLen = s->len[t][i];
            if (s->len[t][i] < minLen) minLen = s->len[t][i];
         }
         BZ2_hbCreateDecodeTables ( 
            &(s->limit[t][0]), 
            &(s->base[t][0]), 
            &(s->perm[t][0]), 
            &(s->len[t][0]),
            minLen, maxLen, alphaSize
         );
         s->minLens[t] = minLen;
      }

      /*--- Now the MTF values ---*/

      EOB      = s->nInUse+1;
      nblockMAX = 100000 * s->blockSize100k;
      groupNo  = -1;
      groupPos = 0;

      for (i = 0; i <= 255; i++) s->unzftab[i] = 0;

      /*-- MTF init --*/
      {
         Int32 ii, jj, kk;
         kk = MTFA_SIZE-1;
         for (ii = 256 / MTFL_SIZE - 1; ii >= 0; ii--) {
            for (jj = MTFL_SIZE-1; jj >= 0; jj--) {
               s->mtfa[kk] = (UChar)(ii * MTFL_SIZE + jj);
               kk--;
            }
            s->mtfbase[ii] = kk + 1;
         }
      }
      /*-- end MTF init --*/

      nblock = 0;
      GET_MTF_VAL(BZ_X_MTF_1, BZ_X_MTF_2, nextSym);

      while (True) {

         if (nextSym == EOB) break;

         if (nextSym == BZ_RUNA || nextSym == BZ_RUNB) {

            es = -1;
            N = 1;
            do {
               /* Check that N doesn't get too big, so that es doesn't
                  go negative.  The maximum value that can be
                  RUNA/RUNB encoded is equal to the block size (post
                  the initial RLE), viz, 900k, so bounding N at 2
                  million should guard against overflow without
                  rejecting any legitimate inputs. */
               if (N >= 2*1024*1024) RETURN(BZ_DATA_ERROR);
               if (nextSym == BZ_RUNA) es = es + (0+1) * N; else
               if (nextSym == BZ_RUNB) es = es + (1+1) * N;
               N = N * 2;
               GET_MTF_VAL(BZ_X_MTF_3, BZ_X_MTF_4, nextSym);
            }
               while (nextSym == BZ_RUNA || nextSym == BZ_RUNB);

            es++;
            uc = s->seqToUnseq[ s->mtfa[s->mtfbase[0]] ];
            s->unzftab[uc] += es;

            if (s->smallDecompress)
               while (es > 0) {
                  if (nblock >= nblockMAX) RETURN(BZ_DATA_ERROR);
                  s->ll16[nblock] = (UInt16)uc;
                  nblock++;
                  es--;
               }
            else
               while (es > 0) {
                  if (nblock >= nblockMAX) RETURN(BZ_DATA_ERROR);
                  s->tt[nblock] = (UInt32)uc;
                  nblock++;
                  es--;
               };

            continue;

         } else {

            if (nblock >= nblockMAX) RETURN(BZ_DATA_ERROR);

            /*-- uc = MTF ( nextSym-1 ) --*/
            {
               Int32 ii, jj, kk, pp, lno, off;
               UInt32 nn;
               nn = (UInt32)(nextSym - 1);

               if (nn < MTFL_SIZE) {
                  /* avoid general-case expense */
                  pp = s->mtfbase[0];
                  uc = s->mtfa[pp+nn];
                  while (nn > 3) {
                     Int32 z = pp+nn;
                     s->mtfa[(z)  ] = s->mtfa[(z)-1];
                     s->mtfa[(z)-1] = s->mtfa[(z)-2];
                     s->mtfa[(z)-2] = s->mtfa[(z)-3];
                     s->mtfa[(z)-3] = s->mtfa[(z)-4];
                     nn -= 4;
                  }
                  while (nn > 0) { 
                     s->mtfa[(pp+nn)] = s->mtfa[(pp+nn)-1]; nn--; 
                  };
                  s->mtfa[pp] = uc;
               } else { 
                  /* general case */
                  lno = nn / MTFL_SIZE;
                  off = nn % MTFL_SIZE;
                  pp = s->mtfbase[lno] + off;
                  uc = s->mtfa[pp];
                  while (pp > s->mtfbase[lno]) { 
                     s->mtfa[pp] = s->mtfa[pp-1]; pp--; 
                  };
                  s->mtfbase[lno]++;
                  while (lno > 0) {
                     s->mtfbase[lno]--;
                     s->mtfa[s->mtfbase[lno]] 
                        = s->mtfa[s->mtfbase[lno-1] + MTFL_SIZE - 1];
                     lno--;
                  }
                  s->mtfbase[0]--;
                  s->mtfa[s->mtfbase[0]] = uc;
                  if (s->mtfbase[0] == 0) {
                     kk = MTFA_SIZE-1;
                     for (ii = 256 / MTFL_SIZE-1; ii >= 0; ii--) {
                        for (jj = MTFL_SIZE-1; jj >= 0; jj--) {
                           s->mtfa[kk] = s->mtfa[s->mtfbase[ii] + jj];
                           kk--;
                        }
                        s->mtfbase[ii] = kk + 1;
                     }
                  }
               }
            }
            /*-- end uc = MTF ( nextSym-1 ) --*/

            s->unzftab[s->seqToUnseq[uc]]++;
            if (s->smallDecompress)
               s->ll16[nblock] = (UInt16)(s->seqToUnseq[uc]); else
               s->tt[nblock]   = (UInt32)(s->seqToUnseq[uc]);
            nblock++;

            GET_MTF_VAL(BZ_X_MTF_5, BZ_X_MTF_6, nextSym);
            continue;
         }
      }

      /* Now we know what nblock is, we can do a better sanity
         check on s->origPtr.
      */
      if (s->origPtr < 0 || s->origPtr >= nblock)
         RETURN(BZ_DATA_ERROR);

      /*-- Set up cftab to facilitate generation of T^(-1) --*/
      /* Check: unzftab entries in range. */
      for (i = 0; i <= 255; i++) {
         if (s->unzftab[i] < 0 || s->unzftab[i] > nblock)
            RETURN(BZ_DATA_ERROR);
      }
      /* Actually generate cftab. */
      s->cftab[0] = 0;
      for (i = 1; i <= 256; i++) s->cftab[i] = s->unzftab[i-1];
      for (i = 1; i <= 256; i++) s->cftab[i] += s->cftab[i-1];
      /* Check: cftab entries in range. */
      for (i = 0; i <= 256; i++) {
         if (s->cftab[i] < 0 || s->cftab[i] > nblock) {
            /* s->cftab[i] can legitimately be == nblock */
            RETURN(BZ_DATA_ERROR);
         }
      }
      /* Check: cftab entries non-descending. */
      for (i = 1; i <= 256; i++) {
         if (s->cftab[i-1] > s->cftab[i]) {
            RETURN(BZ_DATA_ERROR);
         }
      }

      s->state_out_len = 0;
      s->state_out_ch  = 0;
      BZ_INITIALISE_CRC ( s->calculatedBlockCRC );
      s->state = BZ_X_OUTPUT;
      if (s->verbosity >= 2) VPrintf0 ( "rt+rld" );

      if (s->smallDecompress) {

         /*-- Make a copy of cftab, used in generation of T --*/
         for (i = 0; i <= 256; i++) s->cftabCopy[i] = s->cftab[i];

         /*-- compute the T vector --*/
         for (i = 0; i < nblock; i++) {
            uc = (UChar)(s->ll16[i]);
            SET_LL(i, s->cftabCopy[uc]);
            s->cftabCopy[uc]++;
         }

         /*-- Compute T^(-1) by pointer reversal on T --*/
         i = s->origPtr;
         j = GET_LL(i);
         do {
            Int32 tmp = GET_LL(j);
            SET_LL(j, i);
            i = j;
            j = tmp;
         }
            while (i != s->origPtr);

         s->tPos = s->origPtr;
         s->nblock_used = 0;
         if (s->blockRandomised) {
            BZ_RAND_INIT_MASK;
            BZ_GET_SMALL(s->k0); s->nblock_used++;
            BZ_RAND_UPD_MASK; s->k0 ^= BZ_RAND_MASK; 
         } else {
            BZ_GET_SMALL(s->k0); s->nblock_used++;
         }

      } else {

         /*-- compute the T^(-1) vector --*/
         for (i = 0; i < nblock; i++) {
            uc = (UChar)(s->tt[i] & 0xff);
            s->tt[s->cftab[uc]] |= (i << 8);
            s->cftab[uc]++;
         }

         s->tPos = s->tt[s->origPtr] >> 8;
         s->nblock_used = 0;
         if (s->blockRandomised) {
            BZ_RAND_INIT_MASK;
            BZ_GET_FAST(s->k0); s->nblock_used++;
            BZ_RAND_UPD_MASK; s->k0 ^= BZ_RAND_MASK; 
         } else {
            BZ_GET_FAST(s->k0); s->nblock_used++;
         }

      }

      RETURN(BZ_OK);



    endhdr_2:

      GET_UCHAR(BZ_X_ENDHDR_2, uc);
      if (uc != 0x72) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_ENDHDR_3, uc);
      if (uc != 0x45) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_ENDHDR_4, uc);
      if (uc != 0x38) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_ENDHDR_5, uc);
      if (uc != 0x50) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_ENDHDR_6, uc);
      if (uc != 0x90) RETURN(BZ_DATA_ERROR);

      s->storedCombinedCRC = 0;
      GET_UCHAR(BZ_X_CCRC_1, uc);
      s->storedCombinedCRC = (s->storedCombinedCRC << 8) | ((UInt32)uc);
      GET_UCHAR(BZ_X_CCRC_2, uc);
      s->storedCombinedCRC = (s->storedCombinedCRC << 8) | ((UInt32)uc);
      GET_UCHAR(BZ_X_CCRC_3, uc);
      s->storedCombinedCRC = (s->storedCombinedCRC << 8) | ((UInt32)uc);
      GET_UCHAR(BZ_X_CCRC_4, uc);
      s->storedCombinedCRC = (s->storedCombinedCRC << 8) | ((UInt32)uc);

      s->state = BZ_X_IDLE;
      RETURN(BZ_STREAM_END);

      default: AssertH ( False, 4001 );
   }

   AssertH ( False, 4002 );

   save_state_and_return:

   s->save_i           = i;
   s->save_j           = j;
   s->save_t           = t;
   s->save_alphaSize   = alphaSize;
   s->save_nGroups     = nGroups;
   s->save_nSelectors  = nSelectors;
   s->save_EOB         = EOB;
   s->save_groupNo     = groupNo;
   s->save_groupPos    = groupPos;
   s->save_nextSym     = nextSym;
   s->save_nblockMAX   = nblockMAX;
   s->save_nblock      = nblock;
   s->save_es          = es;
   s->save_N           = N;
   s->save_curr        = curr;
   s->save_zt          = zt;
   s->save_zn          = zn;
   s->save_zvec        = zvec;
   s->save_zj          = zj;
   s->save_gSel        = gSel;
   s->save_gMinlen     = gMinlen;
   s->save_gLimit      = gLimit;
   s->save_gBase       = gBase;
   s->save_gPerm       = gPerm;

   return retVal;   
}

#ifdef AOCL_BZIP2_OPT

/* This is the minimum val of s->strm->avail_in which is required 
   to enter the optimized while loop, in this while loop checks are 
   removed for checking s->strm->avail_in inside GET_BITS macro
   i.e, if number of bytes are greater than 53 no need to keep track
   of bytes available inside GET_BITS macro. In a single GET_MTF_VAL
   call inside do...while loop, where
   `if (N >= 2*1024*1024) RETURN(BZ_DATA_ERROR);`
   N is incremented by power of 2 which means the loop will not run
   more than 21 times, in worst case it reads no more than 20bits
   (refer to GET_MTF_VAL macro), so in single iteration worst case 
   it reads 20*21=420 bits , 420 bits=52.5 bytes , 52.5 bytes≈53 bytes.

   In addition to bytes consumed for Huffman decoding, the 64-bit bsBuff
   can hold up to 8 extra bytes for future operations, which must be accounted for
   in buffer size calculations.
*/
#define AOCL_MTF_MAX_BYTES 53      /* Maximum bytes needed for MTF decoding (see explanation above) */
#define AOCL_BSBUFF_ULONGLONG_BYTES 8  /* ULong64 buffer size in bytes */
#define AOCL_WHILE_LIMIT (AOCL_MTF_MAX_BYTES + AOCL_BSBUFF_ULONGLONG_BYTES)
#define AOCL_MTFL_FAST_PATH_LIMIT 128

/* This represents the maximum length of Huffman codes that can be encountered
   in bzip2 streams. We extract this many bits at once for fast decoding.
 */
#define AOCL_HUFFMAN_MAX_CODE_BITS 17

/* Bit mask to extract code length from packed Huffman table data [symbol << 8 | code_length].
   Extracts lower 5 bits containing code length (max 31, fits in 5 bits). */
#define AOCL_HUFFMAN_CODE_LENGTH_MASK 0x1f

/* Macro for direct lookup in primary Huffman table.
   Used for fast decoding of codes <= AOCL_BS_BUFF_BITS length.
   Returns packed value: [symbol << 8 | code_length] or negative secondary table index.
 */
#define AOCL_HUFFMAN_DECODE_DIRECT s->huffman_lookup_table[gSel][raw_huffman_bits >> (AOCL_HUFFMAN_MAX_CODE_BITS - AOCL_BS_BUFF_BITS)]

/* Macro for secondary table lookup for long codes
   Used when primary table returns negative value (indicating long code that overflows primary table)
   Returns packed value: [symbol << 8 | code_length]
 */
#define AOCL_HUFFMAN_DECODE_SECONDARY s->secondary_tables[gSel][s->secondary_table_size[gSel]*index + (raw_huffman_bits >> ((AOCL_HUFFMAN_MAX_CODE_BITS - AOCL_BS_BUFF_BITS) - s->secondary_shift_bits[gSel]))]

#define AOCL_GET_BITS1(lll, vvv, nnn)                                            \
   s->state = lll;                                                               \
   /* Keep filling buffer while it can hold >= 8 bits (bsBuff is 64-bit) */      \
   while (s->bsLive + 8 <= 64) {                                                 \
      s->bsBuff = (s->bsBuff << 8) | ((UInt32)(*((UChar *)(s->strm->next_in)))); \
      s->bsLive += 8;                                                            \
      s->strm->next_in++;                                                        \
      s->strm->avail_in--;                                                       \
      s->strm->total_in_lo32++;                                                  \
      if (s->strm->total_in_lo32 == 0)                                           \
         s->strm->total_in_hi32++;                                               \
   }                                                                             \
   while (True) {                                                                \
      if (s->bsLive >= nnn) {                                                    \
         UInt32 v;                                                               \
         v = (s->bsBuff >> (s->bsLive - nnn)) & ((1 << nnn) - 1);                \
         s->bsLive -= nnn;                                                       \
         vvv = v;                                                                \
         break;                                                                  \
      }                                                                          \
   }

#define AOCL_GET_BIT1(lll,uuu)                    \
   AOCL_GET_BITS1(lll,uuu,1)

/* AOCL_GET_MTF_VAL1: Fast MTF decode via two-tier Huffman lookup (primary + secondary).
   Manages group selection/position, extracts symbol and code length, then consumes bits from the stream. */
#define AOCL_GET_MTF_VAL1(label1, label2, lval)                                        \
{                                                                                      \
   if (groupPos == 0) {                                                                \
      groupNo++;                                                                       \
      if (groupNo >= nSelectors)                                                       \
         RETURN(BZ_DATA_ERROR);                                                        \
      groupPos = BZ_G_SIZE;                                                            \
      gSel = s->selector[groupNo];                                                     \
   }                                                                                   \
   groupPos--;                                                                         \
   /* Extract raw Huffman bits from bit stream */                                      \
   int raw_huffman_bits = (s->bsBuff >> (s->bsLive - AOCL_HUFFMAN_MAX_CODE_BITS))      \
                           & ((1 << AOCL_HUFFMAN_MAX_CODE_BITS) - 1);                  \
   /* Primary table lookup for fast decoding */                                        \
   zn = AOCL_HUFFMAN_DECODE_DIRECT;                                                    \
   if (zn < 0) {                                                                       \
      /* Long code: use secondary table */                                             \
      int index = -1 * (zn + 1);                                                       \
      /* Mask to remaining bits for secondary table indexing */                        \
      raw_huffman_bits &= (1 << (AOCL_HUFFMAN_MAX_CODE_BITS - AOCL_BS_BUFF_BITS)) - 1; \
      zn = AOCL_HUFFMAN_DECODE_SECONDARY;                                              \
   }                                                                                   \
   /* Extract symbol and code length from packed data */                               \
   lval = zn >> 8;                                                                     \
   zn = (zn & AOCL_HUFFMAN_CODE_LENGTH_MASK);                                          \
   /* Consume the decoded number of bits */                                            \
   AOCL_GET_BITS1(label1, zvec, zn);                                                   \
}

#define AOCL_GET_BITS2(lll,vvv,nnn)               \
   s->state = lll;                                \
   UInt32 v;                                      \
   v = (s->bsBuff >>                              \
      (s->bsLive-nnn)) & ((1 << nnn)-1);          \
   s->bsLive -= nnn;                              \
   vvv = v;                                 

#define AOCL_GET_BIT2(lll,uuu)                    \
  AOCL_GET_BITS2(lll,uuu,1)

/* AOCL_GET_MTF_VAL2: Same as AOCL_GET_MTF_VAL1,
   but uses AOCL_GET_BITS2 for bit consumption. */
#define AOCL_GET_MTF_VAL2(label1, label2, lval)                                        \
{                                                                                      \
   if (groupPos == 0) {                                                                \
      groupNo++;                                                                       \
      if (groupNo >= nSelectors)                                                       \
         RETURN(BZ_DATA_ERROR);                                                        \
      groupPos = BZ_G_SIZE;                                                            \
      gSel = s->selector[groupNo];                                                     \
   }                                                                                   \
   groupPos--;                                                                         \
   /* Extract raw Huffman bits from bit stream */                                      \
   int raw_huffman_bits = (s->bsBuff >> (s->bsLive - AOCL_HUFFMAN_MAX_CODE_BITS))      \
                           & ((1 << AOCL_HUFFMAN_MAX_CODE_BITS) - 1);                  \
   /* Primary table lookup for fast decoding */                                        \
   zn = AOCL_HUFFMAN_DECODE_DIRECT;                                                    \
   if (zn < 0) {                                                                       \
      /* Long code: use secondary table */                                             \
      int index = -1 * (zn + 1);                                                       \
      /* Mask to remaining bits for secondary table indexing */                        \
      raw_huffman_bits &= (1 << (AOCL_HUFFMAN_MAX_CODE_BITS - AOCL_BS_BUFF_BITS)) - 1; \
      zn = AOCL_HUFFMAN_DECODE_SECONDARY;                                              \
   }                                                                                   \
   /* Extract symbol and code length from packed data */                               \
   lval = zn >> 8;                                                                     \
   zn = (zn & AOCL_HUFFMAN_CODE_LENGTH_MASK);                                          \
   /* Consume the decoded number of bits */                                            \
   AOCL_GET_BITS2(label1, zvec, zn);                                                   \
}

/*
   Build optimized Huffman code lookup table for fast decoding.

   This function constructs a lookup table that enables fast Huffman decoding 
   by pre-computing decode results for all possible bit patterns. It uses a 
   two-tier approach to handle both short and long Huffman codes efficiently.

   For codes <= AOCL_BS_BUFF_BITS length:
     - Direct lookup table where each entry contains the code length and 
       symbol value packed together
     - Enables single table access for decoding

   For codes > AOCL_BS_BUFF_BITS length:
     - Uses indirect lookup with secondary tables
     - First AOCL_BS_BUFF_BITS bits index into main table to get secondary table index
     - Remaining bits index into the secondary table for final decode
 */
void AOCL_build_huffman_lookup_table( DState* s,
                                 Int32 minLen,
                                 Int32 maxLen,
                                 Int32 alphaSize,
                                 Int32 gSel)
{
   Int32 n, vec, i;
   bz_stream* strm = s->strm;
   
   // Create temporary reference pointers to avoid repetitive s-> accesses
   Int32 *huffman_lookup_table = s->huffman_lookup_table[gSel];
   UChar *length = s->len[gSel];
   Int32 *perm = s->perm[gSel];
   Int32 *base = s->base[gSel];
   Int32 *secondary_table_size = s->secondary_table_size;
   Int32 *secondary_shift_bits = s->secondary_shift_bits;
   Int32 **secondary_tables = s->secondary_tables;

   vec = 0;  // Current canonical Huffman code value
   
   // Phase 1: Build direct lookup table for short codes (length <= AOCL_BS_BUFF_BITS)
   // Each table entry packs: [symbol_value << 8 | code_length]
   for (n = minLen; n <= maxLen && n <= AOCL_BS_BUFF_BITS; n++) {
      for (i = 0; i < alphaSize; i++)
         if (length[i] == n) {
            int len = length[i];
            // Calculate how many table entries this code should fill
            // (replicating the code across all possible suffixes)
            int limit = 1 << (AOCL_BS_BUFF_BITS-len);
            // Left-align the code in the table index space
            int temp_vec = vec << (AOCL_BS_BUFF_BITS-len);
            
            // Fill all table entries for this code (with different suffixes)
            for(int k = 0; k < limit; k++)
            {
               huffman_lookup_table[temp_vec] = len;  // Initialize entry with code length (or -1 for invalid codes)
               if(len != -1)
               {
                  // Extract original code bits and decode symbol
                  int zvec = temp_vec >> (AOCL_BS_BUFF_BITS - len);
                  int lval = perm[zvec - base[len]];  // Look up symbol
                  huffman_lookup_table[temp_vec] |= (lval << 8);     // Store symbol in upper bits
               }
               temp_vec++;
            }
            vec++;  // Move to next code of this length
         }
      vec <<= 1;  // Codes of next length start at vec*2
   }
   
   // Phase 2: Handle long codes (length > AOCL_BS_BUFF_BITS) using secondary tables
   if(maxLen > AOCL_BS_BUFF_BITS)
   {
      // Count how many main table entries need secondary tables (marked as -1)
      int cnt = 0;
      for(int j = 0; j < (1 << AOCL_BS_BUFF_BITS); j++)
      {
        if(huffman_lookup_table[j] == -1)
          cnt++;
      }
      
      // Setup secondary table parameters
      secondary_table_size[gSel] = 1 << (maxLen - AOCL_BS_BUFF_BITS);  // Secondary table size
      secondary_shift_bits[gSel] = maxLen - AOCL_BS_BUFF_BITS;    // Shift bits for secondary table indexing
      
      // Allocate secondary tables for all main table entries that need them
      secondary_tables[gSel] = (int *)BZALLOC(sizeof(int)*secondary_table_size[gSel]*cnt);
      memset(secondary_tables[gSel], 0, sizeof(int)*secondary_table_size[gSel]*cnt);
      
      int index = -1;    // Current secondary table index
      int prev = -1;      // Previous main table index
      // Process remaining long codes
      while(n <= maxLen)
      {
         for (i = 0; i < alphaSize; i++)
            if (length[i] == n) {
               int len = length[i];
               // Replicate this code across all possible suffixes
               int limit = 1 << (maxLen-n);
               int temp_vec = vec << (maxLen-n);
               
               for(int k=0;k<limit;k++)
               {
                  // Extract main table index from upper bits
                  int current = temp_vec >> (maxLen - AOCL_BS_BUFF_BITS);
                  
                  // Assign new secondary table when main table index changes
                  if(prev != current)
                     index++;
                  prev = current;
                  
                  // Mark main table entry to point to secondary table
                  huffman_lookup_table[current] = -1*(index+1);  // Negative value = secondary table index
                  
                  // Calculate position in secondary table
                  int table_index = secondary_table_size[gSel]*index + ((temp_vec) & (secondary_table_size[gSel]-1));
                  
                  // Store code length and symbol in secondary table
                  secondary_tables[gSel][table_index] = n;  // Code length
                  if(len != -1)
                  {
                     int lval = perm[vec - base[len]];           // Decode symbol
                     secondary_tables[gSel][table_index] |= (lval << 8);   // Pack symbol in upper bits
                  }
                  temp_vec++;
               }
               vec++;
            }
         vec <<= 1;
         n++;
      }
   }
}

/*---------------------------------------------------*/
Int32 AOCL_BZ2_decompress ( DState* s )
{
   UChar      uc;
   Int32      retVal;
   Int32      minLen, maxLen;
   bz_stream* strm = s->strm;

   /* stuff that needs to be saved/restored */
   Int32  i;
   Int32  j;
   Int32  t;
   Int32  alphaSize;
   Int32  nGroups;
   Int32  nSelectors;
   Int32  EOB;
   Int32  groupNo;
   Int32  groupPos;
   Int32  nextSym;
   Int32  nblockMAX;
   Int32  nblock;
   Int32  es;
   Int32  N;
   Int32  curr;
   Int32  zt;
   Int32  zn; 
   Int32  zvec;
   Int32  zj;
   Int32  gSel;
   Int32  gMinlen;
   Int32* gLimit;
   Int32* gBase;
   Int32* gPerm;

   if (s->state == BZ_X_MAGIC_1) {
      /*initialise the save area*/
      s->save_i           = 0;
      s->save_j           = 0;
      s->save_t           = 0;
      s->save_alphaSize   = 0;
      s->save_nGroups     = 0;
      s->save_nSelectors  = 0;
      s->save_EOB         = 0;
      s->save_groupNo     = 0;
      s->save_groupPos    = 0;
      s->save_nextSym     = 0;
      s->save_nblockMAX   = 0;
      s->save_nblock      = 0;
      s->save_es          = 0;
      s->save_N           = 0;
      s->save_curr        = 0;
      s->save_zt          = 0;
      s->save_zn          = 0;
      s->save_zvec        = 0;
      s->save_zj          = 0;
      s->save_gSel        = 0;
      s->save_gMinlen     = 0;
      s->save_gLimit      = NULL;
      s->save_gBase       = NULL;
      s->save_gPerm       = NULL;
   }

   /*restore from the save area*/
   i           = s->save_i;
   j           = s->save_j;
   t           = s->save_t;
   alphaSize   = s->save_alphaSize;
   nGroups     = s->save_nGroups;
   nSelectors  = s->save_nSelectors;
   EOB         = s->save_EOB;
   groupNo     = s->save_groupNo;
   groupPos    = s->save_groupPos;
   nextSym     = s->save_nextSym;
   nblockMAX   = s->save_nblockMAX;
   nblock      = s->save_nblock;
   es          = s->save_es;
   N           = s->save_N;
   curr        = s->save_curr;
   zt          = s->save_zt;
   zn          = s->save_zn; 
   zvec        = s->save_zvec;
   zj          = s->save_zj;
   gSel        = s->save_gSel;
   gMinlen     = s->save_gMinlen;
   gLimit      = s->save_gLimit;
   gBase       = s->save_gBase;
   gPerm       = s->save_gPerm;

   retVal = BZ_OK;

   switch (s->state) {

      GET_UCHAR(BZ_X_MAGIC_1, uc);
      if (uc != BZ_HDR_B) RETURN(BZ_DATA_ERROR_MAGIC);

      GET_UCHAR(BZ_X_MAGIC_2, uc);
      if (uc != BZ_HDR_Z) RETURN(BZ_DATA_ERROR_MAGIC);

      GET_UCHAR(BZ_X_MAGIC_3, uc)
      if (uc != BZ_HDR_h) RETURN(BZ_DATA_ERROR_MAGIC);

      GET_BITS(BZ_X_MAGIC_4, s->blockSize100k, 8)
      if (s->blockSize100k < (BZ_HDR_0 + 1) || 
          s->blockSize100k > (BZ_HDR_0 + 9)) RETURN(BZ_DATA_ERROR_MAGIC);
      s->blockSize100k -= BZ_HDR_0;

      if (s->smallDecompress) {
         s->ll16 = BZALLOC( s->blockSize100k * 100000 * sizeof(UInt16) );
         s->ll4  = BZALLOC( 
                      ((1 + s->blockSize100k * 100000) >> 1) * sizeof(UChar) 
                   );
         if (s->ll16 == NULL || s->ll4 == NULL) RETURN(BZ_MEM_ERROR);
      } else {
         s->tt  = BZALLOC( s->blockSize100k * 100000 * sizeof(Int32) );
         if (s->tt == NULL) RETURN(BZ_MEM_ERROR);
      }

      GET_UCHAR(BZ_X_BLKHDR_1, uc);

      if (uc == 0x17) goto endhdr_2;
      if (uc != 0x31) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_BLKHDR_2, uc);
      if (uc != 0x41) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_BLKHDR_3, uc);
      if (uc != 0x59) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_BLKHDR_4, uc);
      if (uc != 0x26) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_BLKHDR_5, uc);
      if (uc != 0x53) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_BLKHDR_6, uc);
      if (uc != 0x59) RETURN(BZ_DATA_ERROR);

      s->currBlockNo++;
      if (s->verbosity >= 2)
         VPrintf1 ( "\n    [%d: huff+mtf ", s->currBlockNo );
 
      s->storedBlockCRC = 0;
      GET_UCHAR(BZ_X_BCRC_1, uc);
      s->storedBlockCRC = (s->storedBlockCRC << 8) | ((UInt32)uc);
      GET_UCHAR(BZ_X_BCRC_2, uc);
      s->storedBlockCRC = (s->storedBlockCRC << 8) | ((UInt32)uc);
      GET_UCHAR(BZ_X_BCRC_3, uc);
      s->storedBlockCRC = (s->storedBlockCRC << 8) | ((UInt32)uc);
      GET_UCHAR(BZ_X_BCRC_4, uc);
      s->storedBlockCRC = (s->storedBlockCRC << 8) | ((UInt32)uc);

      GET_BITS(BZ_X_RANDBIT, s->blockRandomised, 1);

      s->origPtr = 0;
      GET_UCHAR(BZ_X_ORIGPTR_1, uc);
      s->origPtr = (s->origPtr << 8) | ((Int32)uc);
      GET_UCHAR(BZ_X_ORIGPTR_2, uc);
      s->origPtr = (s->origPtr << 8) | ((Int32)uc);
      GET_UCHAR(BZ_X_ORIGPTR_3, uc);
      s->origPtr = (s->origPtr << 8) | ((Int32)uc);

      if (s->origPtr < 0)
         RETURN(BZ_DATA_ERROR);
      if (s->origPtr > 10 + 100000*s->blockSize100k) 
         RETURN(BZ_DATA_ERROR);

      /*--- Receive the mapping table ---*/
      for (i = 0; i < 16; i++) {
         GET_BIT(BZ_X_MAPPING_1, uc);
         if (uc == 1) 
            s->inUse16[i] = True; else 
            s->inUse16[i] = False;
      }

      for (i = 0; i < 256; i++) s->inUse[i] = False;

      for (i = 0; i < 16; i++)
         if (s->inUse16[i])
            for (j = 0; j < 16; j++) {
               GET_BIT(BZ_X_MAPPING_2, uc);
               if (uc == 1) s->inUse[i * 16 + j] = True;
            }
      makeMaps_d ( s );
      if (s->nInUse == 0) RETURN(BZ_DATA_ERROR);
      alphaSize = s->nInUse+2;

      /*--- Now the selectors ---*/
      GET_BITS(BZ_X_SELECTOR_1, nGroups, 3);
      if (nGroups < 2 || nGroups > BZ_N_GROUPS) RETURN(BZ_DATA_ERROR);
      GET_BITS(BZ_X_SELECTOR_2, nSelectors, 15);
      if (nSelectors < 1) RETURN(BZ_DATA_ERROR);
      for (i = 0; i < nSelectors; i++) {
         j = 0;
         while (True) {
            GET_BIT(BZ_X_SELECTOR_3, uc);
            if (uc == 0) break;
            j++;
            if (j >= nGroups) RETURN(BZ_DATA_ERROR);
         }
         /* Having more than BZ_MAX_SELECTORS doesn't make much sense
            since they will never be used, but some implementations might
            "round up" the number of selectors, so just ignore those. */
         if (i < BZ_MAX_SELECTORS)
           s->selectorMtf[i] = j;
      }
      if (nSelectors > BZ_MAX_SELECTORS)
        nSelectors = BZ_MAX_SELECTORS;

      /*--- Undo the MTF values for the selectors. ---*/
      {
         UChar pos[BZ_N_GROUPS], tmp, v;
         for (v = 0; v < nGroups; v++) pos[v] = v;
   
         for (i = 0; i < nSelectors; i++) {
            v = s->selectorMtf[i];
            tmp = pos[v];
            while (v > 0) { pos[v] = pos[v-1]; v--; }
            pos[0] = tmp;
            s->selector[i] = tmp;
         }
      }

      /*--- Now the coding tables ---*/
      for (t = 0; t < nGroups; t++) {
         GET_BITS(BZ_X_CODING_1, curr, 5);
         for (i = 0; i < alphaSize; i++) {
            while (True) {
               if (curr < 1 || curr > 20) RETURN(BZ_DATA_ERROR);
               GET_BIT(BZ_X_CODING_2, uc);
               if (uc == 0) break;
               GET_BIT(BZ_X_CODING_3, uc);
               if (uc == 0) curr++; else curr--;
            }
            s->len[t][i] = curr;
         }
      }

      /*--- Create the Huffman decoding tables ---*/
      for (t = 0; t < nGroups; t++) {
         minLen = 32;
         maxLen = 0;
         for (i = 0; i < alphaSize; i++) {
            if (s->len[t][i] > maxLen) maxLen = s->len[t][i];
            if (s->len[t][i] < minLen) minLen = s->len[t][i];
         }
         BZ2_hbCreateDecodeTables ( 
            &(s->limit[t][0]), 
            &(s->base[t][0]), 
            &(s->perm[t][0]), 
            &(s->len[t][0]),
            minLen, maxLen, alphaSize
         );
         for (int i = 0; i < 1 << AOCL_BS_BUFF_BITS; i++)
           s->huffman_lookup_table[t][i] = -1;
         s->secondary_tables[t] = NULL;
         AOCL_build_huffman_lookup_table(s, minLen, maxLen, alphaSize, t);
         s->minLens[t] = minLen;
      }

      /*--- Now the MTF values ---*/

      EOB      = s->nInUse+1;
      nblockMAX = 100000 * s->blockSize100k;
      groupNo  = -1;
      groupPos = 0;

      for (i = 0; i <= 255; i++) s->unzftab[i] = 0;

      /*
         This pre-populates mtfa with the actual character values, eliminating the need
         for seqToUnseq lookups during decompression and simplifying the MTF decode operation.
      */
      memcpy(s->mtfa, s->seqToUnseq, s->nInUse);

      nblock = 0;
      GET_MTF_VAL(BZ_X_MTF_1, BZ_X_MTF_2, nextSym);
      while (s->strm->avail_in > AOCL_WHILE_LIMIT) {

         if (nextSym == EOB) break;

         if (nextSym == BZ_RUNA || nextSym == BZ_RUNB) {

            es = -1;
            N = 1;
            do {
               /* Check that N doesn't get too big, so that es doesn't
                  go negative.  The maximum value that can be
                  RUNA/RUNB encoded is equal to the block size (post
                  the initial RLE), viz, 900k, so bounding N at 2
                  million should guard against overflow without
                  rejecting any legitimate inputs. */
               if (N >= 2*1024*1024) RETURN(BZ_DATA_ERROR);
               if (nextSym == BZ_RUNA) es = es + (0+1) * N; else
               if (nextSym == BZ_RUNB) es = es + (1+1) * N;
               N = N * 2;

               /* Optimized GET_MTF_VAL for this do..while loop, 
                  so that s->bsBuff is loaded with as many bytes as it can hold at once.*/
               AOCL_GET_MTF_VAL1(BZ_X_MTF_3, BZ_X_MTF_4, nextSym);
            }
               while (nextSym == BZ_RUNA || nextSym == BZ_RUNB);
            es++;
            /*
               Direct access to first MTF element: Since mtfa now contains actual character
               values (not indices), we can directly access mtfa[0] instead of performing
               the indirect lookup s->seqToUnseq[s->mtfa[s->mtfbase[0]]]
            */
            uc = s->mtfa[0];
            s->unzftab[uc] += es;
            if((nblock+es)>=nblockMAX)
               RETURN(BZ_DATA_ERROR);
            if (s->smallDecompress)
               while (es > 0) {
                  s->ll16[nblock] = (UInt16)uc;
                  nblock++;
                  es--;
               }
            else
            {
               do {
                  s->tt[nblock] = (UInt32)uc;
                  s->tt[nblock+1] = (UInt32)uc;
                  s->tt[nblock+2] = (UInt32)uc;
                  s->tt[nblock+3] = (UInt32)uc;
                  s->tt[nblock+4] = (UInt32)uc;
                  s->tt[nblock+5] = (UInt32)uc;
                  s->tt[nblock+6] = (UInt32)uc;
                  s->tt[nblock+7] = (UInt32)uc;
                  s->tt[nblock+8] = (UInt32)uc;
                  s->tt[nblock+9] = (UInt32)uc;
                  s->tt[nblock+10] = (UInt32)uc;
                  s->tt[nblock+11] = (UInt32)uc;
                  s->tt[nblock+12] = (UInt32)uc;
                  s->tt[nblock+13] = (UInt32)uc;
                  s->tt[nblock+14] = (UInt32)uc;
                  s->tt[nblock+15] = (UInt32)uc;
                  nblock += 16;
                  es -= 16;
               } while(es > 0);
               nblock += es;
            }

            continue;

         } else {

            if (nblock >= nblockMAX) RETURN(BZ_DATA_ERROR);

            /*-- uc = MTF ( nextSym-1 ) --*/
            {
               UInt32 nn;
               nn = (UInt32)(nextSym - 1);
               /*
                  Move elements [0..nn-1] one position right,
                  then place the accessed character at position 0. This replaces
                  the complex multi-level array management with mtfbase pointers.
               */
               uc = s->mtfa[nn];
               memmove(&(s->mtfa[1]), &(s->mtfa[0]), nn * sizeof(*s->mtfa));
               s->mtfa[0] = uc;
            }
            /*-- end uc = MTF ( nextSym-1 ) --*/

            s->unzftab[uc]++;
            if (s->smallDecompress)
               s->ll16[nblock] = (UInt16)(uc); else
               s->tt[nblock]   = (UInt32)(uc);
            nblock++;
            
            /* This while loop loads as many bytes as it can into s->bsBuff.*/
            while(s->bsLive+8<=64) {
               s->bsBuff                                   
                  = (s->bsBuff << 8) |                     
                  ((UInt32)                              
                     (*((UChar*)(s->strm->next_in))));   
               s->bsLive += 8;                             
               s->strm->next_in++;                         
               s->strm->avail_in--;                        
               s->strm->total_in_lo32++;                   
               if (s->strm->total_in_lo32 == 0)            
                  s->strm->total_in_hi32++;                
            }

            /* Modified GET_MTF_VAL macro so that the above while loop
               loads enough number of bits into s->bsBuff so as to eliminate
               the need to check if enough number of bits are available
               inside GET_BITS macro which is called 20 times in worst case
               inside GET_MTF_VAL macro.*/
            AOCL_GET_MTF_VAL2(BZ_X_MTF_5, BZ_X_MTF_6, nextSym);
            continue;
         }
      }
      
      gMinlen = s->minLens[gSel];
      gLimit = &(s->limit[gSel][0]);
      gPerm = &(s->perm[gSel][0]);
      gBase = &(s->base[gSel][0]);

      while (True) {

         if (nextSym == EOB) break;

         if (nextSym == BZ_RUNA || nextSym == BZ_RUNB) {

            es = -1;
            N = 1;
            do {
               /* Check that N doesn't get too big, so that es doesn't
                  go negative.  The maximum value that can be
                  RUNA/RUNB encoded is equal to the block size (post
                  the initial RLE), viz, 900k, so bounding N at 2
                  million should guard against overflow without
                  rejecting any legitimate inputs. */
               if (N >= 2*1024*1024) RETURN(BZ_DATA_ERROR);
               if (nextSym == BZ_RUNA) es = es + (0+1) * N; else
               if (nextSym == BZ_RUNB) es = es + (1+1) * N;
               N = N * 2;
               GET_MTF_VAL(BZ_X_MTF_3, BZ_X_MTF_4, nextSym);
            }
               while (nextSym == BZ_RUNA || nextSym == BZ_RUNB);

            es++;
            uc = s->mtfa[0];
            s->unzftab[uc] += es;

            if (s->smallDecompress)
               while (es > 0) {
                  if (nblock >= nblockMAX) RETURN(BZ_DATA_ERROR);
                  s->ll16[nblock] = (UInt16)uc;
                  nblock++;
                  es--;
               }
            else
               while (es > 0) {
                  if (nblock >= nblockMAX) RETURN(BZ_DATA_ERROR);
                  s->tt[nblock] = (UInt32)uc;
                  nblock++;
                  es--;
               };

            continue;

         } else {

            if (nblock >= nblockMAX) RETURN(BZ_DATA_ERROR);

            {
               UInt32 nn;
               nn = (UInt32)(nextSym - 1);
               uc = s->mtfa[nn];
               memmove(&(s->mtfa[1]), &(s->mtfa[0]), nn * sizeof(*s->mtfa));
               s->mtfa[0] = uc;
            }

            s->unzftab[uc]++;
            if (s->smallDecompress)
               s->ll16[nblock] = (UInt16)(uc); else
               s->tt[nblock]   = (UInt32)(uc);
            nblock++;

            GET_MTF_VAL(BZ_X_MTF_5, BZ_X_MTF_6, nextSym);
            continue;
         }
      }

      /* Now we know what nblock is, we can do a better sanity
         check on s->origPtr.
      */
      if (s->origPtr < 0 || s->origPtr >= nblock)
         RETURN(BZ_DATA_ERROR);

      /*-- Set up cftab to facilitate generation of T^(-1) --*/
      /* Check: unzftab entries in range. */
      for (i = 0; i <= 255; i++) {
         if (s->unzftab[i] < 0 || s->unzftab[i] > nblock)
            RETURN(BZ_DATA_ERROR);
      }
      /* Actually generate cftab. */
      s->cftab[0] = 0;
      for (i = 1; i <= 256; i++) s->cftab[i] = s->unzftab[i-1];
      for (i = 1; i <= 256; i++) s->cftab[i] += s->cftab[i-1];
      
      /*
         Create a compacted cumulative frequency table for active characters only.
         This table maps sequence indices (0 to nInUse-1) to cumulative frequencies,
         enabling efficient character lookup during BWT reconstruction.

         The algorithm works as follows:
         1. Use either "Fast path" or "Fallback to precise binary search" to find the character at position x.
         2. If x falls between char_boundaries[i] and char_boundaries[i+1], 
            then the character at position x is seqToUnseq[i]
   
         Example: char_boundaries = [0, 100, 250, 400, 500]
                  If x = 275, it falls in range [250, 400) at index 2
                  So character = seqToUnseq[2]
       */
      Int32 char_boundaries[257] = {0};  // Compacted character boundary lookup table
      for(i = 0; i < s->nInUse; i++)
      {
         // Map sequence index to cumulative frequency of actual character
         char_boundaries[i] = s->cftab[s->seqToUnseq[i]];
      }
      char_boundaries[s->nInUse] = s->cftab[256];  // End boundary for range checks

      /* Check: cftab entries in range. */
      for (i = 0; i <= 256; i++) {
         if (s->cftab[i] < 0 || s->cftab[i] > nblock) {
            /* s->cftab[i] can legitimately be == nblock */
            RETURN(BZ_DATA_ERROR);
         }
      }
      /* Check: cftab entries non-descending. */
      for (i = 1; i <= 256; i++) {
         if (s->cftab[i-1] > s->cftab[i]) {
            RETURN(BZ_DATA_ERROR);
         }
      }

      s->state_out_len = 0;
      s->state_out_ch  = 0;
      BZ_INITIALISE_CRC ( s->calculatedBlockCRC );
      s->state = BZ_X_OUTPUT;
      if (s->verbosity >= 2) VPrintf0 ( "rt+rld" );

      if (s->smallDecompress) {

         /*-- Make a copy of cftab, used in generation of T --*/
         for (i = 0; i <= 256; i++) s->cftabCopy[i] = s->cftab[i];

         /*-- compute the T vector --*/
         for (i = 0; i < nblock; i++) {
            uc = (UChar)(s->ll16[i]);
            SET_LL(i, s->cftabCopy[uc]);
            s->cftabCopy[uc]++;
         }

         /*-- Compute T^(-1) by pointer reversal on T --*/
         i = s->origPtr;
         j = GET_LL(i);
         do {
            Int32 tmp = GET_LL(j);
            SET_LL(j, i);
            i = j;
            j = tmp;
         }
            while (i != s->origPtr);

         s->tPos = s->origPtr;
         s->nblock_used = 0;
         if (s->blockRandomised) {
            BZ_RAND_INIT_MASK;
            BZ_GET_SMALL(s->k0); s->nblock_used++;
            BZ_RAND_UPD_MASK; s->k0 ^= BZ_RAND_MASK; 
         } else {
            BZ_GET_SMALL(s->k0); s->nblock_used++;
         }

      } else {
         /*-- Compute the T^(-1) vector (inverse BWT transform table) --*/
         if(nblock >= (AOCL_RANGE_THRESHOLD) && s->blockRandomised == 0)
         {
            /*
               Optimized path for large, non-randomized blocks
               Uses two-pass algorithm to avoid memory conflicts and improve cache locality
             */
            
            /* Allocate temporary buffer if not already available */
            if(s->temp_tt == NULL)
            {
               s->temp_tt  = BZALLOC( s->blockSize100k * 100000 * sizeof(UInt32) );
               if (s->temp_tt == NULL) RETURN(BZ_MEM_ERROR);
            }
            UInt32 * temp_tt = s->temp_tt;

            /*
               First pass: Build temporary table with original positions
               For each character in the BWT, store its original position (i << 8)
               at the location determined by the cumulative frequency table
             */
            for (i = 0; i < nblock; i++) {
               uc = (UChar)(s->tt[i]);    /* Extract character */
               temp_tt[s->cftab[uc]++] = (i << 8);   /* Store char + position link */
            }
            
            /* Rebuild cumulative frequency table from character frequencies */
            s->cftab[0] = 0;
            for (i = 1; i <= 256; i++) s->cftab[i] = s->unzftab[i-1];
            for (i = 1; i <= 256; i++) s->cftab[i] += s->cftab[i-1];
            
            /*
               Second pass: Combine character data with position data
               Build final transform table where each entry contains:
               - Lower 8 bits: the character value at current position
               - Upper 24 bits: pointer to next-to-next position in sequence
               Note: Character at next position can be determined from current index using temp_cftab
             */
            for (i = 0; i < nblock; i++) {
               uc = (UChar)(s->tt[i]);
               s->tt[s->cftab[uc]++] |= s->temp_tt[i];
            }

            /* Restore original cumulative frequency table for later use */
            for (i = 0; i <= s->nInUse; i++) {
               s->cftab[i] = char_boundaries[i];
            }
            s->tPos = 0;
            s->nblock_used = 0;
         }
         else {
            /*
               Standard single-pass algorithm for smaller blocks or randomized data
               Directly builds the inverse transform table by combining character
               and position information in one step
             */
            for (i = 0; i < nblock; i++) {
               uc = (UChar)(s->tt[i] & 0xff);  /* Extract character */
               s->tt[s->cftab[uc]++] |= (i << 8);  /* Store char + position link */
            }
   
            /* Set initial position for BWT reconstruction */
            s->tPos = s->tt[s->origPtr] >> 8;
            s->nblock_used = 0;
         }
         if (s->blockRandomised) {
            BZ_RAND_INIT_MASK;
            BZ_GET_FAST(s->k0);
            s->nblock_used++;
            BZ_RAND_UPD_MASK;
            s->k0 ^= BZ_RAND_MASK;
         } else {
            if(!(nblock >= (AOCL_RANGE_THRESHOLD)))
            {
               BZ_GET_FAST(s->k0); s->nblock_used++;
            }
         }

      }

      RETURN(BZ_OK);



    endhdr_2:

      GET_UCHAR(BZ_X_ENDHDR_2, uc);
      if (uc != 0x72) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_ENDHDR_3, uc);
      if (uc != 0x45) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_ENDHDR_4, uc);
      if (uc != 0x38) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_ENDHDR_5, uc);
      if (uc != 0x50) RETURN(BZ_DATA_ERROR);
      GET_UCHAR(BZ_X_ENDHDR_6, uc);
      if (uc != 0x90) RETURN(BZ_DATA_ERROR);

      s->storedCombinedCRC = 0;
      GET_UCHAR(BZ_X_CCRC_1, uc);
      s->storedCombinedCRC = (s->storedCombinedCRC << 8) | ((UInt32)uc);
      GET_UCHAR(BZ_X_CCRC_2, uc);
      s->storedCombinedCRC = (s->storedCombinedCRC << 8) | ((UInt32)uc);
      GET_UCHAR(BZ_X_CCRC_3, uc);
      s->storedCombinedCRC = (s->storedCombinedCRC << 8) | ((UInt32)uc);
      GET_UCHAR(BZ_X_CCRC_4, uc);
      s->storedCombinedCRC = (s->storedCombinedCRC << 8) | ((UInt32)uc);

      s->state = BZ_X_IDLE;
      RETURN(BZ_STREAM_END);

      default: AssertH ( False, 4001 );
   }

   AssertH ( False, 4002 );

   save_state_and_return:

   for(int i=0;i<nGroups;i++)
   {
      if(s->secondary_tables[i])
      {
         BZFREE(s->secondary_tables[i]);
         s->secondary_tables[i] = NULL;
      }
   }

   s->save_i           = i;
   s->save_j           = j;
   s->save_t           = t;
   s->save_alphaSize   = alphaSize;
   s->save_nGroups     = nGroups;
   s->save_nSelectors  = nSelectors;
   s->save_EOB         = EOB;
   s->save_groupNo     = groupNo;
   s->save_groupPos    = groupPos;
   s->save_nextSym     = nextSym;
   s->save_nblockMAX   = nblockMAX;
   s->save_nblock      = nblock;
   s->save_es          = es;
   s->save_N           = N;
   s->save_curr        = curr;
   s->save_zt          = zt;
   s->save_zn          = zn;
   s->save_zvec        = zvec;
   s->save_zj          = zj;
   s->save_gSel        = gSel;
   s->save_gMinlen     = gMinlen;
   s->save_gLimit      = gLimit;
   s->save_gBase       = gBase;
   s->save_gPerm       = gPerm;

   return retVal;   
}

#endif

/*-------------------------------------------------------------*/
/*--- end                                      decompress.c ---*/
/*-------------------------------------------------------------*/

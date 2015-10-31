/*
Header Note
-----------
WARNING: In November 1993, someone found a bug in lzrw1-a.68000
which arises if the last item in a compressed data is a copy item of lengh
17 or 18. I haven't checked the code below, but the bug could be in here too.
I'm not sure. So please check for this before using this code.
If yopu don't want to bother with it, just use LZRW1.c.
-Ross.


THE LZRW1-A ALGORITHM IN C
==========================
Author : Ross N. Williams.
Date   : 25-Jun-1991.

1. This is my implementation in  C of my LZRW1-A algorithm. LZRW1-A
is a direct descendant of LZRW1 and was derived by:
   a. Increasing the copy length range from 3..16 to 3..18.
   b. Performing extensive optmizations.

2. This file has been copied into a test harness and works.

3. This code is public domain.  The algorithm is not patented and is a
member  of the  LZ77 class  of algorithms  which seem  to be  clear of
patent challenges.

4. Warning:  This code  is non-deterministic insofar  as it  may yield
different  compressed representations  of the  same file  on different
runs. (However, it will always decompress correctly to the original).

5. If you use this code in anger (e.g. in a product) drop me a note at
ross@spam.ua.oz.au and I will put you  on a mailing list which will be
invoked if anyone finds a bug in this code.

6.   The  internet   newsgroup  comp.compression   might  also   carry
information on this algorithm from time to time.

7. This  code makes use  of a 68000  memory block copy  routine called
fast_copy   which   is   available   in   a   separate   file   called
fast_copy.68000. The first argument is  the source address, the second
the  destination address  and  the third,  the  length  in bytes.  See
fast_copy.68000 for an exact formal definition of the semantics of the
fast_copy procedure.

8. Header files can be found in lzrw_headers.h.

/******************************************************************************/
/*                                                                            */
/*                                  LZRW1-A.C                                 */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/* Author  : Ross Williams.                                                   */
/* Date    : 25 June 1991.                                                    */
/* Release : 1.                                                               */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/* This file contains an implementation of the LZRW1-A data compression       */
/* algorithm in C.                                                            */
/*                                                                            */
/* The algorithm is a general purpose compression algorithm that runs fast    */
/* and gives reasonable compression. The algorithm is a member of the Lempel  */
/* Ziv family of algorithms and bases its compression on the presence in the  */
/* data of repeated substrings.                                               */
/*                                                                            */
/* The algorithm/code is based on the LZRW1 algorithm/code. Changes are:      */
/*    1) The copy length range is now 3..18 instead of 3..16 as in LZRW1.     */
/*    2) The code for both the compressor and decompressor has been optimized */
/*       and made a little more portable.                                     */
/*                                                                            */
/* This algorithm and code is public domain. As the algorithm is based on the */
/* LZ77 class of algorithms, it is unlikely to be the subject of a patent     */
/* challenge.                                                                 */
/*                                                                            */
/* WARNING: This algorithm is non-deterministic. Its compression performance  */
/* may vary slightly from run to run.                                         */
/*                                                                            */
/******************************************************************************/

                            /* INCLUDE FILES                                  */
                            /* =============                                  */
#include "lzrw.h"
#define ULONG uint32_t

/******************************************************************************/

/* The following structure is returned by the "compress" function below when  */
/* the user asks the function to return identifying information.              */
/* The most important field in the record is the working memory field which   */
/* tells the calling program how much working memory should be passed to      */
/* "compress" when it is called to perform a compression or decompression.    */
/* For more information on this structure see "compress.h".                   */

#define U(X) ((ULONG) X)

static struct compress_identity identity =
{
 U(0x4B3E387B),                           /* Algorithm identification number. */
 U(U(4)*U(4096)+U(3)),                    /* Working memory (bytes) to alg.   */
 "LZRW1-A",                               /* Name of algorithm.               */
 "1.0",                                   /* Version number of algorithm.     */
 "22-Jun-1991",                           /* Date of algorithm.               */
 "Public Domain",                         /* Copyright notice.                */
 "Ross N. Williams",                      /* Author of algorithm.             */
 "Renaissance Software",                  /* Affiliation of author.           */
 "Public Domain"                          /* Vendor of algorithm.             */
};

void lzrw1a_compress_compress  (UBYTE *,UBYTE *,ULONG,UBYTE *,ULONG *);
void lzrw1a_compress_decompress(UBYTE *,UBYTE *,ULONG,UBYTE *,ULONG *);

/******************************************************************************/

/* This function is the only function exported by this module.                */
/* Depending on its first parameter, the function can be requested to         */
/* compress a block of memory, decompress a block of memory, or to identify   */
/* itself. For more information, see the specification file "compress.h".     */

EXPORT void lzrw1a_compress
	(UWORD action, UBYTE *wrk_mem,UBYTE *src_adr,ULONG src_len,UBYTE *dst_adr,ULONG* p_dst_len)
{
 switch (action)
   {
//     case COMPRESS_ACTION_IDENTITY:
//        *p_dst_len=(ULONG) &identity;
//        break;
    case COMPRESS_ACTION_COMPRESS:
       lzrw1a_compress_compress(wrk_mem,src_adr,src_len,dst_adr,p_dst_len);
       break;
    case COMPRESS_ACTION_DECOMPRESS:
       lzrw1a_compress_decompress(wrk_mem,src_adr,src_len,dst_adr,p_dst_len);
       break;
   }
}

/******************************************************************************/
/*                                                                            */
/* The remainder of this file contains some definitions and two more          */
/* functions, one for compression and one for decompression. This section     */
/* contains information and definitions common to both algorithms.            */
/* Most of this information relates to the compression format which is common */
/* to both routines.                                                          */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/*                     DEFINITION OF COMPRESSED FILE FORMAT                   */
/*                     ====================================                   */
/*  * A compressed file consists of a COPY FLAG followed by a REMAINDER.      */
/*  * The copy flag CF uses up four bytes with the first byte being the       */
/*    least significant.                                                      */
/*  * If CF=1, then the compressed file represents the remainder of the file  */
/*    exactly. Otherwise CF=0 and the remainder of the file consists of zero  */
/*    or more GROUPS, each of which represents one or more bytes.             */
/*  * Each group consists of two bytes of CONTROL information followed by     */
/*    sixteen ITEMs except for the last group which can contain from one      */
/*    to sixteen items.                                                       */
/*  * An item can be either a LITERAL item or a COPY item.                    */
/*  * Each item corresponds to a bit in the control bytes.                    */
/*  * The first control byte corresponds to the first 8 items in the group    */
/*    with bit 0 corresponding to the first item in the group and bit 7 to    */
/*    the eighth item in the group.                                           */
/*  * The second control byte corresponds to the second 8 items in the group  */
/*    with bit 0 corresponding to the ninth item in the group and bit 7 to    */
/*    the sixteenth item in the group.                                        */
/*  * A zero bit in a control word means that the corresponding item is a     */
/*    literal item. A one bit corresponds to a copy item.                     */
/*  * A literal item consists of a single byte which represents itself.       */
/*  * A copy item consists of two bytes that represent from 3 to 18 bytes.    */
/*  * The first byte in a copy item will be denoted C1.                       */
/*  * The second byte in a copy item will be denoted C2.                      */
/*  * Bits will be selected using square brackets.                            */
/*    For example: C1[0..3] is the low nibble of the first control byte.      */
/*    of copy item C1.                                                        */
/*  * The LENGTH of a copy item is defined to be C1[0..3]+3 which is a number */
/*    in the range [3,18].                                                    */
/*  * The OFFSET of a copy item is defined to be C1[4..7]*256+C2[0..8] which  */
/*    is a number in the range [1,4095] (the value 0 is never used).          */
/*  * A copy item represents the sequence of bytes                            */
/*       text[POS-OFFSET..POS-OFFSET+LENGTH-1] where "text" is the entire     */
/*    text of the uncompressed string, and POS is the index in the text of    */
/*    the character following the string represented by all the items         */
/*    preceeding the item being defined.                                      */
/*                                                                            */
/******************************************************************************/

/* The following define defines the length of the copy flag that appears at   */
/* the start of the compressed file. I have decided on four bytes so as to    */
/* make the source and destination longword aligned in the case where a copy  */
/* operation must be performed.                                               */
/* The actual flag data appears in the first byte. The rest are zero.         */
#define FLAG_BYTES    4     /* How many bytes does the flag use up?           */

/* The following defines define the meaning of the values of the copy         */
/* flag at the start of the compressed file.                                  */
#define FLAG_COMPRESS 0     /* Signals that output was result of compression. */
#define FLAG_COPY     1     /* Signals that output was simply copied over.    */

/******************************************************************************/

LOCAL void lzrw1a_compress_compress
	(UBYTE *p_wrk_mem,UBYTE *p_src_first,ULONG src_len,UBYTE *p_dst_first,ULONG* p_dst_len)
/* Input  : Specify input block using p_src_first and src_len.          */
/* Input  : Point p_dst_first to the start of the output zone (OZ).     */
/* Input  : Point p_dst_len to a ULONG to receive the output length.    */
/* Input  : Input block and output zone must not overlap.               */
/* Output : Length of output block written to *p_dst_len.               */
/* Output : Output block in Mem[p_dst_first..p_dst_first+*p_dst_len-1]. */
/* Output : May write in OZ=Mem[p_dst_first..p_dst_first+src_len+288-1].*/
/* Output : Upon completion guaranteed *p_dst_len<=src_len+FLAG_BYTES.  */
#define PS *p++!=*p_src++  /* Body of inner unrolled matching loop.     */
#define ITEMMAX 18         /* Max number of bytes in an expanded item.  */
#define TOPWORD 0xFFFF0000
{register UBYTE *p_src=p_src_first,*p_dst=p_dst_first;
 UBYTE *p_src_post=p_src_first+src_len,*p_dst_post=p_dst_first+src_len;
 UBYTE *p_src_max1,*p_src_max16;
 /* The following longword aligns the hash table in the working memory. */
 register UBYTE **hash= (UBYTE **) (p_wrk_mem);
 UBYTE *p_control; register ULONG control=TOPWORD;
 p_src_max1=p_src_post-ITEMMAX; p_src_max16=p_src_post-16*ITEMMAX;
 *p_dst=FLAG_COMPRESS; {UWORD i; for (i=1;i<FLAG_BYTES;i++) p_dst[i]=0;}
 p_dst+=FLAG_BYTES; p_control=p_dst; p_dst+=2;
 while (TRUE)
   {register UBYTE *p,**p_entry; register UWORD unroll=16;
    register ULONG offset;
    if (p_dst>p_dst_post) goto overrun;
    if (p_src>p_src_max16)
      {unroll=1;
       if (p_src>p_src_max1)
         {if (p_src==p_src_post) break; goto literal;}}
    begin_unrolled_loop:
       p_entry=&hash
          [((40543*((((p_src[0]<<4)^p_src[1])<<4)^p_src[2]))>>4) & 0xFFF];
       p=*p_entry; *p_entry=p_src; offset=p_src-p;
       if (offset>4095 || p<p_src_first || offset==0 || PS || PS || PS)
         {p_src=*p_entry; literal: *p_dst++=*p_src++; control&=0xFFFEFFFF;}
       else
         {PS || PS || PS || PS || PS || PS || PS || PS ||
          PS || PS || PS || PS || PS || PS || PS || p_src++;
          *p_dst++=((offset&0xF00)>>4)|(--p_src-*p_entry-3);
          *p_dst++=offset&0xFF;}
       control>>=1;
    end_unrolled_loop: if (--unroll) goto begin_unrolled_loop;
    if ((control&TOPWORD) == 0)
      {*p_control=control&0xFF; *(p_control+1)=(control>>8)&0xFF;
       p_control=p_dst; p_dst+=2; control=TOPWORD;}
   }
 while (control&TOPWORD) control>>=1;
 *p_control++=control&0xFF; *p_control++=control>>8;
 if (p_control==p_dst) p_dst-=2;
 *p_dst_len=p_dst-p_dst_first;
 return;
 overrun: fast_copy(p_src_first,p_dst_first+FLAG_BYTES,src_len);
          *p_dst_first=FLAG_COPY; *p_dst_len=src_len+FLAG_BYTES;
}

/******************************************************************************/

LOCAL void lzrw1a_compress_decompress
	(UBYTE *p_wrk_mem,UBYTE *p_src_first,ULONG src_len,UBYTE *p_dst_first,ULONG* p_dst_len)
/* Input  : Specify input block using p_src_first and src_len.          */
/* Input  : Point p_dst_first to the start of the output zone.          */
/* Input  : Point p_dst_len to a ULONG to receive the output length.    */
/* Input  : Input block and output zone must not overlap. User knows    */
/* Input  : upperbound on output block length from earlier compression. */
/* Input  : In any case, maximum expansion possible is nine times.      */
/* Output : Length of output block written to *p_dst_len.               */
/* Output : Output block in Mem[p_dst_first..p_dst_first+*p_dst_len-1]. */
/* Output : Writes only  in Mem[p_dst_first..p_dst_first+*p_dst_len-1]. */
{register UBYTE *p_src=p_src_first+FLAG_BYTES, *p_dst=p_dst_first;
 UBYTE *p_src_post=p_src_first+src_len;
 UBYTE *p_src_max16=p_src_first+src_len-(16*2);
 register ULONG control=1;
 if (*p_src_first==FLAG_COPY)
   {fast_copy(p_src_first+FLAG_BYTES,p_dst_first,src_len-FLAG_BYTES);
    *p_dst_len=src_len-FLAG_BYTES; return;}
 while (p_src!=p_src_post)
   {register UWORD unroll;
    if (control==1) {control=0x10000|*p_src++; control|=(*p_src++)<<8;}
    unroll= p_src<=p_src_max16 ? 16 : 1;
    while (unroll--)
      {if (control&1)
         {register UWORD lenmt; register UBYTE *p;
          lenmt=*p_src++; p=p_dst-(((lenmt&0xF0)<<4)|*p_src++);
          *p_dst++=*p++; *p_dst++=*p++; *p_dst++=*p++;
          lenmt&=0xF; while (lenmt--) *p_dst++=*p++;}
       else
          *p_dst++=*p_src++;
       control>>=1;
      }
   }
 *p_dst_len=p_dst-p_dst_first;
}

/******************************************************************************/
/*                             End of LZRNW1-A.C                              */
/******************************************************************************/

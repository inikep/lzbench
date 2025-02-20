/******************************************************************************/
/*                                                                            */
/*                                   LZRW3-A.C                                */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/* Author  : Ross Williams.                                                   */
/* Date    : 15-Jul-1991.                                                     */
/* Release : 1.                                                               */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/* This file contains an implementation of the LZRW3-A data compression       */
/* algorithm in the C programming language.                                   */
/*                                                                            */
/* The LZRW3-A algorithm has the following features:                          */
/*                                                                            */
/*    1 Requires only 16K of memory (for both compression and decompression). */
/*    2 The compressor   runs about two   times faster than Unix compress's.  */
/*    3 The decompressor runs about three times faster than Unix compress's.  */
/*    4 Yields a few percent better compression than Unix compress for        */
/*      most files.                                                           */
/*    5 Allows you to dial up extra compression at a speed cost in the        */
/*      compressor. The speed of the decompressor is not affected.            */
/*    6 Algorithm is deterministic.                                           */
/*    7 Algorithm is free of patent problems. The algorithm has not been      */
/*      patented (nor will it be) and is of the LZ77 class which is fairly    */
/*      clear of patents.                                                     */
/*    8 This implementation in C is in the public domain.                     */
/*                                                                            */
/* (Timing tests for the speed comparison were performed on a Pyramid 9820.)  */
/*                                                                            */
/* LZRW3-A is LZRW3 with a deepened hash table. This simple change yields     */
/* about a 6% (absolute) improvement in compression.                          */
/*                                                                            */
/* Here are the results of applying this code, compiled under THINK C 4.0     */
/* and running on a Mac-SE (8MHz 68000), to the standard calgary corpus.      */
/*                                                                            */
/*     +----------------------------------------------------------------+     */
/*     | DATA COMPRESSION TEST                                          |     */
/*     | =====================                                          |     */
/*     | Time of run     : Mon 15-Jul-1991 05:29PM                      |     */
/*     | Timing accuracy : One part in 100                              |     */
/*     | Context length  : 262144 bytes (= 256.0000K)                   |     */
/*     | Test suite      : Calgary Corpus Suite                         |     */
/*     | Files in suite  : 14                                           |     */
/*     | Algorithm       : LZRW3-A                                      |     */
/*     | Note: All averages are calculated from the un-rounded values.  |     */
/*     +----------------------------------------------------------------+     */
/*     | File Name   Length  CxB  ComLen  %Remn  Bits  Com K/s  Dec K/s |     */
/*     | ----------  ------  ---  ------  -----  ----  -------  ------- |     */
/*     | rpus:Bib.D  111261    1   49044   44.1  3.53     8.47    31.19 |     */
/*     | us:Book1.D  768771    3  420464   54.7  4.38     7.27    30.07 |     */
/*     | us:Book2.D  610856    3  277955   45.5  3.64     8.51    33.40 |     */
/*     | rpus:Geo.D  102400    1   84218   82.2  6.58     4.23    15.04 |     */
/*     | pus:News.D  377109    2  192880   51.1  4.09     7.08    25.89 |     */
/*     | pus:Obj1.D   21504    1   12651   58.8  4.71     5.23    17.44 |     */
/*     | pus:Obj2.D  246814    1  108044   43.8  3.50     8.01    28.11 |     */
/*     | s:Paper1.D   53161    1   24526   46.1  3.69     8.11    30.24 |     */
/*     | s:Paper2.D   82199    1   39483   48.0  3.84     8.11    32.04 |     */
/*     | rpus:Pic.D  513216    2  111622   21.7  1.74    10.64    49.31 |     */
/*     | us:Progc.D   39611    1   17923   45.2  3.62     8.06    29.01 |     */
/*     | us:Progl.D   71646    1   24362   34.0  2.72    10.74    39.51 |     */
/*     | us:Progp.D   49379    1   16805   34.0  2.72    10.64    37.58 |     */
/*     | us:Trans.D   93695    1   30296   32.3  2.59    11.02    38.06 |     */
/*     +----------------------------------------------------------------+     */
/*     | Average     224401    1  100733   45.8  3.67     8.29    31.21 |     */
/*     +----------------------------------------------------------------+     */
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
/* LZRW3-A uses the same amount of memory during compression and              */
/* decompression. For more information on this structure see "compress.h".    */
/* The alignment fudge below really only needs to be 4 (but I play it safe!). */
/* The id looks non-random, but it really was generated by coin tossing!      */

#define U(X)            ((ULONG) X)
#define SIZE_P_BYTE     (U(sizeof(UBYTE *)))
#define ALIGNMENT_FUDGE (U(16))
#define MEM_REQ ( U(4096)*(SIZE_P_BYTE) + ALIGNMENT_FUDGE )

static struct compress_identity identity =
{
 U(0x01B90B91),                           /* Algorithm identification number. */
 MEM_REQ,                                 /* Working memory (bytes) required. */
 "LZRW3-A",                               /* Name of algorithm.               */
 "1.0",                                   /* Version number of algorithm.     */
 "15-Jul-1990",                           /* Date of algorithm.               */
 "Public Domain",                         /* Copyright notice.                */
 "Ross N. Williams",                      /* Author of algorithm.             */
 "Renaissance Software",                  /* Affiliation of author.           */
 "Public Domain"                          /* Vendor of algorithm.             */
};

void lzrw3a_compress_compress  (UBYTE *,UBYTE *,ULONG,UBYTE *,ULONG *);
void lzrw3a_compress_decompress(UBYTE *,UBYTE *,ULONG,UBYTE *,ULONG *);

/******************************************************************************/

/* This function is the only function exported by this module.                */
/* Depending on its first parameter, the function can be requested to         */
/* compress a block of memory, decompress a block of memory, or to identify   */
/* itself. For more information, see the specification file "compress.h".     */


EXPORT void lzrw3a_compress
	(UWORD action, UBYTE *wrk_mem,UBYTE *src_adr,ULONG src_len,UBYTE *dst_adr,ULONG* p_dst_len)
{
 switch (action)
   {
    case COMPRESS_ACTION_COMPRESS:
       lzrw3a_compress_compress(wrk_mem,src_adr,src_len,dst_adr,p_dst_len);
       break;
    case COMPRESS_ACTION_DECOMPRESS:
       lzrw3a_compress_decompress(wrk_mem,src_adr,src_len,dst_adr,p_dst_len);
       break;
   }
}

/******************************************************************************/
/*                                                                            */
/* BRIEF DESCRIPTION OF THE LZRW3-A ALGORITHM                                 */
/* ==========================================                                 */
/* Note: Before attempting to understand this algorithm, you should first     */
/* understand the LZRW3 algorithm from which this algorithm is derived.       */
/*                                                                            */
/* The LZRW3-A algorithm is identical to the LZRW3 algorithm except that the  */
/* hash table has been "deepened". The LZRW3 algorithm has a hash table of    */
/* 4096 pointers which point to strings in the buffer. LZRW3-A generalizes    */
/* this to 4096/(2^n) partitions each of which contains (2^n) pointers.       */
/* In LZRW3-A, the hash function hashes to a partition number.                */
/*                                                                            */
/* During the processing of each phrase, LZRW3 overwrites the pointer in the  */
/* position selected by the hash function. LZRW3-A overwrites one of the      */
/* pointers in the partition that was selected by the hash function.          */
/*                                                                            */
/* When searching for a match, LZRW3-A matches against all (2^n) strings      */
/* pointed to by the pointers in the target partition.                        */
/*                                                                            */
/* Deep hash tables were used in early versions of LZRW1 in late 1989, but    */
/* were discarded in an effort to increase speed (which was the primary       */
/* requirement for LZRW1). They were revived for use in LZRW3-A in order to   */
/* produce an algorithm with compression performance competitive with Unix    */
/* compress.                                                                  */
/*                                                                            */
/* Until 14-Jul-1991, deep hash tables used in prototype LZRW* algorithms     */
/* used a queue discipline within each partition. Upon the arrival of a new   */
/* pointer, the pointers in the partition would be block copied back one      */
/* position (with the oldest pointer being overwritten) and the new pointer   */
/* being inserted in the space at the front (the youngest position).          */
/* This meant that pointers to the (2^n) most recent phrases corresponding to */
/* each hash was kept. The only flaw in this system was the time-consuming    */
/* block copy operation which was cheap for shallow tables but expensive for  */
/* deep tables.                                                               */
/*                                                                            */
/* The traditional solution to ring buffer block copy problems is to maintain */
/* a cyclic counter which points to the "head" of the queue. However, this    */
/* would have required one counter to be stored for each partition and would  */
/* have been slightly messy. After some thought (on 14-Jul-1991) a better     */
/* solution was found. Instead of maintaining a counter for each partition,   */
/* LZRW3-A maintains a single counter for all partitions! This counter is     */
/* maintained in both the compressor and decompressor and means that the      */
/* algorithm (effectively) overwrites a RANDOM element of the partition to be */
/* updated. The result was to increase the speed of the compressor and        */
/* decompressor, to make the decompressor's speed independent from whatever   */
/* depth was selected, and to impair compression by less than 1% absolute.    */
/*                                                                            */
/* Setting the depth is a speed/compression tradeoff. The table below gives   */
/* the tradeoff observed for a typical 50K text file on a Mac-SE.             */
/* Note: %Rem=Percentage Remaining (after compression).                       */
/*                                                                            */
/*      Depth    %Rem    CmpK/s  DecK/s                                       */
/*          1    45.2    14.77   32.24                                        */
/*          2    42.6    12.12   31.26                                        */
/*          4    40.9    10.28   31.91                                        */
/*          8    40.0     7.81   32.36                                        */
/*         16    39.5     5.30   32.47                                        */
/*         32    39.0     3.23   32.59                                        */
/*                                                                            */
/* I have chosen a depth of 8 as the "default" depth for LZRW3-A. If you use  */
/* a depth different to this (e.g. 4), you should use the name LZRW3-A(4) to  */
/* indicate that a different depth is being used. LZRW3-A(8) is an acceptable */
/* longhand for LZRW3-A.                                                      */
/*                                                                            */
/* To change the depth, search for "HERE IT IS" in the rest of this file.     */
/*                                                                            */
/*                                  +---+                                     */
/*                                  |___|4095                                 */
/*                                  |===|                                     */
/*              +---------------------*_|<---+   /----+---\                   */
/*              |                   |___|    +---|Hash    |                   */
/*              |    512 partitions |___|        |Function|                   */
/*              |    of 8 pointers  |===|        \--------/                   */
/*              |    each (or any   |___|0            ^                       */
/*              |    a*b=4096)      +---+             |                       */
/*              |                   Hash        +-----+                       */
/*              |                   Table       |                             */
/*              |                              ---                            */
/*              v                              ^^^                            */
/*      +-------------------------------------|----------------+              */
/*      ||||||||||||||||||||||||||||||||||||||||||||||||||||||||              */
/*      +-------------------------------------|----------------+              */
/*      |                                     |1......18|      |              */
/*      |<------- Lempel=History ------------>|<--Ziv-->|      |              */
/*      |     (=bytes already processed)      |<-Still to go-->|              */
/*      |<-------------------- INPUT BLOCK ------------------->|              */
/*                                                                            */
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
/*  * The first  byte in a copy item will be denoted C1.                      */
/*  * The second byte in a copy item will be denoted C2.                      */
/*  * Bits will be selected using square brackets.                            */
/*    For example: C1[0..3] is the low nibble of the first control byte.      */
/*    of copy item C1.                                                        */
/*  * The LENGTH of a copy item is defined to be C1[0..3]+3 which is a number */
/*    in the range [3,18].                                                    */
/*  * The INDEX of a copy item is defined to be C1[4..7]*256+C2[0..8] which   */
/*    is a number in the range [0,4095].                                      */
/*  * A copy item represents the sequence of bytes                            */
/*       text[POS-OFFSET..POS-OFFSET+LENGTH-1] where                          */
/*          text   is the entire text of the uncompressed string.             */
/*          POS    is the index in the text of the character following the    */
/*                   string represented by all the items preceeding the item  */
/*                   being defined.                                           */
/*          OFFSET is obtained from INDEX by looking up the hash table.       */
/*                                                                            */
/******************************************************************************/

/* When I first started to get concerned about the portability of my C code,  */
/* I switched over to using only macro defined types UBYTE, UWORD, ULONG and  */
/* one or two others. While, these are useful for most purposes, they impair  */
/* efficiency as, if I have a variable whose range will be [0,1000], I will   */
/* declare it as a UWORD. This will translate into (say) "short int" and      */
/* hence may be less efficient than just an "int" which represents the        */
/* natural size of the machine. Before releasing LZRW3-A, I realized this     */
/* mistake. Unfortunately, I can't access the ftp archive with my portability */
/* header in it in time for this algorithm's release and so I am including an */
/* extra definition. The definition UCARD stands for an unsigned (cardinal)   */
/* type that can hold values in the range [0,32767]. This is within the ANSI  */
/* range of a standard int or unsigned. No assumption about overflow of this  */
/* type is made in the code (i.e. all usages are within range and I do not    */
/* use the value -1 to detect the end of loops.).                             */
/* You can use either "unsigned" or just "int" here depending on which is     */
/* more efficient in your environment (both the same probably).               */
#define UCARD unsigned

/* The following #define defines the length of the copy flag that appears at  */
/* the start of the compressed file. The value of four bytes was chosen       */
/* because the fast_copy routine on my Macintosh runs faster if the source    */
/* and destination blocks are relatively longword aligned.                    */
/* The actual flag data appears in the first byte. The rest are zeroed so as  */
/* to normalize the compressed representation (i.e. not non-deterministic).   */
#define FLAG_BYTES 4

/* The following #defines define the meaning of the values of the copy        */
/* flag at the start of the compressed file.                                  */
#define FLAG_COMPRESS 0     /* Signals that output was result of compression. */
#define FLAG_COPY     1     /* Signals that output was simply copied over.    */

/* The 68000 microprocessor (on which this algorithm was originally developed */
/* is fussy about non-aligned arrays of words. To avoid these problems the    */
/* following macro can be used to "waste" from 0 to 3 bytes so as to align    */
/* the argument pointer.                                                      */
#define ULONG_ALIGN_UP(X) (X)

/* The following constant defines the maximum length of an uncompressed item. */
/* This definition must not be changed; its value is hardwired into the code. */
/* The longest number of bytes that can be spanned by a single item is 18     */
/* for the longest copy item.                                                 */
#define MAX_RAW_ITEM (18)

/* The following constant defines the maximum length of an uncompressed group.*/
/* This definition must not be changed; its value is hardwired into the code. */
/* A group contains at most 16 items which explains this definition.          */
#define MAX_RAW_GROUP (16*MAX_RAW_ITEM)

/* The following constant defines the maximum length of a compressed group.   */
/* This definition must not be changed; its value is hardwired into the code. */
/* A compressed group consists of two control bytes followed by up to 16      */
/* compressed items each of which can have a maximum length of two bytes.     */
#define MAX_CMP_GROUP (2+16*2)

/* This constant defines the number of pointers in the hash table. The number */
/* of partitions multiplied by the number of pointers in each partition must  */
/* multiply out to this value of 4096. In LZRW1, LZRW1-A, and LZRW2, this     */
/* table length value can be changed. However, in LZRW3-A (and LZRW3), the    */
/* table length cannot be changed because it is connected directly to the     */
/* coding scheme which is hardwired (the table index of a single pointer is   */
/* transmitted in the 12-bit index field). So don't change this constant!     */
#define HASH_TABLE_LENGTH (4096)

/* HERE IT IS: THE PLACE TO CHANGE THE HASH TABLE DEPTH!                      */
/* The following definition is the log_2 of the depth of the hash table. This */
/* constant can be in the range [0,1,2,3,...,12]. Increasing the depth        */
/* increases compression at the expense of speed. However, you are not likely */
/* to see much of a compression improvement (e.g. not more than 0.5%) above a */
/* value of 6 and the algorithm will start to get very slow. See the table in */
/* the earlier comments block for an idea of the trade-off involved.          */
/* Note: The parentheses are to avoid macro substitution funnies.             */
/* Note: The LZRW3-A default is a value of (3).                               */
/* Note: If you end up choosing a value of 0, you should use LZRW3 instead.   */
/* Note: Changing the value of HASH_TABLE_DEPTH_BITS is the ONLY thing you    */
/* have to do to change the depth, so go ahead and recompile now!             */
/* Note: I have tested LZRW3-A for DEPTH_BITS=0,1,2,3,4 and a few other       */
/* values. However, I have not tested it for 12 as I can't wait that long!    */
#define HASH_TABLE_DEPTH_BITS (3)      /* Must be in range [0,12].            */

/* The following definitions are all self-explanatory and follow from the     */
/* definition of HASH_TABLE_DEPTH_BITS and the hardwired requirement that the */
/* hash table contain exactly 4096 pointers.                                  */
#define PARTITION_LENGTH_BITS (12-HASH_TABLE_DEPTH_BITS)
#define PARTITION_LENGTH      (1<<PARTITION_LENGTH_BITS)
#define HASH_TABLE_DEPTH      (1<<HASH_TABLE_DEPTH_BITS )
#define HASH_MASK             (PARTITION_LENGTH-1)
#define DEPTH_MASK            (HASH_TABLE_DEPTH-1)

/* LZRW3-A, unlike LZRW1(-A), must initialize its hash table so as to enable  */
/* the compressor and decompressor to stay in step maintaining identical hash */
/* tables. In an early version of LZRW3, the tables were simply               */
/* initialized to zero and a check for zero was included just before the      */
/* matching code. However, this test costs time. A better solution is to      */
/* initialize all the entries in the hash table to point to a constant        */
/* string. The decompressor does the same. This solution requires no extra    */
/* test. The contents of the string do not matter so long as the string is    */
/* the same for the compressor and decompressor and contains at least         */
/* MAX_RAW_ITEM bytes. I chose consecutive decimal digits because they do not */
/* have white space problems (e.g. there is no chance that the compiler will  */
/* replace more than one space by a TAB) and because they make the length of  */
/* the string obvious by inspection.                                          */
#define START_STRING_18 ((UBYTE *) "123456789012345678")

/* The following macro accepts a pointer PTR to three consecutive bytes in    */
/* memory and hashes them into an integer that is a hash table index that     */
/* points to the zeroth (first) element of a partition. Thus, the hash        */
/* function really hashes to a partition number but, for convenience,         */
/* multiplies it up to yield a hash table index. From all this, we see that   */
/* the resultant number is in the range [0,HASH_TABLE_LENGTH-1] and is a      */
/* multiple of HASH_TABLE_DEPTH.                                              */
/* A macro is used, because in LZRW3-A we have to hash more than once.        */
#define HASH(PTR) \
 ( \
     (((40543*(((*(PTR))<<8)^((*((PTR)+1))<<4)^(*((PTR)+2))))>>4) & HASH_MASK) \
  << HASH_TABLE_DEPTH_BITS \
 )

/* Another operation that is performed more than once is the updating of the  */
/* hash table. Here two macros are defined to simplify update operations.     */
/* Updating consists of identifying and overwriting a pointer in a partition  */
/* with a newer pointer and then updating the global cycle value.             */
/* These macros accept the new pointer (NEWPTR) and either a pointer to       */
/* (P_BASE) or the index of (I_BASE) the zeroth (first, or base) pointer in   */
/* the partition that is to be updated. The macros use the 'cycle' variable   */
/* to locate and overwrite a pointer and then update the cycle value.         */
/* Note: Hardcoding 'cycle' in this macro is naughty (it should really be a   */
/* macro parameter), but I have done so because it neatens up the code.       */
#define UPDATE_P(P_BASE,NEWPTR) \
{(P_BASE)[cycle++]=(NEWPTR); cycle&=DEPTH_MASK;}

#define UPDATE_I(I_BASE,NEWPTR) \
{hash[(I_BASE)+cycle++]=(NEWPTR); cycle&=DEPTH_MASK;}

/* This constant supplies a legal (in-range) hash table index for use when    */
/* a legal-but-don't-care index is required.                                  */
#define ANY_HASH_INDEX (0)

/******************************************************************************/

LOCAL void lzrw3a_compress_compress
	(UBYTE *p_wrk_mem,UBYTE *p_src_first,ULONG src_len,UBYTE *p_dst_first,ULONG* p_dst_len)
/* Input  : Hand over the required amount of working memory in p_wrk_mem.     */
/* Input  : Specify input block using p_src_first and src_len.                */
/* Input  : Point p_dst_first to the start of the output zone (OZ).           */
/* Input  : Point p_dst_len to a ULONG to receive the output length.          */
/* Input  : Input block and output zone must not overlap.                     */
/* Output : Length of output block written to *p_dst_len.                     */
/* Output : Output block in Mem[p_dst_first..p_dst_first+*p_dst_len-1]. May   */
/* Output : write in OZ=Mem[p_dst_first..p_dst_first+src_len+MAX_CMP_GROUP-1].*/
/* Output : Upon completion guaranteed *p_dst_len<=src_len+FLAG_BYTES.        */
{
 /* p_src and p_dst step through the source and destination blocks.           */
 UBYTE *p_src = p_src_first;
 UBYTE *p_dst = p_dst_first;

 /* The following variables are never modified and are used in the            */
 /* calculations that determine when the main loop terminates.                */
 UBYTE *p_src_post  = p_src_first+src_len;
 UBYTE *p_dst_post  = p_dst_first+src_len;
 UBYTE *p_src_max1  = p_src_first+src_len-MAX_RAW_ITEM;
 UBYTE *p_src_max16 = p_src_first+src_len-MAX_RAW_ITEM*16;

 /* The variables 'p_control' and 'control' are used to buffer control bits.  */
 /* Before each group is processed, the next two bytes of the output block    */
 /* are set aside for the control word for the group about to be processed.   */
 /* 'p_control' is set to point to the first byte of that word. Meanwhile,    */
 /* 'control' buffers the control bits being generated during the processing  */
 /* of the group. Instead of having a counter to keep track of how many items */
 /* have been processed (=the number of bits in the control word), at the     */
 /* start of each group, the top word of 'control' is filled with 1 bits.     */
 /* As 'control' is shifted for each item, the 1 bits in the top word are     */
 /* absorbed or destroyed. When they all run out (i.e. when the top word is   */
 /* all zero bits, we know that we are at the end of a group.                 */
 #define TOPWORD 0xFFFF0000
 UBYTE *p_control;
 ULONG control=TOPWORD;

 /* The variable 'hash' always points to the first element of the hash table. */
 UBYTE **hash= (UBYTE **) ULONG_ALIGN_UP(p_wrk_mem);

 /* The following two variables represent the literal buffer. p_h1 points to  */
 /* the partition (i.e. the zero'th (first) element of the partition)         */
 /* corresponding to the youngest literal. p_h2 points to the partition       */
 /* corresponding to the second youngest literal.                             */
 /* The value zero denotes an "empty" buffer value with p_h1=0 => p_h2=0.     */
 UBYTE **p_h1=0;
 UBYTE **p_h2=0;

 /* The following variable holds the current 'cycle' value. This value cycles */
 /* through the range [0,HASH_TABLE_DEPTH-1], being incremented every time    */
 /* the hash table is updated. The value gives the within-partition number of */
 /* the next pointer to be overwritten. The decompressor maintains a cycle    */
 /* value in synchrony.                                                       */
 UCARD cycle=0;

 /* To start, we write the flag bytes. Being optimistic, we set the flag to   */
 /* FLAG_COMPRESS. The remaining flag bytes are zeroed so as to keep the      */
 /* algorithm deterministic.                                                  */
 *p_dst++=FLAG_COMPRESS;
 {UCARD i; for (i=2;i<=FLAG_BYTES;i++) *p_dst++=0;}

 /* Reserve the first word of output as the control word for the first group. */
 /* Note: This is undone at the end if the input block is empty.              */
 p_control=p_dst; p_dst+=2;

 /* Initialize all elements of the hash table to point to a constant string.  */
 /* Use of an unrolled loop speeds this up considerably.                      */
 /* These variables should really be declared "register", but I am worried    */
 /* about the possibility that extra register declarations will tempt stupid  */
 /* compilers to allocate all registers before they get to the innermostloop. */
 {UCARD i; UBYTE **p_h=hash;
  #define ZH *p_h++=START_STRING_18
  for (i=0;i<256;i++)     /* 256=HASH_TABLE_LENGTH/16. */
    {ZH;ZH;ZH;ZH;
     ZH;ZH;ZH;ZH;
     ZH;ZH;ZH;ZH;
     ZH;ZH;ZH;ZH;}
 }

 /* The main loop processes either 1 or 16 items per iteration. As its        */
 /* termination logic is complicated, I have opted for an infinite loop       */
 /* structure containing 'break' and 'goto' statements.                       */
 while (TRUE)
   {/* Begin main processing loop. */

    /* Note: All the variables here except unroll should be defined within    */
    /*       the inner loop. Unfortunately the loop hasn't got a block.       */
     UBYTE *p_ziv;              /* Points to first byte of current Ziv.       */
     UCARD unroll;              /* Loop counter for unrolled inner loop.      */
     UCARD index;               /* Index of current partition.                */
     UBYTE **p_h0;              /* Pointer to current partition.              */
     register UCARD d;          /* Depth looping variable.                    */
     register UCARD bestlen;    /* Holds the best length seen so far.         */
     register UCARD bestpos;    /* Holds number of best pointer seen so far.  */

    /* Test for overrun and jump to overrun code if necessary.                */
    if (p_dst>p_dst_post)
       goto overrun;

    /* The following cascade of if statements efficiently catches and deals   */
    /* with varying degrees of closeness to the end of the input block.       */
    /* When we get very close to the end, we stop updating the table and      */
    /* code the remaining bytes as literals. This makes the code simpler.     */
    unroll=16;
    if (p_src>p_src_max16)
      {
       unroll=1;
       if (p_src>p_src_max1)
         {
          if (p_src==p_src_post)
             break;
          else
             {p_h0=&hash[ANY_HASH_INDEX]; /* Avoid undefined pointer. */
              goto literal;}
         }
      }

    /* This inner unrolled loop processes 'unroll' (whose value is either 1   */
    /* or 16) items. I have chosen to implement this loop with labels and     */
    /* gotos to heighten the ease with which the loop may be implemented with */
    /* a single decrement and branch instruction in assembly language and     */
    /* also because the labels act as highly readable place markers.          */
    /* (Also because we jump into the loop for endgame literals (see above)). */

    begin_unrolled_loop:

       p_ziv=p_src;

       /* To process the next phrase, we hash the next three bytes to obtain  */
       /* an index to the zeroth (first) pointer in a target partition. We    */
       /* get the pointer.                                                    */
       index=HASH(p_src);
       p_h0=&hash[index];

       /* This next part runs through the pointers in the partition matching  */
       /* the bytes they point to in the Lempel with the bytes in the Ziv.    */
       /* The length (bestlen) and within-partition pointer number (bestpos)  */
       /* of the longest match so far is maintained and is the output of this */
       /* segment of code. The s[bestlen]==... is an optimization only.       */
       bestlen=0;
       bestpos=0;
       for (d=0;d<HASH_TABLE_DEPTH;d++)
         {
          register UBYTE *s=p_src;
          register UBYTE *p=p_h0[d];
          register UCARD len;
          if (s[bestlen] == p[bestlen])
            {
             #define PS *p++!=*s++
             PS || PS || PS || PS || PS || PS || PS || PS || PS ||
             PS || PS || PS || PS || PS || PS || PS || PS || PS || s++;
             len=s-p_src-1;
             if (len>bestlen)
               {
                bestpos=d;
                bestlen=len;
               }
            }
         }

       /* The length of the longest match determines whether we code a */
       /* literal item or a copy item.                                 */

       if (bestlen<3)
         {
          /* Literal. */

          /* Code the literal byte as itself and a zero control bit.          */
          literal: *p_dst++=*p_src++; control&=0xFFFEFFFF;

          /* We have just coded a literal. If we had two pending ones, that   */
          /* makes three and we can update the hash table.                    */
          if (p_h2!=0)
             {UPDATE_P(p_h2,p_ziv-2);}

          /* In any case, rotate the hash table pointers for next time. */
          p_h2=p_h1; p_h1=p_h0;

         }
       else
         {
          /* Copy */

          /* To code a copy item, we construct a hash table index of the      */
          /* winning pointer (index+=bestpos) and code it and the best length */
          /* into a 2 byte code word. Bump up p_src.                          */
          index+=bestpos;
          *p_dst++=((index&0xF00)>>4)|(bestlen-3);
          *p_dst++=index&0xFF;
          p_src+=bestlen;

          /* As we have just coded three bytes, we are now in a position to   */
          /* update the hash table with the literal bytes that were pending   */
          /* upon the arrival of extra context bytes.                         */
          if (p_h1!=0)
            {
             if (p_h2!=0)
               {UPDATE_P(p_h2,p_ziv-2); p_h2=0;}
             UPDATE_P(p_h1,p_ziv-1); p_h1=0;
            }

          /* In any case, we can update the hash table based on the current   */
          /* position as we just coded at least three bytes in a copy items.  */
          UPDATE_P(p_h0,p_ziv);
         }
       control>>=1;

       /* This loop is all set up for a decrement and jump instruction! */
    end_unrolled_loop: if (--unroll) goto begin_unrolled_loop;

    /* At this point it will nearly always be the end of a group in which     */
    /* case, we have to do some control-word processing. However, near the    */
    /* end of the input block, the inner unrolled loop is only executed once. */
    /* This necessitates the 'if' test.                                       */
    if ((control&TOPWORD)==0)
      {
       /* Write the control word to the place we saved for it in the output. */
       *p_control++=  control     &0xFF;
       *p_control  = (control>>8) &0xFF;

       /* Reserve the next word in the output block for the control word */
       /* for the group about to be processed.                           */
       p_control=p_dst; p_dst+=2;

       /* Reset the control bits buffer. */
       control=TOPWORD;
      }

   } /* End main processing loop. */

 /* After the main processing loop has executed, all the input bytes have     */
 /* been processed. However, the control word has still to be written to the  */
 /* word reserved for it in the output at the start of the most recent group. */
 /* Before writing, the control word has to be shifted so that all the bits   */
 /* are in the right place. The "empty" bit positions are filled with 1s      */
 /* which partially fill the top word.                                        */
 while(control&TOPWORD) control>>=1;
 *p_control++= control     &0xFF;
 *p_control++=(control>>8) &0xFF;

 /* If the last group contained no items, delete the control word too.        */
 if (p_control==p_dst) p_dst-=2;

 /* Write the length of the output block to the dst_len parameter and return. */
 *p_dst_len=p_dst-p_dst_first;
 return;

 /* Jump here as soon as an overrun is detected. An overrun is defined to     */
 /* have occurred if p_dst>p_dst_first+src_len. That is, the moment the       */
 /* length of the output written so far exceeds the length of the input block.*/
 /* The algorithm checks for overruns at least at the end of each group       */
 /* which means that the maximum overrun is MAX_CMP_GROUP bytes.              */
 /* Once an overrun occurs, the only thing to do is to set the copy flag and  */
 /* copy the input over.                                                      */
 overrun:
 *p_dst_first=FLAG_COPY;
 fast_copy(p_src_first,p_dst_first+FLAG_BYTES,src_len);
 *p_dst_len=src_len+FLAG_BYTES;
}

/******************************************************************************/

LOCAL void lzrw3a_compress_decompress
	(UBYTE *p_wrk_mem,UBYTE *p_src_first,ULONG src_len,UBYTE *p_dst_first,ULONG* p_dst_len)
/* Input  : Hand over the required amount of working memory in p_wrk_mem.     */
/* Input  : Specify input block using p_src_first and src_len.                */
/* Input  : Point p_dst_first to the start of the output zone.                */
/* Input  : Point p_dst_len to a ULONG to receive the output length.          */
/* Input  : Input block and output zone must not overlap. User knows          */
/* Input  : upperbound on output block length from earlier compression.       */
/* Input  : In any case, maximum expansion possible is nine times.            */
/* Output : Length of output block written to *p_dst_len.                     */
/* Output : Output block in Mem[p_dst_first..p_dst_first+*p_dst_len-1].       */
/* Output : Writes only  in Mem[p_dst_first..p_dst_first+*p_dst_len-1].       */
{
 /* Byte pointers p_src and p_dst scan through the input and output blocks.   */
 register UBYTE *p_src = p_src_first+FLAG_BYTES;
 register UBYTE *p_dst = p_dst_first;

 /* The following two variables are never modified and are used to control    */
 /* the main loop.                                                            */
 UBYTE *p_src_post  = p_src_first+src_len;
 UBYTE *p_src_max16 = p_src_first+src_len-(MAX_CMP_GROUP-2);

 /* The hash table is the only resident of the working memory. The hash table */
 /* contains HASH_TABLE_LENGTH=4096 pointers to positions in the history. To  */
 /* keep Macintoshes happy, it is longword aligned.                           */
 UBYTE **hash = (UBYTE **) ULONG_ALIGN_UP(p_wrk_mem);

 /* The variable 'control' is used to buffer the control bits which appear in */
 /* groups of 16 bits (control words) at the start of each compressed group.  */
 /* When each group is read, bit 16 of the register is set to one. Whenever   */
 /* a new bit is needed, the register is shifted right. When the value of the */
 /* register becomes 1, we know that we have reached the end of a group.      */
 /* Initializing the register to 1 thus instructs the code to follow that it  */
 /* should read a new control word immediately.                               */
 register ULONG control=1;

 /* The value of 'literals' is always in the range 0..3. It is the number of  */
 /* consecutive literal items just seen. We have to record this number so as  */
 /* to know when to update the hash table. When literals gets to 3, there     */
 /* have been three consecutive literals and we can update at the position of */
 /* the oldest of the three.                                                  */
 register UCARD literals=0;

 /* The following variable holds the current 'cycle' value. This value cycles */
 /* through the range [0,HASH_TABLE_DEPTH-1], being incremented every time    */
 /* the hash table is updated. The value give the within-partition number of  */
 /* the next pointer to be overwritten. The compressor maintains a cycle      */
 /* value in synchrony.                                                       */
 UCARD cycle=0;

 /* Check the leading copy flag to see if the compressor chose to use a copy  */
 /* operation instead of a compression operation. If a copy operation was     */
 /* used, then all we need to do is copy the data over, set the output length */
 /* and return.                                                               */
 if (*p_src_first==FLAG_COPY)
   {
    fast_copy(p_src_first+FLAG_BYTES,p_dst_first,src_len-FLAG_BYTES);
    *p_dst_len=src_len-FLAG_BYTES;
    return;
   }

 /* Initialize all elements of the hash table to point to a constant string.  */
 /* Use of an unrolled loop speeds this up considerably.                      */
 /* The comment about register declarations above similar code in the         */
 /* compressor applies here too.                                              */
 {UCARD i; UBYTE **p_h=hash;
  #define ZJ *p_h++=START_STRING_18
  for (i=0;i<256;i++)     /* 256=HASH_TABLE_LENGTH/16. */
    {ZJ;ZJ;ZJ;ZJ;
     ZJ;ZJ;ZJ;ZJ;
     ZJ;ZJ;ZJ;ZJ;
     ZJ;ZJ;ZJ;ZJ;}
 }

 /* The outer loop processes either 1 or 16 items per iteration depending on  */
 /* how close p_src is to the end of the input block.                         */
 while (p_src!=p_src_post)
   {/* Start of outer loop */

    register UCARD unroll;   /* Counts unrolled loop executions.              */

    /* When 'control' has the value 1, it means that the 16 buffered control  */
    /* bits that were read in at the start of the current group have all been */
    /* shifted out and that all that is left is the 1 bit that was injected   */
    /* into bit 16 at the start of the current group. When we reach the end   */
    /* of a group, we have to load a new control word and inject a new 1 bit. */
    if (control==1)
      {
       control=0x10000|*p_src++;
       control|=(*p_src++)<<8;
      }

    /* If it is possible that we are within 16 groups from the end of the     */
    /* input, execute the unrolled loop only once, else process a whole group */
    /* of 16 items by looping 16 times.                                       */
    unroll= p_src<=p_src_max16 ? 16 : 1;

    /* This inner loop processes one phrase (item) per iteration. */
    while (unroll--)
      { /* Begin unrolled inner loop. */

       /* Process a literal or copy item depending on the next control bit. */
       if (control&1)
         {
          /* Copy item. */

          register UBYTE *p;           /* Points to place from which to copy. */
          register UCARD lenmt;        /* Length of copy item minus three.    */
          register UBYTE *p_ziv=p_dst; /* Pointer to start of current Ziv.    */
          register UCARD index;        /* Index of hash table copy pointer.   */

          /* Read and dismantle the copy word. Work out from where to copy.   */
          lenmt=*p_src++;
          index=((lenmt&0xF0)<<4)|*p_src++;
          p=hash[index];
          lenmt&=0xF;

          /* Now perform the copy using a half unrolled loop. */
          *p_dst++=*p++;
          *p_dst++=*p++;
          *p_dst++=*p++;
          while (lenmt--)
             *p_dst++=*p++;

          /* Because we have just received 3 or more bytes in a copy item     */
          /* (whose bytes we have just installed in the output), we are now   */
          /* in a position to flush all the pending literal hashings that had */
          /* been postponed for lack of bytes.                                */
          if (literals>0)
            {
             register UBYTE *r=p_ziv-literals;;
             UPDATE_I(HASH(r),r);
             if (literals==2)
                {r++; UPDATE_I(HASH(r),r);}
             literals=0;
            }

          /* In any case, we can immediately update the hash table with the   */
          /* current position. We don't need to do a HASH(...) to work out    */
          /* where to put the pointer, as the compressor just told us!!!      */
          UPDATE_I(index&(~DEPTH_MASK),p_ziv);
         }
       else
         {
          /* Literal item. */

          /* Copy over the literal byte. */
          *p_dst++=*p_src++;

          /* If we now have three literals waiting to be hashed into the hash */
          /* table, we can do one of them now (because there are three).      */
          if (++literals == 3)
             {register UBYTE *p=p_dst-3;
              UPDATE_I(HASH(p),p); literals=2;}
         }

       /* Shift the control buffer so the next control bit is in bit 0. */
       control>>=1;

      } /* End unrolled inner loop. */

   } /* End of outer loop */

 /* Write the length of the decompressed data before returning. */
 *p_dst_len=p_dst-p_dst_first;
}

/******************************************************************************/
/*                              End of LZRW3-A.C                              */
/******************************************************************************/


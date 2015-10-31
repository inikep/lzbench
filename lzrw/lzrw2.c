/******************************************************************************/
/*                                                                            */
/*                                    LZRW2.C                                 */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/* Author  : Ross Williams.                                                   */
/* Date    : 29-Jun-1991.                                                     */
/* Release : 1.                                                               */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/* This file contains an implementation of the LZRW2 data compression         */
/* algorithm in C.                                                            */
/*                                                                            */
/* The algorithm is a general purpose compression algorithm that runs fast    */
/* and gives reasonable compression. The algorithm is a member of the Lempel  */
/* Ziv family of algorithms and bases its compression on the presence in the  */
/* data of repeated substrings.                                               */
/*                                                                            */
/* This algorithm is unpatented and the code is public domain. As the         */
/* algorithm is based on the LZ77 class of algorithms, it is unlikely to be   */
/* the subject of a patent challenge.                                         */
/*                                                                            */
/* Unlike the LZRW1 and LZRW1-A algorithms, the LZRW2 algorithm is            */
/* deterministic and is guaranteed to yield the same compressed               */
/* representation for a given file each time it is run.                       */
/*                                                                            */
/* The LZRW2 algorithm was originally designed and implemented                */
/* by Ross Williams on 25-Nov-1990.                                           */

                            /* INCLUDE FILES                                  */
                            /* =============                                  */
#include "lzrw.h"      /* Fast memory copy routine.                      */

/******************************************************************************/

/* The following structure is returned by the "compress" function below when  */
/* the user asks the function to return identifying information.              */
/* The most important field in the record is the working memory field which   */
/* tells the calling program how much working memory should be passed to      */
/* "compress" when it is called to perform a compression or decompression.    */
/* For the LZRW2 algorithm, the decompressor requires less memory than the    */
/* decompressor (so I have defined two constants) but the interface standard  */
/* I am using only allows a single memory size for both compression and       */
/* decompression so I put in the larger: CMP_MEM_REQ.                         */
/* Note: CMP_MEM_REQ~=24K, DEC_MEM_REQ~=16K.                                  */
/* For more information on this structure see "compress.h".                   */
  
#define U(X)            ((uint32_t) X)
#define SIZE_P_BYTE     (U(sizeof(UBYTE *)))
#define SIZE_WORD       (U(sizeof(UWORD  )))
#define ALIGNMENT_FUDGE (U(100))
#define CMP_MEM_REQ ( U(4096)*(SIZE_P_BYTE+SIZE_WORD) + ALIGNMENT_FUDGE )
#define DEC_MEM_REQ ( U(4096)*(SIZE_P_BYTE          ) + ALIGNMENT_FUDGE )

static struct compress_identity identity =
{
 U(0x3D8F733A),                           /* Algorithm identification number. */
 CMP_MEM_REQ,                             /* Working memory (bytes) required. */
 "LZRW2",                                 /* Name of algorithm.               */
 "1.0",                                   /* Version number of algorithm.     */
 "25-Nov-1990",                           /* Date of algorithm.               */
 "Public Domain",                         /* Copyright notice.                */
 "Ross N. Williams",                      /* Author of algorithm.             */
 "Renaissance Software",                  /* Affiliation of author.           */
 "Public Domain"                          /* Vendor of algorithm.             */
};

uint32_t lzrw2_req_mem() { return CMP_MEM_REQ; };

void lzrw2_compress_compress  (UBYTE *,UBYTE *,uint32_t,UBYTE *,uint32_t *);
void lzrw2_compress_decompress(UBYTE *,UBYTE *,uint32_t,UBYTE *,uint32_t *);

/******************************************************************************/

/* This function is the only function exported by this module.                */
/* Depending on its first parameter, the function can be requested to         */
/* compress a block of memory, decompress a block of memory, or to identify   */
/* itself. For more information, see the specification file "compress.h".     */
/*
UWORD     action;      // Action to be performed.                             
UBYTE   *wrk_mem;      // Address of working memory we can use.               
UBYTE   *src_adr;      // Address of input data.                              
uint32_t    src_len;   // Length  of input data.                              
UBYTE   *dst_adr;      // Address to put output data.                         
uint32_t *p_dst_len;   // Address of longword for length of output data.      
*/
EXPORT void lzrw2_compress(UWORD action,UBYTE* wrk_mem,UBYTE* src_adr,uint32_t src_len,UBYTE* dst_adr,uint32_t* p_dst_len)
{
 switch (action)
   {
    case COMPRESS_ACTION_COMPRESS:
       lzrw2_compress_compress(wrk_mem,src_adr,src_len,dst_adr,p_dst_len);
       break;
    case COMPRESS_ACTION_DECOMPRESS:
       lzrw2_compress_decompress(wrk_mem,src_adr,src_len,dst_adr,p_dst_len);
       break;
   }
}

/******************************************************************************/
/*                                                                            */
/* BRIEF DESCRIPTION OF THE LZRW2 ALGORITHM                                   */
/* ========================================                                   */
/* The LZRW2 algorithm is identical to the LZRW1-A algorithm except that it   */
/* employs a PHRASE TABLE. The phrase table contains pointers to the first    */
/* byte of the most recent 4096 phrases processed. A phrase is defined to be  */
/* a sequence of one or more bytes that are coded as a single literal or copy */
/* item. Instead of coding a copy item as a length and an offset as LZRW1     */
/* does, LZRW2 codes a copy item as a length and a phrase table index. The    */
/* result is that LZRW2 can point far further back than LZRW1 but without     */
/* increasing the number of bits allocated to the 'offset' in the coding. The */
/* phrase table is incapable of pointing within phrases, but LZRW1 was        */
/* incapabale of doing that anyway because it only ever updated its hash      */
/* table on phrase boundaries.                                                */
/*                                                                            */
/* Updating the phrase table involves just writing a pointer to the next      */
/* position (which circulates) in the phrase table after each literal or      */
/* copy item is coded. The decompressor needs to maintain a phrase table but  */
/* not a hash table.                                                          */
/*                                                                            */
/* Use of the phrase table yields a 3% absolute to 5% absolute improvement    */
/* over LZRW1-A in compression, does not affect compression speed, but        */
/* reduces decompression speed by about 30%. Thus LZRW2 does not dominate     */
/* LZRW1-A, as LZRW1-A does LZRW1.                                            */
/*                                                                            */
/* An extra 3% absolute compression can be obtained by using a hash table of  */
/* depth two. However, increasing the depth above one incurs a 40% decrease   */
/* in compression speed which was not considered worthwhile. Another reason   */
/* for keeping the depth down to one was to allow easy comparison with the    */
/* LZRW1-A algorithm so as to demonstrate the exact effect of the phrase      */
/* table.                                                                     */
/*                                                                            */
/*                +---+             +---+                                     */
/*                |___|4095         |___|4095                                 */
/*                |___|             |___|                                     */
/*         Next-->|___|       +-------*_|<---+   /----+---\                   */
/*     (circles   |___|       |     |___|    +---|Hash    |                   */
/*      through +---*_|<------+     |___|        |Function|                   */
/*      phrase  | |___|             |___|        \--------/                   */
/*      table)  | |___|0            |___|0            ^                       */
/*              | +---+             +---+             |                       */
/*              | Phrase            Hash        +-----+                       */
/*              | Table             Table       |                             */
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
/*  LZRW2 Algorithm Execution Summary                                         */
/*  ---------------------------------                                         */
/*  1. Hash the first three bytes of the Ziv to yield a hash table index h.   */
/*  2. Look up the hash table yielding phrase table index i.                  */
/*  3. Look up phrase i yielding a pointer p to the Lempel.                   */
/*  4. Overwrite the 'next' cyclic entry in the phrase table with a pointer   */
/*     to the Ziv. Increment the 'next' index (mod 4096).                     */
/*  5. Overwrite hash table entry h with the index of the overwritten         */
/*     position of step 4.                                                    */
/*  6. Match where p points with the Ziv. If there is a match of three or     */
/*     more bytes, code those bytes (in the Ziv) as a copy item, otherwise    */
/*     code the next byte in the Ziv as a literal item.                       */
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
/*          OFFSET is obtained from INDEX by looking up the phrase table.     */
/*                                                                            */
/******************************************************************************/

/* Stylistic note: The LZRW1 algorithm was written in an extremely terse      */
/* style because it had to fit on a single page in a paper. This style        */
/* carried over to the LZRW1-A algorithm.  However, LZRW2 has not been        */
/* published and so I have reverted to my normal programming style.           */

/* Stylistic note: This code could be considerably neated by the use of       */
/* dependent declarations such as {int a=3,b=a+1;}. However I can't locate a  */
/* clause in K&R that guarantees that declarations are evaluated in order.    */
   
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
#define ULONG_ALIGN_UP(X) (X) //((((uint32_t)X)+3)&~3)

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

/* The following constant defines the number of entries in the phrase table.  */
/* This definition must not be changed; its value is hardwired into the code. */
#define PHRASE_TABLE_LENGTH (4096)

/* The following constant defines the number of entries in the hash table.    */
/* This definition must not be changed; its value is hardwired into the code. */
#define HASH_TABLE_LENGTH (4096)

/* At initialization, the hash table entries are all set to point to element  */
/* zero of the phrase table. In order for the algorithm to start up,          */
/* phrase[0] needs to point to a well defined string of at least 18 bytes. At */
/* startup, there is no already-transmitted source text to point to and so    */
/* we have to invent some dummy text to point to. It doesn't matter what is   */
/* in this string so long as it is at least MAX_RAW_ITEM bytes long and is    */
/* the same in the compressor and decompressor. I chose consecutive decimal   */
/* digits because they do not have white space problems (e.g. there is no     */
/* chance that the compiler will replace more than one space by a TAB) and    */
/* because they make the length of the string obvious by inspection.          */
#define START_STRING_18 "123456789012345678"

/******************************************************************************/
                            
LOCAL void lzrw2_compress_compress(UBYTE *p_wrk_mem,UBYTE *p_src_first,uint32_t src_len,UBYTE *p_dst_first,uint32_t *p_dst_len)
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
 register UBYTE *p_src = p_src_first;
 register UBYTE *p_dst = p_dst_first;
 
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
 uint32_t  control=TOPWORD;
 
 UWORD  *hash;           /* Points to the first element of the hash table.    */
 UBYTE **phrase;         /* Points to the first element of the phrase table.  */
 register UWORD next;    /* Index of next phrase entry to be overwritten.     */
 
 /* Set up the hash table and the phrase table in the memory given to         */
 /* the algorithm. These tables are the only occupants of the memory.         */
 hash          = (UWORD *)  ULONG_ALIGN_UP(p_wrk_mem); 
 phrase        = (UBYTE **) ULONG_ALIGN_UP(&hash[HASH_TABLE_LENGTH]);

 /* The variable 'next' cycles through the phrase table indicating the next   */
 /* position in the table to write a new phrase pointer.                      */
 next=0; 
 
 /* To start, we write the flag bytes. Being optimistic, we set the flag to   */
 /* FLAG_COMPRESS. The remaining flag bytes are zeroed so as to keep the      */
 /* algorithm deterministic.                                                  */
 *p_dst++=FLAG_COMPRESS;
 {UWORD i; for (i=2;i<=FLAG_BYTES;i++) *p_dst++=0;}

 /* Reserve the first word of output as the control word for the first group. */
 /* Note: This is undone at the end if the input block is empty.              */
 p_control=p_dst; p_dst+=2;
 
 /* Initialize the hash table and the phrase table. The hash table entries    */
 /* all point to element zero of the phrase table which in turn points to     */
 /* a constant string. The remainder of the phrase table does not need to     */
 /* be initialized as the algorithm is arranged so that no element of the     */
 /* phrase table is ever read before it has been written.                     */
 /* In theory, either the hash table or the phrase table has to be            */
 /* initialized. I chose the hash table as this choice yields determinism.    */
 {register UWORD i,*t=hash;
  #define ZH *t++=0; *t++=0; *t++=0; *t++=0 
  for (i=0;i<256;i++) {ZH;ZH;ZH;ZH;}  /* 256=HASH_TABLE_LENGTH/16 */
 }
 phrase[0]=(UBYTE *) START_STRING_18;
 
 /* The main loop processes either 1 or 16 items per iteration. As its        */
 /* termination logic is complicated, I have opted for an infinite loop       */
 /* structure containing 'break' and 'goto' statements.                       */
 while (TRUE)
   {/* Begin main processing loop. */
   
     register UBYTE *p;         /* Scans through targ phrase during matching. */
     register UBYTE *p_ziv;     /* Points to first byte of current Ziv.       */
     register UWORD phrase_num; /* Current phrase number (index in phr tab).  */
     register UWORD unroll;     /* Loop counter for unrolled inner loop.      */
     register UWORD *p_hte;     /* Pointer to the current hash table entry.   */

    /* Test for overrun and jump to overrun code if necessary. */
    if (p_dst>p_dst_post)
       goto overrun;
       
    /* The following cascade of if statements efficiently catches and deals   */
    /* with varying degrees of closeness to the end of the input block.       */
    /* When we get very close to the end, we stop updating the tables and     */
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
             goto literal;
         }
      }
         
    /* This inner unrolled loop processes 'unroll' (whose value is either 1   */
    /* or 16) items. I have chosen to implement this loop with labels and     */
    /* gotos to heighten the ease with which the loop may be implemented with */
    /* a single decrement and branch instruction in assembly language and     */
    /* also because the labels act as highly readable place markers.          */
    /* (Also because we jump into the loop for endgame literals (see above)). */
    
    begin_unrolled_loop:
    
       /* To process the next phrase, we hash the next three bytes and use    */
       /* the resultant hash table index to look up the hash table. A pointer */
       /* to the entry is stored in p_he. We store a pointer instead of an    */
       /* index so as to avoid array lookups. The hash table entry contains a */
       /* phrase number 'phrase_num' which we use to look up the phrase table */
       /* and get a pointer 'p' to a potential match in the history.          */
       p_hte=&hash[((40543*((p_src[0]<<8)^(p_src[1]<<4)^p_src[2]))>>4) & 0xFFF];
       phrase_num=*p_hte;
       p=phrase[phrase_num];
       
       /* Update the hash table and phrase table. The next element of the     */
       /* phrase table points to the first byte of the current Ziv and the    */
       /* hash table entry we looked up gets overwritten with the phrase      */
       /* table entry number. Wraparound of 'next' is done elsewhere.         */
       *p_hte=next;
       phrase[next++]=p_src;

       /* Having looked up the candidate position, we are in a position to    */
       /* attempt a match. The match loop has been unrolled using the PS      */
       /* macro so that failure within the first three bytes automatically    */
       /* results in the literal branch being taken. The coding is simple.    */
       /* p_ziv saves p_src so we can let p_src wander.                       */
       #define PS *p++!=*p_src++
       p_ziv=p_src;
       if (PS || PS || PS)
         {p_src=p_ziv; literal:*p_dst++=*p_src++; control&=0xFFFEFFFF;}
       else
         {
          PS || PS || PS || PS || PS || PS || PS || PS ||
          PS || PS || PS || PS || PS || PS || PS || p_src++;
          *p_dst++=((phrase_num&0xF00)>>4)|(--p_src-p_ziv-3);
          *p_dst++=phrase_num&0xFF;
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
       
       /* The following statement makes sure that the 'next' phrase table     */
       /* index cycles when it comes to the end of the phrase table. This     */
       /* can be included here within the control word processing because the */
       /* control word processing happens once every 16 items processed and   */
       /* 4096 (the num of entries in the phrase table) is a multiple of 16.  */
       next&=0xFFF;
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

LOCAL void lzrw2_compress_decompress(UBYTE *p_wrk_mem,UBYTE *p_src_first,uint32_t src_len,UBYTE *p_dst_first,uint32_t *p_dst_len)
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
 
 /* The variable 'control' is used to buffer the control bits which appear in */
 /* groups of 16 bits (control words) at the start of each compressed group.  */
 /* When each group is read, bit 16 of the register is set to one. Whenever   */
 /* a new bit is needed, the register is shifted right. When the value of the */
 /* register becomes 1, we know that we have reached the end of a group.      */
 /* Initializing the register to 1 thus instructs the code to follow that it  */
 /* should read a new control word immediately.                               */
 register uint32_t control=1;
 
 /* The phrase table is the same as in the compressor. The decompressor does  */
 /* not need to maintain a hash table, only a phrase table.                   */
 /* The phrase table is the only occupant of the working memory.              */
 UBYTE **phrase = (UBYTE **) ULONG_ALIGN_UP(p_wrk_mem);
 
 /* The next variable cycles through the phrase table always containing the   */
 /* index of the next phrase pointer to be overwritten in the phrase table.   */
 /* Optimization note: I tried using a pointer to cycle through the table but */
 /* this went more slowly than using an explicit index.                       */
 register UWORD next=0;
      
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
   
 /* Whereas the compressor needs to maintain a hash table and a phrase table  */
 /* the decompressor needs to maintain only the phrase table. Only the first  */
 /* entry of the phrase table needs initializing as, apart from this entry,   */
 /* the compressor guarantees not to refer to a table entry until the entry   */
 /* has been written.                                                         */
 phrase[0]=(UBYTE *) START_STRING_18;
    
 /* The outer loop processes either 1 or 16 items per iteration depending on  */
 /* how close p_src is to the end of the input block.                         */
 while (p_src!=p_src_post)
   {/* Start of outer loop */
   
    register UWORD unroll;   /* Counts unrolled loop executions.              */
    
    /* When 'control' has the value 1, it means that the 16 buffered control  */
    /* bits that were read in at the start of the current group have all been */
    /* shifted out and that all that is left is the 1 bit that was injected   */
    /* into bit 16 at the start of the current group. When we reach the end   */
    /* of a group, we have to load a new control word and inject a new 1 bit. */
    if (control==1)
      {
       control=0x10000|*p_src++;
       control|=(*p_src++)<<8;
      
       /* Because 4096 (the number of entries in the phrase table) is a       */
       /* multiple of 16 (the loop unrolling), and 'unroll' has the value 1   */
       /* or 16 and never increases its initial value, this wraparound check  */
       /* need only be done once per main loop. In fact it can even reside    */
       /* within this 'if'.                                                   */
       next&=0xFFF;
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
         { /* Copy item. */
          register UWORD lenmt; /* Length of copy item minus three.           */
          register UBYTE *p;    /* Points to history posn from which to copy. */
          
          /* Read and dismantle the copy word. Work out from where to copy.   */
          lenmt=*p_src++;
          p=phrase[((lenmt&0xF0)<<4)|*p_src++];
          lenmt&=0xF;

          /* Update the phrase table. Don't do this before p=phrase[...]. */
          phrase[next++]=p_dst;
          
          /* Now perform the copy using a half unrolled loop. */
          *p_dst++=*p++;
          *p_dst++=*p++;
          *p_dst++=*p++;
          while (lenmt--)
             *p_dst++=*p++;
         }
       else
         { /* Literal item. */
          phrase[next++]=p_dst;  /* Update the phrase table.    */
          *p_dst++=*p_src++;     /* Copy over the literal byte. */
          }
          
       /* Shift the control buffer so the next control bit is in bit 0. */
       control>>=1;
       
      } /* End unrolled inner loop. */
               
   } /* End of outer loop */
   
 /* Write the length of the decompressed data before returning. */
 *p_dst_len=p_dst-p_dst_first;
}

/******************************************************************************/
/*                               End of LZRW2.C                               */
/******************************************************************************/

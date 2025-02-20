/******************************************************************************/
/*                                                                            */
/*                                 COMPRESS.H                                 */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/* Author : Ross Williams.                                                    */
/* Date   : December 1989.                                                    */
/*                                                                            */
/* This header file defines the interface to a set of functions called        */
/* 'compress', each member of which implements a particular data compression  */
/* algorithm.                                                                 */
/*                                                                            */
/* Normally in C programming, for each .H file, there is a corresponding .C   */
/* file that implements the functions promised in the .H file.                */
/* Here, there are many .C files corresponding to this header file.           */
/* Each comforming implementation file contains a single function             */
/* called 'compress' that implements a single data compression                */
/* algorithm that conforms with the interface specified in this header file.  */
/* Only one algorithm can be linked in at a time in this organization.        */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/*                    DEFINITION OF FUNCTION COMPRESS                         */
/*                    ===============================                         */
/*                                                                            */
/* Summary of Function Compress                                               */
/* ----------------------------                                               */
/* The action that 'compress' takes depends on its first argument called      */
/* 'action'.  The function provides three actions:                            */
/*                                                                            */
/*    - Return information about the algorithm.                               */
/*    - Compress   a block of memory.                                         */
/*    - Decompress a block of memory.                                         */
/*                                                                            */
/* Parameters                                                                 */
/* ----------                                                                 */
/* See the formal C definition later for a description of the parameters.     */
/*                                                                            */
/* Constants                                                                  */
/* ---------                                                                  */
/* COMPRESS_OVERRUN: The constant COMPRESS_OVERRUN defines by how many bytes  */
/* an algorithm is allowed to expand a block during a compression operation.  */
/*                                                                            */
/* Although compression algorithms usually compress data, there will always   */
/* be data that a given compressor will expand (this can be proven).          */
/* Fortunately, the degree of expansion can be limited to a single bit, by    */
/* copying over the input data if the data gets bigger during compression.    */
/* To allow for this possibility, the first bit of a compressed               */
/* representation can be used as a flag indicating whether the                */
/* input data was copied over, or truly compressed. In practice, the first    */
/* byte would be used to store this bit so as to maintain byte alignment.     */
/*                                                                            */
/* Unfortunately, in general, the only way to tell if an algorithm will       */
/* expand a particular block of data is to run the algorithm on the data.     */
/* If the algorithm does not continuously monitor how many output bytes it    */
/* has written, it might write an output block far larger than the input      */
/* block before realizing that it has done so.                                */
/* On the other hand, continuous checks on output length are inefficient.     */
/*                                                                            */
/* To cater for all these problems, this interface definition:                */
/* > Allows a compression algorithm to return an output block that is up to   */
/*   COMPRESS_OVERRUN bytes longer than the input block.                      */
/* > Allows a compression algorithm to write up to COMPRESS_OVERRUN bytes     */
/*   more than the length of the input block to the memory of the output      */
/*   block regardless of the length of the output block eventually returned.  */
/*   This allows an algorithm to overrun the length of the input block in the */
/*   output block by up to COMPRESS_OVERRUN bytes between expansion checks.   */
/*                                                                            */
/* The problem does not arise for decompression.                              */
/*                                                                            */
/* Identity Action                                                            */
/* ---------------                                                            */
/* > action must be COMPRESS_ACTION_IDENTITY.                                 */
/* > p_dst_len must point to a longword to receive a longword address.        */
/* > The value of the other parameters does not matter.                       */
/* > After execution, the longword that p_dst_len points to will be a pointer */
/*   to a structure of type compress_identity.                                */
/*   Thus, for example, after the call, (*p_dst_len)->memory will return the  */
/*   number of bytes of working memory that the algorithm requires to run.    */
/* > The values of the identity structure returned are fixed constant         */
/*   attributes of the algorithm and must not vary from call to call.         */
/*                                                                            */
/* Common Requirements for Compression and Decompression Actions              */
/* -------------------------------------------------------------              */
/* > wrk_mem must point to an unused block of memory of a length specified in */
/*   the algorithm's identity block. The identity block can be obtained by    */
/*   making a separate call to compress, specifying the identity action.      */
/* > The INPUT BLOCK is defined to be Memory[src_addr,src_addr+src_len-1].    */
/* > dst_len will be used to denote *p_dst_len.                               */
/* > dst_len is not read by compress, only written.                           */
/* > The value of dst_len is defined only upon termination.                   */
/* > The OUTPUT BLOCK is defined to be Memory[dst_addr,dst_addr+dst_len-1].   */
/*                                                                            */
/* Compression Action                                                         */
/* ------------------                                                         */
/* > action must be COMPRESS_ACTION_COMPRESS.                                 */
/* > src_len must be in the range [0,COMPRESS_MAX_ORG].                       */
/* > The OUTPUT ZONE is defined to be                                         */
/*      Memory[dst_addr,dst_addr+src_len-1+COMPRESS_OVERRUN].                 */
/* > The function can modify any part of the output zone regardless of the    */
/*   final length of the output block.                                        */
/* > The input block and the output zone must not overlap.                    */
/* > dst_len will be in the range [0,src_len+COMPRESS_OVERRUN].               */
/* > dst_len will be in the range [0,COMPRESS_MAX_COM] (from prev fact).      */
/* > The output block will consist of a representation of the input block.    */
/*                                                                            */
/* Decompression Action                                                       */
/* --------------------                                                       */
/* > action must be COMPRESS_ACTION_DECOMPRESS.                               */
/* > The input block must be the result of an earlier compression operation.  */
/* > If the previous fact is true, the following facts must also be true:     */
/*   > src_len will be in the range [0,COMPRESS_MAX_COM].                     */
/*   > dst_len will be in the range [0,COMPRESS_MAX_ORG].                     */
/* > The input and output blocks must not overlap.                            */
/* > Only the output block is modified.                                       */
/* > Upon termination, the output block will consist of the bytes contained   */
/*   in the input block passed to the earlier compression operation.          */
/*                                                                            */
/******************************************************************************/


#ifndef DONE_PORT       /* Only do this if not previously done.                   */

   #ifdef THINK_C
      #define UBYTE unsigned char      /* Unsigned byte                       */
      #define UWORD unsigned int       /* Unsigned word (2 bytes)             */
      #define uint32_t unsigned long      /* Unsigned word (4 bytes)             */
      #define BOOL  unsigned char      /* Boolean                             */
      #define FOPEN_BINARY_READ  "rb"  /* Mode string for binary reading.     */
      #define FOPEN_BINARY_WRITE "wb"  /* Mode string for binary writing.     */
      #define FOPEN_TEXT_APPEND  "a"   /* Mode string for text appending.     */
      #define REAL double              /* USed for floating point stuff.      */
   #endif

   #define DONE_PORT                   /* Don't do all this again.            */
   #define MALLOC_FAIL NULL            /* Failure status from malloc()        */
   #define LOCAL static                /* For non-exported routines.          */
   #define EXPORT                      /* Signals exported function.          */
   #define then                        /* Useful for aligning ifs.            */

   /* Added by Matt Mahoney, Feb. 14, 2008 */
   typedef unsigned char UBYTE;
   typedef unsigned short UWORD;

   #define TRUE 1
   #define FALSE 0
   #define REGISTER
   #undef LOCAL
   #define LOCAL
   #define ass(p,m) if (!(p)) fprintf(stderr,"%s\n",m),exit(1)
   /* End added section */

#endif

/******************************************************************************/
/*                              End of PORT.H                                 */
/******************************************************************************/


#include <stdint.h>

#define COMPRESS_ACTION_IDENTITY   0
#define COMPRESS_ACTION_COMPRESS   1
#define COMPRESS_ACTION_DECOMPRESS 2

#define COMPRESS_OVERRUN 1024
#define COMPRESS_MAX_COM 0x70000000
#define COMPRESS_MAX_ORG (COMPRESS_MAX_COM-COMPRESS_OVERRUN)

#define COMPRESS_MAX_STRLEN 255

/* The following structure provides information about the algorithm.         */
/* > The top bit of id must be zero. The remaining bits must be chosen by    */
/*   the author of the algorithm by tossing a coin 31 times.                 */
/* > The amount of memory requested by the algorithm is specified in bytes   */
/*   and must be in the range [0,0x70000000].                                */
/* > All strings s must be such that strlen(s)<=COMPRESS_MAX_STRLEN.         */
struct compress_identity
  {
   uint32_t id;           /* Identifying number of algorithm.            */
   uint32_t memory;       /* Number of bytes of working memory required. */

   char  *name;        /* Name of algorithm.                          */
   char  *version;     /* Version number.                             */
   char  *date;        /* Date of release of this version.            */
   char  *copyright;   /* Copyright message.                          */

   char  *author;      /* Author of algorithm.                        */
   char  *affiliation; /* Affiliation of author.                      */
   char  *vendor;      /* Where the algorithm can be obtained.        */
  };

void  lzrw1a_compress(        /* Single function interface to compression algorithm. */
UWORD     action,      /* Action to be performed.                             */
UBYTE   *wrk_mem,      /* Working memory temporarily given to routine to use. */
UBYTE   *src_adr,      /* Address of input  data.                             */
uint32_t    src_len,      /* Length  of input  data.                             */
UBYTE   *dst_adr,      /* Address of output data.                             */
uint32_t *p_dst_len       /* Pointer to a longword where routine will write:     */
                       /*    If action=..IDENTITY   => Adr of id structure.   */
                       /*    If action=..COMPRESS   => Length of output data. */
                       /*    If action=..DECOMPRESS => Length of output data. */
);

void  lzrw2_compress(        /* Single function interface to compression algorithm. */
UWORD     action,      /* Action to be performed.                             */
UBYTE   *wrk_mem,      /* Working memory temporarily given to routine to use. */
UBYTE   *src_adr,      /* Address of input  data.                             */
uint32_t    src_len,      /* Length  of input  data.                             */
UBYTE   *dst_adr,      /* Address of output data.                             */
uint32_t *p_dst_len       /* Pointer to a longword where routine will write:     */
                       /*    If action=..IDENTITY   => Adr of id structure.   */
                       /*    If action=..COMPRESS   => Length of output data. */
                       /*    If action=..DECOMPRESS => Length of output data. */
);

void  lzrw3_compress(        /* Single function interface to compression algorithm. */
UWORD     action,      /* Action to be performed.                             */
UBYTE   *wrk_mem,      /* Working memory temporarily given to routine to use. */
UBYTE   *src_adr,      /* Address of input  data.                             */
uint32_t    src_len,      /* Length  of input  data.                             */
UBYTE   *dst_adr,      /* Address of output data.                             */
uint32_t *p_dst_len       /* Pointer to a longword where routine will write:     */
                       /*    If action=..IDENTITY   => Adr of id structure.   */
                       /*    If action=..COMPRESS   => Length of output data. */
                       /*    If action=..DECOMPRESS => Length of output data. */
);

void  lzrw3a_compress(        /* Single function interface to compression algorithm. */
					 UWORD     action,      /* Action to be performed.                             */
					 UBYTE   *wrk_mem,      /* Working memory temporarily given to routine to use. */
					 UBYTE   *src_adr,      /* Address of input  data.                             */
					 uint32_t    src_len,      /* Length  of input  data.                             */
					 UBYTE   *dst_adr,      /* Address of output data.                             */
					 uint32_t *p_dst_len       /* Pointer to a longword where routine will write:     */
					 /*    If action=..IDENTITY   => Adr of id structure.   */
					 /*    If action=..COMPRESS   => Length of output data. */
					 /*    If action=..DECOMPRESS => Length of output data. */
					 );

uint32_t lzrw1a_req_mem();
uint32_t lzrw2_req_mem();
uint32_t lzrw3_req_mem();

void lzrw1_compress(UBYTE *p_src_first,uint32_t src_len,UBYTE *p_dst_first,uint32_t *p_dst_len);
void lzrw1_decompress(UBYTE *p_src_first,uint32_t src_len,UBYTE *p_dst_first,uint32_t *p_dst_len);

/******************************************************************************/
/*                             End of COMPRESS.H                              */
/******************************************************************************/

void fast_copy(void *src_adr, void *dst_adr, unsigned long src_len);
/* This function copies a block of memory very quickly.                       */
/* The exact speed depends on the relative alignment of the blocks of memory. */
/* PRE  : 0<=src_len<=(2^32)-1 .                                              */
/* PRE  : Source and destination blocks must not overlap.                     */
/* POST : MEM[dst_adr,dst_adr+src_len-1]=MEM[src_adr,src_adr+src_len-1].      */
/* POST : MEM[dst_adr,dst_adr+src_len-1] is the only memory changed.          */

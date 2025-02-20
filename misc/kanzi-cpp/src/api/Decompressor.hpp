/*
Copyright 2011-2024 Frederic Langlet
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once
#ifndef _libAPI_
#define _libAPI_

#ifdef _WIN32
   #define CDECL __cdecl
#else
   #define CDECL
#endif

#include <stdio.h>

#ifdef __cplusplus
   extern "C" {
#endif

   typedef unsigned char BYTE;

   /**
    *  Decompression parameters
    */
   struct dData {
       // Required fields
       unsigned int bufferSize;      /* read buffer size (at least block size) */
       unsigned int jobs;            /* max number of concurrent tasks */
       int headerless;               /* bool to indicate if the bitstream has a header (usually yes) */

       // Optional fields: only required if headerless is true
       char transform[64];           /* name of transforms [None|PACK|BWT|BWTS|LZ|LZX|LZP|ROLZ|ROLZX]
                                                       [RLT|ZRLT|MTFT|RANK|SRT|TEXT|MM|EXE|UTF] */
       char entropy[16];             /* name of entropy codec [None|Huffman|ANS0|ANS1|Range|FPAQ|TPAQ|TPAQX|CM] */
       unsigned int blockSize;       /* size of block in bytes */
       unsigned long originalSize;   /* size of original file in bytes */
       int checksum;                 /* bool to indicate use of block checksum */
       int bsVersion;                /* version of the bitstream */
   };

   /**
    *  Decompression context: encapsulates decompressor state (opaque: could change in future versions)
    */
   struct dContext {
       void* pCis;
       unsigned int bufferSize;
       void* fis;
   };

   /**
    *  Initialize the decompressor internal states.
    *
    *  @param dParam [IN] - the decompression parameters
    *  @param src [IN] - the source stream of compressed data
    *  @param ctx [IN|OUT] - a pointer to the decompression context created by the call
    *
    *  @return 0 in case of success
    */
   int CDECL initDecompressor(struct dData* dParam, FILE* src, struct dContext** ctx);

   /**
    *  Decompress a block of data. The decompressor must have been initialized.
    *
    *  @param ctx [IN] - the decompression context created during initialization
    *  @param dst [IN] - the destination block of decompressed data
    *  @param inSize [OUT] - the number of bytes read from source.
    *  @param outSize [IN|OUT] - the size of the block to decompress.
    *                            Updated to reflect the number of decompressed bytes
    *
    *  @return 0 in case of success
    */
   int CDECL decompress(struct dContext* ctx, BYTE* dst, int* inSize, int* outSize);

   /**
    *  Dispose the decompressor and cleanup memory resources.
    *
    *  @param ctx [IN] - the compression context created during initialization
    *
    *  @return 0 in case of success
    */
   int CDECL disposeDecompressor(struct dContext* ctx);

#ifdef __cplusplus
   }
#endif


#endif


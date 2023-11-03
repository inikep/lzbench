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
    *  Compression parameters
    */
   struct cData {
       // Required fields
       char transform[64];      /* name of transforms [None|PACK|BWT|BWTS|LZ|LZX|LZP|ROLZ|ROLZX]
                                                          [RLT|ZRLT|MTFT|RANK|SRT|TEXT|MM|EXE|UTF] */
       char entropy[16];        /* name of entropy codec [None|Huffman|ANS0|ANS1|Range|FPAQ|TPAQ|TPAQX|CM] */
       unsigned int blockSize;  /* size of block in bytes */
       unsigned int jobs;       /* max number of concurrent tasks */
       int checksum;            /* bool to indicate use of block checksum */
       int headerless;          /* bool to indicate if the bitstream has a header (usually yes) */
   };

   /**
    *  Compression context: encapsulates compressor state (opaque: could change in future versions)
    */
   struct cContext {
       void* pCos;
       unsigned int blockSize;
       void* fos;
   };


    /**
    *  Initialize the compressor internal states.
    *
    *  @param cParam [IN] - the compression parameters
    *  @param dst [IN] - the destination stream of compressed data
    *  @param ctx [IN|OUT] - pointer to the compression context created by the call
    *
    *  @return 0 in case of success
    */
   int CDECL initCompressor(struct cData* cParam, FILE* dst, struct cContext** ctx);

    /**
    *  Compress a block of data. The compressor must have been initialized.
    *
    *  @param ctx [IN] - the compression context created during initialization
    *  @param src [IN] - the source block of data to compress
    *  @param inSize [IN|OUT] - the size of the source block to compress.
                                Updated to reflect the number bytes written to the destination.
    *  @param outSize [OUT] - the size of the compressed data
    *
    *  @return 0 in case of success
    */
   int CDECL compress(struct cContext* ctx, const BYTE* src, int* inSize, int* outSize);

   /**
    *  Dispose the compressor and cleanup memory resources.
    *
    *  @param ctx [IN] - the compression context created during initialization
    *  @param outSize [IN|OUT] - the number of bytes written to the destination
    *                            (the compressor may flush internal data)
    *
    *  @return 0 in case of success
    */
   int CDECL disposeCompressor(struct cContext* ctx, int* outSize);

#ifdef __cplusplus
   }
#endif


#endif


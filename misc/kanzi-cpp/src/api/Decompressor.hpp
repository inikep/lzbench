/*
Copyright 2011-2026 Frederic Langlet
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
#ifndef knz_Decompressor
#define knz_Decompressor

#ifdef _WIN32
   #define CDECL __cdecl

   #ifdef KANZI_EXPORTS
      #define KANZI_API __declspec(dllexport)
   #else
      #define KANZI_API
   #endif
#else
   #define CDECL
   #define KANZI_API
#endif

#include <stdio.h>

#ifdef __cplusplus
    #if __cplusplus >= 201103L
       // C++ 11 or higher
       #define KANZI_NOEXCEPT noexcept
    #else
       #define KANZI_NOEXCEPT
    #endif
#else
   #define KANZI_NOEXCEPT
#endif


#define KANZI_DECOMP_VERSION_MAJOR 1
#define KANZI_DECOMP_VERSION_MINOR 0
#define KANZI_DECOMP_VERSION_PATCH 0


#ifdef __cplusplus
   extern "C" {
#endif
   /**
    *  Decompression context: encapsulates decompressor state (opaque: could change in future versions)
    */
   struct dContext;

   /**
    *  Decompression parameters
    */
   struct dData {
       // Required fields
       size_t bufferSize;            /* read buffer size (at least block size) */
       unsigned int jobs;            /* max number of concurrent tasks */
       int headerless;               /* bool to indicate if the bitstream has a header (usually set to 0) */

       // Optional fields: only required if headerless is true
       char transform[64];           /* name of transforms [None|PACK|BWT|BWTS|LZ|LZX|LZP|ROLZ|ROLZX]
                                                       [RLT|ZRLT|MTFT|RANK|SRT|TEXT|MM|EXE|UTF|DNA] */
       char entropy[16];             /* name of entropy codec [None|Huffman|ANS0|ANS1|Range|FPAQ|TPAQ|TPAQX|CM] */
       unsigned int blockSize;       /* size of block in bytes */
       size_t originalSize;          /* size of original file in bytes */
       int checksum;                 /* 0, 32 or 64 to indicate size of block checksum */
       int bsVersion;                /* version of the bitstream */
   };

   /**
    * @return the version number of the library.
    * Useful for checking for compatibility at runtime.
    */
   KANZI_API unsigned int CDECL getDecompressorVersion(void) KANZI_NOEXCEPT;

   /**
    *  Initialize the decompressor internal states.
    *
    *  @param dParam [IN|OUT] - the decompression parameters. Transform and entropy are
    *                           validated and rewritten.
    *  @param src [IN] - the source stream of compressed data
    *  @param ctx [IN|OUT] - a pointer to the decompression context created by the call
    *
    *  @return 0 in case of success, else see error code in Error.hpp
    */
   KANZI_API int CDECL initDecompressor(struct dData* dParam, FILE* src, struct dContext** ctx) KANZI_NOEXCEPT;

   /**
    *  Decompress a block of data. The decompressor must have been initialized.
    *
    *  @param ctx [IN] - the decompression context created during initialization
    *  @param dst [IN] - the destination block of decompressed data
    *  @param inSize [OUT] - the number of bytes read from source.
    *  @param outSize [IN|OUT] - the size of the block to decompress.
    *                            Updated to reflect the number of decompressed bytes
    *
    *  @return 0 in case of success, else see error code in Error.hpp
    */
   KANZI_API int CDECL decompress(struct dContext* ctx, unsigned char* dst, size_t* inSize, size_t* outSize) KANZI_NOEXCEPT;

   /**
    *  Dispose the decompressor and cleanup memory resources.
    *
    *  @param ctx [IN] - the compression context created during initialization
    *
    *  @return 0 in case of success, else see error code in Error.hpp
    */
   KANZI_API int CDECL disposeDecompressor(struct dContext** ctx) KANZI_NOEXCEPT;

#ifdef __cplusplus
   }
#endif


#endif


/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Block Sorting Compressor                                  */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

   Copyright (c) 2009-2021 Ilya Grebnov <ilya.grebnov@gmail.com>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Please see the file LICENSE for full copyright information and file AUTHORS
for full list of contributors.

See also the bsc and libbsc web site:
  http://libbsc.com/ for more information.

--*/

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <memory.h>

#include "libbsc/libbsc.h"
#include "libbsc/filters.h"
#include "libbsc/platform/platform.h"

#pragma pack(push, 1)

#define LIBBSC_CONTEXTS_AUTODETECT   3

unsigned char bscFileSign[4] = {'b', 's', 'c', 0x31};

typedef struct BSC_BLOCK_HEADER
{
    long long       blockOffset;
    signed char     recordSize;
    signed char     sortingContexts;
} BSC_BLOCK_HEADER;

#pragma pack(pop)

int paramBlockSize                = 25 * 1024 * 1024;
int paramBlockSorter              = LIBBSC_BLOCKSORTER_BWT;
int paramCoder                    = LIBBSC_CODER_QLFC_STATIC;
int paramSortingContexts          = LIBBSC_CONTEXTS_FOLLOWING;

int paramEnableParallelProcessing = 1;
int paramEnableMultiThreading     = 1;
int paramEnableFastMode           = 1;
int paramEnableLargePages         = 0;
int paramEnableCUDA               = 0;
int paramEnableSegmentation       = 0;
int paramEnableReordering         = 0;
int paramEnableLZP                = 1;
int paramLZPHashSize              = 15;
int paramLZPMinLen                = 128;

int paramFeatures()
{
    int features =
        (paramEnableFastMode       ? LIBBSC_FEATURE_FASTMODE       : LIBBSC_FEATURE_NONE) |
        (paramEnableMultiThreading ? LIBBSC_FEATURE_MULTITHREADING : LIBBSC_FEATURE_NONE) |
        (paramEnableLargePages     ? LIBBSC_FEATURE_LARGEPAGES     : LIBBSC_FEATURE_NONE) |
        (paramEnableCUDA           ? LIBBSC_FEATURE_CUDA           : LIBBSC_FEATURE_NONE)
    ;

    return features;
}

#if defined(__GNUC__) && (defined(_GLIBCXX_USE_LFS) || defined(__MINGW32__))
    #define BSC_FSEEK fseeko64
    #define BSC_FTELL ftello64
    #define BSC_FILEOFFSET off64_t
#elif defined(_MSC_VER) && _MSC_VER >= 1400
    #define BSC_FSEEK _fseeki64
    #define BSC_FTELL _ftelli64
    #define BSC_FILEOFFSET __int64
#else
    #define BSC_FSEEK fseek
    #define BSC_FTELL ftell
    #define BSC_FILEOFFSET long
#endif

#if defined(WIN32) || defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__) || defined(_MSC_VER)
  #include <windows.h>
  double BSC_CLOCK() { return 0.001 * GetTickCount(); }
#elif defined (__unix) || defined (__linux__) || defined (__QNX__) || defined (_AIX)  || defined (__NetBSD__) || defined(macintosh) || defined (_MAC)
  #include <sys/time.h>
  double BSC_CLOCK() { timeval tv; gettimeofday(&tv, 0); return tv.tv_sec + tv.tv_usec * 0.000001; }
#else
  double BSC_CLOCK() { return (double)clock() / CLOCKS_PER_SEC; }
#endif

int segmentedBlock[256];

void Compression(char * argv[])
{
    if (!paramEnableLZP)
    {
        paramLZPHashSize = 0;
        paramLZPMinLen = 0;
    }

    FILE * fInput = fopen(argv[2], "rb");
    if (fInput == NULL)
    {
        fprintf(stderr, "Can't open input file: %s!\n", argv[2]);
        exit(1);
    }

    FILE * fOutput = fopen(argv[3], "wb");
    if (fOutput == NULL)
    {
        fprintf(stderr, "Can't create output file: %s!\n", argv[3]);
        exit(1);
    }

    if (BSC_FSEEK(fInput, 0, SEEK_END))
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[2]);
        exit(1);
    }

    BSC_FILEOFFSET fileSize = BSC_FTELL(fInput);
    if (fileSize < 0)
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[2]);
        exit(1);
    }

    if (BSC_FSEEK(fInput, 0, SEEK_SET))
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[2]);
        exit(1);
    }

    if (paramBlockSize > fileSize)
    {
        paramBlockSize = (int)fileSize;
    }

    if (fwrite(bscFileSign, sizeof(bscFileSign), 1, fOutput) != 1)
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[3]);
        exit(1);
    }

    int nBlocks = paramBlockSize > 0 ? (int)((fileSize + paramBlockSize - 1) / paramBlockSize) : 0;
    if (fwrite(&nBlocks, sizeof(nBlocks), 1, fOutput) != 1)
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[3]);
        exit(1);
    }

    double startTime = BSC_CLOCK();

#ifdef LIBBSC_OPENMP

    int numThreads = 1;
    if (paramEnableParallelProcessing)
    {
        numThreads = omp_get_max_threads();
        if (numThreads <= nBlocks) paramEnableMultiThreading = 0;
        if (numThreads >= nBlocks) numThreads = nBlocks;
    }

#endif

    int segmentationStart = 0, segmentationEnd = 0;

#ifdef LIBBSC_OPENMP
    #pragma omp parallel num_threads(numThreads) if(numThreads > 1)
#endif
    {
        unsigned char * buffer = (unsigned char *)bsc_malloc(paramBlockSize + LIBBSC_HEADER_SIZE);
        if (buffer == NULL)
        {
#ifdef LIBBSC_OPENMP
            #pragma omp critical(print)
#endif
            {

                fprintf(stderr, "Not enough memory! Please check README file for more information.\n");
                exit(2);
            }
        }

        while (true)
        {
            BSC_FILEOFFSET  blockOffset     = 0;
            int             dataSize        = 0;

#ifdef LIBBSC_OPENMP
            #pragma omp critical(input)
#endif
            {
                if ((feof(fInput) == 0) && (BSC_FTELL(fInput) != fileSize))
                {
#ifdef LIBBSC_OPENMP
                    #pragma omp master
#endif
                    {
                        double progress = (100.0 * (double)BSC_FTELL(fInput)) / fileSize;
                        fprintf(stdout, "\rCompressing %.55s(%02d%%)", argv[2], (int)progress);
                        fflush(stdout);
                    }

                    blockOffset = BSC_FTELL(fInput);

                    int currentBlockSize = paramBlockSize;
                    if (paramEnableSegmentation)
                    {
                        if (segmentationEnd - segmentationStart > 1) currentBlockSize = segmentedBlock[segmentationStart];
                    }

                    dataSize = (int)fread(buffer, 1, currentBlockSize, fInput);
                    if (dataSize <= 0)
                    {
                        fprintf(stderr, "\nIO error on file: %s!\n", argv[2]);
                        exit(1);
                    }

                    if (paramEnableSegmentation)
                    {
                        bool bSegmentation = false;

                        if (segmentationStart == segmentationEnd) bSegmentation = true;
                        if ((segmentationEnd - segmentationStart == 1) && (dataSize != segmentedBlock[segmentationStart])) bSegmentation = true;

                        if (bSegmentation)
                        {
                            segmentationStart = 0; segmentationEnd = bsc_detect_segments(buffer, dataSize, segmentedBlock, 256, paramFeatures());
                            if (segmentationEnd <= LIBBSC_NO_ERROR)
                            {
                                switch (segmentationEnd)
                                {
                                    case LIBBSC_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough memory! Please check README file for more information.\n"); break;
                                    default                         : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                                }
                                exit(2);
                            }
                        }

                        int newDataSize = segmentedBlock[segmentationStart++];
                        if (dataSize != newDataSize)
                        {
                            BSC_FILEOFFSET pos = BSC_FTELL(fInput) - dataSize + newDataSize;
                            BSC_FSEEK(fInput, pos, SEEK_SET);
                            dataSize = newDataSize;
                        }
                    }
                }
            }

            if (dataSize == 0) break;

            signed char recordSize = 1;
            if (paramEnableReordering)
            {
                recordSize = bsc_detect_recordsize(buffer, dataSize, paramFeatures());
                if (recordSize < LIBBSC_NO_ERROR)
                {
#ifdef LIBBSC_OPENMP
                    #pragma omp critical(print)
#endif
                    {
                        switch (recordSize)
                        {
                            case LIBBSC_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough memory! Please check README file for more information.\n"); break;
                            default                         : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                        }
                        exit(2);
                    }
                }
                if (recordSize > 1)
                {
                    int result = bsc_reorder_forward(buffer, dataSize, recordSize, paramFeatures());
                    if (result != LIBBSC_NO_ERROR)
                    {
#ifdef LIBBSC_OPENMP
                        #pragma omp critical(print)
#endif
                        {
                            switch (result)
                            {
                                case LIBBSC_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough memory! Please check README file for more information.\n"); break;
                                default                         : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                            }
                            exit(2);
                        }
                    }
                }
            }

            signed char sortingContexts = paramSortingContexts;
            if (paramSortingContexts == LIBBSC_CONTEXTS_AUTODETECT)
            {
                sortingContexts = bsc_detect_contextsorder(buffer, dataSize, paramFeatures());
                if (sortingContexts < LIBBSC_NO_ERROR)
                {
#ifdef LIBBSC_OPENMP
                    #pragma omp critical(print)
#endif
                    {
                        switch (sortingContexts)
                        {
                            case LIBBSC_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough memory!\n"); break;
                            default                         : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                        }
                        exit(2);
                    }
                }
            }
            if (sortingContexts == LIBBSC_CONTEXTS_PRECEDING)
            {
                int result = bsc_reverse_block(buffer, dataSize, paramFeatures());
                if (result != LIBBSC_NO_ERROR)
                {
#ifdef LIBBSC_OPENMP
                    #pragma omp critical(print)
#endif
                    {
                        fprintf(stderr, "\nInternal program error, please contact the author!\n");
                        exit(2);
                    }
                }
            }

            int blockSize = bsc_compress(buffer, buffer, dataSize, paramLZPHashSize, paramLZPMinLen, paramBlockSorter, paramCoder, paramFeatures());
            if (blockSize == LIBBSC_NOT_COMPRESSIBLE)
            {
#ifdef LIBBSC_OPENMP
                #pragma omp critical(input)
#endif
                {
                    sortingContexts = LIBBSC_CONTEXTS_FOLLOWING; recordSize = 1;

                    BSC_FILEOFFSET pos = BSC_FTELL(fInput);
                    {
                        BSC_FSEEK(fInput, blockOffset, SEEK_SET);
                        if (dataSize != (int)fread(buffer, 1, dataSize, fInput))
                        {
                            fprintf(stderr, "\nInternal program error, please contact the author!\n");
                            exit(2);
                        }
                    }
                    BSC_FSEEK(fInput, pos, SEEK_SET);
                }

                blockSize = bsc_store(buffer, buffer, dataSize, paramFeatures());
            }
            if (blockSize < LIBBSC_NO_ERROR)
            {
#ifdef LIBBSC_OPENMP
                #pragma omp critical(print)
#endif
                {
                    switch (blockSize)
                    {
                        case LIBBSC_NOT_ENOUGH_MEMORY       : fprintf(stderr, "\nNot enough memory! Please check README file for more information.\n"); break;
                        case LIBBSC_NOT_SUPPORTED           : fprintf(stderr, "\nSpecified compression method is not supported on this platform!\n"); break;
                        case LIBBSC_GPU_ERROR               : fprintf(stderr, "\nGeneral GPU failure! Please check README file for more information.\n"); break;
                        case LIBBSC_GPU_NOT_SUPPORTED       : fprintf(stderr, "\nYour GPU is not supported! Please check README file for more information.\n"); break;
                        case LIBBSC_GPU_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough GPU memory! Please check README file for more information.\n"); break;

                        default                             : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                    }
                    exit(2);
                }
            }

#ifdef LIBBSC_OPENMP
            #pragma omp critical(output)
#endif
            {
                BSC_BLOCK_HEADER header = {blockOffset, recordSize, sortingContexts};

                if (fwrite(&header, sizeof(BSC_BLOCK_HEADER), 1, fOutput) != 1)
                {
                    fprintf(stderr, "\nIO error on file: %s!\n", argv[3]);
                    exit(1);
                }

                if ((int)fwrite(buffer, 1, blockSize, fOutput) != blockSize)
                {
                    fprintf(stderr, "\nIO error on file: %s!\n", argv[3]);
                    exit(1);
                }
            }

        }

        bsc_free(buffer);
    }

    fprintf(stdout, "\r%.55s compressed %.0f into %.0f in %.3f seconds.\n", argv[2], (double)fileSize, (double)BSC_FTELL(fOutput), BSC_CLOCK() - startTime);

    fclose(fInput); fclose(fOutput);
}

void Decompression(char * argv[])
{
    FILE * fInput = fopen(argv[2], "rb");
    if (fInput == NULL)
    {
        fprintf(stderr, "Can't open input file: %s!\n", argv[2]);
        exit(1);
    }

    FILE * fOutput = fopen(argv[3], "wb");
    if (fOutput == NULL)
    {
        fprintf(stderr, "Can't create output file: %s!\n", argv[3]);
        exit(1);
    }

    if (BSC_FSEEK(fInput, 0, SEEK_END))
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[2]);
        exit(1);
    }

    BSC_FILEOFFSET fileSize = BSC_FTELL(fInput);
    if (fileSize < 0)
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[2]);
        exit(1);
    }

    if (BSC_FSEEK(fInput, 0, SEEK_SET))
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[2]);
        exit(1);
    }

    unsigned char inputFileSign[sizeof(bscFileSign)];

    if (fread(inputFileSign, sizeof(bscFileSign), 1, fInput) != 1)
    {
        fprintf(stderr, "This is not bsc archive!\n");
        exit(1);
    }

    if (memcmp(inputFileSign, bscFileSign, sizeof(bscFileSign)) != 0)
    {
        fprintf(stderr, "This is not bsc archive or invalid compression method!\n");
        exit(2);
    }

    int nBlocks = 0;
    if (fread(&nBlocks, sizeof(nBlocks), 1, fInput) != 1)
    {
        fprintf(stderr, "This is not bsc archive!\n");
        exit(1);
    }

    double startTime = BSC_CLOCK();

#ifdef LIBBSC_OPENMP

    int numThreads = 1;
    if (paramEnableParallelProcessing)
    {
        numThreads = omp_get_max_threads();
        if (numThreads <= nBlocks) paramEnableMultiThreading = 0;
        if (numThreads >= nBlocks) numThreads = nBlocks;
    }

    #pragma omp parallel num_threads(numThreads) if(numThreads > 1)
#endif
    {
        int bufferSize = -1; unsigned char * buffer = NULL;

        while (true)
        {
            BSC_FILEOFFSET  blockOffset     = 0;

            signed char     sortingContexts = 0;
            signed char     recordSize      = 0;
            int             blockSize       = 0;
            int             dataSize        = 0;

#ifdef LIBBSC_OPENMP
            #pragma omp critical(input)
#endif
            {
                if ((feof(fInput) == 0) && (BSC_FTELL(fInput) != fileSize))
                {
#ifdef LIBBSC_OPENMP
                    #pragma omp master
#endif
                    {
                        double progress = (100.0 * (double)BSC_FTELL(fInput)) / fileSize;
                        fprintf(stdout, "\rDecompressing %.55s(%02d%%)", argv[2], (int)progress);
                        fflush(stdout);
                    }

                    BSC_BLOCK_HEADER header = {0, 0, 0};
                    if (fread(&header, sizeof(BSC_BLOCK_HEADER), 1, fInput) != 1)
                    {
                        fprintf(stderr, "\nUnexpected end of file: %s!\n", argv[2]);
                        exit(1);
                    }

                    recordSize = header.recordSize;
                    if (recordSize < 1)
                    {
                        fprintf(stderr, "\nThis is not bsc archive or invalid compression method!\n");
                        exit(2);
                    }

                    sortingContexts = header.sortingContexts;
                    if ((sortingContexts != LIBBSC_CONTEXTS_FOLLOWING) && (sortingContexts != LIBBSC_CONTEXTS_PRECEDING))
                    {
                        fprintf(stderr, "\nThis is not bsc archive or invalid compression method!\n");
                        exit(2);
                    }

                    blockOffset = (BSC_FILEOFFSET)header.blockOffset;

                    unsigned char bscBlockHeader[LIBBSC_HEADER_SIZE];

                    if (fread(bscBlockHeader, LIBBSC_HEADER_SIZE, 1, fInput) != 1)
                    {
                        fprintf(stderr, "\nUnexpected end of file: %s!\n", argv[2]);
                        exit(1);
                    }

                    if (bsc_block_info(bscBlockHeader, LIBBSC_HEADER_SIZE, &blockSize, &dataSize, paramFeatures()) != LIBBSC_NO_ERROR)
                    {
                        fprintf(stderr, "\nThis is not bsc archive or invalid compression method!\n");
                        exit(2);
                    }

                    if ((blockSize > bufferSize) || (dataSize > bufferSize))
                    {
                        if (blockSize > bufferSize) bufferSize = blockSize;
                        if (dataSize  > bufferSize) bufferSize = dataSize;

                        if (buffer != NULL) { bsc_free(buffer); } buffer = (unsigned char *)bsc_malloc(bufferSize);
                    }

                    if (buffer == NULL)
                    {
                        fprintf(stderr, "\nNot enough memory! Please check README file for more information.\n");
                        exit(2);
                    }

                    memcpy(buffer, bscBlockHeader, LIBBSC_HEADER_SIZE);

                    if (fread(buffer + LIBBSC_HEADER_SIZE, blockSize - LIBBSC_HEADER_SIZE, 1, fInput) != 1)
                    {
                        fprintf(stderr, "\nUnexpected end of file: %s!\n", argv[2]);
                        exit(1);
                    }
                }
            }

            if (dataSize == 0) break;

            int result = bsc_decompress(buffer, blockSize, buffer, dataSize, paramFeatures());
            if (result < LIBBSC_NO_ERROR)
            {
#ifdef LIBBSC_OPENMP
                #pragma omp critical(print)
#endif
                {
                    switch (result)
                    {
                        case LIBBSC_DATA_CORRUPT            : fprintf(stderr, "\nThe compressed data is corrupted!\n"); break;
                        case LIBBSC_NOT_ENOUGH_MEMORY       : fprintf(stderr, "\nNot enough memory! Please check README file for more information.\n"); break;
                        case LIBBSC_GPU_ERROR               : fprintf(stderr, "\nGeneral GPU failure! Please check README file for more information.\n"); break;
                        case LIBBSC_GPU_NOT_SUPPORTED       : fprintf(stderr, "\nYour GPU is not supported! Please check README file for more information.\n"); break;
                        case LIBBSC_GPU_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough GPU memory! Please check README file for more information.\n"); break;

                        default                             : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                    }
                    exit(2);
                }
            }

            if (sortingContexts == LIBBSC_CONTEXTS_PRECEDING)
            {
                result = bsc_reverse_block(buffer, dataSize, paramFeatures());
                if (result != LIBBSC_NO_ERROR)
                {
#ifdef LIBBSC_OPENMP
                    #pragma omp critical(print)
#endif
                    {
                        fprintf(stderr, "\nInternal program error, please contact the author!\n");
                        exit(2);
                    }
                }
            }

            if (recordSize > 1)
            {
                result = bsc_reorder_reverse(buffer, dataSize, recordSize, paramFeatures());
                if (result != LIBBSC_NO_ERROR)
                {
#ifdef LIBBSC_OPENMP
                    #pragma omp critical(print)
#endif
                    {
                        switch (result)
                        {
                            case LIBBSC_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough memory! Please check README file for more information.\n"); break;
                            default                         : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                        }
                        exit(2);
                    }
                }
            }

#ifdef LIBBSC_OPENMP
            #pragma omp critical(output)
#endif
            {
                if (BSC_FSEEK(fOutput, blockOffset, SEEK_SET))
                {
                    fprintf(stderr, "\nIO error on file: %s!\n", argv[3]);
                    exit(1);
                }

                if ((int)fwrite(buffer, 1, dataSize, fOutput) != dataSize)
                {
                    fprintf(stderr, "\nIO error on file: %s!\n", argv[3]);
                    exit(1);
                }
            }
        }

        if (buffer != NULL) bsc_free(buffer);
    }

    if (BSC_FSEEK(fOutput, 0, SEEK_END))
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[3]);
        exit(1);
    }

    fprintf(stdout, "\r%.55s decompressed %.0f into %.0f in %.3f seconds.\n", argv[2], (double)fileSize, (double)BSC_FTELL(fOutput), BSC_CLOCK() - startTime);

    fclose(fInput); fclose(fOutput);
}

void ShowUsage(void)
{
#if !defined(BSC_DECOMPRESSION_ONLY)
    fprintf(stdout, "Usage: bsc <e|d> inputfile outputfile <options>\n\n");
#elif defined(LIBBSC_CUDA_SUPPORT) || defined(_WIN32) || defined(LIBBSC_OPENMP)
    fprintf(stdout, "Usage: bsc d inputfile outputfile <options>\n\n");
#else
    fprintf(stdout, "Usage: bsc d inputfile outputfile\n\n");
#endif

#if !defined(BSC_DECOMPRESSION_ONLY)
    fprintf(stdout, "Block sorting options:\n");
    fprintf(stdout, "  -b<size> Block size in megabytes, default: -b25\n");
    fprintf(stdout, "             minimum: -b1, maximum: -b2047\n");
    fprintf(stdout, "  -m<algo> Block sorting algorithm, default: -m0\n");
    fprintf(stdout, "             -m0 Burrows Wheeler Transform (default)\n");
#ifdef LIBBSC_SORT_TRANSFORM_SUPPORT
    fprintf(stdout, "             -m3..8 Sort Transform of order n\n");
#endif
    fprintf(stdout, "  -c<ctx>  Contexts for sorting, default: -cf\n");
    fprintf(stdout, "             -cf Following contexts (default)\n");
    fprintf(stdout, "             -cp Preceding contexts\n");
    fprintf(stdout, "             -ca Autodetect (experimental)\n");
    fprintf(stdout, "  -e<algo> Entropy encoding algorithm, default: -e1\n");
    fprintf(stdout, "             -e0 Fast Quantized Local Frequency Coding\n");
    fprintf(stdout, "             -e1 Static Quantized Local Frequency Coding (default)\n");
    fprintf(stdout, "             -e2 Adaptive Quantized Local Frequency Coding (best compression)\n");
   
    fprintf(stdout, "\nPreprocessing options:\n");
    fprintf(stdout, "  -p       Disable all preprocessing techniques\n");
    fprintf(stdout, "  -s       Enable segmentation (adaptive block size), default: disable\n");
    fprintf(stdout, "  -r       Enable structured data reordering, default: disable\n");
    fprintf(stdout, "  -l       Enable Lempel-Ziv preprocessing, default: enable\n");
    fprintf(stdout, "  -H<size> LZP dictionary size in bits, default: -H15\n");
    fprintf(stdout, "             minimum: -H10, maximum: -H28\n");
    fprintf(stdout, "  -M<size> LZP minimum match length, default: -M128\n");
    fprintf(stdout, "             minimum: -M4, maximum: -M255\n\n");
#endif

#if defined(LIBBSC_CUDA_SUPPORT) || defined(_WIN32) || defined(LIBBSC_OPENMP)
    fprintf(stdout, "Platform specific options:\n");
#ifdef LIBBSC_CUDA_SUPPORT
    fprintf(stdout, "  -G       Enable Sort Transform acceleration on NVIDIA GPU, default: disable\n");
#endif
#ifdef _WIN32
    fprintf(stdout, "  -P       Enable large 2MB RAM pages, default: disable\n");
#endif
#ifdef LIBBSC_OPENMP
    fprintf(stdout, "  -t       Disable parallel blocks processing, default: enable\n");
    fprintf(stdout, "  -T       Disable multi-core systems support, default: enable\n");
#endif
    fprintf(stdout, "\n");
#endif

#if !defined(BSC_DECOMPRESSION_ONLY) || defined(LIBBSC_CUDA_SUPPORT) || defined(_WIN32) || defined(LIBBSC_OPENMP)
    fprintf(stdout,"Options may be combined into one, like -b128p -m5e1\n");
#endif

    exit(0);
}

void ProcessSwitch(char * s)
{
    if (*s == 0)
    {
        ShowUsage();
    }

    for (; *s != 0; )
    {
        switch (*s++)
        {
            case 'b':
            {
                char * strNum = s; while ((*s >= '0') && (*s <= '9')) s++;
                paramBlockSize = atoi(strNum) * 1024 * 1024;
                if ((paramBlockSize < 1024 * 1024) || (paramBlockSize > 2047 * 1024 * 1024)) ShowUsage();
                break;
            }

            case 'm':
            {
                char * strNum = s; while ((*s >= '0') && (*s <= '9')) s++;
                switch (atoi(strNum))
                {
                    case 0   : paramBlockSorter = LIBBSC_BLOCKSORTER_BWT; break;

#ifdef LIBBSC_SORT_TRANSFORM_SUPPORT
                    case 3   : paramBlockSorter = LIBBSC_BLOCKSORTER_ST3; break;
                    case 4   : paramBlockSorter = LIBBSC_BLOCKSORTER_ST4; break;
                    case 5   : paramBlockSorter = LIBBSC_BLOCKSORTER_ST5; break;
                    case 6   : paramBlockSorter = LIBBSC_BLOCKSORTER_ST6; break;
                    case 7   : paramBlockSorter = LIBBSC_BLOCKSORTER_ST7; paramEnableCUDA = 1; break;
                    case 8   : paramBlockSorter = LIBBSC_BLOCKSORTER_ST8; paramEnableCUDA = 1; break;
#endif

                    default  : ShowUsage();
                }
                break;
            }

            case 'c':
            {
                switch (*s++)
                {
                    case 'f' : paramSortingContexts = LIBBSC_CONTEXTS_FOLLOWING;  break;
                    case 'p' : paramSortingContexts = LIBBSC_CONTEXTS_PRECEDING;  break;
                    case 'a' : paramSortingContexts = LIBBSC_CONTEXTS_AUTODETECT; break;
                    default  : ShowUsage();
                }
                break;
            }

            case 'e':
            {
                switch (*s++)
                {
                    case '0' : paramCoder = LIBBSC_CODER_QLFC_FAST;     break;
                    case '1' : paramCoder = LIBBSC_CODER_QLFC_STATIC;   break;
                    case '2' : paramCoder = LIBBSC_CODER_QLFC_ADAPTIVE; break;
                    default  : ShowUsage();
                }
                break;
            }

            case 'H':
            {
                char * strNum = s; while ((*s >= '0') && (*s <= '9')) s++;
                paramLZPHashSize = atoi(strNum);
                if ((paramLZPHashSize < 10) || (paramLZPHashSize > 28)) ShowUsage();
                break;
            }

            case 'M':
            {
                char * strNum = s; while ((*s >= '0') && (*s <= '9')) s++;
                paramLZPMinLen = atoi(strNum);
                if ((paramLZPMinLen < 4) || (paramLZPMinLen > 255)) ShowUsage();
                break;
            }

            case 'l': paramEnableLZP            = 1; break;
            case 's': paramEnableSegmentation   = 1; break;
            case 'r': paramEnableReordering     = 1; break;

            case 'p': paramEnableLZP = paramEnableSegmentation = paramEnableReordering = 0; break;

#ifdef LIBBSC_OPENMP
            case 't': paramEnableParallelProcessing = 0; break;
            case 'T': paramEnableParallelProcessing = paramEnableMultiThreading = 0; break;
#endif

#ifdef LIBBSC_CUDA_SUPPORT
            case 'G': paramEnableCUDA           = 1; break;
#endif

#ifdef _WIN32
            case 'P': paramEnableLargePages     = 1; break;
#endif

            default : ShowUsage();
        }
    }
}

void ProcessCommandline(int argc, char * argv[])
{
    if (argc < 4 || strlen(argv[1]) != 1)
    {
        ShowUsage();
    }

    for (int i = 4; i < argc; ++i)
    {
        if (argv[i][0] == '-')
        {
            ProcessSwitch(&argv[i][1]);
        }
        else
        {
            ShowUsage();
        }
    }
}

int main(int argc, char * argv[])
{
    fprintf(stdout, "This is bsc, Block Sorting Compressor. Version 3.2.4. 18 January 2022.\n");
    fprintf(stdout, "Copyright (c) 2009-2021 Ilya Grebnov <Ilya.Grebnov@gmail.com>.\n\n");

#if defined(_OPENMP) && defined(__INTEL_COMPILER)

    kmp_set_warnings_off();

#endif

    ProcessCommandline(argc, argv);

    if (bsc_init(paramFeatures()) != LIBBSC_NO_ERROR)
    {
        fprintf(stderr, "\nInternal program error, please contact the author!\n");
        exit(2);
    }

    switch (*argv[1])
    {
#if !defined(BSC_DECOMPRESSION_ONLY)
        case 'e' : case 'E' : Compression(argv); break;
#endif
        case 'd' : case 'D' : Decompression(argv); break;
        default  : ShowUsage();
    }

    return 0;
}

/*-----------------------------------------------------------*/
/* End                                               bsc.cpp */
/*-----------------------------------------------------------*/

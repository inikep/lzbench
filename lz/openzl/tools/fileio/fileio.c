// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "tools/fileio/fileio.h"

#include <stddef.h> // size_t
#include <stdio.h>  // FILE, fopen, fread
#include <stdlib.h> // malloc

static void fseek_orDie(FILE* f, long int offset, int whence)
{
    int const status = fseek(f, offset, whence);
    ZL_REQUIRE_EQ(status, 0);
    (void)status;
}

static size_t FIO_getFilesize(FILE* f)
{
    fseek_orDie(f, 0, SEEK_END);
    long const fSize = ftell(f);
    ZL_REQUIRE_GT(fSize, 0);
    fseek_orDie(f, 0, SEEK_SET);
    return (size_t)fSize;
}

size_t FIO_sizeof_file(const char* fileName)
{
    FILE* const f = fopen(fileName, "rb");
    ZL_REQUIRE_NN(f);

    size_t const fileSize = FIO_getFilesize(f);
    ZL_REQUIRE_NE(fileSize, 0); // exclude empty and non-existing files

    fclose(f);
    return fileSize;
}

// read file, fill buffer
// if failure : @return = NULL
ZL_Buffer FIO_createBuffer_fromFilename(const char* filename)
{
    FILE* const f = fopen(filename, "rb");
    if (f == NULL) {
        fprintf(stderr, "error: could not open %s \n", filename);
        abort();
    }

    size_t const fileSize = FIO_getFilesize(f);
    ZL_REQUIRE_NE(fileSize, 0); // exclude empty and non-existing files

    ZL_Buffer buffer = ZL_B_create(fileSize);
    ZL_WC* wc        = ZL_B_getWC(&buffer);

    size_t const readSize = fread(ZL_WC_ptr(wc), 1, fileSize, f);
    ZL_REQUIRE_EQ(readSize, fileSize);
    ZL_WC_advance(wc, readSize);

    fclose(f);
    return buffer;
}

ZL_Buffer FIO_createBuffer_fromFilename_orDie(const char* filename)
{
    ZL_Buffer const buffer = FIO_createBuffer_fromFilename(filename);
    ZL_REQUIRE(!ZL_B_isNull(&buffer));
    return buffer;
}

void FIO_loadFile_intoBuffer(ZL_Buffer* buffer, const char* fileName)
{
    ZL_ASSERT_NN(buffer);
    FILE* const f = fopen(fileName, "rb");
    ZL_REQUIRE_NN(f);

    ZL_WC* wc = ZL_B_getWC(buffer);

    size_t const fileSize = FIO_getFilesize(f);
    ZL_REQUIRE_LE(fileSize, ZL_WC_avail(wc));

    size_t const readSize = fread(ZL_WC_ptr(wc), 1, fileSize, f);
    ZL_ASSERT_EQ(readSize, fileSize);
    ZL_WC_advance(wc, readSize);

    fclose(f);
}

char* FIO_createCompressedName(const char* fileName)
{
    size_t fileNameLen           = strlen(fileName);
    const char* outputFileSuffix = ".zs";
    char* outputFilename =
            malloc(strlen(fileName) + strlen(outputFileSuffix) + 1 /* '\0' */);
    ZL_REQUIRE_NN(outputFilename);
    memcpy(outputFilename, fileName, fileNameLen);
    memcpy(outputFilename + fileNameLen,
           outputFileSuffix,
           strlen(outputFileSuffix) + 1);
    return outputFilename;
}

void FIO_writeFile(ZL_ReadCursor src, const char* fileName)
{
    FILE* const f = fopen(fileName, "wb");
    ZL_REQUIRE_NN(f);

    size_t const size = ZL_RC_avail(&src);
    size_t written    = fwrite(ZL_RC_ptr(&src), 1, size, f);
    ZL_REQUIRE_EQ(written, size);

    ZL_REQUIRE_EQ(fclose(f), 0);
}

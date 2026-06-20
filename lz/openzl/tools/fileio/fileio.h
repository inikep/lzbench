// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TESTS_FILEIO_H
#define ZSTRONG_TESTS_FILEIO_H

#include <stddef.h> // size_t

#include "openzl/common/buffer.h" // ZL_Buffer
#include "openzl/common/cursor.h" // ZL_ReadCursor
#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

// read file, fill buffer, always succeeds or abort
ZL_Buffer FIO_createBuffer_fromFilename_orDie(const char* filename);

// read file, fill buffer, may fail
// if failure : @return.ptr == NULL
ZL_Buffer FIO_createBuffer_fromFilename(const char* filename);

size_t FIO_sizeof_file(const char* fileName);

void FIO_loadFile_intoBuffer(ZL_Buffer* buffer, const char* fileName);

// allocates and fills a buffer with filename + ".zs"
char* FIO_createCompressedName(const char* fileName);

void FIO_writeFile(ZL_ReadCursor src, const char* fileName);

ZL_END_C_DECLS

#endif // ZSTRONG_TESTS_FILEIO_H

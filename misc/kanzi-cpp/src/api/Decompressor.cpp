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

#include "Decompressor.hpp"
#include "../types.hpp"
#include "../Error.hpp"
#include "../io/CompressedInputStream.hpp"
#include "../transform/TransformFactory.hpp"
#include "../entropy/EntropyEncoderFactory.hpp"


#ifdef _MSC_VER
   #include <io.h>
   #define FILENO(f) _fileno(f)
   #define READ(fd, buf, n) _read(fd, buf, uint(n))
   #define WRITE(fd, buf, n) _write(fd, buf, uint(n))
#else
   #include <unistd.h>
   #define FILENO(f) fileno(f)
   #define READ(fd, buf, n) read(fd, buf, n)
   #define WRITE(fd, buf, n) write(fd, buf, n)
#endif

using namespace std;
using namespace kanzi;

namespace kanzi {
   class ifstreambuf : public streambuf {
     public:
       ifstreambuf(int fd) : _fd(fd) {
           setg(&_buffer[4], &_buffer[4], &_buffer[4]);
       }

     private:
       static const int BUF_SIZE = 1024 + 4;
       int _fd;
       char _buffer[BUF_SIZE];

       virtual int_type underflow() {
           if (gptr() < egptr())
               return traits_type::to_int_type(*gptr());

           const int pbSz = min(int(gptr() - eback()), BUF_SIZE - 4);
           memmove(&_buffer[BUF_SIZE - pbSz], gptr() - pbSz, pbSz);
           const int r = int(READ(_fd, &_buffer[4], BUF_SIZE - 4));

           if (r <= 0)
               return EOF;

           setg(&_buffer[4 - pbSz], &_buffer[4], &_buffer[4 + r]);
           return traits_type::to_int_type(*gptr());
       }
    };

    class FileInputStream FINAL : public istream
    {
       private:
          ifstreambuf _buf;

       public:
          FileInputStream(int fd) : istream(nullptr), _buf(fd) {
             rdbuf(&_buf);
          }
    };
}


// Create internal dContext and CompressedInputStream
int CDECL initDecompressor(struct dData* pData, FILE* src, struct dContext** pCtx)
{
    if ((pData == nullptr) || (pCtx == nullptr) || (src == nullptr))
        return Error::ERR_INVALID_PARAM;

    if (pData->bufferSize > uint(2) * 1024 * 1024 * 1024) // max buffer size
        return Error::ERR_INVALID_PARAM;

    dContext* dctx = nullptr;
    FileInputStream* fis = nullptr;

    try {
        const int fd = FILENO(src);

        if (fd == -1)
           return Error::ERR_CREATE_DECOMPRESSOR;

        // Create decompression stream and update context
        *pCtx = nullptr;
        fis = new FileInputStream(fd);
        dctx = new dContext();

        if (pData->headerless != 0) {
           // Headerless mode: process params
           string transform = TransformFactory<byte>::getName(TransformFactory<byte>::getType(pData->transform));

           if (transform.length() >= 63) {
               delete fis;
               delete dctx;
               return Error::ERR_INVALID_PARAM;
           }

           strncpy(pData->transform, transform.data(), transform.length());
           pData->transform[transform.length() + 1] = 0;
           string entropy = EntropyEncoderFactory::getName(EntropyEncoderFactory::getType(pData->entropy));

           if (entropy.length() >= 15) {
               delete fis;
               delete dctx;
               return Error::ERR_INVALID_PARAM;
           }

           strncpy(pData->entropy, entropy.data(), entropy.length());
           pData->entropy[entropy.length() + 1] = 0;
           pData->blockSize = (pData->blockSize + 15) & -16;
           bool checksum = pData->checksum == 0 ? false : true;

#ifdef CONCURRENCY_ENABLED
   #if __cplusplus >= 201103L
           dctx->pCis = new CompressedInputStream(*fis, pData->jobs, nullptr, true,
               checksum, pData->blockSize, std::move(transform), std::move(entropy), pData->originalSize, pData->bsVersion);
   #else
           dctx->pCis = new CompressedInputStream(*fis, pData->jobs, nullptr, true,
               checksum, pData->blockSize, transform, entropy, pData->originalSize, pData->bsVersion);
   #endif
#else
           dctx->pCis = new CompressedInputStream(*fis, pData->jobs, true,
               checksum, pData->blockSize, transform, entropy, pData->originalSize, pData->bsVersion);
#endif
        }
        else {
           dctx->pCis = new CompressedInputStream(*fis, pData->jobs);
        }

        dctx->bufferSize = pData->bufferSize;
        dctx->fis = fis;
        *pCtx = dctx;
    }
    catch (exception&) {
        if (fis != nullptr)
           delete fis;

        if (dctx != nullptr)
           delete dctx;

        return Error::ERR_CREATE_DECOMPRESSOR;
    }

    return 0;
}

int CDECL decompress(struct dContext* pCtx, BYTE* dst, int* inSize, int* outSize)
{
    *inSize = 0;
    int res = 0;

    if ((pCtx == nullptr) || (*outSize > int(pCtx->bufferSize))) {
        *outSize = 0;
        return Error::ERR_INVALID_PARAM;
    }

    CompressedInputStream* pCis = (CompressedInputStream*)pCtx->pCis;
    *outSize = 0;

    if (pCis == nullptr)
        return Error::ERR_INVALID_PARAM;

    try {
        const uint64 r = int(pCis->getRead());
        pCis->read((char*)dst, streamsize(*outSize));
        res = pCis->good() ? 0 : Error::ERR_READ_FILE;
        *inSize = int(pCis->getRead() - r);
        *outSize = int(pCis->gcount());
    }
    catch (exception&) {
        return Error::ERR_UNKNOWN;
    }

    return res;
}

// Cleanup allocated internal data structures
int CDECL disposeDecompressor(struct dContext* pCtx)
{
    if (pCtx == nullptr)
        return Error::ERR_INVALID_PARAM;

    CompressedInputStream* pCis = (CompressedInputStream*)pCtx->pCis;

    try {
        if (pCis != nullptr) {
            pCis->close();
        }
    }
    catch (exception&) {
        return Error::ERR_UNKNOWN;
    }

    try {
        if (pCis != nullptr)
            delete pCis;

        if (pCtx->fis != nullptr)
            delete (FileInputStream*)pCtx->fis;

        pCtx->fis = nullptr;
        delete pCtx;
    }
    catch (exception&) {
        if (pCtx->fis != nullptr)
            delete (FileInputStream*)pCtx->fis;

        delete pCtx;
        return Error::ERR_UNKNOWN;
    }

    return 0;
}

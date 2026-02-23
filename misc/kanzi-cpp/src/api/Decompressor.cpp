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

/**
 *  Decompression context: encapsulates decompressor state (opaque: could change in future versions)
 */
struct dContext {
    kanzi::CompressedInputStream* pCis;
    size_t bufferSize;
    void* fis;
};


namespace kanzi {

   class ifstreambuf FINAL : public streambuf {
     public:
       ifstreambuf(int fd) : _fd(fd) {
          // gptr() = egptr() initially forces underflow() on first read
          setg(_buffer + 4, _buffer + 4, _buffer + 4);
       }

     private:
       static const int BUF_SIZE = 1024 + 4;
       int _fd;
       char _buffer[BUF_SIZE];

       virtual int_type underflow() {
           if (gptr() < egptr())
               return traits_type::to_int_type(*gptr());

           // Preserve up to 4 characters for putback
           int putback = int(gptr() - eback());

           if (putback > 4) putback = 4;

           // Only move putback if > 0
           if (putback > 0) {
               std::memmove(_buffer + (4 - putback), gptr() - putback, putback);
           }

           // Read new data
           const int n = int(READ(_fd, _buffer + 4, BUF_SIZE - 4));

           if (n <= 0)
               return EOF;

           // Reset get pointers:
           //   eback = start of buffer (including putback area)
           //   gptr  = first new kanzi::byte
           //   egptr = end of new data
           setg(_buffer + (4 - putback), _buffer + 4, _buffer + 4 + n);
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
KANZI_API int CDECL initDecompressor(struct dData* pData, FILE* src, struct dContext** pCtx) KANZI_NOEXCEPT
{
    if ((pData == nullptr) || (pCtx == nullptr) || (src == nullptr))
        return Error::ERR_INVALID_PARAM;

    // Validate buffer size (sanity check against huge allocations, e.g., > 2GB)
    if (pData->bufferSize > size_t(2) * 1024 * 1024 * 1024)
        return Error::ERR_INVALID_PARAM;

    dContext* dctx = nullptr;
    FileInputStream* fis = nullptr;

    try {
        const int fd = FILENO(src);

        if (fd == -1)
           return Error::ERR_CREATE_DECOMPRESSOR;

        // Create decompression stream and context
        *pCtx = nullptr;
        fis = new FileInputStream(fd);
        dctx = new dContext();
        dctx->pCis = nullptr;
        dctx->fis = nullptr;

        if (pData->headerless != 0) {
           // Headerless mode: process params
           string transform = TransformFactory<kanzi::byte>::getName(TransformFactory<kanzi::byte>::getType(pData->transform));
           string entropy = EntropyEncoderFactory::getName(EntropyEncoderFactory::getType(pData->entropy));

           // Validate sizes
           if ((transform.length() >= sizeof(pData->transform)) ||
               (entropy.length() >= sizeof(pData->entropy))) {
               delete fis;
               delete dctx;
               return Error::ERR_INVALID_PARAM;
           }

           memset(pData->transform, 0, sizeof(pData->transform));
           strncpy(pData->transform, transform.c_str(), sizeof(pData->transform) - 1);
           memset(pData->entropy, 0, sizeof(pData->entropy));
           strncpy(pData->entropy, entropy.c_str(), sizeof(pData->entropy) - 1);

           pData->blockSize = (pData->blockSize + 15) & -16;

           dctx->pCis = new CompressedInputStream(*fis, pData->jobs,
                                                  pData->entropy, pData->transform,
                                                  pData->blockSize, pData->checksum,
                                                  pData->originalSize,
#ifdef CONCURRENCY_ENABLED
                                                  nullptr,
#endif
                                                  true, pData->bsVersion);
        }
        else {
           dctx->pCis = new CompressedInputStream(*fis, pData->jobs);
        }

        dctx->bufferSize = pData->bufferSize;
        dctx->fis = fis;
        *pCtx = dctx;
    }
    catch (const exception&) {
        if (dctx != nullptr) {
            // pCis is managed by dctx, but might not be assigned yet
            if (dctx->pCis)
               delete dctx->pCis;

            delete dctx;
        }

        // fis is usually owned by pCis, but if pCis wasn't created, we delete it
        if (fis != nullptr && (dctx == nullptr || dctx->pCis == nullptr))
           delete fis;

        return Error::ERR_CREATE_DECOMPRESSOR;
    }

    return 0;
}


KANZI_API int CDECL decompress(struct dContext* pCtx, unsigned char* dst,
                               size_t* inSize, size_t* outSize) KANZI_NOEXCEPT
{
    if ((pCtx == nullptr) || (outSize == nullptr)) {
        return Error::ERR_INVALID_PARAM;
    }

    if (*outSize > pCtx->bufferSize) {
         return Error::ERR_INVALID_PARAM;
    }

    if (*outSize == 0)
        return 0;

    if (dst == nullptr) {
        return Error::ERR_INVALID_PARAM;
    }

    if (inSize)
        *inSize = 0;

    CompressedInputStream* pCis = pCtx->pCis;

    if (pCis == nullptr) {
        *outSize = 0;
        return Error::ERR_INVALID_PARAM;
    }

    try {
        const uint64 r = pCis->getRead();
        pCis->read((char*)dst, std::streamsize(*outSize));

        if (!pCis->good() && !pCis->eof())
            return Error::ERR_READ_FILE;

        if (inSize)
            *inSize = size_t(pCis->getRead() - r);

        *outSize = size_t(pCis->gcount());
    }
    catch (const exception&) {
        *outSize = 0;
        return Error::ERR_UNKNOWN;
    }

    return 0;
}

// Cleanup allocated internal data structures
KANZI_API int CDECL disposeDecompressor(struct dContext** ppCtx) KANZI_NOEXCEPT
{
    if ((ppCtx == nullptr) || (*ppCtx == nullptr))
        return Error::ERR_INVALID_PARAM;

    dContext* pCtx = *ppCtx;
    CompressedInputStream* pCis = static_cast<CompressedInputStream*>(pCtx->pCis);

    try {
        if (pCis != nullptr) {
            pCis->close();
            delete pCis;
            pCis = nullptr;
        }

        if (pCtx->fis != nullptr)
            delete (FileInputStream*)pCtx->fis;

        pCtx->fis = nullptr;
        delete pCtx;
        *ppCtx = nullptr;
    }
    catch (const exception&) {
        if (pCis != nullptr)
            delete pCis;

        if (pCtx->fis != nullptr)
            delete (FileInputStream*)pCtx->fis;

        delete pCtx;
        *ppCtx = nullptr;
        return Error::ERR_UNKNOWN;
    }

    return 0;
}

KANZI_API unsigned int CDECL getDecompressorVersion(void) KANZI_NOEXCEPT
{
    return  (KANZI_DECOMP_VERSION_MAJOR << 16) |
            (KANZI_DECOMP_VERSION_MINOR << 8)  |
             KANZI_DECOMP_VERSION_PATCH;
}

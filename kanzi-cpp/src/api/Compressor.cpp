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

#include "Compressor.hpp"
#include "../types.hpp"
#include "../Error.hpp"
#include "../io/CompressedOutputStream.hpp"
#include "../transform/TransformFactory.hpp"
#include "../entropy/EntropyEncoderFactory.hpp"
#include <sys/stat.h>


#ifdef _MSC_VER
   #define FSTAT _fstat64
   #define STAT _stat64
#else
   #if defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__) || defined(__APPLE__) || defined(__MINGW32__)
      #define FSTAT fstat
      #define STAT stat
   #else
      #define FSTAT fstat64
      #define STAT stat64
   #endif
#endif



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
   // Utility classes to map C FILEs to C++ streams
   class ofstreambuf : public streambuf
   {
     public:
        ofstreambuf (int fd) : _fd(fd) { }

     private:
        int _fd; 

        virtual int_type overflow(int_type c) {
            if (c == EOF)
               return EOF;

            char d = char(c);
            return (WRITE(_fd, &d, 1) == 1) ? c : EOF;
        }

        virtual streamsize xsputn(const char* s, streamsize sz) {
            return WRITE(_fd, s, sz);
        }
    };

    class FileOutputStream FINAL : public ostream
    {
       private:
          ofstreambuf _buf;

       public:
          FileOutputStream(int fd) : ostream(nullptr), _buf(fd) {
              rdbuf(&_buf);
          }
    };
}


// Create internal cContext and CompressedOutputStream
int CDECL initCompressor(struct cData* pData, FILE* dst, struct cContext** pCtx)
{
    if ((pData == nullptr) || (pCtx == nullptr) || (dst == nullptr))
        return Error::ERR_INVALID_PARAM;

    FileOutputStream* fos = nullptr;
    cContext* cctx = nullptr;

    try {
        // Process params
        const int fd = FILENO(dst);

        if (fd == -1)
           return Error::ERR_CREATE_COMPRESSOR;

        string transform = TransformFactory<byte>::getName(TransformFactory<byte>::getType(pData->transform));
        
        if (transform.length() >= 63)
            return Error::ERR_INVALID_PARAM;

        strncpy(pData->transform, transform.data(), transform.length());
        pData->transform[transform.length() + 1] = 0;
        string entropy = EntropyEncoderFactory::getName(EntropyEncoderFactory::getType(pData->entropy));

        if (entropy.length() >= 15)
            return Error::ERR_INVALID_PARAM;
        
        strncpy(pData->entropy, entropy.data(), entropy.length());
        pData->entropy[entropy.length() + 1] = 0;
        pData->blockSize = (pData->blockSize + 15) & -16;

        *pCtx = nullptr;
        uint64 fileSize = 0;
        struct STAT sbuf;

        if (FSTAT(fd, &sbuf) == 0) {
           fileSize = uint64(sbuf.st_size);
        }

        // Create compression stream and update context
        fos = new FileOutputStream(fd);
        cctx = new cContext();
        bool checksum = pData->checksum == 0 ? false : true;
        bool headerless = pData->headerless == 0 ? false : true;

#ifdef CONCURRENCY_ENABLED
        cctx->pCos = new CompressedOutputStream(*fos, pData->entropy, pData->transform, pData->blockSize, checksum, pData->jobs, fileSize, nullptr, headerless);
#else
        cctx->pCos = new CompressedOutputStream(*fos, pData->entropy, pData->transform, pData->blockSize, checksum, pData->jobs, fileSize, headerless);
#endif

        cctx->blockSize = pData->blockSize;
        cctx->fos = fos;
        *pCtx = cctx;
    }
    catch (exception&) {
        if (fos != nullptr)
           delete fos;

        if (cctx != nullptr)
           delete cctx;

        return Error::ERR_CREATE_COMPRESSOR;
    }

    return 0;
}

int CDECL compress(struct cContext* pCtx, const BYTE* src, int* inSize, int* outSize)
{
    *outSize = 0;
    int res = 0;

    if ((pCtx == nullptr) || (*inSize > int(pCtx->blockSize))) {
        *inSize = 0;
        return Error::ERR_INVALID_PARAM;
    }

    CompressedOutputStream* pCos = (CompressedOutputStream*)pCtx->pCos;
    *inSize = 0;

    if (pCos == nullptr)
        return Error::ERR_INVALID_PARAM;

    try {
        const uint64 w = pCos->getWritten();
        pCos->write((const char*)src, streamsize(*inSize));
        res = pCos->good() ? 0 : Error::ERR_WRITE_FILE;
        *outSize = int(pCos->getWritten() - w);
    }
    catch (exception&) {
        return Error::ERR_UNKNOWN;
    }

    return res;
}

// Cleanup allocated internal data structures
int CDECL disposeCompressor(struct cContext* pCtx, int* outSize)
{
    *outSize = 0;

    if (pCtx == nullptr)
        return Error::ERR_INVALID_PARAM;

    CompressedOutputStream* pCos = (CompressedOutputStream*)pCtx->pCos;

    try {
        if (pCos != nullptr) {
            const uint64 w = pCos->getWritten();
            pCos->close();
            *outSize = int(pCos->getWritten() - w);
        }
    }
    catch (exception&) {
        return Error::ERR_UNKNOWN;
    }

    try {
        if (pCos != nullptr)
            delete pCos;

        if (pCtx->fos != nullptr)
            delete (FileOutputStream*)pCtx->fos;

        pCtx->fos = nullptr;
        delete pCtx;
    }
    catch (exception&) {
        if (pCtx->fos != nullptr)
            delete (FileOutputStream*)pCtx->fos;

        delete pCtx;
        return Error::ERR_UNKNOWN;
    }

    return 0;
}

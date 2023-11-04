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

    try {
        // Process params
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
        const int fd = FILENO(dst);

        if (fd == -1)
           return Error::ERR_CREATE_COMPRESSOR;

        // Create compression stream and update context
        FileOutputStream* fos = new FileOutputStream(fd);
        cContext* cctx = new cContext();
        cctx->pCos = new CompressedOutputStream(*fos, pData->entropy, pData->transform, pData->blockSize, bool(pData->checksum & 1), pData->jobs);
        cctx->blockSize = pData->blockSize;
        cctx->fos = fos;
        *pCtx = cctx;
    }
    catch (exception&) {
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

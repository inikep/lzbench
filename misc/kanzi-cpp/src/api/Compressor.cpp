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

#include <sys/stat.h>
#include "Compressor.hpp"
#include "../types.hpp"
#include "../Error.hpp"
#include "../io/IOException.hpp"
#include "../io/CompressedOutputStream.hpp"
#include "../transform/TransformFactory.hpp"
#include "../entropy/EntropyEncoderFactory.hpp"


// Note stat64/lstat64 are deprecated on MacOS/Linux
// Use _FILE_OFFSET_BITS and stat/lstat instead

#ifdef _WIN32
   #define FSTAT _fstat64
   #define STAT _stat64
#else
   #define _FILE_OFFSET_BITS 64
   #define FSTAT fstat
   #define STAT stat
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


struct cContext {
   kanzi::CompressedOutputStream* pCos;
   size_t blockSize;
   void* fos;
};


namespace kanzi {
   // Utility classes to map C FILEs to C++ streams
    class ofstreambuf FINAL : public streambuf
    {
    public:
        ofstreambuf(int fd) : _fd(fd), _buffer(65536) {
            // Initialize put pointers to the beginning of the buffer
            setp(&_buffer[0], &_buffer[0] + _buffer.size());
        }

        virtual ~ofstreambuf() {
            // Call the non-virtual implementation directly instead of the virtual sync()
            flush();
        }

    protected:
        // Called when the buffer is full
        virtual int_type overflow(int_type c) {
            if (flush() == EOF) return EOF;

            if (c != EOF) {
                *pptr() = char(c);
                pbump(1);
            }
            return c;
        }

        // Called for explicit sync/flush
        virtual int sync() {
            return (flush() == EOF) ? -1 : 0;
        }

        // Optimized block write
        virtual streamsize xsputn(const char* s, streamsize n) {
            streamsize remaining = n;
            const char* src = s;

            while (remaining > 0) {
                streamsize avail = epptr() - pptr();

                if (avail >= remaining) {
                    // Fits in current buffer
                    memcpy(pptr(), src, remaining);
                    pbump(int(remaining));
                    return n;
                }

                if (avail > 0) {
                    // Fill the rest of the buffer
                    memcpy(pptr(), src, avail);
                    pbump(int(avail));
                    src += avail;
                    remaining -= avail;
                }

                // Flush full buffer
                if (flush() == EOF) return n - remaining;

                // If the remaining chunk is large, write directly to FD to avoid double copy
                if (remaining >= streamsize(_buffer.size())) {
                    streamsize toWrite = remaining;

                    while (toWrite > 0) {
                        const ptrdiff_t written = ptrdiff_t(WRITE(_fd, src, toWrite));

                        if (written <= 0)
                            return n - remaining; // Error

                        src += written;
                        toWrite -= streamsize(written);
                    }

                    remaining = 0;
                }
            }

            return n;
        }

    private:
        int _fd;
        std::vector<char> _buffer;

        int flush() {
            ptrdiff_t n = pptr() - pbase();
            if (n > 0) {
                char* dst = pbase();
                ptrdiff_t remaining = n;

                while (remaining > 0) {
                    const ptrdiff_t written = ptrdiff_t(WRITE(_fd, dst, remaining));

                    if (written <= 0)
                        return EOF;

                    dst += written;
                    remaining -= written;
                }

                pbump(-int(n)); // Reset pbump by subtracting the amount written
            }

            return 0;
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
KANZI_API int CDECL initCompressor(struct cData* pData, FILE* dst, struct cContext** pCtx) KANZI_NOEXCEPT
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

        string transform = TransformFactory<kanzi::byte>::getName(TransformFactory<kanzi::byte>::getType(pData->transform));
        string entropy = EntropyEncoderFactory::getName(EntropyEncoderFactory::getType(pData->entropy));

        if ((transform.length() >= sizeof(pData->transform)) ||
            (entropy.length() >= sizeof(pData->entropy))) {
            return Error::ERR_INVALID_PARAM;
        }

        memset(pData->transform, 0, sizeof(pData->transform));
        strncpy(pData->transform, transform.c_str(), sizeof(pData->transform) - 1);
        memset(pData->entropy, 0, sizeof(pData->entropy));
        strncpy(pData->entropy, entropy.c_str(), sizeof(pData->entropy) - 1);

        pData->blockSize = (pData->blockSize + 15) & -16;

        *pCtx = nullptr;
        size_t fileSize = 0;
        struct STAT sbuf;

        if (FSTAT(fd, &sbuf) == 0) {
           fileSize = size_t(sbuf.st_size);
        }

        // Create compression stream and update context
        fos = new FileOutputStream(fd);
        cctx = new cContext();

        cctx->pCos = new CompressedOutputStream(*fos, pData->jobs,
                                                pData->entropy, pData->transform,
                                                int(pData->blockSize), pData->checksum,
                                                uint64(fileSize),
#ifdef CONCURRENCY_ENABLED
                                                nullptr,
#endif
                                                pData->headerless != 0);

        cctx->blockSize = pData->blockSize;
        cctx->fos = fos;
        *pCtx = cctx;
    }
    catch (const exception&) {
        if (fos != nullptr)
           delete fos;

        if (cctx != nullptr)
           delete cctx;

        return Error::ERR_CREATE_COMPRESSOR;
    }

    return 0;
}

KANZI_API int CDECL compress(struct cContext* pCtx, const unsigned char* src, size_t inSize, size_t* outSize) KANZI_NOEXCEPT
{
    if ((pCtx == nullptr) || (outSize == nullptr)) {
        return Error::ERR_INVALID_PARAM;
    }

    if ((src == nullptr) && (inSize != 0)) {
        return Error::ERR_INVALID_PARAM;
    }

    if (inSize > size_t(pCtx->blockSize)) {
        return Error::ERR_INVALID_PARAM;
    }

    *outSize = 0;
    int res = 0;
    CompressedOutputStream* pCos = pCtx->pCos;

    if (pCos == nullptr) {
        return Error::ERR_INVALID_PARAM;
    }

    try {
        const uint64 w = pCos->getWritten();
        pCos->write((const char*)src, streamsize(inSize));
        res = pCos->good() ? 0 : Error::ERR_WRITE_FILE;
        *outSize = int(pCos->getWritten() - w);
    }
    catch (const IOException& ioe) {
        return ioe.error();
    }
    catch (const exception&) {
        return Error::ERR_UNKNOWN;
    }

    return res;
}

// Cleanup allocated internal data structures
KANZI_API int CDECL disposeCompressor(struct cContext** ppCtx, size_t* outSize) KANZI_NOEXCEPT
{
    if ((ppCtx == nullptr) || (*ppCtx == nullptr) || (outSize == nullptr))
        return Error::ERR_INVALID_PARAM;

    *outSize = 0;
    cContext* pCtx = *ppCtx;
    CompressedOutputStream* pCos = pCtx->pCos;

    try {
        if (pCos != nullptr) {
            const uint64 w = pCos->getWritten();
            pCos->close();
            *outSize = int(pCos->getWritten() - w);
            delete pCos;
            pCos = nullptr;
            pCtx->pCos = nullptr;
        }

        if (pCtx->fos != nullptr)
            delete static_cast<FileOutputStream*>(pCtx->fos);

        pCtx->fos = nullptr;
        delete pCtx;
        *ppCtx = nullptr;
    }
    catch (const IOException& ioe) {
        if (pCos != nullptr) {
            delete pCos;
            pCos = nullptr;
            pCtx->pCos = nullptr;
        }

        if (pCtx->fos != nullptr)
            delete static_cast<FileOutputStream*>(pCtx->fos);

        delete pCtx;
        *ppCtx = nullptr;
        return ioe.error();
    }
    catch (const exception&) {
        if (pCos != nullptr) {
            delete pCos;
            pCos = nullptr;
            pCtx->pCos = nullptr;
        }

        if (pCtx->fos != nullptr)
            delete static_cast<FileOutputStream*>(pCtx->fos);

        delete pCtx;
        *ppCtx = nullptr;
        return Error::ERR_UNKNOWN;
    }

    return 0;
}

KANZI_API unsigned int CDECL getCompressorVersion(void) KANZI_NOEXCEPT
{
    return  (KANZI_COMP_VERSION_MAJOR << 16) |
            (KANZI_COMP_VERSION_MINOR << 8)  |
             KANZI_COMP_VERSION_PATCH;
}

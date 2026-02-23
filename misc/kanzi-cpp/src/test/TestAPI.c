#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../api/Compressor.hpp"
#include "../api/Decompressor.hpp"


// Test the API (pure C code)

#define ASSERT(cond, msg)                                       \
    do {                                                        \
        if (!(cond)) {                                          \
            fprintf(stderr, "ASSERT FAILED: %s (%s:%d)\n",      \
                    msg, __FILE__, __LINE__);                   \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)


#ifndef PORTABLE_FMEMOPEN_H
#define PORTABLE_FMEMOPEN_H

#if defined(_WIN32)

// Windows has no fmemopen
static inline FILE* portable_fmemopen(void* buf, size_t size, const char* mode)
{
    FILE* f = tmpfile();

    if (!f)
        return NULL;

    if (buf && size > 0 && strchr(mode, 'w') == NULL) {
        fwrite(buf, 1, size, f);
        rewind(f);
    }

    return f;
}

#else

// macOS does not have fmemopen, but has funopen
// BSD may not have fmemopen
// Most Linux distros do have fmemopen
static inline FILE* portable_fmemopen(void* buf, size_t size, const char* mode)
{
    (void)mode; // Only "wb" or "rb" used in tests

    // Simple implementation: use a temporary file and preload the buffer
    FILE* f = tmpfile();

    if (!f)
        return NULL;

    if (buf && size > 0 && strchr(mode, 'w') == NULL) {
        // reading: preload buffer
        fwrite(buf, 1, size, f);
        rewind(f);
    }

    return f;
}

#endif

#endif


// Helper functions
static struct cData make_params(void)
{
    struct cData p;
    memset(&p, 0, sizeof(p));

    strcpy(p.transform, "LZP");
    strcpy(p.entropy, "FPAQ");
    p.blockSize = 1024;
    p.jobs = 1;
    p.checksum = 0;
    p.headerless = 0;

    return p;
}

static void fill_buffer(uint8_t* buf, int size)
{
    for (int i = 0; i < size; i++)
        buf[i] = (uint8_t)(i * 17 + 3);
}


// initCompressor invalid parameters
static void test_init_invalid(void)
{
    printf("TEST: initCompressor invalid params...\n");

    struct cContext* ctx;
    struct cData p = make_params();
    FILE* f = portable_fmemopen(NULL, 1000, "wb");

    ASSERT(f != NULL, "fmemopen failed");

    int rc;

    rc = initCompressor(NULL, f, &ctx);
    ASSERT(rc != 0, "init should fail on NULL params");

    rc = initCompressor(&p, NULL, &ctx);
    ASSERT(rc != 0, "init should fail on NULL FILE");

    rc = initCompressor(&p, f, NULL);
    ASSERT(rc != 0, "init should fail on NULL ctx");

    fclose(f);
}


// Basic initialization / disposal
static void test_init_dispose(void)
{
    printf("TEST: init + dispose...\n");

    struct cData p = make_params();
    struct cContext* ctx = NULL;

    FILE* f = portable_fmemopen(NULL, 4096, "wb");
    ASSERT(f != NULL, "fmemopen failed");

    ASSERT(initCompressor(&p, f, &ctx) == 0, "init failed");
    ASSERT(ctx != NULL, "ctx should not be NULL");

    size_t out = (size_t) -1;
    ASSERT(disposeCompressor(&ctx, &out) == 0, "dispose failed");

    fclose(f);
}


// Compress small block
static void test_compress_small(void)
{
    printf("TEST: small compression...\n");

    struct cData p = make_params();
    struct cContext* ctx = NULL;

    FILE* f = portable_fmemopen(NULL, 4096, "wb");
    ASSERT(f != NULL, "fmemopen failed");

    ASSERT(initCompressor(&p, f, &ctx) == 0, "init failed");

    uint8_t data[256];
    fill_buffer(data, 256);

    size_t inSize = 256;
    size_t outSize = 0;

    ASSERT(compress(ctx, data, inSize, &outSize) == 0, "compress failed");

    size_t flushed = 0;
    ASSERT(disposeCompressor(&ctx, &flushed) == 0, "dispose failed");

    fclose(f);
}


// Oversized block
static void test_compress_too_big(void)
{
    printf("TEST: oversized block handling...\n");

    struct cData p = make_params();
    p.blockSize = 1024;

    struct cContext* ctx = NULL;

    FILE* f = portable_fmemopen(NULL, 4096, "wb");
    ASSERT(f != NULL, "fmemopen failed");

    ASSERT(initCompressor(&p, f, &ctx) == 0, "init failed");

    uint8_t big[4096];
    fill_buffer(big, 4096);

    size_t inSize = sizeof(big);
    size_t outSize = 0;

    ASSERT(compress(ctx, big, inSize, &outSize) != 0, "compress should fail on oversized input");
    ASSERT(outSize == 0, "output size must be zero on error");

    size_t flushed = 0;
    ASSERT(disposeCompressor(&ctx, &flushed) == 0, "dispose failed");

    fclose(f);
}


// Two sequential blocks
static void test_compress_two_blocks(void)
{
    printf("TEST: two-block compression...\n");

    struct cData p = make_params();
    p.blockSize = 1024;

    struct cContext* ctx = NULL;

    FILE* f = portable_fmemopen(NULL, 10000, "wb");
    ASSERT(f != NULL, "fmemopen failed");

    ASSERT(initCompressor(&p, f, &ctx) == 0, "init failed");

    uint8_t a[300], b[500];
    fill_buffer(a, 300);
    fill_buffer(b, 500);

    size_t inSize, outSize;

    inSize = 300;
    outSize = 0;
    ASSERT(compress(ctx, a, inSize, &outSize) == 0, "block 1 failed");
    ASSERT(outSize <= 0, "block 1 written bytes invalid");

    inSize = 500;
    outSize = 0;
    ASSERT(compress(ctx, b, inSize, &outSize) == 0, "block 2 failed");

    size_t flushed = 0;
    ASSERT(disposeCompressor(&ctx, &flushed) == 0, "dispose failed");

    fclose(f);
}


// Utilities
static void write_file(const char* path, const unsigned char* data, size_t len)
{
    FILE* f = fopen(path, "wb");
    ASSERT(f != NULL, "failed to write file");
    fwrite(data, 1, len, f);
    fclose(f);
}

// Simple decompression of known data
static void test_basic_decompression(void)
{
    printf("TEST: basic decompression...\n");

    const char* input  = "Hello Kanzi! Hello Compression!";
    const size_t in_len = strlen(input);

    // Step 1: Compress to temporary file
    const char* f_name = "tmp_comp.bin";
    FILE* fcomp = fopen(f_name, "wb");
    ASSERT(fcomp != NULL, "failed to open file");

    struct cData cparams;
    memset(&cparams, 0, sizeof(cparams));

    strcpy(cparams.transform, "LZ");
    strcpy(cparams.entropy,   "ANS0");
    cparams.blockSize  = 1 << 16;
    cparams.jobs       = 1;
    cparams.checksum   = 32;
    cparams.headerless = 0;

    struct cContext* cctx = NULL;
    ASSERT(initCompressor(&cparams, fcomp, &cctx) == 0, "failed to init compressor");

    size_t inSize  = in_len;
    size_t outSize = 0;
    ASSERT(compress(cctx, (const unsigned char*)input, inSize, &outSize) == 0, "failed to compress data");

    size_t flushed = 0;
    ASSERT(disposeCompressor(&cctx, &flushed) == 0, "failed to dispose compressor");

    fclose(fcomp);

    // Step 2: Decompress from temporary file
    FILE* fdec = fopen(f_name, "rb");
    ASSERT(fdec != NULL, "failed to open file for reading");

    struct dData dparams;
    memset(&dparams, 0, sizeof(dparams));

    dparams.bufferSize = 1 << 16;
    dparams.jobs       = 1;
    dparams.headerless = 0;

    struct dContext* dctx = NULL;
    ASSERT(initDecompressor(&dparams, fdec, &dctx) == 0, "failed to init decompressor");

    unsigned char outbuf[1024];
    size_t readComp = 0;
    size_t produced = sizeof(outbuf);

    ASSERT(decompress(dctx, outbuf, &readComp, &produced) == 0, "failed to decompress data");

    // Step 3: Validate output
    ASSERT(produced == in_len, "failed to decompress data: invalid data size");
    ASSERT(memcmp(outbuf, input, in_len) == 0, "failed to decompress data: data differ from original");

    ASSERT(disposeDecompressor(&dctx) == 0, "failed to dispose decompressor");

    fclose(fdec);
    remove(f_name);
}

// Decompress much larger data (multi-block)
static void test_large_multi_block(void)
{
    printf("TEST: large multi blocks\n");

    size_t size = 2 * 1024 * 1024; // 2MB
    unsigned char* data = (unsigned char*)malloc(size);
    ASSERT(data != NULL, "failed to allocate buffer memory");

    for (size_t i = 0; i < size; i++)
        data[i] = (unsigned char)(i * 7);

    const char* f_name = "tmp_large_input.bin";
    write_file(f_name, data, size);

    // Compress
    const char* fcomp_name = "tmp_large_comp.bin";
    FILE* fcomp = fopen(fcomp_name, "wb");
    ASSERT(fcomp, "failed to open file for writing");

    struct cData cparams;
    memset(&cparams, 0, sizeof(cparams));

    strcpy(cparams.transform, "LZ");
    strcpy(cparams.entropy,   "FPAQ");
    cparams.blockSize  = 256 * 1024;
    cparams.jobs       = 1;
    cparams.checksum   = 64;
    cparams.headerless = 0;

    struct cContext* cctx = NULL;
    ASSERT(initCompressor(&cparams, fcomp, &cctx) == 0, "failed to init compressor");

    size_t remaining = size;
    unsigned char* p = data;

    while (remaining > 0) {
        size_t chunk = ((remaining > cparams.blockSize) ?
                         cparams.blockSize : remaining);

        size_t inSize  = chunk;
        size_t outSize = 0;
        ASSERT(compress(cctx, p, inSize, &outSize) == 0, "failed to compress data");

        p         += chunk;
        remaining -= chunk;
    }

    size_t flushed = 0;
    ASSERT(disposeCompressor(&cctx, &flushed) == 0, "failed to dispose compressor");

    fclose(fcomp);

    // Decompress
    FILE* fdec = fopen(fcomp_name, "rb");
    ASSERT(fdec, "failed to open file for reading");

    struct dData dparams;
    memset(&dparams, 0, sizeof(dparams));

    dparams.bufferSize = 256 * 1024;
    dparams.jobs       = 1;
    dparams.headerless = 0;

    struct dContext* dctx = NULL;
    ASSERT(initDecompressor(&dparams, fdec, &dctx) == 0, "failed to init decompressor");

    unsigned char* out = (unsigned char*)malloc(size);
    ASSERT(out, "failed to allocate buffer memory");

    size_t totalOut = 0;
    while (1) {
        size_t inBytes  = 0;
        size_t outBytes = dparams.bufferSize;

        int r = decompress(dctx, out + totalOut, &inBytes, &outBytes);

        if (r != 0)
            break; // expected EOF

        if (outBytes == 0)
            break;

        totalOut += outBytes;
    }

    ASSERT(totalOut == size, "failed to decompress: invalid data size");
    ASSERT(memcmp(out, data, size) == 0, "failed to decompress: data differ from original");

    disposeDecompressor(&dctx);
    fclose(fdec);

    remove(f_name);
    remove(fcomp_name);

    free(out);
    free(data);
}

// Headerless mode
static void test_headerless(void)
{
    printf("TEST: headerless\n");

    const char* input = "HEADERLESS MODE IS ACTIVE";

    const char* f_name = "tmp_hl_input.bin";
    write_file(f_name, (const unsigned char*)input, strlen(input));

    // Compress with headerless = 1
    const char* fcomp_name = "tmp_hl_comp.bin";
    FILE* fcomp = fopen(fcomp_name, "wb");
    ASSERT(fcomp, "failed to open file for writing");

    struct cData cparams;
    memset(&cparams, 0, sizeof(cparams));

    strcpy(cparams.transform, "LZ");
    strcpy(cparams.entropy,   "ANS0");
    cparams.blockSize  = 1 << 15;
    cparams.jobs       = 1;
    cparams.checksum   = 0;
    cparams.headerless = 1;

    struct cContext* cctx = NULL;
    ASSERT(initCompressor(&cparams, fcomp, &cctx) == 0, "failed to init compressor");

    size_t inSize  = strlen(input);
    size_t outSize = 0;
    ASSERT(compress(cctx, (const unsigned char*)input, inSize, &outSize) == 0, "failed to compress data");

    size_t flushed = 0;
    ASSERT(disposeCompressor(&cctx, &flushed) == 0, "failed to dispose compressor");

    fclose(fcomp);

    // Decompress with headerless = 1
    FILE* fdec = fopen(fcomp_name, "rb");
    ASSERT(fdec, "failed to open file for reading");

    struct dData dparams;
    memset(&dparams, 0, sizeof(dparams));

    dparams.bufferSize   = 1 << 15;
    dparams.jobs         = 1;
    dparams.headerless   = 1;

    strcpy(dparams.transform, "LZ");
    strcpy(dparams.entropy,   "ANS0");
    dparams.blockSize     = 1 << 15;
    dparams.originalSize  = strlen(input);
    dparams.checksum      = 0;
    dparams.bsVersion     = 1;

    struct dContext* dctx = NULL;
    ASSERT(initDecompressor(&dparams, fdec, &dctx) == 0, "failed to init decompressor");

    unsigned char outbuf[256];
    size_t inBytes  = 0;
    size_t outBytes = sizeof(outbuf);

    ASSERT(decompress(dctx, outbuf, &inBytes, &outBytes) == 0, "failed to decompress data");
    ASSERT(outBytes == strlen(input), "failed to decompress data: wrong data size");
    ASSERT(memcmp(outbuf, input, strlen(input)) == 0, "failed to decompress data: data differ from original");

    disposeDecompressor(&dctx);
    fclose(fdec);

    remove(f_name);
    remove(fcomp_name);
}

int main(void)
{
    // Compressor
    test_init_invalid();
    test_init_dispose();
    test_compress_small();
    test_compress_too_big();
    test_compress_two_blocks();

    // Decompressor
    test_basic_decompression();
    test_large_multi_block();
    test_headerless();

    printf("All C API tests passed.\n");
    return 0;
}

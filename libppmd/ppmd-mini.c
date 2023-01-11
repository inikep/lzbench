#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <getopt.h>
#include "Ppmd8.h"

#ifdef _WIN32
#define putc_unlocked putc
#define getc_unlocked getc
#endif

static void *pmalloc(ISzAllocPtr ip, size_t size)
{
    (void) ip;
    return malloc(size);
}

static void pfree(ISzAllocPtr ip, void *addr)
{
    (void) ip;
    free(addr);
}

static ISzAlloc ialloc = { pmalloc, pfree };

struct CharWriter {
    /* Inherits from IByteOut */
    void (*Write)(void *p, Byte b);
    FILE *fp;
};

struct CharReader {
    /* Inherits from IByteIn */
    Byte (*Read)(void *p);
    FILE *fp;
    bool eof;
};

static void Write(void *p, Byte b)
{
    struct CharWriter *cw = p;
    putc_unlocked(b, cw->fp);
}

static Byte Read(void *p)
{
    struct CharReader *cr = p;
    if (cr->eof)
        return 0;
    int c = getc_unlocked(cr->fp);
    if (c == EOF) {
        cr->eof = 1;
        return 0;
    }
    return c;
}

static int opt_mem = 16;
static int opt_order = 4;
static int opt_restore = 0;

struct header {
    unsigned magic, attr;
    unsigned short info, fnlen;
    unsigned short date, time;
} hdr = {
#define MAGIC 0x84ACAF8F
    MAGIC, /* FILE_ATTRIBUTE_NORMAL */ 0x80,
    0, 1, 0, 0,
};

static int compress(FILE* in, FILE* out, char* fname)
{
    int fnameLen = strlen(fname) & 0x3FFF;
    hdr.info = (opt_order - 1) | ((opt_mem - 1) << 4) | (('I' - 'A') << 12);
    hdr.fnlen = fnameLen;
    fwrite(&hdr, sizeof hdr, 1, out);
    fwrite(fname, 1, fnameLen, out);

    struct CharWriter cw = { Write, out };
    CPpmd8 ppmd = { .Stream.Out = (IByteOut *) &cw };
    Ppmd8_Construct(&ppmd);
    Ppmd8_Alloc(&ppmd, opt_mem << 20, &ialloc);
    Ppmd8_Init_RangeEnc(&ppmd);
    Ppmd8_Init(&ppmd, opt_order, 0);

    unsigned char buf[BUFSIZ];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, in))) {
        for (size_t i = 0; i < n; i++)
            Ppmd8_EncodeSymbol(&ppmd, buf[i]);
    }
    Ppmd8_EncodeSymbol(&ppmd, -1); /* EndMark */
    Ppmd8_Flush_RangeEnc(&ppmd);
    return fflush(out) != 0 || ferror(in);
}

static int decompress(FILE* in, FILE* out)
{
    if (fread(&hdr, sizeof hdr, 1, in) != 1)
        return 1;
    if (hdr.magic != MAGIC)
        return 1;
    if (hdr.info >> 12 != 'I' - 'A')
        return 1;

    char fname[0x1FF];
    size_t fnlen = hdr.fnlen & 0x1FF;
    if (fread(fname, fnlen, 1, in) != 1)
        return 1;

    opt_restore = hdr.fnlen >> 14;
    opt_order = (hdr.info & 0xf) + 1;
    opt_mem = ((hdr.info >> 4) & 0xff) + 1;

    struct CharReader cr = { Read, in, 0 };
    CPpmd8 ppmd = { .Stream.In = (IByteIn *) &cr };
    Ppmd8_Construct(&ppmd);
    Ppmd8_Alloc(&ppmd, opt_mem << 20, &ialloc);
    Ppmd8_Init_RangeDec(&ppmd);
    Ppmd8_Init(&ppmd, opt_order, opt_restore);

    unsigned char buf[BUFSIZ];
    size_t n = 0;
    int c;
    while (1) {
        c = Ppmd8_DecodeSymbol(&ppmd);
        if (cr.eof || c < 0)
            break;
        buf[n++] = c;
        if (n == sizeof buf) {
            fwrite(buf, 1, sizeof buf, out);
            n = 0;
        }
    }
    if (n)
        fwrite(buf, 1, n, out);
    return fflush(out) != 0 || c != -1 ||
           !Ppmd8_RangeDec_IsFinishedOK(&ppmd) ||
           ferror(in) || getc_unlocked(in) != EOF;
}

int main(int argc, char **argv)
{
    static const struct option longopts[] = {
        { "decompress", 0, NULL, 'd' },
        { "uncompress", 0, NULL, 'd' },
        { "keep",       0, NULL, 'k' },
        { "memory",     1, NULL, 'm' },
        { "order",      1, NULL, 'o' },
        { "help",       0, NULL, 'h' },
        {  NULL,        0, NULL,  0  },
    };
    bool opt_d = 0;
    bool opt_k = 0;
    int c;
    FILE* in = NULL;
    FILE* out = NULL;
    while ((c = getopt_long(argc, argv, "dkm:o:36h", longopts, NULL)) != -1) {
        switch (c) {
        case 'd':
            opt_d = 1;
            break;
        case 'k':
            opt_k = 1;
            break;
        case 'm':
            opt_mem = atoi(optarg);
            break;
        case 'o':
            opt_order = atoi(optarg);
            break;
        case '3':
            opt_mem = 1;
            opt_order = 5;
            break;
        case '6':
            opt_mem = 8;
            opt_order = 6;
            break;
        default:
            goto usage;
        }
    }
    argc -= optind;
    argv += optind;
    if (argc != 1) {
        fputs("ppmid-mini: invalid arguments\n", stderr);
usage:  fputs("Usage: ppmid-mini [-d] [-k] FILE\n", stderr);
        return 1;
    }
    char *fname = argv[0];
    char *fname2 = fname;
    if (fname) {
        char* spos = strrchr(fname2, '/');
        if (spos)
            fname2 = spos + 1;
        spos = strrchr(fname2, '\\');
        if (spos)
            fname2 = spos + 1;
        in = fopen(fname, "rb");
        if (!in) {
            fprintf(stderr, "ppmid-mini: cannot open %s\n", fname);
            return 1;
        }
    }
    if (opt_d) {
        char *dot = strrchr(fname, '.');
        if (dot == NULL || dot[1] != 'p' || strchr(dot, '/')) {
            fprintf(stderr, "ppmid-mini: unknown suffix: %s\n", fname);
            return 1;
        }
        *dot = '\0';
        out = fopen(fname, "wb");
        if (!out) {
            fprintf(stderr, "ppmid-mini: cannot open %s\n", fname);
            return 1;
        }
        *dot = '.';
    }
    if (!opt_d) {
        size_t len = strlen(fname);
        char* outname = malloc(len + 6);
        memcpy(outname, fname, len);
        memcpy(outname + len, ".ppmd", 6);
        out = fopen(outname, "wb");
        if (!out) {
            fprintf(stderr, "ppmid-mini: cannot open %s\n", outname);
            free(outname);
            return 1;
        }
        free(outname);
    }
    int rc = opt_d ? decompress(in, out) : compress(in, out, fname2);
    fclose(in);
    fclose(out);
    if (rc == 0 && !opt_k) {
        remove(fname);
    }
    return rc;
}

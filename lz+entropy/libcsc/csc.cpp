#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <csc_enc.h>
#include <csc_dec.h>
#include <Types.h>


struct StdioSeqStream
{
    union {
        ISeqInStream is;
        ISeqOutStream os;
    };
    FILE *f;
};


int stdio_read(void *p, void *buf, size_t *size)
{
    StdioSeqStream *sss = (StdioSeqStream *)p;
    *size = fread(buf, 1, *size, sss->f);
    return 0;
}

size_t stdio_write(void *p, const void *buf, size_t size)
{
    StdioSeqStream *sss = (StdioSeqStream *)p;
    return fwrite(buf, 1, size, sss->f);
}

int show_progress(void *p, UInt64 insize, UInt64 outsize)
{
    (void)p;
    fprintf(stderr, "\r%llu -> %llu\t\t\t\t", insize, outsize);
    fflush(stdout);
    return 0;
}

void ShowUsage(char *me)
{
    fprintf(stderr, "Usage: %s c/d [options] input output\n", me);
    fprintf(stderr, "       options:\n");
    fprintf(stderr, "        -m{1..5}, compression level from fast to best\n");
    fprintf(stderr, "        -d{###[k|m], dictionary size, ### Bytes [or KB/MB], 32KB <= d < 1GB\n");
    fprintf(stderr, "        -fdelta0, -fexe0, -ftxt0 to turn filters off for data tables, execodes, or English text\n");
    exit(1);
}

int ParseOpt(CSCProps *p, char *argv)
{        
    if (strncmp(argv, "-fdelta0", 8) == 0)
        p->DLTFilter = 0;
    else if (strncmp(argv, "-fexe0", 6) == 0)
        p->EXEFilter = 0;
    else if (strncmp(argv, "-ftxt0", 6) == 0)
        p->TXTFilter = 0;
    return 0;
}

int ParseBasicOpt(char *argv, uint32_t *dict_size, int *level)
{
    if (strncmp(argv, "-m", 2) == 0) {
        if (argv[2])
            *level = argv[2] - '0';
        else
            return -1;
    } else if (strncmp(argv, "-d", 2) == 0) {
        int slen = strlen(argv);
        *dict_size = atoi(argv + 2);
        if ((argv[slen - 1] | 0x20) == 'k') 
            *dict_size *= 1024;
        else if ((argv[slen - 1] | 0x20) == 'm')
            *dict_size *= 1024 * 1024;
        if (*dict_size < 32 * 1024 || *dict_size >= 1024 * 1024 * 1024)
            return -1;
    }
    return 0;
}

#ifdef _WIN32
uint64_t GetFileSize(FILE *f) 
{
    uint64_t size;
    _fseeki64(f, 0, SEEK_END);
    size = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    return size;
}
#else
#include <sys/types.h>
#include <unistd.h>
uint64_t GetFileSize(FILE *f) 
{
    uint64_t size = lseek(fileno(f), 0, SEEK_END);
    lseek(fileno(f), 0, SEEK_SET);
    return size;
}
#endif

int main(int argc, char *argv[])
{
    FILE *fin, *fout;

    if (argc < 4)
        ShowUsage(argv[0]);

    fin = fopen(argv[argc - 2], "rb");
    fout = fopen(argv[argc - 1], "wb");
    if (fin == NULL || fout == NULL) {
        fprintf(stderr, "File open failed\n");
        return 1;
    }

    StdioSeqStream isss, osss;
    isss.f = fin;
    isss.is.Read = stdio_read;
    osss.f = fout;
    osss.os.Write = stdio_write;
    ICompressProgress prog;
    prog.Progress = show_progress;

    if (argv[1][0] == 'c') {
        CSCProps p;
        uint32_t dict_size = 64000000;
        uint64_t filesize = GetFileSize(fin);
        int level = 2;
        for(int i = 2; i < argc - 2; i++) {
            if (ParseBasicOpt(argv[i], &dict_size, &level) < 0)
                ShowUsage(argv[0]);
        }

        if (filesize < dict_size)
            dict_size = filesize;

        // init the default settings
        CSCEncProps_Init(&p, dict_size, level);
        // Then make extra settings
        for(int i = 2; i < argc - 2; i++) {
            if (ParseOpt(&p, argv[i]) < 0)
                ShowUsage(argv[0]);
        }

        fprintf(stderr, "Estimated memory usage: %llu MB\n", CSCEnc_EstMemUsage(&p) / 1048576ull);
        unsigned char buf[CSC_PROP_SIZE];
        CSCEnc_WriteProperties(&p, buf, 0);
        (void)(fwrite(buf, 1, CSC_PROP_SIZE, fout) + 1);
        CSCEncHandle h = CSCEnc_Create(&p, (ISeqOutStream*)&osss, NULL);
        CSCEnc_Encode(h, (ISeqInStream*)&isss, &prog);
        CSCEnc_Encode_Flush(h);
        CSCEnc_Destroy(h);
    } else {
        CSCProps p;
        unsigned char buf[CSC_PROP_SIZE];
        (void)(fread(buf, 1, CSC_PROP_SIZE, fin) + 1);
        CSCDec_ReadProperties(&p, buf);
        CSCDecHandle h = CSCDec_Create(&p, (ISeqInStream*)&isss, NULL);
        if (!h) {
            printf("Invalid csc compressed file\n");
            return 1;
        }
        CSCDec_Decode(h, (ISeqOutStream*)&osss, &prog);
        CSCDec_Destroy(h);
    }
    fclose(fin);
    fclose(fout);

    printf("\n");
    return 0;
}



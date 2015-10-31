#include <stdio.h>
#include <stdlib.h>
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
    printf("\r%llu -> %llu\t\t\t\t", insize, outsize);
    return 0;
}

int main(int argc, char *argv[])
{
    FILE *fin, *fout;
    fin = fopen(argv[1], "rb");
    fout = fopen(argv[2], "wb");
    if (fin == NULL || fout == NULL) {
        fprintf(stderr, "File open failed\n");
        return -1;
    }

    StdioSeqStream isss, osss;
    isss.f = fin;
    isss.is.Read = stdio_read;
    osss.f = fout;
    osss.os.Write = stdio_write;
    ICompressProgress prog;
    prog.Progress = show_progress;

    CSCProps p;
    unsigned char buf[CSC_PROP_SIZE];
    (void)(fread(buf, 1, CSC_PROP_SIZE, fin) + 1);
    CSCDec_ReadProperties(&p, buf);
    CSCDecHandle h = CSCDec_Create(&p, (ISeqInStream*)&isss);
    CSCDec_Decode(h, (ISeqOutStream*)&osss, &prog);
    CSCDec_Destroy(h);
    fclose(fin);
    fclose(fout);

    printf("\n");
    return 0;
}


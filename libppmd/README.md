# ppmd-mini - standalone PPMd compressor

ppmd-mini is a data compression tool with command line syntax similar to gzip.
It uses the PPMd algorithm which provides superior compression for text files.
Specifically, it uses "variant I" of this algorithm which is also used in Zip.
PPMd was developed by Dmitry Shkarin; Igor Pavlov adapted his work for use in
the 7-Zip project.  ppmd-mini employs Ppmd8 compression and decompression
routines from 7-Zip.  Files created by ppmd-mini are compatible with the
original PPMd implementation; they can also be decompressed with 7-Zip.

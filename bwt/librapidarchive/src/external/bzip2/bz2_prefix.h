#pragma once

/* Prefix the vendored librapidarchive bzip2 symbols so the stock bzip2 codec can be linked into the
 * same lzbench binary. */
#define BZ2_bz__AssertH__fail                  lzbench_librapidarchive_bz2_bz__AssertH__fail
#define BZ2_bzCompressInit                     lzbench_librapidarchive_bz2_bzCompressInit
#define BZ2_bzCompress                         lzbench_librapidarchive_bz2_bzCompress
#define BZ2_bzCompressEnd                      lzbench_librapidarchive_bz2_bzCompressEnd
#define BZ2_bzWriteOpen                        lzbench_librapidarchive_bz2_bzWriteOpen
#define BZ2_bzWrite                            lzbench_librapidarchive_bz2_bzWrite
#define BZ2_bzWriteClose                       lzbench_librapidarchive_bz2_bzWriteClose
#define BZ2_bzWriteClose64                     lzbench_librapidarchive_bz2_bzWriteClose64
#define BZ2_bzBuffToBuffCompress               lzbench_librapidarchive_bz2_bzBuffToBuffCompress
#define BZ2_bzBuffToBuffCompressWithCRC        lzbench_librapidarchive_bz2_bzBuffToBuffCompressWithCRC
#define BZ2_bzlibVersion                       lzbench_librapidarchive_bz2_bzlibVersion
#define BZ2_blockSort                          lzbench_librapidarchive_bz2_blockSort
#define BZ2_compressBlock                      lzbench_librapidarchive_bz2_compressBlock
#define BZ2_bsInitWrite                        lzbench_librapidarchive_bz2_bsInitWrite
#define BZ2_hbMakeCodeLengths                  lzbench_librapidarchive_bz2_hbMakeCodeLengths
#define BZ2_hbAssignCodes                      lzbench_librapidarchive_bz2_hbAssignCodes
#define BZ2_hbCreateDecodeTables               lzbench_librapidarchive_bz2_hbCreateDecodeTables
#define BZ2_crc32Table                         lzbench_librapidarchive_bz2_crc32Table
#define BZ2_rNums                              lzbench_librapidarchive_bz2_rNums

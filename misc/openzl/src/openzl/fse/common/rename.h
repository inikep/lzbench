#ifndef ZSTRONG_FSE_COMMON_RENAME_H
#define ZSTRONG_FSE_COMMON_RENAME_H

/**
 * Header to rename every symbol in FSE to avoid duplicate symbol issues.
 */

#ifndef FSE_PREFIX
#    define FSE_PREFIX ZS_
#endif

#define FSE_CONCAT0(x, y) x##y
#define FSE_CONCAT(x, y) FSE_CONCAT0(x, y)
#define FSE_RENAME(name) FSE_CONCAT(FSE_PREFIX, name)

/* common/debug.c */

#define g_debuglevel FSE_RENAME(g_debuglevel)

/* common/error_private.c */

#define ERR_getErrorString FSE_RENAME(ERR_getErrorString)

/* common/entropy_common.c */

#define FSE_versionNumber FSE_RENAME(FSE_versionNumber)
#define FSE_isError FSE_RENAME(FSE_isError)
#define FSE_getErrorName FSE_RENAME(FSE_getErrorName)
#define HUF_isError FSE_RENAME(HUF_isError)
#define HUF_getErrorName FSE_RENAME(HUF_getErrorName)

#define FSE_readNCount_bmi2 FSE_RENAME(FSE_readNCount_bmi2)
#define FSE_readNCount FSE_RENAME(FSE_readNCount)
#define HUF_readStats FSE_RENAME(HUF_readStats)
#define HUF_readStats_wksp FSE_RENAME(HUF_readStats_wksp)

/* common/fse_decompress.c */

#define FSE_createDTable FSE_RENAME(FSE_createDTable)
#define FSE_freeDTable FSE_RENAME(FSE_freeDTable)
#define FSE_buildDTable_wksp FSE_RENAME(FSE_buildDTable_wksp)
#define FSE_buildDTable_constant FSE_RENAME(FSE_buildDTable_constant)
#define FSE_buildDTable_raw FSE_RENAME(FSE_buildDTable_raw)
#define FSE_decompress_usingDTable FSE_RENAME(FSE_decompress_usingDTable)
#define FSE_decompress_wksp FSE_RENAME(FSE_decompress_wksp)
#define FSE_decompress_wksp_bmi2 FSE_RENAME(FSE_decompress_wksp_bmi2)
#define FSE_buildDTable FSE_RENAME(FSE_buildDTable)
#define FSE_decompress FSE_RENAME(FSE_decompress)

/* compress/fse_compress.c */

#define FSE_buildCTable_wksp FSE_RENAME(FSE_buildCTable_wksp)
#define FSE_buildCTable FSE_RENAME(FSE_buildCTable)
#define FSE_NCountWriteBound FSE_RENAME(FSE_NCountWriteBound)
#define FSE_writeNCount FSE_RENAME(FSE_writeNCount)
#define FSE_createCTable FSE_RENAME(FSE_createCTable)
#define FSE_freeCTable FSE_RENAME(FSE_freeCTable)
#define FSE_optimalTableLog_internal FSE_RENAME(FSE_optimalTableLog_internal)
#define FSE_optimalTableLog FSE_RENAME(FSE_optimalTableLog)
#define FSE_normalizeCount FSE_RENAME(FSE_normalizeCount)
#define FSE_buildCTable_raw FSE_RENAME(FSE_buildCTable_raw)
#define FSE_buildCTable_constant FSE_RENAME(FSE_buildCTable_constant)
#define FSE_compress_usingCTable FSE_RENAME(FSE_compress_usingCTable)
#define FSE_compressBound FSE_RENAME(FSE_compressBound)
#define FSE_compress_wksp FSE_RENAME(FSE_compress_wksp)
#define FSE_compress2 FSE_RENAME(FSE_compress2)
#define FSE_compress FSE_RENAME(FSE_compress)

/* compress/hist.c */

#define HIST_isError FSE_RENAME(HIST_isError)
#define HIST_count_simple FSE_RENAME(HIST_count_simple)
#define HIST_countFast_wksp FSE_RENAME(HIST_countFast_wksp)
#define HIST_count_wksp FSE_RENAME(HIST_count_wksp)
#define HIST_countFast FSE_RENAME(HIST_countFast)
#define HIST_count FSE_RENAME(HIST_count)

/* compress/huf_compress.c */

#define HUF_writeCTable_wksp FSE_RENAME(HUF_writeCTable_wksp)
#define HUF_writeCTable FSE_RENAME(HUF_writeCTable)
#define HUF_readCTable FSE_RENAME(HUF_readCTable)
#define HUF_getNbBitsFromCTable FSE_RENAME(HUF_getNbBitsFromCTable)
#define HUF_buildCTable_wksp FSE_RENAME(HUF_buildCTable_wksp)
#define HUF_estimateCompressedSize FSE_RENAME(HUF_estimateCompressedSize)
#define HUF_validateCTable FSE_RENAME(HUF_validateCTable)
#define HUF_compressBound FSE_RENAME(HUF_compressBound)
#define HUF_compress1X_usingCTable FSE_RENAME(HUF_compress1X_usingCTable)
#define HUF_compress1X_usingCTable_bmi2 FSE_RENAME(HUF_compress1X_usingCTable_bmi2)
#define HUF_compress4X_usingCTable FSE_RENAME(HUF_compress4X_usingCTable)
#define HUF_compress4X_usingCTable_bmi2 FSE_RENAME(HUF_compress4X_usingCTable_bmi2)
#define HUF_optimalTableLog FSE_RENAME(HUF_optimalTableLog)
#define HUF_compress1X_wksp FSE_RENAME(HUF_compress1X_wksp)
#define HUF_compress1X_repeat FSE_RENAME(HUF_compress1X_repeat)
#define HUF_compress4X_wksp FSE_RENAME(HUF_compress4X_wksp)
#define HUF_compress4X_repeat FSE_RENAME(HUF_compress4X_repeat)
#define HUF_buildCTable FSE_RENAME(HUF_buildCTable)
#define HUF_compress1X FSE_RENAME(HUF_compress1X)
#define HUF_compress2 FSE_RENAME(HUF_compress2)
#define HUF_compress FSE_RENAME(HUF_compress)

/* decompress/huf_decompress_amd64.S */

#define HUF_decompress4X1_usingDTable_internal_bmi2_asm_loop FSE_RENAME(HUF_decompress4X1_usingDTable_internal_bmi2_asm_loop)
#define HUF_decompress4X2_usingDTable_internal_bmi2_asm_loop FSE_RENAME(HUF_decompress4X2_usingDTable_internal_bmi2_asm_loop)
#define _HUF_decompress4X2_usingDTable_internal_bmi2_asm_loop FSE_CONCAT(_, HUF_decompress4X2_usingDTable_internal_bmi2_asm_loop)
#define _HUF_decompress4X1_usingDTable_internal_bmi2_asm_loop FSE_CONCAT(_, HUF_decompress4X1_usingDTable_internal_bmi2_asm_loop)

/* decompress/huf_decompress.c */

#define HUF_readDTableX1_wksp FSE_RENAME(HUF_readDTableX1_wksp)
#define HUF_readDTableX1_wksp_bmi2 FSE_RENAME(HUF_readDTableX1_wksp_bmi2)
#define HUF_decompress1X1_usingDTable FSE_RENAME(HUF_decompress1X1_usingDTable)
#define HUF_decompress1X1_DCtx_wksp FSE_RENAME(HUF_decompress1X1_DCtx_wksp)
#define HUF_decompress4X1_usingDTable FSE_RENAME(HUF_decompress4X1_usingDTable)
#define HUF_decompress4X1_DCtx_wksp FSE_RENAME(HUF_decompress4X1_DCtx_wksp)
#define HUF_readDTableX2_wksp FSE_RENAME(HUF_readDTableX2_wksp)
#define HUF_readDTableX2_wksp_bmi2 FSE_RENAME(HUF_readDTableX2_wksp_bmi2)
#define HUF_decompress1X2_usingDTable FSE_RENAME(HUF_decompress1X2_usingDTable)
#define HUF_decompress1X2_DCtx_wksp FSE_RENAME(HUF_decompress1X2_DCtx_wksp)
#define HUF_decompress4X2_usingDTable FSE_RENAME(HUF_decompress4X2_usingDTable)
#define HUF_decompress4X2_DCtx_wksp FSE_RENAME(HUF_decompress4X2_DCtx_wksp)
#define HUF_decompress1X_usingDTable FSE_RENAME(HUF_decompress1X_usingDTable)
#define HUF_decompress4X_usingDTable FSE_RENAME(HUF_decompress4X_usingDTable)
#define HUF_selectDecoder FSE_RENAME(HUF_selectDecoder)
#define HUF_decompress4X_hufOnly_wksp FSE_RENAME(HUF_decompress4X_hufOnly_wksp)
#define HUF_decompress1X_DCtx_wksp FSE_RENAME(HUF_decompress1X_DCtx_wksp)
#define HUF_decompress1X_usingDTable_bmi2 FSE_RENAME(HUF_decompress1X_usingDTable_bmi2)
#define HUF_decompress1X1_DCtx_wksp_bmi2 FSE_RENAME(HUF_decompress1X1_DCtx_wksp_bmi2)
#define HUF_decompress4X_usingDTable_bmi2 FSE_RENAME(HUF_decompress4X_usingDTable_bmi2)
#define HUF_decompress4X_hufOnly_wksp_bmi2 FSE_RENAME(HUF_decompress4X_hufOnly_wksp_bmi2)
#define HUF_readDTableX1 FSE_RENAME(HUF_readDTableX1)
#define HUF_decompress1X1_DCtx FSE_RENAME(HUF_decompress1X1_DCtx)
#define HUF_decompress1X1 FSE_RENAME(HUF_decompress1X1)
#define HUF_readDTableX2 FSE_RENAME(HUF_readDTableX2)
#define HUF_decompress1X2_DCtx FSE_RENAME(HUF_decompress1X2_DCtx)
#define HUF_decompress1X2 FSE_RENAME(HUF_decompress1X2)
#define HUF_decompress4X1_DCtx FSE_RENAME(HUF_decompress4X1_DCtx)
#define HUF_decompress4X1 FSE_RENAME(HUF_decompress4X1)
#define HUF_decompress4X2_DCtx FSE_RENAME(HUF_decompress4X2_DCtx)
#define HUF_decompress4X2 FSE_RENAME(HUF_decompress4X2)
#define HUF_decompress FSE_RENAME(HUF_decompress)
#define HUF_decompress4X_DCtx FSE_RENAME(HUF_decompress4X_DCtx)
#define HUF_decompress4X_hufOnly FSE_RENAME(HUF_decompress4X_hufOnly)
#define HUF_decompress1X_DCtx FSE_RENAME(HUF_decompress1X_DCtx)

#endif

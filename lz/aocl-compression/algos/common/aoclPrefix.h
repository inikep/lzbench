/**
 * Copyright (C) 2025-2026, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

 /** @file aoclPrefix.h
 *
 *  @brief Prefix library symbols to avoid conflicts.
 *
 *  @author Ravi Jangra
 */

 #ifndef __AOCL_PREFIX_H
 #define __AOCL_PREFIX_H

 #ifdef AOCL_LLC_PREFIX
 #define AOCL_PREFIX_SET

 /**** ZLIB ****/
 /* all linked symbols and init macros */
#  define _dist_code            AOCL_LLC__dist_code
#  define _length_code          AOCL_LLC__length_code
#  define _tr_align             AOCL_LLC__tr_align
#  define _tr_flush_bits        AOCL_LLC__tr_flush_bits
#  define _tr_flush_block       AOCL_LLC__tr_flush_block
#  define _tr_init              AOCL_LLC__tr_init
#  define _tr_stored_block      AOCL_LLC__tr_stored_block
#  define _tr_tally             AOCL_LLC__tr_tally
#  define adler32               AOCL_LLC_adler32
#  define adler32_combine       AOCL_LLC_adler32_combine
#  define adler32_combine64     AOCL_LLC_adler32_combine64
#  define adler32_z             AOCL_LLC_adler32_z
#  ifndef Z_SOLO
#    define compress              AOCL_LLC_compress
#    define compress2             AOCL_LLC_compress2
#    define compress_z            AOCL_LLC_compress_z
#    define compress2_z           AOCL_LLC_compress2_z
#    define compressBound         AOCL_LLC_compressBound
#    define compressBound_z       AOCL_LLC_compressBound_z
#  endif
#  define crc32                 AOCL_LLC_crc32
#  define crc32_combine         AOCL_LLC_crc32_combine
#  define crc32_combine64       AOCL_LLC_crc32_combine64
#  define crc32_combine_gen     AOCL_LLC_crc32_combine_gen
#  define crc32_combine_gen64   AOCL_LLC_crc32_combine_gen64
#  define crc32_combine_op      AOCL_LLC_crc32_combine_op
#  define crc32_z               AOCL_LLC_crc32_z
#  define deflate               AOCL_LLC_deflate
#  define deflate_medium        AOCL_LLC_deflate_medium
#  define deflate_quick         AOCL_LLC_deflate_quick
#  define deflateBound          AOCL_LLC_deflateBound
#  define deflateBound_z        AOCL_LLC_deflateBound_z
#  define deflateCopy           AOCL_LLC_deflateCopy
#  define deflateEnd            AOCL_LLC_deflateEnd
#  define deflateGetDictionary  AOCL_LLC_deflateGetDictionary
#  define deflateInit           AOCL_LLC_deflateInit
#  define deflateInit2          AOCL_LLC_deflateInit2
#  define deflateInit2_         AOCL_LLC_deflateInit2_
#  define deflateInit_          AOCL_LLC_deflateInit_
#  define deflateParams         AOCL_LLC_deflateParams
#  define deflatePending        AOCL_LLC_deflatePending
#  define deflatePrime          AOCL_LLC_deflatePrime
#  define deflateReset          AOCL_LLC_deflateReset
#  define deflateResetKeep      AOCL_LLC_deflateResetKeep
#  define deflateSetDictionary  AOCL_LLC_deflateSetDictionary
#  define deflateSetHeader      AOCL_LLC_deflateSetHeader
#  define deflateTune           AOCL_LLC_deflateTune
#  define deflateUsed           AOCL_LLC_deflateUsed
#  define deflate_copyright     AOCL_LLC_deflate_copyright
#  define get_crc_table         AOCL_LLC_get_crc_table
#  ifndef Z_SOLO
#    define gz_error              AOCL_LLC_gz_error
#    define gz_intmax             AOCL_LLC_gz_intmax
#    define gz_strwinerror        AOCL_LLC_gz_strwinerror
#    define gzbuffer              AOCL_LLC_gzbuffer
#    define gzclearerr            AOCL_LLC_gzclearerr
#    define gzclose               AOCL_LLC_gzclose
#    define gzclose_r             AOCL_LLC_gzclose_r
#    define gzclose_w             AOCL_LLC_gzclose_w
#    define gzdirect              AOCL_LLC_gzdirect
#    define gzdopen               AOCL_LLC_gzdopen
#    define gzeof                 AOCL_LLC_gzeof
#    define gzerror               AOCL_LLC_gzerror
#    define gzflush               AOCL_LLC_gzflush
#    define gzfread               AOCL_LLC_gzfread
#    define gzfwrite              AOCL_LLC_gzfwrite
#    define gzgetc                AOCL_LLC_gzgetc
#    define gzgetc_               AOCL_LLC_gzgetc_
#    define gzgets                AOCL_LLC_gzgets
#    define gzoffset              AOCL_LLC_gzoffset
#    define gzoffset64            AOCL_LLC_gzoffset64
#    define gzopen                AOCL_LLC_gzopen
#    define gzopen64              AOCL_LLC_gzopen64
#    ifdef _WIN32
#      define gzopen_w              AOCL_LLC_gzopen_w
#    endif
#    define gzprintf              AOCL_LLC_gzprintf
#    define gzputc                AOCL_LLC_gzputc
#    define gzputs                AOCL_LLC_gzputs
#    define gzread                AOCL_LLC_gzread
#    define gzrewind              AOCL_LLC_gzrewind
#    define gzseek                AOCL_LLC_gzseek
#    define gzseek64              AOCL_LLC_gzseek64
#    define gzsetparams           AOCL_LLC_gzsetparams
#    define gztell                AOCL_LLC_gztell
#    define gztell64              AOCL_LLC_gztell64
#    define gzungetc              AOCL_LLC_gzungetc
#    define gzvprintf             AOCL_LLC_gzvprintf
#    define gzwrite               AOCL_LLC_gzwrite
#  endif
#  define inflate               AOCL_LLC_inflate
#  define inflateBack           AOCL_LLC_inflateBack
#  define inflateBackEnd        AOCL_LLC_inflateBackEnd
#  define inflateBackInit       AOCL_LLC_inflateBackInit
#  define inflateBackInit_      AOCL_LLC_inflateBackInit_
#  define inflateCodesUsed      AOCL_LLC_inflateCodesUsed
#  define inflateCopy           AOCL_LLC_inflateCopy
#  define inflateEnd            AOCL_LLC_inflateEnd
#  define inflateGetDictionary  AOCL_LLC_inflateGetDictionary
#  define inflateGetHeader      AOCL_LLC_inflateGetHeader
#  define inflateInit           AOCL_LLC_inflateInit
#  define inflateInit2          AOCL_LLC_inflateInit2
#  define inflateInit2_         AOCL_LLC_inflateInit2_
#  define inflateInit_          AOCL_LLC_inflateInit_
#  define inflateMark           AOCL_LLC_inflateMark
#  define inflatePrime          AOCL_LLC_inflatePrime
#  define inflateReset          AOCL_LLC_inflateReset
#  define inflateReset2         AOCL_LLC_inflateReset2
#  define inflateResetKeep      AOCL_LLC_inflateResetKeep
#  define inflateSetDictionary  AOCL_LLC_inflateSetDictionary
#  define inflateSync           AOCL_LLC_inflateSync
#  define inflateSyncPoint      AOCL_LLC_inflateSyncPoint
#  define inflateUndermine      AOCL_LLC_inflateUndermine
#  define inflateValidate       AOCL_LLC_inflateValidate
#  define inflate_copyright     AOCL_LLC_inflate_copyright
#  define inflate_fast          AOCL_LLC_inflate_fast
#  define inflate_table         AOCL_LLC_inflate_table
#  define inflate_fixed         AOCL_LLC_inflate_fixed
#  ifndef Z_SOLO
#    define uncompress            AOCL_LLC_uncompress
#    define uncompress2           AOCL_LLC_uncompress2
#    define uncompress_z          AOCL_LLC_uncompress_z
#    define uncompress2_z         AOCL_LLC_uncompress2_z
#  endif
#  define zError                AOCL_LLC_zError
#  define z_errmsg              AOCL_LLC_z_errmsg
#  ifndef Z_SOLO
#    define zcalloc               AOCL_LLC_zcalloc
#    define zcfree                AOCL_LLC_zcfree
#  endif
#  define zlibCompileFlags      AOCL_LLC_zlibCompileFlags
#  define zlibVersion           AOCL_LLC_zlibVersion
#  define static_ltree          AOCL_LLC_static_ltree

/* all zlib typedefs in zlib.h and zconf.h */
#  define Byte                  AOCL_LLC_Byte
#  define Bytef                 AOCL_LLC_Bytef
#  define alloc_func            AOCL_LLC_alloc_func
#  define charf                 AOCL_LLC_charf
#  define free_func             AOCL_LLC_free_func
#  ifndef Z_SOLO
#    define gzFile                AOCL_LLC_gzFile
#  endif
#  define gz_header             AOCL_LLC_gz_header
#  define gz_headerp            AOCL_LLC_gz_headerp
#  define in_func               AOCL_LLC_in_func
#  define intf                  AOCL_LLC_intf
#  define out_func              AOCL_LLC_out_func
#  define uInt                  AOCL_LLC_uInt
#  define uIntf                 AOCL_LLC_uIntf
#  define uLong                 AOCL_LLC_uLong
#  define uLongf                AOCL_LLC_uLongf
#  define voidp                 AOCL_LLC_voidp
#  define voidpc                AOCL_LLC_voidpc
#  define voidpf                AOCL_LLC_voidpf
#  define z_stream              AOCL_LLC_z_stream
#  define z_streamp             AOCL_LLC_z_streamp

/* all zlib structs in zlib.h and zconf.h */
#  define gz_header_s           AOCL_LLC_gz_header_s
#  define internal_state        AOCL_LLC_internal_state
#  define z_stream_s            AOCL_LLC_z_stream_s

/**** ZLIB ****/

/**** LZ4 ****/
/* LZ4 core functions */
#  define LZ4_compress                    AOCL_LLC_LZ4_compress
#  define LZ4_decompress_safe             AOCL_LLC_LZ4_decompress_safe
#  define LZ4_decompress_fast             AOCL_LLC_LZ4_decompress_fast
#  define LZ4_compress_default            AOCL_LLC_LZ4_compress_default
#  define LZ4_compress_fast               AOCL_LLC_LZ4_compress_fast
#  define LZ4_compress_fast_extState      AOCL_LLC_LZ4_compress_fast_extState
#  define LZ4_compress_fast_extState_fastReset  AOCL_LLC_LZ4_compress_fast_extState_fastReset
#  define LZ4_compress_destSize           AOCL_LLC_LZ4_compress_destSize
#  define LZ4_compressBound               AOCL_LLC_LZ4_compressBound
#  define LZ4_sizeofState                 AOCL_LLC_LZ4_sizeofState
#  define LZ4_decompress_safe_partial     AOCL_LLC_LZ4_decompress_safe_partial

/* LZ4 streaming functions */
#  define LZ4_createStream                AOCL_LLC_LZ4_createStream
#  define LZ4_freeStream                  AOCL_LLC_LZ4_freeStream
#  define LZ4_resetStream                 AOCL_LLC_LZ4_resetStream
#  define LZ4_resetStream_fast            AOCL_LLC_LZ4_resetStream_fast
#  define LZ4_loadDict                    AOCL_LLC_LZ4_loadDict
#  define LZ4_compress_fast_continue      AOCL_LLC_LZ4_compress_fast_continue
#  define LZ4_saveDict                    AOCL_LLC_LZ4_saveDict
#  define LZ4_createStreamDecode          AOCL_LLC_LZ4_createStreamDecode
#  define LZ4_freeStreamDecode            AOCL_LLC_LZ4_freeStreamDecode
#  define LZ4_setStreamDecode             AOCL_LLC_LZ4_setStreamDecode
#  define LZ4_decompress_safe_continue    AOCL_LLC_LZ4_decompress_safe_continue
#  define LZ4_decompress_fast_continue    AOCL_LLC_LZ4_decompress_fast_continue
#  define LZ4_decompress_safe_usingDict   AOCL_LLC_LZ4_decompress_safe_usingDict
#  define LZ4_decompress_fast_usingDict   AOCL_LLC_LZ4_decompress_fast_usingDict

/* LZ4 version and utility functions */
#  define LZ4_versionNumber               AOCL_LLC_LZ4_versionNumber
#  define LZ4_versionString               AOCL_LLC_LZ4_versionString

/* LZ4 HC (High Compression) functions */
#  define LZ4_compress_HC                 AOCL_LLC_LZ4_compress_HC
#  define LZ4_compress_HC_extStateHC      AOCL_LLC_LZ4_compress_HC_extStateHC
#  define LZ4_compress_HC_extStateHC_fastReset  AOCL_LLC_LZ4_compress_HC_extStateHC_fastReset
#  define LZ4_sizeofStateHC               AOCL_LLC_LZ4_sizeofStateHC
#  define LZ4_compress_HC_destSize        AOCL_LLC_LZ4_compress_HC_destSize
#  define LZ4_createStreamHC              AOCL_LLC_LZ4_createStreamHC
#  define LZ4_freeStreamHC                AOCL_LLC_LZ4_freeStreamHC
#  define LZ4_resetStreamHC               AOCL_LLC_LZ4_resetStreamHC
#  define LZ4_resetStreamHC_fast          AOCL_LLC_LZ4_resetStreamHC_fast
#  define LZ4_loadDictHC                  AOCL_LLC_LZ4_loadDictHC
#  define LZ4_compress_HC_continue        AOCL_LLC_LZ4_compress_HC_continue
#  define LZ4_compress_HC_continue_destSize  AOCL_LLC_LZ4_compress_HC_continue_destSize
#  define LZ4_saveDictHC                  AOCL_LLC_LZ4_saveDictHC
#  define LZ4_initStreamHC                AOCL_LLC_LZ4_initStreamHC
#  define LZ4_setCompressionLevel         AOCL_LLC_LZ4_setCompressionLevel
#  define LZ4_favorDecompressionSpeed     AOCL_LLC_LZ4_favorDecompressionSpeed

/* LZ4 deprecated/legacy functions */
#  define LZ4_compress_limitedOutput      AOCL_LLC_LZ4_compress_limitedOutput
#  define LZ4_compress_limitedOutput_withState  AOCL_LLC_LZ4_compress_limitedOutput_withState
#  define LZ4_compress_withState          AOCL_LLC_LZ4_compress_withState
#  define LZ4_compress_limitedOutput_continue  AOCL_LLC_LZ4_compress_limitedOutput_continue
#  define LZ4_compress_continue           AOCL_LLC_LZ4_compress_continue
#  define LZ4_uncompress                  AOCL_LLC_LZ4_uncompress
#  define LZ4_uncompress_unknownOutputSize  AOCL_LLC_LZ4_uncompress_unknownOutputSize
#  define LZ4_create                      AOCL_LLC_LZ4_create
#  define LZ4_sizeofStreamState           AOCL_LLC_LZ4_sizeofStreamState
#  define LZ4_resetStreamState            AOCL_LLC_LZ4_resetStreamState
#  define LZ4_slideInputBuffer            AOCL_LLC_LZ4_slideInputBuffer
#  define LZ4_compressHC                  AOCL_LLC_LZ4_compressHC
#  define LZ4_compressHC_limitedOutput    AOCL_LLC_LZ4_compressHC_limitedOutput
#  define LZ4_compressHC2                 AOCL_LLC_LZ4_compressHC2
#  define LZ4_compressHC2_limitedOutput   AOCL_LLC_LZ4_compressHC2_limitedOutput
#  define LZ4_compressHC_withStateHC      AOCL_LLC_LZ4_compressHC_withStateHC
#  define LZ4_compressHC_limitedOutput_withStateHC  AOCL_LLC_LZ4_compressHC_limitedOutput_withStateHC
#  define LZ4_compressHC2_withStateHC     AOCL_LLC_LZ4_compressHC2_withStateHC
#  define LZ4_compressHC2_limitedOutput_withStateHC  AOCL_LLC_LZ4_compressHC2_limitedOutput_withStateHC
#  define LZ4_compressHC_continue         AOCL_LLC_LZ4_compressHC_continue
#  define LZ4_compressHC_limitedOutput_continue  AOCL_LLC_LZ4_compressHC_limitedOutput_continue
#  define LZ4_compressHC2_continue        AOCL_LLC_LZ4_compressHC2_continue
#  define LZ4_compressHC2_limitedOutput_continue  AOCL_LLC_LZ4_compressHC2_limitedOutput_continue
#  define LZ4_createHC                    AOCL_LLC_LZ4_createHC
#  define LZ4_freeHC                      AOCL_LLC_LZ4_freeHC
#  define LZ4_slideInputBufferHC          AOCL_LLC_LZ4_slideInputBufferHC
#  define LZ4_sizeofStreamStateHC         AOCL_LLC_LZ4_sizeofStreamStateHC
#  define LZ4_resetStreamStateHC          AOCL_LLC_LZ4_resetStreamStateHC

/* LZ4 internal/static functions */
#  define LZ4_initStream                  AOCL_LLC_LZ4_initStream
#  define LZ4_attach_dictionary           AOCL_LLC_LZ4_attach_dictionary
#  define LZ4_compress_destSize_extState  AOCL_LLC_LZ4_compress_destSize_extState
#  define LZ4_compress_forceExtDict       AOCL_LLC_LZ4_compress_forceExtDict
#  define LZ4_decompress_safe_withPrefix64k  AOCL_LLC_LZ4_decompress_safe_withPrefix64k
#  define LZ4_decompress_fast_withPrefix64k  AOCL_LLC_LZ4_decompress_fast_withPrefix64k
#  define LZ4_decompress_safe_forceExtDict  AOCL_LLC_LZ4_decompress_safe_forceExtDict
#  define LZ4_decompress_safe_partial_forceExtDict  AOCL_LLC_LZ4_decompress_safe_partial_forceExtDict
#  define LZ4_decompress_safe_partial_usingDict  AOCL_LLC_LZ4_decompress_safe_partial_usingDict
#  define LZ4_decoderRingBufferSize       AOCL_LLC_LZ4_decoderRingBufferSize
#  define LZ4_compressBound_st            AOCL_LLC_LZ4_compressBound_st
#  define LZ4_loadDict_internal           AOCL_LLC_LZ4_loadDict_internal
#  define LZ4_loadDictSlow                AOCL_LLC_LZ4_loadDictSlow
#  define LZ4_attach_HC_dictionary        AOCL_LLC_LZ4_attach_HC_dictionary
#  define LZ4HC_searchExtDict             AOCL_LLC_LZ4HC_searchExtDict

/* LZ4 internal implementation functions (usually hidden but may be exported) */
#  define LZ4_compress_fast_continue_internal  AOCL_LLC_LZ4_compress_fast_continue_internal
#  define LZ4_compress_fast_extState_internal  AOCL_LLC_LZ4_compress_fast_extState_internal
#  define LZ4_compress_HC_destSize_internal  AOCL_LLC_LZ4_compress_HC_destSize_internal
#  define LZ4_compress_HC_extStateHC_fastReset_internal  AOCL_LLC_LZ4_compress_HC_extStateHC_fastReset_internal
#  define LZ4_compress_HC_extStateHC_internal  AOCL_LLC_LZ4_compress_HC_extStateHC_internal
#  define LZ4_compress_HC_internal        AOCL_LLC_LZ4_compress_HC_internal
#  define LZ4_decompress_safe_doubleDict_internal  AOCL_LLC_LZ4_decompress_safe_doubleDict_internal
#  define LZ4_decompress_safe_forceExtDict_internal  AOCL_LLC_LZ4_decompress_safe_forceExtDict_internal
#  define LZ4_decompress_safe_partial_forceExtDict_internal  AOCL_LLC_LZ4_decompress_safe_partial_forceExtDict_internal
#  define LZ4_decompress_safe_partial_internal  AOCL_LLC_LZ4_decompress_safe_partial_internal
#  define LZ4_decompress_safe_withPrefix64k_internal  AOCL_LLC_LZ4_decompress_safe_withPrefix64k_internal

/* LZ4 Frame internal functions */
#  define LZ4F_getErrorCode               AOCL_LLC_LZ4F_getErrorCode
#  define LZ4F_getBlockSize               AOCL_LLC_LZ4F_getBlockSize
#  define LZ4F_createCDict_advanced       AOCL_LLC_LZ4F_createCDict_advanced
#  define LZ4F_createCompressionContext_advanced  AOCL_LLC_LZ4F_createCompressionContext_advanced
#  define LZ4F_compressBegin_internal     AOCL_LLC_LZ4F_compressBegin_internal
#  define LZ4F_compressBegin_usingDict    AOCL_LLC_LZ4F_compressBegin_usingDict
#  define LZ4F_compressBegin_usingDictOnce  AOCL_LLC_LZ4F_compressBegin_usingDictOnce
#  define LZ4F_compressFrame_usingCDict   AOCL_LLC_LZ4F_compressFrame_usingCDict
#  define LZ4F_createDecompressionContext_advanced  AOCL_LLC_LZ4F_createDecompressionContext_advanced
#  define LZ4F_headerSize                 AOCL_LLC_LZ4F_headerSize

/* LZ4 Frame functions */
#  define LZ4F_isError                    AOCL_LLC_LZ4F_isError
#  define LZ4F_getErrorName               AOCL_LLC_LZ4F_getErrorName
#  define LZ4F_compressionLevel_max       AOCL_LLC_LZ4F_compressionLevel_max
#  define LZ4F_compressFrameBound         AOCL_LLC_LZ4F_compressFrameBound
#  define LZ4F_compressFrame              AOCL_LLC_LZ4F_compressFrame
#  define LZ4F_getVersion                 AOCL_LLC_LZ4F_getVersion
#  define LZ4F_createCompressionContext   AOCL_LLC_LZ4F_createCompressionContext
#  define LZ4F_freeCompressionContext     AOCL_LLC_LZ4F_freeCompressionContext
#  define LZ4F_compressBegin              AOCL_LLC_LZ4F_compressBegin
#  define LZ4F_compressBound              AOCL_LLC_LZ4F_compressBound
#  define LZ4F_compressUpdate             AOCL_LLC_LZ4F_compressUpdate
#  define LZ4F_flush                      AOCL_LLC_LZ4F_flush
#  define LZ4F_compressEnd                AOCL_LLC_LZ4F_compressEnd
#  define LZ4F_createDecompressionContext AOCL_LLC_LZ4F_createDecompressionContext
#  define LZ4F_freeDecompressionContext   AOCL_LLC_LZ4F_freeDecompressionContext
#  define LZ4F_getFrameInfo               AOCL_LLC_LZ4F_getFrameInfo
#  define LZ4F_decompress                 AOCL_LLC_LZ4F_decompress
#  define LZ4F_resetDecompressionContext  AOCL_LLC_LZ4F_resetDecompressionContext
#  define LZ4F_createCDict                AOCL_LLC_LZ4F_createCDict
#  define LZ4F_freeCDict                  AOCL_LLC_LZ4F_freeCDict
#  define LZ4F_compressBegin_usingCDict   AOCL_LLC_LZ4F_compressBegin_usingCDict
#  define LZ4F_decompress_usingDict       AOCL_LLC_LZ4F_decompress_usingDict
#  define LZ4F_uncompressedUpdate         AOCL_LLC_LZ4F_uncompressedUpdate

/* LZ4 public typedefs and structs (must be prefixed to avoid conflicts) */
#  define LZ4_stream_u                    AOCL_LLC_LZ4_stream_u
#  define LZ4_stream_t                    AOCL_LLC_LZ4_stream_t
#  define LZ4_stream_t_internal           AOCL_LLC_LZ4_stream_t_internal
#  define LZ4_streamDecode_u              AOCL_LLC_LZ4_streamDecode_u
#  define LZ4_streamDecode_t              AOCL_LLC_LZ4_streamDecode_t
#  define LZ4_streamHC_u                  AOCL_LLC_LZ4_streamHC_u
#  define LZ4_streamHC_t                  AOCL_LLC_LZ4_streamHC_t
#  define LZ4HC_CCtx_internal             AOCL_LLC_LZ4HC_CCtx_internal

/* LZ4 Frame typedefs and structs */
#  define LZ4F_errorCode_t                AOCL_LLC_LZ4F_errorCode_t
#  define LZ4F_blockSizeID_t              AOCL_LLC_LZ4F_blockSizeID_t
#  define LZ4F_blockMode_t                AOCL_LLC_LZ4F_blockMode_t
#  define LZ4F_contentChecksum_t          AOCL_LLC_LZ4F_contentChecksum_t
#  define LZ4F_blockChecksum_t            AOCL_LLC_LZ4F_blockChecksum_t
#  define LZ4F_frameType_t                AOCL_LLC_LZ4F_frameType_t
#  define LZ4F_frameInfo_t                AOCL_LLC_LZ4F_frameInfo_t
#  define LZ4F_preferences_t              AOCL_LLC_LZ4F_preferences_t
#  define LZ4F_compressOptions_t          AOCL_LLC_LZ4F_compressOptions_t
#  define LZ4F_decompressOptions_t        AOCL_LLC_LZ4F_decompressOptions_t
#  define LZ4F_cctx_s                     AOCL_LLC_LZ4F_cctx_s
#  define LZ4F_cctx                       AOCL_LLC_LZ4F_cctx
#  define LZ4F_compressionContext_t       AOCL_LLC_LZ4F_compressionContext_t
#  define LZ4F_dctx_s                     AOCL_LLC_LZ4F_dctx_s
#  define LZ4F_dctx                       AOCL_LLC_LZ4F_dctx
#  define LZ4F_decompressionContext_t     AOCL_LLC_LZ4F_decompressionContext_t
#  define LZ4F_CDict_s                    AOCL_LLC_LZ4F_CDict_s
#  define LZ4F_CDict                      AOCL_LLC_LZ4F_CDict
#  define LZ4F_errorCodes                 AOCL_LLC_LZ4F_errorCodes
#  define LZ4F_AllocFunction              AOCL_LLC_LZ4F_AllocFunction
#  define LZ4F_CallocFunction             AOCL_LLC_LZ4F_CallocFunction
#  define LZ4F_FreeFunction               AOCL_LLC_LZ4F_FreeFunction
#  define LZ4F_CustomMem                  AOCL_LLC_LZ4F_CustomMem

/* Deprecated type aliases (for backward compatibility) */
#  define blockSizeID_t                   AOCL_LLC_blockSizeID_t
#  define blockMode_t                     AOCL_LLC_blockMode_t
#  define contentChecksum_t               AOCL_LLC_contentChecksum_t
#  define frameType_t                     AOCL_LLC_frameType_t

/* XXHash types (used by LZ4) */
#  define XXH_errorcode                   AOCL_LLC_LZ4_XXH_errorcode
#  define XXH_OK                          AOCL_LLC_LZ4_XXH_OK
#  define XXH_ERROR                       AOCL_LLC_LZ4_XXH_ERROR
#  define XXH32_hash_t                    AOCL_LLC_LZ4_XXH32_hash_t
#  define XXH64_hash_t                    AOCL_LLC_LZ4_XXH64_hash_t
#  define XXH32_state_t                   AOCL_LLC_LZ4_XXH32_state_t
#  define XXH64_state_t                   AOCL_LLC_LZ4_XXH64_state_t
#  define XXH32_canonical_t               AOCL_LLC_LZ4_XXH32_canonical_t
#  define XXH64_canonical_t               AOCL_LLC_LZ4_XXH64_canonical_t
#  define XXH32_state_s                   AOCL_LLC_LZ4_XXH32_state_s
#  define XXH64_state_s                   AOCL_LLC_LZ4_XXH64_state_s

/**** LZ4 ****/

/**** ZSTD ****/

/* ZSTD functions and symbols */
#  define ERR_getErrorString    AOCL_LLC_ERR_getErrorString
#  define FSE_buildCTable_rle    AOCL_LLC_FSE_buildCTable_rle
#  define FSE_buildCTable_wksp    AOCL_LLC_FSE_buildCTable_wksp
#  define FSE_buildDTable_wksp    AOCL_LLC_FSE_buildDTable_wksp
#  define FSE_compressBound    AOCL_LLC_FSE_compressBound
#  define FSE_compress_usingCTable    AOCL_LLC_FSE_compress_usingCTable
#  define FSE_decompress_wksp_bmi2    AOCL_LLC_FSE_decompress_wksp_bmi2
#  define FSE_getErrorName    AOCL_LLC_FSE_getErrorName
#  define FSE_isError    AOCL_LLC_FSE_isError
#  define FSE_NCountWriteBound    AOCL_LLC_FSE_NCountWriteBound
#  define FSE_normalizeCount    AOCL_LLC_FSE_normalizeCount
#  define FSE_optimalTableLog    AOCL_LLC_FSE_optimalTableLog
#  define FSE_optimalTableLog_internal    AOCL_LLC_FSE_optimalTableLog_internal
#  define FSE_readNCount    AOCL_LLC_FSE_readNCount
#  define FSE_readNCount_bmi2    AOCL_LLC_FSE_readNCount_bmi2
#  define FSE_versionNumber    AOCL_LLC_FSE_versionNumber
#  define FSE_writeNCount    AOCL_LLC_FSE_writeNCount
#  define HIST_add    AOCL_LLC_HIST_add
#  define HIST_count    AOCL_LLC_HIST_count
#  define HIST_countFast    AOCL_LLC_HIST_countFast
#  define HIST_countFast_wksp    AOCL_LLC_HIST_countFast_wksp
#  define HIST_count_simple    AOCL_LLC_HIST_count_simple
#  define HIST_count_wksp    AOCL_LLC_HIST_count_wksp
#  define HIST_isError    AOCL_LLC_HIST_isError
#  define HUF_buildCTable_wksp    AOCL_LLC_HUF_buildCTable_wksp
#  define HUF_cardinality    AOCL_LLC_HUF_cardinality
#  define HUF_compress1X_repeat    AOCL_LLC_HUF_compress1X_repeat
#  define HUF_compress1X_usingCTable    AOCL_LLC_HUF_compress1X_usingCTable
#  define HUF_compress4X_repeat    AOCL_LLC_HUF_compress4X_repeat
#  define HUF_compress4X_usingCTable    AOCL_LLC_HUF_compress4X_usingCTable
#  define HUF_compressBound    AOCL_LLC_HUF_compressBound
#  define HUF_decompress1X1_DCtx_wksp    AOCL_LLC_HUF_decompress1X1_DCtx_wksp
#  define HUF_decompress1X2_DCtx_wksp    AOCL_LLC_HUF_decompress1X2_DCtx_wksp
#  define HUF_decompress1X_DCtx_wksp    AOCL_LLC_HUF_decompress1X_DCtx_wksp
#  define HUF_decompress1X_usingDTable    AOCL_LLC_HUF_decompress1X_usingDTable
#  define HUF_decompress4X_hufOnly_wksp    AOCL_LLC_HUF_decompress4X_hufOnly_wksp
#  define HUF_decompress4X_usingDTable    AOCL_LLC_HUF_decompress4X_usingDTable
#  define HUF_estimateCompressedSize    AOCL_LLC_HUF_estimateCompressedSize
#  define HUF_getErrorName    AOCL_LLC_HUF_getErrorName
#  define HUF_getNbBitsFromCTable    AOCL_LLC_HUF_getNbBitsFromCTable
#  define HUF_isError    AOCL_LLC_HUF_isError
#  define HUF_minTableLog    AOCL_LLC_HUF_minTableLog
#  define HUF_optimalTableLog    AOCL_LLC_HUF_optimalTableLog
#  define HUF_readCTable    AOCL_LLC_HUF_readCTable
#  define HUF_readCTableHeader    AOCL_LLC_HUF_readCTableHeader
#  define HUF_readDTableX1_wksp    AOCL_LLC_HUF_readDTableX1_wksp
#  define HUF_readDTableX2_wksp    AOCL_LLC_HUF_readDTableX2_wksp
#  define HUF_readStats    AOCL_LLC_HUF_readStats
#  define HUF_readStats_wksp    AOCL_LLC_HUF_readStats_wksp
#  define HUF_selectDecoder    AOCL_LLC_HUF_selectDecoder
#  define HUF_validateCTable    AOCL_LLC_HUF_validateCTable
#  define HUF_writeCTable_wksp    AOCL_LLC_HUF_writeCTable_wksp
#  define POOL_add    AOCL_LLC_POOL_add
#  define POOL_create    AOCL_LLC_POOL_create
#  define POOL_create_advanced    AOCL_LLC_POOL_create_advanced
#  define POOL_free    AOCL_LLC_POOL_free
#  define POOL_joinJobs    AOCL_LLC_POOL_joinJobs
#  define POOL_resize    AOCL_LLC_POOL_resize
#  define POOL_sizeof    AOCL_LLC_POOL_sizeof
#  define POOL_tryAdd    AOCL_LLC_POOL_tryAdd
#  define ZDICT_addEntropyTablesFromBuffer    AOCL_LLC_ZDICT_addEntropyTablesFromBuffer
#  define ZDICT_finalizeDictionary    AOCL_LLC_ZDICT_finalizeDictionary
#  define ZDICT_getDictHeaderSize    AOCL_LLC_ZDICT_getDictHeaderSize
#  define ZDICT_getDictID    AOCL_LLC_ZDICT_getDictID
#  define ZDICT_getErrorName    AOCL_LLC_ZDICT_getErrorName
#  define ZDICT_isError    AOCL_LLC_ZDICT_isError
#  define ZDICT_optimizeTrainFromBuffer_cover    AOCL_LLC_ZDICT_optimizeTrainFromBuffer_cover
#  define ZDICT_optimizeTrainFromBuffer_fastCover    AOCL_LLC_ZDICT_optimizeTrainFromBuffer_fastCover
#  define ZDICT_trainFromBuffer    AOCL_LLC_ZDICT_trainFromBuffer
#  define ZDICT_trainFromBuffer_cover    AOCL_LLC_ZDICT_trainFromBuffer_cover
#  define ZDICT_trainFromBuffer_fastCover    AOCL_LLC_ZDICT_trainFromBuffer_fastCover
#  define ZDICT_trainFromBuffer_legacy    AOCL_LLC_ZDICT_trainFromBuffer_legacy
#  define ZSTD_adjustCParams    AOCL_LLC_ZSTD_adjustCParams
#  define ZSTD_buildBlockEntropyStats    AOCL_LLC_ZSTD_buildBlockEntropyStats
#  define ZSTD_buildCTable    AOCL_LLC_ZSTD_buildCTable
#  define ZSTD_buildFSETable    AOCL_LLC_ZSTD_buildFSETable
#  define ZSTD_CCtx_getParameter    AOCL_LLC_ZSTD_CCtx_getParameter
#  define ZSTD_CCtx_loadDictionary    AOCL_LLC_ZSTD_CCtx_loadDictionary
#  define ZSTD_CCtx_loadDictionary_advanced    AOCL_LLC_ZSTD_CCtx_loadDictionary_advanced
#  define ZSTD_CCtx_loadDictionary_byReference    AOCL_LLC_ZSTD_CCtx_loadDictionary_byReference
#  define ZSTD_CCtxParams_getParameter    AOCL_LLC_ZSTD_CCtxParams_getParameter
#  define ZSTD_CCtxParams_init    AOCL_LLC_ZSTD_CCtxParams_init
#  define ZSTD_CCtxParams_init_advanced    AOCL_LLC_ZSTD_CCtxParams_init_advanced
#  define ZSTD_CCtxParams_registerSequenceProducer    AOCL_LLC_ZSTD_CCtxParams_registerSequenceProducer
#  define ZSTD_CCtxParams_reset    AOCL_LLC_ZSTD_CCtxParams_reset
#  define ZSTD_CCtxParams_setParameter    AOCL_LLC_ZSTD_CCtxParams_setParameter
#  define ZSTD_CCtx_refCDict    AOCL_LLC_ZSTD_CCtx_refCDict
#  define ZSTD_CCtx_refPrefix    AOCL_LLC_ZSTD_CCtx_refPrefix
#  define ZSTD_CCtx_refPrefix_advanced    AOCL_LLC_ZSTD_CCtx_refPrefix_advanced
#  define ZSTD_CCtx_refThreadPool    AOCL_LLC_ZSTD_CCtx_refThreadPool
#  define ZSTD_CCtx_reset    AOCL_LLC_ZSTD_CCtx_reset
#  define ZSTD_CCtx_setCParams    AOCL_LLC_ZSTD_CCtx_setCParams
#  define ZSTD_CCtx_setFParams    AOCL_LLC_ZSTD_CCtx_setFParams
#  define ZSTD_CCtx_setParameter    AOCL_LLC_ZSTD_CCtx_setParameter
#  define ZSTD_CCtx_setParametersUsingCCtxParams    AOCL_LLC_ZSTD_CCtx_setParametersUsingCCtxParams
#  define ZSTD_CCtx_setParams    AOCL_LLC_ZSTD_CCtx_setParams
#  define ZSTD_CCtx_setPledgedSrcSize    AOCL_LLC_ZSTD_CCtx_setPledgedSrcSize
#  define ZSTD_CCtx_trace    AOCL_LLC_ZSTD_CCtx_trace
#  define ZSTD_checkContinuity    AOCL_LLC_ZSTD_checkContinuity
#  define ZSTD_checkCParams    AOCL_LLC_ZSTD_checkCParams
#  define ZSTD_compress    AOCL_LLC_ZSTD_compress
#  define ZSTD_compress2    AOCL_LLC_ZSTD_compress2
#  define ZSTD_compress_advanced    AOCL_LLC_ZSTD_compress_advanced
#  define ZSTD_compress_advanced_internal    AOCL_LLC_ZSTD_compress_advanced_internal
#  define ZSTD_compressBegin    AOCL_LLC_ZSTD_compressBegin
#  define ZSTD_compressBegin_advanced    AOCL_LLC_ZSTD_compressBegin_advanced
#  define ZSTD_compressBegin_advanced_internal    AOCL_LLC_ZSTD_compressBegin_advanced_internal
#  define ZSTD_compressBegin_usingCDict    AOCL_LLC_ZSTD_compressBegin_usingCDict
#  define ZSTD_compressBegin_usingCDict_advanced    AOCL_LLC_ZSTD_compressBegin_usingCDict_advanced
#  define ZSTD_compressBegin_usingCDict_deprecated    AOCL_LLC_ZSTD_compressBegin_usingCDict_deprecated
#  define ZSTD_compressBegin_usingDict    AOCL_LLC_ZSTD_compressBegin_usingDict
#  define ZSTD_compressBlock    AOCL_LLC_ZSTD_compressBlock
#  define ZSTD_compressBlock_btlazy2    AOCL_LLC_ZSTD_compressBlock_btlazy2
#  define ZSTD_compressBlock_btlazy2_dictMatchState    AOCL_LLC_ZSTD_compressBlock_btlazy2_dictMatchState
#  define ZSTD_compressBlock_btlazy2_extDict    AOCL_LLC_ZSTD_compressBlock_btlazy2_extDict
#  define ZSTD_compressBlock_btopt    AOCL_LLC_ZSTD_compressBlock_btopt
#  define ZSTD_compressBlock_btopt_dictMatchState    AOCL_LLC_ZSTD_compressBlock_btopt_dictMatchState
#  define ZSTD_compressBlock_btopt_extDict    AOCL_LLC_ZSTD_compressBlock_btopt_extDict
#  define ZSTD_compressBlock_btultra    AOCL_LLC_ZSTD_compressBlock_btultra
#  define ZSTD_compressBlock_btultra2    AOCL_LLC_ZSTD_compressBlock_btultra2
#  define ZSTD_compressBlock_btultra_dictMatchState    AOCL_LLC_ZSTD_compressBlock_btultra_dictMatchState
#  define ZSTD_compressBlock_btultra_extDict    AOCL_LLC_ZSTD_compressBlock_btultra_extDict
#  define ZSTD_compressBlock_deprecated    AOCL_LLC_ZSTD_compressBlock_deprecated
#  define ZSTD_compressBlock_doubleFast    AOCL_LLC_ZSTD_compressBlock_doubleFast
#  define ZSTD_compressBlock_doubleFast_dictMatchState    AOCL_LLC_ZSTD_compressBlock_doubleFast_dictMatchState
#  define ZSTD_compressBlock_doubleFast_extDict    AOCL_LLC_ZSTD_compressBlock_doubleFast_extDict
#  define ZSTD_compressBlock_fast    AOCL_LLC_ZSTD_compressBlock_fast
#  define ZSTD_compressBlock_fast_dictMatchState    AOCL_LLC_ZSTD_compressBlock_fast_dictMatchState
#  define ZSTD_compressBlock_fast_extDict    AOCL_LLC_ZSTD_compressBlock_fast_extDict
#  define ZSTD_compressBlock_greedy    AOCL_LLC_ZSTD_compressBlock_greedy
#  define ZSTD_compressBlock_greedy_dedicatedDictSearch    AOCL_LLC_ZSTD_compressBlock_greedy_dedicatedDictSearch
#  define ZSTD_compressBlock_greedy_dedicatedDictSearch_row    AOCL_LLC_ZSTD_compressBlock_greedy_dedicatedDictSearch_row
#  define ZSTD_compressBlock_greedy_dictMatchState    AOCL_LLC_ZSTD_compressBlock_greedy_dictMatchState
#  define ZSTD_compressBlock_greedy_dictMatchState_row    AOCL_LLC_ZSTD_compressBlock_greedy_dictMatchState_row
#  define ZSTD_compressBlock_greedy_extDict    AOCL_LLC_ZSTD_compressBlock_greedy_extDict
#  define ZSTD_compressBlock_greedy_extDict_row    AOCL_LLC_ZSTD_compressBlock_greedy_extDict_row
#  define ZSTD_compressBlock_greedy_row    AOCL_LLC_ZSTD_compressBlock_greedy_row
#  define ZSTD_compressBlock_lazy    AOCL_LLC_ZSTD_compressBlock_lazy
#  define ZSTD_compressBlock_lazy2    AOCL_LLC_ZSTD_compressBlock_lazy2
#  define ZSTD_compressBlock_lazy2_dedicatedDictSearch    AOCL_LLC_ZSTD_compressBlock_lazy2_dedicatedDictSearch
#  define ZSTD_compressBlock_lazy2_dedicatedDictSearch_row    AOCL_LLC_ZSTD_compressBlock_lazy2_dedicatedDictSearch_row
#  define ZSTD_compressBlock_lazy2_dictMatchState    AOCL_LLC_ZSTD_compressBlock_lazy2_dictMatchState
#  define ZSTD_compressBlock_lazy2_dictMatchState_row    AOCL_LLC_ZSTD_compressBlock_lazy2_dictMatchState_row
#  define ZSTD_compressBlock_lazy2_extDict    AOCL_LLC_ZSTD_compressBlock_lazy2_extDict
#  define ZSTD_compressBlock_lazy2_extDict_row    AOCL_LLC_ZSTD_compressBlock_lazy2_extDict_row
#  define ZSTD_compressBlock_lazy2_row    AOCL_LLC_ZSTD_compressBlock_lazy2_row
#  define ZSTD_compressBlock_lazy_dedicatedDictSearch    AOCL_LLC_ZSTD_compressBlock_lazy_dedicatedDictSearch
#  define ZSTD_compressBlock_lazy_dedicatedDictSearch_row    AOCL_LLC_ZSTD_compressBlock_lazy_dedicatedDictSearch_row
#  define ZSTD_compressBlock_lazy_dictMatchState    AOCL_LLC_ZSTD_compressBlock_lazy_dictMatchState
#  define ZSTD_compressBlock_lazy_dictMatchState_row    AOCL_LLC_ZSTD_compressBlock_lazy_dictMatchState_row
#  define ZSTD_compressBlock_lazy_extDict    AOCL_LLC_ZSTD_compressBlock_lazy_extDict
#  define ZSTD_compressBlock_lazy_extDict_row    AOCL_LLC_ZSTD_compressBlock_lazy_extDict_row
#  define ZSTD_compressBlock_lazy_row    AOCL_LLC_ZSTD_compressBlock_lazy_row
#  define ZSTD_compressBound    AOCL_LLC_ZSTD_compressBound
#  define ZSTD_compressBound_st    AOCL_LLC_ZSTD_compressBound_st
#  define ZSTD_compressCCtx    AOCL_LLC_ZSTD_compressCCtx
#  define ZSTD_compressContinue    AOCL_LLC_ZSTD_compressContinue
#  define ZSTD_compressContinue_public    AOCL_LLC_ZSTD_compressContinue_public
#  define ZSTD_compressEnd    AOCL_LLC_ZSTD_compressEnd
#  define ZSTD_compressEnd_public    AOCL_LLC_ZSTD_compressEnd_public
#  define ZSTD_compressLiterals    AOCL_LLC_ZSTD_compressLiterals
#  define ZSTD_compressRleLiteralsBlock    AOCL_LLC_ZSTD_compressRleLiteralsBlock
#  define ZSTD_compressSequences    AOCL_LLC_ZSTD_compressSequences
#  define ZSTD_compressSequencesAndLiterals    AOCL_LLC_ZSTD_compressSequencesAndLiterals
#  define ZSTD_compressStream    AOCL_LLC_ZSTD_compressStream
#  define ZSTD_compressStream2    AOCL_LLC_ZSTD_compressStream2
#  define ZSTD_compressStream2_simpleArgs    AOCL_LLC_ZSTD_compressStream2_simpleArgs
#  define ZSTD_compressStream2_simpleArgs_internal    AOCL_LLC_ZSTD_compressStream2_simpleArgs_internal
#  define ZSTD_compressSuperBlock    AOCL_LLC_ZSTD_compressSuperBlock
#  define ZSTD_compress_usingCDict    AOCL_LLC_ZSTD_compress_usingCDict
#  define ZSTD_compress_usingCDict_advanced    AOCL_LLC_ZSTD_compress_usingCDict_advanced
#  define ZSTD_compress_usingDict    AOCL_LLC_ZSTD_compress_usingDict
#  define ZSTD_convertBlockSequences    AOCL_LLC_ZSTD_convertBlockSequences
#  define ZSTD_copyCCtx    AOCL_LLC_ZSTD_copyCCtx
#  define ZSTD_copyDCtx    AOCL_LLC_ZSTD_copyDCtx
#  define ZSTD_copyDDictParameters    AOCL_LLC_ZSTD_copyDDictParameters
#  define ZSTD_cParam_getBounds    AOCL_LLC_ZSTD_cParam_getBounds
#  define ZSTD_createCCtx    AOCL_LLC_ZSTD_createCCtx
#  define ZSTD_createCCtx_advanced    AOCL_LLC_ZSTD_createCCtx_advanced
#  define ZSTD_createCCtxParams    AOCL_LLC_ZSTD_createCCtxParams
#  define ZSTD_createCDict    AOCL_LLC_ZSTD_createCDict
#  define ZSTD_createCDict_advanced    AOCL_LLC_ZSTD_createCDict_advanced
#  define ZSTD_createCDict_advanced2    AOCL_LLC_ZSTD_createCDict_advanced2
#  define ZSTD_createCDict_byReference    AOCL_LLC_ZSTD_createCDict_byReference
#  define ZSTD_createCStream    AOCL_LLC_ZSTD_createCStream
#  define ZSTD_createCStream_advanced    AOCL_LLC_ZSTD_createCStream_advanced
#  define ZSTD_createDCtx    AOCL_LLC_ZSTD_createDCtx
#  define ZSTD_createDCtx_advanced    AOCL_LLC_ZSTD_createDCtx_advanced
#  define ZSTD_createDDict    AOCL_LLC_ZSTD_createDDict
#  define ZSTD_createDDict_advanced    AOCL_LLC_ZSTD_createDDict_advanced
#  define ZSTD_createDDict_byReference    AOCL_LLC_ZSTD_createDDict_byReference
#  define ZSTD_createDStream    AOCL_LLC_ZSTD_createDStream
#  define ZSTD_createDStream_advanced    AOCL_LLC_ZSTD_createDStream_advanced
#  define ZSTD_crossEntropyCost    AOCL_LLC_ZSTD_crossEntropyCost
#  define ZSTD_CStreamInSize    AOCL_LLC_ZSTD_CStreamInSize
#  define ZSTD_CStreamOutSize    AOCL_LLC_ZSTD_CStreamOutSize
#  define ZSTD_cycleLog    AOCL_LLC_ZSTD_cycleLog
#  define ZSTD_DCtx_getParameter    AOCL_LLC_ZSTD_DCtx_getParameter
#  define ZSTD_DCtx_loadDictionary    AOCL_LLC_ZSTD_DCtx_loadDictionary
#  define ZSTD_DCtx_loadDictionary_advanced    AOCL_LLC_ZSTD_DCtx_loadDictionary_advanced
#  define ZSTD_DCtx_loadDictionary_byReference    AOCL_LLC_ZSTD_DCtx_loadDictionary_byReference
#  define ZSTD_DCtx_refDDict    AOCL_LLC_ZSTD_DCtx_refDDict
#  define ZSTD_DCtx_refPrefix    AOCL_LLC_ZSTD_DCtx_refPrefix
#  define ZSTD_DCtx_refPrefix_advanced    AOCL_LLC_ZSTD_DCtx_refPrefix_advanced
#  define ZSTD_DCtx_reset    AOCL_LLC_ZSTD_DCtx_reset
#  define ZSTD_DCtx_setFormat    AOCL_LLC_ZSTD_DCtx_setFormat
#  define ZSTD_DCtx_setMaxWindowSize    AOCL_LLC_ZSTD_DCtx_setMaxWindowSize
#  define ZSTD_DCtx_setParameter    AOCL_LLC_ZSTD_DCtx_setParameter
#  define ZSTD_DDict_dictContent    AOCL_LLC_ZSTD_DDict_dictContent
#  define ZSTD_DDict_dictSize    AOCL_LLC_ZSTD_DDict_dictSize
#  define ZSTD_decodeLiteralsBlock_wrapper    AOCL_LLC_ZSTD_decodeLiteralsBlock_wrapper
#  define ZSTD_decodeSeqHeaders    AOCL_LLC_ZSTD_decodeSeqHeaders
#  define ZSTD_decodingBufferSize_min    AOCL_LLC_ZSTD_decodingBufferSize_min
#  define ZSTD_decompress    AOCL_LLC_ZSTD_decompress
#  define ZSTD_decompressBegin    AOCL_LLC_ZSTD_decompressBegin
#  define ZSTD_decompressBegin_usingDDict    AOCL_LLC_ZSTD_decompressBegin_usingDDict
#  define ZSTD_decompressBegin_usingDict    AOCL_LLC_ZSTD_decompressBegin_usingDict
#  define ZSTD_decompressBlock    AOCL_LLC_ZSTD_decompressBlock
#  define ZSTD_decompressBlock_deprecated    AOCL_LLC_ZSTD_decompressBlock_deprecated
#  define ZSTD_decompressBlock_internal    AOCL_LLC_ZSTD_decompressBlock_internal
#  define ZSTD_decompressBound    AOCL_LLC_ZSTD_decompressBound
#  define ZSTD_decompressContinue    AOCL_LLC_ZSTD_decompressContinue
#  define ZSTD_decompressDCtx    AOCL_LLC_ZSTD_decompressDCtx
#  define ZSTD_decompressionMargin    AOCL_LLC_ZSTD_decompressionMargin
#  define ZSTD_decompressSequences_bmi2_fp    AOCL_LLC_ZSTD_decompressSequences_bmi2_fp
#  define ZSTD_decompressSequences_default_fp    AOCL_LLC_ZSTD_decompressSequences_default_fp
#  define ZSTD_decompressStream    AOCL_LLC_ZSTD_decompressStream
#  define ZSTD_decompressStream_simpleArgs    AOCL_LLC_ZSTD_decompressStream_simpleArgs
#  define ZSTD_decompress_usingDDict    AOCL_LLC_ZSTD_decompress_usingDDict
#  define ZSTD_decompress_usingDict    AOCL_LLC_ZSTD_decompress_usingDict
#  define ZSTD_dedicatedDictSearch_lazy_loadDictionary    AOCL_LLC_ZSTD_dedicatedDictSearch_lazy_loadDictionary
#  define ZSTD_defaultCLevel    AOCL_LLC_ZSTD_defaultCLevel
#  define ZSTD_dParam_getBounds    AOCL_LLC_ZSTD_dParam_getBounds
#  define ZSTD_DStreamInSize    AOCL_LLC_ZSTD_DStreamInSize
#  define ZSTD_DStreamOutSize    AOCL_LLC_ZSTD_DStreamOutSize
#  define ZSTD_encodeSequences    AOCL_LLC_ZSTD_encodeSequences
#  define ZSTD_endStream    AOCL_LLC_ZSTD_endStream
#  define ZSTD_estimateCCtxSize    AOCL_LLC_ZSTD_estimateCCtxSize
#  define ZSTD_estimateCCtxSize_usingCCtxParams    AOCL_LLC_ZSTD_estimateCCtxSize_usingCCtxParams
#  define ZSTD_estimateCCtxSize_usingCParams    AOCL_LLC_ZSTD_estimateCCtxSize_usingCParams
#  define ZSTD_estimateCDictSize    AOCL_LLC_ZSTD_estimateCDictSize
#  define ZSTD_estimateCDictSize_advanced    AOCL_LLC_ZSTD_estimateCDictSize_advanced
#  define ZSTD_estimateCStreamSize    AOCL_LLC_ZSTD_estimateCStreamSize
#  define ZSTD_estimateCStreamSize_usingCCtxParams    AOCL_LLC_ZSTD_estimateCStreamSize_usingCCtxParams
#  define ZSTD_estimateCStreamSize_usingCParams    AOCL_LLC_ZSTD_estimateCStreamSize_usingCParams
#  define ZSTD_estimateDCtxSize    AOCL_LLC_ZSTD_estimateDCtxSize
#  define ZSTD_estimateDDictSize    AOCL_LLC_ZSTD_estimateDDictSize
#  define ZSTD_estimateDStreamSize    AOCL_LLC_ZSTD_estimateDStreamSize
#  define ZSTD_estimateDStreamSize_fromFrame    AOCL_LLC_ZSTD_estimateDStreamSize_fromFrame
#  define ZSTD_fillDoubleHashTable    AOCL_LLC_ZSTD_fillDoubleHashTable
#  define ZSTD_fillHashTable    AOCL_LLC_ZSTD_fillHashTable
#  define ZSTD_findDecompressedSize    AOCL_LLC_ZSTD_findDecompressedSize
#  define ZSTD_findFrameCompressedSize    AOCL_LLC_ZSTD_findFrameCompressedSize
#  define ZSTD_flushStream    AOCL_LLC_ZSTD_flushStream
#  define ZSTD_frameHeaderSize    AOCL_LLC_ZSTD_frameHeaderSize
#  define ZSTD_freeCCtx    AOCL_LLC_ZSTD_freeCCtx
#  define ZSTD_freeCCtxParams    AOCL_LLC_ZSTD_freeCCtxParams
#  define ZSTD_freeCDict    AOCL_LLC_ZSTD_freeCDict
#  define ZSTD_freeCStream    AOCL_LLC_ZSTD_freeCStream
#  define ZSTD_freeDCtx    AOCL_LLC_ZSTD_freeDCtx
#  define ZSTD_freeDDict    AOCL_LLC_ZSTD_freeDDict
#  define ZSTD_freeDStream    AOCL_LLC_ZSTD_freeDStream
#  define ZSTD_fseBitCost    AOCL_LLC_ZSTD_fseBitCost
#  define ZSTD_generateSequences    AOCL_LLC_ZSTD_generateSequences
#  define ZSTD_get1BlockSummary    AOCL_LLC_ZSTD_get1BlockSummary
#  define ZSTD_getBlockSize    AOCL_LLC_ZSTD_getBlockSize
#  define ZSTD_getcBlockSize    AOCL_LLC_ZSTD_getcBlockSize
#  define ZSTD_getCParams    AOCL_LLC_ZSTD_getCParams
#  define ZSTD_getCParamsFromCCtxParams    AOCL_LLC_ZSTD_getCParamsFromCCtxParams
#  define ZSTD_getCParamsFromCDict    AOCL_LLC_ZSTD_getCParamsFromCDict
#  define ZSTD_getDecompressedSize    AOCL_LLC_ZSTD_getDecompressedSize
#  define ZSTD_getDictID_fromCDict    AOCL_LLC_ZSTD_getDictID_fromCDict
#  define ZSTD_getDictID_fromDDict    AOCL_LLC_ZSTD_getDictID_fromDDict
#  define ZSTD_getDictID_fromDict    AOCL_LLC_ZSTD_getDictID_fromDict
#  define ZSTD_getDictID_fromFrame    AOCL_LLC_ZSTD_getDictID_fromFrame
#  define ZSTD_getErrorCode    AOCL_LLC_ZSTD_getErrorCode
#  define ZSTD_getErrorName    AOCL_LLC_ZSTD_getErrorName
#  define ZSTD_getErrorString    AOCL_LLC_ZSTD_getErrorString
#  define ZSTD_getFrameContentSize    AOCL_LLC_ZSTD_getFrameContentSize
#  define ZSTD_getFrameHeader    AOCL_LLC_ZSTD_getFrameHeader
#  define ZSTD_getFrameHeader_advanced    AOCL_LLC_ZSTD_getFrameHeader_advanced
#  define ZSTD_getFrameProgression    AOCL_LLC_ZSTD_getFrameProgression
#  define ZSTD_getParams    AOCL_LLC_ZSTD_getParams
#  define ZSTD_getSeqStore    AOCL_LLC_ZSTD_getSeqStore
#  define ZSTD_initCStream    AOCL_LLC_ZSTD_initCStream
#  define ZSTD_initCStream_advanced    AOCL_LLC_ZSTD_initCStream_advanced
#  define ZSTD_initCStream_internal    AOCL_LLC_ZSTD_initCStream_internal
#  define ZSTD_initCStream_srcSize    AOCL_LLC_ZSTD_initCStream_srcSize
#  define ZSTD_initCStream_usingCDict    AOCL_LLC_ZSTD_initCStream_usingCDict
#  define ZSTD_initCStream_usingCDict_advanced    AOCL_LLC_ZSTD_initCStream_usingCDict_advanced
#  define ZSTD_initCStream_usingDict    AOCL_LLC_ZSTD_initCStream_usingDict
#  define ZSTD_initDStream    AOCL_LLC_ZSTD_initDStream
#  define ZSTD_initDStream_usingDDict    AOCL_LLC_ZSTD_initDStream_usingDDict
#  define ZSTD_initDStream_usingDict    AOCL_LLC_ZSTD_initDStream_usingDict
#  define ZSTD_initStaticCCtx    AOCL_LLC_ZSTD_initStaticCCtx
#  define ZSTD_initStaticCDict    AOCL_LLC_ZSTD_initStaticCDict
#  define ZSTD_initStaticCStream    AOCL_LLC_ZSTD_initStaticCStream
#  define ZSTD_initStaticDCtx    AOCL_LLC_ZSTD_initStaticDCtx
#  define ZSTD_initStaticDDict    AOCL_LLC_ZSTD_initStaticDDict
#  define ZSTD_initStaticDStream    AOCL_LLC_ZSTD_initStaticDStream
#  define ZSTD_insertAndFindFirstIndex    AOCL_LLC_ZSTD_insertAndFindFirstIndex
#  define ZSTD_insertBlock    AOCL_LLC_ZSTD_insertBlock
#  define ZSTD_invalidateRepCodes    AOCL_LLC_ZSTD_invalidateRepCodes
#  define ZSTD_isError    AOCL_LLC_ZSTD_isError
#  define ZSTD_isFrame    AOCL_LLC_ZSTD_isFrame
#  define ZSTD_isSkippableFrame    AOCL_LLC_ZSTD_isSkippableFrame
#  define ZSTD_ldm_adjustParameters    AOCL_LLC_ZSTD_ldm_adjustParameters
#  define ZSTD_ldm_blockCompress    AOCL_LLC_ZSTD_ldm_blockCompress
#  define ZSTD_ldm_fillHashTable    AOCL_LLC_ZSTD_ldm_fillHashTable
#  define ZSTD_ldm_generateSequences    AOCL_LLC_ZSTD_ldm_generateSequences
#  define ZSTD_ldm_getMaxNbSeq    AOCL_LLC_ZSTD_ldm_getMaxNbSeq
#  define ZSTD_ldm_getTableSize    AOCL_LLC_ZSTD_ldm_getTableSize
#  define ZSTD_ldm_skipRawSeqStoreBytes    AOCL_LLC_ZSTD_ldm_skipRawSeqStoreBytes
#  define ZSTD_ldm_skipSequences    AOCL_LLC_ZSTD_ldm_skipSequences
#  define ZSTD_loadCEntropy    AOCL_LLC_ZSTD_loadCEntropy
#  define ZSTD_loadDEntropy    AOCL_LLC_ZSTD_loadDEntropy
#  define ZSTD_maxCLevel    AOCL_LLC_ZSTD_maxCLevel
#  define ZSTD_mergeBlockDelimiters    AOCL_LLC_ZSTD_mergeBlockDelimiters
#  define ZSTD_minCLevel    AOCL_LLC_ZSTD_minCLevel
#  define ZSTDMT_compressStream_generic    AOCL_LLC_ZSTDMT_compressStream_generic
#  define ZSTDMT_createCCtx_advanced    AOCL_LLC_ZSTDMT_createCCtx_advanced
#  define ZSTDMT_freeCCtx    AOCL_LLC_ZSTDMT_freeCCtx
#  define ZSTDMT_getFrameProgression    AOCL_LLC_ZSTDMT_getFrameProgression
#  define ZSTDMT_initCStream_internal    AOCL_LLC_ZSTDMT_initCStream_internal
#  define ZSTDMT_nextInputSizeHint    AOCL_LLC_ZSTDMT_nextInputSizeHint
#  define ZSTDMT_sizeof_CCtx    AOCL_LLC_ZSTDMT_sizeof_CCtx
#  define ZSTDMT_toFlushNow    AOCL_LLC_ZSTDMT_toFlushNow
#  define ZSTDMT_updateCParams_whileCompressing    AOCL_LLC_ZSTDMT_updateCParams_whileCompressing
#  define ZSTD_nextInputType    AOCL_LLC_ZSTD_nextInputType
#  define ZSTD_nextSrcSizeToDecompress    AOCL_LLC_ZSTD_nextSrcSizeToDecompress
#  define ZSTD_noCompressLiterals    AOCL_LLC_ZSTD_noCompressLiterals
#  define ZSTD_readSkippableFrame    AOCL_LLC_ZSTD_readSkippableFrame
#  define ZSTD_referenceExternalSequences    AOCL_LLC_ZSTD_referenceExternalSequences
#  define ZSTD_registerSequenceProducer    AOCL_LLC_ZSTD_registerSequenceProducer
#  define ZSTD_reset_compressedBlockState    AOCL_LLC_ZSTD_reset_compressedBlockState
#  define ZSTD_resetCStream    AOCL_LLC_ZSTD_resetCStream
#  define ZSTD_resetDStream    AOCL_LLC_ZSTD_resetDStream
#  define ZSTD_resetSeqStore    AOCL_LLC_ZSTD_resetSeqStore
#  define ZSTD_row_update    AOCL_LLC_ZSTD_row_update
#  define ZSTD_selectBlockCompressor    AOCL_LLC_ZSTD_selectBlockCompressor
#  define ZSTD_selectEncodingType    AOCL_LLC_ZSTD_selectEncodingType
#  define ZSTD_seqToCodes    AOCL_LLC_ZSTD_seqToCodes
#  define ZSTD_sequenceBound    AOCL_LLC_ZSTD_sequenceBound
#  define ZSTD_sizeof_CCtx    AOCL_LLC_ZSTD_sizeof_CCtx
#  define ZSTD_sizeof_CDict    AOCL_LLC_ZSTD_sizeof_CDict
#  define ZSTD_sizeof_CStream    AOCL_LLC_ZSTD_sizeof_CStream
#  define ZSTD_sizeof_DCtx    AOCL_LLC_ZSTD_sizeof_DCtx
#  define ZSTD_sizeof_DDict    AOCL_LLC_ZSTD_sizeof_DDict
#  define ZSTD_sizeof_DStream    AOCL_LLC_ZSTD_sizeof_DStream
#  define ZSTD_splitBlock    AOCL_LLC_ZSTD_splitBlock
#  define ZSTD_toFlushNow    AOCL_LLC_ZSTD_toFlushNow
#  define ZSTD_updateTree    AOCL_LLC_ZSTD_updateTree
#  define ZSTD_versionNumber    AOCL_LLC_ZSTD_versionNumber
#  define ZSTD_versionString    AOCL_LLC_ZSTD_versionString
#  define ZSTD_writeLastEmptyBlock    AOCL_LLC_ZSTD_writeLastEmptyBlock
#  define ZSTD_writeSkippableFrame    AOCL_LLC_ZSTD_writeSkippableFrame
#  define ZSTD_XXH32    AOCL_LLC_ZSTD_XXH32
#  define ZSTD_XXH32_canonicalFromHash    AOCL_LLC_ZSTD_XXH32_canonicalFromHash
#  define ZSTD_XXH32_copyState    AOCL_LLC_ZSTD_XXH32_copyState
#  define ZSTD_XXH32_createState    AOCL_LLC_ZSTD_XXH32_createState
#  define ZSTD_XXH32_digest    AOCL_LLC_ZSTD_XXH32_digest
#  define ZSTD_XXH32_freeState    AOCL_LLC_ZSTD_XXH32_freeState
#  define ZSTD_XXH32_hashFromCanonical    AOCL_LLC_ZSTD_XXH32_hashFromCanonical
#  define ZSTD_XXH32_reset    AOCL_LLC_ZSTD_XXH32_reset
#  define ZSTD_XXH32_update    AOCL_LLC_ZSTD_XXH32_update
#  define ZSTD_XXH64    AOCL_LLC_ZSTD_XXH64
#  define ZSTD_XXH64_canonicalFromHash    AOCL_LLC_ZSTD_XXH64_canonicalFromHash
#  define ZSTD_XXH64_copyState    AOCL_LLC_ZSTD_XXH64_copyState
#  define ZSTD_XXH64_createState    AOCL_LLC_ZSTD_XXH64_createState
#  define ZSTD_XXH64_digest    AOCL_LLC_ZSTD_XXH64_digest
#  define ZSTD_XXH64_freeState    AOCL_LLC_ZSTD_XXH64_freeState
#  define ZSTD_XXH64_hashFromCanonical    AOCL_LLC_ZSTD_XXH64_hashFromCanonical
#  define ZSTD_XXH64_reset    AOCL_LLC_ZSTD_XXH64_reset
#  define ZSTD_XXH64_update    AOCL_LLC_ZSTD_XXH64_update
#  define ZSTD_XXH_versionNumber    AOCL_LLC_ZSTD_XXH_versionNumber

/* Additional ZSTD internal symbols that need prefixing to avoid conflicts with reference ZSTD */
#  define ZSTD_createThreadPool    AOCL_LLC_ZSTD_createThreadPool
#  define ZSTD_freeThreadPool    AOCL_LLC_ZSTD_freeThreadPool
#  define g_debuglevel    AOCL_LLC_g_debuglevel
#  define g_ZSTD_threading_useless_symbol    AOCL_LLC_g_ZSTD_threading_useless_symbol

/**** ZSTD ****/

#endif /* AOCL_LLC_PREFIX */
#endif /* __AOCL_PREFIX_H */

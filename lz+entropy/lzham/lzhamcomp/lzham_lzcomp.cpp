// File: lzham_lzcomp.cpp
// LZHAM is in the Public Domain. Please see the Public Domain declaration at the end of include/lzham.h
#include "lzham_core.h"
#include "lzham.h"
#include "lzham_comp.h"
#include "lzham_lzcomp_internal.h"

using namespace lzham;

namespace lzham
{
   struct lzham_compress_state
   {
      // task_pool requires 8 or 16 alignment
      task_pool m_tp;
      lzcompressor m_compressor;

      uint m_dict_size_log2;

      const uint8 *m_pIn_buf;
      size_t *m_pIn_buf_size;
      uint8 *m_pOut_buf;
      size_t *m_pOut_buf_size;

      size_t m_comp_data_ofs;

      bool m_finished_compression;

      lzham_compress_params m_params;

      lzham_compress_status_t m_status;
   };

   static lzham_compress_status_t create_internal_init_params(lzcompressor::init_params &internal_params, const lzham_compress_params *pParams)
   {
      if ((pParams->m_dict_size_log2 < CLZBase::cMinDictSizeLog2) || (pParams->m_dict_size_log2 > CLZBase::cMaxDictSizeLog2))
         return LZHAM_COMP_STATUS_INVALID_PARAMETER;
			     
      internal_params.m_dict_size_log2 = pParams->m_dict_size_log2;

      if (pParams->m_max_helper_threads < 0)
         internal_params.m_max_helper_threads = lzham_get_max_helper_threads();
      else
         internal_params.m_max_helper_threads = pParams->m_max_helper_threads;
      internal_params.m_max_helper_threads = LZHAM_MIN(LZHAM_MAX_HELPER_THREADS, internal_params.m_max_helper_threads);

      internal_params.m_lzham_compress_flags = pParams->m_compress_flags;

      if (pParams->m_num_seed_bytes)
      {
         if ((!pParams->m_pSeed_bytes) || (pParams->m_num_seed_bytes > (1U << pParams->m_dict_size_log2)))
            return LZHAM_COMP_STATUS_INVALID_PARAMETER;

         internal_params.m_num_seed_bytes = pParams->m_num_seed_bytes;
         internal_params.m_pSeed_bytes = pParams->m_pSeed_bytes;
      }

      switch (pParams->m_level)
      {
         case LZHAM_COMP_LEVEL_FASTEST:   internal_params.m_compression_level = cCompressionLevelFastest; break;
         case LZHAM_COMP_LEVEL_FASTER:    internal_params.m_compression_level = cCompressionLevelFaster; break;
         case LZHAM_COMP_LEVEL_DEFAULT:   internal_params.m_compression_level = cCompressionLevelDefault; break;
         case LZHAM_COMP_LEVEL_BETTER:    internal_params.m_compression_level = cCompressionLevelBetter; break;
         case LZHAM_COMP_LEVEL_UBER:      internal_params.m_compression_level = cCompressionLevelUber; break;
         default:
            return LZHAM_COMP_STATUS_INVALID_PARAMETER;
      };

		if (pParams->m_table_max_update_interval || pParams->m_table_update_interval_slow_rate)
		{
			internal_params.m_table_max_update_interval = pParams->m_table_max_update_interval;
			internal_params.m_table_update_interval_slow_rate = pParams->m_table_update_interval_slow_rate;
		}
		else 
		{
			uint rate = pParams->m_table_update_rate;
			if (!rate)
				rate = LZHAM_DEFAULT_TABLE_UPDATE_RATE;
			rate = math::clamp<uint>(rate, 1, LZHAM_FASTEST_TABLE_UPDATE_RATE) - 1;
			internal_params.m_table_max_update_interval = g_table_update_settings[rate].m_max_update_interval;
			internal_params.m_table_update_interval_slow_rate = g_table_update_settings[rate].m_slow_rate;
		}

      return LZHAM_COMP_STATUS_SUCCESS;
   }

   lzham_compress_state_ptr LZHAM_CDECL lzham_lib_compress_init(const lzham_compress_params *pParams)
   {
      if ((!pParams) || (pParams->m_struct_size != sizeof(lzham_compress_params)))
         return NULL;

      if ((pParams->m_dict_size_log2 < CLZBase::cMinDictSizeLog2) || (pParams->m_dict_size_log2 > CLZBase::cMaxDictSizeLog2))
         return NULL;

      lzcompressor::init_params internal_params;
      lzham_compress_status_t status = create_internal_init_params(internal_params, pParams);
      if (status != LZHAM_COMP_STATUS_SUCCESS)
         return NULL;

      lzham_compress_state *pState = lzham_new<lzham_compress_state>();
      if (!pState)
         return NULL;

      pState->m_params = *pParams;

      pState->m_pIn_buf = NULL;
      pState->m_pIn_buf_size = NULL;
      pState->m_pOut_buf = NULL;
      pState->m_pOut_buf_size = NULL;
      pState->m_status = LZHAM_COMP_STATUS_NOT_FINISHED;
      pState->m_comp_data_ofs = 0;
      pState->m_finished_compression = false;

      if (internal_params.m_max_helper_threads)
      {
         if (!pState->m_tp.init(internal_params.m_max_helper_threads))
         {
            lzham_delete(pState);
            return NULL;
         }
         if (pState->m_tp.get_num_threads() >= internal_params.m_max_helper_threads)
         {
            internal_params.m_pTask_pool = &pState->m_tp;
         }
         else
         {
            internal_params.m_max_helper_threads = 0;
         }
      }

      if (!pState->m_compressor.init(internal_params))
      {
         lzham_delete(pState);
         return NULL;
      }

      return pState;
   }

   lzham_compress_state_ptr LZHAM_CDECL lzham_lib_compress_reinit(lzham_compress_state_ptr p)
   {
      lzham_compress_state *pState = static_cast<lzham_compress_state*>(p);
      if (pState)
      {
         if (!pState->m_compressor.reset())
            return NULL;

         pState->m_pIn_buf = NULL;
         pState->m_pIn_buf_size = NULL;
         pState->m_pOut_buf = NULL;
         pState->m_pOut_buf_size = NULL;
         pState->m_status = LZHAM_COMP_STATUS_NOT_FINISHED;
         pState->m_comp_data_ofs = 0;
         pState->m_finished_compression = false;
      }

      return pState;
   }

   lzham_uint32 LZHAM_CDECL lzham_lib_compress_deinit(lzham_compress_state_ptr p)
   {
      lzham_compress_state *pState = static_cast<lzham_compress_state *>(p);
      if (!pState)
         return 0;

      uint32 adler32 = pState->m_compressor.get_src_adler32();

      lzham_delete(pState);

      return adler32;
   }

   lzham_compress_status_t LZHAM_CDECL lzham_lib_compress(
      lzham_compress_state_ptr p,
      const lzham_uint8 *pIn_buf, size_t *pIn_buf_size,
      lzham_uint8 *pOut_buf, size_t *pOut_buf_size,
      lzham_bool no_more_input_bytes_flag)
   {
      return lzham_lib_compress2(p, pIn_buf, pIn_buf_size, pOut_buf, pOut_buf_size, no_more_input_bytes_flag ? LZHAM_FINISH : LZHAM_NO_FLUSH);
   }

   lzham_compress_status_t LZHAM_CDECL lzham_lib_compress2(
      lzham_compress_state_ptr p,
      const lzham_uint8 *pIn_buf, size_t *pIn_buf_size,
      lzham_uint8 *pOut_buf, size_t *pOut_buf_size,
      lzham_flush_t flush_type)
   {
      lzham_compress_state *pState = static_cast<lzham_compress_state*>(p);

      if ((!pState) || (!pState->m_params.m_dict_size_log2) || (pState->m_status >= LZHAM_COMP_STATUS_FIRST_SUCCESS_OR_FAILURE_CODE) || (!pIn_buf_size) || (!pOut_buf_size))
         return LZHAM_COMP_STATUS_INVALID_PARAMETER;

      if ((*pIn_buf_size) && (!pIn_buf))
         return LZHAM_COMP_STATUS_INVALID_PARAMETER;

      if ((!*pOut_buf_size) || (!pOut_buf))
         return LZHAM_COMP_STATUS_INVALID_PARAMETER;

      byte_vec &comp_data = pState->m_compressor.get_compressed_data();
      size_t num_bytes_written_to_out_buf = 0;
      if (pState->m_comp_data_ofs < comp_data.size())
      {
         size_t n = LZHAM_MIN(comp_data.size() - pState->m_comp_data_ofs, *pOut_buf_size);

         memcpy(pOut_buf, comp_data.get_ptr() + pState->m_comp_data_ofs, n);

         pState->m_comp_data_ofs += n;

         const bool has_no_more_output = (pState->m_comp_data_ofs >= comp_data.size());
         if (has_no_more_output)
         {
            pOut_buf += n;
            *pOut_buf_size -= n;
            num_bytes_written_to_out_buf += n;
         }
         else
         {
            *pIn_buf_size = 0;
            *pOut_buf_size = n;
            pState->m_status = LZHAM_COMP_STATUS_HAS_MORE_OUTPUT;
            return pState->m_status;
         }
      }

      comp_data.try_resize(0);
      pState->m_comp_data_ofs = 0;

      if (pState->m_finished_compression)
      {
         if ((*pIn_buf_size) || (flush_type != LZHAM_FINISH))
         {
            pState->m_status = LZHAM_COMP_STATUS_INVALID_PARAMETER;
            return pState->m_status;
         }

         *pIn_buf_size = 0;
         *pOut_buf_size = num_bytes_written_to_out_buf;

         pState->m_status = LZHAM_COMP_STATUS_SUCCESS;
         return pState->m_status;
      }

      const size_t cMaxBytesToPutPerIteration = 4*1024*1024;
      size_t bytes_to_put = LZHAM_MIN(cMaxBytesToPutPerIteration, *pIn_buf_size);
      const bool consumed_entire_input_buf = (bytes_to_put == *pIn_buf_size);

      if (bytes_to_put)
      {
         if (!pState->m_compressor.put_bytes(pIn_buf, (uint)bytes_to_put))
         {
            *pIn_buf_size = 0;
            *pOut_buf_size = num_bytes_written_to_out_buf;
            pState->m_status = LZHAM_COMP_STATUS_FAILED;
            return pState->m_status;
         }
      }

      if ((consumed_entire_input_buf) && (flush_type != LZHAM_NO_FLUSH))
      {
         if ((flush_type == LZHAM_SYNC_FLUSH) || (flush_type == LZHAM_FULL_FLUSH) || (flush_type == LZHAM_TABLE_FLUSH))
         {
            if (!pState->m_compressor.flush(flush_type))
            {
               *pIn_buf_size = 0;
               *pOut_buf_size = num_bytes_written_to_out_buf;
               pState->m_status = LZHAM_COMP_STATUS_FAILED;
               return pState->m_status;
            }
         }
         else if (!pState->m_finished_compression)
         {
            if (!pState->m_compressor.put_bytes(NULL, 0))
            {
               *pIn_buf_size = 0;
               *pOut_buf_size = num_bytes_written_to_out_buf;
               pState->m_status = LZHAM_COMP_STATUS_FAILED;
               return pState->m_status;
            }
            pState->m_finished_compression = true;
         }
      }

      size_t num_comp_bytes_to_output = LZHAM_MIN(comp_data.size() - pState->m_comp_data_ofs, *pOut_buf_size);
      if (num_comp_bytes_to_output)
      {
         memcpy(pOut_buf, comp_data.get_ptr() + pState->m_comp_data_ofs, num_comp_bytes_to_output);

         pState->m_comp_data_ofs += num_comp_bytes_to_output;
      }

      *pIn_buf_size = bytes_to_put;
      *pOut_buf_size = num_bytes_written_to_out_buf + num_comp_bytes_to_output;

      const bool has_no_more_output = (pState->m_comp_data_ofs >= comp_data.size());
      if ((has_no_more_output) && (flush_type == LZHAM_FINISH) && (pState->m_finished_compression))
         pState->m_status = LZHAM_COMP_STATUS_SUCCESS;
      else if ((has_no_more_output) && (consumed_entire_input_buf) && (flush_type == LZHAM_NO_FLUSH))
         pState->m_status = LZHAM_COMP_STATUS_NEEDS_MORE_INPUT;
      else
         pState->m_status = has_no_more_output ? LZHAM_COMP_STATUS_NOT_FINISHED : LZHAM_COMP_STATUS_HAS_MORE_OUTPUT;

      return pState->m_status;
   }

   lzham_compress_status_t LZHAM_CDECL lzham_lib_compress_memory(const lzham_compress_params *pParams, lzham_uint8* pDst_buf, size_t *pDst_len, const lzham_uint8* pSrc_buf, size_t src_len, lzham_uint32 *pAdler32)
   {
      if ((!pParams) || (!pDst_len))
         return LZHAM_COMP_STATUS_INVALID_PARAMETER;

      if (src_len)
      {
         if (!pSrc_buf)
            return LZHAM_COMP_STATUS_INVALID_PARAMETER;
      }

      if (sizeof(size_t) > sizeof(uint32))
      {
         if (src_len > LZHAM_UINT32_MAX)
            return LZHAM_COMP_STATUS_INVALID_PARAMETER;
      }

      lzcompressor::init_params internal_params;
      lzham_compress_status_t status = create_internal_init_params(internal_params, pParams);
      if (status != LZHAM_COMP_STATUS_SUCCESS)
         return status;

      task_pool *pTP = NULL;
      if (internal_params.m_max_helper_threads)
      {
         pTP = lzham_new<task_pool>();
         if (!pTP->init(internal_params.m_max_helper_threads))
            return LZHAM_COMP_STATUS_FAILED;

         internal_params.m_pTask_pool = pTP;
      }

      lzcompressor *pCompressor = lzham_new<lzcompressor>();
      if (!pCompressor)
      {
         lzham_delete(pTP);
         return LZHAM_COMP_STATUS_FAILED;
      }

      if (!pCompressor->init(internal_params))
      {
         lzham_delete(pTP);
         lzham_delete(pCompressor);
         return LZHAM_COMP_STATUS_INVALID_PARAMETER;
      }

      if (src_len)
      {
         if (!pCompressor->put_bytes(pSrc_buf, static_cast<uint32>(src_len)))
         {
            *pDst_len = 0;
            lzham_delete(pTP);
            lzham_delete(pCompressor);
            return LZHAM_COMP_STATUS_FAILED;
         }
      }

      if (!pCompressor->put_bytes(NULL, 0))
      {
         *pDst_len = 0;
         lzham_delete(pTP);
         lzham_delete(pCompressor);
         return LZHAM_COMP_STATUS_FAILED;
      }

      const byte_vec &comp_data = pCompressor->get_compressed_data();

      size_t dst_buf_size = *pDst_len;
      *pDst_len = comp_data.size();

      if (pAdler32)
         *pAdler32 = pCompressor->get_src_adler32();

      if (comp_data.size() > dst_buf_size)
      {
         lzham_delete(pTP);
         lzham_delete(pCompressor);
         return LZHAM_COMP_STATUS_OUTPUT_BUF_TOO_SMALL;
      }

      memcpy(pDst_buf, comp_data.get_ptr(), comp_data.size());

      lzham_delete(pTP);
      lzham_delete(pCompressor);
      return LZHAM_COMP_STATUS_SUCCESS;
   }

   // ----------------- zlib-style API's

   int lzham_lib_z_deflateInit(lzham_z_streamp pStream, int level)
   {
      return lzham_lib_z_deflateInit2(pStream, level, LZHAM_Z_LZHAM, LZHAM_Z_DEFAULT_WINDOW_BITS, 9, LZHAM_Z_DEFAULT_STRATEGY);
   }

   int lzham_lib_z_deflateInit2(lzham_z_streamp pStream, int level, int method, int window_bits, int mem_level, int strategy)
   {
      LZHAM_NOTE_UNUSED(strategy);

      if (!pStream)
         return LZHAM_Z_STREAM_ERROR;
      if ((mem_level < 1) || (mem_level > 9))
         return LZHAM_Z_PARAM_ERROR;
      if ((method != LZHAM_Z_DEFLATED) && (method != LZHAM_Z_LZHAM))
         return LZHAM_Z_PARAM_ERROR;

      if (level == LZHAM_Z_DEFAULT_COMPRESSION)
         level = 9;

      if (method == LZHAM_Z_DEFLATED)
      {
         // Force Deflate to LZHAM with default window_bits.
         method = LZHAM_Z_LZHAM;
         window_bits = LZHAM_Z_DEFAULT_WINDOW_BITS;
      }

#ifdef LZHAM_Z_API_FORCE_WINDOW_BITS
      window_bits = LZHAM_Z_API_FORCE_WINDOW_BITS;
#endif

      int max_window_bits = LZHAM_64BIT_POINTERS ? LZHAM_MAX_DICT_SIZE_LOG2_X64 : LZHAM_MAX_DICT_SIZE_LOG2_X86;
      if ((labs(window_bits) < LZHAM_MIN_DICT_SIZE_LOG2) || (labs(window_bits) > max_window_bits))
         return LZHAM_Z_PARAM_ERROR;

      lzham_compress_params comp_params;

      utils::zero_object(comp_params);
      comp_params.m_struct_size = sizeof(lzham_compress_params);

      comp_params.m_level = LZHAM_COMP_LEVEL_UBER;
      if (level <= 1)
         comp_params.m_level = LZHAM_COMP_LEVEL_FASTEST;
      else if (level <= 3)
         comp_params.m_level = LZHAM_COMP_LEVEL_FASTER;
      else if (level <= 5)
         comp_params.m_level = LZHAM_COMP_LEVEL_DEFAULT;
      else if (level <= 7)
         comp_params.m_level = LZHAM_COMP_LEVEL_BETTER;

      if (level == 10)
         comp_params.m_compress_flags |= LZHAM_COMP_FLAG_EXTREME_PARSING;

      // Use all CPU's. TODO: This is not always the best idea depending on the dictionary size and the # of bytes to compress.
      comp_params.m_max_helper_threads = -1;

      comp_params.m_dict_size_log2 = static_cast<lzham_uint32>(labs(window_bits));

      if (window_bits > 0)
         comp_params.m_compress_flags |= LZHAM_COMP_FLAG_WRITE_ZLIB_STREAM;

      pStream->data_type = 0;
      pStream->adler = LZHAM_Z_ADLER32_INIT;
      pStream->msg = NULL;
      pStream->reserved = 0;
      pStream->total_in = 0;
      pStream->total_out = 0;

      lzham_compress_state_ptr pComp = lzham_lib_compress_init(&comp_params);
      if (!pComp)
         return LZHAM_Z_PARAM_ERROR;

      pStream->state = (struct lzham_z_internal_state *)pComp;

      return LZHAM_Z_OK;
   }

   int lzham_lib_z_deflateReset(lzham_z_streamp pStream)
   {
      if (!pStream)
         return LZHAM_Z_STREAM_ERROR;

      lzham_compress_state_ptr pComp = (lzham_compress_state_ptr)pStream->state;
      if (!pComp)
         return LZHAM_Z_STREAM_ERROR;

      pComp = lzham_lib_compress_reinit(pComp);
      if (!pComp)
         return LZHAM_Z_STREAM_ERROR;

      pStream->state = (struct lzham_z_internal_state *)pComp;

      return LZHAM_Z_OK;
   }

   int lzham_lib_z_deflate(lzham_z_streamp pStream, int flush)
   {
      if ((!pStream) || (!pStream->state) || (flush < 0) || (flush > LZHAM_Z_FINISH) || (!pStream->next_out))
         return LZHAM_Z_STREAM_ERROR;

      if (!pStream->avail_out)
         return LZHAM_Z_BUF_ERROR;

      if (flush == LZHAM_Z_PARTIAL_FLUSH)
         flush = LZHAM_Z_SYNC_FLUSH;

      int lzham_status = LZHAM_Z_OK;
      lzham_z_ulong orig_total_in = pStream->total_in, orig_total_out = pStream->total_out;
      for ( ; ; )
      {
         size_t in_bytes = pStream->avail_in, out_bytes = pStream->avail_out;

         lzham_compress_state_ptr pComp = (lzham_compress_state_ptr)pStream->state;
         lzham_compress_state *pState = static_cast<lzham_compress_state*>(pComp);

         lzham_compress_status_t status = lzham_lib_compress2(
            pComp,
            pStream->next_in, &in_bytes,
            pStream->next_out, &out_bytes,
            (lzham_flush_t)flush);

         pStream->next_in += (uint)in_bytes;
         pStream->avail_in -= (uint)in_bytes;
         pStream->total_in += (uint)in_bytes;

         pStream->next_out += (uint)out_bytes;
         pStream->avail_out -= (uint)out_bytes;
         pStream->total_out += (uint)out_bytes;

         pStream->adler = pState->m_compressor.get_src_adler32();

         if (status >= LZHAM_COMP_STATUS_FIRST_FAILURE_CODE)
         {
            lzham_status = LZHAM_Z_STREAM_ERROR;
            break;
         }
         else if (status == LZHAM_COMP_STATUS_SUCCESS)
         {
            lzham_status = LZHAM_Z_STREAM_END;
            break;
         }
         else if (!pStream->avail_out)
            break;
         else if ((!pStream->avail_in) && (flush != LZHAM_Z_FINISH))
         {
            if ((flush) || (pStream->total_in != orig_total_in) || (pStream->total_out != orig_total_out))
               break;
            return LZHAM_Z_BUF_ERROR; // Can't make forward progress without some input.
         }
      }
      return lzham_status;
   }

   int lzham_lib_z_deflateEnd(lzham_z_streamp pStream)
   {
      if (!pStream)
         return LZHAM_Z_STREAM_ERROR;

      lzham_compress_state_ptr pComp = (lzham_compress_state_ptr)pStream->state;
      if (pComp)
      {
         pStream->adler = lzham_lib_compress_deinit(pComp);

         pStream->state = NULL;
      }

      return LZHAM_Z_OK;
   }

   lzham_z_ulong lzham_lib_z_deflateBound(lzham_z_streamp pStream, lzham_z_ulong source_len)
   {
      LZHAM_NOTE_UNUSED(pStream);
      return 64 + source_len + ((source_len + 4095) / 4096) * 4;
   }

   int lzham_lib_z_compress2(unsigned char *pDest, lzham_z_ulong *pDest_len, const unsigned char *pSource, lzham_z_ulong source_len, int level)
   {
      int status;
      lzham_z_stream stream;
      memset(&stream, 0, sizeof(stream));

      // In case lzham_z_ulong is 64-bits (argh I hate longs).
      if ((source_len | *pDest_len) > 0xFFFFFFFFU)
         return LZHAM_Z_PARAM_ERROR;

      stream.next_in = pSource;
      stream.avail_in = (uint)source_len;
      stream.next_out = pDest;
      stream.avail_out = (uint)*pDest_len;

      status = lzham_lib_z_deflateInit(&stream, level);
      if (status != LZHAM_Z_OK)
         return status;

      status = lzham_lib_z_deflate(&stream, LZHAM_Z_FINISH);
      if (status != LZHAM_Z_STREAM_END)
      {
         lzham_lib_z_deflateEnd(&stream);
         return (status == LZHAM_Z_OK) ? LZHAM_Z_BUF_ERROR : status;
      }

      *pDest_len = stream.total_out;
      return lzham_lib_z_deflateEnd(&stream);
   }

   int lzham_lib_z_compress(unsigned char *pDest, lzham_z_ulong *pDest_len, const unsigned char *pSource, lzham_z_ulong source_len)
   {
      return lzham_lib_z_compress2(pDest, pDest_len, pSource, source_len, (int)LZHAM_Z_DEFAULT_COMPRESSION);
   }

   lzham_z_ulong lzham_lib_z_compressBound(lzham_z_ulong source_len)
   {
      return lzham_lib_z_deflateBound(NULL, source_len);
   }

} // namespace lzham

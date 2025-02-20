// File: lzham_mem.cpp
// See Copyright Notice and license at the end of include/lzham.h
#include "lzham_core.h"

#ifdef __APPLE__
   #include <malloc/malloc.h>
#elif defined(__FreeBSD__) || defined(__NetBSD__)
   #include <malloc_np.h>
   #if defined(__FreeBSD__)
      #define malloc(size) aligned_alloc((LZHAM_MIN_ALLOC_ALIGNMENT), (size))
   #endif
#else
   #include <malloc.h>
#endif

using namespace lzham;

#define LZHAM_MEM_STATS 0

#ifndef LZHAM_USE_WIN32_API
   #ifndef __APPLE__
      #define _msize malloc_usable_size
   #else
      #define _msize malloc_size
   #endif
#endif

namespace lzham
{
   #if LZHAM_64BIT_POINTERS
      const uint64 MAX_POSSIBLE_BLOCK_SIZE = 0x400000000ULL;
   #else
      const uint32 MAX_POSSIBLE_BLOCK_SIZE = 0x7FFF0000U;
   #endif

   #if LZHAM_MEM_STATS
      #if LZHAM_64BIT_POINTERS
         typedef atomic64_t mem_stat_t;
         #define LZHAM_MEM_COMPARE_EXCHANGE atomic_compare_exchange64
      #else
         typedef atomic32_t mem_stat_t;
         #define LZHAM_MEM_COMPARE_EXCHANGE atomic_compare_exchange32
      #endif

      static volatile atomic32_t g_total_blocks;
      static volatile mem_stat_t g_total_allocated;
      static volatile mem_stat_t g_max_allocated;

      static mem_stat_t update_total_allocated(int block_delta, mem_stat_t byte_delta)
      {
         atomic32_t cur_total_blocks;
         for ( ; ; )
         {
            cur_total_blocks = g_total_blocks;
            atomic32_t new_total_blocks = static_cast<atomic32_t>(cur_total_blocks + block_delta);
            LZHAM_ASSERT(new_total_blocks >= 0);
            if (atomic_compare_exchange32(&g_total_blocks, new_total_blocks, cur_total_blocks) == cur_total_blocks)
               break;
         }

         mem_stat_t cur_total_allocated, new_total_allocated;
         for ( ; ; )
         {
            cur_total_allocated = g_total_allocated;
            new_total_allocated = static_cast<mem_stat_t>(cur_total_allocated + byte_delta);
            LZHAM_ASSERT(new_total_allocated >= 0);
            if (LZHAM_MEM_COMPARE_EXCHANGE(&g_total_allocated, new_total_allocated, cur_total_allocated) == cur_total_allocated)
               break;
         }
         for ( ; ; )
         {
            mem_stat_t cur_max_allocated = g_max_allocated;
            mem_stat_t new_max_allocated = LZHAM_MAX(new_total_allocated, cur_max_allocated);
            if (LZHAM_MEM_COMPARE_EXCHANGE(&g_max_allocated, new_max_allocated, cur_max_allocated) == cur_max_allocated)
               break;
         }
         return new_total_allocated;
      }
   #endif // LZHAM_MEM_STATS

   static void* lzham_default_realloc(void* p, size_t size, size_t* pActual_size, lzham_bool movable, void* pUser_data)
   {
      LZHAM_NOTE_UNUSED(pUser_data);

      void* p_new;

      if (!p)
      {
         p_new = malloc(size);
         LZHAM_ASSERT( (reinterpret_cast<ptr_bits_t>(p_new) & (LZHAM_MIN_ALLOC_ALIGNMENT - 1)) == 0 );

         if (pActual_size)
            *pActual_size = p_new ? _msize(p_new) : 0;
      }
      else if (!size)
      {
         free(p);
         p_new = NULL;

         if (pActual_size)
            *pActual_size = 0;
      }
      else
      {
         void* p_final_block = p;
#ifdef WIN32
         p_new = _expand(p, size);
#else

         p_new = NULL;
#endif

         if (p_new)
         {
            LZHAM_ASSERT( (reinterpret_cast<ptr_bits_t>(p_new) & (LZHAM_MIN_ALLOC_ALIGNMENT - 1)) == 0 );
            p_final_block = p_new;
         }
         else if (movable)
         {
            p_new = realloc(p, size);

            if (p_new)
            {
               LZHAM_ASSERT( (reinterpret_cast<ptr_bits_t>(p_new) & (LZHAM_MIN_ALLOC_ALIGNMENT - 1)) == 0 );
               p_final_block = p_new;
            }
         }

         if (pActual_size)
            *pActual_size = _msize(p_final_block);
      }

      return p_new;
   }

   static size_t lzham_default_msize(void* p, void* pUser_data)
   {
      LZHAM_NOTE_UNUSED(pUser_data);
      return p ? _msize(p) : 0;
   }

   static lzham_realloc_func        g_pRealloc = lzham_default_realloc;
   static lzham_msize_func          g_pMSize   = lzham_default_msize;
   static void*                     g_pUser_data;

   static inline void lzham_mem_error(const char* p_msg)
   {
      lzham_assert(p_msg, __FILE__, __LINE__);
   }

   void* lzham_malloc(size_t size, size_t* pActual_size)
   {
      size = (size + sizeof(uint32) - 1U) & ~(sizeof(uint32) - 1U);
      if (!size)
         size = sizeof(uint32);

      if (size > MAX_POSSIBLE_BLOCK_SIZE)
      {
         lzham_mem_error("lzham_malloc: size too big");
         return NULL;
      }

      size_t actual_size = size;
      uint8* p_new = static_cast<uint8*>((*g_pRealloc)(NULL, size, &actual_size, true, g_pUser_data));

      if (pActual_size)
         *pActual_size = actual_size;

      if ((!p_new) || (actual_size < size))
      {
         lzham_mem_error("lzham_malloc: out of memory");
         return NULL;
      }

      LZHAM_ASSERT((reinterpret_cast<ptr_bits_t>(p_new) & (LZHAM_MIN_ALLOC_ALIGNMENT - 1)) == 0);

#if LZHAM_MEM_STATS
      update_total_allocated(1, static_cast<mem_stat_t>(actual_size));
#endif

      return p_new;
   }

   void* lzham_realloc(void* p, size_t size, size_t* pActual_size, bool movable)
   {
      if ((ptr_bits_t)p & (LZHAM_MIN_ALLOC_ALIGNMENT - 1))
      {
         lzham_mem_error("lzham_realloc: bad ptr");
         return NULL;
      }

      if (size > MAX_POSSIBLE_BLOCK_SIZE)
      {
         lzham_mem_error("lzham_malloc: size too big");
         return NULL;
      }

#if LZHAM_MEM_STATS
      size_t cur_size = p ? (*g_pMSize)(p, g_pUser_data) : 0;
#endif

      size_t actual_size = size;
      void* p_new = (*g_pRealloc)(p, size, &actual_size, movable, g_pUser_data);

      if (pActual_size)
         *pActual_size = actual_size;

      LZHAM_ASSERT((reinterpret_cast<ptr_bits_t>(p_new) & (LZHAM_MIN_ALLOC_ALIGNMENT - 1)) == 0);

#if LZHAM_MEM_STATS
      int num_new_blocks = 0;
      if (p)
      {
         if (!p_new)
            num_new_blocks = -1;
      }
      else if (p_new)
      {
         num_new_blocks = 1;
      }
      update_total_allocated(num_new_blocks, static_cast<mem_stat_t>(actual_size) - static_cast<mem_stat_t>(cur_size));
#endif

      return p_new;
   }

   void lzham_free(void* p)
   {
      if (!p)
         return;

      if (reinterpret_cast<ptr_bits_t>(p) & (LZHAM_MIN_ALLOC_ALIGNMENT - 1))
      {
         lzham_mem_error("lzham_free: bad ptr");
         return;
      }

#if LZHAM_MEM_STATS
      size_t cur_size = (*g_pMSize)(p, g_pUser_data);
      update_total_allocated(-1, -static_cast<mem_stat_t>(cur_size));
#endif

      (*g_pRealloc)(p, 0, NULL, true, g_pUser_data);
   }

   size_t lzham_msize(void* p)
   {
      if (!p)
         return 0;

      if (reinterpret_cast<ptr_bits_t>(p) & (LZHAM_MIN_ALLOC_ALIGNMENT - 1))
      {
         lzham_mem_error("lzham_msize: bad ptr");
         return 0;
      }

      return (*g_pMSize)(p, g_pUser_data);
   }

   void LZHAM_CDECL lzham_lib_set_memory_callbacks(lzham_realloc_func pRealloc, lzham_msize_func pMSize, void* pUser_data)
   {
      if ((!pRealloc) || (!pMSize))
      {
         g_pRealloc = lzham_default_realloc;
         g_pMSize = lzham_default_msize;
         g_pUser_data = NULL;
      }
      else
      {
         g_pRealloc = pRealloc;
         g_pMSize = pMSize;
         g_pUser_data = pUser_data;
      }
   }

   void lzham_print_mem_stats()
   {
#if LZHAM_MEM_STATS
      printf("Current blocks: %u, allocated: %I64u, max ever allocated: %I64i\n", g_total_blocks, (int64)g_total_allocated, (int64)g_max_allocated);
#endif
   }

} // namespace lzham


/*
Copyright 2011-2024 Frederic Langlet
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once
#ifndef _Memory_
#define _Memory_

#include <cstring>
#include "types.hpp"

#if defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__bsdi__) || defined(__DragonFly__) || defined(BSD)
	#include <machine/endian.h>
#elif defined(__linux__) || defined(__linux) || defined(__gnu_linux__)
	#include <endian.h>
#endif


namespace kanzi {

   static inline uint32 bswap32(uint32 x) {
   #if defined(__clang__)
		return __builtin_bswap32(x);
   #elif defined(__GNUC__) && (__GNUC__ >= 5)
		return __builtin_bswap32(x);
   #elif defined(_MSC_VER)
		return uint32(_byteswap_ulong(x));
   #elif defined(__i386__) || defined(__x86_64__)
		uint32 swapped_bytes;
		__asm__ volatile("bswap %0" : "=r"(swapped_bytes) : "0"(x));
		return swapped_bytes;
   #else // fallback
		return uint32((x >> 24) | ((x >> 8) & 0xFF00) | ((x << 8) & 0xFF0000) | (x << 24));
   #endif
	}


   static inline uint16 bswap16(uint16 x) {
   #if defined(__clang__)
		return __builtin_bswap16(x);
   #elif defined(__GNUC__) && (__GNUC__ >= 5)
		return __builtin_bswap16(x);
   #elif defined(_MSC_VER)
		return _byteswap_ushort(x);
   #else // fallback
		return uint16((x >> 8) | ((x & 0xFF) << 8));
   #endif
	}


   static inline uint64 bswap64(uint64 x) {
   #if defined(__clang__)
		return __builtin_bswap64(x);
   #elif defined(__GNUC__) && (__GNUC__ >= 5)
		return __builtin_bswap64(x);
   #elif defined(_MSC_VER)
		return uint64(_byteswap_uint64(x));
   #elif defined(__x86_64__)
		uint64 swapped_bytes;
		__asm__ volatile("bswapq %0" : "=r"(swapped_bytes) : "0"(x));
		return swapped_bytes;
   #else // fallback
		x = ((x & 0xFFFFFFFF00000000ull) >> 32) | ((x & 0x00000000FFFFFFFFull) << 32);
		x = ((x & 0xFFFF0000FFFF0000ull) >> 16) | ((x & 0x0000FFFF0000FFFFull) << 16);
		x = ((x & 0xFF00FF00FF00FF00ull) >> 8) | ((x & 0x00FF00FF00FF00FFull) << 8);
		return x;
   #endif
	}


	#ifndef IS_BIG_ENDIAN
		#if defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN || \
			   defined(__BIG_ENDIAN__) || \
			   defined(__ARMEB__) || \
			   defined(__THUMBEB__) || \
			   defined(__AARCH64EB__) || \
			   defined(_MIBSEB) || defined(__MIBSEB) || defined(__MIBSEB__)
			#define IS_BIG_ENDIAN 1
		#elif defined(_AIX) || defined(__hpux) || (defined(__sun) && defined(__sparc)) || defined(__OS400__) || defined(__MVS__)
			#define IS_BIG_ENDIAN 1
		#elif defined(__BYTE_ORDER) && __BYTE_ORDER == __LITTLE_ENDIAN|| defined(__LITTLE_ENDIAN__)
			#define IS_BIG_ENDIAN 0
		#elif defined(_WIN32)
			#define IS_BIG_ENDIAN 0
		#elif defined(__amd64) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
			#define IS_BIG_ENDIAN 0
		#endif
	#endif


   static inline bool isBigEndian() {
      #if defined(IS_BIG_ENDIAN)
         return IS_BIG_ENDIAN == 1;
      #else
         union { uint32 v; uint8 c[4]; } one = { 0x03020100 };
         return one.c[0] == 0;
         //const union { uint32 u; uint8 c[4]; } one = { 1 };
         //return one.c[3] == 1;
      #endif
   }


	class BigEndian {
	public:
		static int64 readLong64(const byte* p);
		static int32 readInt32(const byte* p);
		static int16 readInt16(const byte* p);

		static void writeLong64(byte* p, int64 val);
		static void writeInt32(byte* p, int32 val);
		static void writeInt16(byte* p, int16 val);
	};

	class LittleEndian {
	public:
		static int64 readLong64(const byte* p);
		static int32 readInt32(const byte* p);
		static int16 readInt16(const byte* p);

		static void writeLong64(byte* p, int64 val);
		static void writeInt32(byte* p, int32 val);
		static void writeInt16(byte* p, int16 val);
	};

	inline int64 BigEndian::readLong64(const byte* p)
	{
      uint64 val;

   #ifdef AGGRESSIVE_OPTIMIZATION
      // !!! unaligned data
		val = *(const uint64*)p;
   #else
		memcpy(&val, p, 8);
   #endif

   #if (!IS_BIG_ENDIAN)
      val = bswap64(val);
   #endif
      return int64(val);
	}


	inline int32 BigEndian::readInt32(const byte* p)
	{
      uint32 val;

   #ifdef AGGRESSIVE_OPTIMIZATION
      // !!! unaligned data
		val = *(const uint32*)p;
   #else
		memcpy(&val, p, 4);
   #endif

   #if (!IS_BIG_ENDIAN)
      val = bswap32(val);
   #endif
      return int32(val);
	}


	inline int16 BigEndian::readInt16(const byte* p)
	{
      uint16 val;

   #ifdef AGGRESSIVE_OPTIMIZATION
      // !!! unaligned data
		val = *(const uint16*)p;
   #else
		memcpy(&val, p, 2);
   #endif

   #if (!IS_BIG_ENDIAN)
      val = bswap16(val);
   #endif
      return int16(val);
	}


	inline void BigEndian::writeLong64(byte* p, int64 val)
	{
   #if (!IS_BIG_ENDIAN)
      val = int64(bswap64(uint64(val)));
   #endif

   #ifdef AGGRESSIVE_OPTIMIZATION
      // !!! unaligned data
		*(int64*)p = val;
   #else
		memcpy(p, &val, 8);
   #endif
	}


	inline void BigEndian::writeInt32(byte* p, int32 val)
	{
   #if (!IS_BIG_ENDIAN)
      val = int32(bswap32(uint32(val)));
   #endif

   #ifdef AGGRESSIVE_OPTIMIZATION
      // !!! unaligned data
		*(int32*)p = val;
   #else
		memcpy(p, &val, 4);
   #endif
	}


	inline void BigEndian::writeInt16(byte* p, int16 val)
	{
   #if (!IS_BIG_ENDIAN)
      val = int16(bswap16(uint16(val)));
   #endif

   #ifdef AGGRESSIVE_OPTIMIZATION
      // !!! unaligned data
		*(int16*)p = val;
   #else
		memcpy(p, &val, 2);
   #endif
	}


	inline int64 LittleEndian::readLong64(const byte* p)
	{
      uint64 val;

   #ifdef AGGRESSIVE_OPTIMIZATION
      // !!! unaligned data
		val = *(const uint64*)p;
   #else
		memcpy(&val, p, 8);
   #endif

   #if (IS_BIG_ENDIAN)
      val = bswap64(val);
   #endif
      return int64(val);
	}


	inline int32 LittleEndian::readInt32(const byte* p)
	{
      uint32 val;

   #ifdef AGGRESSIVE_OPTIMIZATION
      // !!! unaligned data
		val = *(const uint32*)p;
   #else
		memcpy(&val, p, 4);
   #endif

   #if (IS_BIG_ENDIAN)
      val = bswap32(val);
   #endif
      return int32(val);
	}


	inline int16 LittleEndian::readInt16(const byte* p)
	{
      uint16 val;

   #ifdef AGGRESSIVE_OPTIMIZATION
      // !!! unaligned data
		val = *(const uint16*)p;
   #else
		memcpy(&val, p, 2);
   #endif

   #if (IS_BIG_ENDIAN)
      val = bswap16(val);
   #endif
      return int16(val);
	}


	inline void LittleEndian::writeLong64(byte* p, int64 val)
	{
   #if (IS_BIG_ENDIAN)
      val = int64(bswap64(uint64(val)));
   #endif

   #ifdef AGGRESSIVE_OPTIMIZATION
      // !!! unaligned data
		*(int64*)p = val;
   #else
		memcpy(p, &val, 8);
   #endif
	}


	inline void LittleEndian::writeInt32(byte* p, int32 val)
	{
   #if (IS_BIG_ENDIAN)
      val = int32(bswap32(uint32(val)));
   #endif

   #ifdef AGGRESSIVE_OPTIMIZATION
      // !!! unaligned data
		*(int32*)p = val;
   #else
		memcpy(p, &val, 4);
   #endif
	}


	inline void LittleEndian::writeInt16(byte* p, int16 val)
	{
   #if (IS_BIG_ENDIAN)
      val = int16(bswap16(uint16(val)));
   #endif

   #ifdef AGGRESSIVE_OPTIMIZATION
      // !!! unaligned data
		*(int16*)p = val;
   #else
		memcpy(p, &val, 2);
   #endif
	}
}
#endif


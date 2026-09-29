// File: types.h
// LZHAM is in the Public Domain. Please see the Public Domain declaration at the end of include/lzham.h
#pragma once

namespace lzham
{
   typedef unsigned char      uint8;
   typedef signed char        int8;
   typedef unsigned char      uint8;
   typedef unsigned short     uint16;
   typedef signed short       int16;
   typedef unsigned int       uint32;
   typedef uint32             uint;
   typedef signed int         int32;

   #ifdef _MSC_VER
      typedef unsigned __int64      uint64;
      typedef signed __int64        int64;
   #else
      typedef unsigned long long    uint64;
      typedef long long             int64;
   #endif

   const uint8 LZHAM_UINT8_MIN = 0;
   const uint8 LZHAM_UINT8_MAX = 0xFFU;
   const uint16 LZHAM_UINT16_MIN = 0;
   const uint16 LZHAM_UINT16_MAX = 0xFFFFU;
   const uint32 LZHAM_UINT32_MIN = 0;
   const uint32 LZHAM_UINT32_MAX = 0xFFFFFFFFU;
   const uint64 LZHAM_UINT64_MIN = 0;
   const uint64 LZHAM_UINT64_MAX = 0xFFFFFFFFFFFFFFFFULL;    //0xFFFFFFFFFFFFFFFFui64;

   const int8  LZHAM_INT8_MIN  = -128;
   const int8  LZHAM_INT8_MAX  = 127;
   const int16 LZHAM_INT16_MIN = -32768;
   const int16 LZHAM_INT16_MAX = 32767;
   const int32 LZHAM_INT32_MIN = (-2147483647 - 1);
   const int32 LZHAM_INT32_MAX = 2147483647;
   const int64 LZHAM_INT64_MIN = (int64)0x8000000000000000ULL; //(-9223372036854775807i64 - 1);
   const int64 LZHAM_INT64_MAX = (int64)0x7FFFFFFFFFFFFFFFULL; //9223372036854775807i64;

#if LZHAM_64BIT_POINTERS
   typedef uint64 uint_ptr;
   typedef uint64 uint32_ptr;
   typedef int64 signed_size_t;
   typedef uint64 ptr_bits_t;
   const ptr_bits_t PTR_BITS_XOR = 0xDB0DD4415C87DCF7ULL;
#else
   typedef unsigned int uint_ptr;
   typedef unsigned int uint32_ptr;
   typedef signed int signed_size_t;
   typedef uint32 ptr_bits_t;
   const ptr_bits_t PTR_BITS_XOR = 0x5C87DCF7UL;
#endif
   
   enum
   {
      cInvalidIndex = -1
   };

   const uint cIntBits = sizeof(uint) * CHAR_BIT;

   template<typename T> struct int_traits { enum { cMin = INT_MIN, cMax = INT_MAX, cSigned = true }; };
   template<> struct int_traits<int8> { enum { cMin = LZHAM_INT8_MIN, cMax = LZHAM_INT8_MAX, cSigned = true }; };
   template<> struct int_traits<int16> { enum { cMin = LZHAM_INT16_MIN, cMax = LZHAM_INT16_MAX, cSigned = true }; };
   template<> struct int_traits<int32> { enum { cMin = LZHAM_INT32_MIN, cMax = LZHAM_INT32_MAX, cSigned = true }; };

   template<> struct int_traits<uint> { enum { cMin = 0, cMax = UINT_MAX, cSigned = false }; };
   template<> struct int_traits<uint8> { enum { cMin = 0, cMax = LZHAM_UINT8_MAX, cSigned = false }; };
   template<> struct int_traits<uint16> { enum { cMin = 0, cMax = LZHAM_UINT16_MAX, cSigned = false }; };

   struct empty_type { };

} // namespace lzham

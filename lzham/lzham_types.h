// File: types.h
// See Copyright Notice and license at the end of include/lzham.h
#pragma once

// TODO
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__)
   #undef INT8_MIN
   #undef INT8_MAX
   #undef UINT8_MIN
   #undef UINT8_MAX

   #undef INT16_MIN
   #undef INT16_MAX
   #undef UINT16_MIN
   #undef UINT16_MAX

   #undef INT32_MIN
   #undef INT32_MAX
   #undef UINT32_MIN
   #undef UINT32_MAX

   #undef INT64_MIN
   #undef INT64_MAX
   #undef UINT64_MIN
   #undef UINT64_MAX
#endif

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

   const uint8  UINT8_MIN  = 0;
   const uint8  UINT8_MAX  = 0xFFU;
   const uint16 UINT16_MIN = 0;
   const uint16 UINT16_MAX = 0xFFFFU;
   const uint32 UINT32_MIN = 0;
   const uint32 UINT32_MAX = 0xFFFFFFFFU;
   const uint64 UINT64_MIN = 0;
   const uint64 UINT64_MAX = 0xFFFFFFFFFFFFFFFFULL;    //0xFFFFFFFFFFFFFFFFui64;

   const int8  INT8_MIN  = -128;
   const int8  INT8_MAX  = 127;
   const int16 INT16_MIN = -32768;
   const int16 INT16_MAX = 32767;
   const int32 INT32_MIN = (-2147483647 - 1);
   const int32 INT32_MAX = 2147483647;
   const int64 INT64_MIN = (int64)0x8000000000000000ULL; //(-9223372036854775807i64 - 1);
   const int64 INT64_MAX = (int64)0x7FFFFFFFFFFFFFFFULL; //9223372036854775807i64;

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
   template<> struct int_traits<int8> { enum { cMin = INT8_MIN, cMax = INT8_MAX, cSigned = true }; };
   template<> struct int_traits<int16> { enum { cMin = INT16_MIN, cMax = INT16_MAX, cSigned = true }; };
   template<> struct int_traits<int32> { enum { cMin = INT32_MIN, cMax = INT32_MAX, cSigned = true }; };

   template<> struct int_traits<uint> { enum { cMin = 0, cMax = UINT_MAX, cSigned = false }; };
   template<> struct int_traits<uint8> { enum { cMin = 0, cMax = UINT8_MAX, cSigned = false }; };
   template<> struct int_traits<uint16> { enum { cMin = 0, cMax = UINT16_MAX, cSigned = false }; };

   struct empty_type { };

} // namespace lzham

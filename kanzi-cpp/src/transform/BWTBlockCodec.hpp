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
#ifndef _BWTBlockCodec_
#define _BWTBlockCodec_

#include "../transform/BWT.hpp"
#include "../Context.hpp"


namespace kanzi {

   // Utility class to en/de-code a BWT data block and its associated primary index(es)

   // BWT stream format: Header (m bytes) Data (n bytes)
   // Header: For each primary index,
   //   mode (8 bits) + primary index (8,16 or 24 bits)
   //   mode: bits 7-6 contain the size in bits of the primary index :
   //             00: primary index size <=  6 bits (fits in mode byte)
   //             01: primary index size <= 14 bits (1 extra byte)
   //             10: primary index size <= 22 bits (2 extra bytes)
   //             11: primary index size  > 22 bits (3 extra bytes)
   //         bits 5-0 contain 6 most significant bits of primary index
   //   primary index: remaining bits (up to 3 bytes)

   class BWTBlockCodec FINAL : public Transform<byte> {
   public:
       BWTBlockCodec(Context& ctx);

       ~BWTBlockCodec() { delete _pBWT; }

       bool forward(SliceArray<byte>& input, SliceArray<byte>& output, int length) THROW;

       bool inverse(SliceArray<byte>& input, SliceArray<byte>& output, int length) THROW;

       // Required encoding output buffer size
       int getMaxEncodedLength(int srcLen) const
       {
           return srcLen + BWT_MAX_HEADER_SIZE;
       }

   private:
       static const int BWT_MAX_HEADER_SIZE = 8 * 4;

       BWT* _pBWT;
   };
}
#endif


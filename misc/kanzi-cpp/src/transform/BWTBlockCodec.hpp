/*
Copyright 2011-2025 Frederic Langlet
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

   // BWT stream format: Header (mode + primary index(es)) | Data (n bytes)
   //   mode (8 bits): xxxyyyzz
   //   xxx: ignored
   //   yyy: log(chunks)
   //   zz: primary index size - 1 (in bytes)
   //   primary indexes (chunks * (8|16|24|32 bits))

   class BWTBlockCodec FINAL : public Transform<byte> {
   public:

       BWTBlockCodec(Context& ctx);

       ~BWTBlockCodec() { delete _pBWT; }

       bool forward(SliceArray<byte>& input, SliceArray<byte>& output, int length);

       bool inverse(SliceArray<byte>& input, SliceArray<byte>& output, int length);

       // Required encoding output buffer size
       int getMaxEncodedLength(int srcLen) const
       {
           return srcLen + 32 /* max header size */;
       }

   private:
       BWT* _pBWT;
       int _bsVersion;
   };
}
#endif


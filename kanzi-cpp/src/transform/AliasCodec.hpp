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
#ifndef _AliasCodec_
#define _AliasCodec_

#include "../Context.hpp"
#include "../Transform.hpp"

namespace kanzi {
    typedef struct ssAlias
    {
        uint32 val;
        uint32 freq;

        ssAlias(uint32 v, uint32 f) : val(v), freq(f) { }

        friend bool operator< (ssAlias const& lhs, ssAlias const& rhs) {
            int r;
            return ((r = lhs.freq - rhs.freq) != 0) ? r > 0 : lhs.val > rhs.val;
        }
    } sdAlias;


   // Simple codec replacing large symbols with small aliases whenever possible
   class AliasCodec FINAL : public Transform<byte> 
   {

   public:
       AliasCodec(int order = 1) : _pCtx(nullptr),  _order(order) { }

       AliasCodec(Context& ctx);

       ~AliasCodec() {}

       bool forward(SliceArray<byte>& src, SliceArray<byte>& dst, int length) THROW;

       bool inverse(SliceArray<byte>& src, SliceArray<byte>& dst, int length) THROW;


       // Required encoding output buffer size
       int getMaxEncodedLength(int srcLen) const
       {
           return srcLen + 1024;
       }

   private:
       static const int MIN_BLOCK_SIZE = 1024;

       Context* _pCtx;
       int _order;
   };
}

#endif


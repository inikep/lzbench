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
#ifndef _RiceGolombDecoder_
#define _RiceGolombDecoder_

#include "../EntropyDecoder.hpp"

namespace kanzi
{

   class RiceGolombDecoder : public EntropyDecoder
   {
   private:
       InputBitStream& _bitstream;
       bool _signed;
       uint _logBase;

       void _dispose() {}

   public:
       RiceGolombDecoder(InputBitStream& bitstream, uint logBase, bool sgn=true) THROW;

       ~RiceGolombDecoder() { _dispose(); }

       int decode(byte arr[], uint blkptr, uint len);

       InputBitStream& getBitStream() const { return _bitstream; }

       byte decodeByte();

       void dispose() { _dispose(); }

       bool isSigned() const { return _signed; }
   };


   inline byte RiceGolombDecoder::decodeByte()
   {
       int q = 0;

       // quotient is unary encoded
       while (_bitstream.readBit() == 0)
          q++;

       // remainder is binary encoded
       const int res = (q << _logBase) | int(_bitstream.readBits(_logBase));

       if ((_signed == true) && (res != 0))
       {
          if (_bitstream.readBit() == 1)
             return byte(-res);
       }

       return byte(res);
   }
}
#endif


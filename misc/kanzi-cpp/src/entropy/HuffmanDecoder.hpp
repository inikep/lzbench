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
#ifndef _HuffmanDecoder_
#define _HuffmanDecoder_

#include "HuffmanCommon.hpp"
#include "../EntropyDecoder.hpp"


namespace kanzi
{

   // Implementation of a static Huffman coder.
   class HuffmanDecoder : public EntropyDecoder
   {
   public:
       HuffmanDecoder(InputBitStream& bitstream, int chunkSize = HuffmanCommon::MAX_CHUNK_SIZE);

       ~HuffmanDecoder() { _dispose(); delete[] _buffer; }

       int decode(byte block[], uint blkptr, uint len);

       InputBitStream& getBitStream() const { return _bitstream; }

       void dispose() { _dispose(); }

   private:
       static const int DECODING_BATCH_SIZE = 12; // ensures decoding table fits in L1 cache
       static const int TABLE_MASK = (1 << DECODING_BATCH_SIZE) - 1;

       InputBitStream& _bitstream;
       byte* _buffer;
       uint _bufferSize;
       uint16 _codes[256];
       uint _alphabet[256];
       uint16 _sizes[256];
       uint16 _table[TABLE_MASK + 1]; // decoding table: code -> size, symbol
       int _chunkSize;

       int readLengths();

       void buildDecodingTable(int count);

       bool reset();

       void _dispose() const {}
   };


}
#endif


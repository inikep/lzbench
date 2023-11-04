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
#ifndef _XXHash32_
#define _XXHash32_

#include "../Memory.hpp"


namespace kanzi
{

   // XXHash is an extremely fast hash algorithm. It was written by Yann Collet.
   // Original source code: https://github.com/Cyan4973/xxHash

   class XXHash32 {
   private:
       static const uint32 PRIME32_1 = -1640531535;
       static const uint32 PRIME32_2 = -2048144777;
       static const uint32 PRIME32_3 = -1028477379;
       static const uint32 PRIME32_4 = 668265263;
       static const uint32 PRIME32_5 = 374761393;

       int _seed;

       uint32 round(uint32 acc, int32 val);

   public:
       XXHash32() { _seed = int(time(nullptr)); }

       XXHash32(int seed) { _seed = seed; }

       ~XXHash32(){}

       void setSeed(int seed) { _seed = seed; }

       int hash(byte data[], int length);
   };

   inline int XXHash32::hash(byte data[], int length)
   {
       uint32 h32;
       int idx = 0;

       if (length >= 16) {
           const int end16 = length - 16;
           uint32 v1 = _seed + PRIME32_1 + PRIME32_2;
           uint32 v2 = _seed + PRIME32_2;
           uint32 v3 = _seed;
           uint32 v4 = _seed - PRIME32_1;

           do {
               v1 = round(v1, LittleEndian::readInt32(&data[idx]));
               v2 = round(v2, LittleEndian::readInt32(&data[idx + 4]));
               v3 = round(v3, LittleEndian::readInt32(&data[idx + 8]));
               v4 = round(v4, LittleEndian::readInt32(&data[idx + 12]));
               idx += 16;
           } while (idx <= end16);

           h32 = ((v1 << 1) | (v1 >> 31));
           h32 += ((v2 << 7) | (v2 >> 25));
           h32 += ((v3 << 12) | (v3 >> 20));
           h32 += ((v4 << 18) | (v4 >> 14));
       }
       else {
           h32 = _seed + PRIME32_5;
       }

       h32 += uint32(length);

       while (idx <= length - 4) {
           h32 += (uint32(LittleEndian::readInt32(&data[idx])) * PRIME32_3);
           h32 = ((h32 << 17) | (h32 >> 15)) * PRIME32_4;
           idx += 4;
       }

       while (idx < length) {
           h32 += ((uint32(data[idx]) & 0xFF) * PRIME32_5);
           h32 = ((h32 << 11) | (h32 >> 21)) * PRIME32_1;
           idx++;
       }

       h32 ^= (h32 >> 15);
       h32 *= PRIME32_2;
       h32 ^= (h32 >> 13);
       h32 *= PRIME32_3;
       return h32 ^ (h32 >> 16);
   }

   inline uint32 XXHash32::round(uint32 acc, int32 val)
   {
       acc += (uint32(val) * PRIME32_2);
       return ((acc << 13) | (acc >> 19)) * PRIME32_1;
   }

}
#endif


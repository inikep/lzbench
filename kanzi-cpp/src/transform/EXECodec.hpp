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
#ifndef _EXECodec_
#define _EXECodec_

#include "../Context.hpp"
#include "../Transform.hpp"

namespace kanzi
{
   class EXECodec FINAL : public Transform<byte> {
   public:
       EXECodec() { _pCtx = nullptr; }

       EXECodec(Context& ctx) { _pCtx = &ctx; }

       ~EXECodec() {}

       bool forward(SliceArray<byte>& source, SliceArray<byte>& destination, int length) THROW;

       bool inverse(SliceArray<byte>& source, SliceArray<byte>& destination, int length) THROW;

       int getMaxEncodedLength(int inputLen) const;

   private:

       static const byte X86_MASK_JUMP = byte(0xFE);
       static const byte X86_INSTRUCTION_JUMP = byte(0xE8);
       static const byte X86_INSTRUCTION_JCC = byte(0x80);
       static const byte X86_TWO_BYTE_PREFIX = byte(0x0F);
       static const byte X86_MASK_JCC = byte(0xF0);
       static const byte X86_ESCAPE = byte(0x9B);
       static const byte NOT_EXE = byte(0x80);
       static const byte X86 = byte(0x40);
       static const byte ARM64 = byte(0x20);
       static const byte MASK_DT = byte(0x0F);
       static const int X86_ADDR_MASK = (1 << 24) - 1;
       static const int MASK_ADDRESS = 0xF0F0F0F0;
       static const int ARM_B_ADDR_MASK = (1 << 26) - 1;
       static const int ARM_B_OPCODE_MASK = 0xFFFFFFFF ^ ARM_B_ADDR_MASK;
       static const int ARM_B_ADDR_SGN_MASK = 1 << 25;
       static const int ARM_OPCODE_B = 0x14000000;  // 6 bit opcode
       static const int ARM_OPCODE_BL = 0x94000000; // 6 bit opcode       
       static const int ARM_CB_REG_BITS = 5; // lowest bits for register
       static const int ARM_CB_ADDR_MASK = 0x00FFFFE0; // 18 bit addr mask
       static const int ARM_CB_ADDR_SGN_MASK = 1 << 18;
       static const int ARM_CB_OPCODE_MASK = 0x7F000000;
       static const int ARM_OPCODE_CBZ = 0x34000000;  // 8 bit opcode
       static const int ARM_OPCODE_CBNZ = 0x3500000; // 8 bit opcode
       static const int WIN_PE = 0x00004550;
       static const uint16 WIN_X86_ARCH = 0x014C;
       static const uint16 WIN_AMD64_ARCH = 0x8664;
       static const uint16 WIN_ARM64_ARCH = 0xAA64;
       static const int ELF_X86_ARCH = 0x03;
       static const int ELF_AMD64_ARCH = 0x3E;  
       static const int ELF_ARM64_ARCH = 0xB7;
       static const int MAC_AMD64_ARCH = 0x01000007;
       static const int MAC_ARM64_ARCH = 0x0100000C;
       static const int MAC_MH_EXECUTE = 0x02;
       static const int MAC_LC_SEGMENT = 0x01;
       static const int MAC_LC_SEGMENT64 = 0x19;
       static const int MIN_BLOCK_SIZE = 4096;
       static const int MAX_BLOCK_SIZE = (1 << (26 + 2)) - 1; // max offset << 2


       bool forwardARM(SliceArray<byte>& source, SliceArray<byte>& destination, int length, int codeStart, int codeEnd);

       bool forwardX86(SliceArray<byte>& source, SliceArray<byte>& destination, int length, int codeStart, int codeEnd);

       bool inverseARM(SliceArray<byte>& source, SliceArray<byte>& destination, int length) THROW;

       bool inverseX86(SliceArray<byte>& source, SliceArray<byte>& destination, int length) THROW;

       static byte detectType(byte src[], int count, int& codeStart, int& codeEnd);
       
       static bool parseHeader(byte src[], int count, uint magic, int& arch, int& codeStart, int& codeEnd);

       Context* _pCtx;
   };
   
   
    inline int EXECodec::getMaxEncodedLength(int srcLen) const
    {
        // Allocate some extra buffer for incompressible data.
        return (srcLen <= 256) ? srcLen + 32 : srcLen + srcLen / 8;
    }   

}
#endif


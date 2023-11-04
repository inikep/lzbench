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
#ifndef _TextCodec_
#define _TextCodec_

#include "../Context.hpp"
#include "../Transform.hpp"


namespace kanzi {

   class DictEntry FINAL {
   public:
       const byte* _ptr; // text data
       int _hash; // full word hash
       int _data; // packed word length (8 MSB) + index in dictionary (24 LSB)
   
       DictEntry();

       DictEntry(const byte* ptr, int hash, int idx, int length);

#if __cplusplus < 201103L
       DictEntry(const DictEntry& de);

       DictEntry& operator=(const DictEntry& de);

       ~DictEntry() {}
#else
       DictEntry(const DictEntry& de) = delete;

       DictEntry& operator=(const DictEntry& de) = delete;

       DictEntry(DictEntry&& de) = default;

       DictEntry& operator=(DictEntry&& de) = default;

       ~DictEntry() = default;
#endif
   };

   // Encode word indexes using a token
   class TextCodec1 FINAL : public Transform<byte> {
   public:
       TextCodec1();

       TextCodec1(Context&);

       ~TextCodec1()
       {
           if (_dictList != nullptr) delete[] _dictList;
           if (_dictMap != nullptr) delete[] _dictMap;
       }

       bool forward(SliceArray<byte>& src, SliceArray<byte>& dst, int length);

       bool inverse(SliceArray<byte>& src, SliceArray<byte>& dst, int length);

       // Limit to 1 x srcLength and let the caller deal with
       // a failure when the output is too small
       int getMaxEncodedLength(int srcLen) const { return srcLen; }

   private:
       DictEntry** _dictMap;
       DictEntry* _dictList;
       byte _escapes[2];
       int _staticDictSize;
       int _dictSize;
       int _logHashSize;
       int _hashMask;
       bool _isCRLF; // EOL = CR + LF
       Context* _pCtx;

       bool expandDictionary();

       void reset(int count);

       static int emitWordIndex(byte dst[], int val);

       int emitSymbols(const byte src[], byte dst[], const int srcEnd, const int dstEnd);
   };

   // Encode word indexes using a mask (0x80)
   class TextCodec2 FINAL : public Transform<byte> {
   public:
       TextCodec2();

       TextCodec2(Context&);

       ~TextCodec2()
       {
           if (_dictList != nullptr) delete[] _dictList;
           if (_dictMap != nullptr) delete[] _dictMap;
       }

       bool forward(SliceArray<byte>& src, SliceArray<byte>& dst, int length);

       bool inverse(SliceArray<byte>& src, SliceArray<byte>& dst, int length);

       // Limit to 1 x srcLength and let the caller deal with
       // a failure when the output is too small
       int getMaxEncodedLength(int srcLen) const { return srcLen; }

   private:
       DictEntry** _dictMap;
       DictEntry* _dictList;
       int _staticDictSize;
       int _dictSize;
       int _logHashSize;
       int _hashMask;
       bool _isCRLF; // EOL = CR + LF
       Context* _pCtx;

       bool expandDictionary();

       void reset(int count);

       static int emitWordIndex(byte dst[], int val, int mask);

       int emitSymbols(const byte src[], byte dst[], const int srcEnd, const int dstEnd);
   };

   // Simple one-pass text codec that replaces words with indexes.
   // Generates a dynamic dictionary.
   class TextCodec FINAL : public Transform<byte> {
       friend class TextCodec1;
       friend class TextCodec2;

   public:
       static const int MAX_DICT_SIZE = 1 << 19; // must be less than 1<<24
       static const int MAX_WORD_LENGTH = 31; // must be less than 128
       static const int MIN_BLOCK_SIZE = 1024;
       static const int MAX_BLOCK_SIZE = 1 << 30; // 1 GB
       static const byte ESCAPE_TOKEN1 = byte(0x0F); // dictionary word preceded by space symbol
       static const byte ESCAPE_TOKEN2 = byte(0x0E); // toggle upper/lower case of first word char
       static const byte MASK_1F = byte(0x1F);
       static const byte MASK_20 = byte(0x20);
       static const byte MASK_40 = byte(0x40);
       static const byte MASK_80 = byte(0x80);

       TextCodec();

       TextCodec(Context& ctx);

       virtual ~TextCodec()
       {
           delete _delegate;
       }

       bool forward(SliceArray<byte>& src, SliceArray<byte>& dst, int length) THROW;

       bool inverse(SliceArray<byte>& src, SliceArray<byte>& dst, int length) THROW;

       int getMaxEncodedLength(int srcLen) const
       {
           return _delegate->getMaxEncodedLength(srcLen);
       }

       static bool isText(byte val) { return TEXT_CHARS[uint8(val)]; }

       static bool isLowerCase(byte val) { return (val >= byte('a')) && (val <= byte('z')); }

       static bool isUpperCase(byte val) { return (val >= byte('A')) && (val <= byte('Z')); }

       static bool isDelimiter(byte val) { return DELIMITER_CHARS[uint8(val)]; }

   private:
       static const int HASH1 = 0x7FEB352D;
       static const int HASH2 = 0x846CA68B;
       static const byte CR = byte(0x0D);
       static const byte LF = byte(0x0A);
       static const byte SP = byte(0x20);
       static const int THRESHOLD1 = 128;
       static const int THRESHOLD2 = THRESHOLD1 * THRESHOLD1;
       static const int THRESHOLD3 = 32;
       static const int THRESHOLD4 = THRESHOLD3 * 128;
       static const int LOG_HASHES_SIZE = 24; // 16 MB
       static const byte MASK_NOT_TEXT = byte(0x80);
       static const byte MASK_CRLF = byte(0x40);
       static const byte MASK_XML_HTML = byte(0x20);
       static const byte MASK_DT = byte(0x0F);
       static const int MASK_LENGTH = 0x0007FFFF; // 19 bits

       static bool init(bool delims[256], bool text[256]);
       static bool DELIMITER_CHARS[256];
       static bool TEXT_CHARS[256];
       static const bool INIT;

       static bool sameWords(const byte src[], const byte dst[], int length);

       static byte computeStats(const byte block[], int count, uint freqs[], bool strict);

       static byte detectType(uint freqs0[], uint freqs1[], int count);
       
       // Common English words.
       static char DICT_EN_1024[];

       // Static dictionary of 1024 entries.
       static DictEntry STATIC_DICTIONARY[1024];
       static int createDictionary(char words[], int dictSize, DictEntry dict[], int maxWords, int startWord);
       static const int STATIC_DICT_WORDS;

       Transform<byte>* _delegate;
   };

   inline DictEntry::DictEntry()
       : _ptr(nullptr)
       , _hash(0)
       , _data(0)
   { 
   }

   inline DictEntry::DictEntry(const byte* ptr, int hash, int idx, int length = 0)
       : _ptr(ptr)
       , _hash(hash)
       , _data((length << 24) | idx)
   {
   }

#if __cplusplus < 201103L
   inline DictEntry::DictEntry(const DictEntry& de)
   {
       _ptr = de._ptr;
       _hash = de._hash;
       _data = de._data;
   }

   inline DictEntry& DictEntry::operator=(const DictEntry& de)
   {
       _ptr = de._ptr;
       _hash = de._hash;
       _data = de._data;
       return *this;
   }
#endif

   inline bool TextCodec::sameWords(const byte src[], const byte dst[], int length)
   {
       while (length >= 4) {
           length -= 4;

           if (memcmp(&src[length], &dst[length], 4) != 0)
              return false;
       }

       while (length > 0) {
           length--;

           if (dst[length] != src[length])
              return false;
       }

       return true;
   }
}
#endif


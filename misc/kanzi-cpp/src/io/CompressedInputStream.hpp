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
#ifndef _CompressedInputStream_
#define _CompressedInputStream_

#include <cstdio> // definition of EOF
#include <string>
#include <vector>
#include "../concurrent.hpp"
#include "../Context.hpp"
#include "../Event.hpp"
#include "../Listener.hpp"
#include "../InputStream.hpp"
#include "../SliceArray.hpp"
#include "../bitstream/DefaultInputBitStream.hpp"
#include "../util/XXHash.hpp"

#if __cplusplus >= 201103L
   #include <functional>
#endif


namespace kanzi
{

   class DecodingTaskResult FINAL {
   public:
       int _blockId;
       int _decoded;
       byte* _data;
       int _error; // 0 = OK
       std::string _msg;
       uint64 _checksum;
       bool _skipped;
       clock_t _completionTime;

       DecodingTaskResult()
       {
           _blockId = -1;
           _decoded = 0;
           _data = nullptr;
           _error = 0;
           _checksum = 0;
           _skipped = false;
           _completionTime = clock();
       }

       DecodingTaskResult(const SliceArray<byte>& data, int blockId, int decoded, uint64 checksum,
          int error, const std::string& msg, bool skipped = false)
           : _blockId(blockId)
           , _decoded(decoded)
           , _data(data._array)
           , _error(error)
           , _msg(msg)
           , _checksum(checksum)
           , _skipped(skipped)
       {
           _completionTime = clock();
       }

       DecodingTaskResult(const DecodingTaskResult& result)
           : _blockId(result._blockId)
           , _decoded(result._decoded)
           , _data(result._data)
           , _error(result._error)
           , _msg(result._msg)
           , _checksum(result._checksum)
           , _skipped(result._skipped)
           , _completionTime(result._completionTime)
       {
       }

       DecodingTaskResult& operator = (const DecodingTaskResult& result)
       {
           _msg = result._msg;
           _data = result._data;
           _blockId = result._blockId;
           _error = result._error;
           _decoded = result._decoded;
           _checksum = result._checksum;
           _completionTime = result._completionTime;
           _skipped = result._skipped;
           return *this;
       }

       ~DecodingTaskResult() {}
   };

   // A task used to decode a block
   // Several tasks (transform+entropy) may run in parallel
   template <class T>
   class DecodingTask FINAL : public Task<T> {
   private:
       SliceArray<byte>* _data;
       SliceArray<byte>* _buffer;
       int _blockLength;
       DefaultInputBitStream* _ibs;
       XXHash32* _hasher32;
       XXHash64* _hasher64;
       ATOMIC_INT* _processedBlockId;
       std::vector<Listener<Event>*> _listeners;
       Context _ctx;

   public:
       DecodingTask(SliceArray<byte>* iBuffer, SliceArray<byte>* oBuffer,
           int blockSize, DefaultInputBitStream* ibs, XXHash32* hasher32, XXHash64* hasher64,
           ATOMIC_INT* processedBlockId, std::vector<Listener<Event>*>& listeners,
           const Context& ctx);

       ~DecodingTask(){}

       T run();
   };

   class CompressedInputStream : public InputStream {
       friend class DecodingTask<DecodingTaskResult>;

   public:
        // If headerless == false, all provided compression parameters will be overwritten
        // with values read from the bitstream header.
        CompressedInputStream(InputStream& is,
                   int jobs = 1,
                   const std::string& entropy = "NONE",
                   const std::string& transform = "NONE",
                   int blockSize = 4*1024*1024,
                   int checksum = 0,
                   uint64 originalSize = 0,
#ifdef CONCURRENCY_ENABLED
                   ThreadPool* pool = nullptr,
#endif
                   bool headerless = false,
                   int bsVersion = BITSTREAM_FORMAT_VERSION);

      // If headerless == true, the context must contain "entropy", "transform", "checksum" & "blockSize"
      // If "bsVersion" is missing, the current value of BITSTREAM_FORMAT_VERSION is assumed.
       CompressedInputStream(InputStream& is, Context& ctx, bool headerless = false);

       ~CompressedInputStream();

       bool addListener(Listener<Event>& bl);

       bool removeListener(Listener<Event>& bl);

       std::streampos tellg();

       std::istream& seekg(std::streampos pos);

       std::istream& putback(char c);

       std::istream& unget();

       std::istream& read(char* s, std::streamsize n);

       std::streamsize gcount() const { return _gcount; }

       int get();

       int peek();

       void close();

       uint64 getRead() const { return (_ibs->read() + 7) >> 3; }

#if !defined(_MSC_VER) || _MSC_VER > 1500
       bool seek(int64 bitPos);

       int64 tell();
#endif


   protected:

       void readHeader();


   private:
       static const int BITSTREAM_TYPE;
       static const int BITSTREAM_FORMAT_VERSION;
       static const int DEFAULT_BUFFER_SIZE;
       static const int EXTRA_BUFFER_SIZE;
       static const byte COPY_BLOCK_MASK;
       static const byte TRANSFORMS_MASK;
       static const int MIN_BITSTREAM_BLOCK_SIZE;
       static const int MAX_BITSTREAM_BLOCK_SIZE;
       static const int CANCEL_TASKS_ID;
       static const int MAX_CONCURRENCY;
       static const int MAX_BLOCK_ID;

       int _blockSize;
       int _bufferId; // index of current read buffer
       int _maxBufferId; // max index of read buffer
       int _nbInputBlocks;
       int _jobs;
       int _bufferThreshold;
       int64 _available; // decoded not consumed bytes
       int64 _outputSize;
       XXHash32* _hasher32;
       XXHash64* _hasher64;
       SliceArray<byte>** _buffers; // input & output per block
       short _entropyType;
       uint64 _transformType;
       DefaultInputBitStream* _ibs;
       ATOMIC_BOOL _initialized;
       ATOMIC_BOOL _closed;
       ATOMIC_INT _blockId;
       std::vector<Listener<Event>*> _listeners;
       std::streamsize _gcount;
       Context _ctx;
       Context* _parentCtx; // not owner
       bool _headless;
#ifdef CONCURRENCY_ENABLED
       ThreadPool* _pool;
#endif

       int64 processBlock();

       int _get(int inc);

       static void notifyListeners(std::vector<Listener<Event>*>& listeners, const Event& evt);
   };


   inline int CompressedInputStream::get()
   {
       const int res = _get(1);
       _gcount = (res != EOF) ? 1 : 0;
       return res;
   }

   inline int CompressedInputStream::peek()
   {
       return _get(0);
   }

   inline std::streampos CompressedInputStream::tellg()
   {
       throw std::ios_base::failure("Not supported");
   }

   inline std::istream& CompressedInputStream::seekg(std::streampos)
   {
       throw std::ios_base::failure("Not supported");
   }

   inline std::istream& CompressedInputStream::putback(char)
   {
       setstate(std::ios::badbit);
       throw std::ios_base::failure("Not supported");
   }

   inline std::istream& CompressedInputStream::unget()
   {
       setstate(std::ios::badbit);
       throw std::ios_base::failure("Not supported");
   }

#if !defined(_MSC_VER) || _MSC_VER > 1500
   inline bool CompressedInputStream::seek(int64 bitPos)
   {
      // Beware. There be dragons !
      // The only valid bitstream positions are the beginning of a block.
      // Useful to navigate the bitstream block by block without decompressing.
      _available = 0;
      _bufferId = 0;
      return _ibs->seek(bitPos);
   }

   inline int64 CompressedInputStream::tell()
   {
      return _ibs->tell();
   }
#endif
}
#endif


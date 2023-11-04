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
#ifndef _CompressedInputStream_
#define _CompressedInputStream_

#include <cstdio> // definition of EOF
#include <string>
#include <vector>
#include "../concurrent.hpp"
#include "../Context.hpp"
#include "../Listener.hpp"
#include "../InputStream.hpp"
#include "../InputBitStream.hpp"
#include "../SliceArray.hpp"
#include "../util/XXHash32.hpp"

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
       int _checksum;
       bool _skipped;
       clock_t _completionTime;

       DecodingTaskResult()
           : _blockId(-1)
           , _decoded(0)
           , _data(nullptr)
           , _error(0)
           , _msg()
           , _checksum(0)
           , _skipped(false)
           , _completionTime(clock())
       {
       }

       DecodingTaskResult(const SliceArray<byte>& data, int blockId, int decoded, int checksum,
          int error, const std::string& msg, bool skipped = false)
           : _blockId(blockId)
           , _decoded(decoded)
           , _data(data._array)
           , _error(error)
           , _msg(msg)
           , _checksum(checksum)
           , _skipped(skipped)
           , _completionTime(clock())
       {
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
       uint64 _transformType;
       short _entropyType;
       int _blockId;
       InputBitStream* _ibs;
       XXHash32* _hasher;
       ATOMIC_INT* _processedBlockId;
       std::vector<Listener*> _listeners;
       Context _ctx;

   public:
       DecodingTask(SliceArray<byte>* iBuffer, SliceArray<byte>* oBuffer, int blockSize,
           uint64 transformType, short entropyType, int blockId,
           InputBitStream* ibs, XXHash32* hasher,
           ATOMIC_INT* processedBlockId, std::vector<Listener*>& listeners,
           const Context& ctx);

       ~DecodingTask(){}

       T run() THROW;
   };

   class CompressedInputStream FINAL : public InputStream {
       friend class DecodingTask<DecodingTaskResult>;

   private:
       static const int BITSTREAM_TYPE = 0x4B414E5A; // "KANZ"
       static const int BITSTREAM_FORMAT_VERSION = 4;
       static const int DEFAULT_BUFFER_SIZE = 256 * 1024;
       static const int EXTRA_BUFFER_SIZE = 512;
       static const byte COPY_BLOCK_MASK = byte(0x80);
       static const byte TRANSFORMS_MASK = byte(0x10);
       static const int MIN_BITSTREAM_BLOCK_SIZE = 1024;
       static const int MAX_BITSTREAM_BLOCK_SIZE = 1024 * 1024 * 1024;
       static const int CANCEL_TASKS_ID = -1;
       static const int MAX_CONCURRENCY = 64;
       static const int UNKNOWN_NB_BLOCKS = 65536;
       static const int MAX_BLOCK_ID = int((uint(1) << 31) - 1);

       int _blockSize;
       int _bufferId; // index of current read buffer
       int _maxBufferId; // max index of read buffer
       int _nbInputBlocks;
       int _jobs;
       int _bufferThreshold;
       int _available; // decoded not consumed bytes
       XXHash32* _hasher;
       SliceArray<byte>** _buffers; // input & output per block
       short _entropyType;
       uint64 _transformType;
       InputBitStream* _ibs;
       ATOMIC_BOOL _initialized;
       ATOMIC_BOOL _closed;
       ATOMIC_INT _blockId;
       std::vector<Listener*> _listeners;
       std::streamsize _gcount;
       Context _ctx;
#ifdef CONCURRENCY_ENABLED
       ThreadPool* _pool;
#endif

       void readHeader() THROW;

       int processBlock() THROW;

       int _get(int inc);

       static void notifyListeners(std::vector<Listener*>& listeners, const Event& evt);

   public:
#ifdef CONCURRENCY_ENABLED
        CompressedInputStream(InputStream& is, int jobs = 1, ThreadPool* pool = nullptr);
#else
        CompressedInputStream(InputStream& is, int jobs = 1);
#endif

#if __cplusplus >= 201103L
       CompressedInputStream(InputStream& is, Context& ctx,
          std::function<InputBitStream*(InputStream&)>* createBitStream=nullptr);
#else
       CompressedInputStream(InputStream& is, Context& ctx);
#endif

       ~CompressedInputStream();

       bool addListener(Listener& bl);

       bool removeListener(Listener& bl);

       std::streampos tellg();

       std::istream& seekg(std::streampos pos) THROW;

       std::istream& putback(char c) THROW;

       std::istream& unget() THROW;

       std::istream& read(char* s, std::streamsize n) THROW;

       std::streamsize gcount() const { return _gcount; }

       int get() THROW;

       int peek() THROW;

       void close() THROW;

       uint64 getRead() { return (_ibs->read() + 7) >> 3; }
   };


   inline int CompressedInputStream::get() THROW
   {
       const int res = _get(1);
       _gcount = (res != EOF) ? 1 : 0;
       return res;
   }

   inline int CompressedInputStream::peek() THROW
   {
       return _get(0);
   }

   inline std::streampos CompressedInputStream::tellg()
   {
       return uint(getRead());
   }

   inline std::istream& CompressedInputStream::seekg(std::streampos) THROW
   {
       setstate(std::ios::badbit);
       throw std::ios_base::failure("Not supported");
   }

   inline std::istream& CompressedInputStream::putback(char) THROW
   {
       setstate(std::ios::badbit);
       throw std::ios_base::failure("Not supported");
   }

   inline std::istream& CompressedInputStream::unget() THROW
   {
       setstate(std::ios::badbit);
       throw std::ios_base::failure("Not supported");
   }

}
#endif


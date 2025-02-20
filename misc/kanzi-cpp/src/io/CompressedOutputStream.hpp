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
#ifndef _CompressedOutputStream_
#define _CompressedOutputStream_


#include <string>
#include <vector>
#include "../concurrent.hpp"
#include "../Context.hpp"
#include "../Listener.hpp"
#include "../OutputStream.hpp"
#include "../OutputBitStream.hpp"
#include "../SliceArray.hpp"
#include "../util/XXHash32.hpp"

#if __cplusplus >= 201103L
   #include <functional>
#endif


namespace kanzi {

   class EncodingTaskResult FINAL {
   public:
       int _blockId;
       int _error; // 0 = OK
       std::string _msg;

       EncodingTaskResult()
       {
           _blockId = -1;
           _error = 0;
       }

       EncodingTaskResult(int blockId, int error, const std::string& msg)
           : _blockId(blockId)
           , _error(error)
           , _msg(msg)
       {
       }

       EncodingTaskResult(const EncodingTaskResult& result)
           : _blockId(result._blockId)
           , _error(result._error)
           , _msg(result._msg)
       {
       }

       EncodingTaskResult& operator = (const EncodingTaskResult& result)
       {
           _msg = result._msg;
           _blockId = result._blockId;
           _error = result._error;
           return *this;
       }

       ~EncodingTaskResult() {}
   };

   // A task used to encode a block
   // Several tasks (transform+entropy) may run in parallel
   template <class T>
   class EncodingTask FINAL : public Task<T> {
   private:
       SliceArray<byte>* _data;
       SliceArray<byte>* _buffer;
       OutputBitStream* _obs;
       XXHash32* _hasher;
       ATOMIC_INT* _processedBlockId;
       std::vector<Listener*> _listeners;
       Context _ctx;

   public:
       EncodingTask(SliceArray<byte>* iBuffer, SliceArray<byte>* oBuffer,
           OutputBitStream* obs, XXHash32* hasher,
           ATOMIC_INT* processedBlockId, std::vector<Listener*>& listeners,
           const Context& ctx);

       ~EncodingTask(){}

       T run();
   };

   class CompressedOutputStream : public OutputStream {
       friend class EncodingTask<EncodingTaskResult>;

   public:
#ifdef CONCURRENCY_ENABLED
       CompressedOutputStream(OutputStream& os, const std::string& codec, const std::string& transform,
          int blockSize = 4 * 1024 * 1024, bool checksum = false, int jobs = 1,
           uint64 fileSize = 0, ThreadPool* pool = nullptr, bool headerless = false);
#else
       CompressedOutputStream(OutputStream& os, const std::string& codec, const std::string& transform,
          int blockSize = 4 * 1024 * 1024, bool checksum = false, int jobs = 1,
          uint64 fileSize = 0, bool headerless = false);
#endif

#if __cplusplus >= 201103L
       CompressedOutputStream(OutputStream& os, Context& ctx,
          std::function<OutputBitStream*(OutputStream&)>* createBitStream = nullptr);
#else
       CompressedOutputStream(OutputStream& os, Context& ctx);
#endif

       ~CompressedOutputStream();

       bool addListener(Listener& bl);

       bool removeListener(Listener& bl);

       std::ostream& write(const char* s, std::streamsize n);

       std::ostream& put(char c);

       std::ostream& flush();

       std::streampos tellp();

       std::ostream& seekp(std::streampos pos);

       void close();

       uint64 getWritten() const { return (_obs->written() + 7) >> 3; }


  protected:

       void writeHeader();


   private:
       static const int BITSTREAM_TYPE = 0x4B414E5A; // "KANZ"
       static const int BITSTREAM_FORMAT_VERSION = 5;
       static const int DEFAULT_BUFFER_SIZE = 256 * 1024;
       static const byte COPY_BLOCK_MASK = byte(0x80);
       static const byte TRANSFORMS_MASK = byte(0x10);
       static const int MIN_BITSTREAM_BLOCK_SIZE = 1024;
       static const int MAX_BITSTREAM_BLOCK_SIZE = 1024 * 1024 * 1024;
       static const int SMALL_BLOCK_SIZE = 15;
       static const int CANCEL_TASKS_ID = -1;
       static const int MAX_CONCURRENCY = 64;

       int _blockSize;
       int _bufferId; // index of current write buffer
       int _jobs;
       int _bufferThreshold;
       int _nbInputBlocks;
       int64 _inputSize;
       XXHash32* _hasher;
       SliceArray<byte>** _buffers; // input & output per block
       short _entropyType;
       uint64 _transformType;
       OutputBitStream* _obs;
       ATOMIC_BOOL _initialized;
       ATOMIC_BOOL _closed;
       ATOMIC_INT _blockId;
       std::vector<Listener*> _listeners;
       Context _ctx;
       bool _headless;
#ifdef CONCURRENCY_ENABLED
       ThreadPool* _pool;
#endif

       void processBlock();

       static void notifyListeners(std::vector<Listener*>& listeners, const Event& evt);
   };


   inline std::streampos CompressedOutputStream::tellp()
   {
       return uint(getWritten());
   }

   inline std::ostream& CompressedOutputStream::seekp(std::streampos)
   {
       setstate(std::ios::badbit);
       throw std::ios_base::failure("Not supported");
   }

   inline std::ostream& CompressedOutputStream::flush()
   {
       // Let the underlying output stream flush itself when needed
       return *this;
   }

   inline std::ostream& CompressedOutputStream::put(char c)
   {
       try {
           if (_buffers[_bufferId]->_index >= _bufferThreshold) {
               // Current write buffer is full
               const int nbTasks = (_nbInputBlocks == 0) || (_jobs < _nbInputBlocks) ? _jobs : _nbInputBlocks;

               if (_bufferId + 1 < nbTasks) {
                   _bufferId++;
                   const int bSize = _blockSize + (_blockSize >> 6);
                   const int bufSize = (bSize > 65536) ? bSize : 65536;

                   if (_buffers[_bufferId]->_length == 0) {
                       delete[] _buffers[_bufferId]->_array;
                       _buffers[_bufferId]->_array = new byte[bufSize];
                       _buffers[_bufferId]->_length = bufSize;
                   }

                   _buffers[_bufferId]->_index = 0;
               }
               else {
                   if (_closed.load() == true)
                       throw std::ios_base::failure("Stream closed");

                   // If all buffers are full, time to encode
                   processBlock();
               }
           }

           _buffers[_bufferId]->_array[_buffers[_bufferId]->_index++] = byte(c);
           return *this;
       }
       catch (std::exception& e) {
           setstate(std::ios::badbit);
           throw std::ios_base::failure(e.what());
       }
   }
}
#endif


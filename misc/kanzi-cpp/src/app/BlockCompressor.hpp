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
#ifndef _BlockCompressor_
#define _BlockCompressor_

#include <map>
#include <vector>
#include "../InputStream.hpp"
#include "../io/CompressedOutputStream.hpp"

namespace kanzi {

   class FileCompressResult {
   public:
       int _code;
       uint64 _read;
       uint64 _written;
       std::string _errMsg;

       FileCompressResult()
          : _code(0)
          , _read(0)
          , _written(0)
          , _errMsg()
       {
       }

       FileCompressResult(int code, uint64 read, uint64 written, const std::string& errMsg)
           : _code(code)
           , _read(read)
           , _written(written)
           , _errMsg(errMsg)
       {
       }

#if __cplusplus < 201103L
       FileCompressResult(const FileCompressResult& fcr)
           : _code(fcr._code)
           , _read(fcr._read)
           , _written(fcr._written)
           , _errMsg(fcr._errMsg)
       {
       }

       FileCompressResult& operator=(const FileCompressResult& fcr)
       {
           _errMsg = fcr._errMsg;
           _code = fcr._code;
           _read = fcr._read;
           _written = fcr._written;
           return *this;
       }

       ~FileCompressResult() {}
#else
       FileCompressResult(const FileCompressResult& fdr) = delete;

       FileCompressResult& operator=(const FileCompressResult& fdr) = delete;

       FileCompressResult(FileCompressResult&& fdr) = default;

       FileCompressResult& operator=(FileCompressResult&& fdr) = default;

       ~FileCompressResult() = default;
#endif
   };

#ifdef CONCURRENCY_ENABLED
   template <class T, class R>
   class FileCompressWorker FINAL : public Task<R> {
   public:
       FileCompressWorker(BoundedConcurrentQueue<T>* queue) : _queue(queue) { }

       ~FileCompressWorker() {}

       R run();

   private:
       BoundedConcurrentQueue<T>* _queue;
   };
#endif

   template <class T>
   class FileCompressTask FINAL : public Task<T> {
   public:
       static const int DEFAULT_BUFFER_SIZE = 65536;

       FileCompressTask(const Context& ctx, std::vector<Listener*>& listeners);

       ~FileCompressTask();

       T run();

       void dispose();

   private:
       Context _ctx;
       InputStream* _is;
       CompressedOutputStream* _cos;
       std::vector<Listener*> _listeners;
   };


   typedef FileCompressTask<FileCompressResult> FCTask;

   class BlockCompressor {
       friend class FileCompressTask<FileCompressResult>;

   public:
       BlockCompressor(const Context& ctx);

       ~BlockCompressor();

       int compress(uint64& written);

       bool addListener(Listener& bl);

       bool removeListener(Listener& bl);

       void dispose() const {};

   private:
       static const int DEFAULT_BLOCK_SIZE = 4 * 1024 * 1024;
       static const int MIN_BLOCK_SIZE = 1024;
       static const int MAX_BLOCK_SIZE = 1024 * 1024 * 1024;

       int _verbosity;
       bool _overwrite;
       bool _checksum;
       bool _skipBlocks;
       std::string _inputName;
       std::string _outputName;
       std::string _codec;
       std::string _transform;
       int _blockSize;
       bool _autoBlockSize; // derive block size from input size and jobs
       int _jobs;
       std::vector<Listener*> _listeners;
       bool _reorderFiles;
       bool _noDotFiles;
       bool _noLinks;
       Context _ctx;

       static void notifyListeners(std::vector<Listener*>& listeners, const Event& evt);

       static void getTransformAndCodec(int level, std::string tranformAndCodec[2]);
   };
}
#endif


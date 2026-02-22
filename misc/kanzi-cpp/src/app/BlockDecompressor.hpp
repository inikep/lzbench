/*
Copyright 2011-2026 Frederic Langlet
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
#ifndef knz_BlockDecompressor
#define knz_BlockDecompressor

#include <map>
#include <vector>
#include "../OutputStream.hpp"
#include "../io/CompressedInputStream.hpp"

namespace kanzi {
   class FileDecompressResult {
   public:
       int _code;
       uint64 _read;
       std::string _errMsg;

       FileDecompressResult()
          : _code(0)
          , _read(0)
          , _errMsg()
       {
       }

       FileDecompressResult(int code, uint64 read, const std::string& errMsg)
           : _code(code)
           , _read(read)
           , _errMsg(errMsg)
       {
       }

#if __cplusplus < 201103L
       FileDecompressResult(const FileDecompressResult& fdr)
           : _code(fdr._code)
           , _read(fdr._read)
           , _errMsg(fdr._errMsg)
       {
       }

       FileDecompressResult& operator=(const FileDecompressResult& fdr)
       {
           _errMsg = fdr._errMsg;
           _code = fdr._code;
           _read = fdr._read;
           return *this;
       }

       ~FileDecompressResult() {}
#else
       FileDecompressResult(const FileDecompressResult& fcr) = delete;

       FileDecompressResult& operator=(const FileDecompressResult& fcr) = delete;

       FileDecompressResult(FileDecompressResult&& fcr) = default;

       FileDecompressResult& operator=(FileDecompressResult&& fcr) = default;

       ~FileDecompressResult() = default;
#endif
   };

#ifdef CONCURRENCY_ENABLED
   template <class T, class R>
   class FileDecompressWorker FINAL : public Task<R> {
   public:
       FileDecompressWorker(BoundedConcurrentQueue<T>* queue) : _queue(queue) { }

       ~FileDecompressWorker() {}

       R run();

   private:
       BoundedConcurrentQueue<T>* _queue;
   };
#endif

   template <class T>
   class FileDecompressTask FINAL : public Task<T> {
   public:
       static const int DEFAULT_BUFFER_SIZE = 65536;

       FileDecompressTask(const Context& ctx, std::vector<Listener<Event>*>& listeners);

       ~FileDecompressTask();

       T run();

       void dispose();

   private:
       Context _ctx;
       OutputStream* _os;
       CompressedInputStream* _cis;
       std::vector<Listener<Event>*> _listeners;
   };

   typedef FileDecompressTask<FileDecompressResult> FDTask;

   class BlockDecompressor {
       friend class FileDecompressTask<FileDecompressResult>;

   public:
       BlockDecompressor(const Context& ctx);

       ~BlockDecompressor();

       int decompress(uint64& read);

       // Register a copy of the listener
       // Not thread safe
       bool addListener(Listener<Event>& bl);

       // Not thread safe
       bool removeListener(Listener<Event>& bl);

       void dispose() const {};

   private:
       int _verbosity;
       bool _overwrite;
       std::string _inputName;
       std::string _outputName;
       int _blockSize;
       int _jobs;
       std::vector<Listener<Event>*> _listeners;
       bool _noDotFiles;
       bool _noLinks;
       Context _ctx;

       static void notifyListeners(std::vector<Listener<Event>*>& listeners, const Event& evt);
   };
}
#endif


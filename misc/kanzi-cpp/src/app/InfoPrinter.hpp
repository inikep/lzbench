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
#ifndef knz_InfoPrinter
#define knz_InfoPrinter

#include <map>
#include <memory>
#ifdef CONCURRENCY_ENABLED
   #include <mutex>
#endif
#include <time.h>
#include <vector>

#include "../concurrent.hpp"
#include "../Event.hpp"
#include "../Listener.hpp"
#include "../OutputStream.hpp"
#include "../util/Clock.hpp"

namespace kanzi
{

   class BlockInfo {
      public:
          int64 _stage0Size;
          int64 _stage1Size;
          WallTimer::TimeData _timeStamp1;
          WallTimer::TimeData _timeStamp2;
          WallTimer::TimeData _timeStamp3;

       BlockInfo() : _stage0Size(0), _stage1Size(0) {}
   };

   class InfoPrinter : public Listener<Event> {
      public:
          enum Type { COMPRESSION, DECOMPRESSION, INFO };

          InfoPrinter(int infoLevel, InfoPrinter::Type type, OutputStream& os, int firstBlockId = 1);
          ~InfoPrinter() {}

          void processEvent(const Event& evt);

      private:
#ifdef CONCURRENCY_ENABLED
          // Ordered-phase handling
          void processBlockEventOrdered(const Event& evt);
#endif

          // Actual event processing + printing
          void processEventOrdered(const Event& evt);

          // Header-only info
          void processHeaderInfo(const Event& evt);

          OutputStream& _os;
          InfoPrinter::Type _type;
          int _level;
          int _headerInfo;

          // Per-block state
          std::map<int, BlockInfo*> _blocks;

          Event::Type _thresholds[6];
#ifdef CONCURRENCY_ENABLED
          std::mutex _mutex;
#endif
          std::map<int, std::vector<Event> > _pendingBlocks;
          atomic_int_t _nextBlockId;
   };

}

#endif


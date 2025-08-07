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
#ifndef _InfoPrinter_
#define _InfoPrinter_

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
   };

   // An implementation of Listener to display block information (verbose option
   // of the BlockCompressor/BlockDecompressor)
   class InfoPrinter : public Listener<Event> {
   public:
       enum Type {
           ENCODING,
           DECODING
       };

       InfoPrinter(int infoLevel, InfoPrinter::Type type, OutputStream& os);

       ~InfoPrinter() {
          for (int i = 0; i < 1024; i++) {
             if (_map[i] != nullptr)
                delete _map[i];
          }
       }

       void processEvent(const Event& evt);

   private:
       OutputStream& _os;
       BlockInfo* _map[1024];
       Event::Type _thresholds[6];
       InfoPrinter::Type _type;
       int _level;
       Clock _clock12;
       Clock _clock23;
       Clock _clock34;
	   
       static uint hash(uint id) { return (id * 0x1E35A7BD) & 0x03FF; }
   };
}
#endif


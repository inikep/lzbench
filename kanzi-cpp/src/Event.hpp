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
#ifndef _Event_
#define _Event_

#include <string>
#include <time.h>
#include "types.hpp"

namespace kanzi
{

   class Event {
      public:
          enum Type {
              COMPRESSION_START,
              COMPRESSION_END,
              BEFORE_TRANSFORM,
              AFTER_TRANSFORM,
              BEFORE_ENTROPY,
              AFTER_ENTROPY,
              DECOMPRESSION_START,
              DECOMPRESSION_END,
              AFTER_HEADER_DECODING
          };

          Event(Type type, int id, const std::string& msg, clock_t evtTime);

          Event(Type type, int id, int64 size, clock_t evtTime);

          Event(Type type, int id, int64 size, int hash, bool hashing, clock_t evtTime);

          ~Event() {}

          int getId() const { return _id; }

          int64 getSize() const { return _size; }

          Event::Type getType() const { return _type; }

          std::string getTypeAsString() const;

          clock_t getTime() const { return _time; }

          int getHash() const { return _hashing ? _hash : 0; }

          std::string toString() const;

      private:
          Event::Type _type;
          clock_t _time;
          std::string _msg;
          int _id;
          int64 _size;
          int _hash;
          bool _hashing;
      };
}
#endif


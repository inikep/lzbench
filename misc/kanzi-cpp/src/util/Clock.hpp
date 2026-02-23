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
#ifndef knz_Clock
#define knz_Clock


#if __cplusplus >= 201103L || _MSC_VER >= 1700

#include <chrono>

namespace kanzi
{
   class Clock {
   private:
           std::chrono::steady_clock::time_point _start;
           std::chrono::steady_clock::time_point _stop;

   public:
           Clock()
           {
                   start();
                   _stop = _start;
           }

           void start()
           {
                   _start = std::chrono::steady_clock::now();
           }

           void stop()
           {
                   _stop = std::chrono::steady_clock::now();
           }

           double elapsed() const
           {
                   // In millisec
                   return double(std::chrono::duration_cast<std::chrono::milliseconds>(_stop - _start).count());
           }
   };
}
#else

#include <ctime>

namespace kanzi
{

   class Clock {
   private:
           clock_t _start;
           clock_t _stop;

   public:
           Clock()
           {
               start();
               _stop = _start;
           }

           void start()
           {
              _start = clock();
           }

           void stop()
           {
              _stop = clock();
           }

           double elapsed() const
           {
              // In millisec
              return (_stop <= _start) ? 0.0 : double(_stop - _start) / CLOCKS_PER_SEC * 1000.0;
           }
   };
   
}
#endif

#endif


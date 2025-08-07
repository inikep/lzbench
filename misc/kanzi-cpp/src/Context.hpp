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
#ifndef _Context_
#define _Context_

#include <map>
#include <sstream>
#include <string>
#include "concurrent.hpp"
#include "util/strings.hpp"

namespace kanzi
{

   // Poor's man equivalent to std::variant used to support C++98 and up.
   // union cannot be used due to the std:string field.
   // The extra memory used does not matter for the application context since
   // the map is small.
   typedef struct ContextVal {
       int64 lVal;
       std::string sVal;
       bool isString;

       ContextVal(bool b, int64 val, const std::string& str) : lVal(val), sVal(str), isString(b) {}
       ContextVal() { isString = false; lVal = 0; }
   } ctxVal;

   class Context
   {
   public:

#ifdef CONCURRENCY_ENABLED
       Context(ThreadPool* p = nullptr) : _pool(p) {}
       Context(const Context& c) : _map(c._map), _pool(c._pool) {}
       Context(const Context& c, ThreadPool* p) : _map(c._map), _pool(p) {}
       Context& operator=(const Context& c) = default;
#else
       Context() {}
       Context(const Context& c) : _map(c._map) {}
       Context& operator=(const Context& c) { _map = c._map; return *this; };
#endif

       virtual ~Context() {}
       bool has(const std::string& key) const;
       int getInt(const std::string& key, int defValue = 0) const;
       int64 getLong(const std::string& key, int64 defValue = 0) const;
       std::string getString(const std::string& key, const std::string& defValue = "") const;
       void putInt(const std::string& key, int value);
       void putLong(const std::string& key, int64 value);
       void putString(const std::string& key, const std::string& value);

#ifdef CONCURRENCY_ENABLED
       ThreadPool* getPool() const { return _pool; }
#endif

   private:
       std::map<std::string, ContextVal> _map;

#ifdef CONCURRENCY_ENABLED
       ThreadPool* _pool;
#endif
   };


   inline bool Context::has(const std::string& key) const
   {
      return _map.find(key) != _map.end();
   }


   inline int Context::getInt(const std::string& key, int defValue) const
   {
      return int(this->getLong(key, defValue));
   }


   inline int64 Context::getLong(const std::string& key, int64 defValue) const
   {
      std::map<std::string, ContextVal>::const_iterator it = _map.find(key);

      if (it == _map.end())
          return defValue;

      return it->second.isString == true ? defValue : it->second.lVal;
   }


   inline std::string Context::getString(const std::string& key, const std::string& defValue) const
   {
      std::map<std::string, ContextVal>::const_iterator it = _map.find(key);

      if (it == _map.end())
          return defValue;

      return it->second.isString == true ? it->second.sVal : defValue;
   }


   inline void Context::putInt(const std::string& key, int value)
   {
      _map[key] = ctxVal(false, value, "");
   }


   inline void Context::putLong(const std::string& key, int64 value)
   {
      _map[key] = ctxVal(false, value, "");
   }


   inline void Context::putString(const std::string& key, const std::string& value)
   {
      _map[key] = ctxVal(true, 0, value);
   }

}
#endif



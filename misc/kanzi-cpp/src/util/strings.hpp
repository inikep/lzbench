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
#ifndef _strings_
#define _strings_

#include <sstream>
#include <string>
#include <vector>



#if __cplusplus < 201103L
   // to_string() not available before C++ 11
   template <typename T>
   std::string to_string(T value)
   {
       std::ostringstream os;
       os << value;
       return os.str();
   }

   #define TOSTR(v) to_string(v)
#else
   #define TOSTR(v) std::to_string(v)
#endif


inline void to_binary(int num, char* buffer, int length)
{
    for (int i = length - 2; i >= 0; i--) {
        buffer[i] = (num & 1) ? '1' : '0';
        num >>= 1;
    }

    buffer[length - 1] = '\0';
}

// trim from end of string (right)
inline std::string& rtrim(std::string& s)
{
    static const char* whitespaces = " \t\f\v\n\r";
    std::size_t pos = s.find_last_not_of(whitespaces);

    if (pos != std::string::npos)
       s.erase(pos + 1);

    return s;
}

// trim from beginning of string (left)
inline std::string& ltrim(std::string& s)
{
    static const char* whitespaces = " \t\f\v\n\r";
    std::size_t pos = s.find_first_not_of(whitespaces);

    if (pos != std::string::npos)
       s.erase(0, pos);

    return s;
}

// trim from both ends of string (right then left)
inline std::string& trim(std::string& s)
{
    return ltrim(rtrim(s));
}

inline void tokenize(const std::string& str, std::vector<std::string>& v, char token)
{
   std::istringstream ss(str);
   std::string s;    

   while (getline(ss, s, token)) 
      v.push_back(s);   
}    

#endif


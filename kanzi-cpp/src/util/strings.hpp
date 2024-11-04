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
#ifndef _strings_
#define _strings_

#include <sstream>
#include <string>
#include <vector>


template <typename T>
std::string to_string(T value)
{
    std::ostringstream os;
    os << value;
    return os.str();
}


inline std::string __trim(std::string& str, bool left, bool right)
{
    if (str.empty())
        return str;

    std::string::size_type begin = 0;
    std::string::size_type end = str.length() - 1;

    if (left) {
       while (begin <= end && (str[begin] <= 0x20 || str[begin] == 0x7F))
          begin++;
    }

    if (right) {
       while (end > begin && (str[end] <= 0x20 || str[end] == 0x7F))
          end--;
    }

    return str.substr(begin, end - begin + 1);
}


inline std::string trim(std::string& str)  { return __trim(str, true, true); }
inline std::string ltrim(std::string& str) { return __trim(str, true, false); }
inline std::string rtrim(std::string& str) { return __trim(str, false, true); }


inline void tokenize(std::string str, std::vector<std::string>& v, char token)
{
   std::istringstream ss(str);
   std::string s;    

   while (getline(ss, s, token)) 
      v.push_back(s);   
}    

#endif


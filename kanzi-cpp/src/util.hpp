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
#ifndef _util_
#define _util_


#include <iostream>
#include "types.hpp"



// Ahem ... Visual Studio
// This ostreambuf class is required because Microsoft cannot bother to implement
// streambuf::pubsetbuf().
template <typename T>
struct ostreambuf : public std::basic_streambuf<T, std::char_traits<T> >
{
    ostreambuf(T* buffer, std::streamsize length) {
       this->setp(buffer, &buffer[length]);
    }
};

template <typename T>
struct istreambuf : public std::basic_streambuf<T, std::char_traits<T> >
{
    istreambuf(T* buffer, std::streamsize length) {
       this->setg(buffer, buffer, &buffer[length]);
    }
};

#endif


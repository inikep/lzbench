/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NVCOMP_CHECK_H
#define NVCOMP_CHECK_H

#include "nvcomp.h"
#include "nvcomp.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace nvcomp
{

class Check
{
public:
  static void not_null(
      const void* const ptr,
      const std::string& name,
      const std::string& filename,
      const int line);

  template <typename T>
  static void equal(
      const T& a,
      const T& b,
      const std::string& a_name,
      const std::string& b_name,
      const std::string& filename,
      const int line)
  {
    if (!(a == b)) {
      print_fail_position(filename, line);
      std::cerr << a_name << "(" << a << ")"
                << " != " << b_name << "(" << b << ")" << std::endl;
      throw NVCompException(nvcompErrorInternal, "CHECK_EQ Failed");
    }
  }

  static void
  api_call(nvcompError_t err, const std::string& filename, const int line);

  // NOTE: there is no C++11/C++14 standard way to get the function name.
  // In the future we could try to handle major compilers, and get the
  // name that way, as well as use the c++20 method.
  static nvcompError_t
  exception_to_error(const std::exception& e, const std::string& function_name);

private:
  static void print_fail_position(const std::string& filename, const int line);
};

} // namespace nvcomp

#define CHECK_API_CALL(call) Check::api_call(call, __FILE__, __LINE__)

#define CHECK_EQ(a, b) Check::equal(a, b, #a, #b, __FILE__, __LINE__)

#define CHECK_NOT_NULL(ptr) Check::not_null(ptr, #ptr, __FILE__, __LINE__)

#define API_WRAPPER(call, func_name)                                           \
  [](auto err) {                                                               \
    try {                                                                      \
      Check::api_call(err, __FILE__, __LINE__);                                \
    } catch (const std::exception& e) {                                        \
      return Check::exception_to_error(e, func_name);                          \
    }                                                                          \
    return err;                                                                \
  }(call)

#endif

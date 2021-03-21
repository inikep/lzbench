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

#include "Check.h"

namespace nvcomp
{

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

void Check::not_null(
    const void* const ptr,
    const std::string& name,
    const std::string& filename,
    const int line)
{
  if (ptr == nullptr) {
    print_fail_position(filename, line);
    throw std::runtime_error("'" + name + "' must not be null.");
  }
}

void Check::api_call(
    const nvcompError_t err, const std::string& filename, const int line)
{
  if (err != nvcompSuccess) {
    print_fail_position(filename, line);
    throw NVCompException(err, "API CALL FAILED");
  }
}

nvcompError_t Check::exception_to_error(
    const std::exception& e, const std::string& function_name)
{
  std::string context;
  if (!function_name.empty()) {
    context = "In " + function_name + ": ";
  }

  // generic error
  nvcompError_t err = nvcompErrorInvalidValue;

  // NOTE: this depends on RTTI being enabled.
  if (dynamic_cast<const NVCompException*>(&e)) {
    const NVCompException& nve = dynamic_cast<const NVCompException&>(e);
    err = nve.get_error();
  }

  std::cerr << "ERROR: " << context << e.what() << std::endl;
  return err;
}

void Check::print_fail_position(const std::string& filename, const int line)
{
  std::cerr << "CHECK FAILED: " << filename << ":" << line << std::endl;
}

} // namespace nvcomp

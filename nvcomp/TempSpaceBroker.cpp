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

#include "TempSpaceBroker.h"

#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>

namespace nvcomp
{

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

TempSpaceBroker::TempSpaceBroker(void* const space, const size_t bytes) :
    m_base(space),
    m_size(bytes),
    m_offset(0)
{
  assert(space);
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

size_t TempSpaceBroker::spaceLeft() const
{
  return m_size - m_offset;
}

/******************************************************************************
 * PRIVATE METHODS ************************************************************
 *****************************************************************************/

void* TempSpaceBroker::reserve(
    const size_t alignment, const size_t num, const size_t size)
{
  const size_t requiredSize = num * size;

  void* destPtr = next();

  size_t remaining = spaceLeft();
  if (!std::align(alignment, requiredSize, destPtr, remaining)) {
    throw std::runtime_error(
        "Not enough temp space left for " + std::to_string(num)
        + " values aligned to " + std::to_string(alignment) + ". Only "
        + std::to_string(remaining) + " bytes of " + std::to_string(m_size)
        + " bytes remain.");
  }

  const size_t totalSize = spaceLeft() - remaining + requiredSize;
  m_offset += totalSize;

  return destPtr;
}

void* TempSpaceBroker::next() const
{
  return static_cast<char*>(m_base) + m_offset;
}

} // namespace nvcomp

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

#include "Metadata.h"
#include "common.h"

#include <cstddef>

namespace nvcomp
{

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

Metadata::Metadata(
    const nvcompType_t type,
    const size_t uncompressedBytes,
    const size_t compressedBytes,
    const int compressionType) :
    m_type(type),
    m_uncompressedBytes(uncompressedBytes),
    m_compressedBytes(compressedBytes),
    m_compressionType(compressionType)
{
  if (m_uncompressedBytes % sizeOfnvcompType(m_type) != 0) {
    throw std::runtime_error(
        "Number of uncompressed bytes is not a multiple "
        " of the size of the type: "
        + std::to_string(m_uncompressedBytes) + " % "
        + std::to_string(sizeOfnvcompType(m_type)));
  }
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

nvcompType_t Metadata::getValueType() const
{
  return m_type;
}

size_t Metadata::getUncompressedSize() const
{
  return m_uncompressedBytes;
}

size_t Metadata::getCompressedSize() const
{
  return m_compressedBytes;
}

size_t Metadata::getNumUncompressedElements() const
{
  return getUncompressedSize() / sizeOfnvcompType(m_type);
}

int Metadata::getCompressionType() const
{
  return m_compressionType;
}

void Metadata::setUncompressedSize(const size_t bytes)
{
  m_uncompressedBytes = bytes;
}

void Metadata::setCompressedSize(const size_t bytes)
{
  m_compressedBytes = bytes;
}

/******************************************************************************
 * PROTECTED METHODS **********************************************************
 *****************************************************************************/

} // namespace nvcomp

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

#include "CascadedMetadata.h"
#include "common.h"
#include "CascadedCommon.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <iostream>

namespace nvcomp
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

using Header = CascadedMetadata::Header;

namespace
{

constexpr const size_t NULL_OFFSET = static_cast<size_t>(-1);
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

CascadedMetadata::CascadedMetadata(
    const nvcompCascadedFormatOpts opts,
    const nvcompType_t type,
    const size_t uncompressedBytes,
    const size_t compressedBytes) :
    Metadata(type, uncompressedBytes, compressedBytes, COMPRESSION_ID),
    m_formatOpts(opts),
    m_headers(),
    m_dataOffsets(),
    m_dataType(),
    m_isSaved()
{
  if (static_cast<size_t>(m_formatOpts.num_RLEs) > MAX_NUM_RLES) {
    throw std::runtime_error(
        "Invalid number of RLEs: " + std::to_string(m_formatOpts.num_RLEs)
        + ", maximum is " + std::to_string(MAX_NUM_RLES));
  }

  initialize();
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

int CascadedMetadata::getNumRLEs() const
{
  return m_formatOpts.num_RLEs;
}

int CascadedMetadata::getNumDeltas() const
{
  return m_formatOpts.num_deltas;
}

unsigned int CascadedMetadata::getNumInputs() const
{
  // Determine the number of unique data id's that will be present in the
  // compressed data. Currently, the input data gets assigned a value of 0,
  // each RLE produces a runs and values which each take an id, and delta also
  // produces a values which takes an id. Bitpacking is not treated as a layer
  // unless it is the only present operation, which is what the 'max' below is
  // for.
  return static_cast<unsigned int>(
      std::max(1, m_formatOpts.num_RLEs * 2 + m_formatOpts.num_deltas) + 1);
}

bool CascadedMetadata::useBitPacking() const
{
  return m_formatOpts.use_bp;
}

void CascadedMetadata::setHeader(const size_t index, const Header header)
{
  if (index >= getNumInputs()) {
    throw std::runtime_error(
        "Invalid header index to set: " + std::to_string(index) + " / "
        + std::to_string(getNumInputs()));
  }

  assert(index < m_headers.size());

  m_headers[index] = header;
}

void CascadedMetadata::setDataOffset(const size_t index, const size_t offset)
{
  if (index >= getNumInputs()) {
    throw std::runtime_error(
        "Invalid data index to set: " + std::to_string(index) + " / "
        + std::to_string(getNumInputs()));
  }

  m_dataOffsets[index] = offset;
}

Header CascadedMetadata::getHeader(const size_t index) const
{
  if (index >= getNumInputs()) {
    throw std::runtime_error(
        "Invalid header index to set: " + std::to_string(index) + " / "
        + std::to_string(getNumInputs()));
  }
  assert(index < m_headers.size());

  return m_headers[index];
}

size_t CascadedMetadata::getNumElementsOf(const size_t index) const
{
  return m_headers[index].length;
}

size_t CascadedMetadata::getDataOffset(const size_t index) const
{
  if (index >= getNumInputs()) {
    throw std::runtime_error(
        "Invalid data index to set: " + std::to_string(index) + " / "
        + std::to_string(getNumInputs()));
  }

  const size_t offset = m_dataOffsets[index];

  if (offset == NULL_OFFSET) {
    throw std::runtime_error(
        "Cannot get data offset which has not been set: "
        + std::to_string(index));
  }

  return offset;
}

bool CascadedMetadata::haveAnyOffsetsBeenSet() const
{
  for (const size_t offset : m_dataOffsets) {
    if (offset != NULL_OFFSET) {
      return true;
    }
  }

  return false;
}

bool CascadedMetadata::haveAllOffsetsBeenSet() const
{
  for (const size_t offset : m_dataOffsets) {
    if (offset == NULL_OFFSET) {
      return false;
    }
  }

  return true;
}

bool CascadedMetadata::isSaved(const size_t index) const
{
  if (index >= getNumInputs()) {
    throw std::runtime_error(
        "Invalid data index to check if saved: " + std::to_string(index) + " / "
        + std::to_string(getNumInputs()));
  }

  return m_isSaved[index];
}

nvcompType_t CascadedMetadata::getDataType(size_t index) const
{
  if (index >= getNumInputs()) {
    throw std::runtime_error(
        "Invalid data index to get type of: " + std::to_string(index) + " / "
        + std::to_string(getNumInputs()));
  }

  return m_dataType[index];
}

/******************************************************************************
 * PRIVATE METHODS ************************************************************
 *****************************************************************************/

void CascadedMetadata::initialize()
{
  m_headers.resize(getNumInputs());
  m_dataOffsets.resize(getNumInputs(), NULL_OFFSET);
  m_dataType.resize(getNumInputs(), getValueType());
  m_isSaved.resize(getNumInputs(), false);

  // fill out data based on tree
  const int numRLEs = getNumRLEs();
  const int numDeltas = getNumDeltas();
  const bool bitPacking = useBitPacking();

  int vals_id = 0;

  const int numSteps = std::max(numRLEs, numDeltas);
  for (int r = numSteps - 1; r >= 0; r--) {
    int nextValId;

    if (numSteps - r - 1 < numRLEs) {
      const int runId = ++vals_id;
      const int valId = ++vals_id;

      // rle

      if (numRLEs - 1 - r < numDeltas) {
        // delta
        nextValId = ++vals_id;
      } else {
        nextValId = valId;
      }

      // save runs output `runId`
      m_isSaved[runId] = true;
      if (bitPacking) {
        m_dataType[runId] = NVCOMP_TYPE_BITS;
      } else {
        m_dataType[runId] = selectRunsType(getNumUncompressedElements());
      }
    } else {
      // delta only
      nextValId = ++vals_id;
    }

    if (r == 0) {
      // save last layer `nextValId`
      m_isSaved[nextValId] = true;
      if (bitPacking) {
        m_dataType[nextValId] = NVCOMP_TYPE_BITS;
      } else {
        m_dataType[nextValId] = getValueType();
      }
    }
  }

  // If there are no RLEs or Deltas, we will do a single BP step.
  if (numRLEs == 0 && numDeltas == 0) {
    const int nextValId = ++vals_id;

    // bit pack `nextValId`
    m_isSaved[nextValId] = true;
    m_dataType[nextValId] = NVCOMP_TYPE_BITS;
  }
}

} // namespace nvcomp

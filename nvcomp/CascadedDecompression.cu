/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "CascadedCompressionGPU.h"
#include "CascadedMetadata.h"
#include "CascadedMetadataOnGPU.h"

#include "CascadedDecompressionKernels.cuh"
#include "Check.h"
#include "CudaUtils.h"
#include "cascaded.h"
#include "nvcomp.h"
#include "nvcomp.hpp"
#include "type_macros.h"
#include "unpack.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <cub/cub.cuh>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#ifdef USE_RMM
#include <rmm/rmm.h>
#endif

#include <cassert>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

// align all temp allocations by 512B
#define CUDA_MEM_ALIGN(size) (((size) + 0x1FF) & ~0x1FF)

#ifndef RLE_THREAD_BLOCK
#define RLE_THREAD_BLOCK 128
#endif

#ifndef RLE_ELEMS_PER_THREAD
#define RLE_ELEMS_PER_THREAD 4
#endif

#define RLE_ELEMS_PER_BLOCK (RLE_THREAD_BLOCK * RLE_ELEMS_PER_THREAD)

namespace nvcomp
{

// internal representations: one kernel per scheme
enum nvcompScheme_t
{
  NVCOMP_SCHEME_BP,
  NVCOMP_SCHEME_RLE,
  NVCOMP_SCHEME_DELTA,
  NVCOMP_SCHEME_RLE_DELTA, // automatically fused RLE+Delta to reduce mem
                           // traffic
};

struct nvcompLayer_t;

struct nvcompDataNode_t
{
  void* ptr;
  nvcompType_t type;
  int packing;

  nvcompLayer_t* parentLayer;
  size_t length;

  // to enable BP as a separate layer, default -1
  int pointToId;
};

struct nvcompLayer_t
{
  nvcompScheme_t scheme;
  size_t maxOutputSize;

  nvcompDataNode_t* vals;
  nvcompDataNode_t* runs;
  nvcompDataNode_t* output;

  // TODO: can we get rid of those
  int valId;
  int runId;
  int outputId;
};

struct nvcompIntConfig_t
{
  int outputId = 0;
  nvcompType_t outputType = NVCOMP_TYPE_INT;
  size_t maxOutputSize = 0;

  std::list<nvcompLayer_t> layers = {};
  std::map<int, nvcompDataNode_t> nodes
      = {}; // TODO: should we make this nvcompData_t instead of int?

  // compute the workspace size
  size_t getWorkspaceBytes();
  size_t getWorkspaceBytes(nvcompDataNode_t* node);

  // fuse kernels, etc.
  void optimizeLayers();
};

struct nvcompIntTask_t
{
  // TODO: add CUDA event assigned to this task
};

struct nvcompIntHandle_t
{
  std::unique_ptr<nvcompIntConfig_t> config = nullptr;
  cudaStream_t stream = 0;

  // main decomp functions
  template <typename outputT>
  nvcompError_t decompCPU(
      nvcompDataNode_t* node, const void** inputData, const void** h_headers);
  template <typename outputT, typename runT>
  nvcompError_t decompGPU(
      nvcompDataNode_t* node,
      const void** inputData,
      const void** h_headers,
      cudaStream_t stream);

  // workspace memory
  size_t workspaceBytes = 0;
  void* workspaceStorage = nullptr;

  // workspace mem management
  nvcompError_t release();
  nvcompError_t
  allocateAsync(); // new function that splits of pre-allocated memory

  // workspace breakdown
  size_t max_input_len = 0;  // maximum input RLE length
  size_t max_output_len = 0; // maximum output RLE length

  void* temp_val = nullptr;    // temp RLE val expansions
  void* temp_run = nullptr;    // temp RLE run expansions
  void* temp_delta = nullptr;  // temp Delta expansions
  void* temp_output = nullptr; // temp Delta expansions

  // cub scan memory
  size_t temp_scan_bytes = 0;
  void* temp_scan = nullptr;

  // block indices start and offsets
  size_t max_num_blocks = 0;
  size_t* start_ind = nullptr;
  size_t* start_off = nullptr;
};

template <typename keyT, typename valueT>
struct SharedMap
{
  std::map<keyT, valueT> data = {};
  std::mutex m = {};

  // find the next available id
  keyT find_next()
  {
    std::lock_guard<std::mutex> guard(m);
    int id = 0;
    while (data.find(id) != data.end())
      id++;
    return (keyT)id;
  }

  bool exists(const keyT& key)
  {
    std::lock_guard<std::mutex> guard(m);
    return data.find(key) != data.end();
  }

  void insert(const keyT& key, const valueT& val)
  {
    std::lock_guard<std::mutex> guard(m);
    if (data.find(key) == data.end())
      data[key] = val;
  }

  valueT& operator[](const keyT& key)
  {
    std::lock_guard<std::mutex> guard(m);
    return data[key];
  }

  void erase(const keyT& key)
  {
    std::lock_guard<std::mutex> guard(m);
    data.erase(key);
  }
};

// internal collections
SharedMap<nvcompConfig_t, nvcompIntConfig_t> configs;
SharedMap<nvcompHandle_t, nvcompIntHandle_t> handles;

// TODO: can we get rid of these?
std::mutex config_mutex;
std::mutex handle_mutex;

namespace
{

template <typename T>
void cubDeviceScanTempSpace(size_t& temp_scan_bytes, const size_t max_input_len)
{
  void* temp_scan = nullptr;
  T* temp_run = nullptr;

  cub::DeviceScan::InclusiveSum(
      temp_scan, temp_scan_bytes, temp_run, temp_run, max_input_len);
}

void checkCompressSize(const size_t numBytes)
{
  const size_t maxBytes = static_cast<size_t>(std::numeric_limits<int>::max());
  if (numBytes > maxBytes) {
    throw std::runtime_error(
        "Cascaded compression can only compress up to a maximum of "
        + std::to_string(maxBytes) + " bytes at a time (requested "
        + std::to_string(numBytes) + " bytes).");
  }
}

std::unique_ptr<nvcompIntConfig_t> generateConfig(const CascadedMetadata* const metadata)
{
  const int numRLEs = metadata->getNumRLEs();
  const int numDeltas = metadata->getNumDeltas();
  const bool bitPacking = metadata->useBitPacking();

  int vals_id = 0;

  // initialize config
  const nvcompType_t type = metadata->getValueType();

  std::unique_ptr<nvcompIntConfig_t> config(new nvcompIntConfig_t);
  config->outputId = vals_id;
  config->outputType = type;
  config->maxOutputSize = metadata->getUncompressedSize();

  const nvcompType_t runType
      = selectRunsType(metadata->getNumUncompressedElements());

  const size_t maxSegmentSize = metadata->getUncompressedSize();

  config->nodes[0].length = metadata->getNumUncompressedElements();

  // A step can be RLE+Delta, RLE, or Delta, with final outputs conditionally
  // having bit packing applied
  const int numSteps = std::max(numRLEs, numDeltas);
  for (int r = numSteps - 1; r >= 0; r--) {
    const int inputId = vals_id;

    if (numSteps - r - 1 < numRLEs) {
      const int runId = ++vals_id;
      const int valId = ++vals_id;

      // add to config
      nvcompConfigAddRLE_BP(
          config.get(),
          inputId,
          maxSegmentSize,
          valId,
          type,
          bitPacking,
          runId,
          runType,
          bitPacking);
      config->nodes[valId].length = metadata->getNumElementsOf(valId);
      config->nodes[runId].length = metadata->getNumElementsOf(runId);

      // store vals (apply delta if necessary)
      if (numRLEs - 1 - r < numDeltas) {
        const int deltaId = ++vals_id;

        if (r == 0) {
          nvcompConfigAddDelta_BP(
              config.get(), valId, maxSegmentSize, deltaId, type, bitPacking);
        } else {
          nvcompConfigAddDelta_BP(
              config.get(),
              valId,
              maxSegmentSize,
              deltaId,
              type,
              0); // no bitpacking when delta is used as an intermediate step
        }
        config->nodes[deltaId].length = metadata->getNumElementsOf(deltaId);
      }
    } else {
      // RLE-less step
      const int deltaId = ++vals_id;

      if (r == 0) {
        nvcompConfigAddDelta_BP(
            config.get(), inputId, maxSegmentSize, deltaId, type, bitPacking);
      } else {
        nvcompConfigAddDelta_BP(
            config.get(),
            inputId,
            maxSegmentSize,
            deltaId,
            type,
            0); // no bitpacking when delta is used as an intermediate step
      }
      config->nodes[deltaId].length = metadata->getNumElementsOf(deltaId);
    }
  }

  // If there are no RLEs or Deltas, we will do a single BP step.
  if (numRLEs == 0 && numDeltas == 0) {
    const int inputId = vals_id;
    const int bpId = ++vals_id;
    nvcompConfigAddBP(config.get(), inputId, maxSegmentSize, bpId, type);

    config->nodes[bpId].length = metadata->getNumElementsOf(bpId);
  }

  return config;
}

template <typename T>
constexpr bool isFixedWidth()
{
  return std::is_same<T, char>::value || std::is_same<T, int8_t>::value
         || std::is_same<T, uint8_t>::value || std::is_same<T, int16_t>::value
         || std::is_same<T, uint16_t>::value || std::is_same<T, int32_t>::value
         || std::is_same<T, uint32_t>::value || std::is_same<T, int64_t>::value
         || std::is_same<T, uint64_t>::value;
}

template <typename T>
size_t writeFixedWidthData(
    const T* const val,
    void* const ptr,
    const size_t offset,
    const size_t maxSize)
{
  assert(isFixedWidth<T>());

  size_t newOffset = offset + sizeof(*val);

  if (ptr) {
    // only write if we're doing a really output
    if (newOffset > maxSize) {
      throw std::runtime_error(
          "Not enough room to write member, need at least "
          + std::to_string(newOffset) + " bytes, but given only "
          + std::to_string(maxSize));
    }

    memcpy(static_cast<char*>(ptr) + offset, val, sizeof(*val));
  }

  return newOffset;
}

template <typename T>
size_t writeData(
    const T* const val,
    void* const ptr,
    const size_t offset,
    const size_t maxSize)
{
  if (isFixedWidth<T>()) {
    return writeFixedWidthData(val, ptr, offset, maxSize);
  } else if (std::is_same<T, bool>::value) {
    const int8_t typedVal = static_cast<int8_t>(*val);
    return writeData(&typedVal, ptr, offset, maxSize);
  } else if (std::is_same<T, int>::value) {
    // on most systems this will not be used, as int32_t is usually defined as
    // int
    const int32_t typedVal = static_cast<int32_t>(*val);
    return writeData(&typedVal, ptr, offset, maxSize);
  } else if (std::is_same<T, size_t>::value) {
    const uint64_t typedVal = static_cast<uint64_t>(*val);
    return writeData(&typedVal, ptr, offset, maxSize);
  } else {
    throw std::runtime_error("Unsupported type for serialization.");
  }
}

} // namespace


/**************************************************************************************
 *            Older API definitions below.  New API calls rely on them.
 **************************************************************************************/

nvcompIntConfig_t * createConfig(const CascadedMetadata* metadata)
{
  return generateConfig(metadata).release();
}

void destroyConfig(nvcompIntConfig_t* config)
{
  delete config;
}

nvcompError_t nvcompConfigAddRLE_BP(
    nvcompIntConfig_t* const config,
    int outputId,
    size_t maxOutputSize,
    int valId,
    nvcompType_t valType,
    int valPacking,
    int runId,
    nvcompType_t runType,
    int runPacking)
{
  nvcompIntConfig_t& c = *config;

  // setup input nodes if necessary
  if (c.nodes.find(valId) == c.nodes.end()) {
    c.nodes[valId] = {NULL, valType, valPacking, NULL, 0, 0};
  }
  if (c.nodes.find(runId) == c.nodes.end()) {
    c.nodes[runId] = {NULL, runType, runPacking, NULL, 0, 0};
  }

  // create the output node if necessary
  if (c.nodes.find(outputId) == c.nodes.end()) {
    c.nodes[outputId] = {NULL, valType, 0, NULL, 0, 0};
  }

  nvcompLayer_t layer = {NVCOMP_SCHEME_RLE,
                         maxOutputSize,
                         NULL,
                         NULL,
                         NULL,
                         valId,
                         runId,
                         outputId};
  c.layers.push_back(layer);
  c.nodes[outputId].parentLayer = &c.layers.back();

  return nvcompSuccess;
}

nvcompError_t nvcompConfigAddDelta_BP(
    nvcompIntConfig_t* const config,
    int outputId,
    size_t maxOutputSize,
    int valId,
    nvcompType_t valType,
    int valPacking)
{
  nvcompIntConfig_t& c = *config;

  // setup the input node if necessary
  if (c.nodes.find(valId) == c.nodes.end()) {
    c.nodes[valId] = {NULL, valType, valPacking, NULL, 0, 0};
  }

  // create the output node if necessary
  if (c.nodes.find(outputId) == c.nodes.end()) {
    c.nodes[outputId] = {NULL, valType, 0, NULL, 0, 0};
  }

  nvcompLayer_t layer = {NVCOMP_SCHEME_DELTA,
                         maxOutputSize,
                         NULL,
                         NULL,
                         NULL,
                         valId,
                         -1,
                         outputId};
  c.layers.push_back(layer);
  c.nodes[outputId].parentLayer = &c.layers.back();

  return nvcompSuccess;
}

nvcompError_t nvcompConfigAddBP(
    nvcompIntConfig_t* const config,
    int outputId,
    size_t maxOutputSize,
    int valId,
    nvcompType_t valType)
{
  nvcompIntConfig_t& c = *config;

  // setup the input node if necessary
  if (c.nodes.find(valId) == c.nodes.end()) {
    c.nodes[valId] = {NULL, valType, 1, NULL, 0, 0};
  }

  // create the output node if necessary
  if (c.nodes.find(outputId) == c.nodes.end()) {
    c.nodes[outputId] = {NULL, valType, 0, NULL, 0, 0};
  }

  nvcompLayer_t layer = {
      NVCOMP_SCHEME_BP, maxOutputSize, NULL, NULL, NULL, valId, -1, outputId};
  c.layers.push_back(layer);
  c.nodes[outputId].parentLayer = &c.layers.back();

  return nvcompSuccess;
}

size_t nvcompIntConfig_t::getWorkspaceBytes(nvcompDataNode_t* /*node*/)
{
  // TODO: allocate output buffers for each node except the terminal one
  // currently this is done inside decompGPU which will break concurrency (once
  // we add streams)
  return 0;
}

size_t nvcompIntConfig_t::getWorkspaceBytes()
{
  if (nodes.find(outputId) == nodes.end()) {
    throw std::runtime_error(
        "getWorkspaceBytes(): could not find output ID amongst nodes: "
        + std::to_string(outputId) + " with " + std::to_string(nodes.size())
        + " nodes.");
  }
  if (nodes[outputId].parentLayer == NULL) {
    throw std::runtime_error("getWorkspaceBytes(): the output node is not used "
                             "by any compression layers.");
  }

  int numRLEs = 0;
  int numDeltas = 0;

  size_t max_input_len = 0;
  for (const nvcompLayer_t& layer : layers) {
    if (layer.scheme == NVCOMP_SCHEME_RLE
        || layer.scheme == NVCOMP_SCHEME_RLE_DELTA) {
      ++numRLEs;
    }
    if (layer.scheme == NVCOMP_SCHEME_DELTA
        || layer.scheme == NVCOMP_SCHEME_RLE_DELTA) {
      ++numDeltas;
    }

    const size_t layer_len = nodes[layer.valId].length;
    if (layer_len > max_input_len) {
      max_input_len = layer_len;
    }
  }

  const size_t max_output_len = maxOutputSize;

  size_t size = 0;

  // temp vals, runs, delta, output
  if (numRLEs > 0 || numDeltas > 0) {
    size += CUDA_MEM_ALIGN(max_input_len * sizeOfnvcompType(outputType));
    if (numRLEs > 0) {
      size += CUDA_MEM_ALIGN(
          max_input_len * sizeOfnvcompType(selectRunsType(maxOutputSize)));
    }
    size += CUDA_MEM_ALIGN(max_input_len * sizeOfnvcompType(outputType));
    size += CUDA_MEM_ALIGN(max_input_len * sizeOfnvcompType(outputType));
  }

  size_t temp_scan_bytes_run = 0;
  size_t temp_scan_bytes_delta = 0;
  NVCOMP_TYPE_ONE_SWITCH(
      selectRunsType(max_output_len),
      cubDeviceScanTempSpace,
      temp_scan_bytes_run,
      max_input_len);
  NVCOMP_TYPE_ONE_SWITCH(
      outputType, cubDeviceScanTempSpace, temp_scan_bytes_delta, max_input_len);
  size_t temp_scan_bytes = std::max(temp_scan_bytes_run, temp_scan_bytes_delta);

  size += CUDA_MEM_ALIGN(temp_scan_bytes);

  size_t max_num_blocks
      = (max_output_len + RLE_ELEMS_PER_BLOCK - 1) / RLE_ELEMS_PER_BLOCK;
  size += CUDA_MEM_ALIGN((max_num_blocks + 1) * sizeof(size_t));
  size += CUDA_MEM_ALIGN((max_num_blocks + 1) * sizeof(size_t));

  return size;
}

nvcompError_t nvcompIntHandle_t::release()
{
  return nvcompSuccess;
}

// recursively assign memory for all nodes in our DAG
// ** Assumes worspaceStorage is already allocated with sufficient space **
nvcompError_t nvcompIntHandle_t::allocateAsync()
{
  nvcompIntConfig_t& c = *config;

  nvcompType_t outputType = c.outputType;

  // assign member variables for size
  max_output_len = c.maxOutputSize;
  max_input_len = 0;
  int numRLEs = 0;
  int numDeltas = 0;
  for (const nvcompLayer_t& layer : c.layers) {
    if (layer.scheme == NVCOMP_SCHEME_RLE
        || layer.scheme == NVCOMP_SCHEME_RLE_DELTA) {
      ++numRLEs;
    }
    if (layer.scheme == NVCOMP_SCHEME_DELTA
        || layer.scheme == NVCOMP_SCHEME_RLE_DELTA) {
      ++numDeltas;
    }

    const size_t layer_len = c.nodes[layer.valId].length;
    if (layer_len > max_input_len) {
      max_input_len = layer_len;
    }
  }

  unsigned char* ptr = (unsigned char*)workspaceStorage;

  // temporary buffers that can hold RLE expansions and other data, but we will
  // re-use locations
  if (numRLEs > 0 || numDeltas > 0) {
    temp_val = ptr;
    ptr += CUDA_MEM_ALIGN(max_input_len * sizeOfnvcompType(outputType));
    if (numRLEs > 0) {
      temp_run = ptr;
      ptr += CUDA_MEM_ALIGN(
          max_input_len * sizeOfnvcompType(selectRunsType(max_output_len)));
    }
    temp_delta = ptr;
    ptr += CUDA_MEM_ALIGN(max_input_len * sizeOfnvcompType(outputType));

    // one additional buffer for delta expansion
    // TODO: can we get rid of this one?
    temp_output = ptr;
    ptr += CUDA_MEM_ALIGN(max_input_len * sizeOfnvcompType(outputType));
  }

  // allocate temp storage for cub scan using the largest size_t
  // this temp storage will be reused by delta and runs scans of different types
  temp_scan = ptr;

  size_t temp_scan_bytes_run = 0;
  size_t temp_scan_bytes_delta = 0;
  NVCOMP_TYPE_ONE_SWITCH(
      selectRunsType(max_output_len),
      cubDeviceScanTempSpace,
      temp_scan_bytes_run,
      max_input_len);
  NVCOMP_TYPE_ONE_SWITCH(
      outputType, cubDeviceScanTempSpace, temp_scan_bytes_delta, max_input_len);
  temp_scan_bytes = std::max(temp_scan_bytes_run, temp_scan_bytes_delta);
  ptr += CUDA_MEM_ALIGN(temp_scan_bytes);

  // block indices/offsets
  max_num_blocks
      = (max_output_len + RLE_ELEMS_PER_BLOCK - 1) / RLE_ELEMS_PER_BLOCK;
  start_ind = (size_t*)ptr;
  ptr += CUDA_MEM_ALIGN((max_num_blocks + 1) * sizeof(size_t));
  start_off = (size_t*)ptr;
  ptr += CUDA_MEM_ALIGN((max_num_blocks + 1) * sizeof(size_t));

  return nvcompSuccess;
}

// here we do kernel fusion
void nvcompIntConfig_t::optimizeLayers()
{
  for (auto it = layers.begin(); it != layers.end();) {
    if (it->scheme == NVCOMP_SCHEME_DELTA) {
      int valId = it->valId;
      int outputId = it->outputId;
      if (nodes.find(valId) != nodes.end() && nodes[valId].parentLayer != NULL
          && nodes[valId].parentLayer->scheme == NVCOMP_SCHEME_RLE) {
        nodes[outputId].parentLayer = nodes[valId].parentLayer;
        nodes[outputId].parentLayer->scheme = NVCOMP_SCHEME_RLE_DELTA;
        nodes[outputId].parentLayer->outputId = outputId;
        it = layers.erase(it);
        continue;
      }
    }
    it++;
  }
}

/* These functions may not be needed and removed to simplify codebase */
nvcompError_t nvcompSetWorkspace(
    nvcompHandle_t /*handle*/,
    void* /*workspaceStorage*/,
    size_t /*workspaceBytes*/)
{
  std::cerr << "ERROR: nvcompSetWorkspace is not implemented yet!" << std::endl;
  return nvcompErrorNotSupported;
}

nvcompError_t
nvcompGetWorkspaceSize(nvcompHandle_t handle, size_t* workspaceBytes)
{
  *workspaceBytes = handles[handle].workspaceBytes;

  return nvcompSuccess;
}

nvcompError_t nvcompSetStream(nvcompHandle_t handle, cudaStream_t streamId)
{
  handles[handle].stream = streamId;

  return nvcompSuccess;
}

nvcompError_t nvcompGetStream(nvcompHandle_t handle, cudaStream_t* streamId)
{
  *streamId = handles[handle].stream;

  return nvcompSuccess;
}

// if the header is not packed this will shallow copy the pointer
// otherwise unpack into the output buffer
template <typename inputT, typename outputT>
void unpackCpu(
    outputT** output, nvcompDataNode_t* node, const void* hdr, const void* data)
{
  const CascadedMetadata::Header header
      = *static_cast<const CascadedMetadata::Header*>(hdr);
  if (node->packing) {
    for (size_t i = 0; i < header.length; ++i) {
      const inputT minValue
          = *CascadedMetadata::getMinValueLocation<inputT>(&header);
      (*output)[i] = unpackBytes(data, header.numBits, minValue, i);
    }
  } else {
    if (typeid(inputT) == typeid(outputT)) {
      *output = (outputT*)data;
    } else {
      for (size_t i = 0; i < header.length; i++)
        (*output)[i] = (outputT)((inputT*)data)[i];
    }
  }
}

template <typename outputT>
void unpackCpu(
    outputT** output, nvcompDataNode_t* node, const void* hdr, const void* data)
{
  NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY(
      node->type, outputT, unpackCpu, output, node, hdr, data);
}

// if the header is not packed this will shallow copy the pointer if it's
// accessible from the GPU otherwise copy or unpack into the output buffer
template <typename inputT, typename outputT>
void unpackGpu(
    outputT* d_output,
    nvcompDataNode_t* node,
    const void* data,
    const void* h_hdr,
    cudaStream_t stream)
{
  void* d_input = NULL;

  // prepare input data
  cudaPointerAttributes attr;

  cudaError_t err = cudaPointerGetAttributes(&attr, data);
  if (err != cudaSuccess) {
    std::ostringstream oss;
    oss << data;
    throw std::runtime_error(
        "unpackGpu(): Failed to get pointer attributes for " + oss.str()
        + " due to: " + std::to_string(err));
  }

  if (attr.type != cudaMemoryTypeUnregistered) {
    // memory is accessible to the GPU
    d_input = attr.devicePointer;
  } else {
    throw std::runtime_error("unpackGpu(): Data not accessible to the GPU");
  }

  // Get length of run from the host-side header
  size_t length = static_cast<const CascadedMetadata::Header*>(h_hdr)->length;

  CascadedMetadata::Header header
      = *static_cast<const CascadedMetadata::Header*>(h_hdr);
  const unsigned char numBits = header.numBits;
  const inputT minValue
      = *CascadedMetadata::getMinValueLocation<inputT>(&header);

  const dim3 block(512);
  const dim3 grid(roundUpDiv(length, block.x));
  if (node->packing) {
    unpackBytesKernel<<<grid, block, 0, stream>>>(
        d_input, d_output, numBits, minValue, length);
  } else {
    convertKernel<<<grid, block, 0, stream>>>(
        static_cast<const inputT*>(d_input), d_output, length);
  }
}

template <typename outputT>
void unpackGpu(
    outputT* d_output,
    nvcompDataNode_t* node,
    const void* data,
    const void* h_hdr,
    cudaStream_t stream)
{
  NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY(
      node->type, outputT, unpackGpu, d_output, node, data, h_hdr, stream);
}

template <typename outputT>
nvcompError_t nvcompIntHandle_t::decompCPU(
    nvcompDataNode_t* node, const void** inputHdrs, const void** inputData)
{
  size_t maxOutputSize = config->maxOutputSize;

  std::vector<outputT> unpacked_vals;
  std::vector<size_t> unpacked_runs;
  outputT* vals_data = NULL;
  size_t* runs_data = NULL;
  size_t vals_len;

  nvcompLayer_t* layer = node->parentLayer;

  // add BP only layer
  if (layer->scheme == NVCOMP_SCHEME_BP) {
    unpacked_vals.resize(maxOutputSize);
    vals_data = &unpacked_vals[0];
    unpackCpu(
        &vals_data,
        layer->vals,
        inputHdrs[layer->valId],
        inputData[layer->valId]);
    vals_len
        = static_cast<const CascadedMetadata::Header*>(inputHdrs[layer->valId])
              ->length;

    node->length = vals_len;

    // lazy allocation
    // TODO: move to allocate()
    if (node->ptr == NULL)
      node->ptr = new outputT[vals_len];

    // copy and convert type if necessary
    for (int i = 0; i < vals_len; i++) {
      ((outputT*)(node->ptr))[i] = vals_data[i];
    }
    return nvcompSuccess;
  }

  // compute vals
  if (layer->vals->parentLayer != NULL) {
    decompCPU<outputT>(layer->vals, inputHdrs, inputData);
    vals_data = (outputT*)layer->vals->ptr;
    vals_len = layer->vals->length;
  } else {
    unpacked_vals.resize(maxOutputSize);
    vals_data = &unpacked_vals[0];
    unpackCpu(
        &vals_data,
        layer->vals,
        inputHdrs[layer->valId],
        inputData[layer->valId]);
    vals_len
        = static_cast<const CascadedMetadata::Header*>(inputHdrs[layer->valId])
              ->length;
  }

  // compute runs
  if (layer->runs != NULL) {
    if (layer->runs->parentLayer != NULL) {
      decompCPU<size_t>(layer->runs, inputHdrs, inputData);
      runs_data = (size_t*)layer->runs->ptr;
    } else {
      unpacked_runs.resize(maxOutputSize);
      runs_data = &unpacked_runs[0];
      unpackCpu(
          &runs_data,
          layer->runs,
          inputHdrs[layer->runId],
          inputData[layer->runId]);
    }
  }

  // decompress (this is using additional memory)
  std::vector<outputT> next;
  next.clear();
  switch (layer->scheme) {
  case NVCOMP_SCHEME_RLE: {
    for (int i = 0; i < vals_len; i++)
      next.insert(next.end(), runs_data[i], vals_data[i]);
    break;
  }
  case NVCOMP_SCHEME_RLE_DELTA: {
    for (int i = 0; i < vals_len; i++)
      next.insert(next.end(), runs_data[i], vals_data[i]);
    for (int i = 1; i < next.size(); i++)
      next[i] += next[i - 1];
    break;
  }
  case NVCOMP_SCHEME_DELTA: {
    next.resize(vals_len);
    next[0] = vals_data[0];
    for (int i = 1; i < vals_len; i++)
      next[i] = next[i - 1] + vals_data[i];
    break;
  }
  default:
    return nvcompErrorNotSupported;
  }

  node->length = next.size();

  // lazy allocation
  // TODO: move to allocate()
  if (node->ptr == NULL)
    node->ptr = new outputT[next.size()];

  // copy and convert type if necessary
  for (int i = 0; i < next.size(); i++)
    ((outputT*)(node->ptr))[i] = next[i];

  return nvcompSuccess;
}

// Perform Cascaded decompression on the GPU.
// Assumes all workspace is pre-allocated and assigned, inputHdrs and inputData
// are GPU-accessible, and h_headers is CPU-accessible
template <typename outputT, typename runT>
nvcompError_t nvcompIntHandle_t::decompGPU(
    nvcompDataNode_t* node,
    const void** inputData,
    const void** h_headers,
    cudaStream_t stream = NULL)
{
  // prepare device output buffer if necessary
  // TODO: move to the init step
  cudaPointerAttributes attr;
  outputT* out_ptr = NULL;

  // get typed copies of pointers to avoid casting
  outputT* const localOutput = static_cast<outputT*>(temp_output);
  outputT* const localDelta = static_cast<outputT*>(temp_delta);
  runT* const localRun = static_cast<runT*>(temp_run);

  if (node->ptr == nullptr) {
    throw std::runtime_error(
        "nvcompIntHandle_t::decompGPU(): Got node with null ptr.");
  }

  cudaError_t err = cudaPointerGetAttributes(&attr, node->ptr);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "nvcompIntHandle_t::decompGPU(): Failed to get cuda pointer "
        "attributes: "
        + std::to_string(err));
  }

  if (attr.type != cudaMemoryTypeUnregistered) {
    // direct access is possible
    out_ptr = (outputT*)attr.devicePointer;
  } else {
    throw std::runtime_error("nvcompIntHandle_t::decompGPU(): Workspace memory "
                             "not accessible to GPU.");
  }

  nvcompLayer_t* layer = node->parentLayer;

  if (layer->scheme == NVCOMP_SCHEME_BP) {
    // We assume this is the only layer, and we just do it and exit
    layer->vals->ptr = out_ptr;
    unpackGpu(
        (outputT*)layer->vals->ptr,
        layer->vals,
        inputData[layer->valId],
        h_headers[layer->valId],
        stream);
    layer->vals->length
        = static_cast<const CascadedMetadata::Header*>(h_headers[layer->valId])
              ->length;
    assert(layer->vals->length <= max_input_len);

    return nvcompSuccess;
  }

  // prepare inputs
  std::swap(temp_output, temp_val);
  if (layer->vals->parentLayer != NULL) {
    layer->vals->ptr = localOutput;
    // when recursing, swap
    decompGPU<outputT, runT>(layer->vals, inputData, h_headers, stream);
  } else {
    // unpack RLE values
    layer->vals->ptr = localOutput;
    unpackGpu(
        (outputT*)layer->vals->ptr,
        layer->vals,
        inputData[layer->valId],
        h_headers[layer->valId],
        stream);
    layer->vals->length
        = static_cast<const CascadedMetadata::Header*>(h_headers[layer->valId])
              ->length;
    assert(layer->vals->length <= max_input_len);
  }

  if (layer->runs != nullptr) {
    if (layer->runs->parentLayer != nullptr) {
      throw std::runtime_error("decompGPU(): Runs cannot have parent layers.");
    } else {
      // unpack RLE runs
      layer->runs->ptr = localRun;
      unpackGpu(
          (runT*)layer->runs->ptr,
          layer->runs,
          inputData[layer->runId],
          h_headers[layer->runId],
          stream);
      layer->runs->length = static_cast<const CascadedMetadata::Header*>(
                                h_headers[layer->runId])
                                ->length;
    }
  }

  outputT* d_vals = (outputT*)layer->vals->ptr;
  const size_t input_size = layer->vals->length;
  assert(input_size <= max_input_len);

  if (layer->scheme == NVCOMP_SCHEME_DELTA) {
    assert(out_ptr != d_vals);
    cub::DeviceScan::InclusiveSum(
        temp_scan, temp_scan_bytes, d_vals, out_ptr, input_size, stream);
  } else {
    // must be RLE of some form
    runT* d_runs = (runT*)layer->runs->ptr;

    assert(layer->runs->length == input_size);

    if (layer->scheme == NVCOMP_SCHEME_RLE_DELTA) {
      const dim3 block(512);
      const dim3 grid(roundUpDiv(input_size, block.x));
      vecMultKernel<<<grid, block, 0, stream>>>(
          d_vals, d_runs, localDelta, input_size);

      // inclusive scan to compute Delta sums
      cub::DeviceScan::InclusiveSum(
          temp_scan,
          temp_scan_bytes,
          localDelta,
          localDelta,
          input_size,
          stream);
    }

    // inclusive scan to compute RLE offsets
    // TODO: could be merged with the unpack kernel?
    cub::DeviceScan::InclusiveSum(
        temp_scan, temp_scan_bytes, d_runs, d_runs, input_size, stream);

    const size_t output_length = node->length;

    // precompute start/end boundaries for each CUDA block
    size_t output_grid
        = (output_length + RLE_ELEMS_PER_BLOCK - 1) / RLE_ELEMS_PER_BLOCK;
    size_t output_grid_block
        = (output_grid + RLE_THREAD_BLOCK - 1) / RLE_THREAD_BLOCK;
    searchBlockBoundaries<runT, RLE_THREAD_BLOCK, RLE_ELEMS_PER_THREAD>
        <<<output_grid_block, RLE_THREAD_BLOCK, 0, stream>>>(
            start_ind, start_off, output_grid, input_size, d_runs);

    // expand RLE and apply Delta: buf[r] -> buf[r+1]
    // TODO: implement macro to look nicer?
    switch (layer->scheme) {
    case NVCOMP_SCHEME_RLE_DELTA:
      expandRLEDelta<
          outputT,
          outputT,
          runT,
          RLE_THREAD_BLOCK,
          RLE_ELEMS_PER_THREAD,
          true><<<output_grid, RLE_THREAD_BLOCK, 0, stream>>>(
          (outputT*)out_ptr,
          output_length,
          d_vals,
          d_runs,
          localDelta,
          start_ind,
          start_off);
      break;
    case NVCOMP_SCHEME_RLE:
      expandRLEDelta<
          outputT,
          outputT,
          runT,
          RLE_THREAD_BLOCK,
          RLE_ELEMS_PER_THREAD,
          false><<<output_grid, RLE_THREAD_BLOCK, 0, stream>>>(
          (outputT*)out_ptr,
          output_length,
          d_vals,
          d_runs,
          localDelta,
          start_ind,
          start_off);
      break;
    default:
      throw std::runtime_error(
          "Invalid rle scheme: " + std::to_string(layer->scheme));
    }
  }

  return nvcompSuccess;
}

nvcompError_t
nvcompSetNodeLength(nvcompHandle_t handle, int nodeId, size_t output_length)
{
  nvcompIntHandle_t& h = handles[handle];
  nvcompIntConfig_t& c = *h.config;
  c.nodes[nodeId].length = output_length;
  return nvcompSuccess;
}

// Main function that sets up Cascaded decompression from the old API.
// the new cascaded decompression API call is just a wrapper around this (though
// heavily modified to be asynchronous).
template <typename outputType, typename runType>
nvcompError_t nvcompDecompressLaunch(
    nvcompHandle_t handle,
    void* outputData,
    const size_t outputSize,
    const void** inputData,
    const void** h_headers)
{
  nvcompIntHandle_t& h = handles[handle];

  nvcompIntConfig_t& c = *h.config;

  // TODO: assign all the buffers
  nvcompDataNode_t* terminal_node = &c.nodes[c.outputId];
  terminal_node->ptr = outputData;

  nvcompError_t ret = h.decompGPU<outputType, runType>(
      terminal_node, inputData, h_headers, h.stream);

  const size_t neededBytes = terminal_node->length * sizeof(outputType);
  if (outputSize < neededBytes) {
    std::cerr << "Insufficient space to write decompressed date: given "
              << outputSize << " bytes but need " << neededBytes << " bytes."
              << std::endl;
    return nvcompErrorInvalidValue;
  }

  // this is to enable the correct result for multi-chunk execuation
  for (auto it = c.nodes.begin(); it != c.nodes.end(); it++) {
    it->second.length = 0;
  }

  return ret;
}

nvcompError_t nvcompDecompressLaunch(
    nvcompHandle_t handle,
    const size_t numUncompressedElements,
    void* const outputData,
    const size_t outputSize,
    const void** const inputData,
    const void** const h_headers)
{
  const nvcompType_t outputType = handles[handle].config->outputType;
  const nvcompType_t runType = selectRunsType(numUncompressedElements);

  NVCOMP_TYPE_TWO_SWITCH_RETURN(
      outputType,
      runType,
      nvcompDecompressLaunch,
      handle,
      outputData,
      outputSize,
      inputData,
      h_headers);
}

nvcompError_t nvcompDestroyHandle(nvcompHandle_t handle)
{
  nvcompIntHandle_t& h = handles[handle];
  nvcompIntConfig_t& c = *h.config;

  // free temp memory
  h.release();

  // clear all local nodes attached to this config
  c.nodes.clear();

  // remove the handle from the list
  handles.erase(handle);

  return nvcompSuccess;
}

// Modified version of handle creation function from previous API to now be
// asynchronous Assumes workspaceStorage is already allocated.
nvcompError_t nvcompCreateHandleAsync(
    nvcompHandle_t* handle,
    std::unique_ptr<nvcompIntConfig_t> config,
    void* workspaceStorage,
    const size_t workspaceBytes,
    cudaStream_t stream)
{

  std::lock_guard<std::mutex> guard(handle_mutex);

  nvcompIntConfig_t& c = *config;

  // first - optimize the plan
  c.optimizeLayers();
  // assign pointers - at this point the nodes map is set
  for (auto it = c.layers.begin(); it != c.layers.end(); it++) {
    it->vals = &c.nodes[it->valId];
    it->output = &c.nodes[it->outputId];
    if (it->runId >= 0)
      it->runs = &c.nodes[it->runId];
  }

  if (workspaceBytes < c.getWorkspaceBytes()) {
    std::cerr << "Insufficient workspace size: got " << workspaceBytes
              << " but need " << c.getWorkspaceBytes() << std::endl;
    return nvcompErrorInvalidValue;
  }

  // find the next available id
  nvcompHandle_t id = handles.find_next();
  *handle = id;
  nvcompIntHandle_t& h = handles[id];

  h.config = std::move(config);
  h.stream = stream;

  h.workspaceBytes = workspaceBytes;
  h.workspaceStorage = workspaceStorage;

  h.allocateAsync();

  return nvcompSuccess;
}

} // namespace nvcomp

using namespace nvcomp;

nvcompError_t nvcompCascadedDecompressGetMetadata(
    const void* in_ptr,
    const size_t in_bytes,
    void** metadata_ptr,
    cudaStream_t stream)
{
  try {
    CHECK_NOT_NULL(in_ptr);
    CHECK_NOT_NULL(metadata_ptr);

    CascadedMetadataOnGPU gpuMetadata((void*)in_ptr, in_bytes);
    *metadata_ptr = new CascadedMetadata(gpuMetadata.copyToHost(stream));
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompCascadedDecompressGetMetadata()");
  }

  return nvcompSuccess;
}

void nvcompCascadedDecompressDestroyMetadata(void* const metadata_ptr)
{
  CascadedMetadata* metadata = static_cast<CascadedMetadata*>(metadata_ptr);
  ::operator delete(metadata);
}

// TODO: improve estimate with a more sophistocated approach.
nvcompError_t nvcompCascadedDecompressGetTempSize(
    const void* metadata_ptr, size_t* temp_bytes)
{
  try {
    CHECK_NOT_NULL(metadata_ptr);
    CHECK_NOT_NULL(temp_bytes);

    CascadedMetadata* metadata = (CascadedMetadata*)metadata_ptr;

    std::unique_ptr<nvcompIntConfig_t> c = generateConfig(metadata);

    // first - optimize the plan
    c->optimizeLayers();
    // assign pointers - at this point the nodes map is set
    for (auto& layer : c->layers) {
      layer.vals = &c->nodes[layer.valId];
      layer.output = &c->nodes[layer.outputId];
      if (layer.runId >= 0) {
        layer.runs = &c->nodes[layer.runId];
      }
    }

    // Return the required temp storage size
    *temp_bytes = c->getWorkspaceBytes();
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompCascadedDecompressGetTempSize()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompCascadedDecompressGetOutputSize(
    const void* metadata_ptr, size_t* output_bytes)
{
  try {
    *output_bytes = static_cast<const CascadedMetadata*>(metadata_ptr)
                        ->getUncompressedSize();
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompCascadedDecompressionGetOutputSize()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompCascadedDecompressAsync(
    const void* const in_ptr,
    const size_t in_bytes,
    void* const temp_ptr,
    const size_t temp_bytes,
    const void* const metadata_ptr,
    void* const out_ptr,
    const size_t out_bytes,
    cudaStream_t stream)
{
  nvcompHandle_t handle = -1;
  try {
    CHECK_NOT_NULL(metadata_ptr);

    const CascadedMetadata* const metadata
        = static_cast<const CascadedMetadata*>(metadata_ptr);

    if (in_bytes < metadata->getCompressedSize()) {
      throw NVCompException(
          nvcompErrorInvalidValue,
          "in_bytes is smaller than compressed data size: "
              + std::to_string(in_bytes) + " < "
              + std::to_string(metadata->getCompressedSize()));
    }

    std::unique_ptr<nvcompIntConfig_t> c = generateConfig(metadata);

    // first - optimize the plan
    c->optimizeLayers();
    // assign pointers - at this point the nodes map is set
    for (auto& layer : c->layers) {
      layer.vals = &c->nodes[layer.valId];
      layer.output = &c->nodes[layer.outputId];
      if (layer.runId >= 0) {
        layer.runs = &c->nodes[layer.runId];
      }
    }

    CHECK_API_CALL(
        nvcompCreateHandleAsync(&handle, std::move(c), temp_ptr, temp_bytes, stream));
    assert(handle >= 0);

    // Pointers to different portions of compressed data
    std::vector<void*> inputData(metadata->getNumInputs(), nullptr);

    std::vector<CascadedMetadata::Header> inputHdrs;
    std::vector<CascadedMetadata::Header*> cpuHdrs;
    for (size_t i = 0; i < metadata->getNumInputs(); i++) {
      inputHdrs.emplace_back(metadata->getHeader(i));
      inputData[i] = &((char*)in_ptr)[metadata->getDataOffset(i)];
    }

    for (CascadedMetadata::Header& hdr : inputHdrs) {
      cpuHdrs.emplace_back(&hdr);
    }

    nvcompDecompressLaunch(
        handle,
        metadata->getNumUncompressedElements(),
        out_ptr,
        out_bytes,
        (const void**)inputData.data(),
        (const void**)cpuHdrs.data());
    nvcompDestroyHandle(handle);
  } catch (const std::exception& e) {
    if (handle >= 0) {
      nvcompDestroyHandle(handle);
    }
    return Check::exception_to_error(e, "nvcompCascadedDecompressAsync()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompCascadedCompressGetTempSize(
    const void* const in_ptr,
    const size_t in_bytes,
    const nvcompType_t in_type,
    const nvcompCascadedFormatOpts* const format_opts,
    size_t* const temp_bytes)
{
  try {
    checkCompressSize(in_bytes);

    nvcompCascadedCompressionGPU::computeWorkspaceSize(
        in_ptr, in_bytes, in_type, format_opts, temp_bytes);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompCascadedCompressGetTempSize()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompCascadedCompressGetOutputSize(
    const void* in_ptr,
    const size_t in_bytes,
    const nvcompType_t in_type,
    const nvcompCascadedFormatOpts* format_opts,
    void* const temp_ptr,
    const size_t temp_bytes,
    size_t* const out_bytes,
    const int exact_out_bytes)
{
  try {
    checkCompressSize(in_bytes);

    if (exact_out_bytes) {
      throw std::runtime_error("Exact output bytes is unimplemented at "
                               "this time.");
    }

    nvcompCascadedCompressionGPU::generateOutputUpperBound(
        in_ptr,
        in_bytes,
        in_type,
        format_opts,
        temp_ptr,
        temp_bytes,
        out_bytes);
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "nvcompCascadedCompressGetOutputSize()");
  }

  return nvcompSuccess;
}

nvcompError_t nvcompCascadedCompressAsync(
    const void* const in_ptr,
    const size_t in_bytes,
    const nvcompType_t in_type,
    const nvcompCascadedFormatOpts* const format_opts,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const out_ptr,
    size_t* const out_bytes,
    cudaStream_t stream)
{
  try {
    checkCompressSize(in_bytes);

    CHECK_NOT_NULL(out_bytes);

    if (*out_bytes == 0) {
      throw NVCompException(
          nvcompErrorInvalidValue,
          "Output size cannot be zero. Make sure "
          "to set the size of out_bytes to size of output space allocated "
          "for compressed output.");
    }

    nvcompCascadedCompressionGPU::compressAsync(
        in_ptr,
        in_bytes,
        in_type,
        format_opts,
        temp_ptr,
        temp_bytes,
        out_ptr,
        out_bytes,
        stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompCascadedCompressAsync()");
  }

  return nvcompSuccess;
}


/*****************************************************************************
 * Definitions of API calls for automatically selected compression
 ****************************************************************************/
nvcompError_t nvcompCascadedCompressAutoGetTempSize(
    const void* const in_ptr,
    const size_t in_bytes,
    const nvcompType_t in_type,
    size_t* const temp_bytes) 
{

  // Assume the scheme that requires the most temp space
  nvcompCascadedFormatOpts biggest_opts;
  biggest_opts.num_RLEs = 2;
  biggest_opts.num_deltas = 2;
  biggest_opts.use_bp = 1;

  return API_WRAPPER(nvcompCascadedCompressGetTempSize(
             in_ptr,
             in_bytes,
             in_type,
             &biggest_opts,
             temp_bytes), "nvcompCascadedCompressAutoGetTempSize()");
}

nvcompError_t nvcompCascadedCompressAutoGetOutputSize(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    void* temp_ptr,
    size_t temp_bytes,
    size_t* out_bytes)
{
  // Assume the scheme that can result in the largest output
  nvcompCascadedFormatOpts biggest_opts;
  biggest_opts.num_RLEs = 2;
  biggest_opts.num_deltas = 2;
  biggest_opts.use_bp = 1;

  return API_WRAPPER(nvcompCascadedCompressGetOutputSize(
             in_ptr,
             in_bytes,
             in_type,
             &biggest_opts,
             temp_ptr,
             temp_bytes,
             out_bytes,
             0), "nvcompCascadedCompressAutoGetOutputSize()");
}

nvcompError_t nvcompCascadedCompressAuto(
    const void* in_ptr,
    size_t in_bytes,
    nvcompType_t in_type,
    void* temp_ptr,
    size_t temp_bytes,
    void* out_ptr,
    size_t* out_bytes,
    unsigned seed,
    cudaStream_t stream)
{
  try {
    nvcompCascadedSelectorOpts selector_opts;
    selector_opts.sample_size = 1024;
    selector_opts.num_samples = 100;
    selector_opts.seed = seed;

    size_t type_bytes = sizeOfnvcompType(in_type);

    // Adjust sample size if input is too small
    if(in_bytes < (selector_opts.sample_size * selector_opts.num_samples * type_bytes)) {
      selector_opts.sample_size = in_bytes / (10*type_bytes);
      selector_opts.num_samples = 10;
    }
    
    nvcompCascadedFormatOpts format_opts;
    double est_ratio;

    // Run selector to get format opts for compression
    CHECK_API_CALL(nvcompCascadedSelectorSelectConfig(
             in_ptr,
             in_bytes,
             in_type,
             selector_opts,
             temp_ptr,
             temp_bytes,
             &format_opts,
             &est_ratio,
             stream));
    CudaUtils::sync(stream);

    // Run compression
    CHECK_API_CALL(nvcompCascadedCompressAsync(
             in_ptr,
             in_bytes,
             in_type,
             &format_opts,
             temp_ptr,
             temp_bytes,
             out_ptr,
             out_bytes,
             stream));
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "nvcompCascadedCompressAuto()");
  }

  return nvcompSuccess;
}

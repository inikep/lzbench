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

#ifndef VERBOSE
#define VERBOSE 0
#endif

#include "nvcomp/lz4.hpp"
#include "nvcomp/nvcompManager.hpp"

#include "benchmark_common.h"

#include <algorithm>
#include <iostream>
#include <nvml.h>
#include <string.h>

using namespace nvcomp;

// Max number of streams used by each GPU for compression/decompression
#define MAX_STREAMS 8

static void print_usage()
{
  printf("Usage: benchmark_binary [OPTIONS]\n");
  printf("  %-35s Binary dataset filename (required).\n", "-f, --filename");
  printf("  %-35s Number of GPUs (default 2)\n", "-g, --gpu");
  printf(
      "  %-35s Number of data chunks (default number of GPUs)\n",
      "-h, --chunks");
  printf(
      "  %-35s Compression type (None, Cascaded, LZ4)\n", "-c, --compression");
  printf("  %-35s *If Cascaded* Number lf RLEs (default 1)\n", "-r, --rles");
  printf(
      "  %-35s *If Cascaded* Number of Deltas (default 0)\n", "-d, --deltas");
  printf(
      "  %-35s *If Cascaded* Bitpacking enabled (default 1)\n",
      "-b, --bitpack");
  printf(
      "  %-35s *If Cascaded* Datatype (int or long, default int)\n",
      "-t, --type");
}

// Check that the output of the benchmark matches the input
template <typename T>
static void
check_output(T*** outputs, T* h_data, int gpus, int chunks, size_t* chunk_sizes)
{
  std::vector<T> result_buffer(chunk_sizes[0]);

  for (int i = 0; i < gpus; ++i) {
    size_t idx = 0;
    for (int chunkId = 0; chunkId < chunks; ++chunkId) {
      CUDA_CHECK(cudaMemcpy(
          result_buffer.data(),
          outputs[i][chunkId],
          chunk_sizes[chunkId] * sizeof(T),
          cudaMemcpyDeviceToHost));
      for (size_t j = 0; j < chunk_sizes[chunkId]; ++j) {
        if (result_buffer[j] != h_data[idx + j]) {
          std::cout << "Incorrect result, elemenet number:" << j
                    << " - expected:" << h_data[j]
                    << ", found:" << result_buffer[j] << std::endl;
          exit(1);
        }
      }
      idx += chunk_sizes[chunkId];
    }
  }
}

// Load the input from a binary file, split it unti chunks, and distribute them
// to the GPU to prepare for benchmark
template <typename T>
static void load_chunks_to_devices(
    char* fname,
    int gpus,
    int chunks,
    T** dev_ptrs,
    size_t* data_sizes,
    std::vector<T>* h_data)
{

  size_t input_elts = 0;
  *h_data = load_dataset_from_binary<T>(fname, &input_elts);

  int chunks_per_gpu = chunks / gpus;
  if (chunks_per_gpu * gpus != chunks) {
    std::cout << "chunks does not evenly divide number of GPUs." << std::endl;
    exit(1);
  }

  size_t chunk_size = 1 + ((input_elts - 1) / chunks);

  std::cout << "Loading data - " << input_elts
            << " elements, chunk size:" << chunk_size
            << ", total chunks:" << chunks
            << ", number of gpus being used:" << gpus << std::endl;

  size_t offset;

  for (int i = 0; i < gpus; i++) {
    CUDA_CHECK(cudaSetDevice(i));

    // Make sure dataset fits on GPU to benchmark total compression
    size_t freeMem;
    size_t totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    if (freeMem < chunk_size * chunks_per_gpu * sizeof(T)) {
      std::cout << "Insufficient GPU memory to perform compression."
                << std::endl;
      exit(1);
    }

    for (int j = 0; j < chunks_per_gpu; j++) {
      offset = i * chunks_per_gpu + j;
      data_sizes[offset] = chunk_size;
      if (((size_t)(offset + 1) * chunk_size) > h_data->size()) {
        data_sizes[offset] = h_data->size() - (offset * chunk_size);
      }
      CUDA_CHECK(cudaMalloc(&dev_ptrs[offset], data_sizes[offset] * sizeof(T)));
      CUDA_CHECK(cudaMemcpy(
          dev_ptrs[offset],
          &((h_data->data())[(offset)*chunk_size]),
          data_sizes[offset] * sizeof(T),
          cudaMemcpyHostToDevice));
    }
  }
}

// Perform the all-gather operation using memory copies between GPUs
template <typename T>
void copy_to_all(
    int gpus,
    int chunks,
    T** dev_ptrs,
    size_t* chunk_bytes,
    std::vector<T*>* dest_ptrs,
    const std::vector<std::vector<cudaStream_t>>& streams,
    int STREAMS_PER_GPU)
{
  //  int chunks_per_gpu = chunks/gpus;

  for (int i = 0; i < chunks; ++i) {
    for (int j = 0; j < gpus; ++j) {
      if (i != j) {
        CUDA_CHECK(cudaMemcpyAsync(
            dest_ptrs[j][i],
            dev_ptrs[i],
            chunk_bytes[i],
            cudaMemcpyDeviceToDevice,
            streams[j][i % STREAMS_PER_GPU]));
      }
    }
  }
}

// Create all needed streams one each GPU
static void create_gpu_streams(
    std::vector<std::vector<cudaStream_t>>* streams,
    const int gpus,
    int STREAMS_PER_GPU)
{
  streams->resize(gpus);
  for (int i = 0; i < gpus; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    (*streams)[i].resize(STREAMS_PER_GPU);
    for (int j = 0; j < STREAMS_PER_GPU; ++j) {
      cudaStreamCreateWithFlags(&((*streams)[i][j]), cudaStreamNonBlocking);
    }
  }
}

// Synchronize all streams on all GPUs
static void sync_all_streams(
    std::vector<std::vector<cudaStream_t>>* streams,
    int gpus,
    int STREAMS_PER_GPU)
{
  for (int i = 0; i < gpus; ++i) {
    for (int j = 0; j < STREAMS_PER_GPU; ++j) {
      cudaStreamSynchronize((*streams)[i][j]);
    }
  }
}

// Benchmark the All-gather operation on totally uncompressed data
template <typename T>
static void run_uncompressed_benchmark(
    const int gpus,
    const int chunks,
    T** dev_ptrs,
    size_t* chunk_sizes,
    std::vector<T>* h_data)
{

  const int chunks_per_gpu = chunks / gpus;
  const int STREAMS_PER_GPU = std::min(chunks_per_gpu, MAX_STREAMS);

  std::vector<std::vector<cudaStream_t>> streams;
  create_gpu_streams(&streams, gpus, STREAMS_PER_GPU);

  std::vector<std::vector<T*>> dest_ptrs(gpus);
  for (int i = 0; i < gpus; ++i) { // Allocate full data size on each GPU
    CUDA_CHECK(cudaSetDevice(i));
    dest_ptrs[i].resize(gpus);
    for (int j = 0; j < chunks; ++j) {
      CUDA_CHECK(cudaMalloc(&dest_ptrs[i][j], chunk_sizes[j] * sizeof(T)));
    }
  }

  auto start = std::chrono::steady_clock::now();

  copy_to_all<T>(
      gpus,
      chunks,
      dev_ptrs,
      chunk_sizes,
      dest_ptrs.data(),
      streams,
      STREAMS_PER_GPU);
  for (int gpu = 0; gpu < gpus; ++gpu) {
    CUDA_CHECK(cudaMemcpyAsync(
        dest_ptrs[gpu][gpu],
        dev_ptrs[gpu],
        chunk_sizes[gpu] * sizeof(T),
        cudaMemcpyDeviceToDevice,
        streams[gpu][gpu % STREAMS_PER_GPU]));
  }

  sync_all_streams(&streams, gpus, STREAMS_PER_GPU);

  auto end = std::chrono::steady_clock::now();
  std::cout << "Full data size (B): " << h_data->size() * sizeof(T) << std::endl
            << "Per-GPU benchmark throughput (GB/s): "
            << gbs(start,
                   end,
                   h_data->size() * (((double)gpus - 1.0) / (double)gpus)
                       * sizeof(T))
            << std::endl;
  std::cout << "Total data transferred across system (B): "
            << h_data->size() * (gpus - 1) * sizeof(T) << std::endl
            << "Total system throughput (GB/s): "
            << gbs(start, end, h_data->size() * (gpus - 1) * sizeof(T))
            << std::endl;

  // Test for correctness
  std::vector<T**> outputs(gpus);
  for (int i = 0; i < gpus; ++i) {
    outputs[i] = dest_ptrs[i].data();
  }
  check_output<T>(outputs.data(), h_data->data(), gpus, chunks, chunk_sizes);

  for (int i = 0; i < gpus; ++i) { // Allocate full data size on each GPU
    CUDA_CHECK(cudaSetDevice(i));
    for (int j = 0; j < chunks; ++j) {
      CUDA_CHECK(cudaFree(dest_ptrs[i][j]));
    }
  }
}

template <typename T>
static void run_nvcomp_benchmark(
    int gpus,
    int chunks,
    T** dev_ptrs,
    size_t* chunk_sizes,
    std::vector<T>* h_data,
    nvcompManagerBase** managers,
    std::vector<std::vector<cudaStream_t>>& streams)
{
  const int chunks_per_gpu = chunks / gpus;
  const int STREAMS_PER_GPU = std::min(chunks_per_gpu, MAX_STREAMS);

  std::vector<size_t> chunk_bytes(chunks, 0);
  for (int chunkIdx = 0; chunkIdx < chunks; ++chunkIdx) {
    chunk_bytes[chunkIdx] = chunk_sizes[chunkIdx] * sizeof(T);
  }

  // Create a compression result for each chunk
  std::vector<uint8_t*> d_comp_out(chunks);
  size_t* comp_out_bytes;
  CUDA_CHECK(cudaMallocHost(&comp_out_bytes, chunks * sizeof(size_t)));

  std::vector<CompressionConfig> comp_configs;
  comp_configs.reserve(chunks * gpus);
  
  // Allocate all memory buffers necessary for compression of each chunk
  for (int gpu = 0; gpu < gpus; ++gpu) {
    CUDA_CHECK(cudaSetDevice(gpu));

    // Create compressor for the chunk and allocate necessary memory
    for (int chunkIdx = 0; chunkIdx < chunks_per_gpu; ++chunkIdx) {
      const int idx = gpu * chunks_per_gpu + chunkIdx;

      auto comp_config = managers[idx]->configure_compression(chunk_bytes[idx]);

      // Allocate output buffers for each chunk on the GPU
      comp_out_bytes[idx] = comp_config.max_compressed_buffer_size;
      CUDA_CHECK(cudaMalloc(&d_comp_out[idx], comp_out_bytes[idx]));

      comp_configs.push_back(std::move(comp_config));
    }
  }

  std::vector<std::vector<T*>> dest_ptrs(gpus);
  for (int gpu = 0; gpu < gpus; ++gpu) { // Allocate full data size on each GPU
    CUDA_CHECK(cudaSetDevice(gpu));
    dest_ptrs[gpu].resize(chunks);
    for (int chunkIdx = 0; chunkIdx < chunks; ++chunkIdx) {
      CUDA_CHECK(
          cudaMalloc(&dest_ptrs[gpu][chunkIdx], comp_out_bytes[chunkIdx]));
    }
  }

  // Allocate all memory buffers for decompression
  std::vector<size_t> decomp_out_bytes;
  decomp_out_bytes.resize(chunks * gpus);

  // output buffers for each chunk on each gpu
  std::vector<T**> d_decomp_out;
  for (int gpu = 0; gpu < gpus; ++gpu) {
    d_decomp_out.push_back(new T*[chunks]);
    CUDA_CHECK(cudaSetDevice(gpu));
    for (int chunkId = 0; chunkId < chunks; ++chunkId) {
      CUDA_CHECK(cudaMalloc(&d_decomp_out[gpu][chunkId], chunk_bytes[chunkId]));
    }
  }

  auto start = std::chrono::steady_clock::now();

  // Issue compression calls
  for (int gpu = 0; gpu < gpus; ++gpu) {
    CUDA_CHECK(cudaSetDevice(gpu));
    for (int chunkIdx = 0; chunkIdx < chunks_per_gpu; ++chunkIdx) {
      const int idx = gpu * chunks_per_gpu + chunkIdx;
      managers[idx]->compress(
          dev_ptrs[idx],
          d_comp_out[idx],
          comp_configs[idx]);
    }
  }

  sync_all_streams(&streams, gpus, STREAMS_PER_GPU);

  size_t total_comp_bytes = 0;
  for (int i = 0; i < gpus * chunks_per_gpu; ++i) {
    total_comp_bytes += comp_out_bytes[i];
  }

  //  Copy compressed data to all GPUs
  copy_to_all<T>(
      gpus,
      chunks,
      (T**)(d_comp_out.data()),
      comp_out_bytes,
      dest_ptrs.data(),
      streams,
      STREAMS_PER_GPU);
  for (int gpu = 0; gpu < gpus; ++gpu) {
    CUDA_CHECK(cudaMemcpyAsync(
        d_decomp_out[gpu][gpu],
        dev_ptrs[gpu],
        chunk_bytes[gpu],
        cudaMemcpyDeviceToDevice,
        streams[gpu][gpu % STREAMS_PER_GPU]));
  }

  // Create decompressors for each chunk on each gpu
  std::vector<DecompressionConfig> decomp_configs;
  decomp_configs.reserve(chunks * gpus);

  for (int gpu = 0; gpu < gpus; ++gpu) {
    CUDA_CHECK(cudaSetDevice(gpu));
    for (int chunkIdx = 0; chunkIdx < chunks; ++chunkIdx) {
      const int idx = gpu * chunks + chunkIdx;
      auto decomp_config = managers[idx]->configure_decompression(dest_ptrs[gpu][chunkIdx]);
      decomp_configs.push_back(decomp_config);

      if (gpu != chunkIdx) {
        decomp_out_bytes[idx] = decomp_config.decomp_data_size;
      } else {
        decomp_out_bytes[idx] = chunk_bytes[chunkIdx];
      }
    }
  }

  // Issue decompression calls
  for (int gpu = 0; gpu < gpus; ++gpu) {
    CUDA_CHECK(cudaSetDevice(gpu));
    for (int chunkIdx = 0; chunkIdx < chunks; ++chunkIdx) {
      if (gpu != chunkIdx) {
        const int idx = gpu * chunks + chunkIdx;
        managers[idx]->decompress(
            d_decomp_out[gpu][chunkIdx],
            dest_ptrs[gpu][chunkIdx],
            decomp_configs[idx]);
      }
    }
  }

  sync_all_streams(&streams, gpus, STREAMS_PER_GPU);

  auto end = std::chrono::steady_clock::now();

  // Test for correctness
  check_output<T>(
      d_decomp_out.data(), h_data->data(), gpus, chunks, chunk_sizes);

  // Cleanup
  for (int i = 0; i < gpus; ++i) {
    for (int j = 0; j < chunks; ++j) {
      CUDA_CHECK(cudaFree(d_decomp_out[i][j]));
      CUDA_CHECK(cudaFree(dest_ptrs[i][j]));
    }

    for (int j = 0; j < chunks_per_gpu; ++j) {
      const int idx = i * chunks_per_gpu + j;
      CUDA_CHECK(cudaFree(d_comp_out[idx]));
    }
  }
  CUDA_CHECK(cudaFreeHost(comp_out_bytes));

  for (int gpu = 0; gpu < gpus; ++gpu) {
    delete[] d_decomp_out[gpu];
  }

  std::cout << "Full data size (B): " << h_data->size() * sizeof(T) << std::endl
            << "Per-GPU benchmark throughput (GB/s): "
            << gbs(start,
                   end,
                   h_data->size() * (((double)gpus - 1.0) / (double)gpus)
                       * sizeof(T))
            << std::endl;
  std::cout << "Compressed data size (B): " << total_comp_bytes
            << ", compression ratio: "
            << (double)h_data->size() * sizeof(T) / (double)total_comp_bytes
            << std::endl;
  std::cout << "Total data distributed across system (B): "
            << h_data->size() * (gpus - 1) * sizeof(T) << std::endl
            << "Total system throughput (GB/s): "
            << gbs(start, end, h_data->size() * (gpus - 1) * sizeof(T))
            << std::endl;
}

// Benchmark the performance of the All-gather operation using LZ4
// compression/decompression to reduce data transfers
static void run_lz4_benchmark(
    const int gpus,
    const int chunks,
    uint8_t** dev_ptrs,
    size_t* chunk_sizes,
    std::vector<uint8_t>* h_data)
{
  using T = uint8_t;

  const int chunks_per_gpu = chunks / gpus;
  const int STREAMS_PER_GPU = std::min(chunks_per_gpu, MAX_STREAMS);

  std::vector<std::vector<cudaStream_t>> streams;
  create_gpu_streams(&streams, gpus, STREAMS_PER_GPU);

  nvcompManagerBase** managers = new nvcompManagerBase*[gpus * chunks_per_gpu];
  // Allocate all memory buffers necessary for compression of each chunk
  for (int gpu = 0; gpu < gpus; ++gpu) {
    // Create compressor each chunk
    for (int chunkIdx = 0; chunkIdx < chunks_per_gpu; ++chunkIdx) {
      const int idx = gpu * chunks_per_gpu + chunkIdx;
      managers[idx] = new LZ4Manager{1 << 16, NVCOMP_TYPE_CHAR, streams[gpu][chunkIdx]};
    }
  }

  run_nvcomp_benchmark<T>(
      gpus, chunks, dev_ptrs, chunk_sizes, h_data, managers, streams);

  for (int gpu = 0; gpu < gpus; ++gpu) {
    for (int chunkIdx = 0; chunkIdx < chunks_per_gpu; ++chunkIdx) {
      const int idx = gpu * chunks_per_gpu + chunkIdx;
      delete managers[idx];
    }
  }
  delete[] managers;
}

// Benchmark the performance of the All-gather operation using LZ4
// compression/decompression to reduce data transfers
// template <typename T>
// static void run_cascaded_benchmark(
//     int gpus,
//     int chunks,
//     T** dev_ptrs,
//     size_t* chunk_sizes,
//     std::vector<T>* h_data,
//     int RLEs,
//     int deltas,
//     int bitPacking)
// {
//   const int chunks_per_gpu = chunks / gpus;

//   Compressor** compressors = new Compressor*[gpus];
//   for (int gpu = 0; gpu < gpus; ++gpu) {
//     for (int chunkIdx = 0; chunkIdx < chunks_per_gpu; ++chunkIdx) {
//       const int idx = gpu * chunks_per_gpu + chunkIdx;
//       compressors[idx] = new CascadedCompressor(
//           nvcomp::TypeOf<T>(), RLEs, deltas, bitPacking);
//     }
//   }

//   // Create decompressors for each chunk on each gpu
//   Decompressor** decompressors = new Decompressor*[chunks * gpus];
//   for (int gpu = 0; gpu < gpus; ++gpu) {
//     for (int chunkIdx = 0; chunkIdx < chunks; ++chunkIdx) {
//       const int idx = gpu * chunks + chunkIdx;
//       if (gpu != chunkIdx) {
//         // Create compressor for the chunk and allocate necessary memory
//         decompressors[idx] = new CascadedDecompressor;
//       }
//     }
//   }

//   run_nvcomp_benchmark<T>(
//       gpus, chunks, dev_ptrs, chunk_sizes, h_data, compressors, decompressors);

//   for (int gpu = 0; gpu < gpus; ++gpu) {
//     for (int chunkIdx = 0; chunkIdx < chunks_per_gpu; ++chunkIdx) {
//       const int idx = gpu * chunks_per_gpu + chunkIdx;
//       delete compressors[idx];
//     }
//     for (int chunkIdx = 0; chunkIdx < chunks; ++chunkIdx) {
//       const int idx = gpu * chunks + chunkIdx;
//       if (gpu != chunkIdx) {
//         delete decompressors[idx];
//       }
//     }
//   }
//   delete[] compressors;
//   delete[] decompressors;
// }

static void enable_nvlink(int gpus)
{
  for (int i = 0; i < gpus; ++i) {
    for (int j = 0; j < gpus; ++j) {
      int can_access_A = 0;
      cudaDeviceCanAccessPeer(&can_access_A, i, j);
      if (can_access_A) {
        cudaSetDevice(i);
        cudaDeviceEnablePeerAccess(j, 0);
      }
    }
  }
}

int main(int argc, char* argv[])
{
  char* fname = NULL;
  int gpu_num = 2;
  int chunks = 0;
  std::string comp_type = "none";
  int RLEs = 1;
  int deltas = 0;
  int bitPacking = 1;
  std::string dtype = "int";

  // Parse command-line arguments

  char** argv_end = argv + argc;
  argv += 1;
  while (argv != argv_end) {
    char* arg = *argv++;
    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-?") == 0) {
      print_usage();
      return 1;
    }

    // all arguments below require at least a second value in argv
    if (argv >= argv_end) {
      print_usage();
      return 1;
    }

    char* optarg = *argv++;
    if (strcmp(arg, "--filename") == 0 || strcmp(arg, "-f") == 0) {
      fname = optarg;
      continue;
    }
    if (strcmp(arg, "--gpu") == 0 || strcmp(arg, "-g") == 0) {
      gpu_num = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--chunks") == 0 || strcmp(arg, "-h") == 0) {
      chunks = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--comp") == 0 || strcmp(arg, "-c") == 0) {
      comp_type = optarg;
      continue;
    }
    if (strcmp(arg, "--rles") == 0 || strcmp(arg, "-r") == 0) {
      RLEs = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--deltas") == 0 || strcmp(arg, "-d") == 0) {
      deltas = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--bitpack") == 0 || strcmp(arg, "-b") == 0) {
      bitPacking = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--type") == 0 || strcmp(arg, "-t") == 0) {
      dtype = optarg;
      continue;
    }
    print_usage();
    return 1;
  }

  printf("For cascaded, RLEs %d deltas %d bitpack %d\n", RLEs, deltas, bitPacking);

  // TODO check if all options have been found
  if (fname == NULL) {
    std::cerr << "Missing filename." << std::endl;
    print_usage();
    return 1;
  }

  if (gpu_num < 2) {
    std::cerr << "The number of GPUs specified by `--gpu` must be at least 2."
              << std::endl;
    return 1;
  }

  if (chunks == 0)
    chunks = gpu_num;

  if (comp_type == "lz4" || comp_type == "LZ4" || comp_type == "none"
      || comp_type == "None") {
    dtype = "uint8"; // LZ4 only works on byte-level
  }

  enable_nvlink(gpu_num);

  int rv = 0;
  if (dtype == "int") {
    // std::vector<int32_t*> data_ptrs(chunks);
    // std::vector<size_t> data_sizes(chunks);
    // std::vector<int32_t> h_data;
    // load_chunks_to_devices<int32_t>(
    //     fname,
    //     gpu_num,
    //     chunks,
    //     data_ptrs.data(),
    //     data_sizes.data(),
    //     &h_data);
    // run_cascaded_benchmark<int32_t>(
    //     gpu_num,
    //     chunks,
    //     data_ptrs.data(),
    //     data_sizes.data(),
    //     &h_data,
    //     RLEs,
    //     deltas,
    //     bitPacking);
    // for (int chunk = 0; chunk < chunks; ++chunk) {
    //   CUDA_CHECK(cudaFree(data_ptrs[chunk]));
    // }
  } else if (dtype == "long") {
    // std::vector<int64_t*> data_ptrs(chunks);
    // std::vector<size_t> data_sizes(chunks);
    // std::vector<int64_t> h_data;
    // load_chunks_to_devices<int64_t>(
    //     fname,
    //     gpu_num,
    //     chunks,
    //     data_ptrs.data(),
    //     data_sizes.data(),
    //     &h_data);
    // run_cascaded_benchmark<int64_t>(
    //     gpu_num,
    //     chunks,
    //     data_ptrs.data(),
    //     data_sizes.data(),
    //     &h_data,
    //     RLEs,
    //     deltas,
    //     bitPacking);
    // for (int chunk = 0; chunk < chunks; ++chunk) {
    //   CUDA_CHECK(cudaFree(data_ptrs[chunk]));
    // }
  } else if (dtype == "int8") {
    // std::vector<int8_t*> data_ptrs(chunks);
    // std::vector<size_t> data_sizes(chunks);
    // std::vector<int8_t> h_data;
    // load_chunks_to_devices<int8_t>(
    //     fname,
    //     gpu_num,
    //     chunks,
    //     data_ptrs.data(),
    //     data_sizes.data(),
    //     &h_data);
    // run_cascaded_benchmark<int8_t>(
    //     gpu_num,
    //     chunks,
    //     data_ptrs.data(),
    //     data_sizes.data(),
    //     &h_data,
    //     RLEs,
    //     deltas,
    //     bitPacking);
    // for (int chunk = 0; chunk < chunks; ++chunk) {
    //   CUDA_CHECK(cudaFree(data_ptrs[chunk]));
    // }
  } else if (dtype == "byte" || dtype == "uint8") {
    std::vector<uint8_t*> data_ptrs(chunks);
    std::vector<size_t> data_sizes(chunks);
    std::vector<uint8_t> h_data;
    load_chunks_to_devices<uint8_t>(
        fname,
        gpu_num,
        chunks,
        data_ptrs.data(),
        data_sizes.data(),
        &h_data);

    if (comp_type == "lz4" || comp_type == "LZ4") {
      // Run LZ4 benchmark
      run_lz4_benchmark(
          gpu_num, chunks, data_ptrs.data(), data_sizes.data(), &h_data);
    } else if (comp_type == "none" || comp_type == "None") {
      // Run no-comp benchmark
      run_uncompressed_benchmark<uint8_t>(
          gpu_num, chunks, data_ptrs.data(), data_sizes.data(), &h_data);
    } else {
      std::cerr << "Invalid compression benchmark selected." << std::endl;
      print_usage();
      rv = 1;
    }

    for (int chunk = 0; chunk < chunks; ++chunk) {
      CUDA_CHECK(cudaFree(data_ptrs[chunk]));
    }
  } else {
    std::cerr << "Invalid datatype selected." << std::endl;
    print_usage();
    rv = 1;
  }

  return rv;
}

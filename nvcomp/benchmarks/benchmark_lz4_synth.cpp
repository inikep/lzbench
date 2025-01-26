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

#include "benchmark_hlif.hpp"

#include <string.h>
#include <string>
#include <vector>

using namespace nvcomp;

namespace
{

constexpr const size_t CHUNK_SIZE = 1 << 16;

void print_usage()
{
  printf("Usage: benchmark_lz4_synth [OPTIONS]\n");
  printf("  %-35s GPU device number (default 0)\n", "-g, --gpu");
  exit(1);
}

void run_tests(std::mt19937& rng)
{
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  LZ4Manager batch_manager{CHUNK_SIZE, data_type, stream};

  // test all zeros
  for (size_t b = 0; b < 14; ++b) {
    run_benchmark(gen_data(0, CHUNK_SIZE << b, rng), batch_manager, false, stream);
  }

  // test random bytes
  for (size_t b = 0; b < 14; ++b) {
    run_benchmark(gen_data(255, CHUNK_SIZE << b, rng), batch_manager, false, stream);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace

int main(int argc, char* argv[])
{
  int gpu_num = 0;

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
    if (strcmp(arg, "--gpu") == 0 || strcmp(arg, "-g") == 0) {
      gpu_num = atoi(optarg);
      continue;
    }
    print_usage();
    return 1;
  }
  CUDA_CHECK(cudaSetDevice(gpu_num));

  std::mt19937 rng(0);

  run_tests(rng);

  return 0;
}

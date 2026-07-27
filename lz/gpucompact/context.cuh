/*
 * GPUCompact
 * Copyright (C) 2026 UDPSendToFailed
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

struct LaunchConfig {
  int macro_mb = 4;
  int mini_size = 1024;
  int L = 2048;
  int threads_comp = 32;
  int threads_decomp = 128;
};

class CompressionContext {
public:
  int macro_size;
  int mini_size;
  int L;
  int max_chunks;
  int max_words;

  cudaStream_t stream = nullptr;

  unsigned char *host_in = nullptr;
  unsigned char *host_out = nullptr;
  int *host_last_rank = nullptr;

  uint8_t *d_data = nullptr;
  int *d_rank = nullptr;
  int *d_sa = nullptr;
  int *d_sa_alt = nullptr;
  uint64_t *d_keys = nullptr;
  uint64_t *d_keys_alt = nullptr;
  int *d_diff = nullptr;
  int *d_unique_ranks = nullptr;
  uint8_t *d_bwt = nullptr;
  int *d_primary_idx = nullptr;

  uint16_t *d_out_symbols = nullptr;
  uint32_t *d_chunk_sym_lens = nullptr;
  uint32_t *d_hist = nullptr;
  int *d_p = nullptr;
  int *d_prefix_p = nullptr;
  int *d_max_x = nullptr;
  int *d_symbol_spread = nullptr;
  int *d_enc_table = nullptr;
  int *d_dec_table = nullptr;
  int *d_dec_symbol = nullptr;
  int *d_next_state = nullptr;
  uint64_t *d_out_words = nullptr;
  uint32_t *d_chunk_bit_lens = nullptr;
  uint32_t *d_chunk_word_lens = nullptr;
  uint32_t *d_word_offsets = nullptr;
  uint64_t *d_dense_words = nullptr;

  uint8_t *d_payload = nullptr;
  uint64_t *d_gpu_hash = nullptr;

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cudaEvent_t e_start = nullptr, e_end = nullptr;

  int bytes_read = 0;
  int is_raw = 0;
  int num_chunks = 0;
  int total_words = 0;
  int primary_idx = 0;
  uint64_t gpu_hash = 0;
  int comp_size = 0;

  CompressionContext(int macro_bytes, int mini_bytes, int state_L);
  ~CompressionContext();

  void compress_chunk(int threads_comp, int mini_chunk_size);
};

class DecompressionContext {
public:
  int macro_size;
  int mini_size;
  int L;
  int max_chunks;
  int max_words;

  cudaStream_t stream = nullptr;
  cudaEvent_t e_start = nullptr, e_end = nullptr;

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;

  unsigned char *host_in = nullptr;
  unsigned char *host_out = nullptr;
  uint64_t *host_calc_hash = nullptr;

  // Pinned Host Buffers for Async Copy Safety
  int *host_uncomp_size = nullptr;
  int *host_primary_idx = nullptr;

  uint8_t *d_data = nullptr;
  uint8_t *d_bwt = nullptr;
  uint64_t *d_keys = nullptr;
  uint64_t *d_keys_alt = nullptr;
  int *d_primary_idx = nullptr;

  int *d_global_LF = nullptr;
  int *d_J_in = nullptr;
  int *d_D_in = nullptr;
  int *d_J_out = nullptr;
  int *d_D_out = nullptr;

  uint32_t *d_chunk_bit_lengths = nullptr;
  uint32_t *d_chunk_word_lens = nullptr;
  int *d_decoding_table = nullptr;
  int *d_decoding_symbol = nullptr;
  uint32_t *d_word_offsets = nullptr;
  uint64_t *d_dense_words = nullptr;

  uint64_t *d_offsets = nullptr;
  int *d_sizes = nullptr;

  uint8_t *d_payload = nullptr;
  uint64_t *d_gpu_hash = nullptr;

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  int comp_size = 0;
  int uncomp_size = 0;
  int is_raw = 0;
  int num_chunks = 0;
  int total_words = 0;
  int primary_idx = 0;
  uint64_t gpu_hash = 0;

  DecompressionContext(int macro_bytes, int mini_bytes, int state_L);
  ~DecompressionContext();

  bool decompress_chunk(int threads_decomp, int mini_chunk_size);
};
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
#include "context.cuh"
#include "kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cub/cub.cuh>
#include <iostream>
#include <stdexcept>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

CompressionContext::CompressionContext(int macro_bytes, int mini_bytes,
                                       int state_L) {
  macro_size = macro_bytes;
  mini_size = mini_bytes;
  L = state_L;

  max_chunks = (macro_size + mini_size - 1) / mini_size;
  max_words = (int)(mini_size * 1.5 / 8) + 80;

  size_t payload_alloc_size =
      macro_size + (size_t)max_chunks * max_words * sizeof(uint64_t) + 65536;

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  CUDA_CHECK(cudaMallocHost(&host_in, macro_size));
  CUDA_CHECK(cudaMallocHost(&host_out, payload_alloc_size));
  CUDA_CHECK(cudaMallocHost(&host_last_rank, sizeof(int)));

  CUDA_CHECK(cudaMalloc(&d_data, macro_size));
  CUDA_CHECK(cudaMalloc(&d_rank, macro_size * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_sa, macro_size * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_sa_alt, macro_size * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_keys, macro_size * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_keys_alt, macro_size * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_diff, macro_size * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_unique_ranks, macro_size * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_bwt, macro_size));
  CUDA_CHECK(cudaMalloc(&d_primary_idx, sizeof(int)));

  CUDA_CHECK(
      cudaMalloc(&d_out_symbols, (macro_size + 65536) * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&d_chunk_sym_lens, max_chunks * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_hist, 257 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_p, 257 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_prefix_p, 257 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_max_x, 257 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_symbol_spread, L * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_enc_table, L * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_dec_table, L * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_dec_symbol, L * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_next_state, 257 * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc(&d_out_words, max_chunks * max_words * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_chunk_bit_lens, max_chunks * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_chunk_word_lens, max_chunks * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_word_offsets, (max_chunks + 1) * sizeof(uint32_t)));
  CUDA_CHECK(
      cudaMalloc(&d_dense_words, max_chunks * max_words * sizeof(uint64_t)));

  CUDA_CHECK(cudaMalloc(&d_payload, payload_alloc_size));
  CUDA_CHECK(cudaMalloc(&d_gpu_hash, sizeof(uint64_t)));

  size_t sort_bytes = 0, scan_bytes = 0;
  cub::DoubleBuffer<uint64_t> d_keys_db(d_keys, d_keys_alt);
  cub::DoubleBuffer<int> d_sa_db(d_sa, d_sa_alt);
  cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, d_keys_db, d_sa_db,
                                  macro_size, 0, 64, stream);
  cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, d_diff, d_unique_ranks,
                                macro_size, stream);

  temp_storage_bytes = std::max(sort_bytes, scan_bytes) + 65536;
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  CUDA_CHECK(cudaEventCreate(&e_start));
  CUDA_CHECK(cudaEventCreate(&e_end));
}

CompressionContext::~CompressionContext() {
  if (stream)
    cudaStreamSynchronize(stream);

  cudaFreeHost(host_in);
  cudaFreeHost(host_out);
  cudaFreeHost(host_last_rank);

  cudaFree(d_data);
  cudaFree(d_rank);
  cudaFree(d_sa);
  cudaFree(d_sa_alt);
  cudaFree(d_keys);
  cudaFree(d_keys_alt);
  cudaFree(d_diff);
  cudaFree(d_unique_ranks);
  cudaFree(d_bwt);
  cudaFree(d_primary_idx);
  cudaFree(d_out_symbols);
  cudaFree(d_chunk_sym_lens);
  cudaFree(d_hist);
  cudaFree(d_p);
  cudaFree(d_prefix_p);
  cudaFree(d_max_x);
  cudaFree(d_symbol_spread);
  cudaFree(d_enc_table);
  cudaFree(d_dec_table);
  cudaFree(d_dec_symbol);
  cudaFree(d_next_state);
  cudaFree(d_out_words);
  cudaFree(d_chunk_bit_lens);
  cudaFree(d_chunk_word_lens);
  cudaFree(d_word_offsets);
  cudaFree(d_dense_words);
  cudaFree(d_payload);
  cudaFree(d_gpu_hash);
  cudaFree(d_temp_storage);

  cudaEventDestroy(e_start);
  cudaEventDestroy(e_end);
  if (stream)
    cudaStreamDestroy(stream);
}

void CompressionContext::compress_chunk(int threads_comp, int mini_chunk_size) {
  int n = bytes_read;
  if (n <= 1) {
    is_raw = 1;
    comp_size = n;
    std::memcpy(host_out, host_in, n);
    CUDA_CHECK(cudaMemsetAsync(d_gpu_hash, 0, sizeof(uint64_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_data, host_in, std::max(1, n),
                               cudaMemcpyHostToDevice, stream));
    gpu_hash_kernel<<<1, 256, 0, stream>>>(d_data, d_gpu_hash, std::max(1, n));
    CUDA_CHECK(cudaMemcpyAsync(&gpu_hash, d_gpu_hash, sizeof(uint64_t),
                               cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    return;
  }

  bool all_equal = true;
  uint8_t first_byte = host_in[0];
  for (int i = 1; i < n; i++) {
    if (host_in[i] != first_byte) {
      all_equal = false;
      break;
    }
  }

  if (all_equal) {
    is_raw = 2;
    comp_size = 1;
    host_out[0] = host_in[0];
    CUDA_CHECK(cudaMemsetAsync(d_gpu_hash, 0, sizeof(uint64_t), stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_data, host_in, 1, cudaMemcpyHostToDevice, stream));
    gpu_hash_kernel<<<1, 256, 0, stream>>>(d_data, d_gpu_hash, 1);
    CUDA_CHECK(cudaMemcpyAsync(&gpu_hash, d_gpu_hash, sizeof(uint64_t),
                               cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    return;
  }

  CUDA_CHECK(cudaEventRecord(e_start, stream));
  CUDA_CHECK(
      cudaMemcpyAsync(d_data, host_in, n, cudaMemcpyHostToDevice, stream));

  int blocks = (n + 255) / 256;
  cast_uint8_to_int32_kernel<<<blocks, 256, 0, stream>>>(d_data, d_rank, n);

  int k = 1;
  int max_iter = (int)std::ceil(std::log2(n)) + 2;

  cub::DoubleBuffer<uint64_t> d_keys_db(d_keys, d_keys_alt);
  cub::DoubleBuffer<int> d_sa_db(d_sa, d_sa_alt);

  // ASYNC SA DOUBLING WITH PINNED MEMORY EARLY EXIT CHECK
  for (int iter = 0; iter < max_iter; iter++) {
    sa_key_kernel<<<blocks, 256, 0, stream>>>(d_rank, d_sa_db.Current(),
                                              d_keys_db.Current(), n, k);

    size_t t_bytes = temp_storage_bytes;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, t_bytes, d_keys_db, d_sa_db,
                                    n, 0, 64, stream);

    sa_diff_kernel<<<blocks, 256, 0, stream>>>(d_keys_db.Current(), d_diff, n);

    t_bytes = temp_storage_bytes;
    cub::DeviceScan::InclusiveSum(d_temp_storage, t_bytes, d_diff,
                                  d_unique_ranks, n, stream);

    sa_rank_kernel<<<blocks, 256, 0, stream>>>(d_unique_ranks,
                                               d_sa_db.Current(), d_rank, n);

    // Asynchronous transfer to pinned memory + stream synchronization
    CUDA_CHECK(cudaMemcpyAsync(host_last_rank, d_unique_ranks + n - 1,
                               sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (*host_last_rank == n) {
      break; // Early exit when Suffix Array is 100% sorted
    }

    k *= 2;
  }

  extract_bwt_kernel<<<blocks, 256, 0, stream>>>(d_sa_db.Current(), d_data,
                                                 d_bwt, d_primary_idx, n);
  CUDA_CHECK(cudaMemcpyAsync(&primary_idx, d_primary_idx, sizeof(int),
                             cudaMemcpyDeviceToHost, stream));

  num_chunks = (n + mini_chunk_size - 1) / mini_chunk_size;
  int z_blocks = (num_chunks + threads_comp - 1) / threads_comp;

  CUDA_CHECK(cudaMemsetAsync(d_hist, 0, 257 * sizeof(uint32_t), stream));

  zrle_encode_kernel<<<z_blocks, threads_comp, threads_comp * 260, stream>>>(
      d_bwt, d_out_symbols, d_chunk_sym_lens, n, mini_chunk_size, threads_comp);
  compute_histogram_kernel<<<z_blocks, threads_comp, 0, stream>>>(
      d_out_symbols, d_chunk_sym_lens, d_hist, num_chunks, mini_chunk_size);
  build_tans_all_kernel<<<1, 257, 0, stream>>>(
      d_hist, d_p, d_prefix_p, d_max_x, d_symbol_spread, d_enc_table,
      d_dec_table, d_dec_symbol, d_next_state, L, 257);
  tabled_encode_kernel<<<z_blocks, threads_comp, (L + 514) * sizeof(int),
                         stream>>>(d_out_symbols, d_chunk_sym_lens, d_out_words,
                                   d_chunk_bit_lens, d_p, d_prefix_p, d_max_x,
                                   d_enc_table, num_chunks, mini_chunk_size,
                                   max_words, L);

  int c_blocks = (num_chunks + 255) / 256;
  bit_to_word_kernel<<<c_blocks, 256, 0, stream>>>(
      d_chunk_bit_lens, d_chunk_word_lens, num_chunks);

  CUDA_CHECK(cudaMemsetAsync(d_word_offsets, 0, sizeof(uint32_t), stream));
  size_t t_bytes = temp_storage_bytes;
  cub::DeviceScan::InclusiveSum(d_temp_storage, t_bytes, d_chunk_word_lens,
                                d_word_offsets + 1, num_chunks, stream);

  uint32_t total_w = 0;
  CUDA_CHECK(cudaMemcpyAsync(&total_w, d_word_offsets + num_chunks,
                             sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  total_words = (int)total_w;

  uint32_t b_len_bits = num_chunks * 4;
  uint32_t b_len_dec = L * 4;
  uint32_t b_len_words = total_words * 8;
  uint32_t header_total = 4 + b_len_bits + (b_len_dec * 2) + b_len_words;

  if (header_total >= (uint32_t)n) {
    is_raw = 1;
    comp_size = n;
    std::memcpy(host_out, host_in, n);
    CUDA_CHECK(cudaMemsetAsync(d_gpu_hash, 0, sizeof(uint64_t), stream));
    int hash_threads = 256;
    int hash_blocks = ((n + 7) / 8 + hash_threads - 1) / hash_threads;
    gpu_hash_kernel<<<hash_blocks, hash_threads, 0, stream>>>(d_data,
                                                              d_gpu_hash, n);
    CUDA_CHECK(cudaMemcpyAsync(&gpu_hash, d_gpu_hash, sizeof(uint64_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(e_end, stream));
    cudaStreamSynchronize(stream);
    return;
  }

  if (num_chunks > 0) {
    dense_pack_kernel<<<z_blocks, threads_comp, 0, stream>>>(
        d_out_words, d_word_offsets, d_chunk_word_lens, d_dense_words,
        num_chunks, max_words);
  }

  is_raw = 0;

  uint32_t n_chunks_val = num_chunks;
  CUDA_CHECK(cudaMemcpyAsync(d_payload, &n_chunks_val, 4,
                             cudaMemcpyHostToDevice, stream));
  int offset = 4;

  if (b_len_bits > 0) {
    CUDA_CHECK(cudaMemcpyAsync(d_payload + offset, d_chunk_bit_lens, b_len_bits,
                               cudaMemcpyDeviceToDevice, stream));
    offset += b_len_bits;
  }
  CUDA_CHECK(cudaMemcpyAsync(d_payload + offset, d_dec_table, b_len_dec,
                             cudaMemcpyDeviceToDevice, stream));
  offset += b_len_dec;
  CUDA_CHECK(cudaMemcpyAsync(d_payload + offset, d_dec_symbol, b_len_dec,
                             cudaMemcpyDeviceToDevice, stream));
  offset += b_len_dec;

  if (b_len_words > 0) {
    CUDA_CHECK(cudaMemcpyAsync(d_payload + offset, d_dense_words, b_len_words,
                               cudaMemcpyDeviceToDevice, stream));
    offset += b_len_words;
  }

  comp_size = offset;

  CUDA_CHECK(cudaMemsetAsync(d_gpu_hash, 0, sizeof(uint64_t), stream));
  int hash_threads = 256;
  int hash_blocks = ((comp_size + 7) / 8 + hash_threads - 1) / hash_threads;
  gpu_hash_kernel<<<hash_blocks, hash_threads, 0, stream>>>(
      d_payload, d_gpu_hash, comp_size);

  CUDA_CHECK(cudaMemcpyAsync(host_out, d_payload, comp_size,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(&gpu_hash, d_gpu_hash, sizeof(uint64_t),
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaEventRecord(e_end, stream));
  cudaStreamSynchronize(stream);
}

DecompressionContext::DecompressionContext(int macro_bytes, int mini_bytes,
                                           int state_L) {
  macro_size = macro_bytes;
  mini_size = mini_bytes;
  L = state_L;

  max_chunks = (macro_size + mini_size - 1) / mini_size;
  max_words = (int)(mini_size * 1.5 / 8) + 80;

  size_t payload_alloc_size =
      macro_size + (size_t)max_chunks * max_words * sizeof(uint64_t) + 65536;

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  CUDA_CHECK(cudaMallocHost(&host_in, payload_alloc_size));
  CUDA_CHECK(cudaMallocHost(&host_out, macro_size));
  CUDA_CHECK(cudaMallocHost(&host_calc_hash, sizeof(uint64_t)));

  // Pinned host scalars for async copy safety
  CUDA_CHECK(cudaMallocHost(&host_uncomp_size, sizeof(int)));
  CUDA_CHECK(cudaMallocHost(&host_primary_idx, sizeof(int)));

  CUDA_CHECK(cudaMalloc(&d_data, macro_size));
  CUDA_CHECK(cudaMalloc(&d_bwt, macro_size));
  CUDA_CHECK(cudaMalloc(&d_keys, macro_size * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_keys_alt, macro_size * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_primary_idx, sizeof(int)));

  CUDA_CHECK(cudaMalloc(&d_global_LF, macro_size * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_J_in, macro_size * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_D_in, macro_size * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_J_out, macro_size * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_D_out, macro_size * sizeof(int)));

  CUDA_CHECK(cudaMalloc(&d_chunk_bit_lengths, max_chunks * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_chunk_word_lens, max_chunks * sizeof(uint32_t)));

  size_t dec_bytes = L * sizeof(int) * 2;
  CUDA_CHECK(cudaMalloc(&d_decoding_table, dec_bytes));
  d_decoding_symbol = d_decoding_table + L;

  CUDA_CHECK(cudaMalloc(&d_word_offsets, (max_chunks + 1) * sizeof(uint32_t)));
  CUDA_CHECK(
      cudaMalloc(&d_dense_words, max_chunks * max_words * sizeof(uint64_t)));

  CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemsetAsync(d_offsets, 0, sizeof(uint64_t), stream));
  CUDA_CHECK(cudaMalloc(&d_sizes, sizeof(int)));

  CUDA_CHECK(cudaMalloc(&d_payload, payload_alloc_size));
  CUDA_CHECK(cudaMalloc(&d_gpu_hash, sizeof(uint64_t)));

  size_t sort_bytes = 0, scan_bytes = 0;
  cub::DoubleBuffer<uint64_t> d_keys_db(d_keys, d_keys_alt);
  cub::DoubleBuffer<int> d_sa_db(d_J_in, d_J_out);
  cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, d_keys_db, d_sa_db,
                                  macro_size, 0, 64, stream);
  cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, d_chunk_word_lens,
                                d_word_offsets, max_chunks, stream);

  temp_storage_bytes = std::max(sort_bytes, scan_bytes) + 65536;
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // OPTIMIZATION FIX: Set-aside exactly dec_bytes (16 KB) rather than 32 MB!
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, dec_bytes));

  cudaStreamAttrValue attr;
  std::memset(&attr, 0, sizeof(attr));
  attr.accessPolicyWindow.base_ptr = (void *)d_decoding_table;
  attr.accessPolicyWindow.num_bytes = dec_bytes;
  attr.accessPolicyWindow.hitRatio = 1.0f; // 100% L2 Persistence Preference
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

  CUDA_CHECK(cudaStreamSetAttribute(
      stream, cudaStreamAttributeAccessPolicyWindow, &attr));

  // -------------------------------------------------------------------------
  // PRE-CAPTURE: Inverse BWT CUDA Graph for full-size macro chunks
  // Eliminates ~30 kernel launch overheads per decompression (~150-300us)
  // -------------------------------------------------------------------------
  CUDA_CHECK(cudaStreamSynchronize(stream));
  {
    int blocks_u = (macro_size + 255) / 256;
    int steps = (int)std::ceil(std::log2((double)macro_size));

    *host_uncomp_size = macro_size;
    *host_primary_idx = 0;

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    CUDA_CHECK(cudaMemcpyAsync(d_sizes, host_uncomp_size, sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_primary_idx, host_primary_idx, sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    dim3 grid_k(blocks_u, 1);
    fill_key_kernel<<<grid_k, 256, 0, stream>>>(d_offsets, d_sizes, d_bwt,
                                                d_keys, 1);
    fill_sequence_kernel<<<blocks_u, 256, 0, stream>>>(d_J_in, macro_size);

    cub::DoubleBuffer<uint64_t> keys_db(d_keys, d_keys_alt);
    cub::DoubleBuffer<int> f_to_l_db(d_J_in, d_J_out);
    size_t t_bytes = temp_storage_bytes;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, t_bytes, keys_db, f_to_l_db,
                                    macro_size, 0, 64, stream);

    build_lf_kernel<<<blocks_u, 256, 0, stream>>>(f_to_l_db.Current(),
                                                  d_global_LF, macro_size);

    dim3 grid_p(blocks_u, 1);
    dim3 block_p(256, 1);
    jump_init_kernel<<<grid_p, block_p, 0, stream>>>(
        d_global_LF, d_primary_idx, d_offsets, d_sizes, d_J_in, d_D_in);

    int *curr_J = d_J_in, *curr_D = d_D_in;
    int *next_J = d_J_out, *next_D = d_D_out;
    for (int s = 0; s < steps; s++) {
      jump_step_kernel<<<grid_p, block_p, 0, stream>>>(
          curr_J, curr_D, next_J, next_D, d_offsets, d_sizes);
      std::swap(curr_J, next_J);
      std::swap(curr_D, next_D);
    }

    jump_scatter_kernel<<<grid_p, block_p, 0, stream>>>(
        curr_D, d_bwt, d_data, d_primary_idx, d_offsets, d_sizes);

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec, graph, 0));
  }

  CUDA_CHECK(cudaEventCreate(&e_start));
  CUDA_CHECK(cudaEventCreate(&e_end));
}

DecompressionContext::~DecompressionContext() {
  if (stream) {
    cudaStreamSynchronize(stream);
  }

  if (graph_exec) {
    cudaGraphExecDestroy(graph_exec);
    graph_exec = nullptr;
  }
  if (graph) {
    cudaGraphDestroy(graph);
    graph = nullptr;
  }
  if (stream) {
    cudaStreamDestroy(stream);
  }

  cudaFreeHost(host_in);
  cudaFreeHost(host_out);
  cudaFreeHost(host_calc_hash);
  cudaFreeHost(host_uncomp_size);
  cudaFreeHost(host_primary_idx);

  cudaFree(d_data);
  cudaFree(d_bwt);
  cudaFree(d_keys);
  cudaFree(d_keys_alt);
  cudaFree(d_primary_idx);
  cudaFree(d_global_LF);
  cudaFree(d_J_in);
  cudaFree(d_D_in);
  cudaFree(d_J_out);
  cudaFree(d_D_out);
  cudaFree(d_chunk_bit_lengths);
  cudaFree(d_chunk_word_lens);
  cudaFree(d_decoding_table);
  cudaFree(d_word_offsets);
  cudaFree(d_dense_words);
  cudaFree(d_offsets);
  cudaFree(d_sizes);
  cudaFree(d_payload);
  cudaFree(d_gpu_hash);
  cudaFree(d_temp_storage);

  cudaEventDestroy(e_start);
  cudaEventDestroy(e_end);
}

bool DecompressionContext::decompress_chunk(int threads_decomp,
                                            int mini_chunk_size) {
  CUDA_CHECK(cudaEventRecord(e_start, stream));

  CUDA_CHECK(cudaMemcpyAsync(d_payload, host_in, comp_size,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemsetAsync(d_gpu_hash, 0, sizeof(uint64_t), stream));

  int hash_threads = 256;
  int hash_blocks = ((comp_size + 7) / 8 + hash_threads - 1) / hash_threads;
  gpu_hash_kernel<<<hash_blocks, hash_threads, 0, stream>>>(
      d_payload, d_gpu_hash, comp_size);

  CUDA_CHECK(cudaMemcpyAsync(host_calc_hash, d_gpu_hash, sizeof(uint64_t),
                             cudaMemcpyDeviceToHost, stream));

  if (is_raw == 1) {
    CUDA_CHECK(cudaMemcpyAsync(d_data, d_payload, comp_size,
                               cudaMemcpyDeviceToDevice, stream));
  } else if (is_raw == 2) {
    CUDA_CHECK(cudaMemsetAsync(d_data, host_in[0], uncomp_size, stream));
  } else {
    num_chunks = (uncomp_size + mini_chunk_size - 1) / mini_chunk_size;

    int offset = 4;
    int b_len_bits = num_chunks * 4;
    if (b_len_bits > 0) {
      CUDA_CHECK(cudaMemcpyAsync(d_chunk_bit_lengths, d_payload + offset,
                                 b_len_bits, cudaMemcpyDeviceToDevice, stream));
    }
    offset += b_len_bits;

    int b_len_dec = L * 4;
    CUDA_CHECK(cudaMemcpyAsync(d_decoding_table, d_payload + offset, b_len_dec,
                               cudaMemcpyDeviceToDevice, stream));
    offset += b_len_dec;

    CUDA_CHECK(cudaMemcpyAsync(d_decoding_symbol, d_payload + offset, b_len_dec,
                               cudaMemcpyDeviceToDevice, stream));
    offset += b_len_dec;

    if (num_chunks > 0) {
      int c_blocks = (num_chunks + 255) / 256;
      bit_to_word_kernel<<<c_blocks, 256, 0, stream>>>(
          d_chunk_bit_lengths, d_chunk_word_lens, num_chunks);

      CUDA_CHECK(cudaMemsetAsync(d_word_offsets, 0, sizeof(uint32_t), stream));
      size_t t_bytes = temp_storage_bytes;
      cub::DeviceScan::InclusiveSum(d_temp_storage, t_bytes, d_chunk_word_lens,
                                    d_word_offsets + 1, num_chunks, stream);

      int header_bytes = 4 + b_len_bits + (2 * b_len_dec);
      int b_len_words =
          (comp_size > header_bytes) ? (comp_size - header_bytes) : 0;
      if (b_len_words > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_dense_words, d_payload + offset,
                                   b_len_words, cudaMemcpyDeviceToDevice,
                                   stream));
      }

      int blocks = (num_chunks + threads_decomp - 1) / threads_decomp;
      // DYNAMIC SMEM SIZE: (2 * L * sizeof(int)) + (threads_decomp * 256 bytes)
      // for alphabet
      size_t smem_size = (2 * L * sizeof(int)) + (size_t)threads_decomp * 256;

      if (smem_size > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(
            (const void *)tabled_decode_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
      }

      tabled_decode_kernel<<<blocks, threads_decomp, smem_size, stream>>>(
          d_dense_words, d_word_offsets, d_chunk_bit_lengths, d_bwt,
          d_decoding_table, d_decoding_symbol, uncomp_size, mini_chunk_size, L);
    }
  }

  if (is_raw == 0) {
    // Write scalars into pinned memory BEFORE graph launch reads them
    *host_uncomp_size = uncomp_size;
    *host_primary_idx = primary_idx;

    if (uncomp_size == macro_size && graph_exec) {
      // CUDA GRAPH FAST PATH: Replay pre-captured inverse BWT graph
      CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    } else {
      // FALLBACK PATH: Inline launches for partial/last macro chunk
      CUDA_CHECK(cudaMemcpyAsync(d_sizes, host_uncomp_size, sizeof(int),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_primary_idx, host_primary_idx, sizeof(int),
                                 cudaMemcpyHostToDevice, stream));

      int blocks_u = (uncomp_size + 255) / 256;

      dim3 grid_k(blocks_u, 1);
      fill_key_kernel<<<grid_k, 256, 0, stream>>>(d_offsets, d_sizes, d_bwt,
                                                  d_keys, 1);
      fill_sequence_kernel<<<blocks_u, 256, 0, stream>>>(d_J_in, uncomp_size);

      cub::DoubleBuffer<uint64_t> keys_db(d_keys, d_keys_alt);
      cub::DoubleBuffer<int> f_to_l_db(d_J_in, d_J_out);
      size_t t_bytes = temp_storage_bytes;
      cub::DeviceRadixSort::SortPairs(d_temp_storage, t_bytes, keys_db,
                                      f_to_l_db, uncomp_size, 0, 64, stream);

      build_lf_kernel<<<blocks_u, 256, 0, stream>>>(f_to_l_db.Current(),
                                                    d_global_LF, uncomp_size);

      dim3 grid_p(blocks_u, 1);
      dim3 block_p(256, 1);
      jump_init_kernel<<<grid_p, block_p, 0, stream>>>(
          d_global_LF, d_primary_idx, d_offsets, d_sizes, d_J_in, d_D_in);

      int *curr_J = d_J_in, *curr_D = d_D_in;
      int *next_J = d_J_out, *next_D = d_D_out;
      int steps = (int)std::ceil(std::log2(uncomp_size));

      for (int s = 0; s < steps; s++) {
        jump_step_kernel<<<grid_p, block_p, 0, stream>>>(
            curr_J, curr_D, next_J, next_D, d_offsets, d_sizes);
        std::swap(curr_J, next_J);
        std::swap(curr_D, next_D);
      }

      jump_scatter_kernel<<<grid_p, block_p, 0, stream>>>(
          curr_D, d_bwt, d_data, d_primary_idx, d_offsets, d_sizes);
    }
  }

  CUDA_CHECK(cudaMemcpyAsync(host_out, d_data, uncomp_size,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(e_end, stream));
  return true;
}
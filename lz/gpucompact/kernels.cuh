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
#include <device_launch_parameters.h>

// -------------------------------------------------------------------------
// PTX ASYNC COPY INTRINSICS (sm_80 / sm_89 Hardware DMA)
// -------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ inline void cp_async_16(void *smem_ptr, const void *glob_ptr) {
  uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr),
               "l"(glob_ptr));
}

__device__ inline void cp_async_4(void *smem_ptr, const void *glob_ptr) {
  uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_addr),
               "l"(glob_ptr));
}

__device__ inline void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

__device__ inline void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n");
}
#endif

// -------------------------------------------------------------------------
// HELPER UTILITY KERNEL PROTOTYPES
// -------------------------------------------------------------------------
__global__ void cast_uint8_to_int32_kernel(const unsigned char *__restrict__ in,
                                           int *__restrict__ out, int n);

__global__ void bit_to_word_kernel(const unsigned int *__restrict__ bits,
                                   unsigned int *__restrict__ words, int n);

__global__ void fill_sequence_kernel(int *__restrict__ out, int n);

// -------------------------------------------------------------------------
// BWT & TANS KERNEL PROTOTYPES
// -------------------------------------------------------------------------
__global__ void sa_key_kernel(const int *__restrict__ rank,
                              int *__restrict__ sa,
                              unsigned long long *__restrict__ keys, int n,
                              int k);

__global__ void
sa_diff_kernel(const unsigned long long *__restrict__ sorted_keys,
               int *__restrict__ diff, int n);

__global__ void sa_rank_kernel(const int *__restrict__ unique_ranks,
                               const int *__restrict__ sa_sorted,
                               int *__restrict__ rank, int n);

__global__ void extract_bwt_kernel(const int *__restrict__ sa,
                                   const unsigned char *__restrict__ data,
                                   unsigned char *__restrict__ bwt,
                                   int *__restrict__ primary_idx, int n);

__global__ void
zrle_encode_kernel(const unsigned char *__restrict__ bwt,
                   unsigned short *__restrict__ out_symbols,
                   unsigned int *__restrict__ chunk_symbol_lengths,
                   int total_size, int chunk_size, int threads_comp);

__global__ void
compute_histogram_kernel(const unsigned short *__restrict__ symbols,
                         const unsigned int *__restrict__ lengths,
                         unsigned int *__restrict__ hist, int num_chunks,
                         int chunk_size);

__global__ void build_tans_all_kernel(
    const unsigned int *__restrict__ hist, int *__restrict__ p_out,
    int *__restrict__ prefix_p_out, int *__restrict__ max_x,
    int *__restrict__ symbol_spread, int *__restrict__ encoding_table,
    int *__restrict__ decoding_table, int *__restrict__ decoding_symbol,
    int *__restrict__ next_state, int L, int alphabet_size);

__global__ void tabled_encode_kernel(const unsigned short *__restrict__ symbols,
                                     const unsigned int *__restrict__ lengths,
                                     unsigned long long *__restrict__ out_words,
                                     unsigned int *__restrict__ bit_lengths,
                                     const int *__restrict__ p,
                                     const int *__restrict__ prefix_p,
                                     const int *__restrict__ max_x,
                                     const int *__restrict__ enc_table,
                                     int num_chunks, int chunk_size,
                                     int max_words, int L);

__global__ void tabled_decode_kernel(
    const unsigned long long *__restrict__ in_words,
    const unsigned int *__restrict__ word_offsets,
    const unsigned int *__restrict__ bit_lengths,
    unsigned char *__restrict__ bwt, const int *__restrict__ dec_table,
    const int *__restrict__ dec_symbol, int total_size, int chunk_size, int L);

__global__ void dense_pack_kernel(const unsigned long long *__restrict__ in,
                                  const unsigned int *__restrict__ offsets,
                                  const unsigned int *__restrict__ lens,
                                  unsigned long long *__restrict__ out, int n,
                                  int max_w);

__global__ void fill_key_kernel(const unsigned long long *__restrict__ offsets,
                                const int *__restrict__ sizes,
                                const unsigned char *__restrict__ bwt,
                                unsigned long long *__restrict__ key,
                                int num_chunks);

__global__ void build_lf_kernel(const int *__restrict__ F_to_L,
                                int *__restrict__ LF, int n);

__global__ void
jump_init_kernel(const int *__restrict__ LF,
                 const int *__restrict__ primary_indices,
                 const unsigned long long *__restrict__ chunk_offsets,
                 const int *__restrict__ chunk_sizes, int *__restrict__ J,
                 int *__restrict__ D);

__global__ void jump_step_kernel(const int *__restrict__ J_in,
                                 const int *__restrict__ D_in,
                                 int *__restrict__ J_out,
                                 int *__restrict__ D_out,
                                 const unsigned long long *__restrict__ offsets,
                                 const int *__restrict__ sizes);

__global__ void jump_scatter_kernel(
    const int *__restrict__ D, const unsigned char *__restrict__ bwt,
    unsigned char *__restrict__ out, const int *__restrict__ primary,
    const unsigned long long *__restrict__ offsets,
    const int *__restrict__ sizes);

__global__ void gpu_hash_kernel(const unsigned char *__restrict__ data,
                                unsigned long long *__restrict__ d_hash,
                                int size);
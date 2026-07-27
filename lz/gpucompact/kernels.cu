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
#include "kernels.cuh"

__global__ void cast_uint8_to_int32_kernel(const unsigned char *__restrict__ in,
                                           int *__restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = (int)in[idx];
}

__global__ void bit_to_word_kernel(const unsigned int *__restrict__ bits,
                                   unsigned int *__restrict__ words, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    words[idx] = (bits[idx] + 63) / 64;
}

__global__ void fill_sequence_kernel(int *__restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = idx;
}

__global__ void sa_key_kernel(const int *__restrict__ rank,
                              int *__restrict__ sa,
                              unsigned long long *__restrict__ keys, int n,
                              int k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  unsigned long long r1 = rank[idx];
  unsigned long long r2 = rank[(idx + k) % n];
  keys[idx] = (r1 << 32) | r2;
  sa[idx] = idx;
}

__global__ void
sa_diff_kernel(const unsigned long long *__restrict__ sorted_keys,
               int *__restrict__ diff, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  if (idx == 0)
    diff[idx] = 1;
  else
    diff[idx] = (sorted_keys[idx] != sorted_keys[idx - 1]) ? 1 : 0;
}

__global__ void sa_rank_kernel(const int *__restrict__ unique_ranks,
                               const int *__restrict__ sa_sorted,
                               int *__restrict__ rank, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  rank[sa_sorted[idx]] = unique_ranks[idx] - 1;
}

__global__ void extract_bwt_kernel(const int *__restrict__ sa,
                                   const unsigned char *__restrict__ data,
                                   unsigned char *__restrict__ bwt,
                                   int *__restrict__ primary_idx, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  int s = sa[idx];
  if (s == 0)
    *primary_idx = idx;
  bwt[idx] = data[(s == 0) ? n - 1 : s - 1];
}

__global__ void
zrle_encode_kernel(const unsigned char *__restrict__ bwt,
                   unsigned short *__restrict__ out_symbols,
                   unsigned int *__restrict__ chunk_symbol_lengths,
                   int total_size, int chunk_size, int threads_comp) {
  int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
  int start = chunk_id * chunk_size;
  if (start >= total_size)
    return;
  int end = start + chunk_size;
  if (end > total_size)
    end = total_size;

  extern __shared__ unsigned char s_alphabet[];
  unsigned char *alphabet = &s_alphabet[threadIdx.x * 260];
  for (int i = 0; i < 256; i++)
    alphabet[i] = (unsigned char)i;

  int zero_run = 0, sym_offset = 0, out_base = chunk_id * chunk_size;
  for (int i = start; i < end; i++) {
    unsigned char c = bwt[i], idx = 0;
    for (int j = 0; j < 256; j++) {
      if (alphabet[j] == c) {
        idx = (unsigned char)j;
        break;
      }
    }
    for (int j = idx; j > 0; j--)
      alphabet[j] = alphabet[j - 1];
    alphabet[0] = c;

    if (idx == 0) {
      zero_run++;
    } else {
      if (zero_run > 0) {
        int temp = zero_run;
        while (temp > 0) {
          if (temp % 2 == 1) {
            out_symbols[out_base + sym_offset++] = 0;
            temp = (temp - 1) / 2;
          } else {
            out_symbols[out_base + sym_offset++] = 1;
            temp = (temp - 2) / 2;
          }
        }
        zero_run = 0;
      }
      out_symbols[out_base + sym_offset++] = idx + 1;
    }
  }
  if (zero_run > 0) {
    int temp = zero_run;
    while (temp > 0) {
      if (temp % 2 == 1) {
        out_symbols[out_base + sym_offset++] = 0;
        temp = (temp - 1) / 2;
      } else {
        out_symbols[out_base + sym_offset++] = 1;
        temp = (temp - 2) / 2;
      }
    }
  }
  chunk_symbol_lengths[chunk_id] = sym_offset;
}

__global__ void
compute_histogram_kernel(const unsigned short *__restrict__ symbols,
                         const unsigned int *__restrict__ lengths,
                         unsigned int *__restrict__ hist, int num_chunks,
                         int chunk_size) {
  int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk_id >= num_chunks)
    return;
  int length = lengths[chunk_id], base = chunk_id * chunk_size;
  for (int i = 0; i < length; i++) {
    atomicAdd(&hist[symbols[base + i]], 1);
  }
}

__global__ void build_tans_all_kernel(
    const unsigned int *__restrict__ hist, int *__restrict__ p_out,
    int *__restrict__ prefix_p_out, int *__restrict__ max_x,
    int *__restrict__ symbol_spread, int *__restrict__ encoding_table,
    int *__restrict__ decoding_table, int *__restrict__ decoding_symbol,
    int *__restrict__ next_state, int L, int alphabet_size) {
  int tid = threadIdx.x;
  if (tid >= alphabet_size)
    return;

  __shared__ unsigned int s_hist[257];
  __shared__ int s_p[257];
  s_hist[tid] = hist[tid];
  s_p[tid] = (hist[tid] > 0) ? 1 : 0;
  __syncthreads();

  int p_sum = 0;
  double weight_sum = 0.0;
  for (int i = 0; i < alphabet_size; i++) {
    p_sum += s_p[i];
    weight_sum += (double)s_hist[i];
  }
  int remaining = L - p_sum;
  if (remaining > 0) {
    double weight = (double)s_hist[tid];
    int add = (weight_sum > 0.0) ? (int)((remaining * weight) / weight_sum)
                                 : (tid == 0 ? remaining : 0);
    s_p[tid] += add;
  }
  __syncthreads();

  p_sum = 0;
  double max_weight = -1.0;
  int max_idx = 0;
  for (int i = 0; i < alphabet_size; i++) {
    p_sum += s_p[i];
    if ((double)s_hist[i] > max_weight) {
      max_weight = (double)s_hist[i];
      max_idx = i;
    }
  }
  if (L - p_sum > 0 && tid == max_idx && weight_sum > 0.0)
    s_p[tid] += (L - p_sum);
  __syncthreads();

  p_out[tid] = s_p[tid];
  max_x[tid] = 2 * s_p[tid] - 1;
  int prefix = 0;
  for (int i = 0; i < tid; i++)
    prefix += s_p[i];
  prefix_p_out[tid] = prefix;

  int step = (L / 2) + 3;
  for (int k = 0; k < s_p[tid]; k++) {
    symbol_spread[((prefix + k) * step) % L] = tid;
  }
  __syncthreads();

  if (tid == 0) {
    for (int i = 0; i < alphabet_size; i++)
      next_state[i] = s_p[i];
    for (int x = L; x < 2 * L; x++) {
      int s = symbol_spread[x - L];
      decoding_symbol[x - L] = s;
      int state = next_state[s];
      decoding_table[x - L] = state;
      encoding_table[prefix_p_out[s] + state - s_p[s]] = x;
      next_state[s] = state + 1;
    }
  }
}

__global__ void tabled_encode_kernel(const unsigned short *__restrict__ symbols,
                                     const unsigned int *__restrict__ lengths,
                                     unsigned long long *__restrict__ out_words,
                                     unsigned int *__restrict__ bit_lengths,
                                     const int *__restrict__ p,
                                     const int *__restrict__ prefix_p,
                                     const int *__restrict__ max_x,
                                     const int *__restrict__ enc_table,
                                     int num_chunks, int chunk_size,
                                     int max_words, int L) {
  int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ unsigned char s_mem_th[];
  int *s_enc = (int *)s_mem_th;
  int *s_p = (int *)(s_mem_th + L * sizeof(int));
  int *s_prefix_p = (int *)(s_mem_th + L * sizeof(int) + 257 * sizeof(int));

  // Coalesced, branch-free shared memory pre-staging
  for (int i = threadIdx.x; i < L; i += blockDim.x)
    s_enc[i] = enc_table[i];
  for (int i = threadIdx.x; i < 257; i += blockDim.x) {
    s_p[i] = p[i];
    s_prefix_p[i] = prefix_p[i];
  }
  __syncthreads();

  if (chunk_id >= num_chunks)
    return;

  int len = lengths[chunk_id], in_base = chunk_id * chunk_size,
      out_base = chunk_id * max_words;
  unsigned long long bit_buf = 0;
  int bit_cnt = 0, word_off = 0, x = L;

  for (int i = len - 1; i >= 0; i--) {
    unsigned short s = symbols[in_base + i];
    int mx = max_x[s];
    while (x > mx) {
      bit_buf |= ((unsigned long long)(x & 1) << bit_cnt++);
      if (bit_cnt == 64) {
        out_words[out_base + word_off++] = bit_buf;
        bit_cnt = 0;
        bit_buf = 0;
      }
      x >>= 1;
    }
    x = s_enc[s_prefix_p[s] + x - s_p[s]];
  }

  int state_bits = 32 - __clz(L);
  for (int b = 0; b < state_bits; b++) {
    bit_buf |= ((unsigned long long)((x >> b) & 1) << bit_cnt++);
    if (bit_cnt == 64) {
      out_words[out_base + word_off++] = bit_buf;
      bit_cnt = 0;
      bit_buf = 0;
    }
  }
  if (bit_cnt > 0)
    out_words[out_base + word_off++] = bit_buf;
  bit_lengths[chunk_id] = (word_off * 64) - (64 - bit_cnt);
  if (bit_cnt == 0)
    bit_lengths[chunk_id] = word_off * 64;
}

__global__ void tabled_decode_kernel(
    const unsigned long long *__restrict__ in_words,
    const unsigned int *__restrict__ word_offsets,
    const unsigned int *__restrict__ bit_lengths,
    unsigned char *__restrict__ bwt, const int *__restrict__ dec_table,
    const int *__restrict__ dec_symbol, int total_size, int chunk_size, int L) {
  extern __shared__ int s_dec_mem[];
  int *s_dec_table = s_dec_mem;
  int *s_dec_symbol = &s_dec_mem[L];

  // OPTIMIZATION FIX: Move MTF alphabet table from local stack (DRAM) into
  // Shared Memory!
  unsigned char *s_alphabets = (unsigned char *)&s_dec_mem[2 * L];
  unsigned char *alphabet = &s_alphabets[threadIdx.x * 256];

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  for (int i = threadIdx.x * 4; i < L; i += blockDim.x * 4) {
    if (i + 3 < L) {
      cp_async_16(&s_dec_table[i], &dec_table[i]);
      cp_async_16(&s_dec_symbol[i], &dec_symbol[i]);
    } else {
      for (int k = i; k < L; k++) {
        s_dec_table[k] = dec_table[k];
        s_dec_symbol[k] = dec_symbol[k];
      }
    }
  }
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();
#else
  for (int i = threadIdx.x; i < L; i += blockDim.x) {
    s_dec_table[i] = dec_table[i];
    s_dec_symbol[i] = dec_symbol[i];
  }
  __syncthreads();
#endif

  for (int i = 0; i < 256; i++)
    alphabet[i] = (unsigned char)i;

  int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
  int start = chunk_id * chunk_size;
  if (start >= total_size)
    return;
  int end = start + chunk_size;
  if (end > total_size)
    end = total_size;

  int word_base = word_offsets[chunk_id];
  int current_bit_idx = (int)bit_lengths[chunk_id] - 1;
  int x = 0;
  int state_bits = 32 - __clz(L);

  for (int b = state_bits - 1; b >= 0; b--) {
    unsigned int bit = 0;
    if (current_bit_idx >= 0) {
      bit = (unsigned int)((in_words[word_base + (current_bit_idx >> 6)] >>
                            (current_bit_idx & 63)) &
                           1ULL);
      current_bit_idx--;
    }
    x |= (bit << b);
  }

  int out_idx = start, run_length = 0, power = 1;
  while (out_idx + run_length < end) {
    int s = s_dec_symbol[x - L];
    x = s_dec_table[x - L];
    while (x < L) {
      unsigned int bit = 0;
      if (current_bit_idx >= 0) {
        bit = (unsigned int)((in_words[word_base + (current_bit_idx >> 6)] >>
                              (current_bit_idx & 63)) &
                             1ULL);
        current_bit_idx--;
      }
      x = (x << 1) | bit;
    }
    if (s == 0) {
      run_length += power * 1;
      power *= 2;
    } else if (s == 1) {
      run_length += power * 2;
      power *= 2;
    } else {
      if (run_length > 0) {
        for (int i = 0; i < run_length && out_idx < end; i++)
          bwt[out_idx++] = alphabet[0];
        run_length = 0;
        power = 1;
      }
      if (out_idx < end) {
        unsigned int mtf = s - 1;
        unsigned char c = alphabet[mtf];
        bwt[out_idx++] = c;
        for (int j = mtf; j > 0; j--)
          alphabet[j] = alphabet[j - 1];
        alphabet[0] = c;
      }
    }
  }
  if (run_length > 0) {
    for (int i = 0; i < run_length && out_idx < end; i++)
      bwt[out_idx++] = alphabet[0];
  }
}

__global__ void dense_pack_kernel(const unsigned long long *__restrict__ in,
                                  const unsigned int *__restrict__ offsets,
                                  const unsigned int *__restrict__ lens,
                                  unsigned long long *__restrict__ out, int n,
                                  int max_w) {
  int cid = blockIdx.x * blockDim.x + threadIdx.x;
  if (cid >= n)
    return;
  int len = lens[cid], in_b = cid * max_w, out_b = offsets[cid];

  uintptr_t in_ptr = reinterpret_cast<uintptr_t>(in + in_b);
  uintptr_t out_ptr = reinterpret_cast<uintptr_t>(out + out_b);

  // SAFE ALIGNMENT CHECK: Ensure both pointers are 16-byte aligned before
  // vector load/store
  if ((in_ptr % 16 == 0) && (out_ptr % 16 == 0)) {
    const ulonglong2 *in_128 = reinterpret_cast<const ulonglong2 *>(in + in_b);
    ulonglong2 *out_128 = reinterpret_cast<ulonglong2 *>(out + out_b);
    int vec_len = len / 2;
    for (int i = 0; i < vec_len; i++) {
      out_128[i] = in_128[i];
    }
    if (len % 2) {
      out[out_b + len - 1] = in[in_b + len - 1];
    }
  } else {
    for (int i = 0; i < len; i++) {
      out[out_b + i] = in[in_b + i];
    }
  }
}

__global__ void fill_key_kernel(const unsigned long long *__restrict__ offsets,
                                const int *__restrict__ sizes,
                                const unsigned char *__restrict__ bwt,
                                unsigned long long *__restrict__ key,
                                int num_chunks) {
  int cid = blockIdx.y;
  if (cid >= num_chunks)
    return;

  int lid = blockIdx.x * blockDim.x + threadIdx.x;
  int size = sizes[cid];
  if (lid >= size)
    return;

  unsigned long long off = offsets[cid];
  key[off + lid] = (((unsigned long long)cid) << 48) |
                   (((unsigned long long)bwt[off + lid]) << 32) |
                   (unsigned long long)lid;
}

__global__ void build_lf_kernel(const int *__restrict__ F_to_L,
                                int *__restrict__ LF, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    LF[F_to_L[idx]] = idx;
  }
}

__global__ void
jump_init_kernel(const int *__restrict__ LF,
                 const int *__restrict__ primary_indices,
                 const unsigned long long *__restrict__ chunk_offsets,
                 const int *__restrict__ chunk_sizes, int *__restrict__ J,
                 int *__restrict__ D) {
  int cid = blockIdx.y, lid = blockIdx.x * blockDim.x + threadIdx.x,
      size = chunk_sizes[cid];
  if (lid >= size)
    return;
  unsigned long long off = chunk_offsets[cid];
  int gid = off + lid, prim = off + primary_indices[cid],
      g_target = off + LF[gid];
  if (gid == prim) {
    J[gid] = gid;
    D[gid] = 0;
  } else {
    J[gid] = g_target;
    D[gid] = 1;
  }
}

__global__ void jump_step_kernel(const int *__restrict__ J_in,
                                 const int *__restrict__ D_in,
                                 int *__restrict__ J_out,
                                 int *__restrict__ D_out,
                                 const unsigned long long *__restrict__ offsets,
                                 const int *__restrict__ sizes) {
  int cid = blockIdx.y, lid = blockIdx.x * blockDim.x + threadIdx.x;
  if (lid >= sizes[cid])
    return;
  int gid = offsets[cid] + lid, my_J = J_in[gid];
  J_out[gid] = J_in[my_J];
  D_out[gid] = D_in[gid] + D_in[my_J];
}

__global__ void jump_scatter_kernel(
    const int *__restrict__ D, const unsigned char *__restrict__ bwt,
    unsigned char *__restrict__ out, const int *__restrict__ primary,
    const unsigned long long *__restrict__ offsets,
    const int *__restrict__ sizes) {
  int cid = blockIdx.y, lid = blockIdx.x * blockDim.x + threadIdx.x,
      size = sizes[cid];
  if (lid >= size)
    return;
  unsigned long long off = offsets[cid];
  int gid = off + lid, prim = off + primary[cid],
      d_forw = (gid == prim) ? 0 : (size - D[gid]);
  out[off + size - 1 - d_forw] = bwt[gid];
}

__global__ void gpu_hash_kernel(const unsigned char *__restrict__ data,
                                unsigned long long *__restrict__ d_hash,
                                int size) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  unsigned long long local_hash = 0;

  int num_words = (size + 7) / 8;

  if (idx < num_words) {
    int start_byte = idx * 8;
    int remain = size - start_byte;
    unsigned long long val = 0;

    if (remain >= 8) {
      val = *reinterpret_cast<const unsigned long long *>(data + start_byte);
    } else {
      for (int i = 0; i < remain; i++) {
        val |= (static_cast<unsigned long long>(data[start_byte + i]))
               << (i * 8);
      }
    }

    local_hash = val * 0xbf58476d1ce4e5b9ULL;
    local_hash ^= (local_hash >> 31);
  }

  for (int offset = 16; offset > 0; offset /= 2) {
    local_hash ^= __shfl_down_sync(0xFFFFFFFF, local_hash, offset);
  }

  __shared__ unsigned long long shared_hash[32];
  if (tid % 32 == 0)
    shared_hash[tid / 32] = local_hash;
  __syncthreads();

  if (tid < 32) {
    local_hash = (tid < ((blockDim.x + 31) / 32)) ? shared_hash[tid] : 0;
    for (int offset = 16; offset > 0; offset /= 2) {
      local_hash ^= __shfl_down_sync(0xFFFFFFFF, local_hash, offset);
    }
    if (tid == 0 && local_hash != 0) {
      atomicXor(d_hash, local_hash);
    }
  }
}
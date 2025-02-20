/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "common.h"
#include "nvcomp.h"
#include "nvcomp/cascaded.h"

#include <assert.h>
#include <stdint.h>

#include <cub/cub.cuh>

namespace nvcomp
{

template <typename T>
__device__ inline nvcompType_t d_TypeOf()
{
  if (std::is_same<T, int8_t>::value) {
    return NVCOMP_TYPE_CHAR;
  } else if (std::is_same<T, uint8_t>::value) {
    return NVCOMP_TYPE_UCHAR;
  } else if (std::is_same<T, int16_t>::value) {
    return NVCOMP_TYPE_SHORT;
  } else if (std::is_same<T, uint16_t>::value) {
    return NVCOMP_TYPE_USHORT;
  } else if (std::is_same<T, int32_t>::value) {
    return NVCOMP_TYPE_INT;
  } else if (std::is_same<T, uint32_t>::value) {
    return NVCOMP_TYPE_UINT;
  } else if (std::is_same<T, int64_t>::value) {
    return NVCOMP_TYPE_LONGLONG;
  } else if (std::is_same<T, uint64_t>::value) {
    return NVCOMP_TYPE_ULONGLONG;
  } else {
    return NVCOMP_TYPE_CHAR;
  }

  // TODO - perform error checking and notify user if incorrect type is given
}

constexpr int default_chunk_size = 4096;
// Partition metadata contains 8B: 4B for the numbers of every cascaded
// compression layers and another 4B for uncompressed bytes.
constexpr size_t partition_metadata_size = 8;
constexpr size_t num_bits_per_byte = 8;

constexpr int cascaded_compress_threadblock_size = 128;
constexpr int cascaded_decompress_threadblock_size = 128;

/**
 * Helper function to calculate the size in byte of the chunk metadata. The size
 * is guaranteed to be a multiple of the data type size, and a multiple of 4.
 */
template <typename data_type>
__device__ int get_chunk_metadata_size(int num_RLEs, int num_deltas)
{
  return roundUpTo(4 + 4 * (num_RLEs + 1), sizeof(data_type))
         + roundUpTo(sizeof(data_type) * num_deltas, 4);
}

/**
 * Perform RLE compression on a single threadblock.
 *
 * Note: \p num_outputs is written by the last thread of the threadblock, and
 * must be a shared memory location so that the change can be propagated to
 * other threads in the threadblock.
 *
 * @param[in] input_buffer Uncompressed input buffer.
 * @param[in] num_inputs Number of elements in \p input_buffer.
 * @param[out] val_buffer Value array after the RLE compression.
 * @param[out] count_buffer Count array after the RLE compression.
 * @param[out] num_outputs Number of output elements. This is the array size of
 * \p val_buffer or \p count_buffer. The buffer must locate in shared memory.
 * @param[in] tmp_buffer Temporary buffer that needs to hold at least
 * \p num_inputs elements.
 */
template <
    typename data_type,
    typename size_type,
    typename run_type,
    int threadblock_size>
__device__ void block_rle_compress(
    const data_type* input_buffer,
    const size_type num_inputs,
    data_type* val_buffer,
    run_type* count_buffer,
    size_type* num_outputs,
    run_type* tmp_buffer)
{
  // In this kernel, we assign `num_inputs_per_thread` consecutive elements to a
  // thread. Then, if an input element is the last element of a run, the thread
  // stores the value and the count to the output buffer. Note that an input
  // element is the last element of a run if and only if
  //   a) The value of this element is different from the value of the next
  //   element.
  //   or b) This element is the last element.
  //
  // The algorithm consists of the following steps.
  //   1. Each thread counts the number of last elements.
  //   2. Use prefix sum on the counts to calculate the output location of each
  //   thread.
  //   3. For each last element, store the value in `val_buffer` and the input
  //   index in `tmp_buffer`.
  //   4. Calculate the adjacent differences of input indices in `tmp_buffer` to
  //   get the counts, and store into `count_buffer`.
  //
  // Note: `tmp_buffer` is used because we cannot calculate adjacent differences
  // in place.

  typedef cub::BlockScan<size_type, threadblock_size> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  const size_type num_inputs_per_thread
      = roundUpDiv(num_inputs, threadblock_size);

  // Step 1: Count the number of last elements of the current thread

  size_type num_outputs_current_thread = 0;

  data_type val = input_buffer[threadIdx.x * num_inputs_per_thread];
  data_type next_val;

  for (int ielement = 0; ielement < num_inputs_per_thread; ielement++) {
    const int idx = threadIdx.x * num_inputs_per_thread + ielement;
    if (idx >= num_inputs)
      break;

    if (idx + 1 == num_inputs) {
      num_outputs_current_thread++;
      break;
    }

    next_val = input_buffer[idx + 1];
    num_outputs_current_thread += next_val != val;
    val = next_val;
  }

  __syncthreads();

  // Step 2: Use prefix sum to get the output location

  size_type output_idx;
  BlockScan(temp_storage).ExclusiveSum(num_outputs_current_thread, output_idx);

  __syncthreads();

  // Step 3: For each last element, store the index and the value

  val = input_buffer[threadIdx.x * num_inputs_per_thread];

  for (int ielement = 0; ielement < num_inputs_per_thread; ielement++) {
    const int idx = threadIdx.x * num_inputs_per_thread + ielement;
    if (idx >= num_inputs)
      break;

    if (idx + 1 == num_inputs) {
      tmp_buffer[output_idx] = idx + 1;
      val_buffer[output_idx] = input_buffer[idx];
      output_idx++;
      break;
    }

    data_type next_val = input_buffer[idx + 1];
    if (next_val != val) {
      tmp_buffer[output_idx] = idx + 1;
      val_buffer[output_idx] = input_buffer[idx];
      output_idx++;
    }
    val = next_val;
  }

  if (threadIdx.x == threadblock_size - 1) {
    // After step 2, `output_idx` of the last thread is the sum of the number of
    // last elements in all threads except itself (since we use ExclusiveSum).
    // During step 3, the number of last elements of the current thread is added
    // to `num_outputs`. Therefore, `output_idx` here is the number of runs
    // in total.
    *num_outputs = output_idx;
  }

  // syncthreads is necessary here to make `num_outputs` avaiable on all threads
  // in the current threadblock.
  __syncthreads();

  // Step 4: Calculate the adjacent differences between indices, which is the
  // counts of every runs.

  for (int ioutput = 1 + threadIdx.x; ioutput < *num_outputs;
       ioutput += threadblock_size)
    count_buffer[ioutput] = tmp_buffer[ioutput] - tmp_buffer[ioutput - 1];

  if (threadIdx.x == 0)
    count_buffer[0] = tmp_buffer[0];
}

/**
 * Perform RLE decompression on a single threadblock.
 *
 * @param[in] val_buffer Values of the runs.
 * @param[in] count_buffer Counts of the runs.
 * @param[in] num_runs Number of runs. This is also the size in terms of number
 * of elements for \p val_buffer and \p count_buffer.
 * @param[out] output_buffer Pointer to the output uncompressed buffer.
 * @param[out] output_num_elements Number of uncompressed elements. This
 * argument should be unique per thread. The output should be the sum of
 * \p count_buffer.
 */
template <
    typename data_type,
    typename size_type,
    typename run_type,
    int threadblock_size>
__device__ void block_rle_decompress(
    const data_type* val_buffer,
    const run_type* count_buffer,
    size_type num_runs,
    data_type* output_buffer,
    size_type* output_num_elements)
{
  // In this kernel, we assign runs to threads in a round-robin fashion. The
  // algorithm is divided into rounds, where each round handles
  // `threadblock_size` runs, with one run per thread. During each round, we
  // first use prefix sum to calculate the output offsets, and then each thread
  // stores the value of the run into the output locations.

  typedef cub::BlockScan<run_type, threadblock_size> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  *output_num_elements = 0;

  for (int round = 0; round < roundUpDiv(num_runs, threadblock_size); round++) {
    const int idx = round * threadblock_size + threadIdx.x;

    run_type current_count = 0;
    if (idx < num_runs)
      current_count = count_buffer[idx];

    run_type output_offset;
    run_type aggregate;
    BlockScan(temp_storage)
        .ExclusiveSum(current_count, output_offset, aggregate);

    if (idx < num_runs) {
      const auto current_val = val_buffer[idx];
      for (int element_idx = 0; element_idx < current_count; element_idx++) {
        output_buffer[*output_num_elements + output_offset + element_idx]
            = current_val;
      }
    }

    *output_num_elements += aggregate;

    // syncthreads is necessary to make sure temporary storage is not
    // overwritten in the next iteration until all threads finish for the
    // current iteration.
    __syncthreads();
  }
}

/**
 * Perform delta compression on a single threadblock.
 *
 * This function calculate the adjacent differences between consecutive elements
 * of the input buffer.
 *
 * @param[in] input_buffer Array of size \p input_size of the input elements.
 * @param[out] output_buffer Array of size (\p input_size - 1) of the adjacent
 * differences.
 */
template <typename data_type, typename size_type>
__device__ void block_delta_compress(
    const data_type* input_buffer,
    size_type input_size,
    data_type* output_buffer)
{
  for (size_type element_idx = threadIdx.x; element_idx < input_size - 1;
       element_idx += blockDim.x) {
    output_buffer[element_idx]
        = input_buffer[element_idx + 1] - input_buffer[element_idx];
  }
}

/**
 * Perform delta decompression on a single threadblock.
 *
 * This function calculate the prefix sum of the input elements, i.e. the output
 * sequence should be `initial_value`, `initial_value + input_buffer[0]`,
 * `initial_value + input_buffer[0] + input_buffer[1]`, etc.
 *
 * @param[in] input_buffer Array of size \p input_num_elements of the input
 * elements.
 * @param[in] initial_value The first element of the uncompressed buffer.
 * @param[out] output_buffer Array of size (\p input_num_elements + 1) of the
 * prefix sum output.
 */
template <typename data_type, typename size_type, int threadblock_size>
__device__ void block_delta_decompress(
    const data_type* input_buffer,
    data_type initial_value,
    size_type input_num_elements,
    data_type* output_buffer)
{
  typedef cub::BlockScan<data_type, threadblock_size> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  const int num_rounds = roundUpDiv(input_num_elements, threadblock_size);

  for (int round = 0; round < num_rounds; round++) {
    const size_type idx = round * threadblock_size + threadIdx.x;

    data_type input_val = 0;
    if (idx < input_num_elements)
      input_val = input_buffer[idx];

    data_type output_val;
    data_type aggregate;
    BlockScan(temp_storage)
        .ExclusiveScan(
            input_val, output_val, initial_value, cub::Sum(), aggregate);
    initial_value += aggregate;

    if (idx < input_num_elements)
      output_buffer[idx] = output_val;

    __syncthreads();
  }

  if (threadIdx.x == 0)
    output_buffer[input_num_elements] = initial_value;
}

/**
 * Helper function to calculate the frame of reference and bitwidth in
 * bitpacking layer.
 *
 * @param[in] input Array of size \p num_elements of the input elements.
 * @param[out] frame_of_reference Frame of reference in the bitpacking layer.
 * Currently this is the smallest element of \p input. This argument will be set
 * by thread 0 so the memory location should be accessible by thread 0 (e.g., in
 * shared memory).
 * @param[out] bitwidth_ptr The highest 16 bits of this field store the number
 * of bits needed in the bitpacked buffer to represent a single input element.
 * The lowest 16 bits store the number of elements. This argument will be set
 * by thread 0 so the memory location should be accessible by thread 0 (e.g., in
 * shared memory).
 */
template <typename data_type, typename size_type, int threadblock_size>
__device__ void get_for_bitwidth(
    const data_type* input,
    size_type num_elements,
    data_type* frame_of_reference,
    uint32_t* bitwidth_ptr)
{
  // Use signed data type since input could store negative values, e.g. the
  // output of delta layer. Although the signed type and the unsigned type have
  // the same raw bits, the interpretation of the smallest element is different
  // for negative values.
  using signed_data_type = std::make_signed_t<data_type>;

  // First, we calculate the maximum and the minimum of the input elements. We
  // process input elements in rounds, where each round processes
  // `threadblock_size` elements, with one element per thread.

  typedef cub::BlockReduce<signed_data_type, threadblock_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  signed_data_type thread_data;
  int num_valid = min(num_elements, static_cast<size_type>(threadblock_size));
  if (threadIdx.x < num_elements) {
    thread_data = input[threadIdx.x];
  }

  signed_data_type minimum
      = BlockReduce(temp_storage).Reduce(thread_data, cub::Min(), num_valid);
  __syncthreads();
  signed_data_type maximum
      = BlockReduce(temp_storage).Reduce(thread_data, cub::Max(), num_valid);
  __syncthreads();

  const int num_rounds = roundUpDiv(num_elements, threadblock_size);

  for (int round = 1; round < num_rounds; round++) {
    num_valid = min(
        num_elements - round * threadblock_size,
        static_cast<size_type>(threadblock_size));
    if (threadIdx.x < num_valid) {
      thread_data = input[threadIdx.x + round * threadblock_size];
    }

    const signed_data_type local_min
        = BlockReduce(temp_storage).Reduce(thread_data, cub::Min(), num_valid);
    __syncthreads();
    const signed_data_type local_max
        = BlockReduce(temp_storage).Reduce(thread_data, cub::Max(), num_valid);
    __syncthreads();

    if (threadIdx.x == 0 && local_min < minimum)
      minimum = local_min;
    if (threadIdx.x == 0 && local_max > maximum)
      maximum = local_max;
  }

  // Next, we store the frame of reference, the bitwidth and the number of
  // elements into the desired location.

  if (threadIdx.x == 0) {
    *frame_of_reference = minimum;

    uint32_t bitwidth;
    // calculate bit-width
    if (sizeof(data_type) > sizeof(int)) {
      const long long int range
          = static_cast<uint64_t>(maximum) - static_cast<uint64_t>(minimum);
      // need 64 bit clz
      bitwidth = sizeof(long long int) * num_bits_per_byte - __clzll(range);
    } else {
      const int range
          = static_cast<uint32_t>(maximum) - static_cast<uint32_t>(minimum);
      // can use 32 bit clz
      bitwidth = sizeof(int) * num_bits_per_byte - __clz(range);
    }
    *bitwidth_ptr = (bitwidth << 16) | static_cast<uint32_t>(num_elements);
  }
}

/**
 * Perform bitpacking on a single threadblock.
 *
 * @param[in] input Uncompressed input buffer.
 * @param[in] num_elements Number of input elements in \p input.
 * @param[out] output Bitpacked data including metadata.
 * @param[out] out_bytes Size of the bitpacked data in bytes. This argument
 * should be unique to each thread.
 */
template <typename data_type, typename size_type, int threadblock_size>
__device__ void block_bitpack(
    const data_type* input,
    size_type num_elements,
    uint32_t* output,
    size_type* out_bytes)
{
  // First, we need to use unsigned type during bitpacking because bit-shift on
  // negative values has undefined behavior. Next, we need to consider two
  // cases. If the bitwidth is larger than 32 bits, we need to use the unsigned
  // version of the `data_type` to make sure the difference between the data and
  // FOR can fit. If bitwidth is smaller than 32 bits, we need to use `uint32_t`
  // instead of the `data_type` to avoid left-shift beyond `data_type` limit.
  using unsigned_data_type = std::make_unsigned_t<data_type>;
  using padded_data_type = larger_t<unsigned_data_type, uint32_t>;

  auto for_ptr = reinterpret_cast<data_type*>(output);
  uint32_t* current_ptr = roundUpToAlignment<uint32_t>(for_ptr + 1);

  get_for_bitwidth<data_type, size_type, threadblock_size>(
      input, num_elements, for_ptr, current_ptr);

  __syncthreads();

  const data_type frame_of_reference = *for_ptr;
  const uint32_t bitwidth = (*current_ptr & 0xFFFF0000) >> 16;
  current_ptr = reinterpret_cast<uint32_t*>(
      roundUpToAlignment<data_type>(current_ptr + 1));

  const int num_output_elements = roundUpDiv(
      num_elements * bitwidth, sizeof(uint32_t) * num_bits_per_byte);
  if (out_bytes != nullptr) {
    // The bitpacking metadata consists of FOR (data_type size) and bitwidth and
    // the number of elements (4B). The start of the bitpacking data needs to be
    // both 4B and data_type aligned.
    *out_bytes = roundUpTo(
                     sizeof(data_type) + 4,
                     max(static_cast<size_t>(4), sizeof(data_type)))
                 + num_output_elements * sizeof(uint32_t);
  }

  for (int out_idx = threadIdx.x; out_idx < num_output_elements;
       out_idx += threadblock_size) {
    // The kernel works by assigning each 4B element of the bitpacked data to a
    // thread. The kernel then iterates over chunks of input, filling the bits
    // for each element, and then writing the stored bits to the output.
    const int out_bit_start = out_idx * sizeof(uint32_t) * num_bits_per_byte;
    const int out_bit_end
        = out_bit_start + sizeof(uint32_t) * num_bits_per_byte;
    const int input_idx_start = out_bit_start / bitwidth;
    const int input_idx_end = roundUpDiv(out_bit_end, bitwidth);

    uint32_t output_val = 0;
    for (int input_idx = input_idx_start; input_idx < input_idx_end;
         input_idx++) {
      unsigned_data_type input_val = 0;
      if (input_idx < num_elements)
        input_val = static_cast<unsigned_data_type>(
            input[input_idx] - frame_of_reference);

      auto padded_val = static_cast<padded_data_type>(input_val);
      const int offset = input_idx * bitwidth - out_bit_start;
      if (offset > 0) {
        padded_val <<= offset;
      } else {
        padded_val >>= -offset;
      }
      output_val |= static_cast<uint32_t>(padded_val);
    }
    current_ptr[out_idx] = output_val;
  }
}

/**
 * Perform bitunpacking on a single threadblock.
 *
 * @param[in] input Bitpacked input data.
 * @param[out] output Unpacked data.
 * @param[out] out_num_elements Number of `data_type` elements in the unpacked
 * data. This argument should be unique to each thread.
 */
template <typename data_type, typename size_type>
__device__ void block_bitunpack(
    const uint32_t* input, data_type* output, size_type* out_num_elements)
{
  const data_type* for_ptr = reinterpret_cast<const data_type*>(input);
  const uint32_t* current_ptr = roundUpToAlignment<uint32_t>(for_ptr + 1);

  const data_type frame_of_reference = *for_ptr;
  const uint32_t bitwidth = (*current_ptr & 0xFFFF0000) >> 16;
  const uint32_t num_elements = *current_ptr & 0x0000FFFF;

  if (out_num_elements != nullptr)
    *out_num_elements = static_cast<size_type>(num_elements);

  // Casting to unsigned type since bit shifting on negative numbers is
  // undefined.
  typedef typename std::make_unsigned<data_type>::type unsigned_type;
  const unsigned_type* data_ptr
      = roundUpToAlignment<unsigned_type>(current_ptr + 1);

  // Assign each output `data_type` element to a thread.
  for (int out_idx = threadIdx.x; out_idx < num_elements;
       out_idx += blockDim.x) {
    if (bitwidth == 0) {
      output[out_idx] = frame_of_reference;
    } else {
      // Shifting by width of the type is UB
      const unsigned_type mask
          = bitwidth < sizeof(unsigned_type) * num_bits_per_byte
                ? (static_cast<unsigned_type>(1) << bitwidth) - 1
                : static_cast<unsigned_type>(-1);

      // The current output element needs bits from at most two `data_type`
      // input elements, with indices `low_idx` and `high_idx`.
      constexpr size_t num_bits_data_type
          = sizeof(data_type) * num_bits_per_byte;
      const int low_idx = (out_idx * bitwidth) / num_bits_data_type;
      const int high_idx = (out_idx + 1) * bitwidth / num_bits_data_type;
      const int offset = out_idx * bitwidth - low_idx * num_bits_data_type;

      // Load and shift bits from `low_idx` input element
      unsigned_type base_value = data_ptr[low_idx] >> offset;

      if (low_idx < high_idx && offset != 0) {
        // If the current output element crosses input element boundary (i.e. it
        // corresponds to two input elements), we load and shift from `high_idx
        // as well.
        base_value += data_ptr[high_idx] << (num_bits_data_type - offset);
      }

      base_value &= mask;

      output[out_idx] = static_cast<data_type>(base_value) + frame_of_reference;
    }
  }
}

enum class BlockIOStatus
{
  success,
  out_of_bound
};

/**
 * Write a buffer with a single threadblock, optionally with bitpacking.
 *
 * For the current implementation, this function is used to write a layer
 * output from the shared memory to the global memory.
 *
 * @param[in] input Pointer to the uncompressed input buffer.
 * @param[in] num_elements Number of input elements.
 * @param[out] output Pointer to the output buffer.
 * @param[in] output_limit Pointer to the location just past the end of the
 * output buffer. If the output need to overflow beyond this pointer,
 * `BlockIOStatus::out_of_bound` will be returned. This argument must belong to
 * the same array as \p output. Otherwise, the behavior is undefined.
 * @param[out] out_bytes Number of bytes written to \p output. This argument
 * should be unique to each thread.
 * @param[in] temp_storage Temporary storage used for holding the bitpacked
 * data. Users should guarantee this storage has enough space for the combined
 * of the bitpacked metadata and the bitpacked data.
 * @param[in] use_bp Whether bitpacking should be used.
 */
template <typename data_type, typename size_type, int threadblock_size>
__device__ BlockIOStatus block_write(
    const data_type* input,
    size_type num_elements,
    uint32_t* output,
    const uint32_t* output_limit,
    size_type* out_bytes,
    uint32_t* temp_storage,
    bool use_bp)
{
  const uint32_t* source = nullptr;

  if (use_bp) {
    block_bitpack<data_type, size_type, threadblock_size>(
        input, num_elements, temp_storage, out_bytes);
    __syncthreads();
    source = temp_storage;
  } else {
    *out_bytes = num_elements * sizeof(data_type);
    source = reinterpret_cast<const uint32_t*>(input);
  }

  const size_type padded_out_bytes = roundUpTo(*out_bytes, sizeof(uint32_t));
  if (output + padded_out_bytes / sizeof(uint32_t) > output_limit) {
    return BlockIOStatus::out_of_bound;
  }

  for (int element_idx = threadIdx.x;
       element_idx < padded_out_bytes / sizeof(uint32_t);
       element_idx += blockDim.x) {
    output[element_idx] = source[element_idx];
  }

  return BlockIOStatus::success;
}

/**
 * Read a buffer with a single threadblock, optionally with bitunpacking.
 *
 * For the current implementation, this function is used to load compressed data
 * from the global memory to the shared memory.
 *
 * @param[in] input Compressed input buffer.
 * @param[in] in_byte Number of bytes of the compressed input.
 * @param[in] input_limit Pointer past the end of the input buffer. If this
 * function needs to load beyond this pointer, the function will return
 * `BlockIOStatus::out_of_bound`. This pointer must belong to the same array as
 * \p input, otherwise the behavior is undefined.
 * @param[out] output Pointer to the output buffer.
 * @param[out] out_num_elements Number of `data_type` elements written to the
 * output buffer. This argument should be unique to each thread.
 * @param[in] temp_storage Temporary storage to hold input data before
 * bitpacking. User needs to guarantee that this buffer has size at least
 * `in_byte` (rounded up to a multiple of 4) bytes.
 * @param[in] use_bp Whether bitpacking should be used.
 */
template <typename data_type, typename size_type, int threadblock_size>
__device__ BlockIOStatus block_read(
    const uint32_t* input,
    size_type in_byte,
    const uint32_t* input_limit,
    data_type* output,
    size_type* out_num_elements,
    uint32_t* temp_storage,
    bool use_bp)
{
  if (input_limit && input + roundUpDiv(in_byte, 4) > input_limit)
    return BlockIOStatus::out_of_bound;

  uint32_t* dest_ptr;
  if (use_bp) {
    dest_ptr = temp_storage;
  } else {
    dest_ptr = reinterpret_cast<uint32_t*>(output);
  }

  for (int element_idx = threadIdx.x; element_idx < roundUpDiv(in_byte, 4);
       element_idx += threadblock_size) {
    dest_ptr[element_idx] = input[element_idx];
  }
  __syncthreads();

  if (use_bp) {
    block_bitunpack<data_type, size_type>(
        temp_storage, output, out_num_elements);
  } else {
    if (out_num_elements != nullptr)
      *out_num_elements = in_byte / sizeof(data_type);
  }

  return BlockIOStatus::success;
}

/**
 * @brief Batched cascaded compression kernel.
 *
 * All cascaded compression layers are fused together in this single kernel.
 *
 * @tparam data_type Data type of each element.
 * @tparam threadblock_size Number of threads in a threadblock. This argument
 * must match the configuration specified when launching this kernel.
 * @tparam chunk_size Input size that is loaded into shared memory at a time.
 * This argument must be a multiple of the size of `data_type`.
 *
 * @param[in] batch_size Number of partitions to compress.
 * @param[in] uncompressed_data Array with size \p batch_size of pointers to
 * input uncompressed partitions.
 * @param[in] uncompressed_bytes Sizes of input uncompressed partitions in
 * bytes.
 * @param[out] compressed_data Array with size \p batch_size of output locations
 * of the compressed buffers. Each compressed buffer must start at a location
 * aligned with both 4B and the data type.
 * @param[out] compressed_bytes Number of bytes decompressed of all partitions.
 * @param[in] comp_opts Compression format used.
 */
template <
    typename data_type,
    typename size_type,
    int threadblock_size,
    int chunk_size = default_chunk_size>
__device__ void do_cascaded_compression_kernel(
    int batch_size,
    int batch_start,
    int batch_stride,
    const data_type* const* uncompressed_data,
    const size_type* uncompressed_bytes,
    void* const* compressed_data,
    size_type* compressed_bytes,
    nvcompBatchedCascadedOpts_t comp_opts)
{
  using run_type = uint16_t;
  constexpr int chunk_num_elements = chunk_size / sizeof(data_type);

  // We need to guarantee the `chunk_num_elements` is smaller than the limit of
  // uint16_t for two reasons:
  // 1. We use uint16_t to represent run counts.
  // 2. We use 16 bits to represent number of elements in the bitpacking layer.
  assert(chunk_num_elements < 65536);

  // `shared_element_storage_0` and `shared_element_storage_1` are shared memory
  // storage used for holding input and output of the current layer.
  // `shared_storage_type` is used to make sure the storage is both 4B and
  // data_type aligned.
  typedef larger_t<data_type, uint32_t> shared_storage_type;

  // Allocate `4 + sizeof(data_type)` in addition to `chunk_size` to accommodate
  // bitpacking metadata. The extra `sizeof(data_type)` is used for holding
  // frame of reference, while the extra 4B is used to hold the bitwidth and the
  // number of elements.
  constexpr size_t storage_num_elements = roundUpDiv(
      chunk_size + 4 + sizeof(data_type), sizeof(shared_storage_type));

  __shared__ shared_storage_type shared_element_storage_0[storage_num_elements];
  data_type* shared_element_buffer_0
      = reinterpret_cast<data_type*>(shared_element_storage_0);

  __shared__ shared_storage_type shared_element_storage_1[storage_num_elements];
  data_type* shared_element_buffer_1
      = reinterpret_cast<data_type*>(shared_element_storage_1);

  constexpr int shared_counts_storage_size
      = roundUpTo(chunk_num_elements * sizeof(run_type), 4);
  // Shared memory buffer used by RLE for holding run counts
  __shared__ uint32_t
      shared_count_buffer[shared_counts_storage_size / sizeof(uint32_t)];
  // Temporary storage used by RLE
  // Allocate extra 8B for holding bitpacking metadata. The metadata consists of
  // frame of reference (2B) and bit-width and number of elements (4B). So, in
  // total the metadata needs 6B. 8B is allocated for 4B alignment.
  __shared__ uint32_t shared_tmp_buffer
      [shared_counts_storage_size / sizeof(uint32_t) + 8 / sizeof(uint32_t)];

  // `chunk_metadata` is a shared-memory staging buffer for the metadata of the
  // current chunk before flushing it to global memory. Here we assume the chunk
  // metadata is at most 64B large.
  constexpr int max_chunk_metadata_size = 64;
  __shared__ uint32_t
      chunk_metadata[max_chunk_metadata_size / sizeof(uint32_t)];
  const int chunk_metadata_size = get_chunk_metadata_size<data_type>(
      comp_opts.num_RLEs, comp_opts.num_deltas);
  assert(chunk_metadata_size <= max_chunk_metadata_size);

  // Pointer to the delta section of chunk metadata in shared memory. Padding
  // will be added if necessary to make the pointer `data_type` aligned.
  // Explanation of the math here: from the start of a chunk metadata, we need
  // to skip the size of the chunk (4B), and (num_RLEs + 1) RLE offsets
  // (4B each) to get to the start of the delta header.
  data_type* const delta_header = roundUpToAlignment<data_type>(
      chunk_metadata + 1 + comp_opts.num_RLEs + 1);

  // Number of output elements for the RLE layer
  __shared__ size_type num_outputs;
  size_type out_bytes;

  for (int partition_idx = batch_start; partition_idx < batch_size;
       partition_idx += batch_stride) {
    const auto input_buffer = uncompressed_data[partition_idx];
    const auto input_bytes = uncompressed_bytes[partition_idx];
    assert(input_bytes <= UINT32_MAX);
    const size_type num_input_elements = input_bytes / sizeof(data_type);
    auto output_buffer = static_cast<uint32_t*>(compressed_data[partition_idx]);
    // `output_limit` points to the end of the output compressed buffer of the
    // current partition. The size of the compressed buffer should be at least
    // 8B larger than the input uncompressed buffer. It needs to be 8B larger
    // because in the fallback path, the compressed buffer still needs to store
    // the metadata. It is users responsibility to guarantee this requirement.
    uint32_t* output_limit
        = output_buffer + roundUpDiv(partition_metadata_size, sizeof(uint32_t))
          + roundUpDiv(input_bytes, sizeof(uint32_t));

    if (input_buffer == nullptr || input_bytes == 0) {
      if (threadIdx.x == 0)
        compressed_bytes[partition_idx] = 0;
      continue;
    }

    // Global flag on whether we will compress the current partition. If
    // compressed size is larger than the uncompressed size (i.e. compression
    // ratio < 1), we will use the fallback path of directly copying from the
    // input buffer to the compressed buffer, and set this flag to false.
    bool use_compression = true;

    if (comp_opts.num_RLEs == 0 && comp_opts.num_deltas == 0
        && comp_opts.use_bp == 0)
      use_compression = false;

    // Pointer to the first chunk of the current partition
    auto current_output_ptr
        = reinterpret_cast<uint32_t*>(roundUpToAlignment<data_type>(
            output_buffer
            + roundUpDiv(partition_metadata_size, sizeof(uint32_t))));

    const int num_chunks = roundUpDiv(num_input_elements, chunk_num_elements);

    for (int chunk_idx = 0; chunk_idx < num_chunks && use_compression;
         chunk_idx++) {
      // Save a pointer at the start of the chunk
      uint32_t* chunk_start_ptr = current_output_ptr;

      // Move current output pointer as the end of chunk metadata
      current_output_ptr += chunk_metadata_size / sizeof(uint32_t);

      auto input_buffer_current_chunk
          = input_buffer + chunk_num_elements * chunk_idx;
      size_type num_elements_current_chunk = min(
          num_input_elements - chunk_idx * chunk_num_elements,
          static_cast<size_type>(chunk_num_elements));

      // Threadblock collectively loads current chunk from input uncompressed
      // buffer to shared memory buffer
      for (int element_idx = threadIdx.x;
           element_idx < num_elements_current_chunk;
           element_idx += blockDim.x) {
        shared_element_buffer_0[element_idx]
            = input_buffer_current_chunk[element_idx];
      }
      __syncthreads();

      int rle_remaining = comp_opts.num_RLEs;
      int delta_remaining = comp_opts.num_deltas;

      data_type* shared_input_buffer = shared_element_buffer_0;
      data_type* shared_output_buffer = shared_element_buffer_1;

      for (int layer_idx = 0;
           layer_idx < max(comp_opts.num_RLEs, comp_opts.num_deltas);
           layer_idx++) {
        if (rle_remaining > 0) {
          // Run RLE
          block_rle_compress<data_type, size_type, run_type, threadblock_size>(
              shared_input_buffer,
              num_elements_current_chunk,
              shared_output_buffer,
              reinterpret_cast<run_type*>(shared_count_buffer),
              &num_outputs,
              reinterpret_cast<run_type*>(shared_tmp_buffer));
          __syncthreads();

          // Save run counts to the compressed buffer
          if (block_write<run_type, size_type, threadblock_size>(
                  reinterpret_cast<run_type*>(shared_count_buffer),
                  num_outputs,
                  current_output_ptr,
                  output_limit,
                  &out_bytes,
                  shared_tmp_buffer,
                  comp_opts.use_bp)
              != BlockIOStatus::success) {
            use_compression = false;
            goto afterlastchunk;
          }

          current_output_ptr += roundUpDiv(out_bytes, 4);

          // Store the size into chunk metadata
          if (threadIdx.x == 0) {
            chunk_metadata[comp_opts.num_RLEs - rle_remaining + 1] = out_bytes;
          }

          // Revert the role of input and ouput buffer
          auto temp_ptr = shared_output_buffer;
          shared_output_buffer = shared_input_buffer;
          shared_input_buffer = temp_ptr;

          num_elements_current_chunk = num_outputs;

          rle_remaining--;
        }

        if (delta_remaining > 0) {
          // Run Delta
          block_delta_compress<data_type, size_type>(
              shared_input_buffer,
              num_elements_current_chunk,
              shared_output_buffer);

          if (threadIdx.x == 0) {
            delta_header[comp_opts.num_deltas - delta_remaining]
                = shared_input_buffer[0];
          }

          // Revert the role of input and ouput buffer
          auto temp_ptr = shared_output_buffer;
          shared_output_buffer = shared_input_buffer;
          shared_input_buffer = temp_ptr;

          // Number of elements is decreased by 1 since the first element is
          // excluded for the subsequent operations.
          num_elements_current_chunk -= 1;

          delta_remaining--;
        }

        __syncthreads();
      }

      // Save final output to output buffer
      auto final_output_ptr = reinterpret_cast<uint32_t*>(
          roundUpToAlignment<data_type>(current_output_ptr));

      if (block_write<data_type, size_type, threadblock_size>(
              shared_input_buffer,
              num_elements_current_chunk,
              final_output_ptr,
              output_limit,
              &out_bytes,
              reinterpret_cast<uint32_t*>(shared_output_buffer),
              comp_opts.use_bp)
          != BlockIOStatus::success) {
        use_compression = false;
        break;
      }

      current_output_ptr = final_output_ptr + roundUpDiv(out_bytes, 4);
      current_output_ptr = reinterpret_cast<uint32_t*>(
          roundUpToAlignment<data_type>(current_output_ptr));

      // Flush chunk header from shared memory to output buffer
      if (threadIdx.x == 0) {
        const uint32_t chunk_output_size
            = reinterpret_cast<uintptr_t>(current_output_ptr)
              - reinterpret_cast<uintptr_t>(chunk_start_ptr);
        chunk_metadata[0] = chunk_output_size;
        chunk_metadata[comp_opts.num_RLEs + 1] = out_bytes;

        for (int idx = 0; idx < chunk_metadata_size / 4; idx++) {
          chunk_start_ptr[idx] = chunk_metadata[idx];
        }
      }

      __syncthreads();
    }

  afterlastchunk:
    if (!use_compression) {
      // Compressed size is larger than uncompressed size, so we fallback to
      // directly copy input array to output
      data_type* direct_output_buffer
          = roundUpToAlignment<data_type>(output_buffer + 2);
      for (int element_idx = threadIdx.x; element_idx < num_input_elements;
           element_idx += blockDim.x) {
        direct_output_buffer[element_idx] = input_buffer[element_idx];
      }
    }

    // Save the metadata of the current partition
    if (threadIdx.x == 0) {
      auto partition_metadata_ptr = reinterpret_cast<uint8_t*>(output_buffer);

      partition_metadata_ptr[3] = (d_TypeOf<data_type>());

      if (use_compression) {
        partition_metadata_ptr[0] = comp_opts.num_RLEs;
        partition_metadata_ptr[1] = comp_opts.num_deltas;
        partition_metadata_ptr[2] = comp_opts.use_bp;
        compressed_bytes[partition_idx]
            = reinterpret_cast<uintptr_t>(current_output_ptr)
              - reinterpret_cast<uintptr_t>(output_buffer);
      } else {
        // If compression is not used, we set all of num_RLEs, num_deltas and
        // bit-packing fields to 0
        partition_metadata_ptr[0] = 0;
        partition_metadata_ptr[1] = 0;
        partition_metadata_ptr[2] = 0;
        compressed_bytes[partition_idx]
            = roundUpTo(partition_metadata_size, sizeof(data_type))
              + roundUpTo(num_input_elements * sizeof(data_type), 4);
      }
      output_buffer[1]
          = static_cast<uint32_t>(num_input_elements * sizeof(data_type));
    }
  }
}

/**
 * @brief Helper function to determine the shared memory requirement
 * based on the chunk size and datatype...
 * @tparam chunk_size Size of the chunk in bytes
 * @tparam width Width of the datatype being used (in bytes)
 * @tparam storage_width Width in bytes of the storage type used
 * to compute offset.  This should be a minimum of 4 bytes, or 8
 * bytes if the width is also 8 bytes.
 */
template <int chunk_size, int width, int storage_width>
__device__ constexpr int compute_smem_size()
{
  constexpr int storage_num_elts
      = roundUpDiv(chunk_size + 4 + width, storage_width);
  constexpr int tot_elt_storage = 2 * (storage_num_elts * storage_width);
  constexpr int run_width = 2;
  constexpr int chunk_num_elements = chunk_size / width;
  constexpr int tot_count_bytes = 2 * (chunk_num_elements * run_width);

  return 64 + tot_elt_storage + tot_count_bytes + (4 * 8);
}

/**
 * @brief Device function to perform batched cascaded decompression for
 * a given datatype
 *
 * @tparam data_type Data type of each uncompressed element.
 * @tparam threadblock_size Number of threads in a threadblock. This argument
 * must match the configuration specified when launching this kernel.
 * @tparam chunk_size Number of bytes for each uncompressed chunk to fit inside
 * shared memory. This argument must match the chunk size specified during
 * compression.
 *
 * @param[in] batch_size Number of partitions to decompress.
 * @param[in] compressed_data Array of size \p batch_size where each element is
 * a pointer to the compressed data of a partition.
 * @param[in] compressed_bytes Sizes of the compressed buffers corresponding to
 * \p compressed_data.
 * @param[out] decompressed_data Pointers to the output decompressed buffers.
 * @param[in] decompressed_buffer_bytes Sizes of the decompressed buffers in
 * bytes.
 * @param[out] actual_decompressed_bytes Actual number of bytes decompressed for
 * all partitions.
 * @param[in] shmem Allocated shared memory buffer for use in decompression.
 * @param[out] statuses Whether the compressions are successful.
 */
template <
    typename data_type,
    typename size_type,
    int threadblock_size,
    int chunk_size = default_chunk_size>
__device__ void cascaded_decompression_fcn(
    int batch_size,
    int batch_start,
    int batch_stride,
    const void* const* compressed_data,
    const size_type* compressed_bytes,
    void* const* decompressed_data,
    const size_type* decompressed_buffer_bytes,
    size_type* actual_decompressed_bytes,
    void* shmem,
    nvcompStatus_t* statuses)
{

  using run_type = uint16_t;
  constexpr int chunk_num_elements = chunk_size / sizeof(data_type);

  // Shared memory storage for chunk metadata. Chunk metadata consists of
  // 1. size of the chunk (4B)
  // 2. The sizes in bytes of all RLE count arrays (4B per RLE layer)
  // 3. The size in byte of the final array (4B)
  // 4. The first elements of each delta layer (data type size per Delta layer)
  // We assume the data type is at most 8B large, so we use uint64_t here to
  // make sure the storage starts at an 8-byte alignment location.
  // Here we assume the metadata is at most 64B.
  uint64_t* chunk_metadata_storage = static_cast<uint64_t*>(shmem);
  auto chunk_metadata = reinterpret_cast<uint32_t*>(chunk_metadata_storage);
  shmem = static_cast<void*>((static_cast<uint8_t*>(shmem)) + 64);

  // `shared_element_storage_0` and `shared_element_storage_1` are shared memory
  // storage used for the data arrays of the input and the output of the current
  // layer. `shared_storage_type` is used to make sure the storage is both 4B
  // and data_type aligned.
  typedef larger_t<data_type, uint32_t> shared_storage_type;

  // Allocate `4 + sizeof(data_type)` in addition to `chunk_size` to accommodate
  // bitpacking metadata.
  constexpr size_t storage_num_elements = roundUpDiv(
      chunk_size + 4 + sizeof(data_type), sizeof(shared_storage_type));

  shared_storage_type* shared_element_storage_0
      = static_cast<shared_storage_type*>(shmem);
  shmem = static_cast<void*>(
      static_cast<shared_storage_type*>(shmem) + storage_num_elements);

  data_type* shared_element_buffer_0
      = reinterpret_cast<data_type*>(shared_element_storage_0);

  shared_storage_type* shared_element_storage_1
      = static_cast<shared_storage_type*>(shmem);
  shmem = static_cast<void*>(
      static_cast<shared_storage_type*>(shmem) + storage_num_elements);

  data_type* shared_element_buffer_1
      = reinterpret_cast<data_type*>(shared_element_storage_1);

  // `count_array` is the shared memory storage for RLE count arrays.
  // `temp_count_array` is used for bit-unpacking when loading RLE counts. Since
  // run_type should be no larger than 4B, we use `uint32_t` to guarantee 4B
  // aligned (which implies run_type aligned as well).
  uint32_t* count_array = static_cast<uint32_t*>(shmem);
  shmem = static_cast<void*>(
      static_cast<uint8_t*>(shmem) + (chunk_num_elements * sizeof(run_type)));

  uint32_t* temp_count_array = static_cast<uint32_t*>(shmem);
  shmem = static_cast<void*>(
      static_cast<uint8_t*>(shmem) + (chunk_num_elements * sizeof(run_type)));

  // RLE offsets
  uint32_t* rle_offsets = static_cast<uint32_t*>(shmem);

  for (int partition_idx = batch_start; partition_idx < batch_size;
       partition_idx += batch_stride) {
    if (compressed_data[partition_idx] == nullptr
        || compressed_bytes[partition_idx] < partition_metadata_size) {
      // Compressed buffer should at least have enough space for partition
      // metadata.
      if (threadIdx.x == 0) {
        statuses[partition_idx] = nvcompErrorCannotDecompress;
        actual_decompressed_bytes[partition_idx] = 0;
      }
      continue;
    }

    const uint32_t* partition_start_ptr
        = static_cast<const uint32_t*>(compressed_data[partition_idx]);
    const uint32_t* partition_end_ptr
        = partition_start_ptr + compressed_bytes[partition_idx] / 4;
    data_type* decompressed_ptr
        = ((data_type* const*)decompressed_data)[partition_idx];
    size_type decompressed_num_elements = 0;

    const uint8_t* partition_metadata_ptr
        = reinterpret_cast<const uint8_t*>(partition_start_ptr);
    int num_RLEs = partition_metadata_ptr[0];
    int num_deltas = partition_metadata_ptr[1];
    int bitpacking = partition_metadata_ptr[2];

    // Max number of RLE layers is 7
    assert(num_RLEs <= 7);

    const uint32_t num_uncompressed_elements
        = partition_start_ptr[1] / sizeof(data_type);

    if (decompressed_buffer_bytes[partition_idx]
        < sizeof(data_type) * num_uncompressed_elements) {
      // The output buffer is not large enough to hold all uncompressed
      // elements, so we report failure.
      if (threadIdx.x == 0) {
        actual_decompressed_bytes[partition_idx] = 0;
        statuses[partition_idx] = nvcompErrorCannotDecompress;
      }
      continue;
    }

    if (num_RLEs == 0 && num_deltas == 0 && bitpacking == 0) {
      // No compression is used. This could be the result of user specification
      // or compression ratio less than 1. In this case, we copy the compressed
      // data directly to output buffer.

      if (compressed_bytes[partition_idx]
          < roundUpTo(partition_metadata_size, sizeof(data_type))
                + sizeof(data_type) * num_uncompressed_elements) {
        // Compressed buffer does not have enough space to hold all uncompressed
        // data, so we report failure.
        if (threadIdx.x == 0) {
          actual_decompressed_bytes[partition_idx] = 0;
          statuses[partition_idx] = nvcompErrorCannotDecompress;
        }
      } else {
        const data_type* direct_compressed_buffer
            = roundUpToAlignment<data_type>(partition_start_ptr + 2);
        for (int element_idx = threadIdx.x;
             element_idx < num_uncompressed_elements;
             element_idx += blockDim.x) {
          decompressed_ptr[element_idx] = direct_compressed_buffer[element_idx];
        }
        if (threadIdx.x == 0) {
          actual_decompressed_bytes[partition_idx]
              = sizeof(data_type) * num_uncompressed_elements;
          statuses[partition_idx] = nvcompSuccess;
        }
      }
      continue;
    }

    // Start location of the first elements of delta layers in shared memory
    // storage of chunk metadata.
    const data_type* const delta_header
        = roundUpToAlignment<data_type>(chunk_metadata + 1 + num_RLEs + 1);

    // `chunk_ptr` points to the start location of the current chunk in global
    // memory. Here we initialize it to the start location of the first chunk.
    const uint32_t* chunk_ptr = reinterpret_cast<const uint32_t*>(
        roundUpToAlignment<data_type>(partition_start_ptr + 2));

    bool is_decompression_successful = true;

    while (chunk_ptr < partition_end_ptr) {
      // Load chunk metadata to the shared memory storage
      const int chunk_metadata_size
          = get_chunk_metadata_size<data_type>(num_RLEs, num_deltas);
      if (chunk_ptr + chunk_metadata_size / 4 > partition_end_ptr) {
        // Compressed buffer does not have enough space for the current chunk
        // metadata. This means the compressed data is corrupt, so we report
        // failure.
        is_decompression_successful = false;
        break;
      }
      for (int element_idx = threadIdx.x; element_idx < chunk_metadata_size / 4;
           element_idx += threadblock_size) {
        chunk_metadata[element_idx] = chunk_ptr[element_idx];
      }
      __syncthreads();

      // Chunk size is the first element of metadata
      const int compressed_chunk_size = chunk_metadata[0];

      // Calculate RLE count array / final array location offsets from array
      // sizes. The calculation is a prefix sum on array sizes with alignment
      // paddings.
      if (threadIdx.x == 0) {
        rle_offsets[0] = 0;
        if (num_RLEs > 0) {
          for (int rle_idx = 0; rle_idx < num_RLEs - 1; rle_idx++) {
            // The count arrays start at alignment of 4.
            rle_offsets[rle_idx + 1] = roundUpTo(
                rle_offsets[rle_idx] + chunk_metadata[rle_idx + 1], 4);
          }
          // The final array start at location both aligned with data_type and
          // aligned with 4B.
          rle_offsets[num_RLEs] = roundUpTo(
              rle_offsets[num_RLEs - 1] + chunk_metadata[num_RLEs],
              max(static_cast<size_t>(4), sizeof(data_type)));
        }
      }
      __syncthreads();

      data_type* shared_input_buffer = shared_element_buffer_0;
      data_type* shared_output_buffer = shared_element_buffer_1;
      const uint32_t* rle0_ptr = chunk_ptr + chunk_metadata_size / 4;

      // Load array after final layer to shared memory
      const uint32_t* final_array_ptr = rle0_ptr + rle_offsets[num_RLEs] / 4;
      size_type num_elements;
      if (block_read<data_type, size_type, threadblock_size>(
              final_array_ptr,
              chunk_metadata[1 + num_RLEs],
              partition_end_ptr,
              shared_input_buffer,
              &num_elements,
              reinterpret_cast<uint32_t*>(shared_output_buffer),
              bitpacking)
          != BlockIOStatus::success) {
        is_decompression_successful = false;
        break;
      }
      __syncthreads();

      int rle_remaining = num_RLEs;
      int delta_remaining = num_deltas;

      for (int layer_idx = 0; layer_idx < max(num_RLEs, num_deltas);
           layer_idx++) {
        if (delta_remaining > 0 && delta_remaining >= rle_remaining) {
          // Decompress the delta layer
          block_delta_decompress<data_type, size_type, threadblock_size>(
              shared_input_buffer,
              delta_header[delta_remaining - 1],
              num_elements,
              shared_output_buffer);
          __syncthreads();

          // Revert the role of input and ouput buffer
          auto temp_ptr = shared_output_buffer;
          shared_output_buffer = shared_input_buffer;
          shared_input_buffer = temp_ptr;

          // Decompressing delta layer adds one extra element (the first
          // element).
          num_elements++;
          delta_remaining--;
        }

        if (rle_remaining > 0 && rle_remaining >= delta_remaining) {
          // Load the count array from global memory to shared memory
          if (block_read<run_type, size_type, threadblock_size>(
                  rle0_ptr + rle_offsets[rle_remaining - 1] / 4,
                  chunk_metadata[rle_remaining],
                  partition_end_ptr,
                  reinterpret_cast<run_type*>(count_array),
                  nullptr,
                  temp_count_array,
                  bitpacking)
              != BlockIOStatus::success) {
            is_decompression_successful = false;
            goto afterlastchunk;
          }
          __syncthreads();

          // Decompress the RLE layer
          size_type output_num_elements;
          block_rle_decompress<
              data_type,
              size_type,
              run_type,
              threadblock_size>(
              shared_input_buffer,
              reinterpret_cast<run_type*>(count_array),
              num_elements,
              shared_output_buffer,
              &output_num_elements);
          num_elements = output_num_elements;

          // Revert the role of input and ouput buffer
          auto temp_ptr = shared_output_buffer;
          shared_output_buffer = shared_input_buffer;
          shared_input_buffer = temp_ptr;

          rle_remaining--;
        }
      }

      // Save the current chunk to the output buffer

      if (decompressed_num_elements + num_elements
          > num_uncompressed_elements) {
        // If the number of decompressed elements after the current chunk is
        // more than the total number of uncompressed elements, the compressed
        // data must be corrupted, so we report failure.
        is_decompression_successful = false;
        break;
      }

      for (int element_idx = threadIdx.x; element_idx < num_elements;
           element_idx += threadblock_size) {
        decompressed_ptr[element_idx] = shared_input_buffer[element_idx];
      }
      decompressed_ptr += num_elements;
      decompressed_num_elements += num_elements;

      // Update `chunk_ptr` to the start location of the next chunk
      chunk_ptr = reinterpret_cast<const uint32_t*>(
          roundUpToAlignment<data_type>(chunk_ptr + compressed_chunk_size / 4));
    }

  afterlastchunk:
    if (num_uncompressed_elements != decompressed_num_elements) {
      // The number of decompressed elements does not match the uncompressed
      // element stored in the compressed buffer. This means the compressed
      // data is corrupted, so we report failure.
      is_decompression_successful = false;
    }

    if (threadIdx.x == 0) {
      if (is_decompression_successful) {
        actual_decompressed_bytes[partition_idx]
            = decompressed_num_elements * sizeof(data_type);
        statuses[partition_idx] = nvcompSuccess;
      } else {
        actual_decompressed_bytes[partition_idx] = 0;
        statuses[partition_idx] = nvcompErrorCannotDecompress;
      }
    }
  }
}



} // namespace nvcomp

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


namespace nvcomp {

/**
 * @brief Compress a batch of memory locations.
 *
 * @param decomp_data The batch items to compress.
 * @param decomp_sizes The size of each batch item to compress.
 * @param batch_size The number of items in the batch.
 * @param max_chunk_size The number of uncompressed bytes per LZ4 compressed
 * chunk.
 * @param temp_data The temporary memory to use.
 * @param temp_bytes The size of the temporary memory.
 * @param comp_data The output location of each batch item.
 * @param comp_prefixes The size of each compressed chunk (output).
 * @param comp_prefix_offset_host
 * @param stream The stream to operate on.
 */
void lz4CompressBatch(
    const uint8_t* const* decomp_data_device,
    const size_t* decomp_sizes_device,
    const size_t* decomp_sizes_host,
    const size_t batch_size,
    const size_t max_chunk_size,
    uint8_t* temp_data,
    size_t temp_bytes,
    uint8_t* const* comp_data_device,
    size_t* const* comp_prefixes_device,
    const size_t* const comp_prefix_offset_device,
    cudaStream_t stream);

void lz4DecompressBatches(
    void* const temp_space,
    const size_t temp_size,
    void* const* decompData,
    const uint8_t* const* compData,
    int batch_size,
    const size_t** compPrefix,
    int chunk_size,
    int* chunks_in_item,
    cudaStream_t stream);

size_t lz4ComputeChunksInBatch(
    const size_t* const decomp_data_size,
    const size_t batch_size,
    const size_t chunk_size);

size_t lz4CompressComputeTempSize(
    const size_t max_chunks_in_batch, const size_t chunk_size);

size_t lz4DecompressComputeTempSize(
    const size_t max_chunks_in_batch, const size_t chunk_size);

size_t lz4ComputeMaxSize(const size_t chunk_size);

size_t lz4MinChunkSize();

size_t lz4MaxChunkSize();
}


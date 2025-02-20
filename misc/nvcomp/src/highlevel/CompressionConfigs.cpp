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

#include "CompressionConfigs.hpp"

namespace nvcomp {

CompressionConfig::CompressionConfigImpl::CompressionConfigImpl(PinnedPtrPool<nvcompStatus_t>& pool)
  : status(pool.allocate())
{
  *get_status() = nvcompSuccess;
}

nvcompStatus_t* CompressionConfig::CompressionConfigImpl::get_status() const {
  return status->get_ptr();
}

CompressionConfig::CompressionConfig(PinnedPtrPool<nvcompStatus_t>& pool, size_t uncompressed_buffer_size)
  : impl(std::make_shared<CompressionConfig::CompressionConfigImpl>(pool)),
    uncompressed_buffer_size(uncompressed_buffer_size),
    max_compressed_buffer_size(0),
    num_chunks(0)
{}

nvcompStatus_t* CompressionConfig::get_status() const {
  return impl->get_status();
}

CompressionConfig::~CompressionConfig() {}

CompressionConfig::CompressionConfig(CompressionConfig&& other)
  : impl(std::move(other.impl)),
    uncompressed_buffer_size(other.uncompressed_buffer_size),
    max_compressed_buffer_size(other.max_compressed_buffer_size),
    num_chunks(other.num_chunks)
{}

CompressionConfig::CompressionConfig(const CompressionConfig& other)
  : impl(other.impl),
    uncompressed_buffer_size(other.uncompressed_buffer_size),
    max_compressed_buffer_size(other.max_compressed_buffer_size),
    num_chunks(other.num_chunks)
{}

CompressionConfig& CompressionConfig::operator=(const CompressionConfig& other) 
{
  impl = other.impl;
  uncompressed_buffer_size = other.uncompressed_buffer_size;
  max_compressed_buffer_size = other.max_compressed_buffer_size;
  num_chunks = other.num_chunks;
  return *this;
}

CompressionConfig& CompressionConfig::operator=(CompressionConfig&& other) 
{
  impl = std::move(other.impl);
  uncompressed_buffer_size = other.uncompressed_buffer_size;
  max_compressed_buffer_size = other.max_compressed_buffer_size;
  num_chunks = other.num_chunks;
  return *this;
}

/**
 * @brief Construct the config given an nvcompStatus_t memory pool
 */
DecompressionConfig::DecompressionConfigImpl::DecompressionConfigImpl(PinnedPtrPool<nvcompStatus_t>& pool)
  : status(pool.allocate()),
    decomp_data_size(),
    num_chunks()
{
  *get_status() = nvcompSuccess;
}

/**
 * @brief Get the raw nvcompStatus_t*
 */
nvcompStatus_t* DecompressionConfig::DecompressionConfigImpl::get_status() const {
  return status->get_ptr();
}

DecompressionConfig::DecompressionConfig(PinnedPtrPool<nvcompStatus_t>& pool)
  : impl(std::make_shared<DecompressionConfig::DecompressionConfigImpl>(pool)),
    decomp_data_size(0),
    num_chunks(0)
{}

nvcompStatus_t* DecompressionConfig::get_status() const {
  return impl->get_status();
}

DecompressionConfig::~DecompressionConfig() {}

DecompressionConfig::DecompressionConfig(DecompressionConfig&& other)
  : impl(std::move(other.impl)),
    decomp_data_size(other.decomp_data_size),
    num_chunks(other.num_chunks)
{}

DecompressionConfig& DecompressionConfig::operator=(const DecompressionConfig& other) 
{
  impl = other.impl;
  decomp_data_size = other.decomp_data_size;
  num_chunks = other.num_chunks;
  return *this;
}

DecompressionConfig& DecompressionConfig::operator=(DecompressionConfig&& other) 
{
  impl = std::move(other.impl);
  decomp_data_size = other.decomp_data_size;
  num_chunks = other.num_chunks;
  return *this;
}

DecompressionConfig::DecompressionConfig(const DecompressionConfig& other)
  : impl(other.impl),
    decomp_data_size(other.decomp_data_size),
    num_chunks(other.num_chunks)
{}

} // namespace nvcomp
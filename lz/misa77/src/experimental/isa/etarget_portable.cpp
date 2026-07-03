// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

// Built on every target

#include "experimental/ecompress_dispatch.h"
#include "experimental/ecompress_impl.h"
#include "isa/lib_portable.h"

#include <cstdint>

namespace misa77
{
    namespace experimental
    {
        uint64_t compress_tuned_portable(const uint8_t* __restrict src,
                                         uint64_t src_size,
                                         uint8_t* __restrict dst,
                                         uint64_t dst_cap,
                                         const param& given)
        {
            return compress_tuned_impl<lib_portable>(src, src_size, dst, dst_cap, given);
        }
    } // namespace experimental
} // namespace misa77

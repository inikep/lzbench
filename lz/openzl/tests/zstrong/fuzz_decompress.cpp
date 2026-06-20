// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <algorithm>
#include <memory>

#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/common/logging.h"
#include "openzl/zl_decompress.h"
#include "openzl/zl_errors.h"

// We're using a raw LLVM fuzzer in this case so that we can
// pass a seed corpus that is just a bunch of zstrong compressed
// data, without having to worry about converting the data to
// seeds.
extern "C" int LLVMFuzzerTestOneInput(void const* data, size_t size)
{
    ZL_g_logLevel            = ZL_LOG_LVL_ALWAYS;
    ZL_Report const dstSizeR = ZL_getDecompressedSize(data, size);
    if (ZL_isError(dstSizeR))
        return 0;
    size_t const dstSize             = ZL_validResult(dstSizeR);
    size_t const maxDstSize          = std::min<size_t>(10 << 20, size * 100);
    size_t const dstCapacity         = std::min<size_t>(dstSize, maxDstSize);
    auto dst                         = std::make_unique<uint8_t[]>(dstCapacity);
    ZL_Report const reportHeaderSize = ZL_getHeaderSize(data, size);
    ZL_DCtx* dctx                    = ZL_DCtx_create();
    ZL_ASSERT_NN(dctx);
    ZL_Report const report =
            ZL_DCtx_decompress(dctx, dst.get(), dstCapacity, data, size);
    if (ZL_isError(reportHeaderSize) && !ZL_isError(report)) {
        // We cannot succeed in decoding while failing in decoding header for
        // header size
        ZL_REQUIRE_SUCCESS(reportHeaderSize);
    }
    if (ZL_isError(report) && report._code == ZL_ErrorCode_logicError) {
        ZL_REQUIRE_SUCCESS(report);
    }
    if (!ZL_isError(report)) {
        const auto decompressedSize = ZL_getDecompressedSize(data, size);
        ZL_REQUIRE_SUCCESS(decompressedSize);
        ZL_REQUIRE_EQ(ZL_validResult(report), ZL_validResult(decompressedSize));
    }
    ZL_DCtx_free(dctx);
    return 0;
}

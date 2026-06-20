// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/cpp/CustomDecoder.hpp"
#include "openzl/cpp/CustomEncoder.hpp"

namespace openzl::tests {

class PlusOneEncoder : public CustomEncoder {
   public:
    MultiInputCodecDescription multiInputDescription() const override;

    void encode(EncoderState& encoderState) const override;
};

class PlusOneDecoder : public CustomDecoder {
   public:
    MultiInputCodecDescription multiInputDescription() const override;

    void decode(DecoderState& decoderState) const override;
};
} // namespace openzl::tests

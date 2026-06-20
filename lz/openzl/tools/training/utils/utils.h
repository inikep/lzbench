// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/cpp/CCtx.hpp"
#include "openzl/cpp/Compressor.hpp"
#include "tools/io/InputSet.h"

namespace openzl::training {

/**
 * @brief Create a CCtx for training the compressor. The cctx is configured
 * so that if training is called multiple times, the parameters will not be
 * reset.
 */
CCtx refCCtxForTraining(const Compressor& compressor);

class MultiInput {
   public:
    explicit MultiInput(std::vector<Input>&& inputs = {})
            : inputs_(std::make_shared<std::vector<Input>>(std::move(inputs)))
    {
    }

    std::vector<Input>& operator*()
    {
        return *inputs_;
    }

    const std::vector<Input>& operator*() const
    {
        return *inputs_;
    }

    std::vector<Input>* operator->()
    {
        return inputs_.get();
    }

    const std::vector<Input>* operator->() const
    {
        return inputs_.get();
    }

    // Adds input while not owning the buffer the input references
    void add(Input&& input)
    {
        inputs_->emplace_back(std::move(input));
    }

    // Adds input and ensures that the buffer the input references which is
    // owned by io::Input stays around by adding a reference to the shared ptr
    void add(std::shared_ptr<tools::io::Input> input)
    {
        inputSources_.emplace_back(input);
        add(openzl::Input::refSerial(input->contents()));
    }

   private:
    std::vector<std::shared_ptr<tools::io::Input>> inputSources_;
    std::shared_ptr<std::vector<Input>> inputs_;
};

/**
 * @brief Convert a set of inputs to a vector of MultiInputs. It is assumed that
 * each input is serial in @p inputs.
 */
std::vector<MultiInput> inputSetToMultiInputs(tools::io::InputSet& inputs);

} // namespace openzl::training

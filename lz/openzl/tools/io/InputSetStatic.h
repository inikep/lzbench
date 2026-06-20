// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <vector>

#include "tools/io/InputSet.h"

namespace openzl::tools::io {

class InputSetStatic : public InputSet {
   private:
    class IteratorStateStatic;

   public:
    explicit InputSetStatic(std::vector<std::shared_ptr<Input>> inputs);

    static InputSetStatic from_input_set(InputSet& input_set);

    size_t size() const;

    const std::shared_ptr<Input>& operator[](size_t idx) const;

   private:
    std::unique_ptr<IteratorState> begin_state() const override;

    const std::vector<std::shared_ptr<Input>> inputs_;
};

} // namespace openzl::tools::io

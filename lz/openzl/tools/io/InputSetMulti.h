// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <vector>

#include "tools/io/InputSet.h"

namespace openzl::tools::io {

class InputSetMulti : public InputSet {
   private:
    class IteratorStateMulti;

   public:
    explicit InputSetMulti(std::vector<std::unique_ptr<InputSet>> input_sets);

    const InputSet& operator[](size_t idx) const;

   private:
    std::unique_ptr<IteratorState> begin_state() const override;

    const std::vector<std::unique_ptr<InputSet>> input_sets_;
};

} // namespace openzl::tools::io

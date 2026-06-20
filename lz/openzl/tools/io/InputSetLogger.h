// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "tools/io/InputSet.h"

namespace openzl::tools::io {

/// Helper/debug class for inspecting input set discovery / traversal.
class InputSetLogger : public InputSet {
   private:
    class IteratorStateLogger;

   public:
    InputSetLogger(std::unique_ptr<InputSet> input_set, int verbosity = 1);

   private:
    std::unique_ptr<IteratorState> begin_state() const override;

    const std::unique_ptr<InputSet> input_set_;
    const int verbosity_;
};

} // namespace openzl::tools::io

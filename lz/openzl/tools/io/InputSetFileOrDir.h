// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/cpp/poly/Optional.hpp"

#include "tools/io/InputSet.h"
#include "tools/io/InputSetDir.h"

namespace openzl::tools::io {

/// Accepts a single path and either returns a single input representing that
/// file, if it's a file, or if it's a directory, delegates to @ref InputSetDir.
class InputSetFileOrDir : public InputSetDir {
   private:
    class IteratorStateSingleFile;

   public:
    explicit InputSetFileOrDir(std::string path, bool recursive);

   private:
    std::unique_ptr<IteratorState> begin_state() const override;

    const bool is_dir_;
};

} // namespace openzl::tools::io

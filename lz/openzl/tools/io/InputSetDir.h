// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>

#include "tools/io/InputSet.h"

namespace openzl::tools::io {

class InputSetFileOrDir;

/// Non-recursive traversal of regular files in the given directory.
class InputSetDir : public InputSet {
   private:
    template <typename IterT>
    class IteratorStateDir;

   public:
    explicit InputSetDir(std::string path, bool recursive);

   protected:
    std::unique_ptr<IteratorState> begin_state() const override;

    const std::string path_;
    const bool recursive_;
};

} // namespace openzl::tools::io

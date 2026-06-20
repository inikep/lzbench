// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>

#include "tools/io/Input.h"

namespace openzl::tools::io {

/**
 * Input backed by a file.
 */
class InputFile : public Input {
   public:
    explicit InputFile(std::string filename);

    poly::string_view name() const override;

    poly::optional<size_t> size() override;

    poly::string_view contents() override;

   private:
    void read();

    std::string filename_;
    poly::optional<std::string> contents_;
};

} // namespace openzl::tools::io

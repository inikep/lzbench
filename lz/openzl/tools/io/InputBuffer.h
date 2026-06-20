// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <stddef.h>
#include <string>

#include "tools/io/Input.h"

namespace openzl::tools::io {

/**
 * Input backed by an in-memory buffer.
 */
class InputBuffer : public Input {
   public:
    explicit InputBuffer(std::string contents, std::string name = "[buffer]")
            : name_(std::move(name)), contents_(std::move(contents))
    {
    }

    poly::string_view name() const override
    {
        return name_;
    }

    poly::optional<size_t> size() override
    {
        return contents_.size();
    }

    poly::string_view contents() override
    {
        return contents_;
    }

   private:
    std::string name_;
    std::string contents_;
};

} // namespace openzl::tools::io

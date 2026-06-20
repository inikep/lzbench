// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <sstream>
#include <string>

#include "tools/io/Input.h"
#include "tools/io/InputBuffer.h"
#include "tools/io/Output.h"

namespace openzl::tools::io {

/**
 * Output backed by an in-memory buffer.
 */
class OutputBuffer : public Output {
   public:
    explicit OutputBuffer(std::ostringstream& os, std::string name = "[buffer]")
            : os_(os), name_(std::move(name))
    {
    }

    poly::string_view name() const override
    {
        return name_;
    }

    void write(poly::string_view contents) override
    {
        os_ << contents;
    }

    std::ostream& get_ostream() override
    {
        return os_;
    }

    std::unique_ptr<Input> to_input() const
    {
        return std::make_unique<InputBuffer>(os_.str(), name_);
    }

   private:
    std::ostringstream& os_;
    std::string name_;
};

} // namespace openzl::tools::io

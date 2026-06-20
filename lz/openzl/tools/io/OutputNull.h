// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <fstream>

#include "tools/io/Output.h"

namespace openzl::tools::io {

/**
 * No-op output.
 */
class OutputNull : public Output {
   public:
    poly::string_view name() const override;

    void open() override {}

    void close() override {}

    void write(poly::string_view) override {}

    std::ostream& get_ostream() override;
};

} // namespace openzl::tools::io

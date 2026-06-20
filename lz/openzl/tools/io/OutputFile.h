// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <fstream>
#include <string>

#include "openzl/cpp/poly/Optional.hpp"

#include "tools/io/Output.h"

namespace openzl::tools::io {

/**
 * Output backed by a file.
 */
class OutputFile : public Output {
   public:
    explicit OutputFile(std::string filename);

    poly::string_view name() const override;

    void open() override;

    void close() override;

    void write(poly::string_view contents) override;

    std::ostream& get_ostream() override;

   private:
    std::string filename_;
    poly::optional<std::ofstream> os_;
};

} // namespace openzl::tools::io

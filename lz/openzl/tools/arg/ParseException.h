// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <stdexcept>

namespace openzl::arg {

class ParseException : public std::runtime_error {
   public:
    explicit ParseException(const std::string& msg) : std::runtime_error(msg) {}
};

} // namespace openzl::arg

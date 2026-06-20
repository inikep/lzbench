// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>

namespace openzl::arg {

struct Command {
    int cmd; // an int-valued identifier for this command
    std::string name;
    char shortName; // 0 if no short name
};

} // namespace openzl::arg

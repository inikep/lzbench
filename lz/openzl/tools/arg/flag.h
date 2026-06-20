// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>

namespace openzl::arg {

struct Flag {
    std::string name;
    char shortName; // 0 if no short name
    bool immediate; // whether this is meant to print something out and
                    // immediately exit the CLI, e.g. --help
    bool hasVal;    // true if the option takes an argument
    std::string help;
};

} // namespace openzl::arg

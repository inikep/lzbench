// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "tools/arg/arg_parser.h"

namespace openzl::cli {

// these should remain stable-ish, as they're passed as ints to the ArgParser
enum Cmd : int {
    UNSPECIFIED   = arg::ArgParser::CMD_UNSPECIFIED,
    COMPRESS      = 1,
    DECOMPRESS    = 2,
    TRAIN         = 3,
    BENCHMARK     = 4,
    INSPECT       = 5,
    LIST_PROFILES = 6,
};

} // namespace openzl::cli

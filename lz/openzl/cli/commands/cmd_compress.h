// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cli/args/CompressArgs.h"

namespace openzl::cli {

/// @return a return code for the CLI to return to the shell
int cmdCompress(CompressArgs args);

} // namespace openzl::cli

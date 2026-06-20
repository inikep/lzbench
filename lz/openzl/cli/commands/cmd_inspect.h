// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cli/args/InspectArgs.h"

namespace openzl::cli {

/// @return a return code for the CLI to return to the shell
int cmdInspect(const InspectArgs& args);

} // namespace openzl::cli

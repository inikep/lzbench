// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cli/args/GlobalArgs.h"

namespace openzl {
namespace cli {

class ListProfilesArgs : GlobalArgs {
   public:
    static void addArgs(arg::ArgParser& parser)
    {
        // Add the command
        parser.addCommand(cmd(), "list-profiles", 'l');
    }

    explicit ListProfilesArgs(const arg::ParsedArgs& parsed)
            : GlobalArgs(parsed)
    {
    }

    static Cmd cmd()
    {
        return Cmd::LIST_PROFILES;
    }
};

} // namespace cli
} // namespace openzl

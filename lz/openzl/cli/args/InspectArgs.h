// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include "tools/io/InputFile.h"
#include "tools/io/OutputFile.h"

#include "cli/args/GlobalArgs.h"

namespace openzl::cli {

class InspectArgs : GlobalArgs {
   public:
    static void addArgs(arg::ArgParser& parser)
    {
        // Add the command
        parser.addCommand(cmd(), "inspect", 'i');

        // Add the args
        parser.addCommandFlag(cmd(), kOutput, 'o', true, "Output file path.");
        parser.addCommandPositional(
                cmd(), kInput, "Serialized compressor file path.");
    }

    explicit InspectArgs(const arg::ParsedArgs& parsed) : GlobalArgs(parsed)
    {
        auto inputPath = parsed.cmdPositional(cmd(), kInput);
        input = std::make_unique<tools::io::InputFile>(std::move(inputPath));

        auto outputPath = parsed.cmdFlag(cmd(), kOutput);
        if (outputPath) {
            output = std::make_unique<tools::io::OutputFile>(
                    std::move(outputPath).value());
        }
    }

    static Cmd cmd()
    {
        return Cmd::INSPECT;
    }

    std::unique_ptr<tools::io::Input> input;
    std::unique_ptr<tools::io::Output> output;

   private:
    inline static const std::string kInput  = "input";
    inline static const std::string kOutput = "output";
};

} // namespace openzl::cli

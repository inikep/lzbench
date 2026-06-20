// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/base64.h>
#include <iostream>
#include "custom_transforms/thrift/parse_config.h"

using namespace zstrong::thrift;

// Edit this function to build your config, then run the binary to print it out!
std::string buildEncoderConfig()
{
    EncoderConfigBuilder builder;

    // Uncomment for TulipV2 format. Leave commented for TCompact and TBinary
    // formats.
    //
    // builder.setShouldParseTulipV2();

    // In very rare situations, we may want to change the root type.
    // In common cases, the default value (T_STRUCT) is correct.
    //
    // builder.setRootType(TType::T_LIST);

    // If building a non-trivial config, call methods like
    // EncoderConfigBuilder::addPath(), etc. If no paths are added, the config
    // will instruct the parser to type-split the data. This should be the
    // default config for new categories.

    return builder.finalize();
}

int main()
{
    const std::string configBytes = buildEncoderConfig();
    std::cout << folly::base64Encode(configBytes) << std::endl;
    return 0;
}

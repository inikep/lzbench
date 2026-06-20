// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "tools/io/OutputNull.h"

namespace openzl::tools::io {

poly::string_view OutputNull::name() const
{
    return "[no output]";
}

std::ostream& OutputNull::get_ostream()
{
    static auto devnull = std::ofstream{ "/dev/null" };
    return devnull;
}

} // namespace openzl::tools::io

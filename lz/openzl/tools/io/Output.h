// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/cpp/poly/StringView.hpp"

namespace openzl::tools::io {

class Output {
   public:
    virtual ~Output() = default;

    /**
     * Retrieves a meaningful name for this output, suitable e.g. for
     * identifying it to the user in status messages and errors.
     */
    virtual poly::string_view name() const = 0;

    /**
     * Optional method for outputs that require setup. Optional to call, if you
     * want to check that setup succeeds early.
     */
    virtual void open() {}

    /**
     * Optional method for outputs that require setup. Optional to call, if you
     * want to close output earlier than just destroying this object.
     */
    virtual void close() {}

    /**
     * Write the given contents to this output.
     */
    virtual void write(poly::string_view contents) = 0;

    virtual std::ostream& get_ostream() = 0;
};

} // namespace openzl::tools::io

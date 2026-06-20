// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <stddef.h>

#include "openzl/cpp/poly/Optional.hpp"
#include "openzl/cpp/poly/StringView.hpp"

namespace openzl::tools::io {

class Input {
   public:
    virtual ~Input() = default;

    /**
     * Retrieves a meaningful name for this input, suitable e.g. for
     * identifying it to the user in status messages and errors.
     */
    virtual poly::string_view name() const = 0;

    /**
     * Retrieve the size of the input.
     */
    virtual poly::optional<size_t> size() = 0;

    /**
     * Retrieve the contents of this input.
     */
    virtual poly::string_view contents() = 0;
};

} // namespace openzl::tools::io

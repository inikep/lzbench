// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "tools/io/InputSetBuilder.h"

#include "tools/io/InputSetFileOrDir.h"
#include "tools/io/InputSetLogger.h"
#include "tools/io/InputSetMulti.h"
#include "tools/io/InputSetStatic.h"

namespace openzl::tools::io {

InputSetBuilder::InputSetBuilder(bool recursive, bool verbose)
        : recursive_(recursive), verbose_(verbose)
{
}

InputSetBuilder& InputSetBuilder::add_path(std::string path) &
{
    std::unique_ptr<InputSet> input_set =
            std::make_unique<InputSetFileOrDir>(std::move(path), recursive_);
    input_sets_.push_back(std::move(input_set));
    return *this;
}
InputSetBuilder&& InputSetBuilder::add_path(std::string path) &&
{
    return std::move(add_path(std::move(path)));
}

InputSetBuilder& InputSetBuilder::add_path(
        std::optional<std::string> path_opt) &
{
    if (path_opt) {
        return add_path(std::move(path_opt).value());
    } else {
        return *this;
    }
}
InputSetBuilder&& InputSetBuilder::add_path(
        std::optional<std::string> path_opt) &&
{
    return std::move(add_path(std::move(path_opt)));
}

std::unique_ptr<InputSet> InputSetBuilder::build() &&
{
    std::unique_ptr<InputSet> result;
    if (input_sets_.size() != 1) {
        result = std::make_unique<InputSetMulti>(std::move(input_sets_));
    } else {
        result = std::move(input_sets_[0]);
    }
    if (verbose_) {
        result = std::make_unique<InputSetLogger>(std::move(result));
    }
    return result;
}

std::unique_ptr<InputSet> InputSetBuilder::build_static() &&
{
    auto input_set = std::move(*this).build();
    return std::make_unique<InputSetStatic>(
            InputSetStatic::from_input_set(*input_set));
}

} // namespace openzl::tools::io

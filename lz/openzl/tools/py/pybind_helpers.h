// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <optional>
#include <string_view>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "openzl/zl_data.h"
#include "openzl/zl_output.h"

namespace zstrong::pybind {
namespace py = pybind11;

/// @returns the integer size if the format is a native integer, otherwise
/// std::nullopt
std::optional<size_t> getNativeIntegerSize(std::string_view format);

/// @returns true if the buffer is laid out in contiguous C strides.
bool bufferIsContiguousCStrides(py::buffer_info const& info);

/// @returns a numpy array matching the specified type and data
/// current implementation copies the data and may be optimized
/// in the future
py::array
toNumpyArray(ZL_Type type, size_t nbElts, size_t eltWidth, void const* ptr);

/// @returns a numpy array matching the given stream
/// current implementation copies the data and may be optimized
/// in the future
py::array toNumpyArray(ZL_Input const* stream);

/// @returns a list of the stream content
py::list toList(ZL_Input const* stream);

/// @returns a ZL_Data matching a given python buffer_info and
/// a stream type. @p createStreamFn must copy the data (in the future
/// we could support referencing when possible to reduce number of
/// copies).
ZL_Output* bufferToStream(
        py::buffer_info const& buffer,
        ZL_Type streamType,
        const std::function<ZL_Output*(size_t nbElts, size_t eltWidth)>&
                createStreamFn);

/// @returns a ZL_Data matching a given numpy array and
/// a stream type. @p createStreamFn must copy the data (in the future
/// we could support referencing when possible to reduce number of
/// copies).
ZL_Output* arrayToStream(
        py::array const& array,
        ZL_Type streamType,
        const std::function<ZL_Output*(size_t nbElts, size_t eltWidth)>&
                createStreamFn);

/// @returns a ZL_Data matching a given numpy array and
/// a stream type. Created ZL_Data isn't associated with a
/// context.
ZL_Output* arrayToStream(py::array const& array, ZL_Type streamType);

/// @returns a ZL_Data given a list of str's for variable size fields.
ZL_Output* listToStream(
        py::list const& list,
        ZL_Type streamType,
        const std::function<ZL_Output*(size_t contentSize)>& createStreamFn);

} // namespace zstrong::pybind

// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <type_traits>

#include "openzl/cpp/Type.hpp"
#include "openzl/cpp/detail/NonNullUniqueCPtr.hpp"
#include "openzl/cpp/poly/Span.hpp"
#include "openzl/zl_output.h"

namespace openzl {
class Input;

class Output {
   public:
    Output(const Output&) = delete;
    Output(Output&&)      = default;

    Output& operator=(const Output&) = delete;
    Output& operator=(Output&&)      = default;

    ~Output() = default;

    explicit Output();

    static Output wrapSerial(void* buffer, size_t size);
    static Output wrapSerial(poly::span<char> output)
    {
        return wrapSerial(output.data(), output.size());
    }

    static Output wrapStruct(void* buffer, size_t eltWidth, size_t numElts);

    template <typename T>
    static Output wrapStruct(T* output, size_t numElts)
    {
        static_assert(
                std::is_standard_layout<T>{} && std::is_trivial<T>{},
                "Must be POD");
        return wrapStruct(output, sizeof(T), numElts);
    }

    template <typename T>
    static Output wrapStruct(poly::span<T> output)
    {
        return wrapStruct(output.data(), output.size());
    }

    static Output wrapNumeric(void* buffer, size_t eltWidth, size_t numElts);

    template <typename T>
    static Output wrapNumeric(T* output, size_t numElts)
    {
        static_assert(std::is_arithmetic<T>{}, "Must be arithmetic");
        return wrapNumeric(output, sizeof(T), numElts);
    }

    template <typename T>
    static Output wrapNumeric(poly::span<T> output)
    {
        return wrapNumeric(output.data(), output.size());
    }

    ZL_Output* get()
    {
        return output_.get();
    }

    const ZL_Output* get() const
    {
        return output_.get();
    }

    /// @returns true iff the data in the this Output is exactly equal to
    /// @p other. Metadata is ignored during equality testing.
    bool operator==(const Output& other) const;
    bool operator!=(const Output& other) const
    {
        return !(*this == other);
    }

    /// @returns true iff the data in the this Output is exactly equal to
    /// @p other. Metadata is ignored during equality testing.
    bool operator==(const Input& other) const;
    bool operator!=(const Input& other) const
    {
        return !(*this == other);
    }

    int id() const;
    Type type() const;
    size_t eltWidth() const;
    size_t contentSize() const;
    size_t numElts() const;
    size_t eltsCapacity() const;
    size_t contentCapacity() const;
    void* ptr();
    const void* ptr() const;
    const uint32_t* stringLens() const;

    uint32_t* reserveStringLens(size_t numElts);
    void commit(size_t numElts);
    void setIntMetadata(int key, int value);
    poly::optional<int> getIntMetadata(int key) const;

    size_t unwrap(
            ZL_Report report,
            poly::string_view msg = {},
            poly::source_location location =
                    poly::source_location::current()) const
    {
        return openzl::unwrap(
                report, std::move(msg), nullptr, std::move(location));
    }

   protected:
    Output(ZL_Output* output,
           detail::NonNullUniqueCPtr<ZL_Output>::DeleterFn deleter)
            : output_(output, deleter)
    {
    }

   private:
    explicit Output(ZL_Output* output);

    detail::NonNullUniqueCPtr<ZL_Output> output_;
};

class OutputRef : public Output {
   public:
    explicit OutputRef(ZL_Output* output) : Output(output, nullptr) {}
};

} // namespace openzl

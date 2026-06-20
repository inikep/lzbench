// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <type_traits>

#include "openzl/cpp/Type.hpp"
#include "openzl/cpp/detail/NonNullUniqueCPtr.hpp"
#include "openzl/cpp/poly/Optional.hpp"
#include "openzl/cpp/poly/SourceLocation.hpp"
#include "openzl/cpp/poly/Span.hpp"
#include "openzl/cpp/poly/StringView.hpp"
#include "openzl/zl_compress.h"
#include "openzl/zl_input.h"

namespace openzl {
class Output;

class Input {
   public:
    Input(const Input&) = delete;
    Input(Input&&)      = default;

    Input& operator=(const Input&) = delete;
    Input& operator=(Input&&)      = default;

    ~Input() = default;

    static Input refSerial(const void* buffer, size_t size);
    static Input refSerial(poly::string_view data)
    {
        return refSerial(data.data(), data.size());
    }

    static Input refStruct(const void* buffer, size_t eltWidth, size_t numElts);

    template <typename T>
    static Input refStruct(const T* input, size_t numElts)
    {
        static_assert(
                std::is_standard_layout<T>{} && std::is_trivial<T>{},
                "Must be POD");
        return refStruct(input, sizeof(T), numElts);
    }

    template <typename T>
    static Input refStruct(poly::span<const T> input)
    {
        return refStruct(input.data(), input.size());
    }

    static Input
    refNumeric(const void* buffer, size_t eltWidth, size_t numElts);

    template <typename T>
    static Input refNumeric(const T* input, size_t numElts)
    {
        static_assert(std::is_arithmetic<T>{}, "Must be arithmetic");
        return refNumeric(input, sizeof(T), numElts);
    }

    template <typename T>
    static Input refNumeric(poly::span<const T> input)
    {
        return refNumeric(input.data(), input.size());
    }

    static Input refString(
            const void* content,
            size_t contentSize,
            const uint32_t* lengths,
            size_t numElts);

    static Input refString(
            poly::string_view content,
            poly::span<const uint32_t> lengths)
    {
        return refString(
                content.data(), content.size(), lengths.data(), lengths.size());
    }

    static Input refOutput(const Output& output);

    const ZL_Input* get() const
    {
        return input_.get();
    }
    ZL_Input* get()
    {
        return input_.get();
    }

    int id() const
    {
        return ZL_Input_id(get()).sid;
    }

    Type type() const
    {
        return Type(ZL_Input_type(get()));
    }

    size_t numElts() const
    {
        return ZL_Input_numElts(get());
    }

    size_t eltWidth() const
    {
        return ZL_Input_eltWidth(get());
    }

    size_t contentSize() const
    {
        return ZL_Input_contentSize(get());
    }

    const void* ptr() const
    {
        return ZL_Input_ptr(get());
    }

    const uint32_t* stringLens() const
    {
        auto lens = ZL_Input_stringLens(get());
        if (lens == nullptr) {
            throw Exception("Input: Called stringLens() on non-string type");
        }
        return lens;
    }

    poly::optional<int> getIntMetadata(int key) const
    {
        auto val = ZL_Input_getIntMetadata(get(), key);
        if (!val.isPresent) {
            return poly::nullopt;
        } else {
            return val.mValue;
        }
    }

    void setIntMetadata(int key, int value);

    /// @returns true iff the data in the this Input is exactly equal to
    /// @p other. Metadata is ignored during equality testing.
    bool operator==(const Input& other) const;

    bool operator!=(const Input& other) const
    {
        return !(*this == other);
    }

    /// @returns true iff the data in the this Input is exactly equal to
    /// @p other. Metadata is ignored during equality testing.
    bool operator==(const Output& other) const
    {
        return *this == Input::refOutput(other);
    }

    bool operator!=(const Output& other) const
    {
        return !(*this == other);
    }

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
    Input(ZL_Input* input,
          detail::NonNullUniqueCPtr<ZL_Input>::DeleterFn deleter)
            : input_(input, deleter)
    {
    }

   private:
    explicit Input(ZL_Input* input);

    detail::NonNullUniqueCPtr<ZL_Input> input_;
};

class InputRef : public Input {
   public:
    explicit InputRef(ZL_Input* input) : Input(input, nullptr) {}
};

} // namespace openzl

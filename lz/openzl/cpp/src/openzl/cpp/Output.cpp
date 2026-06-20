// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/cpp/Output.hpp"

#include <assert.h>
#include "openzl/cpp/Input.hpp"
#include "openzl/zl_decompress.h"

namespace openzl {
Output::Output() : Output(ZL_TypedBuffer_create()) {}

Output::Output(ZL_Output* output) : Output(output, ZL_TypedBuffer_free) {}

/* static */ Output Output::wrapSerial(void* buffer, size_t size)
{
    return Output(ZL_TypedBuffer_createWrapSerial(buffer, size));
}

/* static */ Output
Output::wrapStruct(void* buffer, size_t eltWidth, size_t numElts)
{
    return Output(ZL_TypedBuffer_createWrapStruct(buffer, eltWidth, numElts));
}

/* static */ Output
Output::wrapNumeric(void* buffer, size_t eltWidth, size_t numElts)
{
    return Output(ZL_TypedBuffer_createWrapNumeric(buffer, eltWidth, numElts));
}

int Output::id() const
{
    return ZL_Output_id(get()).sid;
}

Type Output::type() const
{
    auto type = ZL_Output_type(get());
    if (type == ZL_Type_unassigned) {
        throw Exception("Output: Illegal to call type() on an empty output");
    }

    return Type(type);
}

size_t Output::eltWidth() const
{
    auto width = unwrap(ZL_Output_eltWidth(get()));
    if (width == 0) {
        if (type() == Type::String) {
            throw Exception(
                    "Output: Illegal to call eltWidth() on string type");
        }
        assert(false);
    }
    return width;
}

size_t Output::contentSize() const
{
    return unwrap(ZL_Output_contentSize(get()));
}

size_t Output::numElts() const
{
    return unwrap(ZL_Output_numElts(get()));
}

size_t Output::eltsCapacity() const
{
    return unwrap(ZL_Output_eltsCapacity(get()));
}

size_t Output::contentCapacity() const
{
    return unwrap(ZL_Output_contentCapacity(get()));
}

void* Output::ptr()
{
    auto ptr = ZL_Output_ptr(get());
    if (ptr == nullptr) {
        throw Exception("Output: Illegal to call ptr() on an empty output");
    }
    return ptr;
}

const void* Output::ptr() const
{
    auto ptr = ZL_Output_constPtr(get());
    if (ptr == nullptr) {
        throw Exception("Output: Illegal to call ptr() on an empty output");
    }
    return ptr;
}

const uint32_t* Output::stringLens() const
{
    auto lens = ZL_Output_constStringLens(get());
    if (lens == nullptr) {
        if (type() != Type::String) {
            throw Exception(
                    "Output: Illegal to call stringLens() on a non-string type");
        }
        throw Exception(
                "Output: Illegal to call stringLens() before reserveStringLens()");
    }
    return lens;
}

uint32_t* Output::reserveStringLens(size_t numElts)
{
    auto lens = ZL_Output_reserveStringLens(get(), numElts);
    if (lens == nullptr) {
        if (type() != Type::String) {
            throw Exception(
                    "Output: Illegal to call reserveStringLens() on non-string type");
        }
        throw ExceptionBuilder("Output: reserveStringLens() failed")
                .withErrorCode(ZL_ErrorCode_allocation)
                .build();
    }
    return lens;
}

void Output::commit(size_t numElts)
{
    unwrap(ZL_Output_commit(get(), numElts));
}

void Output::setIntMetadata(int key, int value)
{
    unwrap(ZL_Output_setIntMetadata(get(), key, value));
}

poly::optional<int> Output::getIntMetadata(int key) const
{
    auto meta = ZL_Output_getIntMetadata(get(), key);
    if (meta.isPresent) {
        return { meta.mValue };
    } else {
        return poly::nullopt;
    }
}

bool Output::operator==(const Output& other) const
{
    return Input::refOutput(*this) == other;
}

bool Output::operator==(const Input& other) const
{
    return Input::refOutput(*this) == other;
}

} // namespace openzl

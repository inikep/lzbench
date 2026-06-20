// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/cpp/Input.hpp"

#include "openzl/cpp/Output.hpp"

namespace openzl {

Input::Input(ZL_Input* input) : input_(input, ZL_TypedRef_free) {}

/* static */ Input Input::refSerial(const void* buffer, size_t size)
{
    return Input(ZL_TypedRef_createSerial(buffer, size));
}

/* static */ Input
Input::refStruct(const void* buffer, size_t eltWidth, size_t numElts)
{
    return Input(ZL_TypedRef_createStruct(buffer, eltWidth, numElts));
}
/* static */ Input
Input::refNumeric(const void* buffer, size_t eltWidth, size_t numElts)
{
    return Input(ZL_TypedRef_createNumeric(buffer, eltWidth, numElts));
}
/* static */ Input Input::refString(
        const void* content,
        size_t contentSize,
        const uint32_t* lengths,
        size_t numElts)
{
    return Input(
            ZL_TypedRef_createString(content, contentSize, lengths, numElts));
}

/* static */ Input Input::refOutput(const Output& output)
{
    switch (output.type()) {
        case Type::Serial:
            return Input::refSerial(output.ptr(), output.numElts());
        case Type::Struct:
            return Input::refStruct(
                    output.ptr(), output.eltWidth(), output.numElts());
        case Type::Numeric:
            return Input::refNumeric(
                    output.ptr(), output.eltWidth(), output.numElts());
        case Type::String:
            return Input::refString(
                    output.ptr(),
                    output.contentSize(),
                    output.stringLens(),
                    output.numElts());
    }
    throw Exception("Input: Invalid output type");
}

void Input::setIntMetadata(int key, int value)
{
    unwrap(ZL_Input_setIntMetadata(get(), key, value));
}

bool Input::operator==(const Input& other) const
{
    if (type() != other.type()) {
        return false;
    }
    if (numElts() != other.numElts()) {
        return false;
    }
    if (type() == Type::String) {
        if (contentSize() != other.contentSize()) {
            return false;
        }
        if (numElts() > 0
            && memcmp(stringLens(),
                      other.stringLens(),
                      sizeof(uint32_t) * numElts())
                    != 0) {
            return false;
        }
    } else {
        if (eltWidth() != other.eltWidth()) {
            return false;
        }
    }
    if (contentSize() > 0 && memcmp(ptr(), other.ptr(), contentSize()) != 0) {
        return false;
    }
    return true;
}

} // namespace openzl

// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "examples/example_utils.h"

namespace zstrong {
namespace examples {
std::string readFile(const char* filename)
{
    FILE* f = fopen(filename, "rb");
    abortIf(f == nullptr, "Failed to read file");
    fseek(f, 0, SEEK_END);
    const size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string data(size, '\0');
    const size_t read = fread(data.data(), 1, data.size(), f);
    abortIf(read != size, "Failed to read file");
    fclose(f);
    return data;
}
} // namespace examples
} // namespace zstrong

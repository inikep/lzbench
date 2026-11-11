#pragma once

#include <cassert>
#include <cstddef>
#include <optional>

namespace rapidgzip
{
class BlockFinderInterface
{
public:
    enum class GetReturnCode
    {
        SUCCESS,
        TIMEOUT,
        FAILURE,
    };

public:
    virtual
    ~BlockFinderInterface() = default;

    [[nodiscard]] virtual size_t
    size() const = 0;

    [[nodiscard]] virtual bool
    finalized() const = 0;

    [[nodiscard]] virtual std::pair<std::optional<size_t>, GetReturnCode>
    get( size_t blockIndex,
         double timeoutInSeconds ) = 0;

    [[nodiscard]] virtual std::optional<size_t>
    get( size_t blockIndex )
    {
        const auto [result, returnCode] = get( blockIndex, -1.0 );
        assert( returnCode != GetReturnCode::TIMEOUT );
        return result;
    }

    [[nodiscard]] virtual size_t
    find( size_t encodedBlockOffsetInBits ) const = 0;
};
}  // namespace rapidgzip

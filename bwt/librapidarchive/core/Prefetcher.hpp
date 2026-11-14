#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <deque>
#include <iterator>
#include <numeric>
#include <optional>
#include <set>
#include <vector>

#include <core/common.hpp>       // interleave


namespace rapidgzip::FetchingStrategy
{
class FetchingStrategy
{
public:
    virtual
    ~FetchingStrategy() = default;

    /**
     * Overriding methods must call this base method!
     */
    virtual void
    fetch( size_t index )
    {
        m_lastFetched = index;
    }

    [[nodiscard]] std::optional<size_t>
    lastFetched() const
    {
        return m_lastFetched;
    }

    /**
     * @return true if all of the memorized last access were sequential. On true, more optimizations may be enabled.
     */
    [[nodiscard]] virtual bool
    isSequential() const noexcept
    {
        return false;
    }

    [[nodiscard]] virtual std::vector<size_t>
    prefetch( size_t maxAmountToPrefetch ) const = 0;

private:
    std::optional<size_t> m_lastFetched;
};


/**
 * Simply prefetches the next n indexes after the last access.
 */
class FetchNextFixed :
    public FetchingStrategy
{
public:
    [[nodiscard]] std::vector<size_t>
    prefetch( size_t maxAmountToPrefetch ) const override
    {
        if ( !lastFetched() ) {
            return {};
        }

        std::vector<size_t> toPrefetch( maxAmountToPrefetch );
        std::iota( toPrefetch.begin(), toPrefetch.end(), *lastFetched() + 1 );
        return toPrefetch;
    }
};


/**
 * Similar to @ref FetchNextFixed but the amount of returned subsequent indexes is relative to the amount of
 * consecutive accesses in the memory.
 * If all are consecutive, returns the specified amount to prefetch.
 * Else, if there are only random access in memory, then it will return nothing to prefetch to avoid wasted computation.
 * Between, interpolate exponentially, i.e., for a memory size of 3 and 4 requested prefetch indexes:
 *   1 consecutive pair -> 1
 *   2 consecutive pair -> 2
 *   3 consecutive pair -> 4
 */
class FetchNextAdaptive :
    public FetchingStrategy
{
public:
    explicit
    FetchNextAdaptive( size_t memorySize = 3 ) :
        m_memorySize( memorySize )
    {}

    void
    fetch( size_t index ) override
    {
        FetchingStrategy::fetch( index );

        /* Ignore duplicate accesses, which in the case of bzip2 blocks most likely means
         * that the caller reads only small parts from the block per call. */
        if ( !m_previousIndexes.empty() && ( m_previousIndexes.front() == index ) ) {
            return;
        }

        m_previousIndexes.push_front( index );
        while ( m_previousIndexes.size() > m_memorySize ) {
            m_previousIndexes.pop_back();
        }
    }

    [[nodiscard]] bool
    isSequential() const noexcept override
    {
        /* Returning true for the empty case and only one element or very few is desired behavior. */
        for ( size_t i = 0; i + 1 < m_previousIndexes.size(); ++i ) {
            if ( m_previousIndexes[i + 1] + 1 != m_previousIndexes[i] ) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] static std::vector<size_t>
    extrapolateForward( const size_t highestValue,
                        const size_t consecutiveValues,
                        const size_t saturationCount,
                        const size_t maxExtrapolation )
    {
        /** 0 <= consecutiveRatio <= 1 */
        const auto consecutiveRatio = saturationCount == 0
                                      ? 1.0
                                      : static_cast<double>( std::min( consecutiveValues, saturationCount ) )
                                        / static_cast<double>( saturationCount );
        /** 1 <= maxAmountToPrefetch +- floating point errors */
        const auto amountToPrefetch = std::round( std::exp2( consecutiveRatio * std::log2( maxExtrapolation ) ) );

        assert( amountToPrefetch >= 0 );
        assert( static_cast<size_t>( amountToPrefetch ) <= maxExtrapolation );

        std::vector<size_t> toPrefetch( static_cast<size_t>( std::max( 0.0, amountToPrefetch ) ) );
        std::iota( toPrefetch.begin(), toPrefetch.end(), highestValue + 1 );
        return toPrefetch;
    }

    template<typename Iterator>
    [[nodiscard]] static std::vector<size_t>
    extrapolate( const Iterator rangeBegin,
                 const Iterator rangeEnd,
                 const size_t   maxAmountToPrefetch )
    {
        const auto size = std::distance( rangeBegin, rangeEnd );
        if ( ( size == 0 ) || ( maxAmountToPrefetch == 0 ) ) {
            return {};
        }

        /** This avoids division by 0 further down below! */
        if ( size == 1 ) {
            /* This enables parallel sequential decoding with only a single cache miss.
             * I think this is an often enough encountered use case to have this special case justified. */
            std::vector<size_t> toPrefetch( maxAmountToPrefetch );
            std::iota( toPrefetch.begin(), toPrefetch.end(), *rangeBegin + 1 );
            return toPrefetch;
        }

        /* Count adjacent decreasing elements (because the most recent is on the left!). */
        if ( countAdjacentIf( rangeBegin, rangeEnd, [] ( auto a, auto b ) { return a == b + 1; } ) == 0 ) {
            /* Handle special case of only random accesses. */
            return {};
        }

        /* The most recent values are assumed to be at begin! */
        size_t lastConsecutiveCount = 0;
        for ( auto it = rangeBegin, nit = std::next( it ); nit != rangeEnd; ++it, ++nit ) {
            if ( *it == *nit + 1 ) {
                lastConsecutiveCount = lastConsecutiveCount == 0 ? 2 : lastConsecutiveCount + 1;
            } else {
                break;
            }
        }

        return extrapolateForward( *rangeBegin, lastConsecutiveCount, size, maxAmountToPrefetch );
    }

    [[nodiscard]] std::vector<size_t>
    prefetch( size_t maxAmountToPrefetch ) const override
    {
        return extrapolate( m_previousIndexes.begin(), m_previousIndexes.end(), maxAmountToPrefetch );
    }

    /**
     * Updates indexes such that the given indexes is interpreted as four separate indexes from now on.
     * This also shifts all larger indexes accordingly and might evict some indexes because of the new split ones.
     */
    void
    splitIndex( size_t indexToSplit,
                size_t splitCount )
    {
        if ( splitCount <= 1 ) {
            return;
        }

        std::deque<size_t> newIndexes;
        for ( const auto index : m_previousIndexes ) {
            if ( index == indexToSplit ) {
                for ( size_t i = 0; i < splitCount; ++i ) {
                    newIndexes.push_back( index + splitCount - 1 - i );
                }
            } else if ( index > indexToSplit ) {
                newIndexes.push_back( index + splitCount - 1 );
            } else {
                newIndexes.push_back( index );
            }
        }

        m_previousIndexes = std::move( newIndexes );
    }

protected:
    const size_t m_memorySize;
    /** Contains the most recent index at the front. */
    std::deque<size_t> m_previousIndexes;
};


/**
 * Similar to @ref FetchNextAdaptive but it is able to detect multiple interleaved consecutive accesses as might happen
 * for parallel processes accessing the cache.
 * Detection works by sorting all last indexes and looking for consecutive sequences there and return predictions
 * for them at the same time.
 */
class FetchMultiStream :
    public FetchNextAdaptive
{
public:
    explicit
    FetchMultiStream( size_t memorySize = 3,
                      size_t maxStreamCount = 16U ) :
        FetchNextAdaptive( maxStreamCount * memorySize ),
        m_memorySizePerStream( memorySize )
    {}

    [[nodiscard]] std::vector<size_t>
    prefetch( size_t maxAmountToPrefetch ) const override
    {
        const auto& previousIndexes = this->m_previousIndexes;
        if ( previousIndexes.empty() ) {
            return {};
        }

        if ( previousIndexes.size() == 1 ) {
            /* This enables parallel sequential decoding with only a single cache miss.
             * I think this is an often enough encountered use case to have this special case justified. */
            std::vector<size_t> toPrefetch( maxAmountToPrefetch );
            std::iota( toPrefetch.begin(), toPrefetch.end(), previousIndexes.front() + 1 );
            return toPrefetch;
        }

        const auto sortedIndexes =
            [&previousIndexes] () {
                auto result = previousIndexes;
                std::sort( result.begin(), result.end() );
                return result;
            }();

        std::vector<std::vector<size_t> > subsequencePrefetches;

        const auto extrapolateSubsequence =
            [&, this] ( const auto begin,
                        const auto end )
            {
                /* We only have the deduplicated and sorted indexes. We need to recheck with the real saved indexes
                 * whether it is an increasing sequence, else a perfect backward pattern might be detected as a forward
                 * pattern because of the sorting. */
                const auto highestValue = *std::prev( end );
                size_t sequenceLength{ 0 };
                auto indexesBegin = previousIndexes.begin();
                for ( auto currentHighestValue = std::reverse_iterator( end );
                      currentHighestValue != std::reverse_iterator( begin );
                      ++currentHighestValue )
                {
                    indexesBegin = std::find( indexesBegin, previousIndexes.end(), *currentHighestValue );
                    if ( indexesBegin == previousIndexes.end() ) {
                        break;
                    }
                    ++sequenceLength;
                }

                /* Do not prefetch for completely random access. */
                if ( memoryFull() && ( sequenceLength == 1 ) ) {
                    return;
                }

                /* For the very first prefetches, fewer than the memory size, extrapolate them fully for faster
                 * first-time decoding. */
                const auto consecutiveValues = sequenceLength <= 1 ? 0 : sequenceLength;
                const auto saturationCount = !memoryFull() && ( consecutiveValues > 0 )
                                             ? consecutiveValues : m_memorySizePerStream;
                subsequencePrefetches.emplace_back(
                    extrapolateForward( highestValue, consecutiveValues, saturationCount, maxAmountToPrefetch ) );
            };

        auto lastRangeBegin = sortedIndexes.begin();
        for ( auto it = sortedIndexes.begin(), nit = std::next( it ); ; ++it, ++nit ) {
            if ( ( nit == sortedIndexes.end() ) || ( *it + 1 != *nit ) ) {
                extrapolateSubsequence( lastRangeBegin, nit );
                lastRangeBegin = nit;
                if ( nit == sortedIndexes.end() ) {
                    break;
                }
            }
        }

        auto result = interleave( subsequencePrefetches );
        const auto newEnd = std::remove_if(
            result.begin(), result.end(),
            [this] ( auto value ) {
                return std::find( m_previousIndexes.begin(), m_previousIndexes.end(), value )
                       != m_previousIndexes.end();
            } );
        result.resize( std::min( static_cast<size_t>( std::distance( result.begin(), newEnd ) ),
                                 maxAmountToPrefetch ) );
        return result;
    }

    [[nodiscard]] bool
    memoryFull() const
    {
        return m_previousIndexes.size() >= m_memorySize;
    }

private:
    const size_t m_memorySizePerStream;
};
}  // namespace rapidgzip::FetchingStrategy

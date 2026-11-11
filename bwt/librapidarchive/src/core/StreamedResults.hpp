#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "BlockFinderInterface.hpp"


namespace rapidgzip
{
/**
 * Stores results in the order they are pushed and also stores a flag signaling that nothing will be pushed anymore.
 * The blockfinder will push block offsets and other actors, e.g., the prefetcher, may wait for and read the offsets.
 * Results will never be deleted, so you can assume the size to only grow.
 */
template<typename Value>
class StreamedResults
{
public:
    /**
     * std::vector would work as well but the reallocations during appending might slow things down.
     * For the index access operations, a container with random access iterator, would yield better performance.
     */
    using Values = std::deque<Value>;

    using GetReturnCode = BlockFinderInterface::GetReturnCode;

    class ResultsView
    {
    public:
        ResultsView( const Values* results,
                     std::mutex*   mutex ) :
            m_results( results ),
            m_lock( *mutex )
        {
            if ( m_results == nullptr ) {
                throw std::invalid_argument( "Arguments may not be nullptr!" );
            }
        }

        [[nodiscard]] const Values&
        results() const
        {
            return *m_results;
        }

    private:
        Values const * const m_results;
        std::scoped_lock<std::mutex> const m_lock;
    };

public:
    [[nodiscard]] size_t
    size() const
    {
        const std::scoped_lock lock( m_mutex );
        return m_results.size();
    }

    /**
     * @param timeoutInSeconds Negative waits indefinitely, 0 does not wait at all, and a positive value is a
     *        timeout. Do not use infinity as a sentinel: -ffast-math implies -ffinite-math-only, which would
     *        make an `std::isfinite` check on the sentinel return true and turn the infinite wait into an
     *        immediate timeout.
     * @return the result at the requested position.
     */
    [[nodiscard]] std::pair<std::optional<size_t>, GetReturnCode>
    get( size_t position,
         double timeoutInSeconds = -1.0 ) const
    {
        std::unique_lock lock( m_mutex );

        const auto predicate = [&] () { return m_finalized || ( position < m_results.size() ); };

        if ( timeoutInSeconds < 0 ) {
            m_changed.wait( lock, predicate );
        } else if ( timeoutInSeconds > 0 ) {
            const auto timeout = std::chrono::nanoseconds( static_cast<size_t>( timeoutInSeconds * 1e9 ) );
            m_changed.wait_for( lock, timeout, predicate );
        }

        if ( position < m_results.size() ) {
            return { m_results[position], GetReturnCode::SUCCESS };
        }
        return { std::nullopt, m_finalized ? GetReturnCode::FAILURE : GetReturnCode::TIMEOUT };
    }

    void
    push( Value value )
    {
        const std::scoped_lock lock( m_mutex );

        if ( m_finalized ) {
            throw std::invalid_argument( "You may not push to finalized StreamedResults!" );
        }

        m_results.emplace_back( std::move( value ) );
        m_changed.notify_all();
    }

    void
    finalize( std::optional<size_t> resultsCount = {} )
    {
        const std::scoped_lock lock( m_mutex );

        if ( resultsCount ) {
            if ( *resultsCount > m_results.size() ) {
                throw std::invalid_argument( "You may not finalize to a size larger than the current results buffer!" );
            }

            m_results.resize( *resultsCount );
        }

        m_finalized = true;
        m_changed.notify_all();
    }

    [[nodiscard]] bool
    finalized() const
    {
        return m_finalized;
    }

    /** @return a view to the results, which also locks access to it using RAII. */
    [[nodiscard]] ResultsView
    results() const
    {
        return ResultsView( &m_results, &m_mutex );
    }

    void
    setResults( Values results )
    {
        const std::scoped_lock lock( m_mutex );

        m_results = std::move( results );
        m_finalized = true;
        m_changed.notify_all();
    }

private:
    mutable std::mutex m_mutex;
    mutable std::condition_variable m_changed;

    Values m_results;
    std::atomic<bool> m_finalized = false;
};
}  // namespace rapidgzip

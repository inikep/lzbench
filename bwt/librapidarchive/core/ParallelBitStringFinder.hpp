#pragma once

#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <list>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "AffinityHelpers.hpp"
#include "BitStringFinder.hpp"
#include "common.hpp"
#include "ThreadPool.hpp"


namespace rapidgzip
{
/**
 * No matter the input, the data is read from an input buffer.
 * If a file is given, then that input buffer will be refilled when the input buffer empties.
 * It is less a file object and acts more like an iterator.
 * It offers a @ref find method returning the next match or std::numeric_limits<size_t>::max() if the end was reached.
 */
template<uint8_t bitStringSize>
class ParallelBitStringFinder :
    public BitStringFinder<bitStringSize>
{
public:
    using BaseType = BitStringFinder<bitStringSize>;

    static_assert( bitStringSize > 0, "Bit string to find must have positive length!" );

public:
    ParallelBitStringFinder( UniqueFileReader fileReader,
                             uint64_t         bitStringToFind,
                             size_t           parallelization = std::max( 1U, availableCores() / 8U ),
                             size_t           requestedBytes = 0,
                             size_t           fileBufferSizeBytes = 1_Mi ) :
        BaseType( std::move( fileReader ),
                  bitStringToFind,
                  chunkSize( fileBufferSizeBytes, requestedBytes, parallelization ) ),
        m_threadPool( parallelization )
    {}

    /** @note This overload is used for the tests but can also be useful for other things. */
    ParallelBitStringFinder( const char* buffer,
                             size_t      size,
                             uint64_t    bitStringToFind,
                             size_t      parallelization ) :
        BaseType( UniqueFileReader(), bitStringToFind ),
        m_threadPool( parallelization )
    {
        this->m_buffer.assign( buffer, buffer + size );
    }

    ~ParallelBitStringFinder() override = default;

    void
    stopThreads() override
    {
        m_threadPool.stop();
    }

    /**
     * @return the next match and the requested bytes or std::numeric_limits<size_t>::max() if at end of file.
     */
    [[nodiscard]] size_t
    find() override;

private:
    /**
     * The worker pushes found offsets during which it locks the mutex and the reader locks and pops from the
     * queue. After the worker has finished, which can be queried with the future, and when the found offsets
     * have all been read, this struct can be deleted or reused. The worker thread sets finished and notifies
     * with the condition variable. It also notifies when pushing to the queue.
     */
    struct ThreadResults
    {
        std::queue<size_t>      foundOffsets;
        std::mutex              mutex;
        std::future<void>       future;
        std::condition_variable changed;
    };

private:
    [[nodiscard]] static constexpr size_t
    chunkSize( size_t const fileBufferSizeBytes,
               size_t const requestedBytes,
               size_t const parallelization )
    {
        /* This implementation has the limitation that it might at worst try to read as many as bitStringSize
         * bits from the buffered chunk. It makes no sense to remove this limitation. It might slow things down. */
        const auto result = std::max( fileBufferSizeBytes,
                                      static_cast<size_t>( ceilDiv( bitStringSize, 8 ) ) * parallelization );
        /* With the current implementation it is impossible to have a chunk size smaller than the requested bytes
         * and have it work for non-seekable inputs. In the worst case, the bit string is at the end, so we have to
         * read almost everything of the next chunk. */
        return std::max( result, requestedBytes );
    }

    /**
     * Calls findBitStrings and handles all the parallel synchronization stuff like sending
     * the results to the reading thread through the result buffer.
     * When it is finished, it will return std::numeric_limits<size_t>::max() to signal that the future can be waited
     * for.
     *
     * @param buffer pointer data to look for bitstrings. [buffer, buffer+size) will be searched.
     * @param firstBitsToIgnore This will effectively force matches to only be returned if
     *                          foundOffset >= firstBitsToIgnore
     * @param bitOffsetToAdd All found offsets relative to @p buffer should add this in order to return the global
     *                       bit offsets of interest.
     */
    static void
    workerMain( char   const * const buffer,
                size_t         const bufferSizeInBytes,
                uint8_t        const firstBitsToIgnore,
                uint64_t       const bitStringToFind,
                size_t         const bitOffsetToAdd,
                ThreadResults* const result )
    {
        auto offsets = BaseType::findBitStrings( { buffer, bufferSizeInBytes }, bitStringToFind );
        std::sort( offsets.begin(), offsets.end() );

        const std::lock_guard<std::mutex> lock( result->mutex );
        for ( const auto offset : offsets ) {
            if ( offset >= firstBitsToIgnore ) {
                result->foundOffsets.push( bitOffsetToAdd + offset );
            }
        }
        result->foundOffsets.push( std::numeric_limits<size_t>::max() );
        result->changed.notify_one();
    }

private:
    /** Return at least this amount of bytes after and including the found bit strings. */
    const size_t m_requestedBytes = 0;

    std::list<ThreadResults> m_threadResults;

    ThreadPool m_threadPool;
};


/**
 * Idea:
 *   1. Load one chunk if first iteration
 *   2. Use the serial BitStringFinder in parallel on equal-sized sized sub chunks.
 *   3. Filter out results we already could have found in the chunk before if more than bitStringSize-1
 *      bits were loaded from it.
 *   4. Translate the returned bit offsets of the BitStringFinders to global offsets.
 *   5. Copy requested bytes after match into result buffer.
 *   6. Load the next chunk plus at least the last bitStringSize-1 bits from the chunk before.
 *   7. Use that new chunk to append more of the requested bytes after matches to the result buffer.
 *      More than one chunk should not be necessary for this! This is ensured in the chunkSize method.
 */
template<uint8_t bitStringSize>
size_t
ParallelBitStringFinder<bitStringSize>::find()
{
#ifdef WITH_PYTHON_SUPPORT
    const ScopedGILUnlock unlockedGIL;
#endif

    while ( !BaseType::eof() || !m_threadResults.empty() )
    {
        /* Check whether there are results available and return those. Take care to return results in order! */
        while ( !m_threadResults.empty() ) {
            auto& result = m_threadResults.front();
            using namespace std::chrono;

            /* Check whether some results are already calculated. */
            std::unique_lock<std::mutex> lock( result.mutex );
            while ( !result.foundOffsets.empty() || result.future.valid() ) {
                /* In the easiest case we have something to return already. */
                if ( !result.foundOffsets.empty() ) {
                    if ( result.foundOffsets.front() == std::numeric_limits<size_t>::max() ) {
                        result.foundOffsets.pop();
                        if ( result.future.valid() ) {
                            result.future.get();
                        }
                        break;
                    }
                    const auto foundOffset = result.foundOffsets.front();
                    result.foundOffsets.pop();
                    return foundOffset;
                }

                /* Wait for thread to finish or push new results. Note that this may hang if the worker thread
                 * crashes because the predicate check is only done on condition variable notifies and on spurious
                 * wakeups but those are not guaranteed. */
                result.changed.wait(
                    lock,
                    [&result] () {
                        return !result.foundOffsets.empty() ||
                               ( result.future.wait_for( 0s ) == std::future_status::ready );
                    } );

                if ( result.future.wait_for( 0s ) == std::future_status::ready ) {
                    result.future.get();
                }
            }
            lock = {};  /* release result.mutex before popping result! */

            if ( result.future.valid() || !result.foundOffsets.empty() ) {
                throw std::logic_error( "Should have gotten future and emptied offsets!" );
            }
            m_threadResults.pop_front();
        }

        /* Constructor might fill buffer already making a buffer refill unnecessary the first time! */
        if ( this->bufferEof() ) {
            const auto nBytesRead = BaseType::refillBuffer();
            if ( nBytesRead == 0 ) {
                return std::numeric_limits<size_t>::max();
            }
        }

        /* For very sub chunk sizes, it is more sensible to not parallelize them using threads! */
        const auto minSubChunkSizeInBytes = std::max<size_t>( 8UL * bitStringSize, 4096 );
        const auto subChunkStrideInBytes =
            std::max<size_t>( minSubChunkSizeInBytes, ceilDiv( this->m_buffer.size(), m_threadPool.capacity() ) );

        /* Start worker threads using the thread pool and the current buffer. */
        for ( ; !this->bufferEof(); this->m_bufferBitsRead += subChunkStrideInBytes * CHAR_BIT ) {
            /* Try to seek m_movingBitsToKeep back in order to find bit strings going over the previous border.
             * On the very first iteration after refilling the buffer, the number of read bits might be non-zero
             * but smaller than the bit string to find because BitReader prepends some bytes of the last buffer
             * in order to find bit strings across buffer boundaries. In this case, i.e., in the first iteration,
             * We do not need to seek back by m_movingBitsToKeep only after the first chunk is this necessary just
             * like it is necessary for BitReader::refillBuffer. */
            auto const bufferOffsetInBits = this->m_bufferBitsRead > this->m_movingBitsToKeep
                                            ? this->m_bufferBitsRead - this->m_movingBitsToKeep
                                            : this->m_bufferBitsRead;
            auto const bufferOffsetInBytes  = bufferOffsetInBits / CHAR_BIT;
            auto const subChunkOffsetInBits = static_cast<uint8_t>( bufferOffsetInBits % CHAR_BIT );

            auto const subChunkSizeInBits = this->m_bufferBitsRead - bufferOffsetInBits
                                            + subChunkStrideInBytes * CHAR_BIT;
            auto const subChunkSizeInBytes = std::min( ceilDiv( subChunkSizeInBits, CHAR_BIT ),
                                                       this->m_buffer.size() - bufferOffsetInBytes );

            //std::cerr << "  Find from offset " << bufferOffsetInBytes << "B " << subChunkOffsetInBits << "b "
            //          << "sub chunk size " << subChunkSizeInBytes << " B, "
            //          << "sub chunk stride: " << subChunkStrideInBytes << "B, "
            //          << "buffer size: " << this->m_buffer.size() << " B\n";

            auto& result = m_threadResults.emplace_back();
            result.future = m_threadPool.submit(
                [=, &result] ()
                {
                    workerMain(
                        /* sub chunk buffer */ this->m_buffer.data() + bufferOffsetInBytes,
                        subChunkSizeInBytes,
                        subChunkOffsetInBits,
                        this->m_bitStringToFind,
                        ( this->m_nTotalBytesRead + bufferOffsetInBytes ) * CHAR_BIT,
                        &result
                    );
                } );
        }
    }

    return std::numeric_limits<size_t>::max();
}
}  // namespace rapidgzip

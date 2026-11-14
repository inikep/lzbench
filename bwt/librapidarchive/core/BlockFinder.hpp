#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

#include <core/BlockFinderInterface.hpp>
#include <core/JoiningThread.hpp>
#include <core/StreamedResults.hpp>
#ifdef WITH_PYTHON_SUPPORT
    #include <core/ScopedGIL.hpp>
#endif


namespace rapidgzip
{
/**
 * This is a future-like wrapper around a given actual block finder, which is running asynchronously.
 * The results are not only computed in parallel but also prefetched up to a certain distance to allow full
 * utilization of parallelism for the asynchronous computation.
 * This class also acts as a database (StreamedResults is the actual database) after all results have been computed
 * and can be initialized with the results to avoid recomputing them.
 *
 * @tparam RawBlockFinder Must implement a `size_t find()` method which returns block offsets.
 */
template<typename T_RawBlockFinder>
class BlockFinder final :
    public BlockFinderInterface
{
public:
    using RawBlockFinder = T_RawBlockFinder;
    using BlockOffsets = StreamedResults<size_t>::Values;

public:
    explicit
    BlockFinder( std::unique_ptr<RawBlockFinder> rawBlockFinder ) :
        m_rawBlockFinder( std::move( rawBlockFinder ) )
    {}

    ~BlockFinder() override
    {
        const std::scoped_lock lock( m_mutex );
        m_cancelThread = true;
        m_changed.notify_all();
    }

public:
    void
    startThreads()
    {
        if ( !m_rawBlockFinder ) {
            throw std::invalid_argument( "You may not start the block finder without a valid bit string finder!" );
        }

        if ( !m_blockFinder ) {
            m_blockFinder = std::make_unique<JoiningThread>( [this] () { blockFinderMain(); } );
        }
    }

    void
    stopThreads()
    {
        {
            const std::scoped_lock lock( m_mutex );
            m_cancelThread = true;
            m_changed.notify_all();
        }

        if ( m_blockFinder && m_blockFinder->joinable() ) {
        #ifdef WITH_PYTHON_SUPPORT
            /* The finder thread reacquires the GIL when its find() call returns. A caller such as close()
             * or joinThreads() may hold it, which would deadlock this join. */
            const ScopedGILUnlock unlockedGIL;
        #endif
            m_blockFinder->join();
        }
        m_blockFinder.reset();
        m_cancelThread = false;

        if ( m_rawBlockFinder ) {
            m_rawBlockFinder->stopThreads();
        }
    }

    [[nodiscard]] size_t
    size() const override
    {
        return m_blockOffsets.size();
    }

    /** Finalizes and will only keep the first @param blockCount blocks. */
    void
    finalize( std::optional<size_t> blockCount = {} )
    {
        stopThreads();
        m_rawBlockFinder = {};
        m_blockOffsets.finalize( blockCount );
    }

    [[nodiscard]] bool
    finalized() const override
    {
        return m_blockOffsets.finalized();
    }

    using BlockFinderInterface::get;

    /**
     * This call will track the requested block so that the finder loop will look up to that block.
     * Per default, with the infinite timeout, either a result can be returned or if not it means
     * we are finalized and the requested block is out of range!
     */
    [[nodiscard]] std::pair<std::optional<size_t>, GetReturnCode>
    get( size_t blockNumber,
         double timeoutInSeconds ) override
    {
    #ifdef WITH_PYTHON_SUPPORT
        const ScopedGILUnlock unlockedGIL;
    #endif

        if ( !m_blockOffsets.finalized() ) {
            startThreads();
        }

        {
            const std::scoped_lock lock( m_mutex );
            m_highestRequestedBlockNumber = std::max( m_highestRequestedBlockNumber, blockNumber );
            m_changed.notify_all();
        }

        return m_blockOffsets.get( blockNumber, timeoutInSeconds );
    }

    /** @return Index for the block at the requested offset. */
    [[nodiscard]] size_t
    find( size_t encodedBlockOffsetInBits ) const override
    {
        const std::scoped_lock lock( m_mutex );

        /* m_blockOffsets is effectively double-locked but that's the price of abstraction. */
        const auto lockedOffsets = m_blockOffsets.results();
        const auto& blockOffsets = lockedOffsets.results();

        /* Find in sorted vector by bisection. */
        const auto match = std::lower_bound( blockOffsets.begin(), blockOffsets.end(), encodedBlockOffsetInBits );
        if ( ( match == blockOffsets.end() ) || ( *match != encodedBlockOffsetInBits ) ) {
            throw std::out_of_range( "No block with the specified offset exists in the gzip block finder map!" );
        }

        return std::distance( blockOffsets.begin(), match );
    }

    void
    setBlockOffsets( BlockOffsets blockOffsets )
    {
        /* First we need to cancel the asynchronous block finder thread. */
        stopThreads();
        m_rawBlockFinder = {};

        /* Setting the results also finalizes them. No locking necessary because all threads have shut down. */
        m_blockOffsets.setResults( std::move( blockOffsets ) );
    }

private:
    void
    blockFinderMain()
    {
        while ( !m_cancelThread ) {
            std::unique_lock lock( m_mutex );
            /* m_blockOffsets.size() will only grow, so we don't need to be notified when it changes! */
            m_changed.wait(
                lock,
                [this] {
                    return m_cancelThread
                           || ( m_blockOffsets.size() <= m_highestRequestedBlockNumber + m_prefetchCount );
                } );
            if ( m_cancelThread ) {
                break;
            }

            /**
             * The time for this find method should be bounded and responsive enough for reacting to cancellations.
             * During this compute intensive task, the lock should be unlocked!
             * Or else, the getter and other functions will never be able to acquire the lock
             * until this thread has finished reading the whole file!
             */
            lock.unlock();  // Unlock for a little while so that others can acquire the lock!
            const auto blockOffset = m_rawBlockFinder->find();
            if ( blockOffset == std::numeric_limits<size_t>::max() ) {
                m_blockOffsets.finalize();
                break;
            }

            lock.lock();
            m_blockOffsets.push( blockOffset );
        }

        /* Do not finalize on cancellation. A later startThreads() must be able to resume the finder. */
    }

private:
    mutable std::mutex m_mutex;  /**< Only variables accessed by the asynchronous main loop need to be locked. */
    std::condition_variable m_changed;

    StreamedResults<size_t> m_blockOffsets;

    size_t m_highestRequestedBlockNumber{ 0 };

    /**
     * Only hardware_concurrency slows down decoding! I guess because in the worst case all decoding
     * threads finish at the same time and now the bit string finder would need to find n new blocks
     * in the time it takes to decode one block! In general, the higher this number, the higher the
     * longer will be the initial CPU utilization.
     */
    const size_t m_prefetchCount = 3ULL * std::thread::hardware_concurrency();

    std::unique_ptr<RawBlockFinder> m_rawBlockFinder;
    std::atomic<bool> m_cancelThread{ false };

    std::unique_ptr<JoiningThread> m_blockFinder;
};
}  // namespace rapidgzip

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

#include <core/AffinityHelpers.hpp>
#include <core/BlockFinderInterface.hpp>
#include <core/Cache.hpp>
#include <core/common.hpp>
#include <core/Prefetcher.hpp>
#include <core/ThreadPool.hpp>

#ifdef WITH_PYTHON_SUPPORT
    #include <core/ScopedGIL.hpp>  // For unlocking the GIL in case the block fetch code uses PythonFileReader
#endif


namespace rapidgzip
{
/**
 * Manages block data access. Calls to members are not thread-safe!
 * Requested blocks are cached and accesses may trigger prefetches,
 * which will be fetched in parallel using a thread pool.
 */
template<typename T_BlockFinder,
         typename T_BlockData,
         typename FetchingStrategy>
class BlockFetcher
{
public:
    using BlockFinder = T_BlockFinder;
    using BlockData = T_BlockData;
    using BlockCache = Cache</** block offset in bits */ size_t, std::shared_ptr<BlockData> >;

    static_assert( std::is_base_of_v<BlockFinderInterface, BlockFinder>,
                   "Block finder must derive from the abstract interface." );

    using GetPartitionOffset = std::function<size_t( size_t )>;

    struct Statistics
    {
    public:
        [[nodiscard]] double
        cacheHitRate() const
        {
            return static_cast<double>( cache.hits + prefetchCache.hits + prefetchDirectHits )
                   / static_cast<double>( gets );
        }

        [[nodiscard]] double
        uselessPrefetches() const
        {
            const auto totalFetches = prefetchCount + onDemandFetchCount;
            if ( totalFetches == 0 ) {
                return 0;
            }
            return static_cast<double>( prefetchCache.unusedEntries ) / static_cast<double>( totalFetches );
        }

        [[nodiscard]] std::string
        print() const
        {
            std::stringstream existingBlocks;
            existingBlocks << ( blockCountFinalized ? "" : ">=" ) << blockCount;

            const auto decodeDuration = decodeBlockStartTime && decodeBlockEndTime
                                        ? duration( *decodeBlockStartTime, *decodeBlockEndTime )
                                        : 0.0;
            const auto optimalDecodeDuration = decodeBlockTotalTime / parallelization;
            /* The pool efficiency only makes sense when the thread pool is smaller or equal the CPU cores. */
            const auto poolEfficiency = optimalDecodeDuration / decodeDuration;

            std::stringstream out;
            out << "\n    Parallelization                   : " << parallelization
                << "\n    Cache"
                << "\n        Hits                          : " << cache.hits
                << "\n        Misses                        : " << cache.misses
                << "\n        Unused Entries                : " << cache.unusedEntries
                << "\n        Maximum Fill Size             : " << cache.maxSize
                << "\n        Capacity                      : " << cache.capacity
                << "\n    Prefetch Cache"
                << "\n        Hits                          : " << prefetchCache.hits
                << "\n        Misses                        : " << prefetchCache.misses
                << "\n        Unused Entries                : " << prefetchCache.unusedEntries
                << "\n        Prefetch Queue Hit            : " << prefetchDirectHits
                << "\n        Maximum Fill Size             : " << prefetchCache.maxSize
                << "\n        Capacity                      : " << prefetchCache.capacity
                << "\n    Cache Hit Rate                    : " << cacheHitRate() * 100 << " %"
                << "\n    Useless Prefetches                : " << uselessPrefetches() * 100 << " %"
                << "\n    Access Patterns"
                << "\n        Total Accesses                : " << gets
                << "\n        Duplicate Block Accesses      : " << repeatedBlockAccesses
                << "\n        Sequential Block Accesses     : " << sequentialBlockAccesses
                << "\n        Block Seeks Back              : " << backwardBlockAccesses
                << "\n        Block Seeks Forward           : " << forwardBlockAccesses
                << "\n    Blocks"
                << "\n        Total Existing                : " << existingBlocks.str()
                << "\n        Total Fetched                 : " << prefetchCount + onDemandFetchCount
                << "\n        Prefetched                    : " << prefetchCount
                << "\n        Fetched On-demand             : " << onDemandFetchCount
                << "\n    Prefetch Stall by BlockFinder     : " << waitOnBlockFinderCount
                << "\n    Time spent in:"
                << "\n        decodeBlock                   : " << decodeBlockTotalTime   << " s"
                << "\n        std::future::get              : " << futureWaitTotalTime    << " s"
                << "\n        get                           : " << getTotalTime           << " s"
                << "\n    Thread Pool Utilization:"
                << "\n        Total Real Decode Duration    : " << decodeDuration << " s"
                << "\n        Theoretical Optimal Duration  : " << optimalDecodeDuration << " s"
                << "\n        Pool Efficiency (Fill Factor) : " << poolEfficiency * 100 << " %";
            return out.str();
        }

        void
        recordBlockIndexGet( size_t blockindex )
        {
            ++gets;

            if ( !lastAccessedBlock ) {
                lastAccessedBlock = blockindex;  // effectively counts the first access as sequential
            }

            if ( blockindex > *lastAccessedBlock + 1 ) {
                forwardBlockAccesses++;
            } else if ( blockindex < *lastAccessedBlock ) {
                backwardBlockAccesses++;
            } else if ( blockindex == *lastAccessedBlock ) {
                repeatedBlockAccesses++;
            } else {
                sequentialBlockAccesses++;
            }

            lastAccessedBlock = blockindex;
        }

    public:
        size_t parallelization{ 0 };
        size_t blockCount{ 0 };
        bool blockCountFinalized{ false };

        typename BlockCache::Statistics cache;
        typename BlockCache::Statistics prefetchCache;

        size_t gets{ 0 };
        std::optional<size_t> lastAccessedBlock;
        size_t repeatedBlockAccesses{ 0 };
        size_t sequentialBlockAccesses{ 0 };
        size_t backwardBlockAccesses{ 0 };
        size_t forwardBlockAccesses{ 0 };

        size_t onDemandFetchCount{ 0 };
        size_t prefetchCount{ 0 };
        size_t prefetchDirectHits{ 0 };
        size_t waitOnBlockFinderCount{ 0 };

        std::optional<std::decay_t<decltype( now() )> > decodeBlockStartTime;
        std::optional<std::decay_t<decltype( now() )> > decodeBlockEndTime;

        double decodeBlockTotalTime{ 0 };
        double futureWaitTotalTime{ 0 };
        double getTotalTime{ 0 };
    };

protected:
    BlockFetcher( std::shared_ptr<BlockFinder> blockFinder,
                  size_t                       parallelization ) :
        m_parallelization( parallelization == 0 ? std::max<size_t>( 1U, availableCores() ) : parallelization ),
        m_blockFinder( std::move( blockFinder ) ),
        m_cache( std::max( size_t( 16 ), m_parallelization ) ),
        m_prefetchCache( 2 * m_parallelization /* Only m_parallelization would lead to lot of cache pollution! */ ),
        m_failedPrefetchCache( m_prefetchCache.capacity() ),
        /* If parallelization is 1, then do not start any thread even if the main thread is not doing much work. */
        m_threadPool( m_parallelization == 1 ? 0 : m_parallelization )
    {
        if ( !m_blockFinder ) {
            throw std::invalid_argument( "BlockFinder must be valid!" );
        }
        m_statistics.parallelization = m_parallelization;
    }

public:
    virtual
    ~BlockFetcher()
    {
        if ( m_showProfileOnDestruction ) {
            /* Clear caches while updating the unused entries statistic. */
            m_cache.shrinkTo( 0 );
            m_prefetchCache.shrinkTo( 0 );
            std::cerr << ( ThreadSafeOutput() << "[BlockFetcher::~BlockFetcher]" << statistics().print() );
        }
    }

    void
    setStatisticsEnabled( bool enabled )
    {
        m_statisticsEnabled = enabled;
    }

    [[nodiscard]] bool
    statisticsEnabled() const
    {
        return m_statisticsEnabled;
    }

    /**
     * @note Only will work if m_statisticsEnabled is true.
     */
    void
    setShowProfileOnDestruction( bool showProfileOnDestruction )
    {
        m_showProfileOnDestruction = showProfileOnDestruction;
    }

    [[nodiscard]] bool
    test( const size_t blockOffset ) const
    {
        return ( m_prefetching.find( blockOffset ) != m_prefetching.end() )
               || m_cache.test( blockOffset )
               || m_prefetchCache.test( blockOffset );
    }

    /**
     * Fetches, prefetches, caches, and returns result.
     * @param dataBlockIndex Only used to determine which block indexes to prefetch. If not specified, will
     *        query BlockFinder for the block offset. This started as a performance optimization, to avoid
     *        unnecessary BlockFinder lookups but when looking up the partition offset, it might be necessary
     *        or else the BlockFinder::find call might throw because it can't find the given offset.
     * @param getPartitionOffsetFromOffset Returns the partition offset to a given blockOffset. This is used to look up
     *        existence of blocks in the cache to avoid duplicate prefetches (one for the partition offset
     *        and another one for the real offset).
     * @return BlockData to requested blockOffset. Undefined what happens for an invalid blockOffset as input.
     */
    [[nodiscard]] std::shared_ptr<BlockData>
    get( const size_t                blockOffset,
         const std::optional<size_t> dataBlockIndex = std::nullopt,
         const GetPartitionOffset&   getPartitionOffsetFromOffset = {} )
    {
        [[maybe_unused]] const auto tGetStart = now();

    #ifdef WITH_PYTHON_SUPPORT
        /* The GIL needs to be unlocked for the worker threads to not wait infinitely when calling methods
         * on a given Python file object. In theory, it suffices to call this unlock here to avoid deadlocks
         * because it is the only method that waits for results from the worker threads. But, it might be
         * more efficient to unlock the GIL outside to avoid many unlock/lock cycles and to leave it unlocked
         * for longer so as to not hinder the worker threads. */
        const ScopedGILUnlock unlockedGIL;
    #endif

        /* Not using capture bindings here because C++ is too dumb to capture those, yet.
         * @see https://stackoverflow.com/a/46115028/2191065 */
        auto resultFromCaches = getFromCaches( blockOffset );
        auto& cachedResult = resultFromCaches.first;
        auto& queuedResult = resultFromCaches.second;

        const auto validDataBlockIndex = dataBlockIndex ? *dataBlockIndex : m_blockFinder->find( blockOffset );
        const auto nextBlockOffset = m_blockFinder->get( validDataBlockIndex + 1 );

        if ( m_statisticsEnabled ) {
            m_statistics.recordBlockIndexGet( validDataBlockIndex );
        }

        /* Start requested calculation if necessary. */
        if ( !cachedResult.has_value() && !queuedResult.valid() ) {
            queuedResult = submitOnDemandTask( blockOffset, nextBlockOffset );
        }

        const auto lastFetchedIndex = m_fetchingStrategy.lastFetched();
        m_fetchingStrategy.fetch( validDataBlockIndex );

        const auto resultIsReady =
            [&cachedResult, &queuedResult] () {
                using namespace std::chrono_literals;
                return cachedResult.has_value() ||
                       ( queuedResult.valid() && ( queuedResult.wait_for( 0s ) == std::future_status::ready ) );
            };

        /* The prefetch below only is called when the result future times out. When all futures are ready,
         * this prefetch call would only be called when trying to access the next non-prefetched block.
         * This would introduce a large latency, which is not necessary. This call here is necessary to avoid
         * that. This prefetch results in pipelined behavior, i.e., the next block will be prefetched, when
         * the earliest block can get removed from the cache, e.g., during sequential access!
         * However, prefetchNewBlocks is very expensive for some reason, therefore only call it when the
         * accessed block index actually has changed. This yields orders of magnitudes speedups when
         * ParallelGzipReader::read is called with nBytesToRead < 32 KiB. */
        if ( !lastFetchedIndex || ( lastFetchedIndex.value() != validDataBlockIndex ) ) {
            prefetchNewBlocks( getPartitionOffsetFromOffset, resultIsReady );
        }

        /* Return result */
        if ( cachedResult.has_value() ) {
            assert( !queuedResult.valid() );
            if ( m_statisticsEnabled ) {
                const std::scoped_lock lock( m_analyticsMutex );
                m_statistics.getTotalTime += duration( tGetStart );
            }
            return *std::move( cachedResult );
        }

        [[maybe_unused]] const auto tFutureGetStart = now();
        using namespace std::chrono_literals;
        /* At ~4 MiB compressed blocks and ~200 MB/s compressed bandwidth for base64, one block might take ~20ms. */
        while ( queuedResult.wait_for( 1ms ) == std::future_status::timeout ) {
            prefetchNewBlocks( getPartitionOffsetFromOffset, resultIsReady );
        }
        std::shared_ptr<BlockData> result;
        try {
            result = std::make_shared<BlockData>( queuedResult.get() );
        } catch ( ... ) {
            /* This function owns the only future referencing the worker's shared state. It is destroyed while
             * unwinding, and the worker's task state may be the last owner of the stored exception. Keep the
             * exception alive in a member until the next failure so the caller can read exception.what(). */
            m_exceptionBeingRethrown = std::current_exception();
            std::rethrow_exception( m_exceptionBeingRethrown );
        }
        [[maybe_unused]] const auto futureGetDuration = duration( tFutureGetStart );

        insertIntoCache( blockOffset, result );

        if ( m_statisticsEnabled ) {
            const std::scoped_lock lock( m_analyticsMutex );
            m_statistics.futureWaitTotalTime += futureGetDuration;
            m_statistics.getTotalTime += duration( tGetStart );
        }

        return result;
    }

    /**
     * Clears all caches and waits for outstanding prefetches. Use this when the decoding configuration
     * changes, e.g., when enabling checksum verification, because prefetched chunks decoded with the old
     * configuration must not be reused. The wait is a quiescence barrier, not result collection: dropping
     * the futures would leave the worker tasks running against the old configuration.
     */
    virtual void
    clearCache()
    {
    #ifdef WITH_PYTHON_SUPPORT
        /* Worker threads may need the GIL to read from a Python file object. */
        const ScopedGILUnlock unlockedGIL;
    #endif
        for ( auto& [blockOffset, future] : m_prefetching ) {
            if ( future.valid() ) {
                try {
                    (void)future.get();
                } catch ( const std::exception& ) {
                    /* Errors are rethrown when the block is requested directly. */
                }
            }
        }
        m_prefetching.clear();
        m_cache.clear();
        m_prefetchCache.clear();
        {
            const std::scoped_lock lock( m_failedPrefetchCacheMutex );
            m_failedPrefetchCache.clear();
        }
    }

    [[nodiscard]] Statistics
    statistics() const
    {
        auto result = m_statistics;
        if ( m_blockFinder ) {
            result.blockCountFinalized = m_blockFinder->finalized();
            result.blockCount = m_blockFinder->size();
        }
        result.cache = m_cache.statistics();
        result.prefetchCache = m_prefetchCache.statistics();
        return result;
    }

private:
    void
    insertIntoCache( const size_t               blockOffset,
                     std::shared_ptr<BlockData> blockData )
    {
        if ( m_fetchingStrategy.isSequential() ) {
            m_cache.clear();
        }
        m_cache.insert( blockOffset, std::move( blockData ) );
    }

    [[nodiscard]] bool
    isInCacheOrQueue( const size_t blockOffset ) const
    {
        return ( m_prefetching.find( blockOffset ) != m_prefetching.end() )
               || m_cache.test( blockOffset )
               || m_prefetchCache.test( blockOffset );
    }

    [[nodiscard]] bool
    isFailedPrefetch( const size_t blockOffset ) const
    {
        const std::scoped_lock lock( m_failedPrefetchCacheMutex );
        return m_failedPrefetchCache.test( blockOffset );
    }

    /**
     * @return either a shared_ptr from the caches or a future from the prefetch queue. The prefetch future
     *         is taken from the queue, i.e., it should not be discarded. Either reinsert it into the queue
     *         or wait for the result and insert it into a cache.
     */
    [[nodiscard]] std::pair<std::optional<std::shared_ptr<BlockData> >, std::future<BlockData> >
    getFromCaches( const size_t blockOffset )
    {
        /* In case of a late prefetch, this might return an unfinished future. */
        auto resultFuture = takeFromPrefetchQueue( blockOffset );

        /* Access cache before data might get evicted!
         * Access cache after prefetch queue to avoid incrementing the cache misses counter. */
        std::optional<std::shared_ptr<BlockData> > result;
        if ( !resultFuture.valid() ) {
            result = m_cache.get( blockOffset );
            if ( !result ) {
                /* On prefetch cache hit, move the value into the normal cache. */
                result = m_prefetchCache.get( blockOffset );
                if ( result ) {
                    m_prefetchCache.evict( blockOffset );
                    insertIntoCache( blockOffset, *result );
                }
            }
        }

        return std::make_pair( std::move( result ), std::move( resultFuture ) );
    }

    [[nodiscard]] std::future<BlockData>
    takeFromPrefetchQueue( size_t blockOffset )
    {
        /* Check whether the desired offset is prefetched. */
        std::future<BlockData> resultFuture;
        const auto match = m_prefetching.find( blockOffset );

        if ( match != m_prefetching.end() ) {
            resultFuture = std::move( match->second );
            m_prefetching.erase( match );
            assert( resultFuture.valid() );

            if ( m_statisticsEnabled ) {
                ++m_statistics.prefetchDirectHits;
            }
        }

        return resultFuture;
    }

    /* Check for ready prefetches and move them to cache. */
    void
    processReadyPrefetches()
    {
        using namespace std::chrono_literals;

        for ( auto it = m_prefetching.begin(); it != m_prefetching.end(); ) {
            auto& [prefetchedBlockOffset, prefetchedFuture] = *it;

            if ( prefetchedFuture.valid() && ( prefetchedFuture.wait_for( 0s ) == std::future_status::ready ) ) {
                BlockData result;
                try {
                    result = prefetchedFuture.get();
                    m_prefetchCache.insert( prefetchedBlockOffset, std::make_shared<BlockData>( std::move( result ) ) );
                } catch ( ... ) {
                    /* Prefetching failed, ignore result and error. If the error was a real one, then it will
                     * will be rethrown when the task is requested directly and run directly. */
                    const std::scoped_lock lock( m_failedPrefetchCacheMutex );
                    m_failedPrefetchCache.insert( prefetchedBlockOffset, /* value does not matter */ true );
                }
                it = m_prefetching.erase( it );
            } else {
                ++it;
            }
        }
    }

    /**
     * Fills m_prefetching up with a maximum of m_parallelization-1 new tasks predicted
     * based on the given last accessed block index(es).
     * @param stopPrefetching The prefetcher might wait a bit on the block finder but when stopPrefetching returns true
     *                        it will stop that and return before having completely filled the prefetch queue.
     */
    void
    prefetchNewBlocks( const GetPartitionOffset&    getPartitionOffsetFromOffset,
                       const std::function<bool()>& stopPrefetching )
    {
        /* Make space for new asynchronous prefetches. */
        processReadyPrefetches();

        const auto threadPoolSaturated =
            [&] () {
                return m_prefetching.size() + /* thread with the requested block */ 1 >= m_threadPool.capacity();
            };

        if ( threadPoolSaturated() ) {
            return;
        }

        auto blockIndexesToPrefetch = m_fetchingStrategy.prefetch( m_prefetchCache.capacity() );

        std::vector<size_t> blockOffsetsToPrefetch;
        blockOffsetsToPrefetch.reserve( 2 * blockIndexesToPrefetch.size() );
        for ( auto blockIndexToPrefetch : blockIndexesToPrefetch ) {
            /* If we don't find the offset in the timeout of 0, then we very likely also don't have it cached yet. */
            const auto [blockOffset, _] = m_blockFinder->get( blockIndexToPrefetch, /* timeout */ 0 );
            if ( !blockOffset ) {
                continue;
            }

            blockOffsetsToPrefetch.emplace_back( *blockOffset );
            if ( getPartitionOffsetFromOffset ) {
                const auto partitionOffset = getPartitionOffsetFromOffset( *blockOffset );
                if ( *blockOffset != partitionOffset ) {
                    blockOffsetsToPrefetch.emplace_back( partitionOffset );
                }
            }
        }

        /* Touch all blocks to be prefetched to avoid evicting them while doing the prefetching of other blocks! */
        for ( auto offset = blockOffsetsToPrefetch.rbegin(); offset != blockOffsetsToPrefetch.rend(); ++offset ) {
            m_prefetchCache.touch( *offset );
            m_cache.touch( *offset );
        }

        for ( auto blockIndexToPrefetch : blockIndexesToPrefetch ) {
            if ( threadPoolSaturated() ) {
                break;
            }

            if ( m_blockFinder->finalized() && ( blockIndexToPrefetch >= m_blockFinder->size() ) ) {
                continue;
            }

            /* If the block with the requested index has not been found yet and if we have to wait on the requested
             * result future anyway, then wait a non-zero amount of time on the BlockFinder! */
            using GetReturnCode = BlockFinderInterface::GetReturnCode;
            std::optional<size_t> prefetchBlockOffset;
            auto prefetchGetReturnCode = GetReturnCode::FAILURE;
            std::optional<size_t> nextPrefetchBlockOffset;
            auto nextPrefetchGetReturnCode = GetReturnCode::FAILURE;
            do
            {
                std::tie( prefetchBlockOffset, prefetchGetReturnCode ) =
                    m_blockFinder->get( blockIndexToPrefetch, stopPrefetching() ? 0 : 0.0001 );
                std::tie( nextPrefetchBlockOffset, nextPrefetchGetReturnCode ) =
                    m_blockFinder->get( blockIndexToPrefetch + 1, stopPrefetching() ? 0 : 0.0001 );
            }
            while ( !prefetchBlockOffset && ( prefetchGetReturnCode != GetReturnCode::FAILURE )
                    && !nextPrefetchBlockOffset && ( nextPrefetchGetReturnCode != GetReturnCode::FAILURE )
                    && !stopPrefetching() );

            if ( m_statisticsEnabled ) {
                if ( !prefetchBlockOffset.has_value() ) {
                    m_statistics.waitOnBlockFinderCount++;
                }
            }

            /* The next block offset is used as an upper bound while decoding. The last block of a finalized
             * input has no next data block, and waiting for one would prevent it from ever being prefetched.
             * Decoding a last block without a bound is already done for on-demand requests, so prefetch it
             * with the same bound. This matters for inputs with only one or two data blocks, where otherwise
             * the second block cannot be decoded in parallel with the first one. */
            const auto isLastKnownBlock =
                m_blockFinder->finalized() && ( blockIndexToPrefetch + 1 >= m_blockFinder->size() );

            /* Do not prefetch already cached/prefetched blocks or block indexes which are not yet in the block map. */
            if ( !prefetchBlockOffset.has_value()
                 || ( prefetchGetReturnCode == GetReturnCode::FAILURE )
                 || ( !nextPrefetchBlockOffset.has_value() && !isLastKnownBlock )
                 || isInCacheOrQueue( *prefetchBlockOffset )
                 || ( getPartitionOffsetFromOffset
                      && isInCacheOrQueue( getPartitionOffsetFromOffset( *prefetchBlockOffset ) ) )
                 || isFailedPrefetch( *prefetchBlockOffset ) )
            {
                continue;
            }

            /* Avoid cache pollution by stopping prefetching when we would evict usable results.
             * Note that we have to also account for m_prefetching.size() evictions before our eviction of interest! */
            if ( const auto offsetToBeEvicted = m_prefetchCache.nextNthEviction( m_prefetching.size() + 1 );
                 offsetToBeEvicted.has_value() ) {
                if ( contains( blockOffsetsToPrefetch, offsetToBeEvicted ) ) {
                    break;
                }
            }

            ++m_statistics.prefetchCount;
            auto prefetchedFuture = m_threadPool.submit(
                [this, offset = *prefetchBlockOffset,
                 nextOffset = nextPrefetchBlockOffset ? *nextPrefetchBlockOffset
                                                      : std::numeric_limits<size_t>::max()] () {
                    return decodeAndMeasureBlock( offset, nextOffset );
                }, /* priority */ 0 );
            const auto [_, wasInserted] = m_prefetching.emplace( *prefetchBlockOffset, std::move( prefetchedFuture ) );
            if ( !wasInserted ) {
                throw std::logic_error( "Submitted future could not be inserted to prefetch queue!" );
            }
        }

        /* Note that only m_parallelization-1 blocks will be prefetched. Meaning that even with the unconditionally
         * submitted requested block, the thread pool should never contain more than m_parallelization tasks!
         * All tasks submitted to the thread pool, should either exist in m_prefetching or only temporary inside
         * 'resultFuture' in the 'read' method. */
        if ( m_threadPool.unprocessedTasksCount( 0 ) > m_parallelization ) {
            throw std::logic_error( "The thread pool should not have more tasks than there are prefetching futures!" );
        }
    }

    [[nodiscard]] std::future<BlockData>
    submitOnDemandTask( const size_t                blockOffset,
                        const std::optional<size_t> nextBlockOffset )
    {
        if ( m_statisticsEnabled ) {
            ++m_statistics.onDemandFetchCount;
        }
        auto resultFuture = m_threadPool.submit(
            [this, blockOffset, nextBlockOffset] ()
            {
                return decodeAndMeasureBlock(
                    blockOffset, nextBlockOffset ? *nextBlockOffset : std::numeric_limits<size_t>::max() );
            },
            /* priority */ 0 );
        assert( resultFuture.valid() );
        return resultFuture;
    }

protected:
    [[nodiscard]] virtual BlockData
    decodeBlock( size_t blockOffset,
                 size_t nextBlockOffset ) const = 0;

    /**
     * This must be called before variables that are used by @ref decodeBlock are destructed, i.e.,
     * it must be called in the inheriting class.
     */
    void
    stopThreadPool()
    {
        m_threadPool.stop();
    }

public:
    /**
     * Drains outstanding prefetches and caches, then stops and joins all worker threads. Queued tasks therefore
     * run before the pool is stopped. The rest of the state is kept so that a later read continues from the same
     * block map and finder state. This is intended for use before forking. The pool restarts on the next submit.
     */
    virtual void
    joinThreads()
    {
        clearCache();
        m_threadPool.stop();
    }

protected:
    template<class T_Functor, std::enable_if_t<std::is_invocable_v<T_Functor>, void>* = nullptr>
    std::future<decltype( std::declval<T_Functor>()() )>
    submitTaskWithHighPriority( T_Functor task )
    {
        return m_threadPool.submit( std::move( task ), /* priority */ -1 );
    }

    [[nodiscard]] const auto&
    cache() const noexcept
    {
        return m_cache;
    }

    [[nodiscard]] auto&
    cache() noexcept
    {
        return m_cache;
    }

    [[nodiscard]] const auto&
    prefetchCache() const noexcept
    {
        return m_prefetchCache;
    }

private:
    [[nodiscard]] BlockData
    decodeAndMeasureBlock( size_t blockOffset,
                           size_t nextBlockOffset ) const
    {
        [[maybe_unused]] const auto tDecodeStart = now();
        auto blockData = decodeBlock( blockOffset, nextBlockOffset );
        if ( m_statisticsEnabled ) {
            const auto tDecodeEnd = now();

            const std::scoped_lock lock( m_analyticsMutex );

            const auto& minStartTime = m_statistics.decodeBlockStartTime;
            m_statistics.decodeBlockStartTime.emplace(
                minStartTime ? std::min( *minStartTime, tDecodeStart ) : tDecodeStart );

            const auto& maxEndTime = m_statistics.decodeBlockEndTime;
            m_statistics.decodeBlockEndTime.emplace(
                maxEndTime ? std::max( *maxEndTime, tDecodeEnd ) : tDecodeEnd );

            m_statistics.decodeBlockTotalTime += duration( tDecodeStart, tDecodeEnd );
        }
        return blockData;
    }

private:
    mutable Statistics m_statistics;
    std::atomic<bool> m_statisticsEnabled{ false };
    mutable std::mutex m_analyticsMutex;

protected:
    const size_t m_parallelization;

    FetchingStrategy m_fetchingStrategy;

    bool m_showProfileOnDestruction{ false };

private:
    /**
     * The block finder is used to prefetch blocks among others.
     * But, in general, it only returns unconfirmed guesses for block offsets (at first)!
     * Confirmed block offsets are written to the BlockMap but adding that in here seems a bit overkill
     * and would need further logic to get the next blocks given a specific one.
     * Therefore, the idea is to update and confirm the blocks inside the block finder, which would invalidate
     * the block indexes! In order for that, to not lead to problems, the block finder should only be used by
     * the managing thread not by the worker threads!
     */
    const std::shared_ptr<BlockFinder> m_blockFinder;

    BlockCache m_cache;
    BlockCache m_prefetchCache;
    Cache</* block offset in bits */ size_t, bool> m_failedPrefetchCache;
    mutable std::mutex m_failedPrefetchCacheMutex;

    std::map</* block offset */ size_t, std::future<BlockData> > m_prefetching;
    ThreadPool m_threadPool;

    /* Keeps a rethrown worker exception alive after the future and task state owning it are destroyed. */
    std::exception_ptr m_exceptionBeingRethrown;
};
}  // namespace rapidgzip

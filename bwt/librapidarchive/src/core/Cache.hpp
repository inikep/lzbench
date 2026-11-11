#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rapidgzip
{
namespace CacheStrategy
{
template<typename Index>
class CacheStrategy
{
public:
    virtual
    ~CacheStrategy() = default;

    virtual void
    touch( Index index ) = 0;

    /**
     * @return The next eviction no matter whether the cache is currently full. Only returns nothing if the cache
     *         is empty, i.e., there is nothing to evict.
     */
    [[nodiscard]] virtual std::optional<Index>
    nextEviction() const = 0;

    [[nodiscard]] virtual std::optional<Index>
    nextNthEviction( size_t countToEmplaceHypothetically ) const = 0;

    /**
     * @param indexToEvict If an index is given, that index will be removed if it exists instead of using
     *                     the cache strategy.
     */
    virtual std::optional<Index>
    evict( std::optional<Index> indexToEvict = {} ) = 0;

    /** Forget all tracked keys without affecting cache statistics. */
    virtual void
    clear() = 0;
};


template<typename Index>
class LeastRecentlyUsed :
    public CacheStrategy<Index>
{
public:
    using Nonce = uint64_t;

public:
    LeastRecentlyUsed() = default;

    void
    touch( Index index ) override
    {
        ++usageNonce;
        auto [match, wasInserted] = m_lastUsage.try_emplace( std::move( index ), usageNonce );
        if ( !wasInserted ) {
            m_sortedIndexes.erase( match->second );
            match->second = usageNonce;
        }
        m_sortedIndexes.emplace( usageNonce, index );
    }

    [[nodiscard]] std::optional<Index>
    nextEviction() const override
    {
        return m_sortedIndexes.empty() ? std::nullopt : std::make_optional( m_sortedIndexes.begin()->second );
    }

    [[nodiscard]] std::optional<Index>
    nextNthEviction( size_t countToEmplaceHypothetically ) const override
    {
        return ( countToEmplaceHypothetically == 0 ) || ( countToEmplaceHypothetically > m_sortedIndexes.size() )
               ? std::nullopt
               : std::make_optional( std::next( m_sortedIndexes.begin(), countToEmplaceHypothetically - 1 )->second );
    }

    std::optional<Index>
    evict( std::optional<Index> indexToEvict = {} ) override
    {
        auto evictedIndex = indexToEvict ? indexToEvict : nextEviction();
        if ( evictedIndex ) {
            const auto existingEntry = m_lastUsage.find( *evictedIndex );
            if ( existingEntry != m_lastUsage.end() ) {
                m_sortedIndexes.erase( existingEntry->second );
                m_lastUsage.erase( existingEntry );
            }
        }
        return evictedIndex;
    }

    void
    clear() override
    {
        m_lastUsage.clear();
        m_sortedIndexes.clear();
        usageNonce = 0;
    }

private:
    /* With this, inserting will be relatively fast but eviction make take longer
     * because we have to go over all elements. */
    std::unordered_map<Index, Nonce> m_lastUsage;

    /**
     * Keep a map of values sorted by nonce, i.e., by timestamp. A multimap is not necessary because nonces should
     * should be unique. m_sortedIndexes.begin holds the least recent index.
     */
    std::map<Nonce, Index> m_sortedIndexes;

    Nonce usageNonce{ 0 };
};
}


/**
 * @ref get and @ref insert should be sufficient for simple cache usages.
 * For advanced control, there are also @ref touch, @ref clear, @ref evict, and @ref test available.
 */
template<
    typename Key,
    typename Value,
    typename CacheStrategy = CacheStrategy::LeastRecentlyUsed<Key>
>
class Cache
{
public:
    struct Statistics
    {
        size_t hits{ 0 };
        size_t misses{ 0 };
        size_t unusedEntries{ 0 };
        size_t capacity{ 0 };
        size_t maxSize{ 0 };
    };

public:
    explicit
    Cache( size_t maxCacheSize ) :
        m_maxCacheSize( maxCacheSize )
    {}

    [[nodiscard]] std::optional<Value>
    get( const Key& key )
    {
        if ( const auto match = m_cache.find( key ); match != m_cache.end() ) {
            ++m_statistics.hits;
            ++m_accesses[key];
            m_cacheStrategy.touch( key );
            return match->second;
        }

        ++m_statistics.misses;
        return std::nullopt;
    }

    void
    insert( Key   key,
            Value value )
    {
        if ( capacity() == 0 ) {
            return;
        }

        /* If an entry with the same key already exists, then we can simply replace it without evicting anything.
         * Do not use try_emplace here because that could temporarily exceed the allotted capacity. */
        if ( const auto existingEntry = m_cache.find( key ); existingEntry == m_cache.end() ) {
            shrinkTo( capacity() - 1 );
            m_cache.emplace( key, std::move( value ) );
            m_statistics.maxSize = std::max( m_statistics.maxSize, m_cache.size() );
        } else {
            existingEntry->second = std::move( value );
        }

        if ( const auto match = m_accesses.find( key ); match == m_accesses.end() ) {
            m_accesses[key] = 0;
        }

        m_cacheStrategy.touch( key );
    }

    /* Advanced Control and Usage */

    void
    touch( const Key& key )
    {
        if ( test( key ) ) {
            m_cacheStrategy.touch( key );
        }
    }

    [[nodiscard]] bool
    test( const Key& key ) const
    {
        return m_cache.find( key ) != m_cache.end();
    }

    void
    clear()
    {
        m_cache.clear();
        m_accesses.clear();
        m_cacheStrategy.clear();
    }

    void
    evict( const Key& key )
    {
        m_cacheStrategy.evict( key );
        m_cache.erase( key );
    }

    /**
     * @return The next eviction, if any is necessary, when hypothetically inserting the specified index.
     */
    [[nodiscard]] std::optional<Key>
    nextEviction( const std::optional<Key>& key = std::nullopt ) const
    {
        if ( ( m_cache.size() < capacity() ) || ( key.has_value() && ( m_cache.find( *key ) == m_cache.end() ) ) ) {
            return std::nullopt;
        }
        return m_cacheStrategy.nextEviction();
    }

    [[nodiscard]] std::optional<Key>
    nextNthEviction( size_t countToBeInserted ) const
    {
        const auto freeCapacity = capacity() - m_cache.size();
        return countToBeInserted <= freeCapacity
               ? std::nullopt
               : m_cacheStrategy.nextNthEviction( countToBeInserted - freeCapacity );
    }

    void
    shrinkTo( size_t newSize )
    {
        while ( m_cache.size() > newSize ) {
            const auto toEvict = m_cacheStrategy.evict();
            assert( toEvict );
            const auto keyToEvict = toEvict ? *toEvict : m_cache.begin()->first;
            m_cache.erase( keyToEvict );

            if ( const auto match = m_accesses.find( keyToEvict ); match != m_accesses.end() ) {
                if ( match->second == 0 ) {
                    m_statistics.unusedEntries++;
                }
                m_accesses.erase( match );
            }
        }
    }

    /* Analytics */

    [[nodiscard]] Statistics
    statistics() const
    {
        auto result = m_statistics;
        result.capacity = capacity();
        return result;
    }

    void
    resetStatistics()
    {
        m_statistics = Statistics{};
        m_accesses.clear();
    }

    [[nodiscard]] size_t
    capacity() const
    {
        return m_maxCacheSize;
    }

    [[nodiscard]] size_t
    size() const
    {
        return m_cache.size();
    }

    [[nodiscard]] const CacheStrategy&
    cacheStrategy() const
    {
        return m_cacheStrategy;
    }

    [[nodiscard]] const auto&
    contents() const noexcept
    {
        return m_cache;
    }

private:
    CacheStrategy m_cacheStrategy;
    size_t const m_maxCacheSize;
    std::unordered_map<Key, Value> m_cache;

    /* Analytics */
    Statistics m_statistics;
    std::unordered_map<Key, size_t> m_accesses;
};
}  // namespace rapidgzip

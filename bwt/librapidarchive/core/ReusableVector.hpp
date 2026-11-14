#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace rapidgzip
{
/**
 * Allocated storage with a capacity but no size. Neither the pool nor @ref ReusableVector constructs or
 * value-initializes the elements on reuse.
 */
template<typename T>
struct ReusableBuffer
{
    std::unique_ptr<T[]> data;
    size_t capacity{ 0 };
};


/**
 * A small free list of buffers shared by all users of a @ref ReusableVector. Reusing the largest allocation
 * avoids mapping and faulting it in again. The pool is intentionally bounded because live vectors already
 * retain their buffers.
 */
template<typename T>
class ReusableBufferPool
{
public:
    struct Statistics
    {
        size_t acquireCalls{ 0 };
        size_t acquireHits{ 0 };
        size_t acquireMisses{ 0 };
        size_t allocatedBytes{ 0 };
        size_t reusedBytes{ 0 };
        size_t releaseCalls{ 0 };
        size_t releaseStored{ 0 };
        size_t releaseEvicted{ 0 };
        size_t freeBufferCount{ 0 };
        size_t freeBufferBytes{ 0 };
        size_t peakFreeBufferBytes{ 0 };

        [[nodiscard]] std::string
        print() const
        {
            std::stringstream out;
            out << "\n        Acquires                      : " << acquireCalls
                << "\n        Acquire Hits                  : " << acquireHits
                << "\n        Acquire Misses                : " << acquireMisses
                << "\n        Allocated Bytes               : " << allocatedBytes
                << "\n        Reused Bytes                  : " << reusedBytes
                << "\n        Release Calls                 : " << releaseCalls
                << "\n        Stored                        : " << releaseStored
                << "\n        Evicted                       : " << releaseEvicted
                << "\n        Free Buffers                  : " << freeBufferCount
                << "\n        Free Bytes                    : " << freeBufferBytes
                << "\n        Peak Free Bytes               : " << peakFreeBufferBytes;
            return out.str();
        }
    };

    ReusableBufferPool()
    {
        m_free.reserve( MAX_BUFFERS );
    }

    void
    setStatisticsEnabled( bool enabled )
    {
        m_statisticsEnabled = enabled;
    }

    [[nodiscard]] bool
    statisticsEnabled() const noexcept
    {
        return m_statisticsEnabled;
    }

    void
    setShowProfileOnDestruction( bool show )
    {
        m_showProfileOnDestruction = show;
    }

    ~ReusableBufferPool()
    {
        if ( !m_showProfileOnDestruction ) {
            return;
        }

        /* Destructors are implicitly noexcept. A lock or I/O failure must not terminate the process. */
        try {
            std::cerr << "[ReusableBufferPool]" << statistics().print() << '\n';
        } catch ( ... ) {}
    }

    [[nodiscard]] ReusableBuffer<T>
    acquire( size_t minCapacity )
    {
        const std::scoped_lock lock( m_mutex );
        if ( m_statisticsEnabled ) {
            ++m_statistics.acquireCalls;
        }

        auto best = m_free.end();
        for ( auto it = m_free.begin(); it != m_free.end(); ++it ) {
            if ( ( it->capacity >= minCapacity ) &&
                 ( ( best == m_free.end() ) || ( it->capacity < best->capacity ) ) ) {
                best = it;
            }
        }
        if ( best == m_free.end() ) {
            if ( m_statisticsEnabled ) {
                ++m_statistics.acquireMisses;
                m_statistics.allocatedBytes += minCapacity * sizeof( T );
            }
            return {};
        }

        if ( m_statisticsEnabled ) {
            ++m_statistics.acquireHits;
            m_statistics.reusedBytes += best->capacity * sizeof( T );
        }
        auto result = std::move( *best );
        m_free.erase( best );
        updateFreeStatistics();
        return result;
    }

    void
    release( ReusableBuffer<T> buffer ) noexcept
    {
        if ( !buffer.data ) {
            return;
        }

        /* Destructors must not terminate, so a failed bookkeeping allocation simply frees the buffer. */
        try {
            const std::scoped_lock lock( m_mutex );
            if ( m_statisticsEnabled ) {
                ++m_statistics.releaseCalls;
            }
            if ( m_free.size() >= MAX_BUFFERS ) {
                if ( m_statisticsEnabled ) {
                    ++m_statistics.releaseEvicted;
                }
                m_free.erase( m_free.begin() );
            }
            m_free.emplace_back( std::move( buffer ) );
            if ( m_statisticsEnabled ) {
                ++m_statistics.releaseStored;
            }
            updateFreeStatistics();
        } catch ( ... ) {}
    }

    [[nodiscard]] Statistics
    statistics() const
    {
        const std::scoped_lock lock( m_mutex );
        auto result = m_statistics;
        result.freeBufferCount = m_free.size();
        result.freeBufferBytes = freeBufferBytes();
        result.peakFreeBufferBytes = std::max( result.peakFreeBufferBytes, result.freeBufferBytes );
        return result;
    }

private:
    void
    updateFreeStatistics() noexcept
    {
        const auto bytes = freeBufferBytes();
        m_statistics.freeBufferCount = m_free.size();
        m_statistics.freeBufferBytes = bytes;
        m_statistics.peakFreeBufferBytes = std::max( m_statistics.peakFreeBufferBytes, bytes );
    }

    [[nodiscard]] size_t
    freeBufferBytes() const noexcept
    {
        size_t result = 0;
        for ( const auto& buffer : m_free ) {
            result += buffer.capacity * sizeof( T );
        }
        return result;
    }

private:
    static constexpr size_t MAX_BUFFERS = 8;

    mutable std::mutex m_mutex;
    std::vector<ReusableBuffer<T> > m_free;
    Statistics m_statistics;
    std::atomic<bool> m_statisticsEnabled{ false };
    bool m_showProfileOnDestruction{ false };
};


/**
 * Resizable buffer whose allocation can be recycled through a @ref ReusableBufferPool. In contrast to
 * std::vector::resize, growing does not value-initialize the new elements. The caller must overwrite every
 * element up to the requested size before reading it.
 */
template<typename T>
class ReusableVector
{
public:
    ReusableVector() = default;

    explicit
    ReusableVector( std::shared_ptr<ReusableBufferPool<T> > pool ) :
        m_pool( std::move( pool ) )
    {}

    ~ReusableVector()
    {
        release();
    }

    ReusableVector( const ReusableVector& ) = delete;

    ReusableVector&
    operator=( const ReusableVector& ) = delete;

    ReusableVector( ReusableVector&& other ) noexcept :
        m_pool( std::move( other.m_pool ) ),
        m_data( std::move( other.m_data ) ),
        m_size( other.m_size ),
        m_capacity( other.m_capacity )
    {
        other.m_size = 0;
        other.m_capacity = 0;
    }

    ReusableVector&
    operator=( ReusableVector&& other ) noexcept
    {
        if ( this != &other ) {
            release();
            m_pool = std::move( other.m_pool );
            m_data = std::move( other.m_data );
            m_size = other.m_size;
            m_capacity = other.m_capacity;
            other.m_size = 0;
            other.m_capacity = 0;
        }
        return *this;
    }

    void
    resize( size_t size )
    {
        if ( size > m_capacity ) {
            grow( size );
        }
        m_size = size;
    }

    [[nodiscard]] T*
    data() noexcept
    {
        return m_data.get();
    }

    [[nodiscard]] const T*
    data() const noexcept
    {
        return m_data.get();
    }

    [[nodiscard]] size_t
    size() const noexcept
    {
        return m_size;
    }

    [[nodiscard]] bool
    empty() const noexcept
    {
        return m_size == 0;
    }

private:
    void
    grow( size_t minCapacity )
    {
        /* Round the allocation to a 4 KiB granularity so that it is page-aligned and the pool does not
         * accumulate near-identical capacities. The logical size remains the requested one; the tail is
         * never read. */
        constexpr size_t ALLOCATION_GRANULARITY = 4 * 1024;
        const auto capacityInBytes = ( minCapacity * sizeof( T ) + ALLOCATION_GRANULARITY - 1 )
                                     / ALLOCATION_GRANULARITY * ALLOCATION_GRANULARITY;
        const auto capacity = ( capacityInBytes + sizeof( T ) - 1 ) / sizeof( T );

        auto buffer = m_pool ? m_pool->acquire( capacity ) : ReusableBuffer<T>{};
        if ( !buffer.data ) {
            /* Default-initialized, i.e., the caller overwrites every element before it is read. */
            buffer.data = std::unique_ptr<T[]>( new T[capacity] );
            buffer.capacity = capacity;
        }

        if ( m_data ) {
            std::copy_n( m_data.get(), std::min( m_size, buffer.capacity ), buffer.data.get() );
        }

        if ( m_pool && m_data ) {
            m_pool->release( { std::move( m_data ), m_capacity } );
        }

        m_data = std::move( buffer.data );
        m_capacity = buffer.capacity;
        m_size = std::min( m_size, m_capacity );
    }

    void
    release() noexcept
    {
        if ( m_pool && m_data ) {
            m_pool->release( { std::move( m_data ), m_capacity } );
        } else {
            m_data.reset();
        }
        m_size = 0;
        m_capacity = 0;
    }

private:
    std::shared_ptr<ReusableBufferPool<T> > m_pool;
    std::unique_ptr<T[]> m_data;
    size_t m_size{ 0 };
    size_t m_capacity{ 0 };
};
}  // namespace rapidgzip

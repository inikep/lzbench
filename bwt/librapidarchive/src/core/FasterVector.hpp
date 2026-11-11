#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#if !defined( LIBRAPIDARCHIVE_WITH_RPMALLOC )
    #include <cstdlib>
#endif
#include <iterator>
#include <optional>
#include <stdexcept>
#include <vector>

#include "common.hpp"  // ceilDiv

#ifdef LIBRAPIDARCHIVE_WITH_RPMALLOC
    #include <rpmalloc.h>
#endif

namespace rapidgzip
{
#ifdef LIBRAPIDARCHIVE_WITH_RPMALLOC
class RpmallocInit
{
public:
    RpmallocInit()
    {
        rpmalloc_initialize( nullptr );
    }

    ~RpmallocInit()
    {
    #ifdef LIBRAPIDARCHIVE_RPMALLOC_STATISTICS
        rpmalloc_dump_statistics( stderr );
    #endif
        rpmalloc_finalize();
    }
};


/* It must be the very first static variable so that rpmalloc is initialized before any usage of malloc
 * when overriding operator new (when including rpnew.h). And, this only works if everything is header-only
 * because else the static variable initialization order becomes undefined across different compile units.
 * That's why we avoid overriding operators new and delete and simply use it as a custom allocator in the
 * few places we know to be performance-critical */
inline static const RpmallocInit rpmallocInit{};


class RpmallocThreadInit
{
public:
    RpmallocThreadInit()
    {
        rpmalloc_thread_initialize();
    }

    ~RpmallocThreadInit()
    {
    #ifdef LIBRAPIDARCHIVE_RPMALLOC_STATISTICS
        rpmalloc_dump_statistics( stderr );
    #endif
        /* Collecting first releases blocks other threads freed into this thread's heaps. Finalizing alone
         * only queues the heap, and a queued heap that is never adopted keeps its free pages committed. */
        rpmalloc_thread_collect();
        rpmalloc_thread_finalize();
    }
};


inline void*
rpmalloc_ensuring_initialization( size_t nBytes = 0 )
{
    static const thread_local RpmallocThreadInit rpmallocThreadInit{};
    if ( nBytes == 0 ) {
        return nullptr;
    }
    return rpmalloc( nBytes );
}


template<typename ElementType>
class RpmallocAllocator
{
public:
    using value_type = ElementType;

    using is_always_equal = std::true_type;

public:
    [[nodiscard]] constexpr ElementType*
    allocate( std::size_t nElementsToAllocate )
    {
        if ( nElementsToAllocate > std::numeric_limits<std::size_t>::max() / sizeof( ElementType ) ) {
            throw std::bad_array_new_length();
        }

        auto const nBytesToAllocate = nElementsToAllocate * sizeof( ElementType );
        return reinterpret_cast<ElementType*>( rpmalloc_ensuring_initialization( nBytesToAllocate ) );
    }

    constexpr void
    deallocate( ElementType*                 allocatedPointer,
                [[maybe_unused]] std::size_t nElementsAllocated )
    {
        rpfree( allocatedPointer );
    }

    /* I don't understand why this is still necessary even with is_always_equal = true_type.
     * Defining it to true type should be necessary because the default implementation of
     * is_always_equal == is_empty should also be true because this class does not have any members. */
    [[nodiscard]] bool
    operator==( const RpmallocAllocator& /* unused */ ) const
    {
        return true;
    }

    [[nodiscard]] bool
    operator!=( const RpmallocAllocator& /* unused */ ) const
    {
        return false;
    }
};


static_assert( std::is_empty_v<RpmallocAllocator<char> > );

#endif

#if 1

#ifdef LIBRAPIDARCHIVE_WITH_RPMALLOC
template<typename T>
using FasterVector = std::vector<T, RpmallocAllocator<T> >;
#else
template<typename T>
using FasterVector = std::vector<T>;
#endif

#else

[[nodiscard]] constexpr bool
isPowerOf2( size_t value )
{
    return ( value > 0 ) && ( ( value & ( value - 1U ) ) == 0U );
}


template<typename T>
using RequireInputIterator = typename std::enable_if_t<
    std::is_convertible_v<typename std::iterator_traits<T>::iterator_category, std::input_iterator_tag> >;


/**
 * This was supposed to be a faster std::vector alternative that saves time by not initializing its contents
 * on resize as introduced in 093805ab24c93b7150b56d16bde418e51c0dd970. However, it leads to almost double
 * the memory usage with wikidata.json (12 GB -> 16 GB) even when disabling rounding to powers of 2 when
 * reserving in @ref insert and when disabling alignment and disabling the shrink_to_fit check.
 * @verbatim
 * make rapidgzip
 * /usr/bin/time -v src/tools/rapidgzip -v -P 24 --io-read-method sequential --export-index "$file"{.index,}
 * With std::vector using rpmalloc allocator:
 *     Maximum resident set size (kbytes): 11631884
 * FasterVector:
 *     Maximum resident set size (kbytes): 16360928
 */
template<typename T>
class FasterVector
{
public:
    static_assert( isPowerOf2( sizeof( T ) ), "Size of element type must be a power of 2 for alignment purpose!" );
    static constexpr size_t ALIGNMENT = std::max<size_t>( sizeof( T ), 512U / 8U );

    using value_type = T;

public:
    FasterVector() = default;

    explicit
    FasterVector( size_t                  size,
                  const std::optional<T>& initialValue = {} )
    {
        resize( size, initialValue );
    }

    template<typename InputIt,
             typename = RequireInputIterator<InputIt> >
    FasterVector( InputIt inputBegin,
                  InputIt inputEnd )
    {
        insert( end(), inputBegin, inputEnd );
    }

    FasterVector( FasterVector&& other ) noexcept :
        m_data( other.m_data ),
        m_capacity( other.m_capacity ),
        m_size( other.m_size )
    {
        other.m_data = nullptr;
        other.m_capacity = 0;
        other.m_size = 0;
    }

    FasterVector&
    operator=( FasterVector&& other ) noexcept
    {
        m_data = other.m_data;
        m_capacity = other.m_capacity;
        m_size = other.m_size;

        other.m_data = nullptr;
        other.m_capacity = 0;
        other.m_size = 0;

        return *this;
    }

    /* Forbid copies because they are expensive and because they have unexpected behavior like not copying
     * the capacity. */
    FasterVector( const FasterVector& ) = delete;
    FasterVector& operator=( const FasterVector& ) = delete;

    ~FasterVector()
    {
        free();
    }

    void
    resize( size_t                  size,
            const std::optional<T>& initialValue = {} )
    {
        if ( size > m_size ) {
            reserve( size );
            if ( initialValue ) {
                std::fill( m_data + m_size, m_data + size, *initialValue );
            }
        }
        m_size = size;
    }

    void
    reserve( size_t newCapacity )
    {
        if ( newCapacity > m_capacity ) {
            reallocate( newCapacity );
        }
    }

    void clear() { m_size = 0; }
    void shrink_to_fit()
    {
        if ( m_size < static_cast<size_t>( 0.9 * m_capacity ) ) {
            reallocate( m_size );
        }
    }

    [[nodiscard]] constexpr size_t capacity() const noexcept { return m_capacity; }
    [[nodiscard]] constexpr size_t size() const noexcept { return m_size; }
    [[nodiscard]] constexpr bool empty() const noexcept { return m_size == 0; }

    [[nodiscard]] constexpr const T* data() const noexcept { return m_data; }
    [[nodiscard]] constexpr T* data() noexcept { return m_data; }

    [[nodiscard]] constexpr const T* cbegin() const noexcept { return m_data; }
    [[nodiscard]] constexpr const T* begin() const noexcept { return m_data; }
    [[nodiscard]] constexpr T* begin() noexcept { return m_data; }

    [[nodiscard]] constexpr const T* cend() const noexcept { return m_data + m_size; }
    [[nodiscard]] constexpr const T* end() const noexcept { return m_data + m_size; }
    [[nodiscard]] constexpr T* end() noexcept { return m_data + m_size; }

    [[nodiscard]] constexpr auto crbegin() const noexcept { return std::reverse_iterator<const T*>( m_data + m_size ); }
    [[nodiscard]] constexpr auto rbegin() const noexcept { return std::reverse_iterator<const T*>( m_data + m_size ); }
    [[nodiscard]] constexpr auto rbegin() noexcept { return std::reverse_iterator<T*>( m_data + m_size ); }

    [[nodiscard]] constexpr auto crend() const noexcept { return std::reverse_iterator<const T*>( m_data ); }
    [[nodiscard]] constexpr auto rend() const noexcept { return std::reverse_iterator<const T*>( m_data ); }
    [[nodiscard]] constexpr auto rend() noexcept { return std::reverse_iterator<T*>( m_data ); }

    [[nodiscard]] const T&
    operator[]( size_t i ) const
    {
        assert( i < m_size );
        return m_data[i];
    }

    [[nodiscard]] T&
    operator[]( size_t i )
    {
        assert( i < m_size );
        return m_data[i];
    }

    template<typename InputIt,
             typename = RequireInputIterator<InputIt> >
    void
    insert( const T*       position,
            const InputIt& inputBegin,
            const InputIt& inputEnd )
    {
        const auto inputDistance = inputEnd - inputBegin;
        if ( inputDistance <= 0 ) {
            return;
        }
        const auto inputSize = static_cast<size_t>( inputDistance );

        const auto positionDistance = position - m_data;
        if ( ( positionDistance < 0 ) || ( static_cast<size_t>( positionDistance ) > m_size ) ) {
            throw std::logic_error( "The insertion position must be inside the valid range of this vector or end()!" );
        }
        const auto positionIndex = static_cast<size_t>( positionDistance );

        /* Beware that reserve may invalidate "position"! Do not reallocate when there is enough space.
         * The same guarantee that std::vector provides! */
        if ( size() + inputSize > m_capacity ) {
            reserve( size_t( 1U ) << static_cast<size_t>( std::ceil( std::log2( size() + inputSize ) ) ) );
        }
        if ( positionIndex < m_size ) {
            std::memmove( m_data + positionIndex + inputSize, m_data + positionIndex, inputSize * sizeof( T ) );
        }
        std::copy( inputBegin, inputEnd, m_data + positionIndex );
        m_size += inputSize;
    }

    [[nodiscard]] bool
    operator==( const FasterVector& other ) const
    {
        return std::equal( begin(), end(), other.begin(), other.end() );
    }

    [[nodiscard]] bool
    operator!=( const FasterVector& other ) const
    {
        return !std::equal( begin(), end(), other.begin(), other.end() );
    }

    [[nodiscard]] const T&
    front() const
    {
        requireNonEmpty();
        return m_data[0];
    }

    [[nodiscard]] T&
    front()
    {
        requireNonEmpty();
        return m_data[0];
    }

    [[nodiscard]] const T&
    back() const
    {
        requireNonEmpty();
        return m_data[m_size - 1];
    }

    [[nodiscard]] T&
    back()
    {
        requireNonEmpty();
        return m_data[m_size - 1];
    }

private:
    void
    requireNonEmpty()
    {
        if ( empty() ) {
            throw std::out_of_range( "Cannot get last element of empty vector!" );
        }
    }

    void
    reallocate( const size_t newCapacity )
    {
        if ( newCapacity == m_capacity ) {
            return;
        }

        if ( newCapacity == 0 ) {
            free();
        } else {
        #ifdef LIBRAPIDARCHIVE_WITH_RPMALLOC
        #if 1
            if ( m_data == nullptr ) {
                rpmalloc_ensuring_initialization();
                m_data = static_cast<T*>( rpaligned_alloc( ALIGNMENT, newCapacity * sizeof( T ) ) );
            } else {
                m_data = static_cast<T*>( rpaligned_realloc( m_data, ALIGNMENT, newCapacity * sizeof( T ),
                                                             m_capacity * sizeof( T ), /* flags */ 0 ) );
            }
        #else
            if ( m_data == nullptr ) {
                m_data = static_cast<T*>( rpmalloc_ensuring_initialization( newCapacity * sizeof( T ) ) );
            } else {
                m_data = static_cast<T*>( rprealloc( m_data, newCapacity * sizeof( T ) ) );
            }
        #endif
        #else
            /* > If ptr is a null pointer, the behavior is the same as calling std::malloc(new_size). */
            m_data = static_cast<T*>( std::realloc( m_data, newCapacity * sizeof( T ) ) );  // NOLINT
        #endif
        }

        m_capacity = newCapacity;
    }

    void
    free()
    {
    #ifdef LIBRAPIDARCHIVE_WITH_RPMALLOC
        rpfree( m_data );
    #else
        std::free( m_data );  // NOLINT
    #endif
        m_data = nullptr;
    }

private:
    T* m_data{ nullptr };
    size_t m_capacity{ 0 };
    size_t m_size{ 0 };
};


template<class T, class Alloc>
[[nodiscard]] bool
operator==( const FasterVector<T>&       lhs,
            const std::vector<T, Alloc>& rhs )
{
    return std::equal( lhs.begin(), lhs.end(), rhs.begin(), rhs.end() );
}


template<class T, class Alloc>
[[nodiscard]] bool
operator==( const std::vector<T, Alloc>& lhs,
            const FasterVector<T>&       rhs )
{
    return std::equal( lhs.begin(), lhs.end(), rhs.begin(), rhs.end() );
}


template<class T, class Alloc>
[[nodiscard]] bool
operator!=( const FasterVector<T>&       lhs,
            const std::vector<T, Alloc>& rhs )
{
    return !std::equal( lhs.begin(), lhs.end(), rhs.begin(), rhs.end() );
}


template<class T, class Alloc>
[[nodiscard]] bool
operator!=( const std::vector<T, Alloc>& lhs,
            const FasterVector<T>&       rhs )
{
    return !std::equal( lhs.begin(), lhs.end(), rhs.begin(), rhs.end() );
}
#endif
}  // namespace rapidgzip

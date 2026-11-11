#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef WITH_PYTHON_SUPPORT
    #include <Python.h>
#endif

/* Platform dependent stuff */

#ifdef _MSC_VER
    #include <io.h>

    /* MSVC still has a fileno alias even though it is deprecated. Using define is not a good idea because some
     * methods are named fileno! Instead, disable the deprecation warning for this and if a more recent compiler
     * removes the alias, then use: const auto fileno = _fileno; */
    #pragma warning(disable : 4996)

    /* Ignore warnings about [[likely]], [[unlikely]], which will only work with C++20. */
    #pragma warning(disable : 5051)

    #include <sys/stat.h>

    #define S_IFIFO _S_IFIFO
    #define S_IFMT _S_IFMT

namespace rapidgzip
{
template<typename FileMode, typename FileType>
[[nodiscard]] bool
testFileType( FileMode fileMode,
              FileType fileType )
{
    return ( fileMode & S_IFMT ) == fileType;
}
}      // namespace rapidgzip;

    #define S_ISFIFO( m ) testFileType( m, S_IFIFO )

    #define CONSTEXPR_EXCEPT_MSVC
#else
    #include <unistd.h>

    #define CONSTEXPR_EXCEPT_MSVC constexpr
#endif


#if defined( _MSC_VER )
    #define forceinline __forceinline
#elif defined( __clang__ ) || defined( __GNUC__ )
/* https://stackoverflow.com/questions/38499462/how-to-tell-clang-to-stop-pretending-to-be-other-compilers */
/* *INDENT-OFF* */
    #define forceinline __attribute__(( always_inline )) inline
/* *INDENT-ON* */
#endif

namespace rapidgzip
{
template<typename I1,
         typename I2,
         typename Enable = typename std::enable_if_t<std::is_integral_v<I1> && std::is_integral_v<I2>> >
[[nodiscard]] constexpr I1
ceilDiv( I1 dividend,
         I2 divisor ) noexcept
{
    return ( dividend + divisor - 1 ) / divisor;
}


/**
 * Absolute difference function that also works for unsigned types for which std::abs(a-b) wouldn't work.
 */
template<typename T, typename V>
[[nodiscard]] T
absDiff( const T& a,
         const V& b )
{
    return a < b ? b - a : a - b;
}


template<typename U,
         std::enable_if_t<std::is_unsigned_v<U> >* = nullptr>
[[nodiscard]] constexpr U
saturatingAddition( const U a,
                    const U b )
{
    return a > std::numeric_limits<U>::max() - b ? std::numeric_limits<U>::max() : a + b;
}


template<typename U,
         std::enable_if_t<std::is_signed_v<U> >* = nullptr>
[[nodiscard]] constexpr U
saturatingAddition( const U a,
                    const U b )
{
    /* Underflow or overflow should only be possible when both values have the same sign! */
    if ( ( a > 0 ) && ( b > 0 ) ) {
        return a > std::numeric_limits<U>::max() - b ? std::numeric_limits<U>::max() : a + b;
    }

    if ( ( a < 0 ) && ( b < 0 ) ) {
        return a < std::numeric_limits<U>::lowest() - b ? std::numeric_limits<U>::lowest() : a + b;
    }

    return a + b;
}


template<typename S, typename T>
std::ostream&
operator<<( std::ostream&   out,
            std::pair<S, T> pair )
{
    out << "(" << pair.first << "," << pair.second << ")";
    return out;
}


template<typename T>
std::ostream&
operator<<( std::ostream&         out,
            const std::vector<T>& vector )
{
    if ( vector.empty() ) {
        out << "{}";
        return out;
    }

    out << "{ ";
    for ( auto value = vector.begin(); value != vector.end(); ++value ) {
        if ( value != vector.begin() ) {
            out << ", ";
        }
        if constexpr ( std::is_same_v<T, uint8_t> ) {
            out << static_cast<uint16_t>( *value );
        } else {
            out << *value;
        }
    }
    out << " }";

    return out;
}


template<typename S, typename T>
[[nodiscard]] constexpr bool
startsWith( const S& fullString,
            const T& prefix,
            bool     caseSensitive = true ) noexcept
{
    if ( fullString.size() < prefix.size() ) {
        return false;
    }

    if ( caseSensitive ) {
        return std::equal( prefix.begin(), prefix.end(), fullString.begin() );
    }

    return std::equal( prefix.begin(), prefix.end(), fullString.begin(),
                       [] ( auto a, auto b ) { return std::tolower( a ) == std::tolower( b ); } );
}


template<typename S, typename T>
[[nodiscard]] constexpr bool
endsWith( const S& fullString,
          const T& suffix,
          bool     caseSensitive = true ) noexcept
{
    if ( fullString.size() < suffix.size() ) {
        return false;
    }

    if ( caseSensitive ) {
        return std::equal( suffix.rbegin(), suffix.rend(), fullString.rbegin() );
    }

    return std::equal( suffix.rbegin(), suffix.rend(), fullString.rbegin(),
                       [] ( auto a, auto b ) { return std::tolower( a ) == std::tolower( b ); } );
}


[[nodiscard]] inline std::vector<std::string_view>
split( const std::string_view toSplit,
       const char             separator )
{
    std::vector<std::string_view> result;
    auto start = toSplit.begin();  // NOLINT(readability-qualified-auto)
    for ( auto it = toSplit.begin(); it != toSplit.end(); ++it ) {  // NOLINT(readability-qualified-auto)
        if ( *it == separator ) {
            result.emplace_back( toSplit.data() + std::distance( toSplit.begin(), start ),
                                 std::distance( start, it ) );
            start = it;
            ++start;
        }
    }

    if ( start != toSplit.end() ) {
        result.emplace_back( toSplit.data() + std::distance( toSplit.begin(), start ),
                             std::distance( start, toSplit.end() ) );
    }

    return result;
}


[[nodiscard]] inline std::string
formatBits( const uint64_t value )
{
    return std::to_string( value / 8 ) + " B " + std::to_string( value % 8 ) + " b";
}


[[nodiscard]] inline std::string
formatBytes( const uint64_t value )
{
    const std::array<std::pair<std::string_view, uint64_t>, 7U> UNITS{ {
        /* 64-bit maximum is 16 EiB, so these units cover all cases. */
        { "EiB", 1024ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL },
        { "PiB", 1024ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL },
        { "TiB", 1024ULL * 1024ULL * 1024ULL * 1024ULL },
        { "GiB", 1024ULL * 1024ULL * 1024ULL },
        { "MiB", 1024ULL * 1024ULL },
        { "KiB", 1024ULL },
        { "B", 1ULL },
    } };

    std::stringstream result;
    for ( const auto& [unit, multiplier] : UNITS ) {
        const auto remainder = ( value / multiplier ) % 1024ULL;
        if ( remainder != 0 ) {
            if ( result.tellp() > 0 ) {
                result << " ";
            }
            result << remainder << " " << unit;
        }
    }

    if ( result.tellp() == 0 ) {
        return "0 B";
    }

    return std::move( result ).str();
}


[[nodiscard]] inline std::chrono::time_point<std::chrono::high_resolution_clock>
now() noexcept
{
    return std::chrono::high_resolution_clock::now();
}


/**
 * @return duration in seconds
 */
template<typename T>
[[nodiscard]] double
duration( const T& t0,
          const T& t1 = now() ) noexcept
{
    return std::chrono::duration<double>( t1 - t0 ).count();
}


[[nodiscard]] inline uint64_t
unixTimeInNanoseconds() noexcept
{
    const auto currentTime = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<uint64_t>( std::chrono::duration_cast<std::chrono::nanoseconds>( currentTime ).count() );
}


/**
 * Use like this:
 * @verbatim
 * std::cerr << ( ThreadSafeOutput() << "Hello" << i << "there" ).str();
 * @endverbatim
 */
class ThreadSafeOutput
{
public:
    ThreadSafeOutput()
    {
        using namespace std::chrono;
        const auto time = system_clock::now();
        const auto timePoint = system_clock::to_time_t( time );
        const auto subseconds = duration_cast<milliseconds>( time.time_since_epoch() ).count() % 1000;
        m_out << "[" << std::put_time( std::localtime( &timePoint ), "%H:%M:%S" ) << "." << subseconds << "]"
              << "[0x" << std::hex << std::this_thread::get_id() << std::dec << "]";
    }

    template<typename T>
    ThreadSafeOutput&
    operator<<( const T& value )
    {
        m_out << " " << value;
        return *this;
    }

    operator std::string() const
    {
        return m_out.str() + "\n";
    }

    [[nodiscard]] std::string
    str() const
    {
        return m_out.str() + "\n";
    }

private:
    std::stringstream m_out;
};


inline std::ostream&
operator<<( std::ostream&           out,
            const ThreadSafeOutput& output )
{
    out << output.str();
    return out;
}


[[nodiscard]] inline std::string
toString( std::future_status status ) noexcept
{
    switch ( status )
    {
    case std::future_status::ready:
        return "ready";
    case std::future_status::deferred:
        return "deferred";
    case std::future_status::timeout:
        return "timeout";
    }
    return "unknown future states";
}


/**
 * RAII based notify at the end of a scope, which will also be triggered e.g. when throwing exceptions!
 * std::notify_all_at_thread_exit is no alternative to this because the thread does not exit because we are using
 * a thread pool but we might still want to notify someone when the packaged task throws.
 */
class FinallyNotify
{
public:
    explicit
    FinallyNotify( std::condition_variable& toNotify ) :
        m_toNotify( toNotify )
    {}

    ~FinallyNotify()
    {
        m_toNotify.notify_all();
    }

private:
    std::condition_variable& m_toNotify;
};


class Finally
{
public:
    explicit
    Finally( std::function<void()> cleanup ) :
        m_cleanup( std::move( cleanup ) )
    {}

    /**
     * Runs the cleanup on both normal exit and stack unwinding. If the cleanup throws during unwinding,
     * the original exception is preserved; on normal exit, cleanup errors propagate like a Python/Java
     * finally block.
     */
    ~Finally() noexcept( false )
    {
        if ( !m_cleanup ) {
            return;
        }
        if ( std::uncaught_exceptions() > 0 ) {
            try {
                m_cleanup();
            } catch ( ... ) {}
        } else {
            m_cleanup();
        }
    }

private:
    std::function<void()> m_cleanup;
};


template<typename T>
[[nodiscard]] constexpr typename T::value_type
getMinPositive( const T& container )
{
    if ( container.empty() ) {
        throw std::invalid_argument( "Container must not be empty!" );
    }

    auto result = container.front();
    for ( const auto value : container ) {
        if ( value > 0 ) {
            if ( result > 0 ) {
                result = std::min( result, value );
            } else {
                result = value;
            }
        }
    }
    return result;
}


template<typename T>
[[nodiscard]] constexpr typename T::value_type
getMax( const T& container )
{
    const auto match = std::max_element( container.begin(), container.end() );
    if ( match == container.end() ) {
        throw std::invalid_argument( "Container must not be empty!" );
    }
    return *match;
}


template<typename Container,
         typename Value>
[[nodiscard]] constexpr bool
contains( const Container& container,
          Value            value )
{
    return std::find( container.begin(), container.end(), value ) != container.end();
}


template<typename Iterator,
         typename T = const decltype( *std::declval<Iterator>() ) &>
[[nodiscard]] constexpr size_t
countAdjacentIf( const Iterator                                                              rangeBegin,
                 const Iterator                                                              rangeEnd,
                 const std::function<bool( const decltype( *std::declval<Iterator>() )&,
                                           const decltype( *std::declval<Iterator>() ) & )>& equal )
{
    size_t result{ 0 };
    if ( rangeBegin != rangeEnd ) {
        for ( auto it = rangeBegin, nit = std::next( it ); nit != rangeEnd; ++it, ++nit ) {
            if ( equal( *it, *nit ) ) {
                ++result;
            }
        }
    }
    return result;
}


/**
 * Returns iterators to the first sequence of elements that are increasing by one each.
 */
template<typename Iterator>
[[nodiscard]] constexpr std::pair<Iterator, Iterator>
findAdjacentIf( const Iterator                                                              rangeBegin,
                const Iterator                                                              rangeEnd,
                const std::function<bool( const decltype( *std::declval<Iterator>() )&,
                                          const decltype( *std::declval<Iterator>() ) & )>& equal )
{
    auto sequenceBegin = rangeEnd;
    if ( rangeBegin != rangeEnd ) {
        for ( auto it = rangeBegin, nit = std::next( it ); nit != rangeEnd; ++it, ++nit ) {
            if ( equal( *it, *nit ) ) {
                if ( sequenceBegin == rangeEnd ) {
                    sequenceBegin = it;
                }
            } else if ( sequenceBegin != rangeEnd ) {
                return { sequenceBegin, nit };
            }
        }
    }
    return { sequenceBegin, rangeEnd };
}


template<typename Container>
[[nodiscard]] Container
interleave( const std::vector<Container>& values )
{
    const auto size = std::accumulate( values.begin(), values.end(), size_t{ 0 },
                                       [] ( size_t sum, const auto& container ) { return sum + container.size(); } );
    Container result;
    result.reserve( size );
    for ( size_t i = 0; i < size; ++i ) {
        for ( const auto& container : values ) {
            if ( i < container.size() ) {
                result.emplace_back( container[i] );
            }
        }
    }
    return result;
}


[[nodiscard]] inline bool
testFlags( const uint64_t value,
           const uint64_t flags )
{
    return ( value & flags ) != 0;
}


// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#if defined( __GNUC__ )
    #define LIKELY( x ) ( __builtin_expect( static_cast<bool>( x ), 1 ) )
    #define UNLIKELY( x ) ( __builtin_expect( static_cast<bool>( x ), 0 ) )
#else
    #define LIKELY( x ) ( x )
    #define UNLIKELY( x ) ( x )
#endif
// NOLINTEND(cppcoreguidelines-macro-usage)


enum class Endian
{
    LITTLE,
    BIG,
    UNKNOWN,
};


/**
 * g++-dM -E -x c++ /dev/null | grep -i endian
 * > #define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
 * clang++ -dM -E -x c++ /dev/null | grep -i little
 * > #define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
 */
constexpr Endian ENDIAN =
#if defined( __BYTE_ORDER__ ) && defined( __ORDER_LITTLE_ENDIAN__ ) && ( __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ )
    Endian::LITTLE
#elif defined( __BYTE_ORDER__ ) && defined( __ORDER_BIG_ENDIAN__ ) && ( __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__ )
    Endian::BIG
#else
    Endian::UNKNOWN
#endif
;


/**
 * This should compile to a single load on modern compilers instead of a function call. Test e.g. on godbolt.org.
 * Note that we cannot use reinterpret_cast from char* to uint64_t* because it would result in unedefined behavior
 * because of strict-aliasing rules! @see https://en.cppreference.com/w/cpp/language/reinterpret_cast#Type_aliasing
 */
template<typename T>
[[nodiscard]] constexpr T
loadUnaligned( const void* data )
{
    T result{ 0 };
    if constexpr ( ENDIAN == Endian::LITTLE ) {
        std::memcpy( &result, data, sizeof( result ) );
    } else {
        const auto* bytes = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < sizeof(T); ++i) {
            result |= static_cast<T>(bytes[i]) << (i * 8);
        }
    }
    return result;
}


[[nodiscard]] inline size_t
countNewlines( const std::string_view& view )
{
    return std::count( view.begin(), view.end(), '\n' );
}


struct FindNthNewlineResult
{
    std::size_t position{ std::string_view::npos };
    uint64_t remainingLineCount{ 0 };

    [[nodiscard]] bool
    operator==( const FindNthNewlineResult& other ) const noexcept
    {
        return ( position == other.position ) && ( remainingLineCount == other.remainingLineCount );
    }

    [[nodiscard]] bool
    operator!=( const FindNthNewlineResult& other ) const noexcept
    {
        return !( *this == other );
    }
};


inline std::ostream&
operator<<( std::ostream&               out,
            const FindNthNewlineResult& result )
{
    out << "( position: " << result.position << ", remaining line count: " << result.remainingLineCount << ")";
    return out;
}


/**
 * @return Position of the nth newline character or std::string_view::npos and the remaining line count.
 *         If @p lineCount is 0, always returns 0 by definition.
 */
[[nodiscard]] constexpr FindNthNewlineResult
findNthNewline( const std::string_view& view,
                const uint64_t          lineCount,
                const char              newlineCharacter = '\n' )
{
    const std::string_view toFind( std::addressof( newlineCharacter ), 1 );

    FindNthNewlineResult result;
    result.remainingLineCount = lineCount;
    while ( result.remainingLineCount > 0 ) {
        result.position = view.find( toFind, result.position == std::string_view::npos ? 0 : result.position + 1 );
        if ( result.position == std::string_view::npos ) {
            return result;
        }
        --result.remainingLineCount;
    }

    return result;
}


[[nodiscard]] constexpr uint64_t
operator ""_Ki( unsigned long long int value ) noexcept
{
    return value * 1024ULL;
}


[[nodiscard]] constexpr uint64_t
operator ""_Mi( unsigned long long int value ) noexcept
{
    return value * 1024ULL * 1024ULL;
}


[[nodiscard]] constexpr uint64_t
operator ""_Gi( unsigned long long int value ) noexcept
{
    return value * 1024ULL * 1024ULL * 1024ULL;
}


/**
 * @param rangeA Closed interval given by two numbers.
 */
template<typename Pair,
         typename Value>
[[nodiscard]] constexpr bool
rangeContains( const Pair&  range,
               const Value& value ) noexcept
{
    return ( range.first <= value ) && ( value <= range.second );
}


template<typename PairA,
         typename PairB>
[[nodiscard]] constexpr bool
rangesIntersect( const PairA& rangeA,
                 const PairB& rangeB ) noexcept
{
    /**
     * Cases:
     * @verbatim
     * A     +------+
     *       |      |
     * B  +-+|      |
     *    +--+      |
     *    +----+    |
     *    +------------+
     *       +--+   |
     *       |+-+   |
     *       |+-----+
     *       |+--------+
     *       |      +--+
     *       |      |
     * @endverbatim
     */
    return rangeContains( rangeA, rangeB.first )
           || rangeContains( rangeA, rangeB.second )
           || rangeContains( rangeB, rangeA.first )
           || rangeContains( rangeB, rangeA.second );
}


constexpr std::string_view BASE64_SYMBOLS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+/";


template<typename CharT>
[[nodiscard]] bool
isBase64( std::basic_string_view<CharT> data )
{
    static const auto base64Symbols =
        [] ()
        {
            std::basic_string<CharT> result;
            result.resize( BASE64_SYMBOLS.size() );
            std::transform( BASE64_SYMBOLS.begin(), BASE64_SYMBOLS.end(), result.begin(),
                            [] ( const auto x ) { return static_cast<CharT>( x ); } );
            result.push_back( static_cast<CharT>( '\n' ) );
            return result;
        }();

    return data.find_first_not_of( base64Symbols ) == std::string_view::npos;
}
}  // namespace rapidgzip

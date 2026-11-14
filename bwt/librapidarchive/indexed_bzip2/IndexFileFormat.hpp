#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <filereader/FileReader.hpp>


namespace indexed_bzip2
{
/**
 * A bzip2 block offset index is a plain text file with one block per line:
 *
 * @verbatim
 * <compressed offset in bits>,<uncompressed offset in bytes>
 * @endverbatim
 *
 * The compressed offset points at the block magic bits, which do not have to be byte-aligned. The
 * uncompressed offset is the cumulative size of all preceding decoded blocks. Empty lines and lines starting
 * with '#' are ignored, so that the files can be annotated by hand.
 *
 * The format deliberately has no magic bytes or binary fields so that it stays readable with standard tools
 * and remains compatible with the CSV-like files written by older ibzip2 versions.
 */
using BlockOffsets = std::map<size_t, size_t>;

using CheckedWrite = std::function<void ( const void*, size_t )>;


namespace detail
{
[[nodiscard]] inline std::string_view
trim( std::string_view text )
{
    const auto isWhitespace = [] ( const char character ) {
                                  return ( character == ' ' ) || ( character == '\t' ) || ( character == '\r' );
                              };

    while ( !text.empty() && isWhitespace( text.front() ) ) {
        text.remove_prefix( 1 );
    }
    while ( !text.empty() && isWhitespace( text.back() ) ) {
        text.remove_suffix( 1 );
    }
    return text;
}


[[nodiscard]] inline size_t
parseOffset( const std::string_view text,
             const size_t           lineNumber )
{
    const auto trimmed = trim( text );
    if ( trimmed.empty() ) {
        throw std::runtime_error( "Invalid bzip2 block offset index line " + std::to_string( lineNumber )
                                  + ": empty offset!" );
    }
    if ( !std::all_of( trimmed.begin(), trimmed.end(), [] ( const char character ) {
                           return ( character >= '0' ) && ( character <= '9' );
                       } ) )
    {
        throw std::runtime_error( "Invalid bzip2 block offset index line " + std::to_string( lineNumber )
                                  + ": '" + std::string( trimmed ) + "' is not a non-negative number!" );
    }

    size_t parsedCharacters{ 0 };
    unsigned long long int value{ 0 };
    try {
        value = std::stoull( std::string( trimmed ), &parsedCharacters );
    } catch ( const std::exception& ) {
        throw std::runtime_error( "Invalid bzip2 block offset index line " + std::to_string( lineNumber )
                                  + ": '" + std::string( trimmed ) + "' is not a number!" );
    }

    if ( parsedCharacters != trimmed.size() ) {
        throw std::runtime_error( "Invalid bzip2 block offset index line " + std::to_string( lineNumber )
                                  + ": '" + std::string( trimmed ) + "' has trailing characters!" );
    }
    if ( value > std::numeric_limits<size_t>::max() ) {
        throw std::runtime_error( "Invalid bzip2 block offset index line " + std::to_string( lineNumber )
                                  + ": offset does not fit into size_t!" );
    }
    return static_cast<size_t>( value );
}


inline void
parseLine( const std::string_view line,
           const size_t           lineNumber,
           BlockOffsets&          offsets )
{
    const auto trimmed = trim( line );
    if ( trimmed.empty() || ( trimmed.front() == '#' ) ) {
        return;
    }

    const auto comma = trimmed.find( ',' );
    if ( comma == std::string_view::npos ) {
        throw std::runtime_error( "Invalid bzip2 block offset index line " + std::to_string( lineNumber )
                                  + ": missing comma in '" + std::string( trimmed ) + "'!" );
    }
    if ( trimmed.find( ',', comma + 1 ) != std::string_view::npos ) {
        throw std::runtime_error( "Invalid bzip2 block offset index line " + std::to_string( lineNumber )
                                  + ": more than one comma in '" + std::string( trimmed ) + "'!" );
    }

    offsets.emplace( parseOffset( trimmed.substr( 0, comma ), lineNumber ),
                     parseOffset( trimmed.substr( comma + 1 ), lineNumber ) );
}
}  // namespace detail


[[nodiscard]] inline BlockOffsets
readBzip2Index( rapidgzip::UniqueFileReader indexFile )
{
    if ( !indexFile ) {
        throw std::invalid_argument( "Index file reader must be valid!" );
    }

    BlockOffsets offsets;
    std::string pending;
    std::array<char, 64 * 1024> buffer{};
    size_t lineNumber{ 0 };

    while ( true ) {
        const auto nBytesRead = indexFile->read( buffer.data(), buffer.size() );
        if ( nBytesRead == 0 ) {
            break;
        }
        pending.append( buffer.data(), nBytesRead );

        size_t lineStart{ 0 };
        while ( true ) {
            const auto newline = pending.find( '\n', lineStart );
            if ( newline == std::string::npos ) {
                break;
            }
            ++lineNumber;
            detail::parseLine( std::string_view( pending ).substr( lineStart, newline - lineStart ),
                               lineNumber, offsets );
            lineStart = newline + 1;
        }
        pending.erase( 0, lineStart );
    }

    if ( !pending.empty() ) {
        ++lineNumber;
        detail::parseLine( pending, lineNumber, offsets );
    }

    return offsets;
}


inline void
writeBzip2IndexEntry( const CheckedWrite& checkedWrite,
                      const size_t        compressedOffset,
                      const size_t        uncompressedOffset )
{
    const auto line = std::to_string( compressedOffset ) + "," + std::to_string( uncompressedOffset ) + "\n";
    checkedWrite( line.data(), line.size() );
}


inline void
writeBzip2IndexEntry( std::ostream& out,
                      const size_t  compressedOffset,
                      const size_t  uncompressedOffset )
{
    out << compressedOffset << "," << uncompressedOffset << "\n";
}


inline void
writeBzip2Index( std::ostream&       out,
                 const BlockOffsets& offsets )
{
    for ( const auto& [compressedOffset, uncompressedOffset] : offsets ) {
        writeBzip2IndexEntry( out, compressedOffset, uncompressedOffset );
    }
    if ( !out ) {
        throw std::runtime_error( "Failed to write the bzip2 block offset index!" );
    }
}


inline void
writeBzip2Index( const BlockOffsets&    offsets,
                 const CheckedWrite&    checkedWrite )
{
    if ( !checkedWrite ) {
        throw std::invalid_argument( "Write functor must be valid!" );
    }

    for ( const auto& [compressedOffset, uncompressedOffset] : offsets ) {
        writeBzip2IndexEntry( checkedWrite, compressedOffset, uncompressedOffset );
    }
}
}  // namespace indexed_bzip2

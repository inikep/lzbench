#pragma once

#include <cstddef>
#include <cstdio>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <core/common.hpp>


namespace rapidgzip
{
class FileReader;

using UniqueFileReader = std::unique_ptr<FileReader>;


/**
 * Thrown when a reader has no data left.
 *
 * An exception is used instead of std::optional for BitReader::peek because reaching EOF happens at most once per
 * stream, so a try-catch block acts as a zero-cost conditional in the common case.
 */
class EndOfFileReached :
    public std::exception
{
public:
    [[nodiscard]] const char*
    what() const noexcept override
    {
        return "End of file reached!";
    }
};


/**
 * Thrown when a read or seek is attempted on a closed reader.
 *
 * Deliberately not derived from EndOfFileReached: a closed reader is a state misuse, not data exhaustion, and
 * existing EOF catch sites should not swallow it.
 */
class ReaderClosed :
    public std::exception
{
public:
    [[nodiscard]] const char*
    what() const noexcept override
    {
        return "Cannot read from closed file!";
    }
};


/**
 * This file interface is heavily inspired by IOBase from Python because it is to be used from Python.
 * However, it strips anything related to file modification resulting a read-only file object.
 * @see https://docs.python.org/3/library/io.html#class-hierarchy
 */
class FileReader
{
public:
    FileReader() = default;

    virtual
    ~FileReader() = default;

    /* Delete copy constructors and assignments for performance and to avoid slicing. */

    FileReader( const FileReader& ) = delete;

    FileReader&
    operator=( const FileReader& ) = delete;

    /* Move constructors also are affected by slicing, but for performance, do not delete them to avoid
     * defaulted move constructores in derived classes not being created! We also do not want to manually
     * implement move constructors because it is error-prone. The only choice here is between bad and worse :(. */

    FileReader( FileReader&& ) = default;

    /* I almost never use move assignments, but move constructors occur during function calls. */
    FileReader&
    operator=( FileReader&& ) = delete;

    /**
     * @return A std::unique_ptr to a copy of this FileReader. The copied file reader should have the same
     *         file position if possible, i.e., if not closed.
     */
    [[nodiscard]] UniqueFileReader
    clone() const
    {
        auto fileReader = cloneRaw();
        if ( !fileReader->closed() && ( fileReader->tell() != tell() ) ) {
            fileReader->seekTo( tell() );
        }
        return fileReader;
    }

    virtual void
    close() = 0;

    [[nodiscard]] virtual bool
    closed() const = 0;

    [[nodiscard]] virtual bool
    eof() const = 0;

    [[nodiscard]] virtual bool
    fail() const = 0;

    [[nodiscard]] virtual int
    fileno() const = 0;

    [[nodiscard]] virtual bool
    seekable() const = 0;

    [[nodiscard]] virtual size_t
    read( char*  buffer,
          size_t nMaxBytesToRead ) = 0;

    virtual size_t
    seek( long long int offset,
          int           origin = SEEK_SET ) = 0;

    size_t
    seekTo( uint64_t offset )
    {
        if ( offset > static_cast<uint64_t>( std::numeric_limits<long long int>::max() ) ) {
            throw std::invalid_argument( "Value " + std::to_string( offset ) + " out of range of long long int!" );
        }
        return seek( static_cast<long long int>( offset ) );
    }

    [[nodiscard]] virtual std::optional<size_t>
    size() const = 0;

    [[nodiscard]] virtual size_t
    tell() const = 0;

    virtual void
    clearerr() = 0;

protected:
    /**
     * Derived classes only need to construct a usable copy. Ideally, the state should be the same as the original.
     * The real @ref clone will call this and will ensure that the file position is transferred to the new object
     * if not already done.
     */
    [[nodiscard]] virtual UniqueFileReader
    cloneRaw() const
    {
        throw std::logic_error( "Not implemented!" );
    }

    /**
     * Resolve an offset and origin to an absolute offset without clamping it to the file size. A negative
     * result means the target is before the start. SEEK_END requires a known size and throws otherwise.
     */
    [[nodiscard]] long long int
    resolveOffset( long long int offset,
                   int           origin ) const
    {
        switch ( origin )
        {
        case SEEK_CUR:
            return saturatingAddition( static_cast<long long int>( tell() ), offset );
        case SEEK_SET:
            return offset;
        case SEEK_END:
            if ( const auto fileSize = size(); fileSize.has_value() ) {
                return saturatingAddition( static_cast<long long int>( *fileSize ), offset );
            }
            throw std::logic_error( "File size is not available to seek from end!" );
        default:
            break;
        }
        throw std::invalid_argument( "Invalid seek origin supplied: " + std::to_string( origin ) );
    }

    /**
     * Resolve an offset and clamp it to [0, size()]. Streaming readers may not know their size yet and
     * therefore do not clamp. Readers that must reject out-of-range seeks should use resolveOffset().
     */
    [[nodiscard]] size_t
    effectiveOffset( long long int offset,
                     int           origin ) const
    {
        const auto positiveOffset = static_cast<size_t>( std::max( resolveOffset( offset, origin ), 0LL ) );
        /* Streaming file readers may return nullopt until EOF has been reached. */
        const auto fileSize = size();
        return fileSize.has_value() ? std::min( positiveOffset, *fileSize ) : positiveOffset;
    }
};
}  // namespace rapidgzip

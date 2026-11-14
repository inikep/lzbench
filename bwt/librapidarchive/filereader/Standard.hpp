#pragma once

#include <algorithm>
#include <cstdio>       // fread
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>

#if defined( _WIN32 )
    #include <stdio.h>  // stdin
    #include <io.h>     // _setmode
    #include <fcntl.h>
#endif

#include <core/common.hpp>     // unistd, S_ISFIFO, fstat, ...
#include <core/FileUtils.hpp>  // unique_file_ptr, throwingOpen

#include "FileReader.hpp"


namespace rapidgzip
{
class StandardFileReader :
    public FileReader
{
public:
    explicit
    StandardFileReader( std::string filePath ) :
        m_file( throwingOpen( filePath, "rb" ) ),
        m_fileDescriptor( ::fileno( fp() ) ),
        m_filePath( std::move( filePath ) ),
        m_seekable( determineSeekable( m_fileDescriptor ) ),
        m_fileSizeBytes( m_seekable ? std::filesystem::file_size( m_filePath ) : 0 )
    {
        init();
    }

#if !defined(__APPLE_CC__ ) || ( defined(MAC_OS_X_VERSION_MIN_REQUIRED) \
    && MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_15 )
    explicit
    StandardFileReader( const std::filesystem::path& filePath ) :
        StandardFileReader( filePath.string() )
    {}

    /* Add this to avoid ambiguity for const char*, which would otherwise be the case with only
     * std::string and std::filesystem::path constructors. */
    explicit
    StandardFileReader( const char* filePath ) :
        StandardFileReader( std::string( filePath ) )
    {}
#endif

    explicit
    StandardFileReader( int fileDescriptor ) :
        /* Use dup here so that the following fclose will not close the original file descriptor,
         * which probably is still in use by the caller! Note that dup will not guarantee independent
         * file positions though, so restore the previous position after closing. */
        m_file( throwingOpen( dup( fileDescriptor ), "rb" ) ),
        m_fileDescriptor( ::fileno( fp() ) ),
        m_filePath( fdFilePath( m_fileDescriptor ) ),
        m_seekable( determineSeekable( m_fileDescriptor ) ),
        m_fileSizeBytes( m_seekable ? fileSize( m_fileDescriptor ) : 0 )
    {
        init();
    }

    ~StandardFileReader() override
    {
        StandardFileReader::close();
    }

    [[nodiscard]] UniqueFileReader
    cloneRaw() const override
    {
        throw std::invalid_argument( "Cloning file path reader not allowed because the internal file position "
                                     "should not be modified by multiple owners!" );
    }

    /* Copying is simply not allowed because that might interfere with the file position state, use SharedFileReader! */

    void
    close() override
    {
        if ( !m_file ) {
            return;
        }

        /* Try to restore the file position the file had before it was given to us. */
        if ( m_seekable ) {
            std::fsetpos( m_file.get(), &m_initialPosition );
        }

        m_file.reset();
    }

    [[nodiscard]] bool
    closed() const override
    {
        return !m_file;
    }

    [[nodiscard]] bool
    eof() const override
    {
        return m_seekable ? m_currentPosition >= m_fileSizeBytes : !m_lastReadSuccessful;
    }

    [[nodiscard]] bool
    fail() const override
    {
        return std::ferror( fp() ) != 0;
    }

    [[nodiscard]] int
    fileno() const override
    {
        if ( m_file ) {
            return m_fileDescriptor;
        }
        throw std::invalid_argument( "Trying to get fileno of an invalid file!" );
    }

    [[nodiscard]] bool
    seekable() const override
    {
        return m_seekable;
    }

    [[nodiscard]] size_t
    read( char*  buffer,
          size_t nMaxBytesToRead ) override
    {
        if ( !m_file ) {
            throw std::invalid_argument( "Invalid or file can't be seeked!" );
        }

        if ( nMaxBytesToRead == 0 ) {
            return 0;
        }

        /**
         * @see https://pubs.opengroup.org/onlinepubs/009696899/functions/fseek.html
         * > The fseek() function shall allow the file-position indicator to be set beyond the end of existing
         * > data in the file. If data is later written at this point, subsequent reads of data in the gap shall
         * > return bytes with the value 0 until data is actually written into the gap.
         * Because of this, it is not possible to simply use std::fseek. Because then, we would not be able to infer
         * whether we read past the end nor how many bytes we would have been able to read if the buffer was valid!
         */
        size_t nBytesRead = 0;
        if ( buffer == nullptr ) {
            if ( seekable() ) {
                /* Seeking beyond EOF is allowed, so the remaining byte count must not underflow. */
                nBytesRead = std::min( nMaxBytesToRead,
                                       m_currentPosition < m_fileSizeBytes
                                       ? m_fileSizeBytes - m_currentPosition
                                       : 0 );
                fileSeek( m_file.get(), static_cast<long long int>( nBytesRead ), SEEK_CUR );
            } else {
                std::array<char, 16_Ki> tmpBuffer{};
                while ( nBytesRead < nMaxBytesToRead ) {
                    /* Only request the remaining bytes. Otherwise the last chunk could consume up to a whole
                     * temporary buffer more than the skip requested, so the returned count would exceed it. */
                    const auto nBytesToRead = std::min( tmpBuffer.size(), nMaxBytesToRead - nBytesRead );
                    const auto nBytesReadPerCall =
                        std::fread( tmpBuffer.data(), /* element size */ 1, nBytesToRead, m_file.get() );
                    if ( nBytesReadPerCall == 0 ) {
                        break;
                    }
                    nBytesRead += nBytesReadPerCall;
                }
            }
        } else {
            nBytesRead = std::fread( buffer, /* element size */ 1, nMaxBytesToRead, m_file.get() );
        }

        if ( nBytesRead == 0 ) {
        #if 1
            /* fread returning 0 might traditionally be a valid case if the file position was after the last byte.
             * EOF is only set after reading after the end not when the file position is at the end. */
            m_lastReadSuccessful = false;
            return 0;
        #else
            /* I had some very weird issues of fread returning 0 bytes read even thought ftell was 0 and no EOF
             * bit nor fail bit was set. After that read no error was set but the EOF bit. It makes not sense to me.
             * It almost looks like ftell is broken and the actual position was not moved from the file end.
             * The problem disappeared after trimming down the seek method to basically directly call std::fseek
             * instead of manually transforming the offset and doing some preemptive returns, so maybe it was that. */
            std::stringstream message;
            message
                << "[StandardFileReader] Read call failed (" << nBytesRead << " B read)!\n"
                << "  Buffer: " << (void*)buffer << "\n"
                << "  nMaxBytesToRead: " << nMaxBytesToRead << " B\n"
                << "  File pointer: " << (void*)m_file.get() << "\n"
                << "  File size: " << m_fileSizeBytes << " B\n"
                << "  EOF: " << std::feof( m_file.get() ) << "\n"
                << "  ferror: " << std::ferror( m_file.get() ) << "\n"
                << "  fileno: " << m_fileDescriptor << "\n"
                << "  file path: " << m_filePath << "\n"
                << "  m_currentPosition: " << m_currentPosition << "\n"
                << "  ftell: " << std::ftell( m_file.get() ) << "\n"
                << "\n";
            throw std::domain_error( std::move( message ).str() );
        #endif
        }

        m_currentPosition += nBytesRead;
        m_lastReadSuccessful = nBytesRead == nMaxBytesToRead;

        return nBytesRead;
    }

    size_t
    seek( long long int offset,
          int           origin = SEEK_SET ) override
    {
        if ( !m_file || !m_seekable ) {
            throw std::invalid_argument( "Invalid or file can't be seeked!" );
        }

        fileSeek( m_file.get(), offset, origin );

        if ( origin == SEEK_SET ) {
            m_currentPosition = static_cast<size_t>( std::max( 0LL, offset ) );
        } else {
            /* Note that the file must be seekable at this point, meaning std::ftell will work! */
            m_currentPosition = filePosition( m_file.get() );
        }

        return m_currentPosition;
    }

    [[nodiscard]] std::optional<size_t>
    size() const override
    {
        /* A non-seekable stream such as a pipe reports st_size == 0, which is not its length.
         * Returning nullopt also keeps effectiveOffset() from clamping seeks to a bogus zero size. */
        if ( !m_seekable && ( m_fileSizeBytes == 0 ) ) {
            return std::nullopt;
        }
        return m_fileSizeBytes;
    }

    [[nodiscard]] size_t
    tell() const override
    {
        if ( m_seekable ) {
            return filePosition( fp() );
        }
        return m_currentPosition;
    }

    void
    clearerr() override
    {
        std::clearerr( fp() );
    }

private:
    void
    init()
    {
        std::fgetpos( fp(), &m_initialPosition );

        /* On macOS opening special files like /dev/fd/3 might result in the file position
         * not being 0 in the case it has been seeked or read from somewhere else! */
        if ( m_seekable ) {
            StandardFileReader::seek( 0, SEEK_SET );
        }
    }

    [[nodiscard]] static bool
    determineSeekable( int fileNumber )
    {
        struct stat fileStats{};
        if ( fstat( fileNumber, &fileStats ) != 0 ) {
            return false;
        }
    #if defined( _MSC_VER )
        return !S_ISFIFO( fileStats.st_mode );
    #else
        /* Only regular files and block devices have a meaningful size and support fseek.
         * Pipes, sockets, character devices, and directories do not. */
        return S_ISREG( fileStats.st_mode ) || S_ISBLK( fileStats.st_mode );
    #endif
    }

    [[nodiscard]] FILE*
    fp() const
    {
        if ( m_file ) {
            return m_file.get();
        }
        throw std::invalid_argument( "Operation not allowed on an invalid file!" );
    }

protected:
    unique_file_ptr m_file;
    const int m_fileDescriptor;
    const std::string m_filePath;

    std::fpos_t m_initialPosition{};
    const bool m_seekable;
    const size_t m_fileSizeBytes;

    size_t m_currentPosition{ 0 };  /**< Only necessary for unseekable files. */
    bool m_lastReadSuccessful{ true };
};


[[nodiscard]] inline UniqueFileReader
openFileOrStdin( const std::string& inputFilePath )
{
    if ( !inputFilePath.empty() && ( inputFilePath != "-" ) ) {
        return std::make_unique<StandardFileReader>( inputFilePath );
    }

#ifdef _WIN32
    const auto stdinHandle = _fileno( stdin );
    _setmode( stdinHandle, _O_BINARY );
#else
    const auto stdinHandle = STDIN_FILENO;
#endif

    return std::make_unique<StandardFileReader>( stdinHandle );
}
}  // namespace rapidgzip

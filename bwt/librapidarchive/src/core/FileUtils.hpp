#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __APPLE_CC__
    #include <AvailabilityMacros.h>
#endif

#ifdef _MSC_VER
    #define NOMINMAX
    #include <Windows.h>

    #include <fcntl.h>     // _O_BINARY
    #include <stdio.h>     // stdout
    #include <sys/stat.h>  // _fstat64
    #include <io.h>        // _setmode
#else
    #ifndef _GNU_SOURCE
        #define _GNU_SOURCE
    #endif

    #include <errno.h>
    #include <fcntl.h>
    #include <limits.h>         // IOV_MAX
    #include <sys/stat.h>
    #if defined(__linux__) || defined(__APPLE__)
        #include <sys/uio.h>
    #endif
    #include <unistd.h>

    #if defined( LIBRAPIDARCHIVE_WITH_VMSPLICE ) && ( !defined( __linux__ ) || !defined( F_GETPIPE_SZ ) )
        #error "LIBRAPIDARCHIVE_WITH_VMSPLICE requires Linux and F_GETPIPE_SZ."
    #endif

    #if not defined( HAVE_IOVEC ) and defined( __linux__ )
        #define HAVE_IOVEC
    #endif
#endif

#if defined( _WIN32 ) && !defined( _MSC_VER )
    #include <io.h>  // _setmode
#endif

/**
 * vmsplice is opt-in via LIBRAPIDARCHIVE_WITH_VMSPLICE because it is still experimental. The kernel does
 * not copy the pages at the vmsplice call. They must not be modified or reused until the reader has
 * consumed them. The current implementation uses FIONREAD to release chunks only after the reader has
 * consumed them, and intentionally leaks the final tail at process exit.
 *
 * A future implementation can use mmap-allocated buffers and unmap them directly after vmsplice.
 * SPLICE_F_GIFT is only needed if a later splice should steal the pages.
 * https://github.com/mxmlnkn/rapidgzip/issues/39
 */


#if defined( LIBRAPIDARCHIVE_WITH_VMSPLICE )
    #include <sys/ioctl.h>

    #include <algorithm>
    #include <any>
    #include <cstdint>
    #include <deque>
    #include <memory>
    #include <tuple>
#endif


namespace rapidgzip
{
class SpliceVault;

struct BrokenPipeException :
    public std::exception
{
    [[nodiscard]] const char*
    what() const noexcept override
    {
        return "Broken pipe";
    }
};


#ifdef _MSC_VER
[[nodiscard]] inline bool
stdinHasInput()
{
    return _isatty( _fileno( stdin ) ) == 0;
}


[[nodiscard]] inline bool
stdoutIsDevNull()
{
    /**
     * @todo Figure this out on Windows in a reasonable readable manner:
     * @see https://stackoverflow.com/a/21070689/2191065
     */
    return false;
}

#else

[[nodiscard]] inline bool
stdinHasInput()
{
    return isatty( STDIN_FILENO ) == 0;
}


[[nodiscard]] inline bool
stdoutIsDevNull()
{
    struct stat devNull{};
    struct stat stdOut{};
    return ( fstat( STDOUT_FILENO, &stdOut ) == 0 ) &&
           ( stat( "/dev/null", &devNull ) == 0 ) &&
           S_ISCHR( stdOut.st_mode ) &&  // NOLINT
           ( devNull.st_dev == stdOut.st_dev ) &&
           ( devNull.st_ino == stdOut.st_ino );
}
#endif


[[nodiscard]] inline std::ios_base::seekdir
toSeekdir( int origin )
{
    switch ( origin )
    {
    case SEEK_SET:
        return std::ios_base::beg;
    case SEEK_CUR:
        return std::ios_base::cur;
    case SEEK_END:
        return std::ios_base::end;
    default:
        break;
    }

    throw std::invalid_argument( "Unknown origin" );
}


[[nodiscard]] inline const char*
originToString( int origin )
{
    switch ( origin )
    {
    case SEEK_SET:
        return "SEEK_SET";
    case SEEK_CUR:
        return "SEEK_CUR";
    case SEEK_END:
        return "SEEK_END";
    default:
        break;
    }

    throw std::invalid_argument( "Unknown origin" );
}


inline bool
fileExists( const std::string& filePath )
{
    return std::ifstream( filePath, std::ios_base::in | std::ios_base::binary ).good();
}


inline uintmax_t
fileSize( const std::string& filePath )
{
    return std::filesystem::file_size( filePath );
}


inline uintmax_t
filePosition( std::FILE* file )
{
    if ( file == nullptr ) {
        throw std::runtime_error( "File pointer to call tell on must not be null!" );
    }

#if defined(_MSC_VER)
    /* On Windows, std::ftell STILL (2025!) cannot handle files > 4 GiB because 'long int' is 32-bit! */
    const auto offset = _ftelli64( file );
#else
    const auto offset = std::ftell( file );
#endif
    if ( offset < 0 ) {
        throw std::runtime_error( "Could not get the file position!" );
    }
    return static_cast<uintmax_t>( offset );
}


inline void
fileSeek( std::FILE*    file,
          long long int offset,
          int           origin )
{
    if ( file == nullptr ) {
        throw std::runtime_error( "File pointer to call seek on must not be null!" );
    }

#if defined(_MSC_VER)
    /* On Windows, std::fseek STILL (2025!) cannot handle files > 4 GiB because 'long int' is 32-bit! */
    const auto returnCode = _fseeki64( file, offset, origin );
#else
    if ( offset > static_cast<long long int>( std::numeric_limits<long int>::max() ) ) {
        throw std::out_of_range( "std::fseek only takes long int, try compiling for 64 bit." );
    }

    const auto returnCode = std::fseek( file, static_cast<long int>( offset ), origin );
#endif

    if ( returnCode != 0 ) {
        std::stringstream message;
        message << "Seeking to " << offset << " from origin " << originToString( origin ) << " failed with code: "
                << returnCode << ", " << std::strerror( errno ) << "!";
        throw std::runtime_error( std::move( message ).str() );
    }
}


#ifndef _MSC_VER
struct unique_file_descriptor
{
    explicit
    unique_file_descriptor( int fd ) :
        m_fd( fd )
    {}

    ~unique_file_descriptor()
    {
        close();
    }

    unique_file_descriptor() = default;

    unique_file_descriptor( const unique_file_descriptor& ) = delete;

    unique_file_descriptor&
    operator=( const unique_file_descriptor& ) = delete;

    unique_file_descriptor( unique_file_descriptor&& other ) noexcept :
        m_fd( other.m_fd )
    {
        other.m_fd = -1;
    }

    unique_file_descriptor&
    operator=( unique_file_descriptor&& other ) noexcept
    {
        close();
        m_fd = other.m_fd;
        other.m_fd = -1;
        return *this;
    }

    [[nodiscard]] constexpr int
    operator*() const noexcept
    {
        return m_fd;
    }

    void
    close()
    {
        if ( m_fd >= 0 ) {
            ::close( m_fd );
            m_fd = -1;
        }
    }

    void
    release()
    {
        m_fd = -1;
    }

private:
    int m_fd{ -1 };
};
#endif  // ifndef _MSC_VER


using unique_file_ptr = std::unique_ptr<std::FILE, std::function<void ( std::FILE* )> >;

inline unique_file_ptr
make_unique_file_ptr( std::FILE* file )
{
    return {
        file,
        [] ( auto* ownedFile ) {
            if ( ownedFile != nullptr ) {
                std::fclose( ownedFile );  // NOLINT
            }
        }
    };
}


inline unique_file_ptr
make_unique_file_ptr( char const* const filePath,
                      char const* const mode )
{
    if ( ( filePath == nullptr ) || ( mode == nullptr ) || ( std::strlen( filePath ) == 0 ) ) {
        return {};
    }
    return make_unique_file_ptr( std::fopen( filePath, mode ) );  // NOLINT
}


inline unique_file_ptr
make_unique_file_ptr( int         fileDescriptor,
                      char const* mode )
{
    if ( fileDescriptor < 0 ) {
        return nullptr;
    }
    return make_unique_file_ptr( fdopen( fileDescriptor, mode ) );
}


inline unique_file_ptr
throwingOpen( const std::string& filePath,
              const char*        mode )
{
    if ( mode == nullptr ) {
        throw std::invalid_argument( "Mode must be a C-String and not null!" );
    }

    auto file = make_unique_file_ptr( filePath.c_str(), mode );
    if ( file == nullptr ) {
        std::stringstream msg;
        msg << "Opening file '" << filePath << "' with mode '" << mode << "' failed!";
        throw std::invalid_argument( std::move( msg ).str() );
    }

    return file;
}


inline unique_file_ptr
throwingOpen( int         fileDescriptor,
              const char* mode )
{
    if ( mode == nullptr ) {
        throw std::invalid_argument( "Mode must be a C-String and not null!" );
    }

    auto file = make_unique_file_ptr( fileDescriptor, mode );
    if ( file == nullptr ) {
        std::stringstream msg;
        msg << "Opening file descriptor " << fileDescriptor << " with mode '" << mode << "' failed!";
        throw std::invalid_argument( std::move( msg ).str() );
    }

    return file;
}


/** dup is not strong enough to be able to independently seek in the old and the dup'ed fd! */
[[nodiscard]] inline std::string
fdFilePath( int fileDescriptor )
{
    std::stringstream filename;
    filename << "/dev/fd/" << fileDescriptor;
    return filename.str();
}


template<typename Container = std::vector<char> >
[[nodiscard]] Container
readFile( const std::string& fileName )
{
    Container contents( fileSize( fileName ) );
    const auto file = throwingOpen( fileName, "rb" );
    const auto nBytesRead = std::fread( contents.data(), sizeof( contents[0] ), contents.size(), file.get() );

    if ( nBytesRead != contents.size() ) {
        throw std::logic_error( "Did read less bytes than file is large!" );
    }

    return contents;
}


/* Missing std::filesystem::path support in wheels.
 * https://opensource.apple.com/source/xnu/xnu-2050.7.9/EXTERNAL_HEADERS/AvailabilityMacros.h.auto.html */
#if !defined(__APPLE_CC__ ) || ( defined(MAC_OS_X_VERSION_MIN_REQUIRED) \
    && MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_15 )
[[nodiscard]] inline std::string
findParentFolderContaining( const std::string& folder,
                            const std::string& relativeFilePath )
{
    auto parentFolder = std::filesystem::absolute( folder );
    while ( !parentFolder.empty() )
    {
        const auto filePath = parentFolder / relativeFilePath;
        if ( std::filesystem::exists( filePath ) ) {
            return parentFolder.string();
        }

        if ( parentFolder.parent_path() == parentFolder ) {
            break;
        }
        parentFolder = parentFolder.parent_path();
    }

    return {};
}


inline unique_file_ptr
throwingOpen( const std::filesystem::path& filePath,
              const char*                  mode )
{
    return throwingOpen( filePath.string(), mode );
}


inline size_t
fileSize( const std::filesystem::path& filePath )
{
    return fileSize( filePath.string() );
}


template<typename Container = std::vector<char> >
[[nodiscard]] Container
readFile( const std::filesystem::path& filePath )
{
    return readFile<Container>( filePath.string() );
}
#endif


inline uintmax_t
fileSize( const int fileDescriptor )
{
#if defined(_MSC_VER)
    /* On Windows, fstat STILL (2025!) cannot handle files > 4 GiB! */
    struct _stat64 fileStats{};
    const auto result = _fstat64( fileDescriptor, &fileStats );
#else
    struct stat fileStats{};
    const auto result = fstat( fileDescriptor, &fileStats );
#endif

    if ( result == -1 ) {
        std::stringstream message;
        message << "Failed to get file size because of: " << strerror( errno ) << " (" << errno << ")";
        throw std::runtime_error( std::move( message ).str() );
    }
    if ( fileStats.st_size < 0 ) {
        throw std::runtime_error( "The file size is negative!" );
    }
    return static_cast<uintmax_t>( fileStats.st_size );
}


/**
 * Removes duplicate file descriptors, including different integers that refer to the same stream.
 * fstat is called at most once per descriptor. The relative order of the remaining descriptors is kept.
 */
inline void
deduplicateFileDescriptors( std::vector<int>& fileDescriptors )
{
    struct FileIdentity
    {
        uint64_t device;
        uint64_t inode;

        [[nodiscard]] bool
        operator==( const FileIdentity& other ) const noexcept
        {
            return ( device == other.device ) && ( inode == other.inode );
        }
    };

    struct FileIdentityHash
    {
        [[nodiscard]] size_t
        operator()( const FileIdentity& identity ) const noexcept
        {
            return std::hash<uint64_t>{}( identity.device )
                   ^ ( std::hash<uint64_t>{}( identity.inode ) << 1 );
        }
    };

    std::unordered_set<int> seenFileDescriptors;
    std::unordered_set<FileIdentity, FileIdentityHash> seenIdentities;
    size_t outputIndex = 0;
    for ( const auto fileDescriptor : fileDescriptors ) {
        if ( !seenFileDescriptors.insert( fileDescriptor ).second ) {
            continue;
        }

    #if defined( _MSC_VER )
        struct _stat64 fileStats{};
        const auto hasIdentity = ( _fstat64( fileDescriptor, &fileStats ) == 0 );
    #else
        struct stat fileStats{};
        const auto hasIdentity = ( ::fstat( fileDescriptor, &fileStats ) == 0 );
    #endif
        if ( hasIdentity
             && !seenIdentities.insert( { static_cast<uint64_t>( fileStats.st_dev ),
                                          static_cast<uint64_t>( fileStats.st_ino ) } ).second ) {
            continue;
        }

        fileDescriptors[outputIndex++] = fileDescriptor;
    }
    fileDescriptors.erase( fileDescriptors.begin() + outputIndex, fileDescriptors.end() );
}


#if defined( LIBRAPIDARCHIVE_WITH_VMSPLICE )

/**
 * Short overview of syscalls that optimize copies by instead copying full page pointers into the
 * pipe buffers inside the kernel:
 * - splice: <fd (pipe or not)> <-> <pipe>
 * - vmsplice: memory -> <pipe>
 * - mmap: <fd> -> memory
 * - sendfile: <fd that supports mmap> -> <fd (before Linux 2.6.33 (2010-02-24) it had to be a socket fd)>
 *
 * I think the underlying problem with wrong output data for small chunk sizes
 * is that vmsplice is not as "synchronous" as I thought it to be:
 *
 * https://lwn.net/Articles/181169/
 *
 *  - Determining whether it is safe to write to a vmspliced buffer is
 *    suggested to be done implicitly by splicing more than the maximum
 *    number of pages that can be inserted into the pipe buffer.
 *    That number was supposed to be queryable with fcntl F_GETPSZ.
 *    -> This is probably why I didn't notice problems with larger chunk
 *       sizes.
 *  - Even that might not be safe enough when there are multiple pipe
 *    buffers.
 *
 * https://stackoverflow.com/questions/70515745/how-do-i-use-vmsplice-to-correctly-output-to-a-pipe
 * https://codegolf.stackexchange.com/questions/215216/high-throughput-fizz-buzz/239848#239848
 *
 *  - the safest way to use vmsplice seems to be mmap -> vmplice with
 *    SPLICE_F_GIFT -> munmap. munmap can be called directly after the
 *    return from vmplice and this works in a similar way to aio_write
 *    but actually a lot faster.
 *
 * I think using std::vector with vmsplice is NOT safe when it is
 * destructed too soon! The problem here is that the memory is probably not
 * returned to the system, which would be fine, but is actually reused by
 * the c/C++ standard library's implementation of malloc/free/new/delete:
 *
 * https://stackoverflow.com/a/1119334
 *
 *  - In many malloc/free implementations, free does normally not return
 *    the memory to the operating system (or at least only in rare cases).
 *    [...] Free will put the memory block in its own free block list.
 *    Normally it also tries to meld together adjacent blocks in the
 *    address space.
 *
 * https://mazzo.li/posts/fast-pipes.html
 * https://github.com/bitonic/pipes-speed-test.git
 *
 *  - Set pipe size and double buffer. (Similar to the lwn article
 *    but instead of querying the pipe size, it is set.)
 *  - fcntl(STDOUT_FILENO, F_SETPIPE_SZ, options.pipe_size);
 *
 * I think I will have to implement a container with a custom allocator
 * that uses mmap and munmap to get back my vmsplice speeds :/(.
 * Or maybe try setting the pipe buffer size to some forced value and
 * then only free the last data after pipe size more has been written.
 *
 * @return The number of bytes accepted by vmsplice and an error code. bytesWritten can be positive even
 *         when the splice failed part-way through.
 */
struct SpliceResult
{
    size_t bytesWritten{ 0 };
    int errorCode{ 0 };
};


[[nodiscard]] inline SpliceResult
writeAllSpliceUnsafe( const int         outputFileDescriptor,
                      const void* const dataToWrite,
                      const size_t      dataToWriteSize )
{
    ::iovec dataToSplice{};
    /* The const_cast should be safe because vmsplice should not modify the input data. */
    dataToSplice.iov_base = const_cast<void*>( dataToWrite );
    dataToSplice.iov_len = dataToWriteSize;

    size_t totalWritten = 0;
    while ( dataToSplice.iov_len > 0 ) {
        const auto nBytesWritten = ::vmsplice( outputFileDescriptor, &dataToSplice, 1, /* flags */ 0 );
        if ( nBytesWritten < 0 ) {
            if ( errno == EINTR ) {
                continue;
            }
            return { totalWritten, errno };
        }
        if ( nBytesWritten == 0 ) {
            return { totalWritten, EIO };
        }

        totalWritten += static_cast<size_t>( nBytesWritten );
        dataToSplice.iov_base = static_cast<char*>( dataToSplice.iov_base ) + nBytesWritten;
        dataToSplice.iov_len -= static_cast<size_t>( nBytesWritten );
    }
    return { totalWritten, 0 };
}


[[nodiscard]] inline SpliceResult
writeAllSpliceUnsafe( const int                   outputFileDescriptor,
                      const std::vector<::iovec>& dataToWrite )
{
    size_t totalWritten = 0;
    for ( size_t i = 0; i < dataToWrite.size(); ) {
        const auto segmentCount = std::min( static_cast<size_t>( IOV_MAX ), dataToWrite.size() - i );
        const auto vmspliceResult = ::vmsplice( outputFileDescriptor, &dataToWrite[i], segmentCount, /* flags */ 0 );
        if ( vmspliceResult < 0 ) {
            if ( errno == EINTR ) {
                continue;
            }
            return { totalWritten, errno };
        }
        if ( vmspliceResult == 0 ) {
            return { totalWritten, EIO };
        }

        auto nBytesWritten = static_cast<size_t>( vmspliceResult );
        totalWritten += nBytesWritten;

        /* Skip over buffers that were written fully. */
        for ( ; ( i < dataToWrite.size() ) && ( dataToWrite[i].iov_len <= nBytesWritten ); ++i ) {
            nBytesWritten -= dataToWrite[i].iov_len;
        }

        /* Write out last partially written buffer if necessary so that we can resume full vectorized writing
         * from the next iovec buffer. */
        if ( ( i < dataToWrite.size() ) && ( nBytesWritten > 0 ) ) {
            const auto& iovBuffer = dataToWrite[i];

            assert( iovBuffer.iov_len > nBytesWritten );
            const auto remainingSize = iovBuffer.iov_len - nBytesWritten;

            const auto remainingData = static_cast<char*>( iovBuffer.iov_base ) + nBytesWritten;
            const auto result = writeAllSpliceUnsafe( outputFileDescriptor, remainingData, remainingSize );
            totalWritten += result.bytesWritten;
            if ( result.errorCode != 0 ) {
                return { totalWritten, result.errorCode };
            }
            ++i;
        }
    }

    return { totalWritten, 0 };
}


/**
 * Keeps shared pointers to spliced objects until the reader has consumed the corresponding bytes from the
 * pipe. Entries are released using FIONREAD so that pipe resizing and partial writes do not invalidate the
 * accounting. The vaults and their final tail are intentionally leaked at process exit because the pipe can
 * still reference the pages after the writer has closed its file descriptor.
 */
class SpliceVault
{
public:
    explicit
    SpliceVault( const int fileDescriptor ) :
        m_fileDescriptor( fileDescriptor )
    {
        struct stat fileStats{};
        m_isPipe = ( ::fstat( fileDescriptor, &fileStats ) == 0 ) && S_ISFIFO( fileStats.st_mode );
    }

    /* The pipe can still reference the last spliced pages after the writer has closed its end. Leak the
     * retained tail at process exit. Library users that know the reader has drained can call drain(). */
    ~SpliceVault()
    {
        if ( m_splicedData && !m_splicedData->empty() ) {
            m_splicedData.release();
        }
    }

    /** Release entries that the reader has already consumed according to FIONREAD. */
    void
    drain()
    {
        releaseConsumed();
    }

    /**
     * @param dataToWrite A pointer to the start of the data to write. This pointer should be part of @p splicedData!
     * @param splicedData This owning shared pointer will be stored until the reader has consumed the data.
     */
    template<typename T>
    [[nodiscard]] SpliceResult
    splice( const void* const         dataToWrite,
            const size_t              dataToWriteSize,
            const std::shared_ptr<T>& splicedData )
    {
        if ( !m_isPipe ) {
            return { 0, EINVAL };
        }
        if ( dataToWriteSize == 0 ) {
            return { 0, 0 };
        }

        const auto wasAppended = retain( splicedData );
        const auto result = writeAllSpliceUnsafe( m_fileDescriptor, dataToWrite, dataToWriteSize );
        finishSplice( result.bytesWritten, wasAppended );
        return result;
    }

    /**
     * Overload that works for iovec structures directly.
     */
    template<typename T>
    [[nodiscard]] SpliceResult
    splice( const std::vector<::iovec>& buffersToWrite,
            const std::shared_ptr<T>&   splicedData )
    {
        if ( !m_isPipe ) {
            return { 0, EINVAL };
        }
        if ( buffersToWrite.empty() ) {
            return { 0, 0 };
        }

        const auto wasAppended = retain( splicedData );
        const auto result = writeAllSpliceUnsafe( m_fileDescriptor, buffersToWrite );
        finishSplice( result.bytesWritten, wasAppended );
        return result;
    }

private:
    using Entry = std::tuple</* packed RAII resource */ std::any,
                             /* raw pointer of RAII resource for comparison */ const void*,
                             /* cumulative end offset in the pipe */ uint64_t>;

    template<typename T>
    bool
    retain( const std::shared_ptr<T>& splicedData )
    {
        if ( !m_splicedData->empty() && ( std::get<1>( m_splicedData->back() ) == splicedData.get() ) ) {
            return false;
        }

        /* Add the provisional entry before writing so that even a partial write keeps the data alive. */
        m_splicedData->emplace_back( splicedData, splicedData.get(), m_bytesWritten );
        return true;
    }

    void
    finishSplice( const size_t bytesWritten,
                  const bool   wasAppended )
    {
        if ( bytesWritten > 0 ) {
            m_bytesWritten += bytesWritten;
            if ( !m_splicedData->empty() ) {
                std::get<2>( m_splicedData->back() ) = m_bytesWritten;
            }
        } else if ( wasAppended ) {
            m_splicedData->pop_back();
        }

        releaseConsumed();
    }

    void
    releaseConsumed()
    {
        int pendingBytes = 0;
        if ( ( ::ioctl( m_fileDescriptor, FIONREAD, &pendingBytes ) != 0 ) || ( pendingBytes < 0 ) ) {
            /* If the exact pending size is unknown, retain conservatively. */
            return;
        }
        if ( static_cast<uint64_t>( pendingBytes ) > m_bytesWritten ) {
            /* Another writer has put data into the same pipe. Retain conservatively. */
            return;
        }

        const auto consumedBytes = m_bytesWritten - static_cast<uint64_t>( pendingBytes );
        while ( !m_splicedData->empty() && ( std::get<2>( m_splicedData->front() ) <= consumedBytes ) ) {
            m_splicedData->pop_front();
        }
    }

private:
    const int m_fileDescriptor;
    bool m_isPipe{ false };

    /** Contains the shared_ptr to extend the lifetime and the cumulative pipe offset up to which the
     *  corresponding bytes have been spliced. */
    std::unique_ptr<std::deque<Entry>> m_splicedData = std::make_unique<std::deque<Entry>>();
    /** Total number of bytes spliced into this pipe. */
    uint64_t m_bytesWritten{ 0 };
};


template<typename T>
[[nodiscard]] inline SpliceResult
writeAllSplice( SpliceVault&                vault,
                const std::shared_ptr<T>&   splicedData,
                const std::vector<::iovec>& buffersToWrite )
{
    return vault.splice( buffersToWrite, splicedData );
}
#endif  // LIBRAPIDARCHIVE_WITH_VMSPLICE


/**
 * Posix write is not guaranteed to write everything and in fact was encountered to not write more than
 * 0x7ffff000 (2'147'479'552) B. To avoid this, it has to be looped over.
 */
[[nodiscard]] inline int
writeAllToFd( const int         outputFileDescriptor,
              const void* const dataToWrite,
              const uint64_t    dataToWriteSize )
{
    for ( uint64_t nTotalWritten = 0; nTotalWritten < dataToWriteSize; ) {
        const auto* const currentBufferPosition = reinterpret_cast<const uint8_t*>( dataToWrite ) + nTotalWritten;

        const auto nBytesToWritePerCall =
            static_cast<unsigned int>(
                std::min( static_cast<uint64_t>( std::numeric_limits<unsigned int>::max() ),
                          dataToWriteSize - nTotalWritten ) );

        const auto nBytesWritten = ::write( outputFileDescriptor, currentBufferPosition, nBytesToWritePerCall );
        if ( nBytesWritten < 0 ) {
            if ( errno == EINTR ) {
                continue;
            }
            return errno;
        }
        if ( nBytesWritten == 0 ) {
            return EIO;
        }
        nTotalWritten += static_cast<uint64_t>( nBytesWritten );
    }

    return 0;
}


#ifdef HAVE_IOVEC
inline void
pwriteAllToFd( const int         outputFileDescriptor,
               const void* const dataToWrite,
               const uint64_t    dataToWriteSize,
               const uint64_t    fileOffset )
{
    for ( uint64_t nTotalWritten = 0; nTotalWritten < dataToWriteSize; ) {
        const auto* const currentBufferPosition = reinterpret_cast<const uint8_t*>( dataToWrite ) + nTotalWritten;
        const auto nBytesWritten = ::pwrite( outputFileDescriptor,
                                             currentBufferPosition,
                                             dataToWriteSize - nTotalWritten,
                                             fileOffset + nTotalWritten );  // NOLINT
        if ( ( nBytesWritten < 0 ) && ( errno == EINTR ) ) {
            continue;
        }
        if ( nBytesWritten <= 0 ) {
            std::stringstream message;
            message << "Unable to write all data to the given file descriptor. Wrote " << nTotalWritten << " out of "
                    << dataToWriteSize << " (" << strerror( errno ) << ").";
            throw std::runtime_error( std::move( message ).str() );
        }

        nTotalWritten += static_cast<uint64_t>( nBytesWritten );
    }
}


[[nodiscard]] inline int
writeAllToFdVector( const int                   outputFileDescriptor,
                    const std::vector<::iovec>& dataToWrite )
{
    for ( size_t i = 0; i < dataToWrite.size(); ) {
        const auto segmentCount = std::min( static_cast<size_t>( IOV_MAX ), dataToWrite.size() - i );
        auto nBytesWritten = ::writev( outputFileDescriptor, &dataToWrite[i], segmentCount );  // NOLINT

        if ( nBytesWritten < 0 ) {
            if ( errno == EINTR ) {
                continue;
            }
            return errno;
        }
        if ( nBytesWritten == 0 ) {
            return EIO;
        }

        /* Skip over buffers that were written fully. */
        for ( ; ( i < dataToWrite.size() ) && ( dataToWrite[i].iov_len <= static_cast<size_t>( nBytesWritten ) );
              ++i ) {
            nBytesWritten -= dataToWrite[i].iov_len;  // NOLINT
        }

        /* Write out last partially written buffer if necessary so that we can resume full vectorized writing
         * from the next iovec buffer. */
        if ( ( i < dataToWrite.size() ) && ( nBytesWritten > 0 ) ) {
            const auto& iovBuffer = dataToWrite[i];

            /* The loop above stops at the first buffer that was not fully written, so more bytes are
             * remaining in it than writev reported. */
            assert( iovBuffer.iov_len > static_cast<size_t>( nBytesWritten ) );
            const auto remainingSize = iovBuffer.iov_len - nBytesWritten;
            auto* const remainingData = reinterpret_cast<char*>( iovBuffer.iov_base ) + nBytesWritten;
            const auto errorCode = writeAllToFd( outputFileDescriptor, remainingData, remainingSize );
            if ( errorCode != 0 ) {
                return errorCode;
            }

            ++i;
        }
    }

    return 0;
}
#endif  // HAVE_IOVEC


[[nodiscard]] inline int
writeAll( const int         outputFileDescriptor,
          void* const       outputBuffer,
          const void* const dataToWrite,
          const uint64_t    dataToWriteSize )
{
    if ( dataToWriteSize == 0 ) {
        return 0;
    }

    if ( outputFileDescriptor >= 0 ) {
        return writeAllToFd( outputFileDescriptor, dataToWrite, dataToWriteSize );
    }

    if ( outputBuffer != nullptr ) {
        if ( dataToWriteSize > std::numeric_limits<size_t>::max() ) {
            throw std::invalid_argument( "Too much data to write!" );
        }
        std::memcpy( outputBuffer, dataToWrite, dataToWriteSize );
    }

    return 0;
}


/**
 * Wrapper to open either stdout/stderr, a given existing file without truncation for better performance, or a new file.
 */
class OutputFile
{
public:
    explicit
    OutputFile( const std::string& filePath,
                const bool         useStderr = false ) :
        m_writingToStdout( filePath.empty() && !useStderr )
    {
        if ( filePath.empty() ) {
        #ifdef _WIN32
            m_fileDescriptor = _fileno( useStderr ? stderr : stdout );
            _setmode( m_fileDescriptor, _O_BINARY );
        #else
            m_fileDescriptor = ::fileno( useStderr ? stderr : stdout );
        #endif
        } else {
        #ifndef _MSC_VER
            if ( fileExists( filePath ) ) {
                try {
                    m_oldOutputFileSize = fileSize( filePath );
                } catch ( const std::filesystem::filesystem_error& ) {
                    /* This happens, e.g., when trying to write to /dev/null. */
                }
                /* Opening an existing file and overwriting its data can be much slower because posix_fallocate
                 * can be relatively slow compared to the decoding speed and memory bandwidth! Note that std::fopen
                 * would open a file with O_TRUNC, deallocating all its contents before it has to be reallocated. */
            #if defined( _WIN32 )
                m_fileDescriptor = ::open( filePath.c_str(), O_WRONLY | O_BINARY );  // NOLINT
            #else
                m_fileDescriptor = ::open( filePath.c_str(), O_WRONLY );  // NOLINT
            #endif
                m_ownedFd = unique_file_descriptor( m_fileDescriptor );
            }
        #endif

            if ( m_fileDescriptor == -1 ) {
                m_outputFile = make_unique_file_ptr( filePath.c_str(), "wb" );
                if ( !m_outputFile ) {
                    std::cerr << "Could not open output file: " << filePath << " for writing!\n";
                    throw std::runtime_error( "File could not be opened." );
                }
                m_fileDescriptor = ::fileno( m_outputFile.get() );
            }
        }
    }

    void
    write( const void*  data,
           const size_t size )
    {
        const auto errorCode = writeAllToFd( m_fileDescriptor, data, size );
        if ( errorCode == EPIPE ) {
            throw BrokenPipeException();
        }

        if ( errorCode != 0 ) {
            std::stringstream message;
            message << "Failed to write all bytes because of: " << strerror( errorCode )
                    << " (" << errorCode << ")";
            throw std::runtime_error( std::move( message ).str() );
        }
        m_bytesWritten += size;
    }

    void
    truncateToWrittenSize()
    {
        truncate( m_bytesWritten );
    }

    void
    truncate( size_t size )  // NOLINT
    {
    #ifndef _MSC_VER
        if ( ( m_fileDescriptor != -1 ) && ( size < m_oldOutputFileSize ) ) {
            if ( ::ftruncate( m_fileDescriptor, size ) == -1 ) {  // NOLINT
                std::cerr << "[Error] Failed to truncate file because of: " << strerror( errno )
                          << " (" << errno << ")\n";
            }
        }
    #endif
    }

    [[nodiscard]] bool
    writingToStdout() const noexcept
    {
        return m_writingToStdout;
    }

    [[nodiscard]] int
    fd() const noexcept
    {
        return m_fileDescriptor;
    }

private:
    const bool m_writingToStdout;

    int m_fileDescriptor{ -1 };  // Use this for file access.

    /** This is used to decide whether to truncate the file to a smaller (decompressed) size before closing. */
    size_t m_oldOutputFileSize{ 0 };

    /** Bytes written through write(), used to trim a reused output file to the new content. */
    size_t m_bytesWritten{ 0 };

    /**
     * These should not be used. They are only for automatic closing!
     * Two of them because a file may either be opened from an existing file without truncation,
     * which does not fit into unique_file_ptr, or it might be newly created.
     */
    unique_file_ptr m_outputFile;
#ifndef _MSC_VER
    unique_file_descriptor m_ownedFd;  // This should not be used, it is only for automatic closing!
#endif
};
}  // namespace rapidgzip

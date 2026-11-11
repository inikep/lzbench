#pragma once

#include <algorithm>
#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

#ifndef _MSC_VER
    #include <unistd.h>
#endif

#include <core/common.hpp>
#include <core/Statistics.hpp>

#include "FileReader.hpp"
#include "SinglePass.hpp"

#ifdef WITH_PYTHON_SUPPORT
    #include "Python.hpp"
#endif

#ifndef _MSC_VER
    #include "Standard.hpp"
#endif


namespace rapidgzip
{
class SharedFileReader final :
    public FileReader
{
private:
    /**
     * Create a new shared file reader from an existing FileReader. Takes ownership of the given FileReader!
     */
    explicit
    SharedFileReader( FileReader* file ) :
        m_statistics( dynamic_cast<SharedFileReader*>( file ) == nullptr
                      ? std::make_shared<AccessStatistics>()
                      : dynamic_cast<SharedFileReader*>( file )->m_statistics ),
        m_mutex( dynamic_cast<SharedFileReader*>( file ) == nullptr
                 ? std::make_shared<std::mutex>()
                 : dynamic_cast<SharedFileReader*>( file )->m_mutex ),
        m_fileSizeBytes( file == nullptr ? std::make_optional<size_t>( 0 ) : file->size() ),
        m_currentPosition( file == nullptr ? 0 : file->tell() )
    {
        if ( file == nullptr ) {
            throw std::invalid_argument( "File reader may not be null!" );
        }

    #ifndef _MSC_VER
        if ( auto* const sharedFile = dynamic_cast<StandardFileReader*>( file ); sharedFile != nullptr ) {
            m_fileDescriptor = file->fileno();
        }
    #endif

        /* Fall back to a clone-like copy of all members if the source is also a SharedFileReader.
         * Most of the members are already copied inside the member initializer list. */
        if ( auto* const sharedFile = dynamic_cast<SharedFileReader*>( file ); sharedFile != nullptr ) {
            m_sharedFile = sharedFile->m_sharedFile;
            return;
        }

        if ( !file->seekable() ) {
            throw std::invalid_argument( "This class heavily relies on seeking and won't work with unseekable files!" );
        }

        m_sharedFile =
            std::shared_ptr<FileReader>(
                file,
                [] ( auto* const p ) {
                    if ( ( p != nullptr ) && !p->closed() ) {
                        p->close();
                    }
                    delete p;  // NOLINT(cppcoreguidelines-owning-memory)
                }
            );
    }

public:
    explicit
    SharedFileReader( UniqueFileReader file ) :
        SharedFileReader( file.release() )
    {}

    ~SharedFileReader() override
    {
        if ( m_statistics && m_statistics->showProfileOnDestruction && ( m_statistics.use_count() == 1 ) ) {
            const auto nTimesRead = m_fileSizeBytes
                                    ? m_statistics->read.sum / static_cast<double>( *m_fileSizeBytes )
                                    : 0;

            std::cerr << ( ThreadSafeOutput()
                << "[SharedFileReader::~SharedFileReader]\n"
                << "   seeks back    : (" << m_statistics->seekBack.formatAverageWithUncertainty( true )
                << " ) B (" << m_statistics->seekBack.count << "calls )\n"
                << "   seeks forward : (" << m_statistics->seekForward.formatAverageWithUncertainty( true )
                << " ) B (" << m_statistics->seekForward.count << "calls )\n"
                << "   reads         : (" << m_statistics->read.formatAverageWithUncertainty( true )
                << " ) B (" << m_statistics->read.count << "calls )\n"
                << "   locks         :" << m_statistics->locks << "\n"
                << "   read in total" << static_cast<uint64_t>( m_statistics->read.sum )
                << "B out of" << size().value_or( 0 ) << "B,"
                << "i.e., read the file" << nTimesRead << "times\n"
                << "   time spent seeking and reading:" << m_statistics->readingTime << "s\n"
            );
        }
    }

    /**
     * Creates a shallow copy of this file reader with an independent file position to access the underlying file.
     */
    [[nodiscard]] UniqueFileReader
    cloneRaw() const override
    {
        /* Cannot use std::make_unique because the copy constructor is private! */
        return UniqueFileReader( new SharedFileReader( *this ) );
    }

    void
    setStatisticsEnabled( bool enabled )
    {
        if ( m_statistics ) {
            m_statistics->enabled = enabled;
        }
    }

    void
    setShowProfileOnDestruction( bool showProfileOnDestruction )
    {
        if ( m_statistics ) {
            m_statistics->showProfileOnDestruction = showProfileOnDestruction;
        }
    }

private:
    /**
     * Create a new shared file reader from an existing SharedFileReader by copying shared pointers.
     * The underlying file and mutex are held as a shared_ptr and therefore not copied itself!
     * Make it private because only @ref clone should call this. It cannot be defaulted because the FileReader
     * base class deleted its copy constructor.
     */
    SharedFileReader( const SharedFileReader& other ) :
        m_statistics( other.m_statistics ),
        m_sharedFile( other.m_sharedFile ),
        m_fileDescriptor( other.m_fileDescriptor ),
        m_mutex( other.m_mutex ),
        m_fileSizeBytes( other.m_fileSizeBytes ),
        m_currentPosition( other.m_currentPosition )
    {}

public:
    void
    close() override
    {
        /* This is a shared file. Closing the underlying file while it might be used by another process,
         * seems like a bug-prone functionality. If you really want to close a file, delete all SharedFileReader
         * classes using it. It will be closed by the last destructor @see deleter set in the constructor. */
        const auto lock = getLock();
        m_sharedFile.reset();
    }

    [[nodiscard]] bool
    closed() const override
    {
        const auto lock = getLock();
        return !m_sharedFile || m_sharedFile->closed();
    }

    [[nodiscard]] bool
    eof() const override
    {
        /* m_sharedFile->eof() won't work because some other thread might set the EOF bit on the underlying file! */
        const auto fileSize = size();
        return fileSize ? m_currentPosition >= *fileSize : false;
    }

    [[nodiscard]] bool
    fail() const override
    {
        const auto lock = getLock();
        return !m_sharedFile || m_sharedFile->fail();
    }

    [[nodiscard]] int
    fileno() const override
    {
        if ( m_fileDescriptor >= 0 ) {
            return m_fileDescriptor;
        }

        const auto lock = getLock();
        if ( m_sharedFile ) {
            return m_sharedFile->fileno();
        }
        throw std::invalid_argument( "Invalid or closed SharedFileReader has no associated fileno!" );
    }

    [[nodiscard]] bool
    seekable() const override
    {
        return true;
    }

    [[nodiscard]] std::optional<size_t>
    size() const override
    {
        if ( m_fileSizeBytes.has_value() ) {
            return m_fileSizeBytes;
        }

        const auto lock = getLock();
        return m_sharedFile ? m_sharedFile->size() : std::nullopt;
    }

    size_t
    seek( long long int offset,
          int           origin = SEEK_SET ) override
    {
        if ( ( origin == SEEK_END ) && !size().has_value() ) {
            const auto fileLock = getLock();
            m_currentPosition = m_sharedFile->seek( offset, origin );
            /* File size must have become available when seeking relative to end. */
            m_fileSizeBytes = m_sharedFile->size();
            if ( const auto fileSize = size(); fileSize ) {
                m_currentPosition = std::min( m_currentPosition, *fileSize );
            }
        } else {
            m_currentPosition = effectiveOffset( offset, origin );
        }

        return m_currentPosition;
    }

    [[nodiscard]] size_t
    read( char*  buffer,
          size_t nMaxBytesToRead ) override
    {
        if ( buffer == nullptr ) {
            throw std::invalid_argument( "Buffer may not be nullptr!" );
        }

        if ( nMaxBytesToRead == 0 ) {
            return 0;
        }

        /* Copy the shared pointer in order to avoid race-conditions with @ref close, which might clear
         * @ref m_sharedFile. This copy avoids having to keep m_mutex locked the whole time. Else, we would
         * have to recheck m_sharedFile before after each relock. Note that there are two semantics here:
         * 1. Locking access to the underlying file
         * 2. Locking access to the shared_ptr member. */
        const auto sharedFile =
            [this] () {
                const auto lock = getLock();
                return std::shared_ptr<FileReader>( m_sharedFile );
            } ();
        if ( !sharedFile ) {
            throw std::invalid_argument( "Invalid SharedFileReader cannot be read from!" );
        }

        const auto t0 = now();
        size_t nBytesRead{ 0 };
    #if defined(__linux__) || defined(__APPLE__)
        const auto fileSize = size();
        if ( m_usePread && ( m_fileDescriptor >= 0 ) && fileSize.has_value() && sharedFile->seekable() ) {
            /* This statistic only approximates the actual pread behavior. The OS can probably reorder
             * concurrent pread calls and we would have to enclose pread itself in a lock, which defeats
             * the purpose of pread for speed. */
            if ( m_statistics && m_statistics->enabled ) {
                const std::scoped_lock lock{ m_statistics->mutex };

                auto oldOffset = static_cast<size_t>( m_statistics->lastAccessOffset );
                auto newOffset = m_currentPosition;
                if ( m_fileSizeBytes ) {
                    oldOffset = std::min( oldOffset, *m_fileSizeBytes );
                    newOffset = std::min( newOffset, *m_fileSizeBytes );
                }

                if ( newOffset > oldOffset ) {
                    m_statistics->seekForward.merge( newOffset - oldOffset );
                } else if ( newOffset < oldOffset ) {
                    m_statistics->seekBack.merge( oldOffset - newOffset );
                }
                m_statistics->lastAccessOffset = newOffset;
            }

            nMaxBytesToRead = std::min( nMaxBytesToRead, *fileSize - m_currentPosition );
            const auto nBytesReadWithPread = ::pread( sharedFile->fileno(), buffer, nMaxBytesToRead,
                                                      m_currentPosition );  // NOLINT
            if ( ( nBytesReadWithPread == 0 ) && !m_fileSizeBytes.has_value() ) {
                /* EOF reached. A lock should not be necessary because the file size should not change after EOF has
                 * has been reached but, as it will only be locked once, the performance overhead is negligible
                 * for the amount of security it brings against weird implementations for m_sharedFile. */
                const auto fileLock = getLock();
                m_fileSizeBytes = sharedFile->size();
            }
            if ( nBytesReadWithPread < 0 ) {
                throw std::runtime_error( "Failed to read from file!" );
            }
            nBytesRead = static_cast<size_t>( nBytesReadWithPread );
        } else
    #endif
        {
            const auto fileLock = getLock();

            if ( m_statistics && m_statistics->enabled ) {
                const std::scoped_lock lock{ m_statistics->mutex };
                const auto oldOffset = sharedFile->tell();
                if ( m_currentPosition > oldOffset ) {
                    m_statistics->seekForward.merge( m_currentPosition - oldOffset );
                } else if ( m_currentPosition < oldOffset ) {
                    m_statistics->seekBack.merge( oldOffset - m_currentPosition );
                }
            }

            /* Seeking alone does not clear the EOF nor fail bit if the last read did set it. */
            sharedFile->clearerr();
            sharedFile->seekTo( m_currentPosition );
            nBytesRead = sharedFile->read( buffer, nMaxBytesToRead );
            if ( ( nBytesRead == 0 ) && !m_fileSizeBytes ) {
                m_fileSizeBytes = sharedFile->size();
            }
        }

        if ( m_statistics && m_statistics->enabled ) {
            const std::scoped_lock lock{ m_statistics->mutex };
            m_statistics->read.merge( nBytesRead );
            m_statistics->readingTime += duration( t0 );
        }

        m_currentPosition += nBytesRead;
        return nBytesRead;
    }

    [[nodiscard]] size_t
    tell() const override
    {
        /* Do not use m_sharedFile->tell() because another thread might move that internal file position! */
        return m_currentPosition;
    }

    void
    clearerr() override
    {
        throw std::invalid_argument( "Not implemented because after clearing error another thread might "
                                     "set an error again right away, which makes this interface useless." );
    }

    /**
     * In order to avoid deadlocks where:
     * - Thread 1 has the GIL and tries to acquire the SharedFileReader lock
     * - Thread 2 has the SharedFileReader lock and tries to acquire the GIL
     * this enforces a fixed lock order
     * The order per se locks the std::mutex first and only then acquires the GIL.
     * The other way around has been proven to not work because the GIL gets released at seemingly
     * arbitrary points, in the given test case inside PythonFileReader::read -> PyObject_Call somewhere
     * between PyObject_Call and _Py_Read, which tries to lock the GIL again, which might deadlock.
     * But then there is the Python main thread, which has the GIL already locked when calling into this function!
     * This messes up our lock order, so therefore, we need to try to unlock the GIL before trying to lock
     * @ref m_fileLock. In case it already is unlocked, then ScopedGILUnlock will not do anything.
     */
    struct FileLock
    {
        explicit
        FileLock( std::mutex& mutex ) :
            m_fileLock( mutex )
        {}

    private:
    #ifdef WITH_PYTHON_SUPPORT
        const ScopedGILUnlock m_globalInterpreterUnlock;
    #endif
        const std::unique_lock<std::mutex> m_fileLock;
    #ifdef WITH_PYTHON_SUPPORT
        const ScopedGILLock m_globalInterpreterLock;
    #endif
    };

    /**
     * @return Raw pointer to underlying FileReader and lock, which acts as a kind of borrow together.
     *         The raw pointer must not be used after the lock has been destroyed.
     */
    [[nodiscard]] std::pair<std::unique_ptr<FileLock>, FileReader*>
    underlyingFile()
    {
        return { getUniqueLock(), m_sharedFile.get() };
    }

    void
    setUsePread( bool use )
    {
        m_usePread = use;
    }

    [[nodiscard]] bool
    usePread() const noexcept
    {
        return m_usePread;
    }

private:
    /**
     * This should be used for internal locks to avoid the allocations by std::unique_ptr for each access
     * to @ref size() for example.
     */
    [[nodiscard]] FileLock
    getLock() const
    {
        if ( m_statistics && m_statistics->enabled ) {
            ++m_statistics->locks;
        }
        return FileLock( *m_mutex );
    }

    [[nodiscard]] std::unique_ptr<FileLock>
    getUniqueLock() const
    {
        if ( m_statistics && m_statistics->enabled ) {
            ++m_statistics->locks;
        }
        return std::make_unique<FileLock>( *m_mutex );
    }

private:
    struct AccessStatistics {
        bool showProfileOnDestruction{ false };
        bool enabled{ false };
        uint64_t lastAccessOffset{ 0 };  // necessary for pread because tell() won't work
        Statistics<uint64_t> read;
        Statistics<uint64_t> seekBack;
        Statistics<uint64_t> seekForward;
        double readingTime{ 0 };
        std::atomic<uint64_t> locks{ 0 };
        std::mutex mutex{};
    };

private:
    const std::shared_ptr<AccessStatistics> m_statistics;

    std::shared_ptr<FileReader> m_sharedFile;
    int m_fileDescriptor{ -1 };
    const std::shared_ptr<std::mutex> m_mutex;

    /** This is only for performance to avoid querying the file. */
    std::optional<size_t> m_fileSizeBytes;

    /**
     * This is the independent file pointer that this class offers! Each seek call will only update this and
     * each read call will seek to this offset in an atomic manner before reading from the underlying file.
     */
    size_t m_currentPosition{ 0 };

    bool m_usePread{ true };
};


[[nodiscard]] inline std::unique_ptr<SharedFileReader>
ensureSharedFileReader( UniqueFileReader&& fileReader )
{
    if ( !fileReader ) {
        throw std::invalid_argument( "File reader must not be null!" );
    }

    if ( auto* const casted = dynamic_cast<SharedFileReader*>( fileReader.get() ); casted != nullptr ) {
        fileReader.release();  // NOLINT(bugprone-unused-return-value)
        return std::unique_ptr<SharedFileReader>( casted );
    }

    if ( !fileReader->seekable() ) {
        return std::make_unique<SharedFileReader>( std::make_unique<SinglePassFileReader>( std::move( fileReader ) ) );
    }

    return std::make_unique<SharedFileReader>( std::move( fileReader ) );
}
}  // namespace rapidgzip

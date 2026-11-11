#pragma once

#include <algorithm>


#ifndef __linux__

#include <thread>

namespace rapidgzip
{
inline void
pinThreadToLogicalCore( int /* logicalCoreId */ )
{
    /** @todo */
}


[[nodiscard]] inline unsigned int
availableCores()
{
    return std::thread::hardware_concurrency();
}


[[nodiscard]] inline unsigned int
availablePhysicalCores()
{
    return std::max( 1U, availableCores() );
}
}  // namespace rapidgzip

#else

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>

#ifndef _GNU_SOURCE
    #define _GNU_SOURCE
#endif
#include <sched.h>

namespace rapidgzip
{
[[nodiscard]] inline int
getRequiredBitMaskSize()
{
    const auto nCpusPerElement = sizeof( cpu_set_t ) * 8;
    int nCpus = 0;
    bool einvalError = true;

    while ( einvalError ) {
        nCpus += nCpusPerElement;

        auto* pCpuSet = CPU_ALLOC( nCpus );
        if ( pCpuSet == nullptr ) {
            std::stringstream msg;
            msg << "Could not allocate cpu set for " << nCpus << " CPUs!";
            throw std::runtime_error( std::move( msg ).str() );
        }
        const auto cpuSetSize = CPU_ALLOC_SIZE( nCpus );

        const auto result = sched_getaffinity( /* calling thread */ 0, cpuSetSize, pCpuSet );
        CPU_FREE( pCpuSet );

        if ( ( result != 0 ) && ( errno != EINVAL ) ) {
            std::stringstream msg;
            msg << "An unexpected error occurred on schet_getaffinity: " << result << " with errno " << errno
                << " (" << strerror( errno ) << ")";
            throw std::runtime_error( std::move( msg ).str() );
        }
        einvalError = ( result != 0 ) && ( errno == EINVAL );
    }

    return nCpus;
}


/**
 * Pins the calling thread to the given logical core / processing unit / hardware thread.
 */
inline void
pinThreadToLogicalCore( int logicalCoreId )
{
    /* @see "Handling systems with large CPU affinity masks"
     * https://manpages.courier-mta.org/htmlman2/sched_setaffinity.2.html
     * "If the kernel CPU affinity mask is larger than 1024, then calls of the form:
     *     sched_getaffinity(pid, sizeof(cpu_set_t), &mask);
     * fail with the error EINVAL" -> observed on SGI-UV 2000
     */

    const auto nCpusForSufficientMask = getRequiredBitMaskSize();
    auto* pCpuSet = CPU_ALLOC( nCpusForSufficientMask );
    const auto cpuSetSize = CPU_ALLOC_SIZE( nCpusForSufficientMask );
    CPU_ZERO_S( cpuSetSize, pCpuSet );
    CPU_SET_S( logicalCoreId, cpuSetSize, pCpuSet );
    const auto result = sched_setaffinity( /* set affinity for calling thread */ 0, cpuSetSize, pCpuSet );
    CPU_FREE( pCpuSet );

    if ( result != 0 ) {
        std::stringstream msg;
        msg << "When trying to pin current thread running on logical core "
            << sched_getcpu() << " to " << logicalCoreId << ", sched_setaffinity returned " << result
            << " and errno " << errno << " (" << std::strerror( errno ) << "). "
            << "A bitmask sized " << nCpusForSufficientMask << " was allocated.";
        throw std::runtime_error( std::move( msg ).str() );
    }
}


[[nodiscard]] inline unsigned int
availableCores()
{
    const auto nCpusForSufficientMask = getRequiredBitMaskSize();
    auto* pCpuSet = CPU_ALLOC( nCpusForSufficientMask );
    const auto cpuSetSize = CPU_ALLOC_SIZE( nCpusForSufficientMask );
    CPU_ZERO_S( cpuSetSize, pCpuSet );
    const auto result = sched_getaffinity( /* get affinity for calling thread */ 0, cpuSetSize, pCpuSet );

    if ( result != 0 ) {
        std::stringstream msg;
        msg << "Failed to get affinity, sched_getaffinity returned " << result
            << " and errno " << errno << " (" << std::strerror( errno ) << "). "
            << "A bitmask sized " << nCpusForSufficientMask << " was allocated.";
        throw std::runtime_error( std::move( msg ).str() );
    }

    const auto coreCount = CPU_COUNT( pCpuSet );
    CPU_FREE( pCpuSet );
    return static_cast<unsigned int>( coreCount );
}


/**
 * @return a best-effort number of physical cores available to the calling thread.
 *
 * There is no portable C++ API for this. Reading the topology of every CPU is too expensive on large
 * systems, so only read the kernel's single SMT-active flag and assume a two-way SMT width. If the flag
 * is unavailable, keep the logical count. Restricted affinity masks that expose only one sibling of each
 * core are underestimated, which is acceptable for a best-effort default.
 */
[[nodiscard]] inline unsigned int
availablePhysicalCores()
{
    const auto logicalCores = availableCores();
    std::ifstream smtActive( "/sys/devices/system/cpu/smt/active" );
    int active = 0;
    if ( smtActive >> active ) {
        return active == 0 ? logicalCores : std::max( 1U, logicalCores / 2U );
    }
    return logicalCores;
}
}  // namespace rapidgzip

#endif

#pragma once

#ifdef LIBRAPIDARCHIVE_ENABLE_KERNEL_OVERRIDE
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <string_view>
#endif


/* The translation unit stays on the baseline ISA. The SIMD kernels are compiled with function target
 * attributes and selected at runtime, so the same binary also runs on CPUs without AVX2 or AVX-512. */
#if ( defined( __GNUC__ ) || defined( __clang__ ) ) && ( defined( __x86_64__ ) || defined( __i386__ ) )
    #include <immintrin.h>
    #define LIBRAPIDARCHIVE_COMPILE_X86_SIMD_KERNELS 1
#else
    #define LIBRAPIDARCHIVE_COMPILE_X86_SIMD_KERNELS 0
#endif

/* Some kernels are only compiled for 64-bit x86. */
#if LIBRAPIDARCHIVE_COMPILE_X86_SIMD_KERNELS && defined( __x86_64__ )
    #define LIBRAPIDARCHIVE_COMPILE_X86_64_SIMD_KERNELS 1
#else
    #define LIBRAPIDARCHIVE_COMPILE_X86_64_SIMD_KERNELS 0
#endif


namespace rapidgzip
{
namespace simd
{
/**
 * Runtime-selectable SIMD implementations shared by the dispatch points in the library. Individual
 * components may not implement every variant, so they combine this with their own isSupported().
 */
enum class SimdKernel
{
    AUTOMATIC,
    PORTABLE,
    SSE2,
    SSE41,
    AVX2,
    AVX512,
    NEON,
};


[[nodiscard]] constexpr const char*
toString( const SimdKernel kernel ) noexcept
{
    switch ( kernel )
    {
    case SimdKernel::AUTOMATIC:
        return "automatic";
    case SimdKernel::PORTABLE:
        return "portable";
    case SimdKernel::SSE2:
        return "sse2";
    case SimdKernel::SSE41:
        return "sse4.1";
    case SimdKernel::AVX2:
        return "avx2";
    case SimdKernel::AVX512:
        return "avx512";
    case SimdKernel::NEON:
        return "neon";
    }
    return "unknown";
}


#ifdef LIBRAPIDARCHIVE_ENABLE_KERNEL_OVERRIDE
/**
 * Returns the candidate named by the environment variable. The caller still has to check whether its
 * component implements that kernel.
 */
template<typename Kernel>
[[nodiscard]] inline std::optional<Kernel>
kernelFromEnvironment( const char* const             environmentVariable,
                       std::initializer_list<Kernel> candidates ) noexcept
{
    const auto* const value = std::getenv( environmentVariable );
    if ( value == nullptr ) {
        return std::nullopt;
    }

    const std::string_view name{ value };
    for ( const auto candidate : candidates ) {
        if ( name == toString( candidate ) ) {
            return candidate;
        }
    }
    return std::nullopt;
}
#endif
}  // namespace simd
}  // namespace rapidgzip

/* Some basic macro definitions for high performance.
 * You can use a build system (such as autotools or cmake) to test for
 * these features and define HAVE_* or NO_* to turn them on or off.
 * Otherwise, attempt to determine support based on checking macros
 * defined by the compiler and/or libc. */

/* Define to 1 if the compiler supports __builtin_ctz and friends.
   Clang and Intel both define __GNUC__ (and support __builtin_ctz and
   __builtin_expect) */
#if !defined(NO_BUILTIN_CTZ) && !defined(HAVE_BUILTIN_CTZ)
#  if defined(__GNUC__)
#    define HAVE_BUILTIN_CTZ 1
#  endif
#endif

/* Define to 1 if the compiler supports __builtin_expect. */
#if !defined(NO_BUILTIN_EXPECT) && !defined(HAVE_BUILTIN_EXPECT)
#  if defined(__GNUC__)
#    define HAVE_BUILTIN_EXPECT 1
#  endif
#endif

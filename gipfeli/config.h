/* Some basic macro definitions for high performance.
 * If something is not supported by your system, turn it off.
 * This will be substituted by automatic script in future.*/

/* Define to 1 if the compiler supports __builtin_ctz and friends. */
#ifndef _MSC_VER
	#define HAVE_BUILTIN_CTZ 1
#endif

/* Define to 1 if the compiler supports __builtin_expect. */
#ifndef _MSC_VER
	#define HAVE_BUILTIN_EXPECT 1
#endif

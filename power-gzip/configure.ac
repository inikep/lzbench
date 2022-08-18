# -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

# The Automake init of 1.10 requires at least autoconf 2.62
AC_PREREQ([2.69])

AC_INIT([libnxz],[0.63],[https://github.com/libnxz/power-gzip])
# Update this value for every release: (A:B:C will map to foo.so.(A-C).C.B)
# http://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
LTCURRENT=$(echo "$PACKAGE_VERSION" | cut -d . -f 1)
LTREVISION=$(echo "$PACKAGE_VERSION" | cut -d . -f 2)
LTAGE=0
SOVERSION="$LTCURRENT:$LTREVISION:$LTAGE"
AC_SUBST(SOVERSION)

# Defines host_cpu, host_vendor, and host_os variables.
AC_CANONICAL_HOST
AC_CANONICAL_TARGET

AC_CONFIG_SRCDIR([lib/nx_zlib.c])
AC_CONFIG_FILES([oct/config.mk])
AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([doc/Makefile])
AC_CONFIG_FILES([inc_nx/Makefile])
AC_CONFIG_FILES([lib/Makefile])
AC_CONFIG_FILES([oct/config.sh])
AC_CONFIG_FILES([oct/Makefile])
AC_CONFIG_FILES([samples/Makefile])
AC_CONFIG_FILES([test/Makefile])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config.h])
AM_INIT_AUTOMAKE([1.10 no-define foreign subdir-objects])
AM_MAINTAINER_MODE([disable])
LT_INIT

: ${CFLAGS="-O3 -Wall -Werror -g"}

# Checks for programs.
AC_CHECK_TARGET_TOOL([AR], [ar], [:])
AM_PROG_AS
AC_PROG_AWK
AC_PROG_CC
AC_PROG_CPP
AC_PROG_GREP
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_SED
AX_WITH_PROG(BASH, bash)
AX_WITH_PROG(GZIP, gzip)
AX_WITH_PROG(SHA256SUM, sha256sum)
AX_WITH_PROG(VALGRIND, valgrind, valgrind)
AX_WITH_PROG(WGET, wget)

# Checks for header files.
AC_CHECK_HEADERS([fcntl.h])
AC_CHECK_HEADERS([limits.h])
AC_CHECK_HEADERS([malloc.h])
AC_CHECK_HEADERS([inttypes.h])
AC_CHECK_HEADERS([stddef.h])
AC_CHECK_HEADERS([sys/file.h])
AC_CHECK_HEADERS([sys/ioctl.h])
AC_CHECK_HEADERS([sys/param.h])
AC_CHECK_HEADERS([sys/time.h])
AC_CHECK_HEADERS([string.h])
AC_CHECK_HEADERS([syslog.h])
AC_CHECK_HEADERS([unistd.h])
AC_CHECK_HEADERS([zlib.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_INT16_T
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_OFF_T
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T

# Checks for header files.
AC_HEADER_STDC

# Checks for typedefs, structures, and compiler characteristics.
AC_C_VOLATILE
AC_CHECK_TYPES([ptrdiff_t])

# Checks for library functions.
AC_FUNC_FORK
AC_FUNC_LSTAT_FOLLOWS_SLASHED_SYMLINK
AC_FUNC_MALLOC
AC_FUNC_MMAP
AC_CHECK_FUNCS([ftruncate])
AC_CHECK_FUNCS([gettimeofday])
AC_CHECK_FUNCS([memmove])
AC_CHECK_FUNCS([memset])
AC_CHECK_FUNCS([munmap])
AC_CHECK_FUNCS([strchr])
AC_CHECK_FUNCS([strerror])
AC_CHECK_FUNCS([strrchr])
AC_CHECK_FUNCS([strstr])
AC_CHECK_FUNCS([strtol])
AC_CHECK_FUNCS([strtoull])
# Pthread check (on m4/acx_pthread.m4)
ACX_PTHREAD

AX_CHECK_COMPILE_FLAG([-mcpu=power9],
		      [CFLAGS="$CFLAGS -mcpu=power9"])

# GNU C11 is mandatory in order to build libnxz, however older compilers may
# not enable GNU C11 by default.  Guarantee that GNU C11 is enabled and
# supported by the current compiler.
AC_MSG_CHECKING([whether $CC supports GNU C11 by default])
AC_PREPROC_IFELSE(
[AC_LANG_PROGRAM([[#if !defined __STDC_VERSION__ || __STDC_VERSION__ < 201112L
		   # error "GNU C11 is not available"
		   #endif
]])],
[AC_MSG_RESULT([yes])],
[AC_MSG_RESULT([no])
 AX_CHECK_COMPILE_FLAG([-std=gnu11],
		       [CFLAGS="$CFLAGS -std=gnu11"],
		       [AC_MSG_ERROR([$CC does not support -std=gnu11])])
])

AC_ARG_ENABLE([zlib-api],
	      [  --disable-zlib-api      Enable compatibility with the Zlib API (default enabled)])
AS_IF([test "x$enableval" != xno],
      [CFLAGS="$CFLAGS -DZLIB_API"])

# Doxygen support
DX_HTML_FEATURE(ON)
DX_MAN_FEATURE(ON)
DX_PDF_FEATURE(OFF)
DX_INIT_DOXYGEN($PACKAGE_NAME, [doc/libnxz.doxy])

AC_OUTPUT

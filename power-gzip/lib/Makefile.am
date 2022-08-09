ZLIB_PATH ?= libz.so

SHAREDSONAMELIB = $(PACKAGE_NAME).so.$(SOVERSION)
SHAREDLIB = $(PACKAGE_NAME).so.$(PACKAGE_VERSION)
LIBLINK = $(PACKAGE_NAME).so

lib_LTLIBRARIES = libnxz.la

libnxz_la_SOURCES = crc32_ppc.c \
		    crc32_ppc.h \
		    crc32_ppc_constants.h \
		    crc32_ppc_asm.S \
		    gzip_vas.c \
		    nx_adler32.c \
		    nx_compress.c \
		    nx_crc.c \
		    nx_deflate.c \
		    nx_dht.c \
		    nx_dhtgen.c \
		    nx_dht_builtin.c \
		    nx_inflate.c \
		    nx_uncompr.c \
		    nx_utils.c \
		    nx_zlib.c \
		    nx_zlib.h \
		    ppc-asm.h \
		    ppc-opcode.h \
		    sw_zlib.c
libnxz_la_LIBADD = $(PTHREAD_LIBS) -ldl
libnxz_la_LDFLAGS = -version-info $(SOVERSION) \
		    -Wl,--version-script=$(srcdir)/Versions

AM_CPPFLAGS = -I$(top_srcdir)/inc_nx -DZLIB_PATH=\"$(ZLIB_PATH)\"

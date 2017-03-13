# ################################################################
# Lizard library - Makefile
# Copyright (C) 2011-2015, Yann Collet
# Copyright (C) 2016-2017, Przemyslaw Skibinski <inikep@gmail.com>
# All rights reserved.
#
# This Makefile is validated for Linux, macOS, *BSD, Hurd, Solaris, MSYS2 targets
#
# BSD license
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# You can contact the author at :
#  - Lizard source repository : https://github.com/inikep/lizard
# ################################################################

# Version numbers
LIBVER_MAJOR_SCRIPT:=`sed -n '/define LIZARD_VERSION_MAJOR/s/.*[[:blank:]]\([0-9][0-9]*\).*/\1/p' < ./lizard_compress.h`
LIBVER_MINOR_SCRIPT:=`sed -n '/define LIZARD_VERSION_MINOR/s/.*[[:blank:]]\([0-9][0-9]*\).*/\1/p' < ./lizard_compress.h`
LIBVER_PATCH_SCRIPT:=`sed -n '/define LIZARD_VERSION_RELEASE/s/.*[[:blank:]]\([0-9][0-9]*\).*/\1/p' < ./lizard_compress.h`
LIBVER_SCRIPT:= $(LIBVER_MAJOR_SCRIPT).$(LIBVER_MINOR_SCRIPT).$(LIBVER_PATCH_SCRIPT)
LIBVER_MAJOR := $(shell echo $(LIBVER_MAJOR_SCRIPT))
LIBVER_MINOR := $(shell echo $(LIBVER_MINOR_SCRIPT))
LIBVER_PATCH := $(shell echo $(LIBVER_PATCH_SCRIPT))
LIBVER := $(shell echo $(LIBVER_SCRIPT))

BUILD_STATIC:= yes

CPPFLAGS= -DXXH_NAMESPACE=Lizard_
CFLAGS ?= -O3
CFLAGS += -Wall -Wextra -Wcast-qual -Wcast-align -Wshadow \
          -Wswitch-enum -Wdeclaration-after-statement -Wstrict-prototypes -Wundef \
          -Wpointer-arith -Wstrict-aliasing=1
CFLAGS  += $(MOREFLAGS) 
FLAGS   = $(CPPFLAGS) $(CFLAGS) $(LDFLAGS)



# OS X linker doesn't support -soname, and use different extension
# see : https://developer.apple.com/library/mac/documentation/DeveloperTools/Conceptual/DynamicLibraries/100-Articles/DynamicLibraryDesignGuidelines.html
ifeq ($(shell uname), Darwin)
	SHARED_EXT = dylib
	SHARED_EXT_MAJOR = $(LIBVER_MAJOR).$(SHARED_EXT)
	SHARED_EXT_VER = $(LIBVER).$(SHARED_EXT)
	SONAME_FLAGS = -install_name $(PREFIX)/lib/liblizard.$(SHARED_EXT_MAJOR) -compatibility_version $(LIBVER_MAJOR) -current_version $(LIBVER)
else
	SONAME_FLAGS = -Wl,-soname=liblizard.$(SHARED_EXT).$(LIBVER_MAJOR)
	SHARED_EXT = so
	SHARED_EXT_MAJOR = $(SHARED_EXT).$(LIBVER_MAJOR)
	SHARED_EXT_VER = $(SHARED_EXT).$(LIBVER)
endif

default: lib

all: lib

lib: liblizard.a liblizard

liblizard.a: *.c xxhash/*.c entropy/*.c
ifeq ($(BUILD_STATIC),yes)
	@echo compiling static library
	@$(CC) $(FLAGS) -c $^
	@$(AR) rcs $@ *.o
endif

liblizard: *.c xxhash/*.c entropy/*.c
	@echo compiling dynamic library $(LIBVER)
ifneq (,$(filter Windows%,$(OS)))
	@$(CC) $(FLAGS) -DLIZARD_DLL_EXPORT=1 -shared $^ -o dll\$@.dll
	dlltool -D dll\liblizard.dll -d dll\liblizard.def -l dll\liblizard.lib
else
	@$(CC) $(FLAGS) -shared $^ -fPIC $(SONAME_FLAGS) -o $@.$(SHARED_EXT_VER)
	@echo creating versioned links
	@ln -sf $@.$(SHARED_EXT_VER) $@.$(SHARED_EXT_MAJOR)
	@ln -sf $@.$(SHARED_EXT_VER) $@.$(SHARED_EXT)
endif

clean:
	@rm -f core entropy/*.o xxhash/*.o *.o *.a *.$(SHARED_EXT) *.$(SHARED_EXT).* liblizard.pc dll/liblizard.dll dll/liblizard.lib
	@echo Cleaning library completed


#-----------------------------------------------------------------------------
# make install is validated only for Linux, OSX, BSD, Hurd, Haiku and Solaris targets
#-----------------------------------------------------------------------------
ifneq (,$(filter $(shell uname),Linux Darwin GNU/kFreeBSD GNU OpenBSD FreeBSD NetBSD DragonFly SunOS Haiku))

ifneq (,$(filter $(shell uname),SunOS))
INSTALL ?= ginstall
else
INSTALL ?= install
endif

PREFIX     ?= /usr/local
DESTDIR    ?=
LIBDIR     ?= $(PREFIX)/lib
INCLUDEDIR ?= $(PREFIX)/include

ifneq (,$(filter $(shell uname),OpenBSD FreeBSD NetBSD DragonFly))
PKGCONFIGDIR ?= $(PREFIX)/libdata/pkgconfig
else
PKGCONFIGDIR ?= $(LIBDIR)/pkgconfig
endif

INSTALL_LIB  ?= $(INSTALL) -m 755
INSTALL_DATA ?= $(INSTALL) -m 644

liblizard.pc: liblizard.pc.in Makefile
	@echo creating pkgconfig
	@sed -e 's|@PREFIX@|$(PREFIX)|' \
            -e 's|@LIBDIR@|$(LIBDIR)|' \
            -e 's|@INCLUDEDIR@|$(INCLUDEDIR)|' \
            -e 's|@VERSION@|$(LIBVER)|' \
             $< >$@

install: lib liblizard.pc
	@$(INSTALL) -d -m 755 $(DESTDIR)$(PKGCONFIGDIR)/ $(DESTDIR)$(INCLUDEDIR)/
	@$(INSTALL_DATA) liblizard.pc $(DESTDIR)$(PKGCONFIGDIR)/
	@echo Installing libraries
ifeq ($(BUILD_STATIC),yes)
	@$(INSTALL_LIB) liblizard.a $(DESTDIR)$(LIBDIR)/liblizard.a
endif
	@$(INSTALL_LIB) liblizard.$(SHARED_EXT_VER) $(DESTDIR)$(LIBDIR)
	@ln -sf liblizard.$(SHARED_EXT_VER) $(DESTDIR)$(LIBDIR)/liblizard.$(SHARED_EXT_MAJOR)
	@ln -sf liblizard.$(SHARED_EXT_VER) $(DESTDIR)$(LIBDIR)/liblizard.$(SHARED_EXT)
	@echo Installing includes
	@$(INSTALL_DATA) lizard_compress.h $(DESTDIR)$(INCLUDEDIR)/lizard_compress.h
	@$(INSTALL_DATA) lizard_common.h $(DESTDIR)$(INCLUDEDIR)/lizard_common.h
	@$(INSTALL_DATA) lizard_frame.h $(DESTDIR)$(INCLUDEDIR)/lizard_frame.h
	@echo lizard static and shared libraries installed

uninstall:
	@$(RM) $(DESTDIR)$(PKGCONFIGDIR)/liblizard.pc
	@$(RM) $(DESTDIR)$(LIBDIR)/liblizard.$(SHARED_EXT)
	@$(RM) $(DESTDIR)$(LIBDIR)/liblizard.$(SHARED_EXT_MAJOR)
	@$(RM) $(DESTDIR)$(LIBDIR)/liblizard.$(SHARED_EXT_VER)
	@$(RM) $(DESTDIR)$(LIBDIR)/liblizard.a
	@$(RM) $(DESTDIR)$(INCLUDEDIR)/lizard_compress.h
	@$(RM) $(DESTDIR)$(INCLUDEDIR)/lizard_common.h
	@$(RM) $(DESTDIR)$(INCLUDEDIR)/lizard_frame.h
	@echo lizard libraries successfully uninstalled

endif

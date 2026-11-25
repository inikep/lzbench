TOPDIR     := $(PWD)
DESTDIR    :=
PREFIX     := /usr/local
LIBDIR     := $(PREFIX)/lib
MANDIR     := $(PREFIX)/share/man

CROSS_COMPILE :=
CC         := $(CROSS_COMPILE)gcc
PLATFORM   := $(shell $(CC) -dumpmachine | sed -e 's,-gnu[^-]*,,; s,.*-,,')

# BUILD=release is the default. Setting it to anything else (e.g. "make BUILD="
# or "make BUILD=dev") makes a development build, which defines SLZ_DEV_BUILD
# in every C file to expose debugging features. Its name is passed in
# SLZ_BUILD. Run "make clean" when changing it, the objects don't depend on it.
BUILD      := release
ifeq ($(BUILD),release)
BLD_CFLAGS := -DSLZ_BUILD='"$(BUILD)"'
else
BLD_CFLAGS := -DSLZ_BUILD='"$(BUILD)"' -DSLZ_DEV_BUILD
endif

OPT_CFLAGS := -O3
CPU_CFLAGS := -fomit-frame-pointer
DEB_CFLAGS := -Wall -g
DEF_CFLAGS :=
USR_CFLAGS :=
LIB_CFLAGS := -fPIC -DPRECOMPUTE_TABLES
INC_CFLAGS := -I$(TOPDIR)/include
CFLAGS     := $(OPT_CFLAGS) $(CPU_CFLAGS) $(DEB_CFLAGS) $(DEF_CFLAGS) $(BLD_CFLAGS) $(USR_CFLAGS) $(INC_CFLAGS)

LD         := $(CC)
DEB_LFLAGS := -g
USR_LFLAGS :=
LIB_LFLAGS :=
LDFLAGS    := $(DEB_LFLAGS) $(USR_LFLAGS) $(LIB_LFLAGS)

AR         := $(CROSS_COMPILE)ar
STRIP      := $(CROSS_COMPILE)strip
BINS       := zdecode slz zenc zdec
STATIC     := libslz.a
ifneq ($(filter darwin%, $(PLATFORM)),)
	SHARED := libslz.dylib
	SONAME := libslz.1.dylib
else
	SHARED := libslz.so
	SONAME := $(SHARED).1
endif
OBJS       :=
OBJS       += $(patsubst %.c,%.o,$(wildcard src/*.c))
OBJS       += $(patsubst %.S,%.o,$(wildcard src/*.S))

# retrieve the version from the current Git tree unless IGNOREGIT is set
ifeq ($(IGNOREGIT),)
  VERSION := $(shell [ -d .git/. ] && (git describe --tags --match 'v*' --abbrev=0 | cut -c 2-) 2>/dev/null)
  ifneq ($(VERSION),)
    # OK git is there and works.
    SUBVERS := $(shell comms=`git log --format=oneline --no-merges v$(VERSION).. 2>/dev/null | wc -l | tr -d '[:space:]'`; commit=`(git log -1 --pretty=%h --abbrev=6) 2>/dev/null`; [ $$comms -gt 0 ] && echo "-$$commit-$$comms")
  endif
endif

ifeq ($(VERSION),)
  VERSION := $(shell cat VERSION 2>/dev/null)
  SUBVERS :=
endif

all: static shared tools

static: $(STATIC)

shared: $(SHARED)

tools: $(BINS)

zdecode: tools/zdecode.o
	$(LD) $(LDFLAGS) -o $@ $^

slz: tools/zenc.o src/slz_common.o src/slz.o src/uslz.o
	$(LD) $(LDFLAGS) -o $@ $^

zenc zdec: slz
	ln -sf $^ $@

$(STATIC): src/slz_common.o src/slz.o src/uslz.o
	$(AR) rv $@ $^

$(SONAME): src/slz-pic.o
ifneq ($(filter darwin%, $(PLATFORM)),)
	$(LD) -dynamiclib $(LDFLAGS) -install_name $@ -o $@ $^
else
ifneq ($(filter aix%, $(PLATFORM)),)
	$(LD) -shared $(LDFLAGS) -Wl,-G -o $@ $^
else
	$(LD) -shared $(LDFLAGS) -Wl,-soname,$@ -o $@ $^
endif
endif

$(SHARED): $(SONAME)
	ln -sf $^ $@

%.o: %.c src/slz.h src/tables.h
	$(CC) $(CFLAGS) -o $@ -DSLZ_VERSION='"'$(VERSION)$(SUBVERS)'"' -c $<

%-pic.o: %.c src/slz.h src/tables.h
	$(CC) $(CFLAGS) $(LIB_CFLAGS) -o $@ -c $<

install: install-headers install-static install-shared install-tools install-man

install-headers:
	[ -d "$(DESTDIR)$(PREFIX)/include/." ] || mkdir -p -m 0755 $(DESTDIR)$(PREFIX)/include
	cp src/slz.h $(DESTDIR)$(PREFIX)/include/slz.h
	chmod 644 $(DESTDIR)$(PREFIX)/include/slz.h

install-static: static
	[ -d "$(DESTDIR)$(LIBDIR)/." ] || mkdir -p -m 0755 $(DESTDIR)$(LIBDIR)
	cp $(STATIC) $(DESTDIR)$(LIBDIR)/$(STATIC)
	chmod 644 $(DESTDIR)$(LIBDIR)/$(STATIC)

install-shared: shared
	[ -d "$(DESTDIR)$(LIBDIR)/." ] || mkdir -p -m 0755 $(DESTDIR)$(LIBDIR)
	cp    $(SONAME) $(DESTDIR)$(LIBDIR)/$(SONAME)
	cp -P $(SHARED) $(DESTDIR)$(LIBDIR)/$(SHARED)
	chmod 644 $(DESTDIR)$(LIBDIR)/$(SONAME)

install-tools: tools
	$(STRIP) slz
	[ -d "$(DESTDIR)$(PREFIX)/bin/." ] || mkdir -p -m 0755 $(DESTDIR)$(PREFIX)/bin
	cp zdecode $(DESTDIR)$(PREFIX)/bin/zdecode
	cp slz $(DESTDIR)$(PREFIX)/bin/slz
	chmod 755 $(DESTDIR)$(PREFIX)/bin/zdecode
	chmod 755 $(DESTDIR)$(PREFIX)/bin/slz

install-man:
	[ -d "$(DESTDIR)$(MANDIR)/man1/." ] || mkdir -p -m 0755 $(DESTDIR)$(MANDIR)/man1
	cp tools/slz.1 $(DESTDIR)$(MANDIR)/man1/slz.1
	chmod 644 $(DESTDIR)$(MANDIR)/man1/slz.1

clean:
	-rm -f $(BINS) $(OBJS) $(STATIC) $(SHARED) *.[oa] *.so *.so.* *.dylib *~ */*.[oa] */*~

snapshot:
	ver=$$(git describe --tags --match 'v*' HEAD | cut -c2-); ver=$${ver%-g*}; \
	git archive --format=tar --prefix="libslz-$${ver}/" HEAD | gzip -9 > libslz-$${ver}.tar.gz

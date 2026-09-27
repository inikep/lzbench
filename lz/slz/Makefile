TOPDIR     := $(PWD)
DESTDIR    :=
PREFIX     := /usr/local
LIBDIR     := $(PREFIX)/lib

CROSS_COMPILE :=
CC         := $(CROSS_COMPILE)gcc
PLATFORM   := $(shell $(CC) -dumpmachine | sed -e 's,-gnu[^-]*,,; s,.*-,,')

OPT_CFLAGS := -O3
CPU_CFLAGS := -fomit-frame-pointer
DEB_CFLAGS := -Wall -g
DEF_CFLAGS :=
USR_CFLAGS :=
LIB_CFLAGS := -fPIC -DPRECOMPUTE_TABLES
INC_CFLAGS := -I$(TOPDIR)/include
CFLAGS     := $(OPT_CFLAGS) $(CPU_CFLAGS) $(DEB_CFLAGS) $(DEF_CFLAGS) $(USR_CFLAGS) $(INC_CFLAGS)

LD         := $(CC)
DEB_LFLAGS := -g
USR_LFLAGS :=
LIB_LFLAGS :=
LDFLAGS    := $(DEB_LFLAGS) $(USR_LFLAGS) $(LIB_LFLAGS)

AR         := $(CROSS_COMPILE)ar
STRIP      := $(CROSS_COMPILE)strip
BINS       := zdecode zdec zenc
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

zdecode: src/zdecode.o
	$(LD) $(LDFLAGS) -o $@ $^

zdec: src/zdec.o src/slz_common.o src/uslz.o
	$(LD) $(LDFLAGS) -o $@ $^

zenc: src/zenc.o src/slz_common.o src/slz.o
	$(LD) $(LDFLAGS) -o $@ $^

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

install: install-headers install-static install-shared install-tools

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
	$(STRIP) zenc
	[ -d "$(DESTDIR)$(PREFIX)/bin/." ] || mkdir -p -m 0755 $(DESTDIR)$(PREFIX)/bin
	cp zdecode $(DESTDIR)$(PREFIX)/bin/zdecode
	cp zdecode $(DESTDIR)$(PREFIX)/bin/zdec
	cp zenc $(DESTDIR)$(PREFIX)/bin/zenc
	chmod 755 $(DESTDIR)$(PREFIX)/bin/zdecode
	chmod 755 $(DESTDIR)$(PREFIX)/bin/zdec
	chmod 755 $(DESTDIR)$(PREFIX)/bin/zenc

clean:
	-rm -f $(BINS) $(OBJS) $(STATIC) $(SHARED) *.[oa] *.so *.so.* *.dylib *~ */*.[oa] */*~

snapshot:
	ver=$$(git describe --tags --match 'v*' HEAD | cut -c2-); ver=$${ver%-g*}; \
	git archive --format=tar --prefix="libslz-$${ver}/" HEAD | gzip -9 > libslz-$${ver}.tar.gz

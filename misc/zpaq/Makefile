CXX=g++
CPPFLAGS+=-Dunix
# CPPFLAGS+=NOJIT
CXXFLAGS=-O3 -march=native
PREFIX=/usr/local
BINDIR=$(PREFIX)/bin
MANDIR=$(PREFIX)/share/man

all: zpaq zpaq.1

libzpaq.o: libzpaq.cpp libzpaq.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c libzpaq.cpp

zpaq.o: zpaq.cpp libzpaq.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c zpaq.cpp -pthread

zpaq: zpaq.o libzpaq.o
	$(CXX) $(LDFLAGS) -o $@ zpaq.o libzpaq.o -pthread

zpaq.1: zpaq.pod
	pod2man $< >$@

install: zpaq zpaq.1
	install -m 0755 -d $(DESTDIR)$(BINDIR)
	install -m 0755 zpaq $(DESTDIR)$(BINDIR)
	install -m 0755 -d $(DESTDIR)$(MANDIR)/man1
	install -m 0644 zpaq.1 $(DESTDIR)$(MANDIR)/man1

clean:
	rm -f zpaq.o libzpaq.o zpaq zpaq.1 archive.zpaq zpaq.new

check: zpaq
	./zpaq add archive.zpaq zpaq
	./zpaq extract archive.zpaq zpaq -to zpaq.new
	cmp zpaq zpaq.new
	rm archive.zpaq zpaq.new

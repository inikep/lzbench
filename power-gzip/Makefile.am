EMPTY_AUTOMAKE_TARGETS=selftest

ACLOCAL_AMFLAGS = -I m4

include aminclude.am

# List of directories built by default.
SUBDIRS = lib test samples

DIST_SUBDIRS = doc inc_nx oct

dist_doc_DATA = licenses/APACHE-2.0.txt licenses/gpl-2.0.txt

include_HEADERS = libnxz.h

if DX_COND_doc
doc_target = doc
endif DX_COND_doc

VALGRIND_FLAGS="--leak-check=full \
		--suppressions=$(abs_top_srcdir)/test/valgrind.supp \
		--error-exitcode=1"

all: $(SUBDIRS) $(doc_target)

$(SUBDIRS):
	@if [ -d $@ ]; then				\
		$(MAKE) -C $@ || exit 1;	 	\
	fi

bench: $(SUBDIRS)
	$(MAKE) -C samples $@

valgrind-check: $(SUBDIRS)
	$(MAKE) -C test check \
	  LOG_COMPILER="${VALGRIND}" LOG_FLAGS=$(VALGRIND_FLAGS)

if DX_COND_doc
doc: doc/libnxz.doxy
	doxygen $<

MOSTLYCLEANFILES = $(DX_CLEANFILES)
endif DX_COND_doc

selftest:
	$(MAKE) $(AM_MAKEFLAGS)-C $(abs_srcdir)/$@ run_tests

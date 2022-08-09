AM_CPPFLAGS = -I$(top_srcdir)/inc_nx
LDFLAGS = -static-libtool-libs

STATIC_LIBNXZ = ../lib/libnxz.la

EXTRATESTS = gunzip_nx_test gzip_nxfht_test gzip_nxdht_test compdecomp_th \
	bad_irq_check rand_pfault_check
NXFLAGS = #-DNXDBG  #-DNXDBG -DNX_MMAP

EXTRA_PROGRAMS = bad_irq_check \
		 bench_initend \
		 compdecomp \
		 compdecomp_th \
		 crc_perf_test_vmx \
		 crc_perf_test_zlib \
		 gunzip_nx_test \
		 gzip_nxdht_test \
		 gzip_nxfht_test \
		 gzm \
		 makedata \
		 nx_gzip \
		 rand_pfault_check \
		 zpipe

# Does nothing by default.
all:

bench: $(EXTRATESTS) zpipe gzm nx_gzip bench_initend

bad_irq_check_SOURCES = bad_irq_check.c compdecomp_th.c
bad_irq_check_LDADD   = $(STATIC_LIBNXZ) $(PTHREAD_LIBS) -ldl
bad_irq_check_LDFLAGS = -Wl,-wrap,nx_touch_pages -Wl,-wrap,nx_touch_pages_dde

bench_initend_SOURCES = bench_initend.c
bench_initend_LDADD   = -lz

compdecomp_SOURCES = compdecomp.c
compdecomp_LDADD   = $(STATIC_LIBNXZ) $(PTHREAD_LIBS) -ldl

compdecomp_th_SOURCES = compdecomp_th.c
compdecomp_th_LDADD   = $(STATIC_LIBNXZ) $(PTHREAD_LIBS) -ldl

crc_perf_test_zlib_SOURCES = crc_perf_test.c
crc_perf_test_zlib_LDADD   = -lz

crc_perf_test_vmx_SOURCES = crc_perf_test.c
crc_perf_test_vmx_LDADD   = $(STATIC_LIBNXZ) $(PTHREAD_LIBS) -ldl

gunzip_nx_test_SOURCES = gunzip_nx.c
gunzip_nx_test_LDADD   = $(STATIC_LIBNXZ) $(PTHREAD_LIBS) -ldl

gzip_nxfht_test_SOURCES = gzip_nxfht.c
gzip_nxfht_test_LDADD   = $(STATIC_LIBNXZ) $(PTHREAD_LIBS) -ldl

gzip_nxdht_test_SOURCES = gzip_nxdht.c
gzip_nxdht_test_LDADD   = $(STATIC_LIBNXZ) $(PTHREAD_LIBS) -ldl

gzm_SOURCES = gzm.c
gzm_LDADD   = $(STATIC_LIBNXZ) $(PTHREAD_LIBS) -ldl

makedata_SOURCES = makedata.c

nx_gzip_SOURCES = nx_gzip.c
nx_gzip_LDADD   = $(STATIC_LIBNXZ) -ldl

rand_pfault_check_SOURCES = rand_pfault_check.c compdecomp_th.c
rand_pfault_check_LDADD   = $(STATIC_LIBNXZ) $(PTHREAD_LIBS) -ldl
rand_pfault_check_LDFLAGS = -Wl,-wrap,nx_touch_pages \
			    -Wl,-wrap,nx_touch_pages_dde

zpipe_SOURCES = zpipe.c
zpipe_LDADD   = $(STATIC_LIBNXZ) $(PTHREAD_LIBS) -ldl

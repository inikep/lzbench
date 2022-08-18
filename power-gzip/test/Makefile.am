AM_COLOR_TESTS=always
AM_CFLAGS = -I$(top_srcdir)/lib -I$(top_srcdir)/inc_nx
LDADD = ../lib/libnxz.la $(PTHREAD_LIBS) -ldl

test_scripts = test_abi

test_exe = test_adler32 \
	   test_buf_error \
	   test_crc32 \
	   test_inflatesyncpoint \
	   test_multithread_stress \
	   test_pid_reuse

selector_tests = test_stress test_deflate test_inflate test_dict
other_tests = test_zeroinput

TESTS = $(test_exe) $(test_scripts) \
	$(selector_tests:=.auto) \
	$(selector_tests:=.sw) \
	$(selector_tests:=.nx) \
	$(selector_tests:=.mix) \
	$(selector_tests:=.mix2) \
	test_zeroinput.nx

check_PROGRAMS = $(test_exe) $(selector_tests) $(other_tests)
check_SCRIPTS = $(test_scripts)

test_adler32_SOURCES = test_adler32.c

test_buf_error_SOURCES = test_buf_error.c test_utils.c

test_crc32_SOURCES = test_crc32.c

test_deflate_SOURCES = test_deflate.c \
		       test_utils.c \
		       deflate/0.c \
		       deflate/compress.c \
		       deflate/hello.c \
		       deflate/perf.c \
		       deflate/random_buffer.c \
		       deflate/z_finish.c

test_inflate_SOURCES = test_inflate.c \
		       test_utils.c \
		       inflate/random_buffer.c

test_inflatesyncpoint_SOURCES = test_inflatesyncpoint.c test_utils.c

test_multithread_stress_SOURCES = test_multithread_stress.c

test_pid_reuse_SOURCES = test_pid_reuse.c

test_stress_SOURCES = test_stress.c test_utils.c

test_zeroinput_SOURCES = test_zeroinput.c test_utils.c

test_dict_SOURCES = test_dict.c test_utils.c

ABIDW_FLAGS = --no-corpus-path --drop-undefined-syms --drop-private-types \
	      --no-comp-dir-path --no-show-locs

%.auto:
	@$(srcdir)/gen_test.sh 0 $(*F) > $@
	@chmod +x $@

%.sw:
	@$(srcdir)/gen_test.sh 1 $(*F) > $@
	@chmod +x $@

%.nx:
	@$(srcdir)/gen_test.sh 2 $(*F) > $@
	@chmod +x $@

%.mix:
	@$(srcdir)/gen_test.sh 3 $(*F) > $@
	@chmod +x $@

%.mix2:
	@$(srcdir)/gen_test.sh 4 $(*F) > $@
	@chmod +x $@

regen-abi:
	@rm -f $(srcdir)/libnxz.abi
	abidw $(ABIDW_FLAGS) ../lib/.libs/libnxz.so > $(srcdir)/libnxz.abi
	@echo "To regenerate libz.abi, build zlib or find system's libz.so and do:"
	@echo "    abidw $(ABIDW_FLAGS) /path/to/libz.so > libz.abi"

build-TESTS: $(check_PROGRAMS) $(check_SCRIPTS)

# Set the value of source_files depending on variables passed to GNU Make.
# We cannot leave this code in Makefile.am because Automake tries to interpret
# it.

# List of pre-created files.
source_files = at \
	       calgary \
	       initrd \
	       linux \
	       Packages \
	       patch \
	       silesia-dickens \
	       silesia-mozilla \
	       silesia-mr \
	       silesia-nci \
	       silesia-ooffice \
	       silesia-osdb \
	       silesia-reymont \
	       silesia-samba \
	       silesia-sao \
	       silesia-webster \
	       silesia-xml \
	       silesia-x-ray \
	       vmlinux

# Do not enable large files by default.
# They consume a lot of disk space and increase the execution time.
ifdef LARGEFILES
source_files += db2
endif

# List of all files.  Including the auto-generated files.
files = $(source_files) \
	empty \
	random4k random13M \
	sparse10M sparse1000M \
	zero4k zero13M
uncomp_files = $(addsuffix .uncompressed,$(files))

test_types = deflate gzip
levels = 1 2 3 4 5 6 7 8 9
compress_tests = $(foreach t,$(test_types),\
		   $(foreach f,$(files),$(f).6.compress.$(t).test))

decompress_tests = $(foreach t,$(test_types),\
		     $(foreach l,$(levels),\
		       $(foreach f,$(files),$(f).$(l).decompress.$(t).test)))

compdecomp_tests = $(foreach t,$(test_types),\
		     $(foreach f,$(files),$(f).6.compdecomp.$(t).test))

TESTS += $(compress_tests) $(decompress_tests) $(compdecomp_tests)
check_SCRIPTS += $(compress_tests) $(decompress_tests) $(compdecomp_tests)
check_FILES += $(addsuffix .checksum,$(files))

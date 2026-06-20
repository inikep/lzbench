# Copyright (c) Meta Platforms, Inc. and affiliates.


ROOT = ../../../..
LDFLAGS += -I$(ROOT)/include

default: test

check_lossless: ../encode_ycocg_kernel.c ../decode_ycocg_kernel.c

test_ycocg_codec: LDLIBS +=-lzl
test_ycocg_codec: LDFLAGS += -L$(ROOT)
test_ycocg_codec: ../encode_ycocg_binding.c ../encode_ycocg_kernel.c ../decode_ycocg_binding.c ../decode_ycocg_kernel.c $(ROOT)/libzl.so

$(ROOT)/libzl.so:
	$(MAKE) -C $(ROOT) lib

.PHONY: test
test: test_ycocg_codec
	LD_LIBRARY_PATH=$(ROOT):$(LD_LIBRARY_PATH) DYLD_LIBRARY_PATH=$(ROOT):$(DYLD_LIBRARY_PATH) ./test_ycocg_codec

clean:
	$(RM) check_lossless test_ycocg_codec

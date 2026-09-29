# tornado
CODECS += TORNADO
TORNADO_OBJS := lz+entropy/tornado/tor_test.o

# tornado dereferences unaligned 16/32/64-bit values directly (see value32() in
# lz+entropy/tornado/Common.h). On RISC-V such an access is either performed by the
# hardware, or trapped and emulated by the kernel (orders of magnitude slower),
# or not supported at all (SIGBUS), so it is only worth benchmarking where the
# hardware handles it at full speed. lz+entropy/tornado/check_riscv_fast_unaligned.c asks
# the kernel through the hwprobe syscall and exits 0 only on a clear "fast";
# anything else, including a probe that cannot be built or run, leaves tornado
# disabled.
ifneq (,$(filter riscv%,$(HOST_ARCH)))
    ifeq "$(DONT_BUILD_TORNADO)" ""
        ifeq ($(detected_OS),Linux)
            RISCV_FAST_UNALIGNED := $(shell t=$$(mktemp "$${TMPDIR:-/tmp}/lzbench_hwprobe.XXXXXX" 2>/dev/null) && \
                $(CC) $(SOURCE_PATH)lz+entropy/tornado/check_riscv_fast_unaligned.c -o "$$t" 2>/dev/null && "$$t"; \
                r=$$?; rm -f "$$t"; [ "$$r" = 0 ] && echo 1 || echo 0)
        endif

        ifeq ($(RISCV_FAST_UNALIGNED),1)
            $(info RISC-V: misaligned scalar access is fast, benchmarking tornado)
        else
            $(info RISC-V: misaligned scalar access is slow, emulated or unsupported, disabling tornado)
            DONT_BUILD_TORNADO := 1
        endif
    endif
endif

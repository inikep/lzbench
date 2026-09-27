# zpaq
CODECS += ZPAQ
ZPAQ_OBJS  := misc/zpaq/libzpaq.o
ZPAQ_FLAGS := -Imisc/zpaq

# zpaq's JIT emits x86 machine code and crashes on other CPUs (SIGSEGV on 32-bit
# ARM, SIGILL on aarch64). On non-x86 targets build it with -DNOJIT so it uses
# its portable (slower) interpreter instead.
ifeq (,$(filter x86_64% amd64% i%86,$(TARGET_ARCH)))
    ZPAQ_FLAGS += -DNOJIT
endif

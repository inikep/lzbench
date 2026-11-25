# misa77: needs C++20 and a 64-bit little-endian target
CODECS += MISA77
MISA77_UPSTREAM := git-tags https://github.com/welcome-to-the-sunny-side/misa77

# The probe is skipped when misa77 is already disabled with DONT_BUILD_MISA77=1.
ifneq ($(DONT_BUILD_MISA77),1)
    MISA77_OK := $(shell printf 'int main(){static_assert(__BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__);static_assert(sizeof(void*)==8);return 0;}' | $(CXX) $(CODE_FLAGS) -std=c++20 -fsyntax-only -x c++ - 2>/dev/null && echo ok)
    ifneq ($(MISA77_OK),ok)
        DONT_BUILD_MISA77 ?= 1
        ifeq "$(DONT_BUILD_MISA77)" "1"
            $(info C++20 and a 64-bit little-endian target required – skipping misa77 build)
        endif
    endif
endif

MISA77_OBJS := $(addprefix lz+entropy/misa77/src/, compress.o decompress.o isa/target_portable.o)
# 64-bit x86 only (the probe above already rejected 32-bit and big-endian targets):
ifneq (,$(filter x86_64% amd64%,$(TARGET_ARCH)))
    MISA77_OBJS += lz+entropy/misa77/src/isa/target_sse2.o lz+entropy/misa77/src/isa/target_avx2.o
endif
# 64-bit ARM only:
ifneq (,$(filter arm64% aarch64%,$(TARGET_ARCH)))
    MISA77_OBJS += lz+entropy/misa77/src/isa/target_neon.o
endif
MISA77_FLAGS = -std=c++20 -I$(SRC)lz+entropy/misa77/include -I$(SRC)lz+entropy/misa77/src $(MISA77_ASM_DEFINE) $(MISA77_ISA_FLAGS)

# target_avx2.cpp is the only TU needing extra ISA flags: SSE2 and NEON are baseline on
# 64-bit x86 and ARM, and the probe above already disabled misa77 on 32-bit targets. A
# 32-bit x86 port would have to add -msse2 back for target_sse2.cpp, as i686 has neither
# __SSE__ nor __SSE2__ by default.
lz+entropy/misa77/%_avx2.o: MISA77_ISA_FLAGS := -mavx2

# Assembly decode kernels (src/decompressor_zoo/*_kernels/*.S), as misa77's CMakeLists.txt
# builds them: x86-64 ELF and AArch64 ELF or Mach-O targets. Elsewhere (Windows, x86-64
# macOS, other CPUs) and with "make MISA77_ASM=0", misa77 uses its C++ decoders.
MISA77_ASM ?= 1
MISA77_TARGET_OS := $(subst -, ,$(TARGET_MACHINE))
ifeq ($(MISA77_ASM)$(filter mingw% cygwin% windows%,$(MISA77_TARGET_OS)),1)
    ifneq (,$(filter x86_64% amd64%,$(TARGET_ARCH)))
        ifeq (,$(filter darwin%,$(MISA77_TARGET_OS)))
            MISA77_ASM_OBJS := $(foreach k,light heavy entropy, \
                lz+entropy/misa77/src/decompressor_zoo/$(k)_safe_kernels/x86_64_avx2.o \
                lz+entropy/misa77/src/decompressor_zoo/$(k)_safe_kernels/x86_64_sse2.o)
        endif
    endif
    ifneq (,$(filter arm64% aarch64%,$(TARGET_ARCH)))
        MISA77_ASM_OBJS := $(foreach k,light heavy entropy, \
            lz+entropy/misa77/src/decompressor_zoo/$(k)_safe_kernels/aarch64_neon.o)
    endif
endif
ifneq (,$(MISA77_ASM_OBJS))
    MISA77_OBJS += $(MISA77_ASM_OBJS)
    MISA77_ASM_DEFINE := -DMISA77_ASM_KERNELS=1
endif

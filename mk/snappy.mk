# snappy
CODECS += SNAPPY
SNAPPY_OBJS := $(addprefix lz/snappy/, snappy-sinksource.o snappy-stubs-internal.o snappy.o)

# __builtin_ctz is efficient on CPUs with bit-manipulation support (e.g., RISC-V
# Zbb, x86 BMI1/TZCNT, ARM).
HAVE_BUILTIN_CTZ := $(shell echo 'int main(void){return __builtin_ctz(8);}' \
    | $(CC) $(CFLAGS) -x c -o /dev/null - 2>/dev/null && echo 1 || echo 0)
ifeq ($(HAVE_BUILTIN_CTZ),1)
    SNAPPY_FLAGS += -DHAVE_BUILTIN_CTZ
endif

# Detect RISC-V Vector (RVV) support in the compiler and header files. Snappy has
# an RVV-accelerated path whose source uses two different spellings:
#   1. With __riscv_ prefix (new spec, e.g. __riscv_vsetvl_e8m1)
#   2. Without prefix       (old spec, e.g. vsetvl_e8m1)
# A one-line C file is generated on the fly with printf. $(pound) holds a literal
# '#', inserted before the shell sees the command.
pound := \#
rvv_prefix = __riscv_
SNAPPY_RVV = printf '%s\n' \
        '$(pound)include <riscv_vector.h>' \
        '$(pound)include <stdint.h>' \
        '$(pound)include <stddef.h>' \
        'int main() {' \
        '    uint8_t val = 3;' \
        '    size_t vl = $(rvv_prefix)vsetvl_e8m1(8);' \
        '    vuint8m1_t v = $(rvv_prefix)vmv_v_x_u8m1(val, vl);' \
        '    (void)v;' \
        '    return 0;' \
        '}' \
    | $(CC) $(CFLAGS) -x c -o /dev/null - 2>/dev/null \
    && echo 1 || echo 0
SNAPPY_RVV_1 := $(shell $(SNAPPY_RVV))
rvv_prefix =
SNAPPY_RVV_0_7 := $(shell $(SNAPPY_RVV))

ifeq ($(SNAPPY_RVV_1),1)
    SNAPPY_FLAGS += -DSNAPPY_RVV_1
endif
ifeq ($(SNAPPY_RVV_0_7),1)
    SNAPPY_FLAGS += -DSNAPPY_RVV_0_7
endif

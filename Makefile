# Multi-threaded build:
#	make -j$(nproc)
#
# To order static or dynamic linking set `BUILD_STATIC` to 1, or 0 respectively:
#	make BUILD_STATIC=1
#
# For debug:
#	make BUILD_TYPE=debug
#
# For 32-bit compilation:
#	make BUILD_ARCH=32-bit
#
# For non-default compiler:
#	make CC=gcc-14 CXX=g++-14
#
# For an optimized but non-portable build, use:
#	make MOREFLAGS="-march=native"
# or
#	make USER_CFLAGS="-march=native" USER_CXXFLAGS="-march=native"
#
# To leave a codec out (the names are the ones used in mk/*.mk):
#	make DONT_BUILD_LZHAM=1
#
# CUDA codecs (nvcomp, bsc_cuda, aceapex_cuda, gpucompact):
#	make ENABLE_CUDA=1 [CUDA_BASE=/usr/local/cuda]
#
# Every codec is described by its own file in mk/, see "Codecs" below.

# Out-of-tree builds are supported: "make -f /path/to/lzbench/Makefile" builds in
# the current directory. vpath finds the sources for the rules, and paths into
# the source tree that are used in commands (-I flags, wildcards, cd) are
# prefixed with $(SRC), which is empty for an in-tree build.
SOURCE_PATH := $(dir $(lastword $(MAKEFILE_LIST)))
SRC := $(filter-out ./,$(SOURCE_PATH))
vpath
vpath %.c   $(SOURCE_PATH)
vpath %.cc  $(SOURCE_PATH)
vpath %.cpp $(SOURCE_PATH)
vpath %.S   $(SOURCE_PATH)
vpath %.cu  $(SOURCE_PATH)
vpath %.h   $(SOURCE_PATH)
vpath %.zig $(SOURCE_PATH)


#------------------------------------------------------------------------------
# Toolchain and platform
#------------------------------------------------------------------------------

ifeq ($(BUILD_ARCH),32-bit)
    CODE_FLAGS += -m32
    LDFLAGS += -m32
endif

COMPILER = $(shell $(CC) -v 2>&1 | grep -q "clang version" && echo clang || echo gcc)
GCC_VERSION = $(shell echo | $(CC) -dM -E - | grep __VERSION__  | sed -e 's:\#define __VERSION__ "\([0-9.]*\).*:\1:' -e 's:\.\([0-9][0-9]\):\1:g' -e 's:\.\([0-9]\):0\1:g')
CLANG_VERSION = $(shell $(CC) -v 2>&1 | grep "clang version" | sed -e 's:.*version \([0-9.]*\).*:\1:' -e 's:\.\([0-9][0-9]\):\1:g' -e 's:\.\([0-9]\):0\1:g')

HOST_ARCH      := $(shell uname -m)
TARGET_MACHINE := $(shell $(CXX) -dumpmachine)
TARGET_ARCH    := $(firstword $(subst -, ,$(TARGET_MACHINE)))
# 1 for a Windows target (e.g. x86_64-w64-mingw32), also when cross-compiling
TARGET_WINDOWS := $(if $(filter mingw% cygwin% windows%,$(subst -, ,$(TARGET_MACHINE))),1)
HOST_WINDOWS   := $(if $(filter Windows%,$(OS)),1)
# 1 when the target CPU or OS is not the build machine's. A MinGW build from
# Linux targets x86_64 like the host, so the CPU alone does not tell.
ifneq ($(HOST_ARCH),$(TARGET_ARCH))
    CROSS_BUILD := 1
else ifneq ($(HOST_WINDOWS),$(TARGET_WINDOWS))
    CROSS_BUILD := 1
endif

# detect thread model for gcc or clang
THREAD_MODEL := $(shell $(CXX) -v 2>&1 | grep '^Thread model:' | awk '{print $$3}')
$(info Detected thread model: $(THREAD_MODEL))

ifneq (,$(filter Windows%,$(OS)))
    THREAD_MODEL := $(or $(THREAD_MODEL),win32)
    BUILD_STATIC ?= 1
    ifeq ($(BUILD_STATIC),1)
        LDFLAGS += -lshell32 -lole32 -loleaut32 -static
    endif
else
    THREAD_MODEL := $(or $(THREAD_MODEL),posix)
    detected_OS := $(shell uname -s)
    UNAME_P     := $(shell uname -p)

    ifneq (,$(filter riscv%,$(HOST_ARCH)))
        MOREFLAGS += -mno-strict-align
    endif

    # some compressors use dlopen(), which requires linking with -ldl on glibc
    # 2.33 and older, and other libc libraries. Use -ldl only when dlopen()
    # links with it but not without it: a MinGW cross build from Linux has no
    # dlopen() at all, and no libdl either.
    # GNU Make 3.8.x fails to parse \# inside the $(shell ...) function.
    LIBDL_TEST_SRC := \#include <dlfcn.h>\nint main(){dlopen(0,0);return 0;}\n
    LIBDL := $(shell printf '${LIBDL_TEST_SRC}' | $(CXX) -x c - -o /dev/null 2>/dev/null || \
               { printf '${LIBDL_TEST_SRC}' | $(CXX) -x c - -ldl -o /dev/null 2>/dev/null && echo "-ldl"; })

    ifneq ($(THREAD_MODEL), win32)
        DEFINES += -Dunix
    endif

    ifeq ($(BUILD_STATIC),1)
        LDFLAGS	+= -static -static-libstdc++
    endif
endif


#------------------------------------------------------------------------------
# Compiler flags
#------------------------------------------------------------------------------

DEFINES     += -I$(or $(SRC),.)
CODE_FLAGS  += -Wno-unknown-pragmas -Wno-sign-compare -Wno-conversion

# don't use "-ffast-math" for clang < 10.0
ifeq (1, $(shell [ "$(COMPILER)" = "clang" ] && expr $(CLANG_VERSION) \< 100000 ))
    OPT_FLAGS   ?= -fomit-frame-pointer -fstrict-aliasing
else
    OPT_FLAGS   ?= -fomit-frame-pointer -fstrict-aliasing -ffast-math
endif

ifeq ($(BUILD_TYPE),debug)
    OPT_FLAGS_O2 = $(OPT_FLAGS) -O0 -g
    OPT_FLAGS_O3 = $(OPT_FLAGS) -O0 -g
else
    OPT_FLAGS_O2 = $(OPT_FLAGS) -O2 -DNDEBUG
    OPT_FLAGS_O3 = $(OPT_FLAGS) -O3 -DNDEBUG
endif

# Everything is built with -O3, except the objects of codecs that set
# <NAME>_OPT := O2 in their mk file (OPT_LEVEL is set per object below).
OPT_LEVEL = O3

# Automatic header dependencies. -MMD writes "<object>.d" next to each object,
# listing every file the translation unit included; -MP adds a phony target for
# each of them so that deleting or renaming a header does not break the next
# build. Without this, `make` only knows about the one source file named in the
# rule, so editing a header -- or a .cpp that another .cpp #includes, as
# lz/aceapex, lz/lzo, lz/ucl, lz/tamp and misc/7-zip do -- silently relinks a
# stale object. The flags are a gcc/clang/mingw extension, so probe for them and
# fall back to the old behaviour on a compiler that does not understand them.
DEPFLAGS := $(shell printf 'int main(){return 0;}' | $(CXX) -x c++ - -MMD -MP -MF /dev/null -c -o /dev/null 2>/dev/null && printf -- '-MMD -MP')

CXXFLAGS  = $(CODE_FLAGS) $(OPT_FLAGS_$(OPT_LEVEL)) $(DEFINES) $(MOREFLAGS) $(USER_CXXFLAGS) $(DEPFLAGS)
CFLAGS    = $(CODE_FLAGS) $(OPT_FLAGS_$(OPT_LEVEL)) $(DEFINES) $(MOREFLAGS) $(USER_CFLAGS) $(DEPFLAGS)
# nvcc does not reliably accept -MMD/-MP, so CUDA rules use the host flags without them
CUDA_HOST_CXXFLAGS = $(filter-out $(DEPFLAGS),$(CXXFLAGS))
LDFLAGS  += -pthread $(MOREFLAGS) $(USER_LDFLAGS)
ifeq ($(detected_OS), Darwin)
    CXXFLAGS += -std=c++14
endif


#------------------------------------------------------------------------------
# Threading
#------------------------------------------------------------------------------

ifeq "$(DISABLE_THREADING)" "1"
    DEFINES += -DDISABLE_THREADING
else
    OMP_TEST_CODE = \#include <omp.h>\nint main(){return 0;}\n
    HAVE_OPENMP := $(shell printf '$(OMP_TEST_CODE)' | $(CXX) -x c++ - -fopenmp -o /dev/null 2>/dev/null && echo 1 || echo 0)

    ifeq ($(HAVE_OPENMP),1)
        $(info OpenMP found: compiling bsc with OMP multithreading)
        OPENMP_CXXFLAGS = -fopenmp
        LDFLAGS += -fopenmp
    else
        $(info OpenMP not found: compiling bsc without multithreading)
    endif
endif


#------------------------------------------------------------------------------
# CUDA toolkit (make ENABLE_CUDA=1)
#------------------------------------------------------------------------------

ifeq "$(ENABLE_CUDA)" "1"
    CUDA_BASE ?= /usr/local/cuda
    LIBCUDART = $(wildcard $(CUDA_BASE)/lib64/libcudart.so)
    CUDA_H    = $(wildcard $(CUDA_BASE)/include/cuda.h)

    ifeq "$(and $(LIBCUDART),$(CUDA_H))" ""
        $(info CUDA Toolkit not found at $(CUDA_BASE), CUDA support will be disabled.)
        $(info Run "make CUDA_BASE=..." to use a different path.)
        CUDA_BASE =
        LIBCUDART =
        CUDA_H =
    else
        HAVE_CUDA := 1
        DEFINES += -DBENCH_HAS_CUDA -I$(CUDA_BASE)/include
        LDFLAGS += -L$(CUDA_BASE)/lib64 -lcudart -Wl,-rpath=$(CUDA_BASE)/lib64
        CUDA_COMPILER = nvcc
        CUDA_CC = $(CUDA_BASE)/bin/nvcc --compiler-bindir $(CXX)
        # ("?define" rather than "#define": GNU make 3.81 takes the '#' for a comment)
        CUDA_VERSION := $(shell awk '$$1 ~ /^.define$$/ && $$2 == "CUDA_VERSION" { print $$3; exit;}' $(CUDA_H))
        ifeq "$(CUDA_VERSION)" ""
            $(error Could not determine CUDA_VERSION from $(CUDA_H))
        endif
        CUDA_ARCH := $(shell \
          if [ $(CUDA_VERSION) -ge 13000 ]; then \
              echo 75 80 86 89 90 100 120; \
          elif [ $(CUDA_VERSION) -ge 12080 ]; then \
              echo 50 52 60 61 70 75 80 86 89 90 100 120; \
          elif [ $(CUDA_VERSION) -ge 11080 ]; then \
              echo 50 52 60 61 70 75 80 86 89 90; \
          elif [ $(CUDA_VERSION) -ge 11010 ]; then \
              echo 50 52 60 61 70 75 80 86; \
          elif [ $(CUDA_VERSION) -ge 11000 ]; then \
              echo 50 52 60 61 70 75 80; \
          else \
              echo 50 52 60 61 70 75; fi)
        CUDA_CXXSTD := $(shell \
          if [ $(CUDA_VERSION) -ge 13000 ]; then \
              echo c++17; \
          else \
              echo c++14; \
          fi)
        CUDA_CXXFLAGS = -x cu -std=$(CUDA_CXXSTD) -O3 $(foreach ARCH, $(CUDA_ARCH), --generate-code=arch=compute_$(ARCH),code=[compute_$(ARCH),sm_$(ARCH)]) --expt-extended-lambda -forward-unknown-to-host-compiler -Wno-deprecated-gpu-targets
    endif
endif


#------------------------------------------------------------------------------
# Rust toolchain (for the Rust codecs, see "Rust codecs" below)
#------------------------------------------------------------------------------

HAVE_CARGO := $(shell command -v cargo >/dev/null 2>&1 && echo 1 || echo 0)
ifneq ($(HAVE_CARGO),1)
    $(info Cargo not found – skipping Rust codecs (density, mbrotli, pulsar))
else
    CARGO_VERSION := $(shell cargo --version | awk '{print $$2}')
    # Only build Rust codecs if native build, not 32-bit, not Windows
    ifeq ($(CROSS_BUILD),1)                # Skip cross-compilation
    else ifeq ($(BUILD_ARCH),32-bit)       # Skip user requested 32-bit compilation
    else ifeq ($(TARGET_WINDOWS),1)        # Skip Windows builds due to undefined reference errors on linking even when adding required native static libs to linking dependencies
    else
        HAVE_RUST := 1
    endif
endif

# $(call cargo_at_least,1.85.0) is 1 when cargo is at least that version
cargo_at_least = $(shell printf "%s\n$(1)\n" "$(CARGO_VERSION)" | sort -V | head -n1 | grep -qx $(1) && echo 1)


#------------------------------------------------------------------------------
# Codecs
#------------------------------------------------------------------------------
#
# mk/<name>.mk describes one codec NAME (e.g. mk/zlib-ng.mk is ZLIB_NG):
#
#   CODECS     += NAME
#   NAME_OBJS  := objects to compile and link into lzbench
#   NAME_FLAGS := extra compiler flags for NAME_OBJS (optional)
#   NAME_OPT   := O2 to build NAME_OBJS with -O2 instead of -O3 (optional)
#
# "make DONT_BUILD_NAME=1" leaves NAME_OBJS out and defines BENCH_REMOVE_NAME,
# which removes the codec from bench/*.cpp. A mk file may also:
#   - disable its codec on some platforms with "DONT_BUILD_NAME ?= 1",
#   - add to DEFINES, LDFLAGS, LINK_DEPS, CLEAN_FILES or CLEAN_DIRS,
#   - for a Rust codec, add its cargo feature to RUST_FEATURES and its sources
#     to RUST_DEPS (see "Rust codecs" below),
#   - add rules for objects the generic %.o rules below cannot build,
#   - add objects that DONT_BUILD_NAME does not switch off as a separate
#     group: "OBJ_GROUPS += GROUP" with GROUP_OBJS and GROUP_FLAGS.
#
# To add a codec, add mk/<name>.mk and the codec's entry in bench/.

MKDIR = mkdir -p

# the mk files define rules too, so name the default target explicitly
.DEFAULT_GOAL := lzbench

include $(sort $(wildcard $(SOURCE_PATH)mk/*.mk))

CODECS_OFF := $(foreach c,$(CODECS),$(if $(filter 1,$(DONT_BUILD_$(c))),$(c)))
CODECS_ON  := $(filter-out $(CODECS_OFF),$(CODECS))
DEFINES    += $(addprefix -DBENCH_REMOVE_,$(CODECS_OFF))

# Give each group's objects its compiler flags and optimization level.
$(foreach g,$(CODECS_ON) $(OBJ_GROUPS),$(if $($(g)_OBJS), \
    $(eval $$($(g)_OBJS): CODEC_FLAGS = $$($(g)_FLAGS)) \
    $(if $($(g)_OPT),$(eval $$($(g)_OBJS): OPT_LEVEL = $($(g)_OPT)))))

CODEC_OBJS := $(foreach g,$(CODECS_ON) $(OBJ_GROUPS),$($(g)_OBJS))


#------------------------------------------------------------------------------
# Rust codecs
#------------------------------------------------------------------------------
#
# The Rust codecs (density, mbrotli, pulsar) are built into one library,
# liblzbench_rust, from misc/rust-codecs, which has a cargo feature per codec:
# two Rust staticlibs each carry their own copy of std and cannot be linked into
# the same binary.

CLEAN_DIRS += $(SRC)misc/rust-codecs/target/

# CPU the Rust codecs are compiled for. "native" optimizes them for the build
# machine, and the binary may then die with SIGILL on other CPUs (e.g. AVX-512
# code on a CPU without it); for binaries that are run elsewhere, such as
# releases, use e.g. RUST_TARGET_CPU=x86-64, the baseline the C codecs are
# built for. Empty leaves the choice to rustc.
RUST_TARGET_CPU ?= native

ifneq ($(strip $(RUST_FEATURES)),)
    RUST_SRC_DIR := $(SRC)misc/rust-codecs/
    ifeq ($(BUILD_STATIC),1)
        RUST_BUILD_TYPE := staticlib
        RUST_LIB := $(RUST_SRC_DIR)target/release/liblzbench_rust.a
    else
        RUST_BUILD_TYPE := cdylib
        RUST_LIB := $(RUST_SRC_DIR)target/release/liblzbench_rust$(if $(filter Darwin,$(detected_OS)),.dylib,.so)
    endif

    # RUST_LIB is rebuilt when a source of an enabled codec changes, and when the
    # crate type, target CPU or set of codecs does: RUST_STAMP records those and is
    # rewritten, while the Makefile is read, whenever they differ.
    RUST_STAMP  := $(RUST_SRC_DIR)target/lzbench-config
    RUST_CONFIG := $(RUST_BUILD_TYPE) cpu=$(RUST_TARGET_CPU) $(strip $(RUST_FEATURES))
    ifneq ($(shell cat $(RUST_STAMP) 2>/dev/null),$(RUST_CONFIG))
        $(shell mkdir -p $(RUST_SRC_DIR)target && echo '$(RUST_CONFIG)' > $(RUST_STAMP))
    endif
    RUST_DEPS += $(RUST_STAMP) $(addprefix $(RUST_SRC_DIR),Cargo.toml Cargo.lock lib.rs .cargo/config.toml)

    # linked with -l, but lzbench is relinked when the library changes
    LINK_DEPS += $(RUST_LIB)
    LDFLAGS += -Wl,-rpath,$(RUST_SRC_DIR)target/release -L$(RUST_SRC_DIR)target/release -llzbench_rust
endif

# cargo leaves an up-to-date library alone, so touch it: otherwise it would stay
# older than a source edited without effect on the output, and cargo would run on
# every make. --offline: the dependencies are vendored in misc/rust-codecs/vendor.
ifneq ($(RUST_LIB),)
$(RUST_LIB): $(RUST_DEPS)
	@echo "Building Rust codecs ($(strip $(RUST_FEATURES)))..."
	cd $(RUST_SRC_DIR) && \
	RUSTFLAGS="$(if $(RUST_TARGET_CPU),-C target-cpu=$(RUST_TARGET_CPU) )-C linker=$(lastword $(CXX))" \
	cargo rustc --locked --offline --features "$(strip $(RUST_FEATURES))" --crate-type=$(RUST_BUILD_TYPE) --release -- --print=native-static-libs
	touch $@
endif


#------------------------------------------------------------------------------
# lzbench itself
#------------------------------------------------------------------------------

BENCH_OBJS := bench/lz_codecs.o bench/buggy_codecs.o bench/symmetric_codecs.o bench/lzbench.o bench/misc_codecs.o
ifneq "$(DISABLE_THREADING)" "1"
    BENCH_OBJS += bench/threadpool.o
endif

bench/lz_codecs.o:        CODEC_FLAGS = $(addprefix -I$(SRC),lz lz/brotli/include lz/openzl/include lz/zxc/src/lib/vendors lz/misa77/include)
bench/buggy_codecs.o:     CODEC_FLAGS = -I$(SRC)lz/libcsc
bench/symmetric_codecs.o: CODEC_FLAGS = $(OPENMP_CXXFLAGS)
bench/lzbench.o:          CODEC_FLAGS = $(OPENMP_CXXFLAGS)

bench/lzbench.o: bench/lzbench.cpp bench/lzbench.h bench/threadpool.h bench/codecs.h

# bench/*.cpp compile each codec in or out with BENCH_REMOVE_* (and the CUDA
# codecs with BENCH_HAS_*), so they are rebuilt when that set changes, e.g. with
# DONT_BUILD_<codec>=1: BENCH_STAMP records it and is rewritten, while the
# Makefile is read, whenever it differs.
BENCH_STAMP  := bench/codecs.stamp
BENCH_CONFIG := codecs: $(sort $(filter -DBENCH_%,$(subst ",,$(DEFINES))))
ifneq ($(shell cat $(BENCH_STAMP) 2>/dev/null),$(BENCH_CONFIG))
    $(shell mkdir -p $(dir $(BENCH_STAMP)) && echo '$(BENCH_CONFIG)' > $(BENCH_STAMP))
endif
$(BENCH_OBJS): $(BENCH_STAMP)
CLEAN_FILES += $(BENCH_STAMP)

# static libraries (skim) go last, after the objects that use them
LZBENCH_OBJS = $(filter-out %.a,$(CODEC_OBJS)) $(BENCH_OBJS) $(filter %.a,$(CODEC_OBJS))

# LINK_DEPS: libraries that are linked with -l (the Rust codecs), but still have
# to be built first and trigger a relink when they change
lzbench: $(LZBENCH_OBJS) $(LINK_DEPS)
	$(CXX) $(filter-out $(LINK_DEPS),$^) -o $@ $(LDFLAGS) $(LDFLAGS_LIBDL)
	@echo Linked GCC_VERSION=$(GCC_VERSION) CLANG_VERSION=$(CLANG_VERSION) COMPILER=$(COMPILER)


#------------------------------------------------------------------------------
# Rules
#------------------------------------------------------------------------------

COMPILE_C   = $(CC) $(CFLAGS) $(CODEC_FLAGS) $< -c -o $@
COMPILE_CXX = $(CXX) $(CXXFLAGS) $(CODEC_FLAGS) $< -c -o $@
COMPILE_CU  = $(CUDA_CC) $(CUDA_CXXFLAGS) $(CUDA_HOST_CXXFLAGS) $(CODEC_FLAGS) -c $< -o $@

# disable the implicit rule for making a binary out of a single object file
%: %.o

%.o: %.c
	@$(MKDIR) $(dir $@)
	$(COMPILE_C)

%.o: %.cc
	@$(MKDIR) $(dir $@)
	$(COMPILE_CXX)

%.o: %.cpp
	@$(MKDIR) $(dir $@)
	$(COMPILE_CXX)

%.o: %.S
	@$(MKDIR) $(dir $@)
	$(COMPILE_C)

# CUDA objects are named after their source (foo.cu.o, and foo.cpp.o in nvcomp)
%.cu.o: %.cu
	@$(MKDIR) $(dir $@)
	$(COMPILE_CU)

%.cpp.o: %.cpp
	@$(MKDIR) $(dir $@)
	$(COMPILE_CXX)

clean:
	rm -rf lzbench lzbench.exe
	find . -type f -name "*.o" -exec rm -f {} +
	find . -type f -name "*.d" -exec rm -f {} +
	rm -rf $(CLEAN_DIRS)
	rm -f $(CLEAN_FILES)

# Pull in the header dependencies generated by $(DEPFLAGS). Missing .d files
# (a fresh tree, or a CUDA object built by nvcc) are silently ignored.
DEPFILES := $(patsubst %.o,%.d,$(filter %.o,$(LZBENCH_OBJS)))
-include $(DEPFILES)

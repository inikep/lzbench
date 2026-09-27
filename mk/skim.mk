# skim: a Zig library, needs the zig compiler
CODECS += SKIM

# zig build-lib is not given a -target, so libskim.a is built for the build
# machine: it cannot be linked into a cross, -m32 or Windows build.
HAVE_ZIG := $(shell command -v zig >/dev/null 2>&1 && echo 1 || echo 0)
ifneq ($(HAVE_ZIG),1)
    $(info Zig not found – skipping skim build)
    DONT_BUILD_SKIM ?= 1
else ifeq ($(CROSS_BUILD),1)
    DONT_BUILD_SKIM ?= 1
else ifeq ($(BUILD_ARCH),32-bit)
    DONT_BUILD_SKIM ?= 1
else ifeq ($(TARGET_WINDOWS),1)
    DONT_BUILD_SKIM ?= 1
endif

SKIM_OBJS := misc/skim/libskim.a
CLEAN_FILES += misc/skim/libskim.a

misc/skim/libskim.a: misc/skim/src/root.zig
	@echo "Building Skim (Zig)..."
	@$(MKDIR) $(dir $@)
	cd $(SRC)misc/skim && zig build-lib -O ReleaseFast -femit-bin=$(abspath $@) src/root.zig -lc

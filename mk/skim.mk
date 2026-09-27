# skim: a Zig library, needs the zig compiler
CODECS += SKIM

HAVE_ZIG := $(shell command -v zig >/dev/null 2>&1 && echo 1 || echo 0)
ifneq ($(HAVE_ZIG),1)
    $(info Zig not found – skipping skim build)
    DONT_BUILD_SKIM ?= 1
endif

SKIM_OBJS := misc/skim/libskim.a
CLEAN_FILES += misc/skim/libskim.a

misc/skim/libskim.a: misc/skim/src/root.zig
	@echo "Building Skim (Zig)..."
	cd misc/skim && zig build-lib -O ReleaseFast -femit-bin=libskim.a src/root.zig -lc

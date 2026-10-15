# Go 1.22+ builds go-brrr as a C archive.
CODECS += GOBRRR

GOBRRR_OBJS  := lz+entropy/gobrrr/libgobrrr.a
GOBRRR_STAMP := lz+entropy/gobrrr/libgobrrr.config
CLEAN_FILES  += lz+entropy/gobrrr/libgobrrr.a lz+entropy/gobrrr/libgobrrr.h $(GOBRRR_STAMP)

# The archive targets the host. Skip cross, 32-bit and Windows builds.
ifneq ($(DONT_BUILD_GOBRRR),1)
    ifeq ($(CROSS_BUILD),1)
        DONT_BUILD_GOBRRR ?= 1
    else ifeq ($(BUILD_ARCH),32-bit)
        DONT_BUILD_GOBRRR ?= 1
    else ifeq ($(TARGET_WINDOWS),1)
        DONT_BUILD_GOBRRR ?= 1
    else ifneq ($(shell command -v go >/dev/null 2>&1 && echo 1),1)
        $(info Go not found – skipping go-brrr build)
        DONT_BUILD_GOBRRR ?= 1
    else ifneq ($(shell printf '%s\n1.22\n' "$$(go env GOVERSION | sed 's/^go//')" | sort -V | head -n1),1.22)
        $(info Go is older than 1.22 – skipping go-brrr build)
        DONT_BUILD_GOBRRR ?= 1
    endif
endif

# The stamp tracks the C compiler and Go settings across make runs.
ifneq ($(DONT_BUILD_GOBRRR),1)
    GOBRRR_CONFIG := cc=$(CC) $(shell go env GOVERSION GOARCH GOAMD64 GOARM64 GOFLAGS CGO_CFLAGS CGO_LDFLAGS)
    ifneq ($(shell cat $(GOBRRR_STAMP) 2>/dev/null),$(strip $(GOBRRR_CONFIG)))
        $(shell mkdir -p $(dir $(GOBRRR_STAMP)) && echo '$(strip $(GOBRRR_CONFIG))' > $(GOBRRR_STAMP))
    endif
endif

# Build offline with the local Go toolchain. GOAMD64 defaults to x86-64-v1.
lz+entropy/gobrrr/libgobrrr.a: $(GOBRRR_STAMP) $(shell find $(SRC)lz+entropy/gobrrr -name '*.go' -o -name '*.s' -o -name '*.bin' -o -name go.mod)
	@echo "Building go-brrr (Go)..."
	@$(MKDIR) $(dir $@)
	cd $(SRC)lz+entropy/gobrrr && CC="$(CC)" CGO_ENABLED=1 GOTOOLCHAIN=local GOPROXY=off GOWORK=off \
	go build -buildmode=c-archive -trimpath -o $(abspath $@) ./ffi

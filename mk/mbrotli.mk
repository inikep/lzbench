# mbrotli: a Rust crate, built into the Rust codecs' library (see "Rust codecs"
# in the Makefile) and called through its C ABI (mbrotli-ffi)
CODECS += MBROTLI

ifneq ($(DONT_BUILD_MBROTLI),1)
    ifneq ($(HAVE_RUST),1)
        DONT_BUILD_MBROTLI := 1
    else ifneq ($(call cargo_at_least,1.89.0),1)
        $(info Cargo $(CARGO_VERSION) is older than 1.89 – skipping mbrotli build)
        DONT_BUILD_MBROTLI := 1
    else
        RUST_FEATURES += mbrotli
        RUST_DEPS += $(shell find $(addprefix $(SRC)lz/mbrotli/,Cargo.toml src mbrotli-ffi) -type f)
    endif
endif

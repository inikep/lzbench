# mzstd: a Rust crate, built into the Rust codecs' library (see "Rust codecs"
# in the Makefile) and called through its C ABI (mzstd-ffi)
CODECS += MZSTD

ifneq ($(DONT_BUILD_MZSTD),1)
    ifneq ($(HAVE_RUST),1)
        DONT_BUILD_MZSTD := 1
    else ifneq ($(call cargo_at_least,1.89.0),1)
        $(info Cargo $(CARGO_VERSION) is older than 1.89 – skipping mzstd build)
        DONT_BUILD_MZSTD := 1
    else
        RUST_FEATURES += mzstd
        RUST_DEPS += $(shell find $(addprefix $(SRC)lz+entropy/mzstd/,Cargo.toml src mzstd-ffi) -type f)
    endif
endif

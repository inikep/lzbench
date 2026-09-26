# density: a Rust crate, built into the Rust codecs' library (see "Rust codecs"
# in the Makefile)
CODECS += DENSITY

ifneq ($(DONT_BUILD_DENSITY),1)
    ifneq ($(HAVE_RUST),1)
        DONT_BUILD_DENSITY := 1
    # density uses edition 2024, stable since Rust 1.85
    else ifneq ($(call cargo_at_least,1.85.0),1)
        $(info Cargo $(CARGO_VERSION) is older than 1.85 – skipping Density build)
        DONT_BUILD_DENSITY := 1
    else
        RUST_FEATURES += density
        RUST_DEPS += $(shell find $(addprefix $(SRC)misc/density/src/,Cargo.toml src) -type f)
    endif
endif

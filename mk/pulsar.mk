# pulsar: a Rust crate, built into the Rust codecs' library (see "Rust codecs"
# in the Makefile)
CODECS += PULSAR

ifneq ($(DONT_BUILD_PULSAR),1)
    ifneq ($(HAVE_RUST),1)
        DONT_BUILD_PULSAR := 1
    # pulsar itself is edition 2021, but it is built through misc/rust-codecs,
    # which is edition 2024 (rust-version 1.85)
    else ifneq ($(call cargo_at_least,1.85.0),1)
        $(info Cargo $(CARGO_VERSION) is older than 1.85 – skipping pulsar build)
        DONT_BUILD_PULSAR := 1
    else
        RUST_FEATURES += pulsar
        RUST_DEPS += $(shell find $(addprefix $(SRC)bwt/pulsar/,Cargo.toml src) -type f)
    endif
endif

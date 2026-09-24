# GNU make reads GNUmakefile before Makefile.
# Override pulsar hookup to match density without rewriting the 68k Makefile.
include Makefile

# Drop the old hardcoded Linux-only pulsar libs if the included Makefile added them.
LDFLAGS := $(filter-out -lpulsar -ldl,$(LDFLAGS))

PULSAR_SRC_DIR ?= bwt/pulsar/
DONT_BUILD_PULSAR := 1
ifeq ($(HAVE_CARGO),1)
  ifeq ($(HOST_ARCH),$(TARGET_ARCH))
    ifneq ($(BUILD_ARCH),32-bit)
      ifeq ($(filter Windows%,$(OS)),)
        ifeq ($(BUILD_STATIC),1)
          PULSAR_BUILD_TYPE := staticlib
        else
          PULSAR_BUILD_TYPE := cdylib
        endif
        LDFLAGS += -Wl,-rpath,$(PULSAR_SRC_DIR)target/release -L$(PULSAR_SRC_DIR)target/release -lpulsar
        DONT_BUILD_PULSAR := 0
      endif
    endif
  endif
endif

ifeq ($(DONT_BUILD_PULSAR),1)
  DEFINES += -DBENCH_REMOVE_PULSAR
endif

PULSAR_LIB:
ifneq ($(DONT_BUILD_PULSAR),1)
	@echo "Building Pulsar..."
	cd $(PULSAR_SRC_DIR) && \
	RUSTFLAGS="-C target-cpu=native -C linker=$$(lastword $(CXX))" \
	cargo rustc --lib --crate-type=$(PULSAR_BUILD_TYPE) --release -- --print=native-static-libs
endif

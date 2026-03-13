#CC=clang
#CXX=clang++
OBJ_DIR = obj
TEST_OBJ_DIR=$(OBJ_DIR)/test
APP=kanzi
APP_STATIC=$(APP)_static
APP_DYNAMIC=$(APP)_dynamic

ifeq ($(TCMALLOC_ENABLED), 1)
	LDFLAGS += -lpthread -ltcmalloc_minimal
else
	LDFLAGS += -lpthread
endif

ifeq ($(CONCURRENCY_DISABLED), 1)
	CONCURRENCY_FLAG = -DCONCURRENCY_DISABLED
endif

ifndef CXX_STD
    CXX_STD = c++17
endif

ifndef C_STD
    C_STD = c11
endif

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
else
    DETECTED_OS := $(shell uname -s)
endif

CFLAGS += -c -std=$(C_STD) -Wall -Wextra -O3 -fPIC -pedantic -march=native

ifeq ($(DETECTED_OS),Windows)
	CXXFLAGS += -c -std=$(CXX_STD) -Wall -Wextra -O3 -fomit-frame-pointer -fPIC -DNDEBUG -pedantic -march=native -fno-rtti $(CONCURRENCY_FLAG)
else
	ARCH ?= $(shell uname -m)

	# Check for both x86_64 (Linux) and amd64 (FreeBSD)
	ifneq ($(filter x86_64 amd64,$(ARCH)),)
		#CXXFLAGS += -c -std=$(CXX_STD) -fsanitize=undefined -ftrapv -D_FORTIFY_SOURCE=3 -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST -fstrict-aliasing -Wall -Wextra -O3 -fomit-frame-pointer -fPIC -DNDEBUG -pedantic -march=native -fno-rtti $(CONCURRENCY_FLAG)
		CXXFLAGS += -c -std=$(CXX_STD) -fstrict-aliasing -Wall -Wextra -O3 -fomit-frame-pointer -fPIC -DNDEBUG -pedantic -march=native -fno-rtti $(CONCURRENCY_FLAG)
	else
		#CXXFLAGS += -c -std=$(CXX_STD) -fsanitize=signed-integer-overflow -ftrapv -D_FORTIFY_SOURCE=3 -Wall -Wextra -O3 -fPIC -DNDEBUG -pedantic -fno-rtti $(CONCURRENCY_FLAG)
		CXXFLAGS += -c -std=$(CXX_STD) -fstrict-aliasing -Wall -Wextra -Wpedantic -Wdeprecated -O3 -fPIC -DNDEBUG -pedantic -fno-rtti $(CONCURRENCY_FLAG)
	endif
endif	

LIB_COMMON_SOURCES=Global.cpp \
	Event.cpp \
	util/WallTimer.cpp \
	entropy/EntropyUtils.cpp \
	entropy/HuffmanCommon.cpp \
	entropy/CMPredictor.cpp \
	entropy/TPAQPredictor.cpp \
	transform/AliasCodec.cpp \
	transform/BWT.cpp \
	transform/BWTS.cpp \
	transform/DivSufSort.cpp \
	transform/SBRT.cpp \
	transform/BWTBlockCodec.cpp \
	transform/LZCodec.cpp \
	transform/FSDCodec.cpp \
	transform/ROLZCodec.cpp \
	transform/RLT.cpp \
	transform/SRT.cpp \
	transform/TextCodec.cpp \
	transform/UTFCodec.cpp \
	transform/EXECodec.cpp \
	transform/ZRLT.cpp

LIB_COMP_SOURCES=api/Compressor.cpp \
	bitstream/DebugOutputBitStream.cpp \
	bitstream/DefaultOutputBitStream.cpp \
	io/CompressedOutputStream.cpp \
	entropy/ANSRangeEncoder.cpp \
	entropy/BinaryEntropyEncoder.cpp \
	entropy/ExpGolombEncoder.cpp \
	entropy/FPAQEncoder.cpp \
	entropy/HuffmanEncoder.cpp \
	entropy/RangeEncoder.cpp

LIB_DECOMP_SOURCES=api/Decompressor.cpp \
	bitstream/DebugInputBitStream.cpp \
	bitstream/DefaultInputBitStream.cpp \
	io/CompressedInputStream.cpp \
	entropy/ANSRangeDecoder.cpp \
	entropy/BinaryEntropyDecoder.cpp \
	entropy/ExpGolombDecoder.cpp \
	entropy/FPAQDecoder.cpp \
	entropy/HuffmanDecoder.cpp \
	entropy/RangeDecoder.cpp

LIB_SOURCES=$(LIB_COMMON_SOURCES) $(LIB_COMP_SOURCES) $(LIB_DECOMP_SOURCES)

# Define library object files
LIB_OBJECTS=$(filter-out $(OBJ_DIR)/test/%.o,$(LIB_COMMON_OBJECTS) $(LIB_COMP_OBJECTS) $(LIB_DECOMP_OBJECTS))


TEST_SOURCES=test/TestEntropyCodec.cpp \
	test/TestBWT.cpp \
	test/TestCompressedStream.cpp \
	test/TestDefaultBitStream.cpp \
	test/TestFactories.cpp \
	test/TestMalformedStream.cpp \
	test/TestTransforms.cpp \
	test/TestAPI.c

APP_SOURCES=app/Kanzi.cpp \
	app/InfoPrinter.cpp \
	app/BlockCompressor.cpp \
	app/BlockDecompressor.cpp

SOURCES=$(LIB_SOURCES) $(APP_SOURCES)

# Function to create object file paths, preserving the directory structure
OBJ = $(OBJ_DIR)/$(patsubst %.cpp,%.o,$(1))

LIB_COMMON_OBJECTS=$(foreach src,$(LIB_COMMON_SOURCES),$(call OBJ,$(src)))
LIB_COMP_OBJECTS=$(foreach src,$(LIB_COMP_SOURCES),$(call OBJ,$(src)))
LIB_DECOMP_OBJECTS=$(foreach src,$(LIB_DECOMP_SOURCES),$(call OBJ,$(src)))
LIB_OBJECTS=$(LIB_COMMON_OBJECTS) $(LIB_COMP_OBJECTS) $(LIB_DECOMP_OBJECTS)
#TEST_OBJECTS=$(foreach src,$(TEST_SOURCES),$(call OBJ,$(src)))
APP_OBJECTS=$(foreach src,$(APP_SOURCES),$(call OBJ,$(src)))
OBJECTS=$(LIB_OBJECTS) $(APP_OBJECTS) $(TEST_OBJECTS)
RPTS=$(SOURCES:.cpp=.optrpt)


STATIC_LIB_SUFFIX := .a
SHARED_LIB_SUFFIX := .so
PROG_SUFFIX       :=
# Default to -shared (Linux/GNU)
SHARED_OPTION     := -shared

# Detect Windows or macOS
ifeq ($(DETECTED_OS),Windows)
    STATIC_LIB_SUFFIX := .lib
    SHARED_LIB_SUFFIX := .dll
    PROG_SUFFIX       := .exe
    SHARED_OPTION     := -shared
else ifeq ($(DETECTED_OS),Darwin)
    # macOS Specifics
    SHARED_LIB_SUFFIX := .dylib
    SHARED_OPTION     := -dynamiclib
endif

STATIC_LIB := lib$(APP)$(STATIC_LIB_SUFFIX)
SHARED_LIB := lib$(APP)$(SHARED_LIB_SUFFIX)
STATIC_COMP_LIB := lib$(APP)comp$(STATIC_LIB_SUFFIX)
STATIC_DECOMP_LIB := lib$(APP)decomp$(STATIC_LIB_SUFFIX)
SHARED_COMP_LIB := lib$(APP)comp$(SHARED_LIB_SUFFIX)
SHARED_DECOMP_LIB := lib$(APP)decomp$(SHARED_LIB_SUFFIX)

libcomp: $(STATIC_COMP_LIB) $(SHARED_COMP_LIB)

libdecomp: $(STATIC_DECOMP_LIB) $(SHARED_DECOMP_LIB)

lib: $(STATIC_LIB) $(SHARED_LIB)

# Create static libraries
$(STATIC_LIB):$(LIB_OBJECTS)
	$(AR) cr ../lib/$@ $+

$(STATIC_COMP_LIB):$(LIB_COMP_OBJECTS)
	$(AR) cr ../lib/$@ $+

$(STATIC_DECOMP_LIB):$(LIB_DECOMP_OBJECTS)
	$(AR) cr ../lib/$@ $+

# Create shared libraries
$(SHARED_LIB):$(LIB_OBJECTS)
	$(CXX) -o ../lib/$@ $(LDFLAGS) $(SHARED_OPTION) $+

$(SHARED_COMP_LIB):$(LIB_COMP_OBJECTS)
	$(CXX) -o ../lib/$@ $(LDFLAGS) $(SHARED_OPTION) $+

$(SHARED_DECOMP_LIB):$(LIB_DECOMP_OBJECTS)
	$(CXX) -o ../lib/$@ $(LDFLAGS) $(SHARED_OPTION) $+


# Targets for test executables
testAPI: $(LIB_OBJECTS) $(TEST_OBJ_DIR)/TestAPI.o
	$(CXX) $^ -o ../bin/$@ $(LDFLAGS)

testBWT: $(LIB_OBJECTS) $(TEST_OBJ_DIR)/TestBWT.o
	$(CXX) $^ -o ../bin/$@ $(LDFLAGS)

testTransforms: $(LIB_OBJECTS) $(TEST_OBJ_DIR)/TestTransforms.o
	$(CXX) $^ -o ../bin/$@ $(LDFLAGS)

testEntropyCodec: $(LIB_OBJECTS) $(TEST_OBJ_DIR)/TestEntropyCodec.o
	$(CXX) $^ -o ../bin/$@ $(LDFLAGS)

testDefaultBitStream: $(LIB_OBJECTS) $(TEST_OBJ_DIR)/TestDefaultBitStream.o
	$(CXX) $^ -o ../bin/$@ $(LDFLAGS)

testFactories: $(LIB_OBJECTS) $(TEST_OBJ_DIR)/TestFactories.o
	$(CXX) $^ -o ../bin/$@ $(LDFLAGS)

testMalformedStream: $(LIB_OBJECTS) $(TEST_OBJ_DIR)/TestMalformedStream.o
	$(CXX) $^ -o ../bin/$@ $(LDFLAGS)

testCompressedStream: $(LIB_OBJECTS) $(TEST_OBJ_DIR)/TestCompressedStream.o
	$(CXX) $^ -o ../bin/$@ $(LDFLAGS)

test: testAPI testBWT testTransforms testEntropyCodec testDefaultBitStream testFactories testMalformedStream testCompressedStream

# Default executable target
kanzi: $(LIB_OBJECTS) $(APP_OBJECTS)
	$(CXX) $^ -o ../bin/$@ $(LDFLAGS)

# Statically linked executable
ifeq ($(DETECTED_OS),Darwin)
# macOS does not support -static => disable static target
kanzi_static:
	@echo "Static linking is not supported on macOS"
else
kanzi_static: LDFLAGS += -static-libstdc++ -static-libgcc
kanzi_static: $(LIB_OBJECTS) $(APP_OBJECTS)
	$(CXX) $^ -o ../bin/kanzi_static$(PROG_SUFFIX) $(LDFLAGS)
endif

# Dynamically linked executable
kanzi_dynamic: LDFLAGS := $(filter-out -static,$(LDFLAGS))
kanzi_dynamic: $(LIB_OBJECTS) $(APP_OBJECTS)
	$(CXX) $^ -o ../bin/kanzi_dynamic$(PROG_SUFFIX) $(LDFLAGS)

all: lib test $(APP_STATIC) $(APP_DYNAMIC)

# Install / uninstall (may require sudo or admin rights)
ifeq ($(DETECTED_OS),Windows)
INSTALL_DIR=C:/Program\ Files
else
INSTALL_DIR=/usr/local
MAN_DIR=$(INSTALL_DIR)/share/man/man1
endif

print-vars:
	@echo "OS=$(DETECTED_OS)"
	@echo "APP=$(APP)"
	@echo "CXX=$(CXX)"
	@echo "CXXFLAGS=$(CXXFLAGS)"
	@echo "LDFLAGS=$(LDFLAGS)"
	@echo "INSTALL_DIR=$(INSTALL_DIR)"


install: $(STATIC_LIB) $(SHARED_LIB) $(APP)
ifeq ($(DETECTED_OS),Windows)
	copy /Y ..\bin\$(APP)$(PROG_SUFFIX) $(INSTALL_DIR)
else
	install -d $(INSTALL_DIR)/lib
	install -m 644 ../lib/$(STATIC_LIB) $(INSTALL_DIR)/lib
	install -m 644 ../lib/$(SHARED_LIB) $(INSTALL_DIR)/lib
	install -d $(INSTALL_DIR)/include/kanzi
	install -m 644 ./*.hpp $(INSTALL_DIR)/include/kanzi
	install -d $(INSTALL_DIR)/include/kanzi/api
	install -m 644 ./api/*.hpp $(INSTALL_DIR)/include/kanzi/api
	install -d $(INSTALL_DIR)/include/kanzi/io
	install -m 644 ./io/*.hpp $(INSTALL_DIR)/include/kanzi/io
	install -d $(INSTALL_DIR)/include/kanzi/entropy
	install -m 644 ./entropy/*.hpp $(INSTALL_DIR)/include/kanzi/entropy
	install -d $(INSTALL_DIR)/include/kanzi/bitstream
	install -m 644 ./bitstream/*.hpp $(INSTALL_DIR)/include/kanzi/bitstream
	install -d $(INSTALL_DIR)/include/kanzi/transform
	install -m 644 ./transform/*.hpp $(INSTALL_DIR)/include/kanzi/transform
	install -d $(INSTALL_DIR)/include/kanzi/util
	install -m 644 ./util/*.hpp $(INSTALL_DIR)/include/kanzi/util
	install -d $(INSTALL_DIR)/bin
	install -m577 ../bin/$(APP)$(PROG_SUFFIX) $(INSTALL_DIR)/bin
	install -d $(MAN_DIR)
	if [ -f ../kanzi.1.gz ]; then \
		install -m 644 ../kanzi.1.gz $(MAN_DIR)/$(APP).1.gz; \
	elif [ -f ../kanzi.1 ]; then \
		gzip -n -c ../kanzi.1 > $(MAN_DIR)/$(APP).1.gz; \
		chmod 644 $(MAN_DIR)/$(APP).1.gz; \
	else \
		echo "Error: missing ../kanzi.1.gz or ../kanzi.1"; \
		exit 1; \
	fi
endif

# Uninstall may require sudo or admin rights
uninstall:
ifeq ($(DETECTED_OS),Windows)
	del /Q $(INSTALL_DIR)\$(APP)$(PROG_SUFFIX)
else
	rm -f -r $(INSTALL_DIR)/include/$(APP)
	rm -f $(INSTALL_DIR)/lib/$(STATIC_LIB)
	rm -f $(INSTALL_DIR)/lib/$(SHARED_LIB)
	rm -f $(INSTALL_DIR)/bin/$(APP_STATIC)$(PROG_SUFFIX)
	rm -f $(INSTALL_DIR)/bin/$(APP_DYNAMIC)$(PROG_SUFFIX)
	rm -f $(INSTALL_DIR)/bin/$(APP)$(PROG_SUFFIX)
	rm -f $(MAN_DIR)/$(APP).1.gz
endif

clean:
ifeq ($(DETECTED_OS),Windows)
	del /S $(OBJ_DIR)\*.o
	..\bin\$(APP)$(PROG_SUFFIX) \
	..\bin\$(APP_STATIC)$(PROG_SUFFIX) \
	..\bin\$(APP_DYNAMIC)$(PROG_SUFFIX) ..\bin\test*$(PROG_SUFFIX) \
	..\lib\$(STATIC_LIB) ..\lib\$(SHARED_LIB) \
	..\lib\$(STATIC_COMP_LIB) ..\lib\$(STATIC_DECOMP_LIB) \
	..\lib\$(SHARED_COMP_LIB) ..\lib\$(SHARED_DECOMP_LIB)
else
	rm -f ../bin/test*$(PROG_SUFFIX) $(OBJECTS) $(RPTS) \
	../bin/$(APP)$(PROG_SUFFIX) \
	../bin/$(APP_STATIC)$(PROG_SUFFIX) \
	../bin/$(APP_DYNAMIC)$(PROG_SUFFIX) \
	../lib/$(STATIC_LIB) ../lib/$(SHARED_LIB) \
	../lib/$(STATIC_COMP_LIB) ../lib/$(STATIC_DECOMP_LIB) \
	../lib/$(SHARED_COMP_LIB) ../lib/$(SHARED_DECOMP_LIB)
	rm -rf $(OBJ_DIR)
endif

ifeq ($(DETECTED_OS),Windows)
MKDIR = if not exist $(subst /,\,$(dir $@)) mkdir $(subst /,\,$(dir $@))
else
MKDIR = mkdir -p $(dir $@)
endif

$(OBJ_DIR)/%.o: %.cpp
	@$(MKDIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(OBJ_DIR)/%.o: %.c
	@$(MKDIR)
	$(CC) $(CFLAGS) $< -o $@

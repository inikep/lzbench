cmake_minimum_required(VERSION 3.10)
project(kanzi)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED True)
#set(CMAKE_CXX_COMPILER "clang++")

if(CONCURRENCY_DISABLED)
    add_definitions(-DCONCURRENCY_DISABLED)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -fomit-frame-pointer -fPIC -DNDEBUG -pedantic -march=native -fno-rtti")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# Source files
set(LIB_COMMON_SOURCES
    Global.cpp
    Event.cpp
    entropy/EntropyUtils.cpp
    entropy/HuffmanCommon.cpp
    entropy/CMPredictor.cpp
    entropy/TPAQPredictor.cpp
    transform/AliasCodec.cpp
    transform/BWT.cpp
    transform/BWTS.cpp
    transform/DivSufSort.cpp
    transform/SBRT.cpp
    transform/BWTBlockCodec.cpp
    transform/LZCodec.cpp
    transform/FSDCodec.cpp
    transform/ROLZCodec.cpp
    transform/RLT.cpp
    transform/SRT.cpp
    transform/TextCodec.cpp
    transform/UTFCodec.cpp
    transform/EXECodec.cpp
    transform/ZRLT.cpp
)

set(LIB_COMP_SOURCES
    api/Compressor.cpp
    bitstream/DebugOutputBitStream.cpp
    bitstream/DefaultOutputBitStream.cpp
    io/CompressedOutputStream.cpp
    entropy/ANSRangeEncoder.cpp
    entropy/BinaryEntropyEncoder.cpp
    entropy/ExpGolombEncoder.cpp
    entropy/FPAQEncoder.cpp
    entropy/HuffmanEncoder.cpp
    entropy/RangeEncoder.cpp
)

set(LIB_DECOMP_SOURCES
    api/Decompressor.cpp
    bitstream/DebugInputBitStream.cpp
    bitstream/DefaultInputBitStream.cpp
    io/CompressedInputStream.cpp
    entropy/ANSRangeDecoder.cpp
    entropy/BinaryEntropyDecoder.cpp
    entropy/ExpGolombDecoder.cpp
    entropy/FPAQDecoder.cpp
    entropy/HuffmanDecoder.cpp
    entropy/RangeDecoder.cpp
)

set(TEST_SOURCES
    test/TestEntropyCodec.cpp
    test/TestBWT.cpp
    test/TestCompressedStream.cpp
    test/TestDefaultBitStream.cpp
    test/TestTransforms.cpp
)

set(APP_SOURCES
    app/Kanzi.cpp
    app/InfoPrinter.cpp
    app/BlockCompressor.cpp
    app/BlockDecompressor.cpp
)

# Libraries
add_library(libkanzi STATIC ${LIB_COMMON_SOURCES} ${LIB_COMP_SOURCES} ${LIB_DECOMP_SOURCES})
add_library(libkanzi_shared SHARED ${LIB_COMMON_SOURCES} ${LIB_COMP_SOURCES} ${LIB_DECOMP_SOURCES})
set_target_properties(libkanzi PROPERTIES OUTPUT_NAME "kanzi")
set_target_properties(libkanzi_shared PROPERTIES OUTPUT_NAME "kanzi")

#add_library(libkanzi_comp STATIC ${LIB_COMP_SOURCES})
#add_library(libkanzi_decomp STATIC ${LIB_DECOMP_SOURCES})

#add_library(libkanzi_comp_shared SHARED ${LIB_COMP_SOURCES})
#add_library(libkanzi_decomp_shared SHARED ${LIB_DECOMP_SOURCES})

# Test executables
add_executable(testBWT test/TestBWT.cpp)
target_link_libraries(testBWT libkanzi)

add_executable(testTransforms test/TestTransforms.cpp)
target_link_libraries(testTransforms libkanzi)

add_executable(testEntropyCodec test/TestEntropyCodec.cpp)
target_link_libraries(testEntropyCodec libkanzi)

add_executable(testDefaultBitStream test/TestDefaultBitStream.cpp)
target_link_libraries(testDefaultBitStream libkanzi)

add_executable(testCompressedStream test/TestCompressedStream.cpp)
target_link_libraries(testCompressedStream libkanzi)

# Main executable
add_executable(kanzi ${APP_SOURCES})
target_link_libraries(kanzi libkanzi)

# Custom target to build all tests
add_custom_target(test
    DEPENDS testBWT testTransforms testEntropyCodec testDefaultBitStream testCompressedStream
)
# Custom target to build static libraries
add_custom_target(static_lib
    DEPENDS libkanzi #libkanzi_comp libkanzi_decomp
)

# Custom target to build shared libraries
add_custom_target(shared_lib
    DEPENDS libkanzi_shared #libkanzi_comp_shared libkanzi_decomp_shared
)

# Custom target to build all libraries (static and shared)
add_custom_target(lib
    DEPENDS static_lib shared_lib
)

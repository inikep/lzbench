# High-level C++ Quick Start Guide

NVCOMP provides a C++ interface, which simplifies use of the library by
throwing exceptions and managing state and temporary memory allocation 
inside of nvcompManager objects.

The high level interface provides the following features:
* Compression settings are stored in the nvcompManager object
* Users can decompress nvcomp-compressed buffers without knowing how the buffer was compressed
* The nvcompManager can automatically split a single, uncompressed, contiguous buffer into chunks
  to allow the algorithms to exploit available parallelism. 

To use NVCOMP's C++ interface, you will need to include `nvcomp.hpp`
and the headers of the specific compressors you will be using.  For example, 
for the LZ4 compression scheme used in `high_level_quickstart_example.cpp`, we need to include 
```c++
#include "nvcomp/lz4.hpp"
#include "nvcomp.hpp"
```

All nvCOMP APIs are declared within the `nvcomp` namespace. For ease of use, we suggest to specify the following in the appropriate scope:
```c++
using namespace nvcomp;
```

In the below we introduce the interface and summarize the declarations of relevant member functions of the nvcompManager class hierarchy. For fully worked examples of the same, please view `examples/high_level_quickstart_example.cpp`.

## Manager Construction

The user has two options for constructing an nvcompManager. In either case the user can specify a CUDA stream to use for all nvcompManager GPU operations. If a stream is not specified, the default stream will be used.

#### 1) Construction from an nvcomp-compressed buffer

The user can construct a manager using a compressed buffer. This is the recommended way of constructing a manager for decompression,
as it is less error-prone.

In order to use the `create_manager` factory, the user must include `nvcomp/nvcompManagerFactory.hpp`

```c++
cudaStream_t stream;
CUDA_CHECK(cudaStreamCreate(&stream));

std::shared_ptr<nvcompManagerBase> decomp_nvcomp_manager = create_manager(comp_buffer, stream);
```

A complete worked example using this approach is provided in the `decomp_compressed_with_manager_factory_example` within `high_level_quickstart_example.cpp`.

#### 2) Direct construction

In direct construction, the user must specify the parameters of the particular compressor they wish to use for compression or decompression.
If manually specifying the manager for decompression, care must be taken to ensure that the configuration of the manager matches the configuration
used to compress the buffers.

Chunk size is a common parameter that determines the size of the chunking internally. Some compressors may provide a higher compression ratio 
if given a larger chunk size. For example in LZ4, the larger the chunk size the larger the lookback window the algorithm can use to find matches.

```c++
cudaStream_t stream;
CUDA_CHECK(cudaStreamCreate(&stream));

const int chunk_size = 1 << 16;
nvcompType_t data_type = NVCOMP_TYPE_CHAR;

LZ4Manager nvcomp_manager{chunk_size, data_type, stream};
```

## Compression

Compression consists of two steps: `Configuration` then `Compression`. 

### Step 1) Configuration

The configuration stage provides the maximum size of the compressed buffer. It also performs internal setup for the
compression.

```c++
/*
decomp_buffer_size: The size of the uncompressed buffer
*/
CompressionConfig configure_compression(const size_t decomp_buffer_size)
```

### Step 2) Compression

Compression takes the result of configure_compression, a const input buffer and a result buffer.
The result buffer shoudl be allocated based on the result of configure_compression, which includes
the maximum possible compressed size.

```c++
/*
decomp_buffer: The uncompressed input buffer
comp_buffer: The output result buffer
comp_config: The configure result from configure_compression
*/
void compress(
    const uint8_t* decomp_buffer, 
    uint8_t* comp_buffer,
    const CompressionConfig& comp_config)
```

## Decompression

Decompression consists of two steps: Configuration then Decompression.

### Step 1) Configuration
To configure the decompression, the user has two options.
#### 1) configure using a compressed buffer

If when decompressing a compressed buffer the user doesn't have the CompressionConfig used to compress the buffer, 
the user must use the configure API. This API synchronizes the stream provided at construction of the manager, because 
the decompression needs information that may only be accessible on the GPU.
```c++
/*
comp_buffer: The compressed buffer under consideration
*/
DecompressionConfig configure_decompression(const uint8_t* comp_buffer)
```

#### 2) configure using a compression config
Sometimes, the user will retain the CompressionConfig object that was used to compress the buffer. In this case,
the DecompressionConfig can be constructed from the CompressionConfig. Since the CompressionConfig resides in
host memory, this configuration can happen without synchronizing the stream. 

```c++
/*
comp_config: The configure result from the compression that produced this compressed buffer
*/
DecompressionConfig configure_decompression(const CompressionConfig& comp_config)
```

### Step 2) Decompression

Decompression utilizes a result `decomp_buffer` that should be provided by the user. The size of the decompressed buffer
is provided by the previous configuration step.

```c++
/**
   decomp_buffer: The location to output the decompressed data to (GPU accessible).
   comp_buffer The compressed input data (GPU accessible).
   decomp_config: Resulted from configure_decompression
   */
  void decompress(
      uint8_t* decomp_buffer, 
      const uint8_t* comp_buffer,
      const DecompressionConfig& decomp_config);
```

## HLIF Compression / Decompression Examples - LZ4

`examples/high_level_quickstart_example.cpp` provides worked examples of 
  - constructing the manager from arguments 
  - constructing the manager from a compressed buffer
  - Streamed compression and decompression of multiple buffers

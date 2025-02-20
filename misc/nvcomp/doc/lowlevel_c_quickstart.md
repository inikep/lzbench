# Low-level C Quick Start Guide

Some applications require compressing or decompressing multiple small inputs,
so we provide an additional API to do this efficiently. These API calls combine
all compression/decompression
into a single execution, greatly improving performance compared with running
each input individually.  This API relies on the user to
split the data into chunks, as well as manage metadata information such as
compressed and uncompressed chunk sizes. When splitting data, for best
performance, chunks should be relatively equal size to achieve good
load-balancing as well as extract sufficient parallelism. So in the case that
there are multiple inputs to compress, it may still be best to break each one
up into smaller chunks.

The low level batched C API provides a set of functions to do batched decompression
and compression. 

In the following API description, replace * with the desired compression algorithm. For example, for LZ4,  nvCompBatched\*CompressAsync becomes nvCompBatchedLZ4CompressAsync and nvcompBatched*DecompressAsync becomes nvcompBatchedLZ4DecompressAsync.

## Compression API  

To do batched compression, a temporary workspace is required to be allocated in device memory. The size of this space is computed using:
```c++
/*
 batch_size: Number of chunks to compress
 max_chunk_bytes: Size in bytes of the largest chunk
 format_opts: The compression options to use
 temp_bytes: The output temporary size required by the compression algorithm
*/
nvcompStatus_t nvcompBatched*CompressGetTempSize(
    size_t batch_size,
    size_t max_chunk_bytes,
    nvcompBatched*FormatOpts format_opts,
    size_t * temp_bytes);
```

Then compression is done using:
```c++
/*
 device_uncompressed_ptrs: The pointers on the GPU to uncompressed batched items.
 device_uncompressed_bytes: The size of each uncompressed batch item on the GPU.
 max_uncompressed_chunk_bytes: The maximum size in bytes of the largest chunk in the batch.
 batch_size: The number of chunks in the batch.
 device_temp_ptr: The temporary GPU workspace.
 temp_bytes: The size of the temporary GPU workspace.
 device_compressed_ptrs: The pointers on the GPU, to the output location for each compressed batch item (output). 
 device_compressed_bytes: The compressed size of each chunk on the GPU(output). 
 format_opts: The compression options to use.
 stream: The CUDA stream to operate on.
*/
nvcompStatus_t nvcompBatched*CompressAsync(  
  const void* const* device_uncompressed_ptrs,    
  const size_t* device_uncompressed_bytes,  
  size_t max_uncompressed_chunk_bytes,  
  size_t batch_size,  
  void* device_temp_ptr,  
  size_t temp_bytes,  
  void* const* device_compressed_ptrs,  
  size_t* device_compressed_bytes,  
  nvcompBatched*Opts_t format_opts,  
  cudaStream_t stream);
```

## Decompression API

Decompression also requires a temporary workspace. This is computed using:
```c++
/*
 batch_size: Number of chunks to decompress
 max_chunk_bytes: Size in bytes of the largest chunk
 temp_bytes: The output temporary size required by the compression algorithm
*/
nvcompStatus_t nvcompBatched*DecompressGetTempSize(
    size_t batch_size,
    size_t max_chunk_bytes,
    size_t * temp_bytes);
```

During decompression, device memory buffers that are large enough to hold the decompression result must be provided. There are three possible workflows that are supported:

#### 1) Uncompressed size for each buffer is known exactly (e.g. Apache Parquet https://github.com/apache/parquet-format )

#### 2) Only maximum uncompressed size across all buffers is known (e.g. Apache ORC)

#### 3) No information about the uncompressed sizes is provided (e.g. Apache Avro)

For case 3), nvCOMP provides an API for pre-processing the compressed file 
  to determine the proper sizes for the decompressed output buffers. This API is as follows:

```c++
/**
 * @brief Calculates the decompressed size of each chunk asynchronously. This is
 * needed when we do not know the expected output size. All pointers must be GPU
 * accessible. Note, if the stream is corrupt, the sizes will be garbage.
 *
 * device_compress_ptrs: The compressed chunks of data. 
 * device_compressed_bytes: The size of each compressed chunk
 * device_uncompressed_bytes: The calculated decompressed size of each chunk. 
 * batch_size: The number of chunks in the batch
 * stream: The CUDA stream to operate on.
 */
nvcompStatus_t nvcompBatched*GetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    cudaStream_t stream);
```

With the decompressed sizes known, we can now use the decompression API:

```c++
/*
  device_compressed_ptrs: The pointers on the GPU, to the compressed chunks. 
  device_compressed_bytes: The size of each compressed chunk on the GPU.
  device_uncompressed_bytes: The decompressed buffer size. This is needed to prevent OOB accesses.
  device_actual_uncompressed_bytes: The actual calculated decompressed size of each chunk.
  batch_size: The number of chunks in the batch.
  device_temp_ptr: The temporary GPU space.
  temp_bytes: The size of the temporary GPU space.
  device_uncompressed_ptrs: The pointers on the GPU, to where to uncompress each chunk (output). 
  device_statuses: The status for each chunk of whether it was decompressed or not. 
  stream: The CUDA stream to operate on.
*/
nvcompStatus_t nvcompBatched*DecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr,
    size_t temp_bytes,
    void* const* device_uncompressed_ptrs,
    nvcompStatus_t* device_statuses,
    cudaStream_t stream);
```

Note that the *device_actual_uncompressed_bytes* and *device_statuses* can both be specified as nullptr for LZ4, Snappy, and GDeflate. If these are nullptr, these methods will not compute these values. In particular, if device_statuses is nullptr then out of bounds (OOB) error checking is disabled. This can lead to significant increases in decompression throughput.

## Batched Compression / Decompression Example - LZ4

For an example of batched compression and decompression using LZ4, please see the examples/low_level_quickstart_example.cpp.


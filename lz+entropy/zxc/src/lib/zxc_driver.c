// SPDX-License-Identifier: BSD-3-Clause
/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) Bertrand Lebonnois and contributors.
 */

/**
 * @file zxc_driver.c
 * @brief Userspace @c FILE*-flavored driver: multi-threaded streaming and
 *        the seekable @c FILE* open helper.
 *
 * Two distinct subsystems live in this translation unit because they share
 * the same userspace-only host requirements (@c <stdio.h>, threading, and
 * platform file-descriptor extraction): keeping them together means a
 * single TU to exclude when building for kernel / freestanding targets.
 *
 *   1. Streaming engine: a ring-buffer producer / worker / consumer
 *      pipeline that parallelises block processing over @c FILE* streams.
 *      Public API: @ref zxc_stream_compress, @ref zxc_stream_decompress,
 *      @ref zxc_stream_get_decompressed_size.
 *
 *   2. Seekable @c FILE* wrapper: builds a @ref zxc_reader_t whose
 *      @c read_at uses @c pread / @c ReadFile on the file descriptor
 *      extracted from a @c FILE*, then delegates to
 *      @ref zxc_seekable_open_reader.  Public API:
 *      @ref zxc_seekable_open_file.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "../../include/zxc_buffer.h"
#include "../../include/zxc_dict.h"
#include "../../include/zxc_error.h"
#include "../../include/zxc_seekable.h"
#include "../../include/zxc_stream.h"
#include "zxc_internal.h"
#include "zxc_threads.h"

// ============================================================================
// PLATFORM SHIMS
// ============================================================================
// Threading comes from zxc_threads.h (POSIX names everywhere, Win32 underneath);
// only the file-positioning names are remapped here, next to their users.
#if defined(_WIN32)
#include <io.h> /* _get_osfhandle, _fileno */
#include <malloc.h>
#include <sys/types.h>
#include <windows.h>

// Map POSIX file positioning functions to Windows equivalents
#define fseeko _fseeki64
#define ftello _ftelli64
#else
#include <errno.h>
#include <sys/stat.h>
#endif

// Below this block size the per-block system calls cost more than they save.
#define ZXC_STREAM_PREAD_MIN_BLOCK ((size_t)64 * 1024)

// Output per regular-file job: the thread handoff is paid once per batch. Not for
// a pipe, whose output would wait for a full batch.
#define ZXC_STREAM_BATCH_BYTES ((size_t)256 * 1024)

// Jobs per worker when they carry several blocks each.
#define ZXC_STREAM_RING_BATCHED 2U

// ============================================================================
// STREAMING ENGINE (Producer / Worker / Consumer)
// ============================================================================
// Implements a Ring Buffer architecture to parallelize block processing.

/**
 * @enum job_status_t
 * @brief Represents the lifecycle states of a processing job within the ring
 * buffer.
 *
 * @var JOB_STATUS_FREE
 *      The job slot is empty and available to be filled with new data by the
 * writer.
 * @var JOB_STATUS_FILLED
 *      The job slot has been populated with input data and is ready for
 * processing by a worker.
 * @var JOB_STATUS_PROCESSED
 *      The worker has finished processing the data; the result is ready to be
 * consumed/written out.
 */
typedef enum { JOB_STATUS_FREE, JOB_STATUS_FILLED, JOB_STATUS_PROCESSED } job_status_t;

/**
 * @struct zxc_stream_job_t
 * @brief Represents a single unit of work (a chunk of data) to be processed.
 *
 * This structure holds the input and output buffers for a specific chunk of
 * data, along with its processing status. It is padded to align with cache
 * lines to prevent false sharing in a multi-threaded environment.
 *
 * @var zxc_stream_job_t::in_buf
 *      Raw input; NULL with positioned reads.
 * @var zxc_stream_job_t::in_cap
 *      The total allocated capacity of the input buffer.
 * @var zxc_stream_job_t::in_sz
 *      The actual size of the valid data currently in the input buffer.
 * @var zxc_stream_job_t::out_buf
 *      Pointer to the buffer where processed (compressed/decompressed) data is
 * stored.
 * @var zxc_stream_job_t::out_cap
 *      The total allocated capacity of the output buffer.
 * @var zxc_stream_job_t::result_sz
 *      The actual size of the valid data produced in the output buffer.
 * @var zxc_stream_job_t::block_index
 *      Frame position of the job's first block: seeds the block checksums.
 * @var zxc_stream_job_t::in_off
 *      Positioned reads: file offset of the job's first block.
 * @var zxc_stream_job_t::job_id
 *      A unique identifier for the job, often used for ordering or debugging.
 * @var zxc_stream_job_t::status
 *      The current state of this job (Free, Filled, or Processed).
 * @var zxc_stream_job_t::pad
 *      Padding bytes to ensure the structure size aligns with the cache line
 * size (@c ZXC_CACHE_LINE_SIZE), minimizing cache contention between threads
 * accessing adjacent jobs.
 */
typedef struct {
    uint8_t* in_buf;
    size_t in_cap;
    size_t in_sz;
    uint8_t* out_buf;
    size_t out_cap;
    size_t result_sz;
    uint64_t block_index;
    uint64_t in_off;
    int job_id;
    ZXC_ATOMIC job_status_t status;  // Atomic for lock-free status updates
    char pad[ZXC_CACHE_LINE_SIZE];   // Prevent False Sharing
} zxc_stream_job_t;

/**
 * @struct zxc_stream_ctx_t
 * @brief The main context structure managing the streaming
 * compression/decompression state.
 *
 * This structure orchestrates the producer-consumer workflow. It manages the
 * ring buffer of jobs, the worker queue, synchronization primitives (mutexes
 * and condition variables), and configuration settings for the compression
 * algorithm.
 *
 * @var zxc_stream_ctx_t::jobs
 *      Array of job structures acting as the ring buffer.
 * @var zxc_stream_ctx_t::ring_size
 *      The total number of slots in the jobs array.
 * @var zxc_stream_ctx_t::worker_queue
 *      A circular queue containing indices of jobs ready to be picked up by
 * worker threads.
 * @var zxc_stream_ctx_t::wq_head
 *      Index of the head of the worker queue (where workers take jobs).
 * @var zxc_stream_ctx_t::wq_tail
 *      Index of the tail of the worker queue (where the writer adds jobs).
 * @var zxc_stream_ctx_t::wq_count
 *      Current number of items in the worker queue.
 * @var zxc_stream_ctx_t::lock
 *      Mutex used to protect access to shared resources (queue indices, status
 * changes).
 * @var zxc_stream_ctx_t::cond_reader
 *      Condition variable to signal the output thread (reader) that processed
 * data is available.
 * @var zxc_stream_ctx_t::cond_worker
 *      Condition variable to signal worker threads that new work is available.
 * @var zxc_stream_ctx_t::cond_writer
 *      Condition variable to signal the input thread (writer) that job slots
 * are free.
 * @var zxc_stream_ctx_t::shutdown_workers
 *      Flag indicating that worker threads should terminate.
 * @var zxc_stream_ctx_t::compression_mode
 *      Indicates the operation mode (e.g., compression or decompression).
 * @var zxc_stream_ctx_t::io_error
 *      Atomic flag telling every thread to stop. Set for any failure, not just
 *      I/O, because the wait loops poll it without holding the lock.
 * @var zxc_stream_ctx_t::fail_code
 *      The first failure's actual error code, kept so a corrupt archive is not
 *      reported as an I/O problem. Written under @c lock, first writer wins.
 * @var zxc_stream_ctx_t::write_idx
 *      The index of the next job slot to be written to by the main thread.
 * @var zxc_stream_ctx_t::compression_level
 *      The configured level of compression (trading off speed vs. ratio).
 * @var zxc_stream_ctx_t::chunk_size
 *      The size of each data chunk to be processed.
 * @var zxc_stream_ctx_t::checksum_enabled
 *      Flag indicating whether checksum verification/generation is active.
 * @var zxc_stream_ctx_t::file_has_checksum
 *     Flag indicating whether the input file includes checksums.
 * @var zxc_stream_ctx_t::file_has_seek
 *     Flag indicating whether the file header announces a seek table.
 * @var zxc_stream_ctx_t::frame_in
 *     Decompression: bytes of the frame read so far, which place its footer.
 * @var zxc_stream_ctx_t::progress_cb
 *     Optional callback function for reporting progress during processing.
 * @var zxc_stream_ctx_t::progress_user_data
 *    User data pointer to be passed to the progress callback function.
 * @var zxc_stream_ctx_t::total_input_bytes
 *     Total size of the input data in bytes, used for progress tracking.
 * @var zxc_stream_ctx_t::dict
 *     Pointer to the optional dictionary buffer used to prime
 *     compression/decompression, NULL when no dictionary is in use.
 * @var zxc_stream_ctx_t::dict_size
 *     Size of the dictionary in bytes, 0 when no dictionary is in use.
 * @var zxc_stream_ctx_t::dict_huf
 *     Shared dictionary literal Huffman table (128-byte packed code-lengths
 *     header), NULL when absent.
 * @var zxc_stream_ctx_t::in_fd
 *     Descriptor for positioned reads, -1 when the input goes through stdio.
 * @var zxc_stream_ctx_t::in_pos
 *     Positioned reads: the reader's file offset.
 * @var zxc_stream_ctx_t::in_alloc
 *     Positioned reads: size of each worker's input buffer.
 * @var zxc_stream_ctx_t::batch
 *     Most blocks a decompression job carries (@c ZXC_STREAM_BATCH_BYTES).
 * @var zxc_stream_ctx_t::ahead
 *     Positioned reads: the next block header, read with the previous checksum.
 * @var zxc_stream_ctx_t::ahead_len
 *     Bytes of @c ahead read (short at the end of the file), -1 when none.
 * @var zxc_stream_ctx_t::src_total
 *     Compression: source bytes read.
 * @var zxc_stream_ctx_t::digest
 *     Decompression: the frame's archive digest, folded by the reader.
 */
typedef struct {
    zxc_stream_job_t* jobs;
    size_t ring_size;
    int* worker_queue;
    int wq_head;
    int wq_tail;
    int wq_count;
    pthread_mutex_t lock;
    pthread_cond_t cond_reader;
    pthread_cond_t cond_worker;
    pthread_cond_t cond_writer;
    int shutdown_workers;
    int compression_mode;
    ZXC_ATOMIC int io_error;
    int fail_code;
    int write_idx;
    int compression_level;
    size_t chunk_size;
    int checksum_enabled;
    int file_has_checksum;
    int file_has_seek;
    uint64_t frame_in;
    zxc_progress_callback_t progress_cb;
    void* progress_user_data;
    uint64_t total_input_bytes;
    const uint8_t* dict;
    size_t dict_size;
    const uint8_t* dict_huf; /**< Shared dictionary literal table (128-byte packed
                                  code-lengths header), NULL when absent. */
    int in_fd;
    uint64_t in_pos;
    size_t in_alloc;
    int batch;
    uint8_t ahead[ZXC_BLOCK_HEADER_SIZE];
    int ahead_len;
    uint64_t src_total;
    uint64_t digest;
} zxc_stream_ctx_t;

/** @brief The ring slot after @p idx. */
static ZXC_ALWAYS_INLINE int zxc_ring_next(const zxc_stream_ctx_t* ctx, const int idx) {
    return (int)(((size_t)idx + 1) % ctx->ring_size);
}

/**
 * @struct writer_args_t
 * @brief Structure containing arguments for the writer callback function.
 *
 * This structure is used to pass necessary context and state information
 * to the function responsible for writing compressed or decompressed data
 * to a file stream.
 *
 * @var writer_args_t::ctx
 * Pointer to the ZXC stream context, holding the state of the
 * compression/decompression stream.
 *
 * @var writer_args_t::f
 * Pointer to the output file stream where data will be written.
 *
 * @var writer_args_t::total_bytes
 * Accumulator for the total number of bytes written to the file so far.
 *
 * @var writer_args_t::bytes_processed
 * The number of bytes processed so far, used for progress reporting.
 *
 * @var writer_args_t::seek_comp
 * Array of compressed block sizes for seek table construction.
 *
 * @var writer_args_t::seek_count
 * Number of entries in the seek table.
 *
 * @var writer_args_t::seek_cap
 * Capacity of the seek table array.
 */
typedef struct {
    zxc_stream_ctx_t* ctx;
    FILE* f;
    int64_t total_bytes;
    uint64_t digest;           // archive digest (compress side)
    uint64_t bytes_processed;  // For progress callback
    uint32_t* seek_comp;
    uint64_t seek_count;
    uint64_t seek_cap;
} writer_args_t;

/**
 * @brief A started engine: job ring, workers, writer. Decompression keeps it
 *        across frames of one block size and checksum flag.
 */
typedef struct {
    zxc_stream_ctx_t ctx;
    writer_args_t w;
    pthread_t* workers;
    int n_workers;
    pthread_t writer;
    uint8_t* mem_block;
    int read_idx;  // Next ring slot the reader fills
} zxc_stream_engine_t;

#if !defined(_WIN32)
/**
 * @brief Thread-safe @c pread of @p len bytes, retrying @c EINTR and partial
 *        transfers. A 32-bit @c off_t rejects offsets past 2 GiB.
 *
 * @return Bytes read, short only at the end of the file, or -1 on error.
 */
static int64_t zxc_pread_full(const int fd, void* dst, const size_t len, const uint64_t offset) {
    if (sizeof(off_t) < sizeof(uint64_t) && (offset > INT32_MAX || len > INT32_MAX - offset))
        return -1;  // LCOV_EXCL_LINE
    size_t done = 0;
    while (done < len) {
        const ssize_t r = pread(fd, (uint8_t*)dst + done, len - done, (off_t)(offset + done));
        if (r > 0)
            done += (size_t)r;
        else if (r == 0)
            break;
        else if (errno != EINTR)
            return -1;  // LCOV_EXCL_LINE
    }
    return (int64_t)done;
}
#endif

/** @brief Descriptor of a regular-file stream, -1 for anything else (may be live). */
static int zxc_stream_regular_fd(FILE* f) {
#if defined(_WIN32)
    // LCOV_EXCL_START - Windows path, not reachable on POSIX CI
    const int fd = _fileno(f);
    if (fd < 0) return -1;
    const intptr_t handle = _get_osfhandle(fd);
    return (handle != -1 && GetFileType((HANDLE)handle) == FILE_TYPE_DISK) ? fd : -1;
    // LCOV_EXCL_STOP
#else
    struct stat st;
    const int fd = fileno(f);
    return (fd >= 0 && fstat(fd, &st) == 0 && S_ISREG(st.st_mode)) ? fd : -1;
#endif
}

/** @brief Switches a regular file to positioned reads from its current position. */
static void zxc_stream_open_positioned(zxc_stream_ctx_t* ctx, FILE* f_in, const int fd) {
#if defined(_WIN32)
    (void)ctx;
    (void)f_in;
    (void)fd;
#else
    if (sizeof(off_t) < sizeof(uint64_t)) return;  // LCOV_EXCL_LINE
    const long long pos = ftello(f_in);
    if (UNLIKELY(pos < 0)) return;  // LCOV_EXCL_LINE
    ctx->in_fd = fd;
    ctx->in_pos = (uint64_t)pos;
    ctx->ahead_len = -1;
#endif
}

/**
 * @brief Positioned read of the input, safe from any thread.
 *
 * @return Bytes read, short only at the end of the file, or @ref ZXC_ERROR_IO.
 */
static int64_t zxc_stream_read_at(const zxc_stream_ctx_t* ctx, void* dst, const size_t len,
                                  const uint64_t offset) {
#if defined(_WIN32)
    (void)ctx;
    (void)dst;
    (void)len;
    (void)offset;
    return ZXC_ERROR_IO;
#else
    const int64_t r = zxc_pread_full(ctx->in_fd, dst, len, offset);
    return r < 0 ? ZXC_ERROR_IO : r;
#endif
}

/**
 * @brief Reads up to @p len input bytes on the reader thread: by position while
 *        the engine reads positioned, through stdio otherwise.
 *
 * @param[in]  ctx Started engine, or NULL to read through stdio.
 * @param[in]  f   The input stream.
 * @param[out] dst Destination.
 * @param[in]  len Bytes wanted.
 * @return Bytes read, short only at the end of the input, or @ref ZXC_ERROR_IO.
 */
static int64_t zxc_stream_in_read(zxc_stream_ctx_t* ctx, FILE* f, void* dst, const size_t len) {
    if (ctx && ctx->in_fd >= 0) {
        const int64_t r = zxc_stream_read_at(ctx, dst, len, ctx->in_pos);
        if (LIKELY(r > 0)) ctx->in_pos += (uint64_t)r;
        return r;
    }
    const size_t got = fread(dst, 1, len, f);
    // A short read without EOF is a read error. Not ferror(): the caller may have
    // left a stale error indicator on the stream.
    if (UNLIKELY(got < len && !feof(f))) return ZXC_ERROR_IO;  // LCOV_EXCL_LINE
    return (int64_t)got;
}

/** @brief Records a reader failure; workers write @c fail_code under the same lock. */
static void zxc_stream_reader_fail(zxc_stream_ctx_t* ctx, const int code) {
    pthread_mutex_lock(&ctx->lock);
    if (code && !ctx->fail_code) ctx->fail_code = code;
    ctx->io_error = 1;
    pthread_mutex_unlock(&ctx->lock);
}

/**
 * @brief Runs one chunk through the codec the stream's mode selects.
 *
 * Both codecs are called by name: the compressor takes a mutable context, the
 * decoder a const one, so no single function pointer type fits both.
 */
static ZXC_ALWAYS_INLINE int zxc_stream_process(const zxc_stream_ctx_t* ctx, zxc_cctx_t* cctx,
                                                const uint8_t* in, const size_t in_sz, uint8_t* out,
                                                const size_t out_cap, const uint64_t block_index) {
    return ctx->compression_mode
               ? zxc_compress_chunk_wrapper(cctx, in, in_sz, out, out_cap, block_index)
               : zxc_decompress_chunk_wrapper(cctx, in, in_sz, out, out_cap, block_index);
}

/**
 * @brief Decodes a job's blocks into its output.
 *
 * Headers are re-validated: with positioned reads the file may have changed since
 * the reader measured the job.
 *
 * @return Bytes decoded, or a negative @ref zxc_error_t.
 */
static int zxc_stream_decode_job(const zxc_stream_ctx_t* ctx, zxc_cctx_t* cctx,
                                 const zxc_stream_job_t* job, const uint8_t* in,
                                 uint8_t* dict_work) {
    const size_t checksum_sz = ctx->file_has_checksum ? ZXC_BLOCK_CHECKSUM_SIZE : 0;
    // One block of output plus the fast decoder's margin.
    const size_t room = ctx->chunk_size + ZXC_DECOMPRESS_TAIL_PAD;
    // Dictionary: decode behind the prefix so back-references resolve.
    uint8_t* const scratch = dict_work ? dict_work + ctx->dict_size : NULL;
    uint64_t block_index = job->block_index;
    size_t ip = 0;
    size_t op = 0;

    while (ip < job->in_sz) {
        const size_t rem = job->in_sz - ip;
        zxc_block_header_t bh;
        if (UNLIKELY(rem < ZXC_BLOCK_HEADER_SIZE ||
                     zxc_read_block_header(in + ip, ZXC_BLOCK_HEADER_SIZE, &bh) != ZXC_OK))
            return ZXC_ERROR_CORRUPT_DATA;
        const uint64_t block_sz = (uint64_t)ZXC_BLOCK_HEADER_SIZE + bh.comp_size + checksum_sz;
        if (UNLIKELY(block_sz > rem || job->out_cap - op < room)) return ZXC_ERROR_CORRUPT_DATA;

        const int res =
            zxc_stream_process(ctx, cctx, in + ip, (size_t)block_sz,
                               scratch ? scratch : job->out_buf + op, room, block_index++);
        if (UNLIKELY(res < 0)) return res;
        if (scratch) ZXC_MEMCPY(job->out_buf + op, scratch, (size_t)res);
        ip += (size_t)block_sz;
        op += (size_t)res;
    }
    return (int)op;
}

/**
 * @brief Worker thread: pull a job, process it, hand it to the writer.
 *
 * Sleeps on @c cond_worker until @c worker_queue has a job, then runs
 * @ref zxc_stream_process over it and marks it @c JOB_STATUS_PROCESSED. Each worker
 * owns a thread-local @c zxc_cctx_t so the parallel part never touches a shared
 * context. With positioned reads it also reads each job's blocks itself.
 *
 * The writer is signalled only when the finished job is the one it is waiting
 * for (@c jid == @c ctx->write_idx). Jobs completing out of order stay silent,
 * which keeps the writer from waking up on work it cannot yet emit.
 *
 * @param[in] arg The shared @c zxc_stream_ctx_t.
 * @return Always NULL.
 */
static void* zxc_stream_worker(void* arg) {
    zxc_stream_ctx_t* const ctx = (zxc_stream_ctx_t*)arg;
    zxc_cctx_t cctx;

    const int unified_chk = (ctx->compression_mode == 1)
                                ? ctx->checksum_enabled
                                : (ctx->file_has_checksum && ctx->checksum_enabled);

    const size_t eff_chunk = (ctx->dict_size > 0 && ctx->compression_mode == 1)
                                 ? zxc_block_size_ceil(ctx->dict_size + ctx->chunk_size)
                                 : ctx->chunk_size;
    // Per-worker input buffer for positioned reads: stays in this core's cache.
    uint8_t* const in_work = ctx->in_fd >= 0 ? (uint8_t*)ZXC_MALLOC(ctx->in_alloc) : NULL;
    if (UNLIKELY(zxc_cctx_init(&cctx, eff_chunk, ctx->compression_mode, ctx->compression_level,
                               unified_chk, ctx->dict_size) != ZXC_OK ||
                 zxc_cctx_attach_dict_huf(&cctx, ctx->dict_huf) != ZXC_OK ||
                 (ctx->in_fd >= 0 && !in_work))) {
        // LCOV_EXCL_START
        ZXC_FREE(in_work);
        zxc_cctx_free(&cctx);
        pthread_mutex_lock(&ctx->lock);
        ctx->io_error = 1;
        pthread_cond_broadcast(&ctx->cond_writer);
        pthread_cond_broadcast(&ctx->cond_reader);
        pthread_mutex_unlock(&ctx->lock);
        return NULL;
        // LCOV_EXCL_STOP
    }

    cctx.compression_level = ctx->compression_level;

    // Per-worker dict buffer for assembling [dict | block_data]
    const size_t dsz = ctx->dict_size;
    uint8_t* const dict_work = cctx.dict_buffer;
    if (dict_work) ZXC_MEMCPY(dict_work, ctx->dict, dsz);

    while (1) {
        zxc_stream_job_t* job = NULL;
        pthread_mutex_lock(&ctx->lock);
        while (ctx->wq_count == 0 && !ctx->shutdown_workers) {
            pthread_cond_wait(&ctx->cond_worker, &ctx->lock);
        }
        if (ctx->shutdown_workers && ctx->wq_count == 0) {
            pthread_mutex_unlock(&ctx->lock);
            break;
        }
        const int jid = ctx->worker_queue[ctx->wq_tail];
        ctx->wq_tail = zxc_ring_next(ctx, ctx->wq_tail);
        ctx->wq_count--;
        job = &ctx->jobs[jid];
        pthread_mutex_unlock(&ctx->lock);

        // A short read (the file shrank) is an I/O error.
        const uint8_t* const in = in_work ? in_work : job->in_buf;
        int res;
        if (in_work && UNLIKELY(zxc_stream_read_at(ctx, in_work, job->in_sz, job->in_off) !=
                                (int64_t)job->in_sz)) {
            res = ZXC_ERROR_IO;
        } else if (ctx->compression_mode == 0) {
            res = zxc_stream_decode_job(ctx, &cctx, job, in, dict_work);
        } else if (dict_work) {
            ZXC_MEMCPY(dict_work + dsz, in, job->in_sz);
            res = zxc_stream_process(ctx, &cctx, dict_work, dsz + job->in_sz, job->out_buf,
                                     job->out_cap, job->block_index);
        } else {
            res = zxc_stream_process(ctx, &cctx, in, job->in_sz, job->out_buf, job->out_cap,
                                     job->block_index);
        }

        pthread_mutex_lock(&ctx->lock);
        job->result_sz = UNLIKELY(res < 0) ? 0 : (size_t)res;
        job->status = JOB_STATUS_PROCESSED;
        if (UNLIKELY(res < 0)) {
            // Keep the codec's own diagnosis; io_error only stops the others.
            if (!ctx->fail_code) ctx->fail_code = res;
            ctx->io_error = 1;
            pthread_cond_broadcast(&ctx->cond_writer);
            pthread_cond_broadcast(&ctx->cond_reader);
        } else if (jid == ctx->write_idx) {
            pthread_cond_signal(&ctx->cond_writer);
        }
        pthread_mutex_unlock(&ctx->lock);
    }
    ZXC_FREE(in_work);
    zxc_cctx_free(&cctx);
    return NULL;
}

/**
 * @brief Asynchronous writer thread function.
 *
 * This function runs as a separate thread responsible for writing processed
 * data chunks to the output file. It operates on a ring buffer of jobs shared
 * with the reader and worker threads.
 *
 * **Ordering Enforcement:**
 * The writer MUST write blocks in the exact order they were read. Even if
 * worker threads finish jobs out of order (e.g., job 2 finishes before job 1),
 * the writer waits for `ctx->write_idx` (job 1) to be `JOB_STATUS_PROCESSED`.
 *
 * **Workflow:**
 * 1. **Wait:** Sleeps on `cond_writer` until the job at `ctx->write_idx` is
 * ready.
 * 2. **Write:** Writes the `out_buf` to the file.
 * 3. **Release:** Sets the job status to `JOB_STATUS_FREE` and signals
 * `cond_reader`, allowing the main thread to reuse this slot for new input.
 * 4. **Advance:** Increments `ctx->write_idx` to wait for the next sequential
 * block.
 *
 * @param[in] arg Pointer to a `writer_args_t` structure containing the stream
 * context, the output file handle, and a counter for total bytes written.
 * @return Always returns NULL.
 */
static void* zxc_async_writer(void* arg) {
    writer_args_t* const args = (writer_args_t*)arg;
    zxc_stream_ctx_t* const ctx = args->ctx;
    while (1) {
        zxc_stream_job_t* const job = &ctx->jobs[ctx->write_idx];
        pthread_mutex_lock(&ctx->lock);
        while (job->status != JOB_STATUS_PROCESSED && !ctx->io_error)
            pthread_cond_wait(&ctx->cond_writer, &ctx->lock);
        // Woken by an error: the job may still be decoding, so leave it alone.
        if (UNLIKELY(job->status != JOB_STATUS_PROCESSED)) {
            pthread_mutex_unlock(&ctx->lock);
            break;
        }

        const size_t result_sz = job->result_sz;
        const size_t in_sz = job->in_sz;
        pthread_mutex_unlock(&ctx->lock);

        if (result_sz == (size_t)-1) break;

        if (args->f && result_sz > 0) {
            if (fwrite(job->out_buf, 1, result_sz, args->f) != result_sz) {
                pthread_mutex_lock(&ctx->lock);
                ctx->io_error = 1;
                pthread_cond_signal(&ctx->cond_reader);
                pthread_mutex_unlock(&ctx->lock);
            }
        }
        if (UNLIKELY(ctx->io_error)) {
            pthread_mutex_lock(&ctx->lock);
            job->status = JOB_STATUS_FREE;
            pthread_cond_signal(&ctx->cond_reader);
            pthread_mutex_unlock(&ctx->lock);
            break;
        }
        args->total_bytes += (int64_t)result_sz;
        if (ctx->checksum_enabled && ctx->compression_mode == 1 &&
            LIKELY(result_sz >= ZXC_BLOCK_CHECKSUM_SIZE))
            args->digest = zxc_digest_combine(
                args->digest, zxc_le32(job->out_buf + result_sz - ZXC_BLOCK_CHECKSUM_SIZE));

        // Seekable: record compressed block size
        if (args->seek_comp && ctx->compression_mode == 1) {
            if (UNLIKELY(args->seek_count >= args->seek_cap)) {
                const uint64_t max_cap = SIZE_MAX / sizeof(uint32_t);
                uint32_t* nc = NULL;
                if (LIKELY(args->seek_cap < max_cap)) {
                    args->seek_cap = args->seek_cap < max_cap / 2 ? args->seek_cap * 2 : max_cap;
                    nc = (uint32_t*)ZXC_REALLOC(args->seek_comp,
                                                (size_t)args->seek_cap * sizeof(uint32_t));
                }
                // LCOV_EXCL_START
                if (UNLIKELY(!nc)) {
                    pthread_mutex_lock(&ctx->lock);
                    if (!ctx->fail_code) ctx->fail_code = ZXC_ERROR_MEMORY;
                    ctx->io_error = 1;
                    job->status = JOB_STATUS_FREE;
                    pthread_cond_signal(&ctx->cond_reader);
                    pthread_mutex_unlock(&ctx->lock);
                    break;
                }
                // LCOV_EXCL_STOP
                args->seek_comp = nc;
            }
            args->seek_comp[args->seek_count++] = (uint32_t)result_sz;
        }

        if (ctx->progress_cb) {
            // LCOV_EXCL_START
            args->bytes_processed += ctx->compression_mode == 1 ? in_sz : result_sz;
            ctx->progress_cb(args->bytes_processed, ctx->total_input_bytes,
                             ctx->progress_user_data);
            // LCOV_EXCL_STOP
        }

        pthread_mutex_lock(&ctx->lock);
        job->status = JOB_STATUS_FREE;
        ctx->write_idx = zxc_ring_next(ctx, ctx->write_idx);
        pthread_cond_signal(&ctx->cond_reader);
        pthread_mutex_unlock(&ctx->lock);
    }
    return NULL;
}

/**
 * @brief Stops the threads, then frees the ring and its synchronisation objects.
 *
 * The writer is told to stop by a terminator job at @c read_idx, once the jobs
 * before it are written. @p writer_started is 0 when setup failed before it.
 */
static void zxc_stream_engine_stop(zxc_stream_engine_t* e, const int writer_started) {
    zxc_stream_ctx_t* const ctx = &e->ctx;
    if (writer_started) {
        zxc_stream_job_t* const end_job = &ctx->jobs[e->read_idx];
        pthread_mutex_lock(&ctx->lock);
        while (end_job->status != JOB_STATUS_FREE && !ctx->io_error)
            pthread_cond_wait(&ctx->cond_reader, &ctx->lock);
        end_job->result_sz = (size_t)-1;
        end_job->status = JOB_STATUS_PROCESSED;
        pthread_cond_broadcast(&ctx->cond_writer);
        pthread_mutex_unlock(&ctx->lock);
        pthread_join(e->writer, NULL);
    }
    pthread_mutex_lock(&ctx->lock);
    ctx->shutdown_workers = 1;
    pthread_cond_broadcast(&ctx->cond_worker);
    pthread_mutex_unlock(&ctx->lock);
    for (int i = 0; i < e->n_workers; i++) pthread_join(e->workers[i], NULL);

    pthread_cond_destroy(&ctx->cond_writer);
    pthread_cond_destroy(&ctx->cond_worker);
    pthread_cond_destroy(&ctx->cond_reader);
    pthread_mutex_destroy(&ctx->lock);
    ZXC_FREE(e->workers);
    ZXC_ALIGNED_FREE(e->mem_block);
}

/**
 * @brief Allocates the job ring and starts the workers and the writer.
 *
 * **Architecture: Producer-Consumer with Ring Buffer**
 * - **Ring Buffer:** A fixed-size array of `zxc_stream_job_t` structures.
 * - **Producer (Main Thread):** Reads chunks from `f_in` and fills "Free" slots
 *   in the ring buffer. It blocks if no slots are free (backpressure).
 * - **Workers:** Pick up "Filled" jobs from a queue, process them, and mark
 * them as "Processed".
 * - **Consumer (Writer Thread):** Waits for the *next sequential* job to be
 *   "Processed", writes it to `f_out`, and marks the slot as "Free".
 *
 * **Double-Buffering & Zero-Copy:**
 * We allocate `alloc_in` and `alloc_out` buffers for each job. The reader reads
 * directly into `in_buf`, and the writer writes directly from `out_buf`,
 * minimizing memory copies.
 *
 * **Regular-file Input (decompression):**
 * Small blocks travel several to a job (@c ZXC_STREAM_BATCH_BYTES); with two
 * threads or more, larger ones are read by the workers and all `n_threads`
 * decode. Pipes and memory streams keep one block per job through `fread`.
 *
 * The caller fills the run's fields of @c e->ctx (mode, level, block size,
 * checksums, dictionary, progress) and of @c e->w (output, seek table, counters).
 *
 * @param[in,out] e         The engine, its run fields filled.
 * @param[in]     f_in      The input stream.
 * @param[in]     n_threads Thread count; 0 or less auto-detects the online processors.
 * @return @ref ZXC_OK, or @ref ZXC_ERROR_MEMORY with nothing left to free.
 */
static int zxc_stream_engine_start(zxc_stream_engine_t* e, FILE* f_in, const int n_threads) {
    zxc_stream_ctx_t* const ctx = &e->ctx;
    const int mode = ctx->compression_mode;
    const size_t chunk = ctx->chunk_size;

    int num_threads = (n_threads > 0) ? n_threads : zxc_num_procs();
    if (num_threads > ZXC_MAX_THREADS) num_threads = ZXC_MAX_THREADS;
    // Positioned reads need two threads: a lone worker is faster fed by the reader.
    ctx->in_fd = -1;
    const int in_fd = (mode == 0) ? zxc_stream_regular_fd(f_in) : -1;
    ctx->batch =
        (in_fd >= 0 && chunk < ZXC_STREAM_BATCH_BYTES) ? (int)(ZXC_STREAM_BATCH_BYTES / chunk) : 1;
    if (in_fd >= 0 && num_threads > 1 && chunk >= ZXC_STREAM_PREAD_MIN_BLOCK)
        zxc_stream_open_positioned(ctx, f_in, in_fd);
    // One thread for the reader and writer, none with positioned reads.
    const int num_workers = ctx->in_fd >= 0 ? num_threads : (num_threads > 1) ? num_threads - 1 : 1;
    // Batched jobs hold several blocks: fewer keep the workers fed.
    ctx->ring_size = (size_t)num_workers * (ctx->batch > 1 ? ZXC_STREAM_RING_BATCHED : 4U);

    const uint64_t max_out = zxc_compress_bound(chunk);
    const size_t raw_alloc_in =
        (size_t)((mode ? chunk : max_out * (uint64_t)ctx->batch) + ZXC_PAD_SIZE);
    const size_t alloc_in = (raw_alloc_in + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;
    ctx->in_alloc = alloc_in;

    const size_t raw_alloc_out =
        (size_t)((mode ? max_out : chunk * (size_t)ctx->batch + ZXC_DECOMPRESS_TAIL_PAD) +
                 ZXC_PAD_SIZE);
    const size_t alloc_out = (raw_alloc_out + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;

    // Positioned reads use per-worker input buffers.
    const size_t ring_in = ctx->in_fd >= 0 ? 0 : alloc_in;
    const size_t per_job_sz = sizeof(zxc_stream_job_t) + sizeof(int) + ring_in + alloc_out;
    if (UNLIKELY(per_job_sz > SIZE_MAX / ctx->ring_size))
        return ZXC_ERROR_MEMORY;  // LCOV_EXCL_LINE

    uint8_t* const mem_block = ZXC_ALIGNED_MALLOC(ctx->ring_size * per_job_sz, ZXC_CACHE_LINE_SIZE);
    if (UNLIKELY(!mem_block)) return ZXC_ERROR_MEMORY;  // LCOV_EXCL_LINE
    e->mem_block = mem_block;

    uint8_t* ptr = mem_block;
    ctx->jobs = (zxc_stream_job_t*)ptr;
    ptr += ctx->ring_size * sizeof(zxc_stream_job_t);
    ctx->worker_queue = (int*)ptr;
    ptr += ctx->ring_size * sizeof(int);
    uint8_t* buf_in = ptr;
    ptr += ctx->ring_size * ring_in;
    uint8_t* buf_out = ptr;

    ZXC_MEMSET(mem_block, 0, ctx->ring_size * (sizeof(zxc_stream_job_t) + sizeof(int)));

    for (size_t i = 0; i < ctx->ring_size; i++) {
        ctx->jobs[i].job_id = (int)i;
        ctx->jobs[i].status = JOB_STATUS_FREE;
        ctx->jobs[i].in_buf = ring_in ? buf_in + (i * ring_in) : NULL;
        ctx->jobs[i].in_cap = alloc_in - ZXC_PAD_SIZE;
        ctx->jobs[i].in_sz = 0;
        ctx->jobs[i].out_buf = buf_out + (i * alloc_out);
        ctx->jobs[i].out_cap = alloc_out - ZXC_PAD_SIZE;
        ctx->jobs[i].result_sz = 0;
    }

    pthread_mutex_init(&ctx->lock, NULL);
    pthread_cond_init(&ctx->cond_reader, NULL);
    pthread_cond_init(&ctx->cond_worker, NULL);
    pthread_cond_init(&ctx->cond_writer, NULL);

    e->workers = ZXC_MALLOC((size_t)num_workers * sizeof(pthread_t));
    if (LIKELY(e->workers)) {
        while (e->n_workers < num_workers &&
               pthread_create(&e->workers[e->n_workers], NULL, zxc_stream_worker, ctx) == 0)
            e->n_workers++;
    }
    e->w.ctx = ctx;
    if (UNLIKELY(e->n_workers == 0 ||
                 pthread_create(&e->writer, NULL, zxc_async_writer, &e->w) != 0)) {
        zxc_stream_engine_stop(e, 0);  // LCOV_EXCL_LINE
        return ZXC_ERROR_MEMORY;       // LCOV_EXCL_LINE
    }
    return ZXC_OK;
}

/** @brief Stops a decompression engine; moves the @c FILE* to where positioned reads ended. */
static int zxc_stream_engine_close(zxc_stream_engine_t* e, FILE* f_in) {
    zxc_stream_engine_stop(e, 1);
    if (e->ctx.in_fd < 0) return ZXC_OK;
    e->ctx.in_fd = -1;
    return fseeko(f_in, (long long)e->ctx.in_pos, SEEK_SET) == 0 ? ZXC_OK : ZXC_ERROR_IO;
}

/** @brief Waits until the writer has written every job queued so far. */
static void zxc_stream_engine_drain(zxc_stream_engine_t* e) {
    zxc_stream_ctx_t* const ctx = &e->ctx;
    // The writer frees jobs in order: the last one queued is free last.
    const zxc_stream_job_t* const last =
        &ctx->jobs[((size_t)e->read_idx + ctx->ring_size - 1) % ctx->ring_size];
    pthread_mutex_lock(&ctx->lock);
    while (last->status != JOB_STATUS_FREE && !ctx->io_error)
        pthread_cond_wait(&ctx->cond_reader, &ctx->lock);
    pthread_mutex_unlock(&ctx->lock);
}

/**
 * @brief Fills a decompression job with up to @c ctx->batch consecutive blocks.
 *
 * A failure sets @c io_error and drops the blocks gathered: nothing is queued.
 *
 * @return Bytes the job holds, 0 when there is nothing to queue.
 */
static size_t zxc_stream_read_blocks(zxc_stream_ctx_t* ctx, FILE* f_in, zxc_stream_job_t* job,
                                     int* n_blocks, int* eof) {
    const int positioned = ctx->in_fd >= 0;
    const size_t checksum_sz = ctx->file_has_checksum ? ZXC_BLOCK_CHECKSUM_SIZE : 0;
    // Folded only when finish_decompress will compare it: like the block
    // checksums, the file flag and the caller's switch both.
    const int fold = ctx->file_has_checksum && ctx->checksum_enabled;
    size_t fill = 0;
    int code = 0;  // 0: a read error

    for (*n_blocks = 0; *n_blocks < ctx->batch; (*n_blocks)++) {
        uint8_t bh_buf[ZXC_BLOCK_HEADER_SIZE];
        const uint64_t block_off = ctx->in_pos;
        int64_t h_read;
        if (positioned && ctx->ahead_len >= 0) {
            h_read = ctx->ahead_len;
            ZXC_MEMCPY(bh_buf, ctx->ahead, (size_t)h_read);
            ctx->ahead_len = -1;
        } else if (positioned) {
            h_read = zxc_stream_read_at(ctx, bh_buf, sizeof(bh_buf), ctx->in_pos);
        } else {
            h_read = zxc_stream_in_read(NULL, f_in, bh_buf, sizeof(bh_buf));
        }
        if (UNLIKELY(h_read < 0)) goto _failed;  // LCOV_EXCL_LINE
        if (positioned) ctx->in_pos += (uint64_t)h_read;
        if (UNLIKELY(h_read < ZXC_BLOCK_HEADER_SIZE)) {
            *eof = 1;
            return fill;
        }

        zxc_block_header_t bh;
        if (UNLIKELY(zxc_read_block_header(bh_buf, ZXC_BLOCK_HEADER_SIZE, &bh) != ZXC_OK)) {
            code = ZXC_ERROR_CORRUPT_DATA;  // LCOV_EXCL_LINE
            goto _failed;                   // LCOV_EXCL_LINE
        }

        if (bh.block_type == ZXC_BLOCK_EOF) {
            if (UNLIKELY(zxc_check_eof_header(bh_buf) != ZXC_OK)) {
                code = ZXC_ERROR_BAD_HEADER;
                goto _failed;
            }
            ctx->frame_in += ZXC_BLOCK_HEADER_SIZE;
            *eof = 1;
            return fill;
        }

        if (UNLIKELY((uint64_t)bh.comp_size > (uint64_t)ctx->chunk_size)) {
            code = ZXC_ERROR_BAD_BLOCK_SIZE;  // LCOV_EXCL_LINE
            goto _failed;                     // LCOV_EXCL_LINE
        }

        const size_t body_total = (size_t)bh.comp_size + checksum_sz;
        const uint64_t total_len = (uint64_t)body_total + ZXC_BLOCK_HEADER_SIZE;
        if (UNLIKELY(total_len > (uint64_t)(job->in_cap - fill))) goto _failed;

        uint8_t tail[ZXC_BLOCK_CHECKSUM_SIZE + ZXC_BLOCK_HEADER_SIZE];
        const uint8_t* stored = tail;
        int got = 1;

        if (positioned) {
            // The worker reads the bodies; the digest needs only the checksum,
            // read with the next header.
            if (fill == 0) job->in_off = block_off;
            ctx->in_pos += body_total;
            if (fold) {
                const int64_t r = zxc_stream_read_at(ctx, tail, sizeof(tail),
                                                     ctx->in_pos - ZXC_BLOCK_CHECKSUM_SIZE);
                got = r >= ZXC_BLOCK_CHECKSUM_SIZE;
                if (LIKELY(got)) {
                    ctx->ahead_len = (int)(r - ZXC_BLOCK_CHECKSUM_SIZE);
                    ZXC_MEMCPY(ctx->ahead, tail + ZXC_BLOCK_CHECKSUM_SIZE, (size_t)ctx->ahead_len);
                }
            }
        } else {
            uint8_t* const block = job->in_buf + fill;
            ZXC_MEMCPY(block, bh_buf, ZXC_BLOCK_HEADER_SIZE);
            stored = block + ZXC_BLOCK_HEADER_SIZE + bh.comp_size;

            // Single fread for body + checksum (reduces syscalls)
            got = fread(block + ZXC_BLOCK_HEADER_SIZE, 1, body_total, f_in) == body_total;
        }

        if (UNLIKELY(!got)) goto _failed;
        if (fold) ctx->digest = zxc_digest_combine(ctx->digest, zxc_le32(stored));
        fill += (size_t)total_len;
        ctx->frame_in += total_len;
    }
    return fill;

_failed:
    zxc_stream_reader_fail(ctx, code);
    *eof = 1;
    return 0;
}

/** @brief Fills a compression job with the next chunk of the source. */
static size_t zxc_stream_read_chunk(zxc_stream_ctx_t* ctx, FILE* f_in, zxc_stream_job_t* job,
                                    int* eof) {
    const size_t got = fread(job->in_buf, 1, ctx->chunk_size, f_in);
    ctx->src_total += got;
    if (got < ctx->chunk_size) *eof = 1;
    return got;
}

/**
 * @brief Reads the input and feeds the ring until the stream ends.
 *
 * Fills the ring from slot @p read_idx on; returns the slot after the last job.
 */
static int zxc_stream_read_loop(zxc_stream_ctx_t* ctx, FILE* f_in, int read_idx) {
    int read_eof = 0;
    uint64_t block_index = 0;

    while (!read_eof && !ctx->io_error) {
        zxc_stream_job_t* const job = &ctx->jobs[read_idx];
        pthread_mutex_lock(&ctx->lock);
        while (job->status != JOB_STATUS_FREE && !ctx->io_error)
            pthread_cond_wait(&ctx->cond_reader, &ctx->lock);
        pthread_mutex_unlock(&ctx->lock);

        if (UNLIKELY(ctx->io_error)) break;

        int n_blocks = 1;
        const size_t read_sz = ctx->compression_mode
                                   ? zxc_stream_read_chunk(ctx, f_in, job, &read_eof)
                                   : zxc_stream_read_blocks(ctx, f_in, job, &n_blocks, &read_eof);
        if (UNLIKELY(read_eof && read_sz == 0)) break;

        job->in_sz = read_sz;
        job->block_index = block_index;
        block_index += (uint64_t)n_blocks;
        pthread_mutex_lock(&ctx->lock);
        job->status = JOB_STATUS_FILLED;
        ctx->worker_queue[ctx->wq_head] = read_idx;
        ctx->wq_head = zxc_ring_next(ctx, ctx->wq_head);
        ctx->wq_count++;
        read_idx = zxc_ring_next(ctx, read_idx);
        pthread_cond_signal(&ctx->cond_worker);
        pthread_mutex_unlock(&ctx->lock);
    }
    return read_idx;
}

/**
 * @brief Closes a compressed stream: EOF block, optional seek table, footer.
 *
 * Runs once the writer thread has drained, so it appends straight to the file.
 */
static void zxc_stream_finish_compress(zxc_stream_ctx_t* ctx, writer_args_t* w, FILE* f_out) {
    // EOF block
    uint8_t eof_buf[ZXC_BLOCK_HEADER_SIZE];
    const zxc_block_header_t eof_bh = {
        .block_type = ZXC_BLOCK_EOF, .block_flags = 0, .reserved = 0, .comp_size = 0};
    zxc_write_block_header(eof_buf, ZXC_BLOCK_HEADER_SIZE, &eof_bh);
    if (UNLIKELY(f_out &&
                 fwrite(eof_buf, 1, ZXC_BLOCK_HEADER_SIZE, f_out) != ZXC_BLOCK_HEADER_SIZE))
        ctx->io_error = 1;  // LCOV_EXCL_LINE
    else
        w->total_bytes += ZXC_BLOCK_HEADER_SIZE;

    // Seekable: write SEK block between EOF and footer, empty when no block was read
    if (!ctx->io_error && w->seek_comp) {
        // Header, then one group at a time (seek_comp itself stays resident).
        uint8_t st_buf[ZXC_SEEK_GROUP_BYTES];
        const int h = zxc_seek_table_header(st_buf, sizeof(st_buf), w->seek_count);
        if (UNLIKELY(h < 0 || (f_out && fwrite(st_buf, 1, (size_t)h, f_out) != (size_t)h)))
            ctx->io_error = 1;  // LCOV_EXCL_LINE
        else
            w->total_bytes += h;

        uint64_t anchor = ZXC_FILE_HEADER_SIZE;
        for (uint64_t i = 0; i < w->seek_count && !ctx->io_error; i += ZXC_SEEK_GROUP) {
            const size_t bytes =
                zxc_seek_write_group(st_buf, &anchor, w->seek_comp + i,
                                     zxc_seek_group_len(w->seek_count, i / ZXC_SEEK_GROUP));
            if (UNLIKELY(f_out && fwrite(st_buf, 1, bytes, f_out) != bytes))
                ctx->io_error = 1;  // LCOV_EXCL_LINE
            else
                w->total_bytes += (int64_t)bytes;
        }
    }

    // Footer (FORMAT.md 8)
    uint8_t footer_buf[ZXC_FOOTER_MAX_SIZE_WITH_DIGEST];
    const size_t footer_len =
        (size_t)zxc_write_file_footer(footer_buf, sizeof(footer_buf), (uint64_t)w->total_bytes,
                                      ctx->src_total, w->digest, ctx->checksum_enabled);
    if (UNLIKELY(f_out && fwrite(footer_buf, 1, footer_len, f_out) != footer_len))
        ctx->io_error = 1;
    else
        w->total_bytes += (int64_t)footer_len;
}

/**
 * @brief Consumes and validates the trailer of a decompressed stream.
 */
static void zxc_stream_finish_decompress(zxc_stream_ctx_t* ctx, const uint64_t frame_out,
                                         FILE* f_in) {
    // After the EOF block: the SEK block when the header announced one, then the
    // footer, whose length follows from the bytes read and the bytes produced.

    if (ctx->file_has_seek) {
        uint8_t sek[ZXC_BLOCK_HEADER_SIZE];
        uint64_t remaining = 0;
        uint64_t sek_bytes = 0;
        // A short read is a truncation, unless the bytes read already disagree.
        const int64_t got = zxc_stream_in_read(ctx, f_in, sek, sizeof(sek));
        const int src_rc = got < 0 ? ZXC_ERROR_IO
                                   : zxc_check_seek_header(sek, (size_t)got, frame_out,
                                                           ctx->chunk_size, &remaining);
        if (UNLIKELY(src_rc != ZXC_OK)) {
            if (src_rc == ZXC_ERROR_CORRUPT_DATA && !ctx->fail_code) ctx->fail_code = src_rc;
            ctx->io_error = 1;
        }
        sek_bytes = remaining;
        // Skip the SEK payload; positioned, a cut table shows at the footer read.
        if (ctx->in_fd >= 0) {
            ctx->in_pos += remaining;
            remaining = 0;
        }
        uint8_t discard[512];
        while (remaining > 0 && !ctx->io_error) {
            const size_t chunk = remaining < sizeof(discard) ? (size_t)remaining : sizeof(discard);
            if (UNLIKELY(fread(discard, 1, chunk, f_in) != chunk)) ctx->io_error = 1;
            remaining -= chunk;
        }
        ctx->frame_in += ZXC_BLOCK_HEADER_SIZE + sek_bytes;
    }
    const zxc_footer_layout_t fl =
        zxc_footer_layout(ctx->frame_in, frame_out, ctx->file_has_checksum);
    uint64_t stored_digest = 0;
    if (!ctx->io_error) {
        uint8_t footer[ZXC_FOOTER_MAX_SIZE_WITH_DIGEST];
        const int64_t got = zxc_stream_in_read(ctx, f_in, footer, fl.len);
        const int frc =
            got < 0 ? ZXC_ERROR_IO
                    : zxc_check_file_footer(footer, (size_t)got, &fl, frame_out, &stored_digest);
        if (UNLIKELY(frc != ZXC_OK)) {
            if (frc == ZXC_ERROR_CORRUPT_DATA && !ctx->fail_code) ctx->fail_code = frc;
            ctx->io_error = 1;
        }
    }
    if (!ctx->io_error && ctx->file_has_checksum && ctx->checksum_enabled &&
        UNLIKELY(stored_digest != ctx->digest)) {
        if (!ctx->fail_code) ctx->fail_code = ZXC_ERROR_BAD_CHECKSUM;
        ctx->io_error = 1;
    }
    // Whatever follows is the next frame's, or an error: see zxc_stream_decompress().
}

/**
 * @brief Reads exactly @p len bytes, telling a clean end from a short read.
 *
 * @param[in]  ctx        Started engine, or NULL before the first frame.
 * @param[in]  f          The input stream.
 * @param[out] dst        Destination.
 * @param[in]  len        Bytes to read.
 * @param[in]  short_code Returned when the input ends inside the range.
 * @return @ref ZXC_OK; 1 when the input ended before the first byte;
 *         @p short_code when it ended inside the range; @ref ZXC_ERROR_IO on
 *         a read error.
 */
static int zxc_stream_read_exact(zxc_stream_ctx_t* ctx, FILE* f, void* dst, const size_t len,
                                 const int short_code) {
    const int64_t got = zxc_stream_in_read(ctx, f, dst, len);
    if (LIKELY(got == (int64_t)len)) return ZXC_OK;
    if (UNLIKELY(got < 0)) return ZXC_ERROR_IO;  // LCOV_EXCL_LINE
    return got == 0 ? 1 : short_code;
}

/**
 * @brief Reads and checks a frame header whose magic word is already read.
 *
 * @param[in]  ctx          Started engine, or NULL before the first frame.
 * @param[in]  f_in         The input stream, just past the magic word.
 * @param[in]  magic        The magic word's 4 bytes.
 * @param[in]  dict_id      Id of the caller's dictionary, 0 without one.
 * @param[out] chunk        The frame's block size.
 * @param[out] has_checksum Whether the frame carries checksums.
 * @param[out] has_seek     Whether it ends with a seek table.
 * @return @ref ZXC_OK, or a negative @ref zxc_error_t.
 */
static int zxc_stream_read_frame_header(zxc_stream_ctx_t* ctx, FILE* f_in, const uint8_t* magic,
                                        const uint32_t dict_id, size_t* chunk, int* has_checksum,
                                        int* has_seek) {
    uint8_t h[ZXC_FILE_HEADER_SIZE];
    uint32_t header_dict_id = 0;
    ZXC_MEMCPY(h, magic, sizeof(uint32_t));
    const int rrc =
        zxc_stream_read_exact(ctx, f_in, h + sizeof(uint32_t),
                              ZXC_FILE_HEADER_SIZE - sizeof(uint32_t), ZXC_ERROR_SRC_TOO_SMALL);
    if (UNLIKELY(rrc != ZXC_OK)) return rrc == 1 ? ZXC_ERROR_SRC_TOO_SMALL : rrc;

    const int hrc = zxc_read_file_header(h, ZXC_FILE_HEADER_SIZE, chunk, has_checksum,
                                         &header_dict_id, has_seek);
    if (UNLIKELY(hrc != ZXC_OK)) return hrc;
    if (header_dict_id != 0) {
        if (UNLIKELY(dict_id == 0)) return ZXC_ERROR_DICT_REQUIRED;
        if (UNLIKELY(dict_id != header_dict_id)) return ZXC_ERROR_DICT_MISMATCH;
    }
    return ZXC_OK;
}

/**
 * @brief Decodes one frame's blocks, SEK block and footer through a started engine.
 *
 * @return Bytes the frame produced, or a negative @ref zxc_error_t.
 */
static int64_t zxc_stream_decode_frame(zxc_stream_engine_t* e, FILE* f_in, const int has_seek) {
    zxc_stream_ctx_t* const ctx = &e->ctx;
    const int64_t base = e->w.total_bytes;
    ctx->file_has_seek = has_seek;
    ctx->frame_in = ZXC_FILE_HEADER_SIZE;
    ctx->digest = 0;
    ctx->ahead_len = -1;

    e->read_idx = zxc_stream_read_loop(ctx, f_in, e->read_idx);
    zxc_stream_engine_drain(e);
    if (!ctx->io_error)
        zxc_stream_finish_decompress(ctx, (uint64_t)(e->w.total_bytes - base), f_in);

    // fail_code holds the cause, else a read/write failed; a worker may still set it.
    if (UNLIKELY(ctx->io_error)) {
        pthread_mutex_lock(&ctx->lock);
        const int code = ctx->fail_code;
        pthread_mutex_unlock(&ctx->lock);
        return code ? code : ZXC_ERROR_IO;
    }
    return e->w.total_bytes - base;
}

/** @brief The size of @p f; its position goes to @p saved, for the caller to restore. */
static int zxc_file_span(FILE* f, long long* saved, uint64_t* size) {
    *saved = ftello(f);
    if (UNLIKELY(*saved < 0 || fseeko(f, 0, SEEK_END) != 0)) return ZXC_ERROR_IO;
    const long long n = ftello(f);
    if (UNLIKELY(n < 0)) {
        // LCOV_EXCL_START
        fseeko(f, *saved, SEEK_SET);
        return ZXC_ERROR_IO;
        // LCOV_EXCL_STOP
    }
    *size = (uint64_t)n;
    return ZXC_OK;
}

/**
 * @brief Compresses a @c FILE* stream to another @c FILE* stream.
 *
 * Public API; full contract in @c zxc_stream.h. Resolves the options (threads,
 * level, block size, checksums, seekable, dictionary) with their defaults, then
 * runs the engine over the input.
 */
int64_t zxc_stream_compress(FILE* f_in, FILE* f_out, const zxc_compress_opts_t* opts) {
    if (UNLIKELY(!f_in)) return ZXC_ERROR_NULL_INPUT;

    const int n_threads = opts ? opts->n_threads : 0;
    const int checksum_enabled = opts ? opts->checksum_enabled : 0;
    const int seekable = opts ? opts->seekable : 0;
    const int level = ZXC_OPTS_LEVEL(opts, ZXC_LEVEL_DEFAULT);
    const size_t block_size = ZXC_OPTS_BLOCK_SIZE(opts, ZXC_BLOCK_SIZE_DEFAULT);
    const uint8_t* dict = opts ? (const uint8_t*)opts->dict : NULL;
    const size_t dict_size = ZXC_OPTS_DICT_SIZE(opts);
    zxc_progress_callback_t cb = opts ? opts->progress_cb : NULL;
    void* ud = opts ? opts->user_data : NULL;

    if (UNLIKELY(!zxc_validate_block_size(block_size))) return ZXC_ERROR_BAD_BLOCK_SIZE;

    if (UNLIKELY(dict_size > ZXC_DICT_SIZE_MAX)) return ZXC_ERROR_DICT_TOO_LARGE;

    const uint8_t* dict_huf = ZXC_OPTS_DICT_HUF(opts);
    zxc_stream_engine_t e;
    ZXC_MEMSET(&e, 0, sizeof(e));
    zxc_stream_ctx_t* const ctx = &e.ctx;
    ctx->compression_mode = 1;
    ctx->compression_level = level;
    ctx->chunk_size = block_size;
    ctx->checksum_enabled = checksum_enabled;
    ctx->file_has_checksum = checksum_enabled;
    ctx->progress_cb = cb;
    ctx->progress_user_data = ud;
    ctx->dict = dict;
    ctx->dict_size = dict_size;
    ctx->dict_huf = dict_huf;

    // Input size for progress tracking; for decompression the CLI passes it.
    long long saved_pos = 0;
    uint64_t in_size = 0;
    if (cb && zxc_file_span(f_in, &saved_pos, &in_size) == ZXC_OK) {
        ctx->total_input_bytes = in_size;   // LCOV_EXCL_LINE
        fseeko(f_in, saved_pos, SEEK_SET);  // LCOV_EXCL_LINE
    }

    // Seekable: initial block-size tracking array
    if (seekable) {
        e.w.seek_cap = 64;
        e.w.seek_comp = (uint32_t*)ZXC_MALLOC(e.w.seek_cap * sizeof(uint32_t));
        if (UNLIKELY(!e.w.seek_comp)) return ZXC_ERROR_MEMORY;  // LCOV_EXCL_LINE
    }
    e.w.f = f_out;
    const int src = zxc_stream_engine_start(&e, f_in, n_threads);
    if (UNLIKELY(src != ZXC_OK)) {
        ZXC_FREE(e.w.seek_comp);  // LCOV_EXCL_LINE
        return src;               // LCOV_EXCL_LINE
    }

    // The writer waits for a job: the header goes first.
    if (f_out) {
        uint8_t h[ZXC_FILE_HEADER_SIZE];
        zxc_write_file_header(h, ZXC_FILE_HEADER_SIZE, block_size, checksum_enabled,
                              zxc_dict_id(dict, dict_size, dict_huf), seekable);
        if (UNLIKELY(fwrite(h, 1, ZXC_FILE_HEADER_SIZE, f_out) != ZXC_FILE_HEADER_SIZE))
            ctx->io_error = 1;
    }
    e.w.total_bytes = ZXC_FILE_HEADER_SIZE;

    e.read_idx = zxc_stream_read_loop(ctx, f_in, 0);
    zxc_stream_engine_stop(&e, 1);

    if (!ctx->io_error && e.w.total_bytes >= 0) zxc_stream_finish_compress(ctx, &e.w, f_out);
    // stdio defers write errors to flush time: push the last buffer to the OS
    // so a failure (ENOSPC, EFBIG, EPIPE) is reported here instead of being
    // left to the caller's fclose, after a success count was returned.
    if (UNLIKELY(!ctx->io_error && f_out && fflush(f_out) != 0)) ctx->io_error = 1;
    ZXC_FREE(e.w.seek_comp);

    if (UNLIKELY(ctx->io_error)) return ctx->fail_code ? ctx->fail_code : ZXC_ERROR_IO;
    return e.w.total_bytes;
}

/**
 * @brief Decompresses a @c FILE* stream to another @c FILE* stream.
 *
 * Public API; see @c zxc_stream.h. Before each frame, 4 bytes: the end of the
 * input, a magic word or an error. Block size and level come from each frame
 * header, not from @p opts. Consecutive frames with the same block size and
 * checksum flag share one engine.
 */
int64_t zxc_stream_decompress(FILE* f_in, FILE* f_out, const zxc_decompress_opts_t* opts) {
    if (UNLIKELY(!f_in)) return ZXC_ERROR_NULL_INPUT;

    const int n_threads = opts ? opts->n_threads : 0;
    const int checksum_enabled = opts ? opts->checksum_enabled : 0;
    const uint8_t* dict = opts ? (const uint8_t*)opts->dict : NULL;
    const size_t dict_size = ZXC_OPTS_DICT_SIZE(opts);
    const uint8_t* dict_huf = ZXC_OPTS_DICT_HUF(opts);
    if (UNLIKELY(dict_size > ZXC_DICT_SIZE_MAX)) return ZXC_ERROR_DICT_TOO_LARGE;
    const uint32_t dict_id = zxc_dict_id(dict, dict_size, dict_huf);

    zxc_stream_engine_t e;
    int running = 0;
    uint8_t magic[sizeof(uint32_t)];
    int rc = zxc_stream_read_exact(NULL, f_in, magic, sizeof(magic), ZXC_ERROR_SRC_TOO_SMALL);

    int64_t total = 0;
    for (int frames = 0;; frames++) {
        if (rc == 1) {
            // A clean end of input, after at least one frame.
            rc = frames ? ZXC_OK : ZXC_ERROR_SRC_TOO_SMALL;
            break;
        }
        if (UNLIKELY(rc != ZXC_OK)) break;
        if (UNLIKELY(zxc_le32(magic) != ZXC_MAGIC_WORD)) {
            rc = frames ? ZXC_ERROR_CORRUPT_DATA : ZXC_ERROR_BAD_MAGIC;
            break;
        }
        size_t chunk = 0;
        int has_checksum = 0;
        int has_seek = 0;
        rc = zxc_stream_read_frame_header(running ? &e.ctx : NULL, f_in, magic, dict_id, &chunk,
                                          &has_checksum, &has_seek);
        if (UNLIKELY(rc != ZXC_OK)) break;

        // The workers are set up for one block size and checksum flag.
        if (running && (e.ctx.chunk_size != chunk || e.ctx.file_has_checksum != has_checksum)) {
            running = 0;
            rc = zxc_stream_engine_close(&e, f_in);
            if (UNLIKELY(rc != ZXC_OK)) break;  // LCOV_EXCL_LINE
        }
        if (!running) {
            ZXC_MEMSET(&e, 0, sizeof(e));
            e.ctx.chunk_size = chunk;
            e.ctx.checksum_enabled = checksum_enabled;
            e.ctx.file_has_checksum = has_checksum;
            e.ctx.progress_cb = opts ? opts->progress_cb : NULL;
            e.ctx.progress_user_data = opts ? opts->user_data : NULL;
            e.ctx.dict = dict;
            e.ctx.dict_size = dict_size;
            e.ctx.dict_huf = dict_huf;
            e.w.f = f_out;
            e.w.bytes_processed = (uint64_t)total;  // Progress counts across frames
            rc = zxc_stream_engine_start(&e, f_in, n_threads);
            if (UNLIKELY(rc != ZXC_OK)) break;  // LCOV_EXCL_LINE
            running = 1;
        }

        const int64_t r = zxc_stream_decode_frame(&e, f_in, has_seek);
        if (UNLIKELY(r < 0)) {
            rc = (int)r;
            break;
        }
        total += r;
        rc = zxc_stream_read_exact(&e.ctx, f_in, magic, sizeof(magic), ZXC_ERROR_CORRUPT_DATA);
    }
    if (running) {
        const int crc = zxc_stream_engine_close(&e, f_in);
        if (rc == ZXC_OK) rc = crc;
    }
    // stdio defers write errors to flush time (see zxc_stream_compress).
    if (rc == ZXC_OK && f_out && UNLIKELY(fflush(f_out) != 0)) rc = ZXC_ERROR_IO;
    return rc != ZXC_OK ? rc : total;
}

/** @brief @ref zxc_scan_src_t over a @c FILE*. */
static int zxc_file_scan_read(const zxc_scan_src_t* src, const uint64_t off, void* dst,
                              const size_t len) {
    FILE* const f = (FILE*)src->ctx;
    if (UNLIKELY(fseeko(f, (long long)off, SEEK_SET) != 0 || fread(dst, 1, len, f) != len))
        return ZXC_ERROR_IO;  // LCOV_EXCL_LINE
    return ZXC_OK;
}

/**
 * @brief Reads the header and the end of the file in @p f_in, from offset 0,
 *        and validates the frame they describe; restores the stream position.
 */
static int zxc_stream_read_frame_info(FILE* f_in, zxc_frame_info_t* info) {
    long long saved = 0;
    uint64_t size = 0;
    int rc = zxc_file_span(f_in, &saved, &size);
    if (UNLIKELY(rc != ZXC_OK)) return rc;
    const zxc_scan_src_t src = {zxc_file_scan_read, NULL, f_in, size};
    rc = zxc_scan_whole_frame(&src, info, NULL);
    fseeko(f_in, saved, SEEK_SET);
    return rc;
}

/**
 * @brief Reads the total decompressed size from the footers of a file.
 *
 * Public API; see @c zxc_stream.h. @ref zxc_scan_container over the whole file,
 * as @ref zxc_get_decompressed_size; the stream position is restored.
 */
int64_t zxc_stream_get_decompressed_size(FILE* f_in) {
    if (UNLIKELY(!f_in)) return ZXC_ERROR_NULL_INPUT;
    long long saved = 0;
    uint64_t size = 0;
    int rc = zxc_file_span(f_in, &saved, &size);
    if (UNLIKELY(rc != ZXC_OK)) return rc;
    zxc_container_info_t info;
    const zxc_scan_src_t src = {zxc_file_scan_read, NULL, f_in, size};
    rc = zxc_scan_container(&src, 0, &info);
    fseeko(f_in, saved, SEEK_SET);
    if (UNLIKELY(rc != ZXC_OK)) return rc;
    if (UNLIKELY(info.dsize > (uint64_t)INT64_MAX)) return ZXC_ERROR_CORRUPT_DATA;
    return (int64_t)info.dsize;
}

/**
 * @brief Reads a file's frame header and footer, without decoding.
 *
 * Public API; see @c zxc_stream.h.
 */
// cppcheck-suppress unusedFunction
int zxc_stream_get_frame_info(FILE* f_in, zxc_frame_info_t* info, const size_t info_size) {
    if (UNLIKELY(!f_in || !info)) return ZXC_ERROR_NULL_INPUT;
    zxc_frame_info_t got;
    const int rc = zxc_stream_read_frame_info(f_in, &got);
    if (rc == ZXC_OK) zxc_frame_info_copy(info, info_size, &got);
    return rc;
}

/**
 * @brief Reads the frame that ends at offset @p end of a file, without decoding.
 *
 * Public API; see @c zxc_stream.h.
 */
// cppcheck-suppress unusedFunction
int zxc_stream_get_last_frame_info(FILE* f_in, const uint64_t end, zxc_frame_info_t* info,
                                   const size_t info_size) {
    if (UNLIKELY(!f_in || !info)) return ZXC_ERROR_NULL_INPUT;
    long long saved = 0;
    uint64_t size = 0;
    int rc = zxc_file_span(f_in, &saved, &size);
    if (UNLIKELY(rc != ZXC_OK)) return rc;
    zxc_frame_info_t got;
    if (UNLIKELY(end > size)) {
        rc = ZXC_ERROR_SRC_TOO_SMALL;
    } else {
        const zxc_scan_src_t src = {zxc_file_scan_read, NULL, f_in, end};
        uint64_t start = 0;
        rc = zxc_scan_frame(&src, end, &start, &got, NULL);
    }
    fseeko(f_in, saved, SEEK_SET);
    if (rc == ZXC_OK) zxc_frame_info_copy(info, info_size, &got);
    return rc;
}

// ============================================================================
// SEEKABLE FILE* WRAPPER
// ============================================================================
// Adapts a FILE* into a thread-safe zxc_reader_t (pread, or ReadFile +
// OVERLAPPED on Windows). It lives here rather than in zxc_seekable.c so that
// file stays freestanding.

#if defined(_WIN32)
/** @brief Reader context for the Win32 @c FILE* adapter (OS file handle + size). */
typedef struct {
    HANDLE handle; /* OS handle from _get_osfhandle(_fileno(f)) */
    uint64_t size; /* total file size in bytes */
} zxc_stdio_ctx_t;

/**
 * @brief Thread-safe positioned read backing the seekable @c FILE* reader.
 *
 * Win32 implementation: a positioned @c ReadFile via @c OVERLAPPED, so
 * concurrent worker threads never race on a shared file cursor.
 *
 * @param[in]  vctx    `zxc_stdio_ctx_t` carrying the file handle.
 * @param[out] dst     Destination buffer (at least @p len bytes).
 * @param[in]  len     Number of bytes to read.
 * @param[in]  offset  Absolute byte offset to read from.
 * @return @p len on a full read, otherwise @ref ZXC_ERROR_IO.
 */
// LCOV_EXCL_START - Windows I/O path, not reachable on POSIX CI
static int64_t zxc_stdio_read_at(void* vctx, void* dst, size_t len, uint64_t offset) {
    zxc_stdio_ctx_t* const ctx = (zxc_stdio_ctx_t*)vctx;
    OVERLAPPED ov;
    ZXC_MEMSET(&ov, 0, sizeof(ov));
    ov.Offset = (DWORD)(offset & 0xFFFFFFFFU);
    ov.OffsetHigh = (DWORD)(offset >> 32);
    DWORD bytes_read = 0;
    if (!ReadFile(ctx->handle, dst, (DWORD)len, &bytes_read, &ov)) return ZXC_ERROR_IO;
    return (bytes_read == (DWORD)len) ? (int64_t)len : ZXC_ERROR_IO;
}
// LCOV_EXCL_STOP

#else  /* POSIX */
/** @brief Reader context for the POSIX @c FILE* adapter (file descriptor + size). */
typedef struct {
    int fd;        /* descriptor from fileno(f) */
    uint64_t size; /* total file size in bytes */
} zxc_stdio_ctx_t;

/**
 * @brief Thread-safe positioned read backing the seekable @c FILE* reader.
 *
 * POSIX implementation: @ref zxc_pread_full, which carries its own offset and
 * so is safe to call concurrently from multiple worker threads on one descriptor.
 *
 * @param[in]  vctx    `zxc_stdio_ctx_t` carrying the file descriptor.
 * @param[out] dst     Destination buffer (at least @p len bytes).
 * @param[in]  len     Number of bytes to read.
 * @param[in]  offset  Absolute byte offset to read from.
 * @return @p len on a full read, otherwise @ref ZXC_ERROR_IO.
 */
// cppcheck-suppress constParameterCallback ; the reader callback type takes void*
static int64_t zxc_stdio_read_at(void* vctx, void* dst, size_t len, uint64_t offset) {
    const zxc_stdio_ctx_t* const ctx = (const zxc_stdio_ctx_t*)vctx;
    return zxc_pread_full(ctx->fd, dst, len, offset) == (int64_t)len ? (int64_t)len : ZXC_ERROR_IO;
}
#endif /* _WIN32 */

/**
 * @brief Opens a seekable archive backed by an open @c FILE*.
 *
 * Public API; full contract in @c zxc_stream.h. Snapshots and restores the
 * file position, measures the file, wraps it in a thread-safe positioned
 * reader (@c pread on POSIX, @c ReadFile + @c OVERLAPPED on Windows), and
 * delegates to @ref zxc_seekable_open_reader. The reader context is heap-owned
 * and handed to the returned handle via @ref zxc_seekable_attach_owned_ctx, so
 * @ref zxc_seekable_free releases it.
 */
zxc_seekable* zxc_seekable_open_file(FILE* f) {
    if (UNLIKELY(!f)) return NULL;

    // Snapshot the caller's file position so we can restore it.
    const long long saved_pos = ftello(f);
    if (UNLIKELY(saved_pos < 0)) return NULL;  // LCOV_EXCL_LINE

    // LCOV_EXCL_START - ftello/fseeko failure paths not reachable in CI
    if (UNLIKELY(fseeko(f, 0, SEEK_END) != 0)) return NULL;
    const long long file_size = ftello(f);
    (void)fseeko(f, saved_pos, SEEK_SET);
    if (UNLIKELY(file_size <= 0)) return NULL;
    // LCOV_EXCL_STOP

    zxc_stdio_ctx_t* const ctx = (zxc_stdio_ctx_t*)ZXC_MALLOC(sizeof(*ctx));
    if (UNLIKELY(!ctx)) return NULL;  // LCOV_EXCL_LINE

#if defined(_WIN32)
    ctx->handle = (HANDLE)(intptr_t)_get_osfhandle(_fileno(f));  // LCOV_EXCL_LINE
#else
    ctx->fd = fileno(f);
#endif
    ctx->size = (uint64_t)file_size;

    const zxc_reader_t reader = {
        .read_at = zxc_stdio_read_at, .ctx = ctx, .size = (uint64_t)file_size};

    zxc_seekable* const s = zxc_seekable_open_reader(&reader);
    if (UNLIKELY(!s)) {
        ZXC_FREE(ctx);
        return NULL;
    }

    // Hand the ctx lifetime over to the seekable handle.
    zxc_seekable_attach_owned_ctx(s, ctx);
    return s;
}

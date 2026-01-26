/*
 * Copyright (c) 2025-2026, Bertrand Lebonnois
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "../../include/zxc_buffer.h"
#include "../../include/zxc_sans_io.h"
#include "../../include/zxc_stream.h"
#include "zxc_internal.h"

/*
 * ============================================================================
 * WINDOWS THREADING EMULATION
 * ============================================================================
 * Maps POSIX pthread calls to Windows Native API (CriticalSection,
 * ConditionVariable, Threads). Allows the same threading logic to compile on
 * Linux/macOS and Windows.
 */
#if defined(_WIN32)
#include <malloc.h>
#include <process.h>
#include <sys/types.h>
#include <windows.h>

// Map POSIX file positioning functions to Windows equivalents
#define fseeko _fseeki64
#define ftello _ftelli64

// Simple sysconf emulation to get core count
static int zxc_get_num_procs(void) {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
}

typedef CRITICAL_SECTION pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
typedef HANDLE pthread_t;

#define pthread_mutex_init(m, a) InitializeCriticalSection(m)
#define pthread_mutex_destroy(m) DeleteCriticalSection(m)
#define pthread_mutex_lock(m) EnterCriticalSection(m)
#define pthread_mutex_unlock(m) LeaveCriticalSection(m)

#define pthread_cond_init(c, a) InitializeConditionVariable(c)
#define pthread_cond_destroy(c) (void)(0)
#define pthread_cond_wait(c, m) SleepConditionVariableCS(c, m, INFINITE)
#define pthread_cond_signal(c) WakeConditionVariable(c)
#define pthread_cond_broadcast(c) WakeAllConditionVariable(c)

typedef struct {
    void* (*func)(void*);
    void* arg;
} zxc_win_thread_arg_t;

static unsigned __stdcall zxc_win_thread_entry(void* p) {
    zxc_win_thread_arg_t* a = (zxc_win_thread_arg_t*)p;
    void* (*f)(void*) = a->func;
    void* arg = a->arg;
    free(a);
    f(arg);
    return 0;
}

static int pthread_create(pthread_t* thread, const void* attr, void* (*start_routine)(void*),
                          void* arg) {
    zxc_win_thread_arg_t* wrapper = malloc(sizeof(zxc_win_thread_arg_t));
    if (UNLIKELY(!wrapper)) return -1;
    wrapper->func = start_routine;
    wrapper->arg = arg;
    uintptr_t handle = _beginthreadex(NULL, 0, zxc_win_thread_entry, wrapper, 0, NULL);
    if (UNLIKELY(handle == 0)) {
        free(wrapper);
        return -1;
    }
    *thread = (HANDLE)handle;
    return 0;
}

static int pthread_join(pthread_t thread, void** retval) {
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return 0;
}

#define sysconf(x) zxc_get_num_procs()
#define _SC_NPROCESSORS_ONLN 0

#else
#include <pthread.h>
#include <unistd.h>
#endif

/*
 * ============================================================================
 * STREAMING ENGINE (Producer / Worker / Consumer)
 * ============================================================================
 * Implements a Ring Buffer architecture to parallelize block processing.
 */

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
 *      Pointer to the buffer containing raw input data.
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
 * @var zxc_stream_job_t::job_id
 *      A unique identifier for the job, often used for ordering or debugging.
 * @var zxc_stream_job_t::status
 *      The current state of this job (Free, Filled, or Processed).
 * @var zxc_stream_job_t::pad
 *      Padding bytes to ensure the structure size aligns with typical cache
 * lines (64 bytes), minimizing cache contention between threads accessing
 * adjacent jobs.
 */
typedef struct {
    uint8_t* in_buf;
    size_t in_cap, in_sz;
    uint8_t* out_buf;
    size_t out_cap, result_sz;
    int job_id;
    ZXC_ATOMIC job_status_t status;  // Atomic for lock-free status updates
    char pad[ZXC_CACHE_LINE_SIZE];   // Prevent False Sharing
} zxc_stream_job_t;

/**
 * @typedef zxc_chunk_processor_t
 * @brief Function pointer type for processing a chunk of data.
 *
 * This type defines the signature for internal functions responsible for
 * processing (compressing or transforming) a specific chunk of input data.
 *
 * @param ctx     Pointer to the compression context containing state and
 * configuration.
 * @param in      Pointer to the input data buffer.
 * @param in_sz   Size of the input data in bytes.
 * @param out     Pointer to the output buffer where processed data will be
 * written.
 * @param out_cap Capacity of the output buffer in bytes.
 *
 * @return The number of bytes written to the output buffer on success, or a
 * negative error code on failure.
 */
typedef int (*zxc_chunk_processor_t)(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT in,
                                     const size_t in_sz, uint8_t* RESTRICT out,
                                     const size_t out_cap);

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
 *      Atomic flag to signal if an I/O error occurred during processing.
 * @var zxc_stream_ctx_t::processor
 *      Function pointer or object responsible for the actual chunk processing
 * logic.
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
 * @var zxc_stream_ctx_t::progress_cb
 *     Optional callback function for reporting progress during processing.
 * @var zxc_stream_ctx_t::progress_user_data
 *    User data pointer to be passed to the progress callback function.
 * @var zxc_stream_ctx_t::total_input_bytes
 *     Total size of the input data in bytes, used for progress tracking.
 */
typedef struct {
    zxc_stream_job_t* jobs;
    int ring_size;
    int* worker_queue;
    int wq_head, wq_tail, wq_count;
    pthread_mutex_t lock;
    pthread_cond_t cond_reader, cond_worker, cond_writer;
    int shutdown_workers;
    int compression_mode;
    ZXC_ATOMIC int io_error;
    zxc_chunk_processor_t processor;
    int write_idx;
    int compression_level;
    size_t chunk_size;
    int checksum_enabled;
    int file_has_checksum;
    zxc_progress_callback_t progress_cb;
    void* progress_user_data;
    uint64_t total_input_bytes;
} zxc_stream_ctx_t;

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
 * @var writer_args_t::global_hash
 * The global hash accumulated during processing.
 *
 * @var writer_args_t::bytes_processed
 * The number of bytes processed so far, used for progress reporting.
 */
typedef struct {
    zxc_stream_ctx_t* ctx;
    FILE* f;
    int64_t total_bytes;
    uint32_t global_hash;
    uint64_t bytes_processed;  // For progress callback
} writer_args_t;

/**
 * @brief Worker thread function for parallel stream processing.
 *
 * This function serves as the entry point for worker threads in the ZXC
 * streaming compression/decompression context. It continuously retrieves jobs
 * from a shared work queue, processes them using a thread-local compression
 * context (`zxc_cctx_t`), and signals the writer thread upon completion.
 *
 * **Worker Lifecycle & Synchronization:**
 * 1. **Initialization:** Allocates a thread-local `zxc_cctx_t` to avoid lock
 * contention during compression/decompression.
 * 2. **Wait Loop:** Uses `pthread_cond_wait` on `cond_worker` to sleep until a
 * job is available in the `worker_queue`.
 * 3. **Job Retrieval:** Dequeues a job ID from the ring buffer. The
 * `worker_queue` acts as a load balancer.
 * 4. **Processing:** Calls `ctx->processor` (the compression/decompression
 * function) on the job's data. This is the CPU-intensive part and runs in
 * parallel.
 * 5. **Completion:** Updates `job->status` to `JOB_STATUS_PROCESSED`.
 * 6. **Signaling:** If the processed job is the *next* one expected by the
 * writer
 *    (`jid == ctx->write_idx`), it signals `cond_writer`. This optimization
 * prevents unnecessary wake-ups of the writer thread for out-of-order
 * completions.
 *
 * @param[in] arg A pointer to the shared stream context (`zxc_stream_ctx_t`).
 * @return Always returns NULL.
 */
static void* zxc_stream_worker(void* arg) {
    zxc_stream_ctx_t* ctx = (zxc_stream_ctx_t*)arg;
    zxc_cctx_t cctx;

    int unified_chk = (ctx->compression_mode == 1)
                          ? ctx->checksum_enabled
                          : (ctx->file_has_checksum && ctx->checksum_enabled);

    if (zxc_cctx_init(&cctx, ctx->chunk_size, ctx->compression_mode, ctx->compression_level,
                      unified_chk) != 0) {
        zxc_cctx_free(&cctx);
        return NULL;
    }

    cctx.compression_level = ctx->compression_level;

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
        ctx->wq_tail = (ctx->wq_tail + 1) % ctx->ring_size;
        ctx->wq_count--;
        job = &ctx->jobs[jid];
        pthread_mutex_unlock(&ctx->lock);

        const int res = ctx->processor(&cctx, job->in_buf, job->in_sz, job->out_buf, job->out_cap);
        job->result_sz = UNLIKELY(res < 0) ? 0 : (size_t)res;
        job->status = JOB_STATUS_PROCESSED;

        pthread_mutex_lock(&ctx->lock);
        if (UNLIKELY(res < 0)) {
            ctx->io_error = 1;
            pthread_cond_broadcast(&ctx->cond_writer);
            pthread_cond_broadcast(&ctx->cond_reader);
        } else if (jid == ctx->write_idx) {
            pthread_cond_signal(&ctx->cond_writer);
        }
        pthread_mutex_unlock(&ctx->lock);
    }
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
    writer_args_t* args = (writer_args_t*)arg;
    zxc_stream_ctx_t* ctx = args->ctx;
    while (1) {
        zxc_stream_job_t* job = &ctx->jobs[ctx->write_idx];
        pthread_mutex_lock(&ctx->lock);
        while (job->status != JOB_STATUS_PROCESSED)
            pthread_cond_wait(&ctx->cond_writer, &ctx->lock);

        if (job->result_sz == (size_t)-1) {
            pthread_mutex_unlock(&ctx->lock);
            break;
        }
        pthread_mutex_unlock(&ctx->lock);

        if (args->f && job->result_sz > 0) {
            if (fwrite(job->out_buf, 1, job->result_sz, args->f) != job->result_sz) {
                ctx->io_error = 1;
            } else if (ctx->checksum_enabled && ctx->compression_mode == 1) {
                // Update Global Hash (Rotation + XOR)
                if (LIKELY(job->result_sz >= ZXC_GLOBAL_CHECKSUM_SIZE)) {
                    uint32_t block_hash =
                        zxc_le32(job->out_buf + job->result_sz - ZXC_GLOBAL_CHECKSUM_SIZE);
                    args->global_hash = zxc_hash_combine_rotate(args->global_hash, block_hash);
                }
            }
        }
        if (UNLIKELY(ctx->io_error)) {
            pthread_mutex_lock(&ctx->lock);
            job->status = JOB_STATUS_FREE;
            pthread_cond_signal(&ctx->cond_reader);
            pthread_mutex_unlock(&ctx->lock);
            break;
        }
        args->total_bytes += (int64_t)job->result_sz;

        // Update progress callback
        if (ctx->progress_cb) {
            args->bytes_processed += ctx->compression_mode == 1 ? job->in_sz : job->result_sz;
            ctx->progress_cb(args->bytes_processed, ctx->total_input_bytes,
                             ctx->progress_user_data);
        }

        pthread_mutex_lock(&ctx->lock);
        job->status = JOB_STATUS_FREE;
        ctx->write_idx = (ctx->write_idx + 1) % ctx->ring_size;
        pthread_cond_signal(&ctx->cond_reader);
        pthread_mutex_unlock(&ctx->lock);
    }
    return NULL;
}

/**
 * @brief Orchestrates the multithreaded streaming compression or decompression
 * engine.
 *
 * This function initializes the stream context, allocates the necessary ring
 * buffer memory for jobs and I/O buffers, and spawns the worker threads and the
 * asynchronous writer thread. It acts as the main "producer" (reader) loop.
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
 * @param[in] f_in      Pointer to the input file stream (source).
 * @param[out] f_out     Pointer to the output file stream (destination).
 * @param[in] n_threads Number of worker threads to spawn. If set to 0 or less, the
 * function automatically detects the number of online processors.
 * @param[in] mode      Operation mode: 1 for compression, 0 for decompression.
 * @param[in] level     Compression level to be applied (relevant for compression
 * mode).
 * @param[in] checksum_enabled  Flag indicating whether to enable checksum
 * generation/verification.
 * @param[in] func      Function pointer to the chunk processor (compression or
 * decompression logic).
 *
 * @return The total number of bytes written to the output stream on success, or
 * -1 if an initialization or I/O error occurred.
 */
static int64_t zxc_stream_engine_run(FILE* f_in, FILE* f_out, const int n_threads, const int mode,
                                     const int level, const int checksum_enabled,
                                     zxc_chunk_processor_t func,
                                     zxc_progress_callback_t progress_cb, void* user_data) {
    zxc_stream_ctx_t ctx;
    ZXC_MEMSET(&ctx, 0, sizeof(ctx));

    size_t runtime_chunk_sz = ZXC_BLOCK_SIZE;
    int file_has_chk = 0;

    // Try to get input file size for progress tracking (compression mode only)
    // For decompression, the CLI precomputes the size and passes it via user_data
    uint64_t total_file_size = 0;
    if (mode == 1 && progress_cb) {
        long long saved_pos = ftello(f_in);
        if (saved_pos >= 0) {
            if (fseeko(f_in, 0, SEEK_END) == 0) {
                long long size = ftello(f_in);
                if (size > 0) total_file_size = (uint64_t)size;
                fseeko(f_in, saved_pos, SEEK_SET);
            }
        }
    }

    if (mode == 0) {
        // Decompression Mode: Read and validate file header
        uint8_t h[ZXC_FILE_HEADER_SIZE];
        if (UNLIKELY(fread(h, 1, ZXC_FILE_HEADER_SIZE, f_in) != ZXC_FILE_HEADER_SIZE ||
                     zxc_read_file_header(h, ZXC_FILE_HEADER_SIZE, &runtime_chunk_sz,
                                          &file_has_chk) != 0))
            return -1;
    }

    const int num_threads = (n_threads > 0) ? n_threads : (int)sysconf(_SC_NPROCESSORS_ONLN);
    // Reserve 1 thread for Writer/Reader overhead if possible
    const int num_workers = (num_threads > 1) ? num_threads - 1 : 1;

    ctx.compression_mode = mode;
    ctx.processor = func;
    ctx.io_error = 0;
    ctx.compression_level = level;
    ctx.ring_size = num_workers * 4;
    ctx.chunk_size = runtime_chunk_sz;
    ctx.checksum_enabled = checksum_enabled;
    ctx.file_has_checksum = mode == 1 ? checksum_enabled : file_has_chk;
    ctx.progress_cb = progress_cb;
    ctx.progress_user_data = user_data;
    ctx.total_input_bytes = total_file_size;

    uint32_t d_global_hash = 0;

    size_t max_out = zxc_compress_bound(runtime_chunk_sz);
    size_t raw_alloc_in = ((mode) ? runtime_chunk_sz : max_out) + ZXC_PAD_SIZE;
    size_t alloc_in = (raw_alloc_in + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;

    size_t raw_alloc_out = ((mode) ? max_out : runtime_chunk_sz) + ZXC_PAD_SIZE;
    size_t alloc_out = (raw_alloc_out + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;

    size_t alloc_size =
        ctx.ring_size * (sizeof(zxc_stream_job_t) + sizeof(int) + alloc_in + alloc_out);
    uint8_t* mem_block = zxc_aligned_malloc(alloc_size, ZXC_CACHE_LINE_SIZE);
    if (UNLIKELY(!mem_block)) return -1;

    uint8_t* ptr = mem_block;
    ctx.jobs = (zxc_stream_job_t*)ptr;
    ptr += ctx.ring_size * sizeof(zxc_stream_job_t);
    ctx.worker_queue = (int*)ptr;
    ptr += ctx.ring_size * sizeof(int);
    uint8_t* buf_in = ptr;
    ptr += ctx.ring_size * alloc_in;
    uint8_t* buf_out = ptr;

    ZXC_MEMSET(buf_in, 0, ctx.ring_size * alloc_in);

    for (int i = 0; i < ctx.ring_size; i++) {
        ctx.jobs[i].job_id = i;
        ctx.jobs[i].status = JOB_STATUS_FREE;
        ctx.jobs[i].in_buf = buf_in + (i * alloc_in);
        ctx.jobs[i].in_cap = alloc_in - ZXC_PAD_SIZE;
        ctx.jobs[i].in_sz = 0;
        ctx.jobs[i].out_buf = buf_out + (i * alloc_out);
        ctx.jobs[i].out_cap = alloc_out - ZXC_PAD_SIZE;
        ctx.jobs[i].result_sz = 0;
    }

    pthread_mutex_init(&ctx.lock, NULL);
    pthread_cond_init(&ctx.cond_reader, NULL);
    pthread_cond_init(&ctx.cond_worker, NULL);
    pthread_cond_init(&ctx.cond_writer, NULL);

    pthread_t* workers = malloc(num_workers * sizeof(pthread_t));
    if (UNLIKELY(!workers)) {
        zxc_aligned_free(mem_block);
        return -1;
    }
    for (int i = 0; i < num_workers; i++)
        pthread_create(&workers[i], NULL, zxc_stream_worker, &ctx);

    writer_args_t w_args = {&ctx, f_out, 0, 0, 0};

    if (mode == 1 && f_out) {
        uint8_t h[ZXC_FILE_HEADER_SIZE];
        zxc_write_file_header(h, ZXC_FILE_HEADER_SIZE, checksum_enabled);
        if (UNLIKELY(fwrite(h, 1, ZXC_FILE_HEADER_SIZE, f_out) != ZXC_FILE_HEADER_SIZE))
            ctx.io_error = 1;

        w_args.total_bytes = ZXC_FILE_HEADER_SIZE;
    }
    pthread_t writer_th;
    pthread_create(&writer_th, NULL, zxc_async_writer, &w_args);

    int read_idx = 0;
    int read_eof = 0;
    uint64_t total_src_bytes = 0;

    // Reader Loop: Reads from file, prepares jobs, pushes to worker queue.
    while (!read_eof && !ctx.io_error) {
        zxc_stream_job_t* job = &ctx.jobs[read_idx];
        pthread_mutex_lock(&ctx.lock);
        while (job->status != JOB_STATUS_FREE) pthread_cond_wait(&ctx.cond_reader, &ctx.lock);
        pthread_mutex_unlock(&ctx.lock);

        if (UNLIKELY(ctx.io_error)) break;

        size_t read_sz = 0;
        if (mode == 1) {
            read_sz = fread(job->in_buf, 1, ZXC_BLOCK_SIZE, f_in);
            total_src_bytes += read_sz;
            if (UNLIKELY(read_sz == 0)) read_eof = 1;
        } else {
            uint8_t bh_buf[ZXC_BLOCK_HEADER_SIZE];
            size_t h_read = fread(bh_buf, 1, ZXC_BLOCK_HEADER_SIZE, f_in);
            if (UNLIKELY(h_read < ZXC_BLOCK_HEADER_SIZE)) {
                read_eof = 1;
            } else {
                zxc_block_header_t bh;
                if (UNLIKELY(zxc_read_block_header(bh_buf, ZXC_BLOCK_HEADER_SIZE, &bh) != 0)) {
                    read_eof = 1;
                    goto _job_prepared;
                }

                if (bh.block_type == ZXC_BLOCK_EOF) {
                    if (UNLIKELY(bh.comp_size != 0)) {
                        ctx.io_error = 1;
                        goto _job_prepared;
                    }
                    read_eof = 1;
                    read_sz = 0;
                    goto _job_prepared;
                }

                const int has_crc = ctx.file_has_checksum;
                const size_t checksum_sz = (has_crc ? ZXC_BLOCK_CHECKSUM_SIZE : 0);
                const size_t body_total = bh.comp_size + checksum_sz;
                const size_t total_len = ZXC_BLOCK_HEADER_SIZE + body_total;

                if (UNLIKELY(total_len > job->in_cap)) {
                    ctx.io_error = 1;
                    break;
                }

                ZXC_MEMCPY(job->in_buf, bh_buf, ZXC_BLOCK_HEADER_SIZE);

                // Single fread for body + checksum (reduces syscalls)
                const size_t body_read =
                    fread(job->in_buf + ZXC_BLOCK_HEADER_SIZE, 1, body_total, f_in);

                if (UNLIKELY(body_read != body_total)) {
                    read_eof = 1;
                } else if (has_crc) {
                    // Update Global Hash for Decompression
                    const uint32_t b_crc =
                        zxc_le32(job->in_buf + ZXC_BLOCK_HEADER_SIZE + bh.comp_size);
                    d_global_hash = zxc_hash_combine_rotate(d_global_hash, b_crc);
                }
                read_sz = ZXC_BLOCK_HEADER_SIZE + body_read;
            }
        }
    _job_prepared:
        if (UNLIKELY(read_eof && read_sz == 0)) break;

        job->in_sz = read_sz;
        pthread_mutex_lock(&ctx.lock);
        job->status = JOB_STATUS_FILLED;
        ctx.worker_queue[ctx.wq_head] = read_idx;
        ctx.wq_head = (ctx.wq_head + 1) % ctx.ring_size;
        ctx.wq_count++;
        read_idx = (read_idx + 1) % ctx.ring_size;
        pthread_cond_signal(&ctx.cond_worker);
        pthread_mutex_unlock(&ctx.lock);

        if (UNLIKELY(read_sz < ZXC_BLOCK_SIZE && mode == 1)) read_eof = 1;
    }

    zxc_stream_job_t* end_job = &ctx.jobs[read_idx];
    pthread_mutex_lock(&ctx.lock);
    while (end_job->status != JOB_STATUS_FREE) pthread_cond_wait(&ctx.cond_reader, &ctx.lock);
    end_job->result_sz = -1;
    end_job->status = JOB_STATUS_PROCESSED;
    pthread_cond_broadcast(&ctx.cond_writer);
    pthread_mutex_unlock(&ctx.lock);

    pthread_join(writer_th, NULL);
    pthread_mutex_lock(&ctx.lock);
    ctx.shutdown_workers = 1;
    pthread_cond_broadcast(&ctx.cond_worker);
    pthread_mutex_unlock(&ctx.lock);
    for (int i = 0; i < num_workers; i++) pthread_join(workers[i], NULL);

    // Write EOF Block if compression and no error
    if (mode == 1 && !ctx.io_error && w_args.total_bytes >= 0) {
        uint8_t final_buf[ZXC_BLOCK_HEADER_SIZE + ZXC_FILE_FOOTER_SIZE];
        uint8_t* eof_buf = final_buf;
        uint8_t* footer = final_buf + ZXC_BLOCK_HEADER_SIZE;

        zxc_block_header_t eof_bh = {
            .block_type = ZXC_BLOCK_EOF, .block_flags = 0, .reserved = 0, .comp_size = 0};
        zxc_write_block_header(eof_buf, ZXC_BLOCK_HEADER_SIZE, &eof_bh);
        zxc_write_file_footer(footer, ZXC_FILE_FOOTER_SIZE, total_src_bytes, w_args.global_hash,
                              checksum_enabled);

        if (UNLIKELY(f_out && fwrite(final_buf, 1, sizeof(final_buf), f_out) != sizeof(final_buf)))
            return -1;

        w_args.total_bytes += sizeof(final_buf);
    } else if (mode == 0 && !ctx.io_error) {
        // Verification: Expect 12-byte footer
        uint8_t footer[ZXC_FILE_FOOTER_SIZE];
        if (UNLIKELY(fread(footer, 1, ZXC_FILE_FOOTER_SIZE, f_in) != ZXC_FILE_FOOTER_SIZE)) {
            ctx.io_error = 1;
        } else {
            // Verify Footer Content: Source Size and Global Checksum
            int valid = (zxc_le64(footer) == (uint64_t)w_args.total_bytes);
            if (valid && checksum_enabled && ctx.file_has_checksum)
                valid = (zxc_le32(footer + sizeof(uint64_t)) == d_global_hash);

            if (UNLIKELY(!valid)) ctx.io_error = 1;
        }
    }

    free(workers);
    zxc_aligned_free(mem_block);

    if (UNLIKELY(ctx.io_error)) return -1;

    return w_args.total_bytes;
}

int64_t zxc_stream_compress_ex(FILE* f_in, FILE* f_out, const int n_threads, const int level,
                               const int checksum_enabled, zxc_progress_callback_t progress_cb,
                               void* user_data) {
    if (UNLIKELY(!f_in)) return -1;

    return zxc_stream_engine_run(f_in, f_out, n_threads, 1, level, checksum_enabled,
                                 zxc_compress_chunk_wrapper, progress_cb, user_data);
}

int64_t zxc_stream_compress(FILE* f_in, FILE* f_out, const int n_threads, const int level,
                            const int checksum_enabled) {
    return zxc_stream_compress_ex(f_in, f_out, n_threads, level, checksum_enabled, NULL, NULL);
}

int64_t zxc_stream_decompress_ex(FILE* f_in, FILE* f_out, const int n_threads,
                                 const int checksum_enabled, zxc_progress_callback_t progress_cb,
                                 void* user_data) {
    if (UNLIKELY(!f_in)) return -1;

    return zxc_stream_engine_run(f_in, f_out, n_threads, 0, 0, checksum_enabled,
                                 (zxc_chunk_processor_t)zxc_decompress_chunk_wrapper, progress_cb,
                                 user_data);
}

int64_t zxc_stream_decompress(FILE* f_in, FILE* f_out, const int n_threads,
                              const int checksum_enabled) {
    return zxc_stream_decompress_ex(f_in, f_out, n_threads, checksum_enabled, NULL, NULL);
}

int64_t zxc_stream_get_decompressed_size(FILE* f_in) {
    if (UNLIKELY(!f_in)) return -1;

    long long saved_pos = ftello(f_in);
    if (UNLIKELY(saved_pos < 0)) return -1;

    // Get file size
    if (fseeko(f_in, 0, SEEK_END) != 0) return -1;
    long long file_size = ftello(f_in);
    if (UNLIKELY(file_size < (long long)(ZXC_FILE_HEADER_SIZE + ZXC_FILE_FOOTER_SIZE))) {
        fseeko(f_in, saved_pos, SEEK_SET);
        return -1;
    }

    uint8_t header[ZXC_FILE_HEADER_SIZE];
    if (UNLIKELY(fseeko(f_in, 0, SEEK_SET) != 0 ||
                 fread(header, 1, ZXC_FILE_HEADER_SIZE, f_in) != ZXC_FILE_HEADER_SIZE)) {
        fseeko(f_in, saved_pos, SEEK_SET);
        return -1;
    }

    if (UNLIKELY(zxc_le32(header) != ZXC_MAGIC_WORD)) {
        fseeko(f_in, saved_pos, SEEK_SET);
        return -1;
    }

    uint8_t footer[ZXC_FILE_FOOTER_SIZE];
    if (UNLIKELY(fseeko(f_in, file_size - ZXC_FILE_FOOTER_SIZE, SEEK_SET) != 0 ||
                 fread(footer, 1, ZXC_FILE_FOOTER_SIZE, f_in) != ZXC_FILE_FOOTER_SIZE)) {
        fseeko(f_in, saved_pos, SEEK_SET);
        return -1;
    }

    fseeko(f_in, saved_pos, SEEK_SET);

    return (int64_t)zxc_le64(footer);
}

/*
Copyright 2011-2025 Frederic Langlet
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <sstream>
#include "CompressedInputStream.hpp"
#include "IOException.hpp"
#include "../Error.hpp"
#include "../entropy/EntropyDecoderFactory.hpp"
#include "../transform/TransformFactory.hpp"

#ifdef CONCURRENCY_ENABLED
#include <future>
#endif

using namespace kanzi;
using namespace std;


const int CompressedInputStream::BITSTREAM_TYPE = 0x4B414E5A; // "KANZ"
const int CompressedInputStream::BITSTREAM_FORMAT_VERSION = 6;
const int CompressedInputStream::DEFAULT_BUFFER_SIZE = 256 * 1024;
const int CompressedInputStream::EXTRA_BUFFER_SIZE = 512;
const byte CompressedInputStream::COPY_BLOCK_MASK = byte(0x80);
const byte CompressedInputStream::TRANSFORMS_MASK = byte(0x10);
const int CompressedInputStream::MIN_BITSTREAM_BLOCK_SIZE = 1024;
const int CompressedInputStream::MAX_BITSTREAM_BLOCK_SIZE = 1024 * 1024 * 1024;
const int CompressedInputStream::CANCEL_TASKS_ID = -1;
const int CompressedInputStream::MAX_CONCURRENCY = 64;
const int CompressedInputStream::MAX_BLOCK_ID = int((uint(1) << 31) - 1);


CompressedInputStream::CompressedInputStream(InputStream& is,
                   int tasks,
                   const string& entropy,
                   const string& transform,
                   int blockSize,
                   int checksum,
                   uint64 originalSize,
#ifdef CONCURRENCY_ENABLED
                   ThreadPool* pool,
#endif
                   bool headerless,
                   int bsVersion)
    : InputStream(is.rdbuf())
    , _parentCtx(nullptr)
{
#ifdef CONCURRENCY_ENABLED
    if ((tasks <= 0) || (tasks > MAX_CONCURRENCY)) {
        stringstream ss;
        ss << "The number of jobs must be in [1.." << MAX_CONCURRENCY << "], got " << tasks;
        throw invalid_argument(ss.str());
    }

    _pool = pool; // may be null
#else
    if (tasks != 1)
        throw invalid_argument("The number of jobs is limited to 1 in this version");
#endif

    _hasher32 = nullptr;
    _hasher64 = nullptr;
    _blockId = 0;
    _bufferId = 0;
    _maxBufferId = 0;
    _blockSize = blockSize;
    _bufferThreshold = 0;
    _available = 0;
    _entropyType = EntropyDecoderFactory::getType(entropy.c_str()); // throws on error
    _transformType = TransformFactory<byte>::getType(transform.c_str()); // throws on error
    _initialized = false;
    _closed = false;
    _gcount = 0;
    _ibs = new DefaultInputBitStream(is, DEFAULT_BUFFER_SIZE);
    _jobs = tasks;
    _outputSize = originalSize;
    _nbInputBlocks = 0;
    _buffers = new SliceArray<byte>*[2 * _jobs];
    _headless = headerless;

    if (_headless == true) {
       if ((_blockSize < MIN_BITSTREAM_BLOCK_SIZE) || (_blockSize > MAX_BITSTREAM_BLOCK_SIZE)) {
           stringstream ss;
           ss << "Invalid or missing block size: " << _blockSize;
           throw invalid_argument(ss.str());
       }

       _ctx.putInt("bsVersion", bsVersion);
       _ctx.putString("entropy", entropy);
       _ctx.putString("transform", transform);
       _ctx.putInt("blockSize", blockSize);

       if (checksum == 32) {
          _hasher32 = new XXHash32(BITSTREAM_TYPE);
          _hasher64 = nullptr;
       }
       else if (checksum == 64) {
          _hasher32 = nullptr;
          _hasher64 = new XXHash64(BITSTREAM_TYPE);
       }
       else if (checksum != 0) {
           throw invalid_argument("The block checksum size must be 0, 32 or 64");
       }
    }

    for (int i = 0; i < 2 * _jobs; i++)
        _buffers[i] = new SliceArray<byte>(nullptr, 0, 0);
}

CompressedInputStream::CompressedInputStream(InputStream& is, Context& ctx, bool headerless)
    : InputStream(is.rdbuf())
    , _ctx(ctx)
    , _parentCtx(&ctx)
{
    int tasks = _ctx.getInt("jobs", 1);

#ifdef CONCURRENCY_ENABLED
    if ((tasks <= 0) || (tasks > MAX_CONCURRENCY)) {
        stringstream ss;
        ss << "The number of jobs must be in [1.." << MAX_CONCURRENCY << "], got " << tasks;
        throw invalid_argument(ss.str());
    }

    _pool = _ctx.getPool(); // may be null
#else
    if (tasks != 1)
        throw invalid_argument("The number of jobs is limited to 1 in this version");
#endif
    _blockId = 0;
    _bufferId = 0;
    _maxBufferId = 0;
    _blockSize = 0;
    _bufferThreshold = 0;
    _available = 0;
    _entropyType = EntropyDecoderFactory::NONE_TYPE;
    _transformType = TransformFactory<byte>::NONE_TYPE;
    _initialized = false;
    _closed = false;
    _gcount = 0;
    _ibs = new DefaultInputBitStream(is, DEFAULT_BUFFER_SIZE);
    _jobs = tasks;
    _hasher32 = nullptr;
    _hasher64 = nullptr;
    _outputSize = 0;
    _nbInputBlocks = 0;
    _headless = headerless;

    if (_headless == true) {
        // Validation of required values
        // Optional bsVersion
        const int bsVersion = _ctx.getInt("bsVersion", BITSTREAM_FORMAT_VERSION);

        if (bsVersion > BITSTREAM_FORMAT_VERSION) {
            stringstream ss;
            ss << "Invalid or missing bitstream version, cannot read this version of the stream: " << bsVersion;
            throw invalid_argument(ss.str());
        }

        _ctx.putInt("bsVersion", BITSTREAM_FORMAT_VERSION);
        string entropy = _ctx.getString("entropy");
        _entropyType = EntropyDecoderFactory::getType(entropy.c_str()); // throws on error

        string transform = _ctx.getString("transform");
        _transformType = TransformFactory<byte>::getType(transform.c_str()); // throws on error

        _blockSize = _ctx.getInt("blockSize", 0);

        if ((_blockSize < MIN_BITSTREAM_BLOCK_SIZE) || (_blockSize > MAX_BITSTREAM_BLOCK_SIZE)) {
            stringstream ss;
            ss << "Invalid or missing block size: " << _blockSize;
            throw invalid_argument(ss.str());
        }

        _bufferThreshold = _blockSize;

        // Optional outputSize
        if (_ctx.has("outputSize")) {
            _outputSize = _ctx.getLong("outputSize", 0);

            if ((_outputSize < 0) || (_outputSize >= (int64(1) << 48)))
                _outputSize = 0; // not provided
        }

        const int nbBlocks = int((_outputSize + int64(_blockSize - 1)) / int64(_blockSize));
        _nbInputBlocks = min(nbBlocks, MAX_CONCURRENCY - 1);

        // Optional checksum
        int checksum = ctx.getInt("checksum", 0);

        if (checksum == 0) {
            _hasher32 = nullptr;
            _hasher64 = nullptr;
        }
        else if (checksum == 32) {
            _hasher32 = new XXHash32(BITSTREAM_TYPE);
            _hasher64 = nullptr;
        }
        else if (checksum == 64) {
            _hasher32 = nullptr;
            _hasher64 = new XXHash64(BITSTREAM_TYPE);
        }
        else {
            throw invalid_argument("The block checksum size must be 0, 32 or 64");
        }
    }

    _buffers = new SliceArray<byte>*[2 * _jobs];

    for (int i = 0; i < 2 * _jobs; i++)
        _buffers[i] = new SliceArray<byte>(nullptr, 0, 0);
}

CompressedInputStream::~CompressedInputStream()
{
    try {
        close();
    }
    catch (exception&) {
        // Ignore and continue
    }

    for (int i = 0; i < 2 * _jobs; i++) {
        if (_buffers[i]->_array != nullptr)
            delete[] _buffers[i]->_array;

        delete _buffers[i];
    }

    delete[] _buffers;
    delete _ibs;

    if (_hasher32 != nullptr) {
        delete _hasher32;
        _hasher32 = nullptr;
    }

    if (_hasher64 != nullptr) {
        delete _hasher64;
        _hasher64 = nullptr;
    }
}

void CompressedInputStream::readHeader()
{
    if ((_headless == true) || (_initialized.exchange(true, memory_order_relaxed)))
        return;

    // Read stream type
    const int type = int(_ibs->readBits(32));

    // Sanity check
    if (type != BITSTREAM_TYPE) {
        throw IOException("Invalid stream type", Error::ERR_INVALID_FILE);
    }

    // Read stream version
    const int bsVersion = int(_ibs->readBits(4));

    // Sanity check
    if (bsVersion > BITSTREAM_FORMAT_VERSION) {
        stringstream ss;
        ss << "Invalid bitstream, cannot read this version of the stream: " << bsVersion;
        throw IOException(ss.str(), Error::ERR_STREAM_VERSION);
    }

    _ctx.putInt("bsVersion", bsVersion);
    uint64 ckSize = 0;

    // Read block checksum
    if (bsVersion >= 6) {
        ckSize = _ibs->readBits(2);

        if (ckSize == 1) {
            _hasher32 = new XXHash32(BITSTREAM_TYPE);
        }
        else if (ckSize == 2) {
            _hasher64 = new XXHash64(BITSTREAM_TYPE);
        }
        else if (ckSize == 3) {
           throw IOException("Invalid bitstream, incorrect block checksum size",
               Error::ERR_INVALID_FILE);
        }
    }
    else {
       if (_ibs->readBit() == 1)
           _hasher32 = new XXHash32(BITSTREAM_TYPE);
    }

    try {
        // Read entropy codec
        _entropyType = short(_ibs->readBits(5));
        _ctx.putString("entropy", EntropyDecoderFactory::getName(_entropyType));
    }
    catch (invalid_argument&) {
        stringstream err;
        err << "Invalid bitstream, unknown entropy type: " << _entropyType;
        throw IOException(err.str(), Error::ERR_INVALID_CODEC);
    }

    try {
        // Read transform: 8*6 bits
        _transformType = _ibs->readBits(48);
        _ctx.putString("transform", TransformFactory<byte>::getName(_transformType));
    }
    catch (invalid_argument&) {
        stringstream err;
        err << "Invalid bitstream, unknown transform type: " << _transformType;
        throw IOException(err.str(), Error::ERR_INVALID_CODEC);
    }

    // Read block size
    _blockSize = int(_ibs->readBits(28)) << 4;
    _ctx.putInt("blockSize", _blockSize);
    _bufferThreshold = _blockSize;

    if ((_blockSize < MIN_BITSTREAM_BLOCK_SIZE) || (_blockSize > MAX_BITSTREAM_BLOCK_SIZE)) {
        stringstream ss;
        ss << "Invalid bitstream, incorrect block size: " << _blockSize;
        throw IOException(ss.str(), Error::ERR_BLOCK_SIZE);
    }

    // Read original size
    // 0 -> not provided, <2^16 -> 1, <2^32 -> 2, <2^48 -> 3
    const int szMask = int(_ibs->readBits(2));

    if (szMask != 0) {
        _outputSize = _ibs->readBits(16 * szMask);

        if (_parentCtx != nullptr)
            _parentCtx->putLong("outputSize", _outputSize);

        const int nbBlocks = int((_outputSize + int64(_blockSize - 1)) / int64(_blockSize));
        _nbInputBlocks = min(nbBlocks, MAX_CONCURRENCY - 1);
    }

    if (bsVersion >= 6) {
       // Padding
       _ibs->readBits(15);
    }

    // Read & verify checksum
    const int crcSize = bsVersion <= 5 ? 16 : 24;
    const uint32 cksum1 = uint32(_ibs->readBits(crcSize));

    uint32 seed = (bsVersion >= 6 ? 0x01030507 : 1) * uint32(bsVersion);
    const uint32 HASH = 0x1E35A7BD;

    uint32 cksum2 = HASH * seed;

    if (bsVersion >= 6)
        cksum2 ^= (HASH * uint32(~ckSize));

    cksum2 ^= (HASH * uint32(~_entropyType));
    cksum2 ^= (HASH * uint32((~_transformType) >> 32));
    cksum2 ^= (HASH * uint32(~_transformType));
    cksum2 ^= (HASH * uint32(~_blockSize));

    if (szMask != 0) {
        cksum2 ^= (HASH * uint32((~_outputSize) >> 32));
        cksum2 ^= (HASH * uint32(~_outputSize));
    }

    cksum2 = (cksum2 >> 23) ^ (cksum2 >> 3);

    if (cksum1 != (cksum2 & ((1 << crcSize) - 1)))
        throw IOException("Invalid bitstream, header checksum mismatch", Error::ERR_CRC_CHECK);

    if (_listeners.size() > 0) {
        stringstream ss;
        ss << "Bitstream version: " << bsVersion << endl;
        string ckSize = "NONE";

        if (_hasher32 != nullptr)
            ckSize = "32 bits";
        else if (_hasher64 != nullptr)
            ckSize = "64 bits";

        ss << "Block checksum: " << ckSize<< endl;
        ss << "Block size: " << _blockSize << " bytes" << endl;
        string w1 = EntropyDecoderFactory::getName(_entropyType);
        ss << "Using " << ((w1 == "NONE") ? "no" : w1) << " entropy codec (stage 1)" << endl;
        string w2 = TransformFactory<byte>::getName(_transformType);
        ss << "Using " << ((w2 == "NONE") ? "no" : w2) << " transform (stage 2)" << endl;

        if (szMask != 0) {
            ss << "Original size: " << _outputSize;
            ss << (_outputSize < 2 ? " byte" : " bytes") << endl;
        }

        // Protect against future concurrent modification of the list of block listeners
        vector<Listener<Event>*> blockListeners(_listeners);
        Event evt(Event::AFTER_HEADER_DECODING, 0, ss.str(), clock());
        CompressedInputStream::notifyListeners(blockListeners, evt);
    }
}

bool CompressedInputStream::addListener(Listener<Event>& bl)
{
    _listeners.push_back(&bl);
    return true;
}

bool CompressedInputStream::removeListener(Listener<Event>& bl)
{
    std::vector<Listener<Event>*>::iterator it = find(_listeners.begin(), _listeners.end(), &bl);

    if (it == _listeners.end())
        return false;

    _listeners.erase(it);
    return true;
}

int CompressedInputStream::_get(int inc)
{
    try {
        if (_available == 0) {
            if (_closed.load(memory_order_relaxed) == true)
                throw ios_base::failure("Stream closed");

            _available = processBlock();

            if (_available == 0) {
                // Reached end of stream
                setstate(ios::eofbit);
                return EOF;
            }
        }

        int res = int(_buffers[_bufferId]->_array[_buffers[_bufferId]->_index]);

        if (inc == 0)
            return res;

        _available -= inc;
        _buffers[_bufferId]->_index += inc;

        // Is current read buffer empty ?
        if ((_bufferId < _maxBufferId) && (_buffers[_bufferId]->_index >= _blockSize))
            _bufferId++;

        return res;
    }
    catch (IOException&) {
        setstate(ios::badbit);
        throw; // rethrow
    }
    catch (exception&) {
        setstate(ios::badbit);
        throw; // rethrow
    }
}

istream& CompressedInputStream::read(char* data, streamsize length)
{
    int remaining = int(length);

    if (remaining < 0)
        throw ios_base::failure("Invalid buffer size");

    _gcount = 0;

    while (remaining > 0) {
        // Limit to number of available bytes in current buffer
        const int lenChunk = min(remaining, int(min(_available, int64(_bufferThreshold - _buffers[_bufferId]->_index))));

        if (lenChunk > 0) {
            // Process a chunk of in-buffer data. No access to bitstream required
            memcpy(&data[_gcount], &_buffers[_bufferId]->_array[_buffers[_bufferId]->_index], lenChunk);
            _buffers[_bufferId]->_index += lenChunk;
            _gcount += lenChunk;
            remaining -= lenChunk;
            _available -= lenChunk;

            if ((_bufferId < _maxBufferId) && (_buffers[_bufferId]->_index >= _blockSize)) {
                if (_bufferId + 1 >= _jobs)
                    break;

                _bufferId++;
            }

            if (remaining == 0)
                break;
        }

        // Buffer empty, time to decode
        int c2 = _get(1);

        // EOF ?
        if (c2 == EOF)
            break;

        data[_gcount++] = char(c2);
        remaining--;
    }

    return *this;
}

int64 CompressedInputStream::processBlock()
{
    readHeader();

    // Protect against future concurrent modification of the list of block listeners
    vector<Listener<Event>*> blockListeners(_listeners);
    vector<DecodingTask<DecodingTaskResult>*> tasks;

    try {
        // Add a padding area to manage any block temporarily expanded
        const int blkSize = max(_blockSize + EXTRA_BUFFER_SIZE, _blockSize + (_blockSize >> 4));
        int64 decoded = 0;
        int nbTasks = _jobs;
        int jobsPerTask[MAX_CONCURRENCY];

        // Assign optimal number of tasks and jobs per task (if the number of blocks is available)
        if (nbTasks > 1) {
            // Limit the number of tasks if there are fewer blocks that _jobs
            if (_nbInputBlocks != 0)
                nbTasks = min(_nbInputBlocks, _jobs);

            Global::computeJobsPerTask(jobsPerTask, _jobs, nbTasks);
        }
        else {
            jobsPerTask[0] = _jobs;
        }

        const int bufSize = max(_blockSize + EXTRA_BUFFER_SIZE, _blockSize + (_blockSize >> 4));

        while (true) {
            const int firstBlockId = _blockId.load(memory_order_acquire);

            for (int taskId = 0; taskId < nbTasks; taskId++) {
                if (_buffers[taskId]->_length < bufSize) {
                    if (_buffers[taskId]->_array != nullptr)
                       delete[] _buffers[taskId]->_array;

                    _buffers[taskId]->_array = new byte[bufSize];
                    _buffers[taskId]->_length = bufSize;
                }

                Context copyCtx(_ctx);
                copyCtx.putInt("jobs", jobsPerTask[taskId]); // jobs for current task
                copyCtx.putInt("tasks", nbTasks); // overall number of tasks
                copyCtx.putLong("tType", _transformType);
                copyCtx.putInt("eType", _entropyType);
                copyCtx.putInt("blockId", firstBlockId + taskId + 1);

                _buffers[taskId]->_index = 0;
                _buffers[_jobs + taskId]->_index = 0;

                DecodingTask<DecodingTaskResult>* task = new DecodingTask<DecodingTaskResult>(_buffers[taskId],
                    _buffers[_jobs + taskId], blkSize,
                    _ibs, _hasher32, _hasher64, &_blockId,
                    blockListeners, copyCtx);
                tasks.push_back(task);
            }

            int skipped = 0;
            _maxBufferId = nbTasks - 1;

            if (tasks.size() == 1) {
                // Synchronous call
                DecodingTask<DecodingTaskResult>* task = tasks.back();
                tasks.pop_back();
                DecodingTaskResult res = task->run();
                delete task;
                decoded += res._decoded;

                if (res._error != 0)
                    throw IOException(res._msg, res._error); // deallocate in catch block

                if (res._skipped == true)
                    skipped++;

                if (decoded > _blockSize) {
                    stringstream ss;
                    ss << "Block " << res._blockId << " incorrectly decompressed";
                    throw IOException(ss.str(), Error::ERR_PROCESS_BLOCK); // deallocate in catch code
                }

                if (_buffers[_bufferId]->_array != res._data)
                   memcpy(&_buffers[_bufferId]->_array[0], &res._data[0], res._decoded);

                _buffers[_bufferId]->_index = 0;

                if (blockListeners.size() > 0) {
                    Event::HashType hashType = Event::NO_HASH;

                    if (_hasher32 != nullptr)
                        hashType = Event::SIZE_32;
                    else if (_hasher64 != nullptr)
                        hashType = Event::SIZE_64;

                    // Notify after transform ... in block order !
                    Event evt(Event::AFTER_TRANSFORM, res._blockId,
                        int64(res._decoded), res._completionTime, res._checksum, hashType);
                    CompressedInputStream::notifyListeners(blockListeners, evt);
                }
            }
#ifdef CONCURRENCY_ENABLED
            else {
                vector<future<DecodingTaskResult> > futures;

                // Register task futures and launch tasks in parallel
                for (uint i = 0; i < tasks.size(); i++) {
                    if (_pool == nullptr)
                        futures.push_back(async(&DecodingTask<DecodingTaskResult>::run, tasks[i]));
                    else
                        futures.push_back(_pool->schedule(&DecodingTask<DecodingTaskResult>::run, tasks[i]));
                }

                int error = 0;
                string msg;

                // Wait for tasks completion and check results
                for (uint i = 0; i < futures.size(); i++) {
                    DecodingTaskResult res = futures[i].get();

                    if (error != 0)
                        continue;

                    if (res._skipped == true) {
                        skipped++;
                        continue;
                    }

                    if (res._decoded > _blockSize) {
                        error = Error::ERR_PROCESS_BLOCK;
                        stringstream ss;
                        ss << "Block " << res._blockId << " incorrectly decompressed";
                        msg = ss.str();
                        continue;
                    }

                    if (res._error == 0) {
                       decoded += res._decoded;

                       if (_buffers[i]->_array != res._data)
                           memcpy(&_buffers[i]->_array[0], &res._data[0], res._decoded);

                        _buffers[i]->_index = 0;

                        if (blockListeners.size() > 0) {
                           Event::HashType hashType = Event::NO_HASH;

                           if (_hasher32 != nullptr)
                               hashType = Event::SIZE_32;
                           else if (_hasher64 != nullptr)
                               hashType = Event::SIZE_64;

                           // Notify after transform ... in block order !
                           Event evt(Event::AFTER_TRANSFORM, res._blockId,
                               int64(res._decoded), res._completionTime, res._checksum, hashType);
                           CompressedInputStream::notifyListeners(blockListeners, evt);
                        }
                    }

                    // Capture first error but continue getting results from other tasks
                    // instead of exiting early, otherwise it is possible that the error
                    // management code is going to deallocate memory used by other tasks
                    // before they are completed.
                    error = res._error;
                    msg = res._msg;
                }

                if (error != 0)
                    throw IOException(msg, error); // deallocate in catch block
            }

            for (vector<DecodingTask<DecodingTaskResult>*>::iterator it = tasks.begin(); it != tasks.end(); ++it)
                delete *it;

            tasks.clear();
#endif

            _bufferId = 0;

            // Unless all blocks were skipped, exit the loop (usual case)
            if (skipped != nbTasks)
                break;
        }

        return decoded;
    }
    catch (IOException&) {
        for (vector<DecodingTask<DecodingTaskResult>*>::iterator it = tasks.begin(); it != tasks.end(); ++it)
            delete *it;

        tasks.clear();
        throw;
    }
    catch (exception& e) {
        for (vector<DecodingTask<DecodingTaskResult>*>::iterator it = tasks.begin(); it != tasks.end(); ++it)
            delete *it;

        tasks.clear();
        throw IOException(e.what(), Error::ERR_UNKNOWN);
    }
}

void CompressedInputStream::close()
{
    if (_closed.exchange(true, memory_order_relaxed))
        return;

    try {
        _ibs->close();
    }
    catch (BitStreamException& e) {
        throw IOException(e.what(), e.error());
    }

    _available = 0;
    _bufferThreshold = 0;

    // Release resources, force error on any subsequent write attempt
    for (int i = 0; i < 2 * _jobs; i++) {
        if (_buffers[i]->_array != nullptr)
           delete[] _buffers[i]->_array;

        _buffers[i]->_array = nullptr;
        _buffers[i]->_length = 0;
        _buffers[i]->_index = 0;
    }
}

void CompressedInputStream::notifyListeners(vector<Listener<Event>*>& listeners, const Event& evt)
{
    for (vector<Listener<Event>*>::iterator it = listeners.begin(); it != listeners.end(); ++it)
        (*it)->processEvent(evt);
}

template <class T>
DecodingTask<T>::DecodingTask(SliceArray<byte>* iBuffer, SliceArray<byte>* oBuffer,
    int blockSize, DefaultInputBitStream* ibs, XXHash32* hasher32, XXHash64* hasher64,
    ATOMIC_INT* processedBlockId, vector<Listener<Event>*>& listeners,
    const Context& ctx)
    : _listeners(listeners)
    , _ctx(ctx)
{
    _blockLength = blockSize;
    _data = iBuffer;
    _buffer = oBuffer;
    _ibs = ibs;
    _hasher32 = hasher32;
    _hasher64 = hasher64;
    _processedBlockId = processedBlockId;
}

// Decode mode + transformed entropy coded data
// mode | 0b1yy0xxxx => copy block
//      | 0b0yy00000 => size(size(block))-1
//  case 4 transforms or less
//      | 0b0001xxxx => transform sequence skip flags (1 means skip)
//  case more than 4 transforms
//      | 0b0yy00000 0bxxxxxxxx => transform sequence skip flags in next byte (1 means skip)
template <class T>
T DecodingTask<T>::run()
{
    int blockId = _ctx.getInt("blockId");

    // Lock free synchronization
    while (true) {
        const int taskId = _processedBlockId->load(memory_order_acquire);

        if (taskId == CompressedInputStream::CANCEL_TASKS_ID) {
            // Skip, an error occurred
            return T(*_data, blockId, 0, 0, 0, "Canceled");
        }

        if (taskId == blockId - 1)
            break;

        // Back-off improves performance
        CPU_PAUSE();
    }

    uint64 checksum1 = 0;
    EntropyDecoder* ed = nullptr;
    InputBitStream* ibs = nullptr;
    TransformSequence<byte>* transform = nullptr;
    bool streamPerTask = _ctx.getInt("tasks") > 1;
    uint64 tType = _ctx.getLong("tType");
    short eType = short(_ctx.getInt("eType"));

    try {
        // Read shared bitstream sequentially (each task is gated by _processedBlockId)
#if !defined(_MSC_VER) || _MSC_VER > 1500
        const uint64 blockOffset = _ibs->tell();
#endif
        const uint lr = 3 + uint(_ibs->readBits(5));
        uint64 read = _ibs->readBits(lr);

        if (read == 0) {
            _processedBlockId->store(CompressedInputStream::CANCEL_TASKS_ID, memory_order_release);
            return T(*_data, blockId, 0, 0, 0, "Success");
        }

        if (read > (uint64(1) << 34)) {
            _processedBlockId->store(CompressedInputStream::CANCEL_TASKS_ID, memory_order_release);
            return T(*_data, blockId, 0, 0, Error::ERR_BLOCK_SIZE, "Invalid block size");
        }

        const int r = int((read + 7) >> 3);

        if (streamPerTask == true) {
            if (_data->_length < max(_blockLength, r)) {
                _data->_length = max(_blockLength, r);
                delete[] _data->_array;
                _data->_array = new byte[_data->_length];
            }

            for (int n = 0; read > 0; ) {
                const uint chkSize = uint(min(read, uint64(1) << 30));
                _ibs->readBits(&_data->_array[n], chkSize);
                n += ((chkSize + 7) >> 3);
                read -= uint64(chkSize);
            }
        }

        // After completion of the bitstream reading, increment the block id.
        // It unblocks the task processing the next block (if any)
        _processedBlockId->store(blockId, memory_order_release);

        const int from = _ctx.getInt("from", 1);
        const int to = _ctx.getInt("to", CompressedInputStream::MAX_BLOCK_ID);

        // Check if the block must be skipped
        if (blockId < from) {
            return T(*_data, blockId, 0, 0, 0, "Skipped", true);
        }
        else if (blockId >= to) {
            return T(*_data, blockId, 0, 0, 0, "Success");
        }

        istreambuf<char> buf(reinterpret_cast<char*>(&_data->_array[0]), streamsize(r));
        iostream ios(&buf);
        ibs = (streamPerTask == true) ? new DefaultInputBitStream(ios) : _ibs;

        // Extract block header from bitstream
        byte mode = byte(ibs->readBits(8));
        byte skipFlags = byte(0);

        if ((mode & CompressedInputStream::COPY_BLOCK_MASK) != byte(0)) {
            tType = TransformFactory<byte>::NONE_TYPE;
            eType = EntropyDecoderFactory::NONE_TYPE;
        }
        else {
            if ((mode & CompressedInputStream::TRANSFORMS_MASK) != byte(0))
                skipFlags = byte(ibs->readBits(8));
            else
                skipFlags = (mode << 4) | byte(0x0F);
        }

        const int dataSize = 1 + (int(mode >> 5) & 0x03);
        const int length = dataSize << 3;
        const uint64 mask = (uint64(1) << length) - 1;
        const int preTransformLength = int(ibs->readBits(length) & mask);
        const int maxTransformSize = min(max(_blockLength + _blockLength / 2, 2048), CompressedInputStream::MAX_BITSTREAM_BLOCK_SIZE);

        if ((preTransformLength <= 0) || (preTransformLength > maxTransformSize)) {
            // Error => cancel concurrent decoding tasks
            _processedBlockId->store(CompressedInputStream::CANCEL_TASKS_ID, memory_order_release);
            stringstream ss;
            ss << "Invalid compressed block length: " << preTransformLength;

            if (streamPerTask == true)
                delete ibs;

            return T(*_data, blockId, 0, checksum1, Error::ERR_READ_FILE, ss.str());
        }

        Event::HashType hashType = Event::NO_HASH;

        // Extract checksum from bitstream (if any)
        if (_hasher32 != nullptr) {
            checksum1 = ibs->readBits(32);
            hashType = Event::SIZE_32;
        }
        else if (_hasher64 != nullptr) {
            checksum1 = ibs->readBits(64);
            hashType = Event::SIZE_64;
        }

        if (_listeners.size() > 0) {
#if !defined(_MSC_VER) || _MSC_VER > 1500
            if (_ctx.getInt("verbosity", 0) > 4) {
                Event evt1(Event::BLOCK_INFO, blockId, int64(r), clock(), checksum1,
                           hashType, blockOffset, uint8(skipFlags));
                CompressedInputStream::notifyListeners(_listeners, evt1);
            }
#endif

            // Notify before entropy
            Event evt2(Event::BEFORE_ENTROPY, blockId, int64(r), clock(), checksum1, hashType);
            CompressedInputStream::notifyListeners(_listeners, evt2);
        }

        const int bufferSize = max(_blockLength, preTransformLength + CompressedInputStream::EXTRA_BUFFER_SIZE);

        if (_buffer->_length < bufferSize) {
            _buffer->_length = bufferSize;
            if (_buffer->_array != nullptr)
               delete[] _buffer->_array;

            _buffer->_array = new byte[_buffer->_length];
        }

        const int savedIdx = _data->_index;
        _ctx.putInt("size", preTransformLength);

        // Each block is decoded separately
        // Rebuild the entropy decoder to reset block statistics
        ed = EntropyDecoderFactory::newDecoder(*ibs, _ctx, eType);

        // Block entropy decode
        if (ed->decode(_buffer->_array, 0, preTransformLength) != preTransformLength) {
            // Error => cancel concurrent decoding tasks
            delete ed;
            _processedBlockId->store(CompressedInputStream::CANCEL_TASKS_ID, memory_order_release);
            return T(*_data, blockId, 0, checksum1, Error::ERR_PROCESS_BLOCK,
                "Entropy decoding failed");
        }

        if (streamPerTask == true) {
            delete ibs;
            ibs = nullptr;
        }

        delete ed;
        ed = nullptr;

        if (_listeners.size() > 0) {
            // Notify after entropy
            Event evt1(Event::AFTER_ENTROPY, blockId,
                int64(preTransformLength), clock(), checksum1, hashType);
            CompressedInputStream::notifyListeners(_listeners, evt1);

            // Notify before transform (block size after entropy decoding)
            Event evt2(Event::BEFORE_TRANSFORM, blockId,
                int64(preTransformLength), clock(), checksum1, hashType);
            CompressedInputStream::notifyListeners(_listeners, evt2);
        }

        transform = TransformFactory<byte>::newTransform(_ctx, tType);
        transform->setSkipFlags(skipFlags);
        _buffer->_index = 0;

        // Inverse transform
        bool res = transform->inverse(*_buffer, *_data, preTransformLength);
        delete transform;
        transform = nullptr;

        if (res == false) {
            return T(*_data, blockId, 0, checksum1, Error::ERR_PROCESS_BLOCK,
                "Transform inverse failed");
        }

        const int decoded = _data->_index - savedIdx;

        // Verify checksum
        if (_hasher32 != nullptr) {
            const uint32 checksum2 = _hasher32->hash(&_data->_array[savedIdx], decoded);

            if (checksum2 != uint32(checksum1)) {
                stringstream ss;
                ss << "Corrupted bitstream: expected checksum " << std::hex << checksum1 << ", found " << std::hex << checksum2;
                return T(*_data, blockId, decoded, checksum1, Error::ERR_CRC_CHECK, ss.str());
            }
        }
        else if (_hasher64 != nullptr) {
            const uint64 checksum2 = _hasher64->hash(&_data->_array[savedIdx], decoded);

            if (checksum2 != checksum1) {
                stringstream ss;
                ss << "Corrupted bitstream: expected checksum " << std::hex << checksum1 << ", found " << std::hex << checksum2;
                return T(*_data, blockId, decoded, checksum1, Error::ERR_CRC_CHECK, ss.str());
            }
        }

        return T(*_data, blockId, decoded, checksum1, 0, "Success");
    }
    catch (exception& e) {
        // Make sure to unfreeze next block
        if (_processedBlockId->load(memory_order_acquire) == blockId - 1)
            _processedBlockId->store(blockId, memory_order_release);

        if (transform != nullptr)
            delete transform;

        if (ed != nullptr)
            delete ed;

        if ((streamPerTask == true) && (ibs != nullptr))
            delete ibs;

        return T(*_data, blockId, 0, checksum1, Error::ERR_PROCESS_BLOCK, e.what());
    }
}

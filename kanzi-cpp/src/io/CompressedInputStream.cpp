/*
Copyright 2011-2024 Frederic Langlet
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
#include "../bitstream/DefaultInputBitStream.hpp"
#include "../entropy/EntropyDecoderFactory.hpp"
#include "../transform/TransformFactory.hpp"

#ifdef CONCURRENCY_ENABLED
#include <future>
#endif

using namespace kanzi;
using namespace std;


#ifdef CONCURRENCY_ENABLED
CompressedInputStream::CompressedInputStream(InputStream& is, int tasks, ThreadPool* pool)
#else
CompressedInputStream::CompressedInputStream(InputStream& is, int tasks)
#endif
    : InputStream(is.rdbuf())
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
    _hasher = nullptr;
    _nbInputBlocks = UNKNOWN_NB_BLOCKS;
    _buffers = new SliceArray<byte>*[2 * _jobs];

    for (int i = 0; i < 2 * _jobs; i++)
        _buffers[i] = new SliceArray<byte>(new byte[0], 0, 0);
}

#if __cplusplus >= 201103L
CompressedInputStream::CompressedInputStream(InputStream& is, Context& ctx,
    std::function<InputBitStream* (InputStream&)>* createBitStream)
#else
CompressedInputStream::CompressedInputStream(InputStream& is, Context& ctx)
#endif
    : InputStream(is.rdbuf())
    , _ctx(ctx)
{
    int tasks = ctx.getInt("jobs", 1);

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

#if __cplusplus >= 201103L
    // A hook can be provided by the caller to customize the instantiation of the
    // input bitstream.
    _ibs = (createBitStream == nullptr) ? new DefaultInputBitStream(is, DEFAULT_BUFFER_SIZE) : (*createBitStream)(is);
#else
    _ibs = new DefaultInputBitStream(is, DEFAULT_BUFFER_SIZE);
#endif

    _jobs = tasks;
    _hasher = nullptr;
    _nbInputBlocks = UNKNOWN_NB_BLOCKS;
    _buffers = new SliceArray<byte>*[2 * _jobs];

    for (int i = 0; i < 2 * _jobs; i++)
        _buffers[i] = new SliceArray<byte>(new byte[0], 0, 0);
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
        delete[] _buffers[i]->_array;
        delete _buffers[i];
    }

    delete[] _buffers;
    delete _ibs;

    if (_hasher != nullptr) {
        delete _hasher;
        _hasher = nullptr;
    }
}

void CompressedInputStream::readHeader() THROW
{
    // Read stream type
    const int type = int(_ibs->readBits(32));

    // Sanity check
    if (type != BITSTREAM_TYPE) {
        throw IOException("Invalid stream type", Error::ERR_INVALID_FILE);
    }

    // Read stream version
    int bsVersion = int(_ibs->readBits(4));

    // Sanity check
    if (bsVersion != BITSTREAM_FORMAT_VERSION) {
        stringstream ss;
        ss << "Invalid bitstream, cannot read this version of the stream: " << bsVersion;
        throw IOException(ss.str(), Error::ERR_STREAM_VERSION);
    }

    _ctx.putInt("bsVersion", bsVersion);

    // Read block checksum
    if (_ibs->readBit() == 1)
        _hasher = new XXHash32(BITSTREAM_TYPE);

    try {
        // Read entropy codec
        _entropyType = short(_ibs->readBits(5));
        _ctx.putString("codec", EntropyDecoderFactory::getName(_entropyType));
        _ctx.putString("extra", _entropyType == EntropyDecoderFactory::TPAQX_TYPE ? STR_TRUE : STR_FALSE);
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

    // Read number of blocks in input. 
    _nbInputBlocks = int(_ibs->readBits(6));

    // 0 means 'unknown' and 63 means 63 or more.
    if (_nbInputBlocks == 0)
        _nbInputBlocks = UNKNOWN_NB_BLOCKS;

    // Read checksum
    const uint32 cksum1 = uint32(_ibs->readBits(4));

    // Verify checksum
    const uint32 HASH = 0x1E35A7BD;
    uint32 cksum2 = HASH * uint32(bsVersion);
    cksum2 ^= (HASH * uint32(_entropyType));
    cksum2 ^= (HASH * uint32(_transformType >> 32));
    cksum2 ^= (HASH * uint32(_transformType));
    cksum2 ^= (HASH * uint32(_blockSize));
    cksum2 ^= (HASH * uint32(_nbInputBlocks));
    cksum2 = (cksum2 >> 23) ^ (cksum2 >> 3);

    if (cksum1 != (cksum2 & 0x0F))
        throw IOException("Invalid bitstream, corrupted header", Error::ERR_CRC_CHECK);

    if (_listeners.size() > 0) {
        stringstream ss;
        ss << "Checksum set to " << (_hasher != nullptr ? "true" : "false") << endl;
        ss << "Block size set to " << _blockSize << " bytes" << endl;
        string w1 = EntropyDecoderFactory::getName(_entropyType);
        ss << "Using " << ((w1 == "NONE") ? "no" : w1) << " entropy codec (stage 1)" << endl;
        string w2 = TransformFactory<byte>::getName(_transformType);
        ss << "Using " << ((w2 == "NONE") ? "no" : w2) << " transform (stage 2)" << endl;

        // Protect against future concurrent modification of the list of block listeners
        vector<Listener*> blockListeners(_listeners);
        Event evt(Event::AFTER_HEADER_DECODING, 0, ss.str(), clock());
        CompressedInputStream::notifyListeners(blockListeners, evt);
    }
}

bool CompressedInputStream::addListener(Listener& bl)
{
    _listeners.push_back(&bl);
    return true;
}

bool CompressedInputStream::removeListener(Listener& bl)
{
    std::vector<Listener*>::iterator it = find(_listeners.begin(), _listeners.end(), &bl);

    if (it == _listeners.end())
        return false;

    _listeners.erase(it);
    return true;
}

int CompressedInputStream::_get(int inc) THROW
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

istream& CompressedInputStream::read(char* data, streamsize length) THROW
{
    int remaining = int(length);

    if (remaining < 0)
        throw ios_base::failure("Invalid buffer size");

    _gcount = 0;

    while (remaining > 0) {
        // Limit to number of available bytes in current buffer
        const int lenChunk = min(remaining, min(_available, _bufferThreshold - _buffers[_bufferId]->_index));

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

int CompressedInputStream::processBlock() THROW
{
    if (!_initialized.exchange(true, memory_order_acquire))
        readHeader();

    // Protect against future concurrent modification of the list of block listeners
    vector<Listener*> blockListeners(_listeners);
    vector<DecodingTask<DecodingTaskResult>*> tasks;

    try {
        // Add a padding area to manage any block temporarily expanded
        const int blkSize = max(_blockSize + EXTRA_BUFFER_SIZE, _blockSize + (_blockSize >> 4));
        int decoded = 0;

        while (true) {
            const int firstBlockId = _blockId.load(memory_order_relaxed);
            int nbTasks = _jobs;
            int jobsPerTask[MAX_CONCURRENCY];

            // Assign optimal number of tasks and jobs per task
            if (nbTasks > 1) {
                // Limit the number of tasks if there are fewer blocks that _jobs
                // It allows more jobs per task and reduces memory usage.
                nbTasks = min(_nbInputBlocks, _jobs);
                Global::computeJobsPerTask(jobsPerTask, _jobs, nbTasks);
            }
            else {
                jobsPerTask[0] = _jobs;
            }

            const int bufSize = max(_blockSize + EXTRA_BUFFER_SIZE, _blockSize + (_blockSize >> 4));

            // Create as many tasks as empty buffers to decode
            for (int taskId = 0; taskId < nbTasks; taskId++) {
                if (_buffers[taskId]->_length < bufSize) {
                    delete[] _buffers[taskId]->_array;
                    _buffers[taskId]->_array = new byte[bufSize];
                    _buffers[taskId]->_length = bufSize;
                }

                Context copyCtx(_ctx);
                copyCtx.putInt("jobs", jobsPerTask[taskId]); // jobs for current task
                copyCtx.putInt("tasks", nbTasks); // overall number of tasks
                _buffers[taskId]->_index = 0;
                _buffers[_jobs + taskId]->_index = 0;

                DecodingTask<DecodingTaskResult>* task = new DecodingTask<DecodingTaskResult>(_buffers[taskId],
                    _buffers[_jobs + taskId], blkSize, _transformType,
                    _entropyType, firstBlockId + taskId + 1,
                    _ibs, _hasher, &_blockId,
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

                if (decoded > _blockSize)
                    throw IOException("Invalid data", Error::ERR_PROCESS_BLOCK); // deallocate in catch code

                if (_buffers[_bufferId]->_array != res._data)
                   memcpy(&_buffers[_bufferId]->_array[0], &res._data[0], res._decoded);
                
                _buffers[_bufferId]->_index = 0;

                if (blockListeners.size() > 0) {
                    // Notify after transform ... in block order !
                    Event evt(Event::AFTER_TRANSFORM, res._blockId,
                        int64(res._decoded), res._checksum, _hasher != nullptr, res._completionTime);
                    CompressedInputStream::notifyListeners(blockListeners, evt);
                }
            }
#ifdef CONCURRENCY_ENABLED
            else {
                vector<future<DecodingTaskResult> > futures;
                vector<DecodingTaskResult> results;

                // Register task futures and launch tasks in parallel
                for (uint i = 0; i < tasks.size(); i++) {
                    if (_pool == nullptr)
                        futures.push_back(async(&DecodingTask<DecodingTaskResult>::run, tasks[i]));
                    else
                        futures.push_back(_pool->schedule(&DecodingTask<DecodingTaskResult>::run, tasks[i]));
                }

                // Wait for tasks completion and check results
                for (uint i = 0; i < futures.size(); i++) {
                    DecodingTaskResult status = futures[i].get();
                    results.push_back(status);
                    decoded += status._decoded;

                    if (status._error != 0)
                        throw IOException(status._msg, status._error); // deallocate in catch block

                    if (status._decoded > _blockSize)
                        throw IOException("Invalid data", Error::ERR_PROCESS_BLOCK); // deallocate in catch code

                    if (status._skipped == true)
                        skipped++;
                }

                for (uint i = 0; i < results.size(); i++) {
                    DecodingTaskResult res = results[i];

                    if (_buffers[i]->_array != res._data)
                       memcpy(&_buffers[i]->_array[0], &res._data[0], res._decoded);

                    _buffers[i]->_index = 0;

                    if (blockListeners.size() > 0) {
                        // Notify after transform ... in block order !
                        Event evt(Event::AFTER_TRANSFORM, res._blockId,
                            int64(res._decoded), res._checksum, _hasher != nullptr, res._completionTime);
                        CompressedInputStream::notifyListeners(blockListeners, evt);
                    }
                }
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

void CompressedInputStream::close() THROW
{
    if (_closed.exchange(true, memory_order_acquire))
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
        delete[] _buffers[i]->_array;
        _buffers[i]->_array = new byte[0];
        _buffers[i]->_length = 0;
        _buffers[i]->_index = 0;
    }
}

void CompressedInputStream::notifyListeners(vector<Listener*>& listeners, const Event& evt)
{
    vector<Listener*>::iterator it;

    for (it = listeners.begin(); it != listeners.end(); ++it)
        (*it)->processEvent(evt);
}

template <class T>
DecodingTask<T>::DecodingTask(SliceArray<byte>* iBuffer, SliceArray<byte>* oBuffer, int blockSize,
    uint64 transformType, short entropyType, int blockId,
    InputBitStream* ibs, XXHash32* hasher,
    atomic_int* processedBlockId, vector<Listener*>& listeners,
    const Context& ctx)
    : _listeners(listeners)
    , _ctx(ctx)
{
    _blockLength = blockSize;
    _data = iBuffer;
    _buffer = oBuffer;
    _transformType = transformType;
    _entropyType = entropyType;
    _blockId = blockId;
    _ibs = ibs;
    _hasher = hasher;
    _processedBlockId = processedBlockId;
}

// Decode mode + transformed entropy coded data
// mode | 0b10000000 => copy block
//      | 0b0yy00000 => size(size(block))-1
//      | 0b000y0000 => 1 if more than 4 transforms
//  case 4 transforms or less
//      | 0b0000yyyy => transform sequence skip flags (1 means skip)
//  case more than 4 transforms
//      | 0b00000000
//      then 0byyyyyyyy => transform sequence skip flags (1 means skip)
template <class T>
T DecodingTask<T>::run() THROW
{
    // Lock free synchronization
    while (true) {
        const int taskId = _processedBlockId->load(memory_order_relaxed);

        if (taskId == CompressedInputStream::CANCEL_TASKS_ID) {
            // Skip, an error occurred
            return T(*_data, _blockId, 0, 0, 0, "Canceled");
        }

        if (taskId == _blockId - 1)
            break;

        // Back-off improves performance
        CPU_PAUSE();
    }

    uint32 checksum1 = 0;
    EntropyDecoder* ed = nullptr;
    InputBitStream* ibs = nullptr;
    TransformSequence<byte>* transform = nullptr;
    bool streamPerTask = _ctx.getInt("tasks") > 1;

    try {
        // Read shared bitstream sequentially (each task is gated by _processedBlockId)
        const uint lr = 3 + uint(_ibs->readBits(5));
        uint64 read = _ibs->readBits(lr);

        if (read == 0) {
            _processedBlockId->store(CompressedInputStream::CANCEL_TASKS_ID, memory_order_release);
            return T(*_data, _blockId, 0, 0, 0, "Success");
        }

        if (read > (uint64(1) << 34)) {
            _processedBlockId->store(CompressedInputStream::CANCEL_TASKS_ID, memory_order_release);
            return T(*_data, _blockId, 0, 0, Error::ERR_BLOCK_SIZE, "Invalid block size");
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
        (*_processedBlockId)++;

        const int from = _ctx.getInt("from", 1);
        const int to = _ctx.getInt("to", CompressedInputStream::MAX_BLOCK_ID);

        // Check if the block must be skipped
        if ((_blockId < from) || (_blockId >= to)) {
            return T(*_data, _blockId, 0, 0, 0, "Skipped", true);
        }

        istreambuf<char> buf(reinterpret_cast<char*>(&_data->_array[0]), streamsize(r));
        iostream ios(&buf);
        ibs = (streamPerTask == true) ? new DefaultInputBitStream(ios) : _ibs;

        // Extract block header from bitstream
        byte mode = byte(ibs->readBits(8));
        byte skipFlags = byte(0);

        if ((mode & CompressedInputStream::COPY_BLOCK_MASK) != byte(0)) {
            _transformType = TransformFactory<byte>::NONE_TYPE;
            _entropyType = EntropyDecoderFactory::NONE_TYPE;
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

        if (preTransformLength == 0) {
            // Error => cancel concurrent decoding tasks
            _processedBlockId->store(CompressedInputStream::CANCEL_TASKS_ID, memory_order_release);

            if (streamPerTask == true)
                delete ibs;

            return T(*_data, _blockId, 0, checksum1, 0, "Invalid transform block size");
        }

        if ((preTransformLength < 0) || (preTransformLength > CompressedInputStream::MAX_BITSTREAM_BLOCK_SIZE)) {
            // Error => cancel concurrent decoding tasks
            _processedBlockId->store(CompressedInputStream::CANCEL_TASKS_ID, memory_order_release);
            stringstream ss;
            ss << "Invalid compressed block length: " << preTransformLength;

            if (streamPerTask == true)
                delete ibs;

            return T(*_data, _blockId, 0, checksum1, Error::ERR_READ_FILE, ss.str());
        }

        // Extract checksum from bitstream (if any)
        if (_hasher != nullptr)
            checksum1 = uint32(ibs->readBits(32));

        if (_listeners.size() > 0) {
            // Notify before entropy (block size in bitstream is unknown)
            Event evt(Event::BEFORE_ENTROPY, _blockId, int64(-1), checksum1, _hasher != nullptr, clock());
            CompressedInputStream::notifyListeners(_listeners, evt);
        }

        const int bufferSize = max(_blockLength, preTransformLength + CompressedInputStream::EXTRA_BUFFER_SIZE);

        if (_buffer->_length < bufferSize) {
            _buffer->_length = bufferSize;
            delete[] _buffer->_array;
            _buffer->_array = new byte[_buffer->_length];
        }

        const int savedIdx = _data->_index;
        _ctx.putInt("size", preTransformLength);

        // Each block is decoded separately
        // Rebuild the entropy decoder to reset block statistics
        ed = EntropyDecoderFactory::newDecoder(*ibs, _ctx, _entropyType);

        // Block entropy decode
        if (ed->decode(_buffer->_array, 0, preTransformLength) != preTransformLength) {
            // Error => cancel concurrent decoding tasks
            delete ed;
            _processedBlockId->store(CompressedInputStream::CANCEL_TASKS_ID, memory_order_release);
            return T(*_data, _blockId, 0, checksum1, Error::ERR_PROCESS_BLOCK,
                "Entropy decoding failed");
        }

        if (streamPerTask == true) {
            delete ibs;
            ibs = nullptr;
        }

        delete ed;
        ed = nullptr;

        if (_listeners.size() > 0) {
            // Notify after entropy (block size set to size in bitstream)
            Event evt(Event::AFTER_ENTROPY, _blockId,
                int64(r), checksum1, _hasher != nullptr, clock());

            CompressedInputStream::notifyListeners(_listeners, evt);
        }

        if (_listeners.size() > 0) {
            // Notify before transform (block size after entropy decoding)
            Event evt(Event::BEFORE_TRANSFORM, _blockId,
                int64(preTransformLength), checksum1, _hasher != nullptr, clock());

            CompressedInputStream::notifyListeners(_listeners, evt);
        }

        transform = TransformFactory<byte>::newTransform(_ctx, _transformType);
        transform->setSkipFlags(skipFlags);
        _buffer->_index = 0;

        // Inverse transform
        bool res = transform->inverse(*_buffer, *_data, preTransformLength);
        delete transform;
        transform = nullptr;

        if (res == false) {
            return T(*_data, _blockId, 0, checksum1, Error::ERR_PROCESS_BLOCK,
                "Transform inverse failed");
        }

        const int decoded = _data->_index - savedIdx;

        // Verify checksum
        if (_hasher != nullptr) {
            const uint32 checksum2 = _hasher->hash(&_data->_array[savedIdx], decoded);

            if (checksum2 != checksum1) {
                stringstream ss;
                ss << "Corrupted bitstream: expected checksum " << hex << checksum1 << ", found " << hex << checksum2;
                return T(*_data, _blockId, decoded, checksum1, Error::ERR_CRC_CHECK, ss.str());
            }
        }

        return T(*_data, _blockId, decoded, checksum1, 0, "Success");
    }
    catch (exception& e) {
        // Make sure to unfreeze next block
        if (_processedBlockId->load(memory_order_relaxed) == _blockId - 1)
            (*_processedBlockId)++;

        if (transform != nullptr)
            delete transform;

        if (ed != nullptr)
            delete ed;

        if ((streamPerTask == true) && (ibs != nullptr))
            delete ibs;

        return T(*_data, _blockId, 0, checksum1, Error::ERR_PROCESS_BLOCK, e.what());
    }
}

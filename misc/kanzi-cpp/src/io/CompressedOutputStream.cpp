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

#include <algorithm>
#include <sstream>
#include <stdio.h>
#include "CompressedOutputStream.hpp"
#include "IOException.hpp"
#include "../Error.hpp"
#include "../Magic.hpp"
#include "../entropy/EntropyEncoderFactory.hpp"
#include "../entropy/EntropyUtils.hpp"
#include "../transform/TransformFactory.hpp"

#ifdef CONCURRENCY_ENABLED
#include <future>
#endif

using namespace kanzi;
using namespace std;


const int CompressedOutputStream::BITSTREAM_TYPE = 0x4B414E5A; // "KANZ"
const int CompressedOutputStream::BITSTREAM_FORMAT_VERSION = 6;
const int CompressedOutputStream::DEFAULT_BUFFER_SIZE = 256 * 1024;
const byte CompressedOutputStream::COPY_BLOCK_MASK = byte(0x80);
const byte CompressedOutputStream::TRANSFORMS_MASK = byte(0x10);
const int CompressedOutputStream::MIN_BITSTREAM_BLOCK_SIZE = 1024;
const int CompressedOutputStream::MAX_BITSTREAM_BLOCK_SIZE = 1024 * 1024 * 1024;
const int CompressedOutputStream::SMALL_BLOCK_SIZE = 15;
const int CompressedOutputStream::CANCEL_TASKS_ID = -1;
const int CompressedOutputStream::MAX_CONCURRENCY = 64;


CompressedOutputStream::CompressedOutputStream(OutputStream& os,
                   int tasks,
                   const string& entropy,
                   const string& transform,
                   int blockSize,
                   int checksum,
                   uint64 fileSize,
#ifdef CONCURRENCY_ENABLED
                   ThreadPool* pool,
#endif
                   bool headerless)
    : OutputStream(os.rdbuf())
{
#ifdef CONCURRENCY_ENABLED
    if ((tasks <= 0) || (tasks > MAX_CONCURRENCY)) {
        stringstream ss;
        ss << "The number of jobs must be in [1.." << MAX_CONCURRENCY << "], got " << tasks;
        throw invalid_argument(ss.str());
    }

    _pool = pool; // can be null
#else
    if (tasks != 1)
        throw invalid_argument("The number of jobs is limited to 1 in this version");
#endif

    if (blockSize > MAX_BITSTREAM_BLOCK_SIZE) {
        std::stringstream ss;
        ss << "The block size must be at most " << (MAX_BITSTREAM_BLOCK_SIZE >> 20) << " MB";
        throw invalid_argument(ss.str());
    }

    if (blockSize < MIN_BITSTREAM_BLOCK_SIZE) {
        std::stringstream ss;
        ss << "The block size must be at least " << MIN_BITSTREAM_BLOCK_SIZE;
        throw invalid_argument(ss.str());
    }

    if ((blockSize & -16) != blockSize)
        throw invalid_argument("The block size must be a multiple of 16");

    _blockId = 0;
    _bufferId = 0;
    _blockSize = blockSize;
    _bufferThreshold = blockSize;
    _inputSize = fileSize;
    const int nbBlocks = (_inputSize == 0) ? 0 : int((_inputSize + int64(blockSize - 1)) / int64(blockSize));
    _nbInputBlocks = min(nbBlocks, MAX_CONCURRENCY - 1);
    _headless = headerless;
    _initialized = false;
    _closed = false;
    _obs = new DefaultOutputBitStream(os, DEFAULT_BUFFER_SIZE);
    _entropyType = EntropyEncoderFactory::getType(entropy.c_str());
    _transformType = TransformFactory<byte>::getType(transform.c_str());

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

    _jobs = tasks;
    _buffers = new SliceArray<byte>*[2 * _jobs];
    _ctx.putInt("blockSize", _blockSize);
    _ctx.putInt("checksum", checksum);
    _ctx.putString("entropy", entropy);
    _ctx.putString("transform", transform);
    _ctx.putInt("bsVersion", BITSTREAM_FORMAT_VERSION);

    // Allocate first buffer and add padding for incompressible blocks
    const int bufSize = max(_blockSize + (_blockSize >> 3), DEFAULT_BUFFER_SIZE);
    _buffers[0] = new SliceArray<byte>(new byte[bufSize], bufSize, 0);
    _buffers[_jobs] = new SliceArray<byte>(nullptr, 0, 0);

    for (int i = 1; i < _jobs; i++) {
       _buffers[i] = new SliceArray<byte>(nullptr, 0, 0);
       _buffers[i + _jobs] = new SliceArray<byte>(nullptr, 0, 0);
    }
}

CompressedOutputStream::CompressedOutputStream(OutputStream& os, Context& ctx, bool headerless)
    : OutputStream(os.rdbuf())
    , _ctx(ctx)
{
    int tasks = ctx.getInt("jobs", 1);

#ifdef CONCURRENCY_ENABLED
    if ((tasks <= 0) || (tasks > MAX_CONCURRENCY)) {
        stringstream ss;
        ss << "The number of jobs must be in [1.." << MAX_CONCURRENCY << "], got " << tasks;
        throw invalid_argument(ss.str());
    }

    _pool = _ctx.getPool(); // can be null
#else
    if (tasks != 1)
        throw invalid_argument("The number of jobs is limited to 1 in this version");
#endif

    int blockSize = ctx.getInt("blockSize");

    if (blockSize > MAX_BITSTREAM_BLOCK_SIZE) {
        std::stringstream ss;
        ss << "The block size must be at most " << (MAX_BITSTREAM_BLOCK_SIZE >> 20) << " MB";
        throw invalid_argument(ss.str());
    }

    if (blockSize < MIN_BITSTREAM_BLOCK_SIZE) {
        std::stringstream ss;
        ss << "The block size must be at least " << MIN_BITSTREAM_BLOCK_SIZE;
        throw invalid_argument(ss.str());
    }

    if ((blockSize & -16) != blockSize)
        throw invalid_argument("The block size must be a multiple of 16");

    _inputSize = ctx.getLong("fileSize", 0);
    const int nbBlocks = (_inputSize == 0) ? 0 : int((_inputSize + int64(blockSize - 1)) / int64(blockSize));
    _nbInputBlocks = min(nbBlocks, MAX_CONCURRENCY - 1);
    _jobs = tasks;
    _blockId = 0;
    _bufferId = 0;
    _blockSize = blockSize;
    _bufferThreshold = blockSize;
    _initialized = false;
    _closed = false;
    _headless = headerless;
    _obs = new DefaultOutputBitStream(os, DEFAULT_BUFFER_SIZE);
    string entropyCodec = ctx.getString("entropy");
    string transform = ctx.getString("transform");
    _entropyType = EntropyEncoderFactory::getType(entropyCodec.c_str());
    _transformType = TransformFactory<byte>::getType(transform.c_str());
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

    _ctx.putInt("bsVersion", BITSTREAM_FORMAT_VERSION);
    _buffers = new SliceArray<byte>*[2 * _jobs];

    // Allocate first buffer and add padding for incompressible blocks
    const int bufSize = max(_blockSize + (_blockSize >> 3), DEFAULT_BUFFER_SIZE);
    _buffers[0] = new SliceArray<byte>(new byte[bufSize], bufSize, 0);
    _buffers[_jobs] = new SliceArray<byte>(nullptr, 0, 0);

    for (int i = 1; i < _jobs; i++) {
       _buffers[i] = new SliceArray<byte>(nullptr, 0, 0);
       _buffers[i + _jobs] = new SliceArray<byte>(nullptr, 0, 0);
    }
}

CompressedOutputStream::~CompressedOutputStream()
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
    delete _obs;

    if (_hasher32 != nullptr) {
        delete _hasher32;
        _hasher32 = nullptr;
    }

    if (_hasher64 != nullptr) {
        delete _hasher64;
        _hasher64 = nullptr;
    }
}

void CompressedOutputStream::writeHeader()
{
    if ((_headless == true) || (_initialized.exchange(true, memory_order_acquire)))
        return;

    if (_obs->writeBits(BITSTREAM_TYPE, 32) != 32)
        throw IOException("Cannot write bitstream type to header", Error::ERR_WRITE_FILE);

    if (_obs->writeBits(BITSTREAM_FORMAT_VERSION, 4) != 4)
        throw IOException("Cannot write bitstream version to header", Error::ERR_WRITE_FILE);

    uint ckSize = 0;

    if (_hasher32 != nullptr)
        ckSize = 1;
    else if (_hasher64 != nullptr)
        ckSize = 2;

    if (_obs->writeBits(ckSize, 2) != 2)
        throw IOException("Cannot write block checksum size to header", Error::ERR_WRITE_FILE);

    if (_obs->writeBits(_entropyType, 5) != 5)
        throw IOException("Cannot write entropy type to header", Error::ERR_WRITE_FILE);

    if (_obs->writeBits(_transformType, 48) != 48)
        throw IOException("Cannot write transform types to header", Error::ERR_WRITE_FILE);

    if (_obs->writeBits(_blockSize >> 4, 28) != 28)
        throw IOException("Cannot write block size to header", Error::ERR_WRITE_FILE);

    // _inputSize not provided or >= 2^48 -> 0, <2^16 -> 1, <2^32 -> 2, <2^48 -> 3
    const uint szMask = ((_inputSize == 0) || (_inputSize >= (int64(1) << 48))) ? 0
        : (Global::log2(uint64(_inputSize)) >> 4) + 1;

    if (_obs->writeBits(szMask, 2) != 2)
        throw IOException("Cannot write size of input to header", Error::ERR_WRITE_FILE);

    if (szMask != 0) {
        if (_obs->writeBits(_inputSize, 16 * szMask) != 16 * szMask)
            throw IOException("Cannot write size of input to header", Error::ERR_WRITE_FILE);
    }

    const uint64 padding = 0;

    if (_obs->writeBits(padding, 15) != 15)
        throw IOException("Cannot write padding to header", Error::ERR_WRITE_FILE);

    uint32 seed = 0x01030507 * BITSTREAM_FORMAT_VERSION; // no const to avoid VS2008 warning
    const uint32 HASH = 0x1E35A7BD;
    uint32 cksum = HASH * seed;
    cksum ^= (HASH * uint32(~ckSize));
    cksum ^= (HASH * uint32(~_entropyType));
    cksum ^= (HASH * uint32((~_transformType) >> 32));
    cksum ^= (HASH * uint32(~_transformType));
    cksum ^= (HASH * uint32(~_blockSize));

    if (szMask != 0) {
        cksum ^= (HASH * uint32((~_inputSize) >> 32));
        cksum ^= (HASH * uint32(~_inputSize));
    }

    cksum = (cksum >> 23) ^ (cksum >> 3);

    if (_obs->writeBits(cksum, 24) != 24)
        throw IOException("Cannot write checksum to header", Error::ERR_WRITE_FILE);
}

bool CompressedOutputStream::addListener(Listener<Event>& bl)
{
    _listeners.push_back(&bl);
    return true;
}

bool CompressedOutputStream::removeListener(Listener<Event>& bl)
{
    std::vector<Listener<Event>*>::iterator it = find(_listeners.begin(), _listeners.end(), &bl);

    if (it == _listeners.end())
        return false;

    _listeners.erase(it);
    return true;
}

ostream& CompressedOutputStream::write(const char* data, streamsize length)
{
    int remaining = int(length);

    if (remaining < 0)
        throw IOException("Invalid buffer size");

    int off = 0;

    while (remaining > 0) {
        // Limit to number of available bytes in current buffer
        const int lenChunk = min(remaining, _bufferThreshold - _buffers[_bufferId]->_index);

        if (lenChunk > 0) {
            // Process a chunk of in-buffer data. No access to bitstream required
            memcpy(&_buffers[_bufferId]->_array[_buffers[_bufferId]->_index], &data[off], lenChunk);
            _buffers[_bufferId]->_index += lenChunk;
            off += lenChunk;
            remaining -= lenChunk;

            if (_buffers[_bufferId]->_index >= _bufferThreshold) {
                // Current write buffer is full
                const int nbTasks = _nbInputBlocks == 0 ? _jobs : min(_nbInputBlocks, _jobs);

                if (_bufferId + 1 < nbTasks) {
                    _bufferId++;
                    const int bufSize = max(_blockSize + (_blockSize >> 3), DEFAULT_BUFFER_SIZE);

                    if (_buffers[_bufferId]->_length == 0) {
                        if (_buffers[_bufferId]->_array != nullptr)
                           delete[] _buffers[_bufferId]->_array;

                        _buffers[_bufferId]->_array = new byte[bufSize];
                        _buffers[_bufferId]->_length = bufSize;
                    }

                    _buffers[_bufferId]->_index = 0;
                }
                else {
                    // If all buffers are full, time to encode
                    processBlock();
                }
            }

            if (remaining == 0)
                break;
        }

        put(data[off]);
        off++;
        remaining--;
    }

    return *this;
}

void CompressedOutputStream::close()
{
    if (_closed.exchange(true, memory_order_acquire))
        return;

    processBlock();

    try {
        // Write end block of size 0
        _obs->writeBits(uint64(0), 5); // write length-3 (5 bits max)
        _obs->writeBits(uint64(0), 3); // write 0 (3 bits)
        _obs->close();
    }
    catch (exception& e) {
        setstate(ios::badbit);
        throw ios_base::failure(e.what());
    }

    setstate(ios::eofbit);
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

void CompressedOutputStream::processBlock()
{
    writeHeader();

    // All buffers empty, nothing to do
    if (_buffers[0]->_index == 0)
        return;

    // Protect against future concurrent modification of the list of block listeners
    vector<Listener<Event>*> blockListeners(_listeners);
    vector<EncodingTask<EncodingTaskResult>*> tasks;

    try {
        int firstBlockId = _blockId.load(memory_order_relaxed);
        int nbTasks = _jobs;
        int jobsPerTask[MAX_CONCURRENCY];

        // Assign optimal number of tasks and jobs per task (if the number of blocks is available)
        if (nbTasks > 1) {
            // Limit the number of tasks if there are fewer blocks that _jobs
            // It allows more jobs per task and reduces memory usage.
            if (_nbInputBlocks != 0)
                nbTasks = min(_nbInputBlocks, _jobs);

            Global::computeJobsPerTask(jobsPerTask, _jobs, nbTasks);
        }
        else {
            jobsPerTask[0] = _jobs;
        }

        // Create as many tasks as non-empty buffers to encode
        for (int taskId = 0; taskId < nbTasks; taskId++) {
            const int dataLength = _buffers[taskId]->_index;

            if (dataLength == 0)
                break;

            Context copyCtx(_ctx);
            copyCtx.putInt("jobs", jobsPerTask[taskId]);
            copyCtx.putLong("tType", _transformType);
            copyCtx.putInt("eType", _entropyType);
            copyCtx.putInt("blockId", firstBlockId + taskId + 1);
            copyCtx.putInt("size", dataLength); // "size" is the actual block size, "blockSize" the provided one
            _buffers[taskId]->_index = 0;

            EncodingTask<EncodingTaskResult>* task = new EncodingTask<EncodingTaskResult>(_buffers[taskId],
                _buffers[_jobs + taskId],
                _obs, _hasher32, _hasher64,
                &_blockId, blockListeners, copyCtx);
            tasks.push_back(task);
        }

        if (tasks.size() == 1) {
            // Synchronous call
            EncodingTask<EncodingTaskResult>* task = tasks.back();
            tasks.pop_back();
            EncodingTaskResult res = task->run();

            if (res._error != 0)
                throw IOException(res._msg, res._error); // deallocate in catch block

            delete task;
        }
#ifdef CONCURRENCY_ENABLED
        else {
            vector<future<EncodingTaskResult> > futures;

            // Register task futures and launch tasks in parallel
            for (uint i = 0; i < tasks.size(); i++) {
                if (_pool == nullptr)
                    futures.push_back(async(launch::async, &EncodingTask<EncodingTaskResult>::run, tasks[i]));
                else
                    futures.push_back(_pool->schedule(&EncodingTask<EncodingTaskResult>::run, tasks[i]));
            }

            int error = 0;
            string msg;

            // Wait for tasks completion and check results
            for (uint i = 0; i < tasks.size(); i++) {
                EncodingTaskResult res = futures[i].get();

                if (error != 0)
                    continue;

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

        for (vector<EncodingTask<EncodingTaskResult>*>::iterator it = tasks.begin(); it != tasks.end(); ++it)
            delete *it;

        tasks.clear();
#endif

        _bufferId = 0;
    }
    catch (IOException&) {
        for (vector<EncodingTask<EncodingTaskResult>*>::iterator it = tasks.begin(); it != tasks.end(); ++it)
            delete *it;

        tasks.clear();
        throw; // rethrow
    }
    catch (BitStreamException& e) {
        for (vector<EncodingTask<EncodingTaskResult>*>::iterator it = tasks.begin(); it != tasks.end(); ++it)
            delete *it;

        tasks.clear();
        throw IOException(e.what(), e.error());
    }
    catch (exception& e) {
        for (vector<EncodingTask<EncodingTaskResult>*>::iterator it = tasks.begin(); it != tasks.end(); ++it)
            delete *it;

        tasks.clear();
        throw IOException(e.what(), Error::ERR_UNKNOWN);
    }
}

void CompressedOutputStream::notifyListeners(vector<Listener<Event>*>& listeners, const Event& evt)
{
    for (vector<Listener<Event>*>::iterator it = listeners.begin(); it != listeners.end(); ++it)
        (*it)->processEvent(evt);
}

template <class T>
EncodingTask<T>::EncodingTask(SliceArray<byte>* iBuffer, SliceArray<byte>* oBuffer,
    DefaultOutputBitStream* obs, XXHash32* hasher32, XXHash64* hasher64,
    ATOMIC_INT* processedBlockId, vector<Listener<Event>*>& listeners,
    const Context& ctx)
    : _obs(obs)
    , _listeners(listeners)
    , _ctx(ctx)
{
    _data = iBuffer;
    _buffer = oBuffer;
    _hasher32 = hasher32;
    _hasher64 = hasher64;
    _processedBlockId = processedBlockId;
}

// Encode mode + transformed entropy coded data
// mode | 0b1yy0xxxx => copy block
//      | 0b0yy00000 => size(size(block))-1
//  case 4 transforms or less
//      | 0b0001xxxx => transform sequence skip flags (1 means skip)
//  case more than 4 transforms
//      | 0b0yy00000 0bxxxxxxxx => transform sequence skip flags in next byte (1 means skip)
template <class T>
T EncodingTask<T>::run()
{
    const int blockId = _ctx.getInt("blockId");
    const int blockLength = _ctx.getInt("size");
    TransformSequence<byte>* transform = nullptr;
    EntropyEncoder* ee = nullptr;

    try {
        if (blockLength == 0) {
            // Last block (only block with 0 length)
            (*_processedBlockId)++;
            return T(blockId, 0, "Success");
        }

        byte mode = byte(0);
        int postTransformLength = blockLength;
        uint64 checksum = 0;
        uint64 tType = _ctx.getLong("tType");
        short eType = short(_ctx.getInt("eType"));
        Event::HashType hashType = Event::NO_HASH;

        // Compute block checksum
        if (_hasher32 != nullptr) {
            checksum = _hasher32->hash(&_data->_array[_data->_index], blockLength);
            hashType = Event::SIZE_32;
        }
        else if (_hasher64 != nullptr) {
            checksum = _hasher64->hash(&_data->_array[_data->_index], blockLength);
            hashType = Event::SIZE_64;
        }

        if (_listeners.size() > 0) {
            // Notify before transform
            Event evt(Event::BEFORE_TRANSFORM, blockId,
                int64(blockLength), clock(), checksum, hashType);
            CompressedOutputStream::notifyListeners(_listeners, evt);
        }

        if (blockLength <= CompressedOutputStream::SMALL_BLOCK_SIZE) {
            tType = TransformFactory<byte>::NONE_TYPE;
            eType = EntropyEncoderFactory::NONE_TYPE;
            mode |= CompressedOutputStream::COPY_BLOCK_MASK;
        }
        else {
            int checkSkip = _ctx.getInt("skipBlocks", 0);

            if (checkSkip != 0) {
                bool skip = Magic::isCompressed(Magic::getType(&_data->_array[_data->_index]));

                if (skip == false) {
                    uint histo[256] = { 0 };
                    Global::computeHistogram(&_data->_array[_data->_index], blockLength, histo);
                    const int entropy = Global::computeFirstOrderEntropy1024(blockLength, histo);
                    skip = entropy >= EntropyUtils::INCOMPRESSIBLE_THRESHOLD;
                    //_ctx.putString("histo0", toString(histo, 256));
                }

                if (skip == true) {
                    tType = TransformFactory<byte>::NONE_TYPE;
                    eType = EntropyEncoderFactory::NONE_TYPE;
                    mode |= CompressedOutputStream::COPY_BLOCK_MASK;
                }
            }
        }

        _ctx.putInt("size", blockLength);
        transform = TransformFactory<byte>::newTransform(_ctx, tType);
        const int requiredSize = transform->getMaxEncodedLength(blockLength);

        if (blockLength >= 4) {
           uint magic = Magic::getType(&_data->_array[_data->_index]);

           if (Magic::isCompressed(magic) == true)
               _ctx.putInt("dataType", Global::BIN);
           else if (Magic::isMultimedia(magic) == true)
               _ctx.putInt("dataType", Global::MULTIMEDIA);
           else if (Magic::isExecutable(magic) == true)
               _ctx.putInt("dataType", Global::EXE);
        }

        if (_buffer->_length < requiredSize) {
            if (_buffer->_array != nullptr)
               delete[] _buffer->_array;

            _buffer->_array = new byte[requiredSize];
            _buffer->_length = requiredSize;
        }

        // Forward transform (ignore error, encode skipFlags)
        // _data->_length is at least blockLength
        _buffer->_index = 0;
        transform->forward(*_data, *_buffer, blockLength);
        const int nbTransforms = transform->getNbTransforms();
        const byte skipFlags = transform->getSkipFlags();
        delete transform;
        transform = nullptr;
        postTransformLength = _buffer->_index;

        if (postTransformLength < 0) {
            _processedBlockId->store(CompressedOutputStream::CANCEL_TASKS_ID, memory_order_release);
            return T(blockId, Error::ERR_WRITE_FILE, "Invalid transform size");
        }

        _ctx.putInt("size", postTransformLength);
        const int dataSize = (postTransformLength < 256) ? 1 : (Global::_log2(uint32(postTransformLength)) >> 3) + 1;

        if (dataSize > 4) {
            _processedBlockId->store(CompressedOutputStream::CANCEL_TASKS_ID, memory_order_release);
            return T(blockId, Error::ERR_WRITE_FILE, "Invalid block data length");
        }

        // Record size of 'block size' - 1 in bytes
        mode |= byte(((dataSize - 1) & 0x03) << 5);

        if (_listeners.size() > 0) {
            // Notify after transform
            Event evt(Event::AFTER_TRANSFORM, blockId,
                int64(postTransformLength), clock(), checksum, hashType);
            CompressedOutputStream::notifyListeners(_listeners, evt);
        }

        const int bufSize = max(CompressedOutputStream::DEFAULT_BUFFER_SIZE,
                                max(postTransformLength, blockLength + (blockLength >> 3)));

        if (_data->_length < bufSize) {
            // Rare case where the transform expanded the input or
            // entropy coder may expand size.
            delete[] _data->_array;
            _data->_length = bufSize;
            _data->_array = new byte[_data->_length];
        }

        _data->_index = 0;
        ostreambuf<char> buf(reinterpret_cast<char*>(&_data->_array[_data->_index]), streamsize(_data->_length));
        ostream os(&buf);
        DefaultOutputBitStream obs(os);

        // Write block 'header' (mode + compressed length)
        if (((mode & CompressedOutputStream::COPY_BLOCK_MASK) != byte(0)) || (nbTransforms <= 4)) {
            mode |= byte(skipFlags >> 4);
            obs.writeBits(uint64(mode), 8);
        }
        else {
            mode |= CompressedOutputStream::TRANSFORMS_MASK;
            obs.writeBits(uint64(mode), 8);
            obs.writeBits(uint64(skipFlags), 8);
        }

        obs.writeBits(postTransformLength, 8 * dataSize);

        // Write checksum
        if (_hasher32 != nullptr)
            obs.writeBits(checksum, 32);
        else if (_hasher64 != nullptr)
            obs.writeBits(checksum, 64);

        if (_listeners.size() > 0) {
            // Notify before entropy
            Event evt(Event::BEFORE_ENTROPY, blockId,
                int64(postTransformLength), clock(), checksum, hashType);
            CompressedOutputStream::notifyListeners(_listeners, evt);
        }

        // Each block is encoded separately
        // Rebuild the entropy encoder to reset block statistics
        ee = EntropyEncoderFactory::newEncoder(obs, _ctx, eType);

        // Entropy encode block
        if (ee->encode(_buffer->_array, 0, postTransformLength) != postTransformLength) {
            delete ee;
            _processedBlockId->store(CompressedOutputStream::CANCEL_TASKS_ID, memory_order_release);
            return T(blockId, Error::ERR_PROCESS_BLOCK, "Entropy coding failed");
        }

        // Dispose before processing statistics (may write to the bitstream)
        ee->dispose();
        delete ee;
        ee = nullptr;
        obs.close();
        uint64 written = obs.written();

        // Lock free synchronization
        while (true) {
            const int taskId = _processedBlockId->load(memory_order_acquire);

            if (taskId == CompressedOutputStream::CANCEL_TASKS_ID)
                return T(blockId, 0, "Canceled");

            if (taskId == blockId - 1)
                break;

            // Back-off improves performance
            CPU_PAUSE();
        }

        if (_listeners.size() > 0) {
            // Notify after entropy
            Event evt1(Event::AFTER_ENTROPY, blockId,
                int64((written + 7) >> 3), clock(), checksum, hashType);
            CompressedOutputStream::notifyListeners(_listeners, evt1);

#if !defined(_MSC_VER) || _MSC_VER > 1500
            if (_ctx.getInt("verbosity", 0) > 4) {
                string oName = _ctx.getString("outputName");

                if (oName.length() == 4) {
                    std::transform(oName.begin(), oName.end(), oName.begin(), ::toupper);
                }

                const int64 blockOffset = (oName != "NONE") ? _obs->tell() : _obs->written();
                Event evt2(Event::BLOCK_INFO, blockId,
                   int64((written + 7) >> 3), clock(), checksum, hashType, blockOffset, uint8(skipFlags));
                CompressedOutputStream::notifyListeners(_listeners, evt2);
            }
#endif
        }

        // Emit block size in bits (max size pre-entropy is 1 GB = 1 << 30 bytes)
        const uint lw = (written < 8) ? 3 : uint(Global::log2(uint32(written >> 3)) + 4);
        _obs->writeBits(lw - 3, 5); // write length-3 (5 bits max)
        _obs->writeBits(written, lw);

        // Emit data to shared bitstream
        for (uint n = 0; written > 0; ) {
            uint chkSize = uint(min(written, uint64(1) << 30));
            _obs->writeBits(&_data->_array[n], chkSize);
            n += ((chkSize + 7) >> 3);
            written -= uint64(chkSize);
        }

        // After completion of the entropy coding, increment the block id.
        // It unblocks the task processing the next block (if any).
        _processedBlockId->store(blockId, memory_order_release);

        return T(blockId, 0, "Success");
    }
    catch (exception& e) {
        // Make sure to unfreeze next block
        if (_processedBlockId->load(memory_order_acquire) == blockId - 1)
            _processedBlockId->store(blockId, memory_order_release);

        if (transform != nullptr)
            delete transform;

        if (ee != nullptr)
            delete ee;

        return T(blockId, Error::ERR_PROCESS_BLOCK, e.what());
    }
}

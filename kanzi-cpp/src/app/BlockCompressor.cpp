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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <time.h>
#include "BlockCompressor.hpp"
#include "InfoPrinter.hpp"
#include "../SliceArray.hpp"
#include "../transform/TransformFactory.hpp"
#include "../io/IOException.hpp"
#include "../io/IOUtil.hpp"
#include "../io/NullOutputStream.hpp"
#include "../util/Clock.hpp"
#include "../util/Printer.hpp"

#ifdef CONCURRENCY_ENABLED
#include <future>
#endif

using namespace kanzi;
using namespace std;

BlockCompressor::BlockCompressor(map<string, string>& args) THROW
{
    map<string, string>::iterator it;
    it = args.find("level");

    if (it == args.end()) {
        _level = -1;
    }
    else {
        _level = atoi(it->second.c_str());
        args.erase(it);

        if ((_level < 0) || (_level > 9))
            throw invalid_argument("Invalid compression level");
    }

    it = args.find("overwrite");

    if (it == args.end()) {
        _overwrite = false;
    }
    else {
        _overwrite = it->second == STR_TRUE;
        args.erase(it);
    }

    it = args.find("skipBlocks");

    if (it == args.end()) {
        _skipBlocks = false;
    }
    else {
        _skipBlocks = it->second == STR_TRUE;
        args.erase(it);
    }

    it = args.find("inputName");

    if (it == args.end()) {
        throw invalid_argument("Missing input name");
    }

    _inputName = (it->second == "") ? "STDIN" : it->second;
    args.erase(it);
    it = args.find("outputName");

    if (it == args.end()) {
        throw invalid_argument("Missing output name");
    }

    _outputName = ((it->second == "") && (_inputName == "STDIN")) ? "STDOUT" : it->second;
    args.erase(it);
    string strCodec;
    string strTransf;

    it = args.find("entropy");

    if (it == args.end()) {
        strCodec = "ANS0";
    }
    else {
        strCodec = it->second;
        args.erase(it);
    }

    if (_level >= 0) {
        string tranformAndCodec[2];
        getTransformAndCodec(_level, tranformAndCodec);
        strTransf = tranformAndCodec[0];
        strCodec = tranformAndCodec[1];
    }

    _codec = strCodec;
    it = args.find("block");

    if (it == args.end()) {
        switch (_level) {
        case 6:
            _blockSize = 2 * DEFAULT_BLOCK_SIZE;
            break;
        case 7:
            _blockSize = 4 * DEFAULT_BLOCK_SIZE;
            break;
        case 8:
            _blockSize = 4 * DEFAULT_BLOCK_SIZE;
            break;
        case 9:
            _blockSize = 8 * DEFAULT_BLOCK_SIZE;
            break;
        default:
            _blockSize = DEFAULT_BLOCK_SIZE;
        }
    }
    else {
        string strBlkSz = it->second;
        args.erase(it);

#ifdef _MSC_VER
        uint64 bl = uint64(_atoi64(strBlkSz.c_str()));
#else
        uint64 bl = uint64(atoll(strBlkSz.c_str()));
#endif

        if (bl < MIN_BLOCK_SIZE) {
            stringstream sserr;
            sserr << "Minimum block size is " << (MIN_BLOCK_SIZE / 1024) << " KB (";
            sserr << MIN_BLOCK_SIZE << " bytes), got " << strBlkSz.c_str();
            sserr << ((bl > 1) ? " bytes" : " byte");
            throw invalid_argument(sserr.str().c_str());
        }

        if (bl > MAX_BLOCK_SIZE) {
            stringstream sserr;
            sserr << "Maximum block size is " << (MAX_BLOCK_SIZE / (1024 * 1024 * 1024)) << " GB (";
            sserr << MAX_BLOCK_SIZE << " bytes), got " << strBlkSz.c_str() << " bytes";
            throw invalid_argument(sserr.str().c_str());
        }

        _blockSize = min((int(bl) + 15) & -16, MAX_BLOCK_SIZE);
    }

    it = args.find("transform");

    if (it == args.end()) {
        if (strTransf.length() == 0)
            strTransf = "BWT+RANK+ZRLT";
    }
    else {
        if (strTransf.length() == 0) {
            // Extract transform names. Curate input (EG. NONE+NONE+xxxx => xxxx)
            strTransf = TransformFactory<byte>::getName(TransformFactory<byte>::getType(it->second.c_str()));
        }

        args.erase(it);
    }

    _transform = strTransf;
    it = args.find("checksum");

    if (it == args.end()) {
        _checksum = false;
    }
    else {
        string str = it->second;
        _checksum = str == STR_TRUE;
        args.erase(it);
    }

    it = args.find("verbose");

    if (it == args.end()) {
        _verbosity = 1;
    }
    else {
        _verbosity = atoi(it->second.c_str());
        args.erase(it);
    }

    it = args.find("jobs");
    int concurrency = 0;

    if (it != args.end()) {
        concurrency = atoi(it->second.c_str());
        args.erase(it);
    }

#ifndef CONCURRENCY_ENABLED
    if (concurrency > 1)
        throw invalid_argument("The number of jobs is limited to 1 in this version");

    concurrency = 1;
#else
    if (concurrency == 0) {
       int cores = max(int(thread::hardware_concurrency()) / 2, 1); // Defaults to half the cores
       concurrency = min(cores, MAX_CONCURRENCY);   
    }
    else if (concurrency > MAX_CONCURRENCY) {
        stringstream ss;
        ss << "Warning: the number of jobs is too high, defaulting to " << MAX_CONCURRENCY << endl;
        Printer log(cout);
        log.println(ss.str().c_str(), _verbosity > 0);
        concurrency = MAX_CONCURRENCY;
    }
#endif

    _jobs = concurrency;

    it = args.find("fileReorder");

    if (it == args.end()) {
        _reorderFiles = true;
    }
    else {
        string str = it->second;
        _reorderFiles = str == STR_TRUE;
        args.erase(it);
    }

    it = args.find("noDotFile");

    if (it == args.end()) {
        _noDotFile = false;
    }
    else {
        string str = it->second;
        _noDotFile = str == STR_TRUE;
        args.erase(it);
    }

    it = args.find("autoBlock");

    if (it == args.end()) {
        _autoBlockSize = false;
    }
    else {
        _autoBlockSize = it->second == STR_TRUE;
        args.erase(it);
    }

    if ((_verbosity > 0) && (args.size() > 0)) {
        Printer log(cout);

        for (it = args.begin(); it != args.end(); ++it) {
            stringstream ss;
            ss << "Warning: ignoring invalid option [" << it->first << "]";
            log.println(ss.str().c_str(), true);
        }
    }
}

BlockCompressor::~BlockCompressor()
{
    dispose();
    _listeners.clear();
}

void BlockCompressor::dispose()
{
}

int BlockCompressor::compress(uint64& outputSize)
{
    vector<FileData> files;
    Clock stopClock;
    int nbFiles = 1;
    Printer log(cout);
    stringstream ss;
    string str = _inputName;
    transform(str.begin(), str.end(), str.begin(), ::toupper);
    bool isStdIn = str == "STDIN";

    if (isStdIn == false) {
        vector<string> errors;
        string suffix(1, PATH_SEPARATOR);
        suffix += ".";
        bool isRecursive = (_inputName.length() < 2) 
           || (_inputName.substr(_inputName.length() - 2) != suffix);
        FileListConfig cfg = { isRecursive, false, false, _noDotFile };
        createFileList(_inputName, files, cfg, errors);

        if (files.size() == 0) {
            cerr << "Cannot access input file '" << _inputName << "'" << endl;
            return Error::ERR_OPEN_FILE;
        }

        if (errors.size() > 0) {
            for (size_t i = 0; i < errors.size(); i++)
               cerr << errors[i] << endl;

            return Error::ERR_OPEN_FILE;
        }

        nbFiles = int(files.size());
        string strFiles = (nbFiles > 1) ? " files" : " file";
        ss << nbFiles << strFiles << " to compress\n";
        log.println(ss.str().c_str(), _verbosity > 0);
        ss.str(string());
    }

    string upperOutputName = _outputName;
    transform(upperOutputName.begin(), upperOutputName.end(), upperOutputName.begin(), ::toupper);
    bool isStdOut = upperOutputName == "STDOUT";

    // Limit verbosity level when output is stdout
    // Logic is duplicated here to avoid dependency to Kanzi.cpp
    if (isStdOut == true)
        _verbosity = 0;

    // Limit verbosity level when files are processed concurrently
    if ((_verbosity > 1) && (_jobs > 1) && (nbFiles > 1)) {
        log.println("Warning: limiting verbosity to 1 due to concurrent processing of input files.\n", true);
        _verbosity = 1;
    }

    if (_verbosity > 2) {
        if (_autoBlockSize == true)
            ss << "Block size set to 'auto'";
        else
            ss << "Block size set to " << _blockSize << " bytes";

        log.println(ss.str().c_str(), true);
        ss.str(string());
        ss << "Verbosity set to " << _verbosity;
        log.println(ss.str().c_str(), true);
        ss.str(string());
        ss << "Overwrite set to " << (_overwrite ? "true" : "false");
        log.println(ss.str().c_str(), true);
        ss.str(string());
        ss << "Checksum set to " << (_checksum ? "true" : "false");
        log.println(ss.str().c_str(), true);
        ss.str(string());
        string etransform = _transform;
        transform(etransform.begin(), etransform.end(), etransform.begin(), ::toupper);
        ss << "Using " << ((etransform == "NONE") ? "no" : _transform) << " transform (stage 1)";
        log.println(ss.str().c_str(), true);
        ss.str(string());
        string ecodec = _codec;
        transform(ecodec.begin(), ecodec.end(), ecodec.begin(), ::toupper);
        ss << "Using " << ((ecodec == "NONE") ? "no" : _codec) << " entropy codec (stage 2)";
        log.println(ss.str().c_str(), true);
        ss.str(string());
        ss << "Using " << _jobs << " job" << ((_jobs > 1) ? "s" : "");
        log.println(ss.str().c_str(), true);
        ss.str(string());
    }

    InfoPrinter listener(_verbosity, InfoPrinter::ENCODING, cout);

    if (_verbosity > 2)
        addListener(listener);

    int res = 0;
    uint64 read = 0;
    uint64 written = 0;

    bool inputIsDir = false;
    string formattedOutName = _outputName;
    string formattedInName = _inputName;
    bool specialOutput = (isStdOut == true) || (upperOutputName == "NONE");

    // Need to strip path separator at the end to make 'stat()' happy
    if ((formattedOutName.size() > 1) && (formattedOutName[formattedOutName.size() - 1] == PATH_SEPARATOR)) {
        formattedOutName.resize(formattedOutName.size() - 1);
    }

    if (isStdIn == false) {
        struct STAT buffer;

        if ((formattedInName.size() > 1) && (formattedInName[formattedInName.size() - 1] == PATH_SEPARATOR)) {
            formattedInName.resize(formattedInName.size() - 1);
        }

        if (STAT(formattedInName.c_str(), &buffer) != 0) {
            cerr << "Cannot access input file '" << formattedInName << "'" << endl;
            return Error::ERR_OPEN_FILE;
        }

        if ((buffer.st_mode & S_IFDIR) != 0) {
            inputIsDir = true;

            if ((formattedInName.size() != 0) && (formattedInName[formattedInName.size() - 1] == '.')) {
                formattedInName.resize(formattedInName.size() - 1);
            }

            if ((formattedInName.size() != 0) && (formattedInName[formattedInName.size() - 1] != PATH_SEPARATOR)) {
                formattedInName += PATH_SEPARATOR;
            }

            if ((formattedOutName.size() != 0) && (specialOutput == false)) {
                if (STAT(formattedOutName.c_str(), &buffer) != 0) {
                    cerr << "Output must be an existing directory (or 'NONE')" << endl;
                    return Error::ERR_OPEN_FILE;
                }

                if ((buffer.st_mode & S_IFDIR) == 0) {
                    cerr << "Output must be a directory (or 'NONE')" << endl;
                    return Error::ERR_CREATE_FILE;
                }

                formattedOutName += PATH_SEPARATOR;
            }
        }
        else {
            if ((formattedOutName.size() != 0) && (specialOutput == false)) {
                if ((STAT(formattedOutName.c_str(), &buffer) != 0) && ((buffer.st_mode & S_IFDIR) != 0)) {
                    cerr << "Output must be a file (or 'NONE')" << endl;
                    return Error::ERR_CREATE_FILE;
                }
            }
        }
    }

#ifdef CONCURRENCY_ENABLED
    ThreadPool pool(_jobs);
    Context ctx(&pool);
#else
    Context ctx;
#endif
    ctx.putInt("verbosity", _verbosity);
    ctx.putInt("overwrite", (_overwrite == true) ? 1 : 0);
    ctx.putString("skipBlocks", (_skipBlocks == true) ? STR_TRUE : STR_FALSE);
    ctx.putString("checksum", (_checksum == true) ? STR_TRUE : STR_FALSE);
    ctx.putString("codec", _codec);
    ctx.putString("transform", _transform);
    ctx.putString("extra", (_codec == "TPAQX") ? STR_TRUE : STR_FALSE);

    // Run the task(s)
    if (nbFiles == 1) {
        string oName = formattedOutName;
        string iName = "STDIN";

        if (isStdIn == true) {
            if (oName.length() == 0) {
                oName = "STDOUT";
            }
        } else {
            iName = files[0].fullPath();
            ctx.putLong("fileSize", files[0]._size);

            // Set the block size to optimize compression ratio when possible
            if ((_autoBlockSize == true) && (_jobs > 0)) {
                const int64 bl = files[0]._size / _jobs;
                _blockSize = int(max(min((bl + 63) & ~63, int64(MAX_BLOCK_SIZE)), int64(MIN_BLOCK_SIZE)));
            }

            if (oName.length() == 0) {
                oName = iName + ".knz";
            }
            else if ((inputIsDir == true) && (specialOutput == false)) {
                oName = formattedOutName + iName.substr(formattedInName.size()) + ".knz";
            }
        }

        ctx.putString("inputName", iName);
        ctx.putString("outputName", oName);
        ctx.putInt("blockSize", _blockSize);
        ctx.putInt("jobs", _jobs);
        FileCompressTask<FileCompressResult> task(ctx, _listeners);
        FileCompressResult fcr = task.run();
        res = fcr._code;
        read = fcr._read;
        written = fcr._written;

        if (res != 0) {
            cerr << fcr._errMsg << endl;
        }
    }
    else {
        vector<FileCompressTask<FileCompressResult>*> tasks;
        int* jobsPerTask = new int[nbFiles];
        Global::computeJobsPerTask(jobsPerTask, _jobs, nbFiles);
        int n = 0;

        if (_reorderFiles == true)
            sortFilesByPathAndSize(files, true);

        // Create one task per file
        for (int i = 0; i < nbFiles; i++) {
            string oName = formattedOutName;
            string iName = files[i].fullPath();

            if (oName.length() == 0) {
                oName = iName + ".knz";
            }
            else if ((inputIsDir == true) && (specialOutput == false)) {
                oName = formattedOutName + iName.substr(formattedInName.size()) + ".knz";
            }

            // Set the block size to optimize compression ratio when possible
            if ((_autoBlockSize == true) && (_jobs > 0)) {
                const int64 bl = files[i]._size / _jobs;
                _blockSize = int(max(min((bl + 63) & ~63, int64(MAX_BLOCK_SIZE)), int64(MIN_BLOCK_SIZE)));
            }

            Context taskCtx(ctx);
            taskCtx.putLong("fileSize", files[i]._size);
            taskCtx.putString("inputName", iName);
            taskCtx.putString("outputName", oName);
            taskCtx.putInt("blockSize", _blockSize);
            taskCtx.putInt("jobs", jobsPerTask[n++]);
            ss.str(string());
            FileCompressTask<FileCompressResult>* task = new FileCompressTask<FileCompressResult>(taskCtx, _listeners);
            tasks.push_back(task);
        }

        bool doConcurrent = _jobs > 1;

#ifdef CONCURRENCY_ENABLED
        if (doConcurrent) {
            vector<FileCompressWorker<FileCompressTask<FileCompressResult>*, FileCompressResult>*> workers;
            vector<future<FileCompressResult> > results;
            BoundedConcurrentQueue<FileCompressTask<FileCompressResult>*> queue(nbFiles, &tasks[0]);

            // Create one worker per job and run it. A worker calls several tasks sequentially.
            for (int i = 0; i < _jobs; i++) {
                workers.push_back(new FileCompressWorker<FileCompressTask<FileCompressResult>*, FileCompressResult>(&queue));
                results.push_back(pool.schedule(&FileCompressWorker<FileCompressTask<FileCompressResult>*, FileCompressResult>::run, workers[i]));
            }

            // Wait for results
            for (int i = 0; i < _jobs; i++) {
                FileCompressResult fcr = results[i].get();
                res = fcr._code;
                read += fcr._read;
                written += fcr._written;

                if (res != 0) {
                    cerr << fcr._errMsg << endl;
                    // Exit early by telling the workers that the queue is empty
                    queue.clear();
                }
            }

            for (int i = 0; i < _jobs; i++)
                delete workers[i];
        }
#endif

        if (!doConcurrent) {
            for (uint i = 0; i < tasks.size(); i++) {
                FileCompressResult fcr = tasks[i]->run();
                res = fcr._code;
                read += fcr._read;
                written += fcr._written;

                if (res != 0) {
                    cerr << fcr._errMsg << endl;
                    break;
                }
            }
        }

        delete[] jobsPerTask;

        for (int i = 0; i < nbFiles; i++)
            delete tasks[i];
    }

    stopClock.stop();

    if (nbFiles > 1) {
        if (_verbosity > 0) {
            double delta = stopClock.elapsed();
            log.println("", true);
            ss << "Total compression time: ";

            if (delta >= 1e5) {
                ss.precision(1);
                ss.setf(ios::fixed);
                ss << (delta / 1000) << " s";
            }
            else {
                ss << int(delta) << " ms";
            }

            log.println(ss.str().c_str(), true);
            ss.str(string());
            ss << "Total output size: " << written << ((written > 1) ? " bytes" : " byte");
            log.println(ss.str().c_str(), true);
            ss.str(string());
        }

        if (read > 0) {
            ss << "Compression ratio: " << float(written) / float(read);
            log.println(ss.str().c_str(), _verbosity > 0);
            ss.str(string());
        }
    }

    if (_verbosity > 2)
        removeListener(listener);

    outputSize += written;
    return res;
}

bool BlockCompressor::addListener(Listener& bl)
{
    _listeners.push_back(&bl);
    return true;
}

bool BlockCompressor::removeListener(Listener& bl)
{
    std::vector<Listener*>::iterator it = find(_listeners.begin(), _listeners.end(), &bl);

    if (it == _listeners.end())
        return false;

    _listeners.erase(it);
    return true;
}

void BlockCompressor::notifyListeners(vector<Listener*>& listeners, const Event& evt)
{
    for (vector<Listener*>::iterator it = listeners.begin(); it != listeners.end(); ++it)
        (*it)->processEvent(evt);
}

void BlockCompressor::getTransformAndCodec(int level, string tranformAndCodec[2])
{
    switch (level) {
    case 0:
        tranformAndCodec[0] = "NONE";
        tranformAndCodec[1] = "NONE";
        break;

    case 1:
        tranformAndCodec[0] = "PACK+LZ";
        tranformAndCodec[1] = "NONE";
        break;

    case 2:
        tranformAndCodec[0] = "PACK+LZ";
        tranformAndCodec[1] = "HUFFMAN";
        break;

    case 3:
        tranformAndCodec[0] = "TEXT+UTF+PACK+MM+LZX";
        tranformAndCodec[1] = "HUFFMAN";
        break;

    case 4:
        tranformAndCodec[0] = "TEXT+UTF+EXE+PACK+MM+ROLZ";
        tranformAndCodec[1] = "NONE";
        break;

    case 5:
        tranformAndCodec[0] = "TEXT+UTF+BWT+RANK+ZRLT";
        tranformAndCodec[1] = "ANS0";
        break;

    case 6:
        tranformAndCodec[0] = "TEXT+UTF+BWT+SRT+ZRLT";
        tranformAndCodec[1] = "FPAQ";
        break;

    case 7:
        tranformAndCodec[0] = "LZP+TEXT+UTF+BWT+LZP";
        tranformAndCodec[1] = "CM";
        break;

    case 8:
        tranformAndCodec[0] = "EXE+RLT+TEXT+UTF";
        tranformAndCodec[1] = "TPAQ";
        break;

    case 9:
        tranformAndCodec[0] = "EXE+RLT+TEXT+UTF";
        tranformAndCodec[1] = "TPAQX";
        break;

    default:
        tranformAndCodec[0] = "Unknown";
        tranformAndCodec[1] = "Unknown";
    }
}

template <class T>
FileCompressTask<T>::FileCompressTask(Context& ctx, vector<Listener*>& listeners)
    : _ctx(ctx)
{
    _listeners = listeners;
    _is = nullptr;
    _cos = nullptr;
}

template <class T>
T FileCompressTask<T>::run()
{
    Printer log(cout);
    int verbosity = _ctx.getInt("verbosity");
    string inputName = _ctx.getString("inputName");
    string outputName = _ctx.getString("outputName");
    stringstream ss;

    if (verbosity > 2) {
        ss << "Input file name set to '" << inputName << "'";
        log.println(ss.str().c_str(), true);
        ss.str(string());
        ss << "Output file name set to '" << outputName << "'";
        log.println(ss.str().c_str(), true);
        ss.str(string());
    }

    bool overwrite = _ctx.getInt("overwrite") != 0;
    OutputStream* os = nullptr;

    try {
        string str = outputName;
        transform(str.begin(), str.end(), str.begin(), ::toupper);

        if (str.compare(0, 4, "NONE") == 0) {
            os = new NullOutputStream();
        }
        else if (str.compare(0, 6, "STDOUT") == 0) {
            os = &cout;
        }
        else {
            if (samePaths(inputName, outputName)) {
                stringstream sserr;
                sserr << "The input and output files must be different" << endl;
                return T(Error::ERR_CREATE_FILE, 0, 0, sserr.str().c_str());
            }

            struct STAT buffer;
            string path = outputName;
            replace(path.begin(), path.end(), '\\', '/');

            if (STAT(outputName.c_str(), &buffer) == 0) {
                if ((buffer.st_mode & S_IFDIR) != 0) {
                    return T(Error::ERR_OUTPUT_IS_DIR, 0, 0, "The output file is a directory");
                }

                if (overwrite == false) {
                    stringstream sserr;
                    sserr << "File '" << outputName << "' exists and the 'force' command "
                          << "line option has not been provided";
                    return T(Error::ERR_OVERWRITE_FILE, 0, 0, sserr.str().c_str());
                }

                // Delete output file to ensure consistent performance
                remove(outputName.c_str());
            }

            os = new ofstream(outputName.c_str(), ofstream::out | ofstream::binary);

            if (!*os) {
                if (overwrite == true) {
                    // Attempt to create the full folder hierarchy to file
                    string parentDir = outputName;
                    size_t idx = outputName.find_last_of(PATH_SEPARATOR);

                    if (idx != string::npos) {
                        parentDir.resize(idx);
                    }

                    if (mkdirAll(parentDir) == 0) {
                        os = new ofstream(outputName.c_str(), ofstream::binary);
                    }
                }

                if (!*os) {
                    stringstream sserr;
                    sserr << "Cannot open output file '" << outputName << "' for writing";
                    return T(Error::ERR_CREATE_FILE, 0, 0, sserr.str().c_str());
                }
            }
        }

        try {
            _cos = new CompressedOutputStream(*os, _ctx);

            for (uint i = 0; i < _listeners.size(); i++)
                _cos->addListener(*_listeners[i]);
        }
        catch (invalid_argument& e) {
            stringstream sserr;
            sserr << "Cannot create compressed stream: " << e.what();
            return T(Error::ERR_CREATE_COMPRESSOR, 0, 0, sserr.str().c_str());
        }
    }
    catch (exception& e) {
        stringstream sserr;
        sserr << "Cannot open output file '" << outputName << "' for writing: " << e.what();
        return T(Error::ERR_CREATE_FILE, 0, 0, sserr.str().c_str());
    }

    try {
        string str = inputName;
        transform(str.begin(), str.end(), str.begin(), ::toupper);

        if (str.compare(0, 5, "STDIN") == 0) {
            _is = &cin;
        }
        else {
            ifstream* ifs = new ifstream(inputName.c_str(), ifstream::in | ifstream::binary);

            if (!*ifs) {
                stringstream sserr;
                sserr << "Cannot open input file '" << inputName << "'";
                return T(Error::ERR_OPEN_FILE, 0, 0, sserr.str().c_str());
            }

            _is = ifs;
        }
    }
    catch (exception& e) {
        stringstream sserr;
        sserr << "Cannot open input file '" << inputName << "': " << e.what();
        return T(Error::ERR_OPEN_FILE, 0, 0, sserr.str().c_str());
    }

    // Compress
    ss << "\nCompressing " << inputName << " ...";
    log.println(ss.str().c_str(), verbosity > 1);
    log.println("\n", verbosity > 3);
    int64 read = 0;
    byte* buf = new byte[DEFAULT_BUFFER_SIZE];
    SliceArray<byte> sa(buf, DEFAULT_BUFFER_SIZE, 0);

    if (_listeners.size() > 0) {
        Event evt(Event::COMPRESSION_START, -1, int64(0), clock());
        BlockCompressor::notifyListeners(_listeners, evt);
    }

    Clock stopClock;

    try {
        while (true) {
            int len;

            try {
                _is->read(reinterpret_cast<char*>(&sa._array[0]), sa._length);
                len = (*_is) ? sa._length : int(_is->gcount());
            }
            catch (exception& e) {
                stringstream sserr;
                sserr << "Failed to read block from file '" << inputName << "': ";
                sserr << e.what() << endl;
                return T(Error::ERR_READ_FILE, read, _cos->getWritten(), sserr.str().c_str());
            }

            if (len <= 0)
                break;

            // Just write block to the compressed output stream !
            read += len;
            _cos->write(reinterpret_cast<const char*>(&sa._array[0]), len);
        }
    }
    catch (IOException& ioe) {
        delete[] buf;
        return T(ioe.error(), read, _cos->getWritten(), ioe.what());
    }
    catch (exception& e) {
        delete[] buf;
        stringstream sserr;
        sserr << "An unexpected condition happened. Exiting ..." << endl
              << e.what();
        return T(Error::ERR_UNKNOWN, read, _cos->getWritten(), sserr.str().c_str());
    }

    // Close streams to ensure all data are flushed
    dispose();

    uint64 encoded = _cos->getWritten();

    // os destructor will call close if ofstream
    if ((os != &cout) && (os != nullptr))
        delete os;

    // Clean up resources at the end of the method as the task may be
    // recycled in a threadpool and the destructor not called.
    delete _cos;
    _cos = nullptr;

    try {
        if ((_is != nullptr) && (_is != &cin)) {
            delete _is;
        }

        _is = nullptr;
    }
    catch (exception&) {
    }

    if (read == 0) {
        delete[] buf;
        stringstream sserr;
        sserr << "Input file " << inputName << " is empty ... nothing to do";
        log.println(sserr.str().c_str(), verbosity > 0);
        remove(outputName.c_str()); // best effort to delete output file, ignore return code
        return T(0, read, encoded, sserr.str().c_str());
    }

    stopClock.stop();
    double delta = stopClock.elapsed();

    if (verbosity >= 1) {
        log.println("", verbosity > 1);
        ss.str(string());

        if (verbosity > 1) {
            if (delta >= 1e5) {
                ss.precision(1);
                ss.setf(ios::fixed);
                ss << "Compressing:       " << (delta / 1000) << " s";
            }
            else {
                ss << "Compressing:       " << int(delta) << " ms";
            }

            log.println(ss.str().c_str(), true);
            ss.str(string());
            ss << "Input size:        " << read;
            log.println(ss.str().c_str(), true);
            ss.str(string());
            ss << "Output size:       " << encoded;
            log.println(ss.str().c_str(), true);
            ss.str(string());
            ss << "Compression ratio: " << (double(encoded) / double(read));
            log.println(ss.str().c_str(), true);
            ss.str(string());
        }

        if (verbosity == 1) {
            ss << "Compressing " << inputName << ": " << read << " => " << encoded;
            ss.precision(2);
            ss.setf(ios::fixed);
            const double r = double(encoded) / double(read);
            ss << " (" << (100 * r);

            if (delta >= 1e5) {
                ss.precision(1);
                ss << "%) in " << (delta / 1000) << " s";
            }
            else {
                ss << "%) in " << int(delta) << " ms";
            }

            log.println(ss.str().c_str(), true);
            ss.str(string());
        }

        if ((verbosity > 1) && (delta > 0)) {
            double b2KB = double(1000) / double(1024);
            ss << "Throughput (KB/s): " << uint(double(read) * b2KB / delta);
            log.println(ss.str().c_str(), true);
            ss.str(string());
        }

        log.println("", verbosity > 1);
    }

    if (_listeners.size() > 0) {
        Event evt(Event::COMPRESSION_END, -1, int64(encoded), clock());
        BlockCompressor::notifyListeners(_listeners, evt);
    }

    delete[] buf;
    return T(0, read, encoded, "");
}

template <class T>
FileCompressTask<T>::~FileCompressTask()
{
    dispose();

    if (_cos != nullptr) {
        delete _cos;
        _cos = nullptr;
    }

    try {
        if ((_is != nullptr) && (_is != &cin)) {
            delete _is;
        }

        _is = nullptr;
    }
    catch (exception&) {
        // Ignore: best effort
    }
}

// Close and flush streams. Do not deallocate resources. Idempotent.
template <class T>
void FileCompressTask<T>::dispose()
{
    try {
        if (_cos != nullptr) {
            _cos->close();
        }
    }
    catch (exception& e) {
        cerr << "Compression failure: " << e.what() << endl;
        exit(Error::ERR_WRITE_FILE);
    }

    // _is destructor will call close if ifstream
}

#ifdef CONCURRENCY_ENABLED
template <class T, class R>
R FileCompressWorker<T, R>::run()
{
    int res = 0;
    uint64 read = 0;
    uint64 written = 0;
    string errMsg;

    while (res == 0) {
        T* task = _queue->get();

        if (task == nullptr)
            break;

        R result = (*task)->run();
        res = result._code;
        read += result._read;
        written += result._written;

        if (res != 0) {
            errMsg += result._errMsg;
        }
    }

    return R(res, read, written, errMsg);
}
#endif

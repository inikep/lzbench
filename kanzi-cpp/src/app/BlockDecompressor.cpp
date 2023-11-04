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
#include "BlockDecompressor.hpp"
#include "InfoPrinter.hpp"
#include "../Global.hpp"
#include "../SliceArray.hpp"
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

BlockDecompressor::BlockDecompressor(map<string, string>& args)
{
    _blockSize = 0;
    map<string, string>::iterator it;
    it = args.find("overwrite");

    if (it == args.end()) {
        _overwrite = false;
    }
    else {
        _overwrite = it->second == STR_TRUE;
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
    it = args.find("verbose");

    if (it == args.end()) {
        _verbosity = 1;
    }
    else {
        _verbosity = atoi(it->second.c_str());
        args.erase(it);
    }

    it = args.find("from");

    if (it == args.end()) {
        _from = -1;
    }
    else {
        _from = atoi(it->second.c_str());
        args.erase(it);
    }

    it = args.find("to");

    if (it == args.end()) {
        _to = -1;
    }
    else {
        _to = atoi(it->second.c_str());
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

    if ((_verbosity > 0) && (args.size() > 0)) {
        Printer log(cout);

        for (it = args.begin(); it != args.end(); ++it) {
            stringstream ss;
            ss << "Warning: ignoring invalid option [" << it->first << "]";
            log.println(ss.str().c_str(), _verbosity > 0);
        }
    }
}

BlockDecompressor::~BlockDecompressor()
{
    dispose();
    _listeners.clear();
}

void BlockDecompressor::dispose()
{
}

int BlockDecompressor::decompress(uint64& inputSize)
{
    vector<FileData> files;
    uint64 read = 0;
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
        FileListConfig cfg = { isRecursive, false, false, false };
        createFileList(_inputName, files, cfg, errors);
        
        if (errors.size() > 0) {
            for (size_t i = 0; i < errors.size(); i++)
               cerr << errors[i] << endl;

            return Error::ERR_OPEN_FILE;
        }

        if (files.size() == 0) {
            cerr << "Cannot access input file '" << _inputName << "'" << endl;
            return Error::ERR_OPEN_FILE;
        }

        nbFiles = int(files.size());
        string strFiles = (nbFiles > 1) ? " files" : " file";
        ss << nbFiles << strFiles << " to decompress\n";
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
        ss << "Verbosity set to " << _verbosity;
        log.println(ss.str().c_str(), true);
        ss.str(string());
        ss << "Overwrite set to " << (_overwrite ? "true" : "false");
        log.println(ss.str().c_str(), true);
        ss.str(string());
        ss << "Using " << _jobs << " job" << ((_jobs > 1) ? "s" : "");
        log.println(ss.str().c_str(), true);
        ss.str(string());
    }

    InfoPrinter listener(_verbosity, InfoPrinter::DECODING, cout);

    if (_verbosity > 2)
        addListener(listener);

    int res = 0;
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
            inputIsDir = false;

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

    if (_from >= 0)
        ctx.putInt("from", _from);

    if (_to >= 0)
        ctx.putInt("to", _to);

    // Run the task(s)
    if (nbFiles == 1) {
        string oName = formattedOutName;
        string iName = "STDIN";


        if (isStdIn == true) {
            if (oName.length() == 0) {
                oName = "STDOUT";
            }
        }
        else {
            iName = files[0].fullPath();
            ctx.putLong("fileSize", files[0]._size);

            if (oName.length() == 0) {
                oName = iName + ".bak";
            }
            else if ((inputIsDir == true) && (specialOutput == false)) {
                oName = formattedOutName + iName.substr(formattedInName.size()) + ".bak";
            }
        }

        ctx.putString("inputName", iName);
        ctx.putString("outputName", oName);
        ctx.putInt("jobs", _jobs);
        FileDecompressTask<FileDecompressResult> task(ctx, _listeners);
        FileDecompressResult fdr = task.run();
        res = fdr._code;
        read = fdr._read;

        if (res != 0) {
            cerr << fdr._errMsg << endl;
        }
    }
    else {
        vector<FileDecompressTask<FileDecompressResult>*> tasks;
        int* jobsPerTask = new int[nbFiles];
        Global::computeJobsPerTask(jobsPerTask, _jobs, nbFiles);
        int n = 0;
        sortFilesByPathAndSize(files, true);

        //  Create one task per file
        for (int i = 0; i < nbFiles; i++) {
            string oName = formattedOutName;
            string iName = files[i].fullPath();

            if (oName.length() == 0) {
                oName = iName + ".bak";
            }
            else if ((inputIsDir == true) && (specialOutput == false)) {
                oName = formattedOutName + iName.substr(formattedInName.size()) + ".bak";
            }

            Context taskCtx(ctx);
            taskCtx.putLong("fileSize", files[i]._size);
            taskCtx.putString("inputName", iName);
            taskCtx.putString("outputName", oName);
            taskCtx.putInt("jobs", jobsPerTask[n++]);
            FileDecompressTask<FileDecompressResult>* task = new FileDecompressTask<FileDecompressResult>(taskCtx, _listeners);
            tasks.push_back(task);
        }

        bool doConcurrent = _jobs > 1;

#ifdef CONCURRENCY_ENABLED
        if (doConcurrent) {
            vector<FileDecompressWorker<FileDecompressTask<FileDecompressResult>*, FileDecompressResult>*> workers;
            vector<future<FileDecompressResult> > results;
            BoundedConcurrentQueue<FileDecompressTask<FileDecompressResult>*> queue(nbFiles, &tasks[0]);

            // Create one worker per job and run it. A worker calls several tasks sequentially.
            for (int i = 0; i < _jobs; i++) {
                workers.push_back(new FileDecompressWorker<FileDecompressTask<FileDecompressResult>*, FileDecompressResult>(&queue));
                results.push_back(pool.schedule(&FileDecompressWorker<FileDecompressTask<FileDecompressResult>*, FileDecompressResult>::run, workers[i]));
            }

            // Wait for results
            for (int i = 0; i < _jobs; i++) {
                FileDecompressResult fdr = results[i].get();
                res = fdr._code;
                read += fdr._read;

                if (res != 0) {
                    cerr << fdr._errMsg << endl;
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
                FileDecompressResult fdr = tasks[i]->run();
                res = fdr._code;
                read += fdr._read;

                if (res != 0) {
                    cerr << fdr._errMsg << endl;
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
            ss << "Total decompression time: ";

            if (delta >= 1e5) {
                ss.precision(1);
                ss.setf(ios::fixed);
                ss << (delta / 1000) << " s";
            }
            else {
                ss << int(delta) << " ms";
            }

            log.println(ss.str().c_str(), _verbosity > 0);
            ss.str(string());
            ss << "Total output size: " << read << ((read > 1) ? " bytes" : " byte");
            log.println(ss.str().c_str(), _verbosity > 0);
            ss.str(string());
        }
    }

    if (_verbosity > 2)
        removeListener(listener);

    inputSize += read;
    return res;
}

bool BlockDecompressor::addListener(Listener& bl)
{
    _listeners.push_back(&bl);
    return true;
}

bool BlockDecompressor::removeListener(Listener& bl)
{
    std::vector<Listener*>::iterator it = find(_listeners.begin(), _listeners.end(), &bl);

    if (it == _listeners.end())
        return false;

    _listeners.erase(it);
    return true;
}

void BlockDecompressor::notifyListeners(vector<Listener*>& listeners, const Event& evt)
{
    for (vector<Listener*>::iterator it = listeners.begin(); it != listeners.end(); ++it)
        (*it)->processEvent(evt);
}

template <class T>
FileDecompressTask<T>::FileDecompressTask(Context& ctx, vector<Listener*>& listeners)
    : _ctx(ctx)
{
    _listeners = listeners;
    _os = nullptr;
    _cis = nullptr;
}

template <class T>
FileDecompressTask<T>::~FileDecompressTask()
{
    dispose();

    if (_cis != nullptr) {
        delete _cis;
        _cis = nullptr;
    }

    try {
        if ((_os != nullptr) && (_os != &cout)) {
            delete _os;
        }

        _os = nullptr;
    }
    catch (exception&) {
        // Ignore: best effort
    }
}

template <class T>
T FileDecompressTask<T>::run()
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

    int64 read = 0;
    ss << "\nDecompressing " << inputName << " ...";
    log.println(ss.str().c_str(), verbosity > 1);
    log.println("\n", verbosity > 3);

    if (_listeners.size() > 0) {
        Event evt(Event::DECOMPRESSION_START, -1, int64(0), clock());
        BlockDecompressor::notifyListeners(_listeners, evt);
    }

    string str = outputName;
    transform(str.begin(), str.end(), str.begin(), ::toupper);

    if (str.compare(0, 4, "NONE") == 0) {
        _os = new NullOutputStream();
    }
    else if (str.compare(0, 6, "STDOUT") == 0) {
        _os = &cout;
    }
    else {
        try {
            if (samePaths(inputName, outputName)) {
                stringstream sserr;
                sserr << "The input and output files must be different";
                return T(Error::ERR_CREATE_FILE, 0, sserr.str().c_str());
            }

            struct STAT buffer;

            if (STAT(outputName.c_str(), &buffer) == 0) {
                if ((buffer.st_mode & S_IFDIR) != 0) {
                    stringstream sserr;
                    sserr << "The output file is a directory";
                    return T(Error::ERR_OUTPUT_IS_DIR, 0, sserr.str().c_str());
                }

                if (overwrite == false) {
                    stringstream sserr;
                    sserr << "File '" << outputName << "' exists and the 'force' command "
                          << "line option has not been provided";
                    return T(Error::ERR_OVERWRITE_FILE, 0, sserr.str().c_str());
                }

                // Delete output file to ensure consistent performance
                remove(outputName.c_str());
            }

            _os = new ofstream(outputName.c_str(), ofstream::out | ofstream::binary);

            if (!*_os) {
                if (overwrite == true) {
                    // Attempt to create the full folder hierarchy to file
                    string parentDir = outputName;
                    size_t idx = outputName.find_last_of(PATH_SEPARATOR);

                    if (idx != string::npos) {
                        parentDir.resize(idx);
                    }

                    if (mkdirAll(parentDir) == 0) {
                        _os = new ofstream(outputName.c_str(), ofstream::binary);
                    }
                }

                if (!*_os) {
                    stringstream sserr;
                    sserr << "Cannot open output file '" << outputName << "' for writing";
                    return T(Error::ERR_CREATE_FILE, 0, sserr.str().c_str());
                }
            }
        }
        catch (exception& e) {
            stringstream sserr;
            sserr << "Cannot open output file '" << outputName << "' for writing: " << e.what();
            return T(Error::ERR_CREATE_FILE, 0, sserr.str().c_str());
        }
    }

    InputStream* is;

    try {
        str = inputName;
        transform(str.begin(), str.end(), str.begin(), ::toupper);

        if (str.compare(0, 5, "STDIN") == 0) {
            is = &cin;
        }
        else {
            ifstream* ifs = new ifstream(inputName.c_str(), ifstream::in | ifstream::binary);

            if (!*ifs) {
                stringstream sserr;
                sserr << "Cannot open input file '" << inputName << "'";
                return T(Error::ERR_OPEN_FILE, 0, sserr.str().c_str());
            }

            is = ifs;
        }

        try {
            _cis = new CompressedInputStream(*is, _ctx);

            for (uint i = 0; i < _listeners.size(); i++)
                _cis->addListener(*_listeners[i]);
        }
        catch (invalid_argument& e) {
            stringstream sserr;
            sserr << "Cannot create compressed stream: " << e.what();
            return T(Error::ERR_CREATE_DECOMPRESSOR, 0, sserr.str().c_str());
        }
    }
    catch (exception& e) {
        stringstream sserr;
        sserr << "Cannot open input file '" << inputName << "': " << e.what();
        return T(Error::ERR_OPEN_FILE, _cis->getRead(), sserr.str().c_str());
    }

    Clock stopClock;
    byte* buf = new byte[DEFAULT_BUFFER_SIZE];

    try {
        SliceArray<byte> sa(buf, DEFAULT_BUFFER_SIZE, 0);
        int decoded = 0;

        // Decode next block
        do {
            _cis->read(reinterpret_cast<char*>(&sa._array[0]), sa._length);
            decoded = int(_cis->gcount());

            if (decoded < 0) {
                delete[] buf;
                stringstream sserr;
                sserr << "Reached end of stream";
                return T(Error::ERR_READ_FILE, _cis->getRead(), sserr.str().c_str());
            }

            try {
                if (decoded > 0) {
                    _os->write(reinterpret_cast<const char*>(&sa._array[0]), decoded);
                    read += decoded;
                }
            }
            catch (exception& e) {
                delete[] buf;
                stringstream sserr;
                sserr << "Failed to write decompressed block to file '" << outputName << "': " << e.what();
                return T(Error::ERR_READ_FILE, _cis->getRead(), sserr.str().c_str());
            }
        } while (decoded == sa._length);
    }
    catch (IOException& e) {
        // Close streams to ensure all data are flushed
        dispose();
        delete[] buf;
        stringstream sserr;

        if (_cis->eof()) {
            sserr << "Reached end of stream";
            return T(Error::ERR_READ_FILE, _cis->getRead(), sserr.str().c_str());
        }

        sserr << e.what();
        return T(e.error(), _cis->getRead(), sserr.str().c_str());
    }
    catch (exception& e) {
        // Close streams to ensure all data are flushed
        dispose();
        delete[] buf;
        stringstream sserr;

        if (_cis->eof()) {
            sserr << "Reached end of stream";
            return T(Error::ERR_READ_FILE, _cis->getRead(), sserr.str().c_str());
        }

        sserr << "An unexpected condition happened. Exiting ..." << endl
              << e.what();
        return T(Error::ERR_UNKNOWN, _cis->getRead(), sserr.str().c_str());
    }

    // Close streams to ensure all data are flushed
    dispose();

    uint64 decoded = _cis->getRead();

    // is destructor will call close if ifstream
    if ((is != &cin) && (is != nullptr))
        delete is;

    // Clean up resources at the end of the method as the task may be
    // recycled in a threadpool and the destructor not called.
    delete _cis;
    _cis = nullptr;

    try {
        if ((_os != nullptr) && (_os != &cout)) {
            delete _os;
        }

        _os = nullptr;
    }
    catch (exception&) {
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
                ss << "Decompressing:     " << (delta / 1000) << " s";
            }
            else {
                ss << "Decompressing:     " << int(delta) << " ms";
            }

            log.println(ss.str().c_str(), true);
            ss.str(string());
            ss << "Input size:        " << decoded;
            log.println(ss.str().c_str(), true);
            ss.str(string());
            ss << "Output size:       " << read;
            log.println(ss.str().c_str(), true);
            ss.str(string());
        }

        if (verbosity == 1) {
            ss << "Decompressing " << inputName << ": " << decoded << " => " << read;

            if (delta >= 1e5) {
                ss.precision(1);
                ss.setf(ios::fixed);
                ss << " bytes in " << (delta / 1000) << " s";
            }
            else {
                ss << " bytes in " << int(delta) << " ms";
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
        Event evt(Event::DECOMPRESSION_END, -1, int64(decoded), clock());
        BlockDecompressor::notifyListeners(_listeners, evt);
    }

    delete[] buf;
    return T(0, read, "");
}

// Close and flush streams. Do not deallocate resources. Idempotent.
template <class T>
void FileDecompressTask<T>::dispose()
{
    try {
        if (_cis != nullptr) {
            _cis->close();
        }
    }
    catch (exception& e) {
        cerr << "Decompression failure: " << e.what() << endl;
        exit(Error::ERR_WRITE_FILE);
    }

    // _os destructor will call close if ofstream
}

#ifdef CONCURRENCY_ENABLED
template <class T, class R>
R FileDecompressWorker<T, R>::run()
{
    int res = 0;
    uint64 read = 0;
    string errMsg;

    while (res == 0) {
        T* task = _queue->get();

        if (task == nullptr)
            break;

        R result = (*task)->run();
        res = result._code;
        read += result._read;

        if (res != 0) {
            errMsg += result._errMsg;
        }
    }

    return R(res, read, errMsg);
}
#endif

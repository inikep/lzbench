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

BlockDecompressor::BlockDecompressor(const Context& ctx) :
     _ctx(ctx)
{
    _blockSize = 0;
    _overwrite = _ctx.getInt("overwrite", 0) != 0;
    _ctx.putInt("overwrite", _overwrite ? 1 : 0);
    _verbosity = _ctx.getInt("verbosity", 1);
    _ctx.putInt("verbosity", _verbosity);
    _jobs = _ctx.getInt("jobs", 1);
    _ctx.putInt("jobs", _jobs);
    bool remove = _ctx.getInt("remove", 0) != 0;
    _ctx.putInt("remove", remove ? 1 : 0);
    _noDotFiles = _ctx.getInt("noDotFiles", 0) != 0;
    _ctx.putInt("noDotFiles", _noDotFiles ? 1 : 0);
    _noLinks = _ctx.getInt("noLinks", 0) != 0;
    _ctx.putInt("noLinks", _noLinks ? 1 : 0);

    if (_ctx.has("inputName") == false)
        throw invalid_argument("Missing input name");

    _inputName = _ctx.getString("inputName") == "" ? "STDIN" : _ctx.getString("inputName");

    if (Global::isReservedName(_inputName))
        throw invalid_argument("'" + _inputName + "' is a reserved name");

    if (_ctx.has("outputName") == false)
        throw invalid_argument("Missing output name");

    string str = _ctx.getString("outputName");
    _outputName = (str == "") && (_inputName == "STDIN") ? "STDOUT" : str;

    if (Global::isReservedName(_outputName))
        throw invalid_argument("'" + _outputName + "' is a reserved name");
}

BlockDecompressor::~BlockDecompressor()
{
    dispose();
    _listeners.clear();
}

int BlockDecompressor::decompress(uint64& inputSize)
{
    vector<FileData> files;
    uint64 read = 0;
    Clock stopClock;
    int nbFiles = 1;
    Printer log(cout);
    stringstream ss;
    string upperInputName = _inputName;
    transform(upperInputName.begin(), upperInputName.end(), upperInputName.begin(), ::toupper);
    bool isStdIn = upperInputName == "STDIN";

    if (isStdIn == false) {
        vector<string> errors;
        bool isRecursive = (_inputName.length() < 2) ||                                                                                                                  (_inputName[_inputName.length() - 2] != PATH_SEPARATOR) ||
            (_inputName[_inputName.length() - 1] != '.');
        FileListConfig cfg = { isRecursive, _noLinks, false, _noDotFiles };
        createFileList(_inputName, files, cfg, errors);

        if (errors.size() > 0) {
            for (size_t i = 0; i < errors.size(); i++)
               cerr << errors[i] << endl;

            return Error::ERR_OPEN_FILE;
        }

        if (files.size() == 0) {
            cerr << "Cannot find any file to decompress" << endl;
            return Error::ERR_OPEN_FILE;
        }

        nbFiles = int(files.size());
        string strFiles = nbFiles > 1 ? " files" : " file";
        ss << nbFiles << strFiles << " to decompress\n";
        log.println(ss.str(), _verbosity > 0);
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
        ss << "Verbosity: " << _verbosity << endl;
        ss << "Overwrite: " << (_overwrite ? "true" : "false") << endl;
        ss << "Using " << _jobs << " job" << (_jobs > 1 ? "s" : "") << endl;
        log.print(ss.str(), true);
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

    _ctx.putInt("verbosity", _verbosity);

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
            _ctx.putLong("fileSize", files[0]._size);

            if (oName.length() == 0) {
                oName = iName;

                if ((upperInputName.length() >= 4) && (upperInputName.substr(upperInputName.length() - 4) == ".KNZ"))
                    oName.resize(oName.length() - 4);
                else
                    oName = oName + ".bak";
            }
            else if ((inputIsDir == true) && (specialOutput == false)) {
                oName = formattedOutName + iName.substr(formattedInName.size());

                if ((upperInputName.length() >= 4) && (upperInputName.substr(upperInputName.length() - 4) == ".KNZ"))
                    oName.resize(oName.length() - 4);
                else
                    oName = oName + ".bak";
            }
        }

         _ctx.putString("inputName", iName);
         _ctx.putString("outputName", oName);
         FileDecompressTask<FileDecompressResult> task(_ctx, _listeners);
         FileDecompressResult fdr = task.run();
         res = fdr._code;
         read = fdr._read;

         if (res != 0) {
            cerr << fdr._errMsg << endl;
         }
    }
    else {
        vector<FileDecompressTask<FileDecompressResult>*> tasks;
#ifdef CONCURRENCY_ENABLED
        vector<int> jobsPerTask(nbFiles);
        Global::computeJobsPerTask(jobsPerTask.data(), _jobs, nbFiles);
#endif
        sortFilesByPathAndSize(files, true);

        //  Create one task per file
        for (int i = 0; i < nbFiles; i++) {
            string oName = formattedOutName;
            string iName = files[i].fullPath();
            upperInputName = iName;
            transform(upperInputName.begin(), upperInputName.end(), upperInputName.begin(), ::toupper);

            if (oName.length() == 0) {
                oName = iName;

                if ((upperInputName.length() >= 4) && (upperInputName.substr(upperInputName.length() - 4) == ".KNZ"))
                    oName.resize(oName.length() - 4);
                else
                    oName = oName + ".bak";
            }
            else if ((inputIsDir == true) && (specialOutput == false)) {
                oName = formattedOutName + iName.substr(formattedInName.size());

                if ((upperInputName.length() >= 4) && (upperInputName.substr(upperInputName.length() - 4) == ".KNZ"))
                    oName.resize(oName.length() - 4);
                else
                    oName = oName + ".bak";
            }

            Context taskCtx(_ctx);
            taskCtx.putLong("fileSize", files[i]._size);
            taskCtx.putString("inputName", iName);
            taskCtx.putString("outputName", oName);
#ifdef CONCURRENCY_ENABLED
            taskCtx.putInt("jobs", jobsPerTask[i]);
#else
            taskCtx.putInt("jobs", 1);
#endif
            FileDecompressTask<FileDecompressResult>* task = new FileDecompressTask<FileDecompressResult>(taskCtx, _listeners);
            tasks.push_back(task);
        }

        bool doConcurrent = _jobs > 1;

#ifdef CONCURRENCY_ENABLED
        if (doConcurrent) {
            vector<FileDecompressWorker<FDTask*, FileDecompressResult>*> workers;
            vector<future<FileDecompressResult> > results;
            BoundedConcurrentQueue<FDTask*> queue(nbFiles, &tasks[0]);

            // Create one worker per job and run it. A worker calls several tasks sequentially.
            for (int i = 0; i < _jobs; i++) {
                workers.push_back(new FileDecompressWorker<FileDecompressTask<FileDecompressResult>*, FileDecompressResult>(&queue));

                if (_ctx.getPool() == nullptr)
                    results.push_back(async(launch::async, &FileDecompressWorker<FDTask*, FileDecompressResult>::run, workers[i]));
                else
                    results.push_back(_ctx.getPool()->schedule(&FileDecompressWorker<FDTask*, FileDecompressResult>::run, workers[i]));
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

        for (int i = 0; i < nbFiles; i++)
            delete tasks[i];
    }

    stopClock.stop();

    if ((nbFiles > 1) && (_verbosity > 0)) {
        double delta = stopClock.elapsed();
        log.println("", true);
        ss << "Total decompression time: ";

        if (delta >= 1e5) {
            ss.precision(1);
            ss.setf(ios::fixed);
            ss << (delta / 1000) << " s" << endl;
        }
        else {
            ss << int(delta) << " ms" << endl;
        }

        ss << "Total output size: " << read << (read > 1 ? " bytes" : " byte") << endl;
        log.print(ss.str(), _verbosity > 0);
        ss.str(string());
    }

    if (_verbosity > 2)
        removeListener(listener);

    inputSize += read;
    return res;
}

bool BlockDecompressor::addListener(Listener<Event>& bl)
{
    _listeners.push_back(&bl);
    return true;
}

bool BlockDecompressor::removeListener(Listener<Event>& bl)
{
    std::vector<Listener<Event>*>::iterator it = find(_listeners.begin(), _listeners.end(), &bl);

    if (it == _listeners.end())
        return false;

    _listeners.erase(it);
    return true;
}

void BlockDecompressor::notifyListeners(vector<Listener<Event>*>& listeners, const Event& evt)
{
    for (vector<Listener<Event>*>::iterator it = listeners.begin(); it != listeners.end(); ++it)
        (*it)->processEvent(evt);
}

template <class T>
FileDecompressTask<T>::FileDecompressTask(const Context& ctx, vector<Listener<Event>*>& listeners)
    : _ctx(ctx)
    , _listeners(listeners)
{
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
        ss << "Input file name: '" << inputName << "'" << endl;
        ss << "Output file name: '" << outputName << "'" << endl;
        log.print(ss.str(), true);
        ss.str(string());
    }

    bool overwrite = _ctx.getInt("overwrite") != 0;

    int64 read = 0;
    ss << "\nDecompressing " << inputName << " ...";
    log.println(ss.str(), verbosity > 1);
    log.println("\n", verbosity > 3);

    if (_listeners.size() > 0) {
        Event evt(Event::DECOMPRESSION_START, -1, int64(0), clock());
        BlockDecompressor::notifyListeners(_listeners, evt);
    }

    string str = outputName;
    transform(str.begin(), str.end(), str.begin(), ::toupper);

#if defined(WIN32) || defined(_WIN32) || defined(_WIN64)
    bool checkOutputSize = str != "NUL";
#else
    bool checkOutputSize = str != "/DEV/NULL";
#endif

#define CLEANUP_DECOMP_OS  if ((_os != nullptr) && (_os != &cout)) { \
                              delete _os; \
                              _os = nullptr; \
                           }

    if (str == "NONE") {
        _os = new NullOutputStream();
        checkOutputSize = false;
    }
    else if (str == "STDOUT") {
        _os = &cout;
        checkOutputSize = false;
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

            ofstream* ofs = new ofstream(outputName.c_str(), ofstream::out | ofstream::binary);

            if (!*ofs) {
                string errMsg;

                if (overwrite == true) {
                    // Attempt to create the full folder hierarchy to file
                    string parentDir = outputName;
                    size_t idx = outputName.find_last_of(PATH_SEPARATOR);

                    if (idx != string::npos)
                        parentDir.resize(idx);

                    int rmkd = mkdirAll(parentDir);

                    if ((rmkd == 0) || (rmkd == EEXIST))  {
                        delete ofs;
                        ofs = new ofstream(outputName.c_str(), ofstream::binary);
                    }
                    else {
                        errMsg = strerror(rmkd);
                    }
                }

                if (!*ofs) {
                    delete ofs;
                    stringstream sserr;
                    sserr << "Cannot open output file '" << outputName << "' for writing";

                    if (errMsg != "")
                        sserr << ": " << errMsg;

                    return T(Error::ERR_CREATE_FILE, 0, sserr.str().c_str());
                }
            }

            _os = ofs;
        }
        catch (exception& e) {
            stringstream sserr;
            sserr << "Cannot open output file '" << outputName << "' for writing: " << e.what();
            return T(Error::ERR_CREATE_FILE, 0, sserr.str().c_str());
        }
    }

    InputStream* is = nullptr;

#define CLEANUP_DECOMP_IS  if ((is != nullptr) && (is != &cin)) { \
                              delete is; \
                              is = nullptr; \
                           }

    try {
        str = inputName;
        transform(str.begin(), str.end(), str.begin(), ::toupper);

        if (str == "STDIN") {
            is = &cin;
        }
        else {
            ifstream* ifs = new ifstream(inputName.c_str(), ifstream::in | ifstream::binary);

            if (!*ifs) {
                delete ifs;
                CLEANUP_DECOMP_OS
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
            CLEANUP_DECOMP_IS
            CLEANUP_DECOMP_OS
            stringstream sserr;
            sserr << "Cannot create compressed stream: " << e.what();
            return T(Error::ERR_CREATE_DECOMPRESSOR, 0, sserr.str().c_str());
        }
    }
    catch (exception& e) {
        CLEANUP_DECOMP_IS
        CLEANUP_DECOMP_OS
        stringstream sserr;
        sserr << "Cannot open input file '" << inputName << "': " << e.what();
        return T(Error::ERR_OPEN_FILE, _cis->getRead(), sserr.str().c_str());
    }

    Clock stopClock;
    static const int DEFAULT_BUFFER_SIZE = 65536;
    byte* buf = new byte[DEFAULT_BUFFER_SIZE];

    try {
        SliceArray<byte> sa(buf, DEFAULT_BUFFER_SIZE, 0);
        int decoded = 0;

        // Decode next block
        do {
            _cis->read(reinterpret_cast<char*>(&sa._array[0]), sa._length);
            decoded = int(_cis->gcount());

            if (decoded < 0) {
                dispose();
                const uint64 d = _cis->getRead();
                CLEANUP_DECOMP_IS
                CLEANUP_DECOMP_OS
                delete[] buf;
                delete _cis;
                _cis = nullptr;
                stringstream sserr;
                sserr << "Reached end of stream";
                return T(Error::ERR_READ_FILE, d, sserr.str().c_str());
            }

            try {
                if (decoded > 0) {
                    _os->write(reinterpret_cast<const char*>(&sa._array[0]), decoded);
                    read += decoded;
                }
            }
            catch (exception& e) {
                dispose();
                const uint64 d = _cis->getRead();
                CLEANUP_DECOMP_IS
                CLEANUP_DECOMP_OS
                delete[] buf;
                delete _cis;
                _cis = nullptr;
                stringstream sserr;
                sserr << "Failed to write decompressed block to file '" << outputName << "': " << e.what();
                return T(Error::ERR_READ_FILE, d, sserr.str().c_str());
            }
        } while (decoded == sa._length);
    }
    catch (IOException& e) {
        dispose();
        const uint64 d = _cis->getRead();
        bool isEOF = _cis->eof();
        CLEANUP_DECOMP_IS
        CLEANUP_DECOMP_OS
        delete[] buf;
        delete _cis;
        _cis = nullptr;

        if (isEOF == true)
            return T(Error::ERR_READ_FILE, d, "Reached end of stream");

        stringstream sserr;
        sserr << e.what();
        return T(e.error(), d, sserr.str().c_str());
    }
    catch (exception& e) {
        dispose();
        const uint64 d = _cis->getRead();
        bool isEOF = _cis->eof();
        CLEANUP_DECOMP_IS
        CLEANUP_DECOMP_OS
        delete[] buf;
        delete _cis;
        _cis = nullptr;

        if (isEOF == true)
            return T(Error::ERR_READ_FILE, d, "Reached end of stream");

        stringstream sserr;
        sserr << "An unexpected condition happened. Exiting ..." << endl << e.what();
        return T(Error::ERR_UNKNOWN, d, sserr.str().c_str());
    }

    // Close streams to ensure all data are flushed
    dispose();

    const uint64 decoded = _cis->getRead();
    const uint64 written = (checkOutputSize == true) ? uint64(_os->tellp()) : 0;

    // is destructor will call close if ifstream
    CLEANUP_DECOMP_IS

    // Clean up resources at the end of the method as the task may be
    // recycled in a threadpool and the destructor not called.
    delete _cis;
    _cis = nullptr;

    try {
        CLEANUP_DECOMP_OS
        _os = nullptr;
    }
    catch (exception&) {
        // Ignore: best effort
    }

    stopClock.stop();
    double delta = stopClock.elapsed();

    // If the whole input stream has been decoded and the original data size is present,
    // check that the output size matches the original data size.
    if ((checkOutputSize == true) && (_ctx.has("to") == false) && (_ctx.has("from") == false)) {
        const uint64 outputSize = _ctx.getLong("outputSize", 0);

        if ((outputSize != 0) && (written != outputSize)) {
            delete[] buf;
            stringstream sserr;
            sserr << "Corrupted bitstream: invalid output size (expected " << outputSize;
            sserr << ", got " << written << ")";
            return T(Error::ERR_INVALID_FILE, decoded, sserr.str().c_str());
        }
    }

    if (verbosity >= 1) {
        log.println("", verbosity > 1);
        ss.str(string());

        if (verbosity > 1) {
            if (delta >= 1e5) {
                ss.precision(1);
                ss.setf(ios::fixed);
                ss << "Decompression time: " << (delta / 1000) << " s" << endl;
            }
            else {
                ss << "Decompression time: " << int(delta) << " ms" << endl;
            }

            ss << "Input size:         " << decoded << endl;
            ss << "Output size:        " << read << endl;
            log.print(ss.str(), true);
            ss.str(string());
        }

        if (verbosity == 1) {
            ss << "Decompressed " << inputName << ": " << decoded << " => " << read;

            if (delta >= 1e5) {
                ss.precision(1);
                ss.setf(ios::fixed);
                ss << " bytes in " << (delta / 1000) << " s";
            }
            else {
                ss << " bytes in " << int(delta) << " ms";
            }

            log.println(ss.str(), true);
            ss.str(string());
        }

        if ((verbosity > 1) && (delta > 0)) {
            double b2KiB = double(1000) / double(1024);
            ss << "Throughput (KiB/s): " << uint(double(read) * b2KiB / delta);
            log.println(ss.str(), true);
            ss.str(string());
        }

        log.println("", verbosity > 1);
    }

    if (_listeners.size() > 0) {
        Event evt(Event::DECOMPRESSION_END, -1, int64(decoded), clock());
        BlockDecompressor::notifyListeners(_listeners, evt);
    }

    if (_ctx.getInt("remove", 0) != 0) {
        // Delete input file
        if (inputName == "STDIN") {
            log.println("Warning: ignoring remove option with STDIN", verbosity > 0);
        }
        else if (remove(inputName.c_str()) != 0) {
            log.println("Warning: input file could not be deleted", verbosity > 0);
        }
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

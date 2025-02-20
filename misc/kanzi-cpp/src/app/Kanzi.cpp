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
#include <iostream>
#include <map>

#include "BlockCompressor.hpp"
#include "BlockDecompressor.hpp"
#include "../Error.hpp"
#include "../util/Printer.hpp"

#if defined(WIN32) || defined(_WIN32) || defined(_WIN64)
   #include <windows.h>
#endif


using namespace kanzi;
using namespace std;

static const string CMD_LINE_ARGS[14] = {
    "-c", "-d", "-i", "-o", "-b", "-t", "-e", "-j", "-v", "-l", "-s", "-x", "-f", "-h"
};

//static const int ARG_IDX_COMPRESS = 0;
//static const int ARG_IDX_DECOMPRESS = 1;
static const int ARG_IDX_INPUT = 2;
static const int ARG_IDX_OUTPUT = 3;
static const int ARG_IDX_BLOCK = 4;
static const int ARG_IDX_TRANSFORM = 5;
static const int ARG_IDX_ENTROPY = 6;
static const int ARG_IDX_JOBS = 7;
static const int ARG_IDX_VERBOSE = 8;
static const int ARG_IDX_LEVEL = 9;
//static const int ARG_IDX_FROM = 10;
//static const int ARG_IDX_TO = 11;

static const string KANZI_VERSION = "2.3.0";
static const string APP_HEADER = "Kanzi " + KANZI_VERSION + " (c) Frederic Langlet";


#ifdef CONCURRENCY_ENABLED
   static const int MAX_CONCURRENCY = 64;

   mutex Printer::_mtx;
#endif

void printHelp(Printer& log, const string& mode, bool showHeader)
{
   log.println("", true);

   if (showHeader == true) {
       log.println(APP_HEADER, true);
       log.println("", true);
   }

   log.println("Credits: Matt Mahoney, Yann Collet, Jan Ondrus, Yuta Mori, Ilya Muravyov,", true);
   log.println("         Neal Burns, Fabian Giesen, Jarek Duda, Ilya Grebnov", true);
   log.println("", true);
   log.println("   -h, --help", true);
   log.println("        Display this message\n", true);

   if ((mode.compare(0, 1, "c") != 0) && (mode.compare(0, 1, "d") != 0)) {
       log.println("   -c, --compress", true);
       log.println("        Compress mode\n", true);
       log.println("   -d, --decompress", true);
       log.println("        Decompress mode\n", true);
   }

   log.println("   -i, --input=<inputName>", true);
   log.println("        Mandatory name of the input file or directory or 'stdin'", true);
   log.println("        When the source is a directory, all files in it will be processed.", true);
   stringstream ss;
   ss << "        Provide " << PATH_SEPARATOR << ". at the end of the directory name to avoid recursion";
   log.println(ss.str(), true);
   ss.str(string());
   ss << "        (EG: myDir" << PATH_SEPARATOR << ". => no recursion)\n";
   log.println(ss.str(), true);
   ss.str(string());
   log.println("   -o, --output=<outputName>", true);

   if (mode.compare(0, 1, "c") == 0) {
       log.println("        Optional name of the output file or directory (defaults to", true);
       log.println("        <inputName.knz>) or 'none' or 'stdout'. 'stdout' is not valid", true);
       log.println("        when the number of jobs is greater than 1.\n", true);
   }
   else if (mode.compare(0, 1, "d") == 0) {
       log.println("        Optional name of the output file or directory (defaults to", true);
       log.println("        <inputName.bak>) or 'none' or 'stdout'. 'stdout' is not valid", true);
       log.println("        when the number of jobs is greater than 1.\n", true);
   }
   else {
       log.println("        Optional name of the output file or 'none' or 'stdout'.\n", true);
   }

   if (mode.compare(0, 1, "c") == 0) {
       log.println("   -b, --block=<size>", true);
       log.println("        Size of blocks (default 4|8|16|32 MB based on level, max 1 GB, min 1 KB).", true);
       log.println("        'auto' means that the compressor derives the best value", true);
       log.println("        based on input size (when available) and number of jobs.\n", true);
       log.println("   -l, --level=<compression>", true);
       log.println("        Set the compression level [0..9]", true);
       log.println("        Providing this option forces entropy and transform.", true);
       log.println("        Defaults to level 3 if not provided.", true);
       log.println("        0 = NONE&NONE (store)", true);
       log.println("        1 = PACK+LZ&NONE", true);
       log.println("        2 = PACK+LZ&HUFFMAN", true);
       log.println("        3 = TEXT+UTF+PACK+MM+LZX&HUFFMAN", true);
       log.println("        4 = TEXT+UTF+EXE+PACK+MM+ROLZ&NONE", true);
       log.println("        5 = TEXT+UTF+BWT+RANK+ZRLT&ANS0", true);
       log.println("        6 = TEXT+UTF+BWT+SRT+ZRLT&FPAQ", true);
       log.println("        7 = LZP+TEXT+UTF+BWT+LZP&CM", true);
       log.println("        8 = EXE+RLT+TEXT+UTF&TPAQ", true);
       log.println("        9 = EXE+RLT+TEXT+UTF&TPAQX\n", true);
       log.println("   -e, --entropy=<codec>", true);
       log.println("        Entropy codec [None|Huffman|ANS0|ANS1|Range|FPAQ|TPAQ|TPAQX|CM]\n", true);
       log.println("   -t, --transform=<codec>", true);
       log.println("        Transform [None|BWT|BWTS|LZ|LZX|LZP|ROLZ|ROLZX|RLT|ZRLT]", true);
       log.println("                  [MTFT|RANK|SRT|TEXT|MM|EXE|UTF|PACK]", true);
       log.println("        EG: BWT+RANK or BWTS+MTFT\n", true);
       log.println("   -x, --checksum", true);
       log.println("        Enable block checksum\n", true);
       log.println("   -s, --skip", true);
       log.println("        Copy blocks with high entropy instead of compressing them.\n", true);
   }

   log.println("   -j, --jobs=<jobs>", true);
   log.println("        Maximum number of jobs the program may start concurrently", true);
   #ifdef CONCURRENCY_ENABLED
      log.println("        If 0 is provided, use all available cores (maximum is 64).", true);
      log.println("        (default is half of available cores).\n", true);
   #else
      log.println("        (always 1 in this version).\n", true);
   #endif
   log.println("   -v, --verbose=<level>", true);
   log.println("        0=silent, 1=default, 2=display details, 3=display configuration,", true);
   log.println("        4=display block size and timings, 5=display extra information", true);
   log.println("        Verbosity is reduced to 1 when files are processed concurrently", true);
   log.println("        Verbosity is reduced to 0 when the output is 'stdout'\n", true);
   log.println("   -f, --force", true);
   log.println("        Overwrite the output file if it already exists\n", true);
   log.println("   --rm", true);
   log.println("        Remove the input file after successful (de)compression.", true);
   log.println("        If the input is a folder, all processed files under the folder are removed.\n", true);
   log.println("   --no-link", true);
   log.println("        Skip links\n", true);
   log.println("   --no-dot-file", true);
   log.println("        Skip dot files\n", true);

   if (mode.compare(0, 1, "d") == 0) {
       log.println("   --from=blockId", true);
       log.println("        Decompress starting at the provided block (included).", true);
       log.println("        The first block ID is 1.\n", true);
       log.println("   --to=blockId", true);
       log.println("        Decompress ending at the provided block (excluded).\n", true);
       log.println("", true);
       log.println("EG. kanzi -d -i foo.knz -f -v 2 -j 2\n", true);
       log.println("EG. kanzi --decompress --input=foo.knz --force --verbose=2 --jobs=2\n", true);
   }

   if (mode.compare(0, 1, "c") == 0) {
       log.println("", true);
       log.println("EG. kanzi -c -i foo.txt -o none -b 4m -l 4 -v 3\n", true);
       log.println("EG. kanzi -c -i foo.txt -f -t BWT+MTFT+ZRLT -b 4m -e FPAQ -j 4\n", true);
       log.println("EG. kanzi --compress --input=foo.txt --output=foo.knz --force", true);
       log.println("          --transform=BWT+MTFT+ZRLT --block=4m --entropy=FPAQ --jobs=4\n", true);
   }
}

void printHeader(Printer& log, int verbose, bool& showHeader)
{
    if (verbose < 1)
        return;

    log.println("", true);
    log.println(APP_HEADER, true);

    if (verbose >= 4) {
       stringstream extraHeader;

   #ifdef __clang__
       extraHeader << "\nCompiled with clang version ";
       extraHeader << __clang_major__ << "." << __clang_minor__;
   #else
      #ifdef _MSC_VER
         extraHeader << "\nCompiled with Visual Studio";
         #ifdef _MSC_VER_STR // see types.h
         extraHeader << " " << _MSC_VER_STR;
         #endif
      #else
         #ifdef  __INTEL_COMPILER
         extraHeader << "\nCompiled with Intel compiler ";
         extraHeader << "(" << __INTEL_COMPILER_BUILD_DATE << ")";
         #else
            #ifdef  __GNUC__
            extraHeader << "\nCompiled with gcc version ";
            extraHeader << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
            #endif
         #endif
      #endif
   #endif

        if (extraHeader.str().length() > 0) {
    #if defined(__AVX2__)
            extraHeader << " - AVX2";
    #elif defined(__AVX__)
            extraHeader << " - AVX";
    #elif defined(__AVX512F__)
            extraHeader << " - AVX512";
    #elif defined(__SSE4_1__)
            extraHeader << " - SSE4.1";
    #elif defined(__SSE3__)
            extraHeader << " - SSE3";
    #elif defined(__SSE2__)
            extraHeader << " - SSE2";
    #elif defined(__SSE__)
            extraHeader << " - SSE";
    #endif
            log.println(extraHeader.str(), true);
        }
    }

    log.println("", true);
    showHeader = false;
}


#define WARNING_OPT_NOVALUE(opt) \
                 stringstream ss; \
                 ss << "Warning: ignoring option [" << opt << "] with no value."; \
                 log.println(ss.str(), verbose > 0)

#define WARNING_OPT_COMP_ONLY(opt) \
                 stringstream ss; \
                 ss << "Warning: ignoring option [" << opt << "]. Only applicable in compress mode."; \
                 log.println(ss.str(), verbose > 0)

#define WARNING_OPT_DUPLICATE(opt, val) \
                 stringstream ss; \
                 ss << "Warning: ignoring duplicate " << opt << ": " << val;\
                 log.println(ss.str(), verbose > 0)


bool toInt(string& s, int& res)
{
   // Chekc that all characters are valid
   for (size_t i = 0; i < s.length(); i++) {
       if ((s[i] < '0') || (s[i] > '9'))
          return false;
   }

   // Use atoi because stoi can throw
   res = atoi(s.c_str()); 
   return true;
}

int processCommandLine(int argc, const char* argv[], Context& map)
{
    string inputName;
    string outputName;
    int remove = -1;
    int overwrite = -1;
    int checksum = -1;
    int skip = -1;
    int reorder = -1;
    int noDotFiles = -1;
    int noLinks = -1;
    string codec;
    string transf;
    bool verboseFlag = false;
    int verbose = 1;
    int ctx = -1;
    int level = -1;
    int from = -1;
    int to = -1;
    int tasks = -1;
    int blockSize = -1;
    int autoBlockSize = -1;
    string mode;
    Printer log(cout); 
    bool showHeader = true;
    bool showHelp = false;

    for (int i = 1; i < argc; i++) {
        string arg(argv[i]);
        trim(arg);

        if (arg == "-v") {
            ctx = ARG_IDX_VERBOSE;
            continue;
        }

        if (arg == "-i") {
            ctx = ARG_IDX_INPUT;
            continue;
        }

        if (arg == "-o") {
            ctx = ARG_IDX_OUTPUT;
            continue;
        }

        // Extract verbosity, output and mode first
        if ((arg == "-c") || (arg.compare(0, 10, "--compress") == 0)) {
            if (mode == "d") {
                cerr << "Both compression and decompression options were provided." << endl;
                return Error::ERR_INVALID_PARAM;
            }

            mode = "c";
            continue;
        }

        if ((arg == "-d") || (arg.compare(0, 12, "--decompress") == 0)) {
            if (mode == "c") {
                cerr << "Both compression and decompression options were provided." << endl;
                return Error::ERR_INVALID_PARAM;
            }

            mode = "d";
            continue;
        }

        if ((ctx == ARG_IDX_VERBOSE) || (arg.compare(0, 10, "--verbose=") == 0)) {
           if (verboseFlag == true) {
                WARNING_OPT_DUPLICATE("verbosity level", arg);
            }
            else {
               if (ctx != ARG_IDX_VERBOSE)
                  arg = arg.substr(10);

               if ((toInt(arg, verbose) == false) || (verbose < 0) || (verbose > 5)) {
                   cerr << "Invalid verbosity level provided on command line: " << arg << endl;
                   return Error::ERR_INVALID_PARAM;
               }

               verboseFlag = true;
            }
        }
        else if ((ctx == ARG_IDX_OUTPUT) || (arg.compare(0, 9, "--output=") == 0)) {
            if (ctx != ARG_IDX_OUTPUT)
               arg = arg.substr(9);

            outputName = trim(arg);
        }
        else if ((ctx == ARG_IDX_INPUT) || (arg.compare(0, 8, "--input=") == 0)) {
            if (ctx != ARG_IDX_INPUT)
               arg = arg.substr(8);

            inputName = trim(arg);
        }
        else if ((arg == "--help") || (arg == "-h")) {
            showHelp = true;
        }

        ctx = -1;
    }

    if ((argc == 1) || (showHelp == true)) {
        printHeader(log, verbose, showHeader);
        printHelp(log, mode, showHeader);
        return 0;
    }

    // Overwrite verbosity if the output goes to stdout
    if (outputName.length() == 0) {
        if (inputName.length() == 0) {
            verbose = 0;
            verboseFlag = true;
        }
    }
    else {
        string str = outputName;
        transform(str.begin(), str.end(), str.begin(), ::toupper);

        if (str == "STDOUT") {
            verbose = 0;
            verboseFlag = true;
        }
    }

    printHeader(log, verbose, showHeader);
    inputName.clear();
    outputName.clear();
    ctx = -1;

    for (int i = 1; i < argc; i++) {
        string arg(argv[i]);

        if (arg[0] == 0x20) {
           size_t k = 1;

           // Left trim limited to spaces (due to possible unicode chars in names)
           while ((k < arg.length()) && (arg[k] == 0x20))
              k++;

           arg = arg.substr(k);
        }

        if ((arg == "-c") || (arg == "-d") || (arg == "--compress") || (arg == "--decompress")) {
            if (ctx != -1) {
                WARNING_OPT_NOVALUE(CMD_LINE_ARGS[ctx]);
            }

            ctx = -1;
            continue;
        }

        if ((arg == "--force") || (arg == "-f")) {
            if (ctx != -1) {
                WARNING_OPT_NOVALUE(CMD_LINE_ARGS[ctx]);
            }

            overwrite = 1;
            ctx = -1;
            continue;
        }

        if ((arg == "--skip") || (arg == "-s")) {
            if (ctx != -1) {
                WARNING_OPT_NOVALUE(CMD_LINE_ARGS[ctx]);
            }

            skip = 1;
            ctx = -1;
            continue;
        }

        if ((arg == "--checksum") || (arg == "-x")) {
            if (ctx != -1) {
                WARNING_OPT_NOVALUE(CMD_LINE_ARGS[ctx]);
            }

            checksum = 1;
            ctx = -1;
            continue;
        }

        if (arg == "--rm") {
            if (ctx != -1) {
                WARNING_OPT_NOVALUE(CMD_LINE_ARGS[ctx]);
            }

            remove = 1;
            ctx = -1;
            continue;
        }

        if (arg == "--no-file-reorder") {
            if (ctx != -1) {
                WARNING_OPT_NOVALUE(CMD_LINE_ARGS[ctx]);
            }

            ctx = -1;

            if (mode != "c") {
                WARNING_OPT_COMP_ONLY(arg);
                continue;
            }

            reorder = 0;
            continue;
        }

        if (arg == "--no-dot-file") {
            if (ctx != -1) {
                WARNING_OPT_NOVALUE(CMD_LINE_ARGS[ctx]);
            }

            ctx = -1;
            noDotFiles = 1;
            continue;
        }

        if (arg == "--no-link") {
            if (ctx != -1) {
                WARNING_OPT_NOVALUE(CMD_LINE_ARGS[ctx]);
            }

            ctx = -1;
            noLinks = 1;
            continue;
        }

        if (ctx == -1) {
            for (int j = 0; j < 10; j++) {
                if (arg == CMD_LINE_ARGS[j]) {
                    ctx = j;
                    break;
                }
            }

            if (ctx != -1) 
                continue;
        }

        if ((ctx == ARG_IDX_OUTPUT) || (arg.compare(0, 9, "--output=") == 0)) {
            if (ctx != ARG_IDX_OUTPUT)
               arg = arg.substr(9);

            if (outputName != "") {
                WARNING_OPT_DUPLICATE("output name", arg);
            } else {
                if ((arg.length() >= 2) && (arg[0] == '.') && (arg[1] == PATH_SEPARATOR)) {
                   arg = (arg.length() == 2) ? arg.substr(0, 1) : arg.substr(2);
                }

                outputName = arg;
            }

            ctx = -1;
            continue;
        }

        if ((ctx == ARG_IDX_INPUT) || (arg.compare(0, 8, "--input=") == 0)) {
            if (ctx != ARG_IDX_INPUT)
               arg = arg.substr(8);

            if (inputName != "") {
                WARNING_OPT_DUPLICATE("input name", arg);
            } else {
                if ((arg.length() >= 2) && (arg[0] == '.') && (arg[1] == PATH_SEPARATOR)) {
                   arg = (arg.length() == 2) ? arg.substr(0, 1) : arg.substr(2);
                }

                inputName = arg;
            }

            ctx = -1;
            continue;
        }

        if ((ctx == ARG_IDX_ENTROPY) || (arg.compare(0, 10, "--entropy=") == 0)) {
            if (ctx != ARG_IDX_ENTROPY)
               arg = arg.substr(10);

            if (codec != "") {
                WARNING_OPT_DUPLICATE("entropy", arg);
            } else {
                if (arg.length() == 0) {
                    cerr << "Invalid empty entropy provided on command line" << endl;
                    return Error::ERR_INVALID_PARAM;
                }

                codec = arg;
                transform(codec.begin(), codec.end(), codec.begin(), ::toupper);
            }

            ctx = -1;
            continue;
        }

        if ((ctx == ARG_IDX_TRANSFORM) || (arg.compare(0, 12, "--transform=") == 0)) {
            if (ctx != ARG_IDX_TRANSFORM)
                arg = arg.substr(12);

            if (transf != "") {
                WARNING_OPT_DUPLICATE("transform", arg);
            } else {
                if (arg.length() == 0) {
                    cerr << "Invalid empty transform provided on command line" << endl;
                    return Error::ERR_INVALID_PARAM;
                }

                transf = arg;
                transform(transf.begin(), transf.end(), transf.begin(), ::toupper);
            }

            while ((transf.length() > 0) && (transf[0] == '+')) {
                transf = transf.substr(1);
            }

            while ((transf.length() > 0) && (transf[transf.length() - 1] == '+')) {
                transf.resize(transf.length() - 1);
            }

            ctx = -1;
            continue;
        }

        if ((ctx == ARG_IDX_LEVEL) || (arg.compare(0, 8, "--level=") == 0)) {
            if (mode != "c"){
                log.println("Warning: ignoring level (only valid for compression)", verbose > 0);
                ctx = -1;
                continue;
            }

            if (ctx != ARG_IDX_LEVEL)
               arg = arg.substr(8);

            if (level >= 0) {
                WARNING_OPT_DUPLICATE("level", arg);
            } else {
                if ((toInt(arg, level) == false) || ((level < 0) || (level > 9))) {
                    cerr << "Invalid compression level provided on command line: " << arg << endl;
                    return Error::ERR_INVALID_PARAM;
                }
            }

            ctx = -1;
            continue;
        }

        if ((ctx == ARG_IDX_BLOCK) || (arg.compare(0, 8, "--block=") == 0)) {
            if (ctx != ARG_IDX_BLOCK)
               arg = arg.substr(8);

            if (arg.length() == 0) {
                cerr << "Invalid block size provided on command line: " << arg << endl;
                return Error::ERR_INVALID_PARAM;
            }

            if ((blockSize >= 0) || (autoBlockSize >= 0)) {
                WARNING_OPT_DUPLICATE("block size", arg);
                ctx = -1;
                continue;
            }

            transform(arg.begin(), arg.end(), arg.begin(), ::toupper);

            if (arg == "AUTO") {
                autoBlockSize = 1;
            }
            else {
                uint64 scale = 1;
                char lastChar = arg[arg.length() - 1];

                // Process K or M or G suffix
                if ('K' == lastChar) {
                    scale = 1024;
                    arg.resize(arg.length() - 1);
                }
                else if ('M' == lastChar) {
                    scale = 1024 * 1024;
                    arg.resize(arg.length() - 1);
                }
                else if ('G' == lastChar) {
                    scale = 1024 * 1024 * 1024;
                    arg.resize(arg.length() - 1);
                }

                if (toInt(arg, blockSize) == false) {
                    cerr << "Invalid block size provided on command line: " << arg << endl;
                    return Error::ERR_INVALID_PARAM;
                }

                stringstream ss1;
                ss1 << arg;
                ss1 >> blockSize;
                blockSize = int(uint64(blockSize) * scale);
            }

            ctx = -1;
            continue;
        }

        if ((ctx == ARG_IDX_JOBS) || (arg.compare(0, 7, "--jobs=") == 0)) {
            if (ctx != ARG_IDX_JOBS)
               arg = arg.substr(7);

            if (tasks >= 0) {
                WARNING_OPT_DUPLICATE("jobs", arg);
                ctx = -1;
                continue;
            }

            if ((toInt(arg, tasks) == false) || (tasks < 0)) {
                cerr << "Invalid number of jobs provided on command line: " << arg << endl;
                return Error::ERR_INVALID_PARAM;
            }

            ctx = -1;
            continue;
        }

        if ((arg.compare(0, 7, "--from=") == 0) && (ctx == -1)) {
            arg = arg.substr(7);

            if (mode != "d"){
                log.println("Warning: ignoring start block (only valid for decompression)", verbose > 0);
                continue;
            }

            if (from >= 0) {
                WARNING_OPT_DUPLICATE("start block", arg);
            } else {
                if ((toInt(arg, from) == false) || (from < 0)) {
                    cerr << "Invalid start block provided on command line: " << arg << endl;

                    if (from == 0) {
                        cerr << "The first block ID is 1." << endl;
                    }

                    return Error::ERR_INVALID_PARAM;
                }
            }

            continue;
        }

        if ((arg.compare(0, 5, "--to=") == 0) && (ctx == -1)) {
            arg = arg.substr(5);

            if (mode != "d"){
                log.println("Warning: ignoring end block (only valid for decompression)", verbose > 0);
                continue;
            }

            if (to >= 0) {
                WARNING_OPT_DUPLICATE("end block", arg);
            } else {
                if ((toInt(arg, to) == false) || (to <= 0)) { // Must be > 0 (0 means nothing to do)
                    cerr << "Invalid end block provided on command line: " << arg << endl;
                    return Error::ERR_INVALID_PARAM;
                }
            }

            continue;
        }

        if ((arg.compare(0, 10, "--verbose=") != 0) && (ctx == -1)) {
            stringstream ss;
            ss << "Warning: ignoring unknown option [" << arg << "]";
            log.println(ss.str(), verbose > 0);
        }

        ctx = -1;
    }

    if (ctx != -1) {
        stringstream ss;
        ss << "Warning: ignoring option with missing value [" << CMD_LINE_ARGS[ctx] << "]";
        log.println(ss.str(), verbose > 0);
    }

    if (level >= 0) {
        if (codec.length() > 0) {
            stringstream ss;
            ss << "Warning: providing the 'level' option forces the entropy codec. Ignoring [" << codec << "]";
            log.println(ss.str(), verbose > 0);
        }

        if (transf.length() > 0) {
            stringstream ss;
            ss << "Warning: providing the 'level' option forces the transform. Ignoring [" << transf << "]";
            log.println(ss.str(), verbose > 0);
        }
    }

    if (blockSize >= 0)
        map.putInt("blockSize", blockSize);

    map.putInt("verbosity", (verboseFlag == false) ? 1 : verbose);
    map.putString("mode", mode);
    map.putString("inputName", inputName);
    map.putString("outputName", outputName);

    if (autoBlockSize == 1)
        map.putInt("autoBlock", 1);

    if ((mode == "c") && (level >= 0))
        map.putInt("level", level);

    if (overwrite == 1)
        map.putInt("overwrite", 1);

    if (remove == 1)
        map.putInt("remove", 1);

    if (codec.length() > 0)
        map.putString("entropy", codec);

    if (transf.length() > 0)
        map.putString("transform", transf);

    if (checksum == 1)
        map.putInt("checksum", 1);

    if (skip == 1)
        map.putInt("skipBlocks", 1);

    if (reorder == 0)
        map.putInt("fileReorder", 0);

    if (noDotFiles == 1)
        map.putInt("noDotFiles", 1);

    if (noLinks == 1)
        map.putInt("noLinks", 1);

    if (from >= 0)
        map.putInt("from", from);

    if (to >= 0)
        map.putInt("to", to);

    if (tasks >= 0)
        map.putInt("jobs", tasks);

    return 0;
}

int main(int argc, const char* argv[])
{
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64)
    // Users can provide a custom code page to properly display some non ASCII file names
    // eg. 1252 for ANSI Latin-1 or 65001 for utf-8
    const char* env_cp = getenv("KANZI_CODE_PAGE");

    if (env_cp != nullptr) {
        string s(env_cp);
        int cp;

        if (toInt(s, cp) == true) {
           SetConsoleCP(cp);
           SetConsoleOutputCP(cp);
        }
    }

#endif
    Context args;
    int status = processCommandLine(argc, argv, args);

    // Command line processing error ?
    if (status != 0)
       exit(status);

    // Help mode only ?
    if (args.has("mode") == false)
       exit(0);

    string mode = args.getString("mode");
    int jobs = args.getInt("jobs", -1);

    try {
#ifndef CONCURRENCY_ENABLED
        if (jobs > 1) {
            const int verbosity = args.getInt("verbosity");
            stringstream ss;
            ss << "Warning: the number of jobs is  limited to 1 in this version";
            Printer log(cout);
            log.println(ss.str(), verbosity > 0);
        }

        jobs = 1;
        Context ctx(args);
#else
        if (jobs == 0) {
           int cores = max(int(thread::hardware_concurrency()), 1); // User provided 0 => use all the cores
           jobs = min(cores, MAX_CONCURRENCY);
        }
        else if (jobs == -1) {
           int cores = max(int(thread::hardware_concurrency()) / 2, 1); // Defaults to half the cores
           jobs = min(cores, MAX_CONCURRENCY);
        }
        else if (jobs > MAX_CONCURRENCY) {
            const int verbosity = args.getInt("verbosity");
            stringstream ss;
            ss << "Warning: the number of jobs is too high, defaulting to " << MAX_CONCURRENCY;
            Printer log(cout);
            log.println(ss.str(), verbosity > 0);
            jobs = MAX_CONCURRENCY;
        }

        ThreadPool pool(jobs);
        Context ctx(args, &pool);
#endif
        ctx.putInt("jobs", jobs);

        if (mode == "c") {
            try {
                BlockCompressor bc(ctx);
                uint64 written = 0;
                int code = bc.compress(written);
                exit(code);
            }
            catch (exception& e) {
                cerr << "Could not create the compressor: " << e.what() << endl;
                exit(Error::ERR_CREATE_COMPRESSOR);
            }
        }

        if (mode == "d") {
            try {
                BlockDecompressor bd(ctx);
                uint64 read = 0;
                int code = bd.decompress(read);
                exit(code);
            }
            catch (exception& e) {
                cerr << "Could not create the decompressor: " << e.what() << endl;
                exit(Error::ERR_CREATE_DECOMPRESSOR);
            }
        }

        cout << "Missing arguments: try --help or -h" << endl;
        return Error::ERR_MISSING_PARAM;
    }
    catch (invalid_argument& e) {
       // May be thrown by ThreadPool
       cout << e.what() << endl;
       exit(Error::ERR_INVALID_PARAM);
    }
    catch (runtime_error& e) {
       // May be thrown by ThreadPool
       cout << e.what() << endl;
       exit(Error::ERR_INVALID_PARAM);
    }
}

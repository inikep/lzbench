/*
Copyright 2011-2026 Frederic Langlet
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

#include <cstdlib>
#include <iomanip>
#include <ios>
#include <sstream>

#include "InfoPrinter.hpp"
#include "../util/strings.hpp"

using namespace kanzi;
using namespace std;

InfoPrinter::InfoPrinter(int infoLevel, InfoPrinter::Type type, OutputStream& os, int firstBlockId)
    : _os(os)
    , _type(type)
    , _level(infoLevel)
    , _headerInfo(0)
{
    STORE_ATOMIC(_nextBlockId, firstBlockId);

    if (type == InfoPrinter::COMPRESSION) {
        _thresholds[0] = Event::COMPRESSION_START;
        _thresholds[1] = Event::BEFORE_TRANSFORM;
        _thresholds[2] = Event::AFTER_TRANSFORM;
        _thresholds[3] = Event::BEFORE_ENTROPY;
        _thresholds[4] = Event::AFTER_ENTROPY;
        _thresholds[5] = Event::COMPRESSION_END;
    }
    else {
        _thresholds[0] = Event::DECOMPRESSION_START;
        _thresholds[1] = Event::BEFORE_ENTROPY;
        _thresholds[2] = Event::AFTER_ENTROPY;
        _thresholds[3] = Event::BEFORE_TRANSFORM;
        _thresholds[4] = Event::AFTER_TRANSFORM;
        _thresholds[5] = Event::DECOMPRESSION_END;
    }
}


void InfoPrinter::processEvent(const Event& evt)
{
    if (_type == InfoPrinter::INFO) {
        processHeaderInfo(evt);
        return;
    }

#ifdef CONCURRENCY_ENABLED
    Event::Type t = evt.getType();

    if (t == Event::BEFORE_TRANSFORM || t == Event::AFTER_TRANSFORM ||
        t == Event::BEFORE_ENTROPY || t == Event::AFTER_ENTROPY) {
        processBlockEventOrdered(evt);
        return;
    }
#endif

    processEventOrdered(evt);
}


#ifdef CONCURRENCY_ENABLED
void InfoPrinter::processBlockEventOrdered(const Event& evt)
{
    const int blockId = evt.getId();
    const Event::Type type = evt.getType();
    bool blockComplete = false;

    // Determine completion condition
    if (_type == InfoPrinter::COMPRESSION) {
        blockComplete = (type == Event::AFTER_ENTROPY);
    }
    else if (_type == InfoPrinter::DECOMPRESSION) {
        blockComplete = (type == Event::AFTER_TRANSFORM);
    }

    {
#ifdef CONCURRENCY_ENABLED
        std::lock_guard<std::mutex> lock(_mutex);
#endif
        _pendingBlocks[blockId].push_back(evt);

        // Do not attempt to release unless this block is complete
        if (blockComplete == false)
            return;
    }

    // Try to release completed blocks in strict blockId order
    for (;;)
    {
        std::vector<Event> events;

        {
#ifdef CONCURRENCY_ENABLED
            std::lock_guard<std::mutex> lock(_mutex);
#endif
            int expectedId = LOAD_ATOMIC(_nextBlockId);
            std::map<int, std::vector<Event> >::iterator it = _pendingBlocks.find(expectedId);

            if (it == _pendingBlocks.end())
                return;

            // The block must be complete before release
            bool complete = false;
            const std::vector<Event>& evts = it->second;

            for (size_t i = 0; i < evts.size(); i++) {
                if ((_type == InfoPrinter::COMPRESSION) && (evts[i].getType() == Event::AFTER_ENTROPY)) {
                    complete = true;
                    break;
                }

                if ((_type == InfoPrinter::DECOMPRESSION) && (evts[i].getType() == Event::AFTER_TRANSFORM)) {
                    complete = true;
                    break;
                }
            }

            if (complete == false)
                return;

            // Release block
            events.swap(it->second);
            _pendingBlocks.erase(it);
            STORE_ATOMIC(_nextBlockId, expectedId + 1);
        }

        // Process all events for this block in arrival order
#ifdef CONCURRENCY_ENABLED
        std::lock_guard<std::mutex> lock(_mutex);
#endif

        for (size_t i = 0; i < events.size(); i++) {
            processEventOrdered(events[i]);
        }
    }
}
#endif


void InfoPrinter::processEventOrdered(const Event& evt)
{
    const int blockId = evt.getId();
    const Event::Type type = evt.getType();
    string msg;

    if (type == _thresholds[1]) {
        BlockInfo* bi = new BlockInfo();
        bi->_timeStamp1 = evt.getTime();
        bi->_stage0Size = evt.getSize();
        _blocks[blockId] = bi;

        if (_level >= 5) {
            msg = evt.toString();
        }
    }
    else if (type == _thresholds[2]) {
        std::map<int, BlockInfo*>::iterator it = _blocks.find(blockId);

        if (it == _blocks.end())
            return;

        BlockInfo& bi = *it->second;
        bi._timeStamp2 = evt.getTime();

        if (_level >= 5) {
            double elapsed = WallTimer::calculateDifference(bi._timeStamp1, bi._timeStamp2);
            std::stringstream ss;
            ss << evt.toString() << " [" << int64(elapsed) << " ms]";
            msg = ss.str();
        }
    }
    else if (type == _thresholds[3]) {
        std::map<int, BlockInfo*>::iterator it = _blocks.find(blockId);

        if (it == _blocks.end())
           return;

        BlockInfo& bi = *it->second;
        bi._timeStamp3 = evt.getTime();
        bi._stage1Size = evt.getSize();

        if (_level >= 5)
            msg = evt.toString();
    }
    else if (type == _thresholds[4]) {
        std::map<int, BlockInfo*>::iterator it = _blocks.find(blockId);

        if (it == _blocks.end())
            return;

        const BlockInfo& bi = *it->second;
        stringstream ss;

        if (_level >= 5) {
            ss << evt.toString() << endl;
        }

        // Display block info
        if (_level >= 4) {
            double elapsed1 = WallTimer::calculateDifference(bi._timeStamp1, bi._timeStamp2);
            double elapsed2 = WallTimer::calculateDifference(bi._timeStamp3, evt.getTime());
            ss << "Block " << blockId << ": "
               << bi._stage0Size
               << " => "
               << bi._stage1Size
               << " [" << int64(elapsed1) << " ms] => " << evt.getSize()
               << " [" << int64(elapsed2) << " ms]";

            // Add compression ratio for encoding
            if ((_type == InfoPrinter::COMPRESSION) && (bi._stage0Size != 0)) {
                ss << " (" << uint(double(evt.getSize()) * 100.0 / double(bi._stage0Size))
                   << "%)";
            }

            // Optionally add hash
            if (evt.getHash() != 0) {
                ss << std::uppercase << std::hex << " [" << evt.getHash() << "]";
            }

            msg = ss.str();
        }

        delete it->second;
        _blocks.erase(it);
    }
    else if ((evt.getType() == Event::AFTER_HEADER_DECODING) && (_level >= 3)) {
        Event::HeaderInfo* info = evt.getInfo();

        if (info == nullptr)
            return;

        if (_level >= 5) {
            // JSON output
            msg = evt.toString();
        }
        else {
            // Raw text output
            stringstream ss;
            ss << "Bitstream version: " << info->bsVersion << endl;
            string strCk = "NONE";

            if (info->checksumSize == 32)
               strCk = "32 bits";
            else if (info->checksumSize == 64)
               strCk = "64 bits";

            ss << "Block checksum: " << strCk << endl;
            ss << "Block size: " << info->blockSize << " bytes" << endl;
            string strE = info->entropyType == "NONE" ? "no" : info->entropyType;
            ss << "Using " << strE << " entropy codec (stage 1)" << endl;
            string strF = info->transformType == "NONE" ? "no" : info->transformType;
            ss << "Using " << strF << " transform (stage 2)" << endl;

            if (info->originalSize >= 0)
               ss << "Original size: " << info->originalSize << " byte(s)" << endl;

            msg = ss.str();
        }
    }
    else if (_level >= 5) {
        msg = evt.toString();
    }

    if (msg.size() > 0) {
        _os << msg << endl;
        _os.flush();
    }
}


void InfoPrinter::processHeaderInfo(const Event& evt)
{
    if ((_level == 0) || (evt.getType() != Event::AFTER_HEADER_DECODING))
        return;

    const Event::HeaderInfo* info = evt.getInfo();

    if (info == nullptr)
        return;

    stringstream ss;

    if (_headerInfo++ == 0) {
        ss << endl;
        ss << "|" << "     File Name      ";
        ss << "|" << "Ver";
        ss << "|" << "Check";
        ss << "|" << "Block Size";
        ss << "|" << "  File Size ";
        ss << "|" << " Orig. Size ";
        ss << "|" << " Ratio ";

        if (_level >= 4) {
            ss << "|" << " Entropy";
            ss << "|" << "        Transforms        ";
        }

        ss << "|" << endl;
    }

    ss << "|";

    string inputName = info->inputName;
    size_t idx = inputName.find_last_of(PATH_SEPARATOR);

    if (idx != string::npos)
        inputName.erase(0, idx + 1);

    if (inputName.length() > 20)
        inputName.replace(18, string::npos, "..");

    ss << left << setw(20) << inputName << "|" << right;
    ss << setw(3) << info->bsVersion << "|";
    ss << setw(5) << info->checksumSize << "|";
    ss << setw(10) << info->blockSize << "|";

    if (info->fileSize >= 0)
        ss << setw(12) << formatSize(double(info->fileSize)) << "|";
    else
        ss << setw(12) << "    N/A    |";

    if (info->originalSize >= 0)
        ss << setw(12) << formatSize(double(info->originalSize)) << "|";
    else
        ss << setw(12) << "    N/A    |";

    if ((info->originalSize >= 0) && (info->fileSize >= 0)) {
        double compSz = double(info->fileSize);
        double origSz = double(info->originalSize);

        if (origSz == 0.0)
            ss << setw(7) << "  N/A  |";
        else
            ss << setw(7) << fixed << setprecision(3)
               << (compSz / origSz) << "|";
    }
    else {
        ss << setw(7) << "  N/A  |";
    }

    if (_level >= 4) {
        ss << setw(8) << info->entropyType << "|";
        string t = info->transformType;

        if (t.length() > 26)
            t.replace(24, string::npos, "..");

        ss << setw(26) << t << "|";
    }

    _os << ss.str() << endl;
}


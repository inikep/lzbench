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

#include <iomanip>
#include <sstream>
#include "InfoPrinter.hpp"
#include <ios>

using namespace kanzi;
using namespace std;

InfoPrinter::InfoPrinter(int infoLevel, InfoPrinter::Type type, OutputStream& os)
    : _os(os)
    , _type(type)
    , _level(infoLevel)
{
    if (type == InfoPrinter::ENCODING) {
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
	
	for (int i = 0; i < 1024; i++)
		_map[i] = nullptr;
}

void InfoPrinter::processEvent(const Event& evt)
{
    int currentBlockId = evt.getId();

    if (evt.getType() == _thresholds[1]) {
        // Register initial block size
        BlockInfo* bi = new BlockInfo();
        _clock12.start();

        if (_type == InfoPrinter::ENCODING)
            bi->_stage0Size = evt.getSize();

        _map[hash(currentBlockId)] = bi;

        if (_level >= 5) {
            _os << evt.toString() << endl;
        }
    }
    else if (evt.getType() == _thresholds[2]) {
        BlockInfo* bi = _map[hash(currentBlockId)];

        if (bi == nullptr)
            return;

        if (_type == InfoPrinter::DECODING)
            bi->_stage0Size = evt.getSize();

        _clock12.stop();
        _clock23.start();

        if (_level >= 5) {
            stringstream ss;
            ss << evt.toString() << " [" << int64(_clock12.elapsed()) << " ms]";
            _os << ss.str() << endl;
        }
    }
    else if (evt.getType() == _thresholds[3]) {
        BlockInfo* bi = _map[hash(currentBlockId)];

        if (bi == nullptr)
            return;

        _clock23.stop();
        _clock34.start();
        bi->_stage1Size = evt.getSize();

        if (_level >= 5) {
            _os << evt.toString() << endl;
        }
    }
    else if (evt.getType() == _thresholds[4]) {
        BlockInfo* bi = _map[hash(currentBlockId)];

        if (bi == nullptr)
            return;

        if (_level < 3) {
            delete bi;
            _map[hash(currentBlockId)] = nullptr;
            return;
        }

        int64 stage2Size = evt.getSize();
        _clock34.stop();
        stringstream ss;

        if (_level >= 5) {
            ss << evt.toString() << endl;
        }

        // Display block info
        if (_level >= 4) {
            ss << "Block " << currentBlockId << ": " << bi->_stage0Size << " => ";
            ss << bi->_stage1Size << " [" << int64(_clock12.elapsed()) << " ms] => " << stage2Size;
            ss << " [" << int64(_clock34.elapsed()) << " ms]";

            // Add compression ratio for encoding
            if ((_type == InfoPrinter::ENCODING) && (bi->_stage0Size != 0)) {
                ss << " (" << uint(double(stage2Size) * double(100) / double(bi->_stage0Size));
                ss << "%)";
            }

            // Optionally add hash
            if (evt.getHash() != 0) {
                ss << std::uppercase << std::hex << " [" << evt.getHash() << "]";
            }

            _os << ss.str() << endl;
        }

        delete bi;
        _map[hash(currentBlockId)] = nullptr;
    }
    else if ((evt.getType() == Event::AFTER_HEADER_DECODING) && (_level >= 3)) {
        _os << evt.toString() << endl;
    }
    else if (_level >= 5) {
        _os << evt.toString() << endl;
    }
}

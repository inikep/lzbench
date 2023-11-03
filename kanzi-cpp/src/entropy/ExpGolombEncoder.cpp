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

#include "ExpGolombEncoder.hpp"

using namespace kanzi;

ExpGolombEncoder::ExpGolombEncoder(OutputBitStream& bitstream, bool sgn)
    : _bitstream(bitstream), _signed((sgn == true) ? 1 : 0)
{
}

int ExpGolombEncoder::encode(const byte block[], uint blkptr, uint len)
{
    const byte* buf = &block[blkptr];
    const uint len8 = len & uint(-8);

    for (uint i = 0; i < len8; i += 8) {
        encodeByte(buf[i]);
        encodeByte(buf[i+1]);
        encodeByte(buf[i+2]);
        encodeByte(buf[i+3]);
        encodeByte(buf[i+4]);
        encodeByte(buf[i+5]);
        encodeByte(buf[i+6]);
        encodeByte(buf[i+7]);
    }

    for (uint i = len8; i < len; i++)
        encodeByte(buf[i]);

    return len;
}


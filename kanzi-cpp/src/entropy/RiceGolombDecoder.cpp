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

#include <stdexcept>
#include "RiceGolombDecoder.hpp"

using namespace kanzi;
using namespace std;

RiceGolombDecoder::RiceGolombDecoder(InputBitStream& bitstream, uint logBase, bool sgn) THROW
    : _bitstream(bitstream)
{
    if ((logBase < 1) || (logBase > 12))
       throw invalid_argument("Invalid logBase value (must be in [1..12])");

    _signed = sgn;
    _logBase = logBase;
}


int RiceGolombDecoder::decode(byte block[], uint blkptr, uint len)
{
    const uint end = blkptr + len;

    for (uint i = blkptr; i < end; i++)
        block[i] = decodeByte();

    return len;
}

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

#pragma once
#ifndef _Error_
#define _Error_

namespace kanzi
{

   class Error
   {
   public:
       static const int ERR_MISSING_PARAM = 1;
       static const int ERR_BLOCK_SIZE = 2;
       static const int ERR_INVALID_CODEC = 3;
       static const int ERR_CREATE_COMPRESSOR = 4;
       static const int ERR_CREATE_DECOMPRESSOR = 5;
       static const int ERR_OUTPUT_IS_DIR = 6;
       static const int ERR_OVERWRITE_FILE = 7;
       static const int ERR_CREATE_FILE = 8;
       static const int ERR_CREATE_BITSTREAM = 9;
       static const int ERR_OPEN_FILE = 10;
       static const int ERR_READ_FILE = 11;
       static const int ERR_WRITE_FILE = 12;
       static const int ERR_PROCESS_BLOCK = 13;
       static const int ERR_CREATE_CODEC = 14;
       static const int ERR_INVALID_FILE = 15;
       static const int ERR_STREAM_VERSION = 16;
       static const int ERR_CREATE_STREAM = 17;
       static const int ERR_INVALID_PARAM = 18;
       static const int ERR_CRC_CHECK = 19;
       static const int ERR_UNKNOWN = 127;
   };

}
#endif


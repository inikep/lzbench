#ifndef _CSC_TYPEDEF_H_
#define _CSC_TYPEDEF_H_

#include <stdint.h>

const uint32_t KB = 1024;
const uint32_t MB = 1048576;
const uint32_t MinBlockSize = 8 * KB;


const uint32_t MaxDictSize = 1024 * MB;//Don't change
const uint32_t MinDictSize = 32 * KB;//Don't change

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))


/******Block Type*************/
const uint32_t DT_NONE = 0x00;
const uint32_t DT_NORMAL = 0x01;
const uint32_t DT_ENGTXT = 0x02;
const uint32_t DT_EXE = 0x03;
const uint32_t DT_FAST = 0x04;

///////////////////////////
const uint32_t DT_NO_LZ = 0x05;

//const uint32_t DT_AUDIO = 0x06;
//const uint32_t DT_AUDIO = 0x06;
const uint32_t DT_ENTROPY = 0x07;
const uint32_t DT_BAD = 0x08;
const uint32_t SIG_EOF = 0x09;
const uint32_t DT_DLT = 0x10;
const uint32_t DLT_CHANNEL_MAX = 5;
const uint32_t DltIndex[DLT_CHANNEL_MAX]={1,2,3,4,8};

// DT_SKIP means same with last one
const uint32_t DT_SKIP = 0x1E;
const uint32_t DT_MAXINVALID = 0x1F;
/******Block Type*************/


#endif

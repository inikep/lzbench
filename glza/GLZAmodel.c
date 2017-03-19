/***********************************************************************

Copyright 2015 - 2016 Kennon Conrad

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

***********************************************************************/

#include <inttypes.h>
#include "GLZAmodel.h"
#include <stdio.h>

uint32_t InCharNum, OutCharNum;
uint32_t RangeLow, RangeHigh, count;
uint32_t low, code, range, extra_range;
uint8_t * InBuffer;
uint8_t FoundIndex;
uint8_t SymbolFirstChar[4][0x100][0x100];
uint8_t RangeScaleERG[3], RangeScaleWordTag;
uint8_t FreqERG[3], FreqWordTag, *OutBuffer;
uint16_t RangeScaleMtfQueueNum[2], RangeScaleMtfQueuePos[2][14], RangeScaleMtfgQueuePos[2];
uint16_t RangeScaleSID[2], RangeScaleINST[2][16];
uint16_t RangeScaleFirstChar[4][0x100];
uint16_t RangeScaleFirstCharSection[0x100][7];
uint16_t FreqSymType[4][4], FreqMtfQueueNum[2][14], FreqMtfQueuePos[2][14][64], FreqMtfgQueuePos[2][256];
uint16_t FreqSID[2][16], FreqINST[2][16][38], FreqFirstChar[4][0x100][0x100], FreqFirstCharBinary[0x100][0x100];
const uint32_t MTF_QUEUE_SIZE = 64;

uint32_t ReadLow() {return(low);}
uint32_t ReadRange() {return(range);}

void StartModelSymType(uint8_t use_mtf, uint8_t use_mtfg) {
  uint8_t i = 3;
  do {
    if (use_mtf) {
      if (use_mtfg) {
        FreqSymType[i][0] = 0x1C00;
        FreqSymType[i][1] = 0x2000;
        FreqSymType[i][2] = 0x200;
        FreqSymType[i][3] = 0x200;
      }
      else {
        FreqSymType[i][0] = 0x1E00;
        FreqSymType[i][1] = 0x2000;
        FreqSymType[i][2] = 0;
        FreqSymType[i][3] = 0x200;
      }
    }
    else {
      FreqSymType[i][0] = 0x2000;
      FreqSymType[i][1] = 0x2000;
      FreqSymType[i][2] = 0;
      FreqSymType[i][3] = 0;
    }
  } while (i--);
  return;
}

void StartModelMtfQueueNum() {
  uint8_t i = 1;
  do {
    uint8_t j = 13;
    do {
      FreqMtfQueueNum[i][j] = 4;
    } while (j--);
    RangeScaleMtfQueueNum[i] = 56;
  } while (i--);
  return;
}

void StartModelMtfQueuePos() {
  uint8_t i = 1;
  do {
    uint8_t j = 13;
    do {
      RangeScaleMtfQueuePos[i][j] = 0;
      uint8_t k = 63;
      do {
        FreqMtfQueuePos[i][j][k] = 64 / (k + 1);
        RangeScaleMtfQueuePos[i][j] += FreqMtfQueuePos[i][j][k];
      } while (k--);
    } while (j--);
  } while (i--);
  return;
}

void StartModelMtfgQueuePos(uint8_t max_regular_code_length) {
  uint32_t max_value;
  if (max_regular_code_length >= 17)
    max_value = 0x100;
  else if (max_regular_code_length == 16)
    max_value = 0xC0;
  else if (max_regular_code_length == 15)
    max_value = 0x80;
  else if (max_regular_code_length == 14)
    max_value = 0x40;
  else if (max_regular_code_length == 13)
    max_value = 0x20;
  else
    max_value = 0x10;
  uint8_t i = 1;
  do {
    RangeScaleMtfgQueuePos[i] = 0;
    uint16_t j = 0;
    do {
      FreqMtfgQueuePos[i][j] = 2 * (max_value + 1) / (j + 2);
      RangeScaleMtfgQueuePos[i] += FreqMtfgQueuePos[i][j];
    } while (++j != max_value);
    while (j < 0x100)
      FreqMtfgQueuePos[i][j++] = 0;
  } while (i--);
  return;
}

void StartModelSID() {
  uint8_t i = 1;
  do {
    uint8_t j = 15;
    do {
      FreqSID[i][j] = 1;
    } while (j--);
    RangeScaleSID[i] = 16;
  } while (i--);
  return;
}

void StartModelINST(uint8_t num_inst_codes) {
  uint8_t i = 1;
  do {
    uint8_t j = 15;
    do {
      uint8_t k = num_inst_codes - 1;
      do {
        FreqINST[i][j][k] = 1;
      } while (k--);
      RangeScaleINST[i][j] = num_inst_codes;
    } while (j--);
  } while (i--);
  return;
}

void StartModelERG() {
  FreqERG[0] = 1;
  RangeScaleERG[0] = 2;
  FreqERG[1] = 1;
  RangeScaleERG[1] = 2;
  FreqERG[2] = 1;
  RangeScaleERG[2] = 2;
  return;
}

void StartModelWordTag() {
  FreqWordTag = 1;
  RangeScaleWordTag = 2;
  return;
}

void StartModelFirstChar() {
  uint8_t i = 0xFF;
  do {
    uint8_t j = 0xFF;
    do {
      FreqFirstChar[0][i][j] = 0;
      FreqFirstChar[1][i][j] = 0;
      FreqFirstChar[2][i][j] = 0;
      FreqFirstChar[3][i][j] = 0;
    } while (j--);
    RangeScaleFirstChar[0][i] = 0;
    RangeScaleFirstChar[1][i] = 0;
    RangeScaleFirstChar[2][i] = 0;
    RangeScaleFirstChar[3][i] = 0;
  } while (i--);
  return;
}

void StartModelFirstCharBinary() {
  uint8_t i = 0xFF;
  do {
    uint8_t j = 0xFF;
    do {
      FreqFirstCharBinary[i][j] = 0;
    } while (j--);
    j = 6;
    do {
      RangeScaleFirstCharSection[i][j] = 0;
    } while (j--);
    RangeScaleFirstChar[0][i] = 0;
  } while (i--);
  return;
}

void rescaleMtfQueueNum(uint8_t Context) {
  uint8_t i = 12;
  RangeScaleMtfQueueNum[Context] = FreqMtfQueueNum[Context][13] = (FreqMtfQueueNum[Context][13] + 4) >> 1;
  do {
    RangeScaleMtfQueueNum[Context] += FreqMtfQueueNum[Context][i] = (FreqMtfQueueNum[Context][i] + 4) >> 1;
  } while (i--);
  return;
}

void rescaleMtfQueuePos(uint8_t Context, uint8_t mtf_queue_number) {
  uint8_t i = 62;
  RangeScaleMtfQueuePos[Context][mtf_queue_number] = FreqMtfQueuePos[Context][mtf_queue_number][63]
      = (FreqMtfQueuePos[Context][mtf_queue_number][63] + 1) >> 1;
  do {
    RangeScaleMtfQueuePos[Context][mtf_queue_number] += FreqMtfQueuePos[Context][mtf_queue_number][i]
        = (FreqMtfQueuePos[Context][mtf_queue_number][i] + 1) >> 1;
  } while (i--);
  return;
}

void rescaleMtfQueuePosQ0(uint8_t Context) {
  uint8_t i = 62;
  RangeScaleMtfQueuePos[Context][0] = (FreqMtfQueuePos[Context][0][63]
      = (FreqMtfQueuePos[Context][0][63] + 1) >> 1);
  do {
    RangeScaleMtfQueuePos[Context][0] += FreqMtfQueuePos[Context][0][i]
        = (FreqMtfQueuePos[Context][0][i] + 1) >> 1;
  } while (i--);
  return;
}

void rescaleMtfgQueuePos(uint8_t Context) {
  uint8_t i = 1;
  RangeScaleMtfgQueuePos[Context] = FreqMtfgQueuePos[Context][0] = (FreqMtfgQueuePos[Context][0] + 1) >> 1;
  do {
    RangeScaleMtfgQueuePos[Context] += FreqMtfgQueuePos[Context][i] = (FreqMtfgQueuePos[Context][i] + 1) >> 1;
  } while (++i);
  return;
}

void rescaleSID(uint8_t Context) {
  uint8_t i = 14;
  RangeScaleSID[Context] = FreqSID[Context][15] = (FreqSID[Context][15] + 1) >> 1;
  do {
    RangeScaleSID[Context] += FreqSID[Context][i] = (FreqSID[Context][i] + 1) >> 1;
  } while (i--);
  return;
}

void rescaleINST(uint8_t Context, uint8_t SIDSymbol) {
  uint8_t i = 34;
  RangeScaleINST[Context][SIDSymbol] = (FreqINST[Context][SIDSymbol][35] = (FreqINST[Context][SIDSymbol][35] + 1) >> 1);
  do {
    RangeScaleINST[Context][SIDSymbol] += FreqINST[Context][SIDSymbol][i] = (FreqINST[Context][SIDSymbol][i] + 1) >> 1;
  } while (i--);
  return;
}

void rescaleFirstChar(uint8_t SymType, uint8_t Context) {
  uint8_t i = 0xFE;
  RangeScaleFirstChar[SymType][Context] = FreqFirstChar[SymType][Context][0xFF]
      = (FreqFirstChar[SymType][Context][0xFF] + 1) >> 1;
  do {
    RangeScaleFirstChar[SymType][Context] += FreqFirstChar[SymType][Context][i]
        = (FreqFirstChar[SymType][Context][i] + 1) >> 1;
  } while (i--);
  return;
}

void rescaleFirstCharBinary(uint8_t Context) {
  RangeScaleFirstChar[0][Context] = FreqFirstCharBinary[Context][0] = (FreqFirstCharBinary[Context][0] + 1) >> 1;
  uint8_t i = 1;
  do {
    RangeScaleFirstChar[0][Context] += FreqFirstCharBinary[Context][i] = (FreqFirstCharBinary[Context][i] + 1) >> 1;
  } while (++i != 0x20);
  RangeScaleFirstCharSection[Context][0] = RangeScaleFirstChar[0][Context];
  do {
    RangeScaleFirstChar[0][Context] += FreqFirstCharBinary[Context][i] = (FreqFirstCharBinary[Context][i] + 1) >> 1;
  } while (++i != 0x40);
  RangeScaleFirstCharSection[Context][1] = RangeScaleFirstChar[0][Context];
  do {
    RangeScaleFirstChar[0][Context] += FreqFirstCharBinary[Context][i] = (FreqFirstCharBinary[Context][i] + 1) >> 1;
  } while (++i != 0x60);
  RangeScaleFirstCharSection[Context][2] = RangeScaleFirstChar[0][Context] - RangeScaleFirstCharSection[Context][1];
  do {
    RangeScaleFirstChar[0][Context] += FreqFirstCharBinary[Context][i] = (FreqFirstCharBinary[Context][i] + 1) >> 1;
  } while (++i != 0x80);
  RangeScaleFirstCharSection[Context][3] = RangeScaleFirstChar[0][Context];
  do {
    RangeScaleFirstChar[0][Context] += FreqFirstCharBinary[Context][i] = (FreqFirstCharBinary[Context][i] + 1) >> 1;
  } while (++i != 0xA0);
  RangeScaleFirstCharSection[Context][4] = RangeScaleFirstChar[0][Context] - RangeScaleFirstCharSection[Context][3];
  do {
    RangeScaleFirstChar[0][Context] += FreqFirstCharBinary[Context][i] = (FreqFirstCharBinary[Context][i] + 1) >> 1;
  } while (++i != 0xC0);
  RangeScaleFirstCharSection[Context][5] = RangeScaleFirstChar[0][Context] - RangeScaleFirstCharSection[Context][3];
  do {
    RangeScaleFirstChar[0][Context] += FreqFirstCharBinary[Context][i] = (FreqFirstCharBinary[Context][i] + 1) >> 1;
  } while (++i != 0xE0);
  RangeScaleFirstCharSection[Context][6] = RangeScaleFirstChar[0][Context] - RangeScaleFirstCharSection[Context][5]
      - RangeScaleFirstCharSection[Context][3];
  do {
    RangeScaleFirstChar[0][Context] += FreqFirstCharBinary[Context][i] = (FreqFirstCharBinary[Context][i] + 1) >> 1;
  } while (++i);
  return;
}

void InitSymbolFirstChar(uint8_t trailing_char, uint8_t leading_char) {
  SymbolFirstChar[0][trailing_char][leading_char] = leading_char;
  SymbolFirstChar[1][trailing_char][leading_char] = leading_char;
  SymbolFirstChar[2][trailing_char][leading_char] = leading_char;
  SymbolFirstChar[3][trailing_char][leading_char] = leading_char;
  return;
}

void InitFreqFirstChar(uint8_t trailing_char, uint8_t leading_char) {
  FreqFirstChar[0][trailing_char][leading_char] = 1;
  FreqFirstChar[1][trailing_char][leading_char] = 1;
  FreqFirstChar[2][trailing_char][leading_char] = 1;
  FreqFirstChar[3][trailing_char][leading_char] = 1;
  RangeScaleFirstChar[0][trailing_char]++;
  RangeScaleFirstChar[1][trailing_char]++;
  RangeScaleFirstChar[2][trailing_char]++;
  RangeScaleFirstChar[3][trailing_char]++;
  return;
}

void InitFirstCharBin(uint8_t trailing_char, uint8_t leading_char, uint8_t code_length, uint8_t cap_symbol_defined,
    uint8_t cap_lock_symbol_defined) {
  if (RangeScaleFirstChar[0][trailing_char]
      || ((trailing_char == 'C') && (cap_symbol_defined || cap_lock_symbol_defined))) {
    uint8_t j2 = leading_char;
    while (SymbolFirstChar[0][trailing_char][j2] != (uint8_t)leading_char)
      j2++;
    if (code_length < 8) {
      FreqFirstChar[0][trailing_char][j2] = 1 << (8 - code_length);
      RangeScaleFirstChar[0][trailing_char] += 1 << (8 - code_length);
    }
    else {
      FreqFirstChar[0][trailing_char][j2] = 1;
      RangeScaleFirstChar[0][trailing_char] += 1;
    }
    if (RangeScaleFirstChar[0][trailing_char] > FREQ_FIRST_CHAR_BOT)
      rescaleFirstChar(0, trailing_char);
    j2 = leading_char;
    while (SymbolFirstChar[1][trailing_char][j2] != (uint8_t)leading_char)
      j2++;
    if (code_length < 8) {
      FreqFirstChar[1][trailing_char][j2] = 1 << (8 - code_length);
      RangeScaleFirstChar[1][trailing_char] += 1 << (8 - code_length);
    }
    else {
      FreqFirstChar[1][trailing_char][j2] = 1;
      RangeScaleFirstChar[1][trailing_char] += 1;
    }
    if (RangeScaleFirstChar[1][trailing_char] > FREQ_FIRST_CHAR_BOT)
      rescaleFirstChar(1, trailing_char);
    j2 = leading_char;
    while (SymbolFirstChar[2][trailing_char][j2] != (uint8_t)leading_char)
      j2++;
    if (code_length < 8) {
      FreqFirstChar[2][trailing_char][j2] = 1 << (8 - code_length);
      RangeScaleFirstChar[2][trailing_char] += 1 << (8 - code_length);
    }
    else {
      FreqFirstChar[2][trailing_char][j2] = 1;
      RangeScaleFirstChar[2][trailing_char] += 1;
    }
    if (RangeScaleFirstChar[2][trailing_char] > FREQ_FIRST_CHAR_BOT)
      rescaleFirstChar(2, trailing_char);
    j2 = leading_char;
    while (SymbolFirstChar[3][trailing_char][j2] != (uint8_t)leading_char)
      j2++;
    if (code_length < 8) {
      FreqFirstChar[3][trailing_char][j2] = 1 << (8 - code_length);
      RangeScaleFirstChar[3][trailing_char] += 1 << (8 - code_length);
    }
    else {
      FreqFirstChar[3][trailing_char][j2] = 1;
      RangeScaleFirstChar[3][trailing_char] += 1;
    }
    if (RangeScaleFirstChar[3][trailing_char] > FREQ_FIRST_CHAR_BOT)
      rescaleFirstChar(3, trailing_char);
  }
  return;
}

void InitFirstCharBinBinary(uint8_t trailing_char, uint8_t leading_char, uint8_t code_length) {
  if (RangeScaleFirstChar[0][trailing_char]) {
    if (code_length < 8) {
      FreqFirstCharBinary[trailing_char][leading_char] = 1 << (8 - code_length);
      RangeScaleFirstChar[0][trailing_char] += 1 << (8 - code_length);
      if (leading_char < 0x80) {
        RangeScaleFirstCharSection[trailing_char][3] += 1 << (8 - code_length);
        if (leading_char < 0x40) {
          RangeScaleFirstCharSection[trailing_char][1] += 1 << (8 - code_length);
          if (leading_char < 0x20)
            RangeScaleFirstCharSection[trailing_char][0] += 1 << (8 - code_length);
        }
        else if (leading_char < 0x60)
          RangeScaleFirstCharSection[trailing_char][2] += 1 << (8 - code_length);
      }
      else if (leading_char < 0xC0) {
        RangeScaleFirstCharSection[trailing_char][5] += 1 << (8 - code_length);
        if (leading_char < 0xA0)
          RangeScaleFirstCharSection[trailing_char][4] += 1 << (8 - code_length);
      }
      else if (leading_char < 0xE0)
        RangeScaleFirstCharSection[trailing_char][6] += 1 << (8 - code_length);
    }
    else {
      FreqFirstCharBinary[trailing_char][leading_char] = 1;
      RangeScaleFirstChar[0][trailing_char] += 1;
      if (leading_char < 0x80) {
        RangeScaleFirstCharSection[trailing_char][3] += 1;
        if (leading_char < 0x40) {
          RangeScaleFirstCharSection[trailing_char][1] += 1;
          if (leading_char < 0x20)
            RangeScaleFirstCharSection[trailing_char][0] += 1;
        }
        else if (leading_char < 0x60)
          RangeScaleFirstCharSection[trailing_char][2] += 1;
      }
      else if (leading_char < 0xC0) {
        RangeScaleFirstCharSection[trailing_char][5] += 1;
        if (leading_char < 0xA0)
          RangeScaleFirstCharSection[trailing_char][4] += 1;
      }
      else if (leading_char < 0xE0)
        RangeScaleFirstCharSection[trailing_char][6] += 1;
    }
    if (RangeScaleFirstChar[0][trailing_char] > FREQ_FIRST_CHAR_BOT)
      rescaleFirstCharBinary(trailing_char);
  }
  return;
}

void InitTrailingCharBin(uint8_t trailing_char, uint8_t leading_char, uint8_t code_length) {
  if (code_length < 8) {
    uint16_t init_freq = 1 << (8 - code_length);
    FreqFirstChar[0][trailing_char][leading_char] = init_freq;
    RangeScaleFirstChar[0][trailing_char] += init_freq;
    FreqFirstChar[1][trailing_char][leading_char] = init_freq;
    RangeScaleFirstChar[1][trailing_char] += init_freq;
    FreqFirstChar[2][trailing_char][leading_char] = init_freq;
    RangeScaleFirstChar[2][trailing_char] += init_freq;
    FreqFirstChar[3][trailing_char][leading_char] = init_freq;
    RangeScaleFirstChar[3][trailing_char] += init_freq;
  }
  else {
    InitFreqFirstChar(trailing_char, leading_char);
  }
  return;
}

void InitTrailingCharBinary(uint32_t trailing_char, uint8_t * symbol_lengths) {
  uint8_t leading_char = 0xFF;
  do {
    uint16_t init_freq;
    if (symbol_lengths[leading_char] < 8)
      init_freq = 1 << (8 - symbol_lengths[leading_char]);
    else
      init_freq = 1;
    if (RangeScaleFirstChar[0][leading_char] || (leading_char == trailing_char)) {
      FreqFirstCharBinary[trailing_char][leading_char] = init_freq;
      RangeScaleFirstChar[0][trailing_char] += init_freq;
      if (leading_char < 0x80) {
        RangeScaleFirstCharSection[trailing_char][3] += init_freq;
        if (leading_char < 0x40) {
          RangeScaleFirstCharSection[trailing_char][1] += init_freq;
          if (leading_char < 0x20)
            RangeScaleFirstCharSection[trailing_char][0] += init_freq;
        }
        else if (leading_char < 0x60)
          RangeScaleFirstCharSection[trailing_char][2] += init_freq;
      }
      else if (leading_char < 0xC0) {
        RangeScaleFirstCharSection[trailing_char][5] += init_freq;
        if (leading_char < 0xA0)
          RangeScaleFirstCharSection[trailing_char][4] += init_freq;
      }
      else if (leading_char < 0xE0)
        RangeScaleFirstCharSection[trailing_char][6] += init_freq;
    }
  } while (leading_char-- != 0);
  return;
}

void InitBaseSymbolCap(uint8_t BaseSymbol, uint8_t max_symbol, uint8_t new_symbol_code_length,
    uint8_t * cap_symbol_defined_ptr, uint8_t * cap_lock_symbol_defined_ptr, uint8_t * symbol_lengths) {
  uint8_t j1 = max_symbol;
  do {
    InitFirstCharBin(j1, BaseSymbol, new_symbol_code_length, *cap_symbol_defined_ptr, *cap_lock_symbol_defined_ptr);
  } while (--j1 != 'Z');
  j1 = 'A' - 1;
  do {
    InitFirstCharBin(j1, BaseSymbol, new_symbol_code_length, *cap_symbol_defined_ptr, *cap_lock_symbol_defined_ptr);
  } while (j1--);
  if ((BaseSymbol & 0xFE) == 0x42) {
    j1 = 'z';
    if ((*cap_symbol_defined_ptr | *cap_lock_symbol_defined_ptr) == 0) {
      do {
        InitSymbolFirstChar('C', j1);
        if (RangeScaleFirstChar[0][j1])
          InitTrailingCharBin('C', j1, symbol_lengths[j1]);
      } while (j1-- != 'a');
      do {
        InitSymbolFirstChar('C', j1);
      } while (j1-- != 0);
    }
    if (BaseSymbol == 'C')
      *cap_symbol_defined_ptr = 1;
    else
      *cap_lock_symbol_defined_ptr = 1;
  }
  else {
    if ((BaseSymbol >= 'a') && (BaseSymbol <= 'z'))
      InitFirstCharBin('C', BaseSymbol, new_symbol_code_length, *cap_symbol_defined_ptr, *cap_lock_symbol_defined_ptr);
    j1 = max_symbol;
    do {
      InitSymbolFirstChar(BaseSymbol, j1);
      if (symbol_lengths[j1])
        InitTrailingCharBin(BaseSymbol, j1, symbol_lengths[j1]);
    } while (j1--);
  }
  return;
}

void UpFreqMtfQueueNum(uint8_t Context, uint8_t mtf_queue_number) {
  FreqMtfQueueNum[Context][mtf_queue_number] += RangeScaleMtfQueueNum[Context] >> 5;
  if ((RangeScaleMtfQueueNum[Context] += RangeScaleMtfQueueNum[Context] >> 5) > FREQ_MTF_QUEUE_NUM_BOT)
    rescaleMtfQueueNum(Context);
}

void DoubleRange() {
  range <<= 1;
}

void DoubleRangeDown() {
  low -= range;
  range <<= 1;
}

void DecreaseLow(uint32_t range_factor) {
  low -= range * range_factor;
}

void IncreaseLow(uint32_t range_factor) {
  low += range * range_factor;
}

void MultiplyRange(uint32_t range_factor) {
  range *= range_factor;
}

void SetOutBuffer(uint8_t * bufptr) {
  OutBuffer = bufptr;
}

void WriteOutBuffer(uint8_t value) {
  OutBuffer[OutCharNum++] = value;
}

void NormalizeEncoder(uint32_t bot) {
  while ((low ^ (low + range)) < TOP || (range < (bot) && ((range = -low & ((bot) - 1)), 1))) {
    OutBuffer[OutCharNum++] = (uint8_t)(low >> 24);
    range <<= 8;
    low <<= 8;
  }
  return;
}

void EncodeDictType(uint8_t Context) {
  NormalizeEncoder(FREQ_SYM_TYPE_BOT);
  range = FreqSymType[Context][0] * (range >> 14);
  uint16_t sum = 0;
  uint16_t sub;
  sum += (sub = FreqSymType[Context][1] >> 6);
  FreqSymType[Context][1] -= sub;
  sum += (sub = FreqSymType[Context][2] >> 6);
  FreqSymType[Context][2] -= sub;
  sum += (sub = FreqSymType[Context][3] >> 6);
  FreqSymType[Context][3] -= sub;
  FreqSymType[Context][0] += sum;
  return;
}

void EncodeNewType(uint8_t Context) {
  NormalizeEncoder(FREQ_SYM_TYPE_BOT);
  low += FreqSymType[Context][0] * (range >>= 14);
  range *= FreqSymType[Context][1];
  uint16_t sum = 0;
  uint16_t sub;
  sum += (sub = FreqSymType[Context][0] >> 6);
  FreqSymType[Context][0] -= sub;
  sum += (sub = FreqSymType[Context][2] >> 6);
  FreqSymType[Context][2] -= sub;
  sum += (sub = FreqSymType[Context][3] >> 6);
  FreqSymType[Context][3] -= sub;
  FreqSymType[Context][1] += sum;
  return;
}

void EncodeMtfgType(uint8_t Context) {
  NormalizeEncoder(FREQ_SYM_TYPE_BOT);
  low += (FreqSymType[Context][0] + FreqSymType[Context][1]) * (range >>= 14);
  range *= FreqSymType[Context][2];
  uint16_t sum = 0;
  uint16_t sub;
  sum += (sub = FreqSymType[Context][0] >> 6);
  FreqSymType[Context][0] -= sub;
  sum += (sub = FreqSymType[Context][1] >> 6);
  FreqSymType[Context][1] -= sub;
  sum += (sub = FreqSymType[Context][3] >> 6);
  FreqSymType[Context][3] -= sub;
  FreqSymType[Context][2] += sum;
  return;
}

void EncodeMtfType(uint8_t Context) {
  NormalizeEncoder(FREQ_SYM_TYPE_BOT);
  uint32_t saved_low = low;
  low += (FREQ_SYM_TYPE_BOT - FreqSymType[Context][3]) * (range >> 14);
  range -= low - saved_low;
  uint16_t sum = 0;
  uint16_t sub;
  sum += (sub = FreqSymType[Context][0] >> 6);
  FreqSymType[Context][0] -= sub;
  sum += (sub = FreqSymType[Context][1] >> 6);
  FreqSymType[Context][1] -= sub;
  sum += (sub = FreqSymType[Context][2] >> 6);
  FreqSymType[Context][2] -= sub;
  FreqSymType[Context][3] += sum;
  return;
}

void EncodeMtfQueueNum(uint8_t Context, uint8_t mtf_queue_number) {
  NormalizeEncoder(FREQ_MTF_QUEUE_NUM_BOT);
  if (mtf_queue_number == 0) {
    range = FreqMtfQueueNum[Context][0] * (range / RangeScaleMtfQueueNum[Context]);
    FreqMtfQueueNum[Context][0] += RangeScaleMtfQueueNum[Context] >> 5;
  }
  else {
    RangeLow = FreqMtfQueueNum[Context][0];
    FoundIndex = 1;
    while (FoundIndex != mtf_queue_number)
      RangeLow += FreqMtfQueueNum[Context][FoundIndex++];
    low += RangeLow * (range /= RangeScaleMtfQueueNum[Context]);
    range *= FreqMtfQueueNum[Context][FoundIndex];
    FreqMtfQueueNum[Context][FoundIndex] += RangeScaleMtfQueueNum[Context] >> 5;
  }
  if ((RangeScaleMtfQueueNum[Context] += RangeScaleMtfQueueNum[Context] >> 5) > FREQ_MTF_QUEUE_NUM_BOT)
    rescaleMtfQueueNum(Context);
  return;
}

void EncodeMtfQueueNumLastSymbol(uint8_t Context, uint8_t mtf_queue_number) {
  NormalizeEncoder(FREQ_MTF_QUEUE_NUM_BOT);
  if (mtf_queue_number == 0)
    range = FreqMtfQueueNum[Context][0] * (range / RangeScaleMtfQueueNum[Context]);
  else {
    RangeLow = FreqMtfQueueNum[Context][0];
    FoundIndex = 1;
    while (FoundIndex != mtf_queue_number)
      RangeLow += FreqMtfQueueNum[Context][FoundIndex++];
    low += RangeLow * (range /= RangeScaleMtfQueueNum[Context]);
    range *= FreqMtfQueueNum[Context][FoundIndex];
  }
  return;
}

void EncodeMtfQueuePos(uint8_t Context, uint8_t mtf_queue_number, uint8_t * mtf_queue_size, uint8_t queue_position) {
  NormalizeEncoder(FREQ_MTF_QUEUE_POS_BOT);
  uint16_t RangeScale = RangeScaleMtfQueuePos[Context][mtf_queue_number];
  if (mtf_queue_size[mtf_queue_number+2] != MTF_QUEUE_SIZE) {
    uint8_t tqp = MTF_QUEUE_SIZE - 1;
    do {
      RangeScale -= FreqMtfQueuePos[Context][mtf_queue_number][tqp];
    } while (tqp-- != mtf_queue_size[mtf_queue_number+2]);
  }
  if (queue_position == 0) {
    range = FreqMtfQueuePos[Context][mtf_queue_number][0] * (range / RangeScale);
    FreqMtfQueuePos[Context][mtf_queue_number][0] += UP_FREQ_MTF_QUEUE_POS;
  }
  else {
    RangeLow = FreqMtfQueuePos[Context][mtf_queue_number][0];
    FoundIndex = 1;
    while (FoundIndex != queue_position)
      RangeLow += FreqMtfQueuePos[Context][mtf_queue_number][FoundIndex++];
    low += RangeLow * (range /= RangeScale);
    range *= FreqMtfQueuePos[Context][mtf_queue_number][FoundIndex];
    if (FoundIndex >= 4) {
      if (FoundIndex == 4) {
        FreqMtfQueuePos[Context][mtf_queue_number][FoundIndex] += UP_FREQ_MTF_QUEUE_POS - 1;
        FreqMtfQueuePos[Context][mtf_queue_number][FoundIndex+1] += 1;
      }
      else if (FoundIndex == 63) {
        FreqMtfQueuePos[Context][mtf_queue_number][FoundIndex-1] += 1;
        FreqMtfQueuePos[Context][mtf_queue_number][FoundIndex] += UP_FREQ_MTF_QUEUE_POS - 1;
      }
      else {
        FreqMtfQueuePos[Context][mtf_queue_number][FoundIndex-1] += 1;
        FreqMtfQueuePos[Context][mtf_queue_number][FoundIndex] += UP_FREQ_MTF_QUEUE_POS - 2;
        FreqMtfQueuePos[Context][mtf_queue_number][FoundIndex+1] += 1;
      }
    }
    else
      FreqMtfQueuePos[Context][mtf_queue_number][FoundIndex] += UP_FREQ_MTF_QUEUE_POS;
  }
  if ((RangeScaleMtfQueuePos[Context][mtf_queue_number] += UP_FREQ_MTF_QUEUE_POS) > FREQ_MTF_QUEUE_POS_BOT)
    rescaleMtfQueuePos(Context, mtf_queue_number);
  return;
}

void EncodeMtfgQueuePos(uint8_t Context, uint8_t queue_position) {
  NormalizeEncoder(FREQ_MTFG_QUEUE_POS_BOT);
  if (queue_position == 0) {
    range = FreqMtfgQueuePos[Context][0] * (range / RangeScaleMtfgQueuePos[Context]);
    FreqMtfgQueuePos[Context][0] += UP_FREQ_MTFG_QUEUE_POS;
  }
  else {
    RangeLow = FreqMtfgQueuePos[Context][0];
    FoundIndex = 1;
    while (FoundIndex != queue_position)
      RangeLow += FreqMtfgQueuePos[Context][FoundIndex++];
    low += RangeLow * (range /= RangeScaleMtfgQueuePos[Context]);
    range *= FreqMtfgQueuePos[Context][FoundIndex];
    if (FoundIndex >= 4) {
      if (FoundIndex == 4) {
        FreqMtfgQueuePos[Context][FoundIndex] += UP_FREQ_MTFG_QUEUE_POS - 2;
        FreqMtfgQueuePos[Context][FoundIndex+1] += 2;
      }
      else if (FoundIndex == 255) {
        FreqMtfgQueuePos[Context][FoundIndex-1] += 2;
        FreqMtfgQueuePos[Context][FoundIndex] += UP_FREQ_MTFG_QUEUE_POS - 2;
      }
      else {
        FreqMtfgQueuePos[Context][FoundIndex-1] += 2;
        FreqMtfgQueuePos[Context][FoundIndex] += UP_FREQ_MTFG_QUEUE_POS - 4;
        FreqMtfgQueuePos[Context][FoundIndex+1] += 2;
      }
    }
    else
      FreqMtfgQueuePos[Context][FoundIndex] += UP_FREQ_MTFG_QUEUE_POS;
  }
  if ((RangeScaleMtfgQueuePos[Context] += UP_FREQ_MTFG_QUEUE_POS) > FREQ_MTFG_QUEUE_POS_BOT)
    rescaleMtfgQueuePos(Context);
  return;
}

void EncodeSID(uint8_t Context, uint8_t SIDSymbol) {
  NormalizeEncoder(FREQ_SID_BOT);
  if (SIDSymbol == 0) {
    range = FreqSID[Context][0] * (range / RangeScaleSID[Context]);
    FreqSID[Context][0] += UP_FREQ_SID;
  }
  else {
    RangeLow = FreqSID[Context][0];
    uint8_t Symbol = 1;
    while (Symbol != SIDSymbol)
      RangeLow += FreqSID[Context][Symbol++];
    low += RangeLow * (range /= RangeScaleSID[Context]);
    range *= FreqSID[Context][SIDSymbol];
    FreqSID[Context][SIDSymbol] += UP_FREQ_SID;
  }
  if ((RangeScaleSID[Context] += UP_FREQ_SID) > FREQ_SID_BOT)
    rescaleSID(Context);
  return;
}

void EncodeExtraLength(uint8_t Symbol) {
  NormalizeEncoder(1 << 2);
  range >>= 2;
  low += Symbol * range;
  return;
}

void EncodeINST(uint8_t Context, uint8_t SIDSymbol, uint8_t Symbol) {
  NormalizeEncoder(FREQ_INST_BOT);
  if (Symbol == 0) {
    range = FreqINST[Context][SIDSymbol][0] * (range / RangeScaleINST[Context][SIDSymbol]);
    if (RangeScaleINST[Context][SIDSymbol] >= (FREQ_INST_BOT >> 1)) {
      FreqINST[Context][SIDSymbol][0] += RangeScaleINST[Context][SIDSymbol] >> 11;
      if ((RangeScaleINST[Context][SIDSymbol] += (RangeScaleINST[Context][SIDSymbol]) >> 11) > FREQ_INST_BOT)
        rescaleINST(Context, SIDSymbol);
    }
    else {
      FreqINST[Context][SIDSymbol][0] += UP_FREQ_INST;
      RangeScaleINST[Context][SIDSymbol] += UP_FREQ_INST;
    }
  }
  else {
    RangeLow = FreqINST[Context][SIDSymbol][0];
    FoundIndex = 1;
    while (FoundIndex != Symbol)
      RangeLow += FreqINST[Context][SIDSymbol][FoundIndex++];
    low += RangeLow * (range /= RangeScaleINST[Context][SIDSymbol]);
    range *= FreqINST[Context][SIDSymbol][FoundIndex];
    if (RangeScaleINST[Context][SIDSymbol] >= (FREQ_INST_BOT >> 1)) {
      FreqINST[Context][SIDSymbol][FoundIndex] += RangeScaleINST[Context][SIDSymbol] >> 11;
      if ((RangeScaleINST[Context][SIDSymbol] += (RangeScaleINST[Context][SIDSymbol]) >> 11) > FREQ_INST_BOT)
        rescaleINST(Context, SIDSymbol);
    }
    else {
      FreqINST[Context][SIDSymbol][FoundIndex] += UP_FREQ_INST;
      RangeScaleINST[Context][SIDSymbol] += UP_FREQ_INST;
    }
  }
  return;
}

void EncodeERG(uint8_t Context, uint8_t Symbol) {
  NormalizeEncoder(FREQ_ERG_BOT);
  if ((Symbol) == 0) {
    range = FreqERG[Context] * (range / RangeScaleERG[Context]);
    FreqERG[Context] += UP_FREQ_ERG;
  }
  else {
    low += FreqERG[Context] * (range /= RangeScaleERG[Context]);
    range *= RangeScaleERG[Context] - FreqERG[Context];
  }
  if ((RangeScaleERG[Context] += UP_FREQ_ERG) > FREQ_ERG_BOT) {
    RangeScaleERG[Context] = (FREQ_ERG_BOT >> 1) + 1;
    FreqERG[Context] = (FreqERG[Context] + 1) >> 1;
  }
  return;
}

void EncodeWordTag(uint8_t Symbol) {
  NormalizeEncoder(FREQ_WORD_TAG_BOT);
  if (Symbol == 0) {
    range = FreqWordTag * (range / RangeScaleWordTag);
    FreqWordTag += UP_FREQ_WORD_TAG;
  }
  else {
    low += FreqWordTag * (range /= RangeScaleWordTag);
    range *= RangeScaleWordTag - FreqWordTag;
  }
  if ((RangeScaleWordTag += UP_FREQ_WORD_TAG) > FREQ_WORD_TAG_BOT) {
    RangeScaleWordTag = (FREQ_WORD_TAG_BOT >> 1) + 1;
    FreqWordTag = (FreqWordTag + 1) >> 1;
  }
  return;
}

void EncodeShortDictionarySymbol(uint8_t Length, uint16_t BinNum, uint16_t DictionaryBins, uint16_t CodeBins) {
  NormalizeEncoder((uint32_t)1 << 12);
  low += BinNum * (range /= DictionaryBins);
  range = (uint32_t)CodeBins * (range << (12 - (Length)));
  return;
}

void EncodeLongDictionarySymbol(uint32_t BinCode, uint16_t BinNum, uint16_t DictionaryBins, uint8_t CodeLength,
    uint16_t CodeBins) {
  NormalizeEncoder((uint32_t)1 << 12);
  low += BinNum * (range /= DictionaryBins);
  NormalizeEncoder((uint32_t)1 << (CodeLength - 12));
  low += BinCode * (range >>= CodeLength - 12);
  range *= (uint32_t)CodeBins;
  return;
}

void EncodeBaseSymbol(uint32_t BaseSymbol, uint8_t Bits, uint32_t NumBaseSymbols) {
  NormalizeEncoder((uint32_t)1 << Bits);
  low += BaseSymbol * (range /= NumBaseSymbols);
  return;
}

void EncodeFirstChar(uint8_t Symbol, uint8_t SymType, uint8_t LastChar) {
  NormalizeEncoder(FREQ_FIRST_CHAR_BOT);
  if (Symbol == SymbolFirstChar[SymType][LastChar][0]) {
    range = FreqFirstChar[SymType][LastChar][0] * (range / RangeScaleFirstChar[SymType][LastChar]);
    if (RangeScaleFirstChar[SymType][LastChar] >= (FREQ_FIRST_CHAR_BOT >> 1)) {
      FreqFirstChar[SymType][LastChar][0] += RangeScaleFirstChar[SymType][LastChar] >> 9;
      if ((RangeScaleFirstChar[SymType][LastChar] += (RangeScaleFirstChar[SymType][LastChar] >> 9)) > FREQ_FIRST_CHAR_BOT)
        rescaleFirstChar(SymType, LastChar);
    }
    else {
      FreqFirstChar[SymType][LastChar][0] += UP_FREQ_FIRST_CHAR;
      RangeScaleFirstChar[SymType][LastChar] += UP_FREQ_FIRST_CHAR;
    }
  }
  else {
    RangeLow = FreqFirstChar[SymType][LastChar][0];
    FoundIndex = 1;
    while (SymbolFirstChar[SymType][LastChar][FoundIndex] != Symbol)
      RangeLow += FreqFirstChar[SymType][LastChar][FoundIndex++];
    low += RangeLow * (range /= RangeScaleFirstChar[SymType][LastChar]);
    range *= FreqFirstChar[SymType][LastChar][FoundIndex];
    if (RangeScaleFirstChar[SymType][LastChar] >= (FREQ_FIRST_CHAR_BOT >> 1)) {
      FreqFirstChar[SymType][LastChar][FoundIndex] += RangeScaleFirstChar[SymType][LastChar] >> 9;
      if ((RangeScaleFirstChar[SymType][LastChar] += (RangeScaleFirstChar[SymType][LastChar] >> 9)) > FREQ_FIRST_CHAR_BOT)
        rescaleFirstChar(SymType, LastChar);
    }
    else {
      FreqFirstChar[SymType][LastChar][FoundIndex] += UP_FREQ_FIRST_CHAR;
      RangeScaleFirstChar[SymType][LastChar] += UP_FREQ_FIRST_CHAR;
    }
    if (FreqFirstChar[SymType][LastChar][FoundIndex] > FreqFirstChar[SymType][LastChar][FoundIndex-1]) {
      uint16_t SavedFreq = FreqFirstChar[SymType][LastChar][FoundIndex];
      do {
        FreqFirstChar[SymType][LastChar][FoundIndex] = FreqFirstChar[SymType][LastChar][FoundIndex-1];
        SymbolFirstChar[SymType][LastChar][FoundIndex] = SymbolFirstChar[SymType][LastChar][FoundIndex-1];
      } while ((--FoundIndex) && (SavedFreq > FreqFirstChar[SymType][LastChar][FoundIndex-1]));
      FreqFirstChar[SymType][LastChar][FoundIndex] = SavedFreq;
      SymbolFirstChar[SymType][LastChar][FoundIndex] = Symbol;
    }
  }
  return;
}

void EncodeFirstCharBinary(uint8_t Symbol, uint8_t LastChar) {
  NormalizeEncoder(FREQ_FIRST_CHAR_BOT);
  if (RangeScaleFirstCharSection[LastChar][3] > count) {
    RangeScaleFirstCharSection[LastChar][3] += UP_FREQ_FIRST_CHAR;
    if (RangeScaleFirstCharSection[LastChar][1] > count) {
      RangeScaleFirstCharSection[LastChar][1] += UP_FREQ_FIRST_CHAR;
      if (RangeScaleFirstCharSection[LastChar][0] > count) {
        RangeScaleFirstCharSection[LastChar][0] += UP_FREQ_FIRST_CHAR;
        if (Symbol == 0) {
          range = FreqFirstCharBinary[LastChar][0] * (range / RangeScaleFirstChar[0][LastChar]);
          FreqFirstCharBinary[LastChar][0] += UP_FREQ_FIRST_CHAR;
        }
        else {
          RangeLow = FreqFirstCharBinary[LastChar][0];
          FoundIndex = 1;
          while (FoundIndex != Symbol)
            RangeLow += FreqFirstCharBinary[LastChar][FoundIndex++];
          low += RangeLow * (range /= RangeScaleFirstChar[0][LastChar]);
          range *= FreqFirstCharBinary[LastChar][FoundIndex];
          FreqFirstCharBinary[LastChar][FoundIndex] += UP_FREQ_FIRST_CHAR;
        }
      }
      else {
        RangeLow = RangeScaleFirstCharSection[LastChar][0];
        FoundIndex = 0x20;
        while (FoundIndex != Symbol)
          RangeLow += FreqFirstCharBinary[LastChar][FoundIndex++];
        low += RangeLow * (range /= RangeScaleFirstChar[0][LastChar]);
        range *= FreqFirstCharBinary[LastChar][FoundIndex];
        FreqFirstCharBinary[LastChar][FoundIndex] += UP_FREQ_FIRST_CHAR;
      }
    }
    else {
      RangeLow = RangeScaleFirstCharSection[LastChar][1];
      if (RangeScaleFirstCharSection[LastChar][2] > count) {
        RangeScaleFirstCharSection[LastChar][2] += UP_FREQ_FIRST_CHAR;
        FoundIndex = 0x40;
        while (FoundIndex != Symbol)
          RangeLow += FreqFirstCharBinary[LastChar][FoundIndex++];
        low += RangeLow * (range /= RangeScaleFirstChar[0][LastChar]);
        range *= FreqFirstCharBinary[LastChar][FoundIndex];
        FreqFirstCharBinary[LastChar][FoundIndex] += UP_FREQ_FIRST_CHAR;
      }
      else {
        RangeLow += RangeScaleFirstCharSection[LastChar][2];
        FoundIndex = 0x60;
        while (FoundIndex != Symbol)
          RangeLow += FreqFirstCharBinary[LastChar][FoundIndex++];
        low += RangeLow * (range /= RangeScaleFirstChar[0][LastChar]);
        range *= FreqFirstCharBinary[LastChar][FoundIndex];
        FreqFirstCharBinary[LastChar][FoundIndex] += UP_FREQ_FIRST_CHAR;
      }
    }
  }
  else {
    RangeLow = RangeScaleFirstCharSection[LastChar][3];
    if (RangeLow + RangeScaleFirstCharSection[LastChar][5] > count) {
      RangeScaleFirstCharSection[LastChar][5] += UP_FREQ_FIRST_CHAR;
      if (RangeScaleFirstCharSection[LastChar][4] > count) {
        RangeScaleFirstCharSection[LastChar][4] += UP_FREQ_FIRST_CHAR;
        FoundIndex = 0x80;
        while (FoundIndex != Symbol)
          RangeLow += FreqFirstCharBinary[LastChar][FoundIndex++];
        low += RangeLow * (range /= RangeScaleFirstChar[0][LastChar]);
        range *= FreqFirstCharBinary[LastChar][FoundIndex];
        FreqFirstCharBinary[LastChar][FoundIndex] += UP_FREQ_FIRST_CHAR;
      }
      else {
        RangeLow += RangeScaleFirstCharSection[LastChar][4];
        FoundIndex = 0xA0;
        while (FoundIndex != Symbol)
          RangeLow += FreqFirstCharBinary[LastChar][FoundIndex++];
        low += RangeLow * (range /= RangeScaleFirstChar[0][LastChar]);
        range *= FreqFirstCharBinary[LastChar][FoundIndex];
        FreqFirstCharBinary[LastChar][FoundIndex] += UP_FREQ_FIRST_CHAR;
      }
    }
    else {
      RangeLow += RangeScaleFirstCharSection[LastChar][5];
      if (RangeScaleFirstCharSection[LastChar][6] > count) {
        RangeScaleFirstCharSection[LastChar][6] += UP_FREQ_FIRST_CHAR;
        FoundIndex = 0xC0;
        while (FoundIndex != Symbol)
          RangeLow += FreqFirstCharBinary[LastChar][FoundIndex++];
        low += RangeLow * (range /= RangeScaleFirstChar[0][LastChar]);
        range *= FreqFirstCharBinary[LastChar][FoundIndex];
        FreqFirstCharBinary[LastChar][FoundIndex] += UP_FREQ_FIRST_CHAR;
      }
      else {
        RangeLow += RangeScaleFirstCharSection[LastChar][6];
        FoundIndex = 0xE0;
        while (FoundIndex != Symbol)
          RangeLow += FreqFirstCharBinary[LastChar][FoundIndex++];
        low += RangeLow * (range /= RangeScaleFirstChar[0][LastChar]);
        range *= FreqFirstCharBinary[LastChar][FoundIndex];
        FreqFirstCharBinary[LastChar][FoundIndex] += UP_FREQ_FIRST_CHAR;
      }
    }
  }
  if ((RangeScaleFirstChar[0][LastChar] += UP_FREQ_FIRST_CHAR) > FREQ_FIRST_CHAR_BOT)
    rescaleFirstCharBinary(LastChar);
  return;
}

void WriteInCharNum(uint32_t value) {
  InCharNum = value;
}

void WriteOutCharNum(uint32_t value) {
  OutCharNum = value;
}

uint32_t ReadOutCharNum() {
  return(OutCharNum);
}

void InitEncoder(uint8_t max_regular_code_length, uint8_t num_inst_codes, uint8_t cap_encoded, uint8_t UTF8_compliant,
    uint8_t use_mtf, uint8_t use_mtfg) {
  low = 0, range = -1;
  StartModelSymType(use_mtf, use_mtfg);
  StartModelMtfQueueNum();
  StartModelMtfQueuePos();
  StartModelMtfgQueuePos(max_regular_code_length);
  StartModelSID();
  StartModelINST(num_inst_codes);
  StartModelERG();
  StartModelWordTag();
  if (cap_encoded || UTF8_compliant) {
    StartModelFirstChar();
  }
  else {
    StartModelFirstCharBinary();
  }
}

void FinishEncoder() {
  while (low ^ (low + range)) {
    OutBuffer[OutCharNum++] = (uint8_t)(low >> 24);
    low <<= 8;
    range <<= 8;
  }
}

#define NormalizeDecoder(bot) {                                                                  \
  while ((low ^ (low + range)) < TOP || (range < (bot) && ((range = -low & ((bot) - 1)), 1))) {  \
    code = (code << 8) | InBuffer[InCharNum++];                                                  \
    low <<= 8;                                                                                   \
    range <<= 8;                                                                                 \
  }                                                                                              \
}

void DecodeSymTypeStart(uint8_t Context) {
  NormalizeDecoder(FREQ_SYM_TYPE_BOT);
  extra_range = range & (FREQ_SYM_TYPE_BOT - 1);
  count = (code - low) / (range >>= 14);
  return;
}

uint8_t DecodeSymTypeCheckDict(uint8_t Context) {
  if (FreqSymType[Context][0] > count)
    return(1);
  else
    return(0);
}

void DecodeSymTypeFinishDict(uint8_t Context) {
  range *= FreqSymType[Context][0];
  uint8_t sum = 0;
  uint8_t sub;
  sum += (sub = FreqSymType[Context][1] >> 6);
  FreqSymType[Context][1] -= sub;
  sum += (sub = FreqSymType[Context][2] >> 6);
  FreqSymType[Context][2] -= sub;
  sum += (sub = FreqSymType[Context][3] >> 6);
  FreqSymType[Context][3] -= sub;
  FreqSymType[Context][0] += sum;
  return;
}

uint8_t DecodeSymTypeCheckNew(uint8_t Context) {
  if ((RangeHigh = FreqSymType[Context][0] + FreqSymType[Context][1]) > count)
    return(1);
  else
    return(0);
}

void DecodeSymTypeFinishNew(uint8_t Context) {
  low += range * FreqSymType[Context][0];
  range *= FreqSymType[Context][1];
  uint8_t sum = 0;
  uint8_t sub;
  sum += (sub = FreqSymType[Context][0] >> 6);
  FreqSymType[Context][0] -= sub;
  sum += (sub = FreqSymType[Context][2] >> 6);
  FreqSymType[Context][2] -= sub;
  sum += (sub = FreqSymType[Context][3] >> 6);
  FreqSymType[Context][3] -= sub;
  FreqSymType[Context][1] += sum;
  return;
}

uint8_t DecodeSymTypeCheckMtfg(uint8_t Context) {
  if ((RangeHigh + FreqSymType[Context][2]) > count)
    return(1);
  else
    return(0);
}

void DecodeSymTypeFinishMtfg(uint8_t Context) {
  low += range * RangeHigh;
  range *= FreqSymType[Context][2];
  uint8_t sum = 0;
  uint8_t sub;
  sum += (sub = FreqSymType[Context][0] >> 6);
  FreqSymType[Context][0] -= sub;
  sum += (sub = FreqSymType[Context][1] >> 6);
  FreqSymType[Context][1] -= sub;
  sum += (sub = FreqSymType[Context][3] >> 6);
  FreqSymType[Context][3] -= sub;
  FreqSymType[Context][2] += sum;
  return;
}

void DecodeSymTypeFinishMtf(uint8_t Context) {
  low += range * (RangeHigh + FreqSymType[Context][2]);
  range *= FreqSymType[Context][3];
  range += extra_range;
  uint8_t sum = 0;
  uint8_t sub;
  sum += (sub = FreqSymType[Context][0] >> 6);
  FreqSymType[Context][0] -= sub;
  sum += (sub = FreqSymType[Context][1] >> 6);
  FreqSymType[Context][1] -= sub;
  sum += (sub = FreqSymType[Context][2] >> 6);
  FreqSymType[Context][2] -= sub;
  FreqSymType[Context][3] += sum;
  return;
}

void DecodeMtfQueueNumStart(uint8_t Context) {
  NormalizeDecoder(FREQ_MTF_QUEUE_NUM_BOT);
  count = (code - low) / (range /= RangeScaleMtfQueueNum[Context]);
  return;
}

uint8_t DecodeMtfQueueNumCheck0(uint8_t Context) {
  if ((RangeHigh = FreqMtfQueueNum[Context][0]) > count)
    return(1);
  else
    return(0);
}

void DecodeMtfQueueNumFinish0(uint8_t Context) {
  range *= RangeHigh;
  return;
}

uint8_t DecodeMtfQueueNumFinish(uint8_t Context) {
  uint8_t mtf_queue_number = 1;
  while ((RangeHigh += FreqMtfQueueNum[Context][mtf_queue_number]) <= count)
    mtf_queue_number++;
  low += range * (RangeHigh - FreqMtfQueueNum[Context][mtf_queue_number]);
  range *= FreqMtfQueueNum[Context][mtf_queue_number];
  return(mtf_queue_number);
}

void DecodeMtfQueuePosStart(uint8_t Context, uint8_t mtf_queue_number, uint8_t * mtf_queue_size) {
  NormalizeDecoder(FREQ_MTF_QUEUE_POS_BOT);
  uint16_t RangeScale = RangeScaleMtfQueuePos[Context][mtf_queue_number];
  if (mtf_queue_size[mtf_queue_number+2] != MTF_QUEUE_SIZE) {
    uint8_t tqp = MTF_QUEUE_SIZE - 1;
    do {
      RangeScale -= FreqMtfQueuePos[Context][mtf_queue_number][tqp];
    } while (tqp-- != mtf_queue_size[mtf_queue_number+2]);
  }
  count = (code - low) / (range /= RangeScale);
  return;
}

uint8_t DecodeMtfQueuePosCheck0(uint8_t Context, uint8_t mtf_queue_number) {
  if ((RangeHigh = FreqMtfQueuePos[Context][mtf_queue_number][0]) > count)
    return(1);
  else
    return(0);
}

void DecodeMtfQueuePosFinish0(uint8_t Context, uint8_t mtf_queue_number) {
  range *= RangeHigh;
  FreqMtfQueuePos[Context][mtf_queue_number][0] = RangeHigh + UP_FREQ_MTF_QUEUE_POS;
  if ((RangeScaleMtfQueuePos[Context][mtf_queue_number] += UP_FREQ_MTF_QUEUE_POS) > FREQ_MTF_QUEUE_POS_BOT)
    rescaleMtfQueuePos(Context, mtf_queue_number);
}

uint8_t DecodeMtfQueuePosFinish(uint8_t Context, uint8_t mtf_queue_number) {
  uint32_t Symbol = 1;
  while ((RangeHigh += FreqMtfQueuePos[Context][mtf_queue_number][Symbol]) <= count)
    Symbol++;
  low += range * (RangeHigh - FreqMtfQueuePos[Context][mtf_queue_number][Symbol]);
  range *= FreqMtfQueuePos[Context][mtf_queue_number][Symbol];
  if (Symbol >= 4) {
    if (Symbol == 4) {
      FreqMtfQueuePos[Context][mtf_queue_number][Symbol] += UP_FREQ_MTF_QUEUE_POS - 1;
      FreqMtfQueuePos[Context][mtf_queue_number][Symbol+1] += 1;
    }
    else if (Symbol == 63) {
      FreqMtfQueuePos[Context][mtf_queue_number][Symbol-1] += 1;
      FreqMtfQueuePos[Context][mtf_queue_number][Symbol] += UP_FREQ_MTF_QUEUE_POS - 1;
    }
    else {
      FreqMtfQueuePos[Context][mtf_queue_number][Symbol-1] += 1;
      FreqMtfQueuePos[Context][mtf_queue_number][Symbol] += UP_FREQ_MTF_QUEUE_POS - 2;
      FreqMtfQueuePos[Context][mtf_queue_number][Symbol+1] += 1;
    }
  }
  else
    FreqMtfQueuePos[Context][mtf_queue_number][Symbol] += UP_FREQ_MTF_QUEUE_POS;
  if ((RangeScaleMtfQueuePos[Context][mtf_queue_number] += UP_FREQ_MTF_QUEUE_POS) > FREQ_MTF_QUEUE_POS_BOT)
    rescaleMtfQueuePos(Context, mtf_queue_number);
  return(Symbol);
}

void DecodeMtfgQueuePosStart(uint8_t Context) {
  NormalizeDecoder(FREQ_MTFG_QUEUE_POS_BOT);
  count = (code - low) / (range /= RangeScaleMtfgQueuePos[Context]);
  return;
}

uint8_t DecodeMtfgQueuePosCheck0(uint8_t Context) {
  if ((RangeHigh = FreqMtfgQueuePos[Context][0]) > count)
    return(1);
  else
    return(0);
}

uint8_t DecodeMtfgQueuePosFinish0(uint8_t Context) {
  range *= RangeHigh;
  FreqMtfgQueuePos[Context][0] = RangeHigh + UP_FREQ_MTFG_QUEUE_POS;
  if ((RangeScaleMtfgQueuePos[Context] += UP_FREQ_MTFG_QUEUE_POS) > FREQ_MTFG_QUEUE_POS_BOT)
    rescaleMtfgQueuePos(Context);
  return(0);
}

uint8_t DecodeMtfgQueuePosFinish(uint8_t Context) {
  uint8_t mtfg_queue_position = 1;
  while ((RangeHigh += FreqMtfgQueuePos[Context][mtfg_queue_position]) <= count)
    mtfg_queue_position++;
  low += range * (RangeHigh - FreqMtfgQueuePos[Context][mtfg_queue_position]);
  range *= FreqMtfgQueuePos[Context][mtfg_queue_position];
  if (mtfg_queue_position >= 4) {
    if (mtfg_queue_position == 4) {
      FreqMtfgQueuePos[Context][mtfg_queue_position] += UP_FREQ_MTFG_QUEUE_POS - 2;
      FreqMtfgQueuePos[Context][mtfg_queue_position+1] += 2;
    }
    else if (mtfg_queue_position == 255) {
      FreqMtfgQueuePos[Context][mtfg_queue_position-1] += 2;
      FreqMtfgQueuePos[Context][mtfg_queue_position] += UP_FREQ_MTFG_QUEUE_POS - 2;
    }
    else {
      FreqMtfgQueuePos[Context][mtfg_queue_position-1] += 2;
      FreqMtfgQueuePos[Context][mtfg_queue_position] += UP_FREQ_MTFG_QUEUE_POS - 4;
      FreqMtfgQueuePos[Context][mtfg_queue_position+1] += 2;
    }
  }
  else
    FreqMtfgQueuePos[Context][mtfg_queue_position] += UP_FREQ_MTFG_QUEUE_POS;
  if ((RangeScaleMtfgQueuePos[Context] += UP_FREQ_MTFG_QUEUE_POS) > FREQ_MTFG_QUEUE_POS_BOT)
    rescaleMtfgQueuePos(Context);
  return(mtfg_queue_position);
}

void DecodeSIDStart(uint8_t Context) {
  NormalizeDecoder(FREQ_SID_BOT);
  count = (code - low) / (range /= RangeScaleSID[Context]);
  return;
}

uint8_t DecodeSIDCheck0(uint8_t Context) {
  if ((RangeHigh = FreqSID[Context][0]) > count)
    return(1);
  else
    return(0);
}

uint8_t DecodeSIDFinish0(uint8_t Context) {
  range *= RangeHigh;
  FreqSID[Context][0] = RangeHigh + UP_FREQ_SID;
  if ((RangeScaleSID[Context] += UP_FREQ_SID) > FREQ_SID_BOT)
    rescaleSID(Context);
  return(0);
}

uint8_t DecodeSIDFinish(uint8_t Context) {
  uint8_t SIDSymbol = 1;
  while ((RangeHigh += FreqSID[Context][SIDSymbol]) <= count)
    SIDSymbol++;
  low += range * (RangeHigh - FreqSID[Context][SIDSymbol]);
  range *= FreqSID[Context][SIDSymbol];
  FreqSID[Context][SIDSymbol] += UP_FREQ_SID;
  if ((RangeScaleSID[Context] += UP_FREQ_SID) > FREQ_SID_BOT)
    rescaleSID(Context);
  return(SIDSymbol);
}

uint8_t DecodeExtraLength() {
  NormalizeDecoder((uint32_t)1 << 2);
  uint32_t Symbol = (code - low) / (range >>= 2);
  low += range * Symbol;
  return((uint8_t)Symbol);
}

void DecodeINSTStart(uint8_t Context, uint8_t SIDSymbol) {
  NormalizeDecoder(FREQ_INST_BOT);
  count = (code - low) / (range /= RangeScaleINST[Context][SIDSymbol]);
}

uint8_t DecodeINSTCheck0(uint8_t Context, uint8_t SIDSymbol) {
  if ((RangeHigh = FreqINST[Context][SIDSymbol][0]) > count)
    return(1);
  else
    return(0);
}

void DecodeINSTFinish0(uint8_t Context, uint8_t SIDSymbol) {
  range *= RangeHigh;
  if (RangeScaleINST[Context][SIDSymbol] >= (FREQ_INST_BOT >> 1)) {
    FreqINST[Context][SIDSymbol][0] += RangeScaleINST[Context][SIDSymbol] >> 11;
    if ((RangeScaleINST[Context][SIDSymbol] += RangeScaleINST[Context][SIDSymbol] >> 11) > FREQ_INST_BOT)
      rescaleINST(Context, SIDSymbol);
  }
  else {
    FreqINST[Context][SIDSymbol][0] += UP_FREQ_INST;
    RangeScaleINST[Context][SIDSymbol] += UP_FREQ_INST;
  }
  return;
}

uint8_t DecodeINSTFinish(uint8_t Context, uint8_t SIDSymbol) {
  uint8_t Instances = 1;
  while ((RangeHigh += FreqINST[Context][SIDSymbol][Instances]) <= count)
    Instances++;
  low += range * (RangeHigh - FreqINST[Context][SIDSymbol][Instances]);
  range *= FreqINST[Context][SIDSymbol][Instances];
  if (RangeScaleINST[Context][SIDSymbol] >= (FREQ_INST_BOT >> 1)) {
    FreqINST[Context][SIDSymbol][Instances] += RangeScaleINST[Context][SIDSymbol] >> 11;
    if ((RangeScaleINST[Context][SIDSymbol] += (RangeScaleINST[Context][SIDSymbol] >> 11)) > FREQ_INST_BOT)
      rescaleINST(Context, SIDSymbol);
  }
  else {
    FreqINST[Context][SIDSymbol][Instances] += UP_FREQ_INST;
    RangeScaleINST[Context][SIDSymbol] += UP_FREQ_INST;
  }
  return(Instances);
}

uint8_t DecodeERG(uint8_t Context) {
  uint8_t nonergodic;
  NormalizeDecoder(FREQ_ERG_BOT);
  count = (code - low) / (range /= RangeScaleERG[Context]);
  if (FreqERG[Context] > count) {
    range *= FreqERG[Context];
    FreqERG[Context] += UP_FREQ_ERG;
    nonergodic = 0;
  }
  else {
    low += range * FreqERG[Context];
    range *= RangeScaleERG[Context] - FreqERG[Context];
    nonergodic = 1;
  }
  if ((RangeScaleERG[Context] += UP_FREQ_ERG) > FREQ_ERG_BOT) {
    RangeScaleERG[Context] = (FREQ_ERG_BOT >> 1) + 1;
    FreqERG[Context] = (FreqERG[Context] + 1) >> 1;
  }
  return(nonergodic);
}

uint8_t DecodeWordTag() {
  uint8_t Tag;
  NormalizeDecoder(FREQ_WORD_TAG_BOT);
  count = (code - low) / (range /= RangeScaleWordTag);
  if (FreqWordTag > count) {
    range *= FreqWordTag;
    FreqWordTag += UP_FREQ_WORD_TAG;
    Tag = 0;
  }
  else {
    low += range * FreqWordTag;
    range *= RangeScaleWordTag - FreqWordTag;
    Tag = 1;
  }
  if ((RangeScaleWordTag += UP_FREQ_WORD_TAG) > FREQ_WORD_TAG_BOT) {
    RangeScaleWordTag = (FREQ_WORD_TAG_BOT >> 1) + 1;
    FreqWordTag = (FreqWordTag + 1) >> 1;
  }
  return(Tag);
}

uint16_t DecodeDictionaryBin(uint8_t FirstChar, uint8_t * lookup_bits, uint8_t * CodeLengthPtr, uint16_t DictionaryBins,
    uint8_t bin_extra_bits) {
  uint16_t BinNum;
  NormalizeDecoder((uint32_t)1 << 12);
  *CodeLengthPtr = lookup_bits[BinNum = (code - low) / (range /= DictionaryBins)];
  int8_t BitsUnderBinSize = bin_extra_bits - *CodeLengthPtr;
  if (BitsUnderBinSize > 0)
    low += (range <<= BitsUnderBinSize) * (BinNum >> BitsUnderBinSize);
  else
    low += range * BinNum;
  return(BinNum);
}

uint32_t DecodeBinCode(uint8_t Bits) {
  NormalizeDecoder((uint32_t)1 << Bits);
  uint32_t BinCode = (code - low) / (range >>= Bits);
  return(BinCode);
}

uint32_t DecodeBaseSymbol(uint8_t Bits, uint32_t NumBaseSymbols) {
  NormalizeDecoder((uint32_t)1 << Bits);
  range /= NumBaseSymbols;
  uint32_t BaseSymbol;
  low += range * (BaseSymbol = (code - low) / range);
  return(BaseSymbol);
}

uint8_t DecodeFirstChar(uint8_t SymType, uint8_t LastChar) {
  NormalizeDecoder(FREQ_FIRST_CHAR_BOT);
  count = (code - low) / (range /= RangeScaleFirstChar[SymType][LastChar]);
  uint32_t FirstChar;
  if ((RangeHigh = FreqFirstChar[SymType][LastChar][0]) > count) {
    range *= RangeHigh;
    if (RangeScaleFirstChar[SymType][LastChar] >= (FREQ_FIRST_CHAR_BOT >> 1)) {
      FreqFirstChar[SymType][LastChar][0] += RangeScaleFirstChar[SymType][LastChar] >> 9;
      if ((RangeScaleFirstChar[SymType][LastChar] += (RangeScaleFirstChar[SymType][LastChar] >> 9)) > FREQ_FIRST_CHAR_BOT)
        rescaleFirstChar(SymType, LastChar);
    }
    else {
      FreqFirstChar[SymType][LastChar][0] += UP_FREQ_FIRST_CHAR;
      RangeScaleFirstChar[SymType][LastChar] += UP_FREQ_FIRST_CHAR;
    }
    FirstChar = SymbolFirstChar[SymType][LastChar][0];
  }
  else {
    uint16_t * FreqPtr = &FreqFirstChar[SymType][LastChar][1];
    while ((RangeHigh += *FreqPtr) <= count)
      FreqPtr++;
    low += range * (RangeHigh - *FreqPtr);
    range *= *FreqPtr;
    if (RangeScaleFirstChar[SymType][LastChar] >= (FREQ_FIRST_CHAR_BOT >> 1)) {
      *FreqPtr += RangeScaleFirstChar[SymType][LastChar] >> 9;
      if ((RangeScaleFirstChar[SymType][LastChar] += (RangeScaleFirstChar[SymType][LastChar] >> 9)) > FREQ_FIRST_CHAR_BOT)
        rescaleFirstChar(SymType, LastChar);
    }
    else {
      *FreqPtr += UP_FREQ_FIRST_CHAR;
      RangeScaleFirstChar[SymType][LastChar] += UP_FREQ_FIRST_CHAR;
    }
    FoundIndex = FreqPtr - &FreqFirstChar[SymType][LastChar][0];
    FirstChar = SymbolFirstChar[SymType][LastChar][FoundIndex];
    if (*FreqPtr > *(FreqPtr - 1)) {
      uint16_t SavedFreq = *FreqPtr;
      uint8_t * SymbolPtr = &SymbolFirstChar[SymType][LastChar][FoundIndex];
      do {
        *FreqPtr = *(FreqPtr - 1);
        FreqPtr--;
        *SymbolPtr = *(SymbolPtr - 1);
        SymbolPtr--;
      } while ((FreqPtr != &FreqFirstChar[SymType][LastChar][0]) && (SavedFreq > *(FreqPtr - 1)));
      *FreqPtr = SavedFreq;
      *SymbolPtr = FirstChar;
    }
  }
  return(FirstChar);
}

uint8_t DecodeFirstCharBinary(uint8_t LastChar) {
  NormalizeDecoder(FREQ_FIRST_CHAR_BOT);
  count = (code - low) / (range /= RangeScaleFirstChar[0][LastChar]);
  uint16_t * FreqPtr;
  if (RangeScaleFirstCharSection[LastChar][3] > count) {
    RangeScaleFirstCharSection[LastChar][3] += UP_FREQ_FIRST_CHAR;
    if (RangeScaleFirstCharSection[LastChar][1] > count) {
      RangeScaleFirstCharSection[LastChar][1] += UP_FREQ_FIRST_CHAR;
      if (RangeScaleFirstCharSection[LastChar][0] > count) {
        RangeHigh = 0;
        RangeScaleFirstCharSection[LastChar][0] += UP_FREQ_FIRST_CHAR;
        FreqPtr = &FreqFirstCharBinary[LastChar][0];
      }
      else {
        RangeHigh = RangeScaleFirstCharSection[LastChar][0];
        FreqPtr = &FreqFirstCharBinary[LastChar][0x20];
      }
    }
    else {
      RangeHigh = RangeScaleFirstCharSection[LastChar][1];
      if (RangeHigh + RangeScaleFirstCharSection[LastChar][2] > count) {
        RangeScaleFirstCharSection[LastChar][2] += UP_FREQ_FIRST_CHAR;
        FreqPtr = &FreqFirstCharBinary[LastChar][0x40];
      }
      else {
        RangeHigh += RangeScaleFirstCharSection[LastChar][2];
        FreqPtr = &FreqFirstCharBinary[LastChar][0x60];
      }
    }
  }
  else {
    RangeHigh = RangeScaleFirstCharSection[LastChar][3];
    if (RangeHigh + RangeScaleFirstCharSection[LastChar][5] > count) {
      RangeScaleFirstCharSection[LastChar][5] += UP_FREQ_FIRST_CHAR;
      if (RangeHigh + RangeScaleFirstCharSection[LastChar][4] > count) {
        RangeScaleFirstCharSection[LastChar][4] += UP_FREQ_FIRST_CHAR;
        FreqPtr = &FreqFirstCharBinary[LastChar][0x80];
      }
      else {
        RangeHigh += RangeScaleFirstCharSection[LastChar][4];
        FreqPtr = &FreqFirstCharBinary[LastChar][0xA0];
      }
    }
    else {
      RangeHigh += RangeScaleFirstCharSection[LastChar][5];
      if (RangeHigh + RangeScaleFirstCharSection[LastChar][6] > count) {
        RangeScaleFirstCharSection[LastChar][6] += UP_FREQ_FIRST_CHAR;
        FreqPtr = &FreqFirstCharBinary[LastChar][0xC0];
      }
      else {
        RangeHigh += RangeScaleFirstCharSection[LastChar][6];
        FreqPtr = &FreqFirstCharBinary[LastChar][0xE0];
      }
    }
  }
  while ((RangeHigh += *FreqPtr) <= count)
    FreqPtr++;
  uint32_t FirstChar = FreqPtr - &FreqFirstCharBinary[LastChar][0];
  low += range * (RangeHigh - *FreqPtr);
  range *= *FreqPtr;
  *FreqPtr += UP_FREQ_FIRST_CHAR;
  if ((RangeScaleFirstChar[0][LastChar] += UP_FREQ_FIRST_CHAR) > FREQ_FIRST_CHAR_BOT)
    rescaleFirstCharBinary(LastChar);
  return(FirstChar);
}

void InitDecoder(uint8_t max_regular_code_length, uint8_t num_inst_codes, uint8_t cap_encoded, uint8_t UTF8_compliant,
    uint8_t use_mtf, uint8_t use_mtfg, uint8_t * inbuf) {
  InBuffer = inbuf;
  code = 0, range = -1;
  for (low = 4; low != 0; low--)
    code = (code << 8) | InBuffer[InCharNum++];
  StartModelSymType(use_mtf, use_mtfg);
  StartModelMtfQueueNum();
  StartModelMtfQueuePos();
  StartModelMtfgQueuePos(max_regular_code_length);
  StartModelSID();
  StartModelINST(num_inst_codes);
  StartModelERG();
  StartModelWordTag();
  if (cap_encoded || UTF8_compliant)
    StartModelFirstChar();
  else
    StartModelFirstCharBinary();
}
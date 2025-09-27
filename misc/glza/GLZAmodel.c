/***********************************************************************

Copyright 2015 - 2025 Kennon Conrad

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

uint32_t InCharNum, OutCharNum, RangeLow, RangeHigh, count, code, low, range;
uint16_t last_queue_size, last_queue_size_az, last_queue_size_space, last_queue_size_other;
uint16_t rescale_queue_size, rescale_queue_size_az, rescale_queue_size_space, rescale_queue_size_other;
uint16_t unused_queue_freq, unused_queue_freq_az, unused_queue_freq_space, unused_queue_freq_other;
uint16_t RangeScaleSID[2], FreqSID[2][16], RangeScaleINST[2][16], FreqINST[2][16][38];
uint16_t RangeScaleFirstCharSection[0x100][7], FreqFirstCharBinary[0x100][0x100];
uint16_t RangeScaleFirstChar[4][0x100], FreqFirstChar[4][0x100][0x100], FreqWordTag[0x100], FreqERG[341], FreqGoMtf[0x5A0];
uint16_t RangeScaleMtfPos[4], FreqMtfPos[4][0x100], FreqSymTypePriorType[0x34][2], FreqSymTypePriorEnd[0x100][2];
uint16_t FreqMtfFirst[2][3];
uint8_t SymbolFirstChar[4][0x100][0x100];
uint8_t MaxBaseCode, MaxInstCode, *InBuffer, *OutBuffer;

uint32_t ReadLow() {return(low);}
uint32_t ReadRange() {return(range);}


void StartModelSymType(uint8_t use_mtf, uint8_t cap_encoded) {
  if (cap_encoded == 0) {
    uint8_t i = 1;
    do {
      if (use_mtf != 0) {
        FreqSymTypePriorType[i][0] = 0x1C00;
        FreqSymTypePriorType[i][1] = 0x2000;
      }
      else {
        FreqSymTypePriorType[i][0] = 0x2000;
        FreqSymTypePriorType[i][1] = 0x2000;
      }
    } while (i-- != 0);
  }
  else {
    uint8_t i = 0x33;
    do {
      if (use_mtf != 0) {
        FreqSymTypePriorType[i][0] = 0xE00;
        FreqSymTypePriorType[i][1] = 0x1000;
      }
      else {
        FreqSymTypePriorType[i][0] = 0x1000;
        FreqSymTypePriorType[i][1] = 0x1000;
      }
    } while (i-- != 0x2C);
    do {
      if (use_mtf != 0) {
        FreqSymTypePriorType[i][0] = 0x1500;
        FreqSymTypePriorType[i][1] = 0x1800;
      }
      else {
        FreqSymTypePriorType[i][0] = 0x1800;
        FreqSymTypePriorType[i][1] = 0x1800;
      }
    } while (i-- != 4);
    do {
      if (use_mtf != 0) {
        FreqSymTypePriorType[i][0] = 0xE00;
        FreqSymTypePriorType[i][1] = 0x1000;
      }
      else {
        FreqSymTypePriorType[i][0] = 0x1000;
        FreqSymTypePriorType[i][1] = 0x1000;
      }
    } while (i-- != 2);
    do {
      if (use_mtf != 0) {
        FreqSymTypePriorType[i][0] = 0x1500;
        FreqSymTypePriorType[i][1] = 0x1800;
      }
      else {
        FreqSymTypePriorType[i][0] = 0x1800;
        FreqSymTypePriorType[i][1] = 0x1800;
      }
    } while (i-- != 0);
    i = 0xFF;
    do {
      if (use_mtf != 0) {
        FreqSymTypePriorEnd[i][0] = 0xE00;
        FreqSymTypePriorEnd[i][1] = 0x1000;
      }
      else {
        FreqSymTypePriorEnd[i][0] = 0x1000;
        FreqSymTypePriorEnd[i][1] = 0x1000;
      }
    } while (i-- != 0);
  }
  return;
}

void StartModelMtfFirst() {
    FreqMtfFirst[0][0] = 0x900;
    FreqMtfFirst[0][1] = 0x500;
    FreqMtfFirst[0][2] = 0x200;
    FreqMtfFirst[1][0] = 0x100;
    FreqMtfFirst[1][1] = 0xE00;
    FreqMtfFirst[1][2] = 0x100;
}

void StartModelMtfQueuePos(uint8_t max_code_length) {
  RangeScaleMtfPos[0] = 0;
  uint16_t j = 0;
  do {
    FreqMtfPos[0][j] = FreqMtfPos[1][j] = FreqMtfPos[2][j] = FreqMtfPos[3][j] = 0x200 / (j + 2);
    RangeScaleMtfPos[0] += FreqMtfPos[0][j];
  } while (++j != 0x100);
  RangeScaleMtfPos[1] = RangeScaleMtfPos[2] = RangeScaleMtfPos[3] = RangeScaleMtfPos[0];
  unused_queue_freq = unused_queue_freq_az = unused_queue_freq_space = unused_queue_freq_other = RangeScaleMtfPos[0];
  last_queue_size = last_queue_size_az = last_queue_size_space = last_queue_size_other = 0;
  rescale_queue_size = rescale_queue_size_az = rescale_queue_size_space = rescale_queue_size_other = 0;

  return;
}

void StartModelSID() {
  uint8_t i = 1;
  do {
    uint8_t j = 15;
    do {
      FreqSID[i][j] = 1;
    } while (j-- != 8);
    do {
      FreqSID[i][j] = 2;
    } while (j-- != 4);
    FreqSID[i][3] = 4;
    FreqSID[i][2] = 6;
    FreqSID[i][1] = 8;
    FreqSID[i][0] = 4;
    RangeScaleSID[i] = 0;
    j = 15;
    do {
      RangeScaleSID[i] += FreqSID[i][j];
    } while (j-- != 0);
  } while (i-- != 0);
  return;
}

void StartModelINST(uint8_t num_inst_codes) {
  uint8_t i = 1;
  do {
    uint8_t j = 15;
    do {
      uint8_t k = num_inst_codes;
      if (j != 0)
        k--;
      RangeScaleINST[i][j] = k--;
      do {
        FreqINST[i][j][k] = 1;
      } while (k-- != 0);
    } while (j-- != 0);
  } while (i-- != 0);
  return;
}

void StartModelERG() {
  uint16_t i = 340;
  do {
    FreqERG[i] = 0x600;
  } while (i-- != 240);
  do {
    FreqERG[i] = 0x800;
  } while (i-- != 1);
  FreqERG[i] = 0x200;
  return;
}

void StartModelGoQueue() {
  uint16_t i = 0x59F;
  do {
    FreqGoMtf[i] = 0x555;
  } while (i-- != 0);
  return;
}

void StartModelWordTag() {
  uint8_t i = 0xFF;
  do {
    FreqWordTag[i] = 0x800;
  } while (i-- != 0);
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
      SymbolFirstChar[0][i][j] = j;
      SymbolFirstChar[1][i][j] = j;
      SymbolFirstChar[2][i][j] = j;
      SymbolFirstChar[3][i][j] = j;
    } while (j-- != 0);
    RangeScaleFirstChar[0][i] = 0;
    RangeScaleFirstChar[1][i] = 0;
    RangeScaleFirstChar[2][i] = 0;
    RangeScaleFirstChar[3][i] = 0;
  } while (i-- != 0);
  return;
}

void StartModelFirstCharBinary() {
  uint8_t i = 0xFF;
  do {
    uint8_t j = 0xFF;
    do {
      FreqFirstCharBinary[i][j] = 0;
    } while (j-- != 0);
    j = 6;
    do {
      RangeScaleFirstCharSection[i][j] = 0;
    } while (j-- != 0);
    RangeScaleFirstChar[0][i] = 0;
  } while (i-- != 0);
  return;
}

void rescaleMtfQueuePos(uint8_t Context) {
  uint8_t i = 1;
  RangeScaleMtfPos[Context] = FreqMtfPos[Context][0] = (FreqMtfPos[Context][0] + 1) >> 1;
  do {
    RangeScaleMtfPos[Context] += FreqMtfPos[Context][i] = (FreqMtfPos[Context][i] + 1) >> 1;
  } while (++i != 0);
  uint8_t qp = 0xFF;
  if (Context == 0) {
    rescale_queue_size = last_queue_size;
    unused_queue_freq = 0;
    while (qp >= last_queue_size)
      unused_queue_freq += FreqMtfPos[0][qp--];
  }
  else if (Context == 1) {
    rescale_queue_size_az = last_queue_size_az;
    unused_queue_freq_az = 0;
    while (qp >= last_queue_size_az)
      unused_queue_freq_az += FreqMtfPos[1][qp--];
  }
  else if (Context == 2) {
    rescale_queue_size_space = last_queue_size_space;
    unused_queue_freq_space = 0;
    while (qp >= last_queue_size_space)
      unused_queue_freq_space += FreqMtfPos[2][qp--];
  }
  else {
    rescale_queue_size_other = last_queue_size_other;
    unused_queue_freq_other = 0;
    while (qp >= last_queue_size_other)
      unused_queue_freq_other += FreqMtfPos[3][qp--];
  }
  return;
}

void rescaleSID(uint8_t Context) {
  uint8_t i = 14;
  RangeScaleSID[Context] = FreqSID[Context][15] = (FreqSID[Context][15] + 1) >> 1;
  do {
    RangeScaleSID[Context] += FreqSID[Context][i] = (FreqSID[Context][i] + 1) >> 1;
  } while (i-- != 0);
  return;
}

void rescaleINST(uint8_t Context, uint8_t SIDSymbol) {
  RangeScaleINST[Context][SIDSymbol] = 0;
  uint8_t i = MaxInstCode;
  do {
    RangeScaleINST[Context][SIDSymbol] += FreqINST[Context][SIDSymbol][i] = (FreqINST[Context][SIDSymbol][i] + 1) >> 1;
  } while (i-- != 0);
  return;
}

void rescaleFirstChar(uint8_t SymType, uint8_t Context) {
  uint8_t i = MaxBaseCode;
  RangeScaleFirstChar[SymType][Context] = 0;
  do {
    RangeScaleFirstChar[SymType][Context] += FreqFirstChar[SymType][Context][i]
        = (FreqFirstChar[SymType][Context][i] + 1) >> 1;
  } while (i-- != 0);
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
  } while (++i != 0);
  return;
}

void InitFirstCharBin(uint8_t trailing_char, uint8_t leading_char, uint8_t code_length, uint8_t cap_symbol_defined,
    uint8_t cap_lock_symbol_defined) {
  if ((RangeScaleFirstChar[0][trailing_char] != 0)
      || ((trailing_char == 'C') && (cap_symbol_defined || cap_lock_symbol_defined))) {
    uint16_t freq = 1;
    if (code_length < 8)
      freq = 1 << (8 - code_length);
    uint8_t k;
    for (k = 0 ; k < 4 ; k++) {
      uint8_t j2 = leading_char;
      while (SymbolFirstChar[k][trailing_char][j2] != (uint8_t)leading_char)
        j2++;
      FreqFirstChar[k][trailing_char][j2] = freq;
      RangeScaleFirstChar[k][trailing_char] += freq;
      if (RangeScaleFirstChar[k][trailing_char] > FREQ_FIRST_CHAR_BOT)
        rescaleFirstChar(k, trailing_char);
    }
  }
  return;
}

void InitFirstCharBinBinary(uint8_t trailing_char, uint8_t leading_char, uint8_t code_length) {
  if (RangeScaleFirstChar[0][trailing_char] != 0) {
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
  uint16_t freq = 1;
  if (code_length < 8)
    freq = 1 << (8 - code_length);
  FreqFirstChar[0][trailing_char][leading_char] = freq;
  RangeScaleFirstChar[0][trailing_char] += freq;
  FreqFirstChar[1][trailing_char][leading_char] = freq;
  RangeScaleFirstChar[1][trailing_char] += freq;
  FreqFirstChar[2][trailing_char][leading_char] = freq;
  RangeScaleFirstChar[2][trailing_char] += freq;
  FreqFirstChar[3][trailing_char][leading_char] = freq;
  RangeScaleFirstChar[3][trailing_char] += freq;
  return;
}

void InitTrailingCharBinary(uint8_t trailing_char, uint8_t * symbol_lengths) {
  uint8_t leading_char = 0xFF;
  do {
    uint16_t freq = 1;
    if (symbol_lengths[leading_char] < 8)
      freq = 1 << (8 - symbol_lengths[leading_char]);
    if (RangeScaleFirstChar[0][leading_char] || (leading_char == trailing_char)) {
      FreqFirstCharBinary[trailing_char][leading_char] = freq;
      RangeScaleFirstChar[0][trailing_char] += freq;
      if (leading_char < 0x80) {
        RangeScaleFirstCharSection[trailing_char][3] += freq;
        if (leading_char < 0x40) {
          RangeScaleFirstCharSection[trailing_char][1] += freq;
          if (leading_char < 0x20)
            RangeScaleFirstCharSection[trailing_char][0] += freq;
        }
        else if (leading_char < 0x60)
          RangeScaleFirstCharSection[trailing_char][2] += freq;
      }
      else if (leading_char < 0xC0) {
        RangeScaleFirstCharSection[trailing_char][5] += freq;
        if (leading_char < 0xA0)
          RangeScaleFirstCharSection[trailing_char][4] += freq;
      }
      else if (leading_char < 0xE0)
        RangeScaleFirstCharSection[trailing_char][6] += freq;
    }
  } while (leading_char-- != 0);
  return;
}

void InitBaseSymbolCap(uint8_t BaseSymbol, uint8_t new_symbol_code_length, uint8_t * cap_symbol_defined_ptr,
    uint8_t * cap_lock_symbol_defined_ptr, uint8_t * symbol_lengths) {
  uint8_t j1 = MaxBaseCode;
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
        if (RangeScaleFirstChar[0][j1] != 0)
          InitTrailingCharBin('C', j1, symbol_lengths[j1]);
      } while (j1-- != 'a');
    }
    if (BaseSymbol == 'C')
      *cap_symbol_defined_ptr = 1;
    else
      *cap_lock_symbol_defined_ptr = 1;
  }
  else {
    if ((BaseSymbol >= 'a') && (BaseSymbol <= 'z'))
      InitFirstCharBin('C', BaseSymbol, new_symbol_code_length, *cap_symbol_defined_ptr, *cap_lock_symbol_defined_ptr);
    j1 = MaxBaseCode;
    do {
      if (symbol_lengths[j1] != 0)
        InitTrailingCharBin(BaseSymbol, j1, symbol_lengths[j1]);
    } while (j1--);
  }
  return;
}

void IncreaseRange(uint32_t low_ranges, uint32_t ranges) {
  low -= range * low_ranges;
  range *= ranges;
}

void DoubleRange() {
  range *= 2;
}

void DoubleRangeDown() {
  low -= range;
  range *= 2;
}

void SetOutBuffer(uint8_t * bufptr) {
  OutBuffer = bufptr;
}

void WriteOutBuffer(uint8_t value) {
  OutBuffer[OutCharNum++] = value;
}

void NormalizeEncoder(uint32_t bot) {
  while ((low ^ (low + range)) < TOP || (range < bot && ((range = -low & (bot - 1)), 1))) {
    OutBuffer[OutCharNum++] = (uint8_t)(low >> 24);
    range <<= 8;
    low <<= 8;
  }
  return;
}

void EncodeDictType1(uint8_t Context1) {
  NormalizeEncoder(FREQ_SYM_TYPE_BOT1);
  uint32_t extra_range = range & (FREQ_SYM_TYPE_BOT1 - 1);
  range = FreqSymTypePriorType[Context1][0] * (range >> 14) + extra_range;
  uint16_t sum = FreqSymTypePriorType[Context1][1] >> 6;
  FreqSymTypePriorType[Context1][0]
      += sum + ((FREQ_SYM_TYPE_BOT1 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 6);
  FreqSymTypePriorType[Context1][1] -= sum;
  return;
}

void EncodeDictType2(uint8_t Context1, uint8_t Context2) {
  NormalizeEncoder(2 * FREQ_SYM_TYPE_BOT2);
  uint32_t extra_range = range & (2 * FREQ_SYM_TYPE_BOT2 - 1);
  range = (FreqSymTypePriorType[Context1][0] + FreqSymTypePriorType[Context2][0]) * (range >> 14) + extra_range;
  uint16_t sum = FreqSymTypePriorType[Context1][1] >> 4;
  FreqSymTypePriorType[Context1][0]
      += sum + ((FREQ_SYM_TYPE_BOT2 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 4);
  FreqSymTypePriorType[Context1][1] -= sum;
  sum = FreqSymTypePriorType[Context2][1] >> 7;
  FreqSymTypePriorType[Context2][0]
      += sum + ((FREQ_SYM_TYPE_BOT2 - FreqSymTypePriorType[Context2][0] - FreqSymTypePriorType[Context2][1]) >> 7);
  FreqSymTypePriorType[Context2][1] -= sum;
  return;
}

void EncodeDictType3(uint8_t Context1, uint8_t Context2, uint8_t Context3) {
  NormalizeEncoder(8 * FREQ_SYM_TYPE_BOT3);
  uint32_t extra_range = range & (8 * FREQ_SYM_TYPE_BOT3 - 1);
  range = (FreqSymTypePriorType[Context1][0] + FreqSymTypePriorType[Context2][0] + FreqSymTypePriorEnd[Context3][0])
      * (range >> 15) + extra_range;
  uint16_t sum = FreqSymTypePriorType[Context1][1] >> 4;
  FreqSymTypePriorType[Context1][0]
      += sum + ((3 * FREQ_SYM_TYPE_BOT3 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 4);
  FreqSymTypePriorType[Context1][1] -= sum;
  sum = FreqSymTypePriorType[Context2][1] >> 7;
  FreqSymTypePriorType[Context2][0]
      += sum + ((3 * FREQ_SYM_TYPE_BOT3 - FreqSymTypePriorType[Context2][0] - FreqSymTypePriorType[Context2][1]) >> 7);
  FreqSymTypePriorType[Context2][1] -= sum;
  sum = FreqSymTypePriorEnd[Context3][1] >> 3;
  FreqSymTypePriorEnd[Context3][0]
      += sum + ((2 * FREQ_SYM_TYPE_BOT3 - FreqSymTypePriorEnd[Context3][0] - FreqSymTypePriorEnd[Context3][1]) >> 3);
  FreqSymTypePriorEnd[Context3][1] -= sum;
  return;
}

void EncodeNewType1(uint8_t Context1) {
  NormalizeEncoder(FREQ_SYM_TYPE_BOT1);
  uint32_t extra_range = range & (FREQ_SYM_TYPE_BOT1 - 1);
  low += FreqSymTypePriorType[Context1][0] * (range >>= 14) + extra_range;
  range *= FreqSymTypePriorType[Context1][1];
  uint16_t sum = FreqSymTypePriorType[Context1][0] >> 6;
  FreqSymTypePriorType[Context1][1]
      += sum + ((FREQ_SYM_TYPE_BOT1 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 6);
  FreqSymTypePriorType[Context1][0] -= sum;
  return;
}

void EncodeNewType2(uint8_t Context1, uint8_t Context2) {
  NormalizeEncoder(2 * FREQ_SYM_TYPE_BOT2);
  uint32_t extra_range = range & (2 * FREQ_SYM_TYPE_BOT2 - 1);
  low += (FreqSymTypePriorType[Context1][0] + FreqSymTypePriorType[Context2][0]) * (range >>= 14) + extra_range;
  range *= (FreqSymTypePriorType[Context1][1] + FreqSymTypePriorType[Context2][1]);
  uint16_t sum = FreqSymTypePriorType[Context1][0] >> 4;
  FreqSymTypePriorType[Context1][1]
      += sum + ((FREQ_SYM_TYPE_BOT2 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 4);
  FreqSymTypePriorType[Context1][0] -= sum;
  sum = FreqSymTypePriorType[Context2][0] >> 7;
  FreqSymTypePriorType[Context2][1]
      += sum + ((FREQ_SYM_TYPE_BOT2 - FreqSymTypePriorType[Context2][0] - FreqSymTypePriorType[Context2][1]) >> 7);
  FreqSymTypePriorType[Context2][0] -= sum;
  return;
}

void EncodeNewType3(uint8_t Context1, uint8_t Context2, uint8_t Context3) {
  NormalizeEncoder(8 * FREQ_SYM_TYPE_BOT3);
  uint32_t extra_range = range & (8 * FREQ_SYM_TYPE_BOT3 - 1);
  low += (FreqSymTypePriorType[Context1][0] + FreqSymTypePriorType[Context2][0] + FreqSymTypePriorEnd[Context3][0])
      * (range >>= 15) + extra_range;
  range *= FreqSymTypePriorType[Context1][1] + FreqSymTypePriorType[Context2][1] + FreqSymTypePriorEnd[Context3][1];
  uint16_t sum = FreqSymTypePriorType[Context1][0] >> 4;
  FreqSymTypePriorType[Context1][1]
      += sum + ((3 * FREQ_SYM_TYPE_BOT3 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 4);
  FreqSymTypePriorType[Context1][0] -= sum;
  sum = FreqSymTypePriorType[Context2][0] >> 7;
  FreqSymTypePriorType[Context2][1]
      += sum + ((3 * FREQ_SYM_TYPE_BOT3 - FreqSymTypePriorType[Context2][0] - FreqSymTypePriorType[Context2][1]) >> 7);
  FreqSymTypePriorType[Context2][0] -= sum;
  sum = FreqSymTypePriorEnd[Context3][0] >> 3;
  FreqSymTypePriorEnd[Context3][1]
      += sum + ((2 * FREQ_SYM_TYPE_BOT3 - FreqSymTypePriorEnd[Context3][0] - FreqSymTypePriorEnd[Context3][1]) >> 3);
  FreqSymTypePriorEnd[Context3][0] -= sum;
  return;
}

void EncodeMtfType1(uint8_t Context1) {
  NormalizeEncoder(FREQ_SYM_TYPE_BOT1);
  uint32_t extra_range = range & (FREQ_SYM_TYPE_BOT1 - 1);
  uint16_t sum = FreqSymTypePriorType[Context1][0] + FreqSymTypePriorType[Context1][1];
  low += sum * (range >>= 14) + extra_range;
  range *= FREQ_SYM_TYPE_BOT1 - sum;
  FreqSymTypePriorType[Context1][0] -= FreqSymTypePriorType[Context1][0] >> 6;
  FreqSymTypePriorType[Context1][1] -= FreqSymTypePriorType[Context1][1] >> 6;
  return;
}

void EncodeMtfType2(uint8_t Context1, uint8_t Context2) {
  NormalizeEncoder(2 * FREQ_SYM_TYPE_BOT2);
  uint32_t extra_range = range & (2 * FREQ_SYM_TYPE_BOT2 - 1);
  uint16_t sum = FreqSymTypePriorType[Context1][0] + FreqSymTypePriorType[Context2][0]
      + FreqSymTypePriorType[Context1][1] + FreqSymTypePriorType[Context2][1];
  low += sum * (range >>= 14) + extra_range;
  range *= 2 * FREQ_SYM_TYPE_BOT2 - sum;
  FreqSymTypePriorType[Context1][0] -= FreqSymTypePriorType[Context1][0] >> 4;
  FreqSymTypePriorType[Context1][1] -= FreqSymTypePriorType[Context1][1] >> 4;
  FreqSymTypePriorType[Context2][0] -= FreqSymTypePriorType[Context2][0] >> 7;
  FreqSymTypePriorType[Context2][1] -= FreqSymTypePriorType[Context2][1] >> 7;
  return;
}

void EncodeMtfType3(uint8_t Context1, uint8_t Context2, uint8_t Context3) {
  NormalizeEncoder(8 * FREQ_SYM_TYPE_BOT3);
  uint32_t extra_range = range & (8 * FREQ_SYM_TYPE_BOT3 - 1);
  uint32_t sum = FreqSymTypePriorType[Context1][0] + FreqSymTypePriorType[Context1][1]
      + FreqSymTypePriorType[Context2][0] + FreqSymTypePriorType[Context2][1]
      + FreqSymTypePriorEnd[Context3][0] + FreqSymTypePriorEnd[Context3][1];
  low += sum * (range >>= 15) + extra_range;
  range *= 8 * FREQ_SYM_TYPE_BOT3 - sum;
  FreqSymTypePriorType[Context1][0] -= FreqSymTypePriorType[Context1][0] >> 4;
  FreqSymTypePriorType[Context1][1] -= FreqSymTypePriorType[Context1][1] >> 4;
  FreqSymTypePriorType[Context2][0] -= FreqSymTypePriorType[Context2][0] >> 7;
  FreqSymTypePriorType[Context2][1] -= FreqSymTypePriorType[Context2][1] >> 7;
  FreqSymTypePriorEnd[Context3][0] -= FreqSymTypePriorEnd[Context3][0] >> 3;
  FreqSymTypePriorEnd[Context3][1] -= FreqSymTypePriorEnd[Context3][1] >> 3;
  return;
}

void EncodeMtfFirst(uint8_t Context, uint8_t First) {
  uint32_t delta;
  NormalizeEncoder(0x1000);
  uint32_t extra_range = range & 0xFFF;
  if (First == 0) {
    range = FreqMtfFirst[Context][0] * (range >> 12) + extra_range;
    delta = FreqMtfFirst[Context][1] >> 7;
    FreqMtfFirst[Context][1] -= delta;
    FreqMtfFirst[Context][0] += delta;
    delta = FreqMtfFirst[Context][2] >> 7;
    FreqMtfFirst[Context][2] -= delta;
    FreqMtfFirst[Context][0] += delta;
  }
  else if (First == 1) {
    low += FreqMtfFirst[Context][0] * (range >>= 12) + extra_range;
    range *= FreqMtfFirst[Context][1];
    delta = FreqMtfFirst[Context][0] >> 7;
    FreqMtfFirst[Context][0] -= delta;
    FreqMtfFirst[Context][1] += delta;
    delta = FreqMtfFirst[Context][2] >> 7;
    FreqMtfFirst[Context][2] -= delta;
    FreqMtfFirst[Context][1] += delta;
  }
  else {
    low += (FreqMtfFirst[Context][0] + FreqMtfFirst[Context][1]) * (range >>= 12) + extra_range;
    range *= FreqMtfFirst[Context][2];
    delta = FreqMtfFirst[Context][0] >> 7;
    FreqMtfFirst[Context][0] -= delta;
    FreqMtfFirst[Context][2] += delta;
    delta = FreqMtfFirst[Context][1] >> 7;
    FreqMtfFirst[Context][1] -= delta;
    FreqMtfFirst[Context][2] += delta;
  }
}

void EncodeMtfPos(uint8_t position, uint16_t QueueSize) {
  NormalizeEncoder(FREQ_MTF_POS_BOT);
  if (last_queue_size > QueueSize) {
    do {
      unused_queue_freq += FreqMtfPos[0][--last_queue_size];
    } while (last_queue_size != QueueSize);
  }
  else if (last_queue_size < QueueSize) {
    do {
      unused_queue_freq -= FreqMtfPos[0][last_queue_size++];
      if (last_queue_size > rescale_queue_size) {
        rescale_queue_size++;
        FreqMtfPos[0][last_queue_size - 1] += 8;
        RangeScaleMtfPos[0] += 8;
      }
      else {
        FreqMtfPos[0][last_queue_size - 1] += 2;
        RangeScaleMtfPos[0] += 2;
      }
    } while (last_queue_size != QueueSize);
  }
  if (RangeScaleMtfPos[0] > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(0);
  if (position == 0) {
    range = FreqMtfPos[0][0] * (range / (RangeScaleMtfPos[0] - unused_queue_freq));
    FreqMtfPos[0][0] += UP_FREQ_MTF_POS;
  }
  else {
    uint16_t * FreqPtr = &FreqMtfPos[0][0];
    uint16_t * StopFreqPtr = &FreqMtfPos[0][position];
    RangeLow = *FreqPtr++;
    while (FreqPtr != StopFreqPtr)
      RangeLow += *FreqPtr++;
    low += RangeLow * (range /= (RangeScaleMtfPos[0] - unused_queue_freq));
    range *= *FreqPtr;

    if (position >= 4) {
      if (position == 4) {
        *FreqPtr += UP_FREQ_MTF_POS - 1;
        *(FreqPtr + 1) += 1;
        if (position + 1 == QueueSize)
          unused_queue_freq += 1;
      }
      else if (position == 255) {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 1;
      }
      else {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 2;
        *(FreqPtr + 1) += 1;
        if (position + 1 == QueueSize)
          unused_queue_freq += 1;
      }
    }
    else
      *FreqPtr += UP_FREQ_MTF_POS;
  }
  if ((RangeScaleMtfPos[0] += UP_FREQ_MTF_POS) > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(0);
  return;
}


void EncodeMtfPosAz(uint8_t position, uint16_t QueueSize) {
  NormalizeEncoder(FREQ_MTF_POS_BOT);
  if (last_queue_size_az > QueueSize) {
    do {
      unused_queue_freq_az += FreqMtfPos[1][--last_queue_size_az];
    } while (last_queue_size_az != QueueSize);
  }
  else if (last_queue_size_az < QueueSize) {
    do {
      unused_queue_freq_az -= FreqMtfPos[1][last_queue_size_az++];
      if (last_queue_size_az > rescale_queue_size_az) {
        rescale_queue_size_az++;
        FreqMtfPos[1][last_queue_size_az - 1] += 16;
        RangeScaleMtfPos[1] += 16;
      }
      else {
        FreqMtfPos[1][last_queue_size_az - 1] += 4;
        RangeScaleMtfPos[1] += 4;
      }
    } while (last_queue_size_az != QueueSize);
  }
  if (RangeScaleMtfPos[1] > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(1);

  if (position == 0) {
    range = FreqMtfPos[1][0] * (range / (RangeScaleMtfPos[1] - unused_queue_freq_az));
    FreqMtfPos[1][0] += UP_FREQ_MTF_POS;
  }
  else {
    uint16_t * FreqPtr = &FreqMtfPos[1][0];
    uint16_t * StopFreqPtr = &FreqMtfPos[1][position];
    RangeLow = *FreqPtr++;
    while (FreqPtr != StopFreqPtr)
      RangeLow += *FreqPtr++;
    low += RangeLow * (range /= (RangeScaleMtfPos[1] - unused_queue_freq_az));
    range *= *FreqPtr;

    if (position >= 4) {
      if (position == 4) {
        *FreqPtr += UP_FREQ_MTF_POS - 1;
        *(FreqPtr + 1) += 1;
        if (position + 1 == QueueSize)
          unused_queue_freq_az += 1;
      }
      else if (position == 255) {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 1;
      }
      else {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 2;
        *(FreqPtr + 1) += 1;
        if (position + 1 == QueueSize)
          unused_queue_freq_az += 1;
      }
    }
    else
      *FreqPtr += UP_FREQ_MTF_POS;
  }
  if ((RangeScaleMtfPos[1] += UP_FREQ_MTF_POS) > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(1);
  return;
}

void EncodeMtfPosSpace(uint8_t position, uint16_t QueueSize) {
  NormalizeEncoder(FREQ_MTF_POS_BOT);
  if (last_queue_size_space > QueueSize) {
    do {
      unused_queue_freq_space += FreqMtfPos[2][--last_queue_size_space];
    } while (last_queue_size_space != QueueSize);
  }
  else if (last_queue_size_space < QueueSize) {
    do {
      unused_queue_freq_space -= FreqMtfPos[2][last_queue_size_space++];
      if (last_queue_size_space > rescale_queue_size_space) {
        rescale_queue_size_space++;
        FreqMtfPos[2][last_queue_size_space - 1] += 16;
        RangeScaleMtfPos[2] += 16;
      }
      else {
        FreqMtfPos[2][last_queue_size_space - 1] += 4;
        RangeScaleMtfPos[2] += 4;
      }
    } while (last_queue_size_space != QueueSize);
  }
  if (RangeScaleMtfPos[2] > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(2);
  if (position == 0) {
    range = FreqMtfPos[2][0] * (range / (RangeScaleMtfPos[2] - unused_queue_freq_space));
    FreqMtfPos[2][0] += UP_FREQ_MTF_POS;
  }
  else {
    uint16_t * FreqPtr = &FreqMtfPos[2][0];
    uint16_t * StopFreqPtr = &FreqMtfPos[2][position];
    RangeLow = *FreqPtr++;
    while (FreqPtr != StopFreqPtr)
      RangeLow += *FreqPtr++;
    low += RangeLow * (range /= (RangeScaleMtfPos[2] - unused_queue_freq_space));
    range *= *FreqPtr;

    if (position >= 4) {
      if (position == 4) {
        *FreqPtr += UP_FREQ_MTF_POS - 1;
        *(FreqPtr + 1) += 1;
        if (position + 1 == QueueSize)
          unused_queue_freq_space += 1;
      }
      else if (position == 255) {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 1;
      }
      else {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 2;
        *(FreqPtr + 1) += 1;
        if (position + 1 == QueueSize)
          unused_queue_freq_space += 1;
      }
    }
    else
      *FreqPtr += UP_FREQ_MTF_POS;
  }
  if ((RangeScaleMtfPos[2] += UP_FREQ_MTF_POS) > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(2);
  return;
}

void EncodeMtfPosOther(uint8_t position, uint16_t QueueSize) {
  NormalizeEncoder(FREQ_MTF_POS_BOT);
  if (last_queue_size_other > QueueSize) {
    do {
      unused_queue_freq_other += FreqMtfPos[3][--last_queue_size_other];
    } while (last_queue_size_other != QueueSize);
  }
  else if (last_queue_size_other < QueueSize) {
    do {
      unused_queue_freq_other -= FreqMtfPos[3][last_queue_size_other++];
      if (last_queue_size_other > rescale_queue_size_other) {
        rescale_queue_size_other++;
        FreqMtfPos[3][last_queue_size_other - 1] += 16;
        RangeScaleMtfPos[3] += 16;
      }
      else {
        FreqMtfPos[3][last_queue_size_other - 1] += 4;
        RangeScaleMtfPos[3] += 4;
      }
    } while (last_queue_size_other != QueueSize);
  }
  if (RangeScaleMtfPos[3] > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(3);
  if (position == 0) {
    range = FreqMtfPos[3][0] * (range / (RangeScaleMtfPos[3] - unused_queue_freq_other));
    FreqMtfPos[3][0] += UP_FREQ_MTF_POS;
  }
  else {
    uint16_t * FreqPtr = &FreqMtfPos[3][0];
    uint16_t * StopFreqPtr = &FreqMtfPos[3][position];
    RangeLow = *FreqPtr++;
    while (FreqPtr != StopFreqPtr)
      RangeLow += *FreqPtr++;
    low += RangeLow * (range /= (RangeScaleMtfPos[3] - unused_queue_freq_other));
    range *= *FreqPtr;

    if (position >= 4) {
      if (position == 4) {
        *FreqPtr += UP_FREQ_MTF_POS - 1;
        *(FreqPtr + 1) += 1;
        if (position + 1 == QueueSize)
          unused_queue_freq_other += 1;
      }
      else if (position == 255) {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 1;
      }
      else {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 2;
        *(FreqPtr + 1) += 1;
        if (position + 1 == QueueSize)
          unused_queue_freq_other += 1;
      }
    }
    else
      *FreqPtr += UP_FREQ_MTF_POS;
  }
  if ((RangeScaleMtfPos[3] += UP_FREQ_MTF_POS) > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(3);
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
  uint32_t old_range = range;
  range /= RangeScaleINST[Context][SIDSymbol];
  if (Symbol == 0) {
    range = old_range - range * (RangeScaleINST[Context][SIDSymbol] - FreqINST[Context][SIDSymbol][0]);
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
    uint8_t FoundIndex = 1;
    while (FoundIndex != Symbol)
      RangeLow += FreqINST[Context][SIDSymbol][FoundIndex++];
    low += range * RangeLow + old_range - range * RangeScaleINST[Context][SIDSymbol];
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

void EncodeERG(uint16_t Context1, uint16_t Context2, uint8_t Symbol) {
  NormalizeEncoder(FREQ_ERG_BOT);
  uint32_t extra_range = range & (FREQ_ERG_BOT - 1);
  if (Symbol == 0) {
    range = (FreqERG[0] + FreqERG[Context1] + FreqERG[Context2]) * (range >> 13) + extra_range;
    FreqERG[0] += (0x400 - FreqERG[0]) >> 2;
    FreqERG[Context1] += (0x1000 - FreqERG[Context1]) >> 4;
    FreqERG[Context2] += (0xC00 - FreqERG[Context2]) >> 3;
  }
  else {
    low += (FreqERG[0] + FreqERG[Context1] + FreqERG[Context2]) * (range >>= 13) + extra_range;
    range *= 0x2000 - (FreqERG[0] + FreqERG[Context1] + FreqERG[Context2]);
    FreqERG[0] -= FreqERG[0] >> 2;
    FreqERG[Context1] -= FreqERG[Context1] >> 4;
    FreqERG[Context2] -= FreqERG[Context2] >> 3;
  }
  return;
}

void EncodeGoMtf(uint16_t Context1, uint8_t Context2, uint8_t Symbol) {
  NormalizeEncoder(FREQ_GO_MTF_BOT);
  uint32_t extra_range = range & (FREQ_GO_MTF_BOT - 1);
  Context1 += 0xF0 * Context2;
  uint16_t Context3 = Context1 + 0x2D0;
  if (Symbol == 0) {
    range = (FreqGoMtf[Context1] + FreqGoMtf[Context2] + 2 * FreqGoMtf[Context3]) * (range >> 13) + extra_range;
    FreqGoMtf[Context1] += (0x800 - FreqGoMtf[Context1]) >> 2;
    FreqGoMtf[Context2] += (0x800 - FreqGoMtf[Context2]) >> 2;
    FreqGoMtf[Context3] += (0x800 - FreqGoMtf[Context3]) >> 6;
  }
  else {
    low += (FreqGoMtf[Context1] + FreqGoMtf[Context2] + 2 * FreqGoMtf[Context3]) * (range >>= 13) + extra_range;
    range *= 0x2000 - (FreqGoMtf[Context1] + FreqGoMtf[Context2] + 2 * FreqGoMtf[Context3]);
    FreqGoMtf[Context1] -= FreqGoMtf[Context1] >> 2;
    FreqGoMtf[Context2] -= FreqGoMtf[Context2] >> 2;
    FreqGoMtf[Context3] -= FreqGoMtf[Context3] >> 6;
  }
  return;
}

void EncodeWordTag(uint8_t Symbol, uint8_t Context) {
  NormalizeEncoder(FREQ_WORD_TAG_BOT);
  uint32_t extra_range = range & (FREQ_WORD_TAG_BOT - 1);
  if (Symbol == 0) {
    range = FreqWordTag[Context] * (range >> 12) + extra_range;
    FreqWordTag[Context] += (0x1000 - FreqWordTag[Context]) >> 4;
  }
  else {
    low += FreqWordTag[Context] * (range >>= 12) + extra_range;
    range *= 0x1000 - FreqWordTag[Context];
    FreqWordTag[Context] -= FreqWordTag[Context] >> 4;
  }
  return;
}

void EncodeShortDictionarySymbol(uint16_t BinNum, uint16_t DictionaryBins, uint16_t CodeBins) {
  NormalizeEncoder(1 << 12);
  low += BinNum * (range /= DictionaryBins);
  range *= (uint32_t)CodeBins;
  return;
}

void EncodeLongDictionarySymbol(uint32_t BinCode, uint16_t BinNum, uint16_t DictionaryBins, uint8_t CodeLength,
    uint16_t CodeBins) {
  NormalizeEncoder((uint32_t)1 << 12);
  low += BinNum * (range /= DictionaryBins);
  NormalizeEncoder((uint32_t)1 << CodeLength);
  low += BinCode * (range >>= CodeLength);
  range *= (uint32_t)CodeBins;
  return;
}

void EncodeBaseSymbol(uint32_t BaseSymbol, uint32_t NumBaseSymbols, uint32_t NormBaseSymbols) {
  NormalizeEncoder(NormBaseSymbols);
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
    uint8_t FoundIndex = 1;
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
    if (FreqFirstChar[SymType][LastChar][FoundIndex] > FreqFirstChar[SymType][LastChar][FoundIndex - 1]) {
      uint16_t SavedFreq = FreqFirstChar[SymType][LastChar][FoundIndex];
      do {
        FreqFirstChar[SymType][LastChar][FoundIndex] = FreqFirstChar[SymType][LastChar][FoundIndex - 1];
        SymbolFirstChar[SymType][LastChar][FoundIndex] = SymbolFirstChar[SymType][LastChar][FoundIndex - 1];
      } while ((--FoundIndex) && (SavedFreq > FreqFirstChar[SymType][LastChar][FoundIndex - 1]));
      FreqFirstChar[SymType][LastChar][FoundIndex] = SavedFreq;
      SymbolFirstChar[SymType][LastChar][FoundIndex] = Symbol;
    }
  }
  return;
}

void UpdateFirstChar(uint8_t Symbol, uint8_t SymType, uint8_t LastChar) {
  NormalizeEncoder(FREQ_FIRST_CHAR_BOT);
  if (Symbol == SymbolFirstChar[SymType][LastChar][0]) {
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
    uint8_t FoundIndex = 1;
    while (SymbolFirstChar[SymType][LastChar][FoundIndex] != Symbol)
      FoundIndex++;
    if (RangeScaleFirstChar[SymType][LastChar] >= (FREQ_FIRST_CHAR_BOT >> 1)) {
      FreqFirstChar[SymType][LastChar][FoundIndex] += RangeScaleFirstChar[SymType][LastChar] >> 9;
      if ((RangeScaleFirstChar[SymType][LastChar] += (RangeScaleFirstChar[SymType][LastChar] >> 9)) > FREQ_FIRST_CHAR_BOT)
        rescaleFirstChar(SymType, LastChar);
    }
    else {
      FreqFirstChar[SymType][LastChar][FoundIndex] += UP_FREQ_FIRST_CHAR;
      RangeScaleFirstChar[SymType][LastChar] += UP_FREQ_FIRST_CHAR;
    }
    if (FreqFirstChar[SymType][LastChar][FoundIndex] > FreqFirstChar[SymType][LastChar][FoundIndex - 1]) {
      uint16_t SavedFreq = FreqFirstChar[SymType][LastChar][FoundIndex];
      do {
        FreqFirstChar[SymType][LastChar][FoundIndex] = FreqFirstChar[SymType][LastChar][FoundIndex - 1];
        SymbolFirstChar[SymType][LastChar][FoundIndex] = SymbolFirstChar[SymType][LastChar][FoundIndex - 1];
      } while ((--FoundIndex) && (SavedFreq > FreqFirstChar[SymType][LastChar][FoundIndex - 1]));
      FreqFirstChar[SymType][LastChar][FoundIndex] = SavedFreq;
      SymbolFirstChar[SymType][LastChar][FoundIndex] = Symbol;
    }
  }
  return;
}

void EncodeFirstCharBinary(uint8_t Symbol, uint8_t LastChar) {
  NormalizeEncoder(FREQ_FIRST_CHAR_BOT);
  if (Symbol < 0x80) {
    RangeScaleFirstCharSection[LastChar][3] += UP_FREQ_FIRST_CHAR;
    if (Symbol < 0x40) {
      RangeScaleFirstCharSection[LastChar][1] += UP_FREQ_FIRST_CHAR;
      if (Symbol < 0x20) {
        RangeScaleFirstCharSection[LastChar][0] += UP_FREQ_FIRST_CHAR;
        if (Symbol == 0) {
          range = FreqFirstCharBinary[LastChar][0] * (range / RangeScaleFirstChar[0][LastChar]);
          FreqFirstCharBinary[LastChar][0] += UP_FREQ_FIRST_CHAR;
        }
        else {
          RangeLow = FreqFirstCharBinary[LastChar][0];
          uint8_t FoundIndex = 1;
          while (FoundIndex != Symbol)
            RangeLow += FreqFirstCharBinary[LastChar][FoundIndex++];
          low += RangeLow * (range /= RangeScaleFirstChar[0][LastChar]);
          range *= FreqFirstCharBinary[LastChar][FoundIndex];
          FreqFirstCharBinary[LastChar][FoundIndex] += UP_FREQ_FIRST_CHAR;
        }
      }
      else {
        RangeLow = RangeScaleFirstCharSection[LastChar][0];
        uint8_t FoundIndex = 0x20;
        while (FoundIndex != Symbol)
          RangeLow += FreqFirstCharBinary[LastChar][FoundIndex++];
        low += RangeLow * (range /= RangeScaleFirstChar[0][LastChar]);
        range *= FreqFirstCharBinary[LastChar][FoundIndex];
        FreqFirstCharBinary[LastChar][FoundIndex] += UP_FREQ_FIRST_CHAR;
      }
    }
    else {
      RangeLow = RangeScaleFirstCharSection[LastChar][1];
      if (Symbol < 0x60) {
        RangeScaleFirstCharSection[LastChar][2] += UP_FREQ_FIRST_CHAR;
        uint8_t FoundIndex = 0x40;
        while (FoundIndex != Symbol)
          RangeLow += FreqFirstCharBinary[LastChar][FoundIndex++];
        low += RangeLow * (range /= RangeScaleFirstChar[0][LastChar]);
        range *= FreqFirstCharBinary[LastChar][FoundIndex];
        FreqFirstCharBinary[LastChar][FoundIndex] += UP_FREQ_FIRST_CHAR;
      }
      else {
        RangeLow += RangeScaleFirstCharSection[LastChar][2];
        uint8_t FoundIndex = 0x60;
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
    if (Symbol < 0xC0) {
      RangeScaleFirstCharSection[LastChar][5] += UP_FREQ_FIRST_CHAR;
      if (Symbol < 0xA0) {
        RangeScaleFirstCharSection[LastChar][4] += UP_FREQ_FIRST_CHAR;
        uint8_t FoundIndex = 0x80;
        while (FoundIndex != Symbol)
          RangeLow += FreqFirstCharBinary[LastChar][FoundIndex++];
        low += RangeLow * (range /= RangeScaleFirstChar[0][LastChar]);
        range *= FreqFirstCharBinary[LastChar][FoundIndex];
        FreqFirstCharBinary[LastChar][FoundIndex] += UP_FREQ_FIRST_CHAR;
      }
      else {
        RangeLow += RangeScaleFirstCharSection[LastChar][4];
        uint8_t FoundIndex = 0xA0;
        while (FoundIndex != Symbol)
          RangeLow += FreqFirstCharBinary[LastChar][FoundIndex++];
        low += RangeLow * (range /= RangeScaleFirstChar[0][LastChar]);
        range *= FreqFirstCharBinary[LastChar][FoundIndex];
        FreqFirstCharBinary[LastChar][FoundIndex] += UP_FREQ_FIRST_CHAR;
      }
    }
    else {
      RangeLow += RangeScaleFirstCharSection[LastChar][5];
      if (Symbol < 0xE0) {
        RangeScaleFirstCharSection[LastChar][6] += UP_FREQ_FIRST_CHAR;
        uint8_t FoundIndex = 0xC0;
        while (FoundIndex != Symbol)
          RangeLow += FreqFirstCharBinary[LastChar][FoundIndex++];
        low += RangeLow * (range /= RangeScaleFirstChar[0][LastChar]);
        range *= FreqFirstCharBinary[LastChar][FoundIndex];
        FreqFirstCharBinary[LastChar][FoundIndex] += UP_FREQ_FIRST_CHAR;
      }
      else {
        RangeLow += RangeScaleFirstCharSection[LastChar][6];
        uint8_t FoundIndex = 0xE0;
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

uint32_t ReadOutCharNum() {
  return(OutCharNum);
}

void InitEncoder(uint8_t max_code_length, uint8_t max_base_code, uint8_t num_inst_codes, uint8_t cap_encoded,
    uint8_t UTF8_compliant, uint8_t use_mtf) {
  MaxBaseCode = max_base_code;
  MaxInstCode = num_inst_codes - 1;
  OutCharNum = 0;
  low = 0, range = -1;
  StartModelSymType(use_mtf, cap_encoded);
  StartModelMtfFirst();
  StartModelMtfQueuePos(max_code_length);
  StartModelSID();
  StartModelINST(num_inst_codes);
  StartModelERG();
  StartModelGoQueue();
  StartModelWordTag();
  if (cap_encoded || UTF8_compliant)
    StartModelFirstChar();
  else
    StartModelFirstCharBinary();
}

void FinishEncoder() {
  OutBuffer[OutCharNum++] = (uint8_t)(low >> 24);
  OutBuffer[OutCharNum++] = (uint8_t)(low >> 16);
  OutBuffer[OutCharNum++] = (uint8_t)(low >> 8);
  OutBuffer[OutCharNum++] = (uint8_t)low;
}

void NormalizeDecoder(uint32_t bot) {
  while ((low ^ (low + range)) < TOP || (range < (bot) && ((range = -low & ((bot) - 1)), 1))) {
    code = (code << 8) | InBuffer[InCharNum++];
    low <<= 8;
    range <<= 8;
  }
}

uint8_t DecodeSymType1(uint8_t Context1) {
  uint32_t dict_range;
  NormalizeDecoder(FREQ_SYM_TYPE_BOT1);
  uint32_t extra_range = range & (FREQ_SYM_TYPE_BOT1 - 1);
  if ((dict_range = (range >>= 14) * FreqSymTypePriorType[Context1][0] + extra_range) > code - low) {
    range = dict_range;
    uint16_t delta = FreqSymTypePriorType[Context1][1] >> 6;
    FreqSymTypePriorType[Context1][0]
        += delta + ((FREQ_SYM_TYPE_BOT1 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 6);
    FreqSymTypePriorType[Context1][1] -= delta;
    return(0);
  }
  else if (dict_range + range * FreqSymTypePriorType[Context1][1] > code - low) {
    low += dict_range;
    range *= FreqSymTypePriorType[Context1][1];
    uint16_t delta = FreqSymTypePriorType[Context1][0] >> 6;
    FreqSymTypePriorType[Context1][1]
        += delta + ((FREQ_SYM_TYPE_BOT1 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 6);
    FreqSymTypePriorType[Context1][0] -= delta;
    return(1);
  }
  else {
    low += dict_range + range * FreqSymTypePriorType[Context1][1];
    range *= (FREQ_SYM_TYPE_BOT1 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]);
    FreqSymTypePriorType[Context1][0] -= FreqSymTypePriorType[Context1][0] >> 6;
    FreqSymTypePriorType[Context1][1] -= FreqSymTypePriorType[Context1][1] >> 6;
    return(2);
  }
}

uint8_t DecodeSymType2(uint8_t Context1, uint8_t Context2) {
  uint32_t dict_range;
  NormalizeDecoder(2 * FREQ_SYM_TYPE_BOT2);
  uint32_t extra_range = range & (2 * FREQ_SYM_TYPE_BOT2 - 1);
  if ((dict_range = (range >>= 14)
      * (FreqSymTypePriorType[Context1][0] + FreqSymTypePriorType[Context2][0]) + extra_range) > code - low) {
    range = dict_range;
    uint16_t delta = FreqSymTypePriorType[Context1][1] >> 4;
    FreqSymTypePriorType[Context1][0]
        += delta + ((FREQ_SYM_TYPE_BOT2 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 4);
    FreqSymTypePriorType[Context1][1] -= delta;
    delta = FreqSymTypePriorType[Context2][1] >> 7;
    FreqSymTypePriorType[Context2][0]
        += delta + ((FREQ_SYM_TYPE_BOT2 - FreqSymTypePriorType[Context2][0] - FreqSymTypePriorType[Context2][1]) >> 7);
    FreqSymTypePriorType[Context2][1] -= delta;
    return(0);
  }
  else if (dict_range + range * (FreqSymTypePriorType[Context1][1] + FreqSymTypePriorType[Context2][1]) > code - low) {
    low += dict_range;
    range *= (FreqSymTypePriorType[Context1][1] + FreqSymTypePriorType[Context2][1]);
    uint16_t delta = FreqSymTypePriorType[Context1][0] >> 4;
    FreqSymTypePriorType[Context1][1]
        += delta + ((FREQ_SYM_TYPE_BOT2 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 4);
    FreqSymTypePriorType[Context1][0] -= delta;
    delta = FreqSymTypePriorType[Context2][0] >> 7;
    FreqSymTypePriorType[Context2][1]
        += delta + ((FREQ_SYM_TYPE_BOT2 - FreqSymTypePriorType[Context2][0] - FreqSymTypePriorType[Context2][1]) >> 7);
    FreqSymTypePriorType[Context2][0] -= delta;
    return(1);
  }
  else {
    low += dict_range + range * (FreqSymTypePriorType[Context1][1] + FreqSymTypePriorType[Context2][1]);
    range *= (2 * FREQ_SYM_TYPE_BOT2 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context2][0]
        - FreqSymTypePriorType[Context1][1] - FreqSymTypePriorType[Context2][1]);
    FreqSymTypePriorType[Context1][0] -= FreqSymTypePriorType[Context1][0] >> 4;
    FreqSymTypePriorType[Context1][1] -= FreqSymTypePriorType[Context1][1] >> 4;
    FreqSymTypePriorType[Context2][0] -= FreqSymTypePriorType[Context2][0] >> 7;
    FreqSymTypePriorType[Context2][1] -= FreqSymTypePriorType[Context2][1] >> 7;
    return(2);
  }
}

uint8_t DecodeSymType3(uint8_t Context1, uint8_t Context2, uint8_t Context3) {
  uint32_t dict_range;
  NormalizeDecoder(8 * FREQ_SYM_TYPE_BOT3);
  uint32_t extra_range = range & (8 * FREQ_SYM_TYPE_BOT3 - 1);
  if ((dict_range = (FreqSymTypePriorType[Context1][0] + FreqSymTypePriorType[Context2][0]
      + FreqSymTypePriorEnd[Context3][0]) * (range >>= 15) + extra_range) > code - low) {
    range = dict_range;
    uint16_t delta = FreqSymTypePriorType[Context1][1] >> 4;
    FreqSymTypePriorType[Context1][0]
        += delta + ((3 * FREQ_SYM_TYPE_BOT3 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 4);
    FreqSymTypePriorType[Context1][1] -= delta;
    delta = FreqSymTypePriorType[Context2][1] >> 7;
    FreqSymTypePriorType[Context2][0]
        += delta + ((3 * FREQ_SYM_TYPE_BOT3 - FreqSymTypePriorType[Context2][0] - FreqSymTypePriorType[Context2][1]) >> 7);
    FreqSymTypePriorType[Context2][1] -= delta;
    delta = FreqSymTypePriorEnd[Context3][1] >> 3;
    FreqSymTypePriorEnd[Context3][0]
        += delta + ((2 * FREQ_SYM_TYPE_BOT3 - FreqSymTypePriorEnd[Context3][0] - FreqSymTypePriorEnd[Context3][1]) >> 3);
    FreqSymTypePriorEnd[Context3][1] -= delta;
    return(0);
  }
  else if (dict_range + range * (FreqSymTypePriorType[Context1][1] + FreqSymTypePriorType[Context2][1]
      + FreqSymTypePriorEnd[Context3][1]) > code - low) {
    low += dict_range;
    range *= FreqSymTypePriorType[Context1][1] + FreqSymTypePriorType[Context2][1] + FreqSymTypePriorEnd[Context3][1];
    uint16_t delta = FreqSymTypePriorType[Context1][0] >> 4;
    FreqSymTypePriorType[Context1][1]
        += delta + ((3 * FREQ_SYM_TYPE_BOT3 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 4);
    FreqSymTypePriorType[Context1][0] -= delta;
    delta = FreqSymTypePriorType[Context2][0] >> 7;
    FreqSymTypePriorType[Context2][1]
        += delta + ((3 * FREQ_SYM_TYPE_BOT3 - FreqSymTypePriorType[Context2][0] - FreqSymTypePriorType[Context2][1]) >> 7);
    FreqSymTypePriorType[Context2][0] -= delta;
    delta = FreqSymTypePriorEnd[Context3][0] >> 3;
    FreqSymTypePriorEnd[Context3][1]
        += delta + ((2 * FREQ_SYM_TYPE_BOT3 - FreqSymTypePriorEnd[Context3][0] - FreqSymTypePriorEnd[Context3][1]) >> 3);
    FreqSymTypePriorEnd[Context3][0] -= delta;
    return(1);
  }
  else {
    low += dict_range + range * (FreqSymTypePriorType[Context1][1] + FreqSymTypePriorType[Context2][1]
        + FreqSymTypePriorEnd[Context3][1]);
    range *= 8 * FREQ_SYM_TYPE_BOT3 - (FreqSymTypePriorType[Context1][0] + FreqSymTypePriorType[Context1][1])
        - (FreqSymTypePriorType[Context2][0] + FreqSymTypePriorType[Context2][1])
        - (FreqSymTypePriorEnd[Context3][0] + FreqSymTypePriorEnd[Context3][1]);
    FreqSymTypePriorType[Context1][0] -= FreqSymTypePriorType[Context1][0] >> 4;
    FreqSymTypePriorType[Context1][1] -= FreqSymTypePriorType[Context1][1] >> 4;
    FreqSymTypePriorType[Context2][0] -= FreqSymTypePriorType[Context2][0] >> 7;
    FreqSymTypePriorType[Context2][1] -= FreqSymTypePriorType[Context2][1] >> 7;
    FreqSymTypePriorEnd[Context3][0] -= FreqSymTypePriorEnd[Context3][0] >> 3;
    FreqSymTypePriorEnd[Context3][1] -= FreqSymTypePriorEnd[Context3][1] >> 3;
    return(2);
  }
}

uint8_t DecodeMtfFirst(uint8_t Context) {
  uint32_t delta;
  NormalizeDecoder(0x1000);
  uint32_t extra_range = range & 0xFFF;

  if (FreqMtfFirst[Context][0] * (range >>= 12) + extra_range > code - low) {
    range = range * FreqMtfFirst[Context][0] + extra_range;
    delta = FreqMtfFirst[Context][1] >> 7;
    FreqMtfFirst[Context][1] -= delta;
    FreqMtfFirst[Context][0] += delta;
    delta = FreqMtfFirst[Context][2] >> 7;
    FreqMtfFirst[Context][2] -= delta;
    FreqMtfFirst[Context][0] += delta;
    return(0);
  }
  else if ((FreqMtfFirst[Context][0] + FreqMtfFirst[Context][1]) * range + extra_range > code - low) {
    low += range * FreqMtfFirst[Context][0] + extra_range;
    range *= FreqMtfFirst[Context][1];
    delta = FreqMtfFirst[Context][0] >> 7;
    FreqMtfFirst[Context][0] -= delta;
    FreqMtfFirst[Context][1] += delta;
    delta = FreqMtfFirst[Context][2] >> 7;
    FreqMtfFirst[Context][2] -= delta;
    FreqMtfFirst[Context][1] += delta;
    return(1);
  }
  else {
    low += (FreqMtfFirst[Context][0] + FreqMtfFirst[Context][1]) * range + extra_range;
    range *= FreqMtfFirst[Context][2];
    delta = FreqMtfFirst[Context][0] >> 7;
    FreqMtfFirst[Context][0] -= delta;
    FreqMtfFirst[Context][2] += delta;
    delta = FreqMtfFirst[Context][1] >> 7;
    FreqMtfFirst[Context][1] -= delta;
    FreqMtfFirst[Context][2] += delta;
    return(2);
  }
}

uint8_t DecodeMtfPos(uint16_t QueueSize) {
  NormalizeDecoder(FREQ_MTF_POS_BOT);
  if (last_queue_size > QueueSize) {
    do {
      unused_queue_freq += FreqMtfPos[0][--last_queue_size];
    } while (last_queue_size != QueueSize);
  }
  else if (last_queue_size < QueueSize) {
    do {
      unused_queue_freq -= FreqMtfPos[0][last_queue_size++];
      if (last_queue_size > rescale_queue_size) {
        rescale_queue_size++;
        FreqMtfPos[0][last_queue_size - 1] += 8;
        RangeScaleMtfPos[0] += 8;
      }
      else {
        FreqMtfPos[0][last_queue_size - 1] += 2;
        RangeScaleMtfPos[0] += 2;
      }
    } while (last_queue_size != QueueSize);
    if (RangeScaleMtfPos[0] > FREQ_MTF_POS_BOT)
      rescaleMtfQueuePos(0);
  }
  count = (code - low) / (range /= (RangeScaleMtfPos[0] - unused_queue_freq));
  if ((RangeHigh = FreqMtfPos[0][0]) > count) {
    range *= RangeHigh;
    FreqMtfPos[0][0] = RangeHigh + UP_FREQ_MTF_POS;
    if ((RangeScaleMtfPos[0] += UP_FREQ_MTF_POS) > FREQ_MTF_POS_BOT)
      rescaleMtfQueuePos(0);
    return(0);
  }
  else {
    uint16_t * FreqPtr = &FreqMtfPos[0][1];
    while ((RangeHigh += *FreqPtr) <= count)
      FreqPtr++;
    uint8_t position = FreqPtr - &FreqMtfPos[0][0];
    low += range * (RangeHigh - *FreqPtr);
    range *= *FreqPtr;
    if (position >= 4) {
      if (position == 4) {
        *FreqPtr += UP_FREQ_MTF_POS - 1;
        *(FreqPtr + 1) += 1;
        if (position == QueueSize - 1)
          unused_queue_freq += 1;
      }
      else if (position == 255) {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 1;
      }
      else {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 2;
        *(FreqPtr + 1) += 1;
        if (position == QueueSize - 1)
          unused_queue_freq += 1;
      }
    }
    else
       *FreqPtr += UP_FREQ_MTF_POS;
    if ((RangeScaleMtfPos[0] += UP_FREQ_MTF_POS) > FREQ_MTF_POS_BOT)
      rescaleMtfQueuePos(0);
    return(position);
  }
}

uint8_t DecodeMtfPosAz(uint16_t QueueSize) {
  NormalizeDecoder(FREQ_MTF_POS_BOT);
  if (last_queue_size_az > QueueSize) {
    do {
      unused_queue_freq_az += FreqMtfPos[1][--last_queue_size_az];
    } while (last_queue_size_az != QueueSize);
  }
  else if (last_queue_size_az < QueueSize) {
    do {
      unused_queue_freq_az -= FreqMtfPos[1][last_queue_size_az++];
      if (last_queue_size_az > rescale_queue_size_az) {
        rescale_queue_size_az++;
        FreqMtfPos[1][last_queue_size_az - 1] += 16;
        RangeScaleMtfPos[1] += 16;
      }
      else {
        FreqMtfPos[1][last_queue_size_az - 1] += 4;
        RangeScaleMtfPos[1] += 4;
      }
    } while (last_queue_size_az != QueueSize);
    if (RangeScaleMtfPos[1] > FREQ_MTF_POS_BOT)
      rescaleMtfQueuePos(1);
  }
  count = (code - low) / (range /= (RangeScaleMtfPos[1] - unused_queue_freq_az));
  if ((RangeHigh = FreqMtfPos[1][0]) > count) {
    range *= RangeHigh;
    FreqMtfPos[1][0] = RangeHigh + UP_FREQ_MTF_POS;
    if ((RangeScaleMtfPos[1] += UP_FREQ_MTF_POS) > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(1);
    return(0);
  }
  else {
    uint16_t * FreqPtr = &FreqMtfPos[1][1];
    while ((RangeHigh += *FreqPtr) <= count)
      FreqPtr++;
    uint8_t position = FreqPtr - &FreqMtfPos[1][0];
    low += range * (RangeHigh - *FreqPtr);
    range *= *FreqPtr;
    if (position >= 4) {
      if (position == 4) {
        *FreqPtr += UP_FREQ_MTF_POS - 1;
        *(FreqPtr + 1) += 1;
        if (position == QueueSize - 1)
          unused_queue_freq_az += 1;
      }
      else if (position == 255) {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 1;
      }
      else {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 2;
        *(FreqPtr + 1) += 1;
        if (position == QueueSize - 1)
          unused_queue_freq_az += 1;
      }
    }
    else
       *FreqPtr += UP_FREQ_MTF_POS;
    if ((RangeScaleMtfPos[1] += UP_FREQ_MTF_POS) > FREQ_MTF_POS_BOT)
      rescaleMtfQueuePos(1);
    return(position);
  }
}

uint8_t DecodeMtfPosSpace(uint16_t QueueSize) {
  NormalizeDecoder(FREQ_MTF_POS_BOT);
  if (last_queue_size_space > QueueSize) {
    do {
      unused_queue_freq_space += FreqMtfPos[2][--last_queue_size_space];
    } while (last_queue_size_space != QueueSize);
  }
  else if (last_queue_size_space < QueueSize) {
    do {
      unused_queue_freq_space -= FreqMtfPos[2][last_queue_size_space++];
      if (last_queue_size_space > rescale_queue_size_space) {
        rescale_queue_size_space++;
        FreqMtfPos[2][last_queue_size_space - 1] += 16;
        RangeScaleMtfPos[2] += 16;
      }
      else {
        FreqMtfPos[2][last_queue_size_space - 1] += 4;
        RangeScaleMtfPos[2] += 4;
      }
    } while (last_queue_size_space != QueueSize);
    if (RangeScaleMtfPos[2] > FREQ_MTF_POS_BOT)
      rescaleMtfQueuePos(2);
  }
  count = (code - low) / (range /= (RangeScaleMtfPos[2] - unused_queue_freq_space));
  if ((RangeHigh = FreqMtfPos[2][0]) > count) {
    range *= RangeHigh;
    FreqMtfPos[2][0] = RangeHigh + UP_FREQ_MTF_POS;
    if ((RangeScaleMtfPos[2] += UP_FREQ_MTF_POS) > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(2);
    return(0);
  }
  else {
    uint16_t * FreqPtr = &FreqMtfPos[2][1];
    while ((RangeHigh += *FreqPtr) <= count)
      FreqPtr++;
    uint8_t position = FreqPtr - &FreqMtfPos[2][0];
    low += range * (RangeHigh - *FreqPtr);
    range *= *FreqPtr;
    if (position >= 4) {
      if (position == 4) {
        *FreqPtr += UP_FREQ_MTF_POS - 1;
        *(FreqPtr + 1) += 1;
        if (position == QueueSize - 1)
          unused_queue_freq_space += 1;
      }
      else if (position == 255) {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 1;
      }
      else {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 2;
        *(FreqPtr + 1) += 1;
        if (position == QueueSize - 1)
          unused_queue_freq_space += 1;
      }
    }
    else
       *FreqPtr += UP_FREQ_MTF_POS;
    if ((RangeScaleMtfPos[2] += UP_FREQ_MTF_POS) > FREQ_MTF_POS_BOT)
      rescaleMtfQueuePos(2);
    return(position);
  }
}

uint8_t DecodeMtfPosOther(uint16_t QueueSize) {
  NormalizeDecoder(FREQ_MTF_POS_BOT);
  if (last_queue_size_other > QueueSize) {
    do {
      unused_queue_freq_other += FreqMtfPos[3][--last_queue_size_other];
    } while (last_queue_size_other != QueueSize);
  }
  else if (last_queue_size_other < QueueSize) {
    do {
      unused_queue_freq_other -= FreqMtfPos[3][last_queue_size_other++];
      if (last_queue_size_other > rescale_queue_size_other) {
        rescale_queue_size_other++;
        FreqMtfPos[3][last_queue_size_other - 1] += 16;
        RangeScaleMtfPos[3] += 16;
      }
      else {
        FreqMtfPos[3][last_queue_size_other - 1] += 4;
        RangeScaleMtfPos[3] += 4;
      }
    } while (last_queue_size_other != QueueSize);
    if (RangeScaleMtfPos[3] > FREQ_MTF_POS_BOT)
      rescaleMtfQueuePos(3);
  }
  count = (code - low) / (range /= (RangeScaleMtfPos[3] - unused_queue_freq_other));
  if ((RangeHigh = FreqMtfPos[3][0]) > count) {
    range *= RangeHigh;
    FreqMtfPos[3][0] = RangeHigh + UP_FREQ_MTF_POS;
    if ((RangeScaleMtfPos[3] += UP_FREQ_MTF_POS) > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(3);
    return(0);
  }
  else {
    uint16_t * FreqPtr = &FreqMtfPos[3][1];
    while ((RangeHigh += *FreqPtr) <= count)
      FreqPtr++;
    uint8_t position = FreqPtr - &FreqMtfPos[3][0];
    low += range * (RangeHigh - *FreqPtr);
    range *= *FreqPtr;
    if (position >= 4) {
      if (position == 4) {
        *FreqPtr += UP_FREQ_MTF_POS - 1;
        *(FreqPtr + 1) += 1;
        if (position == QueueSize - 1)
          unused_queue_freq_other += 1;
      }
      else if (position == 255) {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 1;
      }
      else {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 2;
        *(FreqPtr + 1) += 1;
        if (position == QueueSize - 1)
          unused_queue_freq_other += 1;
      }
    }
    else
       *FreqPtr += UP_FREQ_MTF_POS;
    if ((RangeScaleMtfPos[3] += UP_FREQ_MTF_POS) > FREQ_MTF_POS_BOT)
      rescaleMtfQueuePos(3);
    return(position);
  }
}

uint8_t DecodeSID(uint8_t Context) {
  NormalizeDecoder(FREQ_SID_BOT);
  count = (code - low) / (range /= RangeScaleSID[Context]);
  if ((RangeHigh = FreqSID[Context][0]) > count) {
    range *= RangeHigh;
    FreqSID[Context][0] = RangeHigh + UP_FREQ_SID;
    if ((RangeScaleSID[Context] += UP_FREQ_SID) > FREQ_SID_BOT)
      rescaleSID(Context);
    return(0);
  }
  else {
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
}

uint8_t DecodeExtraLength() {
  NormalizeDecoder((uint32_t)1 << 2);
  uint32_t Symbol = (code - low) / (range >>= 2);
  low += range * Symbol;
  return((uint8_t)Symbol);
}

uint8_t DecodeINST(uint8_t Context, uint8_t SIDSymbol) {
  NormalizeDecoder(FREQ_INST_BOT);
  uint32_t extra_range = range;
  range /= RangeScaleINST[Context][SIDSymbol];
  extra_range -= range * RangeScaleINST[Context][SIDSymbol];
  RangeHigh = FreqINST[Context][SIDSymbol][0];
  if (RangeHigh * range + extra_range > code - low) {
    range = range * RangeHigh + extra_range;
    if (RangeScaleINST[Context][SIDSymbol] >= (FREQ_INST_BOT >> 1)) {
      FreqINST[Context][SIDSymbol][0] += RangeScaleINST[Context][SIDSymbol] >> 11;
      if ((RangeScaleINST[Context][SIDSymbol] += RangeScaleINST[Context][SIDSymbol] >> 11) > FREQ_INST_BOT)
        rescaleINST(Context, SIDSymbol);
    }
    else {
      FreqINST[Context][SIDSymbol][0] += UP_FREQ_INST;
      RangeScaleINST[Context][SIDSymbol] += UP_FREQ_INST;
    }
    return(0);
  }
  else {
    low += extra_range;
    count = (code - low) / range;
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
}

uint8_t DecodeERG(uint16_t Context1, uint16_t Context2) {
  NormalizeDecoder(FREQ_ERG_BOT);
  uint32_t extra_range = range & (FREQ_ERG_BOT - 1);
  if ((FreqERG[0] + FreqERG[Context1] + FreqERG[Context2]) * (range >>= 13) + extra_range > code - low) {
    range = range * (FreqERG[0] + FreqERG[Context1] + FreqERG[Context2]) + extra_range;
    FreqERG[0] += (0x400 - FreqERG[0]) >> 2;
    FreqERG[Context1] += (0x1000 - FreqERG[Context1]) >> 4;
    FreqERG[Context2] += (0xC00 - FreqERG[Context2]) >> 3;
    return(0);
  }
  else {
    low += range * (FreqERG[0] + FreqERG[Context1] + FreqERG[Context2]) + extra_range;
    range *= 0x2000 - (FreqERG[0] + FreqERG[Context1] + FreqERG[Context2]);
    FreqERG[0] -= FreqERG[0] >> 2;
    FreqERG[Context1] -= FreqERG[Context1] >> 4;
    FreqERG[Context2] -= FreqERG[Context2] >> 3;
    return(1);
  }
}

uint8_t DecodeGoMtf(uint16_t Context1, uint8_t Context2) {
  uint8_t go_mtf;
  NormalizeDecoder(FREQ_GO_MTF_BOT);
  uint32_t extra_range = range & (FREQ_GO_MTF_BOT - 1);
  Context1 += 0xF0 * Context2;
  uint16_t Context3 = Context1 + 0x2D0;
  if ((FreqGoMtf[Context1] + FreqGoMtf[Context2] + 2 * FreqGoMtf[Context3]) * (range >>= 13) + extra_range > code - low) {
    range = range * (FreqGoMtf[Context1] + FreqGoMtf[Context2] + 2 * FreqGoMtf[Context3]) + extra_range;
    FreqGoMtf[Context1] += (0x800 - FreqGoMtf[Context1]) >> 2;
    FreqGoMtf[Context2] += (0x800 - FreqGoMtf[Context2]) >> 2;
    FreqGoMtf[Context3] += (0x800 - FreqGoMtf[Context3]) >> 6;
    go_mtf = 0;
  }
  else {
    low += range * (FreqGoMtf[Context1] + FreqGoMtf[Context2] + 2 * FreqGoMtf[Context3]) + extra_range;
    range *= 0x2000 - (FreqGoMtf[Context1] + FreqGoMtf[Context2] + 2 * FreqGoMtf[Context3]);
    FreqGoMtf[Context1] -= FreqGoMtf[Context1] >> 2;
    FreqGoMtf[Context2] -= FreqGoMtf[Context2] >> 2;
    FreqGoMtf[Context3] -= FreqGoMtf[Context3] >> 6;
    go_mtf = 1;
  }
  return(go_mtf);
}

uint8_t DecodeWordTag(uint8_t Context) {
  uint8_t Tag;
  NormalizeDecoder(FREQ_WORD_TAG_BOT);
  uint32_t extra_range = range & (FREQ_WORD_TAG_BOT - 1);
  if (FreqWordTag[Context] * (range >>= 12) + extra_range > code - low) {
    range = range * FreqWordTag[Context] + extra_range;
    FreqWordTag[Context] += (0x1000 - FreqWordTag[Context]) >> 4;
    Tag = 0;
  }
  else {
    low += FreqWordTag[Context] * range + extra_range;
    range *= 0x1000 - FreqWordTag[Context];
    FreqWordTag[Context] -= FreqWordTag[Context] >> 4;
    Tag = 1;
  }
  return(Tag);
}

uint16_t DecodeBin(uint16_t Bins) {
  uint16_t BinNum;
  NormalizeDecoder((uint32_t)1 << 12);
  BinNum = (code - low) / (range /= Bins);
  low += range * BinNum;
  return(BinNum);
}

uint32_t DecodeBinCode(uint8_t Bits) {
  NormalizeDecoder((uint32_t)1 << Bits);
  uint32_t BinCode = (code - low) / (range >>= Bits);
  low += BinCode * range;
  return(BinCode);
}

uint32_t DecodeBaseSymbol(uint32_t NumBaseSymbols) {
  NormalizeDecoder(NumBaseSymbols);
  range /= NumBaseSymbols;
  uint32_t BaseSymbol = (code - low) / range;
  low += range * BaseSymbol;
  return(BaseSymbol);
}

uint32_t DecodeBaseSymbolCap(uint32_t NumBaseSymbols) {
  NormalizeDecoder(NumBaseSymbols);
  range /= NumBaseSymbols - 24;
  uint32_t BaseSymbol = (code - low) / range;
  low += range * BaseSymbol;
  return(BaseSymbol);
}

uint8_t DecodeFirstChar(uint8_t SymType, uint8_t LastChar) {
  NormalizeDecoder(FREQ_FIRST_CHAR_BOT);
  range /= RangeScaleFirstChar[SymType][LastChar];
  uint16_t * FreqPtr = &FreqFirstChar[SymType][LastChar][0];
  if ((RangeHigh = *FreqPtr) * range > code - low) {
    range *= RangeHigh;
    if (RangeScaleFirstChar[SymType][LastChar] >= (FREQ_FIRST_CHAR_BOT >> 1)) {
      *FreqPtr += RangeScaleFirstChar[SymType][LastChar] >> 9;
      if ((RangeScaleFirstChar[SymType][LastChar] += (RangeScaleFirstChar[SymType][LastChar] >> 9)) > FREQ_FIRST_CHAR_BOT)
        rescaleFirstChar(SymType, LastChar);
    }
    else {
      *FreqPtr += UP_FREQ_FIRST_CHAR;
      RangeScaleFirstChar[SymType][LastChar] += UP_FREQ_FIRST_CHAR;
    }
    return(SymbolFirstChar[SymType][LastChar][0]);
  }
  else {
    count = (code - low) / range;
    uint16_t temp_range;
    while ((temp_range = RangeHigh + *++FreqPtr) <= count)
      RangeHigh = temp_range;
    low += range * RangeHigh;
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
    uint8_t FoundIndex = FreqPtr - &FreqFirstChar[SymType][LastChar][0];
    uint32_t FirstChar = SymbolFirstChar[SymType][LastChar][FoundIndex];
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
    return(FirstChar);
  }
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

void InitDecoder(uint8_t max_code_length, uint8_t max_base_code, uint8_t num_inst_codes, uint8_t cap_encoded,
    uint8_t UTF8_compliant, uint8_t use_mtf, uint8_t * inbuf) {
  MaxBaseCode = max_base_code;
  MaxInstCode = num_inst_codes - 1;
  InBuffer = inbuf;
  code = 0, range = -1;
  for (low = 4; low != 0; low--)
    code = (code << 8) | InBuffer[InCharNum++];
  StartModelSymType(use_mtf, cap_encoded);
  StartModelMtfFirst();
  StartModelMtfQueuePos(max_code_length);
  StartModelSID();
  StartModelINST(num_inst_codes);
  StartModelERG();
  StartModelGoQueue();
  StartModelWordTag();
  if (cap_encoded || UTF8_compliant)
    StartModelFirstChar();
  else
    StartModelFirstCharBinary();
}

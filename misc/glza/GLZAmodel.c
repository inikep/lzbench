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
#include <stdio.h>  // REMOVE !!
#include <stdlib.h>  // REMOVE !!
//#include <math.h>  // REMOVE !!
#include "GLZAmodel.h"

uint32_t code, low, range, count, RangeLow, RangeHigh;
uint32_t InCharNum, OutCharNum;
uint16_t last_queue_size_az, last_queue_size_space, last_queue_size_other;
uint16_t rescale_queue_size_az, rescale_queue_size_space, rescale_queue_size_other;
uint16_t unused_queue_freq_az, unused_queue_freq_space, unused_queue_freq_other;
uint16_t RangeScaleSID[2], FreqSID[2][16], RangeScaleINST[2][16], FreqINST[2][16][38];
uint16_t FreqWordTag[0x100], FreqERG[341], FreqGoMtf[0x5A0];
uint16_t RangeScaleMtfPos[3], FreqMtfPos[3][0x100], FreqSymTypePriorType[0x34][2], FreqSymTypePriorEnd[0x100][2], FreqMtfFirst[2][3];
uint8_t CapEncoded, UTF8Compliant, MaxBaseCode, MaxInstCode, *InBuffer, *OutBuffer;
uint8_t CapInitialized, CapLockInitialized;
uint16_t RangeScaleFirstCharSection[0x100][7], RangeScaleFirstChar[4][0x100];
struct first_char_data {
  union {
    uint32_t all_data;
    struct {
      uint16_t freq;
      uint8_t symbol;
    } data;
  };
};
struct first_char_data FirstCharData[4][0x100][0x100];
uint8_t NumBaseSymbols = 0;

uint32_t ReadLow() {return(low);}
uint32_t ReadRange() {return(range);}

void StartModelSymType(uint8_t use_mtf, uint8_t cap_encoded) {
  if (cap_encoded == 0) {
    uint8_t i = 1;
    do {
      if (use_mtf != 0)
        FreqSymTypePriorType[i][0] = 0x1C00;
      else
        FreqSymTypePriorType[i][0] = 0x2000;
      FreqSymTypePriorType[i][1] = 0x2000;
    } while (i-- != 0);
  } else {
    uint8_t i = 0x33;
    do {
      if (use_mtf != 0)
        FreqSymTypePriorType[i][0] = 0x16C0;
      else
        FreqSymTypePriorType[i][0] = 0x1A00;
      FreqSymTypePriorType[i][1] = 0x1A00;
    } while (i-- != 4);
    do {
      if (use_mtf != 0)
        FreqSymTypePriorType[i][0] = 0x1340;
      else
        FreqSymTypePriorType[i][0] = 0x1600;
      FreqSymTypePriorType[i][1] = 0x1600;
    } while (i-- != 0);
    i = 0xFF;
    do {
      if (use_mtf != 0)
        FreqSymTypePriorEnd[i][0] = 0xE00;
      else
        FreqSymTypePriorEnd[i][0] = 0x1000;
      FreqSymTypePriorEnd[i][1] = 0x1000;
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
  return;
}

void StartModelMtfPos() {
  RangeScaleMtfPos[0] = 0;
  uint16_t j = 0;
  do {
    FreqMtfPos[0][j] = FreqMtfPos[1][j] = FreqMtfPos[2][j] = 0x200 / (j + 2);
    RangeScaleMtfPos[0] += FreqMtfPos[0][j];
  } while (++j != 0x100);
  RangeScaleMtfPos[1] = RangeScaleMtfPos[2] = RangeScaleMtfPos[0];
  unused_queue_freq_az = unused_queue_freq_space = unused_queue_freq_other = RangeScaleMtfPos[0];
  last_queue_size_az = last_queue_size_space = last_queue_size_other = 0;
  rescale_queue_size_az = rescale_queue_size_space = rescale_queue_size_other = 0;
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

void StartModelGoMtf() {
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
      FirstCharData[0][i][j].data.freq = 0;
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
  uint8_t i = 0xFF;
  if (Context == 0) {
    RangeScaleMtfPos[0] = 0;
    if (last_queue_size_other != 0x100)
      do {
        RangeScaleMtfPos[0] += FreqMtfPos[0][i] = (FreqMtfPos[0][i] + 1) >> 1;
      } while (i-- != last_queue_size_other);
    unused_queue_freq_other = RangeScaleMtfPos[0];
    do {
      RangeScaleMtfPos[0] += FreqMtfPos[0][i] = (FreqMtfPos[0][i] + 1) >> 1;
    } while (i-- != 0);
    rescale_queue_size_other = last_queue_size_other;
  } else if (Context == 1) {
    RangeScaleMtfPos[1] = 0;
    if (last_queue_size_space != 0x100)
      do {
        RangeScaleMtfPos[1] += FreqMtfPos[1][i] = (FreqMtfPos[1][i] + 1) >> 1;
      } while (i-- != last_queue_size_space);
    unused_queue_freq_space = RangeScaleMtfPos[1];
    do {
      RangeScaleMtfPos[1] += FreqMtfPos[1][i] = (FreqMtfPos[1][i] + 1) >> 1;
    } while (i-- != 0);
    rescale_queue_size_space = last_queue_size_space;
  } else if (Context == 2) {
    RangeScaleMtfPos[2] = 0;
    if (last_queue_size_az != 0x100)
      do {
        RangeScaleMtfPos[2] += FreqMtfPos[2][i] = (FreqMtfPos[2][i] + 1) >> 1;
      } while (i-- != last_queue_size_az);
    unused_queue_freq_az = RangeScaleMtfPos[2];
    do {
      RangeScaleMtfPos[2] += FreqMtfPos[2][i] = (FreqMtfPos[2][i] + 1) >> 1;
    } while (i-- != 0);
    rescale_queue_size_az = last_queue_size_az;
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

void rescaleFirstChar(uint8_t SymType, uint8_t PriorEnd) {
  uint8_t i = MaxBaseCode;
  RangeScaleFirstChar[SymType][PriorEnd] = 0;
  do {
    RangeScaleFirstChar[SymType][PriorEnd] += FirstCharData[SymType][PriorEnd][i].data.freq
        = (FirstCharData[SymType][PriorEnd][i].data.freq + 1) >> 1;
  } while (i-- != 0);
  return;
}

void rescaleFirstCharBinary(uint8_t PriorEnd) {
  RangeScaleFirstChar[0][PriorEnd] = FirstCharData[0][PriorEnd][0].data.freq = (FirstCharData[0][PriorEnd][0].data.freq + 1) >> 1;
  uint8_t i = 1;
  do {
    RangeScaleFirstChar[0][PriorEnd] += FirstCharData[0][PriorEnd][i].data.freq = (FirstCharData[0][PriorEnd][i].data.freq + 1) >> 1;
  } while (++i != 0x20);
  RangeScaleFirstCharSection[PriorEnd][0] = RangeScaleFirstChar[0][PriorEnd];
  do {
    RangeScaleFirstChar[0][PriorEnd] += FirstCharData[0][PriorEnd][i].data.freq = (FirstCharData[0][PriorEnd][i].data.freq + 1) >> 1;
  } while (++i != 0x40);
  RangeScaleFirstCharSection[PriorEnd][1] = RangeScaleFirstChar[0][PriorEnd];
  do {
    RangeScaleFirstChar[0][PriorEnd] += FirstCharData[0][PriorEnd][i].data.freq = (FirstCharData[0][PriorEnd][i].data.freq + 1) >> 1;
  } while (++i != 0x60);
  RangeScaleFirstCharSection[PriorEnd][2] = RangeScaleFirstChar[0][PriorEnd];
  do {
    RangeScaleFirstChar[0][PriorEnd] += FirstCharData[0][PriorEnd][i].data.freq = (FirstCharData[0][PriorEnd][i].data.freq + 1) >> 1;
  } while (++i != 0x80);
  RangeScaleFirstCharSection[PriorEnd][3] = RangeScaleFirstChar[0][PriorEnd];
  do {
    RangeScaleFirstChar[0][PriorEnd] += FirstCharData[0][PriorEnd][i].data.freq = (FirstCharData[0][PriorEnd][i].data.freq + 1) >> 1;
  } while (++i != 0xA0);
  RangeScaleFirstCharSection[PriorEnd][4] = RangeScaleFirstChar[0][PriorEnd];
  do {
    RangeScaleFirstChar[0][PriorEnd] += FirstCharData[0][PriorEnd][i].data.freq = (FirstCharData[0][PriorEnd][i].data.freq + 1) >> 1;
  } while (++i != 0xC0);
  RangeScaleFirstCharSection[PriorEnd][5] = RangeScaleFirstChar[0][PriorEnd];
  do {
    RangeScaleFirstChar[0][PriorEnd] += FirstCharData[0][PriorEnd][i].data.freq = (FirstCharData[0][PriorEnd][i].data.freq + 1) >> 1;
  } while (++i != 0xE0);
  RangeScaleFirstCharSection[PriorEnd][6] = RangeScaleFirstChar[0][PriorEnd];
  do {
    RangeScaleFirstChar[0][PriorEnd] += FirstCharData[0][PriorEnd][i].data.freq = (FirstCharData[0][PriorEnd][i].data.freq + 1) >> 1;
  } while (++i != 0);
  RangeScaleFirstCharSection[PriorEnd][6] -= RangeScaleFirstCharSection[PriorEnd][5];
  RangeScaleFirstCharSection[PriorEnd][5] -= RangeScaleFirstCharSection[PriorEnd][4];
  RangeScaleFirstCharSection[PriorEnd][4] -= RangeScaleFirstCharSection[PriorEnd][3];
  RangeScaleFirstCharSection[PriorEnd][3] -= RangeScaleFirstCharSection[PriorEnd][2];
  RangeScaleFirstCharSection[PriorEnd][2] -= RangeScaleFirstCharSection[PriorEnd][1];
  RangeScaleFirstCharSection[PriorEnd][1] -= RangeScaleFirstCharSection[PriorEnd][0];
  return;
}

void InitFirstChar(uint8_t FirstChar, uint8_t code_length) {
// adds the new character (FirstChar) as possibilites for all previously defined characters.
//  except previously defined characters 'B' and 'C' only get defined characters a-z as possibilites
  uint8_t freq, i, j, k, max_index;
  uint8_t either_cap_initialized = CapInitialized || CapLockInitialized;

  if (UTF8Compliant != 0)
    max_index = 0x90;
  else
    max_index = 0xFF;

  i = 3;
  do {
    j = max_index;
    do {
      FirstCharData[i][j][NumBaseSymbols].data.symbol = FirstChar;
    } while (j-- != 0);
  } while (i-- != 0);

  if (code_length < 8)
    freq = 1 << (8 - code_length);
  else
    freq = 1;

  i = max_index;
  do {
    if ((CapEncoded == 0) || ((i > 'Z') || (i < 'A') || ((i == 'C') && ((FirstChar >= 'a') && (FirstChar <= 'z'))))) {
      if ((RangeScaleFirstChar[0][i] != 0) || ((i == 'C') && (either_cap_initialized != 0))) {
        uint8_t k;
        for (k = 0 ; k < 4 ; k++) {
          FirstCharData[k][i][NumBaseSymbols].data.freq = freq;
          RangeScaleFirstChar[k][i] += freq;
          if (RangeScaleFirstChar[k][i] > FREQ_FIRST_CHAR_BOT)
            rescaleFirstChar(k, i);
        }
      }
    }
  } while (i-- != 0);
  return;
}

void InitFirstCharBinary(uint8_t FirstChar, uint8_t code_length) {
  uint8_t freq, i, j;

  if (code_length < 8)
    freq = 1 << (8 - code_length);
  else
    freq = 1;
  i = 0xFF;
  do {
    if (RangeScaleFirstChar[0][i] != 0) {
      FirstCharData[0][i][FirstChar].data.freq = freq;
      RangeScaleFirstChar[0][i] += freq;
      if (FirstChar < 0xE0)
        RangeScaleFirstCharSection[i][FirstChar >> 5] += freq;
      if (RangeScaleFirstChar[0][i] > FREQ_FIRST_CHAR_BOT)
        rescaleFirstCharBinary(i);
    }
  } while (i-- != 0);
  return;
}

void InitPriorEnd(uint8_t PriorEnd, uint8_t * SymbolLengths) {
// adds all previously defined characters and the new character as possibilites for the new context(s) (PriorEnd).
//  except skip 'B' - B is never a PriorEnd (not called with 'B')
//  except 'C' only gets defined characters a-z as possibilites
  uint8_t freq, freq2, code_length, i, j, k;

  k = 3;
  uint8_t PriorEnd_is_a_to_z = (PriorEnd >= 'a') && (PriorEnd <= 'z');
  do {
    i = NumBaseSymbols;
    do {
      uint8_t symbol_i = FirstCharData[k][PriorEnd][i].data.symbol;
      if (((CapEncoded != 0)
            && (((symbol_i == 'C') && (CapInitialized != 0))
              || ((symbol_i == 'B') && (CapLockInitialized != 0))
              || ((symbol_i & 0xFE) != 0x42)))
          || (CapEncoded == 0)) {
        if (((CapEncoded != 0) && ((PriorEnd != 'C') || ((PriorEnd == 'C')
              && (symbol_i >= 'a') && (symbol_i <= 'z'))))
            || (CapEncoded == 0)) {
          code_length = SymbolLengths[symbol_i];
          if (code_length < 8)
            freq = 1 << (8 - code_length);
          else
            freq = 1;
          FirstCharData[k][PriorEnd][i].data.freq = freq;
          RangeScaleFirstChar[k][PriorEnd] += freq;
        }
      }
    } while (i-- != 0);
  } while (k-- != 0);
  NumBaseSymbols++;
  return;
}

void InitPriorEndBinary(uint8_t PriorEnd, uint8_t * code_length) {
  uint8_t freq;
  uint8_t FirstChar = 0xFF;
  do {
    if (code_length[FirstChar] < 8)
      freq = 1 << (8 - code_length[FirstChar]);
    else
      freq = 1;
    if ((RangeScaleFirstChar[0][FirstChar] != 0) || (FirstChar == PriorEnd)) {
      FirstCharData[0][PriorEnd][FirstChar].data.freq = freq;
      RangeScaleFirstChar[0][PriorEnd] += freq;
      if (FirstChar < 0xE0)
        RangeScaleFirstCharSection[PriorEnd][FirstChar >> 5] += freq;
    }
  } while (FirstChar-- != 0);
  return;
}

void InitBaseSymbolCap(uint8_t BaseSymbol, uint8_t * symbol_lengths) {
  InitFirstChar(BaseSymbol, symbol_lengths[BaseSymbol]);
  if ((BaseSymbol & 0xFE) == 0x42) {
    if (BaseSymbol == 'C')
      CapInitialized = 1;
    else
      CapLockInitialized = 1;
    if (CapInitialized + CapLockInitialized == 1)
      InitPriorEnd('C', symbol_lengths);
    else
      NumBaseSymbols++;
  } else
    InitPriorEnd(BaseSymbol, symbol_lengths);
  return;
}

void IncreaseRange(uint32_t low_ranges, uint32_t ranges) {
  low -= range * low_ranges;
  range *= ranges;
}

void DoubleRange(uint8_t low_ranges) {
  low -= range * low_ranges;
  range *= 2;
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

void EncodeDictTypeBinary(uint8_t Context1, uint8_t Context2, uint16_t QueueSize) {
  NormalizeEncoder(FREQ_SYM_TYPE_BOT1);
  uint32_t extra_range = range & (FREQ_SYM_TYPE_BOT1 - 1);
  if (QueueSize != 0)
    range = FreqSymTypePriorType[Context1][0] * (range >> 14) + extra_range;
  else
    range = (FREQ_SYM_TYPE_BOT1 - FreqSymTypePriorType[Context1][1]) * (range >> 14) + extra_range;
  uint16_t delta = FreqSymTypePriorType[Context1][1] >> 6;
  FreqSymTypePriorType[Context1][0] += delta + ((FREQ_SYM_TYPE_BOT1 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 6);
  FreqSymTypePriorType[Context1][1] -= delta;
  return;
}

void EncodeDictType(uint8_t Context1, uint8_t Context2, uint8_t Context3, uint16_t QueueSize) {
  NormalizeEncoder(8 * FREQ_SYM_TYPE_BOT3);
  uint32_t extra_range = range & (8 * FREQ_SYM_TYPE_BOT3 - 1);
  if (QueueSize != 0)
    range = (FreqSymTypePriorType[Context1][0] + FreqSymTypePriorType[Context2][0] + FreqSymTypePriorEnd[Context3][0]) * (range >> 15) + extra_range;
  else
    range = (0x8000 - FreqSymTypePriorType[Context1][1] - FreqSymTypePriorType[Context2][1] - FreqSymTypePriorEnd[Context3][1]) * (range >> 15) + extra_range;
  uint16_t delta = FreqSymTypePriorType[Context1][1] >> 4;
  FreqSymTypePriorType[Context1][0] += delta + ((0x2C00 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 4);
  FreqSymTypePriorType[Context1][1] -= delta;
  delta = FreqSymTypePriorType[Context2][1] >> 7;
  FreqSymTypePriorType[Context2][0] += delta + ((0x3400 - FreqSymTypePriorType[Context2][0] - FreqSymTypePriorType[Context2][1]) >> 7);
  FreqSymTypePriorType[Context2][1] -= delta;
  delta = FreqSymTypePriorEnd[Context3][1] >> 4;
  FreqSymTypePriorEnd[Context3][0] += delta + ((0x2000 - FreqSymTypePriorEnd[Context3][0] - FreqSymTypePriorEnd[Context3][1]) >> 4);
  FreqSymTypePriorEnd[Context3][1] -= delta;
  return;
}

void EncodeNewTypeBinary(uint8_t Context1, uint8_t Context2, uint16_t QueueSize) {
  NormalizeEncoder(FREQ_SYM_TYPE_BOT1);
  uint32_t extra_range = range & (FREQ_SYM_TYPE_BOT1 - 1);
  uint16_t FreqMtf = FREQ_SYM_TYPE_BOT1 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1];
  if (QueueSize != 0)
    low += FreqSymTypePriorType[Context1][0] * (range >>= 14) + extra_range;
  else
    low += (FreqSymTypePriorType[Context1][0] + FreqMtf) * (range >>= 14) + extra_range;
  range *= FreqSymTypePriorType[Context1][1];
  uint16_t delta = FreqSymTypePriorType[Context1][0] >> 6;
  FreqSymTypePriorType[Context1][1] += delta + (FreqMtf >> 6);
  FreqSymTypePriorType[Context1][0] -= delta;
  return;
}

void EncodeNewType(uint8_t Context1, uint8_t Context2, uint8_t Context3, uint16_t QueueSize) {
  NormalizeEncoder(8 * FREQ_SYM_TYPE_BOT3);
  uint32_t extra_range = range & (8 * FREQ_SYM_TYPE_BOT3 - 1);
  if (QueueSize != 0)
    low += (FreqSymTypePriorType[Context1][0] + FreqSymTypePriorType[Context2][0] + FreqSymTypePriorEnd[Context3][0]) * (range >>= 15) + extra_range;
  else
    low += (0x8000 - FreqSymTypePriorType[Context1][1] - FreqSymTypePriorType[Context2][1] - FreqSymTypePriorEnd[Context3][1]) * (range >>= 15) + extra_range;
  range *= FreqSymTypePriorType[Context1][1] + FreqSymTypePriorType[Context2][1] + FreqSymTypePriorEnd[Context3][1];
  uint16_t delta = FreqSymTypePriorType[Context1][0] >> 4;
  FreqSymTypePriorType[Context1][1] += delta + ((0x2C00 - (FreqSymTypePriorType[Context1][0] + FreqSymTypePriorType[Context1][1])) >> 4);
  FreqSymTypePriorType[Context1][0] -= delta;
  delta = FreqSymTypePriorType[Context2][0] >> 7;
  FreqSymTypePriorType[Context2][1] += delta + ((0x3400 - (FreqSymTypePriorType[Context2][0] + FreqSymTypePriorType[Context2][1])) >> 7);
  FreqSymTypePriorType[Context2][0] -= delta;
  delta = FreqSymTypePriorEnd[Context3][0] >> 4;
  FreqSymTypePriorEnd[Context3][1] += delta + ((0x2000 - (FreqSymTypePriorEnd[Context3][0] + FreqSymTypePriorEnd[Context3][1])) >> 4);
  FreqSymTypePriorEnd[Context3][0] -= delta;
  return;
}

void EncodeMtfTypeBinary(uint8_t Context1, uint8_t Context2) {
  NormalizeEncoder(FREQ_SYM_TYPE_BOT1);
  uint32_t extra_range = range & (FREQ_SYM_TYPE_BOT1 - 1);
  uint16_t delta = FreqSymTypePriorType[Context1][0] + FreqSymTypePriorType[Context1][1];
  low += delta * (range >>= 14) + extra_range;
  range *= FREQ_SYM_TYPE_BOT1 - delta;
  FreqSymTypePriorType[Context1][0] -= FreqSymTypePriorType[Context1][0] >> 6;
  FreqSymTypePriorType[Context1][1] -= FreqSymTypePriorType[Context1][1] >> 6;
  return;
}

void EncodeMtfType(uint8_t Context1, uint8_t Context2, uint8_t Context3) {
  NormalizeEncoder(8 * FREQ_SYM_TYPE_BOT3);
  uint32_t extra_range = range & (8 * FREQ_SYM_TYPE_BOT3 - 1);
  uint16_t delta = FreqSymTypePriorType[Context1][0] + FreqSymTypePriorType[Context1][1]
      + FreqSymTypePriorType[Context2][0] + FreqSymTypePriorType[Context2][1]
      + FreqSymTypePriorEnd[Context3][0] + FreqSymTypePriorEnd[Context3][1];
  low += delta * (range >>= 15) + extra_range;
  range *= 8 * FREQ_SYM_TYPE_BOT3 - delta;
  FreqSymTypePriorType[Context1][0] -= FreqSymTypePriorType[Context1][0] >> 4;
  FreqSymTypePriorType[Context1][1] -= FreqSymTypePriorType[Context1][1] >> 4;
  FreqSymTypePriorType[Context2][0] -= FreqSymTypePriorType[Context2][0] >> 7;
  FreqSymTypePriorType[Context2][1] -= FreqSymTypePriorType[Context2][1] >> 7;
  FreqSymTypePriorEnd[Context3][0] -= FreqSymTypePriorEnd[Context3][0] >> 4;
  FreqSymTypePriorEnd[Context3][1] -= FreqSymTypePriorEnd[Context3][1] >> 4;
  return;
}

void EncodeMtfFirst(uint8_t Context, uint8_t First, uint16_t QueueSizeOther, uint16_t QueueSizeSpace, uint16_t QueueSizeAz) {
  uint16_t delta;
  NormalizeEncoder(0x1000);
  if (First == 0) {
    if (QueueSizeSpace == 0) {
      if (QueueSizeAz != 0) {
        if (QueueSizeOther >= QueueSizeAz)
          range = (0x1000 - FreqMtfFirst[Context][2]) * (range >> 12);
        else
          range = FreqMtfFirst[Context][0] * (range >> 12);
      } else
        range = 0x1000 * (range >> 12);
    } else if ((QueueSizeAz == 0) && (QueueSizeOther >= QueueSizeSpace))
      range = (0x1000 - FreqMtfFirst[Context][1]) * (range >> 12);
    else
      range = FreqMtfFirst[Context][0] * (range >> 12);
    delta = FreqMtfFirst[Context][1] >> 7;
    FreqMtfFirst[Context][1] -= delta;
    FreqMtfFirst[Context][0] += delta;
    delta = FreqMtfFirst[Context][2] >> 7;
    FreqMtfFirst[Context][2] -= delta;
    FreqMtfFirst[Context][0] += delta;
  } else if (First == 1) {
    if (QueueSizeOther == 0) {
      if (QueueSizeAz != 0) {
        if (QueueSizeSpace >= QueueSizeAz)
          range = (0x1000 - FreqMtfFirst[Context][2]) * (range >> 12);
        else
          range = FreqMtfFirst[Context][1] * (range >> 12);
      }
      else
        range = 0x1000 * (range >> 12);
    } else if (QueueSizeAz == 0) {
      if (QueueSizeSpace > QueueSizeOther) {
        low += FreqMtfFirst[Context][0] * (range >>= 12);
        range *= 0x1000 - FreqMtfFirst[Context][0];
      } else {
        low += (0x1000 - FreqMtfFirst[Context][1]) * (range >>= 12);
        range *= FreqMtfFirst[Context][1];
      }
    } else {
      low += FreqMtfFirst[Context][0] * (range >>= 12);
      range *= FreqMtfFirst[Context][1];
    }
    delta = FreqMtfFirst[Context][0] >> 7;
    FreqMtfFirst[Context][0] -= delta;
    FreqMtfFirst[Context][1] += delta;
    delta = FreqMtfFirst[Context][2] >> 7;
    FreqMtfFirst[Context][2] -= delta;
    FreqMtfFirst[Context][1] += delta;
  } else {
    if (QueueSizeOther == 0) {
      if (QueueSizeSpace != 0) {
        if (QueueSizeAz > QueueSizeSpace) {
          low += FreqMtfFirst[Context][1] * (range >>= 12);
          range *= FreqMtfFirst[Context][0] + FreqMtfFirst[Context][2];
        } else {
          low += (0x1000 - FreqMtfFirst[Context][2]) * (range >>= 12);
          range *= FreqMtfFirst[Context][2];
        }
      }
      else
        range = 0x1000 * (range >> 12);
    } else if ((QueueSizeSpace == 0) && (QueueSizeAz > QueueSizeOther)) {
      low += FreqMtfFirst[Context][0] * (range >>= 12);
      range *= 0x1000 - FreqMtfFirst[Context][0];
    } else {
      low += (0x1000 - FreqMtfFirst[Context][2]) * (range >>= 12);
      range *= FreqMtfFirst[Context][2];
    }
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
  if (last_queue_size_other > QueueSize)
    unused_queue_freq_other += FreqMtfPos[0][--last_queue_size_other];
  else if (last_queue_size_other < QueueSize) {
    do {
      unused_queue_freq_other -= FreqMtfPos[0][last_queue_size_other++];
      if (last_queue_size_other > rescale_queue_size_other) {
        rescale_queue_size_other++;
        FreqMtfPos[0][last_queue_size_other - 1] += 8;
        RangeScaleMtfPos[0] += 8;
      } else {
        FreqMtfPos[0][last_queue_size_other - 1] += 2;
        RangeScaleMtfPos[0] += 2;
      }
    } while (last_queue_size_other != QueueSize);
  }
  if (RangeScaleMtfPos[0] > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(0);
  if (position == 0) {
    range = FreqMtfPos[0][0] * (range / (RangeScaleMtfPos[0] - unused_queue_freq_other));
    FreqMtfPos[0][0] += UP_FREQ_MTF_POS;
  } else {
    uint16_t * FreqPtr = &FreqMtfPos[0][0];
    uint16_t * StopFreqPtr = &FreqMtfPos[0][position];
    RangeLow = *FreqPtr++;

    while (FreqPtr != StopFreqPtr)
      RangeLow += *FreqPtr++;
    low += RangeLow * (range /= (RangeScaleMtfPos[0] - unused_queue_freq_other));
    range *= *FreqPtr;
    if (position >= 4) {
      if (position == 4) {
        *FreqPtr += UP_FREQ_MTF_POS - 1;
        *(FreqPtr + 1) += 1;
        if (position + 1 == QueueSize)
          unused_queue_freq_other += 1;
      } else if (position == 255) {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 1;
      } else {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 2;
        *(FreqPtr + 1) += 1;
        if (position + 1 == QueueSize)
          unused_queue_freq_other += 1;
      }
    } else
      *FreqPtr += UP_FREQ_MTF_POS;
  }
  RangeScaleMtfPos[0] += UP_FREQ_MTF_POS;
  return;
}


void EncodeMtfPosAz(uint8_t position, uint16_t QueueSize) {
  NormalizeEncoder(FREQ_MTF_POS_BOT);
  if (last_queue_size_az > QueueSize)
    unused_queue_freq_az += FreqMtfPos[2][--last_queue_size_az];
  else if (last_queue_size_az < QueueSize) {
    do {
      unused_queue_freq_az -= FreqMtfPos[2][last_queue_size_az++];
      if (last_queue_size_az > rescale_queue_size_az) {
        rescale_queue_size_az++;
        FreqMtfPos[2][last_queue_size_az - 1] += 16;
        RangeScaleMtfPos[2] += 16;
      } else {
        FreqMtfPos[2][last_queue_size_az - 1] += 4;
        RangeScaleMtfPos[2] += 4;
      }
    } while (last_queue_size_az != QueueSize);
  }
  if (RangeScaleMtfPos[2] > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(2);
  if (position == 0) {
    range = FreqMtfPos[2][0] * (range / (RangeScaleMtfPos[2] - unused_queue_freq_az));
    FreqMtfPos[2][0] += UP_FREQ_MTF_POS;
  } else {
    uint16_t * FreqPtr = &FreqMtfPos[2][0];
    uint16_t * StopFreqPtr = &FreqMtfPos[2][position];
    RangeLow = *FreqPtr++;
    while (FreqPtr != StopFreqPtr)
      RangeLow += *FreqPtr++;
    low += RangeLow * (range /= (RangeScaleMtfPos[2] - unused_queue_freq_az));
    range *= *FreqPtr;
    if (position >= 4) {
      if (position == 4) {
        *FreqPtr += UP_FREQ_MTF_POS - 1;
        *(FreqPtr + 1) += 1;
        if (position + 1 == QueueSize)
          unused_queue_freq_az += 1;
      } else if (position == 255) {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 1;
      } else {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 2;
        *(FreqPtr + 1) += 1;
        if (position + 1 == QueueSize)
          unused_queue_freq_az += 1;
      }
    } else
      *FreqPtr += UP_FREQ_MTF_POS;
  }
  RangeScaleMtfPos[2] += UP_FREQ_MTF_POS;
  return;
}

void EncodeMtfPosSpace(uint8_t position, uint16_t QueueSize) {
  NormalizeEncoder(FREQ_MTF_POS_BOT);
  if (last_queue_size_space > QueueSize)
    unused_queue_freq_space += FreqMtfPos[1][--last_queue_size_space];
  else if (last_queue_size_space < QueueSize) {
    do {
      unused_queue_freq_space -= FreqMtfPos[1][last_queue_size_space++];
      if (last_queue_size_space > rescale_queue_size_space) {
        rescale_queue_size_space++;
        FreqMtfPos[1][last_queue_size_space - 1] += 16;
        RangeScaleMtfPos[1] += 16;
      } else {
        FreqMtfPos[1][last_queue_size_space - 1] += 4;
        RangeScaleMtfPos[1] += 4;
      }
    } while (last_queue_size_space != QueueSize);
  }
  if (RangeScaleMtfPos[1] > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(1);
  if (position == 0) {
    range = FreqMtfPos[1][0] * (range / (RangeScaleMtfPos[1] - unused_queue_freq_space));
    FreqMtfPos[1][0] += UP_FREQ_MTF_POS;
  } else {
    uint16_t * FreqPtr = &FreqMtfPos[1][0];
    uint16_t * StopFreqPtr = &FreqMtfPos[1][position];
    RangeLow = *FreqPtr++;
    while (FreqPtr != StopFreqPtr)
      RangeLow += *FreqPtr++;
    low += RangeLow * (range /= (RangeScaleMtfPos[1] - unused_queue_freq_space));
    range *= *FreqPtr;
    if (position >= 4) {
      if (position == 4) {
        *FreqPtr += UP_FREQ_MTF_POS - 1;
        *(FreqPtr + 1) += 1;
        if (position + 1 == QueueSize)
          unused_queue_freq_space += 1;
      } else if (position == 255) {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 1;
      } else {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 2;
        *(FreqPtr + 1) += 1;
        if (position + 1 == QueueSize)
          unused_queue_freq_space += 1;
      }
    } else
      *FreqPtr += UP_FREQ_MTF_POS;
  }
  RangeScaleMtfPos[1] += UP_FREQ_MTF_POS;
  return;
}

void EncodeMtfPosOther(uint8_t position, uint16_t QueueSize) {
  NormalizeEncoder(FREQ_MTF_POS_BOT);
  if (last_queue_size_other > QueueSize)
    unused_queue_freq_other += FreqMtfPos[0][--last_queue_size_other];
  else if (last_queue_size_other < QueueSize) {
    do {
      unused_queue_freq_other -= FreqMtfPos[0][last_queue_size_other++];
      if (last_queue_size_other > rescale_queue_size_other) {
        rescale_queue_size_other++;
        FreqMtfPos[0][last_queue_size_other - 1] += 16;
        RangeScaleMtfPos[0] += 16;
      } else {
        FreqMtfPos[0][last_queue_size_other - 1] += 4;
        RangeScaleMtfPos[0] += 4;
      }
    } while (last_queue_size_other != QueueSize);
  }
  if (RangeScaleMtfPos[0] > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(0);
  if (position == 0) {
    range = FreqMtfPos[0][0] * (range / (RangeScaleMtfPos[0] - unused_queue_freq_other));
    FreqMtfPos[0][0] += UP_FREQ_MTF_POS;
  } else {
    uint16_t * FreqPtr = &FreqMtfPos[0][0];
    uint16_t * StopFreqPtr = &FreqMtfPos[0][position];
    RangeLow = *FreqPtr++;
    while (FreqPtr != StopFreqPtr)
      RangeLow += *FreqPtr++;
    low += RangeLow * (range /= (RangeScaleMtfPos[0] - unused_queue_freq_other));
    range *= *FreqPtr;
    if (position >= 4) {
      if (position == 4) {
        *FreqPtr += UP_FREQ_MTF_POS - 1;
        *(FreqPtr + 1) += 1;
        if (position + 1 == QueueSize)
          unused_queue_freq_other += 1;
      } else if (position == 255) {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 1;
      } else {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 2;
        *(FreqPtr + 1) += 1;
        if (position + 1 == QueueSize)
          unused_queue_freq_other += 1;
      }
    } else
      *FreqPtr += UP_FREQ_MTF_POS;
  }
  RangeScaleMtfPos[0] += UP_FREQ_MTF_POS;
}

void EncodeSID(uint8_t Context, uint8_t SIDSymbol) {
  NormalizeEncoder(FREQ_SID_BOT);
  if (SIDSymbol == 0) {
    range = FreqSID[Context][0] * (range / RangeScaleSID[Context]);
    FreqSID[Context][0] += UP_FREQ_SID;
  } else {
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

void EncodeExtraSID(uint32_t ExtraSymbols) {
  int64_t code;
  uint32_t j;
  uint8_t range_multiplier;
  uint8_t bits = 9;
  if (ExtraSymbols <= 1) {
    code = ExtraSymbols << 6;
    range_multiplier = 0x40;
  } else if (ExtraSymbols <= 5) {
    code = (ExtraSymbols + 2) << 5;
    range_multiplier = 0x20;
  } else if (ExtraSymbols <= 0xD) {
    code = 0x80 + ((ExtraSymbols + 2) << 4);
    range_multiplier = 0x10;
  } else if (ExtraSymbols <= 0x1D) {
    code = 0x140 + ((ExtraSymbols + 2) << 2);
    range_multiplier = 4;
  } else {
    int64_t top, bottom;
    top = 0x200; bottom = 0x20;
    while (ExtraSymbols + 2 >= bottom * 2) {
      top *= 4;
      bottom *= 2;
      bits += 2;
    }
    code = (top - 3 * bottom + ExtraSymbols + 2) << (6 - ((bits - 3) & 7));
    range_multiplier =  1 << (6 - ((bits - 3) & 7));
    bits += 6 - ((bits - 3) & 7);
  }
  NormalizeEncoder(1 << 9);
  uint16_t count = (code >> (bits - 9)) & 0x1FF;
  range >>= 9;
  low += range * count;
  bits -= 9;
  while (bits != 0) {
    NormalizeEncoder(1 << 8);
    count = (code >> (bits - 8)) & 0xFF;
    range >>= 8;
    low += range * count;
    bits -= 8;
  }
  low -= range * (count & (range_multiplier - 1));
  range *= range_multiplier;
  return;
}

void EncodeINST(uint8_t Context, uint8_t SIDSymbol, uint8_t Symbol) {
  NormalizeEncoder(FREQ_INST_BOT);
  uint32_t extra_range = range;
  range /= RangeScaleINST[Context][SIDSymbol];
  extra_range -= range * RangeScaleINST[Context][SIDSymbol];
  if (Symbol == 0) {
    range = range * FreqINST[Context][SIDSymbol][0] + extra_range;
    if (RangeScaleINST[Context][SIDSymbol] >= (FREQ_INST_BOT >> 1)) {
      FreqINST[Context][SIDSymbol][0] += RangeScaleINST[Context][SIDSymbol] >> 11;
      if ((RangeScaleINST[Context][SIDSymbol] += (RangeScaleINST[Context][SIDSymbol]) >> 11) > FREQ_INST_BOT)
        rescaleINST(Context, SIDSymbol);
    } else {
      FreqINST[Context][SIDSymbol][0] += UP_FREQ_INST;
      RangeScaleINST[Context][SIDSymbol] += UP_FREQ_INST;
    }
  } else {
    RangeLow = FreqINST[Context][SIDSymbol][0];
    uint8_t FoundIndex = 1;
    while (FoundIndex != Symbol)
      RangeLow += FreqINST[Context][SIDSymbol][FoundIndex++];
    low += range * RangeLow + extra_range;
    range *= FreqINST[Context][SIDSymbol][FoundIndex];
    if (RangeScaleINST[Context][SIDSymbol] >= (FREQ_INST_BOT >> 1)) {
      FreqINST[Context][SIDSymbol][FoundIndex] += RangeScaleINST[Context][SIDSymbol] >> 11;
      if ((RangeScaleINST[Context][SIDSymbol] += (RangeScaleINST[Context][SIDSymbol]) >> 11) > FREQ_INST_BOT)
        rescaleINST(Context, SIDSymbol);
    } else {
      FreqINST[Context][SIDSymbol][FoundIndex] += UP_FREQ_INST;
      RangeScaleINST[Context][SIDSymbol] += UP_FREQ_INST;
    }
  }
  return;
}

void EncodeERG(uint16_t Context1, uint16_t Context2, uint8_t Symbol) {
  NormalizeEncoder(FREQ_ERG_BOT);
  if (Symbol == 0) {
    range = (FreqERG[0] + FreqERG[Context1] + FreqERG[Context2]) * (range >> 13);
    FreqERG[0] += (0x400 - FreqERG[0]) >> 2;
    FreqERG[Context1] += (0x1000 - FreqERG[Context1]) >> 4;
    FreqERG[Context2] += (0xC00 - FreqERG[Context2]) >> 3;
  } else {
    low += (FreqERG[0] + FreqERG[Context1] + FreqERG[Context2]) * (range >>= 13);
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
  } else {
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
  if (Symbol == 0) {
    range = FreqWordTag[Context] * (range >> 12);
    FreqWordTag[Context] += (0x1000 - FreqWordTag[Context]) >> 4;
  } else {
    low += FreqWordTag[Context] * (range >>= 12);
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

void EncodeFirstChar(uint8_t FirstChar, uint8_t SymType, uint8_t LastChar) {
  uint16_t *RangeScalePtr = &RangeScaleFirstChar[SymType][LastChar];
  uint32_t extra_range;
  struct first_char_data * FCDataPtr = &FirstCharData[SymType][LastChar][0];

  NormalizeEncoder((uint32_t)FREQ_FIRST_CHAR_BOT);
  extra_range = range;
  RangeLow = 0;
  range /= *RangeScalePtr;
  extra_range -= range * *RangeScalePtr;
  while(FCDataPtr->data.symbol != FirstChar) {
    RangeLow += FCDataPtr++->data.freq;
  }
  low += RangeLow * range;
  range *= FCDataPtr->data.freq;
  if (FCDataPtr == &FirstCharData[SymType][LastChar][0])
    range += extra_range;
  else
    low += extra_range;

  if (*RangeScalePtr >= (FREQ_FIRST_CHAR_BOT >> 2)) {
    FCDataPtr->data.freq += *RangeScalePtr >> 10;
    if ((*RangeScalePtr += (*RangeScalePtr >> 10)) >= FREQ_FIRST_CHAR_BOT)
      rescaleFirstChar(SymType, LastChar);
  } else {
    FCDataPtr->data.freq += UP_FREQ_FIRST_CHAR;
    *RangeScalePtr += UP_FREQ_FIRST_CHAR;
  }

  if (FCDataPtr != &FirstCharData[SymType][LastChar][0]) {
    if (FCDataPtr->data.freq > (FCDataPtr - 1)->data.freq) {
      struct first_char_data SavedData;
      SavedData.all_data = FCDataPtr->all_data;
      do {
        FCDataPtr->all_data = (FCDataPtr - 1)->all_data;
        FCDataPtr--;
      } while ((FCDataPtr != &FirstCharData[SymType][LastChar][0]) && (SavedData.data.freq > (FCDataPtr - 1)->data.freq));
      FCDataPtr->all_data = SavedData.all_data;
    }
  }
  return;
}

void EncodeFirstCharBinary(uint8_t FirstChar, uint8_t LastChar) {
  uint8_t SectionIndex = 0;
  uint16_t *RangeScalePtr = &RangeScaleFirstChar[0][LastChar];
  uint32_t extra_range;
  struct first_char_data * FCDataPtr;

  NormalizeEncoder(FREQ_FIRST_CHAR_BOT);
  extra_range = range;
  RangeLow = 0;
  while ((SectionIndex != 7) && (FirstChar >= 0x20 * (SectionIndex + 1))) {
    RangeLow += RangeScaleFirstCharSection[LastChar][SectionIndex];
    SectionIndex++;
  }
  FCDataPtr = &FirstCharData[0][LastChar][FirstChar & 0xE0];
  while (FCDataPtr != &FirstCharData[0][LastChar][FirstChar]) {
    RangeLow += FCDataPtr++->data.freq;
  }
  range /= (uint32_t)*RangeScalePtr;
  extra_range -= range * (uint32_t)*RangeScalePtr;
  low += (uint32_t)RangeLow * range;
  range *= (uint32_t)FCDataPtr->data.freq;
  if (FCDataPtr == &FirstCharData[0][LastChar][0])
    range += extra_range;
  else
    low += extra_range;

  if (*RangeScalePtr >= (FREQ_FIRST_CHAR_BOT >> 2)) {
    FCDataPtr->data.freq += *RangeScalePtr >> 10;
    if (SectionIndex <= 6)
      RangeScaleFirstCharSection[LastChar][SectionIndex] += *RangeScalePtr >> 10;
    if ((*RangeScalePtr += *RangeScalePtr >> 10) >= FREQ_FIRST_CHAR_BOT)
      rescaleFirstCharBinary(LastChar);
  } else {
    FCDataPtr->data.freq += UP_FREQ_FIRST_CHAR >> 1;
    if (SectionIndex <= 6)
      RangeScaleFirstCharSection[LastChar][SectionIndex] += UP_FREQ_FIRST_CHAR >> 1;
    *RangeScalePtr += UP_FREQ_FIRST_CHAR >> 1;
  }
  return;
}

void WriteInCharNum(uint32_t value) {
  InCharNum = value;
}

uint32_t ReadOutCharNum() {
  return(OutCharNum);
}

void InitEncoder(uint8_t max_base_code, uint8_t num_inst_codes, uint8_t cap_encoded, uint8_t UTF8_compliant,
    uint8_t use_mtf, uint8_t * bufptr) {
  uint8_t i, j;
  CapInitialized = 0;
  CapLockInitialized = 0;
  CapEncoded = cap_encoded;
  UTF8Compliant = UTF8_compliant;
  MaxBaseCode = max_base_code;
  MaxInstCode = num_inst_codes - 1;
  OutBuffer = bufptr;
  OutCharNum = 0;
  low = 0, range = -1;
  StartModelSymType(use_mtf, cap_encoded);
  StartModelMtfFirst();
  StartModelMtfPos();
  StartModelSID();
  StartModelINST(num_inst_codes);
  StartModelERG();
  StartModelGoMtf();
  StartModelWordTag();
  if (cap_encoded || UTF8_compliant)
    StartModelFirstChar();
  else
    StartModelFirstCharBinary();

  if (UTF8_compliant != 0) {
    i = 0x90;
    j = 0x90;
  } else {
    i = 0xFF;
    j = 0xFF;
  }
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

uint8_t DecodeSymTypeBinary(uint8_t Context1, uint8_t Context2, uint16_t QueueSize) {
  uint32_t DictRange;
  NormalizeDecoder(FREQ_SYM_TYPE_BOT1);
  uint32_t extra_range = range & (FREQ_SYM_TYPE_BOT1 - 1);
  if (QueueSize != 0) {
    if ((DictRange = (range >>= 14) * FreqSymTypePriorType[Context1][0] + extra_range) > code - low) {
      range = DictRange;
      uint16_t delta = FreqSymTypePriorType[Context1][1] >> 6;
      FreqSymTypePriorType[Context1][0] += delta + ((FREQ_SYM_TYPE_BOT1 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 6);
      FreqSymTypePriorType[Context1][1] -= delta;
      return(0);
    } else if (DictRange + range * FreqSymTypePriorType[Context1][1] > code - low) {
      low += DictRange;
      range *= FreqSymTypePriorType[Context1][1];
      uint16_t delta = FreqSymTypePriorType[Context1][0] >> 6;
      FreqSymTypePriorType[Context1][1] += delta + ((FREQ_SYM_TYPE_BOT1 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 6);
      FreqSymTypePriorType[Context1][0] -= delta;
      return(1);
    } else {
      low += DictRange + range * FreqSymTypePriorType[Context1][1];
      range *= FREQ_SYM_TYPE_BOT1 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1];
      FreqSymTypePriorType[Context1][0] -= FreqSymTypePriorType[Context1][0] >> 6;
      FreqSymTypePriorType[Context1][1] -= FreqSymTypePriorType[Context1][1] >> 6;
      return(2);
    }
  } else {
    DictRange = (range >>= 14) * (FREQ_SYM_TYPE_BOT1 - FreqSymTypePriorType[Context1][1]) + extra_range;
    if (DictRange > code - low) {
      range = DictRange;
      uint16_t delta = FreqSymTypePriorType[Context1][1] >> 6;
      FreqSymTypePriorType[Context1][0] += delta + ((FREQ_SYM_TYPE_BOT1 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 6);
      FreqSymTypePriorType[Context1][1] -= delta;
      return(0);
    } else {
      low += DictRange;
      range *= FreqSymTypePriorType[Context1][1];
      uint16_t delta = FreqSymTypePriorType[Context1][0] >> 6;
      FreqSymTypePriorType[Context1][1] += delta + ((FREQ_SYM_TYPE_BOT1 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 6);
      FreqSymTypePriorType[Context1][0] -= delta;
      return(1);
    }
  }
}

uint8_t DecodeSymType(uint8_t Context1, uint8_t Context2, uint8_t Context3, uint16_t QueueSize) {
  uint32_t DictRange;
  NormalizeDecoder(8 * FREQ_SYM_TYPE_BOT3);
  uint32_t extra_range = range & (8 * FREQ_SYM_TYPE_BOT3 - 1);
  if (QueueSize != 0) {
    if ((DictRange = (FreqSymTypePriorType[Context1][0] + FreqSymTypePriorType[Context2][0]
        + FreqSymTypePriorEnd[Context3][0]) * (range >>= 15) + extra_range) > code - low) {
      range = DictRange;
      uint16_t delta = FreqSymTypePriorType[Context1][1] >> 4;
      FreqSymTypePriorType[Context1][0] += delta + ((0x2C00 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 4);
      FreqSymTypePriorType[Context1][1] -= delta;
      delta = FreqSymTypePriorType[Context2][1] >> 7;
      FreqSymTypePriorType[Context2][0] += delta + ((0x3400 - FreqSymTypePriorType[Context2][0] - FreqSymTypePriorType[Context2][1]) >> 7);
      FreqSymTypePriorType[Context2][1] -= delta;
      delta = FreqSymTypePriorEnd[Context3][1] >> 4;
      FreqSymTypePriorEnd[Context3][0] += delta + ((0x2000 - FreqSymTypePriorEnd[Context3][0] - FreqSymTypePriorEnd[Context3][1]) >> 4);
      FreqSymTypePriorEnd[Context3][1] -= delta;
      return(0);
    } else if (DictRange + range * (FreqSymTypePriorType[Context1][1] + FreqSymTypePriorType[Context2][1]
        + FreqSymTypePriorEnd[Context3][1]) > code - low) {
      low += DictRange;
      range *= FreqSymTypePriorType[Context1][1] + FreqSymTypePriorType[Context2][1] + FreqSymTypePriorEnd[Context3][1];
      uint16_t delta = FreqSymTypePriorType[Context1][0] >> 4;
      FreqSymTypePriorType[Context1][1] += delta + ((0x2C00 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 4);
      FreqSymTypePriorType[Context1][0] -= delta;
      delta = FreqSymTypePriorType[Context2][0] >> 7;
      FreqSymTypePriorType[Context2][1] += delta + ((0x3400 - FreqSymTypePriorType[Context2][0] - FreqSymTypePriorType[Context2][1]) >> 7);
      FreqSymTypePriorType[Context2][0] -= delta;
      delta = FreqSymTypePriorEnd[Context3][0] >> 4;
      FreqSymTypePriorEnd[Context3][1] += delta + ((0x2000 - FreqSymTypePriorEnd[Context3][0] - FreqSymTypePriorEnd[Context3][1]) >> 4);
      FreqSymTypePriorEnd[Context3][0] -= delta;
      return(1);
    } else {
      low += DictRange + range * (FreqSymTypePriorType[Context1][1] + FreqSymTypePriorType[Context2][1] + FreqSymTypePriorEnd[Context3][1]);
      range *= 0x8000 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1] - FreqSymTypePriorType[Context2][0]
          - FreqSymTypePriorType[Context2][1] - FreqSymTypePriorEnd[Context3][0] - FreqSymTypePriorEnd[Context3][1];
      FreqSymTypePriorType[Context1][0] -= FreqSymTypePriorType[Context1][0] >> 4;
      FreqSymTypePriorType[Context1][1] -= FreqSymTypePriorType[Context1][1] >> 4;
      FreqSymTypePriorType[Context2][0] -= FreqSymTypePriorType[Context2][0] >> 7;
      FreqSymTypePriorType[Context2][1] -= FreqSymTypePriorType[Context2][1] >> 7;
      FreqSymTypePriorEnd[Context3][0] -= FreqSymTypePriorEnd[Context3][0] >> 4;
      FreqSymTypePriorEnd[Context3][1] -= FreqSymTypePriorEnd[Context3][1] >> 4;
      return(2);
    }
  } else {
    if ((DictRange = (0x8000 - FreqSymTypePriorType[Context1][1] - FreqSymTypePriorType[Context2][1] - FreqSymTypePriorEnd[Context3][1])
        * (range >>= 15) + extra_range) > code - low) {
      range = DictRange;
      uint16_t delta = FreqSymTypePriorType[Context1][1] >> 4;
      FreqSymTypePriorType[Context1][0] += delta + ((0x2C00 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 4);
      FreqSymTypePriorType[Context1][1] -= delta;
      delta = FreqSymTypePriorType[Context2][1] >> 7;
      FreqSymTypePriorType[Context2][0] += delta + ((0x3400 - FreqSymTypePriorType[Context2][0] - FreqSymTypePriorType[Context2][1]) >> 7);
      FreqSymTypePriorType[Context2][1] -= delta;
      delta = FreqSymTypePriorEnd[Context3][1] >> 4;
      FreqSymTypePriorEnd[Context3][0] += delta + ((0x2000 - FreqSymTypePriorEnd[Context3][0] - FreqSymTypePriorEnd[Context3][1]) >> 4);
      FreqSymTypePriorEnd[Context3][1] -= delta;
      return(0);
    } else {
      low += DictRange;
      range *= FreqSymTypePriorType[Context1][1] + FreqSymTypePriorType[Context2][1] + FreqSymTypePriorEnd[Context3][1];
      uint16_t delta = FreqSymTypePriorType[Context1][0] >> 4;
      FreqSymTypePriorType[Context1][1] += delta + ((0x2C00 - FreqSymTypePriorType[Context1][0] - FreqSymTypePriorType[Context1][1]) >> 4);
      FreqSymTypePriorType[Context1][0] -= delta;
      delta = FreqSymTypePriorType[Context2][0] >> 7;
      FreqSymTypePriorType[Context2][1] += delta + ((0x3400 - FreqSymTypePriorType[Context2][0] - FreqSymTypePriorType[Context2][1]) >> 7);
      FreqSymTypePriorType[Context2][0] -= delta;
      delta = FreqSymTypePriorEnd[Context3][0] >> 4;
      FreqSymTypePriorEnd[Context3][1] += delta + ((0x2000 - FreqSymTypePriorEnd[Context3][0] - FreqSymTypePriorEnd[Context3][1]) >> 4);
      FreqSymTypePriorEnd[Context3][0] -= delta;
      return(1);
    }
  }
}

uint8_t DecodeMtfFirst(uint8_t Context, uint16_t QueueSizeOther, uint16_t QueueSizeSpace, uint16_t QueueSizeAz) {
  uint16_t delta;
  uint16_t Freq0, Freq1;
  NormalizeDecoder(0x1000);
  if (QueueSizeOther == 0) {
    Freq0 = 0;
    if (QueueSizeSpace == 0)
      Freq1 = 0;
    else if (QueueSizeAz == 0)
      Freq1 = 0x1000;
    else if (QueueSizeAz > QueueSizeSpace)
      Freq1 = FreqMtfFirst[Context][1];
    else
      Freq1 = FreqMtfFirst[Context][0] + FreqMtfFirst[Context][1];
  } else if (QueueSizeSpace == 0) {
    Freq1 = 0;
    if (QueueSizeAz == 0)
      Freq0 = 0x1000;
    else if (QueueSizeAz > QueueSizeOther)
      Freq0 = FreqMtfFirst[Context][0];
    else
      Freq0 = FreqMtfFirst[Context][0] + FreqMtfFirst[Context][1];
  } else if (QueueSizeAz == 0) {
    if (QueueSizeSpace > QueueSizeOther) {
      Freq0 = FreqMtfFirst[Context][0];
      Freq1 = 0x1000 - FreqMtfFirst[Context][0];
    } else {
      Freq0 = 0x1000 - FreqMtfFirst[Context][1];
      Freq1 = FreqMtfFirst[Context][1];
    }
  } else {
    Freq0 = FreqMtfFirst[Context][0];
    Freq1 = FreqMtfFirst[Context][1];
  }

  if (Freq0 * (range >>= 12) > code - low) {
    range *= Freq0;
    delta = FreqMtfFirst[Context][1] >> 7;
    FreqMtfFirst[Context][1] -= delta;
    FreqMtfFirst[Context][0] += delta;
    delta = FreqMtfFirst[Context][2] >> 7;
    FreqMtfFirst[Context][2] -= delta;
    FreqMtfFirst[Context][0] += delta;
    return(0);
  } else if ((Freq0 + Freq1) * range > code - low) {
    low += range * Freq0;
    range *= Freq1;
    delta = FreqMtfFirst[Context][0] >> 7;
    FreqMtfFirst[Context][0] -= delta;
    FreqMtfFirst[Context][1] += delta;
    delta = FreqMtfFirst[Context][2] >> 7;
    FreqMtfFirst[Context][2] -= delta;
    FreqMtfFirst[Context][1] += delta;
    return(1);
  } else {
    low += range * (Freq0 + Freq1);
    range *= 0x1000 - Freq0 - Freq1;
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
  if (last_queue_size_other > QueueSize)
    unused_queue_freq_other += FreqMtfPos[0][--last_queue_size_other];
  else if (last_queue_size_other < QueueSize) {
    do {
      unused_queue_freq_other -= FreqMtfPos[0][last_queue_size_other++];
      if (last_queue_size_other > rescale_queue_size_other) {
        rescale_queue_size_other++;
        FreqMtfPos[0][last_queue_size_other - 1] += 8;
        RangeScaleMtfPos[0] += 8;
      } else {
        FreqMtfPos[0][last_queue_size_other - 1] += 2;
        RangeScaleMtfPos[0] += 2;
      }
    } while (last_queue_size_other != QueueSize);
  }
  if (RangeScaleMtfPos[0] > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(0);
  count = (code - low) / (range /= (RangeScaleMtfPos[0] - unused_queue_freq_other));
  if ((RangeHigh = FreqMtfPos[0][0]) > count) {
    range *= RangeHigh;
    FreqMtfPos[0][0] += UP_FREQ_MTF_POS;
    RangeScaleMtfPos[0] += UP_FREQ_MTF_POS;
    return(0);
  } else {
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
          unused_queue_freq_other += 1;
      } else if (position == 255) {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 1;
      } else {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 2;
        *(FreqPtr + 1) += 1;
        if (position == QueueSize - 1)
          unused_queue_freq_other += 1;
      }
    } else
      *FreqPtr += UP_FREQ_MTF_POS;
    RangeScaleMtfPos[0] += UP_FREQ_MTF_POS;
    return(position);
  }
}

uint8_t DecodeMtfPosAz(uint16_t QueueSize) {
  NormalizeDecoder(FREQ_MTF_POS_BOT);
  if (last_queue_size_az > QueueSize)
    unused_queue_freq_az += FreqMtfPos[2][--last_queue_size_az];
  else if (last_queue_size_az < QueueSize) {
    do {
      unused_queue_freq_az -= FreqMtfPos[2][last_queue_size_az++];
      if (last_queue_size_az > rescale_queue_size_az) {
        rescale_queue_size_az++;
        FreqMtfPos[2][last_queue_size_az - 1] += 16;
        RangeScaleMtfPos[2] += 16;
      } else {
        FreqMtfPos[2][last_queue_size_az - 1] += 4;
        RangeScaleMtfPos[2] += 4;
      }
    } while (last_queue_size_az != QueueSize);
  }
  if (RangeScaleMtfPos[2] > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(2);
  count = code - low;
  range /= RangeScaleMtfPos[2] - unused_queue_freq_az;
  if ((RangeHigh = range * FreqMtfPos[2][0]) > count) {
    range *= FreqMtfPos[2][0];
    FreqMtfPos[2][0] += UP_FREQ_MTF_POS;
    RangeScaleMtfPos[2] += UP_FREQ_MTF_POS;
    return(0);
  } else {
    uint16_t * FreqPtr = &FreqMtfPos[2][1];
    while ((RangeHigh += range * *FreqPtr) <= count)
      FreqPtr++;
    uint8_t position = FreqPtr - &FreqMtfPos[2][0];
    low += RangeHigh - range * *FreqPtr;
    range *= *FreqPtr;
    if (position >= 4) {
      if (position == 4) {
        *FreqPtr += UP_FREQ_MTF_POS - 1;
        *(FreqPtr + 1) += 1;
        if (position == QueueSize - 1)
          unused_queue_freq_az += 1;
      } else if (position == 255) {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 1;
      } else {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 2;
        *(FreqPtr + 1) += 1;
        if (position == QueueSize - 1)
          unused_queue_freq_az += 1;
      }
    } else
      *FreqPtr += UP_FREQ_MTF_POS;
    RangeScaleMtfPos[2] += UP_FREQ_MTF_POS;
    return(position);
  }
}

uint8_t DecodeMtfPosSpace(uint16_t QueueSize) {
  NormalizeDecoder(FREQ_MTF_POS_BOT);
  if (last_queue_size_space > QueueSize)
    unused_queue_freq_space += FreqMtfPos[1][--last_queue_size_space];
  else if (last_queue_size_space < QueueSize) {
    do {
      unused_queue_freq_space -= FreqMtfPos[1][last_queue_size_space++];
      if (last_queue_size_space > rescale_queue_size_space) {
        rescale_queue_size_space++;
        FreqMtfPos[1][last_queue_size_space - 1] += 16;
        RangeScaleMtfPos[1] += 16;
      } else {
        FreqMtfPos[1][last_queue_size_space - 1] += 4;
        RangeScaleMtfPos[1] += 4;
      }
    } while (last_queue_size_space != QueueSize);
  }
  if (RangeScaleMtfPos[1] > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(1);
  count = code - low;
  range /= RangeScaleMtfPos[1] - unused_queue_freq_space;
  if ((RangeHigh = range * FreqMtfPos[1][0]) > count) {
    range *= FreqMtfPos[1][0];
    FreqMtfPos[1][0] += UP_FREQ_MTF_POS;
    RangeScaleMtfPos[1] += UP_FREQ_MTF_POS;
    return(0);
  } else {
    uint16_t * FreqPtr = &FreqMtfPos[1][1];
    while ((RangeHigh += range * *FreqPtr) <= count)
      FreqPtr++;
    uint8_t position = FreqPtr - &FreqMtfPos[1][0];
    low += RangeHigh - range * *FreqPtr;
    range *= *FreqPtr;

    if (position >= 4) {
      if (position == 4) {
        *FreqPtr += UP_FREQ_MTF_POS - 1;
        *(FreqPtr + 1) += 1;
        if (position == QueueSize - 1)
          unused_queue_freq_space += 1;
      } else if (position == 255) {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 1;
      } else {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 2;
        *(FreqPtr + 1) += 1;
        if (position == QueueSize - 1)
          unused_queue_freq_space += 1;
      }
    } else
      *FreqPtr += UP_FREQ_MTF_POS;
    RangeScaleMtfPos[1] += UP_FREQ_MTF_POS;
    return(position);
  }
}

uint8_t DecodeMtfPosOther(uint16_t QueueSize) {
  NormalizeDecoder(FREQ_MTF_POS_BOT);
  if (last_queue_size_other > QueueSize)
    unused_queue_freq_other += FreqMtfPos[0][--last_queue_size_other];
  else if (last_queue_size_other < QueueSize) {
    do {
      unused_queue_freq_other -= FreqMtfPos[0][last_queue_size_other++];
      if (last_queue_size_other > rescale_queue_size_other) {
        rescale_queue_size_other++;
        FreqMtfPos[0][last_queue_size_other - 1] += 16;
        RangeScaleMtfPos[0] += 16;
      } else {
        FreqMtfPos[0][last_queue_size_other - 1] += 4;
        RangeScaleMtfPos[0] += 4;
      }
    } while (last_queue_size_other != QueueSize);
  }
  if (RangeScaleMtfPos[0] > FREQ_MTF_POS_BOT)
    rescaleMtfQueuePos(0);
  count = code - low;
  range /= RangeScaleMtfPos[0] - unused_queue_freq_other;
  if ((RangeHigh = range * FreqMtfPos[0][0]) > count) {
    range *= FreqMtfPos[0][0];
    FreqMtfPos[0][0] += UP_FREQ_MTF_POS;
    RangeScaleMtfPos[0] += UP_FREQ_MTF_POS;
    return(0);
  } else {
    uint16_t * FreqPtr = &FreqMtfPos[0][1];
    while ((RangeHigh += range * *FreqPtr) <= count)
      FreqPtr++;
    uint8_t position = FreqPtr - &FreqMtfPos[0][0];
    low += RangeHigh - range * *FreqPtr;
    range *= *FreqPtr;
    if (position >= 4) {
      if (position == 4) {
        *FreqPtr += UP_FREQ_MTF_POS - 1;
        *(FreqPtr + 1) += 1;
        if (position == QueueSize - 1)
          unused_queue_freq_other += 1;
      } else if (position == 255) {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 1;
      } else {
        *(FreqPtr - 1) += 1;
        *FreqPtr += UP_FREQ_MTF_POS - 2;
        *(FreqPtr + 1) += 1;
        if (position == QueueSize - 1)
          unused_queue_freq_other += 1;
      }
    } else
      *FreqPtr += UP_FREQ_MTF_POS;
    RangeScaleMtfPos[0] += UP_FREQ_MTF_POS;
    return(position);
  }
}

uint8_t DecodeSID(uint8_t Context) {
  NormalizeDecoder(FREQ_SID_BOT);
  range /= RangeScaleSID[Context];
  count = code - low;
  if ((RangeHigh = range * FreqSID[Context][0]) > count) {
    range = RangeHigh;
    FreqSID[Context][0] += UP_FREQ_SID;
    if ((RangeScaleSID[Context] += UP_FREQ_SID) > FREQ_SID_BOT)
      rescaleSID(Context);
    return(0);
  } else {
    uint32_t temp;
    uint8_t SIDSymbol = 0;
    while ((temp = RangeHigh + range * FreqSID[Context][++SIDSymbol]) <= count)
      RangeHigh = temp;
    low += RangeHigh;
    range *= FreqSID[Context][SIDSymbol];
    FreqSID[Context][SIDSymbol] += UP_FREQ_SID;
    if ((RangeScaleSID[Context] += UP_FREQ_SID) > FREQ_SID_BOT)
      rescaleSID(Context);
    return(SIDSymbol);
  }
}


uint32_t DecodeExtraSID() {
  uint32_t ExtraSID;
  NormalizeDecoder((uint32_t)1 << 9);
  uint16_t input = (code - low) / (range >>= 9);
  if (input < 0x80) {
    ExtraSID = input >> 6;
    low += range * (input & 0x40);
    range *= 0x40;
  } else if (input < 0x100) {
    ExtraSID = (input >> 5) - 2;
    low += range * (input & 0xE0);
    range *= 0x20;
  } else if (input < 0x180) {
    ExtraSID = (input >> 4) - 0xA;
    low += range * (input & 0x1F0);
    range *= 0x10;
  } else if (input < 0x1C0) {
    ExtraSID = (input >> 2) - 0x52;
    low += range * (input & 0x1FC);
    range *= 4;
  } else if (input < 0x1E0) {
    ExtraSID = input - 0x1A2;
    low += range * input;
  } else {
    low += range * input;
    uint64_t input_code, limit, mask;
    uint8_t count, j;
    j = 0;
    input_code = input;
    do {
      j += 4;
      NormalizeDecoder((uint32_t)1 << 8);
      count = (code - low) / (range >>= 8);
      low += range * count;
      input_code = (input_code << 8) + count;
      limit = ((uint64_t)1 << (4 + 2 * j)) - ((uint64_t)1 << j) << 5;
    } while (input_code >= limit);
    mask = ((uint64_t)1 << (5 + j)) - 1;
    input_code -= ((uint64_t)1 << (4 + 2 * j)) << 5;
    if (input_code < 0 - (uint64_t)1 << (8 + j)) {
      ExtraSID = ((input_code >> 6) & mask) - (0x14 << j) - 2;
      low -= range * (count & 0x3F);
      range *= 0x40;
    } else if (input_code < 0 - (uint64_t)1 << (7 + j)) {
      ExtraSID = ((input_code >> 4) & mask) - (8 << j) - 2;
      low -= range * (count & 0xF);
      range *= 0x10;
    } else if (input_code < 0 - (uint64_t)1 << (6 + j)) {
      ExtraSID = ((input_code >> 2) & mask) + (0x10 << j) - 2;
      low -= range * (count & 0x3);
      range *= 4;
    } else {
      ExtraSID = (input_code & mask) + (0x20 << j) - 2;
    }
  }
  return(ExtraSID);
}


uint8_t DecodeINST(uint8_t Context, uint8_t SIDSymbol) {
  NormalizeDecoder(FREQ_INST_BOT);
  uint32_t extra_range = range;
  range /= RangeScaleINST[Context][SIDSymbol];
  extra_range -= range * RangeScaleINST[Context][SIDSymbol];
  RangeHigh = range * FreqINST[Context][SIDSymbol][0] + extra_range;
  if (RangeHigh > code - low) {
    range = RangeHigh;
    if (RangeScaleINST[Context][SIDSymbol] >= (FREQ_INST_BOT >> 1)) {
      FreqINST[Context][SIDSymbol][0] += RangeScaleINST[Context][SIDSymbol] >> 11;
      if ((RangeScaleINST[Context][SIDSymbol] += RangeScaleINST[Context][SIDSymbol] >> 11) > FREQ_INST_BOT)
        rescaleINST(Context, SIDSymbol);
    } else {
      FreqINST[Context][SIDSymbol][0] += UP_FREQ_INST;
      RangeScaleINST[Context][SIDSymbol] += UP_FREQ_INST;
    }
    return(0);
  } else {
    uint32_t temp;
    count = code - low;
    uint8_t Instances = 0;
    while ((temp = RangeHigh + range * FreqINST[Context][SIDSymbol][++Instances]) <= count)
      RangeHigh = temp;
    low += RangeHigh;
    range *= FreqINST[Context][SIDSymbol][Instances];
    if (RangeScaleINST[Context][SIDSymbol] >= (FREQ_INST_BOT >> 1)) {
      FreqINST[Context][SIDSymbol][Instances] += RangeScaleINST[Context][SIDSymbol] >> 11;
      if ((RangeScaleINST[Context][SIDSymbol] += (RangeScaleINST[Context][SIDSymbol] >> 11)) > FREQ_INST_BOT)
        rescaleINST(Context, SIDSymbol);
    } else {
      FreqINST[Context][SIDSymbol][Instances] += UP_FREQ_INST;
      RangeScaleINST[Context][SIDSymbol] += UP_FREQ_INST;
    }
    return(Instances);
  }
}

uint8_t DecodeERG(uint16_t Context1, uint16_t Context2) {
  NormalizeDecoder(FREQ_ERG_BOT);
  if ((FreqERG[0] + FreqERG[Context1] + FreqERG[Context2]) * (range >>= 13) > code - low) {
    range *= FreqERG[0] + FreqERG[Context1] + FreqERG[Context2];
    FreqERG[0] += (0x400 - FreqERG[0]) >> 2;
    FreqERG[Context1] += (0x1000 - FreqERG[Context1]) >> 4;
    FreqERG[Context2] += (0xC00 - FreqERG[Context2]) >> 3;
    return(0);
  } else {
    low += range * (FreqERG[0] + FreqERG[Context1] + FreqERG[Context2]);
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
  } else {
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
  if (FreqWordTag[Context] * (range >>= 12) > code - low) {
    range = range * FreqWordTag[Context];
    FreqWordTag[Context] += (0x1000 - FreqWordTag[Context]) >> 4;
    Tag = 0;
  } else {
    low += FreqWordTag[Context] * range;
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
  uint8_t FirstChar;
  uint16_t *RangeScalePtr = &RangeScaleFirstChar[SymType][LastChar];
  uint32_t extra_range;
  struct first_char_data * FCDataPtr = &FirstCharData[SymType][LastChar][0];

  NormalizeDecoder((uint32_t)FREQ_FIRST_CHAR_BOT);
  extra_range = range;
  range /= *RangeScalePtr;
  extra_range -= range * *RangeScalePtr;
  RangeHigh = range * FCDataPtr->data.freq + extra_range;
  count = code - low;
  if (RangeHigh > count) {
    range = RangeHigh;
  } else {
    uint32_t temp;
    while ((temp = RangeHigh + range * (++FCDataPtr)->data.freq) <= count)
      RangeHigh = temp;
    low += RangeHigh;
    range *= FCDataPtr->data.freq;
  }
  FirstChar = FCDataPtr->data.symbol;

  if (*RangeScalePtr >= (FREQ_FIRST_CHAR_BOT >> 2)) {
    FCDataPtr->data.freq += *RangeScalePtr >> 10;
    if ((*RangeScalePtr += (*RangeScalePtr >> 10)) >= FREQ_FIRST_CHAR_BOT)
      rescaleFirstChar(SymType, LastChar);
  } else {
    FCDataPtr->data.freq += UP_FREQ_FIRST_CHAR;
    *RangeScalePtr += UP_FREQ_FIRST_CHAR;
  }

  if (FCDataPtr != &FirstCharData[SymType][LastChar][0]) {
    if (FCDataPtr->data.freq > (FCDataPtr - 1)->data.freq) {
      struct first_char_data SavedData;
      SavedData.all_data = FCDataPtr->all_data;
      do {
        FCDataPtr->all_data = (FCDataPtr - 1)->all_data;
        FCDataPtr--;
      } while ((FCDataPtr != &FirstCharData[SymType][LastChar][0]) && (SavedData.data.freq > (FCDataPtr - 1)->data.freq));
      FCDataPtr->all_data = SavedData.all_data;
    }
  }
  return(FirstChar);
}

uint8_t DecodeFirstCharBinary(uint8_t LastChar) {
  uint8_t FirstChar, SectionIndex;
  uint16_t *RangeScalePtr = &RangeScaleFirstChar[0][LastChar];
  uint32_t extra_range, temp;
  struct first_char_data * FCDataPtr = &FirstCharData[0][LastChar][0];

  NormalizeDecoder(FREQ_FIRST_CHAR_BOT);
  extra_range = range;
  SectionIndex = 0;
  range /= (uint32_t)*RangeScalePtr;
  extra_range -= range * ((uint32_t)*RangeScalePtr);
  count = code - low;
  RangeHigh = extra_range;
  while ((SectionIndex != 7)
      && ((temp = RangeHigh + range * RangeScaleFirstCharSection[LastChar][SectionIndex]) <= count)) {
    RangeHigh = temp;
    SectionIndex++;
  }
  FCDataPtr = &FirstCharData[0][LastChar][0x20 * SectionIndex];
  while ((temp = RangeHigh + range * FCDataPtr->data.freq) <= count) {
    RangeHigh = temp;
    FCDataPtr++;
  }
  FirstChar = FCDataPtr - &FirstCharData[0][LastChar][0];
  if (FirstChar == 0)
    range = temp;
  else {
    low += RangeHigh;
    range *= (uint32_t)FCDataPtr->data.freq;
  }

  if (*RangeScalePtr >= (FREQ_FIRST_CHAR_BOT >> 2)) {
    FCDataPtr->data.freq += *RangeScalePtr >> 10;
    if (SectionIndex <= 6)
      RangeScaleFirstCharSection[LastChar][SectionIndex] += *RangeScalePtr >> 10;
    if ((*RangeScalePtr += *RangeScalePtr >> 10) >= FREQ_FIRST_CHAR_BOT)
      rescaleFirstCharBinary(LastChar);
  } else {
    FCDataPtr->data.freq += UP_FREQ_FIRST_CHAR >> 1;
    if (SectionIndex <= 6)
      RangeScaleFirstCharSection[LastChar][SectionIndex] += UP_FREQ_FIRST_CHAR >> 1;
    *RangeScalePtr += UP_FREQ_FIRST_CHAR >> 1;
  }
  return(FirstChar);
}

void InitDecoder(uint8_t max_base_code, uint8_t num_inst_codes, uint8_t cap_encoded, uint8_t UTF8_compliant,
    uint8_t use_mtf, uint8_t * inbuf) {
  uint8_t i, j;
  CapInitialized = 0;
  CapLockInitialized = 0;
  CapEncoded = cap_encoded;
  UTF8Compliant = UTF8_compliant;
  MaxBaseCode = max_base_code;
  MaxInstCode = num_inst_codes - 1;
  InBuffer = inbuf;
  code = 0, range = -1;
  for (low = 4; low != 0; low--)
    code = (code << 8) | InBuffer[InCharNum++];
  StartModelSymType(use_mtf, cap_encoded);
  StartModelMtfFirst();
  StartModelMtfPos();
  StartModelSID();
  StartModelINST(num_inst_codes);
  StartModelERG();
  StartModelGoMtf();
  StartModelWordTag();
  if (cap_encoded || UTF8_compliant)
    StartModelFirstChar();
  else
    StartModelFirstCharBinary();

  if (UTF8_compliant != 0) {
    i = 0x90;
    j = 0x90;
  } else {
    i = 0xFF;
    j = 0xFF;
  }
}

/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Quantized Local Frequency Coding functions                */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

Copyright (c) 2009-2011 Ilya Grebnov <ilya.grebnov@gmail.com>

See file AUTHORS for a full list of contributors.

The bsc and libbsc is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The bsc and libbsc is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the bsc and libbsc. If not, see http://www.gnu.org/licenses/.

Please see the files COPYING and COPYING.LIB for full copyright information.

See also the bsc and libbsc web site:
  http://libbsc.com/ for more information.

--*/

#include <stdlib.h>
#include <memory.h>

#include "qlfc.h"

#include "../libbsc.h"
#include "../platform/platform.h"

#include "core/coder.h"
#include "core/model.h"
#include "core/tables.h"
#include "core/predictor.h"

int bsc_qlfc_init(int features)
{
    return bsc_qlfc_init_static_model();
}

static INLINE int bsc_qlfc_num_blocks(int n)
{
    if (n <       256 * 1024)   return 1;
    if (n <  4 * 1024 * 1024)   return 2;
    if (n < 16 * 1024 * 1024)   return 4;

    return 8;
}

unsigned char * bsc_qlfc_transform(const unsigned char * input, unsigned char * buffer, int n, unsigned char * MTFTable)
{
    unsigned char Flag[ALPHABET_SIZE];

    for (int i = 0; i < ALPHABET_SIZE; ++i) Flag[i] = 0;
    for (int i = 0; i < ALPHABET_SIZE; ++i) MTFTable[i] = i;

    if (input[n - 1] == 0)
    {
        MTFTable[0] = 1; MTFTable[1] = 0;
    }

    int index = n, nSymbols = 0;
    for (int i = n - 1; i >= 0;)
    {
        unsigned char currentChar = input[i--];
        for (; (i >= 0) && (input[i] == currentChar); --i) ;

        unsigned char previousChar = MTFTable[0], rank = 1; MTFTable[0] = currentChar;
        while (true)
        {
            unsigned char temporaryChar0 = MTFTable[rank + 0]; MTFTable[rank + 0] = previousChar;
            if (temporaryChar0 == currentChar) {rank += 0; break; }

            unsigned char temporaryChar1 = MTFTable[rank + 1]; MTFTable[rank + 1] = temporaryChar0;
            if (temporaryChar1 == currentChar) {rank += 1; break; }

            unsigned char temporaryChar2 = MTFTable[rank + 2]; MTFTable[rank + 2] = temporaryChar1;
            if (temporaryChar2 == currentChar) {rank += 2; break; }

            unsigned char temporaryChar3 = MTFTable[rank + 3]; MTFTable[rank + 3] = temporaryChar2;
            if (temporaryChar3 == currentChar) {rank += 3; break; }

            rank += 4; previousChar = temporaryChar3;
        }

        if (Flag[currentChar] == 0)
        {
            Flag[currentChar] = 1;
            rank = nSymbols++;
        }

        buffer[--index] = rank;
    }

    buffer[n - 1] = 1;

    for (int rank = 1; rank < ALPHABET_SIZE; ++rank)
    {
        if (Flag[MTFTable[rank]] == 0)
        {
            MTFTable[rank] = MTFTable[rank - 1];
            break;
        }
    }

    return buffer + index;
}

int bsc_qlfc_encode_slow(const unsigned char * input, unsigned char * output, unsigned char * buffer, int n, BscQlfcModel * model)
{
    unsigned char MTFTable[ALPHABET_SIZE];

    bsc_qlfc_init_model(model);

    int contextRank0 = 0;
    int contextRank4 = 0;
    int contextRun   = 0;
    int maxRank      = 7;
    int avgRank      = 0;

    unsigned char rankHistory[ALPHABET_SIZE], runHistory[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i)
    {
        rankHistory[i] = runHistory[i] = 0;
    }

    unsigned char * rankArray = bsc_qlfc_transform(input, buffer, n, MTFTable);

    BinaryCoder coder;

    coder.InitEncoder(output, n);
    coder.EncodeWord((unsigned int)n);

    unsigned char usedChar[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i) usedChar[i] = 0;

    int prevChar = -1;
    for (int rank = 0; rank < ALPHABET_SIZE; ++rank)
    {
        int currentChar = MTFTable[rank];

        for (int bit = 7; bit >= 0; --bit)
        {
            bool bit0 = false, bit1 = false;

            for (int c = 0; c < ALPHABET_SIZE; ++c)
            {
                if (c == prevChar || usedChar[c] == 0)
                {
                    if ((currentChar >> (bit + 1)) == (c >> (bit + 1)))
                    {
                        if (c & (1 << bit)) bit1 = true; else bit0 = true;
                    }
                }
            }

            if (bit0 && bit1)
            {
                coder.EncodeBit(currentChar & (1 << bit));
            }
        }

        if (currentChar == prevChar)
        {
            maxRank = bsc_log2(rank - 1);
            break;
        }

        prevChar = currentChar; usedChar[currentChar] = 1;
    }

    for (int i = 0; i < n;)
    {
        if (coder.CheckEOB())
        {
            return LIBBSC_UNEXPECTED_EOB;
        }

        int currentChar = input[i++];
        int runSize = 1; for (; (i < n) && (input[i] == currentChar); ++i) runSize++;

        int                 rank            =   *rankArray++;
        int                 history         =   rankHistory[currentChar];
        int                 state           =   model_rank_state(contextRank4, contextRun, history);
        short *             statePredictor  = & model->Rank.StateModel[state];
        short *             charPredictor   = & model->Rank.CharModel[currentChar];
        short *             staticPredictor = & model->Rank.StaticModel;
        ProbabilityMixer *  mixer           = & model->mixerOfRank[currentChar];

        if (avgRank < 32)
        {
            if (rank == 1)
            {
                rankHistory[currentChar] = 0;

                int probability0 = *charPredictor, probability1 = *statePredictor, probability2 = *staticPredictor;

                ProbabilityCounter::UpdateBit0(*statePredictor,  M_RANK_TS_TH0, M_RANK_TS_AR0);
                ProbabilityCounter::UpdateBit0(*charPredictor,   M_RANK_TC_TH0, M_RANK_TC_AR0);
                ProbabilityCounter::UpdateBit0(*staticPredictor, M_RANK_TP_TH0, M_RANK_TP_AR0);

                coder.EncodeBit0(mixer->MixupAndUpdateBit0(probability0, probability1, probability2, M_RANK_TM_LR0, M_RANK_TM_LR1, M_RANK_TM_LR2, M_RANK_TM_TH0, M_RANK_TM_AR0));
            }
            else
            {
                {
                    int probability0 = *charPredictor, probability1 = *statePredictor, probability2 = *staticPredictor;

                    ProbabilityCounter::UpdateBit1(*statePredictor,  M_RANK_TS_TH1, M_RANK_TS_AR1);
                    ProbabilityCounter::UpdateBit1(*charPredictor,   M_RANK_TC_TH1, M_RANK_TC_AR1);
                    ProbabilityCounter::UpdateBit1(*staticPredictor, M_RANK_TP_TH1, M_RANK_TP_AR1);

                    coder.EncodeBit1(mixer->MixupAndUpdateBit1(probability0, probability1, probability2, M_RANK_TM_LR0, M_RANK_TM_LR1, M_RANK_TM_LR2, M_RANK_TM_TH1, M_RANK_TM_AR1));
                }

                int bitRankSize = bsc_log2(rank); rankHistory[currentChar] = bitRankSize;

                statePredictor  = & model->Rank.Exponent.StateModel[state][0];
                charPredictor   = & model->Rank.Exponent.CharModel[currentChar][0];
                staticPredictor = & model->Rank.Exponent.StaticModel[0];
                mixer           = & model->mixerOfRankExponent[history < 1 ? 1 : history][1];

                for (int bit = 1; bit < bitRankSize; ++bit, ++statePredictor, ++charPredictor, ++staticPredictor)
                {
                    int probability0 = *charPredictor, probability1 = *statePredictor, probability2 = *staticPredictor;

                    ProbabilityCounter::UpdateBit1(*statePredictor,  M_RANK_ES_TH1, M_RANK_ES_AR1);
                    ProbabilityCounter::UpdateBit1(*charPredictor,   M_RANK_EC_TH1, M_RANK_EC_AR1);
                    ProbabilityCounter::UpdateBit1(*staticPredictor, M_RANK_EP_TH1, M_RANK_EP_AR1);

                    coder.EncodeBit1(mixer->MixupAndUpdateBit1(probability0, probability1, probability2, M_RANK_EM_LR0, M_RANK_EM_LR1, M_RANK_EM_LR2, M_RANK_EM_TH1, M_RANK_EM_AR1));

                    mixer = & model->mixerOfRankExponent[history <= bit ? bit + 1 : history][bit + 1];
                }
                if (bitRankSize < maxRank)
                {
                    int probability0 = *charPredictor, probability1 = *statePredictor, probability2 = *staticPredictor;

                    ProbabilityCounter::UpdateBit0(*statePredictor,  M_RANK_ES_TH0, M_RANK_ES_AR0);
                    ProbabilityCounter::UpdateBit0(*charPredictor,   M_RANK_EC_TH0, M_RANK_EC_AR0);
                    ProbabilityCounter::UpdateBit0(*staticPredictor, M_RANK_EP_TH0, M_RANK_EP_AR0);

                    coder.EncodeBit0(mixer->MixupAndUpdateBit0(probability0, probability1, probability2, M_RANK_EM_LR0, M_RANK_EM_LR1, M_RANK_EM_LR2, M_RANK_EM_TH0, M_RANK_EM_AR0));
                }

                statePredictor  = & model->Rank.Mantissa[bitRankSize].StateModel[state][0];
                charPredictor   = & model->Rank.Mantissa[bitRankSize].CharModel[currentChar][0];
                staticPredictor = & model->Rank.Mantissa[bitRankSize].StaticModel[0];
                mixer           = & model->mixerOfRankMantissa[bitRankSize];

                for (int context = 1, bit = bitRankSize - 1; bit >= 0; --bit)
                {
                    if (rank & (1 << bit))
                    {
                        int probability0 = charPredictor[context], probability1 = statePredictor[context], probability2 = staticPredictor[context];

                        ProbabilityCounter::UpdateBit1(statePredictor[context],  M_RANK_MS_TH1, M_RANK_MS_AR1);
                        ProbabilityCounter::UpdateBit1(charPredictor[context],   M_RANK_MC_TH1, M_RANK_MC_AR1);
                        ProbabilityCounter::UpdateBit1(staticPredictor[context], M_RANK_MP_TH1, M_RANK_MP_AR1);

                        coder.EncodeBit1(mixer->MixupAndUpdateBit1(probability0, probability1, probability2, M_RANK_MM_LR0, M_RANK_MM_LR1, M_RANK_MM_LR2, M_RANK_MM_TH1, M_RANK_MM_AR1));

                        context += context + 1;
                    }
                    else
                    {
                        int probability0 = charPredictor[context], probability1 = statePredictor[context], probability2 = staticPredictor[context];

                        ProbabilityCounter::UpdateBit0(statePredictor[context],  M_RANK_MS_TH0, M_RANK_MS_AR0);
                        ProbabilityCounter::UpdateBit0(charPredictor[context],   M_RANK_MC_TH0, M_RANK_MC_AR0);
                        ProbabilityCounter::UpdateBit0(staticPredictor[context], M_RANK_MP_TH0, M_RANK_MP_AR0);

                        coder.EncodeBit0(mixer->MixupAndUpdateBit0(probability0, probability1, probability2, M_RANK_MM_LR0, M_RANK_MM_LR1, M_RANK_MM_LR2, M_RANK_MM_TH0, M_RANK_MM_AR0));

                        context += context;
                    }
                }
            }
        }
        else
        {
            rankHistory[currentChar] = bsc_log2(rank);

            statePredictor  = & model->Rank.Escape.StateModel[state][0];
            charPredictor   = & model->Rank.Escape.CharModel[currentChar][0];
            staticPredictor = & model->Rank.Escape.StaticModel[0];

            for (int context = 1, bit = maxRank; bit >= 0; --bit)
            {
                mixer = & model->mixerOfRankEscape[context];

                if (rank & (1 << bit))
                {
                    int probability0 = charPredictor[context], probability1 = statePredictor[context], probability2 = staticPredictor[context];

                    ProbabilityCounter::UpdateBit1(statePredictor[context],  M_RANK_PS_TH1, M_RANK_PS_AR1);
                    ProbabilityCounter::UpdateBit1(charPredictor[context],   M_RANK_PC_TH1, M_RANK_PC_AR1);
                    ProbabilityCounter::UpdateBit1(staticPredictor[context], M_RANK_PP_TH1, M_RANK_PP_AR1);

                    coder.EncodeBit1(mixer->MixupAndUpdateBit1(probability0, probability1, probability2, M_RANK_PM_LR0, M_RANK_PM_LR1, M_RANK_PM_LR2, M_RANK_PM_TH1, M_RANK_PM_AR1));

                    context += context + 1;
                }
                else
                {
                    int probability0 = charPredictor[context], probability1 = statePredictor[context], probability2 = staticPredictor[context];

                    ProbabilityCounter::UpdateBit0(statePredictor[context],  M_RANK_PS_TH0, M_RANK_PS_AR0);
                    ProbabilityCounter::UpdateBit0(charPredictor[context],   M_RANK_PC_TH0, M_RANK_PC_AR0);
                    ProbabilityCounter::UpdateBit0(staticPredictor[context], M_RANK_PP_TH0, M_RANK_PP_AR0);

                    coder.EncodeBit0(mixer->MixupAndUpdateBit0(probability0, probability1, probability2, M_RANK_PM_LR0, M_RANK_PM_LR1, M_RANK_PM_LR2, M_RANK_PM_TH0, M_RANK_PM_AR0));

                    context += context;
                }
            }
        }

        avgRank         =   (avgRank * 124 + rank * 4) >> 7;
        rank            =   rank - 1;
        history         =   runHistory[currentChar];
        state           =   model_run_state(contextRank0, contextRun, rank, history);
        statePredictor  = & model->Run.StateModel[state];
        charPredictor   = & model->Run.CharModel[currentChar];
        staticPredictor = & model->Run.StaticModel;
        mixer           = & model->mixerOfRun[currentChar];

        if (runSize == 1)
        {
            runHistory[currentChar] = (runHistory[currentChar] + 2) >> 2;

            int probability0 = *charPredictor, probability1 = *statePredictor, probability2 = *staticPredictor;

            ProbabilityCounter::UpdateBit0(*statePredictor,  M_RUN_TS_TH0, M_RUN_TS_AR0);
            ProbabilityCounter::UpdateBit0(*charPredictor,   M_RUN_TC_TH0, M_RUN_TC_AR0);
            ProbabilityCounter::UpdateBit0(*staticPredictor, M_RUN_TP_TH0, M_RUN_TP_AR0);

            coder.EncodeBit0(mixer->MixupAndUpdateBit0(probability0, probability1, probability2, M_RUN_TM_LR0, M_RUN_TM_LR1, M_RUN_TM_LR2, M_RUN_TM_TH0, M_RUN_TM_AR0));
        }
        else
        {
            {
                int probability0 = *charPredictor, probability1 = *statePredictor, probability2 = *staticPredictor;

                ProbabilityCounter::UpdateBit1(*statePredictor,  M_RUN_TS_TH1, M_RUN_TS_AR1);
                ProbabilityCounter::UpdateBit1(*charPredictor,   M_RUN_TC_TH1, M_RUN_TC_AR1);
                ProbabilityCounter::UpdateBit1(*staticPredictor, M_RUN_TP_TH1, M_RUN_TP_AR1);

                coder.EncodeBit1(mixer->MixupAndUpdateBit1(probability0, probability1, probability2, M_RUN_TM_LR0, M_RUN_TM_LR1, M_RUN_TM_LR2, M_RUN_TM_TH1, M_RUN_TM_AR1));
            }

            int bitRunSize = bsc_log2(runSize); runHistory[currentChar] = (runHistory[currentChar] + 3 * bitRunSize + 3) >> 2;

            statePredictor  = & model->Run.Exponent.StateModel[state][0];
            charPredictor   = & model->Run.Exponent.CharModel[currentChar][0];
            staticPredictor = & model->Run.Exponent.StaticModel[0];
            mixer           = & model->mixerOfRunExponent[history < 1 ? 1 : history][1];

            for (int bit = 1; bit < bitRunSize; ++bit, ++statePredictor, ++charPredictor, ++staticPredictor)
            {
                int probability0 = *charPredictor, probability1 = *statePredictor, probability2 = *staticPredictor;

                ProbabilityCounter::UpdateBit1(*statePredictor,  M_RUN_ES_TH1, M_RUN_ES_AR1);
                ProbabilityCounter::UpdateBit1(*charPredictor,   M_RUN_EC_TH1, M_RUN_EC_AR1);
                ProbabilityCounter::UpdateBit1(*staticPredictor, M_RUN_EP_TH1, M_RUN_EP_AR1);

                coder.EncodeBit1(mixer->MixupAndUpdateBit1(probability0, probability1, probability2, M_RUN_EM_LR0, M_RUN_EM_LR1, M_RUN_EM_LR2, M_RUN_EM_TH1, M_RUN_EM_AR1));

                mixer = & model->mixerOfRunExponent[history <= bit ? bit + 1 : history][bit + 1];
            }
            {
                int probability0 = *charPredictor, probability1 = *statePredictor, probability2 = *staticPredictor;

                ProbabilityCounter::UpdateBit0(*statePredictor,  M_RUN_ES_TH0, M_RUN_ES_AR0);
                ProbabilityCounter::UpdateBit0(*charPredictor,   M_RUN_EC_TH0, M_RUN_EC_AR0);
                ProbabilityCounter::UpdateBit0(*staticPredictor, M_RUN_EP_TH0, M_RUN_EP_AR0);

                coder.EncodeBit0(mixer->MixupAndUpdateBit0(probability0, probability1, probability2, M_RUN_EM_LR0, M_RUN_EM_LR1, M_RUN_EM_LR2, M_RUN_EM_TH0, M_RUN_EM_AR0));
            }

            statePredictor  = & model->Run.Mantissa[bitRunSize].StateModel[state][0];
            charPredictor   = & model->Run.Mantissa[bitRunSize].CharModel[currentChar][0];
            staticPredictor = & model->Run.Mantissa[bitRunSize].StaticModel[0];
            mixer           = & model->mixerOfRunMantissa[bitRunSize];

            for (int context = 1, bit = bitRunSize - 1; bit >= 0; --bit)
            {
                if (runSize & (1 << bit))
                {
                    int probability0 = charPredictor[context], probability1 = statePredictor[context], probability2 = staticPredictor[context];

                    ProbabilityCounter::UpdateBit1(statePredictor[context],  M_RUN_MS_TH1, M_RUN_MS_AR1);
                    ProbabilityCounter::UpdateBit1(charPredictor[context],   M_RUN_MC_TH1, M_RUN_MC_AR1);
                    ProbabilityCounter::UpdateBit1(staticPredictor[context], M_RUN_MP_TH1, M_RUN_MP_AR1);

                    coder.EncodeBit1(mixer->MixupAndUpdateBit1(probability0, probability1, probability2, M_RUN_MM_LR0, M_RUN_MM_LR1, M_RUN_MM_LR2, M_RUN_MM_TH1, M_RUN_MM_AR1));

                    if (bitRunSize <= 5) context += context + 1; else context++;
                }
                else
                {
                    int probability0 = charPredictor[context], probability1 = statePredictor[context], probability2 = staticPredictor[context];

                    ProbabilityCounter::UpdateBit0(statePredictor[context],  M_RUN_MS_TH0, M_RUN_MS_AR0);
                    ProbabilityCounter::UpdateBit0(charPredictor[context],   M_RUN_MC_TH0, M_RUN_MC_AR0);
                    ProbabilityCounter::UpdateBit0(staticPredictor[context], M_RUN_MP_TH0, M_RUN_MP_AR0);

                    coder.EncodeBit0(mixer->MixupAndUpdateBit0(probability0, probability1, probability2, M_RUN_MM_LR0, M_RUN_MM_LR1, M_RUN_MM_LR2, M_RUN_MM_TH0, M_RUN_MM_AR0));

                    if (bitRunSize <= 5) context += context + 0; else context++;
                }
            }
        }

        contextRank0 = ((contextRank0 << 1) | (rank == 0   ? 1    : 0)) & 0x7;
        contextRank4 = ((contextRank4 << 2) | (rank < 3    ? rank : 3)) & 0xff;
        contextRun   = ((contextRun   << 1) | (runSize < 3 ? 1    : 0)) & 0xf;
    }

    return coder.FinishEncoder();
}

int bsc_qlfc_encode_fast(const unsigned char * input, unsigned char * output, unsigned char * buffer, int n, BscQlfcModel * model)
{
    unsigned char MTFTable[ALPHABET_SIZE];

    bsc_qlfc_init_model(model);

    int contextRank0 = 0;
    int contextRank4 = 0;
    int contextRun   = 0;
    int maxRank      = 7;
    int avgRank      = 0;

    unsigned char rankHistory[ALPHABET_SIZE], runHistory[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i)
    {
        rankHistory[i] = runHistory[i] = 0;
    }

    unsigned char * rankArray = bsc_qlfc_transform(input, buffer, n, MTFTable);

    BinaryCoder coder;

    coder.InitEncoder(output, n);
    coder.EncodeWord((unsigned int)n);

    unsigned char usedChar[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i) usedChar[i] = 0;

    int prevChar = -1;
    for (int rank = 0; rank < ALPHABET_SIZE; ++rank)
    {
        int currentChar = MTFTable[rank];

        for (int bit = 7; bit >= 0; --bit)
        {
            bool bit0 = false, bit1 = false;

            for (int c = 0; c < ALPHABET_SIZE; ++c)
            {
                if (c == prevChar || usedChar[c] == 0)
                {
                    if ((currentChar >> (bit + 1)) == (c >> (bit + 1)))
                    {
                        if (c & (1 << bit)) bit1 = true; else bit0 = true;
                    }
                }
            }

            if (bit0 && bit1)
            {
                coder.EncodeBit(currentChar & (1 << bit));
            }
        }

        if (currentChar == prevChar)
        {
            maxRank = bsc_log2(rank - 1);
            break;
        }

        prevChar = currentChar; usedChar[currentChar] = 1;
    }

    for (int i = 0; i < n;)
    {
        if (coder.CheckEOB())
        {
            return LIBBSC_UNEXPECTED_EOB;
        }

        int currentChar = input[i++];
        int runSize = 1; for (; (i < n) && (input[i] == currentChar); ++i) runSize++;

        int                 rank            =   *rankArray++;
        int                 history         =   rankHistory[currentChar];
        int                 state           =   model_rank_state(contextRank4, contextRun, history);
        short *             statePredictor  = & model->Rank.StateModel[state];
        short *             charPredictor   = & model->Rank.CharModel[currentChar];
        short *             staticPredictor = & model->Rank.StaticModel;

        if (avgRank < 32)
        {
            if (rank == 1)
            {
                rankHistory[currentChar] = 0;

                int probability = ((*charPredictor) * F_RANK_TM_LR0 + (*statePredictor) * F_RANK_TM_LR1 + (*staticPredictor) * F_RANK_TM_LR2) >> 5;

                ProbabilityCounter::UpdateBit0(*statePredictor,  F_RANK_TS_TH0, F_RANK_TS_AR0);
                ProbabilityCounter::UpdateBit0(*charPredictor,   F_RANK_TC_TH0, F_RANK_TC_AR0);
                ProbabilityCounter::UpdateBit0(*staticPredictor, F_RANK_TP_TH0, F_RANK_TP_AR0);

                coder.EncodeBit0(probability);
            }
            else
            {
                {
                    int probability = ((*charPredictor) * F_RANK_TM_LR0 + (*statePredictor) * F_RANK_TM_LR1 + (*staticPredictor) * F_RANK_TM_LR2) >> 5;

                    ProbabilityCounter::UpdateBit1(*statePredictor,  F_RANK_TS_TH1, F_RANK_TS_AR1);
                    ProbabilityCounter::UpdateBit1(*charPredictor,   F_RANK_TC_TH1, F_RANK_TC_AR1);
                    ProbabilityCounter::UpdateBit1(*staticPredictor, F_RANK_TP_TH1, F_RANK_TP_AR1);

                    coder.EncodeBit1(probability);
                }

                int bitRankSize = bsc_log2(rank); rankHistory[currentChar] = bitRankSize;

                statePredictor  = & model->Rank.Exponent.StateModel[state][0];
                charPredictor   = & model->Rank.Exponent.CharModel[currentChar][0];
                staticPredictor = & model->Rank.Exponent.StaticModel[0];

                for (int bit = 1; bit < bitRankSize; ++bit, ++statePredictor, ++charPredictor, ++staticPredictor)
                {
                    int probability = ((*charPredictor) * F_RANK_EM_LR0 + (*statePredictor) * F_RANK_EM_LR1 + (*staticPredictor) * F_RANK_EM_LR2) >> 5;

                    ProbabilityCounter::UpdateBit1(*statePredictor,  F_RANK_ES_TH1, F_RANK_ES_AR1);
                    ProbabilityCounter::UpdateBit1(*charPredictor,   F_RANK_EC_TH1, F_RANK_EC_AR1);
                    ProbabilityCounter::UpdateBit1(*staticPredictor, F_RANK_EP_TH1, F_RANK_EP_AR1);

                    coder.EncodeBit1(probability);
                }
                if (bitRankSize < maxRank)
                {
                    int probability = ((*charPredictor) * F_RANK_EM_LR0 + (*statePredictor) * F_RANK_EM_LR1 + (*staticPredictor) * F_RANK_EM_LR2) >> 5;

                    ProbabilityCounter::UpdateBit0(*statePredictor,  F_RANK_ES_TH0, F_RANK_ES_AR0);
                    ProbabilityCounter::UpdateBit0(*charPredictor,   F_RANK_EC_TH0, F_RANK_EC_AR0);
                    ProbabilityCounter::UpdateBit0(*staticPredictor, F_RANK_EP_TH0, F_RANK_EP_AR0);

                    coder.EncodeBit0(probability);
                }

                statePredictor  = & model->Rank.Mantissa[bitRankSize].StateModel[state][0];
                charPredictor   = & model->Rank.Mantissa[bitRankSize].CharModel[currentChar][0];
                staticPredictor = & model->Rank.Mantissa[bitRankSize].StaticModel[0];

                for (int context = 1, bit = bitRankSize - 1; bit >= 0; --bit)
                {
                    int probability = (charPredictor[context] * F_RANK_MM_LR0 + statePredictor[context] * F_RANK_MM_LR1 + staticPredictor[context] * F_RANK_MM_LR2) >> 5;

                    if (rank & (1 << bit))
                    {
                        ProbabilityCounter::UpdateBit1(statePredictor[context],  F_RANK_MS_TH1, F_RANK_MS_AR1);
                        ProbabilityCounter::UpdateBit1(charPredictor[context],   F_RANK_MC_TH1, F_RANK_MC_AR1);
                        ProbabilityCounter::UpdateBit1(staticPredictor[context], F_RANK_MP_TH1, F_RANK_MP_AR1);

                        coder.EncodeBit1(probability); context += context + 1;
                    }
                    else
                    {
                        ProbabilityCounter::UpdateBit0(statePredictor[context],  F_RANK_MS_TH0, F_RANK_MS_AR0);
                        ProbabilityCounter::UpdateBit0(charPredictor[context],   F_RANK_MC_TH0, F_RANK_MC_AR0);
                        ProbabilityCounter::UpdateBit0(staticPredictor[context], F_RANK_MP_TH0, F_RANK_MP_AR0);

                        coder.EncodeBit0(probability); context += context;
                    }
                }
            }
        }
        else
        {
            rankHistory[currentChar] = bsc_log2(rank);

            statePredictor  = & model->Rank.Escape.StateModel[state][0];
            charPredictor   = & model->Rank.Escape.CharModel[currentChar][0];
            staticPredictor = & model->Rank.Escape.StaticModel[0];

            for (int context = 1, bit = maxRank; bit >= 0; --bit)
            {
                int probability = (charPredictor[context] * F_RANK_PM_LR0 + statePredictor[context] * F_RANK_PM_LR1 + staticPredictor[context] * F_RANK_PM_LR2) >> 5;

                if (rank & (1 << bit))
                {
                    ProbabilityCounter::UpdateBit1(statePredictor[context],  F_RANK_PS_TH1, F_RANK_PS_AR1);
                    ProbabilityCounter::UpdateBit1(charPredictor[context],   F_RANK_PC_TH1, F_RANK_PC_AR1);
                    ProbabilityCounter::UpdateBit1(staticPredictor[context], F_RANK_PP_TH1, F_RANK_PP_AR1);

                    coder.EncodeBit1(probability); context += context + 1;
                }
                else
                {
                    ProbabilityCounter::UpdateBit0(statePredictor[context],  F_RANK_PS_TH0, F_RANK_PS_AR0);
                    ProbabilityCounter::UpdateBit0(charPredictor[context],   F_RANK_PC_TH0, F_RANK_PC_AR0);
                    ProbabilityCounter::UpdateBit0(staticPredictor[context], F_RANK_PP_TH0, F_RANK_PP_AR0);

                    coder.EncodeBit0(probability); context += context;
                }
            }
        }

        avgRank         =   (avgRank * 124 + rank * 4) >> 7;
        rank            =   rank - 1;
        history         =   runHistory[currentChar];
        state           =   model_run_state(contextRank0, contextRun, rank, history);
        statePredictor  = & model->Run.StateModel[state];
        charPredictor   = & model->Run.CharModel[currentChar];
        staticPredictor = & model->Run.StaticModel;

        if (runSize == 1)
        {
            runHistory[currentChar] = (runHistory[currentChar] + 2) >> 2;

            int probability = ((*charPredictor) * F_RUN_TM_LR0 + (*statePredictor) * F_RUN_TM_LR1 + (*staticPredictor) * F_RUN_TM_LR2) >> 5;

            ProbabilityCounter::UpdateBit0(*statePredictor,  F_RUN_TS_TH0, F_RUN_TS_AR0);
            ProbabilityCounter::UpdateBit0(*charPredictor,   F_RUN_TC_TH0, F_RUN_TC_AR0);
            ProbabilityCounter::UpdateBit0(*staticPredictor, F_RUN_TP_TH0, F_RUN_TP_AR0);

            coder.EncodeBit0(probability);
        }
        else
        {
            {
                int probability = ((*charPredictor) * F_RUN_TM_LR0 + (*statePredictor) * F_RUN_TM_LR1 + (*staticPredictor) * F_RUN_TM_LR2) >> 5;

                ProbabilityCounter::UpdateBit1(*statePredictor,  F_RUN_TS_TH1, F_RUN_TS_AR1);
                ProbabilityCounter::UpdateBit1(*charPredictor,   F_RUN_TC_TH1, F_RUN_TC_AR1);
                ProbabilityCounter::UpdateBit1(*staticPredictor, F_RUN_TP_TH1, F_RUN_TP_AR1);

                coder.EncodeBit1(probability);
            }

            int bitRunSize = bsc_log2(runSize); runHistory[currentChar] = (runHistory[currentChar] + 3 * bitRunSize + 3) >> 2;

            statePredictor  = & model->Run.Exponent.StateModel[state][0];
            charPredictor   = & model->Run.Exponent.CharModel[currentChar][0];
            staticPredictor = & model->Run.Exponent.StaticModel[0];

            for (int bit = 1; bit < bitRunSize; ++bit, ++statePredictor, ++charPredictor, ++staticPredictor)
            {
                int probability = ((*charPredictor) * F_RUN_EM_LR0 + (*statePredictor) * F_RUN_EM_LR1 + (*staticPredictor) * F_RUN_EM_LR2) >> 5;

                ProbabilityCounter::UpdateBit1(*statePredictor,  F_RUN_ES_TH1, F_RUN_ES_AR1);
                ProbabilityCounter::UpdateBit1(*charPredictor,   F_RUN_EC_TH1, F_RUN_EC_AR1);
                ProbabilityCounter::UpdateBit1(*staticPredictor, F_RUN_EP_TH1, F_RUN_EP_AR1);

                coder.EncodeBit1(probability);
            }
            {
                int probability = ((*charPredictor) * F_RUN_EM_LR0 + (*statePredictor) * F_RUN_EM_LR1 + (*staticPredictor) * F_RUN_EM_LR2) >> 5;

                ProbabilityCounter::UpdateBit0(*statePredictor,  F_RUN_ES_TH0, F_RUN_ES_AR0);
                ProbabilityCounter::UpdateBit0(*charPredictor,   F_RUN_EC_TH0, F_RUN_EC_AR0);
                ProbabilityCounter::UpdateBit0(*staticPredictor, F_RUN_EP_TH0, F_RUN_EP_AR0);

                coder.EncodeBit0(probability);
            }

            statePredictor  = & model->Run.Mantissa[bitRunSize].StateModel[state][0];
            charPredictor   = & model->Run.Mantissa[bitRunSize].CharModel[currentChar][0];
            staticPredictor = & model->Run.Mantissa[bitRunSize].StaticModel[0];

            for (int context = 1, bit = bitRunSize - 1; bit >= 0; --bit)
            {
                int probability = (charPredictor[context] * F_RUN_MM_LR0 + statePredictor[context] * F_RUN_MM_LR1 + staticPredictor[context] * F_RUN_MM_LR2) >> 5;
                if (runSize & (1 << bit))
                {
                    ProbabilityCounter::UpdateBit1(statePredictor[context],  F_RUN_MS_TH1, F_RUN_MS_AR1);
                    ProbabilityCounter::UpdateBit1(charPredictor[context],   F_RUN_MC_TH1, F_RUN_MC_AR1);
                    ProbabilityCounter::UpdateBit1(staticPredictor[context], F_RUN_MP_TH1, F_RUN_MP_AR1);

                    coder.EncodeBit1(probability); if (bitRunSize <= 5) context += context + 1; else context++;
                }
                else
                {
                    ProbabilityCounter::UpdateBit0(statePredictor[context],  F_RUN_MS_TH0, F_RUN_MS_AR0);
                    ProbabilityCounter::UpdateBit0(charPredictor[context],   F_RUN_MC_TH0, F_RUN_MC_AR0);
                    ProbabilityCounter::UpdateBit0(staticPredictor[context], F_RUN_MP_TH0, F_RUN_MP_AR0);

                    coder.EncodeBit0(probability); if (bitRunSize <= 5) context += context + 0; else context++;
                }
            }
        }

        contextRank0 = ((contextRank0 << 1) | (rank == 0   ? 1    : 0)) & 0x7;
        contextRank4 = ((contextRank4 << 2) | (rank < 3    ? rank : 3)) & 0xff;
        contextRun   = ((contextRun   << 1) | (runSize < 3 ? 1    : 0)) & 0xf;
    }

    return coder.FinishEncoder();
}

int bsc_qlfc_decode_slow(const unsigned char * input, unsigned char * output, BscQlfcModel * model)
{
    BinaryCoder coder;

    unsigned char MTFTable[ALPHABET_SIZE];

    bsc_qlfc_init_model(model);

    int contextRank0 = 0;
    int contextRank4 = 0;
    int contextRun   = 0;
    int maxRank      = 7;
    int avgRank      = 0;

    unsigned char rankHistory[ALPHABET_SIZE], runHistory[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i)
    {
        rankHistory[i] = runHistory[i] = 0;
    }

    coder.InitDecoder(input);
    int n = (int)coder.DecodeWord();

    unsigned char usedChar[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i) usedChar[i] = 0;

    int prevChar = -1;
    for (int rank = 0; rank < ALPHABET_SIZE; ++rank)
    {
        int currentChar = 0;

        for (int bit = 7; bit >= 0; --bit)
        {
            bool bit0 = false, bit1 = false;

            for (int c = 0; c < ALPHABET_SIZE; ++c)
            {
                if (c == prevChar || usedChar[c] == 0)
                {
                    if (currentChar == (c >> (bit + 1)))
                    {
                        if (c & (1 << bit)) bit1 = true; else bit0 = true;
                    }
                }
            }

            if (bit0 && bit1)
            {
                currentChar += currentChar + coder.DecodeBit();
            }
            else
            {
                if (bit0) currentChar += currentChar + 0;
                if (bit1) currentChar += currentChar + 1;
            }
        }

        MTFTable[rank] =  currentChar;

        if (currentChar == prevChar)
        {
            maxRank = bsc_log2(rank - 1);
            break;
        }

        prevChar = currentChar; usedChar[currentChar] = 1;
    }

    for (int i = 0; i < n;)
    {
        int                 currentChar     =   MTFTable[0];
        int                 history         =   rankHistory[currentChar];
        int                 state           =   model_rank_state(contextRank4, contextRun, history);
        short *             statePredictor  = & model->Rank.StateModel[state];
        short *             charPredictor   = & model->Rank.CharModel[currentChar];
        short *             staticPredictor = & model->Rank.StaticModel;
        ProbabilityMixer *  mixer           = & model->mixerOfRank[currentChar];

        int rank = 1;
        if (avgRank < 32)
        {
            if (coder.DecodeBit(mixer->Mixup(*charPredictor, *statePredictor, *staticPredictor)))
            {
                ProbabilityCounter::UpdateBit1(*statePredictor,  M_RANK_TS_TH1, M_RANK_TS_AR1);
                ProbabilityCounter::UpdateBit1(*charPredictor,   M_RANK_TC_TH1, M_RANK_TC_AR1);
                ProbabilityCounter::UpdateBit1(*staticPredictor, M_RANK_TP_TH1, M_RANK_TP_AR1);
                mixer->UpdateBit1(M_RANK_TM_LR0, M_RANK_TM_LR1, M_RANK_TM_LR2, M_RANK_TM_TH1, M_RANK_TM_AR1);

                statePredictor  = & model->Rank.Exponent.StateModel[state][0];
                charPredictor   = & model->Rank.Exponent.CharModel[currentChar][0];
                staticPredictor = & model->Rank.Exponent.StaticModel[0];
                mixer           = & model->mixerOfRankExponent[history < 1 ? 1 : history][1];

                int bitRankSize = 1;
                while (true)
                {
                    if (bitRankSize == maxRank) break;
                    if (coder.DecodeBit(mixer->Mixup(*charPredictor, *statePredictor, *staticPredictor)))
                    {
                        ProbabilityCounter::UpdateBit1(*statePredictor,  M_RANK_ES_TH1, M_RANK_ES_AR1); statePredictor++;
                        ProbabilityCounter::UpdateBit1(*charPredictor,   M_RANK_EC_TH1, M_RANK_EC_AR1); charPredictor++;
                        ProbabilityCounter::UpdateBit1(*staticPredictor, M_RANK_EP_TH1, M_RANK_EP_AR1); staticPredictor++;
                        mixer->UpdateBit1(M_RANK_EM_LR0, M_RANK_EM_LR1, M_RANK_EM_LR2, M_RANK_EM_TH1, M_RANK_EM_AR1);
                        bitRankSize++;
                        mixer = & model->mixerOfRankExponent[history < bitRankSize ? bitRankSize : history][bitRankSize];
                    }
                    else
                    {
                        ProbabilityCounter::UpdateBit0(*statePredictor,  M_RANK_ES_TH0, M_RANK_ES_AR0);
                        ProbabilityCounter::UpdateBit0(*charPredictor,   M_RANK_EC_TH0, M_RANK_EC_AR0);
                        ProbabilityCounter::UpdateBit0(*staticPredictor, M_RANK_EP_TH0, M_RANK_EP_AR0);
                        mixer->UpdateBit0(M_RANK_EM_LR0, M_RANK_EM_LR1, M_RANK_EM_LR2, M_RANK_EM_TH0, M_RANK_EM_AR0);
                        break;
                    }
                }

                rankHistory[currentChar] = bitRankSize;

                statePredictor  = & model->Rank.Mantissa[bitRankSize].StateModel[state][0];
                charPredictor   = & model->Rank.Mantissa[bitRankSize].CharModel[currentChar][0];
                staticPredictor = & model->Rank.Mantissa[bitRankSize].StaticModel[0];
                mixer           = & model->mixerOfRankMantissa[bitRankSize];

                for (int bit = bitRankSize - 1; bit >= 0; --bit)
                {
                    if (coder.DecodeBit(mixer->Mixup(charPredictor[rank], statePredictor[rank], staticPredictor[rank])))
                    {
                        ProbabilityCounter::UpdateBit1(statePredictor[rank],  M_RANK_MS_TH1, M_RANK_MS_AR1);
                        ProbabilityCounter::UpdateBit1(charPredictor[rank],   M_RANK_MC_TH1, M_RANK_MC_AR1);
                        ProbabilityCounter::UpdateBit1(staticPredictor[rank], M_RANK_MP_TH1, M_RANK_MP_AR1);
                        mixer->UpdateBit1(M_RANK_MM_LR0, M_RANK_MM_LR1, M_RANK_MM_LR2, M_RANK_MM_TH1, M_RANK_MM_AR1);
                        rank += rank + 1;
                    }
                    else
                    {
                        ProbabilityCounter::UpdateBit0(statePredictor[rank],  M_RANK_MS_TH0, M_RANK_MS_AR0);
                        ProbabilityCounter::UpdateBit0(charPredictor[rank],   M_RANK_MC_TH0, M_RANK_MC_AR0);
                        ProbabilityCounter::UpdateBit0(staticPredictor[rank], M_RANK_MP_TH0, M_RANK_MP_AR0);
                        mixer->UpdateBit0(M_RANK_MM_LR0, M_RANK_MM_LR1, M_RANK_MM_LR2, M_RANK_MM_TH0, M_RANK_MM_AR0);
                        rank += rank;
                    }
                }
            }
            else
            {
                rankHistory[currentChar] = 0;
                ProbabilityCounter::UpdateBit0(*statePredictor, M_RANK_TS_TH0,  M_RANK_TS_AR0);
                ProbabilityCounter::UpdateBit0(*charPredictor, M_RANK_TC_TH0,   M_RANK_TC_AR0);
                ProbabilityCounter::UpdateBit0(*staticPredictor, M_RANK_TP_TH0, M_RANK_TP_AR0);
                mixer->UpdateBit0(M_RANK_TM_LR0, M_RANK_TM_LR1, M_RANK_TM_LR2, M_RANK_TM_TH0, M_RANK_TM_AR0);
            }
        }
        else
        {
            statePredictor  = & model->Rank.Escape.StateModel[state][0];
            charPredictor   = & model->Rank.Escape.CharModel[currentChar][0];
            staticPredictor = & model->Rank.Escape.StaticModel[0];

            rank = 0;
            for (int context = 1, bit = maxRank; bit >= 0; --bit)
            {
                mixer = & model->mixerOfRankEscape[context];

                if (coder.DecodeBit(mixer->Mixup(charPredictor[context], statePredictor[context], staticPredictor[context])))
                {
                    ProbabilityCounter::UpdateBit1(statePredictor[context],  M_RANK_PS_TH1, M_RANK_PS_AR1);
                    ProbabilityCounter::UpdateBit1(charPredictor[context],   M_RANK_PC_TH1, M_RANK_PC_AR1);
                    ProbabilityCounter::UpdateBit1(staticPredictor[context], M_RANK_PP_TH1, M_RANK_PP_AR1);
                    mixer->UpdateBit1(M_RANK_PM_LR0, M_RANK_PM_LR1, M_RANK_PM_LR2, M_RANK_PM_TH1, M_RANK_PM_AR1);
                    context += context + 1; rank += rank + 1;
                }
                else
                {
                    ProbabilityCounter::UpdateBit0(statePredictor[context],  M_RANK_PS_TH0, M_RANK_PS_AR0);
                    ProbabilityCounter::UpdateBit0(charPredictor[context],   M_RANK_PC_TH0, M_RANK_PC_AR0);
                    ProbabilityCounter::UpdateBit0(staticPredictor[context], M_RANK_PP_TH0, M_RANK_PP_AR0);
                    mixer->UpdateBit0(M_RANK_PM_LR0, M_RANK_PM_LR1, M_RANK_PM_LR2, M_RANK_PM_TH0, M_RANK_PM_AR0);
                    context += context; rank += rank;
                }
            }

            rankHistory[currentChar] = bsc_log2(rank);
        }

        {
            for (int r = 0; r < rank; ++r)
            {
                MTFTable[r] = MTFTable[r + 1];
            }
            MTFTable[rank] = currentChar;
        }

        avgRank         =   (avgRank * 124 + rank * 4) >> 7;
        rank            =   rank - 1;
        history         =   runHistory[currentChar];
        state           =   model_run_state(contextRank0, contextRun, rank, history);
        statePredictor  = & model->Run.StateModel[state];
        charPredictor   = & model->Run.CharModel[currentChar];
        staticPredictor = & model->Run.StaticModel;
        mixer           = & model->mixerOfRun[currentChar];

        int runSize = 1;
        if (coder.DecodeBit(mixer->Mixup(*charPredictor, *statePredictor, *staticPredictor)))
        {
            ProbabilityCounter::UpdateBit1(*statePredictor,  M_RUN_TS_TH1, M_RUN_TS_AR1);
            ProbabilityCounter::UpdateBit1(*charPredictor,   M_RUN_TC_TH1, M_RUN_TC_AR1);
            ProbabilityCounter::UpdateBit1(*staticPredictor, M_RUN_TP_TH1, M_RUN_TP_AR1);
            mixer->UpdateBit1(M_RUN_TM_LR0, M_RUN_TM_LR1, M_RUN_TM_LR2, M_RUN_TM_TH1, M_RUN_TM_AR1);

            statePredictor  = & model->Run.Exponent.StateModel[state][0];
            charPredictor   = & model->Run.Exponent.CharModel[currentChar][0];
            staticPredictor = & model->Run.Exponent.StaticModel[0];
            mixer           = & model->mixerOfRunExponent[history < 1 ? 1 : history][1];

            int bitRunSize = 1;
            while (true)
            {
                if (coder.DecodeBit(mixer->Mixup(*charPredictor, *statePredictor, *staticPredictor)))
                {
                    ProbabilityCounter::UpdateBit1(*statePredictor,  M_RUN_ES_TH1, M_RUN_ES_AR1); statePredictor++;
                    ProbabilityCounter::UpdateBit1(*charPredictor,   M_RUN_EC_TH1, M_RUN_EC_AR1); charPredictor++;
                    ProbabilityCounter::UpdateBit1(*staticPredictor, M_RUN_EP_TH1, M_RUN_EP_AR1); staticPredictor++;
                    mixer->UpdateBit1(M_RUN_EM_LR0, M_RUN_EM_LR1, M_RUN_EM_LR2, M_RUN_EM_TH1, M_RUN_EM_AR1);
                    bitRunSize++; mixer = & model->mixerOfRunExponent[history < bitRunSize ? bitRunSize : history][bitRunSize];
                }
                else
                {
                    ProbabilityCounter::UpdateBit0(*statePredictor,  M_RUN_ES_TH0, M_RUN_ES_AR0);
                    ProbabilityCounter::UpdateBit0(*charPredictor,   M_RUN_EC_TH0, M_RUN_EC_AR0);
                    ProbabilityCounter::UpdateBit0(*staticPredictor, M_RUN_EP_TH0, M_RUN_EP_AR0);
                    mixer->UpdateBit0(M_RUN_EM_LR0, M_RUN_EM_LR1, M_RUN_EM_LR2, M_RUN_EM_TH0, M_RUN_EM_AR0);
                    break;
                }
            }

            runHistory[currentChar] = (runHistory[currentChar] + 3 * bitRunSize + 3) >> 2;

            statePredictor  = & model->Run.Mantissa[bitRunSize].StateModel[state][0];
            charPredictor   = & model->Run.Mantissa[bitRunSize].CharModel[currentChar][0];
            staticPredictor = & model->Run.Mantissa[bitRunSize].StaticModel[0];
            mixer           = & model->mixerOfRunMantissa[bitRunSize];

            for (int context = 1, bit = bitRunSize - 1; bit >= 0; --bit)
            {
                if (coder.DecodeBit(mixer->Mixup(charPredictor[context], statePredictor[context], staticPredictor[context])))
                {
                    ProbabilityCounter::UpdateBit1(statePredictor[context],  M_RUN_MS_TH1, M_RUN_MS_AR1);
                    ProbabilityCounter::UpdateBit1(charPredictor[context],   M_RUN_MC_TH1, M_RUN_MC_AR1);
                    ProbabilityCounter::UpdateBit1(staticPredictor[context], M_RUN_MP_TH1, M_RUN_MP_AR1);
                    mixer->UpdateBit1(M_RUN_MM_LR0, M_RUN_MM_LR1, M_RUN_MM_LR2, M_RUN_MM_TH1, M_RUN_MM_AR1);
                    runSize += runSize + 1; if (bitRunSize <= 5) context += context + 1; else context++;
                }
                else
                {
                    ProbabilityCounter::UpdateBit0(statePredictor[context],  M_RUN_MS_TH0, M_RUN_MS_AR0);
                    ProbabilityCounter::UpdateBit0(charPredictor[context],   M_RUN_MC_TH0, M_RUN_MC_AR0);
                    ProbabilityCounter::UpdateBit0(staticPredictor[context], M_RUN_MP_TH0, M_RUN_MP_AR0);
                    mixer->UpdateBit0(M_RUN_MM_LR0, M_RUN_MM_LR1, M_RUN_MM_LR2, M_RUN_MM_TH0, M_RUN_MM_AR0);
                    runSize += runSize; if (bitRunSize <= 5) context += context; else context++;
                }
            }

        }
        else
        {
            runHistory[currentChar] = (runHistory[currentChar] + 2) >> 2;
            ProbabilityCounter::UpdateBit0(*statePredictor,  M_RUN_TS_TH0, M_RUN_TS_AR0);
            ProbabilityCounter::UpdateBit0(*charPredictor,   M_RUN_TC_TH0, M_RUN_TC_AR0);
            ProbabilityCounter::UpdateBit0(*staticPredictor, M_RUN_TP_TH0, M_RUN_TP_AR0);
            mixer->UpdateBit0(M_RUN_TM_LR0, M_RUN_TM_LR1, M_RUN_TM_LR2, M_RUN_TM_TH0, M_RUN_TM_AR0);
        }

        contextRank0 = ((contextRank0 << 1) | (rank == 0   ? 1    : 0)) & 0x7;
        contextRank4 = ((contextRank4 << 2) | (rank < 3    ? rank : 3)) & 0xff;
        contextRun   = ((contextRun   << 1) | (runSize < 3 ? 1    : 0)) & 0xf;

        for (; runSize > 0; --runSize) output[i++] = currentChar;
    }

    return n;
}

int bsc_qlfc_decode_fast(const unsigned char * input, unsigned char * output, BscQlfcModel * model)
{
    BinaryCoder coder;

    unsigned char MTFTable[ALPHABET_SIZE];

    bsc_qlfc_init_model(model);

    int contextRank0 = 0;
    int contextRank4 = 0;
    int contextRun   = 0;
    int maxRank      = 7;
    int avgRank      = 0;

    unsigned char rankHistory[ALPHABET_SIZE], runHistory[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i)
    {
        rankHistory[i] = runHistory[i] = 0;
    }

    coder.InitDecoder(input);
    int n = (int)coder.DecodeWord();

    unsigned char usedChar[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; ++i) usedChar[i] = 0;

    int prevChar = -1;
    for (int rank = 0; rank < ALPHABET_SIZE; ++rank)
    {
        int currentChar = 0;

        for (int bit = 7; bit >= 0; --bit)
        {
            bool bit0 = false, bit1 = false;

            for (int c = 0; c < ALPHABET_SIZE; ++c)
            {
                if (c == prevChar || usedChar[c] == 0)
                {
                    if (currentChar == (c >> (bit + 1)))
                    {
                        if (c & (1 << bit)) bit1 = true; else bit0 = true;
                    }
                }
            }

            if (bit0 && bit1)
            {
                currentChar += currentChar + coder.DecodeBit();
            }
            else
            {
                if (bit0) currentChar += currentChar + 0;
                if (bit1) currentChar += currentChar + 1;
            }
        }

        MTFTable[rank] =  currentChar;

        if (currentChar == prevChar)
        {
            maxRank = bsc_log2(rank - 1);
            break;
        }

        prevChar = currentChar; usedChar[currentChar] = 1;
    }

    for (int i = 0; i < n;)
    {
        int                 currentChar     =   MTFTable[0];
        int                 history         =   rankHistory[currentChar];
        int                 state           =   model_rank_state(contextRank4, contextRun, history);
        short *             statePredictor  = & model->Rank.StateModel[state];
        short *             charPredictor   = & model->Rank.CharModel[currentChar];
        short *             staticPredictor = & model->Rank.StaticModel;

        int rank = 1;
        if (avgRank < 32)
        {
            if (coder.DecodeBit((*charPredictor * F_RANK_TM_LR0 + *statePredictor * F_RANK_TM_LR1 + *staticPredictor * F_RANK_TM_LR2) >> 5))
            {
                ProbabilityCounter::UpdateBit1(*statePredictor,  F_RANK_TS_TH1, F_RANK_TS_AR1);
                ProbabilityCounter::UpdateBit1(*charPredictor,   F_RANK_TC_TH1, F_RANK_TC_AR1);
                ProbabilityCounter::UpdateBit1(*staticPredictor, F_RANK_TP_TH1, F_RANK_TP_AR1);

                statePredictor  = & model->Rank.Exponent.StateModel[state][0];
                charPredictor   = & model->Rank.Exponent.CharModel[currentChar][0];
                staticPredictor = & model->Rank.Exponent.StaticModel[0];

                int bitRankSize = 1;
                while (true)
                {
                    if (bitRankSize == maxRank) break;
                    if (coder.DecodeBit((*charPredictor * F_RANK_EM_LR0 + *statePredictor * F_RANK_EM_LR1 + *staticPredictor * F_RANK_EM_LR2) >> 5))
                    {
                        ProbabilityCounter::UpdateBit1(*statePredictor,  F_RANK_ES_TH1, F_RANK_ES_AR1); statePredictor++;
                        ProbabilityCounter::UpdateBit1(*charPredictor,   F_RANK_EC_TH1, F_RANK_EC_AR1); charPredictor++;
                        ProbabilityCounter::UpdateBit1(*staticPredictor, F_RANK_EP_TH1, F_RANK_EP_AR1); staticPredictor++;
                        bitRankSize++;
                    }
                    else
                    {
                        ProbabilityCounter::UpdateBit0(*statePredictor,  F_RANK_ES_TH0, F_RANK_ES_AR0);
                        ProbabilityCounter::UpdateBit0(*charPredictor,   F_RANK_EC_TH0, F_RANK_EC_AR0);
                        ProbabilityCounter::UpdateBit0(*staticPredictor, F_RANK_EP_TH0, F_RANK_EP_AR0);
                        break;
                    }
                }

                rankHistory[currentChar] = bitRankSize;

                statePredictor  = & model->Rank.Mantissa[bitRankSize].StateModel[state][0];
                charPredictor   = & model->Rank.Mantissa[bitRankSize].CharModel[currentChar][0];
                staticPredictor = & model->Rank.Mantissa[bitRankSize].StaticModel[0];

                for (int bit = bitRankSize - 1; bit >= 0; --bit)
                {
                    if (coder.DecodeBit((charPredictor[rank] * F_RANK_MM_LR0 + statePredictor[rank] * F_RANK_MM_LR1 + staticPredictor[rank] * F_RANK_MM_LR2) >> 5))
                    {
                        ProbabilityCounter::UpdateBit1(statePredictor[rank],  F_RANK_MS_TH1, F_RANK_MS_AR1);
                        ProbabilityCounter::UpdateBit1(charPredictor[rank],   F_RANK_MC_TH1, F_RANK_MC_AR1);
                        ProbabilityCounter::UpdateBit1(staticPredictor[rank], F_RANK_MP_TH1, F_RANK_MP_AR1);
                        rank += rank + 1;
                    }
                    else
                    {
                        ProbabilityCounter::UpdateBit0(statePredictor[rank],  F_RANK_MS_TH0, F_RANK_MS_AR0);
                        ProbabilityCounter::UpdateBit0(charPredictor[rank],   F_RANK_MC_TH0, F_RANK_MC_AR0);
                        ProbabilityCounter::UpdateBit0(staticPredictor[rank], F_RANK_MP_TH0, F_RANK_MP_AR0);
                        rank += rank;
                    }
                }
            }
            else
            {
                rankHistory[currentChar] = 0;
                ProbabilityCounter::UpdateBit0(*statePredictor,  F_RANK_TS_TH0, F_RANK_TS_AR0);
                ProbabilityCounter::UpdateBit0(*charPredictor,   F_RANK_TC_TH0, F_RANK_TC_AR0);
                ProbabilityCounter::UpdateBit0(*staticPredictor, F_RANK_TP_TH0, F_RANK_TP_AR0);
            }
        }
        else
        {
            statePredictor  = & model->Rank.Escape.StateModel[state][0];
            charPredictor   = & model->Rank.Escape.CharModel[currentChar][0];
            staticPredictor = & model->Rank.Escape.StaticModel[0];

            rank = 0;
            for (int context = 1, bit = maxRank; bit >= 0; --bit)
            {
                if (coder.DecodeBit((charPredictor[context] * F_RANK_PM_LR0 + statePredictor[context] * F_RANK_PM_LR1 + staticPredictor[context] * F_RANK_PM_LR2) >> 5))
                {
                    ProbabilityCounter::UpdateBit1(statePredictor[context],  F_RANK_PS_TH1, F_RANK_PS_AR1);
                    ProbabilityCounter::UpdateBit1(charPredictor[context],   F_RANK_PC_TH1, F_RANK_PC_AR1);
                    ProbabilityCounter::UpdateBit1(staticPredictor[context], F_RANK_PP_TH1, F_RANK_PP_AR1);
                    context += context + 1; rank += rank + 1;
                }
                else
                {
                    ProbabilityCounter::UpdateBit0(statePredictor[context],  F_RANK_PS_TH0, F_RANK_PS_AR0);
                    ProbabilityCounter::UpdateBit0(charPredictor[context],   F_RANK_PC_TH0, F_RANK_PC_AR0);
                    ProbabilityCounter::UpdateBit0(staticPredictor[context], F_RANK_PP_TH0, F_RANK_PP_AR0);
                    context += context; rank += rank;
                }
            }

            rankHistory[currentChar] = bsc_log2(rank);
        }

        {
            for (int r = 0; r < rank; ++r)
            {
                MTFTable[r] = MTFTable[r + 1];
            }
            MTFTable[rank] = currentChar;
        }

        avgRank         =   (avgRank * 124 + rank * 4) >> 7;
        rank            =   rank - 1;
        history         =   runHistory[currentChar];
        state           =   model_run_state(contextRank0, contextRun, rank, history);
        statePredictor  = & model->Run.StateModel[state];
        charPredictor   = & model->Run.CharModel[currentChar];
        staticPredictor = & model->Run.StaticModel;

        int runSize = 1;
        if (coder.DecodeBit((*charPredictor * F_RUN_TM_LR0 + *statePredictor * F_RUN_TM_LR1 + *staticPredictor * F_RUN_TM_LR2) >> 5))
        {
            ProbabilityCounter::UpdateBit1(*statePredictor,  F_RUN_TS_TH1, F_RUN_TS_AR1);
            ProbabilityCounter::UpdateBit1(*charPredictor,   F_RUN_TC_TH1, F_RUN_TC_AR1);
            ProbabilityCounter::UpdateBit1(*staticPredictor, F_RUN_TP_TH1, F_RUN_TP_AR1);

            statePredictor  = & model->Run.Exponent.StateModel[state][0];
            charPredictor   = & model->Run.Exponent.CharModel[currentChar][0];
            staticPredictor = & model->Run.Exponent.StaticModel[0];

            int bitRunSize = 1;
            while (true)
            {
                if (coder.DecodeBit((*charPredictor * F_RUN_EM_LR0 + *statePredictor * F_RUN_EM_LR1 + *staticPredictor * F_RUN_EM_LR2) >> 5))
                {
                    ProbabilityCounter::UpdateBit1(*statePredictor,  F_RUN_ES_TH1, F_RUN_ES_AR1); statePredictor++;
                    ProbabilityCounter::UpdateBit1(*charPredictor,   F_RUN_EC_TH1, F_RUN_EC_AR1); charPredictor++;
                    ProbabilityCounter::UpdateBit1(*staticPredictor, F_RUN_EP_TH1, F_RUN_EP_AR1); staticPredictor++;
                    bitRunSize++;
                }
                else
                {
                    ProbabilityCounter::UpdateBit0(*statePredictor,  F_RUN_ES_TH0, F_RUN_ES_AR0);
                    ProbabilityCounter::UpdateBit0(*charPredictor,   F_RUN_EC_TH0, F_RUN_EC_AR0);
                    ProbabilityCounter::UpdateBit0(*staticPredictor, F_RUN_EP_TH0, F_RUN_EP_AR0);
                    break;
                }
            }

            runHistory[currentChar] = (runHistory[currentChar] + 3 * bitRunSize + 3) >> 2;

            statePredictor  = & model->Run.Mantissa[bitRunSize].StateModel[state][0];
            charPredictor   = & model->Run.Mantissa[bitRunSize].CharModel[currentChar][0];
            staticPredictor = & model->Run.Mantissa[bitRunSize].StaticModel[0];

            for (int context = 1, bit = bitRunSize - 1; bit >= 0; --bit)
            {
                if (coder.DecodeBit((charPredictor[context] * F_RUN_MM_LR0 + statePredictor[context] * F_RUN_MM_LR1 + staticPredictor[context] * F_RUN_MM_LR2) >> 5))
                {
                    ProbabilityCounter::UpdateBit1(statePredictor[context],  F_RUN_MS_TH1, F_RUN_MS_AR1);
                    ProbabilityCounter::UpdateBit1(charPredictor[context],   F_RUN_MC_TH1, F_RUN_MC_AR1);
                    ProbabilityCounter::UpdateBit1(staticPredictor[context], F_RUN_MP_TH1, F_RUN_MP_AR1);
                    runSize += runSize + 1; if (bitRunSize <= 5) context += context + 1; else context++;
                }
                else
                {
                    ProbabilityCounter::UpdateBit0(statePredictor[context],  F_RUN_MS_TH0, F_RUN_MS_AR0);
                    ProbabilityCounter::UpdateBit0(charPredictor[context],   F_RUN_MC_TH0, F_RUN_MC_AR0);
                    ProbabilityCounter::UpdateBit0(staticPredictor[context], F_RUN_MP_TH0, F_RUN_MP_AR0);
                    runSize += runSize; if (bitRunSize <= 5) context += context; else context++;
                }
            }

        }
        else
        {
            runHistory[currentChar] = (runHistory[currentChar] + 2) >> 2;
            ProbabilityCounter::UpdateBit0(*statePredictor,  F_RUN_TS_TH0, F_RUN_TS_AR0);
            ProbabilityCounter::UpdateBit0(*charPredictor,   F_RUN_TC_TH0, F_RUN_TC_AR0);
            ProbabilityCounter::UpdateBit0(*staticPredictor, F_RUN_TP_TH0, F_RUN_TP_AR0);
        }

        contextRank0 = ((contextRank0 << 1) | (rank == 0   ? 1    : 0)) & 0x7;
        contextRank4 = ((contextRank4 << 2) | (rank < 3    ? rank : 3)) & 0xff;
        contextRun   = ((contextRun   << 1) | (runSize < 3 ? 1    : 0)) & 0xf;

        for (; runSize > 0; --runSize) output[i++] = currentChar;
    }

    return n;
}

int bsc_qlfc_encode_block(const unsigned char * input, unsigned char * output, int n, int features)
{
    if (BscQlfcModel * model = (BscQlfcModel *)bsc_malloc(sizeof(BscQlfcModel)))
    {
        if (unsigned char * buffer = (unsigned char *)bsc_malloc(n * sizeof(unsigned char)))
        {
            if (features & LIBBSC_FEATURE_FASTMODE)
            {
                int result = bsc_qlfc_encode_fast(input, output + 1, buffer, n, model);
                if (result >= LIBBSC_NO_ERROR) result = (output[0] = 1, result + 1);
                bsc_free(buffer); bsc_free(model);
                return result;
            }
            else
            {
                int result = bsc_qlfc_encode_slow(input, output + 1, buffer, n, model);
                if (result >= LIBBSC_NO_ERROR) result = (output[0] = 0, result + 1);
                bsc_free(buffer); bsc_free(model);
                return result;
            }
        };
        bsc_free(model);
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

void bsc_qlfc_split_blocks(const unsigned char * input, int n, int nBlocks, int * blockStart, int * blockSize)
{
    int rankSize = 0;
    for (int i = 1; i < n; i += 32)
    {
        if (input[i] != input[i - 1]) rankSize++;
    }

    if (rankSize > nBlocks)
    {
        int blockRankSize = rankSize / nBlocks;

        blockStart[0] = 0; rankSize = 0;
        for (int id = 0, i = 1; i < n; i += 32)
        {
            if (input[i] != input[i - 1])
            {
                rankSize++;
                if (rankSize == blockRankSize)
                {
                    rankSize = 0;

                    blockSize[id] = i - blockStart[id];
                    id++; blockStart[id] = i;

                    if (id == nBlocks - 1) break;
                }
            }
        }
        blockSize[nBlocks - 1] = n - blockStart[nBlocks - 1];
    }
    else
    {
        for (int p = 0; p < nBlocks; ++p)
        {
            blockStart[p] = (n / nBlocks) * p;
            blockSize[p]  = (p != nBlocks - 1) ? n / nBlocks : n - (n / nBlocks) * (nBlocks - 1);
        }
    }
}

int bsc_qlfc_compress_serial(const unsigned char * input, unsigned char * output, int n, int features)
{
    if (bsc_qlfc_num_blocks(n) == 1)
    {
        int result = bsc_qlfc_encode_block(input, output + 1, n, features);
        if (result >= LIBBSC_NO_ERROR) result = (output[0] = 1, result + 1);

        return result;
    }

    int compressedStart[ALPHABET_SIZE];
    int compressedSize[ALPHABET_SIZE];

    int nBlocks   = bsc_qlfc_num_blocks(n);
    int outputPtr = 1 + 8 * nBlocks;

    bsc_qlfc_split_blocks(input, n, nBlocks, compressedStart, compressedSize);

    output[0] = nBlocks;
    for (int blockId = 0; blockId < nBlocks; ++blockId)
    {
        int inputStart  = compressedStart[blockId];
        int inputSize   = compressedSize[blockId];
        int outputSize  = inputSize; if (outputSize > n - outputPtr) outputSize = n - outputPtr;

        int result = bsc_qlfc_encode_block(input + inputStart, output + outputPtr, inputSize, features);
        if (result < LIBBSC_NO_ERROR)
        {
            if (outputPtr + inputSize >= n) return LIBBSC_NOT_COMPRESSIBLE;
            result = inputSize; memcpy(output + outputPtr, input + inputStart, inputSize);
        }

        *(int *)(output + 1 + 8 * blockId + 0) = inputSize;
        *(int *)(output + 1 + 8 * blockId + 4) = result;

        outputPtr += result;
    }

    return outputPtr;
}

#ifdef LIBBSC_OPENMP

int bsc_qlfc_compress_parallel(const unsigned char * input, unsigned char * output, int n, int features)
{
    if (unsigned char * buffer = (unsigned char *)bsc_malloc(n * sizeof(unsigned char)))
    {
        int compressionResult[ALPHABET_SIZE];
        int compressedStart[ALPHABET_SIZE];
        int compressedSize[ALPHABET_SIZE];

        int nBlocks = bsc_qlfc_num_blocks(n);
        int result  = LIBBSC_NO_ERROR;

        output[0] = nBlocks;
        #pragma omp parallel
        {
            if (omp_get_num_threads() == 1)
            {
                result = bsc_qlfc_compress_serial(input, output, n, features);
            }
            else
            {
                #pragma omp single
                {
                    bsc_qlfc_split_blocks(input, n, nBlocks, compressedStart, compressedSize);
                }

                #pragma omp for schedule(dynamic)
                for (int blockId = 0; blockId < nBlocks; ++blockId)
                {
                    int blockStart   = compressedStart[blockId];
                    int blockSize    = compressedSize[blockId];

                    compressionResult[blockId] = bsc_qlfc_encode_block(input + blockStart, buffer + blockStart, blockSize, features);
                    if (compressionResult[blockId] < LIBBSC_NO_ERROR) compressionResult[blockId] = blockSize;

                    *(int *)(output + 1 + 8 * blockId + 0) = blockSize;
                    *(int *)(output + 1 + 8 * blockId + 4) = compressionResult[blockId];
                }

                #pragma omp single
                {
                    result = 1 + 8 * nBlocks;
                    for (int blockId = 0; blockId < nBlocks; ++blockId)
                    {
                        result += compressionResult[blockId];
                    }

                    if (result >= n) result = LIBBSC_NOT_COMPRESSIBLE;
                }

                if (result >= LIBBSC_NO_ERROR)
                {
                    #pragma omp for schedule(dynamic)
                    for (int blockId = 0; blockId < nBlocks; ++blockId)
                    {
                        int blockStart   = compressedStart[blockId];
                        int blockSize    = compressedSize[blockId];

                        int outputPtr = 1 + 8 * nBlocks;
                        for (int p = 0; p < blockId; ++p) outputPtr += compressionResult[p];

                        if (compressionResult[blockId] != blockSize)
                        {
                            memcpy(output + outputPtr, buffer + blockStart, compressionResult[blockId]);
                        }
                        else
                        {
                            memcpy(output + outputPtr, input + blockStart, compressionResult[blockId]);
                        }
                    }
                }
            }
        }

        bsc_free(buffer);

        return result;
    }
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

#endif

int bsc_qlfc_compress(const unsigned char * input, unsigned char * output, int n, int features)
{

#ifdef LIBBSC_OPENMP

    if ((bsc_qlfc_num_blocks(n) != 1) && (features & LIBBSC_FEATURE_MULTITHREADING))
    {
        return bsc_qlfc_compress_parallel(input, output, n, features);
    }

#endif

    return bsc_qlfc_compress_serial(input, output, n, features);
}


int bsc_qlfc_decode_block(const unsigned char * input, unsigned char * output, int features)
{
    if (BscQlfcModel * model = (BscQlfcModel *)bsc_malloc(sizeof(BscQlfcModel)))
    {
        if (input[0] > 0)
        {
            int size = bsc_qlfc_decode_fast(input + 1, output, model);
            bsc_free(model);
            return size;
        }
        else
        {
            int size = bsc_qlfc_decode_slow(input + 1, output, model);
            bsc_free(model);
            return size;
        }
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_qlfc_decompress(const unsigned char * input, unsigned char * output, int features)
{
    int nBlocks = input[0];

    if (nBlocks == 1)
    {
        return bsc_qlfc_decode_block(input + 1, output, features);
    }

    int decompressionResult[ALPHABET_SIZE];

#ifdef LIBBSC_OPENMP

    if (features & LIBBSC_FEATURE_MULTITHREADING)
    {
        #pragma omp parallel for schedule(dynamic)
        for (int blockId = 0; blockId < nBlocks; ++blockId)
        {
            int inputPtr = 0;  for (int p = 0; p < blockId; ++p) inputPtr  += *(int *)(input + 1 + 8 * p + 4);
            int outputPtr = 0; for (int p = 0; p < blockId; ++p) outputPtr += *(int *)(input + 1 + 8 * p + 0);

            inputPtr += 1 + 8 * nBlocks;

            int inputSize  = *(int *)(input + 1 + 8 * blockId + 4);
            int outputSize = *(int *)(input + 1 + 8 * blockId + 0);

            if (inputSize != outputSize)
            {
                decompressionResult[blockId] = bsc_qlfc_decode_block(input + inputPtr, output + outputPtr, features);
            }
            else
            {
                decompressionResult[blockId] = inputSize; memcpy(output + outputPtr, input + inputPtr, inputSize);
            }
        }
    }
    else

#endif

    {
        for (int blockId = 0; blockId < nBlocks; ++blockId)
        {
            int inputPtr = 0;  for (int p = 0; p < blockId; ++p) inputPtr  += *(int *)(input + 1 + 8 * p + 4);
            int outputPtr = 0; for (int p = 0; p < blockId; ++p) outputPtr += *(int *)(input + 1 + 8 * p + 0);

            inputPtr += 1 + 8 * nBlocks;

            int inputSize  = *(int *)(input + 1 + 8 * blockId + 4);
            int outputSize = *(int *)(input + 1 + 8 * blockId + 0);

            if (inputSize != outputSize)
            {
                decompressionResult[blockId] = bsc_qlfc_decode_block(input + inputPtr, output + outputPtr, features);
            }
            else
            {
                decompressionResult[blockId] = inputSize; memcpy(output + outputPtr, input + inputPtr, inputSize);
            }
        }
    }

    int dataSize = 0, result = LIBBSC_NO_ERROR;
    for (int blockId = 0; blockId < nBlocks; ++blockId)
    {
        if (decompressionResult[blockId] < LIBBSC_NO_ERROR) result = decompressionResult[blockId];
        dataSize += decompressionResult[blockId];
    }

    return (result == LIBBSC_NO_ERROR) ? dataSize : result;
}

/*-----------------------------------------------------------*/
/* End                                              qlfc.cpp */
/*-----------------------------------------------------------*/

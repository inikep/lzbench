/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Statistical data compression model for QLFC               */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

   Copyright (c) 2009-2021 Ilya Grebnov <ilya.grebnov@gmail.com>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Please see the file LICENSE for full copyright information and file AUTHORS
for full list of contributors.

See also the bsc and libbsc web site:
  http://libbsc.com/ for more information.

--*/

#ifndef _LIBBSC_QLFC_MODEL_H
#define _LIBBSC_QLFC_MODEL_H

#include "../common/predictor.h"

const int M_RANK_TS_TH0 =    1; const int M_RANK_TS_AR0 =   57;
const int M_RANK_TS_TH1 = -111; const int M_RANK_TS_AR1 =   31;
const int M_RANK_TC_TH0 =  291; const int M_RANK_TC_AR0 =  250;
const int M_RANK_TC_TH1 =  154; const int M_RANK_TC_AR1 =  528;
const int M_RANK_TP_TH0 =  375; const int M_RANK_TP_AR0 =  163;
const int M_RANK_TP_TH1 =  313; const int M_RANK_TP_AR1 =  639;
const int M_RANK_TM_TH0 =  -41; const int M_RANK_TM_AR0 =   96;
const int M_RANK_TM_TH1 =   53; const int M_RANK_TM_AR1 =   49;
const int M_RANK_TM_LR0 =   20; const int M_RANK_TM_LR1 =   47;
const int M_RANK_TM_LR2 =   27;

const int M_RANK_ES_TH0 = -137; const int M_RANK_ES_AR0 =   17;
const int M_RANK_ES_TH1 =  482; const int M_RANK_ES_AR1 =   40;
const int M_RANK_EC_TH0 =   61; const int M_RANK_EC_AR0 =  192;
const int M_RANK_EC_TH1 =  200; const int M_RANK_EC_AR1 =  133;
const int M_RANK_EP_TH0 =   54; const int M_RANK_EP_AR0 = 1342;
const int M_RANK_EP_TH1 =  578; const int M_RANK_EP_AR1 = 1067;
const int M_RANK_EM_TH0 =  -11; const int M_RANK_EM_AR0 =  318;
const int M_RANK_EM_TH1 =  144; const int M_RANK_EM_AR1 =  848;
const int M_RANK_EM_LR0 =   49; const int M_RANK_EM_LR1 =   41;
const int M_RANK_EM_LR2 =   40;

const int M_RANK_MS_TH0 = -145; const int M_RANK_MS_AR0 =   18;
const int M_RANK_MS_TH1 =  114; const int M_RANK_MS_AR1 =   24;
const int M_RANK_MC_TH0 =  -43; const int M_RANK_MC_AR0 =   69;
const int M_RANK_MC_TH1 =  -36; const int M_RANK_MC_AR1 =   78;
const int M_RANK_MP_TH0 =   -2; const int M_RANK_MP_AR0 = 1119;
const int M_RANK_MP_TH1 =   11; const int M_RANK_MP_AR1 = 1181;
const int M_RANK_MM_TH0 = -203; const int M_RANK_MM_AR0 =   20;
const int M_RANK_MM_TH1 = -271; const int M_RANK_MM_AR1 =   15;
const int M_RANK_MM_LR0 =  263; const int M_RANK_MM_LR1 =  175;
const int M_RANK_MM_LR2 =   17;

const int M_RANK_PS_TH0 =  -99; const int M_RANK_PS_AR0 =   32;
const int M_RANK_PS_TH1 =  318; const int M_RANK_PS_AR1 =   42;
const int M_RANK_PC_TH0 =   17; const int M_RANK_PC_AR0 =  101;
const int M_RANK_PC_TH1 = 1116; const int M_RANK_PC_AR1 =  246;
const int M_RANK_PP_TH0 =   22; const int M_RANK_PP_AR0 =  964;
const int M_RANK_PP_TH1 =   -2; const int M_RANK_PP_AR1 = 1110;
const int M_RANK_PM_TH0 = -194; const int M_RANK_PM_AR0 =   21;
const int M_RANK_PM_TH1 = -129; const int M_RANK_PM_AR1 =   20;
const int M_RANK_PM_LR0 =  480; const int M_RANK_PM_LR1 =  202;
const int M_RANK_PM_LR2 =   17;

const int M_RUN_TS_TH0 =  -93; const int M_RUN_TS_AR0 =   34;
const int M_RUN_TS_TH1 =   -4; const int M_RUN_TS_AR1 =   51;
const int M_RUN_TC_TH0 =  139; const int M_RUN_TC_AR0 =  423;
const int M_RUN_TC_TH1 =  244; const int M_RUN_TC_AR1 =  162;
const int M_RUN_TP_TH0 =  275; const int M_RUN_TP_AR0 =  450;
const int M_RUN_TP_TH1 =   -6; const int M_RUN_TP_AR1 =  579;
const int M_RUN_TM_TH0 =  -68; const int M_RUN_TM_AR0 =   25;
const int M_RUN_TM_TH1 =    1; const int M_RUN_TM_AR1 =   64;
const int M_RUN_TM_LR0 =   15; const int M_RUN_TM_LR1 =   50;
const int M_RUN_TM_LR2 =   78;

const int M_RUN_ES_TH0 = -116; const int M_RUN_ES_AR0 =   31;
const int M_RUN_ES_TH1 =   43; const int M_RUN_ES_AR1 =   45;
const int M_RUN_EC_TH0 =  165; const int M_RUN_EC_AR0 =  222;
const int M_RUN_EC_TH1 =   30; const int M_RUN_EC_AR1 =  324;
const int M_RUN_EP_TH0 =  315; const int M_RUN_EP_AR0 =  857;
const int M_RUN_EP_TH1 =  109; const int M_RUN_EP_AR1 =  867;
const int M_RUN_EM_TH0 =  -14; const int M_RUN_EM_AR0 =  215;
const int M_RUN_EM_TH1 =   61; const int M_RUN_EM_AR1 =   73;
const int M_RUN_EM_LR0 =   35; const int M_RUN_EM_LR1 =   37;
const int M_RUN_EM_LR2 =   42;

const int M_RUN_MS_TH0 = -176; const int M_RUN_MS_AR0 =   14;
const int M_RUN_MS_TH1 = -141; const int M_RUN_MS_AR1 =   21;
const int M_RUN_MC_TH0 =   84; const int M_RUN_MC_AR0 =  172;
const int M_RUN_MC_TH1 =   37; const int M_RUN_MC_AR1 =  263;
const int M_RUN_MP_TH0 =    2; const int M_RUN_MP_AR0 =   15;
const int M_RUN_MP_TH1 = -197; const int M_RUN_MP_AR1 =   20;
const int M_RUN_MM_TH0 =  -27; const int M_RUN_MM_AR0 =  142;
const int M_RUN_MM_TH1 = -146; const int M_RUN_MM_AR1 =   27;
const int M_RUN_MM_LR0 =   51; const int M_RUN_MM_LR1 =   44;
const int M_RUN_MM_LR2 =   80;

const int F_RANK_TS_TH0 = -116; const int F_RANK_TS_AR0 =   33;
const int F_RANK_TS_TH1 =  -78; const int F_RANK_TS_AR1 =   34;
const int F_RANK_TC_TH0 =   -2; const int F_RANK_TC_AR0 =  282;
const int F_RANK_TC_TH1 =   12; const int F_RANK_TC_AR1 =  274;
const int F_RANK_TP_TH0 =    4; const int F_RANK_TP_AR0 =  697;
const int F_RANK_TP_TH1 =   55; const int F_RANK_TP_AR1 = 1185;
const int F_RANK_TM_LR0 =   17; const int F_RANK_TM_LR1 =   14;
const int F_RANK_TM_LR2 =    1;

const int F_RANK_ES_TH0 = -177; const int F_RANK_ES_AR0 =   23;
const int F_RANK_ES_TH1 = -370; const int F_RANK_ES_AR1 =   11;
const int F_RANK_EC_TH0 =  -14; const int F_RANK_EC_AR0 =  271;
const int F_RANK_EC_TH1 =    3; const int F_RANK_EC_AR1 =  308;
const int F_RANK_EP_TH0 =   -3; const int F_RANK_EP_AR0 =  788;
const int F_RANK_EP_TH1 =  135; const int F_RANK_EP_AR1 = 1364;
const int F_RANK_EM_LR0 =   22; const int F_RANK_EM_LR1 =    6;
const int F_RANK_EM_LR2 =    4;

const int F_RANK_MS_TH0 = -254; const int F_RANK_MS_AR0 =   16;
const int F_RANK_MS_TH1 = -177; const int F_RANK_MS_AR1 =   20;
const int F_RANK_MC_TH0 =  -55; const int F_RANK_MC_AR0 =   73;
const int F_RANK_MC_TH1 =  -54; const int F_RANK_MC_AR1 =   74;
const int F_RANK_MP_TH0 =   -6; const int F_RANK_MP_AR0 =  575;
const int F_RANK_MP_TH1 = 1670; const int F_RANK_MP_AR1 = 1173;
const int F_RANK_MM_LR0 =   15; const int F_RANK_MM_LR1 =   10;
const int F_RANK_MM_LR2 =    7;

const int F_RANK_PS_TH0 = -126; const int F_RANK_PS_AR0 =   32;
const int F_RANK_PS_TH1 = -126; const int F_RANK_PS_AR1 =   32;
const int F_RANK_PC_TH0 =  -33; const int F_RANK_PC_AR0 =  120;
const int F_RANK_PC_TH1 =  -25; const int F_RANK_PC_AR1 =  157;
const int F_RANK_PP_TH0 =   -6; const int F_RANK_PP_AR0 =  585;
const int F_RANK_PP_TH1 =  150; const int F_RANK_PP_AR1 =  275;
const int F_RANK_PM_LR0 =   16; const int F_RANK_PM_LR1 =   11;
const int F_RANK_PM_LR2 =    5;

const int F_RUN_TS_TH0 =  -68; const int F_RUN_TS_AR0 =   38;
const int F_RUN_TS_TH1 = -112; const int F_RUN_TS_AR1 =   36;
const int F_RUN_TC_TH0 =   -4; const int F_RUN_TC_AR0 =  221;
const int F_RUN_TC_TH1 =  -13; const int F_RUN_TC_AR1 =  231;
const int F_RUN_TP_TH0 =    0; const int F_RUN_TP_AR0 =    0;
const int F_RUN_TP_TH1 =    0; const int F_RUN_TP_AR1 =    0;
const int F_RUN_TM_LR0 =   14; const int F_RUN_TM_LR1 =   18;
const int F_RUN_TM_LR2 =    0;

const int F_RUN_ES_TH0 =  -90; const int F_RUN_ES_AR0 =   45;
const int F_RUN_ES_TH1 =  -92; const int F_RUN_ES_AR1 =   44;
const int F_RUN_EC_TH0 =   -3; const int F_RUN_EC_AR0 =  325;
const int F_RUN_EC_TH1 =  -11; const int F_RUN_EC_AR1 =  341;
const int F_RUN_EP_TH0 =   24; const int F_RUN_EP_AR0 =  887;
const int F_RUN_EP_TH1 =   -4; const int F_RUN_EP_AR1 =  765;
const int F_RUN_EM_LR0 =   14; const int F_RUN_EM_LR1 =   15;
const int F_RUN_EM_LR2 =    3;

const int F_RUN_MS_TH0 = -275; const int F_RUN_MS_AR0 =   14;
const int F_RUN_MS_TH1 = -185; const int F_RUN_MS_AR1 =   22;
const int F_RUN_MC_TH0 =  -18; const int F_RUN_MC_AR0 =  191;
const int F_RUN_MC_TH1 =  -15; const int F_RUN_MC_AR1 =  241;
const int F_RUN_MP_TH0 =  -73; const int F_RUN_MP_AR0 =   54;
const int F_RUN_MP_TH1 = -214; const int F_RUN_MP_AR1 =   19;
const int F_RUN_MM_LR0 =    7; const int F_RUN_MM_LR1 =   15;
const int F_RUN_MM_LR2 =   10;

struct QlfcStatisticalModel1
{

public:

    struct Rank
    {
        short StaticModel;
        short StateModel[ALPHABET_SIZE];
        short CharModel[ALPHABET_SIZE];

        struct Exponent
        {
            short StaticModel[8];
            short StateModel[ALPHABET_SIZE][8];
            short CharModel[ALPHABET_SIZE][8];
        } Exponent;

        struct Mantissa
        {
            short StaticModel[ALPHABET_SIZE];
            short StateModel[ALPHABET_SIZE][ALPHABET_SIZE];
            short CharModel[ALPHABET_SIZE][ALPHABET_SIZE];
        } Mantissa[8];

        struct Escape
        {
            short StaticModel[ALPHABET_SIZE];
            short StateModel[ALPHABET_SIZE][ALPHABET_SIZE];
            short CharModel[ALPHABET_SIZE][ALPHABET_SIZE];
        } Escape;

    } Rank;

    struct Run
    {
        short StaticModel;
        short StateModel[ALPHABET_SIZE];
        short CharModel[ALPHABET_SIZE];

        struct Exponent
        {
            short StaticModel[32];
            short StateModel[ALPHABET_SIZE][32];
            short CharModel[ALPHABET_SIZE][32];
        } Exponent;

        struct Mantissa
        {
            short StaticModel[32];
            short StateModel[ALPHABET_SIZE][32];
            short CharModel[ALPHABET_SIZE][32];
        } Mantissa[32];

    } Run;

    ProbabilityMixer mixerOfRank[ALPHABET_SIZE];
    ProbabilityMixer mixerOfRankExponent[8][8];
    ProbabilityMixer mixerOfRankMantissa[8];
    ProbabilityMixer mixerOfRankEscape[ALPHABET_SIZE];
    ProbabilityMixer mixerOfRun[ALPHABET_SIZE];
    ProbabilityMixer mixerOfRunExponent[32][32];
    ProbabilityMixer mixerOfRunMantissa[32];
};

struct QlfcStatisticalModel2
{

public:

    struct Rank
    {
        short Exponent[ALPHABET_SIZE][8];
        short Mantissa[ALPHABET_SIZE][8][ALPHABET_SIZE];
    } Rank;

    struct Run
    {
        short Exponent[ALPHABET_SIZE][32];
        short Mantissa[ALPHABET_SIZE][32][32];
    } Run;
};

int  bsc_qlfc_init_static_model();
void bsc_qlfc_init_model(QlfcStatisticalModel1 * model);
void bsc_qlfc_init_model(QlfcStatisticalModel2 * model);

#endif

/*-----------------------------------------------------------*/
/* End                                          qlfc_model.h */
/*-----------------------------------------------------------*/

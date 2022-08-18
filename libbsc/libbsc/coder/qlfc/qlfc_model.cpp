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

#include <stdlib.h>
#include <memory.h>

#include "qlfc_model.h"

#include "../../libbsc.h"
#include "../../platform/platform.h"

QlfcStatisticalModel1 g_QlfcStatisticalModel1;
QlfcStatisticalModel2 g_QlfcStatisticalModel2;

void bsc_qlfc_memset(void * dst, int size, short v)
{
    for (int i = 0; i < size / 2; ++i) ((short *)dst)[i] = v;
}

int bsc_qlfc_init_static_model()
{
    for (int mixer = 0; mixer < ALPHABET_SIZE; ++mixer)
    {
        g_QlfcStatisticalModel1.mixerOfRank[mixer].Init();
        g_QlfcStatisticalModel1.mixerOfRankEscape[mixer].Init();
        g_QlfcStatisticalModel1.mixerOfRun[mixer].Init();
    }
    for (int bit = 0; bit < 8; ++bit)
    {
        g_QlfcStatisticalModel1.mixerOfRankMantissa[bit].Init();
        for (int context = 0; context < 8; ++context)
            g_QlfcStatisticalModel1.mixerOfRankExponent[context][bit].Init();
    }
    for (int bit = 0; bit < 32; ++bit)
    {
        g_QlfcStatisticalModel1.mixerOfRunMantissa[bit].Init();
        for (int context = 0; context < 32; ++context)
            g_QlfcStatisticalModel1.mixerOfRunExponent[context][bit].Init();
    }

    bsc_qlfc_memset(&g_QlfcStatisticalModel1.Rank, sizeof(g_QlfcStatisticalModel1.Rank), 2048);
    bsc_qlfc_memset(&g_QlfcStatisticalModel1.Run, sizeof(g_QlfcStatisticalModel1.Run), 2048);

    bsc_qlfc_memset(&g_QlfcStatisticalModel2.Rank, sizeof(g_QlfcStatisticalModel2.Rank), 4096);
    bsc_qlfc_memset(&g_QlfcStatisticalModel2.Run, sizeof(g_QlfcStatisticalModel2.Run), 1024);

    return LIBBSC_NO_ERROR;
}

void bsc_qlfc_init_model(QlfcStatisticalModel1 * model)
{
    memcpy(model, &g_QlfcStatisticalModel1, sizeof(QlfcStatisticalModel1));
}

void bsc_qlfc_init_model(QlfcStatisticalModel2 * model)
{
    memcpy(model, &g_QlfcStatisticalModel2, sizeof(QlfcStatisticalModel2));
}

/*-----------------------------------------------------------*/
/* End                                        qlfc_model.cpp */
/*-----------------------------------------------------------*/

// Nakamichi is 100% FREE LZSS SUPERFAST decompressor.
// Home of Nakamichi: www.sanmayce.com/Nakamichi/index.html
// Also: http://www.sanmayce.com/Hayabusa/
// Also: http://www.codeproject.com/Articles/878593/Slowest-LZSS-Compressor-in-C

// Okamigan is strongest so far and by far.
// Okami is stronger than Okamiko thanks to the additional 6:3 (64KB window), plus, 14:4 (16MB window) instead of 48:3/24:3/12:3/6:3 (64KB window).
// Okamiko is stronger than Zato thanks to the additional 48:3/24:3/12:3/6:3 (64KB window).

// 狼 Okami, the next...
// Nakamichi 'Zato' is successor to 'Tsubame' which is to 'Tengu-Tsuyo' which is to 'Tengu'.
// 座頭 - Zato
// The character's name is actually Ichi. Zatō is a title, the lowest of the four official ranks within the Tōdōza, the historical guild for blind men. (Thus zato also designates a blind person in Japanese slang.) Ichi is therefore properly called Zatō-no-Ichi ("Low-Ranking Blind Person Ichi", approximately), or Zatōichi for short. Massage was a traditional occupation for the blind (as their lack of sight removed the issue of gender), as was playing the biwa or, for blind women (goze), the shamisen. Being lesser Hinin (lit. "non-people"), blind people and masseurs were regarded as among the very lowest of the low in social class, other than Eta or outright criminals; they were generally considered wretches, beneath notice, no better than beggars or even the insane — especially during the Edo period — and it was also commonly thought that the blind were accursed, despicable, severely mentally disabled, deaf and sexually dangerous.
// Source: https://en.wikipedia.org/wiki/Zatoichi

// Compilation:
/*
Intel(R) Parallel Studio XE 2015
Copyright (C) 1985-2014 Intel Corporation. All rights reserved.

Intel(R) MPI Library 5.0 Update 1 Build Environment for Intel(R) 64 applications
Copyright (C) 2007-2014 Intel Corporation. All rights reserved.


Intel(R) Trace Analyzer and Collector 9.0 Update 1 for Windows* OS for Intel(R) 64 applications
Copyright (C) 1996-2014 Intel Corporation. All rights reserved.

Setting environment for using Microsoft Visual Studio 2010 x64 cross tools.

C:\Program Files (x86)\Intel>cd "Composer XE 2015"

C:\Program Files (x86)\Intel\Composer XE 2015>cd bin

C:\Program Files (x86)\Intel\Composer XE 2015\bin>iclvars.bat
Syntax:
 iclvars.bat <arch> [vs]

   <arch> must be is one of the following
       ia32         : Set up for IA-32 host and target
       ia32_intel64 : Set up for IA-32 host and Intel(R)64 target
       intel64      : Set up for Intel(R) 64 host and target
   If specified, <vs> must be one of the following
       vs2010      : Set to use Microsoft Visual Studio 2010
       vs2010shell : Set to use Microsoft Visual Studio Shell 2010
       vs2012      : Set to use Microsoft Visual Studio 2012
       vs2012shell : Set to use Microsoft Visual Studio Shell 2012
       vs2013      : Set to use Microsoft Visual Studio 2013
   If <vs> is not specified, the version of Visual Studio detected at install
   time is used.


C:\Program Files (x86)\Intel\Composer XE 2015\bin>iclvars.bat intel64
Intel(R) Parallel Studio XE 2015
Copyright (C) 1985-2014 Intel Corporation. All rights reserved.
Intel(R) Parallel Studio XE 2015 Composer Edition (package 108)
Setting environment for using Microsoft Visual Studio 2010 x64 tools.


C:\Program Files (x86)\Intel\Composer XE 2015\bin>d:

D:\>cd D:\TEXTUAL_MADNESS_old\Nakamichi_(Zato)_256MB-Sliding-Window_vs_LZSSE2_bawbaw

D:\TEXTUAL_MADNESS_old\Nakamichi_(Zato)_256MB-Sliding-Window_vs_LZSSE2_bawbaw>dir

09/07/2016  10:50 PM            55,397 lzsse2.cpp
09/07/2016  10:50 PM             4,858 lzsse2.h
09/07/2016  10:50 PM             2,481 lzsse2_platform.h
09/07/2016  10:50 PM             1,285 MakeEXEs_Zato.bat
09/07/2016  10:50 PM             1,632 MokujIN 224 prompt.lnk
09/18/2016  02:25 PM           379,652 Nakamichi_Zato.c
09/07/2016  10:50 PM               225 _DENAKAMICHIZE.BAT
09/07/2016  10:50 PM               113 _NAKAMICHIZE.BAT

D:\TEXTUAL_MADNESS_old\Nakamichi_(Zato)_256MB-Sliding-Window_vs_LZSSE2_bawbaw>MakeEXEs_Zato.bat

D:\TEXTUAL_MADNESS_old\Nakamichi_(Zato)_256MB-Sliding-Window_vs_LZSSE2_bawbaw>icl /TP /O3 /QxSSE4.1 Nakamichi_Zato.c lzsse2.cpp -D_N_XMM -D_N_prefetch_4096 -D_N_HIGH_PRIORITY /FAcs
Intel(R) C++ Intel(R) 64 Compiler XE for applications running on Intel(R) 64, Version 15.0.0.108 Build 20140726
Copyright (C) 1985-2014 Intel Corporation.  All rights reserved.

Nakamichi_Zato.c
lzsse2.cpp
Microsoft (R) Incremental Linker Version 10.00.30319.01
Copyright (C) Microsoft Corporation.  All rights reserved.

-out:Nakamichi_Zato.exe
Nakamichi_Zato.obj
lzsse2.obj

D:\TEXTUAL_MADNESS_old\Nakamichi_(Zato)_256MB-Sliding-Window_vs_LZSSE2_bawbaw>icl /TP /O3 /QxAVX Nakamichi_Zato.c lzsse2.cpp -D_N_XMM -D_N_prefetch_4096 -D_N_HIGH_PRIORITY /FAcs
Intel(R) C++ Intel(R) 64 Compiler XE for applications running on Intel(R) 64, Version 15.0.0.108 Build 20140726
Copyright (C) 1985-2014 Intel Corporation.  All rights reserved.

Nakamichi_Zato.c
lzsse2.cpp
Microsoft (R) Incremental Linker Version 10.00.30319.01
Copyright (C) Microsoft Corporation.  All rights reserved.

-out:Nakamichi_Zato.exe
Nakamichi_Zato.obj
lzsse2.obj

D:\TEXTUAL_MADNESS_old\Nakamichi_(Zato)_256MB-Sliding-Window_vs_LZSSE2_bawbaw>dir

09/07/2016  10:50 PM            55,397 lzsse2.cpp
09/07/2016  10:50 PM             4,858 lzsse2.h
09/19/2016  02:12 PM            18,554 lzsse2.obj
09/19/2016  02:12 PM           697,361 lzsse2_64bit_AVX.cod
09/19/2016  02:12 PM           741,193 lzsse2_64bit_SSE41.cod
09/07/2016  10:50 PM             2,481 lzsse2_platform.h
09/07/2016  10:50 PM             1,285 MakeEXEs_Zato.bat
09/07/2016  10:50 PM             1,632 MokujIN 224 prompt.lnk
09/18/2016  02:25 PM           379,652 Nakamichi_Zato.c
09/19/2016  02:12 PM            55,001 Nakamichi_Zato.obj
09/19/2016  02:12 PM         1,412,792 Nakamichi_Zato_XMM_PREFETCH_4096_Intel_15.0_64bit_AVX.cod
09/19/2016  02:12 PM           146,944 Nakamichi_Zato_XMM_PREFETCH_4096_Intel_15.0_64bit_AVX.exe
09/19/2016  02:12 PM         1,408,383 Nakamichi_Zato_XMM_PREFETCH_4096_Intel_15.0_64bit_SSE41.cod
09/19/2016  02:12 PM           154,624 Nakamichi_Zato_XMM_PREFETCH_4096_Intel_15.0_64bit_SSE41.exe
09/07/2016  10:50 PM               225 _DENAKAMICHIZE.BAT
09/07/2016  10:50 PM               113 _NAKAMICHIZE.BAT

D:\TEXTUAL_MADNESS_old\Nakamichi_(Zato)_256MB-Sliding-Window_vs_LZSSE2_bawbaw>
*/

// 2016-Apr-04: Grrr... Stupid bug (due to overlooking) was crushed in 'TT', namely:
/*
//#define Min_Match_Length (32)
  #define Min_Match_Length (4)
*/
// becomes:
/*
//#define Min_Match_Length (32)
//#define Min_Match_Length (4)
  #define Min_Match_Length (16) // Maximum MatchLength is 16, it decides the size of Look-ahead buffer - to avoid search beyound end. This needs more attention in the future - to clarify it fully. Overlapping is also yet to come.
*/

// 2016-Apr-03: Finished the unfinished 2016-Mar-31 draft. Stupid stats bug was fixed TargetSize -> VerifySize in line: k = (((float)1000*VerifySize/(clocks2 - clocks1 + 1))); k=k>>20; k=k*Trials;
// 2016-Mar-31: added the FASTEST decompressor - LZSSE2 - to juxtapose two XMM vs XMM etudes, below the compile lines:
/*
D:\_The_Usual_Suspects\Nakamichi_(Tengu-Tsuyo)_1MB-Sliding-Window_vs_LZSSE2>MakeEXEs_Tengu-Tsuyo.bat

D:\_The_Usual_Suspects\Nakamichi_(Tengu-Tsuyo)_1MB-Sliding-Window_vs_LZSSE2>icl /TP /O3 /QxSSE4.1 Nakamichi_Tengu-Tsuyo.c lzsse2.cpp -D_N_XMM -D_N_prefetch_4096 -D_N_HIGH_PRIORITY /FAcs
Intel(R) C++ Intel(R) 64 Compiler XE for applications running on Intel(R) 64, Version 15.0.0.108 Build 20140726
Copyright (C) 1985-2014 Intel Corporation.  All rights reserved.

Nakamichi_Tengu-Tsuyo.c
lzsse2.cpp
Microsoft (R) Incremental Linker Version 10.00.30319.01
Copyright (C) Microsoft Corporation.  All rights reserved.

-out:Nakamichi_Tengu-Tsuyo.exe
Nakamichi_Tengu-Tsuyo.obj
lzsse2.obj

D:\_The_Usual_Suspects\Nakamichi_(Tengu-Tsuyo)_1MB-Sliding-Window_vs_LZSSE2>icl /TP /O3 /QxAVX Nakamichi_Tengu-Tsuyo.c lzsse2.cpp -D_N_XMM -D_N_prefetch_4096 -D_N_HIGH_PRIORITY /FAcs
Intel(R) C++ Intel(R) 64 Compiler XE for applications running on Intel(R) 64, Version 15.0.0.108 Build 20140726
Copyright (C) 1985-2014 Intel Corporation.  All rights reserved.

Nakamichi_Tengu-Tsuyo.c
lzsse2.cpp
Microsoft (R) Incremental Linker Version 10.00.30319.01
Copyright (C) Microsoft Corporation.  All rights reserved.

-out:Nakamichi_Tengu-Tsuyo.exe
Nakamichi_Tengu-Tsuyo.obj
lzsse2.obj

D:\_The_Usual_Suspects\Nakamichi_(Tengu-Tsuyo)_1MB-Sliding-Window_vs_LZSSE2>dir Nakamichi_Tengu-Tsuyo*
 Volume in drive D is S640_Vol5
 Volume Serial Number is 5861-9E6C

 Directory of D:\_The_Usual_Suspects\Nakamichi_(Tengu-Tsuyo)_1MB-Sliding-Window_vs_LZSSE2

03/31/2016  12:50 AM           107,458 Nakamichi_Tengu-Tsuyo.c
03/27/2016  11:04 PM         1,234,944 Nakamichi_Tengu-Tsuyo.doc
03/31/2016  12:57 AM           948,131 Nakamichi_Tengu-Tsuyo_XMM_PREFETCH_4096_Intel_15.0_64bit_AVX.cod
03/31/2016  12:57 AM           139,264 Nakamichi_Tengu-Tsuyo_XMM_PREFETCH_4096_Intel_15.0_64bit_AVX.exe
03/31/2016  12:57 AM           943,556 Nakamichi_Tengu-Tsuyo_XMM_PREFETCH_4096_Intel_15.0_64bit_SSE41.cod
03/31/2016  12:57 AM           146,432 Nakamichi_Tengu-Tsuyo_XMM_PREFETCH_4096_Intel_15.0_64bit_SSE41.exe
               6 File(s)      3,519,785 bytes
               0 Dir(s)   8,169,893,888 bytes free

D:\_The_Usual_Suspects\Nakamichi_(Tengu-Tsuyo)_1MB-Sliding-Window_vs_LZSSE2>dir lzs*
 Volume in drive D is S640_Vol5
 Volume Serial Number is 5861-9E6C

 Directory of D:\_The_Usual_Suspects\Nakamichi_(Tengu-Tsuyo)_1MB-Sliding-Window_vs_LZSSE2

03/30/2016  10:05 PM            55,397 lzsse2.cpp
03/30/2016  10:17 PM             4,858 lzsse2.h
03/31/2016  12:57 AM           690,697 lzsse2_64bit_AVX.cod
03/31/2016  12:57 AM           734,033 lzsse2_64bit_SSE41.cod
03/27/2016  07:49 AM             2,481 lzsse2_platform.h
               5 File(s)      1,487,466 bytes
               0 Dir(s)   8,169,893,888 bytes free

D:\_The_Usual_Suspects\Nakamichi_(Tengu-Tsuyo)_1MB-Sliding-Window_vs_LZSSE2>
*/

// 2016-Mar-19: Tiny improvement in encoding (just forgot to refresh two define's), namely:
/*
//#define Min_Match_Length (32)
  #define Min_Match_Length (4)
//#define OffsetBITS (32-3)
  #define OffsetBITS (29)
*/

// How to compile:
// icl /O3 /QxSSE2 Nakamichi_Tengu-Tsuyo.c -D_N_XMM -D_N_prefetch_4096 -D_N_HIGH_PRIORITY /FAcs

// Nakamichi_Tengu-Tsuyo.c, using 16B/4KB/1MB or (8-4)bit/(16-4)bit/(24-4)bit windows with 1/2/3 bytes long offsets.
// The variant 'Tsuyo' tries next position and if it gives higher ratio it is used.

// 中道 nakamichi [noun]
// English Meaning(s) for 中道
//  1. road through the middle; middle road
// Meanings for each kanji in 中道
// 中 in; inside; middle; mean; center
// 道 road-way; street; district; journey; course; moral; teachings

// 天狗 tengu
// heavenly dog

// 強 - Tsuyo
// https://glosbe.com/ja/en/%E5%BC%B7
// Translations into English:
//    JMdict:
//    a little more than   
//    JMdict:
//    a little over   
//    JMdict:
//    military build-up   
//    JMdict:
//    one of the biggest   
//    JMdict:
//    one of the most powerful   
//    JMdict:
//    powerhouse

// The first Nakamichi was 'Kaidanji', based on Nobuo Ito's source, thanks Ito.
// The main goal of Nakamichi is to allow supersimple and superfast decoding for English x-grams (mainly) in pure C, or not, heh-heh.
// Natively Nakamichi is targeted as 64bit tool with 16 threads, helping Kazahana to traverse faster when I/O is not superior.
// In short, Nakamichi is intended as x-gram decompressor.

// Eightfold Path ~ the Buddhist path to nirvana, comprising eight aspects in which an aspirant must become practised; 
// eightfold way ~ (Physics), the grouping of hadrons into supermultiplets by means of SU(3)); (b) adverb to eight times the number or quantity: OE.

// Tengu-Tsuyo vs 7-Zip 9.20 Deflate stats:
/*
D:\_KAZE>7za.exe a -tzip -mx9 Silesia_compression_corpus.tar.zip Silesia_compression_corpus.tar
7-Zip (A) 9.20  Copyright (c) 1999-2010 Igor Pavlov  2010-11-18
...

07/06/2015  01:34 AM        33,258,496 Agatha_Christie_89-ebooks_TXT.tar
07/02/2015  01:19 AM        12,081,163 Agatha_Christie_89-ebooks_TXT.tar.Nakamichi
07/06/2015  01:33 AM        11,173,343 Agatha_Christie_89-ebooks_TXT.tar.zip
07/05/2015  05:01 AM       118,737,920 ebooks.adelaide.edu.au_104-HTMs(Complete).tar
07/04/2015  10:19 PM        65,783,540 ebooks.adelaide.edu.au_104-HTMs(Complete).tar.Nakamichi
07/06/2015  01:38 AM        61,720,058 ebooks.adelaide.edu.au_104-HTMs(Complete).tar.zip
07/04/2015  06:38 AM       100,000,000 enwik8
07/02/2015  12:48 PM        40,432,833 enwik8.Nakamichi
07/06/2015  01:40 AM        35,103,012 enwik8.zip
07/02/2015  10:33 PM       211,948,032 Silesia_compression_corpus.tar
07/03/2015  08:43 PM        82,842,620 Silesia_compression_corpus.tar.Nakamichi
07/06/2015  01:46 AM        64,730,815 Silesia_compression_corpus.tar.zip
07/02/2015  04:51 PM         3,265,536 University_of_Canterbury_The_Calgary_Corpus.tar
07/02/2015  05:03 PM         1,333,372 University_of_Canterbury_The_Calgary_Corpus.tar.Nakamichi
07/06/2015  01:53 AM         1,017,820 University_of_Canterbury_The_Calgary_Corpus.tar.zip
07/02/2015  04:51 PM         2,821,120 University_of_Canterbury_The_Canterbury_Corpus.tar
07/02/2015  05:15 PM         1,032,077 University_of_Canterbury_The_Canterbury_Corpus.tar.Nakamichi
07/06/2015  01:53 AM           675,007 University_of_Canterbury_The_Canterbury_Corpus.tar.zip
07/02/2015  04:52 PM        11,162,624 University_of_Canterbury_The_Large_Corpus.tar
07/02/2015  06:13 PM         3,634,266 University_of_Canterbury_The_Large_Corpus.tar.Nakamichi
07/06/2015  01:54 AM         3,042,350 University_of_Canterbury_The_Large_Corpus.tar.zip
02/02/2003  09:16 PM         4,067,439 www.maximumcompression.com_english.dic
07/06/2015  02:16 AM         1,381,492 www.maximumcompression.com_english.dic.Nakamichi
07/06/2015  01:59 AM           890,085 www.maximumcompression.com_english.dic.zip
*/

void x64toaKAZE (      /* stdcall is faster and smaller... Might as well use it for the helper. */
        unsigned long long val,
        char *buf,
        unsigned radix,
        int is_neg
        )
{
        char *p;                /* pointer to traverse string */
        char *firstdig;         /* pointer to first digit */
        char temp;              /* temp char */
        unsigned digval;        /* value of digit */

        p = buf;

        if ( is_neg )
        {
            *p++ = '-';         /* negative, so output '-' and negate */
            val = (unsigned long long)(-(long long)val);
        }

        firstdig = p;           /* save pointer to first digit */

        do {
            digval = (unsigned) (val % radix);
            val /= radix;       /* get next digit */

            /* convert to ascii and store */
            if (digval > 9)
                *p++ = (char) (digval - 10 + 'a');  /* a letter */
            else
                *p++ = (char) (digval + '0');       /* a digit */
        } while (val > 0);

        /* We now have the digit of the number in the buffer, but in reverse
           order.  Thus we reverse them now. */

        *p-- = '\0';            /* terminate string; p points to last digit */

        do {
            temp = *p;
            *p = *firstdig;
            *firstdig = temp;   /* swap *p and *firstdig */
            --p;
            ++firstdig;         /* advance to next two digits */
        } while (firstdig < p); /* repeat until halfway */
}

/* Actual functions just call conversion helper with neg flag set correctly,
   and return pointer to buffer. */

char * _i64toaKAZE (
        long long val,
        char *buf,
        int radix
        )
{
        x64toaKAZE((unsigned long long)val, buf, radix, (radix == 10 && val < 0));
        return buf;
}

char * _ui64toaKAZE (
        unsigned long long val,
        char *buf,
        int radix
        )
{
        x64toaKAZE(val, buf, radix, 0);
        return buf;
}

char * _ui64toaKAZEzerocomma (
        unsigned long long val,
        char *buf,
        int radix
        )
{
                        char *p;
                        char temp;
                        int txpman;
                        int pxnman;
        x64toaKAZE(val, buf, radix, 0);
                        p = buf;
                        do {
                        } while (*++p != '\0');
                        p--; // p points to last digit
                             // buf points to first digit
                        buf[26] = 0;
                        txpman = 1;
                        pxnman = 0;
                        do
                        { if (buf <= p)
                          { temp = *p;
                            buf[26-txpman] = temp; pxnman++;
                            p--;
                            if (pxnman % 3 == 0)
                            { txpman++;
                              buf[26-txpman] = (char) (',');
                            }
                          }
                          else
                          { buf[26-txpman] = (char) ('0'); pxnman++;
                            if (pxnman % 3 == 0)
                            { txpman++;
                              buf[26-txpman] = (char) (',');
                            }
                          }
                          txpman++;
                        } while (txpman <= 26);
        return buf;
}

char * _ui64toaKAZEcomma (
        unsigned long long val,
        char *buf,
        int radix
        )
{
                        char *p;
                        char temp;
                        int txpman;
                        int pxnman;
        x64toaKAZE(val, buf, radix, 0);
                        p = buf;
                        do {
                        } while (*++p != '\0');
                        p--; // p points to last digit
                             // buf points to first digit
                        buf[26] = 0;
                        txpman = 1;
                        pxnman = 0;
                        while (buf <= p)
                        { temp = *p;
                          buf[26-txpman] = temp; pxnman++;
                          p--;
                          if (pxnman % 3 == 0 && buf <= p)
                          { txpman++;
                            buf[26-txpman] = (char) (',');
                          }
                          txpman++;
                        } 
        return buf+26-(txpman-1);
}

char * _ui64toaKAZEzerocomma4 (
        unsigned long long val,
        char *buf,
        int radix
        )
{
                        char *p;
                        char temp;
                        int txpman;
                        int pxnman;
        x64toaKAZE(val, buf, radix, 0);
                        p = buf;
                        do {
                        } while (*++p != '\0');
                        p--; // p points to last digit
                             // buf points to first digit
                        buf[26] = 0;
                        txpman = 1;
                        pxnman = 0;
                        do
                        { if (buf <= p)
                          { temp = *p;
                            buf[26-txpman] = temp; pxnman++;
                            p--;
                            if (pxnman % 4 == 0)
                            { txpman++;
                              buf[26-txpman] = (char) (',');
                            }
                          }
                          else
                          { buf[26-txpman] = (char) ('0'); pxnman++;
                            if (pxnman % 4 == 0)
                            { txpman++;
                              buf[26-txpman] = (char) (',');
                            }
                          }
                          txpman++;
                        } while (txpman <= 26);
        return buf;
}

/* minimum signed 64 bit value */
#define _I64_MIN    (-9223372036854775807i64 - 1)
/* maximum signed 64 bit value */
#define _I64_MAX      9223372036854775807i64
/* maximum unsigned 64 bit value */
#define _UI64_MAX     0xffffffffffffffffui64

/* minimum signed 128 bit value */
#define _I128_MIN   (-170141183460469231731687303715884105727i128 - 1)
/* maximum signed 128 bit value */
#define _I128_MAX     170141183460469231731687303715884105727i128
/* maximum unsigned 128 bit value */
#define _UI128_MAX    0xffffffffffffffffffffffffffffffffui128

      char llTOaDigits[27]; // 9,223,372,036,854,775,807: 1(sign or carry)+19(digits)+1('\0')+6(,)
      // below duplicates are needed because of one_line_invoking need different buffers.
      char llTOaDigits2[27]; // 9,223,372,036,854,775,807: 1(sign or carry)+19(digits)+1('\0')+6(,)
      char llTOaDigits3[27]; // 9,223,372,036,854,775,807: 1(sign or carry)+19(digits)+1('\0')+6(,)
      char llTOaDigits4[27]; // 9,223,372,036,854,775,807: 1(sign or carry)+19(digits)+1('\0')+6(,)

// During compilation use one of these, the granularity of the padded 'memcpy', 4x2x8/2x2x16/1x2x32/1x1x64 respectively as GP/XMM/YMM/ZMM, the maximum literal length reduced from 127 to 63:
//#define _N_GP
#define _N_XMM
//#define _N_YMM
//#define _N_ZMM

//#define _N_prefetch_64
//#define _N_prefetch_128
//#define _N_prefetch_4096

//Only one must be uncommented:
#define _WIN32_ENVIRONMENT_
//#define _POSIX_ENVIRONMENT_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> // uint64_t needed
#include <time.h>
#include <string.h>

      clock_t clocks1, clocks2;

#if defined(_WIN32)
#include <io.h> // needed for Windows' 'lseeki64' and 'telli64'
//Above line must be commented in order to compile with Intel C compiler: an error "can't find io.h" occurs.
#else
#endif /* defined(_WIN32_ENVIRONMENT_)  */

#ifdef _N_XMM
#include <emmintrin.h> // SSE2 intrinsics
#include <smmintrin.h> // SSE4.1 intrinsics
#endif
#ifdef _N_YMM
#include <emmintrin.h> // SSE2 intrinsics
#include <smmintrin.h> // SSE4.1 intrinsics
#include <immintrin.h> // AVX intrinsics
#endif
#ifdef _N_ZMM
#include <emmintrin.h> // SSE2 intrinsics
#include <smmintrin.h> // SSE4.1 intrinsics
#include <immintrin.h> // AVX intrinsics
#include <zmmintrin.h> // AVX2 intrinsics, definitions and declarations for use with 512-bit compiler intrinsics.
#endif

#ifdef _N_XMM
void SlowCopy128bit (const char *SOURCE, char *TARGET) { _mm_storeu_si128((__m128i *)(TARGET), _mm_loadu_si128((const __m128i *)(SOURCE))); }
void NotSoSlowCopy128bit (const char *SOURCE, char *TARGET) { _mm_storeu_si128((__m128i *)(TARGET), _mm_lddqu_si128((const __m128i *)(SOURCE))); }
#endif
#ifdef _N_YMM
void SlowCopy128bit (const char *SOURCE, char *TARGET) { _mm_storeu_si128((__m128i *)(TARGET), _mm_loadu_si128((const __m128i *)(SOURCE))); }
#endif
#ifdef _N_ZMM
void SlowCopy128bit (const char *SOURCE, char *TARGET) { _mm_storeu_si128((__m128i *)(TARGET), _mm_loadu_si128((const __m128i *)(SOURCE))); }
#endif
/*
 * Move Unaligned Packed Integer Values
 * **** VMOVDQU ymm1, m256
 * **** VMOVDQU m256, ymm1
 * Moves 256 bits of packed integer values from the source operand to the
 * destination
 */
//extern __m256i __ICL_INTRINCC _mm256_loadu_si256(__m256i const *);
//extern void    __ICL_INTRINCC _mm256_storeu_si256(__m256i *, __m256i);
#ifdef _N_YMM
void SlowCopy256bit (const char *SOURCE, char *TARGET) { _mm256_storeu_si256((__m256i *)(TARGET), _mm256_loadu_si256((const __m256i *)(SOURCE))); }
#endif
//extern __m512i __ICL_INTRINCC _mm512_loadu_si512(void const*);
//extern void    __ICL_INTRINCC _mm512_storeu_si512(void*, __m512i);
#ifdef _N_ZMM
void SlowCopy512bit (const char *SOURCE, char *TARGET) { _mm512_storeu_si512((__m512i *)(TARGET), _mm512_loadu_si512((const __m512i *)(SOURCE))); }
#endif

#ifndef NULL
#define NULL ((void*)0)
#endif

// Comment it to see how slower 'BruteForce' is, for Wikipedia 100MB the ratio is 41KB/s versus 197KB/s.
#define ReplaceBruteForceWithRailgunSwampshineBailOut

void SearchIntoSlidingWindow(unsigned int* ShortMediumLongOFFSET, unsigned int* retIndex, unsigned int* retMatch, char* refStart,char* refEnd,char* encStart,char* encEnd);
unsigned int SlidingWindowVsLookAheadBuffer(char* refStart, char* refEnd, char* encStart, char* encEnd);
uint64_t NakaCompress(char* ret, char* src, uint64_t srcSize);
//unsigned int NakaDecompress(char* ret, char* src, unsigned int srcSize);
uint64_t NakaDecompress(char* ret, char* src, uint64_t srcSize);
char * Railgun_Trolldom(char * pbTarget, char * pbPattern, uint32_t cbTarget, uint32_t cbPattern);
char * Railgun_Doublet (char * pbTarget, char * pbPattern, uint32_t cbTarget, uint32_t cbPattern);
char * Railgun_BawBaw_reverse(char * pbTarget, char * pbPattern, uint32_t cbTarget, uint32_t cbPattern);
char * Railgun_Baw_reverse(char * pbTarget, char * pbPattern, uint32_t cbTarget, uint32_t cbPattern);

/*
void memcpy_AVX_4K_prefetched (void *dst, const void *src, size_t nbytes) {
// F3 0F 6F /r RM V/V SSE2 Move unaligned packed integer values from xmm2/m128 to xmm1.
// MOVDQU xmm1, xmm2/m128 
// F3 0F 7F /r MR V/V SSE2 Move unaligned packed integer values from xmm1 to xmm2/m128.
// MOVDQU xmm2/m128, xmm1 
// VEX.128.F3.0F.WIG 6F /r RM V/V AVX Move unaligned packed integer values from xmm2/m128 to xmm1.
// VMOVDQU xmm1, xmm2/m128 
// VEX.128.F3.0F.WIG 7F /r MR V/V AVX Move unaligned packed integer values from xmm1 to xmm2/m128.
// VMOVDQU xmm2/m128, xmm1 
// VEX.256.F3.0F.WIG 6F /r RM V/V AVX Move unaligned packed integer values from ymm2/m256 to ymm1.
// VMOVDQU ymm1, ymm2/m256 
// VEX.256.F3.0F.WIG 7F /r MR V/V AVX Move unaligned packed integer values from ymm1 to ymm2/m256.
// VMOVDQU ymm2/m256, ymm1 
if ( (nbytes&0x3f) == 0 ) { // 64bytes per cycle
	__asm{
		mov		rsi, src
	        mov		rdi, dst
	        mov		rcx, nbytes
	        shr		rcx, 6
	main_loop:
		test		rcx, rcx ; 'nbytes' may be 0
	        jz		main_loop_end
	        prefetcht0	[rsi+64*64]
		vmovdqu 	xmm0, [rsi]
	        vmovdqu 	xmm1, [rsi+16]
	        vmovdqu 	xmm2, [rsi+32]
	        vmovdqu 	xmm3, [rsi+48]
		vmovdqu 	[rdi], xmm0
	        vmovdqu 	[rdi+16], xmm1
	        vmovdqu 	[rdi+32], xmm2
	        vmovdqu 	[rdi+48], xmm3
	        add		rsi, 64
	        add		rdi, 64
		dec		rcx
	        jmp		main_loop
	main_loop_end:
	        sfence
	}
} else memcpy(dst, src, nbytes);
}

void memcpy_SSE2_4K_prefetched (void *dst, const void *src, size_t nbytes) {
// F3 0F 6F /r RM V/V SSE2 Move unaligned packed integer values from xmm2/m128 to xmm1.
// MOVDQU xmm1, xmm2/m128 
// F3 0F 7F /r MR V/V SSE2 Move unaligned packed integer values from xmm1 to xmm2/m128.
// MOVDQU xmm2/m128, xmm1 
// VEX.128.F3.0F.WIG 6F /r RM V/V AVX Move unaligned packed integer values from xmm2/m128 to xmm1.
// VMOVDQU xmm1, xmm2/m128 
// VEX.128.F3.0F.WIG 7F /r MR V/V AVX Move unaligned packed integer values from xmm1 to xmm2/m128.
// VMOVDQU xmm2/m128, xmm1 
// VEX.256.F3.0F.WIG 6F /r RM V/V AVX Move unaligned packed integer values from ymm2/m256 to ymm1.
// VMOVDQU ymm1, ymm2/m256 
// VEX.256.F3.0F.WIG 7F /r MR V/V AVX Move unaligned packed integer values from ymm1 to ymm2/m256.
// VMOVDQU ymm2/m256, ymm1 
if ( (nbytes&0x3f) == 0 ) { // 64bytes per cycle
	__asm{
		mov		rsi, src
	        mov		rdi, dst
	        mov		rcx, nbytes
	        shr		rcx, 6
	main_loop:
		test		rcx, rcx ; 'nbytes' may be 0
	        jz		main_loop_end
	        prefetcht0	[rsi+64*64]
		movdqu 		xmm0, [rsi]
	        movdqu 		xmm1, [rsi+16]
	        movdqu 		xmm2, [rsi+32]
	        movdqu 		xmm3, [rsi+48]
		movdqu 		[rdi], xmm0
	        movdqu 		[rdi+16], xmm1
	        movdqu 		[rdi+32], xmm2
	        movdqu 		[rdi+48], xmm3
	        add		rsi, 64
	        add		rdi, 64
		dec		rcx
	        jmp		main_loop
	main_loop_end:
	        sfence
	}
} else memcpy(dst, src, nbytes);
}
*/

#ifdef _N_HIGH_PRIORITY
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms686219.aspx
#include <stdio.h>
#include <windows.h>
#include <tchar.h>
#endif

/*
int main( void )
{
   DWORD dwError, dwPriClass;

   if(!SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS))
   {
      _tprintf(TEXT("Already REALTIME_PRIORITY\n"));
      goto Cleanup;
   } 

   // Display priority class

   dwPriClass = GetPriorityClass(GetCurrentProcess());

   _tprintf(TEXT("Current priority class is 0x%x\n"), dwPriClass);

   if (dwPriClass==0x00000100) printf("Current priority class is REALTIME_PRIORITY_CLASS.\n");


Cleanup:
   // Clean up
   ;
return 0;
}

// IDLE_PRIORITY_CLASS
// 0x00000040
// Process whose threads run only when the system is idle. The threads of the process are preempted by the threads of any process running in a higher priority class. An example is a screen saver. The idle-priority class is inherited by child processes.

// NORMAL_PRIORITY_CLASS
// 0x00000020
// Process with no special scheduling needs.

// HIGH_PRIORITY_CLASS
// 0x00000080
// Process that performs time-critical tasks that must be executed immediately. The threads of the process preempt the threads of normal or idle priority class processes. An example is the Task List, which must respond quickly when called by the user, regardless of the load on the operating system. Use extreme care when using the high-priority class, because a high-priority class application can use nearly all available CPU time.

// REALTIME_PRIORITY_CLASS
// 0x00000100
// Process that has the highest possible priority. The threads of the process preempt the threads of all other processes, including operating system processes performing important tasks. For example, a real-time process that executes for more than a very brief interval can cause disk caches not to flush or cause the mouse to be unresponsive.
*/

// Min_Match_Length=THRESHOLD=4 means 4 and bigger are to be encoded:
#define Min_Match_BAILOUT_Length (8)
#define Min_Match_Length (48)
//#define Min_Match_Length (4)
//#define Min_Match_Length (16) // Maximum MatchLength is 16, it decides the size of Look-ahead buffer - to avoid search beyound end. This needs more attention in the future - to clarify it fully. Overlapping is also yet to come.
#define Min_Match_Length_SHORT (5)
#define OffsetBITS (32-3)
  //#define OffsetBITS (20)
#define LengthBITS (1)

//12bit
//#define REF_SIZE (4095+Min_Match_Length)
//#define REF_SIZE ( ((1<<OffsetBITS)-1) + Min_Match_Length )
#define REF_SIZE ( ((1<<OffsetBITS)-1) )
//3bit
//#define ENC_SIZE (7+Min_Match_Length)
#define ENC_SIZE ( ((1<<LengthBITS)-1) + Min_Match_Length )

#define _rotl_KAZE(x, n) (((x) << (n)) | ((x) >> (32-(n))))
#define _rotl_KAZE64(x, n) (((x) << (n)) | ((x) >> (64-(n))))

uint32_t FNV1A_Hash_YoshimitsuTRIADii(const char *str, uint64_t wrdlen)
{
    const uint32_t PRIME = 709607;
    uint32_t hash32 = 2166136261;
    uint32_t hash32B = 2166136261;
    uint32_t hash32C = 2166136261;
    const char *p = str;
    uint32_t Loop_Counter;
    uint32_t Second_Line_Offset;

if (wrdlen >= 24) {
    Loop_Counter = (wrdlen/24);
    Loop_Counter++;
    Second_Line_Offset = wrdlen-(Loop_Counter)*(3*4);
    for(; Loop_Counter; Loop_Counter--, p += 3*sizeof(uint32_t)) {
		hash32 = (hash32 ^ (_rotl_KAZE(*(uint32_t *)(p+0),5) ^ *(uint32_t *)(p+0+Second_Line_Offset))) * PRIME;        
		hash32B = (hash32B ^ (_rotl_KAZE(*(uint32_t *)(p+4+Second_Line_Offset),5) ^ *(uint32_t *)(p+4))) * PRIME;        
		hash32C = (hash32C ^ (_rotl_KAZE(*(uint32_t *)(p+8),5) ^ *(uint32_t *)(p+8+Second_Line_Offset))) * PRIME;        
    }
		hash32 = (hash32 ^ _rotl_KAZE(hash32C,5) ) * PRIME;
} else {
    // 1111=15; 10111=23
    if (wrdlen & 4*sizeof(uint32_t)) {	
		hash32 = (hash32 ^ (_rotl_KAZE(*(uint32_t *)(p+0),5) ^ *(uint32_t *)(p+4))) * PRIME;        
		hash32B = (hash32B ^ (_rotl_KAZE(*(uint32_t *)(p+8),5) ^ *(uint32_t *)(p+12))) * PRIME;        
		p += 8*sizeof(uint16_t);
    }
    // Cases: 0,1,2,3,4,5,6,7,...,15
    if (wrdlen & 2*sizeof(uint32_t)) {
		hash32 = (hash32 ^ *(uint32_t*)(p+0)) * PRIME;
		hash32B = (hash32B ^ *(uint32_t*)(p+4)) * PRIME;
		p += 4*sizeof(uint16_t);
    }
    // Cases: 0,1,2,3,4,5,6,7
    if (wrdlen & sizeof(uint32_t)) {
		hash32 = (hash32 ^ *(uint16_t*)(p+0)) * PRIME;
		hash32B = (hash32B ^ *(uint16_t*)(p+2)) * PRIME;
		p += 2*sizeof(uint16_t);
    }
    if (wrdlen & sizeof(uint16_t)) {
        hash32 = (hash32 ^ *(uint16_t*)p) * PRIME;
        p += sizeof(uint16_t);
    }
    if (wrdlen & 1) 
        hash32 = (hash32 ^ *p) * PRIME;
}
    hash32 = (hash32 ^ _rotl_KAZE(hash32B,5) ) * PRIME;
    return hash32 ^ (hash32 >> 16);
}

uint32_t FNV1A_Hash_YoshimitsuTRIAD(const char *str, uint64_t wrdlen)
{
    const uint32_t PRIME = 709607;
    uint32_t hash32 = 2166136261;
    uint32_t hash32B = 2166136261;
    uint32_t hash32C = 2166136261;
    //uint32_t hash32D = 2166136261;
    const char *p = str;

    for(; wrdlen >= 3*2*sizeof(uint32_t); wrdlen -= 3*2*sizeof(uint32_t), p += 3*2*sizeof(uint32_t)) {
		hash32 = (hash32 ^ (_rotl_KAZE(*(uint32_t *)(p+0),5) ^ *(uint32_t *)(p+4))) * PRIME;        
		hash32B = (hash32B ^ (_rotl_KAZE(*(uint32_t *)(p+8),5) ^ *(uint32_t *)(p+12))) * PRIME;        
		hash32C = (hash32C ^ (_rotl_KAZE(*(uint32_t *)(p+16),5) ^ *(uint32_t *)(p+20))) * PRIME;        
		//hash32D = (hash32D ^ (_rotl_KAZE(*(uint32_t *)(p+24),5) ^ *(uint32_t *)(p+28))) * PRIME;        

/*
// Microsoft (R) 32-bit C/C++ Optimizing Compiler Version 16.00.30319.01 for 80x86 gave this:
// 12d-0f4+2= 59 bytes, No CARAMBA anymore.
$LL9@FNV1A_Hash@2:

; 160  : 		hash32 = (hash32 ^ (_rotl_KAZE(*(uint32_t *)(p+0),5) ^ *(uint32_t *)(p+4))) * PRIME;        

  000f4	8b 01		 mov	 eax, DWORD PTR [ecx]
  000f6	c1 c0 05	 rol	 eax, 5
  000f9	33 41 04	 xor	 eax, DWORD PTR [ecx+4]
  000fc	83 eb 18	 sub	 ebx, 24			; 00000018H
  000ff	33 f0		 xor	 esi, eax

; 161  : 		hash32B = (hash32B ^ (_rotl_KAZE(*(uint32_t *)(p+8),5) ^ *(uint32_t *)(p+12))) * PRIME;        

  00101	8b 41 08	 mov	 eax, DWORD PTR [ecx+8]
  00104	69 f6 e7 d3 0a
	00		 imul	 esi, 709607		; 000ad3e7H
  0010a	c1 c0 05	 rol	 eax, 5
  0010d	33 41 0c	 xor	 eax, DWORD PTR [ecx+12]
  00110	83 c1 18	 add	 ecx, 24			; 00000018H
  00113	33 f8		 xor	 edi, eax

; 162  : 		hash32C = (hash32C ^ (_rotl_KAZE(*(uint32_t *)(p+16),5) ^ *(uint32_t *)(p+20))) * PRIME;        

  00115	8b 41 f8	 mov	 eax, DWORD PTR [ecx-8]
  00118	69 ff e7 d3 0a
	00		 imul	 edi, 709607		; 000ad3e7H
  0011e	c1 c0 05	 rol	 eax, 5
  00121	33 41 fc	 xor	 eax, DWORD PTR [ecx-4]
  00124	33 e8		 xor	 ebp, eax
  00126	69 ed e7 d3 0a
	00		 imul	 ebp, 709607		; 000ad3e7H
  0012c	4a		 dec	 edx
  0012d	75 c5		 jne	 SHORT $LL9@FNV1A_Hash@2
*/

/*
// Intel(R) C++ Compiler XE for applications running on IA-32, Version 12.1.1.258 Build 20111011 gave this:
// 216a-212f+2= 61 bytes, No CARAMBA anymore.
;;;     for(; wrdlen >= 3*2*sizeof(uint32_t); wrdlen -= 3*2*sizeof(uint32_t), p += 3*2*sizeof(uint32_t)) {

  02127 83 fa 18         cmp edx, 24                            
  0212a 72 43            jb .B4.5 ; Prob 10%                    
                                ; LOE eax edx ecx ebx ebp esi edi
.B4.2:                          ; Preds .B4.1
  0212c 89 34 24         mov DWORD PTR [esp], esi               ;
                                ; LOE eax edx ecx ebx ebp edi
.B4.3:                          ; Preds .B4.2 .B4.3

;;; 		hash32 = (hash32 ^ (_rotl_KAZE(*(uint32_t *)(p+0),5) ^ *(uint32_t *)(p+4))) * PRIME;        

  0212f 8b 31            mov esi, DWORD PTR [ecx]               
  02131 83 c2 e8         add edx, -24                           
  02134 c1 c6 05         rol esi, 5                             
  02137 33 71 04         xor esi, DWORD PTR [4+ecx]             
  0213a 33 de            xor ebx, esi                           

;;; 		hash32B = (hash32B ^ (_rotl_KAZE(*(uint32_t *)(p+8),5) ^ *(uint32_t *)(p+12))) * PRIME;        

  0213c 8b 71 08         mov esi, DWORD PTR [8+ecx]             
  0213f c1 c6 05         rol esi, 5                             
  02142 33 71 0c         xor esi, DWORD PTR [12+ecx]            
  02145 33 fe            xor edi, esi                           

;;; 		hash32C = (hash32C ^ (_rotl_KAZE(*(uint32_t *)(p+16),5) ^ *(uint32_t *)(p+20))) * PRIME;        

  02147 8b 71 10         mov esi, DWORD PTR [16+ecx]            
  0214a c1 c6 05         rol esi, 5                             
  0214d 33 71 14         xor esi, DWORD PTR [20+ecx]            
  02150 83 c1 18         add ecx, 24                            
  02153 33 ee            xor ebp, esi                           
  02155 69 db e7 d3 0a 
        00               imul ebx, ebx, 709607                  
  0215b 69 ff e7 d3 0a 
        00               imul edi, edi, 709607                  
  02161 69 ed e7 d3 0a 
        00               imul ebp, ebp, 709607                  
  02167 83 fa 18         cmp edx, 24                            
  0216a 73 c3            jae .B4.3 ; Prob 82%                   
                                ; LOE eax edx ecx ebx ebp edi
.B4.4:                          ; Preds .B4.3
  0216c 8b 34 24         mov esi, DWORD PTR [esp]               ;
                                ; LOE eax edx ecx ebx ebp esi edi
.B4.5:                          ; Preds .B4.1 .B4.4
*/

    }
	if (p != str) {
		hash32 = (hash32 ^ _rotl_KAZE(hash32C,5) ) * PRIME;
		//hash32B = (hash32B ^ _rotl_KAZE(hash32D,5) ) * PRIME;
	}

    // 1111=15; 10111=23
    if (wrdlen & 4*sizeof(uint32_t)) {	
		hash32 = (hash32 ^ (_rotl_KAZE(*(uint32_t *)(p+0),5) ^ *(uint32_t *)(p+4))) * PRIME;        
		hash32B = (hash32B ^ (_rotl_KAZE(*(uint32_t *)(p+8),5) ^ *(uint32_t *)(p+12))) * PRIME;        
		p += 8*sizeof(uint16_t);
    }
    // Cases: 0,1,2,3,4,5,6,7,...,15
    if (wrdlen & 2*sizeof(uint32_t)) {
		hash32 = (hash32 ^ *(uint32_t*)(p+0)) * PRIME;
		hash32B = (hash32B ^ *(uint32_t*)(p+4)) * PRIME;
		p += 4*sizeof(uint16_t);
    }
    // Cases: 0,1,2,3,4,5,6,7
    if (wrdlen & sizeof(uint32_t)) {
		hash32 = (hash32 ^ *(uint16_t*)(p+0)) * PRIME;
		hash32B = (hash32B ^ *(uint16_t*)(p+2)) * PRIME;
		p += 2*sizeof(uint16_t);
    }
    if (wrdlen & sizeof(uint16_t)) {
        hash32 = (hash32 ^ *(uint16_t*)p) * PRIME;
        p += sizeof(uint16_t);
    }
    if (wrdlen & 1) 
        hash32 = (hash32 ^ *p) * PRIME;

    hash32 = (hash32 ^ _rotl_KAZE(hash32B,5) ) * PRIME;
    return hash32 ^ (hash32 >> 16);
}

#if 0
int main( int argc, char *argv[] ) {
	FILE *fp;
	FILE *fp_outLOG;
	uint32_t Filehash;
	//int SourceSize;
	//int TargetSize;
	uint64_t SourceSize;
	uint64_t TargetSize;
	uint64_t TargetSize1;
	uint64_t TargetSize2;
	uint64_t VerifySize;
	uint64_t DoOffset;
	char* SourceBlock=NULL;
	char* TargetBlock=NULL;
	char* VerifyBlock=NULL;
	char* Nakamichi = ".Nakamichi\0";
	char* LZSSE     = ".L17.LZSSE2\0";
	char NewFileName[256];
	char NewFileName2[256];
//	clock_t clocks1, clocks2;
	uint64_t OneMB=1024*1024;
	uint64_t SevenGB = OneMB*5120;

char *pointerALIGN;
int i, j;
clock_t clocks3, clocks4;
double duration;
double durationGENERIC;
int BandwidthFlag=0;

unsigned long long k;
unsigned long long k1;
unsigned long long k2;
int Trials;

#if defined(_WIN32_ENVIRONMENT_)
      unsigned long long size_inLINESIXFOUR;
      unsigned long long size_inLINESIXFOURlog;
#else
      size_t size_inLINESIXFOUR;
      size_t size_inLINESIXFOURlog;
#endif /* defined(_WIN32_ENVIRONMENT_)  */

#ifdef _N_HIGH_PRIORITY
   DWORD dwError, dwPriClass;
#endif

	printf("Nakamichi 'Okamigan/Lonewolfeye', written by Kaze, based on Nobuo Ito's LZSS source, babealicious suggestion by m^2 enforced, muffinesque suggestion by Jim Dempsey enforced.\n");
	printf("Note1: It is dedicated to Shintaro Katsu's brother Tomisaburo Wakayama animating 'Okami/Lonewolf'.\n");
	printf("Note2: 'Okamiko' predecessors are 'Zato', 'Tsubame', 'Tengu-Tsuyo' and 'Tengu'.\n");
	printf("Note3: This compile can handle files up to 5120MB.\n");
	printf("Note4: The matchfinder/memmem() is 'Railgun_BawBaw_reverse'.\n");
	printf("Note5: Instead of '_mm_loadu_si128' '_mm_lddqu_si128' is used.\n");
	printf("Note6: The lookahead 'Tsuyo' heuristic which looks one char ahead is applied thrice, still not strengthened, though.\n");

#ifdef _N_HIGH_PRIORITY
   if(!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS))
   {
//      _tprintf(TEXT("Already REALTIME_PRIORITY.\n"));
//      goto Cleanup;
   } 
   if(!SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS))
   {
//      _tprintf(TEXT("Already REALTIME_PRIORITY.\n"));
//      goto Cleanup;
   } 
   // Display priority class

   dwPriClass = GetPriorityClass(GetCurrentProcess());

   //_tprintf(TEXT("Current priority class is 0x%x\n"), dwPriClass);

   if (dwPriClass==0x00000080) printf("Current priority class is HIGH_PRIORITY_CLASS.\n");
   if (dwPriClass==0x00000100) printf("Current priority class is REALTIME_PRIORITY_CLASS.\n");
#endif
	
	if (argc==1) {
		printf("Usage: Nakamichi filename\n"); exit(13);
	}
	if (argc==3) BandwidthFlag=1;
	BandwidthFlag=0; // Disable memcpy test
	//if (BandwidthFlag) Trials=256; else Trials=1; 
	if (argc==3) Trials=256; else Trials=1; 
	if ((fp = fopen(argv[1], "rb")) == NULL) {
		printf("Nakamichi: Can't open '%s' file.\n", argv[1]); exit(13);
	}

#if defined(_WIN32_ENVIRONMENT_)
   // 64bit:
_lseeki64( fileno(fp), 0L, SEEK_END );
size_inLINESIXFOUR = _telli64( fileno(fp) );
_lseeki64( fileno(fp), 0L, SEEK_SET );
#else
   // 64bit:
fseeko( fp, 0L, SEEK_END );
size_inLINESIXFOUR = ftello( fp );
fseeko( fp, 0L, SEEK_SET );
#endif /* defined(_WIN32_ENVIRONMENT_)  */
SourceSize = (uint64_t)size_inLINESIXFOUR;

	//fseek(fp, 0, SEEK_END);
	//SourceSize = ftell(fp);
	//fseek(fp, 0, SEEK_SET);

	// If filename ends in '.Nakamichi' then mode is decompression otherwise compression.
	if (strcmp(argv[1]+(strlen(argv[1])-strlen(Nakamichi)), Nakamichi) == 0) {
// DECOMPRESSING [
	printf("Allocating Source-Buffer %s MB ... \n", _ui64toaKAZEcomma((SourceSize+512)>>20, llTOaDigits2, 10) );
	SourceBlock = (char*)malloc(SourceSize+512);
	if( SourceBlock == NULL )
		{ printf("Nakamichi: Needed memory (%luMB) allocation denied!\n", (SourceSize+512)>>20); exit(13); }
//	TargetBlock = (char*)malloc(1111*1024*1024+512); // This was enwik9 setting
//	TargetBlock = (char*)malloc(OneMB*2330+512); // This is GTCC_General_Textual_Compression_Corpus.tar 2,443,181,056 setting
	printf("Allocating Target-Buffer %s MB ... \n", _ui64toaKAZEcomma((SevenGB+512)>>20, llTOaDigits2, 10) );
	TargetBlock = (char*)malloc(SevenGB+512); // 5GB
		if( TargetBlock == NULL )
		{ printf("Nakamichi: Needed memory (%luMB) allocation denied!\n", (SevenGB+512)>>20); free(SourceBlock); exit(13); }
	fread(SourceBlock, 1, SourceSize, fp);
	fclose(fp);
		//printf("Decompressing %lu bytes ...\n", SourceSize );
		printf("Decompressing %s bytes ...\n", _ui64toaKAZEcomma(SourceSize, llTOaDigits2, 10) );
// Warm up... [
	for (i = 1; i <= Trials; i++) {
		TargetSize = NakaDecompress(TargetBlock, SourceBlock, SourceSize);
	}
// Warm up... ]
		//clocks1 = clock();
		//while (clocks1 == clock());
		clocks1 = clock();
	for (i = 1; i <= Trials; i++) {
		TargetSize = NakaDecompress(TargetBlock, SourceBlock, SourceSize);
	}
		clocks2 = clock();
		k = (((double)CLOCKS_PER_SEC*TargetSize/(double)(clocks2 - clocks1 + 1))); k=k*Trials; k=k>>20; 
		printf("RAM-to-RAM performance: %d MB/s.\n", k);
		strcpy(NewFileName, argv[1]);
		*( NewFileName + strlen(argv[1])-strlen(Nakamichi) ) = '\0';
		printf("Source-file-Hash(FNV1A_YoshimitsuTRIAD) = 0x%s\n", _ui64toaKAZEzerocomma4(FNV1A_Hash_YoshimitsuTRIAD(SourceBlock, SourceSize), llTOaDigits2, 16)+(26-8-1) );
		printf("Target-file-Hash(FNV1A_YoshimitsuTRIAD) = 0x%s\n", _ui64toaKAZEzerocomma4(FNV1A_Hash_YoshimitsuTRIAD(TargetBlock, TargetSize), llTOaDigits2, 16)+(26-8-1) );
	if ((fp = fopen(NewFileName, "wb")) == NULL) {
		printf("Nakamichi: Can't write '%s' file.\n", NewFileName); exit(13);
	}
	//fwrite(TargetBlock, 1, TargetSize, fp); // Caramba: It doesn't work when file is 4+GB long!
	if (TargetSize <= OneMB) {
		fwrite(TargetBlock, 1, TargetSize, fp);
	} else {
		for (DoOffset = 0; DoOffset+OneMB < TargetSize; DoOffset=DoOffset+OneMB) {
			fwrite(TargetBlock+DoOffset, 1, OneMB, fp);
		}
		//if (DoOffset+OneMB >= TargetSize) {
			fwrite(TargetBlock+DoOffset, 1, TargetSize - DoOffset, fp);
	}
	fclose(fp);

	} else {
// COMPRESSING [
	Trials=256;
	printf("Allocating Source-Buffer %s MB ... \n", _ui64toaKAZEcomma((SourceSize+512)>>20, llTOaDigits2, 10) );
	SourceBlock = (char*)malloc(SourceSize+512);
	if( SourceBlock == NULL )
		{ printf("Nakamichi: Needed memory (%sMB) allocation denied!\n", _ui64toaKAZEcomma((SourceSize+512)>>20, llTOaDigits2, 10)); exit(13); }
	printf("Allocating Target-Buffer %s MB ... \n", _ui64toaKAZEcomma((SourceSize+512+32*1024*1024)>>20, llTOaDigits2, 10) );
	TargetBlock = (char*)malloc(SourceSize+512+32*1024*1024); //+32*1024*1024, some files may be expanded instead of compressed.
	if( TargetBlock == NULL )
		{ printf("Nakamichi: Needed memory (%sMB) allocation denied!\n", _ui64toaKAZEcomma((SourceSize+512+32*1024*1024)>>20, llTOaDigits2, 10)); free(SourceBlock); exit(13); }
	// Allocating before the compression, TO TRY TO AVOID JUST-MAPPING [
	printf("Allocating Verification-Buffer %s MB ... \n", _ui64toaKAZEcomma((SourceSize+512)>>20, llTOaDigits2, 10) );
	VerifyBlock = (char*)malloc(SourceSize+512);
	if( VerifyBlock == NULL )
	{ printf("Nakamichi: Needed memory (%sMB) allocation denied!\n", _ui64toaKAZEcomma((SourceSize+512)>>20, llTOaDigits2, 10)); exit(13); }
	// Allocating before the compression, TO TRY TO AVOID JUST-MAPPING ]
	fread(SourceBlock, 1, SourceSize, fp);
	fclose(fp);
		//printf("Compressing %lu bytes ...\n", SourceSize );
		printf("Compressing %s bytes ...\n", _ui64toaKAZEcomma(SourceSize, llTOaDigits2, 10) );
		clocks1 = clock();
		while (clocks1 == clock());
		clocks1 = clock();
		TargetSize = NakaCompress(TargetBlock, SourceBlock, SourceSize);
		TargetSize1=TargetSize;
		clocks2 = clock();
		k = (((double)CLOCKS_PER_SEC*SourceSize/(double)(clocks2 - clocks1 + 1))); //k=k>>10;
		printf("RAM-to-RAM performance: %d B/s.\n", k);
		strcpy(NewFileName, argv[1]);
		strcat(NewFileName, Nakamichi);
		//printf("Compressed to %d bytes.\n", TargetSize );
		printf("Compressed to %s bytes.\n", _ui64toaKAZEcomma(TargetSize, llTOaDigits2, 10) );
		Filehash = FNV1A_Hash_YoshimitsuTRIAD(SourceBlock, SourceSize);
		printf("Source-file-Hash(FNV1A_YoshimitsuTRIAD) = 0x%s\n", _ui64toaKAZEzerocomma4(Filehash, llTOaDigits2, 16)+(26-8-1) );
		printf("Target-file-Hash(FNV1A_YoshimitsuTRIAD) = 0x%s\n", _ui64toaKAZEzerocomma4(FNV1A_Hash_YoshimitsuTRIAD(TargetBlock, TargetSize), llTOaDigits2, 16)+(26-8-1) );
	if ((fp = fopen(NewFileName, "wb")) == NULL) {
		printf("Nakamichi: Can't write '%s' file.\n", NewFileName); exit(13);
	}
	//fwrite(TargetBlock, 1, TargetSize, fp); // Caramba: It doesn't work when file is 4+GB long!
	if (TargetSize <= OneMB) {
		fwrite(TargetBlock, 1, TargetSize, fp);
	} else {
		for (DoOffset = 0; DoOffset+OneMB < TargetSize; DoOffset=DoOffset+OneMB) {
			fwrite(TargetBlock+DoOffset, 1, OneMB, fp);
		}
		//if (DoOffset+OneMB >= TargetSize) {
			fwrite(TargetBlock+DoOffset, 1, TargetSize - DoOffset, fp);
	}
	fclose(fp);
		printf("Decompressing %s (being the compressed stream) bytes ...\n", _ui64toaKAZEcomma(TargetSize, llTOaDigits2, 10) );
		clocks1 = clock();
		while (clocks1 == clock());
		clocks1 = clock();
	for (i = 1; i <= Trials; i++) {
		VerifySize = NakaDecompress(VerifyBlock, TargetBlock, TargetSize);
	}
		clocks2 = clock();
		k1 = (((double)CLOCKS_PER_SEC*VerifySize/(double)(clocks2 - clocks1 + 1))); k1=k1*Trials; k1=k1>>20; 
		printf("RAM-to-RAM performance: %d MB/s.\n", k1);
		if(VerifySize == SourceSize) printf("Verification (input and output sizes match) OK.\n"); else printf("Verification (input and output sizes mismatch) FAILED!\n");
		if (memcmp(SourceBlock, VerifyBlock, SourceSize)==0) printf("Verification (input and output blocks match) OK.\n"); else printf("Verification (input and output blocks mismatch) FAILED!\n");
	free(VerifyBlock);
// COMPRESSING ]
// LOG writing [
	if( ( fp_outLOG = fopen( "Nakamichi.LOG", "a+" ) ) == NULL )
	{ printf( "Nakamichi: Can't open file Nakamichi.LOG.\n" ); return( 1 ); }
#if defined(_WIN32_ENVIRONMENT_)
   // 64bit:
_lseeki64( fileno(fp_outLOG), 0L, SEEK_END );
size_inLINESIXFOURlog = _telli64( fileno(fp_outLOG) );
_lseeki64( fileno(fp_outLOG), 0L, SEEK_SET );
#else
   // 64bit:
fseeko( fp_outLOG, 0L, SEEK_END );
size_inLINESIXFOURlog = ftello( fp_outLOG );
fseeko( fp_outLOG, 0L, SEEK_SET );
#endif /* defined(_WIN32_ENVIRONMENT_)  */
if ( (uint64_t)size_inLINESIXFOURlog == 0 )
	fprintf( fp_outLOG, "| #1 Filesize     | #2 Filehash  | #3 Nakamichi 'ZO' c.size/decompressionrate | #4 LZSSE2 c.size/decompressionrate | #5 Filename \n" );
	fprintf( fp_outLOG, "| %s ",  _ui64toaKAZEzerocomma(size_inLINESIXFOUR, llTOaDigits3, 10)+(26-15) );
	fprintf( fp_outLOG, "| 0x%s  ",  _ui64toaKAZEzerocomma4(Filehash, llTOaDigits2, 16)+(26-8-1) );
	fprintf( fp_outLOG, "| %s / %sMB/s                ",  _ui64toaKAZEzerocomma(TargetSize1, llTOaDigits2, 10)+(26-13), _ui64toaKAZEzerocomma(k1, llTOaDigits3, 10)+(26-7) );
	fprintf( fp_outLOG, "| %s / %sMB/s        ",  _ui64toaKAZEzerocomma(TargetSize2, llTOaDigits2, 10)+(26-13), _ui64toaKAZEzerocomma(k2, llTOaDigits3, 10)+(26-7) );
	fprintf( fp_outLOG, "| %s \n", argv[1] );
	fclose(fp_outLOG);
// LOG writing ]
	}

	if (BandwidthFlag) {
// Benchmark memcpy() [
pointerALIGN = TargetBlock + 64 - (((size_t)TargetBlock) % 64);
//offset=64-int((long)data&63);
printf("Memory pool starting address: %p ... ", pointerALIGN);
if (((uintptr_t)(const void *)pointerALIGN & (64 - 1)) == 0) printf( "64 byte aligned, OK\n"); else printf( "NOT 64 byte aligned, FAILURE\n");
clocks3 = clock();
while (clocks3 == clock());
clocks3 = clock();
printf("Copying a %dMB block 1024 times i.e. %dGB READ + %dGB WRITTEN ...\n", 512, 512, 512);
	for (i = 0; i < 1024; i++) {
	memcpy(pointerALIGN+512*1024*1024, pointerALIGN, 512*1024*1024);
	}
clocks4 = clock();
duration = (double) (clocks4 - clocks3 + 1);
durationGENERIC = duration;
printf("memcpy(): (%dMB block); %dMB copied in %d clocks or %.3fMB per clock\n", 512, 1024*( 512 ), (int) duration, (double)1024*( 512 )/ ((int) duration));

/*
#ifndef _N_GP
clocks3 = clock();
while (clocks3 == clock());
clocks3 = clock();
printf("Copying a %dMB block 1024 times i.e. %dGB READ + %dGB WRITTEN ...\n", 512, 512, 512);
	for (i = 0; i < 1024; i++) {
	memcpy_SSE2_4K_prefetched(pointerALIGN+512*1024*1024, pointerALIGN, 512*1024*1024);
	}
clocks4 = clock();
duration = (double) (clocks4 - clocks3 + 1);
printf("memcpy_SSE2_4K_prefetched(): (%dMB block); %dMB copied in %d clocks or %.3fMB per clock\n", 512, 1024*( 512 ), (int) duration, (double)1024*( 512 )/ ((int) duration));
#endif

#ifdef _N_YMM
clocks3 = clock();
while (clocks3 == clock());
clocks3 = clock();
printf("Copying a %dMB block 1024 times i.e. %dGB READ + %dGB WRITTEN ...\n", 512, 512, 512);
	for (i = 0; i < 1024; i++) {
	memcpy_AVX_4K_prefetched(pointerALIGN+512*1024*1024, pointerALIGN, 512*1024*1024);
	}
clocks4 = clock();
duration = (double) (clocks4 - clocks3 + 1);
printf("memcpy_AVX_4K_prefetched(): (%dMB block); %dMB copied in %d clocks or %.3fMB per clock\n", 512, 1024*( 512 ), (int) duration, (double)1024*( 512 )/ ((int) duration));
#endif
*/
// Benchmark memcpy() ]
//k = (((double)1000*TargetSize/(clocks2 - clocks1 + 1))); k=k>>20;
j = (double)1000*1024*( 512 )/ ((int) durationGENERIC);
printf("RAM-to-RAM performance vs memcpy() ratio (bigger-the-better): %d%%\n", (int)((double)k*100/j));
	}

	free(TargetBlock);
	free(SourceBlock);
	exit(0);
}
#endif

void SearchIntoSlidingWindow(unsigned int* ShortMediumLongOFFSET, unsigned int* retIndex, unsigned int* retMatch, char* refStart,char* refEnd,char* encStart,char* encEnd){
	char* FoundAtPosition;
	unsigned int match=0;

// Too lazy to write Railgun-Reverse, it would save many ugly patches...
	char* refStartSW1 = refEnd-(16-1);             // 11b
	char* refStartSW2 = refEnd-(4*8*128-1);        // 10b
	char* refStartSW3 = refEnd-(1024*8*128-1);     // 01b
	char* refStartSW4 = refEnd-(256*1024*8*128-1); // 00b
	char* refStartSW1a = refEnd-(16*1024*8*128-1);  // 11b
	char* refStartSW1b = refEnd-(64*8*128-1);  // 11b
	char* refStartSW1c = refEnd-(256-1);  // 11b
	char* refStartSW1d = refEnd-(256*4-1);  // 11b
	char* refStartSW1e = refEnd-(64*1024*8*128-1);  // 11b

	// In order to avoid the unheardof slowness the 256MB may be reduced to 2MB... // --|
	char* refStartHOT = refEnd-(256*8*128-1);                                      //   |
	char* refStartHOTTER = refEnd-(4*8*128-1);                                     //   |
	char* refStartHOTEST = refEnd-(16-1);                                          //   |
	char* refStartCOLDERbig = refEnd-(1024*8*128-1);                               //   |
	char* refStartCOLDERERbig = refEnd-(2048*8*128-1);                             //   |
	char* refStartCOLDERERERbig = refEnd-(8192*8*128-1);                           //   |
	char* refStartCOLDERERERERbig = refEnd-(512*1024*8*128-1);                     // <-|
	char* refStartCOLDESTbig = refStart;

	*retIndex=0;
	*retMatch=0;
	*ShortMediumLongOFFSET=0;

#ifdef ReplaceBruteForceWithRailgunSwampshineBailOut

// Also, finally it is time to fix the stupid offset (blind for files smaller than the current window) stupidity for small files:
// Simply assign 'refStart' if it is within the current window i.e. between e.g. 'refEnd-(256*8*128-1)' and 'refEnd':

// Nasty bug fixed (pointer getting negative) only here, to be fixed in all the rest variants [
/*
	if ( refStart >= refEnd-(2048*8*128-1) ) refStartHOT = refStart;                //--|
	if ( refStart >= refEnd-(256*8*128-1) ) refStartHOTTER = refStart;              //--|
	if ( refStart >= refEnd-(2*4*8*128-1) ) refStartHOTEST = refStart;              //--|
	if ( refStart >= refEnd-(512*1024*8*128-1) ) refStartCOLDERbig = refStart;      //--|
                                                                                        // \ /
*/
// Nasty bug fixed (pointer getting negative) only here, to be fixed in all the rest variants ]

//	if ( refStart >= refEnd-(256*8*128-1) ) refStartHOT = refStart;
//	if ( refStart >= refEnd-(4*8*128-1) ) refStartHOTTER = refStart;
//	if ( refStart >= refEnd-(16-1) ) refStartHOTEST = refStart;     
//	if ( refStart >= refEnd-(1024*8*128-1) ) refStartCOLDERbig = refStart; 
//	if ( refStart >= refEnd-(2048*8*128-1) ) refStartCOLDERERbig = refStart;

	if ( (256*8*128-1) >= refEnd-refStart ) refStartHOT = refStart;
	if ( (4*8*128-1) >= refEnd-refStart ) refStartHOTTER = refStart;
	if ( (16-1) >= refEnd-refStart ) refStartHOTEST = refStart;     
	if ( (1024*8*128-1) >= refEnd-refStart ) refStartCOLDERbig = refStart; 
	if ( (2048*8*128-1) >= refEnd-refStart ) refStartCOLDERERbig = refStart;

	if ( (8192*8*128-1) >= refEnd-refStart ) refStartCOLDERERERbig = refStart;
	if ( (512*1024*8*128-1) >= refEnd-refStart ) refStartCOLDERERERERbig = refStart;

	//printf("%d\n", refStartCOLDERbig); //debug
	//printf("%p\n", refStartCOLDERbig); //debug

	if ( (16-1) >= refEnd-refStart ) refStartSW1 = refStart;     
	if ( (4*8*128-1) >= refEnd-refStart ) refStartSW2 = refStart;     
	if ( (1024*8*128-1) >= refEnd-refStart ) refStartSW3 = refStart;     
	if ( (256*1024*8*128-1) >= refEnd-refStart ) refStartSW4 = refStart;     
	if ( (16*1024*8*128-1) >= refEnd-refStart ) refStartSW1a = refStart;     
	if ( (64*8*128-1) >= refEnd-refStart ) refStartSW1b = refStart;     
	if ( (256-1) >= refEnd-refStart ) refStartSW1c = refStart;     

	if ( (256*4-1) >= refEnd-refStart ) refStartSW1d = refStart;     
	if ( (64*1024*8*128-1) >= refEnd-refStart ) refStartSW1e = refStart;     

// Sizewise priority:
//  4:1= 4    (16B)   #01 32:2= 16   (256B)
//  8:1= 8    (16B)   #02 48:4= 12   (64MB)
// 12:1= 12   (16B)   #03 24:2= 12   (4KB)
// 16:1= Flag (16B)   #04 12:1= 12   (16B)
//  4:2= 2    (4KB)   #05 32:4= 8    (64MB)
//  6:2= 3    (1KB)   #06 24:3= 8    (1MB)
//  8:2= 4    (4KB)   #07 16:2= 8    (4KB)
// 12:2= 6    (4KB)   #08  8:1= 8    (16B)
// 16:2= 8    (4KB)   #09 24:4= 6    (256MB)
// 24:2= 12   (4KB)   #10 12:2= 6    (4KB)
// 32:2= 16   (256B)  #11 16:3= 5.3  (1MB)
//  4:3= 1.3  (1MB)   #12 16:4= 4    (16MB)F
//  8:3= 2.6  (1MB)   #13 16:4= 4    (64MB)S
// 12:3= 4    (1MB)   #14 12:3= 4    (1MB)
// 16:3= 5.3  (1MB)   #15  8:2= 4    (4KB)
// 24:3= 8    (1MB)   #16  4:1= 4    (16B)
//  6:4= 1.5  (16MB)  #17 14:4= 3.5  (16MB)
//  8:4= 2    (16MB)  #18 12:4= 3    (16MB)
// 10:4= 2.5  (16MB)  #19  6:2= 3    (1KB)
// 12:4= 3    (16MB)  #20  8:3= 2.6  (1MB)
// 14:4= 3.5  (16MB)  #21 10:4= 2.5  (16MB)
// 16:4= 4    (16MB)F #22  8:4= 2    (16MB)
// 16:4= 4    (64MB)S #23  4:2= 2    (4KB)
// 24:4= 6    (256MB) #24  6:4= 1.5  (16MB)
// 32:4= 8    (64MB)  #25  4:3= 1.3  (1MB)
// 48:4= 12   (64MB) 
// |1stLSB    |2ndLSB  |3rdLSB   |
// -------------------------------
// |OO|LL|xxxx|xxxxxxxx|xxxxxx|xx|
// -------------------------------
// [1bit          16bit]    24bit]
// LL = 00b means Long MatchLength, (4-LL)<<2 or 16
// LL = 01b means Long MatchLength, (4-LL)<<2 or 12
// LL = 10b means Long MatchLength, (4-LL)<<2 or 8
// LL = 11b means Long MatchLength, (4-LL)<<2 or 4
// xxxx0011b Literals for xxxx in 1..15-7, Matches for 10..15 i.e. Sliding Window is   4*8-8=24 or  16MB
// OO = 00b MatchOffset, 0xFFFFFFFF>>OO, 4 bytes long i.e. Sliding Window is 4*8-LL-OO=4*8-4=28 or 256MB
// OO = 01b MatchOffset, 0xFFFFFFFF>>OO, 3 bytes long i.e. Sliding Window is 3*8-LL-OO=3*8-4=20 or   1MB    
// OO = 10b MatchOffset, 0xFFFFFFFF>>OO, 2 bytes long i.e. Sliding Window is 2*8-LL-OO=2*8-4=12 or   4KB    
// OO = 11b MatchOffset, 0xFFFFFFFF>>OO, 1 byte long  i.e. Sliding Window is 1*8-LL-OO=1*8-4=4 or   16B     


// 32:2= 16   (256B)

	if (refStartSW1c >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW1c, encStart, (uint32_t)(refEnd-refStartSW1c), 32);	
		if (FoundAtPosition!=NULL) {
				*retMatch=32;
				// The first four bits should be:
				                                                                    // 0011b = 0x3;
				*retIndex=(((refEnd-FoundAtPosition)<<8)&0xFF00)|0x03; // xx ... x[LLOO]
				*ShortMediumLongOFFSET=2;
				return;
		}
	}

// 48:4= 12   (64MB)

	if (refStartSW1e >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW1e, encStart, (uint32_t)(refEnd-refStartSW1e), 48);	
		if (FoundAtPosition!=NULL) {
				*retMatch=48;
				// The first four bits should be:
				                                                                    // 1100b = 0xC; 6<<3
				*retIndex=(((refEnd-FoundAtPosition)<<6)&0xFFFFFFC0)|0x3C; // xx ... x[LLOO]
				*ShortMediumLongOFFSET=4;
				return;
		}
	}

// 24:2= 12   (4KB)
/*
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartHOTTER >= refStart)
	if (refStartHOTTER < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartHOTTER, encStart, (uint32_t)(refEnd-refStartHOTTER), 24);	
		if (FoundAtPosition!=NULL) {
				*retMatch=24;
				// The first four bits should be:
				                                                                  // 1000b = 8
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFF0)|0x0008; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=2;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]
*/
	if (refStartSW2 >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW2, encStart, (uint32_t)(refEnd-refStartSW2), 24);	
		if (FoundAtPosition!=NULL) {
				*retMatch=24;
				// The first four bits should be:
				                                                                  // 1000b = 8
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFF0)|0x0008; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=2;
				return;
		}
	}

// 12:1= 12   (16B)
/*
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartHOTEST >= refStart)
	if (refStartHOTEST < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartHOTEST, encStart, (uint32_t)(refEnd-refStartHOTEST), 12);	
		if (FoundAtPosition!=NULL) {
				*retMatch=12;
				// The first four bits should be:
				                                                                  // 0111b = 7
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFF0)|0x0007; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=1;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]
*/
	if (refStartSW1 >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW1, encStart, (uint32_t)(refEnd-refStartSW1), 12);	
		if (FoundAtPosition!=NULL) {
				*retMatch=12;
				// The first four bits should be:
				                                                                  // 0111b = 7
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xF0)|0x0007; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=1;
				return;
		}
	}

// 32:4= 8    (64MB)

	if (refStartSW1e >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW1e, encStart, (uint32_t)(refEnd-refStartSW1e), 32);	
		if (FoundAtPosition!=NULL) {
				*retMatch=32;
				// The first four bits should be:
				                                                                    // 1100b = 0xC; 6<<3
				*retIndex=(((refEnd-FoundAtPosition)<<6)&0xFFFFFFC0)|0x2C; // xx ... x[LLOO]
				*ShortMediumLongOFFSET=4;
				return;
		}
	}

// 24:3= 8    (1MB)
/*
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartHOT >= refStart)
	if (refStartHOT < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartHOT, encStart, (uint32_t)(refEnd-refStartHOT), 24);	
		if (FoundAtPosition!=NULL) {
				*retMatch=24;
				// The first four bits should be:
				                                                                    // 0100b = 4
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFFFF0)|0x0004; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=3;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]

	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartCOLDERbig >= refStart)
	if (refStartCOLDERbig < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartCOLDERbig, encStart, (uint32_t)(refEnd-refStartCOLDERbig), 24);	
		if (FoundAtPosition!=NULL) {
				*retMatch=24;
				// The first four bits should be:
				                                                                    // 0100b = 4
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFFFF0)|0x0004; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=3;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]
*/
	if (refStartSW3 >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW3, encStart, (uint32_t)(refEnd-refStartSW3), 24);	
		if (FoundAtPosition!=NULL) {
				*retMatch=24;
				// The first four bits should be:
				                                                                    // 0100b = 4
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFFFF0)|0x0004; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=3;
				return;
		}
	}

// 16:2= 8    (4KB)
/*
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartHOTTER >= refStart)
	if (refStartHOTTER < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartHOTTER, encStart, (uint32_t)(refEnd-refStartHOTTER), 16);	
		if (FoundAtPosition!=NULL) {
				*retMatch=16;
				// The first four bits should be:
				                                                                  // 0010b = 2
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFF0)|0x0002; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=2;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]
*/
	if (refStartSW2 >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW2, encStart, (uint32_t)(refEnd-refStartSW2), 16);	
		if (FoundAtPosition!=NULL) {
				*retMatch=16;
				// The first four bits should be:
				                                                                  // 0010b = 2
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFF0)|0x0002; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=2;
				return;
		}
	}

// 8:1= 8    (16B)
/*
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartHOTEST >= refStart)
	if (refStartHOTEST < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartHOTEST, encStart, (uint32_t)(refEnd-refStartHOTEST), 8);	
		if (FoundAtPosition!=NULL) {
				*retMatch=8;
				// The first four bits should be:
				                                                                  // 1011b = B
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFF0)|0x000B; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=1;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]
*/
	if (refStartSW1 >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW1, encStart, (uint32_t)(refEnd-refStartSW1), 8);	
		if (FoundAtPosition!=NULL) {
				*retMatch=8;
				// The first four bits should be:
				                                                                  // 1011b = B
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFF0)|0x000B; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=1;
				return;
		}
	}

// 24:4= 6    (256MB)
	if (refStartSW4 >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW4, encStart, (uint32_t)(refEnd-refStartSW4), 48-24);	
		if (FoundAtPosition!=NULL) {
				*retMatch=48-24;
				// The first four bits should be:
				                                                                    // 0000b = 0x0; 6<<2
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFFFFFF0)|0x00; // xx ... x[LLOO]
				*ShortMediumLongOFFSET=4;
				return;
		}
	}

// 12:2= 6    (4KB)
/*
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartHOTTER >= refStart)
	if (refStartHOTTER < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartHOTTER, encStart, (uint32_t)(refEnd-refStartHOTTER), 12);	
		if (FoundAtPosition!=NULL) {
				*retMatch=12;
				// The first four bits should be:
				                                                                  // 0110b = 6
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFF0)|0x0006; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=2;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]
*/
	if (refStartSW2 >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW2, encStart, (uint32_t)(refEnd-refStartSW2), 12);	
		if (FoundAtPosition!=NULL) {
				*retMatch=12;
				// The first four bits should be:
				                                                                  // 0110b = 6
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFF0)|0x0006; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=2;
				return;
		}
	}

// 16:3= 5.3  (1MB)
/*
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartHOT >= refStart)
	if (refStartHOT < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartHOT, encStart, (uint32_t)(refEnd-refStartHOT), 16);	
		if (FoundAtPosition!=NULL) {
				*retMatch=16;
				// The first four bits should be:
				                                                                    // 0001b = 1
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFFFF0)|0x0001; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=3;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]

	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartCOLDERbig >= refStart)
	if (refStartCOLDERbig < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartCOLDERbig, encStart, (uint32_t)(refEnd-refStartCOLDERbig), 16);	
		if (FoundAtPosition!=NULL) {
				*retMatch=16;
				// The first four bits should be:
				                                                                    // 0001b = 1
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFFFF0)|0x0001; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=3;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]
*/
	if (refStartSW3 >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW3, encStart, (uint32_t)(refEnd-refStartSW3), 16);	
		if (FoundAtPosition!=NULL) {
				*retMatch=16;
				// The first four bits should be:
				                                                                    // 0001b = 1
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFFFF0)|0x0001; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=3;
				return;
		}
	}

// 16:4= 4    (16MB)F

	if (refStartSW1a >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW1a, encStart, (uint32_t)(refEnd-refStartSW1a), 16);	
		if (FoundAtPosition!=NULL) {
				*retMatch=16;
				// The first four bits should be:
				                                                                    // 0011b = 0x3; 11xx0011b, xx==11
				*retIndex=(((refEnd-FoundAtPosition)<<8)&0xFFFFFF00)|0xA3; // xx ... x[LLOO]
				*ShortMediumLongOFFSET=4;
				return;
		}
	}

// 16:4= 4    (64MB)S

	if (refStartSW1e >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW1e, encStart, (uint32_t)(refEnd-refStartSW1e), 16);	
		if (FoundAtPosition!=NULL) {
				*retMatch=16;
				// The first four bits should be:
				                                                                    // 1100b = 0xC; 6<<3
				*retIndex=(((refEnd-FoundAtPosition)<<6)&0xFFFFFFC0)|0x1C; // xx ... x[LLOO]
				*ShortMediumLongOFFSET=4;
				return;
		}
	}

// 12:3= 4    (1MB)
/*
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartHOT >= refStart)
	if (refStartHOT < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartHOT, encStart, (uint32_t)(refEnd-refStartHOT), 12);	
		if (FoundAtPosition!=NULL) {
				*retMatch=12;
				// The first four bits should be:
				                                                                    // 0101b = 5
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFFFF0)|0x0005; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=3;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]

	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartCOLDERbig >= refStart)
	if (refStartCOLDERbig < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartCOLDERbig, encStart, (uint32_t)(refEnd-refStartCOLDERbig), 12);	
		if (FoundAtPosition!=NULL) {
				*retMatch=12;
				// The first four bits should be:
				                                                                    // 0101b = 5
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFFFF0)|0x0005; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=3;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]
*/
	if (refStartSW3 >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW3, encStart, (uint32_t)(refEnd-refStartSW3), 12);	
		if (FoundAtPosition!=NULL) {
				*retMatch=12;
				// The first four bits should be:
				                                                                    // 0101b = 5
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFFFF0)|0x0005; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=3;
				return;
		}
	}

// 8:2= 4    (4KB)
/*
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartHOTTER >= refStart)
	if (refStartHOTTER < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartHOTTER, encStart, (uint32_t)(refEnd-refStartHOTTER), 8);	
		if (FoundAtPosition!=NULL) {
				*retMatch=8;
				// The first four bits should be:
				                                                                  // 1010b = A
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFF0)|0x000A; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=2;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]
*/
	if (refStartSW2 >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW2, encStart, (uint32_t)(refEnd-refStartSW2), 8);	
		if (FoundAtPosition!=NULL) {
				*retMatch=8;
				// The first four bits should be:
				                                                                  // 1010b = A
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFF0)|0x000A; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=2;
				return;
		}
	}

// 4:1= 4    (16B)
/*
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartHOTEST >= refStart)
	if (refStartHOTEST < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartHOTEST, encStart, (uint32_t)(refEnd-refStartHOTEST), 4);	
		if (FoundAtPosition!=NULL) {
				*retMatch=4;
				// The first four bits should be:
				                                                                  // 1111b = F
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFF0)|0x000F; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=1;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]
*/
	if (refStartSW1 >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW1, encStart, (uint32_t)(refEnd-refStartSW1), 4);	
		if (FoundAtPosition!=NULL) {
				*retMatch=4;
				// The first four bits should be:
				                                                                  // 1111b = F
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFF0)|0x000F; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=1;
				return;
		}
	}

// 14:4= 3.5  (16MB)

	if (refStartSW1a >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW1a, encStart, (uint32_t)(refEnd-refStartSW1a), 14);	
		if (FoundAtPosition!=NULL) {
				*retMatch=14;
				// The first four bits should be:
				                                                                    // 0011b = 0x3; 11xx0011b, xx==11
				*retIndex=(((refEnd-FoundAtPosition)<<8)&0xFFFFFF00)|0xB3; // xx ... x[LLOO]
				*ShortMediumLongOFFSET=4;
				return;
		}
	}

// 12:4= 3    (16MB)

	if (refStartSW1a >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW1a, encStart, (uint32_t)(refEnd-refStartSW1a), 12);	
		if (FoundAtPosition!=NULL) {
				*retMatch=12;
				// The first four bits should be:
				                                                                    // 0011b = 0x3; 11xx0011b, xx==11
				*retIndex=(((refEnd-FoundAtPosition)<<8)&0xFFFFFF00)|0xC3; // xx ... x[LLOO]
				*ShortMediumLongOFFSET=4;
				return;
		}
	}

// 6:2= 3    (1KB)

	if (refStartSW1d >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW1d, encStart, (uint32_t)(refEnd-refStartSW1d), 6);	
		if (FoundAtPosition!=NULL) {
				*retMatch=6;
				// The first four bits should be:
				                                                                    // 0011b = 0x3;
				*retIndex=(((refEnd-FoundAtPosition)<<6)&0xFFC0)|0x0C; // xx ... x[LLOO]
				*ShortMediumLongOFFSET=2;
				return;
		}
	}

// 8:3= 2.6  (1MB)
/*
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartHOT >= refStart)
	if (refStartHOT < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartHOT, encStart, (uint32_t)(refEnd-refStartHOT), 8);	
		if (FoundAtPosition!=NULL) {
				*retMatch=8;
				// The first four bits should be:
				                                                                    // 1001b = 9
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFFFF0)|0x0009; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=3;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]

	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartCOLDERbig >= refStart)
	if (refStartCOLDERbig < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartCOLDERbig, encStart, (uint32_t)(refEnd-refStartCOLDERbig), 8);	
		if (FoundAtPosition!=NULL) {
				*retMatch=8;
				// The first four bits should be:
				                                                                    // 1001b = 9
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFFFF0)|0x0009; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=3;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]
*/
	if (refStartSW3 >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW3, encStart, (uint32_t)(refEnd-refStartSW3), 8);	
		if (FoundAtPosition!=NULL) {
				*retMatch=8;
				// The first four bits should be:
				                                                                    // 1001b = 9
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFFFF0)|0x0009; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=3;
				return;
		}
	}

// 10:4= 2.5  (16MB)

	if (refStartSW1a >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW1a, encStart, (uint32_t)(refEnd-refStartSW1a), 10);	
		if (FoundAtPosition!=NULL) {
				*retMatch=10;
				// The first four bits should be:
				                                                                    // 0011b = 0x3; 11xx0011b, xx==11
				*retIndex=(((refEnd-FoundAtPosition)<<8)&0xFFFFFF00)|0xD3; // xx ... x[LLOO]
				*ShortMediumLongOFFSET=4;
				return;
		}
	}

// 8:4= 2  (16MB)

	if (refStartSW1a >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW1a, encStart, (uint32_t)(refEnd-refStartSW1a), 8);	
		if (FoundAtPosition!=NULL) {
				*retMatch=8;
				// The first four bits should be:
				                                                                    // 0011b = 0x3; 11xx0011b, xx==11
				*retIndex=(((refEnd-FoundAtPosition)<<8)&0xFFFFFF00)|0xE3; // xx ... x[LLOO]
				*ShortMediumLongOFFSET=4;
				return;
		}
	}

// 4:2= 2    (4KB)
/*
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartHOTTER >= refStart)
	if (refStartHOTTER < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartHOTTER, encStart, (uint32_t)(refEnd-refStartHOTTER), 4);	
		if (FoundAtPosition!=NULL) {
				*retMatch=4;
				// The first four bits should be:
				                                                                  // 1110b = E
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFF0)|0x000E; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=2;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]
*/
	if (refStartSW2 >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW2, encStart, (uint32_t)(refEnd-refStartSW2), 4);	
		if (FoundAtPosition!=NULL) {
				*retMatch=4;
				// The first four bits should be:
				                                                                  // 1110b = E
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFF0)|0x000E; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=2;
				return;
		}
	}

// 6:4= 1.5  (16MB)

	if (refStartSW1a >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW1a, encStart, (uint32_t)(refEnd-refStartSW1a), 6);	
		if (FoundAtPosition!=NULL) {
				*retMatch=6;
				// The first four bits should be:
				                                                                    // 0011b = 0x3; 11xx0011b, xx==11
				*retIndex=(((refEnd-FoundAtPosition)<<8)&0xFFFFFF00)|0xF3; // xx ... x[LLOO]
				*ShortMediumLongOFFSET=4;
				return;
		}
	}

// 4:3= 1.3  (1MB)
/*
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartHOT >= refStart)
	if (refStartHOT < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartHOT, encStart, (uint32_t)(refEnd-refStartHOT), 4);	
		if (FoundAtPosition!=NULL) {
				*retMatch=4;
				// The first four bits should be:
				                                                                    // 1101b = D
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFFFF0)|0x000D; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=3;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]

	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) [
	if (refStartCOLDERbig >= refStart)
	if (refStartCOLDERbig < refEnd) {
	FoundAtPosition = Railgun_Trolldom(refStartCOLDERbig, encStart, (uint32_t)(refEnd-refStartCOLDERbig), 4);	
		if (FoundAtPosition!=NULL) {
				*retMatch=4;
				// The first four bits should be:
				                                                                    // 1101b = D
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFFFF0)|0x000D; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=3;
				return;
		}
	}
	// Pre-emptive strike, matches should be sought close to the lookahead (cache-friendliness) ]
*/
	if (refStartSW3 >= refStart) {
	FoundAtPosition = Railgun_BawBaw_reverse (refStartSW3, encStart, (uint32_t)(refEnd-refStartSW3), 4);	
		if (FoundAtPosition!=NULL) {
				*retMatch=4;
				// The first four bits should be:
				                                                                    // 1101b = D
				*retIndex=(((refEnd-FoundAtPosition)<<4)&0xFFFFF0)|0x000D; // xx ... x[OOLL]
				*ShortMediumLongOFFSET=3;
				return;
		}
	}

/*
// 3:2 = 1.5

	if (refStartSW1c >= refStart) {
	FoundAtPosition = Railgun_Baw_reverse (refStartSW1c, encStart, (uint32_t)(refEnd-refStartSW1c), 3);	
		if (FoundAtPosition!=NULL) {
				*retMatch=3;
				// The first four bits should be:
				                                                                    // 0011b = 0x3;
				*retIndex=(((refEnd-FoundAtPosition)<<8)&0xFF00)|0x03; // xx ... x[LLOO]
				*ShortMediumLongOFFSET=2;
				return;
		}
	}
*/


#else				
	while(refStart < refEnd){
		match=SlidingWindowVsLookAheadBuffer(refStart,refEnd,encStart,encEnd);
		if(match > *retMatch){
			*retMatch=match;
			*retIndex=refEnd-refStart;
		}
		if(*retMatch >= Min_Match_BAILOUT_Length) break;
		refStart++;
	}
#endif
}

unsigned int SlidingWindowVsLookAheadBuffer( char* refStart, char* refEnd, char* encStart,char* encEnd){
	int ret = 0;
	while(refStart[ret] == encStart[ret]){
		if(&refStart[ret] >= refEnd) break;
		if(&encStart[ret] >= encEnd) break;
		ret++;
		if(ret >= Min_Match_BAILOUT_Length) break;
	}
	return ret;
}

uint64_t NakaCompress(char* ret, char* src, uint64_t srcSize){
	uint64_t srcIndex=0;
	uint64_t retIndex=0;
	unsigned int index=0;
		unsigned int index1=0;// Tsuyo
	unsigned int match=0;
		unsigned int match1=0;// Tsuyo
	unsigned int notMatch=0;
//	unsigned char* notMatchStart=NULL; // This line causes error when /TP is used, so next one is the fix:
	char* notMatchStart=NULL;
	char* refStart=NULL;
		char* refStart1=NULL;
	char* encEnd=NULL;
		char* encEnd1=NULL;
	int Melnitchka=0;
	char *Auberge[4] = {"|\0","/\0","-\0","\\\0"};
	int ProgressIndicator;

	unsigned int NumberOfFullLiterals=0;
	int GLOBALmediumT=0;
	int GLOBALshortT=0;
	int GLOBALtinyT=0;
	int GLOBAL24B=0;
	int GLOBAL32B=0;
	int GLOBAL48B=0;
	int GLOBAL3=0;
	unsigned int ShortMediumLongOFFSET=0;
		unsigned int ShortMediumLongOFFSET1=0;// Tsuyo
		int TsuyoHEURISTIC;
		long SPD;
		long long TsuyoHEURISTICAPPLIED=0;

	while(srcIndex < srcSize){
		if(srcIndex>=REF_SIZE)
			refStart=&src[srcIndex-REF_SIZE];
		else
			refStart=src;
		if(srcIndex>=srcSize-ENC_SIZE)
			encEnd=&src[srcSize];
		else
			encEnd=&src[srcIndex+ENC_SIZE];
		// Fixing the stupid 'search-beyond-end' bug:
		if(srcIndex+ENC_SIZE < srcSize) {
			SearchIntoSlidingWindow(&ShortMediumLongOFFSET,&index,&match,refStart,&src[srcIndex],&src[srcIndex],encEnd);
// Large_traffic_log_file_of_a_popular_website_fp.log:
// 1: Size: 2,000,938; Speed: 15051 B/s; NumberOfFullLiterals (lower-the-better): 579; Matches(24/48): 526,996/142,715
// 2: Size: 2,000,557; Speed: 10839 B/s; NumberOfFullLiterals (lower-the-better): 579; Matches(24/48): 526,920/142,857
// 3: Size: 2,000,557; Speed:  8323 B/s; NumberOfFullLiterals (lower-the-better): 579; Matches(24/48): 526,920/142,857
for (TsuyoHEURISTIC=1; TsuyoHEURISTIC<=3; TsuyoHEURISTIC++) // This is 'Zato' i.e. double 'Tsuyo' - the lookahead heuristic looks one more char ahead. [One level nested i.e. 1..2-1 due to tiny gains.]
{
// Tsuyo [
	if (ShortMediumLongOFFSET != 0)
	if (srcIndex+(1) < srcSize) {
		if(srcIndex+(1)>=REF_SIZE)
			refStart1=&src[srcIndex+(1)-REF_SIZE];
		else
			refStart1=src;
		if(srcIndex+(1)>=srcSize-ENC_SIZE)
			encEnd1=&src[srcSize];
		else
			encEnd1=&src[srcIndex+(1)+ENC_SIZE];
		if(srcIndex+(1)+ENC_SIZE < srcSize) {

				SearchIntoSlidingWindow(&ShortMediumLongOFFSET1,&index1,&match1,refStart1,&src[srcIndex+(1)],&src[srcIndex+(1)],encEnd1);
	if (ShortMediumLongOFFSET1 != 0)
				if (match/ShortMediumLongOFFSET +(1+1)  >= match1/ShortMediumLongOFFSET1) { // a brash heuristic
					break; // Exit the 'for' - simple heuristic, still not handling YES-NO-YES cases...
				}
				else {
					if (TsuyoHEURISTIC==3) TsuyoHEURISTICAPPLIED++;
					match=0; // Pretend nothing to find.

// 1of2
			if(notMatch==0){
				notMatchStart=&ret[retIndex];
				retIndex++;
			}
			//else if (notMatch==(127-64-32)) {
			else if (notMatch==(127-64-32-16 -(7))) {
NumberOfFullLiterals++;
				//*notMatchStart=(unsigned char)((127-64-32)<<3);
				*notMatchStart=(unsigned char)((127-64-32-16 -(7))<<(4));
				*notMatchStart=(unsigned char)((127-64-32-16 -(7))<<(4)) | 0x03; // Entag it as Literal
				notMatch=0;
				notMatchStart=&ret[retIndex];
				retIndex++;
			}
			ret[retIndex]=src[srcIndex];
			retIndex++;
			notMatch++;
			srcIndex++;
			if ((srcIndex-1) % (1<<16) > srcIndex % (1<<16)) {
				ProgressIndicator = (int)( (srcIndex+1)*(double)100/(srcSize+1) );
				SPD=(((double)CLOCKS_PER_SEC*srcIndex/(double)(clock() - clocks1 + 1)));
				if (SPD >999999) SPD=999999;
				//printf("%s; Each rotation means 64KB are encoded; Speed: %sB/s; Done %d%%; Compression Ratio: %.2f:1; Matches(24/32/48): %s/%s/%s    \r", Auberge[Melnitchka++], _ui64toaKAZEzerocomma(SPD, llTOaDigits4, 10)+(26-7) ,ProgressIndicator, (double)(srcIndex) / (double)(retIndex),  _ui64toaKAZEcomma(GLOBAL24B, llTOaDigits2, 10), _ui64toaKAZEcomma(GLOBAL32B, llTOaDigits, 10), _ui64toaKAZEcomma(GLOBAL48B, llTOaDigits3, 10) );
				Melnitchka = Melnitchka & 3; // 0 1 2 3: 00 01 10 11
			}
// 2of2
			ShortMediumLongOFFSET=ShortMediumLongOFFSET1;
			index=index1;
			match=match1;

				}

		}
	}
// Tsuyo ]
}
			if ( ShortMediumLongOFFSET==1 && match==4 ) GLOBALtinyT++;
			if ( ShortMediumLongOFFSET==1 && match==8 ) GLOBALshortT++;
			if ( ShortMediumLongOFFSET==1 && match==12 ) GLOBALmediumT++;
			if ( match==24 ) GLOBAL24B++;
			if ( match==32 ) GLOBAL32B++;
			if ( match==48 ) GLOBAL48B++;
			if ( match==3 ) GLOBAL3++;
		}
		else
			match=0; // Nothing to find.
		//if ( match<Min_Match_Length ) {
		//if ( match<Min_Match_Length || match<8 ) {
  		if ( match==0 ) {
			if(notMatch==0){
				notMatchStart=&ret[retIndex];
				retIndex++;
			}
			//else if (notMatch==(127-64-32)) {
			else if (notMatch==(127-64-32-16 -(7))) {
NumberOfFullLiterals++;
				//*notMatchStart=(unsigned char)((127-64-32)<<3);
				*notMatchStart=(unsigned char)((127-64-32-16 -(7))<<(4));
				*notMatchStart=(unsigned char)((127-64-32-16 -(7))<<(4)) | 0x03; // Entag it as Literal
				notMatch=0;
				notMatchStart=&ret[retIndex];
				retIndex++;
			}
			ret[retIndex]=src[srcIndex];
			retIndex++;
			notMatch++;
			srcIndex++;
			if ((srcIndex-1) % (1<<16) > srcIndex % (1<<16)) {
				ProgressIndicator = (int)( (srcIndex+1)*(double)100/(srcSize+1) );
				SPD=(((double)CLOCKS_PER_SEC*srcIndex/(double)(clock() - clocks1 + 1)));
				if (SPD >999999) SPD=999999;
				//printf("%s; Each rotation means 64KB are encoded; Speed: %sB/s; Done %d%%; Compression Ratio: %.2f:1; Matches(24/32/48): %s/%s/%s    \r", Auberge[Melnitchka++], _ui64toaKAZEzerocomma(SPD, llTOaDigits4, 10)+(26-7) ,ProgressIndicator, (double)(srcIndex) / (double)(retIndex),  _ui64toaKAZEcomma(GLOBAL24B, llTOaDigits2, 10), _ui64toaKAZEcomma(GLOBAL32B, llTOaDigits, 10), _ui64toaKAZEcomma(GLOBAL48B, llTOaDigits3, 10) );
				Melnitchka = Melnitchka & 3; // 0 1 2 3: 00 01 10 11
			}
		} else {
			if(notMatch > 0){
				*notMatchStart=(unsigned char)((notMatch)<<(4));
				*notMatchStart=(unsigned char)((notMatch)<<(4)) | 0x03; // Entag it as Literal
				notMatch=0;
			}
// ---------------------| 
//                     \ /

			//ret[retIndex] = 0x80; // Assuming seventh/fifteenth bit is zero i.e. LONG MATCH i.e. Min_Match_BAILOUT_Length*4
	  		//if ( match==Min_Match_BAILOUT_Length ) ret[retIndex] = 0xC0; // 8bit&7bit set, SHORT MATCH if seventh/fifteenth bit is not zero i.e. Min_Match_BAILOUT_Length
//                     / \
// ---------------------|
/*
			ret[retIndex] = 0x01; // Assuming seventh/fifteenth bit is zero i.e. LONG MATCH i.e. Min_Match_BAILOUT_Length*4
	  		if ( match==Min_Match_BAILOUT_Length ) ret[retIndex] = 0x03; // 2bit&1bit set, LONG MATCH if 2bit is not zero i.e. Min_Match_BAILOUT_Length
*/
// No need of above, during compression we demanded lowest 2bits to be not 00.
			// 1bit+3bits+12bits:
			//ret[retIndex] = ret[retIndex] | ((match-Min_Match_Length)<<4);
			//ret[retIndex] = ret[retIndex] | (((index-Min_Match_Length) & 0x0F00)>>8);
			// 1bit+1bit+14bits:
			//ret[retIndex] = ret[retIndex] | ((match-Min_Match_Length)<<(8-(LengthBITS+1))); // No need to set the matchlength
// The fragment below is outrageously ineffective - instead of 8bit&7bit I have to use the lower TWO bits i.e. 2bit&1bit as flags, thus in decompressing one WORD can be fetched instead of two BYTE loads followed by SHR by 2.
// ---------------------| 
//                     \ /
			//ret[retIndex] = ret[retIndex] | (((index-Min_Match_Length) & 0x3F00)>>8); // 2+4+8=14
			//retIndex++;
			//ret[retIndex] = (char)((index-Min_Match_Length) & 0x00FF);
			//retIndex++;
//                     / \
// ---------------------|
			// Now the situation is like LOW:HIGH i.e. FF:3F i.e. 0x3FFF, 16bit&15bit used as flags,
			// should become LOW:HIGH i.e. FC:FF i.e. 0xFFFC, 2bit&1bit used as flags.
/*
			ret[retIndex] = ret[retIndex] | (((index-Min_Match_Length) & 0x00FF)<<2); // 6+8=14
			//ret[retIndex] = ret[retIndex] | (((index-Min_Match_Length) & 0x00FF)<<1); // 7+8=15
			retIndex++;
			ret[retIndex] = (char)(((index-Min_Match_Length) & 0x3FFF)>>6);
			//ret[retIndex] = (char)(((index-Min_Match_Length) & 0x7FFF)>>7);
			retIndex++;
*/
// No need of above, during compression we demanded lowest 2bits to be not 00, use the full 16bits and get rid of the stupid '+/-' Min_Match_Length.
			//if (index>0xFFFF) {printf ("\nFatal error: Overflow!\n"); exit(13);}
			//memcpy(&ret[retIndex],&index,2+1); // copy lower 2 bytes
			//retIndex++;
			//retIndex++;
			 //retIndex++;
			memcpy(&ret[retIndex],&index,ShortMediumLongOFFSET);
			retIndex = retIndex + ShortMediumLongOFFSET;
//                     / \
// ---------------------|
			srcIndex+=match;
			if ((srcIndex-match) % (1<<16) > srcIndex % (1<<16)) {
				ProgressIndicator = (int)( (srcIndex+1)*(double)100/(srcSize+1) );
				SPD=(((double)CLOCKS_PER_SEC*srcIndex/(double)(clock() - clocks1 + 1)));
				if (SPD >999999) SPD=999999;
				//printf("%s; Each rotation means 64KB are encoded; Speed: %sB/s; Done %d%%; Compression Ratio: %.2f:1; Matches(24/32/48): %s/%s/%s    \r", Auberge[Melnitchka++], _ui64toaKAZEzerocomma(SPD, llTOaDigits4, 10)+(26-7) ,ProgressIndicator, (double)(srcIndex) / (double)(retIndex),  _ui64toaKAZEcomma(GLOBAL24B, llTOaDigits2, 10), _ui64toaKAZEcomma(GLOBAL32B, llTOaDigits, 10), _ui64toaKAZEcomma(GLOBAL48B, llTOaDigits3, 10) );
				Melnitchka = Melnitchka & 3; // 0 1 2 3: 00 01 10 11
			}
		}
	}
	if(notMatch > 0){
		*notMatchStart=(unsigned char)((notMatch)<<(4));
		*notMatchStart=(unsigned char)((notMatch)<<(4)) | 0x03; // Entag it as Literal
	}
	SPD=(((double)CLOCKS_PER_SEC*srcIndex/(double)(clock() - clocks1 + 1)));
	if (SPD >999999) SPD=999999;
				//printf("%s; Each rotation means 64KB are encoded; Speed: %sB/s; Done %d%%; Compression Ratio: %.2f:1; Matches(24/32/48): %s/%s/%s    \n", Auberge[Melnitchka++], _ui64toaKAZEzerocomma(SPD, llTOaDigits4, 10)+(26-7) ,100, (double)(srcIndex) / (double)(retIndex),  _ui64toaKAZEcomma(GLOBAL24B, llTOaDigits2, 10), _ui64toaKAZEcomma(GLOBAL32B, llTOaDigits, 10), _ui64toaKAZEcomma(GLOBAL48B, llTOaDigits3, 10) );

	//printf("NumberOfFullLiterals (lower-the-better): %d\n", NumberOfFullLiterals );
	//printf("Tsuyo_HEURISTIC_APPLIED_thrice_back-to-back: %d\n", TsuyoHEURISTICAPPLIED );
//printf("NumberOf(Micro)Matches[Tiny]Window (%d)[%d]: %d\n", 3, 256, GLOBAL3);
//printf("NumberOf(Tiny)Matches[Micro]Window (%d)[%d]: %d\n", 4, 16, GLOBALtinyT);
//printf("NumberOf(Short)Matches[Tiny]Window (%d): %d\n", 8, GLOBALshortT);
//printf("NumberOf(Medium)Matches[Tiny]Window (%d): %d\n", 12, GLOBALmediumT);
//printf("NumberOf(ExtraLong)Matches[Long]Window (%d): %d\n", 24, GLOBAL24B);
//printf("NumberOf(Huge)Matches[Long]Window (%d): %d\n", 48, GLOBAL48B);
	return retIndex;
}


uint64_t NakaDecompress (char* ret, char* src, uint64_t srcSize) {
	char* retLOCAL = ret;
	char* srcLOCAL = src;
	char* srcEndLOCAL = src+srcSize;
	unsigned int DWORDtrio;
	unsigned int DWORDtrioDumbo;
	unsigned int MatchLen;
	while (srcLOCAL < srcEndLOCAL) {
		DWORDtrio = *(unsigned int*)srcLOCAL;
		MatchLen = (DWORDtrio&0x0C); // 0|4|8|12
/*
#ifndef _N_GP
#ifdef _N_prefetch_64
		_mm_prefetch((char*)(srcLOCAL + 64), _MM_HINT_T0);
#endif
#ifdef _N_prefetch_128
		_mm_prefetch((char*)(srcLOCAL + 64*2), _MM_HINT_T0);
#endif
#ifdef _N_prefetch_4096
		_mm_prefetch((char*)(srcLOCAL + 64*64), _MM_HINT_T0);
#endif
#endif
*/

// Sizewise priority:
//  4:1= 4    (16B)   #01 32:2= 16   (256B)
//  8:1= 8    (16B)   #02 48:4= 12   (64MB)
// 12:1= 12   (16B)   #03 24:2= 12   (4KB)
// 16:1= Flag (16B)   #04 12:1= 12   (16B)
//  4:2= 2    (4KB)   #05 32:4= 8    (64MB)
//  6:2= 3    (1KB)   #06 24:3= 8    (1MB)
//  8:2= 4    (4KB)   #07 16:2= 8    (4KB)
// 12:2= 6    (4KB)   #08  8:1= 8    (16B)
// 16:2= 8    (4KB)   #09 24:4= 6    (256MB)
// 24:2= 12   (4KB)   #10 12:2= 6    (4KB)
// 32:2= 16   (256B)  #11 16:3= 5.3  (1MB)
//  4:3= 1.3  (1MB)   #12 16:4= 4    (16MB)F
//  8:3= 2.6  (1MB)   #13 16:4= 4    (64MB)S
// 12:3= 4    (1MB)   #14 12:3= 4    (1MB)
// 16:3= 5.3  (1MB)   #15  8:2= 4    (4KB)
// 24:3= 8    (1MB)   #16  4:1= 4    (16B)
//  6:4= 1.5  (16MB)  #17 14:4= 3.5  (16MB)
//  8:4= 2    (16MB)  #18 12:4= 3    (16MB)
// 10:4= 2.5  (16MB)  #19  6:2= 3    (1KB)
// 12:4= 3    (16MB)  #20  8:3= 2.6  (1MB)
// 14:4= 3.5  (16MB)  #21 10:4= 2.5  (16MB)
// 16:4= 4    (16MB)F #22  8:4= 2    (16MB)
// 16:4= 4    (64MB)S #23  4:2= 2    (4KB)
// 24:4= 6    (256MB) #24  6:4= 1.5  (16MB)
// 32:4= 8    (64MB)  #25  4:3= 1.3  (1MB)
// 48:4= 12   (64MB) 
// |1stLSB    |2ndLSB  |3rdLSB   |
// -------------------------------
// |OO|LL|xxxx|xxxxxxxx|xxxxxx|xx|
// -------------------------------
// [1bit          16bit]    24bit]
// LL = 00b means Long MatchLength, (4-LL)<<2 or 16
// LL = 01b means Long MatchLength, (4-LL)<<2 or 12
// LL = 10b means Long MatchLength, (4-LL)<<2 or 8
// LL = 11b means Long MatchLength, (4-LL)<<2 or 4
// xxxx0011b Literals for xxxx in 1..15-7, Matches for 10..15 i.e. Sliding Window is   4*8-8=24 or  16MB
// OO = 00b MatchOffset, 0xFFFFFFFF>>OO, 4 bytes long i.e. Sliding Window is 4*8-LL-OO=4*8-4=28 or 256MB
// OO = 01b MatchOffset, 0xFFFFFFFF>>OO, 3 bytes long i.e. Sliding Window is 3*8-LL-OO=3*8-4=20 or   1MB    
// OO = 10b MatchOffset, 0xFFFFFFFF>>OO, 2 bytes long i.e. Sliding Window is 2*8-LL-OO=2*8-4=12 or   4KB    
// OO = 11b MatchOffset, 0xFFFFFFFF>>OO, 1 byte long  i.e. Sliding Window is 1*8-LL-OO=1*8-4=4 or   16B     
		if ( (DWORDtrio & 0x0F) == 0x03 ) {
		if ( ((DWORDtrio & 0xFF)>>4) <= 8 ) { // 9 not used, only 1..8
			if ( ((DWORDtrio & 0xFF)>>4) == 0 ) {
			#ifdef _N_GP
				memcpy(retLOCAL, (const char *)( (uint64_t)(retLOCAL-((DWORDtrio&0xFFFF)>>8))) ), 32); // No need of DWORDtrio&0xFFFFFFFF
			#endif
			#ifdef _N_XMM
				NotSoSlowCopy128bit( (const char *)( (uint64_t)(retLOCAL-((DWORDtrio&0xFFFF)>>8))+16*(0) ), retLOCAL +16*(0));
				NotSoSlowCopy128bit( (const char *)( (uint64_t)(retLOCAL-((DWORDtrio&0xFFFF)>>8))+16*(1) ), retLOCAL +16*(1));
			#endif
			retLOCAL+= 32; 
			srcLOCAL+= 2;
			} else {
//			#ifdef _N_GP
//				memcpy(retLOCAL, (const char *)( (uint64_t)(srcLOCAL+1) ), 16);
//			#endif
//			#ifdef _N_XMM
//				NotSoSlowCopy128bit( (const char *)( (uint64_t)(srcLOCAL+1) ), retLOCAL );
//			#endif
			*(uint64_t*)(retLOCAL) = *(uint64_t*)((const char *)( (uint64_t)(srcLOCAL+1) )); 
			retLOCAL+= ((DWORDtrio & 0xFF)>>4);
			srcLOCAL+= ((DWORDtrio & 0xFF)>>4)+1;
			}
		} else {
			#ifdef _N_GP
				memcpy(retLOCAL, (const char *)( (uint64_t)(retLOCAL-(DWORDtrio>>8))) ), 16); // No need of DWORDtrio&0xFFFFFFFF
			#endif
			#ifdef _N_XMM
				NotSoSlowCopy128bit( (const char *)( (uint64_t)(retLOCAL-(DWORDtrio>>8))+16*(0) ), retLOCAL +16*(0));
			#endif
			retLOCAL+= (18 - ((DWORDtrio & 0xFF)>>4))<<1; 
			srcLOCAL+= 4; // 6|8|10|12|14|16 in 16MB window
		}
		} else if ( (DWORDtrio & 0x0f) == 0x0C ) { 
			if ( (DWORDtrio & 0x30) == 0 ) {
			// 6:2 in 256B x 4 = 1KB window
			*(uint64_t*)(retLOCAL+8*(0)) = *(uint64_t*)((retLOCAL-( (DWORDtrio&(0xFFFF))>>6 ))+8*(0)); 
			retLOCAL+= 6;
			srcLOCAL+= 2;
			} else {
			#ifdef _N_GP
				memcpy(retLOCAL, (const char *)( (uint64_t)(retLOCAL-(DWORDtrio>>6)) ), 48);
			#endif
			#ifdef _N_XMM
				NotSoSlowCopy128bit( (const char *)( (uint64_t)(retLOCAL-(DWORDtrio>>6))+16*(0) ), retLOCAL +16*(0));
				NotSoSlowCopy128bit( (const char *)( (uint64_t)(retLOCAL-(DWORDtrio>>6))+16*(1) ), retLOCAL +16*(1));
				NotSoSlowCopy128bit( (const char *)( (uint64_t)(retLOCAL-(DWORDtrio>>6))+16*(2) ), retLOCAL +16*(2));
			#endif
			retLOCAL+= (DWORDtrio & 0x30); // 16|32|48
			srcLOCAL+= 4; // 48:4
			}
		} else if ( (DWORDtrio & 0x03) == 0x00 ) {
			// MatchLen 0|4|8 <<1 0|8|16
			MatchLen=MatchLen<<1; // To avoid 'LEA'
			*(uint64_t*)(retLOCAL+8*(0)) = *(uint64_t*)((retLOCAL-( (DWORDtrio&(0xFFFFFFFF>>MatchLen))>>4 ))+8*(0)); 
			*(uint64_t*)(retLOCAL+8*(1)) = *(uint64_t*)((retLOCAL-( (DWORDtrio&(0xFFFFFFFF>>MatchLen))>>4 ))+8*(1)); 
			*(uint64_t*)(retLOCAL+8*(2)) = *(uint64_t*)((retLOCAL-( (DWORDtrio&(0xFFFFFFFF>>MatchLen))>>4 ))+8*(2)); 
			retLOCAL+= 24;
			srcLOCAL+= 4-(MatchLen>>3); // 24:2, 24:3, 24:4
		} else {
			DWORDtrioDumbo = (DWORDtrio & 0x03)<<3; // To avoid 'LEA'
			#ifdef _N_GP
				memcpy(retLOCAL, (const char *)( (uint64_t)(retLOCAL-((DWORDtrio&(0xFFFFFFFF>>DWORDtrioDumbo))>>4)) ), 16);
			#endif
			#ifdef _N_XMM
				NotSoSlowCopy128bit( (const char *)( (uint64_t)(retLOCAL-((DWORDtrio&(0xFFFFFFFF>>DWORDtrioDumbo))>>4)) ), retLOCAL );
			#endif
			retLOCAL+= 16-MatchLen;
			srcLOCAL+= 4-(DWORDtrioDumbo>>3);
		}
	}        
	return (uint64_t)(retLOCAL - ret);
}

/*
; 'Okamigan' decompression loop, 191-21+6=374 bytes long, 108 instructions long:
; mark_description "Intel(R) C++ Intel(R) 64 Compiler XE for applications running on Intel(R) 64, Version 15.0.0.108 Build 20140";
; mark_description "-TP -O3 -QxSSE4.1 -D_N_XMM -D_N_prefetch_4096 -D_N_HIGH_PRIORITY -FAcs";

.B2.3::                         
  00021 45 8b 10         mov r10d, DWORD PTR [r8]               
  00024 45 89 d3         mov r11d, r10d                         
  00027 44 89 d5         mov ebp, r10d                          
  0002a 41 83 e3 0f      and r11d, 15                           
  0002e 83 e5 0c         and ebp, 12                            
  00031 41 83 fb 03      cmp r11d, 3                            
  00035 75 75            jne .B2.9 
.B2.4::                         
  00037 41 0f b6 ea      movzx ebp, r10b                        
  0003b c1 ed 04         shr ebp, 4                             
  0003e 83 fd 08         cmp ebp, 8                             
  00041 77 44            ja .B2.8 
.B2.5::                         
  00043 85 ed            test ebp, ebp                          
  00045 74 14            je .B2.7 
.B2.6::                         
  00047 49 8b 48 01      mov rcx, QWORD PTR [1+r8]              
  0004b 48 89 08         mov QWORD PTR [rax], rcx               
  0004e 48 03 c5         add rax, rbp                           
  00051 ff c5            inc ebp                                
  00053 4c 03 c5         add r8, rbp                            
  00056 e9 33 01 00 00   jmp .B2.16 
.B2.7::                         
  0005b 41 0f b7 ca      movzx ecx, r10w                        
  0005f 49 83 c0 02      add r8, 2                              
  00063 c1 e9 08         shr ecx, 8                             
  00066 48 f7 d9         neg rcx                                
  00069 48 03 c8         add rcx, rax                           
  0006c f2 0f f0 01      lddqu xmm0, XMMWORD PTR [rcx]          
  00070 f3 0f 7f 00      movdqu XMMWORD PTR [rax], xmm0         
  00074 f2 0f f0 49 10   lddqu xmm1, XMMWORD PTR [16+rcx]       
  00079 f3 0f 7f 48 10   movdqu XMMWORD PTR [16+rax], xmm1      
  0007e 48 83 c0 20      add rax, 32                            
  00082 e9 07 01 00 00   jmp .B2.16 
.B2.8::                         
  00087 41 c1 ea 08      shr r10d, 8                            
  0008b 49 83 c0 04      add r8, 4                              
  0008f f7 dd            neg ebp                                
  00091 49 f7 da         neg r10                                
  00094 4c 03 d0         add r10, rax                           
  00097 8d 4c 2d 24      lea ecx, DWORD PTR [36+rbp+rbp]        
  0009b f2 41 0f f0 02   lddqu xmm0, XMMWORD PTR [r10]          
  000a0 f3 0f 7f 00      movdqu XMMWORD PTR [rax], xmm0         
  000a4 48 03 c1         add rax, rcx                           
  000a7 e9 e2 00 00 00   jmp .B2.16 
.B2.9::                         
  000ac 41 83 fb 0c      cmp r11d, 12                           
  000b0 75 5e            jne .B2.13 
.B2.10::                        
  000b2 41 f7 c2 30 00 
        00 00            test r10d, 48                          
  000b9 75 20            jne .B2.12 
.B2.11::                        
  000bb 41 0f b7 ca      movzx ecx, r10w                        
  000bf 49 83 c0 02      add r8, 2                              
  000c3 c1 e9 06         shr ecx, 6                             
  000c6 48 f7 d9         neg rcx                                
  000c9 48 03 c8         add rcx, rax                           
  000cc 48 8b 29         mov rbp, QWORD PTR [rcx]               
  000cf 48 89 28         mov QWORD PTR [rax], rbp               
  000d2 48 83 c0 06      add rax, 6                             
  000d6 e9 b3 00 00 00   jmp .B2.16 
.B2.12::                        
  000db 44 89 d1         mov ecx, r10d                          
  000de 49 83 c0 04      add r8, 4                              
  000e2 c1 e9 06         shr ecx, 6                             
  000e5 48 f7 d9         neg rcx                                
  000e8 48 03 c8         add rcx, rax                           
  000eb 49 83 e2 30      and r10, 48                            
  000ef f2 0f f0 01      lddqu xmm0, XMMWORD PTR [rcx]          
  000f3 f3 0f 7f 00      movdqu XMMWORD PTR [rax], xmm0         
  000f7 f2 0f f0 49 10   lddqu xmm1, XMMWORD PTR [16+rcx]       
  000fc f3 0f 7f 48 10   movdqu XMMWORD PTR [16+rax], xmm1      
  00101 f2 0f f0 51 20   lddqu xmm2, XMMWORD PTR [32+rcx]       
  00106 f3 0f 7f 50 20   movdqu XMMWORD PTR [32+rax], xmm2      
  0010b 49 03 c2         add rax, r10                           
  0010e eb 7e            jmp .B2.16 
.B2.13::                        
  00110 44 89 d1         mov ecx, r10d                          
  00113 83 e1 03         and ecx, 3                             
  00116 75 41            jne .B2.15 
.B2.14::                        
  00118 03 ed            add ebp, ebp                           
  0011a 41 bb ff ff ff 
        ff               mov r11d, -1                           
  00120 89 e9            mov ecx, ebp                           
  00122 41 d3 eb         shr r11d, cl                           
  00125 45 23 d3         and r10d, r11d                         
  00128 41 c1 ea 04      shr r10d, 4                            
  0012c 49 f7 da         neg r10                                
  0012f 4c 03 d0         add r10, rax                           
  00132 c1 ed 03         shr ebp, 3                             
  00135 f7 dd            neg ebp                                
  00137 83 c5 04         add ebp, 4                             
  0013a 4d 8b 1a         mov r11, QWORD PTR [r10]               
  0013d 4c 89 18         mov QWORD PTR [rax], r11               
  00140 4d 8b 5a 08      mov r11, QWORD PTR [8+r10]             
  00144 4c 89 58 08      mov QWORD PTR [8+rax], r11             
  00148 4d 8b 52 10      mov r10, QWORD PTR [16+r10]            
  0014c 4c 89 50 10      mov QWORD PTR [16+rax], r10            
  00150 4c 03 c5         add r8, rbp                            
  00153 48 83 c0 18      add rax, 24                            
  00157 eb 35            jmp .B2.16 
.B2.15::                        
  00159 c1 e1 03         shl ecx, 3                             
  0015c 41 bb ff ff ff 
        ff               mov r11d, -1                           
  00162 41 d3 eb         shr r11d, cl                           
  00165 f7 dd            neg ebp                                
  00167 45 23 d3         and r10d, r11d                         
  0016a 83 c5 10         add ebp, 16                            
  0016d 41 c1 ea 04      shr r10d, 4                            
  00171 49 f7 da         neg r10                                
  00174 4c 03 d0         add r10, rax                           
  00177 c1 e9 03         shr ecx, 3                             
  0017a f7 d9            neg ecx                                
  0017c 83 c1 04         add ecx, 4                             
  0017f f2 41 0f f0 02   lddqu xmm0, XMMWORD PTR [r10]          
  00184 f3 0f 7f 00      movdqu XMMWORD PTR [rax], xmm0         
  00188 48 03 c5         add rax, rbp                           
  0018b 4c 03 c1         add r8, rcx                            
.B2.16::                        
  0018e 4d 3b c1         cmp r8, r9                             
  00191 0f 82 8a fe ff 
        ff               jb .B2.3 
*/

/*
; 'Tengu' decompression loop, 7f-1a+2=103 bytes long:
; mark_description "Intel(R) C++ Intel(R) 64 Compiler XE for applications running on Intel(R) 64, Version 12.1.1.258 Build 20111";
; mark_description "-O3 -D_N_GP -FAcs";

.B6.3::                         
  0001a 8b 02            mov eax, DWORD PTR [rdx]               
  0001c 89 c1            mov ecx, eax                           
  0001e 83 e1 03         and ecx, 3                             
  00021 75 1f            jne .B6.6 
.B6.4::                         
  00023 48 8b 6a 01      mov rbp, QWORD PTR [1+rdx]             
  00027 49 89 2a         mov QWORD PTR [r10], rbp               
  0002a 4c 8b 5a 09      mov r11, QWORD PTR [9+rdx]             
  0002e 4d 89 5a 08      mov QWORD PTR [8+r10], r11             
.B6.5::                         
  00032 0f b6 c0         movzx eax, al                          
  00035 c1 e8 04         shr eax, 4                             
  00038 4c 03 d0         add r10, rax                           
  0003b ff c0            inc eax                                
  0003d 48 03 d0         add rdx, rax                           
  00040 eb 3a            jmp .B6.8 
.B6.6::                         
  00042 c1 e1 03         shl ecx, 3                             
  00045 bd ff ff ff ff   mov ebp, -1                            
  0004a d3 ed            shr ebp, cl                            
  0004c 23 e8            and ebp, eax                           
  0004e c1 ed 04         shr ebp, 4                             
  00051 48 f7 dd         neg rbp                                
  00054 49 03 ea         add rbp, r10                           
  00057 4c 8b 5d 00      mov r11, QWORD PTR [rbp]               
  0005b 4d 89 1a         mov QWORD PTR [r10], r11               
  0005e 4c 8b 5d 08      mov r11, QWORD PTR [8+rbp]             
  00062 4d 89 5a 08      mov QWORD PTR [8+r10], r11             
.B6.7::                         
  00066 c1 e9 03         shr ecx, 3                             
  00069 83 e0 0c         and eax, 12                            
  0006c f7 d8            neg eax                                
  0006e f7 d9            neg ecx                                
  00070 83 c0 10         add eax, 16                            
  00073 83 c1 04         add ecx, 4                             
  00076 4c 03 d0         add r10, rax                           
  00079 48 03 d1         add rdx, rcx                           
.B6.8::                         
  0007c 49 3b d0         cmp rdx, r8                            
  0007f 72 99            jb .B6.3 
*/

/*
; 'Tengu' decompression loop, 70-11+2=97 bytes long:
; mark_description "Intel(R) C++ Intel(R) 64 Compiler XE for applications running on Intel(R) 64, Version 12.1.1.258 Build 20111";
; mark_description "-O3 -QxSSE2 -D_N_XMM -FAcs";

.B7.3::                         
  00011 8b 02            mov eax, DWORD PTR [rdx]               
  00013 89 c1            mov ecx, eax                           
  00015 83 e1 03         and ecx, 3                             
  00018 75 1a            jne .B7.5 
.B7.4::                         
  0001a 0f b6 c0         movzx eax, al                          
  0001d f3 0f 6f 42 01   movdqu xmm0, XMMWORD PTR [1+rdx]       
  00022 c1 e8 04         shr eax, 4                             
  00025 f3 41 0f 7f 01   movdqu XMMWORD PTR [r9], xmm0          
  0002a 4c 03 c8         add r9, rax                            
  0002d ff c0            inc eax                                
  0002f 48 03 d0         add rdx, rax                           
  00032 eb 39            jmp .B7.6 
.B7.5::                         
  00034 c1 e1 03         shl ecx, 3                             
  00037 41 bb ff ff ff 
        ff               mov r11d, -1                           
  0003d 41 d3 eb         shr r11d, cl                           
  00040 44 23 d8         and r11d, eax                          
  00043 83 e0 0c         and eax, 12                            
  00046 41 c1 eb 04      shr r11d, 4                            
  0004a f7 d8            neg eax                                
  0004c 83 c0 10         add eax, 16                            
  0004f 49 f7 db         neg r11                                
  00052 4d 03 d9         add r11, r9                            
  00055 c1 e9 03         shr ecx, 3                             
  00058 f7 d9            neg ecx                                
  0005a 83 c1 04         add ecx, 4                             
  0005d f3 41 0f 6f 03   movdqu xmm0, XMMWORD PTR [r11]         
  00062 f3 41 0f 7f 01   movdqu XMMWORD PTR [r9], xmm0          
  00067 4c 03 c8         add r9, rax                            
  0006a 48 03 d1         add rdx, rcx                           
.B7.6::                         
  0006d 49 3b d0         cmp rdx, r8                            
  00070 72 9f            jb .B7.3 
*/

/*
RT;.,:;:,:;;;:,:;:;:;:,:,:;:;:;:;:;:,:,:;:,:,:;:,:,:;:,:;:,:;:;:,:;:;:,:,:;:,:,:,:,:;:,:;:;:,:;:,:,:,:,:;:;:,:;:;:;:,:;:;:;:;:,:,:,:;:,:,:,:,:,:;:;:,:,:,:;:;:;:,:;:;:,:,:,:;:,:;:,:,:;:,:,:,:;:;:;:;:;:,:;:;:,:,:;:,:,:,:,:,:
.:.,:,:;:,.,:,:;:,:;:,:,:;:;:;:,:;:,:,:;:,:;:;:,:,:,:,:;:;:;:,:;:;:;:,:;:;:;:;:;:;:,:;:;:;:;:;:,:,:;:;:,:;:;:;:;:;:,:,:;:,:;:,:,:,:;:,:;:;:;:;:,:,:,:;:,:;:;:;:,:,:,:;:;:;:;:;:;:;:,:,:,:;:,:;:;:,:,:;:,:,:;:,:;:,:,:;:;:,:;:,
,.,.;:;:;:,:,:,:,:,:,:,:;:,:;:;:,:;:;:;:;:;:,:,:,:;:;:,:,:,:,:;:,:;:;:;:;:,:,:;:;:;:;:;:,:,:,:,:;:;:,:,:,:;:,:,:,:,:;:,:;:,:;:,:,:;:;:;:;:,:;:;:,:,:,:,:;:,:;:;:;:;:,:,:,:;:;:;:;:;:;:;:,:;:;:,:;:;:,:;:,:;:,:,:;:;:;:,:;:,:;:
:,:;:;:,:,:;:;:;:;:,:,:,:;:,:;:,:;:;:;:;:,:,:;:,:,:,:;:,:;:,:,:,:,:;:,:,:,:,:,:;:;:,:,:,:,:;:;:,:;:;:,:,:;:,:;:,:;:;:;:;:;:,:;:,:,:,:;:,:;:;:,:,:,:,:;:;:;:;:;:,:,:;:;:,:,:;:;:;:;:;:;:,:;:,:;:;:;:,:,:;:,:,:;:;:;:;:,:,:;:,:,
;:;:,:;:;:;:,:;:,:;:;:;:;:;:;:,:;:,:;:;:;:;:;:,:,:,:,:;:,:;:,:,:,:,:,:;:,:,:;:,:;:;:;:;:,:,:,:;:;:;:,:;:;:;:;:;:,:;:;:;:,:,:,:,:;:,:;:;:,:,:,:,:;:;:;:;:;:,:;:;:,:,:;:;:;:,:;:,:,:;:,:,:;:;:,:;:,:;:;:,:,:,:,:,:;:;:,:;:,:,:,:
:;:;:,:;:;:;:,:;:;:,:,:,:,:,:;:;:,:,:;:;:;:,:;:;:;:,:;:,:;:,:,:,:,:;:,:,:;:;:,:;:;:,:,:,:;:,:,:,:;:;:;:;:;:;:;:,:,:,:,:;:;:,:;:;:;:,:,:,:,:;:;:;:;:,:;:;:,:;:;:,:,:,:,:,:;:;:;:,:;:;:,:,:;:,:,:,:,:,:;:,:,:;:,:;:;:;:;:;:,:;:;
;:;:;:,:;:,:,:,:;:,:;:;:,:,:;:,:,:,:;:,:;:;:;:;:,:;:;:;:,:,:;:;:;:,:;:;:;:;:,:,:,:;:;:,:,:,:,:;:,:,:;:;:;:,:;:,:,:,:;:,:,:,:,:;:;:;:,:,:;:;:;:,:;:,:;:;:;:;:;:,:;:,:,:,:,:,:,:;:;:;:,:,:,:,:,:,:,:;:,:;:,:;:;:,:,:;:;:,:,:,:,:
,,:;:,:;:,:,:,:,:,:;:,:;:;:,:,:,:;:,:,:,:,:;:;:;:,:;:,:,:;:;:;:,:;:,:,:,:,:,:,:;:;:;:;:,:;:,:,:,:;:,:,:;:,:,:,:,:;:;:,:;:,:,:;:;:,:;:,:;:,:,:;:,:;:;:;:;:;:,:;:;:,:,:,:,:;:;:;:;:,:;:;:,:,:;:;:;:;:;:,:;:,:;:;:,:;:,:,:;:,:;:,
;:,:,:;:,:,:;:;:;:,:;:;:;:;:;:;:;:;:,:;:;:,:,:;:,:,:,:;:,:,:,:,:,:;:,:,:,:,:;:,:,:,:;:,:;:;:,:;:;:;:;:,:,:;:;:;:,:;:,:,:,:,:;:;:,:;:,:,:;:,:;:,:;:;:,:;:,:,:;:;:,:;:;:;:,:,:,:;:,:;:;:,:;:,:,:,:;:,:;:;:,:;:,:;:,:;:;:,:,:,:,:
:,:;:;:,:,:,:,:,:;:;:,:;:;:;:,:;:;:;:;:,:;:;:,:,:;:;:;:,:;:,:;:,:,:;:;:;:;:;:,:;:;:,:;:,:;:;:,:;:,:,:,:;:,:;:;:;:,:;:;:;:,:,:,:,:;:;:,:;:,:,:,:,:;:;:,:,:;:,:;:,:;:,:;:,:;:;:;:,:;:,:,:,:;:,:,:;:;:;:;:;:;:,:,:;:;:;:;:;:,:;:;
;:;:;:,:,:,:;:;:;:;:,:,:;:;:,:;:;:;:;:,:,:;:;:;:,:;:,:,:,:,:;:;:,:,:;:,:,:,:;:,:,:;:,:,:,:;:;:,:,:,:;:,:,:,:,:,:,:;:,:;:;:,:,:;:,:,:;:;:,:;:;:,:;:,:,:,:,:,:,:,:;:,:;:,:,:;:;:,:;:;:,:,:;:;:;:;:;:,:,:;:,:,:,:;:;:;:;:;:,:,:;:
,;:,:,:,:;:,:,:,:;:,:,:;:,:,:;:,:;:;:,:,:;:,:;:;:;:,:;:;:,:,:,:,:,:;:;:;:,:;:;:,:;:;:;:,:;:;:,:;:,:;:,:;:;:,:;:;:,:;:,:,:,:,:;:,:,:,:;:;:;:,:,:;:;:;:,:,:,:;:,:;:,:;:;:;:,:;:,:,:,:;:,:,:;:,:;:,:,:;:;:;:;:;:,:,:;:,:;:;:;:,:,
;:;:;:;:;:;:;:;:;:;:;:;:;:;:;:,:,:;:;:,:;:,:,:,:;:;:;:,:,:,:,:,:;:;:,:;:;:,:,:;:;:,:;:,:;:;:,:;:,:,:,:;:;:;:,:;:;:,:,:,:,:;:,:,:,:,:;:;:,:;:;:;:,:,:,:,:,:,:,:;:,:,:;:,:,:,:;:;:;:;:,:;:;:;:;:,:;:;:,:,:,:;:;:;:,:;:;:;:,:,:,:
,;:;:;:,:;:,:;:,:;:;:,:,:,:,:,:;:;:,:,:;:,:,:,:,:;:,:,:,:,:,:;:;:;:,:;:,:,:;:;,,:,.,:;:;:;:,:;:;:,:,:;:;:,:,:;:;:,:,:,:;:,:;:,:;:;:;:;:,:,:;:;:;:,:,:,:,:,:;:;:;:;:;:,:;:;:;:;:;:,:;:,:,:;:,:,:;:;:,:;:,:;:;:;:,:,:;:;:,:;:;:,
;:;:,:;:;:,:,:,:;:,:,:,:;:,:;:,:,:;:,:;:;:,:;:,:;:,:,:,:;:,:,:,:;:,:;:,:,:,,;:;.,:;.;.,.,:,:,:;:;:,:;:;:;:;:;:,:;:,:,:,:,:;:,:,:;:,:,:;:,:;:,:,:,:,:,:,:,:;:;:,:;:,:;:,:;:;:;:;:,:,:;:;:;:;:,:;:;:;:,:,:;:;:,:;:,:,:;:,:;:,:,:
:;:,:;:,:;:,:,:,:,:,:;:;:,:;:;:,:;:;:;:;:;:;:;:;:;:,:,:,:,:,:;:;:;:,:,:,:;.,:,..,i;,.,;,.,:;:,:,:;:;:;:,:,:;:,:;:,:;:,:,:,:,:,:;:;:,:;:;:,:,:,:,:,:,:;:;:;:;:,:;:,:;:,:,:,:;:,:;:,:;:,:,:;:;:,:;:;:,:,:,:,:,:,:;:,:;:,:,:,:,:,
;:,:;:;:;:;:,:,:,:;:;:;:,:;:;:,:;:;:,:,:,:,:,:,:;:,:;:,:;:,:,:;:,:;:;:,:;:,.: ,iyvl;:i; ,.,,,:,:,:;:,:;:,:,:,:,:,:,:;:;:,:;:;:,:;:,:;:;:;:,:;:;:;:;:,:,:,:;:,:,:,:;:;:;:,:,:;:,:,:,:,:;:,:;:,:;:,:,:,:;:,:,:,:,:,:,:,:,:;:;:;:
:,:,:,:;:;:,:,:;:,:,:;:,:,:,:;:,:;:,:,:;:;:,:;:;:,:;:,:;:;:;:,:,:;:,:,.,,,...;vOXViiv3VYi,.,:;:,:;:;:,:;:;:;:;:,:;:,:;:;:,:,:;:;:,:,:;:;:;:;:;:;:;:,:,:;:,:,:,:,:;:,:;:,:;:,:;:;:;:,:;:;:;:,:,:;:,:,:,:,:;:,:,:;:;:;:;:,:;:,:,
;:;:;:;:,:,:,:;:;:,:,:;:;:;:;:;:;:;:;:;:,:,:;:;:;:;:;:,:,:;:,.,:,.,.,.,.,,ivtQ@DFVlv2K8BL...,.,:,:;,;:,:;:,:;:;:;:;:;:;:;:,:,:;:,:,:;:;:,:,:;:;:,:;:;:;:;:;:;:;:,:,:;:,:;:;:,:;:,:,:;:;:,:,:;:;:,:,:,:,:,:;:,:,:,:;:;:,:,:;:,:
,;:;:,:;:,:,:,:,:,:,:,:;:,:;:;:;:;:;:,:,:,:,:;:;:,:,:,:;:,.:.,.. : ..:.;ijOBE@B@E@@@Bt;yVi;;.:.,.,:,.,.,:,:;:;:,:,:,:,:;:;:;:;:,:;:,:;:,:,:;:;:,:,:;:,:,:;:;:;:;:;:,:;:;:;:,:,:;:;:;:;:,:;:,:,:,:,:;:;:,:,:;:,:;:,:;:;:,:;:;:,
;:,:,:,:;:;:;:;:,:,:,:,:;:;:,:,:;:,:;:;:,:,:;:;:,.,:,:,:,.: :ii;;   ,;ijZS8B@8B@@@@B@ECVylLyi;;,,.,.. ..,:;:;:,:;:;:;:,:;:,:,:,:,:,:,:;:;:,:,:,:;:;:,:,:;:,:;:,:,:;:;:;:;:;:;:;:;:;:,:;:,:;:;:;:,:;:;:,:,:;:,:;:;:;:,:;:,:,:,:
,,:;:;:;:;:;:;:;:,:,:,:;:;:,:;:,:;:,:;:;:,:;:,:,:,.,.,.,.;i;lBlyEtjOF16BB@EBQQQ@B88BE@8XvUVlivvi;;. ,yi:.,:;:,:;:,:;:,:;:;:;:,.,:,:;:,:,:;:;:,:;:,:;:;:,:,:,:,:,:;:;:,:,:;:,:,:;:,:,:,:,:;:,:,:;:,:,:,:;:;:,:,:,:,:,:;:,:,:,:,
;:;:;:;:,:,:;:,:,:;:;:;:,:;:;:,:;:,:;:,:;:,:,.,.;,,,;.:.iV336ScX88@B@B@EBB@8S1C3MyF8B8SlLUciyvVi;ii;@B@i..,:;:,:,:,:;:;:,:;.,.. :.;:,:;:;:,:,:,:,:,:,:,:,:;:;:,:;:,:,:;:;:;:;:,:;:,:;:,:,:,:;:,:;:,:;:;:,:,:,:,:,:;:;:,:,:;:,:
,;:,:,:,:,:;:,:,:;:;:,:;:,:,:,:,:;:;:,:;:,:,:;,;;i;Yl;,iYJljlyt$OEE@@@EQZB@@ED32G0KEG2FX1McCG$XVyvi;i@0;.,,,:;:;:;:,:;:,:;:, .i; ::;:;:;:;:;:,:;:;:,:,:,:;:;:,:;:,:,:,:;:,:;:,:,:,:;:,:;:;:,:,:;:,:;:,:;:;:;:;:;:,:;:;:;:,:,:,
;:;:,:;:;:,:;:,:;:;:;:,:;:,:,:;:;:,:;:;:,:;.,;i;yiivl;;;;;yLVlK3IX3D@B@BBB@88Z8@@83KOXIGSICIBBQ1Jly;;;, ::;:,:,,;:;:,:;:;:,. $@@; :.;:;:,:,:,:,:,:;:;:;:,:,:,:;:,:,:,:;:;:,:,:;:;:,:;:,:,:,:,:,:;:,:,:;:;:,:;:;:,:;:,:;:,:,:,:
:;:;:;:;:,:,:;:;:,:;:,:;:,:,:;:;:;:,:;:;,,:;;;it1cyVii;;,;iSE$XMj11tllJ8@@B8@@@Ei. .O@SGZ@DQQ8SKL1VLvi;;.,.;:;:,:,:,:;:;,,.. @B@2, ,:;:;:,:;:;:;:;:,:;:;:;:,:,:;:;:,:,:;:,:;:,:,:;:,:;:;:;:,:,:;:,:;:;:,:,:;:;:;:,:,:;:;:,:;:;
;:,:;:,:;:,:,:;:,:,:,:;:,:,:,:;:;:;:,:,:;:;,;ijVJVFUyii;;,;;U8BXOUi;iiivEBB@@Q;    i@B6Q@BBQDI3XFKMGOii;;.,.;:;:;:,:;:,:,:, .B@B@; .,:,:;:,:;:,:,:;:,:;:;:;:,:;:;:;:;:;:,:;:,:;:;:,:,:,:,:,:,:,:;:;:,:,:;:;:,:,:,:;:;:;:;:;:;:
:,:,:;:,:;:;:,:;:;:,:;:,:,:;:;:;:;:,:,.;:;:;;yiVUJcFUYii;;,..y0QZ1ivc0XVC8@@I     ;@@@@@@@@8O6Z$F68@8L;;;i,:.,:,,,:;:,:,:,. ,@@8@l .:,:,:;:;:;:;:,:,:,:,:;:,:,:;:,:;:;:;:;:;:,:;:,:,:,:,:;:;:;:,:;:;:;:;:,:,:;:;:;:;:;:,:;:,:,
;:;:;:,:;:;:,:;:,:,:,:,:;:,:,:;:,:;,;.;:;:,;lvlc3XVl2cYi;,;::.Y@BMK0803CE@@Di   . jB@B@@@DCUS@@B@B@B@MUi;;v;;:,.,:,.;:;:;.. 0@@X@Z; ,:;:;:;:,:,:,:,:;:;:;:;:;:;:;:,:;:;:;:,:,:;:,:;:,:,:;:,:,:,:;:;:,:;:,:;:;:,:;:;:;:;:,:;:,:
,,:,:;:;:,:,:;:,:,:;:;:;:,:,:;,,:;,;,,:;:,,lvUXCFL;iyXVyiiii;;v@@DI88St2E@8F$8KU.,B@@@B@iSi;V68@@@@@E6SU;iii;;,;.,.,:,:;::  B@8K8@; .,:,:,:;:,:;:,:,:,:;:;:,:,:,:;:;:,:;:;:;:,:;:,:;:,:,:,:;:;:;:,:;:,:;:;:,:;:,:,:,:;:;:;:;:,
;:,:;:,:;:,:;:;:,:,:,:;:;:,:;,;,,.;:;;;:,.iLVYMXLi;;iYJVUlyvv;y8@ZSQ80O2@@@B@B@BUO@@@@@jv1;.;;SB@DB@@0QMVvli;,;;i;;,,.,.,. ;@@GYEBv ::,:;:,:,:,:,:;:;:,:,:,:;:;:;:;:;:,:,:,:;:,:;:,:;:;:;:,:,:;:,:;:,:;:;:;:;:;:,:;:,:;:;:,:;:
:;:,:,:;:;:;:;:,:,:,:,:;:;:,:,.,:,:,:;;;:iUFJVYUiiiiiUcJVVvyviv8B8686SKCB@@@B@@@B@@@ytLVY;.,:iY@6IZBFCQSKSKJi;;;;;;i;;,,. .@@@FUE@v .,:;:;:;:,:;:,:,:,:;:;:;:,:;:;:;:,:,:;:;:;:;:,:;:;:;:;:;:,:,:;:,:;:;:,:;:,:,:,:,:;:,:,:,:,
;:,:,:;:;:;:;:,:;:;:;:;:,:;:;:;,;,;,,.,:;lXV2JYiiitVvjFVVvlvv;vS@8@88SOK@B@@@@@@@B@; ;Vi,.,:;;Y3QFS$L1@DDQ8QCyjl;.;;;;i;iilSECt2@@V ,:;:;:,:,:;:;:;:;:,:;:,:;:,:;:;:;:;:,:;:,:;:,:,:,:;:,:;:,:;:;:;:;:;:,:;:,:;:;:;:,:;:,:;:,:
:,:;:,:,:,:;:,:,:;:,:,:;:;:,:,:;;i;;;i;;;YVylVlll1QFijjYlyiviiv2S@8E8838B@@@@@B@@@U  . ..,:;,;;j308@OQQ3$B8@8ZSZLyyv;;;iv3MjVyFSB@v .;:,:,:,:,:;:;:,:,:,:;:;:,:;:,:;:;:,:;:,:,:,:;:,:;:;:,:,:,:;:,:,:;:;:;:;:;:;:,:,:;:,:,:,:;
;:,:,:,:,:,:,:;:,:;:;:;:,:;,;:;:;,;;iii;yUUiiivlUXQLvyUiYllililVKDQ3BQ$E@@@@@@@E@8; ,,;:;,;,;,;vCZ@B@QDKQB@8BB@8DFS3l;i;;ijJXVK2@B; ,:;,;:;:,:,:;:;:;:;:;:;:,:,:,:;:;:;:;:;:,:;:;:,:;:;:,:,:;:,:;:,:;:,:,:,:,:;:;:;:;:;:,:,:,:
,;:;:;:,:;:,:;:;:,:,:,:,:,:,:;,;,;,;,;;;iLylivvy;illiycLljllvlic0EOOQBZB@@B@B8$S@$ .,;,;,;:;,;,iyZ68B@@QDBEE06Q8OXU$jiii;;;LcXU6@i :.,,;:,:;:;:;:;:;:,:;:,:,:,:,:,:,:,:,:,:;:,:;:;:;:;:,:,:;:,:,:;:,:;:,:;:;:,:;:;:;:;:,:;:;:,
;:;:,:,:;:,:,:;:,:,:,:;:;:,:,:;:;;;;i;i;vyUlY1BJ, ,;;;ccXUjljlc0@QQSEZ6$StCK3ccM@F .;,;,;,;,;,;;vY8B@@B@@@B88QB863D32K1lv;ivjVJI0...,,;:;:,:;:,:;:;:,:;:;:,:;:,:;:;:,:;:;:;:,:;:;:,:;:,:;:;:;:,:,:,:,:;:;:;:;:;:,:;:,:;:;:;:,:
,,:,:;:;:,:,:,:;:;:;:;:;:;:;:,.,:;:;;iillcKEB@@@i  ,;ljXLFJtXt2B$O6B6OFJVVvylyl3@1 :,;,;,;,;,;,iiU@@@88@BQB@B@@@B8QQQE0IUyilyjUEEi :.;,,:,.,.;,,:;:;:,:,:;:;:;:,:;:;:;:,:;:,:;:;:;:;:,:;:;:;:;:,:,:,:,:;:;:;:,:,:,:,:;:;:;:;:,
;:,:;:,:;:;:;:,:,:,:,:;:,:,:;:,:,:;:;;i;ivU$@B@B@;  ilUjlyUcIF$0tS@@BtO1FXKXJUU2@X .;;;,;,;,,.;;iK@@@SV;,;vYFC8B@B@B@8B8S1tvlUXS@; .;,,.,.,;i,;:;:,:;:,:;:;:;:;:;:;:;:,:;:;:;:,:;:,:;:,:;:;:,:,:,:,:,:,:;:;:;:,:,:;:,:,:;:,:,:
:;:;:;:,:;:;:;:,:;:,:,:,:,:;:,:;,;,;:,;;,;;lyFZB@@ti;;iUilyjVCFLtVivLMviiiii;,;M@C .;;,;,;,;:,;;ii;i;;  .:.....;;VE@B@8@BBCXlLF@K. Yi;.,;vJc;,:;,;,;:;:,:,:,:;:;:;:;:,:;:;:;:;:;:;:;:,:,:;:,:,:;:;:;:;:,:;:;:;:;:,:;:;:;:,:;:,
;:;:,:,:,:;:,:;:,:;:;:;:,:,:,:,,,:;:;,;,;;;;lvc$BB@@@0QDSIEO2EEili,  :;.. . . ;G@Oi ;;;,;,;,;,;;v,.   ,.;.,.:..    ,BB@E6GGUVc8B; 30i:;;yv;.::;,;:,:;:;:,:;:;:,:,:,:;:;:;:,:,:;:,:;:;:,:;:;:,:;:,:,:,:,:,:;:,:;:;:,:;:;:;:,:,:
,;:;:,:,:;:;:;:;:,:;:;:,:;:;:,:,:;:;:,:;,;ilivlFXDB@B@@@B@@@B@BEZ@Qy  ;i.,.,..iZXZ3: ;;;,;,;,;,;i;,;:;:,:,:;.,.,..  ;@@Q1F1XL1O3;VGL,,iU;  ;,;,;,,:,:;:;:,:,:;:,:,:,:,:,:,:;:,:,:,:;:,:,:,:;:,:,:;:;:,:,:,:;:;:;:;:;:,:,:,:,:,
;:,:,:,:;:;:,:,:;:,:;:,:;:;:;:;:;:;:,:;:;,;;vvyL3$@B@B@B@@@@@B@B@SBM; ,;;.;,: i0tv6J..;,;;;,;,;;v,;;;,;,;:;:;:,:,.. y@@0OFXlYVB8My; ,yt;;vv;;:,,;,;:;:,:;:,:;:;:;:,:;:;:,:,:;:;:,:;:;:;:;:,:,:;:,:,:,:;:;:;:,:,:;:;:;:,:;:;:;:
:,:,:,:;:,:;:;:,:,:,:,:;:;:,:,:;:;:;:;:,:,.,ililF@B@@@@@B@@@B@@@B@@@i..l;:,;..i6XyXBji,;,;;;;;;30;:i;;,;:;,;:;:,.. iB@I20SCMtE$l;: ;L6M$1i:: ..,.,.,:,:;:;:,:,:,:;:;:;:,:;:;:,:,:;:,:,:,:;:,:,:;:;:;:,:,:;:,:;:;:,:,:;:;:,:;:;
;:;:,:,:,:;:,:,:,:;:;:,:,:;:;:,:,:,:;:;:,:,.;;vvJZ@B@B@@@@@@@@@@@B@Bi.VX;.,.: vG0VivIG3UUUJU6ZB@@;.;i;;,;,;:;,;...l8@tL2$XSBCi;.: ;y30@QLi;,vyVi;.,.,:,:;:;:,:;:;:;:;:;:;:;:;:,:;:,:,:,:,:;:,:,:,:;:;:,:;:;:;:,:,:;:;:;:;:;:,:
:,:,:,:;:,:,:,:;:,:;:;:;:,:;:,:;:,:;:;:;:;:;,;;yyJS@@@@@@@@@@@@@@@B@yi8B;:,;:.,UFvii;l1QSEGtJCO@@Qli,;,;:,:;:, ;iEBDjjYUJ$D1i;:;.;U1I@ZSKGMZ6l;Vv;.. :.,:,:,:;:;:,:,:,:;:,:,:;:;:;:,:,:;:,:,:,:;:,:;:,:,:;:,:;:,:,:;:;:,:;:;:;
;:;:;:,:,:,:;:,:,:,:;:;:;:,:;:,:;:,:;:;,;:;,;;YyyvUO@@@B@@@@@@@B@@@BMX@U:vSv:.:iUyYyjlyi;;;,;.,;M@@B6Vv;;;;;ii1E@EJllyXtGS0i;;;:iXMM@@BSQSS66Vli;.;il;;:;,,:;:;:;:;:;:;:;:,:,:,:,:,:;:;:;:;:;:;:;:,:;:;:,:;:,:;:;:,:,:,:,:,:;:
,;:,:,:,:,:;:;:;:,:,:,:;:;:,:,:;:;:;:;:;:,:,:;;villMQ@@@@@@@@@@@@@B@DEBci@Ei;v;,t@$Kli,;:;,;,;..,vVQB@@@88D@B@E$VviUjX1ZDSyi;;:iMcv8B@@@QEOGE@8B8B8@@t;;,;:;:;:,:;:,:;:,:;:;:,:;:;:;:,:;:,:,:,:;:;:;:,:,:;:;:;:;:;:;:,:;:;:,:;
;:;:,:;:;:;:;:;:,:,:,:,:;:;:;:,:,:;:;:,:;:;:;:,:;;ii10BB@@@@@B@@@@@B@BB8@BI;lOJ;UFl;;:;,;;;;;:;.::i;iiUUttXV1cVvcFM1MXO0ZXv;;,;t0;2@@EBBB88$EE@QY;jXUi;:;:;:;:,:;:;:,:;:,:,:;:;:;:;:,:;:;:,:,:,:,:,:;:,:,:;:;:;:;:,:,:;:,:,:;:
,;:,:;:,:;:,:;:,:,:;:;:,:;:,:;:,:;:;:,:,:,:,:;,;:,,iiicQZB@@@@@@@@@@@@B@B@ZOI@@M,,,;;;;;;;;;:;ii;;iMviilililV1J1FG1O1K$EMY;;,;jBviB@B8SQE8B@Q@Ei     ..,:,:;:;:,:,:,:;:;:;:;:,:,:;:;:;:,:,:,:,:,:,:;:;:,:;:;:;:;:;:;:,:;:;:,:,
;:;:;:;:;:,:,:,:;:,:;:;:,:,:,:;:;:,:,:;:;:;:;,;;;;iivlF2G6BB@@@@@@@B@@@@@B8Z@@8;,;;;;;;;;;;,;USvvVllIFUl2D$cUYLj1t3233D2Li;,;iEG;U@@@EEEBS880ESiXXl  .,.,.,:;.,:,:,:;:;:;:,:,:,:;:;:;:,:;:,:,:,:,:,:,:,:,:;:;:;:;:,:,:,:,:,:;:
,;:,:;:,:;:;:,:,:;:,:,:;:;:;:,:;:,:,:;:;:;:;:,.,:;;iiyLKME8@B@B@@@@@@@8@B@@@@@;.;i;;,;;;;;;;,0BB6@J;i$CGKQBQFcVXFOOO3SMUii;;;tBj.ZB@B@@@@@E86B@QllFL,;;,.,:,:,:;:;:;:,:,:,:;:,:,:,:;:;:,:,:;:;:,:;:;:;:,:,:,:;:;:,:,:;:,:;:,:;
;:;:,:,:,:;:;:,:;:;:,:,:,:;:;:;:;:;:;:;:,:,:;:,:,.,:;iVVtI88@@@B@@@B@B8B@B@@@K,,i;;,;,;;;;;,,iB@@@X.;SBU30EB@SGMO2$GSMJvi;i;vCQi;B@B@@@B@@@B@Q@@@GMSt,;;;,;,;,;,;:,:,:,:,:;:;:;:,:,:;:,:,:;:;:;:,:,:;:;:;:;:,:;:,:,:,:,:;:,:,:
:;:;:,:,:;:,:,:;:;:;:;:;:,:,:;:;:;:,:;:;:;:;:;:,:,,;;ivlivvcC8B@@@B@@@B@@@B@@v:i;;;;;;;i;;;;;;iJ33ii;c8Xy02QQ8OIK2ccjLli;ivviji.i@B@B@QBB@B8EB88E8E@@tlXli,;,;:;:,:;:,:;:,:;:;:,:;:;:,:;:,:,:,:;:,:,:,:,:,:;:;:,:,:;:;:,:,:,:,
;:,:,:;:,:;:;:;:,:;:,:;:;:;:;:,:;:,:;:;:;:;:;:;:,:,.;,;;i;iiJC8B@@@B@B@B@8@@M,;ii;;;;;i;i;vljyyvyvvvlUBJivVlLJFU1F1iivM2GK3Yi,. IB@B@8SS@@B$BB@88DEE@@6l;.. .:;,;:;:;:,:;:;:,:;:,:,:,:,:;:,:,:;:,:;:;:,:;:,:;:;:;:;:;:;:;:;:,:
,,:;:,:,:;:,:;:;:;:;:,:;:;:,:,:;:,:,:,:,:;:,:,:;,;,;,;,;ily1t3I88@@@@@8B8@B@;;ii;;;;;i;iiyVCXUlyvUvLj$BC;i;iiyljUKJVL8B@BB8Ei. :8@@8EB3GQ@@E8@@@BBDE8@B@B80V,,,,:;:,:,:,:;:,:,:,:;:,:,:,:,:;:;:,:,:,:,:,:,:,:,:;:;:;:,:,:;:,:,
;:;:;:,:;:,:,:;:;:;:,:;:,:,:,:;:,:;:;:;:,:,:,:;:,:;,;:;iylVlYlcKDDB8B88QBB@Y;;i;;;;;iiilccI$ZGSO6DSI0IE2l;i;iivvYyLJB@@B@B@8C;.;@@@D8B8GQ8@B868B@B@B@688@B@@$iiii;;,;,;,;:,:,:,:;:;:,:,:;:,:;:;:;:;:;:;:;:;:,:;:;:;:,:;:,:,:;:
:,:;:;:,:;:,:;:;:,:;:,:;:;:,:,:,:,:,:;:;:;:,:;:,:,:,:,:;iiivvyy1FG088B8@8@Fi;iii;;;i;lVO0QDBB@@@B@E8D$M0li;i;iiviyY3B@@@@@BBQGitB@@DQ@@@QB@@88QBB@@@8@88IZEGviiXy;,;,;,;,;:;:;:,:;:,:,:;:;:;:;:,:,:,:;:,:;:,:,:;:,:,:;:;:;:,:;
;:;:;:,:;:,:;:,:,:,:;:;:,:,:,:;:,:,:,:;:;:;:;:,:,:;:,:,:iilvliyllvylyJ0DBSy;iii;;;;;lXQQQGD3I0QGS0$S3lcOV;i;iiiilvVS@@@B@@@E8QEB@@@8B@@@D0@BB8B$E@@@@B@8QO6FX;;il;;,;,,:,:;:;:;:;:,:,:;:;:;:;:;:;:,:,:;:,:;:;:,:;:,:,:;:,:,:;:
:,:,:;:,:;:;:;:;:,:;:,:,:;:,:,:;:,:,:,:,:;:,:,:;:,:;:;:;,;,,.:.,.;;;:;vJcl;iii;i;;iyM8Q868E6JVii;viv;;lSViiiivivivY8QDBBQB@BB@@@@@B@B@B@ct8@S8BQX8B@@@@@Q88@BEY;   ..,:,:,:,:,:;:,:,:;:,:;:;:,:,:;:;:;:,:;:;:;:;:,:,:,:,:;:,:,
;:;:;:;:,:;:,:;:;:;:;:,:;:;:;:,:,:;:;:,:;:;:;:;:;:;:,:,:,.,.,:;:;,,:;;iii;vly;;;;;VI888E8QBEt;,:;,;;;,iVcvi;iiiiiicQDtE88DBE@@@@@@@@@B@@FUEB8Z@@DG@B@B@B@@@@@B@B@6i.. :.,.,:,:,:;:;:,:,:,:,:,:,:,:,:;:,:,:;:;:;:,:;:,:;:,:;:;:
,,:;:;:;:;:;:;:,:,:;:,:;:,:,:,:;:;:,:,:;:,:;:,:;:;:,:,:;:;:;,;,;,;:,,;:;;ilJviii;X$BDEZES3MDL;.;;;;;;;;vjVii;iiiiiVB8$3SS@Q6@@B@B@@@@@B@Uy3EESC@@EQ@@@@@888@B@B8B@@BLVvi,,,,:,:,:;:;:,:;:;:,:;:;:;:,:,:,:;:;:;:;:;:,:,:,:;:;:;
;:,:,:;:,:,:,:,:;:,:,:,:;:;:;:;:;:;:;:;:;:;:;:,:,:;:;:,:;:;,;:;:,:;:;:..cjvVcvll1686$G62XvlcU;,,;;i;i;;;lli;i;i;iit8@8EFSBB2B@@B@@@B@B@BU;cOSE$Q@@@8@B@B@@@EBB@BB$@8Kvi;;,;,,:,:,:,:,:,:,:;:,:;:;:,:;:,:,:,:,:,:;:;:;:;:,:;:,:
,,:;:;:,:,:,:;:;:,:;:;:,:,:,:;:;:,:;:;:,:,:,:;:,:;:;:;:,:,:,:;:,:;:,:,:yUi,y0OtQ8E6$1MUyvviVli:;;;;;;i;;ivii;i;iivLEQ6Q@B@BBB@B@B@B@B@B@Ot26S@BDD@@@@@@@8@B@@@B@B8Q@SCFi ..,.,.;:;:,:;:,:,:;:,:,:;:,:,:;:;:;:;:,:;:;:;:,:;:,:,
;:;:,:;:;:,:;:;:,:;:,:;:;:,:;:,:;:,:;:,:;:;:,:,:,:;:;:,:;:;:;:;:,:,.:.LUi,ilMIS6EOIULllvliilV;;,;;;;;;;;i;vii;iiiiLS8OE@@@@@@@@@@@@@@B@@@B@EBE@BBB@@@B@@BQBB@B@B@@@@8$@3i,;:,:;,;:;:;.;:,:,:;:;:,:;:,:,:;:,:;:,:;:;:,:;:;:;:;:
,,:;:,:;:;:,:;:,:;:,:,:;:,:;:;:;:;:,:,:;:,:,:,:,:;:,:;:,:,:,:;:;:,.:.lU;,vyXUI66KCLVvlvliiiYi;,;;;;;;;;;;iivivii;il$8BE8B@B@@@@@B@B@@@@@B@@BB@@@8BB@@@@@BQBBB@@@@@B@B@@@@Gii;;,;,;,,:,:;:;:;:;:;:;:;:;:,:;:;:,:;:,:,:,:;:;:,:;
;:;:,:,:;:;:,:;:;:;:;:;:;:;:;:,:,:;:,:,:;:;:;:;:;:,:,:,:;:,:;:,:;:,.ll;,llVtD6SK1YjlYvlii;iii,;,;;;;;;;;i;;;;;iiiil$@B@@@B@B@@@B@B@B@@@B@@@@@@EE8S@@@B@B@QB@BD8B@B@B@BBB@63y;,;:;;;,;:,:;:,:,:,:,:;:;:,:,:,:,:;:,:;:,:,:;:,:,:
:,:;:,:,:;:,:;:;:;:;:;:;:,:;:;:;:,:;:,:,:,:;:;:;:;:,:,:;:;:,:,:;:,:il;,iljc00SKFVVllilii;iiyi;,;;;;;;;;iii;i;i;;;;i$@@B@@@@@@@@@@@@@B@@@@@B@@@lX8B@@B@B@@@O88EQ@@@B@B@@@@@QXii;,;;,;,;;;,;,;:;:;:,:,:;:,:,:;:;:,:;:,:,:;:;:,:,
;:;:;:,:;:;:;:,:;:,:,:,:,:;:,:;:,:,:,:,:;:;:;:,:,:;:;:,:,:,:;:;:,:ii;;viycS$ItXyYlliiiiiiilvi,;;;;;;;;;;i;i;iii;i;iV8E8@@B@@@B@@@@@B@B@@@@@B@@@Gi;E8@@@B@@Q68E@@@@@@@B@@@BB8@QUi, :.;:;:;,;:;:,:,:,:;:,:;:,:,:,:;:;:,:,:;:;:;:
,;:,:,:;:;:;:;:,:;:,:;:,:,:,:,:,:,:,:,:;:;:,:,:,:;:,:,:;:;:;:,.,.;i;;vilVG3OtXVVvliviiiiiivv;;,;;;;;;;;;;;;;;iiiii;yD@D@B@@@B@@@@@@@B@@@@@@@@@@@V, :X@B@E@BB8@B@B@B@B@@@B@8@@It$YV;,.;:,:;:;:;:,:;:;:,:;:,:,:;:;:,:;:;:;:;:;:;
;:,:;:,:,:;:;:,:;:;:,:,:,:;:;:,:;:,:,:;:;:;:,:,:,:;:;:,:;:;:,:,.;;;;vvyVFF3XUyYvlilii;iivili;,;;;;;;;;ivv;;;iii;i;iiG@@8@@@@@B@B@@@@@B@B@@@@@@@@@v   t@@@@@@@@@@@@BDBB@@@8B88SSi;ilii;;:;:;:;:;:,:;:,:;:;:,:,:;:,:;:,:;:,:;:,:
:;:;:;:,:;:;:;:;:;:;:,:;:,:;:;:;:;:;:,:;:;:,:;:;:,:;:,:;:,:;,;.,;;;iiiV31CFXyYvlvviiiviiili;,;,;;;;;;;;lvv;;;i;i;i;iyB@8B@@@@@@@@@@@@@@@@@B@@@EIS$;. .8@B@D@@@@@@@8D6BB@BBBBQ@O;, .i;;;;,;:,:;:;:;:;:;:;:,:,:;:,:;:;:;:;:;:;:;
;:;:;:,:;:,:,:,:;:,:;:,:;:;:;:,:;:,:;:;:,:,:,:;:,:,:,:;:;:;,,:,,;;i;vvVcXYcYyiliv;iiiiiiii;;;,;;;;;;;;;;iii;i;;;iii;iG@8B@@@@@@B@@@B@@@@@@@@@BBU;,i,: ;B@2vyBB@B@@@EB8@@@B@8B@6ii,..,:;,;:,:;:,:;:;:;:;:,:;:,:;:,:,:;:,:;:,:;:
:;:;:;:,:;:,:,:;:,:,:;:,:,:,:;:,:,:;:;:;:,:,:;:,:;:,:,:;:;:,.,.;;iiviVvvilvliviviiiiiiiiii;;,;;;;;;;;;;;;i;iii;i;;;i;J@@8@@@@@B@B@B@@@B@B@@@@@@@y:.,.: ,i: ;Z@@@@@@@B@B@@@@@SB@83V:,.,:;:;:,:;:;:,:,:,:;:;:,:;:,:;:;:;:,:;:,:;
;:,:;:,:,:,:;:;:,:;:;:,:;:,:;:;:,:;:;:,:,:;:,:,:,:,:,:;:;,,:,;;iiilivvliiiiivivii;iiiilii;;,;;;;;;;;;;;;;;;;i;i;i;i;;iB@B@@@@@@@@@@@@@@@@B@B@@@@8, .,.: ..;j@@@B@@@BB8@B@@@@QSDK@E;.,:;:;:;:;:,:,:;:;:;:,:,:;:;:,:,:;:,:,:,:,:
,,:,:;:,:,:;:;:;:,:,:,:;:,:;:,:;:,:,:;:,:;:;:,:;:;:;:;:,.;:,;viivyii;iii;;;iiiiiiiilvlii;;,;,;;;;;;;;;;;;i;;;i;i;i;i,iM@B@@@B@@@B@B@@@B@@@@@B@@@Ei.  :..  .2@@@BB@@BE8B@BBB@B8$2Vtyi:,:;:;:,:;:,:;:,:;:,:;:,:,:;:,:,:;:;:,:;:,
;:;:;:;:,:,:,:;:;:,:;:;:,:;:,:;:;:,:;:,:,:;:,:,:;:;:;:,:,.,,;;;;i;;:,,;;;;iii;iiviYyy;;,;;;;;;;;;;;;i;i;;;;;i;i;i;iii;VE@B@@@B@@@@@B@B@@@B@@@@@@@B@J, . ,;I@@8@BE8@88QBBBZ8@@06B; iiV;,.;:;:,:,:;:,:;:;:;:;:,:,:;:;:,:;:,:,:,:
,;:;:,:,:,:;:;:;:,:;:;:,:;:;:;:,:;:;:,:;:,:,:,:,:;:;:,.,.;;;;;;i;;;;,;,,;;,;;;;i;;;;:,.;;i,;ii;;;i;i;;;;;;;;;;;i;i;viii8BB8@@@@@@@@@B@@@@@@@@@@@@@B@8;.:;BB@Q8BQCB@BEB8BBBEB@@BEi. ;i;:;:,:;:;:,:,:;:,:,:;:;:;:;:,:,:,:;:;:,:;
;:;:,:;:;:,:,:,:,:,:;:,:;:;:,:;:;:,:,:,:;:,:;:;:;:,.,.:.;V2J00S2S0$CMJUlylliiii;i;;;;,;,;,;,;,;,;,;;;.,:;;iivi;;iii;i,;6@BBB@B@@@@@@@@@@@B@@@@@@@BZO@BB1I@E0B@@I0B@EB8BB@BB8@88v;.:.,:,:,:,:;:,:;:;:;:,:,:;:;:,:;:,:,:,:;:,:,:
,,:;:,:,:;:,:,:,:,:,:;:;:,:;:;:;:,:,:;:,:;:;:,:,:;:,.;;;icYVjVlLYUYULXLXXKttLXctUXLUVjyYviii;i;;;i;,.,.,:;;;ili;;i;ii;,J8B8B@@B@@@B@@@@@B@@@B@B@B@BMVIKBB@$6SE6DQ@888BE8B@B@BIv@M;.,:,:;:;:,:;:,:,:,:;:;:,:,:;:,:;:;:,:;:,:;:;
;:,:;:,:;:;:;:;:;:,:,:;:;:,:;:,:,:,:,:;:;:;:;:;:;.,,iilili;;;;;.,:,.,,;,;,;,;;;;;;;;iivilivililvyVi.,.:.;;;.,,;;;;iii;;;O0EE@@@B@B@@@@@@@@@@@@@@@@@@Xl0ESM0IS6EQ@@BQB88DBBBE@I;YU;;:;:,:,:,:;:,:,:,:;:;:,:,:;:,:,:,:,:,:,:;:;:
,,:,:,:;:,:,:,:,:;:,:;:;:,:;:;:,:,:,:;:;:;:;:;:,:,,iii;i;i;i;;;;;;,;,;,;:,:,:,.,:,.,:,:,:;,;;;;;;i;;;;:,vCUv;i;;,;;i;i;;UBBB@@QB@@B@@@B@@@@@@@B@B@B@8Q8MKGOZQ@BE8@@BB@8BB@Qcv1K1.:,;:,:;:,:;:;:;:;:,:,:,:;:,:,:;:,:;:;:,:;:;:,
;:;:,:;:,:;:;:;:;:;:;:;:,:,:;:;:,:;:,:,:,:;:,:;:,:;;;;iiivvii;;;;;;;;;;,;;;;;;;;;,;;;,;;;,;,;,;,;,;;i;i;lVF2CjYii;i;;;;;iZ@8@@BE@B@@@@@B@@@@@@@B@8B@@B@$O0S8@@B$@B@Q@BBQ@BBtUiiQi ;:;:;:;:,:;:,:,:,:,:,:,:;:,:,:;:;:,:,:,:;:,:
:,:;:;:,:;:;:;:,:;:;:;:;:;:,:;:,:;:,:,:,:;:;:;:,:;,;,;;;iYlyvviiiiiiii;;;;;;;;;;;;;;,;,;;;;;;;;;;;;;;;;;,;,iyt2$cy;i;;;;,GBZMBB88BB@8@@@B@@@@@@@@@@@@@QQE@8@B@EB@@BZB@DB@@B@Iy.vJ;.;,,:,:;:;:,:;:,:;:,:,:;:;:,:;:;:,:;:;:,:;:,
;:;:;:;:,:,:,:;:,:;:;:,:,:,:;:;:;:,:,:,:,:;:,:;:,:;,;;;;vvYlylYljlyii;iii;i;;;iii;;;;;;,;;;;i;;;i;i;;;;;;;;:;,il2OXvi;;:;yEXl2@DEZE6QDBB@@@@@B@@@@@@@Q$O@B@8@8BB@@@8@EOB@@@@;iV:;i;.;:;:;:,:;:,:,:,:,:;:,:,:,:;:;:;:;:,:,:;:,:
:;:,:;:;:,:;:,:;:,:,:;:;:,:;:,:;:,:,:;:;:,:;:;:,:,,;;iililvyljVUlVlylylYlylYlylVlviviviiii;i;i;i;i;i;iii;;;;;;.,,vtGUi;;,ilyiU6EQ$3888E@@@@@@@@@B@@@BB$8B@BB@@8B8@BQB@ZQB@@O ,;;.,:;:,:,:,:,:;:,:;:;:,:;:;:;:;:;:,:;:;:;:;:;:,
;:;:;:,:;:;:,:;:;:,:,:,:;:,:,:;:;:;:,:,:,:;:;:,:;,;,iVVyJc1cF13XFX1cFK$S3U1XMtKMCUUL1tXJXlYlliiii;ilyii;i;vii;;;;.;l2ci,;;vylvVc8E6Z8ZEB@B@B@@@B@B@@@B8QBB@8@@@8B@8E@0SB@3C1;.,,;.,:;:;:;:,:;:,:,:;:,:,:;:;:,:;:,:,:,:,:,:,:;:
,,:;:,:,:,:,:;:,:,:,:;:;:;:,:,:;:;:;:;:;:,:;,;:,:,;ivUYjVXcXc2ttKGFKKG080ItXUXM8DGX1F3C$CXF0XcVLvvlciiilivlYyv;i;i;;;vi;;;y$Ui,;2@BE06DBB@B@@@@@B@@@BB6EQB8B@@Q8B@QEEZQ@Elij;:,;,;,;:;:,:,:;:;:;:,:,:;:,:;:,:;:,:;:;:,:,:,:;:;
;:,:;:;:,:;:,:;:;:,:;:;:;:;:;:,:;:,:;:,:;:,:,:,:;,;;vvVU1cXjMCG1KVFCC13cG$I1IO60GcXt0FJUJVXFICMUjYFUl;vviiyvlii;li;;i;i;;;lXSv;ijQ@BEDEEBB@@@@@B@@@@DCQ88Q@B@BSS@@@Q@DEB@V,;i.;,;:;:,:,:;:;:,:;:,:;:,:,:;:;:,:,:;:,:,:;:,:,:,:
:;:,:;:,:,:;:;:,:;:;:;:,:,:;:,:;:;:;:;:;:;,,,;:;,;,;;vlXtMX1F0M21F2ZtOD2V13$30I21KFS$ClyL2UUjtVcUK1Xlviliiilvvivvi;iiiii;;iVUyLZtK36BQXQB@@@@@@@@@@@QQ6Q0QB@BB2EB@BQ@BilY; ;;;.;:;:,:,:;:;:;:,:;:;:;:,:,:;:,:;:,:,:,:;:;:;:,:;
;:,:;:;:;:;:,:,:;:,:,:;:;:,:,:;:,:;:;:;:;:,:;:,:,:;;;;iiUJ2XCKO1J2D1UMQFUcZ6SF3FtVtFFYyjIXcVccJjXVjvviyvviVlliiii;i;i;iii;iiUltQCi;iQE0Q@BBB@B@B@B@B@8SIQQ88@BZE@@@6QQIi:.::;,;:,:;:;:;:,:,:,:,:,:;:,:;:,:,:;:;:,:,:;:;:,:,:;:
,;:;:,:,:,:,:;:;:,:,:,:,:,:;:;:;:;:,:;:,:;:,:,:;:;;i;iivV1XXU2O0XZ0KJC1M2SCMXCtJyUcMcLVFFYvylcUK1jviivvyiYVliviv;vyl;i;iil;;UUvKcCli1BBBQ@BB@@@@@@B@@8GE@@E8BBSB@@B8DBc;.,.,:;,;:,:;:;:,:;:,:;:,:,:,:,:,:;:;:,:,:,:;:;:;:;:,:,
;:;:;:,:;:;:,:;:,:,:,:,:,:;:,:;:,:;:,:;:,:;:;:,:,:;ivii;VU3F1XSS0I1J602S0JXyJC$1Jc2FtVUVjyyljVUVUyyvyyjlyyy;iyUiiiyvv;iivv;;lv;;JSMvYOBB8E@8BB@@@@@@@EEQBB@8@8E8@8@Q8Qv :,,:,:,:;:,:,:,:,:;:;:;:;:,:;:;:;:;:;:;:,:;:,:,:,:;:,:
,,:,:;:,:;:;:,:,:,:;:;:;:;:;:,:;:,:;:,:,:;:,:;:;:;,;;iiyyUM01F3SJFXIO3cItVyX3SFtUXJFJJlVljjLJCyliyyVlLlliLliijyv;iiiiyilLi:;;;;,c8yvlME@8E6@88E@B@@@@@BBQQS8EBBBB@88QBU;.;:,:;:;,;:,:;:,:,:;:,:;:,:,:,:;:,:,:,:;:;:,:;:;:;:;:,
;:,:;:,:;:,:;:;:,:,:;:;:;:;:;:,:,:,:,:;:,:;:,:;,;:;;;ivvllttFtIXUU3FJyXFGcUXGcliyVtJXVLlYUOVULYlyvVlllVvllv;vvVllvl;ivyFU:;;;:l33yilOK8QBBB@@SIMSE@@@B@B800Z$$@B@@B8@y;:::;:;:,:;:,:;:;:;:,:,:;:;:,:,:;:;:,:;:,:;:,:;:;:;:,:;:
:;:,:;:,:;:,:;:,:,:,:,:,:;:,:;:,:;:,:,:;:,:;:;:,:;:;;;;i;YJXL3FLyXXMUJLtcFXFUYvUU1KXvllliX0XlYljlVVLvyyyiliiiVVULUvVlyLV.,;i,;V8i..;vVGBSEB@BB$3GB@@@@B8$6DQ1S8B8@QB@v ;.;:,:,:,:;:;:,:;:,:;:,:,:;:;:;:;:,:;:;:,:,:;:;:;:,:;:;
;:;:;:;:;:;:;:;:;:;:,:;:;:;:;:,:;:;:,:,:,:,:,:;:;:,,;,iivYtjctIUCccjMKtYcU1UUVX3GXKYviVvvy1yjccYYvjUYvYvlvyljYliyyVyXtG;.,i;;,ii,.lyyltZ8OE8@B@EEQBB@BBQQQBQ06@Q8BBB@U.:;.;:,:,:;:,:,:,:,:;:;:,:;:;:;:;:;:;:,:;:,:,:;:;:,:,:;:
,,:,:;:,:,:,:;:;:;:,:,:;:;:,:,:,:;:,:;:;:;:;:;:;:,.;;;;iiyVXUXXKjJjtt3MGFtYylJJXVUVLjXLjyXyvjScjlvVFlllUXtVyyUlliV1XJ8L,.;;;:;;;:ycVlXvcQ88B8@B@EEQ@8BQEQBB@ZQ8B8@$B@J ,:;:;:;:,:,:,:,:;:;:,:,:,:,:,:,:;:;:;:,:,:;:,:;:,:;:,:,
;:,:;:,:;:;:;:;:;:,:;:;:;:,:,:;:,:,:;:;:;:,:,:,:;:,,;;;;iijXXVJUyL$O3UJc3FjYMcUyjLUF0tJyVyyiYVllYiLjYvYYCGJilt3LXK1F8V;.,,;;;,,.vUY;;Vv;3O8B@ZQB@EBBBZ88B8@BQOE8@BE1y,..,:,:;:,:,:;:;:;:;:;:;:,:;:;:;:;:;:;:;:,:;:;:,:,:;:,:;:
,,:,:,:;:;:;:;:;:;:,:;:,:,:;:,:;:;:;:;:,:;:,:,:,:,:,;i;;;lVXUJVUUcVJU1VyytVFILvVcFlUylYMyvivilvcjllliylyVMMCUUlVF01EG;.,,;,;,;:;ii;i;;;vivUQ8D3BBB8@SSQBDB@BI$QB@@Bi :.,.,.,:,:,:,:,:,:,:,:;:,:,:,:;:,:;:,:;:,:;:;:;:;:,:;:,:,
;:,:,:,:,:,:,:;:,:,:;:;:;:;:;:,:;:,:,:;:,:;:,:;:;.;:;;;,iiYycJKVjYUjUVUV1JLVJYUJ3VvVUvMMlililVccjycyVc3XCUt0CytI3cBE;.,,;,;,;,;;ly;,;;,;v;;iOQQQ8E8SS$QZQE@QOO88@BBiii;.,:;:,:,:,:,:;:;:,:;:,:;:;:;:,:;:,:,:,:;:,:;:;:,:,:;:,:
:,:,:,:,:;:,:,:;:;:,:,:,:,:;:;:;:,:,:,:,:;:;:;:;:;:,:;;iiiiyVF1MVXccjcU13Jivi3OZFlvUUJCFvVVYyUyViYXXj31UVUU32IXCOBBK:::;,;;;,;;;;J;;;i;iS2,;V8E@BB8EZQZ6S88SMQEBB@8cii;;;;,;:;:,:,.,:,:;:;:,:;:,:,:,:,:;:;:,:,:;:;:;:,:;:;:,:;
;:;:;:;:,:,:,:,:;:,:;:,:,:;:,:;:,:;:;:,:;:;:,:;:;:;:;:;il;ilclyjcjXUcYLj3YlvY6BZMV0cXG0cJVKVYyUJVJOKXjUyJK$SQCXCBBQ,.:;;;,;,;;;,;i;.;,;1@Xi;XDB@@B@DQ$ZOQEEM368Q@BQV;.;;;;;;;,;,;,;:,.,:,:,:,:,:;:;:,:,:;:;:;:;:;:;:,:;:,:;:;:
,;:;:,:,:,:,:;:;:;:;:;:;:;:;:,:;:;:,:,:,:;:,:,:;:,:;:;,iliijKtj1VcJCcVy3O3CIF0M21IEQO8Q3UJIKvyVMXMGZXUJ0SSIIF8@B@@i.:;,;;i;;,;;;,;:;;;:iVy;vUSD@@@BBSG$B8BD6Q@@QQ8Vli;,;;;;;;;;;;;;;,;,;:;:,:,:,:;:,:,:,:,:;:;:;:;:,:,:,:,:;:;
;:,:;:,:;:,:,:;:;:,:;:,:;:,:;:;:;:,:;:;:,:,:;:;:;:;:;:,,iiiiUJ1FIXt1S$DD$K2tFXCK33$FMCQGFC8OLVXjF2S33XKCG1$68B@S0y;.;;;:;;;,;;;,;,;,vi;.ii;;lJOE@@@BEKQBBBB8B@@Xyyv;;;;:;;;;;;i;i;;;;;;;;,;:,.;:;:,:;:;:,:,:,:,:;:;:,:;:,:,:,:
,,:,:;:,:,:,:;:;:,:;:,:;:;:,:,:,:;:;:,:,:;:,:;:;:;:,:,:;;v;;ilyXXFUcXZ$330X1KZ0$MMXUYX3ZCGIFLGSZSBOKMDG0SEE@@Syy;,.;,;,;,;,;,;,;,;:;iXi,;i;ivX0BB@8BQZE@8@BBS8QV;;;;:;;i;;,;;i;i;i;;;i;;,;,;:,:;:,:,:;:,:;:;:;:;:,:,:,:,:;:;:,
;:;:;:,:,:,:;:,:;:;:,:,:;:,:,:,:,:;:,:;:;:,:,:,:;:;:,:,,;;;;viiiJK3JCGSJGSS0EE8S02G3OISGS6ZJK$DIQBBKZE8E@@BB@2;:;:;,;,;,;,;,;,;,;,;,;;;:i;;illO8@B$S8Q@BDQ8ZEIVii;;,;,;il;;;;;iii;iili;,;:;;i,,:;:;:;:;:;:;:,:,:,:,:,:,:;:;:,:
,;:,:;:;:;:;:,:;:;:;:,:;:,:;:;:,:,:,:,:;:;:;:,:,:;:;:,:,:;;i;i;vYXFZ$6G6ZQ$0CO3$200EQE6ZSBZGG8IIQBDDS88@@@@@@J:;;;;;,;,;,;,;,;;;;;;i;;:;;i;VviVIE@QBBB8QKD8@@$viii;;;;:;;;,,,iiviyVKcVi;.:,il;.,:,:,:;:,:,:,:,:,:;:;:,:;:;:,:;
;:,:,:,:,:;:;:,:;:;:,:,:,:;:,:;:,:;:,:;:,:,:;:,:,:,:;:,:;,;,;iiiYYJ3DMOSEZDKtUK3Z0OGQS668QESQSSD@BDS@B@B@@@Bt,,;i;i;;,;,;;;,;,;,;;;;;,;;v;;ltivVMS8@@B@BQS@B@SUvv;;;;;;,;.,,;iUctVt112Sli,;jX;,:,:;:;:,:,:;:,:;:,:,:,:,:,:;:,:
,,:;:,:;:;:;:,:;:;:;:;:,:,:,:;:;:,:,:;:,:;:,:,:;:,:;:,:,:,:;:;;lvVyF1UUccO$Q68D$IQ$SO3tOQBEESEE@B@B@@@@@@BOj;;;;;;;;;;,;,;,;;;,;;;,;,;;;;i;;tZUXv1ZQ8@B@$M$@B$ylvv;;;i;;.iCEQDGQI3SB88@@Q@BS;..,:,:;:,:;:;:,:,:,:;:,:;:,:,:,:;
;:,:,:,:,:,:,:;:,:,:;:,:;:;:;:;:;:;:;:,:;:;:;:,:;:,:;:;:;:,:,:iiivVU1lVl3SD$EEB0S68Q8DE8@BBZEB@B@B@@@B@BBFl;;ii;;;;;;,;;;,;;;;;;;;;,;;;,;vY;l3Z1JC30B8EEB8@BBMVlVii;;;;:.V@B@XlillJ0@B@@8BJ:..,:;:;:,:;:,:;:,:;:;:;:;:;:,:;:,:
:,:,:,:,:,:;:,:,:,:,:;:,:,:;:;:,:,:,:,:;:,:;:;:,:;:,:;:;:;:,:;iiiylXUll3Z88BZEEB88Q88@@@BBB@B@B@@@@@@@82cVivii;i;;;;,;,;,;,;,;,;;i;,i1i;;iVjlIFjyIFDB@8@B8I8EZXVvviiii;; ;Q@BV;;;;,;l@@@,  ..,:;:;:,:;:;:,:,:,:,:;:;:,:,:,:;:;
;:;:,:,:,:;:,:,:,:,:;:;:;:,:;:,:;:;:;:;:,:;:;:;:;:;:,:,:,:;:,,i;iilvllSEQS86QEB8B8B8BB@BB8@@@B@@@@@8DD0vVyyii;i;i;;;;;;,;,;;;,;,;;;,iUXli;vVODXvlGD3BB@@@CK6EI3Vlilii;;;,.V8v;i;i;;,,K@Bv.:.,:;:,:,:,:,:;:;:,:,:,:;:,:,:,:,:;:
,,:;:,:,:;:,:,:;:;:;:;:,:,:,:,:;:;:,:;:,:,:;:;:,:,:;:;:,:,:,:;,,,;vVvV2QO6OIGQ8@68@@B@@@@BB@B@B@B@@QlcXLyjlvii;i;;;;,;;;;;,;;;,;,;,;,;;lli;ivM3CtZ888@B@Q$OESI3$Ylivii;;.,l0llvli;,;;;i$Bc,..,:;:;:,:;:,:,:;:;:,:;:,:,:,:;:,:,
;:;:;:;:,:;:,:;:,:;:;:;:,:,:,:,:;:,:;:,:,:,:,:;:,:;:;:;:;:,:;:,.,,llJcVlcMtlFO8BB8@E@@@8B8@@@B@B@B@83VcYVvvvv;i;i;;;;,;;;;;,;;;;;,;,;:;;UtViiX8GO2BB@@@@@8E68SEQMyyil;;;;:VGXlyll;;;;,;iVi;ii.,:,:;:,:;:;:,:,:;:,:,:,:;:;:;:;:
:,:;:,:,:;:;:,:;:;:;:;:;:;:;:,:;:,:,:;:;:,:,:;:;:,:;:;:,:;:;:,:;,;;iiUvvlUvijSQB8BSZ8@@BQ@@@@@@BEGFD3yiYviivii;i;i;;;;;ii;,;,;,i;;,i;;i;;VX1U$8S2GQB@@@@B@E8@@8821yvii;i,,l6Vyiviv;;;;;iilVl,..,.;:,:,:,:;:;:,:;:,:;:;:;:,:,:,
;:;:,:,:,:;:;:,:;:,:,:,:,:,:,:;:,:,:;:;:;:;:,:,:,:,:;:;:;:;:;:,,;:,;i;iivlvit3$O8BB6BBBB@B@B@8QMJVlvVivllilii;iii;;;;;;,;;;;;;;ii,iliiFl;iVyVX8ZQO8E@@@@@@@@@BB8ZXlii;i;;:yEMLYvlii;i;ivVvjli;i;,:,:;:;:,:;:,:,:;:;:,:,:;:,:;:
,;:,:;:;:;:,:,:,:,:,:,:,:;:;:;:;:,:,:,:,:;:,:;:,:;:;:,:,:,:,:;:;,,.;i;;iiyvyJXL$688B@@68@@B@01VyvylyiVyylyiv;i;iii;;;;;;,;iivyivilyUllUti;iLiJQ8E8B@@@B@B@@@BB8@QMvvii;i;,;@@@K1yl;i;;;iiXFDQGci:,:,:;:,:,:;:;:;:;:,:,:;:;:,:,
;:,:;:;:,:;:,:,:;:;:;:,:;:,:,:;:,:;:,:,:,:;:,:,:,:,:,:;:,:,:,:,:,:;:;.;;iiviyj2G3F6Q@@BS8@BtlivillLlVUclyvi;i;iii;;;;;VU;;v;vVYiivJVv;llv;VOt3BE8B@B@@@B@B@@@QB@8KFlvivi;,,M@@@BEMVii;i;iUXlli;,,:,:,:;:,:;:;:,:,:;:,:,:;:;:;:
:;:;:,:;:,:;:;:;:,:;:,:,:,:,:;:,:,:,:,:;:;:;:;:;:,:,:,:;:;:;:,:,:;:,.,,;:;iyvVLXYUX3F0CUy2Li;villyvyVjvvivilll;iiiii;Ytc;iii;ycV;iYciivVvlX0OBQQ8BZ@@@@@@@@@B@@@B8KUlvii;;.;vi;ilVllii;;,y3ti;...;:;:;:,:,:,:,:,:,:;:;:;:,:;:,
;:;:;:,:,:;:;:,:;:;:,:,:;:;:,:;:;:;:,:;:,:;:,:;:,:;:;:;:,:;:;:;:;:;:,.,:;;liiivvvillYvyivvliljLlLllvlii;vilV1Yviiiyiivl;iYYvllXl;itUVvVYVVFIE8QIEQ68@@@@@@@@@B@@@8E3MVlii;,.;.,:;,;;i;;.,lD2Qv..,:,:,:,:,:;:,:;:,:,:;:,:;:;:;:
,,:;:;:,:;:,:;:;:,:,:;:,:,:,:,:;:,:,:,:,:,:;:;:,:,:,:;:;:,:;:;:,:;:,:,:;,;vlii;iiiiviviylVcCKOJUcIJVvlYcVLvvilvli;;iiv;iv1llvvyLiLGIJXjLU136KQQSOBBBB@@@@@B@BUX@B@@@SSJcUy;,;iii;iiiiiilv0B8Q$i,:,:;:,:,:;:,:,:,:;:;:;:,:,:;:,
;:;:;:,:;:,:,:,:,:;:,:,:;:;:;:;:;:,:,:,:,:;:,:,:;:;:,:,:,:,:,:;:;:;:;:;:,,viyvi;viiiivLcJyOQ8$XV$QDI32E$tUtlVlvlYvyvUFXJjvyvVlLMXVGtUcUY21302ID6BB@8@B@B@@@B@; 1@@@B@EQ0Oyi;;iyivvllXLO@@@@8GJv:,:,:;:;:,:,:;:,:;:,:;:,:,:,:;:
:,:;:,:,:;:;:;:;:;:;:,:;:,:,:,:;:;:;:;:,:,:;:;:;:,:,:,:,:,:;:,:;:;:;:,:,:;;;iLviiylVyVc$MMG8S2VcK$0Q66G3VKGS2XLFtXyvvtMSXLY2SFlJXJSEIDIOI3MSIZSB@@S@@@@@@@@@BQ.,vB@@@@@@QZIJiiiyVYjM3@B@@Bi;:..::;:;:,:,:,:;:,:;:;:;:,:;:,:;:,
;:,:;:;:,:;:;:,:;:,:;:,:;:,:;:,:;:;:;:,:,:,:;:,:,:;:,:;:,:,:;:;:;:,:;:,:;:,.iVVlllXtUlXGO$QSD06DE2$SESGC$3ZGKU3OMVVvjtS$6OG$$jU1QSB@B8B6It66QE86BQD@@B@@@@@@icv  .i1@@@8@E8ZFiil1FZ8@DBSJl; ..,:;:,:,:;:,:,:,:;:;:,:,:,:,:;:,:
:;:,:,:;:;:;:,:;:;:;:;:;:;:,:,:;:,:;:;:;:;:,:,:;:,:,:;:,:,:;:;:,:,:;:;:,:,.;vcLVvjtCylVKK8Z6Z@@@@ESEQBQ8QDGGJcc2XUYJ1$O$OQQZCOIZSDEBQ8Q$LjUK0@B8SD0@@@B@@@S; .;, . ;yOvU@@BB3S3QQ@@@BcvY,..,.,:,:,:,:;:;:,:,:,:;:,:;:,:,:,:,:;
;:;:;:,:,:;:;:;:,:;:;:,:;:,:,:,:,:,:,:;:;:;:;:;:;:;:;:,:;:;:,:;:,:;:,:,:,.,ijVJVVycLYilLSB80EB@@@B@8BB@886ES$3030MIOEQ8QE6D2G$ESSO88B8QMcYt1G8@@@B88@B@@@@@l: ;,,.:   ,.i@@B@@@8@Z@BBSi ..,.,:,:;:;:,:;:;:,:,:;:,:,:,:;:,:,:;:
:,:;:,:;:;:;:;:;:;:,:,:;:;:,:,:;:,:;:;:,:,:,:;:,:;:,:,:,:;:;:,:;:,:,:,:,:;illYlcULlVyviXB@BEQ@@@8@@BE@@@Q88BB@88E8QBB@8@EQZE6D$EQ888Q80Fj1$B$8B@B@B@B@B@B@Gi.,.;,;.,.... ,;Y@@@3286Xyi:..,:;:;:;:;:;:;:,:;:,:;:,:,:;:,:,:,:;:,
;:;:,:;:;:;:,:,:;:,:;:;:,:,:,:;:,:,:,:;:,:;:;:;:,:;:,:;:;:,:;:;:,:,:;:;:,;viivlVCcLUUiv1B@@QB@B8@@@8BB@88E88@@@8B8BB@E88E3EEB$338BDMGC2VLV2SD8@BBQ8BB@@@@@; ..,:;:;:;:,.:   ;O@i:;i:. :.,:;:;:,:,:,:;:;:,:,:,:,:,:,:,:;:;:;:,:
:;:,:,:,:;:,:,:,:,:;:,:,:,:,:;:;:;:,:;:;:;:,:,:,:;:;:,:;:,:;:,:,:;:,:,:,:;;iiylt1ccOUvySQ@B6E@0ZQSGBB@Q6DB8@@@@B8BE@B86B8E$Z88C$E@tVVFK3tCK008B@BB8@8@B@@@i..,.;:,:,:,:,.,.. ... . ..,:;:;:,:;:,:,:,:;:;:;:;:,:;:;:,:;:,:;:,:,
;:;:,:,:,:;:;:;:,:,:,:,:;:;:,:;:,:,:,:;:,:;:;:,:,:,:;:,:;:,:,:,:;:,:;:,:;:;;lljUUy3QOijQ@@BFIS3CSt0EBSGOB@@B@@@BBEB8@EBB@@QI8QDD@B0VUXSDESZ3$Q@@@B@@@BB@@@,.,.,:,:,:,:,:,:,:,.:.,.,:,.,:;:;:,:,:,:,:;:;:,:;:,:;:;:;:;:,:,:,:,:
,;:,:,:,:;:;:;:,:;:,:,:;:;:,:,:;:,:;:,:;:;:,:,:,:,:;:;:,:;:,:,:;:;:;:,:,;;;ivVyLVXZ@QLc@B@8XJZOD03O@BBQ@@@B@B@@@BB8@@@8B8@E8B@S8@@Q$KGSQZE$3CQ8@@@@@B8E@B@y.,;.;:;:;:,:,:;:;:,:,:,:,:;:,:;:,:,:,:,:,:,:;:,:,:;:,:;:;:;:,:;:;:,
;:,:,:,:,:,:,:,:,:,:;:;:;:,:,:;:,:,:;:,:,:;:,:,:,:;:;:;:,:;:;:,:,:;:;,;:;;i;lyUyCS@@@cCB@@BcKEBBBQ@@@@@B@@@@@B@@@ED8@@BZZ3ZE@B8D@@80ICSOEB8MG$BB@@@8@B@B8B@v,,;:;:;:,:,:;:;:;:,:;:,:,:;:;:,:;:;:;:,:;:,:,:;:;:;:;:;:;:;:,:,:;:
:;:;:;:,:,:,:,:,:,:,:,:;:;:,:;:;:,:;:,:;:,:;:;:,:,:,:;:,:,:,:,:,:,:,:;.,:;iivclj3BB@BQSBBBSFC8B@B@@@@@@@@@@@BBB@BBS8@@EDKC08B@888@88QEOD8@8SIS6B8@B@B@@@IEBV.,:;:;:;:,:;:;:,:;:,:,:,:,:;:;:;:;:,:,:;:,:,:;:,:;:,:,:,:;:,:,:,:,
;:;:;:;:,:,:;:,:;:,:;:,:;:,:,:;:;:;:;:,:,:;:;:;:,:;:;:;:,:;:,:,:;:,:,:;;vvviFXj2@@@@@B@8BQI2QB@@@SEB@@@@@@@B@EB8B@@E@BBD21BB@88QQ$8Q88ES8DSSZGZ$Q8@@@@@Z1i; ..,:,:,:,:,:;:;:;:;:;:,:,:;:;:,:;:;:;:;:;:,:;:,:;:;:;:,:,:;:,:,:;:
,,:,:;:;:,:;:,:,:,:,:;:,:;:;:,:,:;:,:;:;:;:,:,:;:;:;:;:;:;:;:,:,:,:;:;,;;ilVVLU8@@@@@@@@B@EBB@@@B@8@B@@@B@@@B88@@@Q6B@BB3C8@B@8BQEBBSBB8DQSDG$K2M8@@B@@v.  :.,:,:;:;:,:,:,:,:;:;:,:,:,:,:;:,:;:,:;:;:,:;:,:;:;:;:,:,:,:;:,:,:,
;:;:,:,:;:,:;:,:;:,:,:,:,:;:,:;:;:,:,:;:;:,:;:,:;:,:,:;:,:,:,:;:;:;:,:;:,;ivyl28@B@@@@@@@@@B@@@@@@@B@@@@@@@B@Q8B@@BSBB@@QG88@88E@@@B88@8@B8GQQBQ608@@@X.:.,:,:,:;:;:;:,:;:;:,:;:;:;:,:,:,:,:,:,:;:,:;:;:;:,:;:,:,:,:,:;:;:,:,:
,,:;:;:;:,:;:,:,:;:,:;:,:;:;:;:;:;:;:,:;:,:;:,:;:,:;:,:,:,:,:;:;:,:;,;:;;;;vV16@6ZQ@B@@@@@@@B@@@B@@@@@@@@@@@@88B8@@B@@B@8B88QBB@888@B@8BB@88D8B@8Q6$li,..,.,:;:;:;:;:,:;:;:;:;:;:;:;:,:;:;:;:;:,:;:,:;:;:,:;:;:,:,:;:;:,:;:,:,
;:;:;:;:;:,:;:,:,:;:;:,:;:;:;:;:,:,:,:;:,:,:;:;:,:,:;:;:,:,:;:;:;:,,;:;.;;iilYFFG$8B@@@@@@@@@@@@@B@@@B@@@@@@@B@8B@@E@B@B@BB88E@@@DBB@BBB@B@@@8BEB8El, ..,.,:;:;:;:,:,:;:,:;:,:,:;:;:;:,:;:,:,:;:,:,:,:,:,:,:,:;:,:,:;:;:;:,:,:
,,:,:,:,:;:,:,:,:;:,:;:,:;:;:;:,:,:,:,:,:,:,:,:;:,:,:,:,:,:,:;:,:;:;:,:,,;;ilUvJSED88@B@@@B@@@@@@@BB@@@@@@B@@@8Q8@@@B@@@8B8@B@B@@BB@8B8B8B888@8BEDMy...,:;:;:,:;:;:,:,:,:,:,:,:,:,:;:;:;:;:,:;:,:;:,:;:,:,:;:,:;:;:;:;:,:;:,:;
;:,:,:,:;:;:,:,:;:;:,:;:;:,:;:;:;:;:;:;:;:;:,:,:;:;:;:,:;:;:;:,:;:,:;,,:;ivillU0B68Q@@@@@@@B@@@B@BBO@B@@@@@@@@B8@B@@@B@@B68B@BBB@@@@BE8E88BQ8EB@@BC.  ,:;:;:;:,:,:;:;:,:,:,:;:,:,:,:,:;:,:;:;:;:,:,:;:,:,:,:;:,:,:,:,:;:;:;:,:
:;:;:;:;:,:;:;:;:;:,:,:;:,:,:;:,:,:;:;:;:,:,:,:,:;:;:;:,:;:,:,:;:,:;:;:,;iii;vX@BB8@@@B@@@@@@@B@@@B@QFM@BB8B@@B@B@@@B@@@BB@@BB8@@BBBQ8Q8QB8BE8SDSBDy ..,:;,,:,:;:,:,:,:,:;:;:,:,:;:;:;:,:,:;:;:;:;:;:;:;:;:;:,:;:;:;:;:,:;:,:;
;:;:;:;:,:,:;:;:,:;:,:,:,:;:;:;:,:;:,:,:;:,:,:;.,:,.,:,.,.,.,.,:,.,.,.,:;:iillSB@B@@@@@B@@@@@@@@@B@Bl .v; ,,;v$D0Lt8@@@@@@@@@B8D88@BBEB888@BBSSGSCS81;,.,:;:;:;:,:;:,:;:;:,:,:,:;:,:;:,:,:;:;:,:,:;:;:,:,:;:,:,:,:;:,:;:,:,:;:
,;:,:,:,:;:,:,:;:,:,:,:;:;:,:,:;,;.,.,:,:;,;:;:,:,:,.,.,:,::.:.:.:.,.,,;;iiyyF6@B@@@@@@@@@@@@@@@B@SV;  .:. :.:,;ii:v086@@@BB8@QE8@B@BB8@B@8802JFJXY2BC;..,:;:;:;:,,,:;:,:,:;:;:;:;:;:,:,:;:,:;:;:;:,:;:,:,:;:;:,:,:,:,:;:;:,:;
;:,:;:;:,:;:;:;:;,,:;,;,;:,,;:,:;:;:,:;,;,;,;,;:;;;,;:;;iiiii,;,. ..;;i;villJFZQ8Q@@@B@@@@@B@@@@@Bv   ii;:,.,.;,;;;;vJS8@BBDB@BDB@BZ8B8SBBB88IFVVj1yUl;.:.:.,.,:,.,,;:,:,:;:,:,:,:,:;:;:,:,:;:,:;:,:;:;:,:;:;:;:;:,:;:,:,:,:,:
,,:,:,:;:,:;:;:;,;:;:;:;:;:,:;,;:;:;:,:;,;:;;;;iiUlYYyivyyvylYVU;iiviiivvylVVKIEQ8@@B@B@@@B@@@B@B@0Xlyi, ,.;:,:;,;;;;cS6Q@B@@@8ZQBQZZ@8$K8B@@@D3UtXXivVJi;,;:,.,.,.,:,:,.;:,:;:;:,:,:,:;:,:,:;:;:;:;:,:,:;:;:,:;:;:,:;:;:;:,:;
;:;:,:;:;:,:,:;:,:,:,,;:;,;:;:;,;:;:,:,:;,iiiilL2cViyVjlUtKyVLVl1X33$JyiUUcJ68@8B@@@@B@B@@@B@B@B8E@@@2yv: ..,:;:;;;,;iFGE88E@@@ZBB@E8@@6$S8Q@B@8ZFMtJyVyVllii;;,;:,.,.,:,:,:,,;:,:,:;:,:,:;:;:;:,:;:;:;:;:;:,:;:,:;:,:,:;:,:;:
,,:,:,:;:,:;:;:;,;:;,;:;:;:;:;:,:;,;;iiviiivii;;;vlliyj1F$KCLGCSQQD@@B0IO6GSE@@@@@@@@B8@B@EB8@EEZCcM8@@@X;...,:;:;;;;vVBBB88B@BB@@8B@@BEG8Q88@@@EDG63FlyljVliliiivii,,:,.,:;:,:;:,:,:,:,:,:,:;:,:;:,:,:;:;:;:,:;:;:,:,:,:,:;:;
;:,:,:;:,:,:;,;:;,;,;,;:;:,.,:,,;;iiyyVivvi;;;iii;lYUyUVFXXXtCO3S08B@@BQ@@@8BB@B@@@@@@@BBS$S@B8Q@SG2D8@@@@X;,.,:,,;;i;lQ@B@B@@@@@B@Q@@@ZSZEO6863B8888OJvllLUUcCUYvliiiii;;;,,.,,;:,.:.,:,:,:;:,:,:;:;:,:;:;:;:,:;:;:,:,:;:;:;:
,,:,:,:,:;:;,;,;,;:;,;.:.::;;;ivivilvlii;;;;;ylLYlvyvviyyUyVlcLcX3GE8BEE8@B@@@@@B@B@B@B@BQ068BSMX8E0LUM8@@Ey;,:;:;:,..;EB@B@B@88ME8BB@@@B@6S3@8FM@QStcVLjXJ2FKUUlyyYvYYVvi;;;i;;,;...,.,.;:;:;:;:,:,:,:,:;:,:,:;:;:,:,:;:,:,:;
;:,:,:,:;,;,;,;,;,;,, :,vyVlllUlyvlilivvylVvvivivivvyvlvyivlLlVU3C$SEQE686EQB8@@@8BE8SQSEEBE86ZXV0SctX1U36SUl,;:;:;,:;I@@8@QZLcUlvtS$USQ8B@BS28UjQSFG30COKFLUYjyliyvVVVvVlylViiii;;iFJ; :.,:;:,:,:,:,:,:;:;:;:,:;:,:;:;:;:;:,:
,;:,:;.;,;,;,;,;,;,,.;V02UlUcUlyyVvvillcLVvYlliviiilllvllvvUccVcXMK3FCXFC$I$IZZ8EBZSFFXMM6B@E88@EBSZQ8I1JMMOl;.;,;,;.,ij;;;ii,;li;;v;ili;lS@U;;;iXv1E8F13$yllccXlVVcVjlyyUylvlvyljjMCEJv;, ....:.,.,:,:,:;:;:,:,:;:;:,:,:,:;:,
;:,:;:,:;,;,;,;,;,, ;I@VlilFKVjyVllljyYyUyyycvylyvyvylYycYVyULUlLcCM3XXUKF3CSGSODSZ3GF2MO3E8@B@@@@@B@@@B@SVly;;,;;;,;.,;;.;;;.,;;;i;;iXlYlMDUillviV3IJyilljKFyVUXLJjXYUVLyUjUVLyjyVyUcG2MLl,,.. . ..:.,:,:,:,:;:;:;:;:,:;:;:;:
:,:;:;,;,;,;,;,;,,.;I8Jv,ijKVcXCLi;ylyvylVvylyyVlYlYlLVUYVvllJVVlUXIM3F3M3C0GZO$3$GD6QO$33KSQ@BB8BB@B@B@B@F. ;;;;;;;;;;ivyii;;;;;;ii;iiVLVivijVVvj1MlVLyi;iZIFJ3tcylilyyiiilyjVJLcVYvllLlX3KJXYyii:. . ..,.,:,:,,;:;:,:,:;:,:;
;:;:,:,,;,;,;,;;;. c@BFi;y0UcXFKi;ycyvlvyivivvlljyVyVYjYjvlvjjVlVUFF3F3K212FGISG$M0SQ$Z$6KG0EEBB@8@B@B@B@B@V;ivivii;iicK2VY;;;i;v;;;i;;,;;viv;;;i;vyjKFlcXGS6GGUyvYllvVlYvlvlvYyUUtX1XKtMXKMIF31G3ILVii,. . ..,:;,;,;:,:;:;:;:
,,:;:;:;,;,;;;;;..;B@B8$MES1VUV1yC8BZSYlivilvYlVlVyVlYvVUUVcctVVYVlVYJJ11GC0GZ06G0I$IG2S$EQB8@B@@@@@@@B@@@B@;ilVljlyvVXCJUYvii;lvli;;;;;,ivcLliv;;il;VX3FQS0cUjXX$68SQ68SZZQF3C$0QSQZ60DSQGIcUlylXXJX0tcVVi;...,:;:,:;:;:;:;:;
;:,:,:;:;,;,;;;,,i@@@@@BBQQMXLLJ6DSQZt8LvillylVvlvYlVlyljYUVcVLllillXMS6B8B8@BB8BB@886ED88BB@B@B@@@@@B@@@@@ZilKlLXcyUX$MXyViVlvyVvjvvii;;;Y12llviiV;,;vlvilCIcMGGLjcCKIF1X0S6SEZEQ8E@B8QB@@B@B@8QS88ESQ$$0EEBX;.;:,:;:;:;:,:,:
,,:;:;:,,;,;,;,;,OB@@@B8ZEB@0OQBS8KXyU8QylvLyyvyvlvlvlvlilvlvvilyM6@B@@@B@@@@@@@B@@@@@B@B@B@@@B@B@B@@@B@B@@OU0KcUtcLL2I2yYVcVUlylviVVYiviy2ZVlvliiiiivvVi: llj$@Qi               . .,;,;;iiiilljYX2D0E@@@@@@QV.,:;,;:;:;:,:,:;
;:,:,:;:;:;,;:;,,;;;;v@@@@@@@@@Q3S2jQB@8ZtMG$CMXKXtLJVcyvivillJK@@@@@B@@@B@@@@@@@@@B@@@B@B@B@@@B@@@@@@@@@B@B@@B8B68S62CVcUylVV1LMjVyVvlyGQ8XylUli;vYliYVV;iyviyY$8Si. ..:.:.,.:.. . . . .         . . ..:.,.. ,:;:;,,,;:,:;:;:
:,:,:,:;,;:;,;,;..   ;6@BE8@B@@SF20@@@@@8ESQQ8$SS6OZ0SO$KGIQB@@@@@B@@@@@@@@@B8SZGG2IKMctjlvlvlilvllVVUjKCSDBB@@@@@@BZSKKccVYvylYlUvv;iivC@$Y;ilyivlliviiiUMFyl;;i1EQXy,,,;,;,;:;,;:,:,:,.;.,.,.,.:.:.. . . ..,.,:;:,,;:,:,:,:;
;:,:;:,:,.,:;,;:;:,.:..    ...V@@@@@@@@@@@@B@B@@@@@B@@@B@@@B@@@@@BQtVii;;:,.:.:.:.: . . . . ..,.,.;,;,;;;;iilY1K$$8EEI3t1cLlyvlivvliiiiiOBZilvyvlilii;iiiiMUVi;i;.;Vti,.;,;:;,;:;,;,;:;,;,;,;:,:,.,:;:,:,.,.,,;,;:;,;:,:;:;:;:
:,:;:,:;:,:,:;:,:,:,:,.. . . .i@B@BBQ88@@@@@B@B@B@@@8BQ8EQ1Lvi;, . . ..:.,.,.,:,.,:,:;:,,;,;,;;;;;;;;;;;;i;i;iii;ivM$$t1Jclliliv;iililvl2@Iylyvlii;;;;iliij0v;;i;,.ilv::.;,;,;:;,;,;,;,;,;,;,;:,,;:;:,,;:;:;:;:,:;:,:,:;:;:;:,
;:,:,:;:;:,:,:;:;:,:;:,.,:,:,..                                 ..,.,:;,;:;:;:;,;,;,;,;,;;;,;;;;;;;;;;i;;;i;i;i;i;;VSCtUMUVviii;;,;iyYYvOB6jVllvviviiiiiyvy1Liyvvii;JKv:::;,;,;:;,;:;,;:;:;,,:;:,,,:,:;:;,;,;:;:;,;,;,,:,:;:,:
,;:,:;:,:,:,:,:;:;,;,;:;:,,;:,.. . . . . . . . . . . . ....:.,:,:,:;,;:;:;:;,;,;,;,;,;,;,;,;,;,;,;;;;;;;;i;;;i;iiii2ZOXJUcvlljlyi;,;illl3@8IUUYVvliv;iilyVlS0UvYlVvVyFSj.::;,;,;:;,;,,,;,;:;,;,;:;:;:,:;,;,;:;:;:,:;:;:;:,:,:,
;:;:;:;:,:;:,:;:,:;:;,;:,:;:;:;:;:;:;:,:,.,.,:,:,.,:;,,,;:,:;,;,,,;,;,;:;,;,;,;,;,;:;:;,;,;,;,;,;,;;;;;;;;;;i;iiyltZE3IjLX$C$DEZ6Ji;iillZ@@QItXyVlvii;llylVQ8KFjYvYlllLcl.,,;,;:;,;,;:;:;:;:;:,:,:;:,:;:;:;:;:;:;:;:;:,:;:;:;:
,;:;:,:,:;:;:,:;:;:;:;,;:;:;:;,;,;:,:;:,,,:;,;:;:,:;:;:;:;,;,;:;:;:;:;:;,;,;,;,;,;:;,;:;:;,;,;,;,;,;,;;;;;;;;iilvUZ@8D33CE8BB@S6QQCFi;iVS@BQM3cjyVvi;vljlyl8BQ2tllvvlViYUi.,,;:;:;,;:;,,,;:;:,:,:;:,,;,;:;:;:;:;:,:;:,:;:;:,:,
;:;:,:;:,:;:;:;:;:;:,:;:;:;:;:,,;,,,;,;:;:;:;:;:,:;:,:,:,,;,,:,:,:;:;:;:,,;,;,;:;:;:;,;,;,;,;,;,;,;,;,;;;;;;i;iilYB@@E8S8B@@@B$F6SEQMlVjE@B2GIMYVlliiilycYUB@StYyii;iiiiLl,.;,;:;:;:;:;,;:;:;:,:;:;:;:;,,:;:;:;:,:;:,:,:,:;:,:
,;:,:;:;:,:,:,:;:;:,:;:;:;:;:,:,:,:,,,:,,;:;:,:;,;:;:;:,:;:;,;:;,;,;,;:;:;:;:,,;,,:;:;,,:;,;,;,;,;,;,;,;,;;;;;;iilC@@@B@8@@@@8cX$88@8608B@B0IQ3MM$CIFC1UlyU@@8jliVVlvyiilV;,:;:;:;:;:;:;:;:;,;:,:,:,:;,;:;,;:,:,:,:;:,:;:;:;:;
;:,:;:;:;:,:,:,:;:;:,:,:;:;:,:,:;:,:,:;:,:;:;:;:;:;:;:,:,:,:;.,:,:;:;:;:;,;:;:,:;,,:;:;:;:,:;,;:;:;:;,;,;,;;;;;;iic8@B@@@@@@@61JD8@BBE@B@@@SQSSS@@@IG6B$XUI@@@SXK6BEEKl;jL;.,,;,;,;,;:;:;:;:;:;:,:;:;:;:;:;:,:,:,:;:;:;:,:;:;:
,;:;:;:,:,:;:;:;:,:;:,:;:;:,:;:,:,:;:;:;:,:,:;:,:;:;:,:,:;:;:,:,:;:,:;:;:,:,:,:,:;:;,;:;:,:,:;:;,,:,:;:;:,:;,;,;;;;l3B@@B@@@B$1D@@@@B@@@B@@BBBQB@@BtySBBG@@@B@QQ8@BO2E2F3j.,,;,,:;:;:;,,:;:;:;:,:;,,:,:;:,:;:;:;:,:,:;:;:;:;:;
;:,:;:;:,:,:;:;:;:;:;:;:,:;:;:,:,:,:,:,:;:,:,:,:,:,:;:,:,:,:,:,:,:;:;:,:,:;:,:;:,:,:,:,:;:,:;:,,;:,:,:,:,:;:;:,:,.: .:;iG@@@@B@@@@@B@B@B@@@@@B@@@@@FC8@@@@@@@@@8@@@1$QB@I:::;,;,;,;:,,;:;:,:,:,:;,;:;:;:;:,:;:,:,:,:,:,:,:;:;:
,;:,:,:,:,:,:,:,:;:;:,:;:;:;:;:,:,:;:,:,:;:,:;:;:;:,:;:;:;:;:;:;:,:,:,:,:;:,:;:;:,:;:,:,:,:,:,:;:,:,:,:;:;:;:,:;:,.:..   .;K@@0Y;;..     .;VMBB@B@@@@@B@MXXB@@@@@@@@B@@I:.:;:;,;,;:;:;,,:;:;,;:;:,:,:,:,:;:,:;:;:;:;:,:,:,:,:;
;:,:;:;:,:,:,:,:;:,:,:,:,:,:;:;:;:,:,:;:,:;:;:,:,:;:;:,:;:;:;:,:,:,:;:;:,:;:,:,:;:;:;:;:;:;:,:,:;:,:,:,:;:;:,,;,,:;:,.,..           . . .      :iV@BJ;;      .;iQ@@B0Jy...;:;:;:,:;,;:;:;:;:;,,:;:;:;:,:,:;:,:;:;:;:,:,:,:,:;:
,,:;:,:,:,:,:,:;:,:;:;:,:,:;:;:,:;:,:,:;:,:,:;:,:;:;:,:;:,:,:,:;:;:,:;:,:,:,:;:;:,:,:;:;:,:;:,:;:,:;:,:,:,:;:;:;:,:,,;,,:,.....:.,.,:,.,.,.:..         ... .       .   ..,:;:;:;:;:;:;:,:;:,:;:,:,:,:;:;:,:;:;:;:;:;:;:,:;:;:;
;:,:;:,:;:,:;:,:;:,:,:;:,:,:;:;:;:,:;:,:,:,:;:,:,:,:;:,:,:,:;:,:,:,:;:;:;:,:;:;:;:,:,:;:,:;:,:;:,:;:,:;:;:,:,:,:,:;:,:,:,:,:,.,:,:;:;:,:;:,:,.,.,.:.:.,:,:;:,.,.. . ..,:,:;:;:,:,,;:;:;:;:;,,:;:;:;:;:;:,:;:,:,:;:;:;:;:,:;:;:
,,:,:;:;:,:,:,:,:;:,:;:;:;:;:;:;:;:;:;:;:,:;:,:,:;:,:,:;:,:;:,:;:;:,:;:,:;:,:;:,:,:;:;:;:;:,:,:,:,:;:;:,:;:,:,:,:,:,:,:;:;:;:,:,:;:;:;:;:,:,:,:,:,.,:,:;:,:;:,:,:,:,:,:;:,:;:;:;:,:;:;:,:,:;:,:,:,:;:;:,:,:;:,:,:;:;:,:;:;:,:,
;:;:;:;:;:,:,:,:;:,:,:,:;:,:,:;:;:;:,:;:;:,:,:,:;:;:;:,:,:;:,:;:,:,:;:,:;:;:,:;:,:,:,:,:,:;:;:,:;:,:;:;:,:,:;:,:,:;:,:;:,:;:;:,:,:;:,:;:,:,:,:;:,:,:;:;:,:,:;:;:,:;:,:,:;:,:;:;:,:,:,:,:;:;:,:,:;:,:,:;:;:,:;:,:,:;:;:,:;:,:,:
:;:,:;:,:;:,:,:,:;:;:,:,:;:,:;:,:,:,:;:,:,:;:;:;:,:,:,:;:,:,:;:,:;:,:,:,:;:,:;:;:,:;:,:;:;:,:;:;:;:,:;:;:,:,:,:,:;:,:,:,:;:;:;:;:;:;:;:;:,:,:;:,:;:;:,:,:;:;:;:;:;:;:;:,:,:;:;:;:,:;:,:,:,:,:;:;:;:;:;:,:,:,:,:,:,:;:;:,:;:,:;
*/
// Railgun_Trolldom (the successor of Railgun_Swampshine_BailOut - avoiding second pattern comparison in BMH2 and pseudo-BMH4), copyleft 2016-Aug-19, Kaze.
// Railgun_Swampshine_BailOut, copyleft 2016-Aug-10, Kaze.
// Internet "home" page: http://www.codeproject.com/Articles/250566/Fastest-strstr-like-function-in-C
// My homepage (homeserver, often down): http://www.sanmayce.com/Railgun/
/*
!!!!!!!!!!!!!!!!!!!!!!!! BENCHMARKING GNU's memmem vs Railgun !!!!!!!!!!!!!!!!!!!!!!!! [
Add-on: 2016-Aug-22

Two things.

First, the fix from the last time was buggy, my apologies, now fixed, quite embarrassing since it is a simple left/right boundary check. It doesn't affect the speed, it appears as rare pattern hit misses.
Since I don't believe in saying "sorry" but in making things right, here my attempt to further disgrace my amateurish work follows:
Two years ago, I didn't pay due attention to adding 'Swampwalker' heuristic to the Railgun_Ennearch, I mean, only quick test was done and no real proofing - this was due not to a blunder of mine, nor carelessness, but overconfidence in my ability to write "on the fly". Stupid, indeed, however, when a coder gets momentum in writing simple etudes he starts gaining false confidence of mastering the subject, not good for sure!
Hopefully, other coders will learn to avoid such full of neglect style.

Second, wanted to present the heaviest testbed for search i.e. memmem() functions: it benefits the benchmarking (speed in real application) as well as bug-control.

The benchmark is downloadable at my INTERNET drive:
https://1drv.ms/u/s!AmWWFXGMzDmEglwjlUtnMJrfhosK

The speed showdown has three facets:
- compares the 64bit code generated from GCC 5.10 versus Intel 15.0 compilers;
- compares four types of datasets - search speed through English texts versus genome ACGT-type data versus binary versus UTF8;
- compares the tweaked Two-Way algorithm (implemented by Eric Blake) and adopted by GLIBC as memmem() versus my Railgun_Swampshine.

Note1: The GLIBC memmem() was taken from latest (2016-08-05) glibc 2.24 tar:
https://www.gnu.org/software/libc/
Note2: Eric Blake says that he enhanced the linearity of Two-Way by adding some sublinear paths, well, Railgun is all about sublinearity, so feel free to experiment with your own testfiles (worst-case-scenarios), just make such a file feed the compressor with it, then we will see how the LINEAR Two-Way behaves versus Railgun_Swampshine.
Note3: Just copy-and-paste 'Railgun_Swampshine' or 'Railgun_Ennearch' from the benchmark's source.

So the result on Core 2 Q9550s @2.83GHz DDR2 @666MHz / i5-2430M @3.00GHz DDR3 @666MHz:
--------------------------------------------------------------------------------------------------------------------------------
| Searcher                                  | GNU/GLIBC memmem()        | Railgun_Swampshine       | Railgun_Trolldom          | 
|--------------------------------------------------------------------------------------------------|---------------------------|
| Testfile\Compiler                         | Intel 15.0 | GCC 5.10     | Intel 15.0 | GCC 5.10    | Intel 15.0  | GCC 5.10    |
|------------------------------------------------------------------------------------------------------------------------------|
| Size: 27,703 bytes                        |     4506/- |   5330/14725 |    13198/- | 11581/15171 | 19105/22449 | 15493/21642 |
| Name: An_Interview_with_Carlos_Castaneda.TXT                          |            |             |             |             |
| LATENCY-WISE: Number of 'memmem()' Invocations: 308,062               |            |             |             |             |
| THROUGHPUT-WISE: Number of Total bytes Traversed: 3,242,492,648       |            |             |             |             |
|------------------------------------------------------------------------------------------------------------------------------|
| Size: 2,347,772 bytes                     |      190/- |      226/244 |     1654/- |   1729/1806 |   1794/1822 |   1743/1809 |
| Name: Gutenberg_EBook_Don_Quixote_996_(ANSI).txt                      |            |             |             |             |
| LATENCY-WISE: Number of 'memmem()' Invocations: 14,316,954            |            |             |             |             |
| THROUGHPUT-WISE: Number of Total bytes Traversed: 6,663,594,719,173   |            |             |             |             |
|------------------------------------------------------------------------------------------------------------------------------|
| Size: 899,425 bytes                       |      582/- |      760/816 |     3094/- |   2898/3088 |   3255/3289 |   2915/3322 |
| Name: Gutenberg_EBook_Dokoe_by_Hakucho_Masamune_(Japanese_UTF8).txt   |            |             |             |             |
| LATENCY-WISE: Number of 'memmem()' Invocations: 3,465,806             |            |             |             |             |
| THROUGHPUT-WISE: Number of Total bytes Traversed: 848,276,034,315     |            |             |             |             |
|------------------------------------------------------------------------------------------------------------------------------|
| Size: 4,487,433 bytes                     |      104/- |      109/116 |      445/- |     458/417 |     450/411 |     467/425 |
| Name: Dragonfly_genome_shotgun_sequence_(ACGT_alphabet).fasta         |            |             |             |             |
| LATENCY-WISE: Number of 'memmem()' Invocations: 20,540,375            |            |             |             |             |
| THROUGHPUT-WISE: Number of Total bytes Traversed: 13,592,530,857,131  |            |             |             |             |
|------------------------------------------------------------------------------------------------------------------------------|
| Size: 954,035 bytes                       |       99/- |      144/144 |      629/- |     580/682 |     634/807 |     585/725 |
| Name: LAOTZU_Wu_Wei_(BINARY).pdf                                      |            |             |             |             |
| LATENCY-WISE: Number of 'memmem()' Invocations: 27,594,933            |            |             |             |             |
| THROUGHPUT-WISE: Number of Total bytes Traversed: 8,702,455,122,519   |            |             |             |             |
|------------------------------------------------------------------------------------------------------------------------------|
| Size: 15,583,440 bytes                    |        -/- |          -/- |        -/- |     663/771 |     675/778 |     663/757 |
| Name: Arabian_Nights_complete.html                                    |            |             |             |             |
| LATENCY-WISE: Number of 'memmem()' Invocations: 72,313,262            |            |             |             |             |
| THROUGHPUT-WISE: Number of Total bytes Traversed: 105,631,163,854,099 |            |             |             |             |
--------------------------------------------------------------------------------------------------------------------------------

Note0: Railgun_Trolldom is slightly faster (both with Intel & GCC) than Railgun_Swampshine, this is mostly due to adding a bitwise BMH order 2 (8KB table overhead instead of 64KB) path - for haystacks <77777 bytes long - the warm-up is faster.
Note1: The numbers represent the rate (bytes/s) at which patterns/needles 4,5,6,8,9,10,12,13,14,16,17,18,24 bytes long are memmemed into 4KB, 256KB, 1MB, 256MB long haystacks.
in fact, these numbers are the compression speed using LZSS and memmem() as matchfinder.
Note2: The Arabian Nights is downloadable at:
https://ebooks.adelaide.edu.au/b/burton/richard/b97b/complete.html
Note3: On i5-2430M, TW is catching up since this CPU crunches instructions faster while the RAM speed is almost the same, Railgun suffers from the slow RAM fetches - the prefetcher and such suck.
Note4: With a simple English text 'Tales of 1001 Nights', 15,583,440 bytes long, the cumulative size of traversed haystack data is nearly 100TB, 105,631,163,854,099 ~ 1024^4 = 1,099,511,627,776.
Note5: With a simple French text 'Agatha_Christie_85-ebooks_(French)_TXT.tar', 32,007,168 bytes long, the cumulative size of traversed haystack data is nearly 200TB ~ 234,427,099,834,376.

Just to see how faster is Yann's Zstd in decompression (its level 12 is 377-331 MB/s faster), on Core 2 Q9550s @2.83GHz DDR2 @666MHz:
[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
D:\Nakamichi_Kintaro++_source_executables_64bit_(GCC510-vs-Intel150)_(TW-vs-RG)_BENCHMARK>Nakamichi_Kintaro++_Intel_15.0_64bit.exe Agatha_Christie_85-ebooks_(French)_TXT.tar
Nakamichi 'Kintaro++', written by Kaze, based on Nobuo Ito's LZSS source, babealicious suggestion by m^2 enforced, muffinesque suggestion by Jim Dempsey enforced.
Note1: This compile can handle files up to 1711MB.
Note2: The matchfinder/memmem() is Railgun_Trolldom.
Current priority class is HIGH_PRIORITY_CLASS.
Compressing 32007168 bytes ...
|; Each rotation means 64KB are encoded; Done 100%; Compression Ratio: 3.53:1
NumberOfFullLiterals (lower-the-better): 164
NumberOfFlushLiteralsHeuristic (bigger-the-better): 184323
Legend: WindowSizes: 1/2/3/4=Tiny/Short/Medium/Long
NumberOf(Tiny)Matches[Short]Window (4)[2]: 226869
NumberOf(Short)Matches[Short]Window (8)[2]: 119810
NumberOf(Medium)Matches[Short]Window (12)[2]: 71202
NumberOf(Long)Matches[Short]Window (16)[2]: 31955
NumberOf(MaxLong)Matches[Short]Window (24)[2]: 7078
NumberOf(Tiny)Matches[Medium]Window (5)[3]: 257313
NumberOf(Short)Matches[Medium]Window (9)[3]: 526493
NumberOf(Medium)Matches[Medium]Window (13)[3]: 285579
NumberOf(Long)Matches[Medium]Window (17)[3]: 158873
NumberOf(MaxLong)Matches[Medium]Window (24)[3]: 51276
NumberOf(Tiny)Matches[Long]Window (6)[4]: 41075
NumberOf(Short)Matches[Long]Window (10)[4]: 240454
NumberOf(Medium)Matches[Long]Window (14)[4]: 258653
NumberOf(Long)Matches[Long]Window (18)[4]: 209007
NumberOf(MaxLong)Matches[Long]Window (24)[4]: 190929
RAM-to-RAM performance: 605 bytes/s.
Compressed to 9076876 bytes.
LATENCY-WISE: Number of 'memmem()' Invocations: 102,091,852
THROUGHPUT-WISE: Number of Total bytes Traversed: 234,427,099,834,376

D:\Nakamichi_Kintaro++_source_executables_64bit_(GCC510-vs-Intel150)_(TW-vs-RG)_BENCHMARK>"Nakamichi_Kintaro++_Intel_15.0_64bit.exe" "Agatha_Christie_85-ebooks_(French)_TXT.tar.Nakamichi"
Nakamichi 'Kintaro++', written by Kaze, based on Nobuo Ito's LZSS source, babealicious suggestion by m^2 enforced, muffinesque suggestion by Jim Dempsey enforced.
Note1: This compile can handle files up to 1711MB.
Note2: The matchfinder/memmem() is Railgun_Trolldom.
Current priority class is HIGH_PRIORITY_CLASS.
Decompressing 9076876 bytes ...
RAM-to-RAM performance: 331 MB/s.
Compression Ratio (bigger-the-better): 3.53:1

D:\Nakamichi_Kintaro++_source_executables_64bit_(GCC510-vs-Intel150)_(TW-vs-RG)_BENCHMARK>zstd-windows-v0.8.1_win64.exe -h
*** zstd command line interface 64-bits v0.8.1, by Yann Collet ***
...

D:\Nakamichi_Kintaro++_source_executables_64bit_(GCC510-vs-Intel150)_(TW-vs-RG)_BENCHMARK>zstd-windows-v0.8.1_win64.exe -b12 "Agatha_Christie_85-ebooks_(French)_TXT.tar"
12#_(French)_TXT.tar :  32007168 ->   8965791 (3.570),   6.7 MB/s , 377.0 MB/s

D:\Nakamichi_Kintaro++_source_executables_64bit_(GCC510-vs-Intel150)_(TW-vs-RG)_BENCHMARK>zstd-windows-v0.8.1_win64.exe -b22 "Agatha_Christie_85-ebooks_(French)_TXT.tar"
22#_(French)_TXT.tar :  32007168 ->   6802321 (4.705),   1.0 MB/s , 260.7 MB/s

D:\Nakamichi_Kintaro++_source_executables_64bit_(GCC510-vs-Intel150)_(TW-vs-RG)_BENCHMARK>
]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

Two-Way is significantly slower than BMH Order 2, the speed-down is in range:
- for TEXTUAL ANSI alphabets: 1729/226= 7.6x
- for TEXTUAL UTF8 alphabets: 2898/760= 3.8x
- for TEXTUAL ACGT alphabets:  458/109= 4.2x
- for BINARY-esque alphabets:  580/144= 4.0x

For faster RAM, than mine @666MHz, and for haystacks multimegabytes long, the speedup goes beyond 8x.

The benchmark shows the real behavior (both latency and raw speed) of the memmem variants, I added also the Thierry Lecroq's Two-Way implementation:
http://www-igm.univ-mlv.fr/~lecroq/string/node26.html#SECTION00260
However, Eric Blake's one is faster, so it was chosen for the speed showdown.

Once I measured the total length of traversed haystacks, and for files 100+MB long, it went ... quintillion of bytes i.e. petabytes - good torture it is.

!!!!!!!!!!!!!!!!!!!!!!!! BENCHMARKING GNU's memmem vs Railgun !!!!!!!!!!!!!!!!!!!!!!!! ]
*/
// 2014-Apr-27: The nasty SIGNED/UNSIGNED bug in 'Swampshines' which I illustrated several months ago in my fuzzy search article now is fixed here too:
/*
The bug is this (the variables 'i' and 'PRIMALposition' are uint32_t):
Next line assumes -19 >= 0 is true:
if ( (i-(PRIMALposition-1)) >= 0) printf ("THE NASTY BUG AGAIN: %d >= 0\n", i-(PRIMALposition-1));
Next line assumes -19 >= 0 is false:
if ( (signed int)(i-(PRIMALposition-1)) >= 0) printf ("THE NASTY BUG AGAIN: %d >= 0\n", i-(PRIMALposition-1));
And the actual fix:
...
// If we miss to hit then no need to compare the original: Needle
if ( count <= 0 ) {
// I have to add out-of-range checks...
// i-(PRIMALposition-1) >= 0
// &pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4
// i-(PRIMALposition-1)+(count-1) >= 0
// &pbTarget[i-(PRIMALposition-1)+(count-1)] <= pbTargetMax - 4

// "FIX" from 2014-Apr-27:
// Because (count-1) is negative, above fours are reduced to next twos:
// i-(PRIMALposition-1)+(count-1) >= 0
// &pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4
	// The line below is BUGGY:
	//if ( (i-(PRIMALposition-1) >= 0) && (&pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4) && (&pbTarget[i-(PRIMALposition-1)+(count-1)] <= pbTargetMax - 4) ) {
	// The line below is NOT OKAY, in fact so stupid, grrr, not a blunder, not carelessness, but overconfidence in writing "on the fly":
	//if ( ((signed int)(i-(PRIMALposition-1)+(count-1)) >= 0) && (&pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4) ) {
// FIX from 2016-Aug-10 (two times failed to do simple boundary checks, pfu):
	if ( ((signed int)(i-(PRIMALposition-1)) >= 0) && (&pbTarget[i-(PRIMALposition-1)]+((PRIMALlengthCANDIDATE-4+1)-1) <= pbTargetMax - 4) ) {
		if ( *(uint32_t *)&pbTarget[i-(PRIMALposition-1)] == *(uint32_t *)(pbPattern-(PRIMALposition-1))) { // This fast check ensures not missing a match (for remainder) when going under 0 in loop below:
			count = PRIMALlengthCANDIDATE-4+1; 
			while ( count > 0 && *(uint32_t *)(pbPattern-(PRIMALposition-1)+count-1) == *(uint32_t *)(&pbTarget[i-(PRIMALposition-1)]+(count-1)) )
				count = count-4;
			if ( count <= 0 ) return(pbTarget+i-(PRIMALposition-1));	
		}
	}
}
...
*/
// Railgun_Swampshine_BailOut, copyleft 2014-Jan-31, Kaze.
// Caution: For better speed the case 'if (cbPattern==1)' was removed, so Pattern must be longer than 1 char.
#define NeedleThreshold2vs4swampLITE 9+10 // Should be bigger than 9. BMH2 works up to this value (inclusive), if bigger then BMH4 takes over. Should be <=255 otherwise the 0|1 BMH2 should be used.
char * Railgun_Trolldom (char * pbTarget, char * pbPattern, uint32_t cbTarget, uint32_t cbPattern)
{
	char * pbTargetMax = pbTarget + cbTarget;
	register uint32_t ulHashPattern;
	signed long count;

	unsigned char bm_Horspool_Order2[256*256]; // Bitwise soon...
	unsigned char bm_Horspool_Order2bitwise[(256*256)>>3]; // Bitwise soon...
	uint32_t i, Gulliver;

	uint32_t PRIMALposition, PRIMALpositionCANDIDATE;
	uint32_t PRIMALlength, PRIMALlengthCANDIDATE;
	uint32_t j, FoundAtPosition;

// Quadruplet [
    //char * pbTargetMax = pbTarget + cbTarget;
    //register unsigned long  ulHashPattern;
    unsigned long ulHashTarget;
    //unsigned long count;
    unsigned long countSTATIC;
    unsigned char SINGLET;
    unsigned long Quadruplet2nd;
    unsigned long Quadruplet3rd;
    unsigned long Quadruplet4th;
    unsigned long  AdvanceHopperGrass;
// Quadruplet ]

	if (cbPattern > cbTarget) return(NULL);

	if ( cbPattern<4 ) { 
		// SSE2 i.e. 128bit Assembly rules here, Mischa knows best:
		// ...
        	pbTarget = pbTarget+cbPattern;
		ulHashPattern = ( (*(char *)(pbPattern))<<8 ) + *(pbPattern+(cbPattern-1));
		if ( cbPattern==3 ) {
			for ( ;; ) {
				if ( ulHashPattern == ( (*(char *)(pbTarget-3))<<8 ) + *(pbTarget-1) ) {
					if ( *(char *)(pbPattern+1) == *(char *)(pbTarget-2) ) return((pbTarget-3));
				}
				if ( (char)(ulHashPattern>>8) != *(pbTarget-2) ) { 
					pbTarget++;
					if ( (char)(ulHashPattern>>8) != *(pbTarget-2) ) pbTarget++;
				}
				pbTarget++;
				if (pbTarget > pbTargetMax) return(NULL);
			}
		} else {
		}
		for ( ;; ) {
			if ( ulHashPattern == ( (*(char *)(pbTarget-2))<<8 ) + *(pbTarget-1) ) return((pbTarget-2));
			if ( (char)(ulHashPattern>>8) != *(pbTarget-1) ) pbTarget++;
			pbTarget++;
			if (pbTarget > pbTargetMax) return(NULL);
		}
	} else { //if ( cbPattern<4 )
		if ( cbPattern<=NeedleThreshold2vs4swampLITE ) { 

// This is the awesome 'Railgun_Quadruplet', it did outperform EVERYWHERE the fastest strstr (back in old GLIBCes ~2003, by the Dutch hacker Stephen R. van den Berg), suitable for short haystacks ~100bytes.
// Caution: For better speed the case 'if (cbPattern==1)' was removed, so Pattern must be longer than 1 char.
// char * Railgun_Quadruplet (char * pbTarget, char * pbPattern, unsigned long cbTarget, unsigned long cbPattern)
// ...
//    if (cbPattern > cbTarget) return(NULL);
//} else { //if ( cbPattern<4)
if (cbTarget<777) // This value is arbitrary(don't know how exactly), it ensures(at least must) better performance than 'Boyer_Moore_Horspool'.
{
        pbTarget = pbTarget+cbPattern;
        ulHashPattern = *(unsigned long *)(pbPattern);
//        countSTATIC = cbPattern-1;

    //SINGLET = *(char *)(pbPattern);
    SINGLET = ulHashPattern & 0xFF;
    Quadruplet2nd = SINGLET<<8;
    Quadruplet3rd = SINGLET<<16;
    Quadruplet4th = SINGLET<<24;

    for ( ;; )
    {
	AdvanceHopperGrass = 0;
	ulHashTarget = *(unsigned long *)(pbTarget-cbPattern);

        if ( ulHashPattern == ulHashTarget ) { // Three unnecessary comparisons here, but 'AdvanceHopperGrass' must be calculated - it has a higher priority.
//         count = countSTATIC;
//         while ( count && *(char *)(pbPattern+1+(countSTATIC-count)) == *(char *)(pbTarget-cbPattern+1+(countSTATIC-count)) ) {
//	       if ( countSTATIC==AdvanceHopperGrass+count && SINGLET != *(char *)(pbTarget-cbPattern+1+(countSTATIC-count)) ) AdvanceHopperGrass++;
//               count--;
//         }
         count = cbPattern-1;
         while ( count && *(char *)(pbPattern+(cbPattern-count)) == *(char *)(pbTarget-count) ) {
	       if ( cbPattern-1==AdvanceHopperGrass+count && SINGLET != *(char *)(pbTarget-count) ) AdvanceHopperGrass++;
               count--;
         }
         if ( count == 0) return((pbTarget-cbPattern));
        } else { // The goal here: to avoid memory accesses by stressing the registers.
    if ( Quadruplet2nd != (ulHashTarget & 0x0000FF00) ) {
         AdvanceHopperGrass++;
         if ( Quadruplet3rd != (ulHashTarget & 0x00FF0000) ) {
              AdvanceHopperGrass++;
              if ( Quadruplet4th != (ulHashTarget & 0xFF000000) ) AdvanceHopperGrass++;
         }
    }
	}

	AdvanceHopperGrass++;

	pbTarget = pbTarget + AdvanceHopperGrass;
        if (pbTarget > pbTargetMax)
            return(NULL);
    }
} else if (cbTarget<77777) { // The warmup/overhead is lowered from 64K down to 8K, however the bitwise additional instructions quickly start hurting the throughput/traversal.
// The below bitwise 0|1 BMH2 gives 1427 bytes/s for 'Don_Quixote' with Intel:
// The below bitwise 0|1 BMH2 gives 1242 bytes/s for 'Don_Quixote' with GCC:
//	} else { //if ( cbPattern<4 )
//		if ( cbPattern<=NeedleThreshold2vs4Decumanus ) { 
			// BMH order 2, needle should be >=4:
			ulHashPattern = *(uint32_t *)(pbPattern); // First four bytes
			//for (i=0; i < 256*256; i++) {bm_Horspool_Order2[i]=0;}
			for (i=0; i < (256*256)>>3; i++) {bm_Horspool_Order2bitwise[i]=0;}
			//for (i=0; i < cbPattern-1; i++) bm_Horspool_Order2[*(unsigned short *)(pbPattern+i)]=1;
			for (i=0; i < cbPattern-2+1; i++) bm_Horspool_Order2bitwise[(*(unsigned short *)(pbPattern+i))>>3]= bm_Horspool_Order2bitwise[(*(unsigned short *)(pbPattern+i))>>3] | (1<<((*(unsigned short *)(pbPattern+i))&0x7));
			i=0;
			while (i <= cbTarget-cbPattern) {
				Gulliver = 1; // 'Gulliver' is the skip
				//if ( bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+cbPattern-1-1]] != 0 ) {
				if ( ( bm_Horspool_Order2bitwise[(*(unsigned short *)&pbTarget[i+cbPattern-1-1])>>3] & (1<<((*(unsigned short *)&pbTarget[i+cbPattern-1-1])&0x7)) ) != 0 ) {
					//if ( bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+cbPattern-1-1-2]] == 0 ) Gulliver = cbPattern-(2-1)-2; else {
					if ( ( bm_Horspool_Order2bitwise[(*(unsigned short *)&pbTarget[i+cbPattern-1-1-2])>>3] & (1<<((*(unsigned short *)&pbTarget[i+cbPattern-1-1-2])&0x7)) ) == 0 ) Gulliver = cbPattern-(2-1)-2; else {
						if ( *(uint32_t *)&pbTarget[i] == ulHashPattern) { // This fast check ensures not missing a match (for remainder) when going under 0 in loop below:
							count = cbPattern-4+1; 
							while ( count > 0 && *(uint32_t *)(pbPattern+count-1) == *(uint32_t *)(&pbTarget[i]+(count-1)) )
								count = count-4;
							if ( count <= 0 ) return(pbTarget+i);
						}
					}
				} else Gulliver = cbPattern-(2-1);
				i = i + Gulliver;
				//GlobalI++; // Comment it, it is only for stats.
			}
			return(NULL);
//		} else { // if ( cbPattern<=NeedleThreshold2vs4Decumanus )
} else { //if (cbTarget<777)
			// BMH order 2, needle should be >=4:
			ulHashPattern = *(uint32_t *)(pbPattern); // First four bytes
			for (i=0; i < 256*256; i++) {bm_Horspool_Order2[i]=0;}
			for (i=0; i < cbPattern-1; i++) bm_Horspool_Order2[*(unsigned short *)(pbPattern+i)]=1;
			i=0;
			while (i <= cbTarget-cbPattern) {
				Gulliver = 1; // 'Gulliver' is the skip
				if ( bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+cbPattern-1-1]] != 0 ) {
					if ( bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+cbPattern-1-1-2]] == 0 ) Gulliver = cbPattern-(2-1)-2; else {
						if ( *(uint32_t *)&pbTarget[i] == ulHashPattern) { // This fast check ensures not missing a match (for remainder) when going under 0 in loop below:
							count = cbPattern-4+1; 
							while ( count > 0 && *(uint32_t *)(pbPattern+count-1) == *(uint32_t *)(&pbTarget[i]+(count-1)) )
								count = count-4;
							if ( count <= 0 ) return(pbTarget+i);
						}
					}
				} else Gulliver = cbPattern-(2-1);
				i = i + Gulliver;
				//GlobalI++; // Comment it, it is only for stats.
			}
			return(NULL);

// Slower than Swampshine's simple 0|1 segment:
/*
PRIMALlength=0;
for (i=0+(1); i < cbPattern-2+1+(1)-(1); i++) { // -(1) because the last BB order 2 has no counterpart(s)
    FoundAtPosition = cbPattern;
    PRIMALpositionCANDIDATE=i;
    while ( PRIMALpositionCANDIDATE <= (FoundAtPosition-1) ) {
        j = PRIMALpositionCANDIDATE + 1;
        while ( j <= (FoundAtPosition-1) ) {
            if ( *(unsigned short *)(pbPattern+PRIMALpositionCANDIDATE-(1)) == *(unsigned short *)(pbPattern+j-(1)) ) FoundAtPosition = j;
            j++;
        }
        PRIMALpositionCANDIDATE++;
    }
    PRIMALlengthCANDIDATE = (FoundAtPosition-1)-i+(2);
    if (PRIMALlengthCANDIDATE >= PRIMALlength) {PRIMALposition=i; PRIMALlength = PRIMALlengthCANDIDATE;}
}
PRIMALlengthCANDIDATE = cbPattern;
cbPattern = PRIMALlength;
pbPattern = pbPattern + (PRIMALposition-1);
if (cbPattern<4) {
	cbPattern = PRIMALlengthCANDIDATE;
	pbPattern = pbPattern - (PRIMALposition-1);
}
if (cbPattern == PRIMALlengthCANDIDATE) {
			// BMH order 2, needle should be >=4:
			ulHashPattern = *(uint32_t *)(pbPattern); // First four bytes
			for (i=0; i < 256*256; i++) {bm_Horspool_Order2[i]=0;}
			for (i=0; i < cbPattern-1; i++) bm_Horspool_Order2[*(unsigned short *)(pbPattern+i)]=1;
			i=0;
			while (i <= cbTarget-cbPattern) {
				Gulliver = 1; // 'Gulliver' is the skip
				if ( bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+cbPattern-1-1]] != 0 ) {
					if ( bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+cbPattern-1-1-2]] == 0 ) Gulliver = cbPattern-(2-1)-2; else {
						if ( *(uint32_t *)&pbTarget[i] == ulHashPattern) { // This fast check ensures not missing a match (for remainder) when going under 0 in loop below:
							count = cbPattern-4+1; 
							while ( count > 0 && *(uint32_t *)(pbPattern+count-1) == *(uint32_t *)(&pbTarget[i]+(count-1)) )
								count = count-4;
							if ( count <= 0 ) return(pbTarget+i);
						}
					}
				} else Gulliver = cbPattern-(2-1);
				i = i + Gulliver;
				//GlobalI++; // Comment it, it is only for stats.
			}
			return(NULL);
} else { //if (cbPattern == PRIMALlengthCANDIDATE) {
// BMH Order 2 [
			ulHashPattern = *(uint32_t *)(pbPattern); // First four bytes
			for (i=0; i < 256*256; i++) {bm_Horspool_Order2[i]= cbPattern-1;} // cbPattern-(Order-1) for Horspool; 'memset' if not optimized
			// The above 'for' gives  1424 bytes/s for 'Don_Quixote' with Intel:
			// The above 'for' gives  1431 bytes/s for 'Don_Quixote' with GCC:
			// The below 'memset' gives  1389 bytes/s for 'Don_Quixote' with Intel:
			// The below 'memset' gives  1432 bytes/s for 'Don_Quixote' with GCC:
			//memset(&bm_Horspool_Order2[0], cbPattern-1, 256*256); // Why why? It is 1700:1000 slower than above 'for'!?
			for (i=0; i < cbPattern-1; i++) bm_Horspool_Order2[*(unsigned short *)(pbPattern+i)]=i; // Rightmost appearance/position is needed
			i=0;
			while (i <= cbTarget-cbPattern) { 
				Gulliver = bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+cbPattern-1-1]];
				if ( Gulliver != cbPattern-1 ) { // CASE #2: if equal means the pair (char order 2) is not found i.e. Gulliver remains intact, skip the whole pattern and fall back (Order-1) chars i.e. one char for Order 2
				if ( Gulliver == cbPattern-2 ) { // CASE #1: means the pair (char order 2) is found
					if ( *(uint32_t *)&pbTarget[i] == ulHashPattern) {
						count = cbPattern-4+1; 
						while ( count > 0 && *(uint32_t *)(pbPattern+count-1) == *(uint32_t *)(&pbTarget[i]+(count-1)) )
							count = count-4;
// If we miss to hit then no need to compare the original: Needle
if ( count <= 0 ) {
// I have to add out-of-range checks...
// i-(PRIMALposition-1) >= 0
// &pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4
// i-(PRIMALposition-1)+(count-1) >= 0
// &pbTarget[i-(PRIMALposition-1)+(count-1)] <= pbTargetMax - 4

// "FIX" from 2014-Apr-27:
// Because (count-1) is negative, above fours are reduced to next twos:
// i-(PRIMALposition-1)+(count-1) >= 0
// &pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4
	// The line below is BUGGY:
	//if ( (i-(PRIMALposition-1) >= 0) && (&pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4) && (&pbTarget[i-(PRIMALposition-1)+(count-1)] <= pbTargetMax - 4) ) {
	// The line below is NOT OKAY, in fact so stupid, grrr, not a blunder, not carelessness, but overconfidence in writing "on the fly":
	//if ( ((signed int)(i-(PRIMALposition-1)+(count-1)) >= 0) && (&pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4) ) {
// FIX from 2016-Aug-10 (two times failed to do simple boundary checks, pfu):
	if ( ((signed int)(i-(PRIMALposition-1)) >= 0) && (&pbTarget[i-(PRIMALposition-1)]+((PRIMALlengthCANDIDATE-4+1)-1) <= pbTargetMax - 4) ) {
		if ( *(uint32_t *)&pbTarget[i-(PRIMALposition-1)] == *(uint32_t *)(pbPattern-(PRIMALposition-1))) { // This fast check ensures not missing a match (for remainder) when going under 0 in loop below:
			count = PRIMALlengthCANDIDATE-4+1; 
			while ( count > 0 && *(uint32_t *)(pbPattern-(PRIMALposition-1)+count-1) == *(uint32_t *)(&pbTarget[i-(PRIMALposition-1)]+(count-1)) )
				count = count-4;
			if ( count <= 0 ) return(pbTarget+i-(PRIMALposition-1));	
		}
	}
}
					}
					Gulliver = 1;
				} else
					Gulliver = cbPattern - Gulliver - 2; // CASE #3: the pair is found and not as suffix i.e. rightmost position
				}
				i = i + Gulliver;
				//GlobalI++; // Comment it, it is only for stats.
			}
			return(NULL);
// BMH Order 2 ]
} //if (cbPattern == PRIMALlengthCANDIDATE) {
*/

/*
So the result on Core 2 Q9550s @2.83GHz:
---------------------------------------------------------------------------------------------------------------------
| testfile\Searcher                         | GNU/GLIBC memmem()    | Railgun_Swampshine    | Railgun_Trolldom      | 
|-------------------------------------------------------------------------------------------|-----------------------|
| Compiler                                  | Intel 15.0 | GCC 5.10 | Intel 15.0 | GCC 5.10 | Intel 15.0 | GCC 5.10 |
|-------------------------------------------------------------------------------------------|-----------------------|
| The_Project_Gutenberg_EBook_of_Don        |        190 |      226 |       1654 |     1729 |       1147 |     1764 |
| _Quixote_996_(ANSI).txt                   |            |          |            |          |            |          |
| 2,347,772 bytes                           |            |          |            |          |            |          |
|-------------------------------------------------------------------------------------------|-----------------------|
| The_Project_Gutenberg_EBook_of_Dokoe      |        582 |      760 |       3094 |     2898 |       2410 |     3036 |
| _by_Hakucho_Masamune_(Japanese_UTF-8).txt |            |          |            |          |            |          |
| 899,425 bytes                             |            |          |            |          |            |          |
|-------------------------------------------------------------------------------------------|-----------------------|
| Dragonfly_genome_shotgun_sequence         |        104 |      109 |        445 |      458 |        484 |      553 |
| _(ACGT_alphabet).fasta                    |            |          |            |          |            |          |
| 4,487,433 bytes                           |            |          |            |          |            |          |
|-------------------------------------------------------------------------------------------|-----------------------|
| LAOTZU_Wu_Wei_(BINARY).pdf                |         99 |      144 |        629 |      580 |        185 |      570 |
| 954,035 bytes                             |            |          |            |          |            |          |
|-------------------------------------------------------------------------------------------|-----------------------|
Below segment (when compiled with Intel) is very slow, see Railgun_Trolldom two sub-columns above, compared to GCC:
*/
/*
// BMH Order 2 [
			ulHashPattern = *(uint32_t *)(pbPattern); // First four bytes
			for (i=0; i < 256*256; i++) {bm_Horspool_Order2[i]= (cbPattern-1);} // cbPattern-(Order-1) for Horspool; 'memset' if not optimized
			// The above 'for' is translated by Intel as:
//.B5.21::                        
//  0013f 83 c0 40         add eax, 64                            
//  00142 66 0f 7f 44 14 
//        60               movdqa XMMWORD PTR [96+rsp+rdx], xmm0  
//  00148 3d 00 00 01 00   cmp eax, 65536                         
//  0014d 66 0f 7f 44 14 
//        70               movdqa XMMWORD PTR [112+rsp+rdx], xmm0 
//  00153 66 0f 7f 84 14 
//        80 00 00 00      movdqa XMMWORD PTR [128+rsp+rdx], xmm0 
//  0015c 66 0f 7f 84 14 
//        90 00 00 00      movdqa XMMWORD PTR [144+rsp+rdx], xmm0 
//  00165 89 c2            mov edx, eax                           
//  00167 72 d6            jb .B5.21 
			//memset(&bm_Horspool_Order2[0], cbPattern-1, 256*256); // Why why? It is 1700:1000 slower than above 'for'!?
			// The above 'memset' is translated by Intel as:
//  00127 41 b8 00 00 01 
//        00               mov r8d, 65536                         
//  0012d 44 8b 26         mov r12d, DWORD PTR [rsi]              
//  00130 e8 fc ff ff ff   call _intel_fast_memset                
			// ! The problem is that 256*256, 64KB, is already too much, going bitwise i.e. 8KB is not that better, when 'cbPattern-1' is bigger than 255 - an unsigned char - then 
			// we must switch to 0|1 table i.e. present or not. Since we are in 'if ( cbPattern<=NeedleThreshold2vs4swampLITE ) {' branch and NeedleThreshold2vs4swampLITE, by default, is 19 - it is okay to use 'memset'. !
			for (i=0; i < cbPattern-1; i++) bm_Horspool_Order2[*(unsigned short *)(pbPattern+i)]=i; // Rightmost appearance/position is needed
			i=0;
			while (i <= cbTarget-cbPattern) { 
				Gulliver = bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+cbPattern-1-1]];
				if ( Gulliver != cbPattern-1 ) { // CASE #2: if equal means the pair (char order 2) is not found i.e. Gulliver remains intact, skip the whole pattern and fall back (Order-1) chars i.e. one char for Order 2
				if ( Gulliver == cbPattern-2 ) { // CASE #1: means the pair (char order 2) is found
					if ( *(uint32_t *)&pbTarget[i] == ulHashPattern) {
						count = cbPattern-4+1; 
						while ( count > 0 && *(uint32_t *)(pbPattern+count-1) == *(uint32_t *)(&pbTarget[i]+(count-1)) )
							count = count-4;
						if ( count <= 0 ) return(pbTarget+i);	
					}
					Gulliver = 1;
				} else
					Gulliver = cbPattern - Gulliver - 2; // CASE #3: the pair is found and not as suffix i.e. rightmost position
				}
				i = i + Gulliver;
				//GlobalI++; // Comment it, it is only for stats.
			}
			return(NULL);
// BMH Order 2 ]
*/
// Above fragment in Assembly:
/*
; mark_description "Intel(R) C++ Intel(R) 64 Compiler XE for applications running on Intel(R) 64, Version 15.0.0.108 Build 20140";
; mark_description "-O3 -QxSSE2 -D_N_XMM -D_N_prefetch_4096 -D_N_Branchfull -D_N_HIGH_PRIORITY -FA";
       ALIGN     16
.B6.1::                         ; Preds .B6.0
        push      rbx                                           ;3435.1
        push      r13                                           ;3435.1
        push      r15                                           ;3435.1
        push      rbp                                           ;3435.1
        mov       eax, 65592                                    ;3435.1
        call      __chkstk                                      ;3435.1
        sub       rsp, 65592                                    ;3435.1
        cmp       r9d, r8d                                      ;3460.18
        ja        .B6.25        ; Prob 28%                      ;3460.18
                                ; LOE rdx rcx rbx rsi rdi r12 r14 r8d r9d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.3::                         ; Preds .B6.1
        mov       r13d, DWORD PTR [rdx]                         ;3491.33
        lea       ebp, DWORD PTR [-1+r9]                        ;3492.67
        movzx     eax, bpl                                      ;3492.67
        xor       r10d, r10d                                    ;3492.4
        movd      xmm0, eax                                     ;3492.67
        xor       eax, eax                                      ;3492.4
        punpcklbw xmm0, xmm0                                    ;3492.67
        punpcklwd xmm0, xmm0                                    ;3492.67
        punpckldq xmm0, xmm0                                    ;3492.67
        punpcklqdq xmm0, xmm0                                   ;3492.67
                                ; LOE rdx rcx rbx rsi rdi r10 r12 r14 eax ebp r8d r9d r13d xmm0 xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.4::                         ; Preds .B6.4 .B6.3
        add       eax, 64                                       ;3492.4
        movdqa    XMMWORD PTR [48+rsp+r10], xmm0                ;3492.33
        cmp       eax, 65536                                    ;3492.4
        movdqa    XMMWORD PTR [64+rsp+r10], xmm0                ;3492.33
        movdqa    XMMWORD PTR [80+rsp+r10], xmm0                ;3492.33
        movdqa    XMMWORD PTR [96+rsp+r10], xmm0                ;3492.33
        mov       r10d, eax                                     ;3492.4
        jb        .B6.4         ; Prob 99%                      ;3492.4
                                ; LOE rdx rcx rbx rsi rdi r10 r12 r14 eax ebp r8d r9d r13d xmm0 xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.5::                         ; Preds .B6.4
        test      ebp, ebp                                      ;3515.28
        je        .B6.12        ; Prob 50%                      ;3515.28
                                ; LOE rdx rcx rbx rsi rdi r12 r14 ebp r8d r9d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.6::                         ; Preds .B6.5
        mov       eax, 1                                        ;3515.4
        lea       r11d, DWORD PTR [-1+r9]                       ;3515.4
        mov       r15d, r11d                                    ;3515.4
        xor       r10d, r10d                                    ;3515.4
        shr       r15d, 1                                       ;3515.4
        test      r15d, r15d                                    ;3515.4
        jbe       .B6.10        ; Prob 15%                      ;3515.4
                                ; LOE rdx rcx rbx rsi rdi r12 r14 eax ebp r8d r9d r10d r11d r13d r15d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.8::                         ; Preds .B6.6 .B6.8
        lea       eax, DWORD PTR [r10+r10]                      ;3515.36
        movzx     ebx, WORD PTR [rax+rdx]                       ;3515.75
        mov       BYTE PTR [48+rsp+rbx], al                     ;3515.36
        lea       eax, DWORD PTR [1+r10+r10]                    ;3515.36
        inc       r10d                                          ;3515.4
        cmp       r10d, r15d                                    ;3515.4
        movzx     ebx, WORD PTR [rax+rdx]                       ;3515.75
        mov       BYTE PTR [48+rsp+rbx], al                     ;3515.36
        jb        .B6.8         ; Prob 64%                      ;3515.4
                                ; LOE rdx rcx rsi rdi r12 r14 ebp r8d r9d r10d r11d r13d r15d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.9::                         ; Preds .B6.8
        lea       eax, DWORD PTR [1+r10+r10]                    ;3515.4
                                ; LOE rdx rcx rbx rsi rdi r12 r14 eax ebp r8d r9d r11d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.10::                        ; Preds .B6.9 .B6.6
        dec       eax                                           ;3515.36
        cmp       eax, r11d                                     ;3515.4
        jae       .B6.12        ; Prob 15%                      ;3515.4
                                ; LOE rax rdx rcx rbx rsi rdi r12 r14 ebp r8d r9d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.11::                        ; Preds .B6.10
        movzx     r10d, WORD PTR [rax+rdx]                      ;3515.75
        mov       BYTE PTR [48+rsp+r10], al                     ;3515.36
                                ; LOE rdx rcx rbx rsi rdi r12 r14 ebp r8d r9d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.12::                        ; Preds .B6.5 .B6.10 .B6.11
        xor       r10d, r10d                                    ;3516.4
        lea       r15d, DWORD PTR [-3+r9]                       ;3522.27
        movsxd    r15, r15d                                     ;3522.7
        sub       r8d, r9d                                      ;3517.16
        lea       r11d, DWORD PTR [-2+r9]                       ;3520.32
                                ; LOE rdx rcx rsi rdi r12 r14 r15 ebp r8d r9d r10d r11d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.13::                        ; Preds .B6.12 .B6.24
        lea       eax, DWORD PTR [-2+r9+r10]                    ;3518.78
        movzx     ebx, WORD PTR [rax+rcx]                       ;3518.55
        movzx     eax, BYTE PTR [48+rsp+rbx]                    ;3518.16
        cmp       eax, ebp                                      ;3519.32
        je        .B6.24        ; Prob 50%                      ;3519.32
                                ; LOE rdx rcx rsi rdi r12 r14 r15 eax ebp r8d r9d r10d r11d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.14::                        ; Preds .B6.13
        cmp       eax, r11d                                     ;3520.32
        jne       .B6.23        ; Prob 62%                      ;3520.32
                                ; LOE rdx rcx rsi rdi r12 r14 r15 eax ebp r8d r9d r10d r11d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.15::                        ; Preds .B6.14
        mov       eax, r10d                                     ;3521.25
        add       rax, rcx                                      ;3521.25
        cmp       r13d, DWORD PTR [rax]                         ;3521.40
        je        .B6.17        ; Prob 50%                      ;3521.40
                                ; LOE rax rdx rcx rsi rdi r12 r14 r15 ebp r8d r9d r10d r11d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.16::                        ; Preds .B6.26 .B6.15
        mov       eax, 1                                        ;3527.6
        jmp       .B6.24        ; Prob 100%                     ;3527.6
                                ; LOE rdx rcx rsi rdi r12 r14 r15 eax ebp r8d r9d r10d r11d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.17::                        ; Preds .B6.15
        mov       rbx, r15                                      ;3522.7
        test      r15, r15                                      ;3523.23
        jle       .B6.22        ; Prob 2%                       ;3523.23
                                ; LOE rax rdx rcx rbx rsi rdi r12 r14 r15 ebp r8d r9d r10d r11d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.18::                        ; Preds .B6.17
        mov       QWORD PTR [32+rsp], rsi                       ;
                                ; LOE rax rdx rcx rbx rdi r12 r14 r15 ebp r8d r9d r10d r11d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.19::                        ; Preds .B6.20 .B6.18
        mov       esi, DWORD PTR [-1+rbx+rdx]                   ;3523.58
        cmp       esi, DWORD PTR [-1+rbx+rax]                   ;3523.79
        jne       .B6.26        ; Prob 20%                      ;3523.79
                                ; LOE rax rdx rcx rbx rdi r12 r14 r15 ebp r8d r9d r10d r11d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.20::                        ; Preds .B6.19
        add       rbx, -4                                       ;3524.22
        test      rbx, rbx                                      ;3523.23
        jg        .B6.19        ; Prob 82%                      ;3523.23
                                ; LOE rax rdx rcx rbx rdi r12 r14 r15 ebp r8d r9d r10d r11d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.21::                        ; Preds .B6.20
        mov       rsi, QWORD PTR [32+rsp]                       ;
                                ; LOE rax rbx rsi rdi r12 r14 xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.22::                        ; Preds .B6.17 .B6.21
        add       rsp, 65592                                    ;3525.32
        pop       rbp                                           ;3525.32
        pop       r15                                           ;3525.32
        pop       r13                                           ;3525.32
        pop       rbx                                           ;3525.32
        ret                                                     ;3525.32
                                ; LOE
.B6.23::                        ; Preds .B6.14
        neg       eax                                           ;3529.17
        add       eax, r9d                                      ;3529.17
        add       eax, -2                                       ;3529.40
                                ; LOE rdx rcx rsi rdi r12 r14 r15 eax ebp r8d r9d r10d r11d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.24::                        ; Preds .B6.16 .B6.23 .B6.13
        add       r10d, eax                                     ;3531.13
        cmp       r10d, r8d                                     ;3517.25
        jbe       .B6.13        ; Prob 82%                      ;3517.25
                                ; LOE rdx rcx rsi rdi r12 r14 r15 ebp r8d r9d r10d r11d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B6.25::                        ; Preds .B6.1 .B6.24
        xor       eax, eax                                      ;3534.10
        add       rsp, 65592                                    ;3534.10
        pop       rbp                                           ;3534.10
        pop       r15                                           ;3534.10
        pop       r13                                           ;3534.10
        pop       rbx                                           ;3534.10
        ret                                                     ;3534.10
                                ; LOE
.B6.26::                        ; Preds .B6.19                  ; Infreq
        mov       rsi, QWORD PTR [32+rsp]                       ;
        jmp       .B6.16        ; Prob 100%                     ;
*/

// GCC 5.10; >gcc -O3 -m64 -fomit-frame-pointer
/*
Railgun_Trolldom:
	pushq	%r15
	.seh_pushreg	%r15
	movl	$65592, %eax
	pushq	%r14
	.seh_pushreg	%r14
	pushq	%r13
	.seh_pushreg	%r13
	pushq	%r12
	.seh_pushreg	%r12
	pushq	%rbp
	.seh_pushreg	%rbp
	pushq	%rdi
	.seh_pushreg	%rdi
	pushq	%rsi
	.seh_pushreg	%rsi
	pushq	%rbx
	.seh_pushreg	%rbx
	call	___chkstk_ms
	subq	%rax, %rsp
	.seh_stackalloc	65592
	.seh_endprologue
	cmpl	%r9d, %r8d
	movq	%rcx, %rbx
	movq	%rdx, %rdi
	movl	%r8d, %r12d
	movl	%r9d, %esi
	jb	.L118
	movl	(%rdx), %ebp
	leal	-1(%r9), %edx
	movl	$65536, %r8d
	leaq	48(%rsp), %rcx
	movzbl	%dl, %edx
	call	memset
	movl	%esi, %r11d
	subl	$1, %r11d
	je	.L119
	xorl	%eax, %eax
	.p2align 4,,10
.L113:
	movzwl	(%rdi,%rax), %edx
	movb	%al, 48(%rsp,%rdx)
	addq	$1, %rax
	cmpl	%eax, %r11d
	ja	.L113
.L112:
	leal	-4(%rsi), %r9d
	movl	%r12d, %r8d
	xorl	%edx, %edx
	leal	-3(%rsi), %eax
	shrl	$2, %r9d
	subl	%esi, %r8d
	leal	-2(%rsi), %r10d
	movslq	%eax, %r14
	negq	%r9
	movl	%eax, 44(%rsp)
	leaq	-1(%r14), %r15
	salq	$2, %r9
	leaq	(%rdi,%r14), %r13
	jmp	.L117
	.p2align 4,,10
.L130:
	movl	%r10d, %eax
	subl	%ecx, %eax
	cmpl	%r10d, %ecx
	je	.L129
.L114:
	addl	%eax, %edx
	cmpl	%r8d, %edx
	ja	.L118
.L117:
	leal	(%rdx,%r10), %eax
	movzwl	(%rbx,%rax), %eax
	movzbl	48(%rsp,%rax), %ecx
	cmpl	%r11d, %ecx
	jne	.L130
	movl	%r11d, %eax
	addl	%eax, %edx
	cmpl	%r8d, %edx
	jbe	.L117
.L118:
	xorl	%eax, %eax
	jmp	.L128
	.p2align 4,,10
.L129:
	movl	%edx, %ecx
	movl	$1, %eax
	leaq	(%rbx,%rcx), %r12
	cmpl	(%r12), %ebp
	jne	.L114
	movl	44(%rsp), %esi
	testl	%esi, %esi
	jle	.L124
	movl	(%r12,%r15), %esi
	cmpl	%esi, (%rdi,%r15)
	jne	.L114
	addq	%r14, %rcx
	xorl	%eax, %eax
	addq	%rbx, %rcx
	jmp	.L116
	.p2align 4,,10
.L132:
	movl	-5(%r13,%rax), %esi
	subq	$4, %rax
	cmpl	-1(%rcx,%rax), %esi
	jne	.L131
.L116:
	cmpq	%rax, %r9
	jne	.L132
.L124:
	movq	%r12, %rax
.L128:
	addq	$65592, %rsp
	popq	%rbx
	popq	%rsi
	popq	%rdi
	popq	%rbp
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	ret
	.p2align 4,,10
.L131:
	movl	$1, %eax
	jmp	.L114
.L119:
	xorl	%r11d, %r11d
	jmp	.L112
*/

} //if (cbTarget<777)

		} else { // if ( cbPattern<=NeedleThreshold2vs4swampLITE )

// Swampwalker_BAILOUT heuristic order 4 (Needle should be bigger than 4) [
// Needle: 1234567890qwertyuiopasdfghjklzxcv            PRIMALposition=01 PRIMALlength=33  '1234567890qwertyuiopasdfghjklzxcv'
// Needle: vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv            PRIMALposition=29 PRIMALlength=04  'vvvv'
// Needle: vvvvvvvvvvBOOMSHAKALAKAvvvvvvvvvv            PRIMALposition=08 PRIMALlength=20  'vvvBOOMSHAKALAKAvvvv'
// Needle: Trollland                                    PRIMALposition=01 PRIMALlength=09  'Trollland'
// Needle: Swampwalker                                  PRIMALposition=01 PRIMALlength=11  'Swampwalker'
// Needle: licenselessness                              PRIMALposition=01 PRIMALlength=15  'licenselessness'
// Needle: alfalfa                                      PRIMALposition=02 PRIMALlength=06  'lfalfa'
// Needle: Sandokan                                     PRIMALposition=01 PRIMALlength=08  'Sandokan'
// Needle: shazamish                                    PRIMALposition=01 PRIMALlength=09  'shazamish'
// Needle: Simplicius Simplicissimus                    PRIMALposition=06 PRIMALlength=20  'icius Simplicissimus'
// Needle: domilliaquadringenquattuorquinquagintillion  PRIMALposition=01 PRIMALlength=32  'domilliaquadringenquattuorquinqu'
// Needle: boom-boom                                    PRIMALposition=02 PRIMALlength=08  'oom-boom'
// Needle: vvvvv                                        PRIMALposition=01 PRIMALlength=04  'vvvv'
// Needle: 12345                                        PRIMALposition=01 PRIMALlength=05  '12345'
// Needle: likey-likey                                  PRIMALposition=03 PRIMALlength=09  'key-likey'
// Needle: BOOOOOM                                      PRIMALposition=03 PRIMALlength=05  'OOOOM'
// Needle: aaaaaBOOOOOM                                 PRIMALposition=02 PRIMALlength=09  'aaaaBOOOO'
// Needle: BOOOOOMaaaaa                                 PRIMALposition=03 PRIMALlength=09  'OOOOMaaaa'
PRIMALlength=0;
for (i=0+(1); i < cbPattern-((4)-1)+(1)-(1); i++) { // -(1) because the last BB (Building-Block) order 4 has no counterpart(s)
	FoundAtPosition = cbPattern - ((4)-1) + 1;
	PRIMALpositionCANDIDATE=i;
	while ( PRIMALpositionCANDIDATE <= (FoundAtPosition-1) ) {
		j = PRIMALpositionCANDIDATE + 1;
		while ( j <= (FoundAtPosition-1) ) {
			if ( *(uint32_t *)(pbPattern+PRIMALpositionCANDIDATE-(1)) == *(uint32_t *)(pbPattern+j-(1)) ) FoundAtPosition = j;
			j++;
		}
		PRIMALpositionCANDIDATE++;
	}
	PRIMALlengthCANDIDATE = (FoundAtPosition-1)-i+1 +((4)-1);
	if (PRIMALlengthCANDIDATE >= PRIMALlength) {PRIMALposition=i; PRIMALlength = PRIMALlengthCANDIDATE;}
	if (cbPattern-i+1 <= PRIMALlength) break;
	if (PRIMALlength > 128) break; // Bail Out for 129[+]
}
// Swampwalker_BAILOUT heuristic order 4 (Needle should be bigger than 4) ]

// Swampwalker_BAILOUT heuristic order 2 (Needle should be bigger than 2) [
// Needle: 1234567890qwertyuiopasdfghjklzxcv            PRIMALposition=01 PRIMALlength=33  '1234567890qwertyuiopasdfghjklzxcv'
// Needle: vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv            PRIMALposition=31 PRIMALlength=02  'vv'
// Needle: vvvvvvvvvvBOOMSHAKALAKAvvvvvvvvvv            PRIMALposition=09 PRIMALlength=13  'vvBOOMSHAKALA'
// Needle: Trollland                                    PRIMALposition=05 PRIMALlength=05  'lland'
// Needle: Swampwalker                                  PRIMALposition=03 PRIMALlength=09  'ampwalker'
// Needle: licenselessness                              PRIMALposition=01 PRIMALlength=13  'licenselessne'
// Needle: alfalfa                                      PRIMALposition=04 PRIMALlength=04  'alfa'
// Needle: Sandokan                                     PRIMALposition=01 PRIMALlength=07  'Sandoka'
// Needle: shazamish                                    PRIMALposition=02 PRIMALlength=08  'hazamish'
// Needle: Simplicius Simplicissimus                    PRIMALposition=08 PRIMALlength=15  'ius Simplicissi'
// Needle: domilliaquadringenquattuorquinquagintillion  PRIMALposition=01 PRIMALlength=19  'domilliaquadringenq'
// Needle: DODO                                         PRIMALposition=02 PRIMALlength=03  'ODO'
// Needle: DODOD                                        PRIMALposition=03 PRIMALlength=03  'DOD'
// Needle: aaaDODO                                      PRIMALposition=02 PRIMALlength=05  'aaDOD'
// Needle: aaaDODOD                                     PRIMALposition=02 PRIMALlength=05  'aaDOD'
// Needle: DODOaaa                                      PRIMALposition=02 PRIMALlength=05  'ODOaa'
// Needle: DODODaaa                                     PRIMALposition=03 PRIMALlength=05  'DODaa'
/*
PRIMALlength=0;
for (i=0+(1); i < cbPattern-2+1+(1)-(1); i++) { // -(1) because the last BB order 2 has no counterpart(s)
    FoundAtPosition = cbPattern;
    PRIMALpositionCANDIDATE=i;
    while ( PRIMALpositionCANDIDATE <= (FoundAtPosition-1) ) {
        j = PRIMALpositionCANDIDATE + 1;
        while ( j <= (FoundAtPosition-1) ) {
            if ( *(unsigned short *)(pbPattern+PRIMALpositionCANDIDATE-(1)) == *(unsigned short *)(pbPattern+j-(1)) ) FoundAtPosition = j;
            j++;
        }
        PRIMALpositionCANDIDATE++;
    }
    PRIMALlengthCANDIDATE = (FoundAtPosition-1)-i+(2);
    if (PRIMALlengthCANDIDATE >= PRIMALlength) {PRIMALposition=i; PRIMALlength = PRIMALlengthCANDIDATE;}
}
*/
// Swampwalker_BAILOUT heuristic order 2 (Needle should be bigger than 2) ]

/*
Legend:
'[]' points to BB forming left or right boundary;
'{}' points to BB being searched for;
'()' position of duplicate and new right boundary;

                       00000000011111111112222222222333
                       12345678901234567890123456789012
Example #1 for Needle: 1234567890qwertyuiopasdfghjklzxcv  NewNeedle = '1234567890qwertyuiopasdfghjklzxcv'
Example #2 for Needle: vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv  NewNeedle = 'vv'
Example #3 for Needle: vvvvvvvvvvBOOMSHAKALAKAvvvvvvvvvv  NewNeedle = 'vvBOOMSHAKALA'


     PRIMALlength=00; FoundAtPosition=33; 
Step 01_00: {}[12]34567890qwertyuiopasdfghjklzxc[v?] ! For position #01 the initial boundaries are PRIMALpositionCANDIDATE=LeftBoundary=01, RightBoundary=FoundAtPosition-1, the CANDIDATE PRIMAL string length is RightBoundary-LeftBoundary+(2)=(33-1)-01+(2)=33 !
Step 01_01: [{12}]34567890qwertyuiopasdfghjklzxc[v?] ! Searching for '12', FoundAtPosition = 33, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(33-1)-01+(2)=33 ! 
Step 01_02: [1{2]3}4567890qwertyuiopasdfghjklzxc[v?] ! Searching for '23', FoundAtPosition = 33, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(33-1)-01+(2)=33 ! 
...
Step 01_30: [12]34567890qwertyuiopasdfghjkl{zx}c[v?] ! Searching for 'zx', FoundAtPosition = 33, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(33-1)-01+(2)=33 ! 
Step 01_31: [12]34567890qwertyuiopasdfghjklz{xc}[v?] ! Searching for 'xc', FoundAtPosition = 33, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(33-1)-01+(2)=33 ! 
     if (PRIMALlengthCANDIDATE >= PRIMALlength) {PRIMALposition=PRIMALpositionCANDIDATE; PRIMALlength = PRIMALlengthCANDIDATE;}
Step 02_00: {}1[23]4567890qwertyuiopasdfghjklzxc[v?] ! For position #02 the initial boundaries are PRIMALpositionCANDIDATE=LeftBoundary=02, RightBoundary=FoundAtPosition-1, the CANDIDATE PRIMAL string length is RightBoundary-LeftBoundary+(2)=(33-1)-02+(2)=32 !
Step 02_01: 1[{23}]4567890qwertyuiopasdfghjklzxc[v?] ! Searching for '23', FoundAtPosition = 33, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(33-1)-02+(2)=32 ! 
Step 02_02: 1[2{3]4}567890qwertyuiopasdfghjklzxc[v?] ! Searching for '34', FoundAtPosition = 33, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(33-1)-02+(2)=32 ! 
...
Step 02_29: 1[23]4567890qwertyuiopasdfghjkl{zx}c[v?] ! Searching for 'zx', FoundAtPosition = 33, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(33-1)-02+(2)=32 ! 
Step 02_30: 1[23]4567890qwertyuiopasdfghjklz{xc}[v?] ! Searching for 'xc', FoundAtPosition = 33, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(33-1)-02+(2)=32 ! 
     if (PRIMALlengthCANDIDATE >= PRIMALlength) {PRIMALposition=PRIMALpositionCANDIDATE; PRIMALlength = PRIMALlengthCANDIDATE;}
...
Step 31_00: {}1234567890qwertyuiopasdfghjklz[xc][v?] ! For position #31 the initial boundaries are PRIMALpositionCANDIDATE=LeftBoundary=31, RightBoundary=FoundAtPosition-1, the CANDIDATE PRIMAL string length is RightBoundary-LeftBoundary+(2)=(33-1)-31+(2)=03 !
Step 31_01: 1234567890qwertyuiopasdfghjklz[{xc}][v?] ! Searching for 'xc', FoundAtPosition = 33, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(33-1)-31+(2)=03 ! 
     if (PRIMALlengthCANDIDATE >= PRIMALlength) {PRIMALposition=PRIMALpositionCANDIDATE; PRIMALlength = PRIMALlengthCANDIDATE;}
     Result:
     PRIMALposition=01 PRIMALlength=33, NewNeedle = '1234567890qwertyuiopasdfghjklzxcv'


     PRIMALlength=00; FoundAtPosition=33; 
Step 01_00: {}[vv]vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv[v?] ! For position #01 the initial boundaries are PRIMALpositionCANDIDATE=LeftBoundary=01, RightBoundary=FoundAtPosition-1, the CANDIDATE PRIMAL string length is RightBoundary-LeftBoundary+(2)=(33-1)-01+(2)=33 !
Step 01_01: [{v(v}]v)vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv  ! Searching for 'vv', FoundAtPosition = 02, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(02-1)-01+(2)=02 ! 
     if (PRIMALlengthCANDIDATE >= PRIMALlength) {PRIMALposition=PRIMALpositionCANDIDATE; PRIMALlength = PRIMALlengthCANDIDATE;}
Step 02_00: {}v[vv]vvvvvvvvvvvvvvvvvvvvvvvvvvvvv[v?] ! For position #02 the initial boundaries are PRIMALpositionCANDIDATE=LeftBoundary=02, RightBoundary=FoundAtPosition-1, the CANDIDATE PRIMAL string length is RightBoundary-LeftBoundary+(2)=(33-1)-02+(2)=32 !
Step 02_01: v[{v(v}]v)vvvvvvvvvvvvvvvvvvvvvvvvvvvvv  ! Searching for 'vv', FoundAtPosition = 03, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(03-1)-02+(2)=02 ! 
     if (PRIMALlengthCANDIDATE >= PRIMALlength) {PRIMALposition=PRIMALpositionCANDIDATE; PRIMALlength = PRIMALlengthCANDIDATE;}
...
Step 31_00: {}vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv[vv][v?] ! For position #31 the initial boundaries are PRIMALpositionCANDIDATE=LeftBoundary=31, RightBoundary=FoundAtPosition-1, the CANDIDATE PRIMAL string length is RightBoundary-LeftBoundary+(2)=(33-1)-31+(2)=03 !
Step 31_01: vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv[{v(v}]v)  ! Searching for 'vv', FoundAtPosition = 32, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(32-1)-31+(2)=02 ! 
     if (PRIMALlengthCANDIDATE >= PRIMALlength) {PRIMALposition=PRIMALpositionCANDIDATE; PRIMALlength = PRIMALlengthCANDIDATE;}
     Result:
     PRIMALposition=31 PRIMALlength=02, NewNeedle = 'vv'


     PRIMALlength=00; FoundAtPosition=33; 
Step 01_00: {}[vv]vvvvvvvvBOOMSHAKALAKAvvvvvvvvv[v?] ! For position #01 the initial boundaries are PRIMALpositionCANDIDATE=LeftBoundary=01, RightBoundary=FoundAtPosition-1, the CANDIDATE PRIMAL string length is RightBoundary-LeftBoundary+(2)=(33-1)-01+(2)=33 !
Step 01_01: [{v(v}]v)vvvvvvvBOOMSHAKALAKAvvvvvvvvvv  ! Searching for 'vv', FoundAtPosition = 02, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(02-1)-01+(2)=02 ! 
     if (PRIMALlengthCANDIDATE >= PRIMALlength) {PRIMALposition=PRIMALpositionCANDIDATE; PRIMALlength = PRIMALlengthCANDIDATE;}
Step 02_00: {}v[vv]vvvvvvvBOOMSHAKALAKAvvvvvvvvv[v?] ! For position #02 the initial boundaries are PRIMALpositionCANDIDATE=LeftBoundary=02, RightBoundary=FoundAtPosition-1, the CANDIDATE PRIMAL string length is RightBoundary-LeftBoundary+(2)=(33-1)-02+(2)=32 !
Step 02_01: v[{v(v}]v)vvvvvvBOOMSHAKALAKAvvvvvvvvvv  ! Searching for 'vv', FoundAtPosition = 03, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(03-1)-02+(2)=02 ! 
     if (PRIMALlengthCANDIDATE >= PRIMALlength) {PRIMALposition=PRIMALpositionCANDIDATE; PRIMALlength = PRIMALlengthCANDIDATE;}
...
Step 09_00: {}vvvvvvvv[vv]BOOMSHAKALAKAvvvvvvvvv[v?] ! For position #09 the initial boundaries are PRIMALpositionCANDIDATE=LeftBoundary=09, RightBoundary=FoundAtPosition-1, the CANDIDATE PRIMAL string length is RightBoundary-LeftBoundary+(2)=(33-1)-09+(2)=25 !
Step 09_01: vvvvvvvv[{vv}]BOOMSHAKALAKA(vv)vvvvvvvv  ! Searching for 'vv', FoundAtPosition = 24, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(24-1)-09+(2)=16 ! 
Step 09_02: vvvvvvvv[v{v]B}OOMSHAKALAKA[vv]vvvvvvvv  ! Searching for 'vB', FoundAtPosition = 24, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(24-1)-09+(2)=16 ! 
Step 09_03: vvvvvvvv[vv]{BO}OMSHAKALAKA[vv]vvvvvvvv  ! Searching for 'BO', FoundAtPosition = 24, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(24-1)-09+(2)=16 ! 
Step 09_04: vvvvvvvv[vv]B{OO}MSHAKALAKA[vv]vvvvvvvv  ! Searching for 'OO', FoundAtPosition = 24, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(24-1)-09+(2)=16 ! 
Step 09_05: vvvvvvvv[vv]BO{OM}SHAKALAKA[vv]vvvvvvvv  ! Searching for 'OM', FoundAtPosition = 24, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(24-1)-09+(2)=16 ! 
Step 09_06: vvvvvvvv[vv]BOO{MS}HAKALAKA[vv]vvvvvvvv  ! Searching for 'MS', FoundAtPosition = 24, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(24-1)-09+(2)=16 ! 
Step 09_07: vvvvvvvv[vv]BOOM{SH}AKALAKA[vv]vvvvvvvv  ! Searching for 'SH', FoundAtPosition = 24, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(24-1)-09+(2)=16 ! 
Step 09_08: vvvvvvvv[vv]BOOMS{HA}KALAKA[vv]vvvvvvvv  ! Searching for 'HA', FoundAtPosition = 24, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(24-1)-09+(2)=16 ! 
Step 09_09: vvvvvvvv[vv]BOOMSH{AK}AL(AK)Avvvvvvvvvv  ! Searching for 'AK', FoundAtPosition = 21, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(21-1)-09+(2)=13 ! 
Step 09_10: vvvvvvvv[vv]BOOMSHA{KA}L[AK]Avvvvvvvvvv  ! Searching for 'KA', FoundAtPosition = 21, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(21-1)-09+(2)=13 ! 
Step 09_11: vvvvvvvv[vv]BOOMSHAK{AL}[AK]Avvvvvvvvvv  ! Searching for 'AL', FoundAtPosition = 21, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(21-1)-09+(2)=13 ! 
Step 09_12: vvvvvvvv[vv]BOOMSHAKA{L[A}K]Avvvvvvvvvv  ! Searching for 'LA', FoundAtPosition = 21, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(21-1)-09+(2)=13 ! 
     if (PRIMALlengthCANDIDATE >= PRIMALlength) {PRIMALposition=PRIMALpositionCANDIDATE; PRIMALlength = PRIMALlengthCANDIDATE;}
...
Step 31_00: {}vvvvvvvv[vv]BOOMSHAKALAKAvvvvvvvvv[v?] ! For position #31 the initial boundaries are PRIMALpositionCANDIDATE=LeftBoundary=31, RightBoundary=FoundAtPosition-1, the CANDIDATE PRIMAL string length is RightBoundary-LeftBoundary+(2)=(33-1)-31+(2)=03 !
Step 31_01: vvvvvvvvvvBOOMSHAKALAKAvvvvvvv[{v(v}]v)  ! Searching for 'vv', FoundAtPosition = 32, PRIMALlengthCANDIDATE=RightBoundary-LeftBoundary+(2)=(32-1)-31+(2)=02 ! 
     if (PRIMALlengthCANDIDATE >= PRIMALlength) {PRIMALposition=PRIMALpositionCANDIDATE; PRIMALlength = PRIMALlengthCANDIDATE;}
     Result:
     PRIMALposition=09 PRIMALlength=13, NewNeedle = 'vvBOOMSHAKALA'
*/

// Here we have 4 or bigger NewNeedle, apply order 2 for pbPattern[i+(PRIMALposition-1)] with length 'PRIMALlength' and compare the pbPattern[i] with length 'cbPattern':
PRIMALlengthCANDIDATE = cbPattern;
cbPattern = PRIMALlength;
pbPattern = pbPattern + (PRIMALposition-1);

// Revision 2 commented section [
/*
if (cbPattern-1 <= 255) {
// BMH Order 2 [
			ulHashPattern = *(uint32_t *)(pbPattern); // First four bytes
			for (i=0; i < 256*256; i++) {bm_Horspool_Order2[i]= cbPattern-1;} // cbPattern-(Order-1) for Horspool; 'memset' if not optimized
			for (i=0; i < cbPattern-1; i++) bm_Horspool_Order2[*(unsigned short *)(pbPattern+i)]=i; // Rightmost appearance/position is needed
			i=0;
			while (i <= cbTarget-cbPattern) { 
				Gulliver = bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+cbPattern-1-1]];
				if ( Gulliver != cbPattern-1 ) { // CASE #2: if equal means the pair (char order 2) is not found i.e. Gulliver remains intact, skip the whole pattern and fall back (Order-1) chars i.e. one char for Order 2
				if ( Gulliver == cbPattern-2 ) { // CASE #1: means the pair (char order 2) is found
					if ( *(uint32_t *)&pbTarget[i] == ulHashPattern) {
						count = cbPattern-4+1; 
						while ( count > 0 && *(uint32_t *)(pbPattern+count-1) == *(uint32_t *)(&pbTarget[i]+(count-1)) )
							count = count-4;
// If we miss to hit then no need to compare the original: Needle
if ( count <= 0 ) {
// I have to add out-of-range checks...
// i-(PRIMALposition-1) >= 0
// &pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4
// i-(PRIMALposition-1)+(count-1) >= 0
// &pbTarget[i-(PRIMALposition-1)+(count-1)] <= pbTargetMax - 4

// "FIX" from 2014-Apr-27:
// Because (count-1) is negative, above fours are reduced to next twos:
// i-(PRIMALposition-1)+(count-1) >= 0
// &pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4
	// The line below is BUGGY:
	//if ( (i-(PRIMALposition-1) >= 0) && (&pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4) && (&pbTarget[i-(PRIMALposition-1)+(count-1)] <= pbTargetMax - 4) ) {
	// The line below is NOT OKAY, in fact so stupid, grrr, not a blunder, not carelessness, but overconfidence in writing "on the fly":
	//if ( ((signed int)(i-(PRIMALposition-1)+(count-1)) >= 0) && (&pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4) ) {
// FIX from 2016-Aug-10 (two times failed to do simple boundary checks, pfu):
	if ( ((signed int)(i-(PRIMALposition-1)) >= 0) && (&pbTarget[i-(PRIMALposition-1)]+((PRIMALlengthCANDIDATE-4+1)-1) <= pbTargetMax - 4) ) {
		if ( *(uint32_t *)&pbTarget[i-(PRIMALposition-1)] == *(uint32_t *)(pbPattern-(PRIMALposition-1))) { // This fast check ensures not missing a match (for remainder) when going under 0 in loop below:
			count = PRIMALlengthCANDIDATE-4+1; 
			while ( count > 0 && *(uint32_t *)(pbPattern-(PRIMALposition-1)+count-1) == *(uint32_t *)(&pbTarget[i-(PRIMALposition-1)]+(count-1)) )
				count = count-4;
			if ( count <= 0 ) return(pbTarget+i-(PRIMALposition-1));	
		}
	}
}
					}
					Gulliver = 1;
				} else
					Gulliver = cbPattern - Gulliver - 2; // CASE #3: the pair is found and not as suffix i.e. rightmost position
				}
				i = i + Gulliver;
				//GlobalI++; // Comment it, it is only for stats.
			}
			return(NULL);
// BMH Order 2 ]
} else {
			// BMH order 2, needle should be >=4:
			ulHashPattern = *(uint32_t *)(pbPattern); // First four bytes
			for (i=0; i < 256*256; i++) {bm_Horspool_Order2[i]=0;}
			for (i=0; i < cbPattern-1; i++) bm_Horspool_Order2[*(unsigned short *)(pbPattern+i)]=1;
			i=0;
			while (i <= cbTarget-cbPattern) {
				Gulliver = 1; // 'Gulliver' is the skip
				if ( bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+cbPattern-1-1]] != 0 ) {
					if ( bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+cbPattern-1-1-2]] == 0 ) Gulliver = cbPattern-(2-1)-2; else {
						if ( *(uint32_t *)&pbTarget[i] == ulHashPattern) { // This fast check ensures not missing a match (for remainder) when going under 0 in loop below:
							count = cbPattern-4+1; 
							while ( count > 0 && *(uint32_t *)(pbPattern+count-1) == *(uint32_t *)(&pbTarget[i]+(count-1)) )
								count = count-4;
// If we miss to hit then no need to compare the original: Needle
if ( count <= 0 ) {
// I have to add out-of-range checks...
// i-(PRIMALposition-1) >= 0
// &pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4
// i-(PRIMALposition-1)+(count-1) >= 0
// &pbTarget[i-(PRIMALposition-1)+(count-1)] <= pbTargetMax - 4

// "FIX" from 2014-Apr-27:
// Because (count-1) is negative, above fours are reduced to next twos:
// i-(PRIMALposition-1)+(count-1) >= 0
// &pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4
	// The line below is BUGGY:
	//if ( (i-(PRIMALposition-1) >= 0) && (&pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4) && (&pbTarget[i-(PRIMALposition-1)+(count-1)] <= pbTargetMax - 4) ) {
	// The line below is NOT OKAY, in fact so stupid, grrr, not a blunder, not carelessness, but overconfidence in writing "on the fly":
	//if ( ((signed int)(i-(PRIMALposition-1)+(count-1)) >= 0) && (&pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4) ) {
// FIX from 2016-Aug-10 (two times failed to do simple boundary checks, pfu):
	if ( ((signed int)(i-(PRIMALposition-1)) >= 0) && (&pbTarget[i-(PRIMALposition-1)]+((PRIMALlengthCANDIDATE-4+1)-1) <= pbTargetMax - 4) ) {
		if ( *(uint32_t *)&pbTarget[i-(PRIMALposition-1)] == *(uint32_t *)(pbPattern-(PRIMALposition-1))) { // This fast check ensures not missing a match (for remainder) when going under 0 in loop below:
			count = PRIMALlengthCANDIDATE-4+1; 
			while ( count > 0 && *(uint32_t *)(pbPattern-(PRIMALposition-1)+count-1) == *(uint32_t *)(&pbTarget[i-(PRIMALposition-1)]+(count-1)) )
				count = count-4;
			if ( count <= 0 ) return(pbTarget+i-(PRIMALposition-1));	
		}
	}
}
						}
					}
				} else Gulliver = cbPattern-(2-1);
				i = i + Gulliver;
				//GlobalI++; // Comment it, it is only for stats.
			}
			return(NULL);
}
*/
// Revision 2 commented section ]

		if ( cbPattern<=NeedleThreshold2vs4swampLITE ) { 

			// BMH order 2, needle should be >=4:
			ulHashPattern = *(uint32_t *)(pbPattern); // First four bytes
			for (i=0; i < 256*256; i++) {bm_Horspool_Order2[i]=0;}
			// Above line is translated by Intel as:
//  0044c 41 b8 00 00 01 
//        00               mov r8d, 65536                         
//  00452 44 89 5c 24 20   mov DWORD PTR [32+rsp], r11d           
//  00457 44 89 54 24 60   mov DWORD PTR [96+rsp], r10d           
//  0045c e8 fc ff ff ff   call _intel_fast_memset                
			for (i=0; i < cbPattern-1; i++) bm_Horspool_Order2[*(unsigned short *)(pbPattern+i)]=1;
			i=0;
			while (i <= cbTarget-cbPattern) {
				Gulliver = 1; // 'Gulliver' is the skip
				if ( bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+cbPattern-1-1]] != 0 ) {
					if ( bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+cbPattern-1-1-2]] == 0 ) Gulliver = cbPattern-(2-1)-2; else {
						if ( *(uint32_t *)&pbTarget[i] == ulHashPattern) { // This fast check ensures not missing a match (for remainder) when going under 0 in loop below:
							count = cbPattern-4+1; 
							while ( count > 0 && *(uint32_t *)(pbPattern+count-1) == *(uint32_t *)(&pbTarget[i]+(count-1)) )
								count = count-4;

	if (cbPattern != PRIMALlengthCANDIDATE) { // No need of same comparison when Needle and NewNeedle are equal!
// If we miss to hit then no need to compare the original: Needle
if ( count <= 0 ) {
// I have to add out-of-range checks...
// i-(PRIMALposition-1) >= 0
// &pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4
// i-(PRIMALposition-1)+(count-1) >= 0
// &pbTarget[i-(PRIMALposition-1)+(count-1)] <= pbTargetMax - 4

// "FIX" from 2014-Apr-27:
// Because (count-1) is negative, above fours are reduced to next twos:
// i-(PRIMALposition-1)+(count-1) >= 0
// &pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4
	// The line below is BUGGY:
	//if ( (i-(PRIMALposition-1) >= 0) && (&pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4) && (&pbTarget[i-(PRIMALposition-1)+(count-1)] <= pbTargetMax - 4) ) {
	// The line below is NOT OKAY, in fact so stupid, grrr, not a blunder, not carelessness, but overconfidence in writing "on the fly":
	//if ( ((signed int)(i-(PRIMALposition-1)+(count-1)) >= 0) && (&pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4) ) {
// FIX from 2016-Aug-10 (two times failed to do simple boundary checks, pfu):
	if ( ((signed int)(i-(PRIMALposition-1)) >= 0) && (&pbTarget[i-(PRIMALposition-1)]+((PRIMALlengthCANDIDATE-4+1)-1) <= pbTargetMax - 4) ) {
		if ( *(uint32_t *)&pbTarget[i-(PRIMALposition-1)] == *(uint32_t *)(pbPattern-(PRIMALposition-1))) { // This fast check ensures not missing a match (for remainder) when going under 0 in loop below:
			count = PRIMALlengthCANDIDATE-4+1; 
			while ( count > 0 && *(uint32_t *)(pbPattern-(PRIMALposition-1)+count-1) == *(uint32_t *)(&pbTarget[i-(PRIMALposition-1)]+(count-1)) )
				count = count-4;
			if ( count <= 0 ) return(pbTarget+i-(PRIMALposition-1));	
		}
	}
}
	} else { //if (cbPattern != PRIMALlengthCANDIDATE)
							if ( count <= 0 ) return(pbTarget+i);
	}
						}
					}
				} else Gulliver = cbPattern-(2-1);
				i = i + Gulliver;
				//GlobalI++; // Comment it, it is only for stats.
			}
			return(NULL);

		} else { // if ( cbPattern<=NeedleThreshold2vs4swampLITE )

			// BMH pseudo-order 4, needle should be >=8+2:
			ulHashPattern = *(uint32_t *)(pbPattern); // First four bytes
			for (i=0; i < 256*256; i++) {bm_Horspool_Order2[i]=0;}
			// In line below we "hash" 4bytes to 2bytes i.e. 16bit table, how to compute TOTAL number of BBs, 'cbPattern - Order + 1' is the number of BBs for text 'cbPattern' bytes long, for example, for cbPattern=11 'fastest fox' and Order=4 we have BBs = 11-4+1=8:
			//"fast"
			//"aste"
			//"stes"
			//"test"
			//"est "
			//"st f"
			//"t fo"
			//" fox"
			//for (i=0; i < cbPattern-4+1; i++) bm_Horspool_Order2[( *(unsigned short *)(pbPattern+i+0) + *(unsigned short *)(pbPattern+i+2) ) & ( (1<<16)-1 )]=1;
			//for (i=0; i < cbPattern-4+1; i++) bm_Horspool_Order2[( (*(uint32_t *)(pbPattern+i+0)>>16)+(*(uint32_t *)(pbPattern+i+0)&0xFFFF) ) & ( (1<<16)-1 )]=1;
			// Above line is replaced by next one with better hashing:
			for (i=0; i < cbPattern-4+1; i++) bm_Horspool_Order2[( (*(uint32_t *)(pbPattern+i+0)>>(16-1))+(*(uint32_t *)(pbPattern+i+0)&0xFFFF) ) & ( (1<<16)-1 )]=1;
			i=0;
			while (i <= cbTarget-cbPattern) {
				Gulliver = 1;
				//if ( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2]>>16)+(*(uint32_t *)&pbTarget[i+cbPattern-1-1-2]&0xFFFF) ) & ( (1<<16)-1 )] != 0 ) { // DWORD #1
				// Above line is replaced by next one with better hashing:
				if ( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2]>>(16-1))+(*(uint32_t *)&pbTarget[i+cbPattern-1-1-2]&0xFFFF) ) & ( (1<<16)-1 )] != 0 ) { // DWORD #1
					//if ( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-4]>>16)+(*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-4]&0xFFFF) ) & ( (1<<16)-1 )] == 0 ) Gulliver = cbPattern-(2-1)-2-4; else {
					// Above line is replaced in order to strengthen the skip by checking the middle DWORD,if the two DWORDs are 'ab' and 'cd' i.e. [2x][2a][2b][2c][2d] then the middle DWORD is 'bc'.
					// The respective offsets (backwards) are: -10/-8/-6/-4 for 'xa'/'ab'/'bc'/'cd'.
					//if ( ( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-6]>>16)+(*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-6]&0xFFFF) ) & ( (1<<16)-1 )] ) + ( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-4]>>16)+(*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-4]&0xFFFF) ) & ( (1<<16)-1 )] ) + ( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-2]>>16)+(*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-2]&0xFFFF) ) & ( (1<<16)-1 )] ) < 3 ) Gulliver = cbPattern-(2-1)-2-4-2; else {
					// Above line is replaced by next one with better hashing:
					// When using (16-1) right shifting instead of 16 we will have two different pairs (if they are equal), the highest bit being lost do the job especialy for ASCII texts with no symbols in range 128-255.
					// Example for genomesque pair TT+TT being shifted by (16-1):
					// T            = 01010100
					// TT           = 01010100 01010100
					// TTTT         = 01010100 01010100 01010100 01010100
					// TTTT>>16     = 00000000 00000000 01010100 01010100
					// TTTT>>(16-1) = 00000000 00000000 10101000 10101000 <--- Due to the left shift by 1, the 8th bits of 1st and 2nd bytes are populated - usually they are 0 for English texts & 'ACGT' data.
					//if ( ( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-6]>>(16-1))+(*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-6]&0xFFFF) ) & ( (1<<16)-1 )] ) + ( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-4]>>(16-1))+(*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-4]&0xFFFF) ) & ( (1<<16)-1 )] ) + ( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-2]>>(16-1))+(*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-2]&0xFFFF) ) & ( (1<<16)-1 )] ) < 3 ) Gulliver = cbPattern-(2-1)-2-4-2; else {
					// 'Maximus' uses branched 'if', again.
					if ( \
					( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-6 +1]>>(16-1))+(*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-6 +1]&0xFFFF) ) & ( (1<<16)-1 )] ) == 0 \
					|| ( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-4 +1]>>(16-1))+(*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-4 +1]&0xFFFF) ) & ( (1<<16)-1 )] ) == 0 \
					) Gulliver = cbPattern-(2-1)-2-4-2 +1; else {
					// Above line is not optimized (several a SHR are used), we have 5 non-overlapping WORDs, or 3 overlapping WORDs, within 4 overlapping DWORDs so:
// [2x][2a][2b][2c][2d]
// DWORD #4
// [2a] (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-6]>>16) =     !SHR to be avoided! <--
// [2x] (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-6]&0xFFFF) =                        |
//     DWORD #3                                                                       |
// [2b] (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-4]>>16) =     !SHR to be avoided!   |<--
// [2a] (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-4]&0xFFFF) = ------------------------  |
//         DWORD #2                                                                      |
// [2c] (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-2]>>16) =     !SHR to be avoided!      |<--
// [2b] (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-2]&0xFFFF) = ---------------------------  |
//             DWORD #1                                                                     |
// [2d] (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-0]>>16) =                                 |
// [2c] (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-0]&0xFFFF) = ------------------------------
//
// So in order to remove 3 SHR instructions the equal extractions are:
// DWORD #4
// [2a] (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-4]&0xFFFF) =  !SHR to be avoided! <--
// [2x] (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-6]&0xFFFF) =                        |
//     DWORD #3                                                                       |
// [2b] (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-2]&0xFFFF) =  !SHR to be avoided!   |<--
// [2a] (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-4]&0xFFFF) = ------------------------  |
//         DWORD #2                                                                      |
// [2c] (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-0]&0xFFFF) =  !SHR to be avoided!      |<--
// [2b] (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-2]&0xFFFF) = ---------------------------  |
//             DWORD #1                                                                     |
// [2d] (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-0]>>16) =                                 |
// [2c] (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-0]&0xFFFF) = ------------------------------
					//if ( ( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-4]&0xFFFF)+(*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-6]&0xFFFF) ) & ( (1<<16)-1 )] ) + ( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-2]&0xFFFF)+(*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-4]&0xFFFF) ) & ( (1<<16)-1 )] ) + ( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-0]&0xFFFF)+(*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-2]&0xFFFF) ) & ( (1<<16)-1 )] ) < 3 ) Gulliver = cbPattern-(2-1)-2-6; else {
// Since the above Decumanus mumbo-jumbo (3 overlapping lookups vs 2 non-overlapping lookups) is not fast enough we go DuoDecumanus or 3x4:
// [2y][2x][2a][2b][2c][2d]
// DWORD #3
//         DWORD #2
//                 DWORD #1
					//if ( ( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-4]>>16)+(*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-4]&0xFFFF) ) & ( (1<<16)-1 )] ) + ( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-8]>>16)+(*(uint32_t *)&pbTarget[i+cbPattern-1-1-2-8]&0xFFFF) ) & ( (1<<16)-1 )] ) < 2 ) Gulliver = cbPattern-(2-1)-2-8; else {
						if ( *(uint32_t *)&pbTarget[i] == ulHashPattern) {
							// Order 4 [
						// Let's try something "outrageous" like comparing with[out] overlap BBs 4bytes long instead of 1 byte back-to-back:
						// Inhere we are using order 4, 'cbPattern - Order + 1' is the number of BBs for text 'cbPattern' bytes long, for example, for cbPattern=11 'fastest fox' and Order=4 we have BBs = 11-4+1=8:
						//0:"fast" if the comparison failed here, 'count' is 1; 'Gulliver' is cbPattern-(4-1)-7
						//1:"aste" if the comparison failed here, 'count' is 2; 'Gulliver' is cbPattern-(4-1)-6
						//2:"stes" if the comparison failed here, 'count' is 3; 'Gulliver' is cbPattern-(4-1)-5
						//3:"test" if the comparison failed here, 'count' is 4; 'Gulliver' is cbPattern-(4-1)-4
						//4:"est " if the comparison failed here, 'count' is 5; 'Gulliver' is cbPattern-(4-1)-3
						//5:"st f" if the comparison failed here, 'count' is 6; 'Gulliver' is cbPattern-(4-1)-2
						//6:"t fo" if the comparison failed here, 'count' is 7; 'Gulliver' is cbPattern-(4-1)-1
						//7:" fox" if the comparison failed here, 'count' is 8; 'Gulliver' is cbPattern-(4-1)
							count = cbPattern-4+1; 
							// Below comparison is UNIdirectional:
							while ( count > 0 && *(uint32_t *)(pbPattern+count-1) == *(uint32_t *)(&pbTarget[i]+(count-1)) )
								count = count-4;

	if (cbPattern != PRIMALlengthCANDIDATE) { // No need of same comparison when Needle and NewNeedle are equal!
// count = cbPattern-4+1 = 23-4+1 = 20
// boomshakalakaZZZZZZ[ZZZZ] 20
// boomshakalakaZZ[ZZZZ]ZZZZ 20-4
// boomshakala[kaZZ]ZZZZZZZZ 20-8 = 12
// boomsha[kala]kaZZZZZZZZZZ 20-12 = 8
// boo[msha]kalakaZZZZZZZZZZ 20-16 = 4

// If we miss to hit then no need to compare the original: Needle
if ( count <= 0 ) {
// I have to add out-of-range checks...
// i-(PRIMALposition-1) >= 0
// &pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4
// i-(PRIMALposition-1)+(count-1) >= 0
// &pbTarget[i-(PRIMALposition-1)+(count-1)] <= pbTargetMax - 4

// "FIX" from 2014-Apr-27:
// Because (count-1) is negative, above fours are reduced to next twos:
// i-(PRIMALposition-1)+(count-1) >= 0
// &pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4
	// The line below is BUGGY:
	//if ( (i-(PRIMALposition-1) >= 0) && (&pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4) && (&pbTarget[i-(PRIMALposition-1)+(count-1)] <= pbTargetMax - 4) ) {
	// The line below is NOT OKAY, in fact so stupid, grrr, not a blunder, not carelessness, but overconfidence in writing "on the fly":
	//if ( ((signed int)(i-(PRIMALposition-1)+(count-1)) >= 0) && (&pbTarget[i-(PRIMALposition-1)] <= pbTargetMax - 4) ) {
// FIX from 2016-Aug-10 (two times failed to do simple boundary checks, pfu):
	if ( ((signed int)(i-(PRIMALposition-1)) >= 0) && (&pbTarget[i-(PRIMALposition-1)]+((PRIMALlengthCANDIDATE-4+1)-1) <= pbTargetMax - 4) ) {
		if ( *(uint32_t *)&pbTarget[i-(PRIMALposition-1)] == *(uint32_t *)(pbPattern-(PRIMALposition-1))) { // This fast check ensures not missing a match (for remainder) when going under 0 in loop below:
			count = PRIMALlengthCANDIDATE-4+1; 
			while ( count > 0 && *(uint32_t *)(pbPattern-(PRIMALposition-1)+count-1) == *(uint32_t *)(&pbTarget[i-(PRIMALposition-1)]+(count-1)) )
				count = count-4;
			if ( count <= 0 ) return(pbTarget+i-(PRIMALposition-1));	
		}
	}
}
	} else { //if (cbPattern != PRIMALlengthCANDIDATE)
							if ( count <= 0 ) return(pbTarget+i);
	}

							// In order to avoid only-left or only-right WCS the memcmp should be done as left-to-right and right-to-left AT THE SAME TIME.
							// Below comparison is BIdirectional. It pays off when needle is 8+++ long:
//							for (count = cbPattern-4+1; count > 0; count = count-4) {
//								if ( *(uint32_t *)(pbPattern+count-1) != *(uint32_t *)(&pbTarget[i]+(count-1)) ) {break;};
//								if ( *(uint32_t *)(pbPattern+(cbPattern-4+1)-count) != *(uint32_t *)(&pbTarget[i]+(cbPattern-4+1)-count) ) {count = (cbPattern-4+1)-count +(1); break;} // +(1) because two lookups are implemented as one, also no danger of 'count' being 0 because of the fast check outwith the 'while': if ( *(uint32_t *)&pbTarget[i] == ulHashPattern)
//							}
//							if ( count <= 0 ) return(pbTarget+i);
								// Checking the order 2 pairs in mismatched DWORD, all the 3:
								//if ( bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+count-1]] == 0 ) Gulliver = count; // 1 or bigger, as it should
								//if ( bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+count-1+1]] == 0 ) Gulliver = count+1; // 1 or bigger, as it should
								//if ( bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+count-1+1+1]] == 0 ) Gulliver = count+1+1; // 1 or bigger, as it should
							//	if ( bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+count-1]] + bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+count-1+1]] + bm_Horspool_Order2[*(unsigned short *)&pbTarget[i+count-1+1+1]] < 3 ) Gulliver = count; // 1 or bigger, as it should, THE MIN(count,count+1,count+1+1)
								// Above compound 'if' guarantees not that Gulliver > 1, an example:
								// Needle:    fastest tax
								// Window: ...fastast tax...
								// After matching ' tax' vs ' tax' and 'fast' vs 'fast' the mismathced DWORD is 'test' vs 'tast':
								// 'tast' when factorized down to order 2 yields: 'ta','as','st' - all the three when summed give 1+1+1=3 i.e. Gulliver remains 1.
								// Roughly speaking, this attempt maybe has its place in worst-case scenarios but not in English text and even not in ACGT data, that's why I commented it in original 'Shockeroo'.
								//if ( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+count-1]>>16)+(*(uint32_t *)&pbTarget[i+count-1]&0xFFFF) ) & ( (1<<16)-1 )] == 0 ) Gulliver = count; // 1 or bigger, as it should
								// Above line is replaced by next one with better hashing:
//								if ( bm_Horspool_Order2[( (*(uint32_t *)&pbTarget[i+count-1]>>(16-1))+(*(uint32_t *)&pbTarget[i+count-1]&0xFFFF) ) & ( (1<<16)-1 )] == 0 ) Gulliver = count; // 1 or bigger, as it should
							// Order 4 ]
						}
					}
				} else Gulliver = cbPattern-(2-1)-2; // -2 because we check the 4 rightmost bytes not 2.
				i = i + Gulliver;
				//GlobalI++; // Comment it, it is only for stats.
			}
			return(NULL);

		} // if ( cbPattern<=NeedleThreshold2vs4swampLITE )
		} // if ( cbPattern<=NeedleThreshold2vs4swampLITE )
	} //if ( cbPattern<4 )
}
/*
// For short needles, and mainly haystacks, 'Doublet' is quite effective. Consider it or 'Quadruplet'.
// Fixed version from 2012-Feb-27.
// Caution: For better speed the case 'if (cbPattern==1)' was removed, so Pattern must be longer than 1 char.
char * Railgun_Doublet (char * pbTarget, char * pbPattern, uint32_t cbTarget, uint32_t cbPattern)
{
	char * pbTargetMax = pbTarget + cbTarget;
	register uint32_t ulHashPattern;
	uint32_t ulHashTarget, count, countSTATIC;

	if (cbPattern > cbTarget) return(NULL);

	countSTATIC = cbPattern-2;

	pbTarget = pbTarget+cbPattern;
	ulHashPattern = (*(uint16_t *)(pbPattern));

	for ( ;; ) {
		if ( ulHashPattern == (*(uint16_t *)(pbTarget-cbPattern)) ) {
			count = countSTATIC;
			while ( count && *(char *)(pbPattern+2+(countSTATIC-count)) == *(char *)(pbTarget-cbPattern+2+(countSTATIC-count)) ) {
				count--;
			}
			if ( count == 0 ) return((pbTarget-cbPattern));
		}
		pbTarget++;
		if (pbTarget > pbTargetMax) return(NULL);
	}
}
*/
/*
; mark_description "Intel(R) C++ Intel(R) 64 Compiler XE for applications running on Intel(R) 64, Version 15.0.0.108 Build 20140";
; mark_description "-O3 -QxSSE2 -D_N_XMM -D_N_prefetch_4096 -D_N_Branchfull -D_N_HIGH_PRIORITY -FA";

_TEXT	SEGMENT      'CODE'
;	COMDAT Railgun_Trolldom
; mark_begin;
       ALIGN     16
	PUBLIC Railgun_Trolldom
Railgun_Trolldom	PROC 
; parameter 1: rcx
; parameter 2: rdx
; parameter 3: r8d
; parameter 4: r9d
.B11.1::                        ; Preds .B11.0
        push      rbx                                           ;3712.1
        push      rsi                                           ;3712.1
        push      rdi                                           ;3712.1
        push      r12                                           ;3712.1
        push      r13                                           ;3712.1
        push      r14                                           ;3712.1
        push      r15                                           ;3712.1
        push      rbp                                           ;3712.1
        mov       eax, 65640                                    ;3712.1
        call      __chkstk                                      ;3712.1
        sub       rsp, 65640                                    ;3712.1
        mov       ebx, r8d                                      ;3712.1
        mov       r13d, ebx                                     ;3713.23
        mov       rdi, rcx                                      ;3712.1
        mov       esi, r9d                                      ;3712.1
        mov       r12, rdx                                      ;3712.1
        cmp       esi, ebx                                      ;3738.18
        lea       r11, QWORD PTR [r13+rdi]                      ;3713.23
        ja        .B11.9        ; Prob 28%                      ;3738.18
                                ; LOE rsi rdi r11 r12 r13 ebx r9d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.2::                        ; Preds .B11.1
        cmp       esi, 4                                        ;3740.17
        jae       .B11.18       ; Prob 50%                      ;3740.17
                                ; LOE rsi rdi r11 r12 r13 ebx r9d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.3::                        ; Preds .B11.2
        movsx     edx, BYTE PTR [r12]                           ;3744.23
        lea       ebp, DWORD PTR [-1+rsi]                       ;3744.74
        shl       edx, 8                                        ;3744.45
        lea       rbx, QWORD PTR [rdi+rsi]                      ;3743.21
        lea       rax, QWORD PTR [-2+rsi+rdi]                   ;
        lea       rcx, QWORD PTR [-3+rsi+rdi]                   ;
        movsx     edi, BYTE PTR [r12+rbp]                       ;3744.53
        add       edx, edi                                      ;3744.53
        mov       r8d, edx                                      ;3750.32
        shr       r8d, 8                                        ;3750.32
        movsx     rbp, r8b                                      ;3750.32
        cmp       esi, 3                                        ;3745.19
        je        .B11.11       ; Prob 50%                      ;3745.19
                                ; LOE rax rcx rbx r11 r12 edx ebp xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.5::                        ; Preds .B11.3 .B11.8
        movsx     ecx, BYTE PTR [rax]                           ;3760.48
        shl       ecx, 8                                        ;3760.53
        movsx     esi, BYTE PTR [-1+rbx]                        ;3760.70
        add       ecx, esi                                      ;3760.70
        cmp       edx, ecx                                      ;3760.70
        je        .B11.41       ; Prob 20%                      ;3760.70
                                ; LOE rax rbx r11 edx ebp esi xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.6::                        ; Preds .B11.5
        cmp       ebp, esi                                      ;3761.48
        je        .B11.8        ; Prob 50%                      ;3761.48
                                ; LOE rax rbx r11 edx ebp xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.7::                        ; Preds .B11.6
        inc       rax                                           ;3761.53
        inc       rbx                                           ;3761.53
                                ; LOE rax rbx r11 edx ebp xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.8::                        ; Preds .B11.7 .B11.6
        inc       rbx                                           ;3762.4
        inc       rax                                           ;3762.4
        cmp       rbx, r11                                      ;3763.19
        jbe       .B11.5        ; Prob 80%                      ;3763.19
                                ; LOE rax rbx r11 edx ebp xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.9::                        ; Preds .B11.1 .B11.8
        xor       eax, eax                                      ;3763.38
        add       rsp, 65640                                    ;3763.38
        pop       rbp                                           ;3763.38
        pop       r15                                           ;3763.38
        pop       r14                                           ;3763.38
        pop       r13                                           ;3763.38
        pop       r12                                           ;3763.38
        pop       rdi                                           ;3763.38
        pop       rsi                                           ;3763.38
        pop       rbx                                           ;3763.38
        ret                                                     ;3763.38
                                ; LOE
.B11.11::                       ; Preds .B11.3 .B11.16
        movsx     edi, BYTE PTR [rcx]                           ;3747.49
        shl       edi, 8                                        ;3747.54
        movsx     esi, BYTE PTR [-1+rbx]                        ;3747.71
        add       edi, esi                                      ;3747.71
        movsx     r8d, BYTE PTR [-2+rbx]                        ;3748.56
        cmp       edx, edi                                      ;3747.71
        jne       .B11.13       ; Prob 50%                      ;3747.71
                                ; LOE rcx rbx r11 r12 edx ebp r8d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.12::                       ; Preds .B11.11
        cmp       r8b, BYTE PTR [1+r12]                         ;3748.56
        je        .B11.185      ; Prob 20%                      ;3748.56
                                ; LOE rcx rbx r11 r12 edx ebp r8d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.13::                       ; Preds .B11.11 .B11.12
        cmp       ebp, r8d                                      ;3750.49
        je        .B11.16       ; Prob 50%                      ;3750.49
                                ; LOE rcx rbx r11 r12 edx ebp xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.14::                       ; Preds .B11.13
        inc       rbx                                           ;3751.6
        inc       rcx                                           ;3751.6
        cmp       bpl, BYTE PTR [-2+rbx]                        ;3752.50
        je        .B11.16       ; Prob 50%                      ;3752.50
                                ; LOE rcx rbx r11 r12 edx ebp xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.15::                       ; Preds .B11.14
        inc       rcx                                           ;3752.55
        inc       rbx                                           ;3752.55
                                ; LOE rcx rbx r11 r12 edx ebp xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.16::                       ; Preds .B11.15 .B11.14 .B11.13
        inc       rbx                                           ;3754.5
        inc       rcx                                           ;3754.5
        cmp       rbx, r11                                      ;3755.20
        jbe       .B11.11       ; Prob 80%                      ;3755.20
                                ; LOE rcx rbx r11 r12 edx ebp xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.17::                       ; Preds .B11.16
        xor       eax, eax                                      ;3755.39
        add       rsp, 65640                                    ;3755.39
        pop       rbp                                           ;3755.39
        pop       r15                                           ;3755.39
        pop       r14                                           ;3755.39
        pop       r13                                           ;3755.39
        pop       r12                                           ;3755.39
        pop       rdi                                           ;3755.39
        pop       rsi                                           ;3755.39
        pop       rbx                                           ;3755.39
        ret                                                     ;3755.39
                                ; LOE
.B11.18::                       ; Preds .B11.2
        cmp       esi, 19                                       ;3766.19
        ja        .B11.82       ; Prob 50%                      ;3766.19
                                ; LOE rsi rdi r11 r12 r13 ebx r9d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.19::                       ; Preds .B11.18
        cmp       ebx, 777                                      ;3774.14
        jb        .B11.67       ; Prob 50%                      ;3774.14
                                ; LOE rsi rdi r11 r12 ebx xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.20::                       ; Preds .B11.19
        mov       ebp, DWORD PTR [r12]                          ;3825.33
        cmp       ebx, 77777                                    ;3819.21
        jae       .B11.45       ; Prob 50%                      ;3819.21
                                ; LOE rsi rdi r12 ebx ebp xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.21::                       ; Preds .B11.20
        xor       edx, edx                                      ;3827.38
        lea       rcx, QWORD PTR [32+rsp]                       ;3827.38
        mov       r8d, 8192                                     ;3827.38
        call      _intel_fast_memset                            ;3827.38
                                ; LOE rsi rdi r12 ebx ebp xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.22::                       ; Preds .B11.21
        mov       r10d, esi                                     ;3829.30
        dec       r10d                                          ;3829.30
        je        .B11.31       ; Prob 50%                      ;3829.30
                                ; LOE rsi rdi r12 ebx ebp r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.23::                       ; Preds .B11.22
        mov       r9d, r10d                                     ;3829.4
        xor       r8d, r8d                                      ;3829.4
        shr       r9d, 4                                        ;3829.4
        mov       eax, 1                                        ;3829.4
        xor       edx, edx                                      ;
        test      r9d, r9d                                      ;3829.4
        jbe       .B11.27       ; Prob 15%                      ;3829.4
                                ; LOE rdx rsi rdi r12 eax ebx ebp r8d r9d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.25::                       ; Preds .B11.23 .B11.25
        mov       r11d, 1                                       ;3829.211
        mov       r15d, 1                                       ;3829.211
        lea       r13d, DWORD PTR [1+rdx]                       ;3829.197
        inc       r8d                                           ;3829.4
        movzx     ecx, WORD PTR [rdx+r12]                       ;3829.197
        mov       eax, ecx                                      ;3829.166
        and       ecx, 7                                        ;3829.211
        shl       r11d, cl                                      ;3829.211
        movzx     ecx, WORD PTR [r13+r12]                       ;3829.151
        mov       r14d, ecx                                     ;3829.166
        and       ecx, 7                                        ;3829.211
        lea       r13d, DWORD PTR [3+rdx]                       ;3829.197
        shl       r15d, cl                                      ;3829.211
        shr       eax, 3                                        ;3829.166
        lea       ecx, DWORD PTR [2+rdx]                        ;3829.197
        shr       r14d, 3                                       ;3829.166
        movzx     ecx, WORD PTR [rcx+r12]                       ;3829.151
        or        BYTE PTR [32+rsp+rax], r11b                   ;3829.211
        mov       eax, ecx                                      ;3829.166
        shr       eax, 3                                        ;3829.166
        and       ecx, 7                                        ;3829.211
        mov       r11d, 1                                       ;3829.211
        shl       r11d, cl                                      ;3829.211
        or        BYTE PTR [32+rsp+r14], r15b                   ;3829.211
        lea       r14d, DWORD PTR [4+rdx]                       ;3829.197
        movzx     ecx, WORD PTR [r13+r12]                       ;3829.151
        or        BYTE PTR [32+rsp+rax], r11b                   ;3829.211
        mov       eax, ecx                                      ;3829.166
        shr       eax, 3                                        ;3829.166
        and       ecx, 7                                        ;3829.211
        mov       r11d, 1                                       ;3829.211
        shl       r11d, cl                                      ;3829.211
        or        BYTE PTR [32+rsp+rax], r11b                   ;3829.211
        lea       r11d, DWORD PTR [5+rdx]                       ;3829.197
        mov       eax, 1                                        ;3829.211
        movzx     ecx, WORD PTR [r14+r12]                       ;3829.151
        mov       r15d, ecx                                     ;3829.166
        and       ecx, 7                                        ;3829.211
        mov       r14d, 1                                       ;3829.211
        shl       eax, cl                                       ;3829.211
        movzx     ecx, WORD PTR [r11+r12]                       ;3829.151
        mov       r13d, ecx                                     ;3829.166
        and       ecx, 7                                        ;3829.211
        mov       r11d, 1                                       ;3829.211
        shl       r14d, cl                                      ;3829.211
        shr       r15d, 3                                       ;3829.166
        lea       ecx, DWORD PTR [6+rdx]                        ;3829.197
        shr       r13d, 3                                       ;3829.166
        movzx     ecx, WORD PTR [rcx+r12]                       ;3829.151
        or        BYTE PTR [32+rsp+r15], al                     ;3829.211
        mov       eax, ecx                                      ;3829.166
        shr       eax, 3                                        ;3829.166
        lea       r15d, DWORD PTR [7+rdx]                       ;3829.197
        and       ecx, 7                                        ;3829.211
        shl       r11d, cl                                      ;3829.211
        or        BYTE PTR [32+rsp+r13], r14b                   ;3829.211
        lea       r13d, DWORD PTR [8+rdx]                       ;3829.197
        movzx     ecx, WORD PTR [r15+r12]                       ;3829.151
        mov       r15d, 1                                       ;3829.211
        or        BYTE PTR [32+rsp+rax], r11b                   ;3829.211
        mov       eax, ecx                                      ;3829.166
        shr       eax, 3                                        ;3829.166
        and       ecx, 7                                        ;3829.211
        mov       r11d, 1                                       ;3829.211
        shl       r11d, cl                                      ;3829.211
        or        BYTE PTR [32+rsp+rax], r11b                   ;3829.211
        lea       r11d, DWORD PTR [9+rdx]                       ;3829.197
        mov       eax, 1                                        ;3829.211
        movzx     ecx, WORD PTR [r13+r12]                       ;3829.151
        mov       r14d, ecx                                     ;3829.166
        and       ecx, 7                                        ;3829.211
        shl       eax, cl                                       ;3829.211
        movzx     ecx, WORD PTR [r11+r12]                       ;3829.151
        mov       r13d, ecx                                     ;3829.166
        and       ecx, 7                                        ;3829.211
        mov       r11d, 1                                       ;3829.211
        shl       r15d, cl                                      ;3829.211
        shr       r14d, 3                                       ;3829.166
        lea       ecx, DWORD PTR [10+rdx]                       ;3829.197
        shr       r13d, 3                                       ;3829.166
        movzx     ecx, WORD PTR [rcx+r12]                       ;3829.151
        or        BYTE PTR [32+rsp+r14], al                     ;3829.211
        mov       eax, ecx                                      ;3829.166
        shr       eax, 3                                        ;3829.166
        lea       r14d, DWORD PTR [11+rdx]                      ;3829.197
        and       ecx, 7                                        ;3829.211
        shl       r11d, cl                                      ;3829.211
        or        BYTE PTR [32+rsp+r13], r15b                   ;3829.211
        lea       r13d, DWORD PTR [12+rdx]                      ;3829.197
        movzx     ecx, WORD PTR [r14+r12]                       ;3829.151
        mov       r14d, 1                                       ;3829.211
        or        BYTE PTR [32+rsp+rax], r11b                   ;3829.211
        mov       eax, ecx                                      ;3829.166
        shr       eax, 3                                        ;3829.166
        and       ecx, 7                                        ;3829.211
        mov       r11d, 1                                       ;3829.211
        shl       r11d, cl                                      ;3829.211
        or        BYTE PTR [32+rsp+rax], r11b                   ;3829.211
        lea       r11d, DWORD PTR [13+rdx]                      ;3829.197
        mov       eax, 1                                        ;3829.211
        movzx     ecx, WORD PTR [r13+r12]                       ;3829.151
        mov       r15d, ecx                                     ;3829.166
        and       ecx, 7                                        ;3829.211
        shl       eax, cl                                       ;3829.211
        movzx     ecx, WORD PTR [r11+r12]                       ;3829.151
        mov       r13d, ecx                                     ;3829.166
        and       ecx, 7                                        ;3829.211
        mov       r11d, 1                                       ;3829.211
        shl       r14d, cl                                      ;3829.211
        shr       r15d, 3                                       ;3829.166
        lea       ecx, DWORD PTR [14+rdx]                       ;3829.197
        shr       r13d, 3                                       ;3829.166
        movzx     ecx, WORD PTR [rcx+r12]                       ;3829.151
        or        BYTE PTR [32+rsp+r15], al                     ;3829.211
        mov       eax, ecx                                      ;3829.166
        shr       eax, 3                                        ;3829.166
        lea       r15d, DWORD PTR [15+rdx]                      ;3829.197
        and       ecx, 7                                        ;3829.211
        add       edx, 16                                       ;3829.4
        shl       r11d, cl                                      ;3829.211
        or        BYTE PTR [32+rsp+r13], r14b                   ;3829.211
        movzx     ecx, WORD PTR [r15+r12]                       ;3829.151
        DB        144                                           ;3829.211
        or        BYTE PTR [32+rsp+rax], r11b                   ;3829.211
        mov       eax, ecx                                      ;3829.166
        shr       eax, 3                                        ;3829.166
        and       ecx, 7                                        ;3829.211
        mov       r11d, 1                                       ;3829.211
        shl       r11d, cl                                      ;3829.211
        or        BYTE PTR [32+rsp+rax], r11b                   ;3829.211
        cmp       r8d, r9d                                      ;3829.4
        jb        .B11.25       ; Prob 99%                      ;3829.4
                                ; LOE rdx rsi rdi r12 ebx ebp r8d r9d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.26::                       ; Preds .B11.25
        shl       r8d, 4                                        ;3829.38
        lea       eax, DWORD PTR [1+r8]                         ;3829.4
                                ; LOE rsi rdi r12 eax ebx ebp r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.27::                       ; Preds .B11.26 .B11.23
        dec       eax                                           ;3829.38
        mov       edx, eax                                      ;3829.4
        cmp       eax, r10d                                     ;3829.4
        jae       .B11.31       ; Prob 15%                      ;3829.4
                                ; LOE rdx rsi rdi r12 eax ebx ebp r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.29::                       ; Preds .B11.27 .B11.29
        movzx     ecx, WORD PTR [r12+rdx]                       ;3829.151
        mov       edx, ecx                                      ;3829.166
        shr       edx, 3                                        ;3829.166
        and       ecx, 7                                        ;3829.211
        mov       r8d, 1                                        ;3829.211
        shl       r8d, cl                                       ;3829.211
        inc       eax                                           ;3829.4
        or        BYTE PTR [32+rsp+rdx], r8b                    ;3829.211
        mov       edx, eax                                      ;3829.4
        cmp       eax, r10d                                     ;3829.4
        jb        .B11.29       ; Prob 93%                      ;3829.4
                                ; LOE rdx rsi rdi r12 eax ebx ebp r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.31::                       ; Preds .B11.22 .B11.29 .B11.27
        xor       edx, edx                                      ;3830.4
        lea       r8d, DWORD PTR [-3+rsi]                       ;3836.192
        movsxd    r9, r8d                                       ;3838.8
        lea       r10d, DWORD PTR [-1+rsi]                      ;3844.33
        sub       ebx, esi                                      ;3831.16
                                ; LOE rsi rdi r9 r12 edx ebx ebp r8d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.32::                       ; Preds .B11.31 .B11.43
        mov       r14d, 1                                       ;3834.145
        lea       eax, DWORD PTR [-2+rsi+rdx]                   ;3834.141
        movzx     ecx, WORD PTR [rax+rdi]                       ;3834.59
        mov       r11d, ecx                                     ;3834.87
        shr       r11d, 3                                       ;3834.87
        and       ecx, 7                                        ;3834.145
        shl       r14d, cl                                      ;3834.145
        movzx     r13d, BYTE PTR [32+rsp+r11]                   ;3834.12
        test      r13d, r14d                                    ;3834.145
        je        .B11.42       ; Prob 50%                      ;3834.156
                                ; LOE rsi rdi r9 r12 edx ebx ebp r8d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.33::                       ; Preds .B11.32
        mov       r14d, 1                                       ;3836.150
        lea       eax, DWORD PTR [-4+rsi+rdx]                   ;3836.146
        movzx     ecx, WORD PTR [rax+rdi]                       ;3836.60
        mov       r11d, ecx                                     ;3836.90
        shr       r11d, 3                                       ;3836.90
        and       ecx, 7                                        ;3836.150
        shl       r14d, cl                                      ;3836.150
        movzx     r13d, BYTE PTR [32+rsp+r11]                   ;3836.13
        test      r13d, r14d                                    ;3836.150
        jne       .B11.35       ; Prob 50%                      ;3836.161
                                ; LOE rsi rdi r9 r12 edx ebx ebp r8d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.34::                       ; Preds .B11.33
        mov       ecx, r8d                                      ;3836.165
        jmp       .B11.43       ; Prob 100%                     ;3836.165
                                ; LOE rsi rdi r9 r12 edx ecx ebx ebp r8d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.35::                       ; Preds .B11.33
        mov       eax, edx                                      ;3837.26
        mov       ecx, 1                                        ;3832.5
        add       rax, rdi                                      ;3837.26
        cmp       ebp, DWORD PTR [rax]                          ;3837.41
        jne       .B11.43       ; Prob 50%                      ;3837.41
                                ; LOE rax rsi rdi r9 r12 edx ecx ebx ebp r8d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.36::                       ; Preds .B11.35
        mov       r11, r9                                       ;3838.8
        test      r9, r9                                        ;3839.24
        jle       .B11.41       ; Prob 2%                       ;3839.24
                                ; LOE rax rsi rdi r9 r11 r12 edx ecx ebx ebp r8d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.38::                       ; Preds .B11.36 .B11.39
        mov       r13d, DWORD PTR [-1+r11+r12]                  ;3839.59
        cmp       r13d, DWORD PTR [-1+r11+rax]                  ;3839.80
        jne       .B11.43       ; Prob 20%                      ;3839.80
                                ; LOE rax rsi rdi r9 r11 r12 edx ecx ebx ebp r8d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.39::                       ; Preds .B11.38
        add       r11, -4                                       ;3840.23
        test      r11, r11                                      ;3839.24
        jg        .B11.38       ; Prob 82%                      ;3839.24
                                ; LOE rax rsi rdi r9 r11 r12 edx ecx ebx ebp r8d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.41::                       ; Preds .B11.5 .B11.36 .B11.39
        add       rsp, 65640                                    ;3841.33
        pop       rbp                                           ;3841.33
        pop       r15                                           ;3841.33
        pop       r14                                           ;3841.33
        pop       r13                                           ;3841.33
        pop       r12                                           ;3841.33
        pop       rdi                                           ;3841.33
        pop       rsi                                           ;3841.33
        pop       rbx                                           ;3841.33
        ret                                                     ;3841.33
                                ; LOE
.B11.42::                       ; Preds .B11.32
        mov       ecx, r10d                                     ;3844.12
                                ; LOE rsi rdi r9 r12 edx ecx ebx ebp r8d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.43::                       ; Preds .B11.38 .B11.34 .B11.35 .B11.42
        add       edx, ecx                                      ;3845.13
        cmp       edx, ebx                                      ;3831.25
        jbe       .B11.32       ; Prob 82%                      ;3831.25
                                ; LOE rsi rdi r9 r12 edx ebx ebp r8d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.44::                       ; Preds .B11.43
        xor       eax, eax                                      ;3848.10
        add       rsp, 65640                                    ;3848.10
        pop       rbp                                           ;3848.10
        pop       r15                                           ;3848.10
        pop       r14                                           ;3848.10
        pop       r13                                           ;3848.10
        pop       r12                                           ;3848.10
        pop       rdi                                           ;3848.10
        pop       rsi                                           ;3848.10
        pop       rbx                                           ;3848.10
        ret                                                     ;3848.10
                                ; LOE
.B11.45::                       ; Preds .B11.20
        xor       edx, edx                                      ;3853.33
        lea       rcx, QWORD PTR [32+rsp]                       ;3853.33
        mov       r8d, 65536                                    ;3853.33
        call      _intel_fast_memset                            ;3853.33
                                ; LOE rsi rdi r12 ebx ebp xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.46::                       ; Preds .B11.45
        mov       r9d, esi                                      ;3854.28
        dec       r9d                                           ;3854.28
        je        .B11.53       ; Prob 50%                      ;3854.28
                                ; LOE rsi rdi r12 ebx ebp r9d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.47::                       ; Preds .B11.46
        mov       edx, r9d                                      ;3854.4
        mov       ecx, 1                                        ;3854.4
        shr       edx, 1                                        ;3854.4
        xor       eax, eax                                      ;3854.4
        test      edx, edx                                      ;3854.4
        jbe       .B11.51       ; Prob 15%                      ;3854.4
                                ; LOE rsi rdi r12 eax edx ecx ebx ebp r9d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.48::                       ; Preds .B11.47
        mov       cl, 1                                         ;3854.36
                                ; LOE rsi rdi r12 eax edx ebx ebp r9d cl xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.49::                       ; Preds .B11.49 .B11.48
        lea       r8d, DWORD PTR [rax+rax]                      ;3854.75
        lea       r11d, DWORD PTR [1+rax+rax]                   ;3854.75
        inc       eax                                           ;3854.4
        cmp       eax, edx                                      ;3854.4
        movzx     r10d, WORD PTR [r8+r12]                       ;3854.75
        movzx     r13d, WORD PTR [r11+r12]                      ;3854.75
        mov       BYTE PTR [32+rsp+r10], cl                     ;3854.36
        mov       BYTE PTR [32+rsp+r13], cl                     ;3854.36
        jb        .B11.49       ; Prob 64%                      ;3854.4
                                ; LOE rsi rdi r12 eax edx ebx ebp r9d cl xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.50::                       ; Preds .B11.49
        lea       ecx, DWORD PTR [1+rax+rax]                    ;3854.4
                                ; LOE rsi rdi r12 ecx ebx ebp r9d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.51::                       ; Preds .B11.50 .B11.47
        dec       ecx                                           ;3854.36
        cmp       ecx, r9d                                      ;3854.4
        jae       .B11.53       ; Prob 15%                      ;3854.4
                                ; LOE rcx rsi rdi r12 ebx ebp r9d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.52::                       ; Preds .B11.51
        movzx     eax, WORD PTR [rcx+r12]                       ;3854.75
        mov       BYTE PTR [32+rsp+rax], 1                      ;3854.36
                                ; LOE rsi rdi r12 ebx ebp r9d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.53::                       ; Preds .B11.46 .B11.51 .B11.52
        xor       edx, edx                                      ;3855.4
        lea       ecx, DWORD PTR [-3+rsi]                       ;3859.113
        movsxd    r8, ecx                                       ;3861.8
        sub       ebx, esi                                      ;3856.16
                                ; LOE rsi rdi r8 r12 edx ecx ebx ebp r9d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.54::                       ; Preds .B11.53 .B11.65
        lea       eax, DWORD PTR [-2+rsi+rdx]                   ;3858.72
        movzx     r10d, WORD PTR [rax+rdi]                      ;3858.49
        cmp       BYTE PTR [32+rsp+r10], 0                      ;3858.79
        je        .B11.64       ; Prob 50%                      ;3858.79
                                ; LOE rsi rdi r8 r12 edx ecx ebx ebp r9d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.55::                       ; Preds .B11.54
        lea       eax, DWORD PTR [-4+rsi+rdx]                   ;3859.75
        movzx     r10d, WORD PTR [rax+rdi]                      ;3859.50
        cmp       BYTE PTR [32+rsp+r10], 0                      ;3859.82
        jne       .B11.57       ; Prob 50%                      ;3859.82
                                ; LOE rsi rdi r8 r12 edx ecx ebx ebp r9d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.56::                       ; Preds .B11.55
        mov       r10d, ecx                                     ;3859.86
        jmp       .B11.65       ; Prob 100%                     ;3859.86
                                ; LOE rsi rdi r8 r12 edx ecx ebx ebp r9d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.57::                       ; Preds .B11.55
        mov       eax, edx                                      ;3860.26
        mov       r10d, 1                                       ;3857.5
        add       rax, rdi                                      ;3860.26
        cmp       ebp, DWORD PTR [rax]                          ;3860.41
        jne       .B11.65       ; Prob 50%                      ;3860.41
                                ; LOE rax rsi rdi r8 r12 edx ecx ebx ebp r9d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.58::                       ; Preds .B11.57
        mov       r11, r8                                       ;3861.8
        test      r8, r8                                        ;3862.24
        jle       .B11.63       ; Prob 2%                       ;3862.24
                                ; LOE rax rsi rdi r8 r11 r12 edx ecx ebx ebp r9d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.60::                       ; Preds .B11.58 .B11.61
        mov       r13d, DWORD PTR [-1+r11+r12]                  ;3862.59
        cmp       r13d, DWORD PTR [-1+r11+rax]                  ;3862.80
        jne       .B11.65       ; Prob 20%                      ;3862.80
                                ; LOE rax rsi rdi r8 r11 r12 edx ecx ebx ebp r9d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.61::                       ; Preds .B11.60
        add       r11, -4                                       ;3863.23
        test      r11, r11                                      ;3862.24
        jg        .B11.60       ; Prob 82%                      ;3862.24
                                ; LOE rax rsi rdi r8 r11 r12 edx ecx ebx ebp r9d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.63::                       ; Preds .B11.58 .B11.61
        add       rsp, 65640                                    ;3864.33
        pop       rbp                                           ;3864.33
        pop       r15                                           ;3864.33
        pop       r14                                           ;3864.33
        pop       r13                                           ;3864.33
        pop       r12                                           ;3864.33
        pop       rdi                                           ;3864.33
        pop       rsi                                           ;3864.33
        pop       rbx                                           ;3864.33
        ret                                                     ;3864.33
                                ; LOE
.B11.64::                       ; Preds .B11.54
        mov       r10d, r9d                                     ;3867.12
                                ; LOE rsi rdi r8 r12 edx ecx ebx ebp r9d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.65::                       ; Preds .B11.60 .B11.56 .B11.57 .B11.64
        add       edx, r10d                                     ;3868.13
        cmp       edx, ebx                                      ;3856.25
        jbe       .B11.54       ; Prob 82%                      ;3856.25
                                ; LOE rsi rdi r8 r12 edx ecx ebx ebp r9d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.66::                       ; Preds .B11.65
        xor       eax, eax                                      ;3871.10
        add       rsp, 65640                                    ;3871.10
        pop       rbp                                           ;3871.10
        pop       r15                                           ;3871.10
        pop       r14                                           ;3871.10
        pop       r13                                           ;3871.10
        pop       r12                                           ;3871.10
        pop       rdi                                           ;3871.10
        pop       rsi                                           ;3871.10
        pop       rbx                                           ;3871.10
        ret                                                     ;3871.10
                                ; LOE
.B11.67::                       ; Preds .B11.19
        mov       ebx, DWORD PTR [r12]                          ;3777.43
        lea       ecx, DWORD PTR [-1+rsi]                       ;3797.28
        movzx     ebp, bl                                       ;3781.5
        mov       edx, esi                                      ;3776.20
        mov       r8d, ebp                                      ;3782.30
        mov       r9d, ebp                                      ;3783.30
        mov       r10d, ebp                                     ;3784.30
        shl       r8d, 8                                        ;3782.30
        add       rdi, rdx                                      ;3776.20
        shl       r9d, 16                                       ;3783.30
        shl       r10d, 24                                      ;3784.30
        mov       QWORD PTR [65608+rsp], r12                    ;3797.28
                                ; LOE rdx rsi rdi r11 ecx ebx ebp r8d r9d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.68::                       ; Preds .B11.80 .B11.67
        mov       rax, rdi                                      ;3789.36
        xor       r12d, r12d                                    ;3788.2
        sub       rax, rdx                                      ;3789.36
        mov       r13d, DWORD PTR [rax]                         ;3789.45
        cmp       ebx, r13d                                     ;3791.31
        jne       .B11.77       ; Prob 50%                      ;3791.31
                                ; LOE rax rdx rsi rdi r11 ecx ebx ebp r8d r9d r10d r12d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.69::                       ; Preds .B11.68
        mov       r14d, ecx                                     ;
        mov       r13d, ecx                                     ;3797.10
        movsxd    r15, ecx                                      ;
        neg       r14d                                          ;
        neg       r15                                           ;
        add       r14d, esi                                     ;
        test      ecx, ecx                                      ;3798.18
        je        .B11.76       ; Prob 10%                      ;3798.18
                                ; LOE rax rdx rsi rdi r11 r14 r15 ecx ebx ebp r8d r9d r10d r12d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.70::                       ; Preds .B11.69
        mov       DWORD PTR [65624+rsp], esi                    ;
        mov       QWORD PTR [40+rsp], rdx                       ;
        mov       QWORD PTR [32+rsp], r11                       ;
        mov       rsi, QWORD PTR [65608+rsp]                    ;
                                ; LOE rax rsi rdi r14 r15 ecx ebx ebp r8d r9d r10d r12d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.71::                       ; Preds .B11.74 .B11.70
        movsx     edx, BYTE PTR [r15+rdi]                       ;3798.88
        cmp       dl, BYTE PTR [r14+rsi]                        ;3798.88
        jne       .B11.169      ; Prob 20%                      ;3798.88
                                ; LOE rax rsi rdi r15 edx ecx ebx ebp r8d r9d r10d r12d r13d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.72::                       ; Preds .B11.71
        lea       r11d, DWORD PTR [r12+r13]                     ;3799.46
        cmp       ecx, r11d                                     ;3799.46
        jne       .B11.74       ; Prob 50%                      ;3799.46
                                ; LOE rax rsi rdi r15 edx ecx ebx ebp r8d r9d r10d r12d r13d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.73::                       ; Preds .B11.72
        cmp       ebp, edx                                      ;3799.94
        lea       r11d, DWORD PTR [1+r12]                       ;3799.94
        cmovne    r12d, r11d                                    ;3799.94
                                ; LOE rax rsi rdi r15 ecx ebx ebp r8d r9d r10d r12d r13d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.74::                       ; Preds .B11.73 .B11.72
        inc       r14d                                          ;3800.16
        inc       r15                                           ;3800.16
        dec       r13d                                          ;3800.16
        jne       .B11.71       ; Prob 82%                      ;3798.18
                                ; LOE rax rsi rdi r14 r15 ecx ebx ebp r8d r9d r10d r12d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.76::                       ; Preds .B11.69 .B11.74
        add       rsp, 65640                                    ;3802.44
        pop       rbp                                           ;3802.44
        pop       r15                                           ;3802.44
        pop       r14                                           ;3802.44
        pop       r13                                           ;3802.44
        pop       r12                                           ;3802.44
        pop       rdi                                           ;3802.44
        pop       rsi                                           ;3802.44
        pop       rbx                                           ;3802.44
        ret                                                     ;3802.44
                                ; LOE
.B11.77::                       ; Preds .B11.68
        mov       eax, r13d                                     ;3804.43
        and       eax, 65280                                    ;3804.43
        cmp       r8d, eax                                      ;3804.43
        je        .B11.80       ; Prob 50%                      ;3804.43
                                ; LOE rdx rsi rdi r11 ecx ebx ebp r8d r9d r10d r12d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.78::                       ; Preds .B11.77
        mov       eax, r13d                                     ;3806.48
        mov       r12d, 1                                       ;3805.10
        and       eax, 16711680                                 ;3806.48
        cmp       r9d, eax                                      ;3806.48
        je        .B11.80       ; Prob 50%                      ;3806.48
                                ; LOE rdx rsi rdi r11 ecx ebx ebp r8d r9d r10d r12d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.79::                       ; Preds .B11.78
        and       r13d, -16777216                               ;3808.53
        mov       eax, 3                                        ;3808.67
        mov       r12d, 2                                       ;3808.67
        cmp       r10d, r13d                                    ;3808.67
        cmovne    r12d, eax                                     ;3808.67
                                ; LOE rdx rsi rdi r11 ecx ebx ebp r8d r9d r10d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.80::                       ; Preds .B11.169 .B11.79 .B11.78 .B11.77
        inc       r12d                                          ;3813.2
        add       rdi, r12                                      ;3815.13
        cmp       rdi, r11                                      ;3816.24
        jbe       .B11.68       ; Prob 80%                      ;3816.24
                                ; LOE rdx rsi rdi r11 ecx ebx ebp r8d r9d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.81::                       ; Preds .B11.80
        xor       eax, eax                                      ;3817.19
        add       rsp, 65640                                    ;3817.19
        pop       rbp                                           ;3817.19
        pop       r15                                           ;3817.19
        pop       r14                                           ;3817.19
        pop       r13                                           ;3817.19
        pop       r12                                           ;3817.19
        pop       rdi                                           ;3817.19
        pop       rsi                                           ;3817.19
        pop       rbx                                           ;3817.19
        ret                                                     ;3817.19
                                ; LOE
.B11.82::                       ; Preds .B11.18
        xor       ebp, ebp                                      ;4360.1
        lea       ecx, DWORD PTR [-3+rsi]                       ;4361.29
        mov       edx, 1                                        ;4361.6
        mov       r8d, -1                                       ;
        mov       r10d, esi                                     ;
        cmp       ecx, 1                                        ;4361.41
        jbe       .B11.184      ; Prob 10%                      ;4361.41
                                ; LOE rsi rdi r12 r13 edx ecx ebx ebp r8d r9d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.83::                       ; Preds .B11.82
        mov       DWORD PTR [32+rsp], ebx                       ;4364.54
        lea       eax, DWORD PTR [-2+rsi]                       ;4362.42
        mov       QWORD PTR [40+rsp], rdi                       ;4364.54
        lea       r11d, DWORD PTR [-3+rsi]                      ;4364.54
        mov       QWORD PTR [48+rsp], r13                       ;4364.54
                                ; LOE rsi r12 eax edx ecx ebp r8d r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.84::                       ; Preds .B11.96 .B11.83
        mov       r14d, edx                                     ;4363.2
        mov       ebx, eax                                      ;4362.2
        cmp       edx, r11d                                     ;4364.54
        ja        .B11.92       ; Prob 10%                      ;4364.54
                                ; LOE rsi r12 r14 eax edx ecx ebx ebp r8d r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.85::                       ; Preds .B11.84
        mov       DWORD PTR [65624+rsp], esi                    ;4365.3
        lea       edi, DWORD PTR [-3+rsi]                       ;4366.33
        mov       r13d, edx                                     ;4365.3
                                ; LOE r12 r14 eax edx ecx ebx ebp edi r8d r9d r10d r11d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.86::                       ; Preds .B11.90 .B11.85
        inc       r13d                                          ;4365.33
        mov       r15d, r13d                                    ;4365.3
        cmp       r13d, edi                                     ;4366.33
        ja        .B11.90       ; Prob 10%                      ;4366.33
                                ; LOE r12 r14 r15 eax edx ecx ebx ebp edi r8d r9d r10d r11d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.87::                       ; Preds .B11.86
        mov       esi, DWORD PTR [-1+r14+r12]                   ;4367.57
                                ; LOE r12 r15 eax edx ecx ebx ebp esi edi r8d r9d r10d r11d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.88::                       ; Preds .B11.88 .B11.87
        lea       r14d, DWORD PTR [-1+r15]                      ;4366.33
        cmp       esi, DWORD PTR [-1+r15+r12]                   ;4367.98
        jne       L26           ; Prob 50%                      ;4366.33
        mov       edi, r14d                                     ;4366.33
L26:                                                            ;
        jne       L27           ; Prob 50%                      ;4367.98
        mov       ebx, r15d                                     ;4367.98
L27:                                                            ;
        inc       r15d                                          ;4368.4
        cmp       r15d, edi                                     ;4366.33
        jbe       .B11.88       ; Prob 82%                      ;4366.33
                                ; LOE r12 r15 eax edx ecx ebx ebp esi edi r8d r9d r10d r11d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.90::                       ; Preds .B11.88 .B11.86
        mov       r14d, r13d                                    ;4370.3
        cmp       r13d, edi                                     ;4364.54
        jbe       .B11.86       ; Prob 82%                      ;4364.54
                                ; LOE r12 r14 eax edx ecx ebx ebp edi r8d r9d r10d r11d r13d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.91::                       ; Preds .B11.90
        mov       esi, DWORD PTR [65624+rsp]                    ;
                                ; LOE r12 eax edx ecx ebx ebp esi r8d r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.92::                       ; Preds .B11.91 .B11.84
        lea       ebx, DWORD PTR [3+r8+rbx]                     ;4372.2
        cmp       ebx, ebp                                      ;4373.31
        jb        .B11.95       ; Prob 50%                      ;4373.31
                                ; LOE rsi r12 eax edx ecx ebx ebp r8d r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.93::                       ; Preds .B11.92
        mov       r9d, edx                                      ;4373.46
        mov       ebp, ebx                                      ;4373.64
        cmp       ebx, r10d                                     ;4374.23
        jae       .B11.97       ; Prob 20%                      ;4374.23
                                ; LOE rsi r12 eax edx ecx ebx ebp r8d r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.94::                       ; Preds .B11.93
        cmp       ebx, 128                                      ;4375.21
        ja        .B11.97       ; Prob 20%                      ;4375.21
        jmp       .B11.96       ; Prob 100%                     ;4375.21
                                ; LOE rsi r12 eax edx ecx ebp r8d r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.95::                       ; Preds .B11.92
        cmp       ebp, r10d                                     ;4374.23
        jae       .B11.97       ; Prob 20%                      ;4374.23
                                ; LOE rsi r12 eax edx ecx ebp r8d r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.96::                       ; Preds .B11.95 .B11.94
        inc       edx                                           ;4361.46
        dec       r10d                                          ;4361.46
        dec       r8d                                           ;4361.46
        cmp       edx, ecx                                      ;4361.41
        jb        .B11.84       ; Prob 82%                      ;4361.41
                                ; LOE rsi r12 eax edx ecx ebp r8d r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.97::                       ; Preds .B11.93 .B11.95 .B11.94 .B11.96
        mov       ebx, DWORD PTR [32+rsp]                       ;
        mov       rdi, QWORD PTR [40+rsp]                       ;
        mov       r13, QWORD PTR [48+rsp]                       ;
                                ; LOE rbx rsi rdi r12 r13 ebx ebp edi r9d r13d bl bh dil r13b xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.98::                       ; Preds .B11.97
        cmp       ebp, 19                                       ;4605.19
        lea       r14d, DWORD PTR [-1+r9]                       ;4500.41
        lea       r15, QWORD PTR [r12+r14]                      ;4500.13
        mov       r10d, DWORD PTR [r15]                         ;4608.33
        ja        .B11.131      ; Prob 50%                      ;4605.19
                                ; LOE rbx rsi rdi r12 r13 r14 r15 ebx ebp edi r9d r10d r13d bl bh dil r13b xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.99::                       ; Preds .B11.184 .B11.98
        xor       edx, edx                                      ;4609.33
        lea       rcx, QWORD PTR [32+rsp]                       ;4609.33
        mov       r8d, 65536                                    ;4609.33
        mov       DWORD PTR [65568+rsp], r10d                   ;4609.33
        mov       DWORD PTR [65616+rsp], r9d                    ;4609.33
        call      _intel_fast_memset                            ;4609.33
                                ; LOE rsi rdi r13 r14 r15 ebx ebp xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.100::                      ; Preds .B11.99
        mov       r11d, ebp                                     ;4616.28
        mov       r9d, DWORD PTR [65616+rsp]                    ;
        dec       r11d                                          ;4616.28
        mov       r10d, DWORD PTR [65568+rsp]                   ;
        je        .B11.107      ; Prob 50%                      ;4616.28
                                ; LOE rsi rdi r9 r10 r13 r14 r15 ebx ebp r9d r10d r11d r9b r10b xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.101::                      ; Preds .B11.100
        mov       eax, 1                                        ;4616.4
        lea       ecx, DWORD PTR [-1+rbp]                       ;4616.4
        mov       r8d, ecx                                      ;4616.4
        xor       edx, edx                                      ;4616.4
        shr       r8d, 1                                        ;4616.4
        test      r8d, r8d                                      ;4616.4
        jbe       .B11.105      ; Prob 15%                      ;4616.4
                                ; LOE rsi rdi r9 r10 r13 r14 r15 eax edx ecx ebx ebp r8d r9d r10d r11d r9b r10b xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.102::                      ; Preds .B11.101
        mov       al, 1                                         ;3854.36
                                ; LOE rsi rdi r13 r14 r15 edx ecx ebx ebp r8d r9d r10d r11d al xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.103::                      ; Preds .B11.103 .B11.102
        lea       r12d, DWORD PTR [rdx+rdx]                     ;4616.75
        movzx     r12d, WORD PTR [r12+r15]                      ;4616.75
        mov       BYTE PTR [32+rsp+r12], al                     ;4616.36
        lea       r12d, DWORD PTR [1+rdx+rdx]                   ;4616.75
        inc       edx                                           ;4616.4
        cmp       edx, r8d                                      ;4616.4
        movzx     r12d, WORD PTR [r12+r15]                      ;4616.75
        mov       BYTE PTR [32+rsp+r12], al                     ;4616.36
        jb        .B11.103      ; Prob 64%                      ;4616.4
                                ; LOE rsi rdi r13 r14 r15 edx ecx ebx ebp r8d r9d r10d r11d al xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.104::                      ; Preds .B11.103
        lea       eax, DWORD PTR [1+rdx+rdx]                    ;4616.4
                                ; LOE rsi rdi r13 r14 r15 eax ecx ebx ebp r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.105::                      ; Preds .B11.104 .B11.101
        dec       eax                                           ;4616.36
        cmp       eax, ecx                                      ;4616.4
        jae       .B11.107      ; Prob 15%                      ;4616.4
                                ; LOE rax rsi rdi r13 r14 r15 ebx ebp r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.106::                      ; Preds .B11.105
        movzx     edx, WORD PTR [rax+r15]                       ;4616.75
        mov       BYTE PTR [32+rsp+rdx], 1                      ;4616.36
                                ; LOE rsi rdi r13 r14 r15 ebx ebp r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.107::                      ; Preds .B11.100 .B11.105 .B11.106
        mov       r8, r15                                       ;4646.70
        lea       eax, DWORD PTR [-4+rsi]                       ;4645.109
        lea       ecx, DWORD PTR [-3+rbp]                       ;4621.113
        sub       r8, r14                                       ;4646.70
        add       rax, rdi                                      ;4645.54
        movsxd    rcx, ecx                                      ;4623.8
        lea       r12d, DWORD PTR [-3+rsi]                      ;4647.36
        mov       QWORD PTR [65576+rsp], rcx                    ;4623.8
        xor       edx, edx                                      ;4617.4
        movsxd    r12, r12d                                     ;4647.4
        sub       ebx, ebp                                      ;4618.16
        mov       QWORD PTR [65584+rsp], rax                    ;4647.4
        lea       r13, QWORD PTR [-4+r13+rdi]                   ;4645.134
        mov       QWORD PTR [65608+rsp], r8                     ;4647.4
        mov       QWORD PTR [65592+rsp], r15                    ;4647.4
        mov       QWORD PTR [65600+rsp], r14                    ;4647.4
                                ; LOE rsi rdi r12 r13 edx ecx ebx ebp r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.108::                      ; Preds .B11.107 .B11.129
        lea       eax, DWORD PTR [-2+rbp+rdx]                   ;4620.72
        movzx     r8d, WORD PTR [rax+rdi]                       ;4620.49
        cmp       BYTE PTR [32+rsp+r8], 0                       ;4620.79
        je        .B11.128      ; Prob 50%                      ;4620.79
                                ; LOE rsi rdi r12 r13 edx ecx ebx ebp r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.109::                      ; Preds .B11.108
        lea       eax, DWORD PTR [-4+rbp+rdx]                   ;4621.75
        movzx     r8d, WORD PTR [rax+rdi]                       ;4621.50
        cmp       BYTE PTR [32+rsp+r8], 0                       ;4621.82
        jne       .B11.111      ; Prob 50%                      ;4621.82
                                ; LOE rsi rdi r12 r13 edx ecx ebx ebp r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.110::                      ; Preds .B11.109
        mov       r14d, ecx                                     ;4621.86
        jmp       .B11.129      ; Prob 100%                     ;4621.86
                                ; LOE rsi rdi r12 r13 edx ecx ebx ebp r9d r10d r11d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.111::                      ; Preds .B11.109
        mov       eax, edx                                      ;4622.26
        mov       r14d, 1                                       ;4619.5
        lea       r8, QWORD PTR [rdi+rax]                       ;4622.26
        cmp       r10d, DWORD PTR [r8]                          ;4622.41
        jne       .B11.129      ; Prob 50%                      ;4622.41
                                ; LOE rax rsi rdi r8 r12 r13 edx ecx ebx ebp r9d r10d r11d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.112::                      ; Preds .B11.111
        mov       r15, QWORD PTR [65576+rsp]                    ;4623.8
        test      r15, r15                                      ;4624.24
        jle       .B11.176      ; Prob 2%                       ;4624.24
                                ; LOE rax rsi rdi r8 r12 r13 r15 edx ecx ebx ebp r9d r10d r11d r14d r15d r15b xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.113::                      ; Preds .B11.112
        mov       DWORD PTR [65616+rsp], r9d                    ;
        mov       DWORD PTR [65624+rsp], esi                    ;
        mov       r9, r15                                       ;
        mov       r15, QWORD PTR [65592+rsp]                    ;
                                ; LOE rax rdi r8 r9 r12 r13 r15 edx ecx ebx ebp r10d r11d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.114::                      ; Preds .B11.115 .B11.113
        mov       esi, DWORD PTR [-1+r9+r15]                    ;4624.59
        cmp       esi, DWORD PTR [-1+r9+r8]                     ;4624.80
        jne       .B11.175      ; Prob 20%                      ;4624.80
                                ; LOE rax rdi r8 r9 r12 r13 r15 edx ecx ebx ebp r10d r11d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.115::                      ; Preds .B11.114
        add       r9, -4                                        ;4625.23
        test      r9, r9                                        ;4624.24
        jg        .B11.114      ; Prob 82%                      ;4624.24
                                ; LOE rax rdi r8 r9 r12 r13 r15 edx ecx ebx ebp r10d r11d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.116::                      ; Preds .B11.115
        mov       esi, DWORD PTR [65624+rsp]                    ;
        cmp       ebp, esi                                      ;4627.19
        mov       r9d, DWORD PTR [65616+rsp]                    ;
        mov       QWORD PTR [65592+rsp], r15                    ;
        je        .B11.127      ; Prob 50%                      ;4627.19
                                ; LOE rax rdi r8 r12 r13 edx ecx ebx ebp esi r9d r10d r11d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.117::                      ; Preds .B11.176 .B11.116
        mov       r8d, edx                                      ;4645.21
        sub       r8d, r9d                                      ;4645.21
        inc       r8d                                           ;4645.21
        js        .B11.129      ; Prob 16%                      ;4645.46
                                ; LOE rax rsi rdi r8 r12 r13 edx ecx ebx ebp r9d r10d r11d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.118::                      ; Preds .B11.117
        mov       r15, QWORD PTR [65584+rsp]                    ;4645.54
        mov       QWORD PTR [65568+rsp], r8                     ;4645.54
        add       r8, r15                                       ;4645.54
        cmp       r13, r8                                       ;4645.134
        jb        .B11.129      ; Prob 50%                      ;4645.134
                                ; LOE rax rsi rdi r12 r13 edx ecx ebx ebp r9d r10d r11d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.119::                      ; Preds .B11.118
        mov       r8, QWORD PTR [65568+rsp]                     ;4646.22
        add       r8, rdi                                       ;4646.22
        mov       QWORD PTR [65568+rsp], r8                     ;4646.22
        mov       r15d, DWORD PTR [r8]                          ;4646.22
        mov       r8, QWORD PTR [65608+rsp]                     ;4646.96
        cmp       r15d, DWORD PTR [r8]                          ;4646.96
        jne       .B11.129      ; Prob 50%                      ;4646.96
                                ; LOE rax rsi rdi r12 r13 edx ecx ebx ebp r9d r10d r11d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.120::                      ; Preds .B11.119
        mov       r8, r12                                       ;4647.4
        test      r12, r12                                      ;4648.20
        jle       .B11.124      ; Prob 2%                       ;4648.20
                                ; LOE rax rsi rdi r8 r12 r13 edx ecx ebx ebp r9d r10d r11d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.121::                      ; Preds .B11.120
        mov       DWORD PTR [65616+rsp], r9d                    ;
        mov       DWORD PTR [65624+rsp], esi                    ;
        mov       r9, QWORD PTR [65568+rsp]                     ;
        mov       r15, QWORD PTR [65608+rsp]                    ;
                                ; LOE rax rdi r8 r9 r12 r13 r15 edx ecx ebx ebp r10d r11d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.122::                      ; Preds .B11.123 .B11.121
        mov       esi, DWORD PTR [-1+r8+r15]                    ;4648.74
        cmp       esi, DWORD PTR [-1+r8+r9]                     ;4648.95
        jne       .B11.173      ; Prob 20%                      ;4648.95
                                ; LOE rax rdi r8 r9 r12 r13 r15 edx ecx ebx ebp r10d r11d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.123::                      ; Preds .B11.122
        add       r8, -4                                        ;4649.19
        test      r8, r8                                        ;4648.20
        jg        .B11.122      ; Prob 82%                      ;4648.20
                                ; LOE rax rdi r8 r9 r12 r13 r15 edx ecx ebx ebp r10d r11d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.124::                      ; Preds .B11.120 .B11.123
        mov       r14, QWORD PTR [65600+rsp]                    ;
                                ; LOE rax rdi r14 r14d r14b xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.125::                      ; Preds .B11.124
        sub       rdi, r14                                      ;4650.29
        add       rax, rdi                                      ;4650.29
        add       rsp, 65640                                    ;4650.56
        pop       rbp                                           ;4650.56
        pop       r15                                           ;4650.56
        pop       r14                                           ;4650.56
        pop       r13                                           ;4650.56
        pop       r12                                           ;4650.56
        pop       rdi                                           ;4650.56
        pop       rsi                                           ;4650.56
        pop       rbx                                           ;4650.56
        ret                                                     ;4650.56
                                ; LOE
.B11.127::                      ; Preds .B11.176 .B11.116
        mov       rax, r8                                       ;4655.33
        add       rsp, 65640                                    ;4655.33
        pop       rbp                                           ;4655.33
        pop       r15                                           ;4655.33
        pop       r14                                           ;4655.33
        pop       r13                                           ;4655.33
        pop       r12                                           ;4655.33
        pop       rdi                                           ;4655.33
        pop       rsi                                           ;4655.33
        pop       rbx                                           ;4655.33
        ret                                                     ;4655.33
                                ; LOE
.B11.128::                      ; Preds .B11.108
        mov       r14d, r11d                                    ;4659.12
                                ; LOE rsi rdi r12 r13 edx ecx ebx ebp r9d r10d r11d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.129::                      ; Preds .B11.175 .B11.173 .B11.110 .B11.119 .B11.118
                                ;       .B11.117 .B11.111 .B11.128
        add       edx, r14d                                     ;4660.13
        cmp       edx, ebx                                      ;4618.25
        jbe       .B11.108      ; Prob 82%                      ;4618.25
                                ; LOE rsi rdi r12 r13 edx ecx ebx ebp r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.130::                      ; Preds .B11.129
        xor       eax, eax                                      ;4663.10
        add       rsp, 65640                                    ;4663.10
        pop       rbp                                           ;4663.10
        pop       r15                                           ;4663.10
        pop       r14                                           ;4663.10
        pop       r13                                           ;4663.10
        pop       r12                                           ;4663.10
        pop       rdi                                           ;4663.10
        pop       rsi                                           ;4663.10
        pop       rbx                                           ;4663.10
        ret                                                     ;4663.10
                                ; LOE
.B11.131::                      ; Preds .B11.98
        xor       edx, edx                                      ;4669.33
        lea       rcx, QWORD PTR [32+rsp]                       ;4669.33
        mov       r8d, 65536                                    ;4669.33
        mov       DWORD PTR [65568+rsp], r10d                   ;4669.33
        mov       DWORD PTR [65616+rsp], r9d                    ;4669.33
        call      _intel_fast_memset                            ;4669.33
                                ; LOE rbx rsi rdi r12 r13 r14 r15 ebx ebp edi r13d bl bh dil r13b xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.132::                      ; Preds .B11.131
        mov       ecx, ebp                                      ;4682.30
        mov       r9d, DWORD PTR [65616+rsp]                    ;
        add       ecx, -3                                       ;4682.30
        mov       r10d, DWORD PTR [65568+rsp]                   ;
        je        .B11.139      ; Prob 50%                      ;4682.30
                                ; LOE rbx rsi rdi r9 r10 r12 r13 r14 r15 ecx ebx ebp edi r9d r10d r13d bl bh dil r9b r10b r13b xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.133::                      ; Preds .B11.132
        mov       r8d, ecx                                      ;4682.4
        mov       eax, 1                                        ;4682.4
        shr       r8d, 1                                        ;4682.4
        xor       edx, edx                                      ;4682.4
        test      r8d, r8d                                      ;4682.4
        jbe       .B11.137      ; Prob 15%                      ;4682.4
                                ; LOE rbx rsi rdi r9 r10 r12 r13 r14 r15 eax edx ecx ebx ebp edi r8d r9d r10d r13d bl bh dil r9b r10b r13b xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.134::                      ; Preds .B11.133
        mov       DWORD PTR [65624+rsp], esi                    ;3854.36
        mov       al, 1                                         ;3854.36
                                ; LOE rdi r12 r13 r14 r15 edx ecx ebx ebp r8d r9d r10d al xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.135::                      ; Preds .B11.135 .B11.134
        lea       esi, DWORD PTR [rdx+rdx]                      ;4682.74
        mov       r11d, DWORD PTR [rsi+r15]                     ;4682.113
        mov       esi, r11d                                     ;4682.90
        shr       esi, 15                                       ;4682.90
        add       esi, r11d                                     ;4682.128
        movzx     r11d, si                                      ;4682.140
        lea       esi, DWORD PTR [1+rdx+rdx]                    ;4682.113
        inc       edx                                           ;4682.4
        mov       BYTE PTR [32+rsp+r11], al                     ;4682.38
        mov       r11d, DWORD PTR [rsi+r15]                     ;4682.74
        mov       esi, r11d                                     ;4682.90
        shr       esi, 15                                       ;4682.90
        add       esi, r11d                                     ;4682.128
        cmp       edx, r8d                                      ;4682.4
        movzx     r11d, si                                      ;4682.140
        mov       BYTE PTR [32+rsp+r11], al                     ;4682.38
        jb        .B11.135      ; Prob 64%                      ;4682.4
                                ; LOE rdi r12 r13 r14 r15 edx ecx ebx ebp r8d r9d r10d al xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.136::                      ; Preds .B11.135
        mov       esi, DWORD PTR [65624+rsp]                    ;
        lea       eax, DWORD PTR [1+rdx+rdx]                    ;4682.4
                                ; LOE rdi r12 r13 r14 r15 eax ecx ebx ebp esi r9d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.137::                      ; Preds .B11.136 .B11.133
        dec       eax                                           ;4682.38
        cmp       eax, ecx                                      ;4682.4
        jae       .B11.139      ; Prob 15%                      ;4682.4
                                ; LOE rax rsi rdi r12 r13 r14 r15 ecx ebx ebp r9d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.138::                      ; Preds .B11.137
        mov       edx, DWORD PTR [rax+r15]                      ;4682.74
        mov       r8d, edx                                      ;4682.90
        shr       r8d, 15                                       ;4682.90
        add       r8d, edx                                      ;4682.128
        movzx     r11d, r8w                                     ;4682.140
        mov       BYTE PTR [32+rsp+r11], 1                      ;4682.38
                                ; LOE rsi rdi r12 r13 r14 r15 ecx ebx ebp r9d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.139::                      ; Preds .B11.137 .B11.132 .B11.138
        movsxd    rcx, ecx                                      ;4754.8
        lea       r8d, DWORD PTR [-4+rsi]                       ;4784.109
        lea       rax, QWORD PTR [-4+r13+rdi]                   ;4784.134
        mov       QWORD PTR [65584+rsp], rcx                    ;4754.8
        lea       r13d, DWORD PTR [-3+rsi]                      ;4786.36
        movsxd    r13, r13d                                     ;4786.4
        xor       edx, edx                                      ;4683.4
        mov       QWORD PTR [65592+rsp], r15                    ;4706.42
        sub       ebx, ebp                                      ;4684.16
        mov       QWORD PTR [65600+rsp], r14                    ;4706.42
        add       r8, rdi                                       ;4784.54
        mov       QWORD PTR [65608+rsp], r12                    ;4706.42
        lea       r11d, DWORD PTR [-8+rbp]                      ;4706.42
                                ; LOE rax rdx rsi rdi r8 r13 ecx ebx ebp r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.140::                      ; Preds .B11.139 .B11.162
        lea       r12d, DWORD PTR [-4+rbp+rdx]                  ;4688.123
        mov       r14d, DWORD PTR [r12+rdi]                     ;4688.46
        mov       r15d, r14d                                    ;4688.75
        shr       r15d, 15                                      ;4688.75
        add       r15d, r14d                                    ;4688.126
        movzx     r12d, r15w                                    ;4688.138
        cmp       BYTE PTR [32+rsp+r12], 0                      ;4688.156
        je        .B11.161      ; Prob 50%                      ;4688.156
                                ; LOE rax rdx rsi rdi r8 r13 ecx ebx ebp r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.141::                      ; Preds .B11.140
        mov       r12d, 1                                       ;4685.5
        lea       r14d, DWORD PTR [-9+rbp+rdx]                  ;4704.131
        mov       r15d, DWORD PTR [r14+rdi]                     ;4704.44
        mov       r14d, r15d                                    ;4704.78
        shr       r14d, 15                                      ;4704.78
        add       r14d, r15d                                    ;4704.134
        movzx     r15d, r14w                                    ;4704.146
        cmp       BYTE PTR [32+rsp+r15], 0                      ;4704.166
        je        .B11.143      ; Prob 50%                      ;4704.166
                                ; LOE rax rdx rsi rdi r8 r13 ecx ebx ebp r9d r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.142::                      ; Preds .B11.141
        lea       r14d, DWORD PTR [-7+rbp+rdx]                  ;4705.134
        mov       r15d, DWORD PTR [r14+rdi]                     ;4705.47
        mov       r14d, r15d                                    ;4705.81
        shr       r14d, 15                                      ;4705.81
        add       r14d, r15d                                    ;4705.137
        movzx     r15d, r14w                                    ;4705.149
        cmp       BYTE PTR [32+rsp+r15], 0                      ;4705.169
        jne       .B11.144      ; Prob 50%                      ;4705.169
                                ; LOE rax rdx rsi rdi r8 r13 ecx ebx ebp r9d r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.143::                      ; Preds .B11.141 .B11.142
        mov       r12d, r11d                                    ;4706.8
        jmp       .B11.162      ; Prob 100%                     ;4706.8
                                ; LOE rax rdx rsi rdi r8 r13 ecx ebx ebp r9d r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.144::                      ; Preds .B11.142
        mov       QWORD PTR [65576+rsp], rdx                    ;4742.26
        lea       r15, QWORD PTR [rdi+rdx]                      ;4742.26
        cmp       r10d, DWORD PTR [r15]                         ;4742.41
        jne       .B11.162      ; Prob 50%                      ;4742.41
                                ; LOE rax rsi rdi r8 r13 r15 edx ecx ebx ebp r9d r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.145::                      ; Preds .B11.144
        mov       r14, QWORD PTR [65584+rsp]                    ;4754.8
        test      r14, r14                                      ;4756.24
        jle       .B11.181      ; Prob 2%                       ;4756.24
                                ; LOE rax rsi rdi r8 r13 r14 r15 edx ecx ebx ebp r9d r10d r11d r12d r14d r14b xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.146::                      ; Preds .B11.145
        mov       DWORD PTR [65616+rsp], r9d                    ;
        mov       DWORD PTR [65624+rsp], esi                    ;
        mov       r9, QWORD PTR [65592+rsp]                     ;
                                ; LOE rax rdi r8 r9 r13 r14 r15 edx ecx ebx ebp r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.147::                      ; Preds .B11.148 .B11.146
        mov       esi, DWORD PTR [-1+r14+r9]                    ;4756.59
        cmp       esi, DWORD PTR [-1+r14+r15]                   ;4756.80
        jne       .B11.180      ; Prob 20%                      ;4756.80
                                ; LOE rax rdi r8 r9 r13 r14 r15 edx ecx ebx ebp r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.148::                      ; Preds .B11.147
        add       r14, -4                                       ;4757.23
        test      r14, r14                                      ;4756.24
        jg        .B11.147      ; Prob 82%                      ;4756.24
                                ; LOE rax rdi r8 r9 r13 r14 r15 edx ecx ebx ebp r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.149::                      ; Preds .B11.148
        mov       esi, DWORD PTR [65624+rsp]                    ;
        cmp       ebp, esi                                      ;4759.19
        mov       QWORD PTR [65592+rsp], r9                     ;
        mov       r9d, DWORD PTR [65616+rsp]                    ;
        je        .B11.160      ; Prob 50%                      ;4759.19
                                ; LOE rax rdi r8 r13 r15 edx ecx ebx ebp esi r9d r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.150::                      ; Preds .B11.181 .B11.149
        mov       r14d, edx                                     ;4784.21
        sub       r14d, r9d                                     ;4784.21
        inc       r14d                                          ;4784.21
        js        .B11.162      ; Prob 16%                      ;4784.46
                                ; LOE rax rsi rdi r8 r13 r14 edx ecx ebx ebp r9d r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.151::                      ; Preds .B11.150
        mov       QWORD PTR [65568+rsp], r14                    ;4784.54
        lea       r15, QWORD PTR [r8+r14]                       ;4784.54
        cmp       rax, r15                                      ;4784.134
        jb        .B11.162      ; Prob 50%                      ;4784.134
                                ; LOE rax rsi rdi r8 r13 r14 edx ecx ebx ebp r9d r10d r11d r12d r14d r14b xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.152::                      ; Preds .B11.151
        add       r14, rdi                                      ;4785.22
        mov       QWORD PTR [65568+rsp], r14                    ;4785.22
        mov       r15d, DWORD PTR [r14]                         ;4785.22
        mov       r14, QWORD PTR [65608+rsp]                    ;4785.96
        cmp       r15d, DWORD PTR [r14]                         ;4785.96
        jne       .B11.162      ; Prob 50%                      ;4785.96
                                ; LOE rax rsi rdi r8 r13 edx ecx ebx ebp r9d r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.153::                      ; Preds .B11.152
        mov       r14, r13                                      ;4786.4
        test      r13, r13                                      ;4787.20
        jle       .B11.157      ; Prob 2%                       ;4787.20
                                ; LOE rax rsi rdi r8 r13 r14 edx ecx ebx ebp r9d r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.154::                      ; Preds .B11.153
        mov       DWORD PTR [65616+rsp], r9d                    ;
        mov       DWORD PTR [65624+rsp], esi                    ;
        mov       r9, QWORD PTR [65568+rsp]                     ;
        mov       r15, QWORD PTR [65608+rsp]                    ;
                                ; LOE rax rdi r8 r9 r13 r14 r15 edx ecx ebx ebp r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.155::                      ; Preds .B11.156 .B11.154
        mov       esi, DWORD PTR [-1+r14+r15]                   ;4787.74
        cmp       esi, DWORD PTR [-1+r14+r9]                    ;4787.95
        jne       .B11.178      ; Prob 20%                      ;4787.95
                                ; LOE rax rdi r8 r9 r13 r14 r15 edx ecx ebx ebp r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.156::                      ; Preds .B11.155
        add       r14, -4                                       ;4788.19
        test      r14, r14                                      ;4787.20
        jg        .B11.155      ; Prob 82%                      ;4787.20
                                ; LOE rax rdi r8 r9 r13 r14 r15 edx ecx ebx ebp r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.157::                      ; Preds .B11.153 .B11.156
        mov       rax, QWORD PTR [65576+rsp]                    ;
        mov       r14, QWORD PTR [65600+rsp]                    ;
                                ; LOE rax rdi r14 eax r14d al ah r14b xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.158::                      ; Preds .B11.157
        sub       rdi, r14                                      ;4789.29
        add       rax, rdi                                      ;4789.29
        add       rsp, 65640                                    ;4789.56
        pop       rbp                                           ;4789.56
        pop       r15                                           ;4789.56
        pop       r14                                           ;4789.56
        pop       r13                                           ;4789.56
        pop       r12                                           ;4789.56
        pop       rdi                                           ;4789.56
        pop       rsi                                           ;4789.56
        pop       rbx                                           ;4789.56
        ret                                                     ;4789.56
                                ; LOE
.B11.160::                      ; Preds .B11.181 .B11.149
        mov       rax, r15                                      ;4794.33
        add       rsp, 65640                                    ;4794.33
        pop       rbp                                           ;4794.33
        pop       r15                                           ;4794.33
        pop       r14                                           ;4794.33
        pop       r13                                           ;4794.33
        pop       r12                                           ;4794.33
        pop       rdi                                           ;4794.33
        pop       rsi                                           ;4794.33
        pop       rbx                                           ;4794.33
        ret                                                     ;4794.33
                                ; LOE
.B11.161::                      ; Preds .B11.140
        mov       r12d, ecx                                     ;4821.12
                                ; LOE rax rdx rsi rdi r8 r13 ecx ebx ebp r9d r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.162::                      ; Preds .B11.180 .B11.178 .B11.152 .B11.151 .B11.150
                                ;       .B11.144 .B11.143 .B11.161
        add       edx, r12d                                     ;4822.13
        cmp       edx, ebx                                      ;4684.25
        jbe       .B11.140      ; Prob 82%                      ;4684.25
                                ; LOE rax rdx rsi rdi r8 r13 ecx ebx ebp r9d r10d r11d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.163::                      ; Preds .B11.162
        xor       eax, eax                                      ;4825.10
        add       rsp, 65640                                    ;4825.10
        pop       rbp                                           ;4825.10
        pop       r15                                           ;4825.10
        pop       r14                                           ;4825.10
        pop       r13                                           ;4825.10
        pop       r12                                           ;4825.10
        pop       rdi                                           ;4825.10
        pop       rsi                                           ;4825.10
        pop       rbx                                           ;4825.10
        ret                                                     ;4825.10
                                ; LOE
.B11.169::                      ; Preds .B11.71                 ; Infreq
        mov       QWORD PTR [65608+rsp], rsi                    ;
        mov       rdx, QWORD PTR [40+rsp]                       ;
        mov       r11, QWORD PTR [32+rsp]                       ;
        mov       esi, DWORD PTR [65624+rsp]                    ;
        jmp       .B11.80       ; Prob 100%                     ;
                                ; LOE rdx rdi r11 ecx ebx ebp esi r8d r9d r10d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.173::                      ; Preds .B11.122                ; Infreq
        mov       QWORD PTR [65608+rsp], r15                    ;
        mov       r9d, DWORD PTR [65616+rsp]                    ;
        mov       esi, DWORD PTR [65624+rsp]                    ;
        jmp       .B11.129      ; Prob 100%                     ;
                                ; LOE rdi r12 r13 edx ecx ebx ebp esi r9d r10d r11d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.175::                      ; Preds .B11.114                ; Infreq
        mov       QWORD PTR [65592+rsp], r15                    ;
        mov       r9d, DWORD PTR [65616+rsp]                    ;
        mov       esi, DWORD PTR [65624+rsp]                    ;
        jmp       .B11.129      ; Prob 100%                     ;
                                ; LOE rdi r12 r13 edx ecx ebx ebp esi r9d r10d r11d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.176::                      ; Preds .B11.112                ; Infreq
        cmp       ebp, esi                                      ;4627.19
        jne       .B11.117      ; Prob 50%                      ;4627.19
        jmp       .B11.127      ; Prob 100%                     ;4627.19
                                ; LOE rax rsi rdi r8 r12 r13 edx ecx ebx ebp r9d r10d r11d r14d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.178::                      ; Preds .B11.155                ; Infreq
        mov       QWORD PTR [65608+rsp], r15                    ;
        mov       r9d, DWORD PTR [65616+rsp]                    ;
        mov       esi, DWORD PTR [65624+rsp]                    ;
        jmp       .B11.162      ; Prob 100%                     ;
                                ; LOE rax rdi r8 r13 edx ecx ebx ebp esi r9d r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.180::                      ; Preds .B11.147                ; Infreq
        mov       QWORD PTR [65592+rsp], r9                     ;
        mov       r9d, DWORD PTR [65616+rsp]                    ;
        mov       esi, DWORD PTR [65624+rsp]                    ;
        jmp       .B11.162      ; Prob 100%                     ;
                                ; LOE rax rdi r8 r13 edx ecx ebx ebp esi r9d r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.181::                      ; Preds .B11.145                ; Infreq
        cmp       ebp, esi                                      ;4759.19
        jne       .B11.150      ; Prob 50%                      ;4759.19
        jmp       .B11.160      ; Prob 100%                     ;4759.19
                                ; LOE rax rsi rdi r8 r13 r15 edx ecx ebx ebp r9d r10d r11d r12d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.184::                      ; Preds .B11.82                 ; Infreq
        lea       r14d, DWORD PTR [-1+r9]                       ;4500.41
        lea       r15, QWORD PTR [r12+r14]                      ;4500.13
        mov       r10d, DWORD PTR [r15]                         ;4608.33
        jmp       .B11.99       ; Prob 100%                     ;4608.33
                                ; LOE rsi rdi r13 r14 r15 ebx ebp r9d r10d xmm6 xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
.B11.185::                      ; Preds .B11.12                 ; Infreq
        mov       rax, rcx                                      ;3748.78
        add       rsp, 65640                                    ;3748.78
        pop       rbp                                           ;3748.78
        pop       r15                                           ;3748.78
        pop       r14                                           ;3748.78
        pop       r13                                           ;3748.78
        pop       r12                                           ;3748.78
        pop       rdi                                           ;3748.78
        pop       rsi                                           ;3748.78
        pop       rbx                                           ;3748.78
        ret                                                     ;3748.78
        ALIGN     16
                                ; LOE
.B11.187::
; mark_end;
Railgun_Trolldom ENDP
;Railgun_Trolldom	ENDS
_TEXT	ENDS
*/

// Fixed version from 2012-Feb-27.
// Caution: For better speed the case 'if (cbPattern==1)' was removed, so Pattern must be longer than 1 char.
char * Railgun_Doublet (char * pbTarget, char * pbPattern, uint32_t cbTarget, uint32_t cbPattern)
{
	char * pbTargetMax = pbTarget + cbTarget;
	register uint32_t ulHashPattern;
	uint32_t ulHashTarget, count, countSTATIC;

	if (cbPattern > cbTarget) return(NULL);

	countSTATIC = cbPattern-2;

	pbTarget = pbTarget+cbPattern;
	ulHashPattern = (*(uint16_t *)(pbPattern));

	for ( ;; ) {
		if ( ulHashPattern == (*(uint16_t *)(pbTarget-cbPattern)) ) {
			count = countSTATIC;
			while ( count && *(char *)(pbPattern+2+(countSTATIC-count)) == *(char *)(pbTarget-cbPattern+2+(countSTATIC-count)) ) {
				count--;
			}
			if ( count == 0 ) return((pbTarget-cbPattern));
		}
		pbTarget++;
		if (pbTarget > pbTargetMax) return(NULL);
	}
}


// Pattern must be >=4
char * Railgun_BawBaw_reverse (char * pbTarget, char * pbPattern, uint32_t cbTarget, uint32_t cbPattern)
{
	register uint32_t ulHashPattern;
	signed long count;

	unsigned char bm_Horspool_Order2[256*256]; // Bitwise soon...
	uint32_t i, Gulliver;

	if (cbPattern > cbTarget) return(NULL);

			// BMH order 2, needle should be >=4:
			ulHashPattern = *(uint32_t *)(pbPattern); // First four bytes
			for (i=0; i < 256*256; i++) {bm_Horspool_Order2[i]=0;}
			for (i=0; i < cbPattern-1; i++) bm_Horspool_Order2[*(unsigned short *)(pbPattern+i)]=1;
			i=cbTarget;
			while (i >= cbPattern) {
				Gulliver = 1; // 'Gulliver' is the skip
				if ( bm_Horspool_Order2[*(unsigned short *)&pbTarget[i-cbPattern]] != 0 ) {
					if ( bm_Horspool_Order2[*(unsigned short *)&pbTarget[i-cbPattern+2]] == 0 ) Gulliver = cbPattern-(2-1)-2; else {
						if ( *(uint32_t *)&pbTarget[i-cbPattern] == ulHashPattern) { // This fast check ensures not missing a match (for remainder) when going under 0 in loop below:
							count = cbPattern-4+1; 
							while ( count > 0 && *(uint32_t *)(pbPattern+count-1) == *(uint32_t *)(&pbTarget[i-cbPattern]+(count-1)) )
								count = count-4;
							if ( count <= 0 ) return(pbTarget+i-cbPattern);
						}
					}
				} else Gulliver = cbPattern-(2-1);
				i = i - Gulliver;
			}
			return(NULL);
}

// Pattern must be >=2
char * Railgun_Baw_reverse (char * pbTarget, char * pbPattern, uint32_t cbTarget, uint32_t cbPattern)
{
	register uint16_t ulHashPattern;
	signed long count;

	unsigned char bm_Horspool_Order2[256*256]; // Bitwise soon...
	uint32_t i, Gulliver;

	if (cbPattern > cbTarget) return(NULL);

			// BMH order 2, needle should be >=4:
			ulHashPattern = *(uint32_t *)(pbPattern); // First four bytes
			for (i=0; i < 256*256; i++) {bm_Horspool_Order2[i]=0;}
			for (i=0; i < cbPattern-1; i++) bm_Horspool_Order2[*(unsigned short *)(pbPattern+i)]=1;
			i=cbTarget;
			while (i >= cbPattern) {
				Gulliver = 1; // 'Gulliver' is the skip
				if ( bm_Horspool_Order2[*(unsigned short *)&pbTarget[i-cbPattern]] != 0 ) {
						if ( *(uint16_t *)&pbTarget[i-cbPattern] == ulHashPattern) { // This fast check ensures not missing a match (for remainder) when going under 0 in loop below:
							count = cbPattern-2+1; 
							while ( count > 0 && *(uint16_t *)(pbPattern+count-1) == *(uint16_t *)(&pbTarget[i-cbPattern]+(count-1)) )
								count = count-2;
							if ( count <= 0 ) return(pbTarget+i-cbPattern);
						}
				} else Gulliver = cbPattern-(2-1);
				i = i - Gulliver;
			}
			return(NULL);
}

// Last change: 2016-Nov-14;
// Last change: 2016-Aug-22; fixed Railgun_Swampshine, after that changed with Trolldom.
// Last change: 2016-Apr-08; small fix in encoding (avoiding possible negative offset if Sliding Window is biggy) from 2015-Mar-03.
// If you want to help me to improve it, email me at: sanmayce@sanmayce.com
// Enfun!

// To Tomisaburo Wakayama:
/*
                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                  UjM                                                                                                                                                                                                                                                                                                           
                                                                                                 Z8MN;                                                                                                                                                                                                                                                                                                          
                                                             :YMTUXUXXrrvv,                     AOmvMv                                                                                                                                                                                                                                                                                                          
                                                           vb6GtHX4AwThMNhXhTV;.               .ttrvr;                                                                                                                                                                                                                                                                                                          
                                                          hOC0jkyVvvvvvYVyLALMrhrV;            vUcyYV;                                                                                                            .Al                                                                                                                                                                                           
                                                         rBejUrvvvvvvvvvvvYvVVyLrLwAL;         vLUMvMv                                                                                                           ;BDj:                      ;vlllvv:                                                                                                                                                            
                                                        .pBcllvvvvvvvvvvvvvvvvvllyVAyAry;    .mVvwhYry                                                                                                          ,GDrv;                ;YXtpCpjmmUUXUhr                                                                                                                                                          
                                                        yZjvvvvvv;v;vvvvvvvvvvvvvvvlYVyyrVv. vkvvvhyyy:                                                                                                         j0v:v:           ,lMmp0bbcHhyvvvvvvvlhH.                                                                                                                                                        
                                                        HqrvvvvvvvvvvvvvvvvvvvvvvvvvvYYVYyyyvvvv;vlwLrv                                                                                                        XZHvvl.        vwHj2CjtUrYvvvvv;vvvvvvvlA                                                                                                                                                        
                                                       .cHVvvvvvvvvvv;v;v;v;vvvvvvvvvvvvYvyVYvv;vvvlwLV;                                                                                                      U#bYvlv     ,vkm2C2tmhVvvvvvvvvvvvv;vvv;;lv                                                                                                                                                       
                                                       ;khvvvvvvvv;v;;,:,:,:;v;vvvvvvvvvvvvYYVYVvv;;vTyy,                                                                                                    mCZLvv:   :VUHjp@CcAyvvvvvvvvvvvvvvvv;vvv;vv.                                                                                                                                                      
                                                       vtrvvv;vvv;;;:;r4mH4yv::,;;vvvvvvvvvvvvyVyvv;;vVYV.                                                                                                 .kj9Tv;;  vhhHtZZpMl;v;v;vvvvvvvvvvvvvvvvvvvvv;                                                                                                                                                      
                                                       VcAvvvvvv;v;:UG99DODODZkv:,;vvvvvvvvvvvvvYyllvv;llY:                                                                                               vtmGX;;;vAAXcDZ0Uy;v;v;v;v;v;v;;;v;v;vvvvvvvvvvv                                                                                                                                                      
                                                       lhVvvvvvvv;,U6qCtpt2200DO0hv,;;vvvvvvvvvvvvVYVvv;vlV;                                                                                             McHZr:;vYANCZOcrv;;vvv;vvv;v;;,,,:,:;vvvvvvvvvvv;                                                                                                                                                      
                                                       y4y;vvvvv;:vOq2j2tpjbjb20@OZHv;;vvlvvvvvvvvvvlyYv;vYlv,                                                                                         vHHcCV;vYrm#9CUv;;v;vvvvvvv;;.:;lUtttA;:vvvvvvvvvvv                                                                                                                                                      
                                                       ymyv;vvv;v,wObjpctj2tpjptpjC@mv;;vvvvvvvvvvvvvvVYvvvvVlv                                                                                      :UHXHplvYwcGGcyv;v;vvv;vvv;v;;;NtZZO#DDe@l:vvvvvvvvvv                                                                                                                                                      
                                                       y4yvvvvvv;:tDjjc2tpt2jjctmjHHwlvv;;;vvvvvvvvvvvvvlVvvvllv:                                                                                  ;hcXNHmlvMbODUv;;;v;v;vvvvv;;;lH##qp0p2p@0gp;;vvvvvvvvv                                                                                                                                                      
                                                       vkyvvvvv;;;@btcctjjbpbttHmHcMlvVYl;;;vvvvlvvvvvvvvvVlvvlvlv;                                                                             .vUkkw4HklMCBOw;;;vvvvv;v;v;;vNkLvvrmmtcjjC20#Ov;vvvvvvvv;                                                                                                                                                      
                                                       vhVvvvv;v;vCCcjctcjtpjpmHmcmkvvvvvYvv;;;vvlvlvvvvvvvVlYvlYvvv:        ,,;;;;vvvvvvvvlvvvvvlvvvvvvvvvv;v;;;;::,,..                     ,vNkUNhMhNhr0BDMv;v;vvv;vvvv;vkkpHAv, vrkmjjpjCp#9N,vvvvvvvv;                                                                                                                                                      
                                                       :4yvvvvvv;;bCpmjcpc2tpccmcHccU;vvvvvvlvv;;;vvlvvvvvvvvvrvlvlvv;v;v;vvlvYlyYyyyVryLyrrLyrLyyrVyVyyryyVVllvllvvlvvvlvVvvvvvv;;,....,;vrUUkXNTTArLhj92T;;;vvvvv;v;v;vvk0cYvv; ;Vwhmcjtpb0#ek;;vvvvvvv:                                                                                                                                                      
                                                       ,hyvvvv;v:v0qmptjctjpjjctmtttp4vvvvvvvvvYvv;vvvvvvvvvvvvyVlvYvv;vlVVrVLVryLLrLrLAyryrLArAyrrTAMrMrArMrryMrAyrrrrArTrAyrVyVVVLLrrwTNhUThAMrAVyXZg2V;;vvv;v;;;;;vvAUj4Y;vvv.,vMhmmjtjjCq#9c,vvvvvvvv:                                                                                                                                                      
                                                        wyvvvvv;;;#CccjttcCjpttmmkHUktUvvvvvvvvvvvlvvvvvlvYvlvvvlYylvvvvVVVvVllVyYyVyYylyVylllyYyVyVVYVYYYYYyVyyryryryALrrAyArTrMrrAMLMyLrMrrrAyryHD8qr;;vv;v;;;vlrLNXUhAvvvvvv,.vrwkHcctt2jCC6k:;vvvvvvv.                                                                                                                                                      
                                                        yMvvvvvv;;j#tptpcjjptptckwLy:;lVvlvvvvvvvvvvvVvYlVvvvvvvvvvVlvvYvlvlvlvVlYvYvVvVvlvlvlllvlvlvvvlvvvlvYvlvlvVvllylYYylllylyYVvYvVlyLyyrLrwDeDr;;vvv;;vvLArArrVlvvvvvvvv:.;rNHkccttCp2t#9H:vvvvvvvv                                                                                                                                                       
                                                yAvvv:  vrl;vvv;;:t@pcjtjjpjpccHHwNYv;;,;vvvvvvvvvvvvvvvlvVvVvVVvvvvVlYvlvlvllylVlyllvVlYvYlYvYvYvlvlvlvVVyvlvlvlvvvvvlvlvlvlvvvvvVlYvYlyVyyrrUt9qy;;;vvv;VrrVyvvvvvvvvvvvvvv;;vTUcctcjjjcCppZGr:vvvvvvvv                                                                                                                                                       
                                               :B0vvvvv;vyvvvvvv;:UOjjtjtCjpctHmHHVvvYlv:;vv;vvvvlvvvvvvvvvvlhkHrAyLrwyrVylVYylVlVlYYVllvVlVlYlVllvlvlvyLryylvvlvlvvvlvvvlvvvYvlvllVlllVYVVryhkpXv;vvvvvvlVlvvvvvvvvvvvvvv;v.;vvvllYyUHccjp0C9Dv:vvvvvvv:                                                                                                                                                       
                                                  ;vv;;;vlYvvvvvv,wO2ttt2p2t2ctHHmUvvlYVyvVyyvvvlvvvvvvvvvlvYUZjpmHUNMryLVyyyVylyYVlyYyVYYVlllllVvVllvAVYvlvlvvvYYlvlvlvlvYvllYvlvYvlvVlYYyyrrNrVvlvvvvvvvvvvvvvvvvvvvvvvvvvvllvyv; ;TmHjcC2@OC;;vvvvvv;.                                                                                                                                                       
                                                    ;LVv;lYlvvvv;;vZ2jjttCtpjjmcHcmXvvVVYyVrryvlvlvlvVvyyMwXwUHHXhrryLyyVyVyyyVVVyVyVyVylyVyYyYyVylYvAyVYVvlvvlvvvvYvVvlvYvYvllYlYvYlVlVvylyVryyVryylVllvvvvvvvvvvvvvvvvvllVllvv;: ;YNHct2CCC6U:vvvvvv;;  :vAk:                                                                                                                                                 
                                                     ;qh;vVlvvv;v;;cZjjtpppjttjmcmmmkv;;VylvlvYlyVVyMrNwU4hrALryryyYyVryyyryrVyVyVVlylyYyYyYylVYVYllMyyVyVVVVvlllvvvYvlvllYvlvYvYvllVlVvYVVYyYyVylVvVVyLyyVvlvvvvvvvvvvvvvv;v;;.,:YM4UmmpjCbOGV:vvvvvv;:,:;Vw.                                                                                                                                                  
                                                      vCv;vyvvvvvv,rOqtptbjjjpcjccctcbmrvLyNMryAM4XHkU4hrAyrLrLArAyyVLyLyLVLVyVyVyVyVyVyVylyYylVYlvAryVyyyyyVVlVvlvvvYllvllVlYlYlVlYvYlYVVYVlYlVllvYvvvVlyVLYyllvvvvvvvv;;;;,,.;vM4kkcctmpjC9Z;;vvvvvv;,.;vY                                                                                                                                                    
                                                       vw;vYYvvvvv;;j0pcjjptbjCjptb29B8BcvvMHHmmjjttmwMrryrrArTLryLyyyyyryLyryyVyyrVyYyVyVyyyVyVYVAyyVyyLyrVyVVYVvllvvvvylYvVlYlyVyvYlVlYlVvVvYvYvlvvvvvlvlvVyyYyllvvvvvv;::;vrwkHcktcjjjjC@eM:;vvvvv;;.,vV                                                                                                                                                     
                                                        YyvYYlvvvv;:VObccjj2jbp2pOB8BGtwryvvvllyTHwhyyLLyArALMLLyLVLyLyryLyLyLVyVyyyyLyyyyYyVyYVYAyyyrVryLyyVyYyYylVvvvvlVYllVvVvyYyYyvVlVlVlYvVvYvlvlvvvvvvvlvllVYyYlvv:vrcHmHmHmmtjbpbtbCGq;;vvvvvv;..;r                                                                                                                                                      
                                                         ylvyYvvvvv;;jZt2t2tbj0Z8B8@UyyyArrVyVrLryyyrrAyrLryrLryryyyyyLVLyryyyryrLryryyyyVyVLyVVryLyryryyyryryrVyVylVlvvvlylyVyVVYllyVVlVlVlYvVvllYvlvvvlvvvvvvvvvvYV;;;;;r0eO0tttjj2jC2CbZGU;vvvvvvv:.:y                                                                                                                                                       
                                                         .LlYyvvvvvv:l#@pjp0COB8g24UThAMAArwMMAhrArryryTyrrMyryLyryLyyyyyLyryyyryVlyVyyrVyVyyyVAyyyryyyryyYVvlvVyyYVYVlYvvlylVVyYyYllVYYlYvYvVlVvlvlllvlvlvvvvvvvvvvvvvvvv;vrCG6qbpCbqCqCq9#;;vvvvvv;,,l.                                                                                                                                                       
                                                          ;UylVvvvvv;,XOCj0ZBe9jNMMrrAwAhrMrAAMAArMrAyryAyryryyyryyyLVyyryLyryLVYvvvvYyyryyyyyryryryyyyyAlyVVvlvlYLVyYVlVvvvVYVvVyVvlvlvVlVlYvVvYYlvlvlvvvlvvvvvvvvvvvvvvvv;::VbB6Dq#C0C@Z6A;;vvvvv;;,v,                                                                                                                                                        
                                                           vyVYvvvvvv;;20CGBZtXTVyyl:vVTATrArTrALTrwrrrALLrryyyyyryryryyVLyryryLyylvvvyyVrVyVALrLLVryLyrVyVyyyYVvvvyVyYyYVvvvYlvvryylVvvvVlyvlvVvlvYvvvlvvvvvvvvvvvvvvvvvvvvv;,:l09Obbp0@et;;vvvvv;;,;.                                                                                                                                                         
                                                            yVyVvvvvv;:;68GjUryVrrTv;;yAwrhAMrTrArAAAyrrryLyLyyVryryryryryrrLVryryrlvvYVyyyVLyryrLAyyrrVLyyVlVyyVvvvyyyVyyyvvvVvvrLyryyvv;llVlVllvYvlvlvlvYvvvvvvvvvvvvvvvvvv;vv:,rCD2bbDOV:vvvvvvv;;,                                                                                                                                                          
                                                            .rVrlvvv;v;;pGHMYyyArwTv;vywTwAwTTAMMwrArrLArAyryryryrLLyAyryLyAyryryLyLylvYVyVArArArALyyryrrAyYAylryyvvvyyLVyyyvvvvvAryyryLVvvvvVlYlllYlYvlvvvlvvvlvvvvvvvvvvvvvvrwyv,vUZCCDN:v;vvvvv;:,                                                                                                                                                           
                                                             vyVyvvv;vyHqkyYLrALAThv;vrrhAAAhrlYTAMrTyryrrrLLyrVryryryyyrrALyLryAyryrLVvvlrArrALrLyyrrMAMyvM8XvYyyLvvvVVyVLyyVv;vMAVLyyyyYVvvvVvVllvlllvlvvvvvvvvvvvlvvvvvvvv;vyXLv:;ljOc:;vv;v;v;;,                                                                                                                                                            
                                                              VyMy;vvNjqUVYArArAAhAv;YrMrArMyyvvlTAMrMrMLALryryrrAyAyryrrArrrryryryryrrY;vATrArAyLyrATAMrr;bB#;LyryyvvvyVrVyVrVv;UrLyyVyyyYYvv;vlVvYvlllllvlvlvvvvvvvvvvvvvvvv;VM4vv;:vUv:;vvvvvv;:                                                                                                                                                             
                                                              ;yVlYAH2CMlyAAALArMTTv;lrAwAALMyV;vlrrTwwM4AArMrArryLyrAArArMAMrAAMrrrryrLlVwAArArrrNwwrTrTy;B8B;YLyryrvvvVVyVyVLVyyrYyVyyLVLVVvvvvvVllvVvVvlvlvlvvvvvvvvvvvvvvvvvTAAvv;;;l;;;vvv;v;                                                                                                                                                              
                                                               VvycGZcLVVMATrAAArMTV;vrhhTVALryylvvVlyywMhTwhwyrrMAArMLrrTAXTwAhwwAArryAAwAwLArAyhhwrwAMrVv8B8vvLLyryAvvvVVyVyyryryyyyVrVyVyYYvv;vlyllvYvYvlvlvlvlvvvvvlvvvvvvv;VrMv;vv;vv;;;v;;;.                                                                                                                                                              
                                                               :yrCqHyYyTrwATMTrAAXAvvyM4LrAhhXThArVYvvvllyrkUUNNMTAMrLr4wNwMAwrMAwMMrryArMrTrrLhhUMwMAAAvAB8BU;ryyVryLvvvVVyyLyLyrYyVLvvvLVVYYYlvvvlvVvlvlvlvlvlvYvvvvvvvvvvvv;yAwvv;vvvvl;;;;;;                                                                                                                                                               
                                                                yAAAVyrMryvvvwMwrArTVvvMAAXkUtjjtCc2mmMrVYvvvyAh4XAUMrMUX4wXrLVlvlrwrAAArMAAArLNNhwhTTrTrvc8B89;yALyyryrvvvVyyVLyyYVYMArvvvYVyVyVVvvvyVVvYvlvVvYvlvlvvvvvvvvvvvvvYLVvvvvvvvlv;;;                                                                                                                                                                
                                                                AVyyLyALyyyv;vrwhrMrhyvvrrHc22CCqb@ZDOO@#tcTAvvvAXUNwUpkUwXAAATArvvLwrAAwrAyryN4UwTMwAwArvBB8B8;VyAyLyLyyvvvyyryLYYvLNTrrlvvvYVYyYVvvvYlYlVvYvlvvvvvvvlvvvlvvvv;;vyYvvvvvvvvv;;,                                                                                                                                                                
                                                               .AArAyrrryAArv;vLTTMwTHM;;wcbjhvyrhHjcb0Z#ZD@Ct4MllYyyjcUThA4hwAhhy;lATrTrArrLUU4AwrAAwrAYvB888BlvryryyyyyrvvvLyrVyvvAkLrLLvvvvvyVyVyYvvllVvlvYvvvlvlvvvlvvvvvlv;:vllvvvvvvvv;;:                                                                                                                                                                 
                                                               ;hrrLArMVrrMrrv;;VAkXkkph;VbCmvLryvvvyAkmpjqC#b@0chyvmpHXwMUcUMhANMl;yAArMyAyhNXThAhMwAMrYY8B888yvrryryryyyyvvlryyVvvXXrLryLvv;vvyVyVyYvvvllvlvvvVvlvvvvvv;vvvv;,;vvvvvvvv;;;v;.                                                                                                                                                                 
                                                               LMAVrLALyyrrTrAv;;vMtcct0mkmjkyHjjjcmLvvvLNkppCjCttmcpckkAhjcMTMwwNTvvLLrLryhhNwhMMATMAAAvhB8B8Bj;yyryyyyVrLVvvYrVVvykhyyVLyyvvvvvyVyVLVv;vvlvYlvvvvlvVllvvvv;v;vvvvv;v;v;;;vv;                                                                                                                                                                  
                                                              ,wMyyyyrAYLrTAwArvv;vym20jCbccpr4pjmC@Z@cAvvvrmcjjjcjttHHwAHZkNTwAMAhrvvyyyyMXNAwATAArMATLvC8B8B8O;VryryyyLVyylvvlyvvhUyyyryyVVvvvvlVYylyllvyYVYVyyyrrryrYyVlvvvvvv;;;v;;;;;;;vv                                                                                                                                                                  
                                                              Y4yyVyywAlyrLAM4NXyv;;vU2Z2tmpbHyCbcccHp2D04v;;YhccbtmcHUNhq2mUXwMAMANYvvVVhhwMTAMrwMMrMrr;e88B8B8;Vyyyryryyyryl;llvymrryrVyVyVYvvvYYyYyYyYylVrArMTUrrVvvlvvvvll;v;;;;;;:;lV;;;l:                                                                                                                                                                 
                                                              ULyyyyrAkyVVrrMwkkH4Y;;:YmZ#CjpcLT9DbcttbjqqZmLvvvMmttjkmAcCjkmUhrTrTAwv;vyHUMMAwAMAArAVV;.e8B8B8B,;vlyyLyLyryLyv;;vmXrVryLyrVyyl;vvVlVvylyyLMNhhXhllvlvlvv;vvyvv;v;;;;;;lyvv;;vv                                                                                                                                                                 
                                                         .Tv ;hLYyVrLTkAvyyATNTmktH4v;,:vUcC2qcVw0O9kryAMmj#@pwlvvLHtc4TmCcHHkAwrwrwwy;;AcMTrAATLryl;;vrh8B8B8B8kLv;,vvyVyyryAy;:rkrVLyyyyVylyYv;vvYlylVyrhmHNVlvvvlvvvv;;;vvvvv;;;v;;vrvv;;;;v,                                                                                                                                                                
                                                         pklVyrVyVyyLMkylVrLNhw4mHjjtkAv;.;vwHCHrAC9UvTrv:;yHp#tkyvvrXUAjttkHXUAwTTrNAv;HNMATAALLv;V#B8B8B8B8B8B8B8BeX;:vlyyLyAYvwwyyVyVyVyVVVyYv;vlylyVMUkryvvvlvvvVllvv;;;vvv;;;v;;vLVv;;:;;v;                                                                                                                                                                
                                                        yZyvvrVVVVyryMwMlyyLrhNHctjtmpjtcUv;,;vTXMwjtVc8Ocv:;vTc2bHrv;vrcjmmXUhwMwrArhyTUArTrArY;wB8B8B888B8B8B8B8B8B8BeY:vVyrVLrwVLlyyryyVVYVVrvv;vlVrhwrvvvYlyrNhTrMvlv;,vYvvv;;;;vyyv;v;:;;;v..                                                                                                                                                              
                                                        c2V;vVylyYyYyTwrVlVVyMkUct2c2cpC0CCchl;:vvvyUTAD88eHv:;v4mbpHVvYjcmUUNhAwATATrhMArArTrv;eB8B8B888B8B888B88888B888ev;lVyVyyyYyyryVVyYVVyyrv;vyrMlvvvlyrUUHNXNhAyv;,vVyvv;;,;vyVv;;;;,;;;;;;v                                                                                                                                                             
                                                        cbVvvyVVYyrLywrAYvVyVwXHm2tCqC0DpC2bcpckVv;;;v;wgBeBOmv;:vAct2kkccUH4XhhTXMhMMrArALrrv;8B88888B8B8ZpmccZB8B8B8B8B8Bc:lVryLVyyyVyVLVyVyyyrY;vvv;vvyLhmtkhvvvrMyv;,;VAvv::,vVyvv;;;;;:;;;;:,;l                                                                                                                                                            
                                                        tHyvvYyYyyLyryyLAvvYLLNktcjjqqOCDqO9#cHt0jUlv;;;jg9G6g@rv;;vNmjmkUUN4ANhAvvVhTTrArArvv8B888B8B8r;,;;;;::;;rB8B8B8B8Bc:YVyVyVVVVYlYyvVYLVwr;:;;YLNXHkwvvvvvv;vv;,;ly;:,;vyyY;;;;;;;;,;;;;;.:v;                                                                                                                                                           
                                                       ,UHVvvyVyVyVyVylrrAvvvLAkkHHpb00ZGG9eD9#jt0CjXy;;;qGOZZD#kLv;;VUjmmUkhAUjXv;vMwyyyAVl;8B8B888BV.;vv,:;;,;;v;:w888B8B8Bl;yVyVYYchYvvvVlyvyywv;YAhk4wlv;vvlvlv;,;;;:;,:;lYylv;;;;;;;;,:;;:;;.,v;                                                                                                                                                           
                                                       .NmAvlYyYyVyyyyVLrrkrv;lAwUccCZZCOOZ@9O6G9OO0qtNv,;O6DbCCZtwl;:;ycmcmU4OCwvvvrTyVrVy:jB8B8B8Bv,lv;v088B8j;;lv:V8B88888t:lyVVvNpkVlvvvVlllVLTrXhhyv;vvVvyMAyY:,,,,;;vYylvvv;;;;;;;;:,,;;;;;.,vv                                                                                                                                                           
                                                       ,rcAyvVlyYyYyLyVAVA4cmA;vlhkcct0O@Z@ZOOOD@#qDOO#0T;vCbC2CbZCcyv:;vUccAjbtAYv;YhVyVyv;B8B8B8BX.Yl;A8B8B8B88L;Vv,k888B888:vYyYvXtUyVVv;vVyVVLrwrY;;vVyyVmmMYv;..,;vYAVyvvvv;;;;;;:::;.:;;:;:::v;                                                                                                                                                           
                                                        yXUVYvYlVVyVVVAVyAUktckLv;lMmjCpCjjZ#Cqq#Cbp009O6ZHhtcjcttb#bMv;:;MANHUkAvv;vMMyrYvv8B888B8;;vvv8B8B8B8B8B;vV:YB8B8888;vyVYvykHYyyV;;vYlyyyv;;vYLVrkcUl;;,.,;vwU4yVvvvvvv;;;;;;:;.::;,;::,;v,                                                                                                                                                           
                                                        VTmVvlvVyyYVlyyVLwNkmmkcHwvvvwkcmj2bt2j2q0t22ptCpqZ2UkUkUmXc2qUv;;;ytcw4rvVv;TwALr;l8888B8B;;V;yB8B8B8B8B8vvv;v8B8B8B8;vvYvvlHUVVrrl;vvvvv;;vrrLLkcUv;;;.,;rhHcTv;vvvv;v;;;;;;::.,:;,;,;,:;v                                                                                                                                                            
                                                        ylHAvvllyVyVyVLlywXNHkU4ctt4V;vyUmmmtcpttjptbmtmkkmUHXUMhXUNkcDkv;:VDHHUHLlv;vhAMylv8B8B888;.vv;BB8B8B888m;vY.hB8B8B8p,vVlvvLt4lVykr;;v;;;vywrrhHAl;;;:.vLHkrv;;;;v;;;;;;;;:;::.,::::,:,,,v:                                                                                                                                                            
                                                        vVvklvvVlylVVyVyVwhUUUMUkmXtjkAlvYrUUHkcmckckHHcHHHUNUwwATwNAhkbwv;LC2HHkUvl;;vAwAl;B8B8B8B8:;vv;0B8B8B8L,vv.,B88888Bv;lvlvvyqLvvyHw;;;;vrMMVryyv;;;;;,vUHY;;;;;;;;;;;;;:;:;,,..:;,,,:,,.v;                                                                                                                                                             
                                                        ;VvwUvvllvVVyVyVrr4wUXkNkwUkjO0: ;;;vYTwUXkUkh4wXXUhUTwwNAMrwTNwmAvytbmmmkrvl;;lrrL;rB8B8B8B8.,;;.;yXhV,:;; ;B8B8B8Bj.vvlvv;ctvvlYUr,.;lUTylVvv;;;;;;:;vv:;;;;;;;;;;;:;:;,,.,.,:;;:,,,,.:;:                                                                                                                                                             
                                                         yvYMlvlvlvYlVYyrAh4wXU4kUXkceH. Yv;;:;;vVrMTXNXNhMhrArArMrArArwTkrTcckckHNyvv,;vrVl:b88B888B8v..,.. ...  ,@B8B888BC.vvvvv;w#rvlvvMY.:AkLlvlv;;;;;;;;;;;:;:;;;;;;;;;::,,....,::;::,:,..,.,.                                                                                                                                                             
                                                         YYvlvvvlvlvVYVyAMUMXT4N4hwhtGN  .v:v;v;;;;;vYrAwAwrNwwAMrArAyrAAyw4cXUUkhUUyvv,;vVlv,0B8B8B8B8Bbrv;;:;rqB8B8B888Bj ;vvvv;Vqkvvvlvy;:Akvvvv;;;;;;;;;;;;;;:;;;;;;;;: .   .,::;:;:;:,.,.,...                                                                                                                                                              
                                                         ;VvvlvvvlvllylryTTNw4w4h4M4c9M  ,vv;;;vvvv;,;;vlVyryArrLryArLyyrALrrAMNUk4hrYlv:;vlvv.UB888B8B8B8B8B8B8B8B8B8B8Br ;vvvv;YjUvvvvvvvvw4v;;;;;;;;;:;:;;;;;:;;;;;;;. .;;::;;;:;:;:;:: ....,.                                                                                                                                                               
                                                          yvlvvvvvYvlvlVrAhTMXXwhwXAt@L .:v;v:.   ,:v;;:;;vlyyryryLyLVLyyyyVyVLrAMwr4rllv;;vvvv.vG8B8B8B888B8B8B8B88888k..vvvvv;lpjvvvvvvvvyL;;;;;;;;;;;;:;;;;;;;::..      Al;;;;:;:;,::;........                                                                                                                                                               
                                                          vYvlvvvvvllYvLyrAMAhMXTNTwXOw  .:;; p8y,    .::::::vlyVylVvVYyVylyVyyyVrMwrryVlv;;;vvv:.;HB8B8B8B8B888B8B8OV..;vvvvv;L2cvvvvvvvvvv;;;;;;;;;;:;;;;;;:.            Y;;;::;:;:;,:,,::....                                                                                                                                                                
                                                           llvvvvlAVvvYyyyTAhAww4MAyhCt.  .,, 8B8B8BH:    .,,...vlLVYlYlyllVVVVlVYVVrLyyyll;;;vvvv;..vUt9B8B8B89mV;. .;vvvvvvvXC4vvvvvv;vvv;;;;;;;;;;:;;v;;.               v;:;::,:::::,:,:v: .                                                                                                                                                                 
                                                          .VvvvvvvVhyv;lyrrrrwLThrTGAtp;   .  j8B8B8B8B8l     .  .;YyyYYlVvYvVlVVllyVyyrYlYyv;;;vvvv;;,. ...., ..::;vvvvvvvvltjr;vvvvv;;;v;;;;;;;;:;;;::                   v;:,;::,:,:,:.,;v..                                                                                                                                                                  
                                                          lAvvvvvvvVNTv;yyALLrArwVhBhr0v.   . ;B8B8B8B8B888y        ,vyyYvlvlvYvYvVllvYlYvYlVlv;vvvvvvvvv;v;;;vvvvvvvvvvvvlwc4v;vvv;;;v;;;;;;;;:;;;::            vZBe     :V;:;,:,:,:,:,.,v,..                                                                                                                                                                  
                                                          .HvvvvvvvlvLAyvyyyyrALyyY8jYpM:      8B8B8B8B8B8B8B8X.       ;vrVYvlvvvlvYvlvYvlvVvVYYvvvvvvvvvvvvvvvvvvvvvvvvlTUHvvvvvvv;vv;;;;:;:;;;:,           lZ8B8B84     vY::::,:,:,,.,.v;.                                                                                                                                                                    
                                                           L4vvvvvvvVvVr4VVyLVyyrv;B8;mkv.     v888B8B888B8B8B8B8H        vVyllvvvvvvvvvlvYvlvYlVlVvvvvvvvvvvvlvVVVlyyAMNrYvvvv;v;v;;;;;;;;;;:,          .CB8B8B8B8B      Av,.,,:,:,,,,.;v,.                                                                                                                                                                    
                                                           .MNvvvvvvvYvvyyYylyyyYY:8BvVtY:      e88888B8B888B8B8B8B8A       ,lVyvvvvvvvvvvvvvvvvvllyVyllvvvv;vYyVrAwArylvv;v;v;v;;;;;;;;:;;;          leB8B888B8B8B8     .L; v;,,,,,,,.;v,                                                                                                                                                                      
                                                            vylvvvvvvvlllvVlVVrVVV;U8#:kt;,      B888B8B8B8B8B8B8B8B88e.       ;VVVvvvvvvvvvvvvvvvvvllyyryryArAyrVyvv;vvvvv;v;;;;;;;;:;;;,         A8B8B8B8B8B8B8B8.     Vy, jv :,,,, :v: .                                                                                                                                                                     
                                                             vvvvvvvvvvlvVlYYylLyLvvB8;vcX;,      B8B8B8B88888B8B888B8888t       .vVYYvvvvvvvvvlvvvlvvvvvlvVvlvvvv;vvvvv;v;v;v;;;;;;;;;.        U8B8B8B8B8B888B8B8H     ;M; ,8: ,,.,.;v:..                                                                                                                                                                      
                                                              yvvvvvvvvvvvlvYYyYyvv,BB8 vHM;       B8B8B8B8B8B8B8B8B888B8B8B:       :vyYvvvvvvvvvvvvlvvvvvvvvvvvvvvvv;v;v;;;;;;;;;;;;        k8B8B8B8B8B888B88888e      yl. tG .,...:;,..                                                                                                                                                                       
                                                              ,yvvvvvlvvvvvvvllYYlv;v8Bt Vmy:       G888B8B8B8B8B8B8B888B8B8B8m        :;vvvvvvvvvvvvvvvvvvvvvvv;v;v;v;v;;;;;;;v;:        T8B8B8B8B8B8B8B8B8B8B8B      yY; .Bm ,.,.,,,..                                                                                                                                                                        
                                                               ,yllvvvvvvvvvvvvvYlYv,C8BN yHA;       :8B888B8B8B8B8B8B8B8B8B8B8B         Vv;vvvvvvvvvvvvvvvvv;v;v;v;v;;;;;;;;:,        v8B8B8B8B888B8B888B8B8B80      yVv  D8. ,,.,....                                                                                                                                                                         
                                                                ;rllllvlvvvvvlvlvlvl;,B8BH vUNv.       vB8B8B8B8B8B8B8B8B8B8B8:         kwvv;v;vvvvvvvvvvvvvvv;v;;;;;;;;,,,,        ,#B8B8B8B8B88888B8B8B8B8B8;     .yYl  r8h .,.,....                                                                          .:.....:                                                                                        
                                                                 ;AYYvvvlvvvvvvvvvlvv,;B8BC ;rhV:         ve888B8B88888B8BO;          ;qUvvvv;v;;;;;;;;;;;v;;;;;;;;:;,:,,,::       ,8B8B8B888B888B8B888B888BA      :yYv. ;8B  ,......                               v      ;UTr                G8v              888B8B8B8                                                                                       
                                                                  :AVYvvvlvvvvvvvvvvvv,;88B8,.vUhv            ;Uqe9ODbl              4jy;vvvvv;v;;;;;;;;:;;:,;::,:,;:;:;;;;v:        8B8B888B8B8B888B8B8B8v       vyvv  ,88, .......                           eB8mGB8B.   8B8B                8Bk                                                                                                              
                                                                   .LyVvvvLyYvvvvvvvvvv.;B8B8y .VwTv.                              vCUv;vvvvvvvvvvvvvvv;v;v;;;v;v;v;vvvvv;vvY:        ,B888B8B88888B8BBM        ;LVY;  v8Bv ..... .                            98B8B8B6B8B8B8B8B8B8L           B8y              rB8B8B8H          lBw                                                      G8B;                 
                                                :                    rrVlvvVVTLv;vvvvvvv,,B8B8Bv ,vNNr;                          lUUv;;vvvvvvvvvvvvvvvvvvvvvvvvvvvvv;vvv;vvvvy;          vp888B8B8#U;         ;yVVv,  U8Bv  ....                               :B8B8B  8B8B6OeZ8B8Bv           8By            B8B8clvUB8B8v       A8c                                                      ;BB                  
                       vwMMyylvv:.            v0hv;                   lyVvvvlvrryvvvvvvvv: H8B8B8l  vrUhTv;.                 ,lktMv;v;vvvvvvvvvvvvvvvvvvvvvvv;vvvvv;v;vvv;v;;;lv                            vrrlY:  ,68B;  .. .  ..                           8B888B8  B8B8CODCB8B8v           B8V          v88r         B8B      yB4                 ,.                 ,       ,,                             
                      rcAMylvVvVvvv;,         cDVv;;.                  yVVvlvvvlllvvvvvlvv;.;8B8B8Bb;  ;YwwhwAv;;;,.,;;;vM4mj2Xv.vyv;vvvvvvvvvvvvvvvvvvvvvvv;vvv;v;v;v;v;v;v;vvvvv                       ;yTyVl;   4B8e,  ..   .,.                            v8B;B8Bj 8B8B8B8B888Bv           8By         .B8            68D     l8T    8B8       8B88888O B8     GB.D8B8B8v .8B8888       88                  
                      ArkB8XYYVvYvvvvv;,.     jtv;;;;,                  yVYllvvvvvvvvvvvvvv;: vB8B8B8Bt;   .;;lVAMhwHhXwwrV;:  vgBy;vvvvvvvvvvvvvvvvvvvvvvvvvvv;v;v;;;v;;;v;;;;;;vlv;                .vyryVvv,   V888c    .   :,                                  8B8B B8B8;vv;0888;           B8y         B8:             B8     V8l  B8B;      r8BV    ;888B     98B8    B8B8j   .8B      8B                  
                      vyU8B9rVYVYVvlvvvvvv;.  m#rv;;::,                  ;VYvlvvvvvvvvvvvvvvvv. vGB8B888B80Xv;            ,vU8B8D;;vvvvvvvlVlVvlvvvvvvvvvvvvvvvvvvvv;v;;;v;;;;;v;v:,;vvv;:.     ,;;Vyylv;;.    UB8Bev  . .   ,                                  B8B8B8 8B8B8B88888B,           88r         8B              8B;    v8rUB8:       v8B        H88     #BB      B8      L8Y     B8                  
                      ,YVc8BCLylVVVllvvvvvv;;,cOkvv;;,:,                  :vVvlvvvvvvvvvvvvvvvv;: :X8B88888B8B8B86GGgB8B8B8B8Bp,.;vvllyVrVLyyvVlVlVvlVyVrVyVyYvvvvvvvvv;v;;;v;;;;y9m;  .:;vvvvvvv;;;;.      MG8B8By   . .                                     8B8B8B8B B8BBL888 lB8w           B8y         B8              B8     vB8B;         8B          8B     O8,      8B      vB4     8B                  
                       vvyj8B2AyvvvvvYvvvvvv;;H9XYvv;:,:.                  .vyvlvlvvvvvvvvvvvvvvvv:. ,yOB8B8B8B8B888B8B8B8BU:..;vVlrATMwrAAyyVlyyLyATMLyvvvlYrrAyVvvvvvvvv;v;v;;;:;@B8by,              :YCB8B8BDv    . .                                      B8Blv8B8 8B8c B8B8B8B8           8Br         vB8            p8B     v888By        B8          B8     98.      B8      v8r     B8                  
                       ,Vvyj8BpAt2A;;vlvvvvv;;tjAVYv;:,...                   ;yvvvvvvvvvvvvvvvvvvvv;;;. .:vAt@BB8B8eGbkV;..,;vLrwUU4UXkAXhryAyyrMrV:            ;lryyllvvvvvvvv;;;: ,U8B888eqHkhNUcCBB8B8B88#l    . .                                          k  vB8B B8B#  B8B8B             88V          UB8.         B8B      VB; 0B8;       B8        B88     G8,      8B      V8w     88                  
                        vVvLteejp888cv;vvlvv;vtcyVll;;,,,,                    .llYvvvvvv;v;vvvvvvvvvvvvv;;,.         ..:;;vyrUkHmcUkUUMrAwyArhMy.                  .vrvvvlvlvvvv;v;;,  .vtB8B8B8B8B888BOkl,      . .                                           ,  B8B8;8B8B8B8B8B8BO;          8By            B8Bel,.vC8B8r       M8c   B8Bv      B8BV  v8B888     8B;      B8      M8c     88                  
                        .VlvVcB6Hk8B8BU;vlvv;v0HyYVvv;:.,;,                     ;VVvlvvvvvv;v;;;vvvvvvvvvvvvvvv;;;v;vvvvVVTTkkcmtkmX4wXhUTUXUX4vvYXUHkmHHkHUkU4ry;:  ,YYlvlvlvlvlvv;;::.     .::v;;,.         . .                                              B8B8B8B8B8B8B8B #888BA          88X              9B888B8Bv         :Bv     8Bv      ;888B8B: 8@     A8       89      ;B;     82                  
                         ;VvvrbeG4A8B8Bp;vvv;y2mVVvlv;::.vY.                      ;rVlvvvvvvvv;v;v;vvvvvvvvvvvvvvvvvvYYyVMNmUcccHHUHUUUHUUUmmtmtjjmHkHUHkHUk4kUUkHHmAlYyvlvllYlYvyVllvvv;;,.           . ... ...                                               cB8B8; B8BCl.     GBV           eBv                                                                                                              
                          vlvvrCB9A28B88t;Yv;AZkYvlvv;;:.,M;                        vVrVYvvvvvvvvvv;v;v;v;v;v;vvvvvvvvyyhXttqq0cckcHcUHHkktUy;,                .,.,;vyArrVVvllVvYyALATwlv;;::,,,:,,.,.,......                                                                                                                                                                                                   
                           vlvvwDB2AB8B88r;v;A#Hvvvlvv;:..Vw.                         ,vvVvvvvvvvvvvvvvvvvvvvvvvvvvYlVVLAHjqZZpjUkUHkHUkmH:                            .;yVlvvvlvyyrAALv;;:;::,:,:,,...... .       .                                                                                                                                                                                            
                           .lvvvkGeX#B8B8ev;vA0Avvvvv;;:: vc;                            YyvvvvvvvvvvvvvvvvvvvvvllYYVvVlyrwktjZ@0ckXHkcjY       vU#888888B8B8B89pr;       vVvvvvvvLyrlv;;:;;:,:........ .         .                                           NLryyyryryryLyLyryrVyVryLyryyyrLryyyyyLyryryryryryryryryryryryryLVLVLyryryryryryrLryLyryLyryryryryryLyryrLALryLyryryryryLyryLyXv                  
                            ;lvvl2eCc888B8N;vTjLvvvlvv;;,.:pr                             :lllvvvvvvvvvvvvvvvvvvvvvvvvvlvlvyyXHCCOZpkUcH    v8B8B8B8B8B8B8B8B8B8B888B8;    ;vvvvvvYrvv;;,::,.:..                 .                                            vvvv;v;vvvvv;v;vvv;v;v;v;v;vvvvv;v;v;v;v;v;v;v;v;v;vvvvv;vvvvv;v;v;v;v;vvv;v;v;v;v;v;v;v;v;v;v;v;v;v;v;v;v;v;vvv;v;v;v;v;vvv;vv:                  
                            .vvvvM9OpB888BD;;Apyvvlvv;;;;.,tt.                               ;VYylvvvvvvvvvvvvvvvvvvvvlvvvvvlYAAHc0q@jjH   B8B888BB9D2@qZ@9OeB8B8B88888By   vvvvyVYvv;;:;::.,.. .               ,,,                                                                                                                                                                                             
                             vvv;vmebBB8B8Bv;ywVvvvvvv;;:.,jb;                                 .;vvvvvvvvv;vvv;vvvvvvvvlvvvlvvvyyNUc2#qOA  t                           vA   vvvyVVvv;;::.,.... .             ..,,;:;:,                                     
                             vlvvvMOG#8B888r;lyvlvvvvv;;;.,m#v                                    :;;:;;;;;;:,;;;;vvv;,vVvlvvvvvYvVyN4cqOU                                 ;LryyY;;,:,,.,....               ,;;,,.::;;;::.                                 
                             vllvvV09#8888Bj;lvlvvvvvv;;;.;b#w.                                  ..,. .. .:;:,   ,:;;v, ;YvvvVYlvvvllVrNm#ObAyLyLhAyvv.      ,,::;.      ;VTAy;;::.,,,.... .           ,...,,:,:.,.:,;:;;;;;:.                             
                             vyvlvVme08B8B89vvlvlvvvvvv;; vOgk,                              ,vvv;;,:.,..  :;98H; ,;;;v. :vvvvvlVvlvvvvvllrXmjtctHkwrYlvv;;vvvvvvvvvlvvvlvv;:...,.,., ...             ..,...,.,.,.,,,,;:;;;;v;;;,                          
                             vylvvvHG9O888B8l;vlvvvvvv;;: Veej, .                          ;lyvvv;;;:;,,,.   mB8B8XyVv;;:.,vv:  yVvvvvvvvvvvvvvvvvvvvvvvvYvvvvvvvv;v;;;;::,,.,.,.... .         .:.     . ......,.,,,.:,:,;;;;v;v;v;,                       
                             vLvVvvw96jB8B8Bk;vvlvvvvv;;: mB9j; ,                       .vrVYvv;v;;;;;:,;,   ,8B8Bb.vvv;v;,:;   ,Yvvvvvvvvlvv;v;;;;;v;v;v;v;v;;;;;;;;:;,:,,,,.,..         . ..:,,.        ....,.,.,,:,:,;:;;;;v;;;vvv;.                    
                             lVYllvhZ8tCB8B8g;;lvvvvvv;;..b8Z#v ..                   .vVyrYYvvvv;;;;,;;;:;,   q8B80;vlvvvv;;;    ;vv;v;vvv,vv;;v;;;;;;;;;;;;;;;;;;;;,:,,,......,       ;Z8C, ,.,,,       . .....,,,:,::;:;;;;;;; .;vvvv;                   
                          wBBylvVllr@BZyBB888y:vvvvvvvv;,:OBZZy ,,           .,.,  vgqyvyyYvvvvvv;;,:;;:;;;   lB88jvyVLVyvv;vv: :::;;;vv;   ;;;;;;;;;;;;;;;:;;;,:::,,,,...... .        ;DegY .,.:,.       . ....,.,,::;:;;;;;;v;   .;;vvv,                 
                         UBeyvvylYvXOBOTl8B8B9:vvvvvvv;v,;98D9w..:        vyrMVV:vB8Hvv;;VrYvlvv::,::Yv;,v;.  ,888hlLryrVLyYvlvrYv;;,;,      ;;;,;:;:;:.  ,;,,,:.,.,               ,..  Aeec .,:,:,.        .,.,.,,:,:...;;;;;;;vl;, .,;vv,                
                         OgrvvylVvYhOB9klV8B8B4,vvvvvvv;,;GB9Om..,, ....:;vvv;, MB8v;;;;vvrAyv;,;;;:;yA;;;, . .B8BryNTMAAyAyylYyMvv;;:        .::,:,:.     .:.... .             ..,.,,. V68D; :,:,::.       ....,,,,::, ,,..,.;ymU4Lv;. ,;;                
                        ;92YvlVVlYYkOBOjwYv#B8By,vvvvv;v.;G8GGc, ,..;vvvvv;;:. @B8,:;;vlllvw4;:;;;;;;;;:..  .  8BblXw4h4MAyryVvywrvv;v;;@8jy,    ...         . .           ..:,:,,.,,:. v8BB; :;::,;,          ..,,::;;;lrYvvYwcHmUkNwyv.                  
                        UOM;vvyYVlLUDgDckAY;Ve8BT;vvvvvv:,DBeGq:.,,..,vvv;;;: 8B8Z .;vVlYvvvvvvvv;;;;::,:.  , ,B8;ywXTUXwrrVVvvVXwyvvvv;;D8B8BeT:     ..,.          ....,,;,,,,,:.:,:,. VB8G; ;:;,::;,          ..::;,,;rAUUmtjtjttctcmNrv;                
                        Z@vvvVYyYVrHZeZjkUrV;,vqkvvvvvvv:,b86eOv :,,...;;;:; 0B8B8BU;Vllvvvvvvvv;v;;;;:;:, .. ;8w,yXwNArrTyryYvvyXAyvlvv: ;8B8B8Bc  :;;;:;;;...,.,:;:;:;::,:,:,:,:,:,:. w8B0 .,;;;,::;,            .. .YMhXmccNryyrUHjttmtmA               
                       YgUvvVYVVVlAHD9OpmXNrVv;:;vvvvvv;; 48BOgA..:,,. ,;;;; g8B886yvvVvvvvvvvvvv;;;;;;::. .,.yt.:YAXMrrMrArLVYvAXXyYvYvv;. XB8B8BG. ,:,;,:,;:;:;,;,:,,::::,:,:,:,::::. j88l ,::;;:,:,;         .,::vvVLNkcUV;, ...,;l4cpcttN              
                       j0V;vlVyyYVrt@O#bHHwAllvvvvvvvvvv;,:eBGep,.,:,, ,;vvv;;U8B0lVllYVvvvvvvvv;v;v;;v;:, .,:v;,;;yVTM4AwAMLyYvYUXAyyYvvv;; ,q8B8B8; ,;;;,::;:;:::::;,;,;:;;;;;;;;;:: ,989. ;:::;:::;:;       .,;;vvVLXUcM; ..,,:,,,:;VUpHccA             
                       @tv;vyVlVrYMH#ZCpcU4ryvvvvvvvvvv;v;.r8BeOv :,,,:,:;vv; AOYvvvVlyVlvvvvvvvl;v;;;;;v:  ,,:,;;vvvrXMArArAyyvvhkMNryvlvv;;. v8B888l ,;;;;;:;;;:;:;;;:;;;;;:;;;;;;;: ;8BX ,:;:;:;,:,;:,       .,;;vVwUtN,  ...,.::;;;;lXjccmr            
                      ;qMvrpcr;vYAUp0ZpjHkhrVlvvvvvvvvvv;v,.c86eU..:.:;;.:;;.mBv;vlyyryrvvvvvv;:;rv;;;;;;v:  ,,;;;;vvLLMAhMAyryyvykXTwyrVYvvvv;  pB8B8l ,;;;;;;;;;:;;;;;;;;;;;;;;;;;;; yB8: ,;:::;:;,;::,        :;vvLwHm;     ..,:;;;;v;vXcHcmv           
                      vAvy6BHv;;lrkm2tcUkXULyvlvvvvvvvv;v;;.:b8O#v .,:;;,.; UBlvyrwrrVVllvvvv;.  ,lvv;;;,;v,  ,,;;;vVVyyTAwrMyLyvvhU4wMVLvlvvvv;: vB8B8v :;;;;;;;;;;;;;;;;;;;;;;;;;;;; r8e  ::;:;:;:::;::,      .,;vyMUmh       ,,::;;;;v;yUcUmU,          
                      ;VvCBZv;;vvMkmHc4XTwMrlvvvvvvvvv;vvv;;.:t90M...;;;,. y8AvryylyyMAryAyy,.:;;..,::.. ,;v,..,:;;vvvVyyMAArryyYvvUhhLryylYvvvv;; .G8B8:,;v;;;;;v;v;v;;;v;v;v;;;v;v;;.yBO .::;v;;:;:;;;,.       :;vrw4tv      ..,,;:;;;;;vhmkXml          
                     .AvhBBT;;;vyrUHkNXTTAAVlvvvvvv;v;v;;;;;; ,wCH; ,,;;:  BB;yrrv,.;ykAy;;. ;LNmH4r;.:;; .;: ..,:;vvvVYrrhrrLylyvvyHwMyLVVllvlvvyl..q88O.;;v;vvv;v;v;v;v;v;v;vvv;v;;;.;8q .;,,vV;;::;;,;       .:vyUwUk,       .,::;:;;;;lUjHUUA.         
                     vyvUBDy;;;vyMANTwrAyyVllvvvvvvv;v;v;;;;:; .lml,.,,;. y8CvVAV;.;cOtkr, ;AHmcmmHcUwLrYv.    . :;vvlvYrwAALryyYVvvwHTMyyYVvVv;;mrv:.O8Bv,vvv;vvv;vvv;;;vvvvv;v;vvv;v:.66 :;: .;v;:,;:;;,       :;rMMhL       ..,,;:;::.;TbcHUUA:         
                     vyvhGOV;;v;LyyVryyYLyVvlvvvvvvvv;;;;;v;;;:..;l:..:,  Aecvrrv,;c8B8B89Z99Z@2pcHkHkkwAyV     .,;;vvVlrMMMAyyVylYvV4UArlYvllv.;tkvv,:B8r,vvvvvvvvvvvvvvvvvvvvvylvvv;;.HB;,;,,..,;,:,::;:      .,;vlly,        ..:::,. ;MbtckUMA,         
                     VVvy0CV;;;vlryrVyYVVVvlvvvvvvvv;v;;A4lv;:,:,,.,,,,, yAvvLMA;,48BBeeGeeGDDCptcUHXUUHUkMV    .::;;vllvrAArrVyVyllvyhXVyvylv::;Uclvv.OBX.vvvvvvvvvvvvvvvvvvvvllAryvvv;;9l.::;;...,:;:;:;       :;;;;.        ..,,,. .vm#pjtmTrv.         
                     lyvYrkv;;v;lrryyvYYyvlvvvvvvvv;v:;kbyv;;::::,,.,.. vTrVyMHv.vG88ee6e6e9Oq0jjHhv;:;;ymbmr.  ..::;:vYVVArAyLVVlYvlvlrMVlv;:;;;vtVvv,N8w:vvvvv;vYvvvvvlvvvYvlv;:vAyvv;;vv:;,;;;,;;;:;,::,                    ....  ;UC#tjjjwrYv          
                     VYvvyhV;;;vvyyyvyrLvvvvvvvvvv;v;;lCUv;;;;,:,:.,., :MTYyk#kv.CB8BBeBgeGGOZ@btU;  . ..;ACpX,    ..,;lVyrryylYYVvYvlvlvl;;;;vv;vhAvl;LBr:vvvv; ;VvvvvvvvvYlYlv:..;vrlv;;:;:,:;;;;;:;:;:;,                    .   ;TCCjkHkkwyvv,          
                     vVvvvyv;;v;vYVvmG@ky;;;v;v;v;v;;;rmr;;;;;;::,,.,  lhvVA6@; VB8BBeBeeee9G@@Cpl  ...   ,vj@m;   ..,;vllyVryVvVvlvlvYvv;;;vvlvv;VrvvvvAvvvlv;.,;VvvvvvvvVVLyrv. ..:;lvv;;:::;;;:;;:,;,;:,                      .ypObHh4wAVlvv;           
                     ;vvvllvvvvvvvvUpck4Yv;;;;;v;v;;;vMHv;;;:;;;:,.,   UV;:vv. ;B88BeBeB6eeeOOZ0jl ...     .;cqc;  ..;;vllvVVVvlvlvvvvvYlYvVllvvvvvLvvvv;vvvv;,;::;rvvvvvvvVyAy;,; ..,:;v;;;.;;;;;:;,,,:,;.                     ;XtjUhTMyyvv;;:            
                    .;vvvvYvv;vvvvALvvvvv;;;;;;;;;;;;vMV;;;;;,;:;,:,, ;tv.,0eC988B8e8BBeBee6eO9Z@w  .     ...;tbm,  .,;;lvllyvlvvvvvvvvvlvYvYvlvvvvvyvvvvvvv;,;;;:;YyvvvvvvvVrV.;Vv,..,,;;;,,;;;;:;;:  .,.                     ;TAAyyVylvvv;;,             
                   88@;vvvvlvv;vvVYvvvvv;v;;;;;;;;;;;vVv;;;;::::,:,:. ;@Y.,6B8B8BBe8B8B8BBeBgGG9q0v        ..:YHHV   :;;vlvVvlvvvvvvvvvvvvvvvvvvvvvvVYvvvvv::v;v;;:vlvvvvvvVvl;.vylv;. ..,..:;;;:::;,                         ;ylvvlvvvv;;,:               
                  vB84;;vvvvv;;vyvvvvvv;v;v;;;;:;;;:;vV;;;;::,:,,.,,, ,yl.,G8eeeBgB88B8B8BBge9G9O#b;        ..vyMv   .;;vvvvlvvvvvvvv;vvvvvvlvvvvvvvlYvvv;::vvvvv;;;vvvvvvvlVv,:Yvvvv;;...,:;;;:;:::,    ;h                   vvv;;;;;;::.                 
                   lMV;v;vvv;;vyvvvvvv;v;v;;;;;;:;:;;Yv;:;::,:,,.,..     vw96gge6Bg8B8B8B8BBee96OD00;        .;vv.   ..;;vvvvvvv;;:yvvvvvvvvvvvvvvvvvvvv;::vvvvvvv;;;vvvvvvv;,.vvvvvvvvv;v;;,::;::::..   29,                   ,,:,:,,..                   
                  .NbM;;v;vvv;vvv;vvvvv;v;;;;;;;;:::vYv;;::,,.,......    U60eG6gB6BB8B8B8e8BegGOO#Z@DT        ,,     .,:;;;;;;:.::;vv;v;v;vvvvvvvvvvv;;,:;v;vvvvvvv:;;v;v;;;,.;vv;vvv;v;;;;:: .::,:..   .ZB;                                               
                  ;rYv;;;v;v;v;v;v;v;v;;;;;;;;:;::,;vl;;,:,,.,.,.. .     .pOGg9e6ee8B8B8B8BBee9GDZ@ZqO@L       :     ...,:.....;;;:vvvvv;vvv;vvv;vvvvvvvvvvv;vvv;vv;;;;;;;::.:;;;vvvvv;;;;:;,   .,..    .e8r                                               
                  :vv;;;;;v;;;v;;;;;;;;;;;;:;:;:;:,;v;:::.,.... .         :pB9G9e6ee8B8B8B8BB9O#DqCCCqDO@w;.  .;            ,:;;;;;;v;v;v;v;vvv;vvvvvvvvvvv;vvv;v;vv;:;:;,. :;;;;;;;;;;;;;;,.            H8b                                               
                   ;;;;;;;;v;v;;;;;;;;;;;;;;:;:::;;v;;,,...... .           ;eBG69G96eBB8B88BgG#Dq@2btpbZOGZCw;:v,        ,,::;;;;;:;;;;v;;;v;;,:;v;vvv;vvv;v;v;;;v;v;;:;:,.;;;;;;;;;;;;;::::           . .p8v                                              
                   .;;;;;;;;;;;;;;;;;:;;;:;,:::,:,;;;,,.,.. .               L8eO9O9O69G96GGOODZq#CCctmtjZZ@mHUVYl   ....,,;:;:;:;:::;;;;;;;:,.,:;vv;v;vvv;;;v;;;;;;;;;;;;;;;;:;;;;;;;:;::,:.          .,, ,CZ.                                             
                    .;;;;;;;;;;;;;;;:;;;,:,:,,,,.,.,.... .                   OB9OO9GO9O9OOOOZZZDb2cckmUkUkUXryvly    ....,,,,:,;,;,:,::;:,..,;;;:vvv;v;;;v;;;;;;;;:;;;;;;;;;;;;;.,:;::,:,:.          .,:,. ,Ur                                             
                      ,:;;v;;;;;;;;;;;;::::,,.,.,.. ...                       ee9D9O9DOZO#Z@ZbqC2tmkk4XTNAAyyv:ym;  . ..,.,,,,:,;,:,,.....:;;;;:;;v;;;;;v;;;;;;;:.;;;;;;;:;:;:;,  ,::.,,...        .,,;:,.. .;.                                            
                      :rVVvv;;,;,;:;:;::,:,,,,......                           9e9qZZZZ#qqbCtjtc4HkUMTAALrYYv..c#r   ....,.:,:.,,;:;::.,,;;;:;:;,;;;;;;;;;;;;;,. .;;:;:::::;:;.    ..,....       ..,:;::,,.. .                                             
                      v4Alv;;::::,:,;,:,,,,.. ... .                             pO@CCb0C0CbcmHHUU4XATrryVlv;, HeBc  ..,.,...,,:,:,::;:;:;:;::::,,.;;;;;;;:;:,.    ;;;::,;::,:.                   .;;;,:,:,:,...                                            
                       vvv;::;:;:,,,.:,,.. . . .                                 r2C22cjctmmU4h4TMrAVVlvvv;. LB8B9..,:,,,:,,,,,:,:::,:,,,:::,,,,...::;:;,,.    .  ,;,:,:,,,.                  .  ::;::,;,,,:,,.                                            
                        ;;;,:,:,:.,,,...                                         ,rAUUckHXXXXANAryylvvv;:. .;y8B8C.,;::::,:,,.:,:,:,:,:,,,:,,,,,... . .       .    ,:,,.,..                ..,:. .:::,;,:,  .,.                                            
                          ,,.......... ..... ..                                  :v:;vlyyyVLVVvvvvvv;,    vXvvB88w ;;;:;,;::,:,,,,,,.,,,.,.,.,......               .,:..                ...,,:::.  :::,;,,   ...                                           
                                ... .:;;;;;;;v;.                                  .;vvvvvv;;;.,..      :AmjUV;e88:,:;:;;;:;::::.,.,...,........ .                   .               . ,.,,:,:,:,;,. ..:,,.... ..                                           
                                   .vyYVYyV4mT;.                                   lvvvvvVyryyvvYVVAhkUC0UAMvvB8V :;;;;;;;;;::,:,,.... .                                         ....,,,,,,:,;,::;;:.. . .,:.. .                                           
                                    vLAAwrUUw;.                                   :YlvvvlvyLXUctb#Z0ZcXcjTAY;j8k.,;;;;;:;;;;;::,;,.                                           ....,.,.,,,,:::,;,:;;vv;;::,:,,.,.                                           
                                       ..,.                                       ;VvlvllyVyAXXcmmpCtcAkckrlYBU.;v;;;v;;;;;;;;,;:,                                           ...,:.,,:,:,,,:::,:,;;;;v;;::,,.,..                                           
                                                                                  vvlvlvYyyyMrHccmjmckMwjkTVyv;;;;v;v;v;;;;;;:;:, . .                                       ....:,,,:,,,:::,:,::::;;;;;::,:,,..                                            
                                                                                 ,vvvlvYlyyArwXmHjtbtHMXcHALlv;vvv;;;;,vv;;;:;,,......                                       ..,,,,:,,,:,:,:,:,;:;:;:;;:::.,                                               
                                                                                 ;YvvvlvyYyrhw44HHjpckhLkkhVYvv;v;;::.;lv;;;;,,.,,,,,..                                       ....:,:,:::,;,;:;;;:;:;,,.                                                   
                                                                                 vvlvlvllyyrTUNXkjcbmmhyAkMrlvv;::,::;;v;;;;,,.,,,,,...                                                . .....                                                             
                                                                                 vvvvvlvVVyrN4HUXc2cmUXAyyAYv;,,;;;;v;vv;;;,..;;;,,....                                                                                                                    
                                                                                 vvvvvvYlyLMMkUjcjccmH4hyyv;,,:;;vvvvv;;::,:;v;;,;,...                                                                                                                     
                                                                                 :yYlvvvYVrAUUmHjCtkHXUwTVv;;vYvvvvvvvv;;;;;;;;::,,.. .                                                                                                                    
                                                                                  ,VVVvllYyTAUctcbjtcHUkNTLMyyvv;vvvvv;;;;;;;;:;,,....                                                                                                                     
                                                                                    ;vAyyVyyrTHmtm2cmUkhwAAryvvvv;v;v;;;;;;;;,,.,....                                                                                                                      
                                                                                       ;vyyATXhkmjHkXUU4rAyylvvv;v;;;;;;;;;;:;::,.                                                                                                                         
                                                                                           .;vyyNUcHHkkAAMAyVvvvvvv;v;v;v;;::..                                                                                                                            
                                                                                                  .,:;;v;vvv;v;;;;;:,,                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                
*/

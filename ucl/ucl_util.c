/* ucl_util.c -- utilities for the UCL library

   This file is part of the UCL data compression library.

   Copyright (C) 1996-2004 Markus Franz Xaver Johannes Oberhumer
   All Rights Reserved.

   The UCL library is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 2 of
   the License, or (at your option) any later version.

   The UCL library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with the UCL library; see the file COPYING.
   If not, write to the Free Software Foundation, Inc.,
   59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.

   Markus F.X.J. Oberhumer
   <markus@oberhumer.com>
   http://www.oberhumer.com/opensource/ucl/
 */


#include "ucl_conf.h"


/***********************************************************************
//
************************************************************************/

UCL_PUBLIC(ucl_bool)
ucl_assert(int expr)
{
    return (expr) ? 1 : 0;
}


/***********************************************************************
//
************************************************************************/

/* If you use the UCL library in a product, you *must* keep this
 * copyright string in the executable of your product.
.*/

static const char __ucl_copyright[] =
    "\r\n\n"
    "UCL data compression library.\n"
    "$Copyright: UCL (C) 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004 Markus Franz Xaver Johannes Oberhumer\n"
    "<markus@oberhumer.com>\n"
    "http://www.oberhumer.com $\n\n"
    "$Id: UCL version: v" UCL_VERSION_STRING ", " UCL_VERSION_DATE " $\n"
    "$Built: " __DATE__ " " __TIME__ " $\n"
    "$Info: " ACC_INFO_OS
#if defined(ACC_INFO_OS_POSIX)
    "/" ACC_INFO_OS_POSIX
#endif
    " " ACC_INFO_ARCH
#if defined(ACC_INFO_ENDIAN)
    "/" ACC_INFO_ENDIAN
#endif
    " " ACC_INFO_MM
    " " ACC_INFO_CC
#if defined(ACC_INFO_CCVER)
    " " ACC_INFO_CCVER
#endif
    " $\n";

UCL_PUBLIC(const ucl_bytep)
ucl_copyright(void)
{
#if (ACC_OS_DOS16 && ACC_CC_TURBOC)
    return (ucl_voidp) __ucl_copyright;
#else
    return (const ucl_bytep) __ucl_copyright;
#endif
}

UCL_PUBLIC(ucl_uint32)
ucl_version(void)
{
    return UCL_VERSION;
}

UCL_PUBLIC(const char *)
ucl_version_string(void)
{
    return UCL_VERSION_STRING;
}

UCL_PUBLIC(const char *)
ucl_version_date(void)
{
    return UCL_VERSION_DATE;
}

UCL_PUBLIC(const ucl_charp)
_ucl_version_string(void)
{
    return UCL_VERSION_STRING;
}

UCL_PUBLIC(const ucl_charp)
_ucl_version_date(void)
{
    return UCL_VERSION_DATE;
}


/***********************************************************************
// adler32 checksum
// adapted from free code by Mark Adler <madler@alumni.caltech.edu>
// see http://www.cdrom.com/pub/infozip/zlib/
************************************************************************/

#define UCL_BASE 65521u /* largest prime smaller than 65536 */
#define UCL_NMAX 5552
/* NMAX is the largest n such that 255n(n+1)/2 + (n+1)(BASE-1) <= 2^32-1 */

#define UCL_DO1(buf,i)  {s1 += buf[i]; s2 += s1;}
#define UCL_DO2(buf,i)  UCL_DO1(buf,i); UCL_DO1(buf,i+1);
#define UCL_DO4(buf,i)  UCL_DO2(buf,i); UCL_DO2(buf,i+2);
#define UCL_DO8(buf,i)  UCL_DO4(buf,i); UCL_DO4(buf,i+4);
#define UCL_DO16(buf,i) UCL_DO8(buf,i); UCL_DO8(buf,i+8);

UCL_PUBLIC(ucl_uint32)
ucl_adler32(ucl_uint32 adler, const ucl_bytep buf, ucl_uint len)
{
    ucl_uint32 s1 = adler & 0xffff;
    ucl_uint32 s2 = (adler >> 16) & 0xffff;
    int k;

    if (buf == NULL)
        return 1;

    while (len > 0)
    {
        k = len < UCL_NMAX ? (int) len : UCL_NMAX;
        len -= k;
        if (k >= 16) do
        {
            UCL_DO16(buf,0);
            buf += 16;
            k -= 16;
        } while (k >= 16);
        if (k != 0) do
        {
            s1 += *buf++;
            s2 += s1;
        } while (--k > 0);
        s1 %= UCL_BASE;
        s2 %= UCL_BASE;
    }
    return (s2 << 16) | s1;
}


/*
vi:ts=4:et
*/

/* ucl_str.c -- string functions for the the UCL library

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

#undef ucl_memcmp
#undef ucl_memcpy
#undef ucl_memmove
#undef ucl_memset


/***********************************************************************
// slow but portable <string.h> stuff, only used in assertions
************************************************************************/

#if !defined(__UCL_MMODEL_HUGE)
#  undef ACC_HAVE_MM_HUGE_PTR
#endif
#define acc_hsize_t             ucl_uint
#define acc_hvoid_p             ucl_voidp
#define acc_hbyte_p             ucl_bytep
#define ACCLIB_PUBLIC(r,f)      UCL_PUBLIC(r) f
#define acc_hmemcmp             ucl_memcmp
#define acc_hmemcpy             ucl_memcpy
#define acc_hmemmove            ucl_memmove
#define acc_hmemset             ucl_memset
#include "acc/acclib/hmemcpy.ch"
#undef ACCLIB_PUBLIC


/*
vi:ts=4:et
*/

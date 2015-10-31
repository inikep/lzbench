/* ucl_ptr.h -- low-level pointer constructs

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


/* WARNING: this file should *not* be used by applications. It is
   part of the implementation of the library and is subject
   to change.
 */


#ifndef __UCL_PTR_H
#define __UCL_PTR_H

#ifdef __cplusplus
extern "C" {
#endif


/***********************************************************************
// Integral types
************************************************************************/

#if !defined(ucl_uintptr_t)
#  define ucl_uintptr_t     acc_uintptr_t
#endif


/***********************************************************************
//
************************************************************************/

/* Always use the safe (=integral) version for pointer-comparisions.
 * The compiler should optimize away the additional casts anyway.
 *
 * Note that this only works if the representation and ordering
 * of the pointer and the integral is the same (at bit level).
 *
 * Most 16-bit compilers have their own view about pointers -
 * fortunately they don't care about comparing pointers
 * that are pointing to Nirvana.
 */

#if (ACC_OS_DOS16 || ACC_OS_OS216 || ACC_OS_WIN16)
#define PTR(a)              ((ucl_bytep) (a))
/* only need the low bits of the pointer -> offset is ok */
#define PTR_ALIGNED_4(a)    ((ACC_FP_OFF(a) & 3) == 0)
#define PTR_ALIGNED2_4(a,b) (((ACC_FP_OFF(a) | ACC_FP_OFF(b)) & 3) == 0)
#else
#define PTR(a)              ((ucl_uintptr_t) (a))
#define PTR_LINEAR(a)       PTR(a)
#define PTR_ALIGNED_4(a)    ((PTR_LINEAR(a) & 3) == 0)
#define PTR_ALIGNED_8(a)    ((PTR_LINEAR(a) & 7) == 0)
#define PTR_ALIGNED2_4(a,b) (((PTR_LINEAR(a) | PTR_LINEAR(b)) & 3) == 0)
#define PTR_ALIGNED2_8(a,b) (((PTR_LINEAR(a) | PTR_LINEAR(b)) & 7) == 0)
#endif

#define PTR_LT(a,b)         (PTR(a) < PTR(b))
#define PTR_GE(a,b)         (PTR(a) >= PTR(b))


UCL_EXTERN(ucl_uintptr_t)
__ucl_ptr_linear(const ucl_voidp ptr);


typedef union
{
    char            a_char;
    unsigned char   a_uchar;
    short           a_short;
    unsigned short  a_ushort;
    int             a_int;
    unsigned int    a_uint;
    long            a_long;
    unsigned long   a_ulong;
    ucl_int         a_ucl_int;
    ucl_uint        a_ucl_uint;
    ucl_int32       a_ucl_int32;
    ucl_uint32      a_ucl_uint32;
    ptrdiff_t       a_ptrdiff_t;
    ucl_uintptr_t   a_ucl_uintptr_t;
    ucl_voidp       a_ucl_voidp;
    void *          a_void_p;
    ucl_bytep       a_ucl_bytep;
    ucl_bytepp      a_ucl_bytepp;
    ucl_uintp       a_ucl_uintp;
    ucl_uint *      a_ucl_uint_p;
    ucl_uint32p     a_ucl_uint32p;
    ucl_uint32 *    a_ucl_uint32_p;
    unsigned char * a_uchar_p;
    char *          a_char_p;
}
ucl_align_t;



#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* already included */

/*
vi:ts=4:et
*/


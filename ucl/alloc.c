/* alloc.c -- memory allocation

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
// implementation
************************************************************************/

#if defined(__UCL_MMODEL_HUGE)

#define acc_hsize_t             ucl_uint
#define acc_hvoid_p             ucl_voidp
#define ACCLIB_PUBLIC(r,f)      static r __UCL_CDECL f
#define acc_halloc              ucl_malloc_internal
#define acc_hfree               ucl_free_internal
#include "acc/acclib/halloc.ch"
#undef ACCLIB_PUBLIC

#else

UCL_PRIVATE(ucl_voidp)
ucl_malloc_internal(ucl_uint size)
{
    ucl_voidp p = NULL;
    if (size < ~(size_t)0)
        p = (ucl_voidp) malloc((size_t) size);
    return p;
}


UCL_PRIVATE(void)
ucl_free_internal(ucl_voidp p)
{
    if (p)
        free(p);
}

#endif


/***********************************************************************
// public interface using the global hooks
************************************************************************/

/* global allocator hooks */
static ucl_malloc_hook_t ucl_malloc_hook = ucl_malloc_internal;
static ucl_free_hook_t ucl_free_hook = ucl_free_internal;

UCL_PUBLIC(void)
ucl_set_malloc_hooks(ucl_malloc_hook_t a, ucl_free_hook_t f)
{
    ucl_malloc_hook = ucl_malloc_internal;
    ucl_free_hook = ucl_free_internal;
    if (a)
        ucl_malloc_hook = a;
    if (f)
        ucl_free_hook = f;
}

UCL_PUBLIC(void)
ucl_get_malloc_hooks(ucl_malloc_hook_t* a, ucl_free_hook_t* f)
{
    if (a)
        *a = ucl_malloc_hook;
    if (f)
        *f = ucl_free_hook;
}


UCL_PUBLIC(ucl_voidp)
ucl_malloc(ucl_uint size)
{
    if (size <= 0)
        return NULL;
    return ucl_malloc_hook(size);
}

UCL_PUBLIC(ucl_voidp)
ucl_alloc(ucl_uint nelems, ucl_uint size)
{
    ucl_uint s = nelems * size;
    if (nelems <= 0 || s / nelems != size)
        return NULL;
    return ucl_malloc(s);
}


UCL_PUBLIC(void)
ucl_free(ucl_voidp p)
{
    if (p)
        ucl_free_hook(p);
}


/*
vi:ts=4:et
*/

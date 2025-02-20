/* n2e_to.c -- implementation of the NRV2E test overlap algorithm

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



/***********************************************************************
// entries for the different bit-buffer sizes
************************************************************************/

#include "ucl_conf.h"
#include <ucl/ucl.h>
#include "getbit.h"

#define SAFE
#define TEST_OVERLAP


UCL_PUBLIC(int)
ucl_nrv2e_test_overlap_8        ( const ucl_bytep src, ucl_uint src_off,
                                        ucl_uint  src_len, ucl_uintp dst_len,
                                        ucl_voidp wrkmem )
{
#define getbit(bb)      getbit_8(bb,src,ilen)
#include "n2e_d.c"
#undef getbit
}


UCL_PUBLIC(int)
ucl_nrv2e_test_overlap_le16     ( const ucl_bytep src, ucl_uint src_off,
                                        ucl_uint  src_len, ucl_uintp dst_len,
                                        ucl_voidp wrkmem )
{
#define getbit(bb)      getbit_le16(bb,src,ilen)
#include "n2e_d.c"
#undef getbit
}


UCL_PUBLIC(int)
ucl_nrv2e_test_overlap_le32     ( const ucl_bytep src, ucl_uint src_off,
                                        ucl_uint  src_len, ucl_uintp dst_len,
                                        ucl_voidp wrkmem )
{
    unsigned bc = 0;
#define getbit(bb)      getbit_le32(bb,bc,src,ilen)
#include "n2e_d.c"
#undef getbit
}


/*
vi:ts=4:et
*/


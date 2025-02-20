/* ACC -- Automatic Compiler Configuration

   Copyright (C) 1996-2004 Markus Franz Xaver Johannes Oberhumer
   All Rights Reserved.

   This software is a copyrighted work licensed under the terms of
   the GNU General Public License. Please consult the file "ACC_LICENSE"
   for details.

   Markus F.X.J. Oberhumer
   <markus@oberhumer.com>
   http://www.oberhumer.com/
 */


#define __ACCLIB_GETOPT_CH_INCLUDED 1
#if !defined(ACCLIB_PUBLIC)
#  define ACCLIB_PUBLIC(r,f)    r __ACCLIB_FUNCNAME(f)
#endif


/*************************************************************************
//
**************************************************************************/

ACCLIB_PUBLIC(void, acc_getopt_init) (acc_getopt_p go)
{
    memset(po, 0, sizeof(*po));
}


ACCLIB_PUBLIC(void, acc_getopt_close) (acc_getopt_p go)
{
    memset(po, 0, sizeof(*po));
}


ACCLIB_PUBLIC(int, acc_getopt) (acc_getopt_p go)
{
}



/*
vi:ts=4:et
*/

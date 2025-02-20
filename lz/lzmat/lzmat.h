/*
**
**  $Revision: 1.5 $
**  $Date: 2008/07/19 15:21:55 $
**
**  $Author: Vitaly $
***************************************************************************
** lzmat.h - header file for the LZMAT real-time data compression library
**
** This file is part of the LZMAT real-time data compression library.
**
** LZMAT ANSI-C encoder/decoder 1.01
** Copyright (C) 2007,2008 Vitaly Evseenko. All Rights Reserved.
**
** The LZMAT library is free software; you can redistribute it and/or
** modify it under the terms of the GNU General Public License as
** published by the Free Software Foundation; either version 2 of
** the License, or (at your option) any later version.
**
** The LZMAT library is distributed WITHOUT ANY WARRANTY;
** without even the implied warranty of MERCHANTABILITY or
** FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
** License for more details.
**
** You should have received a copy of the GNU General Public License
** along with the LZMAT library; see the file GPL.TXT.
** If not, write to the Free Software Foundation, Inc.,
** 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
**
** Vitaly Evseenko
** <ve@matcode.com>
** http://www.matcode.com/lzmat.htm
** lzmat.h
***************************************************************************
*/

#ifndef _LZMAT_H
#define _LZMAT_H

#ifdef _WINDOWS
typedef unsigned __int64	MP_U64;
typedef unsigned long	MP_U32;
typedef __int64	MP_S64;
typedef long	MP_S32;
#else
typedef unsigned long	MP_U64;
typedef unsigned int	MP_U32;
typedef long	MP_S64;
typedef int	MP_S32;
#endif

typedef unsigned short	MP_U16;
typedef unsigned char	MP_U8;
typedef unsigned short	MP_WC;
typedef short	MP_S16;
typedef char	MP_S8;
#if defined(_WINDOWS) // #if defined(_WIN64)
typedef unsigned __int64	MP_PTR, *PMP_PTR;
#else
typedef unsigned long	MP_PTR, *PMP_PTR;
#endif

#define GET_LE64(_p_)	(*((MP_U64 *)(_p_)))
#define GET_LE32(_p_)	(*((MP_U32 *)(_p_)))
#define GET_LE16(_p_)	(*((MP_U16 *)(_p_)))

#define SET_LE64(_p_,_v_)	(*((MP_U64 *)(_p_)) = (_v_))
#define SET_LE32(_p_,_v_)	(*((MP_U32 *)(_p_)) = (_v_))
#define SET_LE16(_p_,_v_)	(*((MP_U16 *)(_p_)) = (_v_))

#ifndef LZMAT_CALLCONV
#define LZMAT_CALLCONV
#endif

#define MAX_LZMAT_ENCODED_SIZE(_sz_)	((_sz_)+(((_sz_)+7)>>3)+0x21)

#ifdef __cplusplus
extern "C" {
#endif

int LZMAT_CALLCONV lzmat_encode(MP_U8 *pbOut, MP_U32 *pcbOut,
		MP_U8 *pbIn, MP_U32 cbIn
#ifdef LZMAT_SMALL_STACK
	, void *pVocabularyBuf
#endif
);

#ifdef mkstub
int __fastcall lzmat_decode(MP_U8 *pbOut, MP_U8 *pbIn, MP_U32 cbIn);
#else //!mkstub
int LZMAT_CALLCONV lzmat_decode(MP_U8 *pbOut, MP_U32 *pcbOut,
	MP_U8 *pbIn, MP_U32 cbIn);
#endif

MP_U32 lzmat_dictionary_size(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

// Error codes
#define LZMAT_STATUS_OK	0
#define LZMAT_STATUS_ERROR	(-1)
#define LZMAT_STATUS_INTEGRITY_FAILURE	0x100
#define LZMAT_STATUS_BUFFER_TOO_SMALL	0x110

#endif // _LZMAT_H

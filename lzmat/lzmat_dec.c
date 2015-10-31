/*
**  $Id: lzmat_dec.c,v 1.1 2008/07/08 16:58:35 Vitaly Exp $
**  $Revision: 1.1 $
**  $Date: 2008/07/08 16:58:35 $
** 
**  $Author: Vitaly $
**
***************************************************************************
** LZMAT ANSI-C decoder 1.01
** Copyright (C) 2007,2008 Vitaly Evseenko. All Rights Reserved.
** lzmat_dec.c
**
** This file is part of the LZMAT real-time data compression library.
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
***************************************************************************
*/


#include "lzmat.h"

#define LZMAT_SAVE


#define LZMAT_DEFAULT_CNT		(0x12)
#define LZMAT_1BYTE_CNT		(0xFF + LZMAT_DEFAULT_CNT)
#define LZMAT_2BYTE_CNT		(0xFFFF + LZMAT_1BYTE_CNT)
#define LZMAT_MAX_2BYTE_CNT	(LZMAT_2BYTE_CNT-1)

#define LZMAT_GET_U4(_p_,_i_,_n_) \
	((_n_^=1)?(((MP_U8 *)(_p_))[_i_]&0xF):(((MP_U8 *)(_p_))[_i_++]>>4))

#define LZMAT_GET_U8(_p_,_n_)	(MP_U8)(((_n_)?((((MP_U8 *)(_p_))[0]>>4)|(((MP_U8 *)(_p_))[1]<<4)):((MP_U8 *)(_p_))[0]))

#define LZMAT_GET_LE16(_p_,_n_) \
 (MP_U16)((_n_)?((((MP_U8 *)(_p_))[0]>>4)|((MP_U16)(GET_LE16((_p_)+1))<<4)):GET_LE16(_p_))

#define MAX_LZMAT_SHORT_DIST0	0x80
#define MAX_LZMAT_SHORT_DIST1	(0x800|MAX_LZMAT_SHORT_DIST0)
#define MAX_LZMAT_LONG_DIST0	0x40
#define MAX_LZMAT_LONG_DIST1	(0x400|MAX_LZMAT_LONG_DIST0)
#define MAX_LZMAT_LONG_DIST2	(0x4000|MAX_LZMAT_LONG_DIST1)
#define MAX_LZMAT_LONG_DIST3	(0x40000|MAX_LZMAT_LONG_DIST2)
#define MAX_LZMAT_GAMMA_DIST	(MAX_LZMAT_LONG_DIST3-1)

#define LZMAT_DIST_MSK0		0x3F
#define LZMAT_DIST_MSK1		0x3FF

#ifdef LZMAT_SAVE

int LZMAT_CALLCONV lzmat_decode(MP_U8 *pbOut, MP_U32 *pcbOut,
	MP_U8 *pbIn, MP_U32 cbIn)
{
	MP_U32  inPos, outPos;
	MP_U32  cbOutBuf = *pcbOut;
	MP_U8 cur_nib;
	*pbOut = *pbIn;
	for(inPos=1, outPos=1, cur_nib=0; inPos<(cbIn-cur_nib);)
	{
		int bc;
		MP_U8 tag;
		tag = LZMAT_GET_U8(pbIn+inPos,cur_nib);
		inPos++;
		for(bc=0; bc<8 && inPos<(cbIn-cur_nib) && outPos<cbOutBuf; bc++, tag<<=1)
		{
			if(tag&0x80) // gamma
			{
				MP_U32 r_pos, r_cnt, dist;
#define cflag	r_cnt
				cflag = LZMAT_GET_LE16(pbIn+inPos,cur_nib);
				inPos++;
				if(outPos>MAX_LZMAT_SHORT_DIST1)
				{
					dist = cflag>>2;
					switch(cflag&3)
					{
					case 0:
						dist=(dist&LZMAT_DIST_MSK0)+1;
						break;
					case 1:
						inPos+=cur_nib;
						dist = (dist&LZMAT_DIST_MSK1)+0x41;
						cur_nib^=1;
						break;
					case 2:
						inPos++;
						dist += 0x441;
						break;
					case 3:
						if((inPos+2+cur_nib)>cbIn)
							return LZMAT_STATUS_INTEGRITY_FAILURE+1;
						inPos++;
						dist = (dist + 
							((MP_U32)LZMAT_GET_U4(pbIn,inPos,cur_nib)<<14))
							+0x4441;
						break;
					}
				}
				else
				{
					dist = cflag>>1;
					if(cflag&1)
					{
						inPos+=cur_nib;
						dist = (dist&0x7FF)+0x81;
						cur_nib^=1;
					}
					else
						dist = (dist&0x7F)+1;
				}
#undef cflag
				r_cnt = LZMAT_GET_U4(pbIn,inPos,cur_nib);
				if(r_cnt!=0xF)
				{
					r_cnt += 3;
				}
				else
				{
					if((inPos+1+cur_nib)>cbIn)
						return LZMAT_STATUS_INTEGRITY_FAILURE+2;
					r_cnt = LZMAT_GET_U8(pbIn+inPos,cur_nib);
					inPos++;
					if(r_cnt!=0xFF)
					{
						r_cnt += LZMAT_DEFAULT_CNT;
					}
					else
					{
						if((inPos+2+cur_nib)>cbIn)
							return LZMAT_STATUS_INTEGRITY_FAILURE+3;
						r_cnt = LZMAT_GET_LE16(pbIn+inPos,cur_nib)+LZMAT_1BYTE_CNT;
						inPos+=2;
						if(r_cnt==LZMAT_2BYTE_CNT)
						{
							// copy chunk
							if(cur_nib)
							{
								r_cnt = ((MP_U32)pbIn[inPos-4]&0xFC)<<5;
								inPos++;
								cur_nib = 0;
							}
							else
							{
								r_cnt = (GET_LE16(pbIn+inPos-5)&0xFC0)<<1;
							}
							r_cnt+=(tag&0x7F)+4;
							r_cnt<<=1;
							if((outPos+(r_cnt<<2))>cbOutBuf)
								return LZMAT_STATUS_BUFFER_TOO_SMALL;
							while(r_cnt-- && outPos<cbOutBuf)
							{
								*(MP_U32 *)(pbOut+outPos)=*(MP_U32 *)(pbIn+inPos);
								inPos+=4;
								outPos+=4;
							}
							break;
						}
					}
				}
				if(outPos<dist)
					return LZMAT_STATUS_INTEGRITY_FAILURE+4;
				if((outPos+r_cnt)>cbOutBuf)
					return LZMAT_STATUS_BUFFER_TOO_SMALL+1;
				r_pos = outPos-dist;
				while(r_cnt-- && outPos<cbOutBuf)
					pbOut[outPos++]=pbOut[r_pos++];
			}
			else
			{
				pbOut[outPos++]=LZMAT_GET_U8(pbIn+inPos,cur_nib);
				inPos++;
			}
		}
	}
	*pcbOut = outPos;
	return LZMAT_STATUS_OK;
}

#else //!LZMAT_SAVE


#ifdef mkstub
int __fastcall lzmat_decode(MP_U8 *pbOut, MP_U8 *pbIn, MP_U32 cbIn)
#else //!mkstub
int LZMAT_CALLCONV lzmat_decode(MP_U8 *pbOut, MP_U32 *pcbOut,
	MP_U8 *pbIn, MP_U32 cbIn)
#endif
{
	MP_U32 inPos, outPos;
	MP_U8  cur_nib;
#ifndef mkstub
	MP_U32 cbOutBuf = *pcbOut;
#endif
	*pbOut = *pbIn;
	for(inPos=1, outPos=1, cur_nib=0; inPos<(cbIn-cur_nib);)
	{
		int bc;
		MP_U8 tag;
		tag = LZMAT_GET_U8(pbIn+inPos,cur_nib);
		inPos++;
#ifdef mkstub
		for(bc=0; bc<8 && inPos<(cbIn-cur_nib); bc++, tag<<=1)
#else //!mkstub
		for(bc=0; bc<8 && inPos<(cbIn-cur_nib) && outPos<cbOutBuf; bc++, tag<<=1)
#endif
		{
			if(tag&0x80) // gamma
			{
				MP_U32 r_pos, r_cnt, dist;
#define cflag	r_cnt
				cflag = LZMAT_GET_LE20_UNSAVE(pbIn+inPos,cur_nib);
				inPos++;
				if(outPos<0x881)
				{
					dist = cflag>>1;
					if(cflag&1)
					{
						inPos+=cur_nib;
						dist = (dist&0x7FF)+0x81;
						cur_nib^=1;
					}
					else
						dist = (dist&0x7F)+1;
				}
				else
				{
					dist = cflag>>2;
					switch(cflag&3)
					{
					case 0:
						dist=(dist&0x3F)+1;
						break;
					case 1:
						inPos+=cur_nib;
						dist = (dist&0x3FF)+0x41;
						cur_nib^=1;
						break;
					case 2:
						dist = (dist&0x3FFF)+0x441;
						inPos++;
						break;
					case 3:
						inPos+=(1+cur_nib);
						dist = (dist&0x3FFFF)+0x4441;
						cur_nib^=1;
						break;
					}
				}
#undef cflag
				r_cnt = LZMAT_GET_LE12_UNSAVE(pbIn+inPos,cur_nib);
				inPos+=cur_nib;
				cur_nib^=1;
				if((r_cnt&0xF)!=0xF)
				{
					r_cnt = (r_cnt&0xF)+3;
				}
				else
				{
					inPos++;
					if(r_cnt!=0xFFF)
					{
						//r_cnt = LZMAT_GET_U8(pbIn+inPos-1,cur_nib)+0x12;
						r_cnt=(r_cnt>>4)+0x12;
					}
					else
					{
						r_cnt = LZMAT_GET_LE16_UNSAVE(pbIn+inPos,cur_nib)+0x111;
						inPos+=2;
						if(r_cnt==LZMAT_2BYTE_CNT)
						{
							// copy chunk
							if(cur_nib)
							{
								r_cnt = ((MP_U32)pbIn[inPos-4]&0xFC)<<5;
								inPos++;
								cur_nib = 0;
							}
							else
							{
								r_cnt = (GET_LE16(pbIn+inPos-5)&0xFC0)<<1;
								//((MP_U32)(pbIn[inPos-5]&0xC0)+(pbIn[inPos-4]<<4))<<1;
							}
							r_cnt+=(tag&0x7F)+4;
							r_cnt<<=1;
#ifdef mkstub
							while(r_cnt--)
#else //!mkstub
							while(r_cnt-- && outPos<cbOutBuf)
#endif
							{
								*(MP_U32 *)(pbOut+outPos)=*(MP_U32 *)(pbIn+inPos);
								inPos+=4;
								outPos+=4;
							}
							break;
						}
					}
				}
				r_pos = outPos-dist;
#ifdef mkstub
				while(r_cnt--)
#else //!mkstub
				while(r_cnt-- && outPos<cbOutBuf)
#endif
					pbOut[outPos++]=pbOut[r_pos++];
			}
			else
			{
				pbOut[outPos++]=LZMAT_GET_U8(pbIn+inPos,cur_nib);
				inPos++;
			}
		}
	}
#ifdef mkstub
	return outPos;
#else //!mkstub
	if(inPos<(cbIn-cur_nib))
		return -1;
	*pcbOut = outPos;
	return LZMAT_STATUS_OK;
#endif
}

#endif


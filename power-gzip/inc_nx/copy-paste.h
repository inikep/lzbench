/*
 * Copyright (C) IBM Corporation, 2016-2020
 *
 * Licenses for GPLv2 and Apache v2.0:
 *
 * GPLv2:
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *
 * Apache v2.0:
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "nx-helpers.h"

/*
 * Macros taken from arch/powerpc/include/asm/ppc-opcode.h and other
 * header files in kernel.
 *
 * TODO: Fix dependencies so we can directly use the kernel headers
 *	 without copying headers/macros to local test directory?
 */
#define ___PPC_RA(a)    (((a) & 0x1f) << 16)
#define ___PPC_RB(b)    (((b) & 0x1f) << 11)

#define PPC_INST_COPY                   0x7c20060c
#define PPC_INST_PASTE                  0x7c20070d

#define PPC_COPY(a, b)          stringify_in_c(.long PPC_INST_COPY | \
						___PPC_RA(a) | ___PPC_RB(b))
#define PPC_PASTE(a, b)         stringify_in_c(.long PPC_INST_PASTE | \
						___PPC_RA(a) | ___PPC_RB(b))
#define CR0_SHIFT	28
#define CR0_MASK	0xF
/*
 * Copy/paste instructions:
 *
 *	copy RA,RB
 *		Copy contents of address (RA) + effective_address(RB)
 *		to internal copy-buffer.
 *
 *	paste RA,RB
 *		Paste contents of internal copy-buffer to the address
 *		(RA) + effective_address(RB)
 */
static inline int vas_copy(void *crb, int offset)
{
	asm volatile(PPC_COPY(%0, %1)";"
		:
		: "b" (offset), "b" (crb)
		: "memory");

	return 0;
}

static inline int vas_paste(void *paste_address, int offset)
{
	u32 cr;

	cr = 0;
	asm volatile(PPC_PASTE(%1, %2)";"
		"mfocrf %0, 0x80;"
		: "=r" (cr)
		: "b" (offset), "b" (paste_address)
		: "memory", "cr0");

	return (cr >> CR0_SHIFT) & CR0_MASK;
}

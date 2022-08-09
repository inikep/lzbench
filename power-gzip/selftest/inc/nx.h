/*
 * Copyright (C) IBM Corporation, 2011-2020
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
#ifndef _NX_H
#define _NX_H

#include <stdbool.h>

#define	NX_FUNC_COMP_842	1
#define NX_FUNC_COMP_GZIP	2

#ifndef __aligned
#define __aligned(x)	__attribute__((aligned(x)))
#endif

struct nx842_func_args {
	bool use_crc;
	bool decompress;		/* true decompress; false compress */
	bool move_data;
	int timeout;			/* seconds */
};

struct nxbuf_t {
	int len;
	char *buf;
};

/* @function should be EFT (aka 842), GZIP etc */
void *nx_function_begin(int function, int pri);

int nx_function(void *handle, struct nxbuf_t *in, struct nxbuf_t *out,
		void *arg);

int nx_function_end(void *handle);

#endif	/* _NX_H */

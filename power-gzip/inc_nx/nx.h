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

/** @file nx.h
 *
 */

#ifndef _NX_H
#define _NX_H

#include <stdbool.h>

#define	NX_FUNC_COMP_842	1
#define NX_FUNC_COMP_GZIP	2

struct nx842_func_args {
	bool use_crc;
	bool decompress;		/* true decompress; false compress */
	bool move_data;
	int timeout;			/* seconds */
};

typedef struct {
	int len;
	char *buf;
} nxbuf_t;

/** \brief Allocate and initialize an nx_handle.
 *
 * @param function Should be EFT (aka 842), GZIP etc
 * @param pri ID of the NX device. Use -1 to pick any.
 *
 * \return A pointer to a dynamically allocated struct nx_handle.
 * \retval NULL In case of error.
 */
void *nx_function_begin(int function, int pri);

int nx_function(void *handle, nxbuf_t *in, nxbuf_t *out, void *arg);

/** \brief Deallocate a previously allocated nx_handle.
 *
 * @param handle A pointer to a struct nx_handle previously initialized by
 *               nx_function_begin().
 *
 * \retval 0 In case of success.
 * \retval -1 In case of failure.  \c errno is set to indicate the cause of the
 *            error.
 */
int nx_function_end(void *handle);

#endif	/* _NX_H */

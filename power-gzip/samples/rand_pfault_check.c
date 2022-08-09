/*
 * Wrappers to randomly touch pages.
 * This will cause a lot of page faults on a stress test, this checks if this
 * is causing the system to let process hanging indefinitely.
 *
 * Copyright (C) 2020 IBM Corporation
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
 *
 */
#include <stdlib.h>

extern int __real_nx_touch_pages(void *buf, long buf_len, long page_len,
				 int wr);
extern int __real_nx_touch_pages_dde(void *ddep, long buf_sz, long page_sz,
				 int wr);

int __wrap_nx_touch_pages(void *buf, long buf_len, long page_len, int wr)
{
	if (!(rand() % 2))
		return __real_nx_touch_pages(buf, buf_len, page_len, wr);
	else
		return 0;
}

int __wrap_nx_touch_pages_dde(void *ddep, long buf_sz, long page_sz, int wr)
{
	if (!(rand() % 2))
		return __real_nx_touch_pages_dde(ddep, buf_sz, page_sz, wr);
	else
		return 0;
}

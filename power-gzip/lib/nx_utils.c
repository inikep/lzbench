/*
 * Copyright (C) IBM Corporation, 2011-2017
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

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include "nx_utils.h"

/* very simple configure file parser */

#define MAX_KEY_LEN 128
#define MAX_CONFIG_LINE 1024
#define MAX_CONFIG_ITEM 1024

struct cfg_item {
	char key[MAX_KEY_LEN];
	char val[MAX_CONFIG_LINE];
};

/* Trim leading and trailing white space in str, and copy to out
 * if the length of str is larger than (len - 1),
 * or str contains only white spaces, it is regarded as illegal input */
static int trim_space(char *out, int len, const char *str)
{
	const char *end;
	int out_size;

	if (!out || !str || len <= 0)
		return -1;

	/* trim leading space */
	while (isspace((unsigned char)*str))
		str++;

	if (*str == 0) {
		/* all spaces */
		return -1;
	}

	/* trim trailing space */
	end = str + strlen(str) - 1;
	while (end > str && isspace((unsigned char)*end))
		end--;
	end++;

	out_size = end - str;
	if (out_size >= len || out_size <= 0 ) {
		/* safeguard, regarded as illegal input */
		return -1;
	}

	/* copy trimmed string and add null terminator */
	memcpy(out, str, out_size);
	out[out_size] = 0;

	return out_size;
}

static int find_cfg(char *key, struct nx_cfg_tab *cfg_table)
{
	int i;
	struct cfg_item *configs;

	if (!cfg_table || !cfg_table->configs || !key)
		return -1;

	configs = cfg_table->configs;
	for (i = 0; i < cfg_table->cfg_num; i++) {
		if (strcmp(key, configs[i].key) == 0)
			return i;
	}
	return -1;

}

/* set configs[cnt].key and configs[cnt].val */
static int set_cfg(struct nx_cfg_tab *cfg_table, char *key, char *val, int cnt)
{
	int ret;
	int key_size;
	int cfg_cnt;
	char trim_key[MAX_KEY_LEN];
	struct cfg_item *configs;

	if (!cfg_table || !cfg_table->configs || !key || !val)
		return -1;

	configs = cfg_table->configs;
	key_size = trim_space(trim_key, MAX_KEY_LEN, key);
	if (key_size < 0) {
		/* key is white space or larger than MAX_KEY_LEN, do nothing */
		return -1;
	}

	/* is the key already in config table? */
	cfg_cnt = find_cfg(trim_key, cfg_table);
	if (cfg_cnt < 0) {
		/* not found, add a new config item */
		if (cnt >= MAX_CONFIG_ITEM || cnt < 0)
			return -1;
		cfg_cnt = cnt;
		ret = 1;
	} else
		ret = 0;

	memcpy(configs[cfg_cnt].key, trim_key, key_size);
	if (trim_space(configs[cfg_cnt].val, MAX_CONFIG_LINE, val) < 0) {
		/* val is all space, or lengh of val larger than MAX_CONFIG_LINE,
		 * store val as an empty string */
		*configs[cfg_cnt].val = '\0';
	}

	return ret;
}

int nx_dump_cfg(struct nx_cfg_tab *cfg_table, FILE *fp)
{
	int i;

	if (!cfg_table || !fp)
		return -1;

	fprintf(fp, "nx-zlib config file ========\n");
	for (i = 0; i < cfg_table->cfg_num; i++) {
		fprintf(fp, "[%d]: %s = %s\n", i, cfg_table->configs[i].key,
			cfg_table->configs[i].val);
	}

	return 0;
}

char* nx_get_cfg(char *key, struct nx_cfg_tab *cfg_table)
{
	int i;
	char *val;

	if (!cfg_table || !cfg_table->configs || !key)
		return NULL;

	i = find_cfg(key, cfg_table);
	if (i < 0)
		return NULL;

	val = cfg_table->configs[i].val;
	if (val && *val == '\0') {
		/* regard empty string as invalid value */
		val = NULL;
	}
	return val;
}

int nx_read_cfg(const char *filename, struct nx_cfg_tab *cfg_table)
{
	int ret;
	FILE *cfg_file;
	char buf[MAX_CONFIG_LINE];
	char *val;
	struct cfg_item *configs;
	int cfg_cnt;

	if (!cfg_table)
		return -1;

	cfg_file = fopen(filename, "r");
	if (cfg_file == NULL) {
		/* configure file is optional. log as notice. */
		syslog(LOG_NOTICE, "cannot open nx-zlib config file: %s: %s\n",
			filename, strerror(errno));
		return -1;
	}

	configs = malloc(sizeof(struct cfg_item) * MAX_CONFIG_ITEM);
	if (!configs) {
		syslog(LOG_ERR, "cannot alloc nx-zlib config memory\n");
		fclose(cfg_file);
		return -1;
	}
	memset(configs, 0, sizeof(struct cfg_item) * MAX_CONFIG_ITEM);
	cfg_table->configs = configs;

	cfg_cnt = 0;
	while (!feof(cfg_file)) {
		if (!fgets(buf, MAX_CONFIG_LINE, cfg_file))
			break;
		/* find the first # (comment) */
		val = strchr(buf, '#');
		/* discard string following # */
		if (val != NULL)
			*val = '\0';
		val = strchr(buf, '=');
		if (val != NULL) {
			*val = '\0';
			val++;
			/* now buf is key, val is value */
			ret = set_cfg(cfg_table, buf, val, cfg_cnt);
			if (ret == 1) {
				cfg_cnt++;
				if (cfg_cnt > MAX_CONFIG_ITEM)
					break;
				cfg_table->cfg_num = cfg_cnt;
			}
		}
	}

	if (cfg_cnt == 0) {
		/* nothing read */
		free(configs);
		cfg_table->configs = NULL;
		fclose(cfg_file);
		return -1;
	}

	fclose(cfg_file);
	return 0;
}

int nx_close_cfg(struct nx_cfg_tab *cfg_table)
{
	if (!cfg_table)
		return -1;

	if (cfg_table->configs)
		free(cfg_table->configs);

	return 0;
}

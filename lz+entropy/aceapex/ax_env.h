/* ax_env.h - the tuning knobs read from the environment (ACEAPEX_BS, ACEAPEX_DUMP, AX_*, FSE_CHUNK, LIT_CHUNK,
 * LIT_LANES, LIT_LANES_DEC, LIT_LEVEL) are read only in builds with ACEAPEX_ENV_TUNING defined: the CLI (ACEAPEX_CLI)
 * and the repository's own tools. A library built without it - the default, e.g. inside lzbench - behaves the same
 * whatever the environment holds: same archive bytes, same threads (lzbench #336). Every read goes through ax_getenv. */
#ifndef AX_ENV_H
#define AX_ENV_H
#include <stdlib.h>
#if defined(ACEAPEX_CLI) && !defined(ACEAPEX_ENV_TUNING)
#define ACEAPEX_ENV_TUNING 1
#endif
#ifdef ACEAPEX_ENV_TUNING
#define ax_getenv(name) getenv(name)
#else
#define ax_getenv(name) ((char*)0)
#endif
#endif

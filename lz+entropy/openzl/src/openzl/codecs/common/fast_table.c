// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/common/fast_table.h"
#include "openzl/shared/portability.h"

#include <stdlib.h>

size_t ZS_FastTable_tableSize(size_t tableLog)
{
    return ((size_t)1 << tableLog) * sizeof(((ZS_FastTable*)NULL)->table[0]);
}

// We are forcing the compiler to not inline this function, so that it won't
// optimize the malloc + memset into a calloc.
ZL_FORCE_NOINLINE void
ZS_FastTable_clear(ZS_FastTable* table, uint32_t tableLog, uint32_t minMatch)
{
    size_t const tableSize = ZS_FastTable_tableSize(tableLog);
    memset(table->table, 0, tableSize);
    table->tableLog = tableLog;
    table->minMatch = minMatch;
}

void ZS_FastTable_init(
        ZS_FastTable* table,
        void* memory,
        uint32_t tableLog,
        uint32_t minMatch)
{
    // malloc + memset is faster than calloc because we have a random access
    // pattern. If we don't memset the table, the pages will be filled into the
    // page table one at a time, whenever we first write to that page. That is
    // a lot less efficient than linearly loading the entire table into the page
    // table, as we memset it.
    ZL_ASSERT_NULL(table->table);
    table->table = memory;
    ZS_FastTable_clear(table, tableLog, minMatch);
}

uint32_t
ZS_FastTable_getAndUpdate(ZS_FastTable* table, uint8_t const* ptr, uint32_t pos)
{
    return ZS_FastTable_getAndUpdateT(table, ptr, pos, table->minMatch);
}

void ZS_FastTable_put(ZS_FastTable* table, uint8_t const* ptr, uint32_t pos)
{
    ZS_FastTable_putT(table, ptr, pos, table->minMatch);
}

uint32_t ZS_FastTable_get(ZS_FastTable* table, uint8_t const* ptr)
{
    return ZS_FastTable_getT(table, ptr, table->minMatch);
}

#include <csc_default_alloc.h>
#include <stdlib.h>


static void *Alloc(void *p, size_t size) 
{
    return malloc(size);
}

static void Free(void *p, void *address) 
{
    free(address);
}


ISzAlloc st_default_alloc = {Alloc, Free};
ISzAlloc *default_alloc = &st_default_alloc;


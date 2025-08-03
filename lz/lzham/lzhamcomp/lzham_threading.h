// File: lzham_threading.h
// LZHAM is in the Public Domain. Please see the Public Domain declaration at the end of include/lzham.h

#if LZHAM_USE_WIN32_API
   #include "lzham_win32_threading.h"
#elif LZHAM_USE_PTHREADS_API
   #include "lzham_pthreads_threading.h"
#else
   #include "lzham_null_threading.h"
#endif



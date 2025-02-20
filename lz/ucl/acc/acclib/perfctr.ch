/* ACC -- Automatic Compiler Configuration

   Copyright (C) 1996-2004 Markus Franz Xaver Johannes Oberhumer
   All Rights Reserved.

   This software is a copyrighted work licensed under the terms of
   the GNU General Public License. Please consult the file "ACC_LICENSE"
   for details.

   Markus F.X.J. Oberhumer
   <markus@oberhumer.com>
   http://www.oberhumer.com/
 */


#define __ACCLIB_PERFCTR_CH_INCLUDED 1
#if !defined(ACCLIB_PUBLIC)
#  define ACCLIB_PUBLIC(r,f)    r __ACCLIB_FUNCNAME(f)
#endif


#if (ACC_OS_POSIX_LINUX)
/* see http://user.it.uu.se/~mikpe/linux/perfctr/ */
#if defined(__cplusplus)
extern "C" {
#include <libperfctr.h>
}
#else
#include <libperfctr.h>
#endif
#endif


/*************************************************************************
//
**************************************************************************/

ACCLIB_PUBLIC(int, acc_perfctr_open) (acc_perfctr_handle_p h)
{
    memset(h, 0, sizeof(*h));
#if (ACC_OS_POSIX_LINUX)
    {
    struct vperfctr* handle;
    struct perfctr_info info;
    struct vperfctr_control control;
    struct perfctr_cpu_control* const cc = &control.cpu_control;
    /* open */
    handle = vperfctr_open();
    if (!handle) goto error;
    /* get info */
    if (vperfctr_info(handle, &info) < 0) goto error;
    h->cpu_type = info.cpu_type;
    h->cpu_features = info.cpu_features;
    h->cpu_khz = info.cpu_khz;
    h->cpu_nrctrs = perfctr_info_nrctrs(&info);
    h->cpu_name = perfctr_info_cpu_name(&info);
    /* setup control */
    memset(&control, 0, sizeof(control));
    switch (h->cpu_type) {
#if (ACC_ARCH_IA32)
    case PERFCTR_X86_WINCHIP_C6:
    case PERFCTR_X86_WINCHIP_2:
        break;      /* no working TSC available */
    case PERFCTR_X86_AMD_K7:
#endif
#if (ACC_ARCH_AMD64 || ACC_ARCH_IA32)
    case PERFCTR_X86_AMD_K8:
    case PERFCTR_X86_AMD_K8C:
        cc->tsc_on = 1; cc->nractrs = 2;
        /* event 0xC0 (RETIRED_INSNS), count at CPL > 0, Enable */
        cc->pmc_map[0] = 0;
        cc->evntsel[0] = 0xC0 | (1 << 16) | (1 << 22);
        /* event 0xC1 (RETIRED_OPS), count at CPL > 0, Enable */
        cc->pmc_map[1] = 1;
        cc->evntsel[1] = 0xC1 | (1 << 16) | (1 << 22);
        break;
#endif
    default:
        cc->tsc_on = 1;
        break;
    }
    if (cc->nractrs > h->cpu_nrctrs) cc->nractrs = h->cpu_nrctrs;
    if (vperfctr_control(handle, &control) < 0) goto error;
    /* success */
    h->h = (void*) handle;
    return 0;
error:
    if (handle) {
        vperfctr_stop(handle);
        vperfctr_close(handle);
    }
    }
#endif
    return -1;
}


ACCLIB_PUBLIC(int, acc_perfctr_close) (acc_perfctr_handle_p h)
{
    if (h->h) {
#if (ACC_OS_POSIX_LINUX)
        struct vperfctr* handle = (struct vperfctr*) h->h;
        vperfctr_stop(handle);
        vperfctr_close(handle);
#endif
        h->h = 0;
    }
    return 0;
}


/*************************************************************************
//
**************************************************************************/

ACCLIB_PUBLIC(void, acc_perfctr_read) (acc_perfctr_handle_p h, acc_perfctr_clock_p c)
{
    if (h->h) {
#if (ACC_OS_POSIX_LINUX)
        struct vperfctr* handle = (struct vperfctr*) h->h;
        vperfctr_read_ctrs(handle, (struct perfctr_sum_ctrs*) c);
#else
        memset(c, 0, sizeof(*c));
#endif
    } else
        memset(c, 0, sizeof(*c));
}


/*************************************************************************
//
**************************************************************************/

ACCLIB_PUBLIC(double, acc_perfctr_get_elapsed) (acc_perfctr_handle_p h, const acc_perfctr_clock_p start, const acc_perfctr_clock_p stop)
{
#if (ACC_OS_POSIX_LINUX)
    acc_uint64l_t tsc = stop->tsc - start->tsc;
    return ((double)tsc / h->cpu_khz) / 1000.0;
#else
    ACC_UNUSED(h); ACC_UNUSED(start); ACC_UNUSED(stop); return 0;
#endif
}


ACCLIB_PUBLIC(double, acc_perfctr_get_elapsed_tsc) (acc_perfctr_handle_p h, acc_uint64l_t tsc)
{
#if (ACC_OS_POSIX_LINUX)
    return ((double)tsc / h->cpu_khz) / 1000.0;
#else
    ACC_UNUSED(h); ACC_UNUSED(tsc); return 0;
#endif
}


/*
vi:ts=4:et
*/

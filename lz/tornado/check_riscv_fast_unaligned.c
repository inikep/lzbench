/* Does this RISC-V machine perform misaligned scalar accesses at full speed?
 *
 * Exit code 0 means yes, 1 means no -- slow, emulated, unsupported, or simply
 * unknown. The lzbench Makefile builds and runs this on RISC-V hosts to decide
 * whether to benchmark tornado, which dereferences unaligned 16/32/64-bit
 * values directly (see value32() and friends in Common.h).
 *
 * The answer comes from the Linux hwprobe syscall. Everything it needs is
 * declared below instead of being taken from <asm/hwprobe.h>, so that the check
 * does not depend on the age of the installed kernel headers:
 *
 *   258 = __NR_riscv_hwprobe                       (Linux 6.4+)
 *     9 = RISCV_HWPROBE_KEY_MISALIGNED_SCALAR_PERF (Linux 6.11+)
 *     5 = RISCV_HWPROBE_KEY_CPUPERF_0, deprecated, but carries the same answer
 *         in its low 3 bits (RISCV_HWPROBE_MISALIGNED_MASK) (Linux 6.4+)
 *     3 = RISCV_HWPROBE_MISALIGNED[_SCALAR]_FAST
 *
 * A key the running kernel does not know comes back as -1, which counts as
 * "not fast", as does a kernel without the syscall at all.
 */

#if defined(__riscv) && defined(__linux__)

extern long syscall(long number, ...);

#define NR_RISCV_HWPROBE            258
#define KEY_MISALIGNED_SCALAR_PERF  9
#define KEY_CPUPERF_0               5
#define MISALIGNED_MASK             7
#define MISALIGNED_FAST             3

struct hwprobe_pair {
    long long          key;
    unsigned long long value;
};

static int fast_misaligned(long long key, unsigned long long mask)
{
    struct hwprobe_pair pair;

    pair.key   = key;
    pair.value = 0;

    /* (pairs, pair_count, cpusetsize, cpus, flags): a zero cpusetsize with a
       NULL cpu set asks about every CPU in the system. */
    if (syscall(NR_RISCV_HWPROBE, &pair, 1UL, 0UL, (void *)0, 0U) != 0)
        return 0;
    if (pair.key < 0)
        return 0;

    return (pair.value & mask) == MISALIGNED_FAST;
}

int main(void)
{
    if (fast_misaligned(KEY_MISALIGNED_SCALAR_PERF, ~0ULL))
        return 0;
    if (fast_misaligned(KEY_CPUPERF_0, MISALIGNED_MASK))
        return 0;
    return 1;
}

#else

int main(void) { return 1; }

#endif

/* Check if RISC-V hardware supports fast unaligned memory access.
 * Uses the hwprobe syscall (Linux 6.4+).
 * Exit code 0: FAST unaligned access supported
 * Exit code 1: Not FAST (slow/emulated/unsupported/unknown)
 */

#include <sys/syscall.h>
#include <unistd.h>

#if defined(__riscv) && defined(__linux__)

#ifdef __has_include
#  if __has_include(<asm/hwprobe.h>)
#    include <asm/hwprobe.h>
#  else
#    define MANUAL_DEFS
#  endif
#else
#  include <asm/hwprobe.h>
#endif

#ifdef MANUAL_DEFS
#define __NR_riscv_hwprobe 258
struct riscv_hwprobe {
    long long key;
    unsigned long long value;
};
#define RISCV_HWPROBE_KEY_MISALIGNED_SCALAR_PERF 4
#define RISCV_HWPROBE_MISALIGNED_FAST 3
#endif

int main(void) {
    struct riscv_hwprobe pairs[] = {
        { .key = RISCV_HWPROBE_KEY_MISALIGNED_SCALAR_PERF, },
    };

    if (syscall(__NR_riscv_hwprobe, pairs, 1, 0, (void*)0, 0) == 0) {
        return (pairs[0].value == RISCV_HWPROBE_MISALIGNED_FAST) ? 0 : 1;
    }
    return 1;
}

#else
int main(void) { return 1; }
#endif

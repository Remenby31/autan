/* Preloaded by bench/ci/ctest-watchdog.sh into the test command it starts, and only there: each
 * process of the test lets a process of the same user trace it. The hosts' Yama ptrace_scope 1
 * otherwise lets a debugger attach only to its own descendants, and the watchdog's gdb is the
 * test's sibling: a hang would leave no stacks. Not set for the runner agent, which a job step
 * must not be able to trace. */
#include <sys/prctl.h>

__attribute__((constructor)) static void autan_allow_ptrace(void) {
  prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
}

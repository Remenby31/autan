/* Preloaded by bench/ci/ctest-watchdog.sh into the test command it starts, and only there: each
 * process of the test lets a process of the same user trace it. The hosts' Yama ptrace_scope 1
 * otherwise lets a debugger attach only to its own descendants, and the watchdog's gdb is the
 * test's sibling: a hang would leave no stacks. The exception survives exec but not fork, so a
 * forked child that does not exec sets it again. Never preloaded into the runner agent. */
#include <pthread.h>
#include <sys/prctl.h>

static void autan_allow_ptrace(void) { prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0); }

__attribute__((constructor)) static void autan_ptracer_init(void) {
  autan_allow_ptrace();
  pthread_atfork(NULL, NULL, autan_allow_ptrace);
}

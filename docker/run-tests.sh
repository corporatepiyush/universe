#!/usr/bin/env bash
# Run the universe test suite natively on Linux inside the container.
# Enforces the kernel-version floor, then builds + runs every test for the
# container's native triple (tests actually EXECUTE here, unlike macOS
# crosscheck which only codegens the Linux objects).
set -euo pipefail

MIN_KERNEL_MAJOR=6
MIN_KERNEL_MINOR=15

kver="$(uname -r)"
kmaj="${kver%%.*}"
krest="${kver#*.}"
kmin="${krest%%.*}"

echo "== universe Linux test container =="
echo "kernel: ${kver}   arch: $(uname -m)   llvm: ${LLVM}"

if [ "${kmaj}" -lt "${MIN_KERNEL_MAJOR}" ] || \
   { [ "${kmaj}" -eq "${MIN_KERNEL_MAJOR}" ] && [ "${kmin}" -lt "${MIN_KERNEL_MINOR}" ]; }; then
  if [ "${ALLOW_OLD_KERNEL:-0}" = "1" ]; then
    echo "WARN: kernel ${kver} is below the ${MIN_KERNEL_MAJOR}.${MIN_KERNEL_MINOR} TARGET." >&2
    echo "      ALLOW_OLD_KERNEL=1 set — running the suite anyway to get real Linux" >&2
    echo "      runtime signal (libc ABI, endianness, atomics lowering). Kernel-" >&2
    echo "      version-gated features (newer io_uring ops) are NOT exercised here." >&2
  else
    echo "FAIL: kernel ${kver} is below the required ${MIN_KERNEL_MAJOR}.${MIN_KERNEL_MINOR}." >&2
    echo "      Containers use the Docker VM kernel; update the host so 'uname -r'" >&2
    echo "      reports >= ${MIN_KERNEL_MAJOR}.${MIN_KERNEL_MINOR}, or re-run with" >&2
    echo "      ALLOW_OLD_KERNEL=1 to run on the available kernel (target unmet)." >&2
    exit 2
  fi
fi

# Build native-Linux artifacts into a container-local build dir so the
# bind-mounted source tree's build/ (macOS objects) is never clobbered.
export LLVM
BUILD_ROOT="${BUILD_ROOT:-/tmp/universe-build}"
rm -rf "${BUILD_ROOT}"
mkdir -p "${BUILD_ROOT}"

# Mirror sources into the build root (cheap; keeps macOS build/ untouched).
cp -a src tests Makefile docs "${BUILD_ROOT}/" 2>/dev/null || true
cd "${BUILD_ROOT}"

echo "== make test (native $(uname -m) Linux) =="
make test

echo "== make lib dylib (static libuniverse.a + shared libuniverse.so) =="
make lib dylib
ls -la build/libuniverse.a build/libuniverse.so

echo "== ALL LINUX TESTS PASSED on kernel ${kver} =="

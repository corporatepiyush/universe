#!/usr/bin/env bash
# Black-box artifact tests: build the SHIPPED library (libuniverse.a / dylib)
# and drive it ONLY through its public C ABI from a separate C harness.
# No IR objects are linked here — this proves the archive that users get.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LIB="$REPO_ROOT/build/libuniverse.a"

# System/Homebrew clang — deliberately NOT the IR toolchain: a black-box
# consumer uses whatever C compiler it has.
CC="${CC:-cc}"

# Extra system libs the archive's modules pull in (libc math, pthreads).
case "$(uname -s)" in
  Darwin) SYSLIBS="-lpthread" ;;
  *)      SYSLIBS="-lpthread -lm" ;;
esac

echo "== building shipped library (make lib dylib) =="
# One make at a time (repo rule). Build from the repo root.
if ! make -C "$REPO_ROOT" lib dylib; then
  echo "FAIL build: make lib dylib failed"
  exit 1
fi
if [ ! -f "$LIB" ]; then
  echo "FAIL build: $LIB not produced"
  exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fails=0
ran=0
echo "== running black-box cases =="
for src in "$SCRIPT_DIR"/bb_*.c; do
  [ -e "$src" ] || continue
  name="$(basename "$src" .c)"
  ran=$((ran + 1))
  bin="$WORK/$name"
  if ! "$CC" -O2 -std=c11 -Wall "$src" "$LIB" $SYSLIBS -o "$bin" 2> "$WORK/$name.cc.log"; then
    echo "FAIL $name (compile/link)"
    sed 's/^/    /' "$WORK/$name.cc.log"
    fails=$((fails + 1))
    continue
  fi
  if "$bin"; then
    echo "ok   $name"
  else
    echo "FAIL $name (exit $?)"
    fails=$((fails + 1))
  fi
done

echo "== summary: $((ran - fails))/$ran passed =="
[ "$ran" -gt 0 ] || { echo "FAIL: no bb_*.c cases found"; exit 1; }
[ "$fails" -eq 0 ] || exit 1
exit 0

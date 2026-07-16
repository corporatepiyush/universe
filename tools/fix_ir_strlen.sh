#!/usr/bin/env bash
# Auto-correct LLVM IR string-constant array lengths.
#
# Every hand-written `... constant [N x i8] c"..."` must have N equal to the
# real byte count of the c"..." payload (each `\XX` hex escape is ONE byte,
# every other character is one byte). Miscounting is the single most common
# authoring error. This tool recomputes N for every such line and rewrites it
# in place, so the length is never wrong again.
#
# Usage:
#   tools/fix_ir_strlen.sh <file.ll> [more.ll ...]   # fix listed files
#   tools/fix_ir_strlen.sh --all                     # fix all tests + src
#   tools/fix_ir_strlen.sh --check <file.ll ...>     # report mismatches, no write (exit 1 if any)
set -euo pipefail

CHECK=0
FILES=()
if [ "${1:-}" = "--all" ]; then
  while IFS= read -r f; do FILES+=("$f"); done < <(find src tests -name '*.ll' 2>/dev/null)
elif [ "${1:-}" = "--check" ]; then
  CHECK=1; shift; FILES=("$@")
else
  FILES=("$@")
fi

[ "${#FILES[@]}" -gt 0 ] || { echo "usage: $0 <file.ll...> | --all | --check <file...>" >&2; exit 2; }

# Compute the byte length of an LLVM c"..." payload passed as $1 (without quotes).
byte_len() {
  local s="$1" n=0 i=0 len=${#1}
  while [ "$i" -lt "$len" ]; do
    local ch="${s:$i:1}"
    if [ "$ch" = "\\" ]; then
      # LLVM escape: backslash + exactly two hex digits = 1 byte
      i=$((i + 3)); n=$((n + 1))
    else
      i=$((i + 1)); n=$((n + 1))
    fi
  done
  echo "$n"
}

rc=0
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "skip (not found): $f" >&2; continue; }
  tmp="$(mktemp)"
  changed=0
  while IFS= read -r line || [ -n "$line" ]; do
    # match: <prefix>[N x i8] c"<payload>"<suffix>
    if [[ "$line" =~ ^(.*)\[([0-9]+)\ x\ i8\]\ c\"(.*)\"(.*)$ ]]; then
      pre="${BASH_REMATCH[1]}"; declared="${BASH_REMATCH[2]}"
      payload="${BASH_REMATCH[3]}"; post="${BASH_REMATCH[4]}"
      real="$(byte_len "$payload")"
      if [ "$declared" != "$real" ]; then
        changed=1
        if [ "$CHECK" -eq 1 ]; then
          echo "$f: declared [$declared] but real [$real]  ->  c\"$payload\""
          rc=1
        fi
        line="${pre}[${real} x i8] c\"${payload}\"${post}"
      fi
    fi
    printf '%s\n' "$line" >> "$tmp"
  done < "$f"
  if [ "$CHECK" -eq 1 ]; then
    rm -f "$tmp"
  elif [ "$changed" -eq 1 ]; then
    mv "$tmp" "$f"; echo "fixed: $f"
  else
    rm -f "$tmp"
  fi
done
exit $rc

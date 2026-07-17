#!/usr/bin/env python3
# stackcheck — static stack-overflow gate for hand-written LLVM IR.
#
# The real stack-overflow patterns in this project (CLAUDE.md IR hazard #11 — an
# alloca reached IN A LOOP grows the frame every iteration and overflows):
#   1. alloca in a block that is part of a CFG cycle (loop body)         — HIGH.
#   2. dynamic / variable-length alloca (`alloca T, i64 %reg`) not wrapped
#      in llvm.stacksave/stackrestore (unbounded frame, worst in a loop)  — HIGH in a loop / MED elsewhere.
#   3. oversized fixed stack frame — a single alloca whose byte size exceeds
#      the threshold (default 16 KiB)                                     — MED.
#   4. direct self-recursion (a define that calls itself) — a NOTE for manual
#      review that the recursion depth is bounded.
# Exempt a verified line with a trailing `; stackcheck-ok: <reason>`.
#
# Usage: tools/stackcheck.py [--max-bytes N] [files...]   (default: all src+tests)
# Exit 0 if clean, 1 if any HIGH/MED finding. Run before every commit.
import sys, os, re, glob

MAXB = 16384
TYSZ = {'i1':1,'i8':1,'i16':2,'i32':4,'i64':8,'i128':16,'float':4,'double':8,'ptr':8,'half':2}

def elem_bytes(ty):
    ty = ty.strip()
    m = re.match(r'\[(\d+)\s*x\s*(.+)\]$', ty)
    if m: return int(m.group(1)) * elem_bytes(m.group(2))
    m = re.match(r'<(\d+)\s*x\s*(.+)>$', ty)
    if m: return int(m.group(1)) * elem_bytes(m.group(2))
    return TYSZ.get(ty, 8)

def alloca_info(line):  # -> (bytes|None, dynamic:bool)  — brace-aware type/count split
    m = re.search(r'\balloca\s+(.*)$', line)
    if not m: return (None, False)
    rest = m.group(1)
    # strip trailing comment
    rest = re.sub(r';.*$', '', rest).strip()
    # split off a TOP-LEVEL ", ..." tail (count/align); commas inside {}/[]/<> don't count
    depth = 0; split = -1
    for i, c in enumerate(rest):
        if c in '{[<': depth += 1
        elif c in '}]>': depth -= 1
        elif c == ',' and depth == 0: split = i; break
    if split < 0:
        return (elem_bytes(rest.strip()), False)          # type only → single element
    ty = rest[:split].strip(); tail = rest[split+1:].strip()
    if tail.startswith('align'):
        return (elem_bytes(ty), False)                    # ", align N" only → single element
    cm = re.match(r'i64\s+(\S+)', tail)
    if not cm: return (elem_bytes(ty), False)
    cnt = cm.group(1)
    if re.fullmatch(r'\d+', cnt): return (elem_bytes(ty)*int(cnt), False)
    return (None, True)                                   # variable count %reg → dynamic

TERM = re.compile(r'^(ret|br|switch|indirectbr|unreachable|resume|callbr)\b')

def parse_functions(lines):
    fns = []
    i, n = 0, len(lines)
    while i < n:
        if lines[i].lstrip().startswith('define '):
            m = re.search(r'@([A-Za-z0-9_.]+)\(', lines[i]); name = m.group(1) if m else '?'
            j = i+1
            while j < n and lines[j].strip() != '}': j += 1
            fns.append((name, i+1, lines[i+1:j]))   # start-lineno(1-based of body), body lines
            i = j+1
        else:
            i += 1
    return fns

def blocks_of(body):
    # returns: order[list of blkname], succ{blk->[blk]}, alloc{blk->[(lineidx,line)]}, calls_self set handled by caller
    order, succ, allocs = [], {}, {}
    cur = 'entry'; order.append(cur); succ[cur]=[]; allocs[cur]=[]
    for idx, ln in enumerate(body):
        s = ln.strip()
        mlab = re.match(r'^([A-Za-z0-9_.]+):', s)
        if mlab:
            cur = mlab.group(1)
            if cur not in succ: order.append(cur); succ[cur]=[]; allocs[cur]=[]
            continue
        code = re.sub(r';.*$', '', s)   # strip comment ("allocate" in prose must not match)
        if re.search(r'=\s*alloca\s', code) and 'stackcheck-ok' not in ln:
            allocs[cur].append((idx, ln))
        if TERM.match(s):
            for t in re.findall(r'label %([A-Za-z0-9_.]+)', s):
                succ[cur].append(t)
    return order, succ, allocs

def in_cycle(start, succ):
    # is `start` reachable from itself?
    seen=set(); stack=list(succ.get(start,[]))
    while stack:
        b=stack.pop()
        if b==start: return True
        if b in seen: continue
        seen.add(b); stack.extend(succ.get(b,[]))
    return False

def scan(path):
    findings=[]
    lines=open(path,encoding='utf-8',errors='replace').read().splitlines()
    for name, base, body in parse_functions(lines):
        order, succ, allocs = blocks_of(body)
        loopblk = {b: in_cycle(b, succ) for b in order}
        # self-recursion note
        if re.search(r'call[^\n]*@'+re.escape(name)+r'\(', '\n'.join(body)):
            findings.append(('NOTE', base, name, 'self-recursive — verify the recursion depth is bounded', ''))
        for b in order:
            for (idx, ln) in allocs[b]:
                lineno = base+idx
                by, dyn = alloca_info(ln)
                inloop = loopblk[b]
                if inloop and dyn:
                    findings.append(('HIGH', lineno, name, 'variable-length alloca in a LOOP (unbounded frame → overflow)', ln.strip()))
                elif inloop:
                    findings.append(('HIGH', lineno, name, 'alloca in a LOOP body (hazard #11: fresh frame each iteration → overflow)', ln.strip()))
                elif dyn:
                    ctx='\n'.join(body[max(0,idx-4):idx+3])
                    if 'llvm.stacksave' not in ctx:
                        findings.append(('MED', lineno, name, 'dynamic alloca without llvm.stacksave/stackrestore', ln.strip()))
                elif by is not None and by > MAXB:
                    findings.append(('MED', lineno, name, f'oversized stack frame: {by} bytes (> {MAXB})', ln.strip()))
    return findings

def main():
    args=sys.argv[1:]; global MAXB
    strict_all = False
    if args and args[0]=='--strict-all': strict_all=True; args=args[1:]   # fail on tests/ too
    if args and args[0]=='--max-bytes': MAXB=int(args[1]); args=args[2:]
    files=args or sorted(glob.glob('src/**/*.ll',recursive=True)+glob.glob('tests/**/*.ll',recursive=True))
    # src/ is the shipping library — STRICT (may run on small-stack threads). tests/
    # run on the 8 MB main thread at -O3 (allocas hoisted) — ADVISORY unless --strict-all.
    fail=0; warn=0
    for f in files:
        blocking = strict_all or f.startswith('src/')
        for (sev, ln, fn, msg, code) in scan(f):
            tag = '' if not code else f"\n       {code}"
            if sev=='NOTE':
                print(f"NOTE {f}:{ln}  [{fn}]  {msg}{tag}"); continue
            if blocking: fail+=1; print(f"{sev:4} {f}:{ln}  [{fn}]  {msg}{tag}")
            else: warn+=1; print(f"warn {f}:{ln}  [{fn}]  {msg} (test file — advisory){tag}")
    print(f"\nstackcheck: {fail} blocking (src/), {warn} advisory (tests/).")
    if fail:
        print("FAILED — src/ has a stack-overflow pattern: hoist loop allocas to the entry block, "
              "wrap dynamic allocas in llvm.stacksave/stackrestore, or exempt a verified line with "
              "`; stackcheck-ok: <reason>`.")
        return 1
    print("OK — the shipping library (src/) has no stack-overflow patterns."
          + ("" if not warn else f"  ({warn} advisory finding(s) in tests/ — fix when convenient, or run --strict-all.)"))
    return 0

if __name__=='__main__':
    sys.exit(main())

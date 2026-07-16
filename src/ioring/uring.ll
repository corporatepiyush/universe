; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     http://www.apache.org/licenses/LICENSE-2.0
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS,
; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
; See the License for the specific language governing permissions and
; limitations under the License.

; io_uring async IO backend  ---  LINUX ONLY (kernel >= 5.1; tested >= 6.15).
;
; This module speaks the raw Linux io_uring kernel ABI directly (syscalls
; io_uring_setup=425 / io_uring_enter=426, plus mmap/munmap/close). It maps the
; SQ ring, CQ ring and SQE array into this process and hands out a single-
; allocation handle that carries every base pointer and the head/tail/mask
; pointers the ring protocol needs. There is NO dependency on any userspace
; io_uring helper library — the ABI (struct layouts + syscall numbers) is the
; public kernel interface and is implemented here from first principles.
;
; On non-Linux targets the module still COMPILES (pure IR, no target lines);
; at runtime `universe_ioring_ring_available` returns false (io_uring_setup is
; not a valid syscall) so callers fall back to a posix path. NEVER assume this
; runs anywhere but Linux.
;
; ---------------------------------------------------------------------------
; KERNEL ABI OFFSETS (bytes) -- the correctness core; verified against the
; public uapi/linux/io_uring.h layout for LP64 little-endian (x86_64/aarch64).
;
; struct io_uring_params (passed to io_uring_setup, total 120 bytes):
;   +0   u32 sq_entries          +20  u32 features
;   +4   u32 cq_entries          +24  u32 wq_fd
;   +8   u32 flags               +28  u32 resv[3] (28,32,36)
;   +12  u32 sq_thread_cpu       +40  struct io_sqring_offsets sq_off (40 B)
;   +16  u32 sq_thread_idle      +80  struct io_cqring_offsets cq_off (40 B)
;
;   sq_off (absolute offsets inside params):
;     +40 head  +44 tail  +48 ring_mask  +52 ring_entries
;     +56 flags +60 dropped +64 array   +68 resv1  +72 user_addr(u64)
;   cq_off (absolute offsets inside params):
;     +80 head  +84 tail  +88 ring_mask  +92 ring_entries
;     +96 overflow +100 cqes +104 flags +108 resv1 +112 user_addr(u64)
;
; struct io_uring_sqe (64 bytes):
;   +0  u8  opcode      +8  u64 off/addr2   +24 u32 len       +40 u16 buf_index
;   +1  u8  flags       +16 u64 addr        +28 u32 op_flags  +42 u16 personality
;   +2  u16 ioprio                          +32 u64 user_data +44 s32 splice_fd_in
;   +4  s32 fd                                                +48 u64 addr3
;                                                             +56 u64 __pad2
;
; struct io_uring_cqe (16 bytes):  +0 u64 user_data  +8 s32 res  +12 u32 flags
;
; mmap regions (PROT_READ|WRITE=3, MAP_SHARED=1, over the ring fd):
;   SQ ring : off IORING_OFF_SQ_RING=0          size = sq_off.array + sq_entries*4
;   CQ ring : off IORING_OFF_CQ_RING=0x8000000  size = cq_off.cqes  + cq_entries*16
;   SQEs    : off IORING_OFF_SQES  =0x10000000  size = sq_entries*64
;
; io_uring_enter flags: IORING_ENTER_GETEVENTS = 1.
; io_uring_op:  READV=1 WRITEV=2 ACCEPT=13 CONNECT=16 CLOSE=19 READ=22
;               WRITE=23 SEND=26 RECV=27.
; ---------------------------------------------------------------------------
;
; MEMORY ORDERING (this is a single producer/consumer *with the kernel* -- the
; kernel runs the other side on another CPU, so the orderings are real; this is
; NOT the deferred lock-free wave, it is the ring protocol the ABI mandates):
;   * SQ tail publish (submit):  the SQEs are written with plain stores, then
;     the shared SQ tail is stored RELEASE. Release guarantees the kernel sees
;     the fully-written SQEs before it observes the advanced tail.
;   * SQ head consume (prep full-check): the kernel-owned SQ head is loaded
;     ACQUIRE before we reuse a slot. Acquire keeps the reuse (overwrite) from
;     being reordered ahead of learning the kernel has freed the slot.
;   * CQ tail (kernel produces): loaded ACQUIRE in peek. Acquire guarantees the
;     CQE payload the kernel wrote is visible before we read the CQE.
;   * CQ head publish (cqe_seen): stored RELEASE so the kernel does not recycle
;     a CQE slot until we have finished reading it.
;   * Our shadow SQ tail (prepared-but-unsubmitted counter) and our own CQ head
;     value are process-local single-writer state: plain / monotonic access.
;
; Handle layout (one malloc, 144 bytes):
;   +0   i32  ring_fd
;   +8   ptr  sq_khead        (kernel-owned SQ head, u32 in ring)
;   +16  ptr  sq_ktail        (we publish, u32 in ring)
;   +24  i32  sq_ring_mask    (constant value copied out of the ring)
;   +28  i32  sq_ring_entries (constant value)
;   +32  ptr  sq_array        (index -> sqe map; identity-filled at setup)
;   +40  ptr  sqes            (SQE array base)
;   +48  i32  sq_tail_shadow  (local prepared counter, free-running)
;   +56  ptr  cq_khead        (we publish, u32 in ring)
;   +64  ptr  cq_ktail        (kernel-owned CQ tail, u32 in ring)
;   +72  i32  cq_ring_mask
;   +76  i32  cq_ring_entries
;   +80  ptr  cqes            (CQE array base)
;   +88  ptr  sq_ring_ptr     +96  i64 sq_ring_sz   (for munmap)
;   +104 ptr  cq_ring_ptr     +112 i64 cq_ring_sz
;   +120 ptr  sqes_ptr        +128 i64 sqes_sz
;
; API (universe_ioring_*):
;   ptr  ring_setup(i64 entries, i64 flags)             ; null on failure
;   i1   ring_available()                                ; probe (Linux+kernel)
;   i32  prep_read (ring, i32 fd, ptr buf,  i32 len, i64 off, i64 ud)
;   i32  prep_write(ring, i32 fd, ptr buf,  i32 len, i64 off, i64 ud)
;   i32  prep_readv (ring, i32 fd, ptr iov, i32 nr, i64 off, i64 ud)
;   i32  prep_writev(ring, i32 fd, ptr iov, i32 nr, i64 off, i64 ud)
;   i32  prep_recv (ring, i32 fd, ptr buf, i32 len, i32 msg_flags, i64 ud)
;   i32  prep_send (ring, i32 fd, ptr buf, i32 len, i32 msg_flags, i64 ud)
;   i32  prep_accept (ring, i32 fd, ptr addr, ptr addrlen, i32 flags, i64 ud)
;   i32  prep_connect(ring, i32 fd, ptr addr, i64 addrlen, i64 ud)
;   i32  prep_close(ring, i32 fd, i64 ud)
;   i32  submit(ring)                                    ; -> n submitted / -errno
;   i32  submit_and_wait(ring, i64 wait_nr)
;   i32  peek_cqe(ring, ptr out16)   ; 0 ok / 4 empty / 1 null ; does not consume
;   i32  wait_cqe(ring, ptr out16)   ; blocks for one completion
;   i32  cqe_seen(ring)              ; advance CQ head by one
;   void ring_destroy(ring)

declare i64 @syscall(i64, ...)
declare ptr @mmap(ptr, i64, i32, i32, i32, i64)
declare i32 @munmap(ptr, i64)
declare i32 @close(i32)
declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)

; ---------------------------------------------------------------------------
; ring_setup: io_uring_setup + map the three regions + build the handle.
; ---------------------------------------------------------------------------
define ptr @universe_ioring_ring_setup(i64 %entries, i64 %flags) local_unnamed_addr #1 {
entry:
  %p = alloca [128 x i8], align 8
  call void @llvm.memset.p0.i64(ptr nonnull %p, i8 0, i64 128, i1 false)
  %p.flags = getelementptr inbounds nuw i8, ptr %p, i64 8
  %flags32 = trunc i64 %flags to i32
  store i32 %flags32, ptr %p.flags, align 4
  %fd64 = call i64 (i64, ...) @syscall(i64 425, i64 %entries, ptr nonnull %p)
  %fd.bad = icmp slt i64 %fd64, 0
  br i1 %fd.bad, label %fail0, label %sizes, !prof !0

sizes:
  %fd = trunc i64 %fd64 to i32
  %sqe.n32 = load i32, ptr %p, align 4
  %p.cqe = getelementptr inbounds nuw i8, ptr %p, i64 4
  %cqe.n32 = load i32, ptr %p.cqe, align 4
  %p.sqarr = getelementptr inbounds nuw i8, ptr %p, i64 64
  %sqoff.array = load i32, ptr %p.sqarr, align 4
  %p.cqcqes = getelementptr inbounds nuw i8, ptr %p, i64 100
  %cqoff.cqes = load i32, ptr %p.cqcqes, align 4
  %sqe.n64 = zext i32 %sqe.n32 to i64
  %cqe.n64 = zext i32 %cqe.n32 to i64
  %sqarr.b = shl i64 %sqe.n64, 2
  %sqoff.array64 = zext i32 %sqoff.array to i64
  %sq.ring.sz = add i64 %sqoff.array64, %sqarr.b
  %cqcqe.b = shl i64 %cqe.n64, 4
  %cqoff.cqes64 = zext i32 %cqoff.cqes to i64
  %cq.ring.sz = add i64 %cqoff.cqes64, %cqcqe.b
  %sqes.sz = shl i64 %sqe.n64, 6
  br label %map.sq

map.sq:
  %sq.ring = call ptr @mmap(ptr null, i64 %sq.ring.sz, i32 3, i32 1, i32 %fd, i64 0)
  %sq.ring.i = ptrtoint ptr %sq.ring to i64
  %sq.mapfail = icmp eq i64 %sq.ring.i, -1
  br i1 %sq.mapfail, label %fail1, label %map.cq, !prof !0

map.cq:
  %cq.ring = call ptr @mmap(ptr null, i64 %cq.ring.sz, i32 3, i32 1, i32 %fd, i64 134217728)
  %cq.ring.i = ptrtoint ptr %cq.ring to i64
  %cq.mapfail = icmp eq i64 %cq.ring.i, -1
  br i1 %cq.mapfail, label %fail2, label %map.sqes, !prof !0

map.sqes:
  %sqes = call ptr @mmap(ptr null, i64 %sqes.sz, i32 3, i32 1, i32 %fd, i64 268435456)
  %sqes.i = ptrtoint ptr %sqes to i64
  %sqes.mapfail = icmp eq i64 %sqes.i, -1
  br i1 %sqes.mapfail, label %fail3, label %build, !prof !0

build:
  %h = call ptr @malloc(i64 144)
  %h.null = icmp eq ptr %h, null
  br i1 %h.null, label %fail4, label %fill.handle, !prof !0

fill.handle:
  ; --- derive SQ field pointers from sq_ring base + sq_off.* ---
  %p.sqhead = getelementptr inbounds nuw i8, ptr %p, i64 40
  %sqoff.head = load i32, ptr %p.sqhead, align 4
  %sqoff.head64 = zext i32 %sqoff.head to i64
  %sq.khead = getelementptr inbounds i8, ptr %sq.ring, i64 %sqoff.head64
  %p.sqtail = getelementptr inbounds nuw i8, ptr %p, i64 44
  %sqoff.tail = load i32, ptr %p.sqtail, align 4
  %sqoff.tail64 = zext i32 %sqoff.tail to i64
  %sq.ktail = getelementptr inbounds i8, ptr %sq.ring, i64 %sqoff.tail64
  %p.sqmask = getelementptr inbounds nuw i8, ptr %p, i64 48
  %sqoff.mask = load i32, ptr %p.sqmask, align 4
  %sqoff.mask64 = zext i32 %sqoff.mask to i64
  %sq.mask.p = getelementptr inbounds i8, ptr %sq.ring, i64 %sqoff.mask64
  %sq.mask = load i32, ptr %sq.mask.p, align 4
  %p.sqents = getelementptr inbounds nuw i8, ptr %p, i64 52
  %sqoff.ents = load i32, ptr %p.sqents, align 4
  %sqoff.ents64 = zext i32 %sqoff.ents to i64
  %sq.ents.p = getelementptr inbounds i8, ptr %sq.ring, i64 %sqoff.ents64
  %sq.ents = load i32, ptr %sq.ents.p, align 4
  %sq.array = getelementptr inbounds i8, ptr %sq.ring, i64 %sqoff.array64
  %sq.ktail.v = load i32, ptr %sq.ktail, align 4
  ; --- derive CQ field pointers from cq_ring base + cq_off.* ---
  %p.cqhead = getelementptr inbounds nuw i8, ptr %p, i64 80
  %cqoff.head = load i32, ptr %p.cqhead, align 4
  %cqoff.head64 = zext i32 %cqoff.head to i64
  %cq.khead = getelementptr inbounds i8, ptr %cq.ring, i64 %cqoff.head64
  %p.cqtail = getelementptr inbounds nuw i8, ptr %p, i64 84
  %cqoff.tail = load i32, ptr %p.cqtail, align 4
  %cqoff.tail64 = zext i32 %cqoff.tail to i64
  %cq.ktail = getelementptr inbounds i8, ptr %cq.ring, i64 %cqoff.tail64
  %p.cqmask = getelementptr inbounds nuw i8, ptr %p, i64 88
  %cqoff.mask = load i32, ptr %p.cqmask, align 4
  %cqoff.mask64 = zext i32 %cqoff.mask to i64
  %cq.mask.p = getelementptr inbounds i8, ptr %cq.ring, i64 %cqoff.mask64
  %cq.mask = load i32, ptr %cq.mask.p, align 4
  %p.cqents = getelementptr inbounds nuw i8, ptr %p, i64 92
  %cqoff.ents = load i32, ptr %p.cqents, align 4
  %cqoff.ents64 = zext i32 %cqoff.ents to i64
  %cq.ents.p = getelementptr inbounds i8, ptr %cq.ring, i64 %cqoff.ents64
  %cq.ents = load i32, ptr %cq.ents.p, align 4
  %cqes = getelementptr inbounds i8, ptr %cq.ring, i64 %cqoff.cqes64
  ; --- store handle fields ---
  store i32 %fd, ptr %h, align 8
  %h.sqkhead = getelementptr inbounds nuw i8, ptr %h, i64 8
  store ptr %sq.khead, ptr %h.sqkhead, align 8
  %h.sqktail = getelementptr inbounds nuw i8, ptr %h, i64 16
  store ptr %sq.ktail, ptr %h.sqktail, align 8
  %h.sqmask = getelementptr inbounds nuw i8, ptr %h, i64 24
  store i32 %sq.mask, ptr %h.sqmask, align 4
  %h.sqents = getelementptr inbounds nuw i8, ptr %h, i64 28
  store i32 %sq.ents, ptr %h.sqents, align 4
  %h.sqarray = getelementptr inbounds nuw i8, ptr %h, i64 32
  store ptr %sq.array, ptr %h.sqarray, align 8
  %h.sqes = getelementptr inbounds nuw i8, ptr %h, i64 40
  store ptr %sqes, ptr %h.sqes, align 8
  %h.shadow = getelementptr inbounds nuw i8, ptr %h, i64 48
  store i32 %sq.ktail.v, ptr %h.shadow, align 4
  %h.cqkhead = getelementptr inbounds nuw i8, ptr %h, i64 56
  store ptr %cq.khead, ptr %h.cqkhead, align 8
  %h.cqktail = getelementptr inbounds nuw i8, ptr %h, i64 64
  store ptr %cq.ktail, ptr %h.cqktail, align 8
  %h.cqmask = getelementptr inbounds nuw i8, ptr %h, i64 72
  store i32 %cq.mask, ptr %h.cqmask, align 4
  %h.cqents = getelementptr inbounds nuw i8, ptr %h, i64 76
  store i32 %cq.ents, ptr %h.cqents, align 4
  %h.cqes = getelementptr inbounds nuw i8, ptr %h, i64 80
  store ptr %cqes, ptr %h.cqes, align 8
  %h.sqring = getelementptr inbounds nuw i8, ptr %h, i64 88
  store ptr %sq.ring, ptr %h.sqring, align 8
  %h.sqringsz = getelementptr inbounds nuw i8, ptr %h, i64 96
  store i64 %sq.ring.sz, ptr %h.sqringsz, align 8
  %h.cqring = getelementptr inbounds nuw i8, ptr %h, i64 104
  store ptr %cq.ring, ptr %h.cqring, align 8
  %h.cqringsz = getelementptr inbounds nuw i8, ptr %h, i64 112
  store i64 %cq.ring.sz, ptr %h.cqringsz, align 8
  %h.sqesptr = getelementptr inbounds nuw i8, ptr %h, i64 120
  store ptr %sqes, ptr %h.sqesptr, align 8
  %h.sqessz = getelementptr inbounds nuw i8, ptr %h, i64 128
  store i64 %sqes.sz, ptr %h.sqessz, align 8
  br label %idmap

idmap:
  ; identity-fill sq_array[i] = i so the SQ tail directly indexes SQEs.
  %i = phi i64 [ 0, %fill.handle ], [ %i.next, %idmap.body ]
  %ents64 = zext i32 %sq.ents to i64
  %i.done = icmp uge i64 %i, %ents64
  br i1 %i.done, label %ok, label %idmap.body

idmap.body:
  %arr.slot = getelementptr inbounds nuw i32, ptr %sq.array, i64 %i
  %i32 = trunc i64 %i to i32
  store i32 %i32, ptr %arr.slot, align 4
  %i.next = add nuw nsw i64 %i, 1
  br label %idmap

ok:
  ret ptr %h

fail4:
  call void @munmap(ptr %sqes, i64 %sqes.sz)
  br label %fail3c
fail3:
  br label %fail3c
fail3c:
  call void @munmap(ptr %cq.ring, i64 %cq.ring.sz)
  br label %fail2c
fail2:
  br label %fail2c
fail2c:
  call void @munmap(ptr %sq.ring, i64 %sq.ring.sz)
  br label %fail1c
fail1:
  br label %fail1c
fail1c:
  %fd.c = trunc i64 %fd64 to i32
  %ign = call i32 @close(i32 %fd.c)
  ret ptr null
fail0:
  ret ptr null
}

; ---------------------------------------------------------------------------
; ring_available: probe whether io_uring_setup works on this host/kernel.
; ---------------------------------------------------------------------------
define i1 @universe_ioring_ring_available() local_unnamed_addr #1 {
entry:
  %p = alloca [128 x i8], align 8
  call void @llvm.memset.p0.i64(ptr nonnull %p, i8 0, i64 128, i1 false)
  %fd64 = call i64 (i64, ...) @syscall(i64 425, i64 2, ptr nonnull %p)
  %ok = icmp sge i64 %fd64, 0
  br i1 %ok, label %close, label %no

close:
  %fd = trunc i64 %fd64 to i32
  %ign = call i32 @close(i32 %fd)
  ret i1 true

no:
  ret i1 false
}

; ---------------------------------------------------------------------------
; internal: fill one SQE at the shadow tail and advance it. err 1 null / 6 full.
; ---------------------------------------------------------------------------
define internal i32 @ioring_fill(ptr %h, i32 %opcode, i32 %fd, i64 %off, i64 %addr, i32 %len, i32 %opflags, i64 %user_data) #0 {
entry:
  %h.null = icmp eq ptr %h, null
  br i1 %h.null, label %err.null, label %load, !prof !0

err.null:
  ret i32 1

load:
  %h.sqkhead = getelementptr inbounds nuw i8, ptr %h, i64 8
  %sq.khead.p = load ptr, ptr %h.sqkhead, align 8
  %h.sqmask = getelementptr inbounds nuw i8, ptr %h, i64 24
  %mask = load i32, ptr %h.sqmask, align 4
  %h.sqents = getelementptr inbounds nuw i8, ptr %h, i64 28
  %ents = load i32, ptr %h.sqents, align 4
  %h.sqes = getelementptr inbounds nuw i8, ptr %h, i64 40
  %sqes = load ptr, ptr %h.sqes, align 8
  %h.shadow = getelementptr inbounds nuw i8, ptr %h, i64 48
  %shadow = load i32, ptr %h.shadow, align 4
  %khead = load atomic i32, ptr %sq.khead.p acquire, align 4
  %inflight = sub i32 %shadow, %khead
  %full = icmp uge i32 %inflight, %ents
  br i1 %full, label %err.full, label %fill, !prof !0

err.full:
  ret i32 6

fill:
  %slot = and i32 %shadow, %mask
  %slot64 = zext i32 %slot to i64
  %sqe.off = shl i64 %slot64, 6
  %sqe = getelementptr inbounds nuw i8, ptr %sqes, i64 %sqe.off
  call void @llvm.memset.p0.i64(ptr %sqe, i8 0, i64 64, i1 false)
  %op8 = trunc i32 %opcode to i8
  store i8 %op8, ptr %sqe, align 1
  %sqe.fd = getelementptr inbounds nuw i8, ptr %sqe, i64 4
  store i32 %fd, ptr %sqe.fd, align 4
  %sqe.off8 = getelementptr inbounds nuw i8, ptr %sqe, i64 8
  store i64 %off, ptr %sqe.off8, align 8
  %sqe.addr = getelementptr inbounds nuw i8, ptr %sqe, i64 16
  store i64 %addr, ptr %sqe.addr, align 8
  %sqe.len = getelementptr inbounds nuw i8, ptr %sqe, i64 24
  store i32 %len, ptr %sqe.len, align 4
  %sqe.opf = getelementptr inbounds nuw i8, ptr %sqe, i64 28
  store i32 %opflags, ptr %sqe.opf, align 4
  %sqe.ud = getelementptr inbounds nuw i8, ptr %sqe, i64 32
  store i64 %user_data, ptr %sqe.ud, align 8
  %shadow.n = add i32 %shadow, 1
  store i32 %shadow.n, ptr %h.shadow, align 4
  ret i32 0
}

; ---------------------------------------------------------------------------
; SQE builders (thin wrappers over ioring_fill). opcodes per the ABI table.
; ---------------------------------------------------------------------------
define i32 @universe_ioring_prep_read(ptr %ring, i32 %fd, ptr %buf, i32 %len, i64 %off, i64 %ud) local_unnamed_addr #0 {
entry:
  %addr = ptrtoint ptr %buf to i64
  %r = tail call i32 @ioring_fill(ptr %ring, i32 22, i32 %fd, i64 %off, i64 %addr, i32 %len, i32 0, i64 %ud)
  ret i32 %r
}

define i32 @universe_ioring_prep_write(ptr %ring, i32 %fd, ptr %buf, i32 %len, i64 %off, i64 %ud) local_unnamed_addr #0 {
entry:
  %addr = ptrtoint ptr %buf to i64
  %r = tail call i32 @ioring_fill(ptr %ring, i32 23, i32 %fd, i64 %off, i64 %addr, i32 %len, i32 0, i64 %ud)
  ret i32 %r
}

define i32 @universe_ioring_prep_readv(ptr %ring, i32 %fd, ptr %iov, i32 %nr, i64 %off, i64 %ud) local_unnamed_addr #0 {
entry:
  %addr = ptrtoint ptr %iov to i64
  %r = tail call i32 @ioring_fill(ptr %ring, i32 1, i32 %fd, i64 %off, i64 %addr, i32 %nr, i32 0, i64 %ud)
  ret i32 %r
}

define i32 @universe_ioring_prep_writev(ptr %ring, i32 %fd, ptr %iov, i32 %nr, i64 %off, i64 %ud) local_unnamed_addr #0 {
entry:
  %addr = ptrtoint ptr %iov to i64
  %r = tail call i32 @ioring_fill(ptr %ring, i32 2, i32 %fd, i64 %off, i64 %addr, i32 %nr, i32 0, i64 %ud)
  ret i32 %r
}

define i32 @universe_ioring_prep_recv(ptr %ring, i32 %fd, ptr %buf, i32 %len, i32 %msg_flags, i64 %ud) local_unnamed_addr #0 {
entry:
  %addr = ptrtoint ptr %buf to i64
  %r = tail call i32 @ioring_fill(ptr %ring, i32 27, i32 %fd, i64 0, i64 %addr, i32 %len, i32 %msg_flags, i64 %ud)
  ret i32 %r
}

define i32 @universe_ioring_prep_send(ptr %ring, i32 %fd, ptr %buf, i32 %len, i32 %msg_flags, i64 %ud) local_unnamed_addr #0 {
entry:
  %addr = ptrtoint ptr %buf to i64
  %r = tail call i32 @ioring_fill(ptr %ring, i32 26, i32 %fd, i64 0, i64 %addr, i32 %len, i32 %msg_flags, i64 %ud)
  ret i32 %r
}

define i32 @universe_ioring_prep_accept(ptr %ring, i32 %fd, ptr %addr, ptr %addrlen, i32 %flags, i64 %ud) local_unnamed_addr #0 {
entry:
  %addr.i = ptrtoint ptr %addr to i64
  %alen.i = ptrtoint ptr %addrlen to i64
  %r = tail call i32 @ioring_fill(ptr %ring, i32 13, i32 %fd, i64 %alen.i, i64 %addr.i, i32 0, i32 %flags, i64 %ud)
  ret i32 %r
}

define i32 @universe_ioring_prep_connect(ptr %ring, i32 %fd, ptr %addr, i64 %addrlen, i64 %ud) local_unnamed_addr #0 {
entry:
  %addr.i = ptrtoint ptr %addr to i64
  %r = tail call i32 @ioring_fill(ptr %ring, i32 16, i32 %fd, i64 %addrlen, i64 %addr.i, i32 0, i32 0, i64 %ud)
  ret i32 %r
}

define i32 @universe_ioring_prep_close(ptr %ring, i32 %fd, i64 %ud) local_unnamed_addr #0 {
entry:
  %r = tail call i32 @ioring_fill(ptr %ring, i32 19, i32 %fd, i64 0, i64 0, i32 0, i32 0, i64 %ud)
  ret i32 %r
}

; ---------------------------------------------------------------------------
; internal: publish the shadow tail (release) and io_uring_enter.
; ---------------------------------------------------------------------------
define internal i64 @ioring_flush_enter(ptr %h, i64 %min_complete, i64 %flags) #1 {
entry:
  %h.null = icmp eq ptr %h, null
  br i1 %h.null, label %err.null, label %go, !prof !0

err.null:
  ret i64 -1

go:
  %fd32 = load i32, ptr %h, align 8
  %fd64 = zext i32 %fd32 to i64
  %h.sqktail = getelementptr inbounds nuw i8, ptr %h, i64 16
  %sq.ktail.p = load ptr, ptr %h.sqktail, align 8
  %h.shadow = getelementptr inbounds nuw i8, ptr %h, i64 48
  %shadow = load i32, ptr %h.shadow, align 4
  %published = load atomic i32, ptr %sq.ktail.p monotonic, align 4
  %to.submit32 = sub i32 %shadow, %published
  store atomic i32 %shadow, ptr %sq.ktail.p release, align 4
  %to.submit = zext i32 %to.submit32 to i64
  %ret = call i64 (i64, ...) @syscall(i64 426, i64 %fd64, i64 %to.submit, i64 %min_complete, i64 %flags, ptr null, i64 0)
  ret i64 %ret
}

define i32 @universe_ioring_submit(ptr %ring) local_unnamed_addr #1 {
entry:
  %r = call i64 @ioring_flush_enter(ptr %ring, i64 0, i64 0)
  %r32 = trunc i64 %r to i32
  ret i32 %r32
}

define i32 @universe_ioring_submit_and_wait(ptr %ring, i64 %wait_nr) local_unnamed_addr #1 {
entry:
  %r = call i64 @ioring_flush_enter(ptr %ring, i64 %wait_nr, i64 1)
  %r32 = trunc i64 %r to i32
  ret i32 %r32
}

; ---------------------------------------------------------------------------
; peek_cqe: copy the head CQE (16 B) to out without consuming. 0/4/1.
; ---------------------------------------------------------------------------
define i32 @universe_ioring_peek_cqe(ptr %ring, ptr %out) local_unnamed_addr #0 {
entry:
  %r.null = icmp eq ptr %ring, null
  %o.null = icmp eq ptr %out, null
  %any.null = or i1 %r.null, %o.null
  br i1 %any.null, label %err.null, label %load, !prof !0

err.null:
  ret i32 1

load:
  %h.cqkhead = getelementptr inbounds nuw i8, ptr %ring, i64 56
  %cq.khead.p = load ptr, ptr %h.cqkhead, align 8
  %h.cqktail = getelementptr inbounds nuw i8, ptr %ring, i64 64
  %cq.ktail.p = load ptr, ptr %h.cqktail, align 8
  %h.cqmask = getelementptr inbounds nuw i8, ptr %ring, i64 72
  %mask = load i32, ptr %h.cqmask, align 4
  %h.cqes = getelementptr inbounds nuw i8, ptr %ring, i64 80
  %cqes = load ptr, ptr %h.cqes, align 8
  %khead = load atomic i32, ptr %cq.khead.p monotonic, align 4
  %ktail = load atomic i32, ptr %cq.ktail.p acquire, align 4
  %empty = icmp eq i32 %khead, %ktail
  br i1 %empty, label %err.empty, label %copy, !prof !0

err.empty:
  ret i32 4

copy:
  %idx = and i32 %khead, %mask
  %idx64 = zext i32 %idx to i64
  %cqe.off = shl i64 %idx64, 4
  %cqe = getelementptr inbounds nuw i8, ptr %cqes, i64 %cqe.off
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %cqe, i64 16, i1 false)
  ret i32 0
}

; ---------------------------------------------------------------------------
; wait_cqe: block until at least one CQE is available, then peek it.
; ---------------------------------------------------------------------------
define i32 @universe_ioring_wait_cqe(ptr %ring, ptr %out) local_unnamed_addr #1 {
entry:
  br label %try

try:
  %pr = call i32 @universe_ioring_peek_cqe(ptr %ring, ptr %out)
  %have = icmp eq i32 %pr, 0
  br i1 %have, label %done, label %check.null

check.null:
  %is.null = icmp eq i32 %pr, 1
  br i1 %is.null, label %ret.null, label %wait

wait:
  %er = call i64 @ioring_flush_enter(ptr %ring, i64 1, i64 1)
  %efail = icmp slt i64 %er, 0
  br i1 %efail, label %ret.io, label %try

done:
  ret i32 0
ret.null:
  ret i32 1
ret.io:
  ret i32 15
}

; ---------------------------------------------------------------------------
; cqe_seen: advance the CQ head by one (release), returning the slot.
; ---------------------------------------------------------------------------
define i32 @universe_ioring_cqe_seen(ptr %ring) local_unnamed_addr #0 {
entry:
  %r.null = icmp eq ptr %ring, null
  br i1 %r.null, label %err.null, label %adv, !prof !0

err.null:
  ret i32 1

adv:
  %h.cqkhead = getelementptr inbounds nuw i8, ptr %ring, i64 56
  %cq.khead.p = load ptr, ptr %h.cqkhead, align 8
  %khead = load atomic i32, ptr %cq.khead.p monotonic, align 4
  %khead.n = add i32 %khead, 1
  store atomic i32 %khead.n, ptr %cq.khead.p release, align 4
  ret i32 0
}

; ---------------------------------------------------------------------------
; ring_destroy: unmap the three regions, close the fd, free the handle.
; ---------------------------------------------------------------------------
define void @universe_ioring_ring_destroy(ptr %ring) local_unnamed_addr #1 {
entry:
  %r.null = icmp eq ptr %ring, null
  br i1 %r.null, label %done, label %do.free, !prof !0

do.free:
  %h.sqring = getelementptr inbounds nuw i8, ptr %ring, i64 88
  %sq.ring = load ptr, ptr %h.sqring, align 8
  %h.sqringsz = getelementptr inbounds nuw i8, ptr %ring, i64 96
  %sq.ring.sz = load i64, ptr %h.sqringsz, align 8
  %u0 = call i32 @munmap(ptr %sq.ring, i64 %sq.ring.sz)
  %h.cqring = getelementptr inbounds nuw i8, ptr %ring, i64 104
  %cq.ring = load ptr, ptr %h.cqring, align 8
  %h.cqringsz = getelementptr inbounds nuw i8, ptr %ring, i64 112
  %cq.ring.sz = load i64, ptr %h.cqringsz, align 8
  %u1 = call i32 @munmap(ptr %cq.ring, i64 %cq.ring.sz)
  %h.sqesptr = getelementptr inbounds nuw i8, ptr %ring, i64 120
  %sqes = load ptr, ptr %h.sqesptr, align 8
  %h.sqessz = getelementptr inbounds nuw i8, ptr %ring, i64 128
  %sqes.sz = load i64, ptr %h.sqessz, align 8
  %u2 = call i32 @munmap(ptr %sqes, i64 %sqes.sz)
  %fd = load i32, ptr %ring, align 8
  %uc = call i32 @close(i32 %fd)
  call void @free(ptr nonnull %ring)
  br label %done

done:
  ret void
}

attributes #0 = { nounwind willreturn }
attributes #1 = { nounwind }

!0 = !{!"branch_weights", i32 1, i32 2000}

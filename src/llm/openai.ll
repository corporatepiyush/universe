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

; Minimal OpenAI-protocol chat-completions client (universe_llm_*). Builds the
; request JSON by hand (there is no JSON encoder in the SDK — only a tokenizer),
; ships it over the plaintext HTTP/1.1 client (src/http), and walks the response
; with the JSON tokenizer (src/parse/json). Handles both non-streaming
; completions and Server-Sent-Events (SSE) streaming deltas.
;
; SCOPE / NO TLS: this client speaks PLAINTEXT HTTP only (TLS is deferred to the
; hardening phase). It therefore targets LOCAL OpenAI-compatible servers on the
; loopback/LAN — llama.cpp, vLLM, Ollama, LM Studio, etc. It is NOT for the
; public api.openai.com endpoint (which requires TLS). The Authorization: Bearer
; header is still emitted for local servers that check a token.
;
; DESIGN:
;   * COMPUTE / MEMORY / IO SEPARATION. Three pure phases, no syscall in any
;     compute loop:
;       - BUILD (compute+memory): @universe_llm_build_request assembles the body
;         into a caller buffer via bounded appends; @universe_llm_json_escape is
;         the string-escaper the tokenizer cannot provide. Zero IO.
;       - IO: the http client (src/http) does the single POST + response read;
;         we never touch a socket inside a parse/build loop.
;       - PARSE (compute+memory): @universe_llm_parse_response drives the
;         zero-copy JSON tokenizer over the response buffer, emitting only
;         offsets/lengths, then unescapes the ONE content span into the caller
;         buffer. SSE lines are split with @universe_simd_find_byte (newline).
;   * RESPONSE WALK is a small hand-written navigator over the pull tokenizer:
;     a top-level object loop dispatches keys "choices"/"usage"/"error"; the
;     choice object is descended to <container>.content where <container> is
;     "message" (non-streaming) or "delta" (streaming). Non-matching values are
;     consumed with a depth-counted @ll_skip (begin/end object/array balance) so
;     any document shape is handled without recursion into the SDK. Navigation
;     nesting is bounded (top->choices->choice->content = 4 frames).
;   * ERROR CODES (i32, per conventions.md): 0 OK, 1 NULL, 2 OOM (scanner),
;     3 SIZE_OVERFLOW (build/escape buffer too small — surfaced as 6 FULL for
;     the response side), 5 NOT_FOUND (no content field), 6 FULL (out buffer too
;     small), 11 INVALID_STATE (a top-level `error` object is present — an API
;     error), 13 PARSE (malformed JSON), 15 IO (transport). SSE process returns
;     4 when the terminating `data: [DONE]` sentinel is seen.
;   * message input (per element, 32 B, caller array):
;       +0 role_ptr  +8 role_len  +16 content_ptr  +24 content_len
;   * config (caller-allocated, 72 B) for the networked convenience path:
;       +0 ip(i32) +4 port(i32) +8 host_ptr +16 host_len +24 path_ptr
;       +32 path_len +40 model_ptr +48 model_len +56 apikey_ptr +64 apikey_len
;   * ZERO-ALLOC on the build/parse hot paths except the one-shot JSON scanner
;     object (reused for the whole document). The SSE path creates one scanner
;     per `data:` line — acceptable at line rate; a pooled/reset scanner is a
;     later optimization.
;
; HARDENING-TODO: no TLS, no auth-token zeroization, no response size caps
; beyond the caller buffers, no retry/backoff. Deferred to the hardening wave.

declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)

; src/simd
declare i64 @universe_simd_find_byte(ptr readonly, i64, i8)

; src/parse/json
declare ptr @universe_parse_json_scanner_create(ptr, i64, i32)
declare void @universe_parse_json_scanner_destroy(ptr)
declare i32 @universe_parse_json_next(ptr, ptr)
declare i64 @universe_parse_json_unescape(ptr, ptr, i64)

; src/http (transport)
declare i32 @universe_http_client_request(ptr, ptr, i64, ptr, i64, i64, ptr, i64, ptr, i64, i32, ptr, ptr, i64)

; ------------------------------------------------------------ string constants
@ll.r_open    = private unnamed_addr constant [10 x i8] c"{\22model\22:\22"
@ll.r_msgs    = private unnamed_addr constant [14 x i8] c"\22,\22messages\22:["
@ll.r_role    = private unnamed_addr constant [9 x i8]  c"{\22role\22:\22"
@ll.r_content = private unnamed_addr constant [13 x i8] c"\22,\22content\22:\22"
@ll.r_endmsg  = private unnamed_addr constant [2 x i8]  c"\22}"
@ll.r_comma   = private unnamed_addr constant [1 x i8]  c","
@ll.r_stream  = private unnamed_addr constant [11 x i8] c"],\22stream\22:"
@ll.r_true    = private unnamed_addr constant [4 x i8]  c"true"
@ll.r_false   = private unnamed_addr constant [5 x i8]  c"false"
@ll.r_close   = private unnamed_addr constant [1 x i8]  c"}"

@ll.k_choices = private unnamed_addr constant [7 x i8]  c"choices"
@ll.k_message = private unnamed_addr constant [7 x i8]  c"message"
@ll.k_delta   = private unnamed_addr constant [5 x i8]  c"delta"
@ll.k_content = private unnamed_addr constant [7 x i8]  c"content"
@ll.k_finish  = private unnamed_addr constant [13 x i8] c"finish_reason"
@ll.k_usage   = private unnamed_addr constant [5 x i8]  c"usage"
@ll.k_error   = private unnamed_addr constant [5 x i8]  c"error"
@ll.k_prompt  = private unnamed_addr constant [13 x i8] c"prompt_tokens"
@ll.k_comp    = private unnamed_addr constant [17 x i8] c"completion_tokens"
@ll.k_total   = private unnamed_addr constant [12 x i8] c"total_tokens"

@ll.h_host    = private unnamed_addr constant [4 x i8]  c"Host"
@ll.h_ctype   = private unnamed_addr constant [12 x i8] c"Content-Type"
@ll.h_auth    = private unnamed_addr constant [13 x i8] c"Authorization"
@ll.v_json    = private unnamed_addr constant [16 x i8] c"application/json"
@ll.v_bearer  = private unnamed_addr constant [7 x i8]  c"Bearer "
@ll.method    = private unnamed_addr constant [4 x i8]  c"POST"

@ll.sse_data  = private unnamed_addr constant [5 x i8]  c"data:"
@ll.sse_done  = private unnamed_addr constant [6 x i8]  c"[DONE]"
@ll.hex       = private unnamed_addr constant [16 x i8] c"0123456789abcdef"

; =========================================================== JSON string escape

; universe_llm_json_escape(dst, dstcap, src, len) -> i64
;   Escape src[0..len) as JSON string CONTENT (no surrounding quotes) into dst.
;   Returns bytes written, or -1 on null args or if dstcap is too small.
define i64 @universe_llm_json_escape(ptr %dst, i64 %dstcap, ptr readonly %src, i64 %len) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %dst, null
  %sn = icmp eq ptr %src, null
  %bad = or i1 %dn, %sn
  br i1 %bad, label %err, label %chk0

chk0:
  %z = icmp eq i64 %len, 0
  br i1 %z, label %ret0, label %loop

ret0:
  ret i64 0

loop:
  %i = phi i64 [ 0, %chk0 ], [ %i.n1, %e1 ], [ %i.n2, %e2 ], [ %i.n6, %e6 ]
  %o = phi i64 [ 0, %chk0 ], [ %o.n1, %e1 ], [ %o.n2, %e2 ], [ %o.n6, %e6 ]
  %sp = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %c = load i8, ptr %sp, align 1
  %cz = zext i8 %c to i32
  switch i32 %cz, label %chkctrl [
    i32 34,  label %esc.q
    i32 92,  label %esc.bs
    i32 10,  label %esc.n
    i32 13,  label %esc.r
    i32 9,   label %esc.t
    i32 8,   label %esc.b
    i32 12,  label %esc.f
  ]

esc.q:
  br label %e2.pre
esc.bs:
  br label %e2.pre
esc.n:
  br label %e2.pre
esc.r:
  br label %e2.pre
esc.t:
  br label %e2.pre
esc.b:
  br label %e2.pre
esc.f:
  br label %e2.pre

e2.pre:
  ; second byte of the 2-char escape: \" \\ \n \r \t \b \f
  %sec = phi i8 [ 34, %esc.q ], [ 92, %esc.bs ], [ 110, %esc.n ], [ 114, %esc.r ], [ 116, %esc.t ], [ 98, %esc.b ], [ 102, %esc.f ]
  %o2end = add i64 %o, 2
  %o2fit = icmp ule i64 %o2end, %dstcap
  br i1 %o2fit, label %e2, label %err

e2:
  %d2a = getelementptr inbounds nuw i8, ptr %dst, i64 %o
  store i8 92, ptr %d2a, align 1
  %o2b = add i64 %o, 1
  %d2b = getelementptr inbounds nuw i8, ptr %dst, i64 %o2b
  store i8 %sec, ptr %d2b, align 1
  %o.n2 = add i64 %o, 2
  %i.n2 = add i64 %i, 1
  %more2 = icmp ult i64 %i.n2, %len
  br i1 %more2, label %loop, label %done

chkctrl:
  %isctrl = icmp ult i32 %cz, 32
  br i1 %isctrl, label %e6.pre, label %e1.pre

e1.pre:
  %o1end = add i64 %o, 1
  %o1fit = icmp ule i64 %o1end, %dstcap
  br i1 %o1fit, label %e1, label %err

e1:
  %d1 = getelementptr inbounds nuw i8, ptr %dst, i64 %o
  store i8 %c, ptr %d1, align 1
  %o.n1 = add i64 %o, 1
  %i.n1 = add i64 %i, 1
  %more1 = icmp ult i64 %i.n1, %len
  br i1 %more1, label %loop, label %done

e6.pre:
  ; \u00XX for control bytes < 0x20
  %o6end = add i64 %o, 6
  %o6fit = icmp ule i64 %o6end, %dstcap
  br i1 %o6fit, label %e6, label %err

e6:
  %d6a = getelementptr inbounds nuw i8, ptr %dst, i64 %o
  store i8 92, ptr %d6a, align 1
  %o6_1 = add i64 %o, 1
  %d6b = getelementptr inbounds nuw i8, ptr %dst, i64 %o6_1
  store i8 117, ptr %d6b, align 1
  %o6_2 = add i64 %o, 2
  %d6c = getelementptr inbounds nuw i8, ptr %dst, i64 %o6_2
  store i8 48, ptr %d6c, align 1
  %o6_3 = add i64 %o, 3
  %d6d = getelementptr inbounds nuw i8, ptr %dst, i64 %o6_3
  store i8 48, ptr %d6d, align 1
  %hi = lshr i32 %cz, 4
  %hix = zext i32 %hi to i64
  %hip = getelementptr inbounds nuw i8, ptr @ll.hex, i64 %hix
  %hic = load i8, ptr %hip, align 1
  %o6_4 = add i64 %o, 4
  %d6e = getelementptr inbounds nuw i8, ptr %dst, i64 %o6_4
  store i8 %hic, ptr %d6e, align 1
  %lo = and i32 %cz, 15
  %lox = zext i32 %lo to i64
  %lop = getelementptr inbounds nuw i8, ptr @ll.hex, i64 %lox
  %loc = load i8, ptr %lop, align 1
  %o6_5 = add i64 %o, 5
  %d6f = getelementptr inbounds nuw i8, ptr %dst, i64 %o6_5
  store i8 %loc, ptr %d6f, align 1
  %o.n6 = add i64 %o, 6
  %i.n6 = add i64 %i, 1
  %more6 = icmp ult i64 %i.n6, %len
  br i1 %more6, label %loop, label %done

done:
  %ofin = phi i64 [ %o.n1, %e1 ], [ %o.n2, %e2 ], [ %o.n6, %e6 ]
  ret i64 %ofin

err:
  ret i64 -1
}

; =============================================================== request builder

; append raw literal; returns new offset or -1 (propagates a prior -1).
define internal i64 @ll_app_raw(ptr %dst, i64 %cap, i64 %o, ptr readonly %s, i64 %n) #1 {
entry:
  %prev = icmp slt i64 %o, 0
  br i1 %prev, label %bad, label %chk
chk:
  %end = add i64 %o, %n
  %fit = icmp ule i64 %end, %cap
  br i1 %fit, label %do, label %bad
do:
  %d = getelementptr inbounds nuw i8, ptr %dst, i64 %o
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 %n, i1 false)
  ret i64 %end
bad:
  ret i64 -1
}

; append escaped string; returns new offset or -1.
define internal i64 @ll_app_esc(ptr %dst, i64 %cap, i64 %o, ptr readonly %s, i64 %n) #1 {
entry:
  %prev = icmp slt i64 %o, 0
  br i1 %prev, label %bad, label %do
do:
  %avail = sub i64 %cap, %o
  %dp = getelementptr inbounds nuw i8, ptr %dst, i64 %o
  %w = call i64 @universe_llm_json_escape(ptr %dp, i64 %avail, ptr %s, i64 %n)
  %wbad = icmp slt i64 %w, 0
  br i1 %wbad, label %bad, label %ok
ok:
  %end = add i64 %o, %w
  ret i64 %end
bad:
  ret i64 -1
}

; universe_llm_build_request(dst, dstcap, model,mlen, msgs, nmsgs, stream) -> i64
;   Assemble the chat-completions body. msgs is an array of 32-byte entries
;   {role_ptr, role_len, content_ptr, content_len}. Returns body length, or -1
;   on null args / buffer overflow.
define i64 @universe_llm_build_request(ptr %dst, i64 %dstcap, ptr readonly %model, i64 %mlen, ptr readonly %msgs, i64 %nmsgs, i32 %stream) local_unnamed_addr #2 {
entry:
  %dn = icmp eq ptr %dst, null
  %mn = icmp eq ptr %model, null
  %sn = icmp eq ptr %msgs, null
  %b0 = or i1 %dn, %mn
  %bad = or i1 %b0, %sn
  br i1 %bad, label %err, label %open

err:
  ret i64 -1

open:
  %o1 = call i64 @ll_app_raw(ptr %dst, i64 %dstcap, i64 0, ptr @ll.r_open, i64 10)
  %o2 = call i64 @ll_app_esc(ptr %dst, i64 %dstcap, i64 %o1, ptr %model, i64 %mlen)
  %o3 = call i64 @ll_app_raw(ptr %dst, i64 %dstcap, i64 %o2, ptr @ll.r_msgs, i64 14)
  %empty = icmp eq i64 %nmsgs, 0
  br i1 %empty, label %tail, label %mloop

mloop:
  %k = phi i64 [ 0, %open ], [ %k.n, %mnext ]
  %o = phi i64 [ %o3, %open ], [ %o.n, %mnext ]
  %first = icmp eq i64 %k, 0
  br i1 %first, label %entrymsg, label %withcomma

withcomma:
  %oc = call i64 @ll_app_raw(ptr %dst, i64 %dstcap, i64 %o, ptr @ll.r_comma, i64 1)
  br label %entrymsg

entrymsg:
  %ostart = phi i64 [ %o, %mloop ], [ %oc, %withcomma ]
  %eoff = shl i64 %k, 5
  %ep = getelementptr inbounds nuw i8, ptr %msgs, i64 %eoff
  %rolep = load ptr, ptr %ep, align 8
  %rlp = getelementptr inbounds nuw i8, ptr %ep, i64 8
  %rolelen = load i64, ptr %rlp, align 8
  %cpp = getelementptr inbounds nuw i8, ptr %ep, i64 16
  %contentp = load ptr, ptr %cpp, align 8
  %clp = getelementptr inbounds nuw i8, ptr %ep, i64 24
  %contentlen = load i64, ptr %clp, align 8
  %oa = call i64 @ll_app_raw(ptr %dst, i64 %dstcap, i64 %ostart, ptr @ll.r_role, i64 9)
  %ob = call i64 @ll_app_esc(ptr %dst, i64 %dstcap, i64 %oa, ptr %rolep, i64 %rolelen)
  %occ = call i64 @ll_app_raw(ptr %dst, i64 %dstcap, i64 %ob, ptr @ll.r_content, i64 13)
  %od = call i64 @ll_app_esc(ptr %dst, i64 %dstcap, i64 %occ, ptr %contentp, i64 %contentlen)
  %o.n = call i64 @ll_app_raw(ptr %dst, i64 %dstcap, i64 %od, ptr @ll.r_endmsg, i64 2)
  br label %mnext

mnext:
  %k.n = add i64 %k, 1
  %more = icmp ult i64 %k.n, %nmsgs
  br i1 %more, label %mloop, label %tail

tail:
  %otail = phi i64 [ %o3, %open ], [ %o.n, %mnext ]
  %os = call i64 @ll_app_raw(ptr %dst, i64 %dstcap, i64 %otail, ptr @ll.r_stream, i64 11)
  %wantstream = icmp ne i32 %stream, 0
  br i1 %wantstream, label %st.t, label %st.f

st.t:
  %ot = call i64 @ll_app_raw(ptr %dst, i64 %dstcap, i64 %os, ptr @ll.r_true, i64 4)
  br label %closej

st.f:
  %of = call i64 @ll_app_raw(ptr %dst, i64 %dstcap, i64 %os, ptr @ll.r_false, i64 5)
  br label %closej

closej:
  %ojoin = phi i64 [ %ot, %st.t ], [ %of, %st.f ]
  %ofin = call i64 @ll_app_raw(ptr %dst, i64 %dstcap, i64 %ojoin, ptr @ll.r_close, i64 1)
  ret i64 %ofin
}

; =============================================================== response walker

; token accessor: read the type after a next(); -1 on scanner error.
define internal i32 @ll_rn(ptr %sc, ptr %tok) #3 {
entry:
  %st = call i32 @universe_parse_json_next(ptr %sc, ptr %tok)
  %ok = icmp eq i32 %st, 0
  br i1 %ok, label %rd, label %bad
rd:
  %ty = load i32, ptr %tok, align 4
  ret i32 %ty
bad:
  ret i32 -1
}

; raw key/literal byte compare.
define internal i1 @ll_keq(ptr readonly %body, i64 %off, i64 %len, ptr readonly %lit, i64 %litlen) #0 {
entry:
  %leneq = icmp eq i64 %len, %litlen
  br i1 %leneq, label %chk, label %no
chk:
  %z = icmp eq i64 %len, 0
  br i1 %z, label %yes, label %loop
loop:
  %k = phi i64 [ 0, %chk ], [ %k.n, %cont ]
  %bi = add i64 %off, %k
  %bp = getelementptr inbounds nuw i8, ptr %body, i64 %bi
  %bc = load i8, ptr %bp, align 1
  %lp = getelementptr inbounds nuw i8, ptr %lit, i64 %k
  %lc = load i8, ptr %lp, align 1
  %eq = icmp eq i8 %bc, %lc
  br i1 %eq, label %cont, label %no
cont:
  %k.n = add i64 %k, 1
  %more = icmp ult i64 %k.n, %len
  br i1 %more, label %loop, label %yes
yes:
  ret i1 true
no:
  ret i1 false
}

; parse a non-negative decimal integer from a byte slice (stops at non-digit).
define internal i64 @ll_pu64(ptr readonly %s, i64 %n) #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %ret0, label %loop
ret0:
  ret i64 0
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %acc = phi i64 [ 0, %entry ], [ %acc.n, %cont ]
  %p = getelementptr inbounds nuw i8, ptr %s, i64 %i
  %c = load i8, ptr %p, align 1
  %cz = zext i8 %c to i64
  %d = sub i64 %cz, 48
  %bad = icmp ugt i64 %d, 9
  br i1 %bad, label %done, label %cont
cont:
  %acc.m = mul i64 %acc, 10
  %acc.n = add i64 %acc.m, %d
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done
done:
  %r = phi i64 [ %acc, %loop ], [ %acc.n, %cont ]
  ret i64 %r
}

; consume the remainder of an already-read value token (vtype). Scalars are no-op;
; objects/arrays are depth-balanced. 0 OK, 13 PARSE.
define internal i32 @ll_skip(ptr %sc, ptr %tok, i32 %vtype) #3 {
entry:
  %isobj = icmp eq i32 %vtype, 0
  %isarr = icmp eq i32 %vtype, 2
  %cont0 = or i1 %isobj, %isarr
  br i1 %cont0, label %loop, label %ret0
ret0:
  ret i32 0
loop:
  %depth = phi i64 [ 1, %entry ], [ %depth.n, %step ]
  %t = call i32 @ll_rn(ptr %sc, ptr %tok)
  %terr = icmp slt i32 %t, 0
  br i1 %terr, label %bad, label %chkend
chkend:
  %isend = icmp eq i32 %t, 10
  br i1 %isend, label %bad, label %step
step:
  %inc0 = icmp eq i32 %t, 0
  %inc2 = icmp eq i32 %t, 2
  %isinc = or i1 %inc0, %inc2
  %dec1 = icmp eq i32 %t, 1
  %dec3 = icmp eq i32 %t, 3
  %isdec = or i1 %dec1, %dec3
  %addv = select i1 %isinc, i64 1, i64 0
  %subv = select i1 %isdec, i64 1, i64 0
  %d1 = add i64 %depth, %addv
  %depth.n = sub i64 %d1, %subv
  %closed = icmp eq i64 %depth.n, 0
  br i1 %closed, label %ret0, label %loop
bad:
  ret i32 13
}

; descend a container object (message/delta), extract "content" -> out.
define internal i32 @ll_content_obj(ptr %sc, ptr %tok, ptr %out, i64 %outcap, ptr %out_len, ptr %foundp) #3 {
entry:
  %body = load ptr, ptr %sc, align 8
  br label %loop
loop:
  %t = call i32 @ll_rn(ptr %sc, ptr %tok)
  %terr = icmp slt i32 %t, 0
  br i1 %terr, label %bad, label %chkend
chkend:
  %isend = icmp eq i32 %t, 1
  br i1 %isend, label %ret0, label %chkkey
chkkey:
  %iskey = icmp eq i32 %t, 9
  br i1 %iskey, label %key, label %bad
key:
  %offp = getelementptr inbounds nuw i8, ptr %tok, i64 8
  %koff = load i64, ptr %offp, align 8
  %lenp = getelementptr inbounds nuw i8, ptr %tok, i64 16
  %klen = load i64, ptr %lenp, align 8
  %isc = call i1 @ll_keq(ptr %body, i64 %koff, i64 %klen, ptr @ll.k_content, i64 7)
  br i1 %isc, label %want, label %other
want:
  %tv = call i32 @ll_rn(ptr %sc, ptr %tok)
  %tverr = icmp slt i32 %tv, 0
  br i1 %tverr, label %bad, label %wchk
wchk:
  %isstr = icmp eq i32 %tv, 4
  br i1 %isstr, label %store, label %wskip
wskip:
  %ws = call i32 @ll_skip(ptr %sc, ptr %tok, i32 %tv)
  %wsbad = icmp ne i32 %ws, 0
  br i1 %wsbad, label %bad, label %loop
store:
  %voffp = getelementptr inbounds nuw i8, ptr %tok, i64 8
  %voff = load i64, ptr %voffp, align 8
  %vlenp = getelementptr inbounds nuw i8, ptr %tok, i64 16
  %vlen = load i64, ptr %vlenp, align 8
  %outn = icmp eq ptr %out, null
  br i1 %outn, label %loop, label %docopy
docopy:
  %over = icmp ugt i64 %vlen, %outcap
  br i1 %over, label %full, label %unesc
unesc:
  %vp = getelementptr inbounds nuw i8, ptr %body, i64 %voff
  %w = call i64 @universe_parse_json_unescape(ptr %out, ptr %vp, i64 %vlen)
  %wbad = icmp slt i64 %w, 0
  br i1 %wbad, label %bad, label %wrote
wrote:
  %oln = icmp eq ptr %out_len, null
  br i1 %oln, label %setfound, label %wlen
wlen:
  store i64 %w, ptr %out_len, align 8
  br label %setfound
setfound:
  store i8 1, ptr %foundp, align 1
  br label %loop
other:
  %ov = call i32 @ll_rn(ptr %sc, ptr %tok)
  %overr = icmp slt i32 %ov, 0
  br i1 %overr, label %bad, label %oskip
oskip:
  %os = call i32 @ll_skip(ptr %sc, ptr %tok, i32 %ov)
  %osbad = icmp ne i32 %os, 0
  br i1 %osbad, label %bad, label %loop
ret0:
  ret i32 0
full:
  ret i32 6
bad:
  ret i32 13
}

; descend one choice object; find <container>.content and (optionally) finish_reason.
define internal i32 @ll_choice_obj(ptr %sc, ptr %tok, ptr %ckey, i64 %ckeylen, ptr %out, i64 %outcap, ptr %out_len, ptr %out_finish, i64 %fcap, ptr %out_flen, ptr %foundp) #3 {
entry:
  %body = load ptr, ptr %sc, align 8
  br label %loop
loop:
  %t = call i32 @ll_rn(ptr %sc, ptr %tok)
  %terr = icmp slt i32 %t, 0
  br i1 %terr, label %bad, label %chkend
chkend:
  %isend = icmp eq i32 %t, 1
  br i1 %isend, label %ret0, label %chkkey
chkkey:
  %iskey = icmp eq i32 %t, 9
  br i1 %iskey, label %key, label %bad
key:
  %offp = getelementptr inbounds nuw i8, ptr %tok, i64 8
  %koff = load i64, ptr %offp, align 8
  %lenp = getelementptr inbounds nuw i8, ptr %tok, i64 16
  %klen = load i64, ptr %lenp, align 8
  %isctr = call i1 @ll_keq(ptr %body, i64 %koff, i64 %klen, ptr %ckey, i64 %ckeylen)
  br i1 %isctr, label %container, label %chkfinish

container:
  %tv = call i32 @ll_rn(ptr %sc, ptr %tok)
  %tverr = icmp slt i32 %tv, 0
  br i1 %tverr, label %bad, label %cchk
cchk:
  %isobj = icmp eq i32 %tv, 0
  br i1 %isobj, label %descend, label %cskip
descend:
  %dr = call i32 @ll_content_obj(ptr %sc, ptr %tok, ptr %out, i64 %outcap, ptr %out_len, ptr %foundp)
  %drbad = icmp ne i32 %dr, 0
  br i1 %drbad, label %prop, label %loop
prop:
  ret i32 %dr
cskip:
  %cs = call i32 @ll_skip(ptr %sc, ptr %tok, i32 %tv)
  %csbad = icmp ne i32 %cs, 0
  br i1 %csbad, label %bad, label %loop

chkfinish:
  %hasf = icmp ne ptr %out_finish, null
  br i1 %hasf, label %finchk, label %other
finchk:
  %isf = call i1 @ll_keq(ptr %body, i64 %koff, i64 %klen, ptr @ll.k_finish, i64 13)
  br i1 %isf, label %fin, label %other
fin:
  %fv = call i32 @ll_rn(ptr %sc, ptr %tok)
  %fverr = icmp slt i32 %fv, 0
  br i1 %fverr, label %bad, label %fchk
fchk:
  %fstr = icmp eq i32 %fv, 4
  br i1 %fstr, label %fstore, label %fskip
fstore:
  %fvoffp = getelementptr inbounds nuw i8, ptr %tok, i64 8
  %fvoff = load i64, ptr %fvoffp, align 8
  %fvlenp = getelementptr inbounds nuw i8, ptr %tok, i64 16
  %fvlen = load i64, ptr %fvlenp, align 8
  %fover = icmp ugt i64 %fvlen, %fcap
  br i1 %fover, label %full, label %fw
fw:
  %fvp = getelementptr inbounds nuw i8, ptr %body, i64 %fvoff
  %fwn = call i64 @universe_parse_json_unescape(ptr %out_finish, ptr %fvp, i64 %fvlen)
  %fwbad = icmp slt i64 %fwn, 0
  br i1 %fwbad, label %bad, label %fwrote
fwrote:
  %fln = icmp eq ptr %out_flen, null
  br i1 %fln, label %loop, label %fwlen
fwlen:
  store i64 %fwn, ptr %out_flen, align 8
  br label %loop
fskip:
  %fs = call i32 @ll_skip(ptr %sc, ptr %tok, i32 %fv)
  %fsbad = icmp ne i32 %fs, 0
  br i1 %fsbad, label %bad, label %loop

other:
  %ov = call i32 @ll_rn(ptr %sc, ptr %tok)
  %overr = icmp slt i32 %ov, 0
  br i1 %overr, label %bad, label %oskip
oskip:
  %os = call i32 @ll_skip(ptr %sc, ptr %tok, i32 %ov)
  %osbad = icmp ne i32 %os, 0
  br i1 %osbad, label %bad, label %loop

ret0:
  ret i32 0
full:
  ret i32 6
bad:
  ret i32 13
}

; handle the "choices" array value (begin-array already consumed by caller).
define internal i32 @ll_choices(ptr %sc, ptr %tok, ptr %ckey, i64 %ckeylen, ptr %out, i64 %outcap, ptr %out_len, ptr %out_finish, i64 %fcap, ptr %out_flen, ptr %foundp) #3 {
entry:
  %t = call i32 @ll_rn(ptr %sc, ptr %tok)
  %terr = icmp slt i32 %t, 0
  br i1 %terr, label %bad, label %chkend
chkend:
  %isend = icmp eq i32 %t, 3
  br i1 %isend, label %ret0, label %first
first:
  %isobj = icmp eq i32 %t, 0
  br i1 %isobj, label %handle, label %skip1
handle:
  %hr = call i32 @ll_choice_obj(ptr %sc, ptr %tok, ptr %ckey, i64 %ckeylen, ptr %out, i64 %outcap, ptr %out_len, ptr %out_finish, i64 %fcap, ptr %out_flen, ptr %foundp)
  %hrbad = icmp ne i32 %hr, 0
  br i1 %hrbad, label %prop, label %drain
prop:
  ret i32 %hr
skip1:
  %s1 = call i32 @ll_skip(ptr %sc, ptr %tok, i32 %t)
  %s1bad = icmp ne i32 %s1, 0
  br i1 %s1bad, label %bad, label %drain
drain:
  %t2 = call i32 @ll_rn(ptr %sc, ptr %tok)
  %t2err = icmp slt i32 %t2, 0
  br i1 %t2err, label %bad, label %dchk
dchk:
  %dend = icmp eq i32 %t2, 3
  br i1 %dend, label %ret0, label %dskip
dskip:
  %ds = call i32 @ll_skip(ptr %sc, ptr %tok, i32 %t2)
  %dsbad = icmp ne i32 %ds, 0
  br i1 %dsbad, label %bad, label %drain
ret0:
  ret i32 0
bad:
  ret i32 13
}

; handle the "usage" object value (begin-object already consumed).
define internal i32 @ll_usage(ptr %sc, ptr %tok, ptr %out_usage) #3 {
entry:
  %body = load ptr, ptr %sc, align 8
  br label %loop
loop:
  %t = call i32 @ll_rn(ptr %sc, ptr %tok)
  %terr = icmp slt i32 %t, 0
  br i1 %terr, label %bad, label %chkend
chkend:
  %isend = icmp eq i32 %t, 1
  br i1 %isend, label %ret0, label %chkkey
chkkey:
  %iskey = icmp eq i32 %t, 9
  br i1 %iskey, label %key, label %bad
key:
  %offp = getelementptr inbounds nuw i8, ptr %tok, i64 8
  %koff = load i64, ptr %offp, align 8
  %lenp = getelementptr inbounds nuw i8, ptr %tok, i64 16
  %klen = load i64, ptr %lenp, align 8
  %isp = call i1 @ll_keq(ptr %body, i64 %koff, i64 %klen, ptr @ll.k_prompt, i64 13)
  %isco = call i1 @ll_keq(ptr %body, i64 %koff, i64 %klen, ptr @ll.k_comp, i64 17)
  %ist = call i1 @ll_keq(ptr %body, i64 %koff, i64 %klen, ptr @ll.k_total, i64 12)
  %w0 = select i1 %isp, i64 0, i64 -1
  %w1 = select i1 %isco, i64 1, i64 %w0
  %which = select i1 %ist, i64 2, i64 %w1
  %tv = call i32 @ll_rn(ptr %sc, ptr %tok)
  %tverr = icmp slt i32 %tv, 0
  br i1 %tverr, label %bad, label %vchk
vchk:
  %isnum = icmp eq i32 %tv, 5
  %recognized = icmp sge i64 %which, 0
  %take = and i1 %isnum, %recognized
  br i1 %take, label %num, label %skip
num:
  %voffp = getelementptr inbounds nuw i8, ptr %tok, i64 8
  %voff = load i64, ptr %voffp, align 8
  %vlenp = getelementptr inbounds nuw i8, ptr %tok, i64 16
  %vlen = load i64, ptr %vlenp, align 8
  %vp = getelementptr inbounds nuw i8, ptr %body, i64 %voff
  %val = call i64 @ll_pu64(ptr %vp, i64 %vlen)
  %slot = getelementptr inbounds nuw i64, ptr %out_usage, i64 %which
  store i64 %val, ptr %slot, align 8
  br label %loop
skip:
  %s = call i32 @ll_skip(ptr %sc, ptr %tok, i32 %tv)
  %sbad = icmp ne i32 %s, 0
  br i1 %sbad, label %bad, label %loop
ret0:
  ret i32 0
bad:
  ret i32 13
}

; ll_walk: top-level driver. ckey selects "message"/"delta". Nullable out params.
;   0 content found, 5 not found, 11 error object present, 13 parse, 2 OOM, 6 full.
define internal i32 @ll_walk(ptr %body, i64 %len, ptr %ckey, i64 %ckeylen, ptr %out, i64 %outcap, ptr %out_len, ptr %out_finish, i64 %fcap, ptr %out_flen, ptr %out_usage) #4 {
entry:
  %tok = alloca [24 x i8], align 8
  %foundp = alloca i8, align 1
  %haderrp = alloca i8, align 1
  store i8 0, ptr %foundp, align 1
  store i8 0, ptr %haderrp, align 1
  %sc = call ptr @universe_parse_json_scanner_create(ptr %body, i64 %len, i32 0)
  %scnull = icmp eq ptr %sc, null
  br i1 %scnull, label %oom, label %top0
oom:
  ret i32 2
top0:
  %t0 = call i32 @ll_rn(ptr %sc, ptr %tok)
  %isobj = icmp eq i32 %t0, 0
  br i1 %isobj, label %loop, label %parsefail

loop:
  %t = call i32 @ll_rn(ptr %sc, ptr %tok)
  %terr = icmp slt i32 %t, 0
  br i1 %terr, label %parsefail, label %chkend
chkend:
  %isend = icmp eq i32 %t, 1
  br i1 %isend, label %finish, label %chkkey
chkkey:
  %iskey = icmp eq i32 %t, 9
  br i1 %iskey, label %key, label %parsefail
key:
  %offp = getelementptr inbounds nuw i8, ptr %tok, i64 8
  %koff = load i64, ptr %offp, align 8
  %lenp = getelementptr inbounds nuw i8, ptr %tok, i64 16
  %klen = load i64, ptr %lenp, align 8
  %ischoices = call i1 @ll_keq(ptr %body, i64 %koff, i64 %klen, ptr @ll.k_choices, i64 7)
  br i1 %ischoices, label %choices, label %chkerr

choices:
  %cv = call i32 @ll_rn(ptr %sc, ptr %tok)
  %cverr = icmp slt i32 %cv, 0
  br i1 %cverr, label %parsefail, label %cvchk
cvchk:
  %isarr = icmp eq i32 %cv, 2
  br i1 %isarr, label %dochoices, label %cvskip
dochoices:
  %chr = call i32 @ll_choices(ptr %sc, ptr %tok, ptr %ckey, i64 %ckeylen, ptr %out, i64 %outcap, ptr %out_len, ptr %out_finish, i64 %fcap, ptr %out_flen, ptr %foundp)
  %chrbad = icmp ne i32 %chr, 0
  br i1 %chrbad, label %prop, label %loop
prop:
  call void @universe_parse_json_scanner_destroy(ptr %sc)
  ret i32 %chr
cvskip:
  %cvs = call i32 @ll_skip(ptr %sc, ptr %tok, i32 %cv)
  %cvsbad = icmp ne i32 %cvs, 0
  br i1 %cvsbad, label %parsefail, label %loop

chkerr:
  %iserr = call i1 @ll_keq(ptr %body, i64 %koff, i64 %klen, ptr @ll.k_error, i64 5)
  br i1 %iserr, label %doerr, label %chkusage
doerr:
  store i8 1, ptr %haderrp, align 1
  %ev = call i32 @ll_rn(ptr %sc, ptr %tok)
  %everr = icmp slt i32 %ev, 0
  br i1 %everr, label %parsefail, label %eskip
eskip:
  %es = call i32 @ll_skip(ptr %sc, ptr %tok, i32 %ev)
  %esbad = icmp ne i32 %es, 0
  br i1 %esbad, label %parsefail, label %loop

chkusage:
  %hasu = icmp ne ptr %out_usage, null
  br i1 %hasu, label %uchk, label %skipkey
uchk:
  %isusage = call i1 @ll_keq(ptr %body, i64 %koff, i64 %klen, ptr @ll.k_usage, i64 5)
  br i1 %isusage, label %dousage, label %skipkey
dousage:
  %uv = call i32 @ll_rn(ptr %sc, ptr %tok)
  %uverr = icmp slt i32 %uv, 0
  br i1 %uverr, label %parsefail, label %uvchk
uvchk:
  %uobj = icmp eq i32 %uv, 0
  br i1 %uobj, label %douobj, label %uskip
douobj:
  %ur = call i32 @ll_usage(ptr %sc, ptr %tok, ptr %out_usage)
  %urbad = icmp ne i32 %ur, 0
  br i1 %urbad, label %parsefail, label %loop
uskip:
  %us = call i32 @ll_skip(ptr %sc, ptr %tok, i32 %uv)
  %usbad = icmp ne i32 %us, 0
  br i1 %usbad, label %parsefail, label %loop

skipkey:
  %sv = call i32 @ll_rn(ptr %sc, ptr %tok)
  %sverr = icmp slt i32 %sv, 0
  br i1 %sverr, label %parsefail, label %svskip
svskip:
  %svs = call i32 @ll_skip(ptr %sc, ptr %tok, i32 %sv)
  %svsbad = icmp ne i32 %svs, 0
  br i1 %svsbad, label %parsefail, label %loop

finish:
  call void @universe_parse_json_scanner_destroy(ptr %sc)
  %haderr = load i8, ptr %haderrp, align 1
  %iserr2 = icmp ne i8 %haderr, 0
  br i1 %iserr2, label %reterr, label %chkfound
reterr:
  ret i32 11
chkfound:
  %found = load i8, ptr %foundp, align 1
  %isfound = icmp ne i8 %found, 0
  %code = select i1 %isfound, i32 0, i32 5
  ret i32 %code

parsefail:
  call void @universe_parse_json_scanner_destroy(ptr %sc)
  ret i32 13
}

; universe_llm_parse_response(body, len, out, outcap, out_len,
;   out_finish, fcap, out_flen, out_usage) -> i32
;   Walk choices[0].message.content. out_finish/out_usage may be null.
;   out_usage, if non-null, must point at 3 i64 slots (prompt, completion, total).
define i32 @universe_llm_parse_response(ptr %body, i64 %len, ptr %out, i64 %outcap, ptr %out_len, ptr %out_finish, i64 %fcap, ptr %out_flen, ptr %out_usage) local_unnamed_addr #2 {
entry:
  %bn = icmp eq ptr %body, null
  br i1 %bn, label %err, label %go
err:
  ret i32 1
go:
  %r = call i32 @ll_walk(ptr %body, i64 %len, ptr @ll.k_message, i64 7, ptr %out, i64 %outcap, ptr %out_len, ptr %out_finish, i64 %fcap, ptr %out_flen, ptr %out_usage)
  ret i32 %r
}

; universe_llm_extract_delta(body, len, out, outcap, out_len) -> i32
;   Walk choices[0].delta.content (one SSE chunk). 5 = no content in this delta.
define i32 @universe_llm_extract_delta(ptr %body, i64 %len, ptr %out, i64 %outcap, ptr %out_len) local_unnamed_addr #2 {
entry:
  %bn = icmp eq ptr %body, null
  br i1 %bn, label %err, label %go
err:
  ret i32 1
go:
  %r = call i32 @ll_walk(ptr %body, i64 %len, ptr @ll.k_delta, i64 5, ptr %out, i64 %outcap, ptr %out_len, ptr null, i64 0, ptr null, ptr null)
  ret i32 %r
}

; =============================================================== SSE processing

; universe_llm_sse_process(buf, len, scratch, scratchcap, cb, userdata, out_consumed) -> i32
;   Split buf on '\n'; for each complete "data:" line extract the delta content
;   into scratch and invoke cb(userdata, text, textlen). Stops on "data: [DONE]".
;   Returns 0 (all complete lines processed, more expected), 4 ([DONE] seen),
;   13 (a data line was malformed JSON), 1 (null). out_consumed (nullable) gets
;   the number of bytes consumed up to the last complete line.
;   cb type: void(ptr userdata, ptr text, i64 textlen).
define i32 @universe_llm_sse_process(ptr %buf, i64 %len, ptr %scratch, i64 %scratchcap, ptr %cb, ptr %userdata, ptr %out_consumed) local_unnamed_addr #2 {
entry:
  %clenp = alloca i64, align 8
  %bn = icmp eq ptr %buf, null
  %cn = icmp eq ptr %cb, null
  %bad = or i1 %bn, %cn
  br i1 %bad, label %err.null, label %loop
err.null:
  ret i32 1

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %advance ]
  %rem = sub i64 %len, %i
  %base = getelementptr inbounds nuw i8, ptr %buf, i64 %i
  %rel = call i64 @universe_simd_find_byte(ptr %base, i64 %rem, i8 10)
  %miss = icmp slt i64 %rel, 0
  br i1 %miss, label %done0, label %haveline

haveline:
  %lineend = add i64 %i, %rel
  ; trim a trailing '\r'
  %hascr = icmp ugt i64 %rel, 0
  br i1 %hascr, label %crchk, label %noline

crchk:
  %lastidx = sub i64 %lineend, 1
  %lastp = getelementptr inbounds nuw i8, ptr %buf, i64 %lastidx
  %lastb = load i8, ptr %lastp, align 1
  %iscr = icmp eq i8 %lastb, 13
  %trimmed = select i1 %iscr, i64 1, i64 0
  %linelen = sub i64 %rel, %trimmed
  br label %proc

noline:
  %linelen0 = phi i64 [ 0, %haveline ]
  br label %proc

proc:
  %ll = phi i64 [ %linelen, %crchk ], [ %linelen0, %noline ]
  ; is it a "data:" line?
  %isdata = icmp uge i64 %ll, 5
  br i1 %isdata, label %datachk, label %advance

datachk:
  %pfx = call i1 @ll_keq(ptr %buf, i64 %i, i64 5, ptr @ll.sse_data, i64 5)
  br i1 %pfx, label %payload, label %advance

payload:
  %ps0 = add i64 %i, 5
  %payloadend = add i64 %i, %ll
  ; skip one optional leading space
  %hasmore = icmp ult i64 %ps0, %payloadend
  br i1 %hasmore, label %spacechk, label %emptypayload

spacechk:
  %ps0p = getelementptr inbounds nuw i8, ptr %buf, i64 %ps0
  %ps0b = load i8, ptr %ps0p, align 1
  %issp = icmp eq i8 %ps0b, 32
  %skipsp = select i1 %issp, i64 1, i64 0
  %ps = add i64 %ps0, %skipsp
  %plen = sub i64 %payloadend, %ps
  ; check for [DONE]
  %isdone = call i1 @ll_keq(ptr %buf, i64 %ps, i64 %plen, ptr @ll.sse_done, i64 6)
  br i1 %isdone, label %done_sentinel, label %parseline

parseline:
  %pz = icmp eq i64 %plen, 0
  br i1 %pz, label %advance, label %doparse
doparse:
  %pp = getelementptr inbounds nuw i8, ptr %buf, i64 %ps
  %st = call i32 @universe_llm_extract_delta(ptr %pp, i64 %plen, ptr %scratch, i64 %scratchcap, ptr %clenp)
  %isok = icmp eq i32 %st, 0
  br i1 %isok, label %emit, label %chkparse
emit:
  %clen = load i64, ptr %clenp, align 8
  call void %cb(ptr %userdata, ptr %scratch, i64 %clen)
  br label %advance
chkparse:
  %ispe = icmp eq i32 %st, 13
  br i1 %ispe, label %reterr, label %advance
reterr:
  %consumed.e = add i64 %lineend, 1
  %ocn.e = icmp eq ptr %out_consumed, null
  br i1 %ocn.e, label %ret.parse, label %wc.e
wc.e:
  store i64 %consumed.e, ptr %out_consumed, align 8
  br label %ret.parse
ret.parse:
  ret i32 13

emptypayload:
  br label %advance

advance:
  %i.n = add i64 %lineend, 1
  %atend = icmp uge i64 %i.n, %len
  br i1 %atend, label %done0, label %loop

done_sentinel:
  %consumed.d = add i64 %lineend, 1
  %ocn.d = icmp eq ptr %out_consumed, null
  br i1 %ocn.d, label %ret.done, label %wc.d
wc.d:
  store i64 %consumed.d, ptr %out_consumed, align 8
  br label %ret.done
ret.done:
  ret i32 4

done0:
  %consumed.0 = phi i64 [ %i, %loop ], [ %i.n, %advance ]
  %ocn.0 = icmp eq ptr %out_consumed, null
  br i1 %ocn.0, label %ret.ok, label %wc.0
wc.0:
  store i64 %consumed.0, ptr %out_consumed, align 8
  br label %ret.ok
ret.ok:
  ret i32 0
}

; =============================================================== networked path

; universe_llm_config_init(cfg, ip, port, host,hlen, path,plen, model,mlen, key,klen)
;   Fill a caller 72-byte config. path may be null (caller supplies its own).
define void @universe_llm_config_init(ptr %cfg, i32 %ip, i32 %port, ptr %host, i64 %hlen, ptr %path, i64 %plen, ptr %model, i64 %mlen, ptr %key, i64 %klen) local_unnamed_addr #2 {
entry:
  %cn = icmp eq ptr %cfg, null
  br i1 %cn, label %done, label %do
do:
  store i32 %ip, ptr %cfg, align 4
  %pp = getelementptr inbounds nuw i8, ptr %cfg, i64 4
  store i32 %port, ptr %pp, align 4
  %hp = getelementptr inbounds nuw i8, ptr %cfg, i64 8
  store ptr %host, ptr %hp, align 8
  %hlp = getelementptr inbounds nuw i8, ptr %cfg, i64 16
  store i64 %hlen, ptr %hlp, align 8
  %pap = getelementptr inbounds nuw i8, ptr %cfg, i64 24
  store ptr %path, ptr %pap, align 8
  %plp = getelementptr inbounds nuw i8, ptr %cfg, i64 32
  store i64 %plen, ptr %plp, align 8
  %mp = getelementptr inbounds nuw i8, ptr %cfg, i64 40
  store ptr %model, ptr %mp, align 8
  %mlp = getelementptr inbounds nuw i8, ptr %cfg, i64 48
  store i64 %mlen, ptr %mlp, align 8
  %kp = getelementptr inbounds nuw i8, ptr %cfg, i64 56
  store ptr %key, ptr %kp, align 8
  %klp = getelementptr inbounds nuw i8, ptr %cfg, i64 64
  store i64 %klen, ptr %klp, align 8
  br label %done
done:
  ret void
}

; universe_llm_complete(conn, cfg, msgs, nmsgs, bodybuf, bodycap,
;   out, outcap, out_len, out_finish, fcap, out_flen, out_usage) -> i32
;   Build a non-streaming request, POST it over conn, and parse the reply.
;   Emits Host, Content-Type: application/json, and (if key set) Authorization.
;   Returns 0 OK, 3 SIZE_OVERFLOW (buffers too small), and the transport/parse
;   codes otherwise. An HTTP status >= 400 yields 11 (INVALID_STATE).
define i32 @universe_llm_complete(ptr %conn, ptr %cfg, ptr %msgs, i64 %nmsgs, ptr %bodybuf, i64 %bodycap, ptr %out, i64 %outcap, ptr %out_len, ptr %out_finish, i64 %fcap, ptr %out_flen, ptr %out_usage) local_unnamed_addr #2 {
entry:
  %hdrs = alloca [96 x i8], align 8
  %authbuf = alloca [1088 x i8], align 8
  %resp_msg = alloca [96 x i8], align 8
  %resp_hdrs = alloca [2048 x i8], align 8
  %cn = icmp eq ptr %conn, null
  %gn = icmp eq ptr %cfg, null
  %anynull = or i1 %cn, %gn
  br i1 %anynull, label %err.null, label %fields
err.null:
  ret i32 1

fields:
  %mp = getelementptr inbounds nuw i8, ptr %cfg, i64 40
  %model = load ptr, ptr %mp, align 8
  %mlp = getelementptr inbounds nuw i8, ptr %cfg, i64 48
  %mlen = load i64, ptr %mlp, align 8
  %hp = getelementptr inbounds nuw i8, ptr %cfg, i64 8
  %host = load ptr, ptr %hp, align 8
  %hlp = getelementptr inbounds nuw i8, ptr %cfg, i64 16
  %hlen = load i64, ptr %hlp, align 8
  %pap = getelementptr inbounds nuw i8, ptr %cfg, i64 24
  %path = load ptr, ptr %pap, align 8
  %plp = getelementptr inbounds nuw i8, ptr %cfg, i64 32
  %plen = load i64, ptr %plp, align 8
  %kp = getelementptr inbounds nuw i8, ptr %cfg, i64 56
  %key = load ptr, ptr %kp, align 8
  %klp = getelementptr inbounds nuw i8, ptr %cfg, i64 64
  %klen = load i64, ptr %klp, align 8

  %blen = call i64 @universe_llm_build_request(ptr %bodybuf, i64 %bodycap, ptr %model, i64 %mlen, ptr %msgs, i64 %nmsgs, i32 0)
  %blbad = icmp slt i64 %blen, 0
  br i1 %blbad, label %overflow, label %hdr0

hdr0:
  ; entry 0: Host
  store ptr @ll.h_host, ptr %hdrs, align 8
  %h0l = getelementptr inbounds nuw i8, ptr %hdrs, i64 8
  store i64 4, ptr %h0l, align 8
  %h0v = getelementptr inbounds nuw i8, ptr %hdrs, i64 16
  store ptr %host, ptr %h0v, align 8
  %h0vl = getelementptr inbounds nuw i8, ptr %hdrs, i64 24
  store i64 %hlen, ptr %h0vl, align 8
  ; entry 1: Content-Type: application/json
  %h1 = getelementptr inbounds nuw i8, ptr %hdrs, i64 32
  store ptr @ll.h_ctype, ptr %h1, align 8
  %h1l = getelementptr inbounds nuw i8, ptr %hdrs, i64 40
  store i64 12, ptr %h1l, align 8
  %h1v = getelementptr inbounds nuw i8, ptr %hdrs, i64 48
  store ptr @ll.v_json, ptr %h1v, align 8
  %h1vl = getelementptr inbounds nuw i8, ptr %hdrs, i64 56
  store i64 16, ptr %h1vl, align 8
  ; optional entry 2: Authorization: Bearer <key>
  %haskey = icmp ne ptr %key, null
  %klok = icmp ugt i64 %klen, 0
  %wantauth = and i1 %haskey, %klok
  br i1 %wantauth, label %auth, label %sendreq

auth:
  %ktoobig = icmp ugt i64 %klen, 1080
  br i1 %ktoobig, label %overflow, label %buildauth
buildauth:
  call void @llvm.memcpy.p0.p0.i64(ptr %authbuf, ptr @ll.v_bearer, i64 7, i1 false)
  %ab7 = getelementptr inbounds nuw i8, ptr %authbuf, i64 7
  call void @llvm.memcpy.p0.p0.i64(ptr %ab7, ptr %key, i64 %klen, i1 false)
  %authlen = add i64 7, %klen
  %h2 = getelementptr inbounds nuw i8, ptr %hdrs, i64 64
  store ptr @ll.h_auth, ptr %h2, align 8
  %h2l = getelementptr inbounds nuw i8, ptr %hdrs, i64 72
  store i64 13, ptr %h2l, align 8
  %h2v = getelementptr inbounds nuw i8, ptr %hdrs, i64 80
  store ptr %authbuf, ptr %h2v, align 8
  %h2vl = getelementptr inbounds nuw i8, ptr %hdrs, i64 88
  store i64 %authlen, ptr %h2vl, align 8
  br label %sendreq

sendreq:
  %hcount = phi i64 [ 2, %hdr0 ], [ 3, %buildauth ]
  %st = call i32 @universe_http_client_request(ptr %conn, ptr @ll.method, i64 4, ptr %path, i64 %plen, i64 1, ptr %hdrs, i64 %hcount, ptr %bodybuf, i64 %blen, i32 1, ptr %resp_msg, ptr %resp_hdrs, i64 64)
  %stbad = icmp ne i32 %st, 0
  br i1 %stbad, label %rettransport, label %parse
rettransport:
  ret i32 %st

parse:
  %codep = getelementptr inbounds nuw i8, ptr %resp_msg, i64 40
  %code = load i64, ptr %codep, align 8
  %bpp = getelementptr inbounds nuw i8, ptr %resp_msg, i64 56
  %bodyptr = load ptr, ptr %bpp, align 8
  %blp = getelementptr inbounds nuw i8, ptr %resp_msg, i64 64
  %rblen = load i64, ptr %blp, align 8
  %pr = call i32 @universe_llm_parse_response(ptr %bodyptr, i64 %rblen, ptr %out, i64 %outcap, ptr %out_len, ptr %out_finish, i64 %fcap, ptr %out_flen, ptr %out_usage)
  %httperr = icmp uge i64 %code, 400
  br i1 %httperr, label %retstate, label %retparse
retstate:
  ret i32 11
retparse:
  ret i32 %pr

overflow:
  ret i32 3
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #1 = { nounwind }
attributes #2 = { nounwind }
attributes #3 = { nounwind }
attributes #4 = { nounwind }

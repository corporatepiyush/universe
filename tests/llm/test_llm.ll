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

; Tests for universe_llm_*: JSON string escaping, request building, non-stream
; response parsing (content + finish_reason), API-error object detection, SSE
; delta extraction, and a LIVE end-to-end call against a local OpenAI-compatible
; server (Ollama on 127.0.0.1:11434) that SKIPS cleanly if unreachable.

declare i64 @universe_llm_json_escape(ptr, i64, ptr, i64)
declare i64 @universe_llm_build_request(ptr, i64, ptr, i64, ptr, i64, i32)
declare i32 @universe_llm_parse_response(ptr, i64, ptr, i64, ptr, ptr, i64, ptr, ptr)
declare i32 @universe_llm_extract_delta(ptr, i64, ptr, i64, ptr)
declare void @universe_llm_config_init(ptr, i32, i32, ptr, i64, ptr, i64, ptr, i64, ptr, i64)
declare i32 @universe_llm_complete(ptr, ptr, ptr, i64, ptr, i64, ptr, i64, ptr, ptr, i64, ptr, ptr)
declare ptr @universe_http_connect(i32, i32, i64)
declare void @universe_http_conn_destroy(ptr)

declare i32 @memcmp(ptr, ptr, i64)
declare i64 @strlen(ptr)
declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i32 @ut_summary()

@g.body  = internal global [8192 x i8] zeroinitializer, align 16
@g.out   = internal global [4096 x i8] zeroinitializer, align 16
@g.esc   = internal global [64 x i8] zeroinitializer, align 16
@g.fin   = internal global [64 x i8] zeroinitializer, align 8
@g.usage = internal global [64 x i8] zeroinitializer, align 8
@g.outlen = internal global i64 0, align 8
@g.finlen = internal global i64 0, align 8

@c.model = private constant [12 x i8] c"qwen2.5:0.5b"
@c.role  = private constant [4 x i8] c"user"
@c.hi    = private constant [2 x i8] c"hi"
@c.m1    = private constant [1 x i8] c"m"
@c.host  = private constant [9 x i8] c"127.0.0.1"
@c.path  = private constant [20 x i8] c"/v1/chat/completions"
@c.prompt = private constant [33 x i8] c"Reply with exactly the word: pong"

; escape: a " b <LF> c  ->  a \ " b \ n c
@esc.in  = private constant [5 x i8] c"a\22b\0Ac"
@esc.exp = private constant [7 x i8] c"a\5C\22b\5Cnc"

; canned non-streaming response (NUL-terminated so strlen gives the JSON length)
@resp = private constant [166 x i8] c"{\22choices\22:[{\22index\22:0,\22message\22:{\22role\22:\22assistant\22,\22content\22:\22Hello!\22},\22finish_reason\22:\22stop\22}],\22usage\22:{\22prompt_tokens\22:1,\22completion_tokens\22:1,\22total_tokens\22:2}}\00"
@exp.hello = private constant [6 x i8] c"Hello!"
@exp.stop  = private constant [4 x i8] c"stop"

; API error object
@errobj = private constant [34 x i8] c"{\22error\22:{\22message\22:\22bad model\22}}\00"

; SSE delta chunk payload JSON
@sse = private constant [41 x i8] c"{\22choices\22:[{\22delta\22:{\22content\22:\22Hi\22}}]}\00"
@exp.hi2 = private constant [2 x i8] c"Hi"

@m.esc   = private unnamed_addr constant [16 x i8] c"json_escape a\22b\00"
@m.build = private unnamed_addr constant [23 x i8] c"build_request nonempty\00"
@m.bcur  = private unnamed_addr constant [20 x i8] c"build starts with {\00"
@m.prc   = private unnamed_addr constant [18 x i8] c"parse_response OK\00"
@m.pcon  = private unnamed_addr constant [20 x i8] c"parse content Hello\00"
@m.pfin  = private unnamed_addr constant [18 x i8] c"parse finish stop\00"
@m.perr  = private unnamed_addr constant [25 x i8] c"error obj -> INV_STATE11\00"
@m.delta = private unnamed_addr constant [18 x i8] c"sse delta == \22Hi\22\00"
@m.live  = private unnamed_addr constant [22 x i8] c"LIVE ollama complete\0A\00"
@f.skip  = private unnamed_addr constant [40 x i8] c"llm: ollama unreachable, skipping live\0A\00"
@f.got   = private unnamed_addr constant [26 x i8] c"llm live content: [%.*s]\0A\00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; ---- json_escape ----
  %en = call i64 @universe_llm_json_escape(ptr @g.esc, i64 64, ptr @esc.in, i64 5)
  %en.ok = icmp eq i64 %en, 7
  %ec = call i32 @memcmp(ptr @g.esc, ptr @esc.exp, i64 7)
  %ec.ok = icmp eq i32 %ec, 0
  %esc.ok = and i1 %en.ok, %ec.ok
  call void @ut_check(i1 %esc.ok, ptr @m.esc)

  ; ---- build_request (one user message "hi") ----
  %msgs = alloca [32 x i8], align 8
  store ptr @c.role, ptr %msgs, align 8
  %m.rl = getelementptr inbounds nuw i8, ptr %msgs, i64 8
  store i64 4, ptr %m.rl, align 8
  %m.cp = getelementptr inbounds nuw i8, ptr %msgs, i64 16
  store ptr @c.hi, ptr %m.cp, align 8
  %m.cl = getelementptr inbounds nuw i8, ptr %msgs, i64 24
  store i64 2, ptr %m.cl, align 8
  %bl = call i64 @universe_llm_build_request(ptr @g.body, i64 8192, ptr @c.m1, i64 1, ptr %msgs, i64 1, i32 0)
  %bl.ok = icmp sgt i64 %bl, 0
  call void @ut_check(i1 %bl.ok, ptr @m.build)
  %b0 = load i8, ptr @g.body, align 1
  %b0.ok = icmp eq i8 %b0, 123           ; '{'
  call void @ut_check(i1 %b0.ok, ptr @m.bcur)

  ; ---- parse_response (canned non-stream) ----
  %rlen = call i64 @strlen(ptr @resp)
  %prc = call i32 @universe_llm_parse_response(ptr @resp, i64 %rlen, ptr @g.out, i64 4096, ptr @g.outlen, ptr @g.fin, i64 64, ptr @g.finlen, ptr @g.usage)
  %prc.ok = icmp eq i32 %prc, 0
  call void @ut_check(i1 %prc.ok, ptr @m.prc)
  %olen = load i64, ptr @g.outlen, align 8
  %olen.ok = icmp eq i64 %olen, 6
  %occ = call i32 @memcmp(ptr @g.out, ptr @exp.hello, i64 6)
  %occ.ok = icmp eq i32 %occ, 0
  %con.ok = and i1 %olen.ok, %occ.ok
  call void @ut_check(i1 %con.ok, ptr @m.pcon)
  %flen = load i64, ptr @g.finlen, align 8
  %flen.ok = icmp eq i64 %flen, 4
  %fcc = call i32 @memcmp(ptr @g.fin, ptr @exp.stop, i64 4)
  %fcc.ok = icmp eq i32 %fcc, 0
  %fin.ok = and i1 %flen.ok, %fcc.ok
  call void @ut_check(i1 %fin.ok, ptr @m.pfin)

  ; ---- API error object -> INVALID_STATE (11) ----
  %elen = call i64 @strlen(ptr @errobj)
  %erc = call i32 @universe_llm_parse_response(ptr @errobj, i64 %elen, ptr @g.out, i64 4096, ptr @g.outlen, ptr @g.fin, i64 64, ptr @g.finlen, ptr @g.usage)
  %erc.ok = icmp eq i32 %erc, 11
  call void @ut_check(i1 %erc.ok, ptr @m.perr)

  ; ---- SSE delta extraction ----
  %slen = call i64 @strlen(ptr @sse)
  %drc = call i32 @universe_llm_extract_delta(ptr @sse, i64 %slen, ptr @g.out, i64 4096, ptr @g.outlen)
  %drc.ok = icmp eq i32 %drc, 0
  %dlen = load i64, ptr @g.outlen, align 8
  %dlen.ok = icmp eq i64 %dlen, 2
  %dcc = call i32 @memcmp(ptr @g.out, ptr @exp.hi2, i64 2)
  %dcc.ok = icmp eq i32 %dcc, 0
  %d1 = and i1 %drc.ok, %dlen.ok
  %d2 = and i1 %d1, %dcc.ok
  call void @ut_check(i1 %d2, ptr @m.delta)

  ; ---- LIVE end-to-end against Ollama (skip if unreachable) ----
  ; 127.0.0.1 = 0x7F000001 = 2130706433, port 11434
  %conn = call ptr @universe_http_connect(i32 2130706433, i32 11434, i64 65536)
  %conn.null = icmp eq ptr %conn, null
  br i1 %conn.null, label %live.skip, label %live.go

live.go:
  %cfg = alloca [72 x i8], align 8
  call void @universe_llm_config_init(ptr %cfg, i32 2130706433, i32 11434, ptr @c.host, i64 9, ptr @c.path, i64 20, ptr @c.model, i64 12, ptr null, i64 0)
  %lmsgs = alloca [32 x i8], align 8
  store ptr @c.role, ptr %lmsgs, align 8
  %lm.rl = getelementptr inbounds nuw i8, ptr %lmsgs, i64 8
  store i64 4, ptr %lm.rl, align 8
  %lm.cp = getelementptr inbounds nuw i8, ptr %lmsgs, i64 16
  store ptr @c.prompt, ptr %lm.cp, align 8
  %lm.cl = getelementptr inbounds nuw i8, ptr %lmsgs, i64 24
  store i64 33, ptr %lm.cl, align 8
  %lrc = call i32 @universe_llm_complete(ptr %conn, ptr %cfg, ptr %lmsgs, i64 1, ptr @g.body, i64 8192, ptr @g.out, i64 4096, ptr @g.outlen, ptr @g.fin, i64 64, ptr @g.finlen, ptr @g.usage)
  %lrc.ok = icmp eq i32 %lrc, 0
  %llen = load i64, ptr @g.outlen, align 8
  %llen.ok = icmp sgt i64 %llen, 0
  %live.ok = and i1 %lrc.ok, %llen.ok
  call void @ut_check(i1 %live.ok, ptr @m.live)
  ; print the model's reply
  %llen32 = trunc i64 %llen to i32
  %pr = call i32 (ptr, ...) @printf(ptr @f.got, i32 %llen32, ptr @g.out)
  call void @universe_http_conn_destroy(ptr %conn)
  br label %fin

live.skip:
  %ps = call i32 (ptr, ...) @printf(ptr @f.skip)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}

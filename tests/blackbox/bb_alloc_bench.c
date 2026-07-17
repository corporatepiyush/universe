/* bb_alloc_bench.c — cross-allocator benchmark (black-box, public C ABI).
 *
 * Times every SDK allocator against libc malloc under several load profiles that
 * mirror real module patterns, to decide (a) the worthy DEFAULT and (b) which
 * allocator suits which workload. Built against build/libuniverse.a and run on
 * BOTH glibc (Debian) and musl (Alpine). Output: CSV "profile,allocator,ns_op"
 * (ns_op = median of reps; -1 = N/A for this allocator/profile).
 *
 * libc malloc/calloc/free here are the HARNESS's own bookkeeping + the baseline
 * under test — the "no native malloc outside src/allocator" rule governs SDK
 * modules, not a test driver.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* ---- SDK allocator C ABI ---- */
void* universe_alloc_arena_create(int64_t);
void* universe_alloc_arena_alloc(void*, int64_t);
void  universe_alloc_arena_reset(void*);
void  universe_alloc_arena_destroy(void*);
void* universe_alloc_pool_create(int64_t, int64_t);
void* universe_alloc_pool_alloc(void*);
void  universe_alloc_pool_free(void*, void*);
void  universe_alloc_pool_destroy(void*);
void* universe_alloc_slab_create(int64_t, int64_t);
void* universe_alloc_slab_alloc(void*);
void  universe_alloc_slab_free(void*, void*);
void  universe_alloc_slab_destroy(void*);
void* universe_alloc_buddy_create(int64_t, int64_t);
void* universe_alloc_buddy_alloc(void*, int64_t);
void  universe_alloc_buddy_free(void*, void*, int64_t);
void  universe_alloc_buddy_destroy(void*);
void* universe_alloc_tlsf_create(void*, int64_t);
void* universe_alloc_tlsf_alloc(void*, int64_t);
void  universe_alloc_tlsf_free(void*, void*);
void  universe_alloc_tlsf_destroy(void*);
void* universe_alloc_hybrid_create(void*);
void* universe_alloc_hybrid_alloc(void*, int64_t);
void  universe_alloc_hybrid_free(void*, void*);
void  universe_alloc_hybrid_destroy(void*);

/* ---- timing + rng ---- */
static uint64_t sink = 0;
static inline uint64_t now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec; }
static inline uint64_t sm(uint64_t*s){ uint64_t z=(*s+=0x9E3779B97F4A7C15ull); z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull; return z^(z>>31); }
static size_t pow2ceil(size_t x){ size_t p=32; while(p<x) p<<=1; return p; }

/* ---- adapter: one uniform interface over every allocator ---- */
enum { FIXED=1, NOFREE=2 };
typedef struct {
  const char* name;
  void*(*mk)(size_t item, size_t n);          /* item=max size, n=working set */
  void*(*al)(void* h, size_t sz);
  void (*fr)(void* h, void* p, size_t sz);
  void (*rs)(void* h);                         /* NULL if none */
  void (*ds)(void* h);
  unsigned flags;
} A;

/* malloc baseline */
static void* m_mk(size_t i,size_t n){(void)i;(void)n;return (void*)1;}
static void* m_al(void* h,size_t s){(void)h;return malloc(s);}
static void  m_fr(void* h,void* p,size_t s){(void)h;(void)s;free(p);}
static void  m_ds(void* h){(void)h;}
/* tlsf (growable; base=null owns+grows) */
static void* t_mk(size_t i,size_t n){ int64_t init=(int64_t)(i*n*2+65536); if(init<65536)init=65536; return universe_alloc_tlsf_create(NULL,init);}
static void* t_al(void* h,size_t s){return universe_alloc_tlsf_alloc(h,(int64_t)s);}
static void  t_fr(void* h,void* p,size_t s){(void)s;universe_alloc_tlsf_free(h,p);}
static void  t_ds(void* h){universe_alloc_tlsf_destroy(h);}
/* hybrid */
static void* h_mk(size_t i,size_t n){(void)i;(void)n;return universe_alloc_hybrid_create(NULL);}
static void* h_al(void* h,size_t s){return universe_alloc_hybrid_alloc(h,(int64_t)s);}
static void  h_fr(void* h,void* p,size_t s){(void)s;universe_alloc_hybrid_free(h,p);}
static void  h_ds(void* h){universe_alloc_hybrid_destroy(h);}
/* buddy (fixed power-of-two region; sized generously; free needs size) */
static void* b_mk(size_t i,size_t n){ size_t reg=pow2ceil(i*(n+16)*4); if(reg<65536)reg=65536; return universe_alloc_buddy_create((int64_t)reg,16);}
static void* b_al(void* h,size_t s){return universe_alloc_buddy_alloc(h,(int64_t)s);}
static void  b_fr(void* h,void* p,size_t s){universe_alloc_buddy_free(h,p,(int64_t)s);}
static void  b_ds(void* h){universe_alloc_buddy_destroy(h);}
/* pool (fixed size) */
static void* p_mk(size_t i,size_t n){return universe_alloc_pool_create((int64_t)i,(int64_t)(n+16));}
static void* p_al(void* h,size_t s){(void)s;return universe_alloc_pool_alloc(h);}
static void  p_fr(void* h,void* p,size_t s){(void)s;universe_alloc_pool_free(h,p);}
static void  p_ds(void* h){universe_alloc_pool_destroy(h);}
/* slab (fixed size, growable) */
static void* s_mk(size_t i,size_t n){(void)n;return universe_alloc_slab_create((int64_t)i,256);}
static void* s_al(void* h,size_t s){(void)s;return universe_alloc_slab_alloc(h);}
static void  s_fr(void* h,void* p,size_t s){(void)s;universe_alloc_slab_free(h,p);}
static void  s_ds(void* h){universe_alloc_slab_destroy(h);}
/* arena (bump; no per-object free; reset) */
static void* a_mk(size_t i,size_t n){ int64_t c=(int64_t)(i*n+4096); return universe_alloc_arena_create(c);}
static void* a_al(void* h,size_t s){return universe_alloc_arena_alloc(h,(int64_t)s);}
static void  a_rs(void* h){universe_alloc_arena_reset(h);}
static void  a_ds(void* h){universe_alloc_arena_destroy(h);}

static A ALLOCS[] = {
  {"malloc", m_mk,m_al,m_fr,NULL,m_ds, 0},
  {"tlsf",   t_mk,t_al,t_fr,NULL,t_ds, 0},
  {"hybrid", h_mk,h_al,h_fr,NULL,h_ds, 0},
  {"buddy",  b_mk,b_al,b_fr,NULL,b_ds, 0},
  {"pool",   p_mk,p_al,p_fr,NULL,p_ds, FIXED},
  {"slab",   s_mk,s_al,s_fr,NULL,s_ds, FIXED},
  {"arena",  a_mk,a_al,NULL,a_rs,a_ds, NOFREE},
};
#define NA (sizeof(ALLOCS)/sizeof(ALLOCS[0]))

static int dcmp(const void*x,const void*y){ double a=*(const double*)x,b=*(const double*)y; return a<b?-1:a>b?1:0; }
static double median(double* v,int n){ qsort(v,n,sizeof(double),dcmp); return v[n/2]; }

/* steady-state churn: prefill W live, then nops (free one + alloc replacement) */
static double churn(A* a,size_t lo,size_t hi,size_t W,size_t nops){
  if(a->flags&NOFREE) return -1;
  if((a->flags&FIXED)&&lo!=hi) return -1;
  void* h=a->mk(hi,W); if(!h) return -2;
  void** live=calloc(W,sizeof(void*)); size_t* sz=calloc(W,sizeof(size_t));
  uint64_t rng=0x2545F4914F6CDD1Dull;
  for(size_t i=0;i<W;i++){ size_t s=lo==hi?lo:lo+sm(&rng)%(hi-lo+1); void* p=a->al(h,s); if(!p){a->ds(h);free(live);free(sz);return -2;} live[i]=p; sz[i]=s; ((char*)p)[0]=(char)i; }
  uint64_t t0=now_ns();
  for(size_t k=0;k<nops;k++){ size_t idx=sm(&rng)%W; a->fr(h,live[idx],sz[idx]); size_t s=lo==hi?lo:lo+sm(&rng)%(hi-lo+1); void* p=a->al(h,s); if(!p){a->ds(h);free(live);free(sz);return -2;} live[idx]=p; sz[idx]=s; ((char*)p)[0]=(char)k; sink^=(uint64_t)p; }
  uint64_t t1=now_ns();
  a->ds(h); free(live); free(sz);
  return (double)(t1-t0)/(double)nops;   /* ns per free+alloc */
}
/* bump+reset: R rounds of (alloc nper, then reset/destroy) — build-scratch pattern */
static double bumpreset(A* a,size_t item,size_t nper,size_t R){
  if(a->flags&FIXED) return -1;
  uint64_t total=0, ops=0;
  if(a->rs){ /* arena: reset */
    void* h=a->mk(item,nper); if(!h)return -2;
    for(size_t r=0;r<R;r++){ uint64_t t0=now_ns(); for(size_t i=0;i<nper;i++){void*p=a->al(h,item); if(p){((char*)p)[0]=1;sink^=(uint64_t)p;}} total+=now_ns()-t0; ops+=nper; a->rs(h);} a->ds(h);
  } else if(a->fr){ /* tlsf/hybrid/buddy/malloc: alloc then bulk-destroy (or free-loop for malloc) */
    for(size_t r=0;r<R;r++){ void* h=a->mk(item,nper); if(!h)return -2; void** live=calloc(nper,sizeof(void*)); uint64_t t0=now_ns(); for(size_t i=0;i<nper;i++){void*p=a->al(h,item); live[i]=p; if(p){((char*)p)[0]=1;sink^=(uint64_t)p;}} total+=now_ns()-t0; ops+=nper;
      if(strcmp(a->name,"malloc")==0){ for(size_t i=0;i<nper;i++) a->fr(h,live[i],item); } else a->ds(h); free(live);}
  } else return -1;
  return (double)total/(double)ops;   /* ns per alloc */
}

static double best_of(double(*f)(A*,size_t,size_t,size_t,size_t),A* a,size_t lo,size_t hi,size_t W,size_t n,int reps){
  double v[16]; int ok=0; for(int r=0;r<reps;r++){ double d=f(a,lo,hi,W,n); if(d<0)return d; v[ok++]=d;} return median(v,ok);
}

int main(void){
  int reps=5;
  printf("profile,allocator,ns_op\n");
  struct { const char* name; int kind; size_t lo,hi,W,n; } P[] = {
    {"node_churn_32B",   0, 32,32,     4096, 2000000},
    {"small_churn_64B",  0, 64,64,     4096, 2000000},
    {"medium_churn_512B",0, 512,512,   2048, 1000000},
    {"large_churn_8KB",  0, 8192,8192, 256,  200000},
    {"mixed_16_2048B",   0, 16,2048,   4096, 1000000},
    {"bumpreset_64B",    1, 64,64,     0,    0},
    {"build_alloc_64B",  2, 64,64,     0,    0},
  };
  for(size_t pi=0; pi<sizeof(P)/sizeof(P[0]); pi++){
    for(size_t ai=0; ai<NA; ai++){
      A* a=&ALLOCS[ai]; double d;
      if(P[pi].kind==0)      d=best_of(churn,a,P[pi].lo,P[pi].hi,P[pi].W,P[pi].n,reps);
      else if(P[pi].kind==1){ double v[16];int ok=0;for(int r=0;r<reps;r++){double x=bumpreset(a,64,100000,10);if(x<0){d=x;goto emit;}v[ok++]=x;} d=median(v,ok);}
      else { double v[16];int ok=0;for(int r=0;r<reps;r++){double x=bumpreset(a,64,500000,1);if(x<0){d=x;goto emit;}v[ok++]=x;} d=median(v,ok);}
      emit:
      printf("%s,%s,%.2f\n", P[pi].name, a->name, d);
    }
  }
  fprintf(stderr,"sink=%llu\n",(unsigned long long)sink);
  return 0;
}

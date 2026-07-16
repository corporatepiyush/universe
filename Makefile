MAKEFLAGS += -j$(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# LLVM toolchain dir and host triple are overridable so the same Makefile
# builds natively on macOS (default) and inside the Linux test container
# (which sets LLVM=/usr/lib/llvm-22/bin and HOST_TRIPLE=<arch>-linux-gnu).
UNAME_S := $(shell uname -s)
LLVM    ?= /opt/homebrew/opt/llvm/bin
CLANG   := $(LLVM)/clang
AR      := $(LLVM)/llvm-ar

ifeq ($(UNAME_S),Darwin)
  HOST_TRIPLE ?= arm64-apple-macosx26.0
  SHLIB       := build/libuniverse.dylib
  SHLIB_FLAGS := -dynamiclib -install_name @rpath/libuniverse.dylib
else
  HOST_ARCH   := $(shell uname -m)
  HOST_TRIPLE ?= $(HOST_ARCH)-unknown-linux-gnu
  SHLIB       := build/libuniverse.so
  SHLIB_FLAGS := -shared -Wl,-soname,libuniverse.so
endif

# Host build (test-runnable). IR files carry no triple; the driver supplies it.
IRFLAGS  := -O3 -target $(HOST_TRIPLE) -Wno-override-module
LDFLAGS  := -lpthread

# Every module must also codegen for these (objects only, not linked/run).
CROSS_TRIPLES := x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu \
                 x86_64-unknown-freebsd15.0 aarch64-unknown-freebsd15.0

# OS-divergent net primitives ship as two symbol-identical twins; the HOST
# build links exactly one (the wrong one would be a duplicate-symbol / wrong-ABI
# link). crosscheck still codegen-checks BOTH (ALLIRSRCS) on every triple.
ifeq ($(UNAME_S),Linux)
  OSCONST_EXCLUDE := src/net/osconst_bsd.ll
else
  OSCONST_EXCLUDE := src/net/osconst_linux.ll
endif
ALLIRSRCS := $(shell find src -name '*.ll' | sort)
IRSRCS  := $(filter-out $(OSCONST_EXCLUDE), $(ALLIRSRCS))
IROBJS  := $(patsubst src/%.ll, build/obj/%.o, $(IRSRCS))
LIB     := build/libuniverse.a
UTOBJ   := build/obj/ut.o
DOMAINS := $(notdir $(shell find src -mindepth 1 -maxdepth 1 -type d | sort))

# Cross-domain test dependencies. A test in domain X links its own domain
# archive; if X's modules also reference symbols from other domains, list those
# domains here so their archives are linked (AFTER X's, so X's references
# resolve against them). Leaf/self-contained domains need no entry.
DEPS_http     := io net simd ioring
DEPS_llm      := http io net simd parse
DEPS_crypto             := bignum
DEPS_docparse           := compress parse
DEPS_ml                 := simd
DEPS_structures_succinct := encoding
DEPS_observ             := concurrent io
DEPS_strings            := simd

TESTSRC := $(shell find tests -name 'test_*.ll' | sort)
TESTBIN := $(patsubst tests/%.ll, build/bin/%, $(TESTSRC))

# Docker: run the suite natively on Linux (kernel >= 6.15 enforced in-container).
DOCKER_IMAGE := universe-linux-test
DOCKER_PLATFORM ?=

.PHONY: all lib dylib test crosscheck clean list docker-build docker-test \
        $(addprefix test-,$(DOMAINS)) $(addprefix crosscheck-,$(DOMAINS))

all: lib dylib $(TESTBIN)

docker-build:
	docker build $(if $(DOCKER_PLATFORM),--platform $(DOCKER_PLATFORM),) \
	  -t $(DOCKER_IMAGE) -f docker/Dockerfile .

# Bind-mounts the source read-only; the container builds into its own tmpfs
# so the macOS build/ is never touched. `make docker-test` = real Linux run.
docker-test: docker-build
	docker run --rm $(if $(DOCKER_PLATFORM),--platform $(DOCKER_PLATFORM),) \
	  -v "$(CURDIR):/universe:ro" $(DOCKER_IMAGE)

lib: $(LIB)

# `dylib` name kept for muscle memory; produces .dylib on macOS, .so on Linux.
dylib: $(SHLIB)

$(LIB): $(IROBJS)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

$(SHLIB): $(IROBJS)
	@mkdir -p $(dir $@)
	$(CLANG) -target $(HOST_TRIPLE) $(SHLIB_FLAGS) $^ $(LDFLAGS) -o $@

$(UTOBJ): tests/support/ut.ll
	@mkdir -p $(dir $@)
	$(CLANG) $(IRFLAGS) -c $< -o $@

build/obj/%.o: src/%.ll
	@mkdir -p $(dir $@)
	$(CLANG) $(IRFLAGS) -c $< -o $@

# Per-domain archive: tests link ONLY their own domain's modules, so domains
# build/test independently (parallel agents can't break each other).
build/libuniverse_%.a: FORCE
	@mkdir -p build
	@$(MAKE) --no-print-directory $(patsubst src/%.ll, build/obj/%.o, $(filter src/$*/%,$(IRSRCS)))
	@$(AR) rcs $@ $(patsubst src/%.ll, build/obj/%.o, $(filter src/$*/%,$(IRSRCS)))

FORCE:

# Per-test fixture objects: a test may embed generated data in sibling
# tests/<domain>/fixture_*.ll files (not test_*.ll, so not their own binaries).
# Link every fixture in the test's directory into that domain's test binaries.
.SECONDEXPANSION:
build/bin/%: tests/%.ll $(UTOBJ) build/libuniverse_$$(firstword $$(subst /, ,$$*)).a \
             $$(foreach d,$$(DEPS_$$(firstword $$(subst /, ,$$*))),build/libuniverse_$$d.a) \
             $$(wildcard tests/$$(dir $$*)fixture_*.ll)
	@mkdir -p $(dir $@)
	$(CLANG) $(IRFLAGS) $< $(wildcard tests/$(dir $*)fixture_*.ll) $(UTOBJ) \
	  build/libuniverse_$(firstword $(subst /, ,$*)).a \
	  $(foreach d,$(DEPS_$(firstword $(subst /, ,$*))),build/libuniverse_$d.a) \
	  $(LDFLAGS) -o $@

# Run all tests; report every failure.
test: $(TESTBIN)
	@rc=0; for t in $(TESTBIN); do \
	  printf '== %s == ' $$t; \
	  if ./$$t; then :; else rc=1; fi; \
	done; exit $$rc

# Per-domain: make test-sort, make test-structures, ...
$(addprefix test-,$(DOMAINS)): test-%:
	@$(MAKE) --no-print-directory $(filter build/bin/$*/%,$(TESTBIN))
	@rc=0; for t in $(filter build/bin/$*/%,$(TESTBIN)); do \
	  printf '== %s == ' $$t; \
	  if ./$$t; then :; else rc=1; fi; \
	done; exit $$rc

# Cross-target codegen validation.
crosscheck:
	@$(MAKE) --no-print-directory crosscheck-run SCOPE='$(ALLIRSRCS)'

$(addprefix crosscheck-,$(DOMAINS)): crosscheck-%:
	@$(MAKE) --no-print-directory crosscheck-run SCOPE='$(filter src/$*/%,$(ALLIRSRCS))'

.PHONY: crosscheck-run
crosscheck-run:
	@rc=0; for triple in $(CROSS_TRIPLES); do \
	  for f in $(SCOPE); do \
	    out=build/cross/$$triple/$${f#src/}; out=$${out%.ll}.o; \
	    mkdir -p $$(dirname $$out); \
	    if ! $(CLANG) -O3 -target $$triple -Wno-override-module -c $$f -o $$out; then \
	      echo "CROSSFAIL $$triple $$f"; rc=1; \
	    fi; \
	  done; \
	done; \
	if [ $$rc -eq 0 ]; then echo "crosscheck OK: $(words $(SCOPE)) modules x $(words $(CROSS_TRIPLES)) triples"; fi; \
	exit $$rc

list:
	@echo "modules:"; for f in $(IRSRCS); do echo "  $$f"; done
	@echo "tests:";   for t in $(TESTSRC); do echo "  $$t"; done

clean:
	rm -rf build

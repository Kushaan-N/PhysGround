# PhysGround. Targets follow spec 10's ordering, so each is a kill-switch for
# the next and costs less than what follows.
#
#   make pilot      # 50 scenes + gates. Do this first; it is minutes.
#   make corpus     # the full 3000 + 1000
#   make features
#   make probes
#   make figures
#
# MUJOCO_GL is exported per-target rather than set in the shell, because MuJoCo
# reads it once at import and a later assignment silently does nothing (spec 14).

SEED       ?= 0
N_BASE     ?= 3000
N_OCCLUDED ?= 1000
SEEDS      ?= 0,1,2,3,4
GEN_SHARDS ?= 8
EXT_SHARDS ?= 8

# osmesa on Linux; macOS uses CGL and rejects osmesa outright.
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  GL_ENV :=
else
  GL_ENV := MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
endif
PY := $(GL_ENV) PYTHONPATH=$(CURDIR):$(CURDIR)/scripts python

.PHONY: help preflight pilot corpus gates features probes figures test clean-outputs

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

preflight:  ## Verify the render backend is what we think it is (spec 17.3)
	$(PY) scripts/preflight_render.py --json

pilot: preflight  ## 50-scene pilot + gates. Run before spending the corpus budget.
	$(PY) scripts/generate_corpus.py --condition base     --n 50 --seed $(SEED) --skip-preflight
	$(PY) scripts/generate_corpus.py --condition occluded --n 20 --seed $(SEED) --skip-preflight
	$(MAKE) gates
	@echo
	@echo "Gate G2 has no automated verdict. Open outputs/gates/contact_sheet.png"
	@echo "and look at it before going further (spec 9.2, 17.6)."

corpus: preflight  ## Generate the full corpus
	$(PY) scripts/generate_corpus.py --condition base     --n $(N_BASE)     --seed $(SEED) --skip-preflight
	$(PY) scripts/generate_corpus.py --condition occluded --n $(N_OCCLUDED) --seed $(SEED) --skip-preflight

# --corpus-scenes is deliberately not $(N_BASE): G1 tests the factor sampler,
# not the generated corpus, and evaluating it at pilot size measures sampling
# noise (see DEVIATIONS.md #1).
gates:  ## G1, G2, G3 (add G4 once features exist)
	$(PY) scripts/run_gates.py --corpus-scenes 3000 --seed $(SEED) --gates 1,2,3

features:  ## Extract frozen features for every encoder and condition
	@for enc in dinov2_b random_b raw_pixel; do \
	  for cond in base occluded; do \
	    for s in $$(seq 0 $$(($(EXT_SHARDS)-1))); do \
	      $(PY) scripts/extract_features.py --encoder $$enc --condition $$cond \
	        --shard $$s --n-shards $(EXT_SHARDS) --no-patches || exit 1; \
	    done; \
	  done; \
	done

probes:  ## A and B are kill-switches and exit non-zero on failure
	$(PY) scripts/run_probes.py --exp A,B --seed $(SEED)
	$(PY) scripts/run_probes.py --exp C,D,E --seed $(SEEDS)

figures:  ## Regenerate every table and figure from the raw archives
	$(PY) scripts/make_figures.py --seeds $(SEEDS)

test:  ## 80 tests, ~8s, no checkpoint downloads
	$(GL_ENV) PYTHONPATH=$(CURDIR) python -m pytest tests/ -q

clean-outputs:  ## Delete everything regenerable. Does not touch code.
	rm -rf outputs/corpus outputs/features outputs/results outputs/figures outputs/gates outputs/tables

# Submit ErwinParTv2 / ParT work to NRP Nautilus.
#
#   make smoke   MODEL=ErwinParTv2              # ~15 min, 0.1% of data
#   make tests                                  # S0-S9 on cluster hardware
#   make train   MODEL=ErwinParTv2 TAG=v2-10pct # the real run, 10% scope
#   make train   MODEL=ParT        TAG=part-10pct
#   make predict MODEL=ErwinParTv2 RUN_ID=<id>  # id from docs/RUNS.md
#   make logs RUN_ID=<id> / make status / make dry-run TAG=x
#
# Every target renders jobs/job.yml with sed and pipes it to kubectl. Nothing
# here deletes a job; `make clean-job RUN_ID=...` exists but must be typed.
SHELL := /bin/bash
NS     ?= cms-ml
REPO_ROOT := $(shell cd .. && pwd)

# ---- what the job runs -----------------------------------------------------
BRANCH ?= ErwinParTCluster
# NOTE: :latest is a moving tag. Resolve it once with `make image-digest` and
# set IMAGE to the sha256 form for a reproducible run.
IMAGE  ?= gitlab-registry.nrp-nautilus.io/jmduarte/hbb_interaction_network:latest
MODEL   ?= ErwinParTv2
FEATURE ?= full

# ---- SCOPE: 10% of JetClass by default -------------------------------------
# --data-fraction caps what each epoch reads from every file (class balance
# preserved). SAMPLES_PER_EPOCH caps compute so one epoch is roughly one pass
# over that 10%, not ten passes over it. Raise only deliberately.
DATA_FRACTION         ?= 0.1
SAMPLES_PER_EPOCH     ?= 1024000
SAMPLES_PER_EPOCH_VAL ?= 128000
EPOCHS                ?= 50
DEADLINE              ?= 172800
EXTRA_ARGS            ?=
RUN_DIR               ?=
TEST_DATA             ?= /output/testdata/JetClass_example_100k.root
SKIP_PREFLIGHT        ?= 0

SHA  := $(shell git -C $(REPO_ROOT) rev-parse --short HEAD)
DATE := $(shell date +%Y%m%d)
TAG  ?= dev
RUN_ID ?= $(shell echo "$(DATE)-$(SHA)-$(TAG)" | tr '[:upper:]' '[:lower:]')

ROLE ?= train

RENDER = sed \
  -e 's|@@ROLE@@|$(ROLE)|g' \
  -e 's|@@RUN_ID@@|$(RUN_ID)|g' \
  -e 's|@@MODEL@@|$(MODEL)|g' \
  -e 's|@@FEATURE@@|$(FEATURE)|g' \
  -e 's|@@BRANCH@@|$(BRANCH)|g' \
  -e 's|@@IMAGE@@|$(IMAGE)|g' \
  -e 's|@@DATA_FRACTION@@|$(DATA_FRACTION)|g' \
  -e 's|@@SAMPLES_PER_EPOCH@@|$(SAMPLES_PER_EPOCH)|g' \
  -e 's|@@SAMPLES_PER_EPOCH_VAL@@|$(SAMPLES_PER_EPOCH_VAL)|g' \
  -e 's|@@EPOCHS@@|$(EPOCHS)|g' \
  -e 's|@@DEADLINE@@|$(DEADLINE)|g' \
  -e 's|@@EXTRA_ARGS@@|$(EXTRA_ARGS)|g' \
  -e 's|@@RUN_DIR@@|$(RUN_DIR)|g' \
  -e 's|@@TEST_DATA@@|$(TEST_DATA)|g' \
  -e 's|@@SKIP_PREFLIGHT@@|$(SKIP_PREFLIGHT)|g' \
  jobs/job.yml

.PHONY: check-pushed dry-run submit train predict smoke tests logs status describe probe image-digest clean-job help

## The single most important guard: a job runs the branch as it exists on
## GitHub, so an uncommitted or unpushed edit is invisible to the cluster.
check-pushed:
	@cd $(REPO_ROOT) && \
	if [ -n "$$(git status --porcelain)" ]; then \
	  echo "REFUSING: working tree is dirty. The cluster clones $(BRANCH) from GitHub,"; \
	  echo "          so uncommitted changes would NOT be in the run."; \
	  git status --short; exit 1; fi; \
	if [ "$$(git rev-parse --abbrev-ref HEAD)" != "$(BRANCH)" ]; then \
	  echo "REFUSING: on branch $$(git rev-parse --abbrev-ref HEAD) but the job clones $(BRANCH)."; exit 1; fi; \
	git fetch -q origin $(BRANCH) 2>/dev/null || true; \
	if [ "$$(git rev-parse HEAD)" != "$$(git rev-parse origin/$(BRANCH) 2>/dev/null)" ]; then \
	  echo "REFUSING: HEAD != origin/$(BRANCH). Push first, or the cluster runs older code."; exit 1; fi; \
	echo "OK: $(BRANCH) @ $(SHA) is pushed and clean."

dry-run:
	@$(RENDER) | kubectl -n $(NS) apply --dry-run=client -f - && echo "OK: manifest valid (run-id $(RUN_ID))"

submit: check-pushed
	@echo ">>> submitting role=$(ROLE) model=$(MODEL) run-id=$(RUN_ID) scope=$(DATA_FRACTION)"
	@$(RENDER) | kubectl -n $(NS) apply -f -
	@echo ">>> kubectl -n $(NS) logs -f job/vleept-$(RUN_ID)"

train:   ; @$(MAKE) submit ROLE=train
predict: ; @if [ -z "$(RUN_DIR)$(RUN_ID_IN)" ]; then echo "usage: make predict MODEL=... RUN_ID=<trained run id>"; exit 1; fi; \
	    $(MAKE) submit ROLE=predict RUN_ID=$(RUN_ID)-pred RUN_DIR=$${RUN_DIR:-/output/runs/$(RUN_ID)} DEADLINE=43200
smoke:   ; @$(MAKE) submit ROLE=smoke TAG=smoke DATA_FRACTION=0.001 EPOCHS=1 \
	    SAMPLES_PER_EPOCH=51200 SAMPLES_PER_EPOCH_VAL=12800 DEADLINE=86400
tests:   ; @$(MAKE) submit ROLE=tests TAG=tests SKIP_PREFLIGHT=1 DEADLINE=7200

logs:    ; kubectl -n $(NS) logs -f job/vleept-$(RUN_ID)
status:  ; @kubectl -n $(NS) get jobs -l jobgroup=erwinpart-cluster
describe:; @kubectl -n $(NS) describe job vleept-$(RUN_ID) | tail -30
probe:   ; kubectl -n $(NS) apply -f pods/probe.yml && echo ">>> kubectl -n $(NS) logs -f vleept-probe"

image-digest:
	@echo "Run inside any pod using the image:"; \
	 echo "  kubectl -n $(NS) get pod <pod> -o jsonpath='{.status.containerStatuses[0].imageID}'"

clean-job:
	@echo "About to DELETE job vleept-$(RUN_ID) (the PVC output is untouched)."
	kubectl -n $(NS) delete job vleept-$(RUN_ID)

help: ; @sed -n 's/^#   //p' Makefile

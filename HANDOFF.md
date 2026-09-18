# Handoff

Written for whoever picks this up next — likely another LLM agent, on a fresh
remote VM. Read this first, then `spec.md`. Everything below is either measured
or points at the file that measures it; where a number is an estimate rather
than an observation, it says so.

**Repo:** `git@github.com:Kushaan-N/PhysGround.git`, branch `main`.
**State at handoff:** working tree clean, 86 tests passing, everything below
reproducible from `main`. `git log --oneline` reads as a narrative — each commit
message says what changed and, where a measurement forced it, what the
measurement was.

> **Update, 2026-09-15 (second handoff).** The full-corpus run is DONE: 3,000 +
> 1,000 scenes, all encoders, Experiments A–F and S at five probe seeds,
> figures and hypothesis tests. Read `RESULTS_full.md` for the numbers and
> `DEVIATIONS.md` §13–14 for the two substantive events: the occluder was
> found to collide (half the matched pairs had divergent physics — fixed,
> occluded corpus regenerated) and Experiment A landed at 0.8989 against the
> 0.90 kill threshold (diagnosed as a variance artifact of the camera
> geometry; the PI directed the run to continue with the number on record).
> The raw arrays, tables, figures, and gate outputs of that run are committed
> under `archive/full-run-2026-09-15/` — the machine that produced them was
> deleted, so the repo is the only copy. §3 and §5 below are updated; new
> traps found during the run are §5.11–5.15. Remaining work is §9.

---

## 1. What this project is, in one screen

Latent world models (DINO-WM and relatives) embed observations with a **frozen**
pretrained encoder and learn dynamics entirely in that latent space. That design
assumes the frozen encoder retains the physical state the dynamics model needs.
If a property isn't in the representation, no amount of predictor capacity
recovers it.

This repo measures which physical state survives the encoder. It generates
MuJoCo push scenes where physical properties are known exactly and
**statistically decorrelated from appearance**, extracts frozen features from
several encoders, and trains **linear probes** to decode each property from each
layer.

Four hypotheses (full statements in `spec.md` §2):

| | Claim | Spec predicted | Pilot measured |
|---|---|---|---|
| H1 | Geometric/kinematic state is linearly decodable | confirmed | **confirmed** |
| H2 | Mass and friction are **not** decodable from single frames | confirmed | **confirmed** |
| H3 | Video encoders partially recover friction | partially | **confirmed** |
| H4 | Contact degrades disproportionately under occlusion | confirmed | **refuted** |

**The most important thing to internalise:** several probes are *supposed* to
land at chance. A low number on mass or friction is the finding, not a bug.
`spec.md` §0.2 is explicit about this. Only the designated positive controls
(object position, §9.1) warrant investigation when they come back low.

---

## 2. Document map

Read in this order:

| File | What it is |
|---|---|
| `HANDOFF.md` | this file |
| `spec.md` | **the source of truth.** If your priors conflict with it, follow it |
| `DEVIATIONS.md` | all 14 departures from the spec, each with the measurement that forced it |
| `RESULTS_full.md` | **the full-corpus numbers** — 5 seeds, clean occluder; supersedes the pilot |
| `RESULTS_pilot.md` | pilot numbers; note its Exp E ran on the colliding occluder (§5.11) |
| `prereg.md` | effect size, layer set, baselines — fixed before Exp D, do not change |
| `README.md` | orientation for a human reader |
| `archive/full-run-2026-09-15/` | raw arrays, tables, figures, gates of the full run — the only surviving copy |

`DEVIATIONS.md` matters more than it looks. Several spec instructions are
internally inconsistent (notably Gate G1's threshold versus its sample size),
and that file records which way each was resolved and why. Do not silently
re-resolve one.

---

## 3. What is done, and what is not

### Done and validated end to end

- Corpus generation, ground truth, rendering, all four gates (G4 passes with
  fp16 costing 3.7e-05 R² against a 0.01 tolerance — the precision question is
  closed)
- Feature extraction for all five encoders (`dinov2_b`, `random_b`,
  `raw_pixel`, `videomae_b`, `vjepa2`)
- Experiments A, B, C, D, E, F — run on 1,500 base + 500 occluded scenes
- Figures 1–4, hypothesis tests with Holm and TOST
- 86 tests, ~9 s, no checkpoint downloads

### Done in the 2026-09-15 full run (previously listed as not-run)

Full corpus (3,000+1,000, with the §5.11 occluder fix), Experiment S at five
seeds, five probe seeds across C–F, G4, and the figures. Results in
`RESULTS_full.md`; raw arrays in `archive/full-run-2026-09-15/`.

### Not done

| Item | Why not | What it needs |
|---|---|---|
| **v2 occluder** (slab in both conditions, placement varied) | design decision pending: third condition vs replacement | scene.py changes + tests, ~2 min corpus, ~20 min GPU, ~15 min probes |
| **Experiment G** (H5 latent dynamics) | optional stretch goal, spec ranks it last | ~4–6 GPU-hr; code in `latent_dynamics.py`, unit-tested, never run at scale |

### The open question of the first handoff, resolved

**Experiment A at the full corpus: mean R² = 0.8989 vs the 0.90 threshold**
(`obj_pos_x` 0.923, `obj_pos_y` 0.875). Diagnosed before proceeding
(`DEVIATIONS.md` §14): the probe's pixel RMSE is *better* on y than x
(6.14 vs 6.37 px); y's lower R² is the camera's foreshortened vertical
variance (0.57× of x) plus airborne frames from the spec-required
`spawn_height` factor, and the threshold sits inside the bootstrap 95% CI
[0.890, 0.907]. The PI directed the pipeline to continue; the number stands
unedited. If you rerun and see ~0.899 again, that is this known geometry
artifact, not a new regression — but if it *drops*, investigate.

---

## 4. Running this on a fresh Linux VM

This is the part where a remote VM differs most from where the pilot ran (macOS,
Apple MPS, hardware CGL). **Read §5 before running anything.**

### 4.1 Machine shape

| Stage | Wants | Why |
|---|---|---|
| Corpus generation | CPU only, many cores, 8 GB RAM | rendering is a minority of per-scene cost; a GPU second costs ~13× a core second for a ~2× speedup (spec §17.2) |
| Feature extraction | 1 GPU, 16 GB RAM | the only stage that needs one |
| Probes and figures | CPU, 32 GB RAM if running Exp S | probes are linear algebra on cached features; 32 GB only because patch tokens go resident |

A single 8–16 core box with one modest GPU (L4, A10, T4) does the whole thing.

### 4.2 Disk

Measured on the pilot, extrapolated to the full 3,000+1,000 corpus:

| Item | Size |
|---|---|
| Corpus (frames + ground truth) | ~0.7 GB |
| Pooled features, all 5 encoders | ~1.1 GB |
| **Subtotal without patch tokens** | **~2 GB** |
| Patch tokens (`dinov2_b`, layers 8 and 11, base only) | ~24 GB |
| **Total if running Experiment S** | **~26 GB** |

Provision **60 GB** and you will not think about it. The spec's §7.3 estimate of
37 GB was conservative; frames pack smaller than assumed.

### 4.3 Setup

```bash
# System libraries. libosmesa6 is what makes headless software rendering work.
sudo apt-get update && sudo apt-get install -y \
    libgl1 libglew-dev libosmesa6 libosmesa6-dev libegl1 libgles2 git

git clone git@github.com:Kushaan-N/PhysGround.git && cd PhysGround
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e '.[encoders,video]'
```

Python 3.11 to match `env.yml` and the Modal image. `conda env create -f env.yml`
also works.

### 4.4 Run order

Each stage is a kill-switch for the next and costs less than what follows
(spec §10). Do not skip ahead.

```bash
# 0. Prove the renderer is what you think it is. Non-negotiable — see §5.1.
python scripts/preflight_render.py --json

# 1. Pilot + gates. Minutes. Then LOOK AT the contact sheet.
make pilot
#    -> open outputs/gates/contact_sheet.png. Gate G2 has no automated verdict.

# 2. Full corpus. Parallelise across shards; ~25 CPU-hr aggregate.
for s in $(seq 0 15); do
  python scripts/generate_corpus.py --condition base --n 3000 --seed 0 \
      --shard $s --n-shards 16 --skip-preflight &
done; wait
for s in $(seq 0 15); do
  python scripts/generate_corpus.py --condition occluded --n 1000 --seed 0 \
      --shard $s --n-shards 16 --skip-preflight &
done; wait

# 3. Confirm generation actually finished. A dead shard leaves a hole and no
#    process reports it — see 5.9. Seconds, read-only.
python scripts/check_corpus.py --condition base --expected 3000
python scripts/check_corpus.py --condition occluded --expected 1000 --paired-with base

# 4. Features. Drop --no-patches for dinov2_b/base if you want Experiment S.
for enc in dinov2_b random_b raw_pixel; do
  for cond in base occluded; do
    for s in $(seq 0 7); do
      python scripts/extract_features.py --encoder $enc --condition $cond \
          --shard $s --n-shards 8 --no-patches
    done
  done
done
# Video encoders are used only by Experiment F, which is base-only (a video
# encoder emits one embedding per scene, so per-frame targets are undefined for
# it and the occlusion analysis cannot use it). Extracting them for `occluded`
# costs ~15 min and nothing reads the result.
for enc in videomae_b vjepa2; do
  for s in $(seq 0 3); do
    python scripts/extract_features.py --encoder $enc --condition base \
        --shard $s --n-shards 4 --no-patches
  done
done

# 5. Kill-switches. These exit non-zero on failure; let them stop you.
python scripts/run_probes.py --exp A,B --seed 0

# 6. The grid, 5 seeds as spec 3.5 requires. Seeds are independent, so --jobs is
#    a straight wall-clock win; keep it at 1 for experiment S (see 5.10).
python scripts/run_probes.py --exp C,D,E,F --seed 0,1,2,3,4 --jobs 5

# 7. Tables, figures, hypothesis tests. No GPU, no model.
python scripts/make_figures.py --seeds 0,1,2,3,4
```

SLURM equivalents are in `slurm/*.sbatch`; Modal in `modal_app.py`
(`modal run modal_app.py`).

### 4.5 Everything resumes; nothing needs babysitting

Every stage is idempotent against a config hash, so a run that dies partway
through redoes only what it lost (spec §0.6). That covers generation,
extraction, **and experiments** — the last of those was a gap until recently:
the experiment runner wrote a completion marker that nothing ever read, so a
grid killed at seed 3 redid seeds 0–2 on restart.

Practically, on a preemptible VM you can re-issue the exact same command after
an interruption and it will pick up where it stopped. Pass `--force` to redo
regardless.

One subtlety worth knowing rather than discovering: Experiments A and B
**recompute their verdict from the cached archive** instead of reporting
"skipped" with no verdict. Otherwise resuming a grid whose kill-switch had
already *failed* would skip it, report nothing wrong, and let the run continue
past a stop condition.

The config hash covers the probe seed and the cell grid, so changing the encoder
set, layers, or views correctly invalidates prior work rather than silently
reusing it.

### 4.6 Where everything lives, and what does *not* come with the clone

**Nothing under `outputs/` is in git.** The repo tracks code and documents only;
the pilot corpus, features, and results lived on the machine that produced them
and are gone. Every number in `RESULTS_pilot.md` is reproducible from `main`
plus a seed, but you have to regenerate it. Budget for that — it is §4.4 in
full, not a download.

**`PHYSGROUND_DATA` is the single knob for where output goes.** Unset,
everything lands in `./outputs/`. Set it, and every path follows, with no call
site hard-coding anything:

```bash
export PHYSGROUND_DATA=/mnt/data/physground     # a mounted volume, not the boot disk
```

This is the only thing Modal overrides (`modal_app.py` points it at `/data`).
On a VM with a small root disk and a big attached volume, set it before anything
else — the corpus and patch tokens are what fill a disk.

Layout underneath it:

```
$PHYSGROUND_DATA/
  corpus/{base,occluded}/{00000,00001,...}/
      frames.npz     10 PNG-packed frames
      gt.npz         per-frame ground truth, frame-aligned
      factors.json   sampled factors + QA record for the scene
  features/{encoder}/{condition}/
      shard_0000.npz        pooled views, all layers, fp16
      shard_0000_patch.npz  patch tokens (only if extracted without --no-patches)
  results/{A..G,S}/seed_00/raw.npz   raw per-item predictions — the deliverable
  gates/          decorrelation.csv, contact_sheet.png, gate_g*.json
  tables/         exp_*_summary.csv, hypothesis_tests.csv
  figures/        fig1..fig4 png
```

**Two different things are called `--seed`.** On `generate_corpus.py` it is the
**master seed** that determines the corpus, and it must stay fixed across every
shard and both conditions or the matched occlusion pairs stop matching. On
`run_probes.py` it is the **probe seed**, which only chooses the train/test scene
partition; spec §3.5 wants five of those over one fixed corpus.

### 4.7 Makefile shortcuts

`make help` lists them. The useful ones:

| target | does |
|---|---|
| `make preflight` | render backend identity + timing check |
| `make pilot` | 50-scene pilot, then gates G1–G3 |
| `make corpus` | full corpus, both conditions |
| `make check` | corpus completeness and per-scene quality flags |
| `make features` | all encoders, both conditions, no patch tokens |
| `make probes` | A,B as kill-switches, then C,D,E,F over `$(SEEDS)`, `JOBS=` workers |
| `make figures` | regenerate every table and figure |
| `make test` | the 86 tests |
| `make clean-outputs` | delete everything regenerable; never touches code |

Override the defaults inline: `N_BASE=3000 SEEDS=0,1,2,3,4 JOBS=5 make probes`.

### 4.8 Timings

Measured on Apple M3 Pro, single process, with hardware GL and MPS. **Linux VM
numbers will differ** — generation is slower under software rendering, extraction
is faster on a real CUDA GPU.

| Stage | Measured | Note |
|---|---|---|
| Generation | 74 ms/scene | expect ~2× slower under `osmesa` (spec §17.2) |
| DINOv2 extraction | 41 frames/s effective | CUDA should be several× faster |
| VideoMAE | 156 s / 1,500 scenes | |
| V-JEPA 2 | 882 s / 1,500 scenes | the long pole; ~4× that for 4,000 scenes on MPS |
| Exp D, 16 cells | 562 s | per seed; logistic probes dominate, not ridge. `--jobs 5` runs five seeds in the time of roughly one |
| Exp F | 12 s | features already cached |
| Test suite | 8 s | |

---

## 5. Traps. Read this before running anything.

Every one of these was hit during implementation. Each produces a **plausible
wrong answer with no error**, which is why they are listed rather than merely
fixed.

### 5.1 `MUJOCO_GL` and the silent software fallback

MuJoCo reads `MUJOCO_GL` **once**, at `import mujoco`. Setting it afterwards
does nothing, silently. Every entry point here resolves it before importing
anything that reaches mujoco; `scripts/preflight_render.py::configure_gl` is the
single place that decides.

The dangerous case (spec §17.1, case 3): a CPU container with Mesa's EGL
installed accepts `MUJOCO_GL=egl` **without error** and hands you llvmpipe
software rasterisation. Nothing raises. You get ~7× slower rendering while
believing you have hardware acceleration, and every wall-clock estimate
downstream is invalid. **Checking that EGL initialised is not verification** —
something answered; the question is what.

`preflight_render.py` checks the renderer's *identity* (GL_RENDERER string) and
its *timing*, because either alone can be fooled. Run it on every worker. On
Linux it defaults to `osmesa`; software rendering is fine, silently falling into
it is not.

### 5.2 Mass must be set explicitly, never derived from density

In MJCF a geom's mass comes from `density × volume` unless `mass` is set. Sample
`obj_size`, leave density at its default, and mass becomes a deterministic
function of size — the decorrelation the entire project depends on is gone,
nothing errors, and the renders look perfect. `rollout.py::generate_scene`
asserts `model.body_mass` equals the sampled value on **every** scene. Do not
remove that assertion.

### 5.3 Splits are by scene, never by frame

Ten frames from one scene share lighting, object identity, colour, mass, and
friction. A frame-level split inflates every number, worst for exactly the
per-scene properties whose *absence* is the claim. A leaked mass probe reads as
evidence against H2. `splits.py` only accepts scene-level partitions; keep it
that way, including for the inner CV folds.

### 5.4 VideoMAE loads with zeroed attention biases

transformers 5.x expects `query/key/value.bias`; VideoMAE follows BEiT and
stores `q_bias`/`v_bias` with no key bias. The loader reports them MISSING and
substitutes zeros. The discarded layer-0 `q_bias` has norm 17.5, and pooled
features differ 19–25% in relative L2 from the correct ones — enough to move
H3's result, subtle enough to read as a real finding.
`encoders/videomae.py::repair_attention_biases` restores them, verifies all 12
layers took, and raises rather than proceeding. It no-ops on any version that
loads them correctly. **If you upgrade transformers, re-check this.**

### 5.5 `mj_multiRay` with a negative cutoff returns nothing

It prunes geoms *further than* cutoff, so `cutoff=-1` prunes the whole scene.
Every ray misses and the occlusion fraction comes back as exactly 0.000 on every
frame — a plausible number, no error. Cross-checked against per-ray `mj_ray` in
`tests/test_ground_truth.py`.

### 5.6 Geom group 3 is not rendered

`MjvOption` defaults to `geomgroup [1,1,1,0,0,0]`. A target placed in group 3
would be absent from every corpus frame while the renders otherwise look normal.
`scene.py::TARGET_GEOM_GROUP` is 2, and a test asserts it.

### 5.7 Stale feature shards

Shard files are named by index. Re-extract with a *smaller* `--n-shards` and the
surplus files from the previous run stay behind; the loader globs the directory
and every row they cover appears twice. `features.py` refuses a shard set with
duplicate `(scene, frame)` pairs, but the clean move is to delete the encoder's
directory before re-extracting with different sharding.

### 5.8 Gate G1's sample size

G1 tests the **factor sampler**, not the corpus, and sampling is free. Under
independence Spearman ρ has SE ≈ 1/√(n−1) = 0.143 at n=50, so a correctly
independent pair clears the 0.10 threshold about half the time. Measured on this
factor table: n=50 gives max |ρ| = 0.446, n=3000 gives 0.053. `gate_g1` raises
its own sample size to 1,000 and says so. Do not pass it a pilot size and
conclude the sampler is broken. Full reasoning in `DEVIATIONS.md` §1.

### 5.9 A dead generation shard is silent

Generation runs as N parallel shards. If one dies — preemption, OOM, a node
going away — the others finish and the run *looks* successful, because no single
process's exit code covers the hole. It surfaces much later as a feature matrix
with fewer rows than expected, pointing nowhere near generation.

`scripts/check_corpus.py` closes this. It is read-only, takes seconds, and also
reports the per-scene quality flags the rollout already records (distractor
touched, target left frame, settled, contact rate). When missing indices form an
arithmetic progression it says so and names the shard to re-run, since that is
exactly the signature of one dead worker. Run it between generation and
extraction.

### 5.10 `--jobs` memory, and why experiment S is different

Probe seeds are independent, so `--jobs N` is a straight wall-clock win — on the
pilot, Experiment D alone was 562 s per seed, so five seeds serially is ~47
minutes that parallelises to ~10.

Each worker holds its own copy of the feature matrix. For the pooled views that
is ~100 MB per worker and irrelevant. For **experiment S it is gigabytes**,
because patch tokens are ~60× the pooled payload — keep `--jobs 1` there, and
budget **64 GB** for the job: the 23 GB fp16 patch cache upcasts to fp32 in
the solver, and 32 GB (this handoff's original estimate) was OOM-killed
during the load.

Workers pin themselves to one BLAS thread **via `setdefault`**, so an
explicitly exported `OMP_NUM_THREADS` wins. With the newton-cholesky solver
(§5.12) the pin is no longer optimal: its Newton steps are dense `gemm` that
scales to ~4 threads, which is why `slurm/grid.sbatch` runs 5 workers × 4
threads on 21 cores. The 1-thread advice stands for anything gemv-bound.

### 5.11 The occluder collided until 2026-09-15

The slab was emitted with MuJoCo's default `contype`/`conaffinity` and stood
inside the push's travel: the actuator squeezed the object against it and
505/1,000 occluded scenes diverged from their matched base pairs, 32 ejected
out of frame entirely. Every occluded render looked perfect. Fixed with
`contype="0" conaffinity="0"` (visual-only slab);
`tests/test_scene.py::test_occluder_leaves_the_physics_untouched` fails on
the old behaviour in 0.7 s. Full measurements in `DEVIATIONS.md` §13. Any
corpus generated before commit `9416ba0` has broken pairing — regenerate.

### 5.12 lbfgs stops converging between 15k and 30k rows

The logistic probes converged under lbfgs at pilot scale and could not reach
tol=1e-4 within 1,000 iterations at the full corpus — even at C=100, at
~0.26 s/iteration. No iteration budget fixes it (500 → 2,000 → 20,000 all
failed); a 21-core grid job burned 3h20 without finishing Experiment C.
`LogisticProbe` now uses `newton-cholesky` with a warm-started ascending C
path: measured step counts [4,4,3,3,2,2,2,2,1] over the 9-C path, a full
fold path in 47 s, and the whole C,D,E,F × 5-seed grid in 39 minutes.
fp32 was tried for the 2× BLAS win and rejected: the solver runs to the cap
instead of converging. Same objective, same tol — the probe definition is
unchanged.

### 5.13 Loading patch tokens the obvious way is 3× peak memory

`load_patches` used to collect fp32 copies of every shard, concatenate, then
fancy-index sort — ~70 GB transient for a layer whose final array is 24 GB.
It now reads the index columns first, computes each row's sorted destination,
and scatter-writes each fp16 shard into one preallocated array. Peak is the
final array plus one shard. Do not "simplify" it back.

### 5.14 Old cluster GPUs silently lack kernels for current torch

torch 2.14+cu130 ships kernels for sm_75+ only (`torch.cuda.get_arch_list()`).
Unity's `gpu` partition still fields Maxwell/Pascal/Volta cards; an
unconstrained job drew a Tesla M40 and died on the first conv with
`cudaErrorNoKernelImageForDevice`. `slurm/extract_all.sbatch` carries the
constraint (`2080_ti|rtx_8000|a100|a40|a4000|a16|l4|l40s|h100`); keep an
equivalent on any new cluster.

### 5.15 `sbatch --wrap` runs under sh, not bash

A wrapped one-liner beginning `set -euo pipefail` dies instantly ("Illegal
option -o pipefail") and takes its dependency chain with it. Write a script
file with a bash shebang instead — that is why `slurm/check.sbatch` exists.

---

## 6. Findings so far, and how to read them

Full numbers in `RESULTS_pilot.md`. Compressed:

**H1.** Object position R² 0.905 vs 0.466 random-init; contact 0.891 balanced
accuracy vs 0.727. Rises from block 2 to block 8, plateaus at 11.

**H2.** Mass is TOST-equivalent to random init (Δ = +0.001 R²). Friction is
*detectable* at p = 0.049 but Δ = 0.012 R², a quarter of the pre-registered
effect size. Both sit at R² ≈ 0 absolutely. `make_figures.py` reports that
conjunction as "detectable but below the threshold" rather than picking whichever
test reads better — that case is precisely why the SESOI is pre-registered.

**H3.** Friction: frame encoders ≈ 0, VideoMAE 0.265, V-JEPA 2 **0.573**. Mass:
recovered by nothing. That split is the physics being right — Coulomb friction
decelerates at `a = μg`, independent of mass, so the deceleration profile the
video encoders read determines μ and carries nothing about m.

**H4 — refuted, and this needs care.** Contact was predicted to degrade most
under occlusion; it degrades *least* (retains 0.83 vs object position's 0.09).
Training the probe *within* the occluded condition recovers almost everything
(object position 0.124 → **0.803**), so the state is still in the frozen
representation. The transfer number was measuring a readout pointed at the wrong
place.

The occluded condition **confounds two treatments**: hiding part of the target,
and adding a large object the probe has never seen. Retained skill *rises* with
occlusion fraction rather than falling, which is what gives it away. Do not
report this as "occlusion destroys contact information."

Experiment E now runs both families — transfer cells and matched-training cells —
so the two explanations stay separable.

**A v2 occluder design should put the slab in both conditions and vary only
where it stands** (beside the contact region versus in front of it), so the
distribution is matched and only the occlusion differs. That is the single
highest-value change to the scene design, and it is not yet implemented.

---

## 7. Rules of engagement

1. **`spec.md` outranks your priors.** Where they conflict, follow the spec and
   flag it. Where you must depart, add an entry to `DEVIATIONS.md` with the
   measurement that forced it.
2. **Do not fix a predicted null.** Mass and friction at chance on single frames
   is the result. The same discipline applies in reverse: H4 did not confirm,
   and that was recorded rather than tuned away by adjusting the occluder until
   it did.
3. **Do not change `prereg.md`.** The effect size, layer set, and baselines were
   fixed before Experiment D ran. A margin chosen after seeing numbers can
   always be set to make a null look decisive.
4. **Gate G2 needs a human.** It returns `passed: None` on purpose. It caught two
   real defects during implementation — an occluder anchored where the object no
   longer was, and the camera seeing the arm draped over the target — neither of
   which any automated check in the pipeline was written to notice. Render the
   contact sheet and actually look at it.
5. **Raw arrays are the deliverable.** Every experiment writes per-item
   predictions to `.npz`; every table and figure is recomputed from those by
   `make_figures.py`. Never compute a statistic that cannot be recomputed
   without re-running a model.
6. **Report faithfully.** If a gate fails, say so with the numbers. If a stage
   was skipped, say that.

---

## 8. Reading the raw results yourself

Every experiment writes one `results/{exp}/seed_XX/raw.npz` holding raw per-item
predictions and nothing else (spec §0.4). Tables and figures are recomputed from
those, so you can revise a statistic, re-seed a bootstrap, or apply a different
correction without re-running a model.

Keys are flat strings — `.npz` has no hierarchy, and building one out of pickled
objects would let a results file execute code on load. The format is eight
`|`-separated fields:

```
encoder | train_condition | Llayer | view | task | eval_condition | target | field
```

- `task` — `real`, `control` (selectivity), `transfer` / `matched` (Exp E),
  `scene_pooled` (Exp F), `spatial` (Exp S)
- `eval_condition` — which condition these rows were *evaluated* on, which for
  Exp E differs from the condition the probe was *trained* on
- `field` — `y_true`, `y_pred`, `y_score` (classification only), `scene_index`

A few keys have fewer fields and are metadata, not predictions:
`ridge_alpha`, `ridge_targets`, `logistic_C|{target}`, and
`{eval_condition}|occlusion_fraction`. The parser skips anything that is not
eight fields, so adding more is safe.

```python
from physground import summarize as S

rows = S.summarize_experiment("D", seed=0)        # bootstrap CI per cell
S.compare_to_baseline("D", 0, "dinov2_b", "random_b", "log_mass")   # paired + TOST
S.occlusion_curve("E", 0, ("contact_state", "obj_pos_x"))           # H4, paired
```

`summarize.py` imports no mujoco and needs no GPU, so analysis runs anywhere.

---

## 9. Suggested order of work

Items 1–3 of the original list (full corpus + Exp A recheck, five seeds
across C–F, Experiment S) were completed in the 2026-09-15 run — see
`RESULTS_full.md`. What remains:

1. **The v2 occluder** (§6). Slab in *both* conditions, varying only its
   placement (beside vs in front of the contact region), so the
   novel-object-presence and visibility-loss explanations of the H4 result
   separate. With the §5.11 fix the slab is visual-only, so a v2 pair is
   physics-identical by construction. Decide first whether it is a third
   condition or replaces `occluded`. Roughly half a day: hours of scene.py +
   test work, minutes of corpus, ~20 min GPU extraction, ~15 min probes.
2. **Experiment G / H5.** Optional, last, ~4–6 GPU-hr.

Practical notes for a fresh machine: the run artifacts live in
`archive/full-run-2026-09-15/` (analysis via `physground.summarize` needs no
corpus, no GPU, no mujoco). Rebuilding the corpus + features from seed 0 is
§4.4 in full — measured on Unity: generation ~30 s × 20 CPU tasks per
condition, all-encoder extraction 48 min on one L4, the probe grid 39 min on
21 cores, Exp S 4h45 at 64 GB. The SLURM files under `slurm/` encode the
working resource shapes; `extract_all.sbatch` runs every encoder sequentially
on one GPU allocation (one queue wait), and `check.sbatch` is the
generation→extraction bridge that stops the chain on an incomplete corpus.

---

## 10. Things I would check first if something looks wrong

- Run `pytest tests/ -q`. 86 tests, ~9 s, no downloads. If any fail, fix that
  before trusting a number.
- Run `python scripts/preflight_render.py`. Most rendering weirdness is this.
- Run `python scripts/validate_occlusion.py`. Pearson r should be > 0.95
  against the double render; it was 0.985 at handoff.
- Check Experiment B. Every selectivity control should sit at chance
  (regression R² within ±0.04 of 0, classification balanced accuracy 0.497–0.508
  at handoff). Anything above chance means scene identity is leaking across the
  split, and every other number is suspect until it is fixed.

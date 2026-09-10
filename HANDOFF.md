# Handoff

Written for whoever picks this up next — likely another LLM agent, on a fresh
remote VM. Read this first, then `spec.md`. Everything below is either measured
or points at the file that measures it; where a number is an estimate rather
than an observation, it says so.

**Repo:** `git@github.com:Kushaan-N/PhysGround.git`, branch `main`.
**State at handoff:** working tree clean, 81 tests passing, everything below
reproducible from `main`. `git log --oneline` reads as a narrative — each commit
message says what changed and, where a measurement forced it, what the
measurement was.

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
| `DEVIATIONS.md` | all 12 departures from the spec, each with the measurement that forced it |
| `RESULTS_pilot.md` | every pilot number, including the H4 refutation |
| `prereg.md` | effect size, layer set, baselines — fixed before Exp D, do not change |
| `README.md` | orientation for a human reader |

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
- 81 tests, ~8 s, no checkpoint downloads

### Implemented and unit-tested, but never run at scale

| Item | Why not | What it needs |
|---|---|---|
| **Experiment S** (spatial contact probe, spec §8.3) | needs the patch-token cache, ~24 GB | extract without `--no-patches` |
| **Experiment G** (H5 latent dynamics) | optional stretch goal | run only after A–F are clean |
| **5 probe seeds** | pilot ran 1 | `--seed 0,1,2,3,4`; spec §3.5 requires 5 for every reported cell |
| **Full corpus** | pilot ran 1,500+500 | spec §5.4 wants 3,000+1,000 |

### The one open question that matters

**Experiment A is at threshold, not over it.** `obj_pos_x` R² = 0.907, but the
mean of x and y is 0.888 against the spec's 0.90 kill condition. The learning
curve was still climbing (0.709 → 0.769 → 0.800 → 0.817 at 60/120/180/240
training scenes), which is why `prereg.md` declares that threshold as applying
at the **full 3,000-scene corpus**. Generating the full corpus and re-checking
Exp A is therefore the first real task. If it still misses 0.90 at 3,000 scenes,
that is a genuine signal something is wrong — do not wave it through.

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

# 3. Features. Drop --no-patches for dinov2_b/base if you want Experiment S.
for enc in dinov2_b random_b raw_pixel videomae_b vjepa2; do
  for cond in base occluded; do
    for s in $(seq 0 7); do
      python scripts/extract_features.py --encoder $enc --condition $cond \
          --shard $s --n-shards 8 --no-patches
    done
  done
done

# 4. Kill-switches. These exit non-zero on failure; let them stop you.
python scripts/run_probes.py --exp A,B --seed 0

# 5. The grid, 5 seeds as spec 3.5 requires.
python scripts/run_probes.py --exp C,D,E,F --seed 0,1,2,3,4

# 6. Tables, figures, hypothesis tests. No GPU, no model.
python scripts/make_figures.py --seeds 0,1,2,3,4
```

SLURM equivalents are in `slurm/*.sbatch`; Modal in `modal_app.py`
(`modal run modal_app.py`).

### 4.5 Timings

Measured on Apple M3 Pro, single process, with hardware GL and MPS. **Linux VM
numbers will differ** — generation is slower under software rendering, extraction
is faster on a real CUDA GPU.

| Stage | Measured | Note |
|---|---|---|
| Generation | 74 ms/scene | expect ~2× slower under `osmesa` (spec §17.2) |
| DINOv2 extraction | 41 frames/s effective | CUDA should be several× faster |
| VideoMAE | 156 s / 1,500 scenes | |
| V-JEPA 2 | 882 s / 1,500 scenes | the long pole; ~4× that for 4,000 scenes on MPS |
| Exp D, 16 cells | 562 s | logistic probes dominate, not ridge |
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

## 8. Suggested order of work

1. **Full corpus + re-check Experiment A.** The only open correctness question.
   If the positive control still misses 0.90 at 3,000 scenes, stop and diagnose.
2. **Five probe seeds across C, D, E, F.** Spec §3.5 requires it for every
   reported cell; the pilot has one.
3. **Experiment S** (spatial contact probe). Needs the patch cache. It is the
   first thing a reviewer asks about a null on pooled features.
4. **The v2 occluder** (§6). Turns the H4 refutation from a confound into a
   clean result.
5. **Experiment G / H5.** Optional, last.

---

## 9. Things I would check first if something looks wrong

- Run `pytest tests/ -q`. 81 tests, 8 s, no downloads. If any fail, fix that
  before trusting a number.
- Run `python scripts/preflight_render.py`. Most rendering weirdness is this.
- Run `python scripts/validate_occlusion.py`. Pearson r should be > 0.95
  against the double render; it was 0.985 at handoff.
- Check Experiment B. Every selectivity control should sit at chance
  (regression R² within ±0.04 of 0, classification balanced accuracy 0.497–0.508
  at handoff). Anything above chance means scene identity is leaking across the
  split, and every other number is suspect until it is fixed.

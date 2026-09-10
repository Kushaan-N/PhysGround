# PhysGround

**What physical state do frozen visual encoders actually encode?**

Latent world models — DINO-WM most clearly — predict future *representations*
rather than future pixels. They embed observations with a frozen pretrained
encoder and learn dynamics entirely in that latent space. The design rests on an
assumption that is almost never tested:

> The frozen encoder retains the physical state information that the dynamics
> model needs to predict.

If a property is not present in the frozen representation, no amount of
predictor capacity recovers it. The encoder is an information bottleneck fixed
before training begins. **A latent world model that cannot represent contact
cannot represent the event that determines what happens next.**

This repository measures which physical state survives the encoder. It generates
MuJoCo scenes where physical properties are known exactly and statistically
decorrelated from appearance, extracts frozen features from several pretrained
encoders, and trains linear probes to decode each property from each layer.

The contribution is a constraint on a design pattern, not a description of a
network.

---

## Hypotheses

| | Claim | Predicted | Pilot (1500+500 scenes, 1 seed) |
|---|---|---|---|
| **H1** | Geometric and kinematic state (object position, end-effector distance, support) is linearly decodable | confirmed | **confirmed** — object position +0.45 R² over random-init |
| **H2** | Mass and friction are **not** decodable from single-frame encoders | confirmed — *a predicted refutation* | **confirmed** — mass TOST-equivalent to random init |
| **H3** | Video encoders partially recover friction, and less so mass, from post-contact deceleration | partially confirmed | not yet run |
| **H4** | Contact state degrades disproportionately under occlusion | confirmed | **refuted** — contact is the *most* robust; see below |

H2 and H4 are the point. Several probes are **expected to land at chance**, and
a low number there is the result, not a bug to fix. Only the designated positive
controls (§9.1 of `spec.md`) warrant investigation when they come back low.

Full pilot numbers, including the gates and the kill-switches, are in
[`RESULTS_pilot.md`](RESULTS_pilot.md).

### H4 came out backwards, and the reason is the interesting part

Contact was predicted to degrade most under occlusion. It degrades *least* —
retaining 0.83 of its skill while object position retains 0.09. Training the
same probe *within* the occluded condition then recovers almost everything
(object position 0.124 → 0.803), so the state is still in the frozen
representation: the transfer number was measuring a linear readout pointed at
the wrong place, not an encoder that discarded information.

The occluded condition as specified confounds two treatments — hiding part of
the target, and adding a large object the probe has never seen. Retained skill
*rises* with occlusion fraction rather than falling, which is what gives it
away. A v2 design should put the occluder in both conditions and vary only where
it stands, so the distribution is matched and only the occlusion differs.

For a latent world model the finding is arguably sharper than the hypothesis it
replaces: a dynamics model trained on clean observations fails under occlusion
not because the encoder stops representing the world, but because its readout is
not robust to the scene changing. That is a different claim, and a fixable
problem.

---

## Quick start

```bash
conda env create -f env.yml && conda activate physground
pip install -e .

# The renderer must be verified, not assumed. See "Rendering" below.
python scripts/preflight_render.py

python scripts/generate_corpus.py --condition base     --n 3000 --seed 0
python scripts/generate_corpus.py --condition occluded --n 1000 --seed 0

python scripts/run_gates.py --pilot-scenes 50 --corpus-scenes 3000
#   ... then open outputs/gates/contact_sheet.png and look at it.

python scripts/extract_features.py --encoder dinov2_b --shard 0 --n-shards 8
python scripts/extract_features.py --encoder random_b --shard 0 --n-shards 8 --no-patches

python scripts/run_probes.py --exp A,B          # kill-switches; exit non-zero on failure
python scripts/run_probes.py --exp C,D,E --seed 0,1,2,3,4
python scripts/make_figures.py --seeds 0,1,2,3,4
```

On a cluster: `slurm/{generate,extract,probes}.sbatch`.
On Modal: `modal run modal_app.py`.

---

## How the design defends its claims

A null result is only interesting if the pipeline could have detected a positive
one. Four things carry that burden.

**Decorrelation is asserted, not assumed.** Every factor is sampled from its own
independent stream, and Gate G1 hard-fails if any physical factor correlates
with any appearance factor. The trap this exists for: in MJCF a geom's mass is
derived from `density × volume` unless set explicitly, so sampling `obj_size`
and leaving density at its default silently makes mass a function of size. Every
downstream number would be void, nothing would error, and the renders would look
perfect. Mass is set explicitly and the assertion runs on every scene.

**Splits are by scene, never by frame.** Ten frames from one scene share
lighting, object identity, colour, mass, and friction. A frame-level split puts
near-duplicates on both sides and inflates every number — most severely for
exactly the per-scene properties whose absence is the claim. A leaked mass probe
would read as evidence *against* H2.

**Every probe carries three baselines.** Random-init encoder, raw pixels, and
trivial. Untrained ViT features are a surprisingly strong probe target, so
without the random-init control a positive result shows only that the
architecture plus the image carries the property, not that pretraining put it
there.

**Absence is tested, not inferred from a null.** H2 claims properties are not
encoded. A non-significant difference is not evidence of absence, so the
comparison against the random-init baseline uses TOST equivalence against a
pre-registered effect size (`prereg.md`, committed before Experiment D).

---

## Rendering

MuJoCo reads `MUJOCO_GL` **once**, at `import mujoco`. Setting it afterwards
silently does nothing. Every entry point here resolves it before importing
anything that could reach mujoco.

The failure worth knowing about: a CPU container with Mesa's EGL installed will
accept `MUJOCO_GL=egl` without error and quietly hand you llvmpipe software
rasterisation. Nothing raises; you simply get ~7× slower rendering while
believing you have hardware acceleration, and every wall-clock estimate
downstream is invalid. **Checking that EGL initialised is not verification** —
something answered, but the question is what.

`scripts/preflight_render.py` therefore checks the renderer's *identity* and its
*timing*, because either alone can be fooled, and runs on every worker. Software
rendering is fine; silently falling back to it is not.

Corpus generation deliberately runs **CPU-only**. Rendering is a minority of
per-scene cost — physics dominates — so hardware EGL buys about 2× end to end,
while a GPU second costs roughly 13× a core second. The GPU is for feature
extraction and nothing else.

---

## Repository layout

```
spec.md            the source of truth (v1)
prereg.md          effect size, layer set, baselines — committed before Exp D
DEVIATIONS.md      every departure from the spec, with the measurement behind it
RESULTS_pilot.md   pilot findings, including the H4 refutation

physground/
  factors.py     factor definitions, deterministic sampling, Gate G1
  scene.py       MJCF construction, planar arm kinematics, occluder placement
  rollout.py     episode execution, phase-based capture, per-scene ground truth
  ground_truth.py  contact, support, image-plane projection, occlusion
  render.py      offscreen rendering with context reuse; frame packing
  features.py    extraction driver, sharding, fp16 caching, alignment
  probes.py      ridge / logistic / spatial probes, selectivity control
  splits.py      scene-level splitting, matched-pair handling
  stats.py       cluster bootstrap, Holm-Bonferroni, TOST
  gates.py       G1-G4
  experiments.py Experiments A-F, plus G (H5) and S (spatial probe)
  summarize.py   raw arrays -> tables, tests, figures
  encoders/      dinov2, random_init, raw_pixel, videomae, vjepa2

scripts/         thin CLIs over the above
slurm/           array jobs, every one with an explicit --time
modal_app.py     Modal deployment
tests/           80 tests, ~8s, no checkpoint downloads
```

Everything under `outputs/` is regenerable from code plus a seed and is not
tracked.

---

## Two bugs this project found in its dependencies

Recorded here because both are silent and both would have corrupted a result.

**VideoMAE loads with zeroed attention biases under transformers 5.x.** VideoMAE
follows BEiT in giving attention a learned `q_bias` and `v_bias` with no key
bias, which is what the published checkpoint stores. transformers 5.x expects
the standard `query/key/value` bias triple, finds none, reports them as MISSING,
and substitutes zeros. The discarded layer-0 `q_bias` has norm 17.5. Pooled
features from the broken load differ from the correct ones by 19–25% in relative
L2 at every probed layer — enough to move H3's result, subtle enough to read as
a real finding about video encoders.
`physground/encoders/videomae.py::repair_attention_biases` restores them and
refuses to proceed if it cannot.

**`mj_multiRay` with a negative cutoff silently returns no hits.** It prunes
geoms *further than* cutoff, so `cutoff=-1` prunes the entire scene. Every ray
reports a miss, and the occlusion fraction comes back as exactly 0.000 on every
frame — a plausible number, no error. Cross-checked against per-ray `mj_ray` in
`tests/test_ground_truth.py`.

---

## Scope

**In:** frozen-encoder extraction, linear probing, occlusion ablation, and a
frame-versus-video comparison, on simulated data.

**Out for v1:** real-world video *(stated as a limitation in the abstract, not
buried in a footnote)*, fine-tuning any encoder (the premise is that it is
frozen), training a full world model, and more than one robot morphology.

---

## A note on the name

`PhysGround` is proposed, not fixed. Search Scholar, arXiv, and Papers with Code
for collisions before any public posting — an occupied name costs credibility
and is free to avoid now. Alternatives: *FrozenPhys*, *LatentGround*, *PG-FE*.

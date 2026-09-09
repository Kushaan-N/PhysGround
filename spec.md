# PhysGround
## What Physical State Do Frozen Visual Encoders Actually Encode?

### Implementation Spec v1 — self-contained, written to be handed to an LLM coding agent

---

## 0. Instructions for the implementing agent

**0.1 — This document is the source of truth.** If your training-data priors about
DINOv2, MuJoCo, or probing methodology conflict with something written here, follow this
document and flag the conflict in a comment. Where this document is silent, ask before
inventing.

**0.2 — Null results are results.** This project has hypotheses it expects to *refute*.
Several probes are predicted to land at chance. Do not "fix" a probe that returns low
accuracy on a property this document predicts is undecodable. Low accuracy there is the
finding. Only investigate low accuracy on the designated positive controls (§9.1).

**0.3 — Validation code is written before generation code.** Section 9 defines four
gates. Implement all four and confirm they pass on a 50-scene pilot before generating
the full corpus. This ordering is not negotiable; §12 explains why.

**0.4 — Raw-array persistence.** Every experiment writes its raw per-item predictions and
targets to disk as `.npz`, not just summary statistics. Summary tables are regenerated
from raw arrays by a separate script. Never compute a statistic that cannot be
recomputed without re-running the model.

**0.5 — Determinism.** Every script takes `--seed`. Set `numpy`, `torch`, and Python
`random` seeds. MuJoCo scene sampling must be reproducible from `(scene_id, seed)` alone,
with no dependence on iteration order or wall-clock.

**0.6 — Idempotence.** Every generation and training step writes to a deterministic path
derived from its config hash. On startup, skip any item whose output file already exists
and passes a checksum. Write to `path.tmp` then `os.replace()` so partial files are never
mistaken for complete ones. This project will run as SLURM array jobs that get preempted.

---

## 1. Thesis

### 1.1 The assumption under test

A growing class of world models — DINO-WM being the clearest example — predict future
**latent representations** rather than future pixels. They take a frozen, pretrained
visual encoder (DINOv2, V-JEPA, VideoMAE), embed observations, and learn dynamics purely
in that latent space. Planning happens by rolling the latent predictor forward and
scoring against a latent goal.

This design rests on an assumption that is almost never tested:

> **The frozen encoder retains the physical state information that the dynamics model
> needs to predict.**

If a property is not present in the frozen representation, no amount of predictor
capacity recovers it. The encoder is an information bottleneck fixed before training
begins. A latent world model that cannot represent contact cannot represent the event
that determines what happens next.

### 1.2 What this project does

Generate MuJoCo scenes in which physical properties are known exactly and **statistically
decorrelated from appearance**. Extract frozen features from several pretrained encoders.
Train linear probes to decode each physical property from each layer. Report what is
linearly available, what is not, and how that degrades under occlusion.

### 1.3 The framing that makes this publishable

Do not frame this as "what does DINOv2 encode." That genre is crowded and reviewers are
jaded about it. Frame it at the **assumption**:

> Latent world models assume frozen encoders preserve the physical state needed for
> dynamics. We measure which state survives the encoder, and find that geometric and
> kinematic state is linearly recoverable while dynamic properties are not — and that
> contact state, the property most load-bearing for manipulation, degrades sharply under
> the partial occlusion that manipulation constantly produces.

The contribution is a constraint on a design pattern, not a description of a network.

---

## 2. Hypotheses

Each is stated with its predicted outcome. **H2 and H4 are predicted refutations of the
implicit assumption in §1.1 and are the point of the paper.**

**H1 — Geometric/kinematic state is linearly decodable.**
Object image-plane position, arm end-effector position, object-arm distance, and support
relation (resting vs. airborne) are recoverable from frozen features with high accuracy
at mid-to-late layers, well above both the random-init and raw-pixel baselines.
*Predicted: confirmed.* H1 is partly a sanity check on the pipeline.

**H2 — Dynamic properties are not linearly decodable from single-frame encoders.**
Mass and sliding-friction coefficient are recoverable at or near the level of the
random-init control from single-frame encoders (DINOv2).
*Predicted: confirmed (i.e. these properties are absent).*
The mechanism is not mysterious: with appearance decorrelated from physics, mass and
friction are *not observable in a static image at all*. This is an
information-theoretic floor, not an encoder failure. Its purpose is to establish that
the pipeline detects genuine absence, and to set up H3.

**H3 — Video encoders partially recover dynamic properties.**
V-JEPA 2 and VideoMAE, given multi-frame clips spanning a push, recover friction and mass
above the single-frame floor — because deceleration profiles after contact are
informative about both.
*Predicted: partially confirmed, with friction recovered more strongly than mass.*
H3 is what makes the frame-vs-video comparison load-bearing rather than decorative.

**H4 — Contact state degrades disproportionately under occlusion.**
Binary arm-object contact is decodable in unoccluded frames but degrades substantially
more than object *presence* does when the contact region is partially occluded.
*Predicted: confirmed.* This is the finding with direct consequences for latent world
models in manipulation, where the manipulator itself occludes the contact.

**H5 (stretch, optional) — Probe decodability predicts downstream latent-dynamics error.**
Across encoder/layer choices, probe accuracy on contact state correlates with the
prediction error of a small DINO-WM-style latent predictor trained on the same features.
*Predicted: unknown.* Run only if Experiments 0–F complete cleanly. See §12.7.

---

## 3. Non-negotiable constraints

1. **Linear probes only** for the headline results. Ridge for regression, logistic
   regression for classification. An MLP probe invites the objection that the probe
   learned the task rather than read it out. If an MLP probe is run at all, it is an
   appendix row, clearly labeled, with the selectivity control (§9.2) reported beside it.

2. **Every probe carries three baselines** (§8.4): random-init encoder, raw-pixel, and
   trivial (mean/majority). A number without all three is uninterpretable.

3. **Splits are by scene, never by frame.** Frames from one scene are highly correlated.
   A frame-level split leaks the test set and inflates every number in the paper. See
   §7.4.

4. **Decorrelation is asserted in code, not assumed.** Gate G1 (§9.1) hard-fails
   generation if any physical property correlates with any appearance factor above
   threshold.

5. **Seeds.** 5 probe seeds (different train/test scene partitions) for every reported
   cell. Report mean and cluster-bootstrap CI, never a single number.

6. **fp16 for cached features.** Halves storage at no measurable cost to linear probe
   accuracy. Verify once on the pilot (§9.4) and then stop thinking about it.

---

## 4. Scope and what is explicitly out

**In scope:** frozen-encoder feature extraction, linear probing, occlusion ablation,
frame-vs-video encoder comparison, on simulated data.

**Out of scope for v1:**
- Real-world video. State this as a limitation in the abstract, not in a footnote where
  a reviewer gets to discover it themselves.
- Fine-tuning any encoder. The entire premise is that the encoder is frozen.
- Training a full world model. H5 uses a small latent predictor only, and is optional.
- More than one robot morphology. One planar pusher.

---

## 5. Scene design

### 5.1 The environment

A single MJCF model, procedurally parameterized:

- **Ground plane**, fixed.
- **Pusher arm**: 2-DOF planar arm (shoulder + elbow hinge joints) with a capsule
  end-effector, visible in frame at all times. Position-controlled via `mocap` or
  `position` actuators driving a scripted trajectory.
- **Target object**: one free body, geom type sampled from {box, sphere, cylinder}.
- **Distractor object** (present in 50% of scenes): a second free body, always off the
  arm's path, never contacted. Its role is to prevent the contact probe from degenerating
  into an "is there an object in frame" probe.
- **Occluder** (present in the occlusion condition only, §5.5): a static vertical slab
  between camera and the contact region.
- **Camera**: fixed pose with small per-scene jitter (§5.3).

### 5.2 The episode

Every scene runs one scripted push:

1. **Approach** — arm moves from canonical rest toward the target along a sampled
   approach angle.
2. **Contact** — end-effector contacts the target object.
3. **Push** — arm continues, object translates and decelerates according to its mass and
   friction.
4. **Retract** — arm returns to canonical rest. Object continues to a stop.

Ten frames are captured per scene at fixed *phase* fractions, not fixed timesteps, so
that contact-state classes are balanced across scenes regardless of episode duration:

| Frame index | Phase |
|---|---|
| 0 | Rest, pre-approach |
| 1–2 | Approach, no contact |
| 3 | First contact frame |
| 4–5 | Sustained contact, pushing |
| 6 | Last contact frame |
| 7 | Immediately post-contact, object still moving |
| 8 | Object decelerating |
| 9 | Object at rest, arm at canonical rest |

Record the actual phase label with every frame. Frames 3–6 are `contact=1`; all others
are `contact=0`. This yields roughly 40% positive class, which is close enough to
balanced that no reweighting is needed, but report the exact rate.

### 5.3 Factors

Two disjoint groups. **Physical factors** are the probe targets. **Appearance factors**
are nuisance variables that must be decorrelated from them.

**Physical factors (probe targets):**

| Name | Type | Range / values | Notes |
|---|---|---|---|
| `mass` | continuous | log-uniform, 0.05–2.0 kg | Set explicitly. See §6.1. |
| `friction_slide` | continuous | uniform, 0.05–1.2 | First component of geom `friction`. |
| `contact_state` | binary | {0, 1} | Derived per-frame, §6.2. |
| `support_state` | binary | {0, 1} | Resting on plane vs. airborne, §6.3. |
| `obj_pos_x`, `obj_pos_y` | continuous | scene-dependent | Image-plane pixel coords. Positive control. |
| `obj_speed` | continuous | derived | Magnitude of object linear velocity. |
| `ee_obj_dist` | continuous | derived | 3D distance, end-effector to object center. |

**Appearance factors (nuisance, must be decorrelated):**

| Name | Type | Range / values |
|---|---|---|
| `obj_size` | continuous | uniform, 0.03–0.09 m characteristic dimension |
| `obj_rgba` | categorical/continuous | uniform over hue, fixed saturation and value |
| `obj_geom_type` | categorical | {box, sphere, cylinder} |
| `floor_rgba` | continuous | uniform over a light-gray band |
| `light_pos` | continuous | small uniform jitter |
| `cam_jitter` | continuous | ±2° azimuth, ±2° elevation, ±3% distance |
| `distractor_present` | binary | {0, 1} |

**All factors are sampled independently.** No factor is a function of any other. §6.1
explains the one place where MuJoCo will silently violate this if you let it.

### 5.4 Sampling and corpus size

- **3,000 scenes** in the base (unoccluded) condition.
- **1,000 scenes** in the occlusion condition (§5.5), with the *same* sampled factor
  values as the first 1,000 base scenes so the comparison is matched and paired.
- 10 frames per scene → **40,000 frames total**.

Scene IDs are `f"{condition}_{index:05d}"`. Factor sampling is a pure function of
`(scene_id, master_seed)` via a per-scene `np.random.default_rng(hash)`. This makes the
matched occlusion pairs trivially reproducible.

### 5.5 The occlusion condition

Identical to the base condition, plus a static box geom positioned to occlude the region
around the predicted contact point, sized to hide roughly 40% of the contact area from
the camera. The occluder is a neutral gray, distinct from both object and floor.

The design intent: the object remains partially visible (so `obj_pos` and presence stay
decodable), but the *contact interface* is hidden. H4 predicts contact-state probe
accuracy collapses while object-position probe accuracy does not.

Record the actual per-frame occlusion fraction of the object's bounding box, computed by
rendering a segmentation pass with and without the occluder. This gives a continuous
covariate rather than a binary condition label, which is far more useful in analysis.

---

## 6. Ground-truth extraction

### 6.1 Mass — the trap that will silently void the paper

**In MJCF, a geom's mass is derived from `density` × volume unless `mass` is set
explicitly.** The default density is 1000. If you sample `obj_size` and leave density at
its default, then `mass` becomes a deterministic function of `obj_size`, your
decorrelation is destroyed, and your "mass probe" is a size probe. Every downstream
number is void, nothing errors, and the renders look perfect.

Set mass explicitly on the geom:

```xml
<geom name="target_geom" type="box" size="0.05 0.05 0.05" mass="0.37" .../>
```

Then **verify it took effect**, every scene, in code:

```python
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
assert abs(model.body_mass[body_id] - sampled_mass) < 1e-6, \
    f"mass not applied: got {model.body_mass[body_id]}, wanted {sampled_mass}"
```

This assertion is cheap and catches the single most expensive failure in the project.

### 6.2 Contact state

MuJoCo exposes active contacts in `data.contact[:data.ncon]`, each with `.geom1` and
`.geom2` geom indices. Naive presence-of-contact is not enough — contacts appear and
vanish across single timesteps, and zero-force contacts register.

Use a force threshold:

```python
def arm_object_contact(model, data, ee_geom_id, obj_geom_ids, force_thresh=1e-3):
    force = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {c.geom1, c.geom2}
        if ee_geom_id in pair and pair & obj_geom_ids:
            mujoco.mj_contactForce(model, data, i, force)
            if np.linalg.norm(force[:3]) > force_thresh:
                return True
    return False
```

**Debounce across timesteps.** A frame is labeled `contact=1` only if contact holds for
at least 3 consecutive physics steps centered on the captured frame. Contact flicker
produces mislabeled frames, and a probe trained on noisy labels reads as a weak probe —
you would misattribute a data bug to an encoder limitation.

### 6.3 Support state

Same routine with the floor geom in place of the end-effector. `support_state=1` when the
object has a force-bearing contact with the ground plane. During a push the object is
supported throughout; during any airborne phase it is not. If your scripted push never
produces airborne frames, `support_state` is constant and must be dropped from the probe
set — do not report a probe on a constant target. Check class balance in Gate G3.

### 6.4 Image-plane object position

Project the object body's world position through the camera matrix to pixel coordinates.
This is the **positive control** target: it is unambiguously present in the image, and
any encoder that fails to encode it indicates a pipeline bug, not a scientific finding.

### 6.5 Occlusion fraction

Render twice per frame in the occlusion condition — once with the occluder geom's
`rgba` alpha set to 0 and once normally — using segmentation rendering to get the
object's visible pixel count in each. The ratio is the occlusion fraction. Cache it as a
per-frame float.

---

## 7. Feature extraction

### 7.1 Encoders

| Key | Model | Input | Notes |
|---|---|---|---|
| `dinov2_b` | `dinov2_vitb14` via torch.hub | single frame, 224×224 | Primary. 12 blocks, patch 14 → 256 patches + CLS. |
| `random_b` | same architecture, randomly initialized | single frame | Baseline, §8.4. |
| `videomae_b` | `MCG-NJU/videomae-base` | 16-frame clip, 224×224 | Video, for H3. |
| `vjepa2` | V-JEPA 2 ViT-L | clip | Video, for H3. **See warning below.** |

**Warning to the implementing agent:** V-JEPA 2's released API and preprocessing
constants postdate much of the training data you were trained on. Do not write its
loading and preprocessing code from memory. Read the model card and config from the
actual HuggingFace repo before writing a line of it, and if the interface is unclear,
stop and ask rather than guessing at method names and normalization constants.

**Staging:** implement and validate `dinov2_b` and `random_b` first, end to end, through
Experiment D. The video encoders are only needed at Experiment F. Do not block the
pipeline on them.

### 7.2 Layers

For each encoder, extract from **four evenly spaced blocks plus the final block**. For
DINOv2 ViT-B (12 blocks): layers `[2, 5, 8, 11]`.

```python
outs = model.get_intermediate_layers(x, n=[2, 5, 8, 11], return_class_token=True)
```

Each element is `(patch_tokens, cls_token)` with `patch_tokens` of shape
`(B, 256, 768)`.

### 7.3 What to cache

Cache two pooled views for every layer of every encoder:

- `cls` — the class token, shape `(768,)`
- `mean` — mean over patch tokens, shape `(768,)`

Cache **full patch tokens only for `dinov2_b`, layers 8 and 11**. These are needed for
the spatial contact probe (§8.3) and nothing else.

Storage arithmetic, at 40,000 frames, fp16:

| Item | Size |
|---|---|
| Pooled, per encoder (4 layers × 2 views × 768 dims) | ~0.5 GB |
| Pooled, all 4 encoders | ~2 GB |
| Patch tokens, dinov2_b, 2 layers (256 × 768) | ~31 GB |
| Raw PNG frames, 224×224 | ~4 GB |
| **Total** | **~37 GB** |

If quota is tight, drop patch tokens to layer 11 only (~16 GB) and note the restriction
in §8.3.

### 7.4 Split discipline

**Split by scene ID, never by frame.** Ten frames from one scene share lighting, object
identity, color, mass, and friction. A frame-level split puts near-duplicates in train
and test and inflates every reported number, most severely for exactly the per-scene
properties (`mass`, `friction_slide`) whose absence is the paper's claim.

```python
scene_ids = sorted(set(meta["scene_id"]))
rng = np.random.default_rng(probe_seed)
rng.shuffle(scene_ids)
n_train = int(0.8 * len(scene_ids))
train_scenes, test_scenes = set(scene_ids[:n_train]), set(scene_ids[n_train:])
train_mask = np.array([s in train_scenes for s in meta["scene_id"]])
```

For the occlusion condition, matched scene pairs must fall on the **same side** of the
split. Partition by base scene index and apply to both conditions.

---

## 8. Probe protocol

### 8.1 Regression probes

Targets: `mass`, `friction_slide`, `obj_pos_x`, `obj_pos_y`, `obj_speed`, `ee_obj_dist`.

Standardize features (fit scaler on train only). `RidgeCV` over
`alphas=np.logspace(-3, 5, 17)`, internal CV on the training scenes, grouped by scene.
Metric: **R² on held-out scenes**. Report against the trivial baseline of predicting the
training mean, which by construction gives R² ≈ 0.

`mass` is probed in log space, since it is sampled log-uniformly.

### 8.2 Classification probes

Targets: `contact_state`, `support_state`.

`LogisticRegression` with L2, `C` swept over `np.logspace(-4, 4, 9)`, grouped CV.
Metrics: **balanced accuracy** and **AUROC**. Report the majority-class rate alongside.

### 8.3 Spatial contact probe (patch tokens)

For `contact_state` only, additionally train a probe on patch tokens: a per-patch logistic
probe with predictions max-pooled over patches. This tests whether contact is encoded
*somewhere* spatially even when pooled representations wash it out — a real possibility,
and one a reviewer will raise if you do not test it.

### 8.4 Mandatory baselines

Every reported cell has three:

1. **Random-init encoder** (`random_b`). Untrained ViT features are a surprisingly strong
   probe target. Without this, you cannot claim *pretraining* encoded anything.
2. **Raw pixel** — image downsampled to 32×32 grayscale, flattened to 1024 dims, same
   probe. If pixels match the encoder, the encoder contributed nothing.
3. **Trivial** — train-set mean for regression, majority class for classification.

### 8.5 Selectivity control

For every probe, run a matched control task: labels shuffled **across scenes**, preserving
the marginal distribution. Report `selectivity = metric_real − metric_control`.

The control must land at chance. If it does not, features are leaking scene identity —
almost always a split-leakage bug (§7.4). Investigate before trusting any real number.

---

## 9. Validation gates

All four run on a **50-scene pilot** before the full corpus is generated. All must pass.

### G1 — Decorrelation

Compute Spearman correlation between every (physical factor, appearance factor) pair.

**Hard-fail if `|rho| > 0.10` for any pair.** Write the full matrix to
`outputs/gates/decorrelation.csv` and render it as a heatmap — this figure goes in the
paper's appendix and it is what a skeptical reviewer will look for first.

Pay attention to `mass` × `obj_size` (§6.1) and `friction_slide` × `obj_geom_type`
(spheres roll rather than slide; if geom type is sampled independently this is fine as a
factor, but check that friction remains *estimable* within each type).

### G2 — Visual inspection

Render a contact sheet of 20 randomly sampled frames, tiled, with each frame's phase
label, contact state, and occlusion fraction printed on it. **Open it and look at it.**

This is the only gate a human must execute. It catches objects intersecting the floor,
the arm out of frame, the occluder in the wrong place, degenerate camera angles, and
the object leaving frame during the push — none of which any automated check you would
think to write will catch.

### G3 — Label sanity

- `contact_state` positive rate between 0.25 and 0.55.
- `support_state` not constant (else drop it, §6.3).
- No NaNs or infinities in any target.
- `mass` and `friction_slide` marginals match their sampling distributions
  (KS test, p > 0.01).

### G4 — Precision and fp16

Extract features for the pilot in both fp32 and fp16, run the positive-control probe on
both, and confirm R² differs by less than 0.01. Then use fp16 everywhere and stop
thinking about it.

---

## 10. Experiment sequence

Ordered so that each step is a kill-switch for everything after it, and costs less than
what follows.

| Exp | What | Cost | Kill condition |
|---|---|---|---|
| **0** | Corpus generation + Gates G1–G4 on 50-scene pilot | CPU, minutes | Any gate fails → fix generation, do not proceed |
| **A** | Positive control: `obj_pos_x/y` from `dinov2_b` L8 | <5 min GPU | R² < 0.90 → pipeline is broken, stop |
| **B** | Selectivity control on all targets | CPU | Control above chance → split leakage, stop |
| **C** | Baselines: `random_b` and raw-pixel across all targets | ~15 min | — |
| **D** | **Main grid**: 2 encoders × 4 layers × 2 views × 7 targets × 5 seeds | ~2 CPU-hr | — |
| **E** | Occlusion: matched pairs, H4 | ~1 CPU-hr | — |
| **F** | Video encoders on clips, H3 | ~1 GPU-hr | — |
| **G** | *(optional)* H5, latent predictor correlation | ~4 GPU-hr | Skip if 0–F ran long |

**Run Experiment A before generating the full corpus.** It uses the 50-scene pilot. If
the positive control fails on 50 scenes it will fail on 3,000, and you will have spent
the generation budget to learn nothing.

---

## 11. Statistics

**Cluster bootstrap over scenes.** Frames within a scene are not independent. Resample
*scenes* with replacement (not frames), recompute the metric, 2,000 iterations, report
the 2.5/97.5 percentiles.

**Multiple comparisons.** The main grid is 2 × 4 × 2 × 7 = 112 cells. For any claim of
the form "property P is decodable above baseline," apply **Holm–Bonferroni** across the
family of tests within that property. State the family explicitly in the paper.

**Claims of absence.** H2 asserts properties are *not* encoded. A non-significant test is
not evidence of absence. Use **equivalence testing**: pre-specify a smallest effect size
of interest (proposed: R² = 0.05 above the random-init baseline) and run a TOST. Reporting
"p > 0.05, therefore absent" is the single most likely way this paper gets rejected.

**Pre-register** the smallest-effect-size threshold, the layer set, and the baseline
choice before running Experiment D. Commit it to the repo with a timestamp.

---

## 12. Repository structure

```
physground/
  README.md
  spec.md                      # this document
  prereg.md                    # committed before Exp D, timestamped
  env.yml
  physground/
    __init__.py
    scene.py                   # MJCF construction, factor sampling
    factors.py                 # factor definitions, sampling, decorrelation check
    rollout.py                 # episode execution, frame capture, GT extraction
    ground_truth.py            # contact/support/position/occlusion extraction
    encoders/
      base.py                  # Encoder ABC: .embed(frames) -> dict[layer][view]
      dinov2.py
      random_init.py
      videomae.py
      vjepa2.py
      raw_pixel.py
    features.py                # extraction driver, caching, fp16, manifest
    probes.py                  # ridge/logistic, CV, selectivity control
    splits.py                  # scene-level splitting, matched-pair handling
    stats.py                   # cluster bootstrap, Holm, TOST
    figures.py
  scripts/
    generate_corpus.py         # --condition {base,occluded} --n --seed
    run_gates.py               # G1-G4, writes outputs/gates/
    extract_features.py        # --encoder --layers --shard --n-shards
    run_probes.py              # --exp {A,B,C,D,E,F} --seed
    make_figures.py
  slurm/
    generate.sbatch            # CPU partition, array
    extract.sbatch             # GPU partition, array
    probes.sbatch              # CPU partition
  outputs/
    corpus/{condition}/{scene_id}/  frames/*.png, gt.npz, factors.json
    features/{encoder}/{layer}/{shard}.npz
    gates/
    results/{exp}/{seed}/raw.npz
    figures/
```

### Encoder interface

Every encoder implements one method, so probes never know which model produced features:

```python
class Encoder(ABC):
    name: str
    layers: list[int]

    @abstractmethod
    def embed(self, frames: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
        """frames: (N, H, W, 3) uint8 for image encoders,
                   (N, T, H, W, 3) uint8 for video encoders.
        Returns {layer: {"cls": (N, D), "mean": (N, D), "patch": (N, P, D) | None}}
        """
```

Video encoders receive the 10 captured frames per scene as a clip and return one
embedding per *scene*, not per frame. Per-scene targets (`mass`, `friction_slide`) probe
directly. Per-frame targets are not probed on video encoders in v1 — note this asymmetry
explicitly in the paper, since it means H3 is tested on dynamic properties only.

---

## 13. Compute and storage budget

| Stage | Compute | Wall-clock (1 GPU / CPU array) |
|---|---|---|
| Corpus generation, 4,000 scenes | ~25 CPU-hr | <1 hr on a CPU array |
| Gates | minutes | minutes |
| Feature extraction, dinov2 + random | ~0.5 GPU-hr | ~30 min |
| Feature extraction, video encoders | ~0.5 GPU-hr | ~30 min |
| Probes, full grid × 5 seeds | ~3 CPU-hr | ~1 hr |
| **Nominal total** | **~1 GPU-hr, ~30 CPU-hr** | |
| **With 5× debugging multiplier** | **~5 GPU-hr, ~150 CPU-hr** | |

Storage: ~37 GB (§7.3).

Every stage is an independent array job. Nothing requires a continuous session. Set
`--time` explicitly on every SLURM submission.

---

## 14. Known failure modes

**`MUJOCO_GL` must be set before `import mujoco`.** Setting it after import silently has
no effect. On a headless cluster node, set `MUJOCO_GL=egl` at the top of the entry script,
before any project import.

**EGL silently falling back to osmesa.** If EGL libraries are missing on a compute node,
some setups fall back to software rendering, which is roughly 10× slower and produces
subtly different output. Verify in a preflight check with an *actual one-frame render*
and a timing assertion — not by checking the environment variable.

**Mass/density coupling** — §6.1. The most expensive failure in the project.

**Frame-level splits** — §7.4. Inflates everything, most severely the numbers whose
smallness is the finding.

**Contact flicker** — §6.2. Mislabeled frames read as encoder weakness.

**Patch-token storage blowup** — §7.3. Caching patch tokens for all encoders and all
layers is ~250 GB and will exhaust your quota mid-run.

**Constant `support_state`** — §6.3. Probing a constant target returns a meaningless
number that looks like a result.

**V-JEPA 2 API drift** — §7.1. Read the model card; do not write it from memory.

**SLURM default time limit.** Omitting `--time` kills jobs at the partition default with
no informative error.

---

## 15. Deliverables

**Figures:**

1. Decorrelation heatmap over all factor pairs (appendix, but load-bearing for trust).
2. Main result: R² / balanced accuracy by property × layer, with random-init and
   raw-pixel baselines overlaid, cluster-bootstrap CIs. This is Figure 1.
3. Frame vs. video encoder on dynamic properties (H3).
4. Occlusion: contact-state accuracy vs. occlusion fraction, with object-position accuracy
   on the same axes as the contrast (H4). This is the figure that carries the paper's
   consequence.
5. Contact sheet of example frames across conditions.

**Artifacts:** corpus generation code, cached pooled features, probe results as raw
`.npz`, and the pre-registration file.

---

## 16. A note on the name

**PhysGround** is proposed, not fixed. Before any public posting, search Google Scholar,
arXiv, and Papers with Code for collisions — an occupied name costs credibility and is
free to avoid now. Alternatives: *FrozenPhys*, *LatentGround*, *PG-FE (Physical Grounding
in Frozen Encoders)*.

---

## 17. Modal deployment and the EGL/osmesa rendering problem

### 17.1 The problem

MuJoCo selects its OpenGL backend from the `MUJOCO_GL` environment variable, read once at
import. `egl` gives hardware-accelerated offscreen rendering. `osmesa` gives software
rasterization, roughly 5–10× slower per frame.

**Hardware EGL requires an NVIDIA driver and its vendor ICD**, normally installed at
`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`. On Modal, the driver is mounted into the
container only when a GPU is attached. A CPU-only container has the EGL *loader* but no
hardware vendor implementation.

This produces three outcomes, and the third is the dangerous one:

1. **GPU container, EGL available** — hardware rendering, ~5 ms/frame at 224×224. Correct.
2. **CPU container, no EGL vendor** — `eglInitialize` fails loudly. Annoying but honest.
3. **CPU container with `libegl-mesa0` installed** — Mesa's *software* EGL takes over.
   `MUJOCO_GL=egl` succeeds, no error is raised, and you get llvmpipe software
   rasterization at ~35 ms/frame while believing you have hardware acceleration.

Case 3 is why **checking that EGL initialized is not sufficient verification**. Something
answered; the question is what.

### 17.2 Cost impact, measured honestly

The penalty applies to rendering, not to physics. PhysGround captures 10 frames per scene
but steps physics for hundreds of timesteps, so rendering is a minority of per-scene cost:

| | Physics | Render (10 frames) | Total/scene |
|---|---|---|---|
| Hardware EGL | ~0.30 s | ~0.05 s | ~0.35 s |
| Software (osmesa/llvmpipe) | ~0.30 s | ~0.35 s | ~0.65 s |

End-to-end penalty is therefore **~2×, not ~10×**. Over the full 4,000-scene corpus:
~0.39 core-hours versus ~0.72 core-hours.

At Modal rates — CPU $0.0000131/core-sec, T4 $0.000164/sec — the counterintuitive result
is that **CPU-only with software rendering is cheaper than a T4 with hardware EGL**
(~$0.06 vs ~$0.35 for the corpus), because a T4-second costs ~13× a core-second and the
speedup is only ~2×.

**Therefore: do not attach a GPU to corpus generation for performance reasons.** Both
options cost under a dollar. Choose on convenience, not price. The GPU is needed for
feature extraction (§7) and nothing else.

### 17.3 Required preflight check

Run this before generating any scenes, on every worker, and fail loudly. It checks the
renderer *identity* and the *timing*, because either alone can be fooled.

```python
# scripts/preflight_render.py
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")      # MUST precede `import mujoco`
if os.environ["MUJOCO_GL"] == "osmesa":
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import time
import numpy as np
import mujoco

_MJCF = """
<mujoco>
  <visual><global offwidth="224" offheight="224"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" castshadow="false"/>
    <geom type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <body pos="0 0 0.1"><freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="0.5" rgba="0.9 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

def preflight(warn_ms=15.0, fail_ms=60.0, n=20):
    model = mujoco.MjModel.from_xml_string(_MJCF)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=224, width=224)
    renderer.update_scene(data)
    renderer.render()                                  # warm up; exclude from timing

    t0 = time.perf_counter()
    for _ in range(n):
        renderer.update_scene(data)
        frame = renderer.render()
    ms = (time.perf_counter() - t0) / n * 1000.0

    assert frame.shape == (224, 224, 3), f"bad frame shape {frame.shape}"
    assert frame.std() > 1.0, "frame is blank - renderer produced no geometry"

    backend = os.environ["MUJOCO_GL"]
    gl_renderer = "unknown"
    try:
        from OpenGL import GL
        gl_renderer = GL.glGetString(GL.GL_RENDERER).decode()
    except Exception:
        pass

    software = any(k in gl_renderer.lower() for k in ("llvmpipe", "softpipe", "swrast"))

    print(f"[preflight] MUJOCO_GL={backend}  GL_RENDERER={gl_renderer}  "
          f"{ms:.1f} ms/frame  software={software}")

    if backend == "egl" and (software or ms > warn_ms):
        raise RuntimeError(
            f"MUJOCO_GL=egl but rendering looks like software "
            f"(GL_RENDERER={gl_renderer}, {ms:.1f} ms/frame). "
            "No NVIDIA EGL vendor ICD in this container. Either attach a GPU or "
            "set MUJOCO_GL=osmesa explicitly so the slow path is a deliberate choice."
        )
    if ms > fail_ms:
        raise RuntimeError(f"rendering unusably slow: {ms:.1f} ms/frame")
    return {"backend": backend, "gl_renderer": gl_renderer, "ms_per_frame": ms}

if __name__ == "__main__":
    preflight()
```

**The rule this encodes:** software rendering is acceptable, but only when it is
*chosen*. A silent fallback from `egl` to llvmpipe is always an error, because it
invalidates every wall-clock estimate downstream.

### 17.4 Making software rendering fast

If the preflight shows rendering (not physics) dominating, apply these in order of
payoff. All are science-neutral — none affects the appearance factors in §5.3.

**1. Disable shadows.** Largest single win under software rasterization; shadow mapping
is an extra full-scene pass.

```xml
<visual>
  <global offwidth="224" offheight="224"/>
  <quality shadowsize="0" offsamples="0"/>
</visual>
<worldbody>
  <light pos="0 0 3" dir="0 0 -1" castshadow="false"/>
</worldbody>
```

Setting `offwidth`/`offheight` to exactly the render size matters independently: MuJoCo's
default offscreen framebuffer is 640×480, and rendering above it errors while rendering
far below it wastes memory.

**2. Reuse the renderer.** GL context creation is expensive under osmesa. Construct
`mujoco.Renderer` **once per worker process**, not once per scene. Since geom types vary
per scene you will rebuild `MjModel`, but keep the render context alive.

```python
class SceneRenderer:
    def __init__(self, height=224, width=224):
        self._r, self._hw = None, (height, width)
    def render(self, model, data, camera):
        if self._r is None or self._r.model is not model:
            if self._r is not None:
                self._r.close()
            self._r = mujoco.Renderer(model, *self._hw)
        self._r.update_scene(data, camera=camera)
        return self._r.render()
```

**3. Replace the double-render occlusion pass with ray casting.** §6.5 as written renders
each occluded frame twice to compute the visible-pixel ratio, doubling render cost for a
quarter of the corpus. Ray casting gives the same continuous covariate in microseconds:

```python
def occlusion_fraction(model, data, cam_pos, obj_body_id, obj_geom_ids, n_samples=64,
                       rng=None):
    """Fraction of sampled object-surface points hidden from the camera."""
    rng = rng or np.random.default_rng(0)
    center = data.xpos[obj_body_id]
    radius = float(model.geom_rbound[list(obj_geom_ids)[0]])

    pts = rng.normal(size=(n_samples, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts = center + radius * pts

    geomid = np.zeros(1, dtype=np.int32)
    hidden = 0
    for p in pts:
        vec = p - cam_pos
        dist_to_pt = np.linalg.norm(vec)
        vec = vec / dist_to_pt
        hit_dist = mujoco.mj_ray(model, data, cam_pos, vec, None, 1, -1, geomid)
        if hit_dist >= 0 and geomid[0] not in obj_geom_ids and hit_dist < dist_to_pt - 1e-4:
            hidden += 1
    return hidden / n_samples
```

Validate it once against the double-render method on ~50 frames (Pearson r > 0.95) before
switching. Record which method produced each value in the corpus metadata.

**4. One thread per worker.** MuJoCo's internal threading fights container-level
parallelism. Set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`, then run one process per
core.

### 17.5 Working Modal image

```python
import os
import modal

BASE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1", "libglew-dev",
        "libosmesa6", "libosmesa6-dev",     # software path
        "libegl1", "libgles2",              # EGL loader, used only with a GPU
    )
    .pip_install(
        "mujoco>=3.1", "numpy", "PyOpenGL", "Pillow",
        "torch", "torchvision", "scikit-learn", "scipy",
    )
    .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
)

# CPU corpus generation: software rendering, chosen deliberately (§17.2)
CPU_IMAGE = BASE.env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"})

# GPU feature extraction: hardware EGL, checkpoints baked in so N containers
# do not each re-download them on cold start
GPU_IMAGE = (
    BASE.env({"MUJOCO_GL": "egl"})
    .run_commands(
        "python -c \"import torch; "
        "torch.hub.load('facebookresearch/dinov2','dinov2_vitb14')\""
    )
)

app = modal.App("physground")
vol = modal.Volume.from_name("physground-data", create_if_missing=True)


@app.function(image=CPU_IMAGE, cpu=1.0, memory=4096,
              volumes={"/data": vol}, timeout=3600)
def generate_shard(shard_id: int, n_shards: int, n_scenes: int, master_seed: int):
    from scripts.preflight_render import preflight
    preflight()                                    # §17.3, every worker, every time

    from physground.rollout import generate_scene
    for idx in range(shard_id, n_scenes, n_shards):
        out = f"/data/corpus/base/{idx:05d}"
        if os.path.exists(f"{out}/gt.npz"):         # idempotence, §0.6
            continue
        generate_scene(idx, master_seed, out)       # writes .tmp then os.replace
    vol.commit()                                    # REQUIRED for cross-container visibility


@app.function(image=GPU_IMAGE, gpu="L4", memory=16384,
              volumes={"/data": vol}, timeout=3600)
def extract_shard(encoder: str, shard_id: int, n_shards: int):
    from physground.features import extract
    extract(encoder, shard_id, n_shards, root="/data")
    vol.commit()


@app.local_entrypoint()
def main():
    # ~20 shards, not 100 - see §17.6 on the cold-start tax
    list(generate_shard.map(
        range(20), kwargs={"n_shards": 20, "n_scenes": 4000, "master_seed": 0}))
    for enc in ("dinov2_b", "random_b"):
        list(extract_shard.map(range(4), kwargs={"encoder": enc, "n_shards": 4}))
```

### 17.6 Modal-specific failure modes

**`MUJOCO_GL` set after import.** Setting it in Python after `import mujoco` silently has
no effect. Set it in the image `.env()`, or at the very top of the entry script before any
project import that transitively imports mujoco.

**Missing `vol.commit()`.** Volume writes are not visible to other containers, or to
`modal volume get`, until committed. A function that returns without committing appears
to have done nothing.

**Volume write races.** Parallel containers writing the same path corrupt each other. Use
per-shard paths only, never shared-file appends, and merge in a separate pass.

**Cold-start tax.** Corpus generation is ~25 minutes of aggregate work. Split across 100
containers that is ~15 s of work each behind a ~20 s cold start — you pay more for booting
than computing. Use 10–20 shards.

**Default timeout.** Modal kills functions at the default timeout with an unhelpful error,
exactly like omitting `--time` on SLURM. Set `timeout=` explicitly on every function.

**Memory over-provisioning.** Memory bills at $0.00000222/GiB-sec, and the probe stage is
the project's largest line item because patch tokens need ~16 GB resident. Request 32 GiB
for probes and 4 GiB for generation. Requesting 64 GiB everywhere doubles the biggest cost
in the project for no benefit.

**Cost multipliers.** Region selection bills at 1.5–1.75× and non-preemptible execution at
3×. This pipeline is idempotent and resumable, so neither is needed. Do not set them.

**Interactive sessions.** A `modal shell` left open on a GPU overnight costs more than the
entire pipeline. This is the only realistic way to lose real money on this project.

**Gate G2 needs a human.** The contact sheet (§9.2) must reach your eyes. Write it to the
volume, `vol.commit()`, then `modal volume get physground-data /outputs/gates/sheet.png`.
Do not treat G2 as satisfiable by an automated check.

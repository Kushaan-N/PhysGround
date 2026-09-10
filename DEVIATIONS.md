# Deviations from the spec

Spec §0.1 makes `spec.md` the source of truth and asks that conflicts be flagged
rather than resolved silently. Each entry below records what the spec says, what
the implementation does, and the measurement that motivated the change. Nothing
here was changed on taste.

Entries are ordered by how much they could affect a reported number.

---

## 1. Gate G1 runs at full corpus size, not on the 50-scene pilot

**Spec:** §9.1 hard-fails if `|rho| > 0.10` for any factor pair; §9 runs all four
gates on a 50-scene pilot.

**Problem:** those two instructions are incompatible. Under independence,
Spearman ρ has standard error ≈ 1/√(n−1). At n = 50 that is 0.143, so a
*correctly independent* pair exceeds 0.10 about 48% of the time. Across the ~240
cross-group pairs in this factor table, a passing run would be a miracle. The
gate as literally written measures pilot size, not decorrelation.

**Measured on this factor table:**

| n | max &#124;ρ&#124; over cross-group pairs | significant after Holm | verdict |
|---|---|---|---|
| 50 | 0.446 | 0 | fails on noise alone |
| 3000 | 0.053 | 0 | passes |

Note that the significance test correctly reports zero real couplings at *both*
sizes — it is only the fixed effect-size threshold that misfires.

**Change:** decorrelation is a property of the sampler, not of a particular
draw, and sampling factors costs microseconds per scene with no physics and no
rendering. G1 therefore evaluates the sampler at the full corpus size, where the
0.10 threshold sits at about 5.5σ. Both criteria are applied and either can
fail a pair: the spec's effect-size bound, and an n-aware Holm-corrected
significance bound.

**Where:** `physground/factors.py::check_decorrelation`, `physground/gates.py::gate_g1`.

---

## 2. Scene seeding uses BLAKE2b, not `hash`

**Spec:** §5.4 suggests a per-scene `np.random.default_rng(hash)`.

**Problem:** Python's `hash` on `str` is salted per interpreter by
`PYTHONHASHSEED`. Scene sampling would return different factors in different
worker processes, violating the determinism requirement of §0.5 — and it would
do so silently, since any single run looks self-consistent.

**Change:** streams are derived from `SeedSequence(entropy=master_seed,
spawn_key=(index, blake2b(factor_name)))`, which is stable across processes,
machines, and Python versions. Each factor also gets its own stream, so adding
or reordering a factor leaves every other factor bit-identical and a
half-generated corpus does not become inconsistent with its second half.

**Where:** `physground/factors.py::scene_rng`.

---

## 3. `spawn_height` added to the physical factors

**Spec:** §5.3's factor table does not include it. §6.3 says that if the
scripted push never produces airborne frames, `support_state` is constant and
must be dropped, and §14 lists a constant `support_state` as a failure mode that
"returns a meaningless number that looks like a result".

**Problem:** a scripted *planar* push never lifts the object. Without an
independent source of airborne frames, `support_state` is constant across the
entire corpus and one of the seven probe targets has to be discarded.

**Change:** the target is spawned at a sampled height above its resting pose,
with a point mass at zero so a substantial fraction of scenes still start flat.
The class is then defined by an actual physical difference rather than by a
monotone function of one continuous nuisance. Measured on the pilot,
`support_state` is 84% supported / 16% airborne — imbalanced but not constant,
which is what §9.3 requires.

**Where:** `physground/factors.py::_spawn_height`.

---

## 4. G1's derived-target table is advisory, not a gate

**Spec:** §5.3 lists derived quantities (`obj_pos_x/y`, `obj_speed`,
`ee_obj_dist`, `contact_state`, `support_state`) among the physical factors, and
§9.1 gates on physical × appearance correlations.

**Problem:** most of a derived quantity's dependence on other factors is
definitional. Measured over 1500 scenes:

| pair | ρ | why |
|---|---|---|
| `obj_pos_y` × `approach_angle` | −0.950 | image position *is* where the object is |
| `obj_speed` × `spawn_height` | +0.852 | a dropped object is moving at frame 0 |
| `occlusion_fraction` × `obj_geom_type=sphere` | −0.707 | a sphere presents less area behind a fixed slab |
| `ee_obj_dist` × `obj_size` | +0.486 | the finger stops at the object's surface |

Failing a gate on these would require the corpus to contradict its own geometry.

**Change:** the derived table is computed, written to CSV, and reported for
transparency, but does not set the gate verdict. The gate remains the property
the paper actually depends on: no *sampled* physical factor is predictable from
appearance.

One derived correlation is worth reading rather than dismissing:
`obj_speed` × `friction_slide` = −0.199. Higher friction really does stop the
object sooner. That is physics, and it is precisely the signal H3 predicts a
video encoder can exploit — evidence the corpus is correct, not contaminated.

**Where:** `physground/gates.py::gate_g1`.

---

## 5. The selectivity control shuffles within scenes as well as across them

**Spec:** §8.5 says "labels shuffled **across scenes**, preserving the marginal
distribution".

**Problem:** taken literally as permuting whole label blocks between scenes,
this is a no-op for the corpus's per-frame targets. Frames are captured at fixed
phases (§5.2), so `contact_state` is literally `[0,0,0,1,1,1,1,0,0,0]` in every
scene and `ee_obj_dist` follows the same stereotyped arc. Permuting identical
vectors changes nothing, and the "control" silently re-runs the real task.

**Measured on the pilot with a block-only shuffle:** `contact_state` control
scored 0.869 balanced accuracy (chance 0.5) and `ee_obj_dist` scored R² 0.688
(chance 0).

**Change:** permute across scenes *and* within each scene's block. Across-scene
handles per-scene targets like `mass`, which are constant within a scene and
immune to within-scene shuffling; within-scene handles per-frame targets with
stereotyped profiles. Both preserve the label multiset exactly, and both
preserve scene-level *memorisability*, which is what lets the control detect the
split leak it exists for. After the change every control lands at chance:
regression R² between −0.04 and 0.00, classification balanced accuracy
0.497–0.508.

**Where:** `physground/probes.py::shuffle_labels_across_scenes`.

---

## 6. Occlusion fraction: camera-facing, area-weighted, real-surface ray casting

**Spec:** §6.5 renders twice per frame with segmentation. §17.4 proposes ray
casting against points on the geom's bounding sphere, and asks for Pearson
r > 0.95 against the double render before switching.

**Changes, all three motivated by the two estimators needing to measure the same
quantity:**

1. Points are sampled on the **real geom surface** (face-area-weighted for
   boxes, side/cap weighted for cylinders) rather than on the `geom_rbound`
   sphere, which is up to √3 larger than a box's true extent — so a large share
   of sampled points sit in empty space beside the object, where whether a ray
   is blocked says nothing about visibility.
2. Only **camera-facing** points count. Back-face points are hidden by the
   object itself regardless of the scene, and counting them puts a large
   constant floor under every measurement.
3. Each sample is weighted by **cos(incidence)/distance²**, the image area it
   actually covers. Unweighted samples over-count grazing regions relative to
   the rendered pixel ratio the estimator stands in for.

The reference implementation also changed: the denominator is the target's
*unoccluded silhouette*, obtained by hiding every other geom via geom group,
rather than the scene with only the slab removed. The arm occludes the target
too, and the covariate H4 needs is how much of the object the encoder can see,
whatever is hiding it.

**Validation:** Pearson r = 0.998 over 120 frames spanning both conditions, mean
absolute difference 0.011 — against the required 0.95.

**Where:** `physground/ground_truth.py::occlusion_fraction`.

---

## 7. Frames are packed one archive per scene, not one PNG per frame

**Spec:** §12 lays out `corpus/{condition}/{scene_id}/frames/*.png`.

**Change:** frames go into a single `frames.npz` per scene, as concatenated PNG
bytes plus an offset index. 4,000 files instead of 40,000, on a Modal volume
where feature extraction reads every frame exactly once and fewer, larger reads
are markedly faster. The offset scheme avoids object arrays, so loading never
needs `allow_pickle` and no corpus file can execute code on read. Per-scene
granularity still keeps parallel writers off shared paths (§17.6).

Pooled features likewise go in one archive per shard covering all layers rather
than one file per (encoder, layer, shard). Layers stay individually addressable
as array keys, and `.npz` reads members lazily, so nothing is loaded that is not
asked for.

**Where:** `physground/render.py::pack_frames`, `physground/paths.py::feature_shard`.

---

## 8. Experiment E gained matched-training cells

**Spec:** §10 describes Exp E as "Occlusion: matched pairs, H4".

**Problem:** a probe fitted on clean frames and evaluated on occluded ones
produces a drop, but the drop alone is ambiguous. Two very different things
cause it: the occluder destroyed the information, or the occluder moved the
representation somewhere the clean-trained readout does not point. Only the
first supports H4's claim about the encoder.

**Measured on the pilot:** object position fell from R² 0.880 to 0.124 under
transfer, and a probe *trained* on occluded frames recovered 0.803. The state
was still in the representation; the transfer number was measuring the readout.

**Change:** Exp E runs both families — transfer cells and matched-training cells
— so the two explanations are separable. See `RESULTS_pilot.md`.

---

## 9. The H4 curve is paired, and uses a fixed variance

**Spec:** §15 figure 4 plots contact-state accuracy against occlusion fraction.

**Problems, both found by looking at the output:**

*Unpaired.* Binning the occluded condition alone and normalising to its own
least-occluded bin discards the matched pairing the corpus exists to provide.
With ~17 scenes per bin, the reference carries enough noise to dominate: the
curve wandered between 0.54 and 1.29 with no trend.

*Variance restriction.* Occlusion varies within a scene as the object moves past
the slab, so binning on occlusion also bins on **phase**. A narrow phase window
contains little positional variance, and an R² whose denominator is the bin's
own variance swings for reasons unrelated to occlusion — `obj_pos_y` read −0.80
in the least-occluded bin and *improved* to +0.67 in the most-occluded one.

**Change:** each occluded frame is compared against its own matched base frame,
and the R² denominator is fixed to the full matched set's variance so bins sit
on one scale and only the numerator responds to occlusion.

---

## 10. Gate G1 refuses to run at pilot sample size

Following from §1: `gate_g1` now raises its own sample size to 1,000 scenes when
handed less, and says so. Sampling factors is free and the quantity does not
depend on the corpus, so a caller who passed a pilot size wanted the gate rather
than a noise measurement. This was a live footgun, not a hypothetical —
`make pilot` passed the pilot size through and a 40-scene run failed on
max |ρ| = 0.445.

---

## 11. Frame packing lives outside the renderer

Decoding a PNG needs PIL and NumPy. Keeping `pack_frames`/`unpack_frames` in
`render.py` meant they dragged in `mujoco`, which resolves `MUJOCO_GL` at import
time — so feature extraction and the gate contact sheet, which only read frames
back, required a working GL backend and failed on machines whose valid backends
differ. They now live in `physground/frames.py`, which imports no mujoco;
`render.py` re-exports them.

Relatedly, `MUJOCO_GL` defaults are now platform-aware (`osmesa` on Linux, the
platform's own on macOS, an explicit value always winning). Hard-coding `osmesa`
made every documented command fail on a Mac.

---

## 12. Modules added beyond the spec's file list

`physground/paths.py` (output layout, atomic writes, completion markers),
`physground/render.py` (renderer reuse and frame packing), `physground/gates.py`
(G1–G4 as importable functions), `physground/experiments.py` (Exp A–F),
`physground/summarize.py` (raw arrays → tables and tests),
`physground/frames.py` (PNG packing, mujoco-free),
`physground/latent_dynamics.py` (the optional H5 predictor).

Each exists so the corresponding `scripts/` entry point stays a thin CLI and the
logic underneath is directly testable. No module in §12's list was removed.

---

## Non-deviations worth recording

These looked like they might need a change and did not:

- **Layer set.** `[2, 5, 8, 11]` for the 12-block ViT-B encoders exactly as
  §7.2 specifies. V-JEPA 2 uses `[5, 11, 17, 23]` only because its checkpoint
  has 24 blocks, which was read from its `config.json`.
- **fp16 caching.** Kept, gated by G4 as §3.6 and §9.4 require.
- **Linear probes only** for headline results, with all three baselines on every
  cell.
- **Scene-level splits**, never frame-level, including for the inner CV folds.
- **Cluster bootstrap over scenes** at 2000 iterations; the fast path is
  algebraically identical to literal resampling (verified to 3.3e-16).

# PhysGround — Pre-registration

**Committed:** 2026-09-09
**Master seed:** 0
**Spec version:** v1 (`spec.md`, committed 2026-09-09)

Spec §11 requires the smallest effect size of interest, the layer set, and the
baseline choice to be fixed **before Experiment D is run**, and committed with a
timestamp. This file is that commitment. Anything decided after Experiment D
that is not listed here is exploratory and will be labelled as such in the
paper.

---

## 1. Hypotheses and their predicted direction

| ID | Claim | Predicted | Type of claim |
|---|---|---|---|
| H1 | Geometric/kinematic state (`obj_pos_x`, `obj_pos_y`, `ee_obj_dist`, `support_state`) is linearly decodable well above both baselines at mid-to-late layers | confirmed | presence |
| H2 | `log_mass` and `friction_slide` are **not** linearly decodable from single-frame encoders, i.e. equivalent to the random-init baseline | confirmed (absence) | **absence** |
| H3 | Video encoders recover `friction_slide`, and to a lesser degree `log_mass`, above the single-frame floor | partially confirmed | presence |
| H4 | `contact_state` degrades disproportionately more than `obj_pos_x` under occlusion | confirmed | interaction |
| H5 | Contact-state probe accuracy correlates with latent-dynamics prediction error | unknown | exploratory, optional |

H2 and H4 are the point of the paper. H1 is partly a pipeline sanity check.

---

## 2. Smallest effect size of interest (SESOI)

**ΔR² = 0.05 above the random-init baseline**, and the corresponding
**Δbalanced accuracy = 0.05** for classification targets.

This is the equivalence margin for the TOST in §4. It is fixed now, before any
Experiment D number is seen, precisely because a margin chosen after the fact
can always be set to make a null result look decisive.

Rationale for 0.05: it is the value proposed in spec §11, and it is small
relative to the effects H1 predicts (object position is expected near R² ≈ 0.9)
while being comfortably larger than the bootstrap standard error at the planned
corpus size. An encoder whose advantage over random initialisation is under 0.05
R² is not carrying physical state a dynamics model could use.

---

## 3. Fixed design choices

**Layer set.** Four evenly spaced blocks plus the final block, per spec §7.2.
- DINOv2 ViT-B/14 and random-init ViT-B/14 (12 blocks): `[2, 5, 8, 11]`
- VideoMAE base (12 blocks): `[2, 5, 8, 11]`
- V-JEPA 2 ViT-L (24 blocks): `[5, 11, 17, 23]`

The V-JEPA 2 set differs because the checkpoint has 24 blocks, not 12. This was
read from `facebook/vjepa2-vitl-fpc64-256/config.json`, not assumed.

**Pooled views.** `cls` and `mean` for encoders with a class token. VideoMAE and
V-JEPA 2 have no class token and are probed on `mean` only. A duplicated view
would double their row count in the grid with a perfectly correlated copy.

**Baselines.** All three of spec §8.4 are reported for every cell:
1. `random_b` — the same ViT-B/14 architecture, randomly initialised. This is
   the baseline the SESOI and the TOST are defined against.
2. `raw_pixel` — 32×32 grayscale, flattened to 1024 dims, identical probe.
3. Trivial — train-set mean (R² = 0 by construction) or majority class
   (balanced accuracy = 0.5).

**Probes.** Linear only for headline results. Ridge with
`alphas = np.logspace(-3, 5, 17)` for regression; L2 logistic with
`C = np.logspace(-4, 4, 9)` for classification. Hyperparameters selected by
5-fold **scene-grouped** CV on the training scenes only.

**Splits.** By scene, never by frame. 80/20. Matched occlusion pairs are
assigned to the same side by scene index. 5 probe seeds per reported cell, each
a different scene partition.

**Metrics.** R² for regression; balanced accuracy as the headline classification
metric with AUROC reported alongside.

---

## 4. Statistical procedures

**Uncertainty.** Cluster bootstrap over scenes, 2000 iterations, 2.5/97.5
percentile interval. Frames are never resampled independently.

**Claims of presence (H1, H3).** One-sided bootstrap test that the encoder
exceeds the random-init baseline, paired on scenes. **Holm–Bonferroni** applied
within the family of tests for that property. The family is stated explicitly in
each table caption.

**Claims of absence (H2).** TOST equivalence against the random-init baseline at
the SESOI of §2, on a paired bootstrap of the difference. A non-significant
superiority test is **not** reported as evidence of absence.

**H4.** Paired comparison at matched scene *and* frame index between the base
and occluded conditions, with the probe fitted on unoccluded training scenes
only. The quantity of interest is the *difference in degradation* between
`contact_state` and `obj_pos_x`, not the degradation of either alone.

---

## 5. Kill conditions, declared in advance

| Stage | Condition | Action |
|---|---|---|
| Gates G1, G3, G4 | any fails | fix generation; do not proceed |
| Gate G2 | contact sheet not inspected by a human | do not proceed |
| Exp A | mean R² on `obj_pos_x/y` < 0.90 | pipeline is broken; stop |
| Exp B | any selectivity control above chance | split leakage; stop |

Exp A's threshold applies at the **full corpus size** (3000 base scenes). The
probe's accuracy on object position is still rising with training-set size at
pilot scale — measured 0.709 → 0.769 → 0.800 → 0.817 R² at 60, 120, 180 and 240
training scenes — so evaluating this threshold on a small pilot would reject a
working pipeline for having too little data. The pilot's role is to check that
the number is rising and that nothing is structurally broken.

---

## 6. Declared deviations from the spec

Recorded in `DEVIATIONS.md` with reasoning and, where relevant, the measurement
that motivated them. The three that touch this pre-registration:

1. **G1 is evaluated at full corpus size, not on the 50-scene pilot.** Under
   independence Spearman ρ has SE ≈ 1/√(n−1) = 0.143 at n = 50, so a correctly
   independent pair clears the 0.10 threshold roughly half the time. Measured on
   this factor table: n = 50 gives max |ρ| = 0.446, n = 3000 gives 0.053.
2. **`spawn_height` added to the physical factors.** Without it a scripted
   planar push never lifts the object, `support_state` is constant, and spec
   §6.3 requires dropping the target.
3. **G1's derived-target table is advisory, not a gate.** Derived quantities
   depend on layout by definition (`obj_pos_y` vs `approach_angle`: ρ = −0.950).
   The gate remains: no *sampled* physical factor predictable from appearance.

---

## 7. What is explicitly exploratory

Anything not listed above, including: per-geom-type breakdowns, the spatial
contact probe of spec §8.3 (reported as a follow-up to a null on pooled
features, not as a pre-planned test), any MLP probe, and H5.

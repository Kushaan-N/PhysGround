# Full-corpus results

**Corpus:** 3,000 base + 1,000 occluded scenes (the spec's full size), master
seed 0, generated 2026-09-15 on Unity with the **visual-only occluder**
(`DEVIATIONS.md` §13 — the pilot's slab collided and diverged half the matched
pairs; every number below is from the regenerated corpus where base/occluded
pairs are bit-identical in physics). **Five probe seeds** per reported cell for
Experiments C–F and S, as spec §3.5 requires. Encoders: `dinov2_b`, `random_b`,
`raw_pixel`, `videomae_b`, `vjepa2`.

Regenerate every table and figure from the raw archives with
`python scripts/make_figures.py --seeds 0,1,2,3,4`. Raw archives, tables, and
figures are also copied to `outputs/` on `/work` (durable), since the scratch
workspace expires.

| Hypothesis | Pilot | Full corpus |
|---|---|---|
| H1 geometric/kinematic state decodable | confirmed | **confirmed** |
| H2 mass and friction absent from single frames | confirmed | **confirmed, cleaner** — friction's marginal p=0.049 signal did not survive |
| H3 video encoders recover friction, not mass | confirmed | **confirmed** — V-JEPA 2 R² 0.593 |
| H4 contact degrades most under occlusion | refuted | **refuted again, now unconfounded by physics** |

---

## Gates and kill-switches

| Check | Result |
|---|---|
| G1 decorrelation | pass — max \|ρ\| = 0.053 at n = 3,000, 0 significant after Holm |
| G2 contact sheet | inspected (agent) after occluder fix; occluder placement, occlusion fractions, and contact labels all consistent |
| G3 label sanity | pass — contact rate 0.400, support 85/15, KS p = 0.62 / 0.87 |
| G4 fp16 precision | pass — \|ΔR²\| = 4e-5 against 0.01 tolerance |
| Exp B selectivity | **pass** — every control at chance (regression −0.015…−0.001, classification 0.494/0.500) |
| Exp A positive control | **0.8989 mean R² against the 0.90 threshold** — see below |

**Experiment A** (`obj_pos_x` 0.923, `obj_pos_y` 0.875) missed the
pre-registered kill threshold by 0.0011, with the threshold inside the
bootstrap 95% CI [0.890, 0.907]. Diagnosis before proceeding
(`DEVIATIONS.md` §14): pixel RMSE is *better* on y than x (6.14 vs 6.37 px on
224 px frames); y's lower R² is entirely the camera's foreshortened vertical
variance (0.57× of x) plus airborne frames from the spec-mandated
`spawn_height` factor. The PI instructed the run to continue; the number
stands unedited in the record.

---

## H1 and H2 — main grid (Exp D, dinov2_b L8, 5 seeds, CI = cluster bootstrap)

| target | dinov2_b (mean view) | Δ vs random_b | verdict (Holm / TOST, all 5 seeds) |
|---|---|---|---|
| obj_pos_x | 0.928 [0.912, 0.939] | +0.43 | above baseline |
| obj_pos_y | 0.897 [0.875, 0.912] | +0.39 | above baseline |
| ee_obj_dist | 0.936 [0.930, 0.941] | +0.31 | above baseline |
| obj_speed | 0.254 [0.212, 0.289] | +0.17 | above baseline |
| contact_state | 0.878 [0.866, 0.890] | +0.13 | above baseline |
| support_state | 0.762 [0.730, 0.795] | +0.16 | above baseline |
| **log_mass** | −0.005 [−0.027, 0.012] | +0.001 avg | **TOST-equivalent to baseline, every seed** |
| **friction_slide** | −0.005 [−0.034, 0.011] | +0.003 avg | **TOST-equivalent to baseline, every seed** |

**H1 confirmed** on every geometric/kinematic target, p_holm = 0 throughout.

**H2 confirmed, and more cleanly than the pilot.** The pilot's friction signal
(p = 0.049, Δ = 0.012) does not reappear at the full corpus with the exact
solver: friction is TOST-equivalent to random init on all five seeds
(per-seed Δ between −0.002 and +0.006, all p_holm ≥ 0.63). Mass likewise.
Both sit at R² ≈ 0 absolutely.

---

## H3 — video encoders (Exp F, scene-pooled, 5 seeds)

| encoder | best layer | friction_slide R² | log_mass R² |
|---|---|---|---|
| dinov2_b (frame floor) | 11 | 0.018 | −0.002 |
| random_b (frame floor) | — | ≈ 0 | ≈ 0 |
| videomae_b | 8 | **0.369** [0.273, 0.454] | −0.001 |
| vjepa2 | 17 | **0.593** [0.511, 0.669] | −0.002 |

**H3 confirmed.** Friction is invisible to every single-frame encoder and
strongly recoverable from clips; mass is recovered by nothing, exactly as
Coulomb friction (a = μg, mass-independent) predicts. Both video numbers rose
slightly from the pilot (0.265 → 0.369, 0.573 → 0.593) with the doubled
corpus.

---

## H4 — occlusion (Exp E, matched pairs with identical physics, 5 seeds)

Transfer (probe trained on base, evaluated on the matched occluded frames),
L8 cls:

| target | base → base | base → occluded | retained |
|---|---|---|---|
| contact_state | 0.890 | 0.828 | **0.93** |
| obj_pos_x | 0.921 | 0.735 | 0.80 |
| obj_pos_y | 0.878 | 0.267 | 0.30 |

**H4 refuted again — and this time the occluded condition differs from base
only in what the camera sees.** Contact remains the *most* robust property;
vertical object position collapses. Training within the occluded condition
recovers nearly everything (obj_pos_y 0.267 → 0.862, obj_pos_x → 0.928,
contact → 0.868), so the state survives in the frozen representation and the
transfer loss is a readout/distribution-shift effect, not an information
bottleneck. With the physics confound removed (pilot corpus had the slab
colliding), the two remaining explanations — visibility loss versus
novel-object presence — are what the v2 occluder design (slab in both
conditions, placement varied) would separate. Notably, the transfer drops are
milder than the pilot's across the board (obj_pos_x retained 0.80 vs 0.64),
consistent with part of the pilot's "degradation" having been the
now-removed physical perturbation.

---

## Experiment S — spatial contact probe (patch tokens, seed 0 shown)

| layer | balanced accuracy | AUROC |
|---|---|---|
| L8 | 0.857 [0.849, 0.865] | 0.937 |
| L11 | 0.879 [0.871, 0.886] | 0.952 |

The spatial probe on patch tokens does not exceed the pooled contact probe
(0.890 at L8 cls) — the pooled null-results elsewhere are not an artifact of
pooling. Reported as exploratory per `prereg.md` §7.

---

## What changed relative to the pilot, and why

1. **Occluder made visual-only** (`DEVIATIONS.md` §13) — the pilot slab
   collided; 505/1000 matched pairs had divergent physics. All Exp E numbers
   above are from the clean corpus.
2. **Exp A verdict recorded at 0.8989** (`DEVIATIONS.md` §14), threshold
   inside the CI; proceeded on PI instruction with the diagnosis on record.
3. **Logistic solver: lbfgs → newton-cholesky with a warm-started C path.**
   lbfgs stopped converging between 15k and 30k rows (could not reach
   tol = 1e-4 in 1,000 iterations even at C = 100); newton-cholesky converges
   in 4–8 steps at every C and the identical objective/tolerance means the
   probe definition is unchanged. Measured: a 9-C fold path in 47 s that
   lbfgs could not finish in hours.
4. **`load_patches` rewritten** to preallocate and scatter-write (~24 GB peak
   instead of ~70 GB) — what made Experiment S runnable at full corpus.

## Timings (Unity, cpu partition)

| stage | cost |
|---|---|
| corpus generation | 20-task array, ~30 s/task per condition |
| feature extraction | 48 min on one L4 (all encoders, sequential) |
| Exp C+D+E+F, 5 seeds | 39 min on 21 cores (`--jobs 5`, newton-cholesky) |
| Exp S, 5 seeds | 4 h 45 on 2 cores, 64 GB |
| figures + tests | ~4 min |

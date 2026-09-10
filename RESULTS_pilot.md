# Pilot results

**Corpus:** 1,500 base + 500 occluded scenes (the spec's target is 3,000 + 1,000),
master seed 0, one probe seed. Encoders: `dinov2_b`, `random_b`, `raw_pixel`,
`videomae_b`, `vjepa2`.

These numbers exist to show the pipeline measures what it claims to, and to
record where the spec's predictions held (H1, H2, H3) and where one did not
(H4). They are **not** the paper's results: one probe seed, half the corpus, and
no multiple-seed aggregation.

Regenerate everything here from the raw archives with
`python scripts/make_figures.py --seeds 0`.

---

## Gates

| Gate | Result |
|---|---|
| G1 decorrelation | **pass** — max &#124;ρ&#124; = 0.053 over 241 cross-group pairs at n = 3000, 0 significant after Holm |
| G2 contact sheet | inspected; caught two real defects (see `DEVIATIONS.md` §4, and the README) |
| G3 label sanity | **pass** — contact positive rate 0.400, support 84/16, all targets finite, KS p = 0.62 (mass) and 0.87 (friction) |
| Occlusion estimator | **pass** — ray cast vs. double render, Pearson r = 0.985 over 72 measurements spanning 0.01–0.99 |

## Kill-switches

**Experiment A — positive control.** `obj_pos_x` R² = **0.907**, `obj_pos_y` =
0.868. The spec's threshold is 0.90 and the mean of the two is 0.888, just under
it. The learning curve is still climbing at this corpus size — 0.709, 0.769,
0.800, 0.817 at 60, 120, 180, 240 training scenes — which is why `prereg.md`
declares this threshold as applying at the full 3,000-scene corpus. Not treated
as a pass.

**Experiment B — selectivity control.** Clean pass. Every control lands at
chance:

| target | control | chance |
|---|---|---|
| contact_state | 0.5001 | 0.5 |
| support_state | 0.4986 | 0.5 |
| log_mass | −0.0000 | 0 |
| friction_slide | +0.0103 | 0 |
| obj_pos_x | −0.0116 | 0 |
| obj_pos_y | −0.0171 | 0 |
| obj_speed | −0.0009 | 0 |
| ee_obj_dist | −0.0024 | 0 |

No scene identity leaks across the split.

---

## H1 and H2 — the main grid

`dinov2_b`, best layer/view, against both baselines. Paired bootstrap, TOST
margin 0.05 as pre-registered.

| target | dinov2_b | random_b | raw_pixel | Δ vs random | verdict |
|---|---|---|---|---|---|
| obj_pos_x | 0.905 | 0.466 | 0.733 | **+0.450** | above baseline |
| obj_pos_y | 0.880 | 0.459 | 0.666 | **+0.436** | above baseline |
| ee_obj_dist | 0.923 | 0.612 | 0.781 | **+0.320** | above baseline |
| obj_speed | 0.244 | 0.057 | 0.197 | **+0.190** | above baseline |
| contact_state | 0.891 | 0.727 | 0.808 | **+0.163** | above baseline |
| support_state | 0.759 | 0.592 | 0.717 | **+0.168** | above baseline |
| **friction_slide** | −0.006 | −0.011 | −0.006 | +0.012 | detectable, **below the 0.05 threshold** |
| **log_mass** | −0.010 | −0.011 | −0.016 | +0.001 | **equivalent to baseline** |

**H1 confirmed.** Geometric and kinematic state is linearly recoverable, rising
from block 2 to block 8 and plateauing at 11.

**H2 confirmed.** Mass is statistically equivalent to random initialisation.
Friction is *detectable* at p = 0.049 but its advantage is 0.012 R², a quarter
of the pre-registered smallest effect size — which is what the SESOI exists to
adjudicate. Both sit at R² ≈ 0 in absolute terms.

Worth noting for the write-up: **raw pixels are a strong baseline**, reaching
0.808 balanced accuracy on contact against DINOv2's 0.891, and 0.197 R² on
object speed against 0.244. Spec §8.4 requires this control precisely so that
0.891 is not read as impressive on its own.

---

## H3 — confirmed, and the mass/friction split is exactly right

Dynamic properties, scene-pooled so the frame encoders are scored on the same
number of rows as the video encoders (one per scene), best layer per encoder:

| encoder | kind | `log_mass` R² | `friction_slide` R² |
|---|---|---|---|
| dinov2_b | frame | −0.005 [−0.032, +0.003] | −0.004 [−0.035, +0.005] |
| random_b | frame | −0.007 [−0.034, −0.002] | −0.009 [−0.041, +0.001] |
| videomae_b | video | −0.008 [−0.037, +0.003] | **+0.265** [+0.172, +0.342] |
| vjepa2 | video | −0.004 [−0.031, +0.004] | **+0.573** [+0.493, +0.641] |

Friction is invisible to a single frame and strongly recoverable from a clip —
V-JEPA 2 reaches R² 0.573 at its final block (L23), VideoMAE 0.265 at L8. Mass
is recovered by **nothing**, video or frame.

That split is not a shortfall; it is the physics being reported correctly.
Coulomb friction decelerates a sliding object at *a = μg*, independent of mass,
so the post-contact deceleration profile the video encoders are reading
determines μ and carries no information about m. The spec predicted "friction
recovered more strongly than mass"; the sharper statement the corpus supports is
that friction is recoverable and mass is not observable at all from this
trajectory.

This also confirms the whole video path end to end, including the VideoMAE
attention-bias repair — a lobotomised VideoMAE would not have reached 0.265.

---

## H4 — refuted as stated, and the reason is interesting

**Prediction:** contact state degrades disproportionately more than object
position under occlusion.

**Measured:** the opposite, decisively. Probe trained on unoccluded frames,
evaluated on the matched occluded frames of the same held-out scenes:

| target | base → base | base → occluded | retained |
|---|---|---|---|
| contact_state | 0.905 | 0.836 | **0.83** |
| obj_pos_x | 0.907 | 0.583 | 0.64 |
| ee_obj_dist | 0.923 | 0.340 | 0.37 |
| support_state | 0.751 | 0.595 | 0.38 |
| **obj_pos_y** | 0.845 | 0.079 | **0.09** |

Contact is the *most* robust property, not the least, and object position — the
thing H4 predicts survives — collapses.

### Why: the drop is distribution shift, not lost information

Training the same probe *within* the occluded condition recovers almost
everything:

| target | train base → test base | train base → test occ | train occ → test occ |
|---|---|---|---|
| obj_pos_x | 0.891 | 0.521 | **0.832** |
| obj_pos_y | 0.880 | 0.124 | **0.803** |
| ee_obj_dist | 0.916 | 0.476 | **0.818** |
| contact_state | 0.873 | 0.767 | **0.844** |
| support_state | 0.761 | 0.575 | **0.753** |

Object position goes from 0.124 back to 0.803 when the readout is fitted on
occluded frames. **The state is still in the frozen representation.** What the
transfer number measures is a linear readout pointed at the wrong place, not an
encoder that discarded the information.

The binned curve says the same thing: retained skill does not fall with
occlusion fraction — it *rises* (contact 0.70 → 0.88, object position 0.17 →
0.86 across occlusion bins). If occlusion magnitude were driving the loss, the
curve would slope the other way. What actually happens is that the slab is a
large unfamiliar object present in *every* occluded frame regardless of how much
of the target it covers, and its mere presence moves the representation.

### What this means, and what it does not

It is a real result about a real question — but not the question H4 asked. The
occluded condition as specified confounds two treatments: hiding part of the
target, and adding an object the probe has never seen. Reporting it as
"occlusion destroys contact information" would be wrong.

For a latent world model the finding is still pointed, and arguably sharper than
the original hypothesis: a dynamics model trained on clean observations does not
fail because the encoder stops representing the world, it fails because its
readout is not robust to the scene changing. That is a fixable problem, and a
different claim from an information bottleneck.

**A v2 design would separate the two**, and the cheapest way is to put the
occluder in *both* conditions and vary only where it stands — beside the contact
region versus in front of it — so the distribution is matched and only the
occlusion differs.

This is recorded rather than fixed. Spec §0.2 says not to tune away a predicted
null; the same discipline applies to a predicted confirmation that did not
arrive.

---

## Timings (Apple M3 Pro, one process)

| stage | cost |
|---|---|
| scene generation | 77 ms/scene → 2,000 scenes in 119 s |
| DINOv2 extraction | 20,000 frames in 491 s (MPS) |
| Experiment D, 16 cells | 562 s |
| Experiment E, 8 cells | 71 s |
| ridge grid (6 targets × 17 alphas × 5 folds) | 1.05 s |
| logistic path (9 C values × 5 folds) | 15 s |
| test suite | 80 tests in 8 s |

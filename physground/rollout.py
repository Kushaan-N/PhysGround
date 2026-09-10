"""Episode execution, phase-based frame capture, and per-scene ground truth.

The scripted push (spec 5.2) runs in five stages: rotate at a retracted radius
to the sampled approach angle, extend outward to a standoff, push through the
target at constant speed, pull back in, and rotate home to canonical rest, then
hold while the object settles.

Rotating *before* extending is what keeps the distractor untouched. If the arm
swung out at full extension it would sweep an arc through the workspace and
sometimes clip the distractor, which is supposed to be a never-contacted control
for object presence (spec 5.1). At the retracted radius the swept arc stays
inside the distractor's minimum radius, and the only outward motion is along the
push ray, which the distractor is provably off (see ``factors.py``).

Two passes
----------
Frames are captured at *phase* fractions rather than fixed timesteps (spec 5.2),
and phases are defined by contact onset and release, which are not known until
the episode has run. So pass one steps the physics and records a compact state
trace, frame indices are chosen from the realised contact profile, and pass two
restores state at exactly those steps and renders.

Restoring is not re-simulating: the recorded ``qpos``/``qvel``/``ctrl`` are
written back and ``mj_forward`` recomputes everything derived, including
contacts and their forces. That costs one forward per captured frame instead of
a second full rollout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from . import ground_truth as gt
from . import paths as P
from .factors import sample_factors, scene_id
from .frames import pack_frames
from .render import SceneRenderer
from .scene import ARM, CAMERA, ArmSpec, RENDER_HW, build_mjcf, planar_ik

__all__ = [
    "Schedule",
    "SCHEDULE",
    "PHASE_LABELS",
    "N_FRAMES",
    "trajectory_target",
    "simulate",
    "select_frames",
    "generate_scene",
    "SceneInvalid",
]

N_FRAMES = 10

#: Spec 5.2's phase table, recorded with every frame.
PHASE_LABELS = (
    "rest_pre_approach",
    "approach",
    "approach",
    "first_contact",
    "sustained_contact",
    "sustained_contact",
    "last_contact",
    "post_contact_moving",
    "decelerating",
    "episode_end",
)


class SceneInvalid(RuntimeError):
    """Raised when a scene cannot yield a usable episode (e.g. contact never occurred)."""


@dataclass(frozen=True)
class Schedule:
    """Episode timing.

    The push runs at a fixed *speed* rather than for a fixed duration, so the
    stage's length varies with how far the end-effector has to travel. Episode
    duration therefore differs between scenes, which is exactly what spec 5.2
    intends by capturing at phase fractions instead of fixed timesteps.

    A fixed duration would have made the push speed proportional to the travel
    distance, which is itself a function of object size and push distance. The
    object would then separate faster in scenes with larger objects, coupling
    release speed to an appearance factor -- precisely the kind of leak the
    decorrelation gate exists to prevent, and one it would not have caught
    because it lives in the trajectory rather than in the factor table.
    """
    dt: float = 0.002
    rotate: float = 0.40        # swing to the approach angle at retracted radius
    extend: float = 0.45        # reach outward to the standoff
    push_speed: float = 0.14    # m/s, constant through contact
    push_ramp: float = 0.30     # s of smooth acceleration before constant speed
    pull_in: float = 0.35       # retract radially
    re_home: float = 0.35       # rotate back to canonical rest
    settle: float = 0.70        # hold while the object comes to rest
    standoff: float = 0.05      # clearance ahead of the target before contact


SCHEDULE = Schedule()


# --------------------------------------------------------------------------- #
# Scripted trajectory
# --------------------------------------------------------------------------- #

def _smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _ramped_distance(t: float, speed: float, ramp: float) -> float:
    """Distance covered by a velocity profile that smoothsteps up then holds.

    Integral of ``speed * smoothstep(t/ramp)``. Keeping the ramp a fixed
    *duration* rather than a fraction of the stage means the acceleration phase
    always completes during the standoff approach, so the end-effector is at
    constant speed by the time it reaches the object in every scene.
    """
    if t <= 0.0:
        return 0.0
    if t < ramp:
        u = t / ramp
        return speed * ramp * (u ** 3 - 0.5 * u ** 4)
    return speed * (0.5 * ramp + (t - ramp))


def _push_duration(span: float, sched: Schedule) -> float:
    """Time for the ramped profile to cover ``span``."""
    ramp_distance = sched.push_speed * 0.5 * sched.push_ramp
    if span <= ramp_distance:
        # Short spans never reach constant speed; solve u^3 - u^4/2 = span/(v*ramp).
        target = span / (sched.push_speed * sched.push_ramp)
        u = _solve_ramp_fraction(target)
        return u * sched.push_ramp
    return sched.push_ramp + (span - ramp_distance) / sched.push_speed


def _solve_ramp_fraction(target: float, iterations: int = 40) -> float:
    """Invert ``u**3 - u**4/2 = target`` on [0, 1] by bisection.

    Monotone on the interval, and called a handful of times per scene at most,
    so bisection is both correct and fast enough to not warrant Newton's method.
    """
    lo, hi = 0.0, 1.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if mid ** 3 - 0.5 * mid ** 4 < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class _Waypoints:
    theta: float
    r_start: float
    r_end: float
    t_rotate: float
    t_extend: float
    t_push: float
    t_pull: float
    t_home: float
    t_end: float


def _waypoints(factors: dict, arm: ArmSpec, sched: Schedule) -> _Waypoints:
    theta = float(factors["approach_angle"])
    size = float(factors["obj_size"])
    r_start = float(factors["obj_radius"]) - size - arm.finger_radius - sched.standoff
    r_start = max(r_start, arm.min_reach + 0.02)
    r_end = min(float(factors["obj_radius"]) + float(factors["push_dist"]), arm.reach - 0.03)

    t_rotate = sched.rotate
    t_extend = t_rotate + sched.extend
    t_push = t_extend + _push_duration(max(r_end - r_start, 1e-4), sched)
    t_pull = t_push + sched.pull_in
    t_home = t_pull + sched.re_home
    return _Waypoints(theta, r_start, r_end, t_rotate, t_extend, t_push, t_pull,
                      t_home, t_home + sched.settle)


def trajectory_target(t: float, wp: _Waypoints, arm: ArmSpec, sched: Schedule) -> tuple[float, float]:
    """Commanded ``(radius, angle)`` for the finger axis at time ``t``."""
    if t < wp.t_rotate:
        u = _smoothstep(t / sched.rotate)
        return arm.home_radius, arm.home_angle + u * (wp.theta - arm.home_angle)
    if t < wp.t_extend:
        u = _smoothstep((t - wp.t_rotate) / sched.extend)
        return arm.home_radius + u * (wp.r_start - arm.home_radius), wp.theta
    if t < wp.t_push:
        travelled = _ramped_distance(t - wp.t_extend, sched.push_speed, sched.push_ramp)
        return min(wp.r_start + travelled, wp.r_end), wp.theta
    if t < wp.t_pull:
        u = _smoothstep((t - wp.t_push) / sched.pull_in)
        return wp.r_end + u * (arm.home_radius - wp.r_end), wp.theta
    if t < wp.t_home:
        u = _smoothstep((t - wp.t_pull) / sched.re_home)
        return arm.home_radius, wp.theta + u * (arm.home_angle - wp.theta)
    return arm.home_radius, arm.home_angle


# --------------------------------------------------------------------------- #
# Pass one: physics
# --------------------------------------------------------------------------- #

def simulate(model: mujoco.MjModel, data: mujoco.MjData, factors: dict, ids: gt.SceneIds,
             arm: ArmSpec = ARM, sched: Schedule = SCHEDULE) -> dict[str, np.ndarray]:
    """Run the episode, recording a compact per-step trace.

    Records enough to restore any step exactly (``qpos``, ``qvel``, ``ctrl``)
    plus the quantities needed to choose frames and label them. Contact is
    recorded raw here; debouncing happens once over the whole trace, which is
    both cheaper and more correct than a running window.
    """
    wp = _waypoints(factors, arm, sched)
    n_steps = int(round(wp.t_end / sched.dt))

    qpos = np.empty((n_steps, model.nq), dtype=np.float64)
    qvel = np.empty((n_steps, model.nv), dtype=np.float64)
    ctrl = np.empty((n_steps, model.nu), dtype=np.float64)
    contact_raw = np.zeros(n_steps, dtype=bool)
    support_raw = np.zeros(n_steps, dtype=bool)
    distractor_hit = np.zeros(n_steps, dtype=bool)
    obj_pos = np.empty((n_steps, 3), dtype=np.float64)
    obj_vel = np.empty((n_steps, 3), dtype=np.float64)
    ee_pos = np.empty((n_steps, 3), dtype=np.float64)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    qv = ids.target_qvel
    for step in range(n_steps):
        t = step * sched.dt
        radius, angle = trajectory_target(t, wp, arm, sched)
        q1, q2 = planar_ik(radius, angle, arm)
        data.ctrl[0] = q1
        data.ctrl[1] = q2

        qpos[step] = data.qpos
        qvel[step] = data.qvel
        ctrl[step] = data.ctrl
        contact_raw[step] = gt.body_contact(model, data, ids.finger_geom, ids.target_geoms)
        support_raw[step] = gt.body_contact(model, data, ids.floor_geom, ids.target_geoms)
        if ids.distractor_geoms:
            distractor_hit[step] = gt.body_contact(model, data, ids.finger_geom, ids.distractor_geoms)
        obj_pos[step] = data.xpos[ids.target_body]
        obj_vel[step] = data.qvel[qv:qv + 3]
        ee_pos[step] = data.site_xpos[ids.ee_site]

        mujoco.mj_step(model, data)

    return {
        "qpos": qpos, "qvel": qvel, "ctrl": ctrl,
        "contact": gt.debounce(contact_raw), "contact_raw": contact_raw,
        "support": gt.debounce(support_raw),
        "distractor_hit": distractor_hit,
        "obj_pos": obj_pos, "obj_vel": obj_vel, "ee_pos": ee_pos,
        "speed": np.linalg.norm(obj_vel, axis=1),
        "n_steps": np.int64(n_steps),
        "waypoints": np.array([wp.t_rotate, wp.t_extend, wp.t_push, wp.t_pull, wp.t_home, wp.t_end]),
    }


# --------------------------------------------------------------------------- #
# Frame selection
# --------------------------------------------------------------------------- #

def select_frames(trace: dict[str, np.ndarray], sched: Schedule = SCHEDULE,
                  settle_speed: float = 0.01) -> np.ndarray:
    """Choose the ten capture steps from the realised contact profile (spec 5.2).

    Raises :class:`SceneInvalid` when contact never occurred: a scene with no
    contact has no frames 3-6 and would silently contribute ten negative
    examples to a probe whose positive rate is the thing being reported.
    """
    contact = trace["contact"]
    speed = trace["speed"]
    n = int(trace["n_steps"])

    active = np.flatnonzero(contact)
    if active.size == 0:
        raise SceneInvalid("no debounced finger-target contact in episode")
    first, last = int(active[0]), int(active[-1])

    idx = np.zeros(N_FRAMES, dtype=np.int64)
    idx[0] = min(int(round(0.02 / sched.dt)), n - 1)
    idx[3], idx[6] = first, last
    # Pre-contact: spaced through the approach rather than at fixed times, so
    # the arm is at a comparable *stage* of its reach across scenes whose
    # approaches differ in length.
    # Frame 1 is pinned to a short fixed delay rather than a fraction of the
    # approach. A drop of at most 0.20 m lands within ~0.2 s, while the approach
    # runs 1.5-2.5 s, so a fractional frame 1 always arrives after the object has
    # settled and every airborne frame in the corpus would be frame 0 alone.
    # Sampling early roughly doubles the minority class for support_state, which
    # spec 6.3 requires be non-constant. Frame 2 stays fractional so the pair
    # still spans the approach.
    span_pre = first - idx[0]
    idx[1] = idx[0] + min(int(round(0.09 / sched.dt)), max(int(0.5 * span_pre), 1))
    idx[2] = idx[0] + int(0.75 * span_pre)

    # Frames 4 and 5 are drawn from steps that are *actually* in contact, not
    # from blind thirds of [first, last]. Contact is genuinely intermittent for
    # a rolling sphere, which briefly outruns the pusher, so an interpolated
    # index can land in a real gap and silently make a "sustained contact" frame
    # a negative example. Labels are still read from the trace rather than
    # assumed from the frame slot, so a scene with too few contact steps
    # degrades honestly instead of being mislabelled.
    idx[4] = int(active[active.size // 3])
    idx[5] = int(active[2 * active.size // 3])

    idx[7] = min(last + int(round(0.05 / sched.dt)), n - 1)

    # "Object at rest" is aspirational, not guaranteed: a low-friction sphere
    # can still be rolling when the episode ends. Frame 8 is placed midway to
    # whenever motion actually stops, falling back to the end of the episode,
    # and the realised speed is recorded so analysis can tell the difference.
    tail = speed[idx[7]:]
    stopped = np.flatnonzero(tail < settle_speed)
    settle_step = idx[7] + int(stopped[0]) if stopped.size else n - 1
    idx[9] = n - 1
    idx[8] = (idx[7] + min(settle_step, n - 1)) // 2

    return _make_strictly_increasing(idx, n)


def _make_strictly_increasing(idx: np.ndarray, n: int) -> np.ndarray:
    """Force strictly increasing indices inside ``[0, n)``.

    Short episodes can collapse adjacent choices onto the same step. Nudging
    forward, then backward from the end if that overruns, keeps ten distinct
    frames without silently returning duplicates -- duplicate frames would
    appear as perfectly correlated rows and inflate any probe trained on them.
    """
    idx = np.clip(idx, 0, n - 1)
    for i in range(1, len(idx)):
        if idx[i] <= idx[i - 1]:
            idx[i] = idx[i - 1] + 1
    if idx[-1] > n - 1:
        idx[-1] = n - 1
        for i in range(len(idx) - 2, -1, -1):
            if idx[i] >= idx[i + 1]:
                idx[i] = idx[i + 1] - 1
    if idx[0] < 0:
        raise SceneInvalid(f"episode too short for {len(idx)} distinct frames")
    return idx


# --------------------------------------------------------------------------- #
# Pass two: capture
# --------------------------------------------------------------------------- #

def _capture(model: mujoco.MjModel, data: mujoco.MjData, trace: dict, frames_idx: np.ndarray,
             ids: gt.SceneIds, renderer: SceneRenderer, occlusion_rng: np.random.Generator,
             render_hw: tuple[int, int], occlusion_samples: int) -> tuple[list[np.ndarray], dict]:
    height, width = render_hw
    images: list[np.ndarray] = []
    rows: dict[str, list] = {k: [] for k in
                             ("obj_pos_x", "obj_pos_y", "obj_depth", "obj_in_frame",
                              "ee_pos_x", "ee_pos_y", "occlusion_fraction")}

    for step in frames_idx:
        data.qpos[:] = trace["qpos"][step]
        data.qvel[:] = trace["qvel"][step]
        data.ctrl[:] = trace["ctrl"][step]
        mujoco.mj_forward(model, data)

        u, v, depth, in_frame = gt.project_to_image(
            model, data, ids.camera, data.xpos[ids.target_body], width, height)
        eu, ev, _, _ = gt.project_to_image(
            model, data, ids.camera, data.site_xpos[ids.ee_site], width, height)

        rows["obj_pos_x"].append(u)
        rows["obj_pos_y"].append(v)
        rows["obj_depth"].append(depth)
        rows["obj_in_frame"].append(in_frame)
        rows["ee_pos_x"].append(eu)
        rows["ee_pos_y"].append(ev)
        rows["occlusion_fraction"].append(
            gt.occlusion_fraction(model, data, ids, occlusion_samples, occlusion_rng))

        images.append(renderer.render(model, data, ids.camera))

    return images, rows


# --------------------------------------------------------------------------- #
# Scene generation
# --------------------------------------------------------------------------- #

def generate_scene(index: int, master_seed: int, condition: str, out_dir: Path,
                   renderer: SceneRenderer, *, arm: ArmSpec = ARM, sched: Schedule = SCHEDULE,
                   render_hw: tuple[int, int] = RENDER_HW,
                   occlusion_samples: int = 192) -> dict[str, Any]:
    """Simulate, capture, and persist one scene. Returns its summary record."""
    factors = sample_factors(index, master_seed, condition)
    occluded = condition == "occluded"

    model = mujoco.MjModel.from_xml_string(
        build_mjcf(factors, occluded=occluded, arm=arm, cam=CAMERA, render_hw=render_hw,
                   timestep=sched.dt))
    data = mujoco.MjData(model)
    ids = gt.resolve_ids(model)

    # Spec 6.1: the single most expensive failure in the project. If mass were
    # derived from density it would be a deterministic function of obj_size, the
    # decorrelation would be gone, and nothing would error.
    applied = float(model.body_mass[ids.target_body])
    if abs(applied - float(factors["mass"])) > 1e-6:
        raise AssertionError(
            f"mass not applied for scene {index}: model has {applied}, sampled {factors['mass']}")

    trace = simulate(model, data, factors, ids, arm, sched)
    frames_idx = select_frames(trace, sched)

    occlusion_rng = np.random.default_rng([master_seed, index, 0xC0FFEE])
    images, rows = _capture(model, data, trace, frames_idx, ids, renderer, occlusion_rng,
                            render_hw, occlusion_samples)

    obj_pos = trace["obj_pos"][frames_idx]
    ee_pos = trace["ee_pos"][frames_idx]
    speed = trace["speed"][frames_idx]

    targets = {
        "frame_index": np.arange(N_FRAMES, dtype=np.int64),
        "step_index": frames_idx.astype(np.int64),
        "time": (frames_idx * sched.dt).astype(np.float32),
        "phase": np.array(PHASE_LABELS),
        "contact_state": trace["contact"][frames_idx].astype(np.int8),
        "support_state": trace["support"][frames_idx].astype(np.int8),
        "obj_speed": speed.astype(np.float32),
        "ee_obj_dist": np.linalg.norm(ee_pos - obj_pos, axis=1).astype(np.float32),
        "obj_world_x": obj_pos[:, 0].astype(np.float32),
        "obj_world_y": obj_pos[:, 1].astype(np.float32),
        "obj_world_z": obj_pos[:, 2].astype(np.float32),
        "obj_radius_world": np.linalg.norm(obj_pos[:, :2], axis=1).astype(np.float32),
    }
    for key, values in rows.items():
        dtype = np.int8 if key == "obj_in_frame" else np.float32
        targets[key] = np.asarray(values, dtype=dtype)

    # Per-scene physical factors, broadcast so every array in gt.npz is
    # frame-aligned and the probe stage never has to join two tables.
    for key in ("mass", "friction_slide", "spawn_height", "obj_size"):
        targets[key] = np.full(N_FRAMES, float(factors[key]), dtype=np.float32)
    targets["log_mass"] = np.log(targets["mass"])

    quality = {
        "contact_detected": True,
        "n_contact_frames": int(targets["contact_state"].sum()),
        "n_airborne_frames": int((targets["support_state"] == 0).sum()),
        "distractor_touched": bool(trace["distractor_hit"].any()),
        "all_frames_in_frame": bool(targets["obj_in_frame"].all()),
        "settled_by_end": bool(speed[-1] < 0.01),
        "max_obj_radius": float(trace["obj_pos"][:, :2].__abs__().max()),
        "final_obj_radius": float(np.linalg.norm(obj_pos[-1, :2])),
        "mean_occlusion": float(np.mean(rows["occlusion_fraction"])),
        "n_steps": int(trace["n_steps"]),
        "episode_seconds": float(int(trace["n_steps"]) * sched.dt),
    }

    record = {
        "scene_id": scene_id(condition, index),
        "scene_index": int(index),
        "condition": condition,
        "master_seed": int(master_seed),
        "factors": {k: v for k, v in factors.items()},
        "quality": quality,
        "occlusion_method": "raycast_surface_weighted",
    }

    out_dir = Path(out_dir)
    cfg_hash = P.config_hash({"factors": factors, "schedule": asdict(sched),
                              "render_hw": list(render_hw), "occluded": occluded})

    gt_path = out_dir / "gt.npz"
    P.atomic_write_npz(gt_path, targets, compress=True)
    P.atomic_write_npz(out_dir / "frames.npz", pack_frames(images))
    P.atomic_write_json(out_dir / "factors.json", record)
    P.write_done_marker(gt_path, cfg_hash=cfg_hash, extra={"scene_id": record["scene_id"]})
    return record


def scene_is_complete(out_dir: Path, index: int, master_seed: int, condition: str,
                      sched: Schedule = SCHEDULE, render_hw: tuple[int, int] = RENDER_HW) -> bool:
    """Idempotence check (spec 0.6): is this scene already generated under this config?"""
    factors = sample_factors(index, master_seed, condition)
    cfg_hash = P.config_hash({"factors": factors, "schedule": asdict(sched),
                              "render_hw": list(render_hw), "occluded": condition == "occluded"})
    out_dir = Path(out_dir)
    return P.is_complete(out_dir / "gt.npz", cfg_hash=cfg_hash) and (out_dir / "frames.npz").exists()

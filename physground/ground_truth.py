"""Ground-truth extraction: contact, support, image-plane projection, occlusion.

Everything a probe is asked to decode is computed here, from simulator state
rather than from the scripted command. That distinction matters: the position
actuators have nonzero tracking error under load, so "where the end-effector was
told to go" and "where it went" are different quantities, and only the second is
ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

__all__ = [
    "SceneIds",
    "resolve_ids",
    "contact_force",
    "body_contact",
    "debounce",
    "project_to_image",
    "surface_points",
    "occlusion_fraction",
    "occlusion_fraction_segmentation",
]


# --------------------------------------------------------------------------- #
# Id resolution
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SceneIds:
    """Model indices resolved once per scene rather than per timestep.

    ``mj_name2id`` does a linear scan over the name table. Calling it inside the
    stepping loop would put a string lookup on the hot path for every frame of
    every scene, so it happens once here.
    """
    target_body: int
    target_geoms: frozenset[int]
    finger_geom: int
    floor_geom: int
    ee_site: int
    camera: int
    target_qpos: int
    target_qvel: int
    distractor_body: int | None
    distractor_geoms: frozenset[int]
    occluder_geom: int | None


def _maybe_id(model, objtype, name: str) -> int | None:
    idx = mujoco.mj_name2id(model, objtype, name)
    return None if idx < 0 else idx


def resolve_ids(model: mujoco.MjModel) -> SceneIds:
    obj = mujoco.mjtObj
    target_body = mujoco.mj_name2id(model, obj.mjOBJ_BODY, "target")
    target_joint = mujoco.mj_name2id(model, obj.mjOBJ_JOINT, "target_free")
    distractor_body = _maybe_id(model, obj.mjOBJ_BODY, "distractor")
    return SceneIds(
        target_body=target_body,
        target_geoms=frozenset(_body_geoms(model, target_body)),
        finger_geom=mujoco.mj_name2id(model, obj.mjOBJ_GEOM, "finger"),
        floor_geom=mujoco.mj_name2id(model, obj.mjOBJ_GEOM, "floor"),
        ee_site=mujoco.mj_name2id(model, obj.mjOBJ_SITE, "ee"),
        camera=mujoco.mj_name2id(model, obj.mjOBJ_CAMERA, "main"),
        target_qpos=int(model.jnt_qposadr[target_joint]),
        target_qvel=int(model.jnt_dofadr[target_joint]),
        distractor_body=distractor_body,
        distractor_geoms=frozenset(_body_geoms(model, distractor_body)) if distractor_body is not None else frozenset(),
        occluder_geom=_maybe_id(model, obj.mjOBJ_GEOM, "occluder"),
    )


def _body_geoms(model: mujoco.MjModel, body_id: int) -> list[int]:
    start = int(model.body_geomadr[body_id])
    return list(range(start, start + int(model.body_geomnum[body_id])))


# --------------------------------------------------------------------------- #
# Contact (spec 6.2, 6.3)
# --------------------------------------------------------------------------- #

_FORCE_BUF = np.zeros(6, dtype=np.float64)


def contact_force(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> float:
    """Magnitude of the contact force on contact ``index``."""
    mujoco.mj_contactForce(model, data, index, _FORCE_BUF)
    return float(np.linalg.norm(_FORCE_BUF[:3]))


def body_contact(model: mujoco.MjModel, data: mujoco.MjData,
                 geom_a: int, geoms_b: frozenset[int], force_thresh: float = 1e-3) -> bool:
    """Whether ``geom_a`` bears a force above threshold against any of ``geoms_b``.

    The threshold is not optional. MuJoCo's solver registers contacts that are
    merely within margin and carry no force, so a plain presence test flickers
    on and off around a near-touch. Spec 6.2 is explicit that mislabelled frames
    would read as a weak probe and be misattributed to the encoder.
    """
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 == geom_a and g2 in geoms_b) or (g2 == geom_a and g1 in geoms_b):
            if contact_force(model, data, i) > force_thresh:
                return True
    return False


def debounce(flags: np.ndarray, width: int = 3) -> np.ndarray:
    """Keep a flag only where it holds across a window of ``width`` steps.

    Binary erosion, centred (spec 6.2). Edges replicate rather than being
    treated as False, so a contact that is genuinely active at the very first or
    last recorded step is not erased by the window running off the end.
    """
    flags = np.asarray(flags, dtype=bool)
    if width <= 1 or flags.size == 0:
        return flags.copy()
    pad = width // 2
    padded = np.concatenate([np.repeat(flags[:1], pad), flags, np.repeat(flags[-1:], pad)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return windows.all(axis=1)


# --------------------------------------------------------------------------- #
# Image-plane projection (spec 6.4)
# --------------------------------------------------------------------------- #

def project_to_image(model: mujoco.MjModel, data: mujoco.MjData, cam_id: int,
                     world_pos: np.ndarray, width: int, height: int) -> tuple[float, float, float, bool]:
    """Project a world point to pixel coordinates.

    Returns ``(u, v, depth, in_frame)`` with ``u`` right and ``v`` down, both in
    pixels, origin at the top-left corner.

    A MuJoCo camera looks down its own -Z with +X right and +Y up, and
    ``cam_xmat`` holds that frame's axes as columns in world coordinates, so the
    world-to-camera rotation is its transpose. ``fovy`` is the *vertical* field
    of view; pixels are square, so the same focal length applies to both axes and
    the horizontal field follows from the aspect ratio.
    """
    cam_pos = np.asarray(data.cam_xpos[cam_id], dtype=float)
    rot = np.asarray(data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
    rel = rot.T @ (np.asarray(world_pos, dtype=float) - cam_pos)

    depth = -float(rel[2])
    if depth <= 1e-6:  # behind the camera; no meaningful projection
        return float("nan"), float("nan"), depth, False

    focal = (height / 2.0) / np.tan(np.radians(float(model.cam_fovy[cam_id])) / 2.0)
    u = width / 2.0 + focal * float(rel[0]) / depth
    v = height / 2.0 - focal * float(rel[1]) / depth
    in_frame = bool(0.0 <= u < width and 0.0 <= v < height)
    return u, v, depth, in_frame


# --------------------------------------------------------------------------- #
# Occlusion (spec 6.5, 17.4)
# --------------------------------------------------------------------------- #

def surface_points(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int,
                   n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample points and outward normals on a geom's actual surface.

    Spec 17.4 samples on a sphere of radius ``geom_rbound`` around the centre.
    That bounding sphere is up to sqrt(3) times the true extent of a box, so a
    large share of the sampled points sit in empty space beside the object,
    where whether a ray is blocked says nothing about whether the object is
    visible. Sampling the real surface — face-area-weighted for boxes, side/cap
    weighted for cylinders — makes the estimate track the rendered visible-pixel
    ratio closely enough to substitute for it (validated in
    ``scripts/validate_occlusion.py``).
    """
    gtype = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)

    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        local = rng.normal(size=(n, 3))
        local /= np.linalg.norm(local, axis=1, keepdims=True)
        normals = local.copy()
        local = local * size[0]

    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
        a, b, c = size[:3]
        areas = np.array([b * c, b * c, a * c, a * c, a * b, a * b])
        face = rng.choice(6, size=n, p=areas / areas.sum())
        uv = rng.uniform(-1.0, 1.0, size=(n, 2))
        local = np.empty((n, 3))
        normals = np.zeros((n, 3))
        axis, sign = face // 2, np.where(face % 2 == 0, 1.0, -1.0)
        for k in range(3):
            m = axis == k
            others = [i for i in range(3) if i != k]
            local[m, k] = sign[m] * size[k]
            local[m, others[0]] = uv[m, 0] * size[others[0]]
            local[m, others[1]] = uv[m, 1] * size[others[1]]
            normals[m, k] = sign[m]

    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius, half_h = float(size[0]), float(size[1])
        side_area = 2 * np.pi * radius * 2 * half_h
        cap_area = 2 * np.pi * radius ** 2
        on_side = rng.random(n) < side_area / (side_area + cap_area)
        phi = rng.uniform(0, 2 * np.pi, size=n)
        local = np.empty((n, 3))
        normals = np.zeros((n, 3))
        # Curved side
        local[on_side, 0] = radius * np.cos(phi[on_side])
        local[on_side, 1] = radius * np.sin(phi[on_side])
        local[on_side, 2] = rng.uniform(-half_h, half_h, size=int(on_side.sum()))
        normals[on_side, 0] = np.cos(phi[on_side])
        normals[on_side, 1] = np.sin(phi[on_side])
        # Caps: sqrt for uniform area density on a disc
        cap = ~on_side
        rad = radius * np.sqrt(rng.random(int(cap.sum())))
        sign = np.where(rng.random(int(cap.sum())) < 0.5, 1.0, -1.0)
        local[cap, 0] = rad * np.cos(phi[cap])
        local[cap, 1] = rad * np.sin(phi[cap])
        local[cap, 2] = sign * half_h
        normals[cap, 2] = sign

    else:
        raise ValueError(f"surface sampling not implemented for geom type {gtype}")

    rot = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    origin = np.asarray(data.geom_xpos[geom_id], dtype=float)
    return local @ rot.T + origin, normals @ rot.T


def occlusion_fraction(model: mujoco.MjModel, data: mujoco.MjData, ids: SceneIds,
                       n_samples: int = 96, rng: np.random.Generator | None = None) -> float:
    """Fraction of the target's camera-facing surface hidden by other geometry.

    Replaces the double segmentation render of spec 6.5, as spec 17.4 suggests,
    with two refinements over the sketch given there:

    * Only *camera-facing* points are considered. Points on the far side of the
      object are hidden by the object itself no matter what else is in the
      scene, so counting them would put a large constant floor under every
      measurement and compress the range the H4 analysis depends on.
    * Rays exclude the target's own body via ``bodyexclude`` rather than
      filtering hits afterwards, so a ray that grazes the object's own front
      face cannot be miscounted as an occlusion.

    Returns 0.0 when nothing is occluding, which is the correct value for the
    base condition, and is also what makes the covariate continuous across both
    conditions rather than a binary condition label (spec 5.5).
    """
    rng = rng or np.random.default_rng(0)
    geom_id = min(ids.target_geoms)
    points, normals = surface_points(model, data, geom_id, n_samples, rng)

    cam_pos = np.asarray(data.cam_xpos[ids.camera], dtype=float)
    to_cam = cam_pos[None, :] - points
    distances = np.linalg.norm(to_cam, axis=1)
    to_cam_unit = to_cam / distances[:, None]

    cos_incidence = np.einsum("ij,ij->i", normals, to_cam_unit)
    facing = cos_incidence > 0.0
    if not facing.any():
        return 0.0

    points, distances = points[facing], distances[facing]
    directions = -to_cam_unit[facing]

    # Weight each sample by the image area it actually covers. A unit of surface
    # seen edge-on projects to almost no pixels, so counting samples uniformly
    # over-weights grazing regions and inflates the estimate relative to the
    # rendered pixel ratio it is standing in for. The differential solid angle
    # per unit surface area is cos(incidence)/distance^2.
    weights = cos_incidence[facing] / (distances ** 2)

    hits = _cast_rays(model, data, cam_pos, directions, bodyexclude=ids.target_body)
    hidden = (hits >= 0.0) & (hits < distances - 1e-4)
    return float(weights[hidden].sum() / weights.sum())


def _cast_rays(model: mujoco.MjModel, data: mujoco.MjData, origin: np.ndarray,
               directions: np.ndarray, bodyexclude: int) -> np.ndarray:
    """Distance to the first hit for each ray, -1 where nothing was hit.

    Prefers ``mj_multiRay``, which amortises the broad-phase setup across all
    rays from a shared origin — exactly this case, since every ray starts at the
    camera. Falls back to per-ray ``mj_ray`` if the batched entry point is
    missing, so the module still works on older MuJoCo builds.
    """
    n = directions.shape[0]
    origin = np.ascontiguousarray(origin, dtype=np.float64)
    directions = np.ascontiguousarray(directions, dtype=np.float64)

    multi = getattr(mujoco, "mj_multiRay", None)
    if multi is not None:
        geomid = np.full(n, -1, dtype=np.int32)
        dist = np.zeros(n, dtype=np.float64)
        args = (model, data, origin, directions.reshape(-1), None, 1, int(bodyexclude), geomid, dist)
        # cutoff must be +inf, not -1. mj_multiRay prunes geoms *further than
        # cutoff*, so a negative value prunes the entire scene and every ray
        # reports a miss -- silently, with no error and a plausible-looking
        # occlusion fraction of exactly 0.0 for every frame. Verified against
        # per-ray mj_ray in tests/test_ground_truth.py.
        try:
            # MuJoCo >= 3.1.4 takes an optional per-ray `normal` output before `nray`.
            multi(*args, None, n, np.inf)
        except TypeError:
            multi(*args, n, np.inf)
        # mj_multiRay leaves dist unspecified where geomid is -1.
        return np.where(geomid >= 0, dist, -1.0)

    out = np.empty(n, dtype=np.float64)
    geomid = np.zeros(1, dtype=np.int32)
    for i in range(n):
        out[i] = mujoco.mj_ray(model, data, origin, directions[i], None, 1, int(bodyexclude), geomid)
    return out


def occlusion_fraction_segmentation(renderer, model: mujoco.MjModel, data: mujoco.MjData,
                                    ids: SceneIds) -> float:
    """Reference measurement: 1 - visible target pixels / unoccluded silhouette.

    This is the double-render of spec 6.5, kept only to validate the ray-cast
    estimator (spec 17.4 asks for Pearson r > 0.95 before switching). It renders
    twice per frame and is far too slow for corpus generation.

    Two departures from the sketch in spec 6.5, both needed for the two
    estimators to measure the same quantity:

    * The denominator is the target's *unoccluded silhouette*, obtained by
      hiding every other geom, rather than the scene with only the slab removed.
      The slab is not the only thing in front of the target -- the arm is too --
      and the covariate that matters for H4 is how much of the object the
      encoder can see, whatever is hiding it.
    * Geoms are hidden by geom group rather than by setting the occluder's alpha
      to 0. Segmentation rendering writes a geom id per pixel and never applies
      transparency, so an alpha-0 occluder still registers and the two passes
      come back identical -- a silent no-op that reads as "no occlusion
      anywhere".

    ``renderer`` must have segmentation rendering enabled.
    """
    from .scene import TARGET_GEOM_GROUP

    def _target_pixels(option) -> int:
        renderer.update_scene(data, camera=ids.camera, scene_option=option)
        seg = renderer.render()
        mask = (seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[:, :, 0], list(ids.target_geoms))
        return int(mask.sum())

    full = mujoco.MjvOption()
    visible = _target_pixels(full)

    solo = mujoco.MjvOption()
    solo.geomgroup[:] = 0
    solo.geomgroup[TARGET_GEOM_GROUP] = 1
    silhouette = _target_pixels(solo)

    if silhouette == 0:
        return 0.0
    return float(np.clip(1.0 - visible / silhouette, 0.0, 1.0))

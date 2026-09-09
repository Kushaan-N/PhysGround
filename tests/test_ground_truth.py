"""Ground-truth extraction (spec 6)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from physground import factors as F
from physground import ground_truth as G
from physground import scene as S


def _settled_scene(index: int = 1, occluded: bool = True, steps: int = 400):
    model = mujoco.MjModel.from_xml_string(
        S.build_mjcf(F.sample_factors(index, 0), occluded=occluded))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return model, data, G.resolve_ids(model)


def test_debounce_erodes_flicker_but_keeps_runs():
    assert not G.debounce(np.array([0, 1, 0], bool)).any()
    assert G.debounce(np.ones(5, bool)).all()
    result = G.debounce(np.array([0, 1, 1, 1, 0, 1, 0], bool))
    assert result.tolist() == [False, False, True, False, False, False, False]


def test_debounce_replicates_edges():
    """Contact genuinely active at the first recorded step must survive."""
    assert G.debounce(np.array([1, 1, 1, 0], bool))[0]


def test_multiray_agrees_with_per_ray_mj_ray():
    """mj_multiRay's cutoff prunes geoms FURTHER than cutoff.

    Passing a negative value prunes the entire scene and every ray reports a
    miss -- silently, yielding a perfectly plausible occlusion fraction of 0.000
    on every frame.
    """
    model, data, ids = _settled_scene()
    rng = np.random.default_rng(0)
    points, _ = G.surface_points(model, data, min(ids.target_geoms), 64, rng)
    camera = np.asarray(data.cam_xpos[ids.camera], dtype=float)
    distances = np.linalg.norm(camera - points, axis=1)
    directions = (points - camera) / distances[:, None]

    batched = G._cast_rays(model, data, camera, directions, bodyexclude=ids.target_body)

    geomid = np.zeros(1, dtype=np.int32)
    single = np.array([
        mujoco.mj_ray(model, data, camera, directions[i], None, 1, int(ids.target_body), geomid)
        for i in range(len(directions))])

    assert (batched >= 0).sum() > 0, "batched ray cast found no geometry at all"
    assert np.allclose(batched, single, atol=1e-6)


@pytest.mark.parametrize("geom_type", ["box", "sphere", "cylinder"])
def test_surface_points_lie_on_the_geom(geom_type):
    factors = F.sample_factors(0, 0)
    factors.update(obj_geom_type=geom_type, obj_size=0.05, spawn_height=0.0)
    model = mujoco.MjModel.from_xml_string(S.build_mjcf(factors, occluded=False))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ids = G.resolve_ids(model)
    geom = min(ids.target_geoms)

    points, normals = G.surface_points(model, data, geom, 400, np.random.default_rng(0))
    centre = np.asarray(data.geom_xpos[geom])
    radius = float(model.geom_rbound[geom])

    # Every sample within the bounding sphere, and not collapsed to the centre.
    offsets = np.linalg.norm(points - centre, axis=1)
    assert offsets.max() <= radius + 1e-6
    assert offsets.min() > 0.2 * radius
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)


def test_projection_matches_the_rendered_silhouette():
    """Spec 6.4's positive-control target must actually be where we say it is."""
    model, data, ids = _settled_scene(index=3, occluded=False)
    renderer = mujoco.Renderer(model, 224, 224)
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=ids.camera)
        segmentation = renderer.render()
    finally:
        renderer.close()

    mask = ((segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
            & np.isin(segmentation[:, :, 0], list(ids.target_geoms)))
    if mask.sum() < 60:
        pytest.skip("target too small in this scene to localise reliably")

    ys, xs = np.nonzero(mask)
    u, v, depth, in_frame = G.project_to_image(
        model, data, ids.camera, data.xpos[ids.target_body], 224, 224)
    assert in_frame and depth > 0
    # The projected centre must fall inside the rendered silhouette's bounding
    # box; the centroid itself is biased by which faces are visible.
    assert xs.min() - 2 <= u <= xs.max() + 2
    assert ys.min() - 2 <= v <= ys.max() + 2


def test_projection_reports_points_behind_the_camera():
    model, data, ids = _settled_scene(index=0, occluded=False)
    behind = np.asarray(data.cam_xpos[ids.camera]) + np.array([0.0, 0.0, 5.0])
    u, v, depth, in_frame = G.project_to_image(model, data, ids.camera, behind, 224, 224)
    assert not in_frame


def test_occlusion_is_zero_without_an_occluder():
    model, data, ids = _settled_scene(index=5, occluded=False)
    assert ids.occluder_geom is None
    value = G.occlusion_fraction(model, data, ids, 256, np.random.default_rng(0))
    assert value < 0.05


def test_raycast_occlusion_tracks_the_segmentation_reference():
    """Spec 17.4 requires Pearson r > 0.95 before substituting the ray cast.

    The object is swept along its push ray rather than being read at a single
    settled pose. All scenes evaluated at the same pose give nearly the same
    occlusion, and a correlation over a range of 0.04 would be measuring noise
    while looking like agreement.
    """
    ray, reference = [], []
    for index in range(8):
        factors = F.sample_factors(index, 0)
        factors["spawn_height"] = 0.0
        model = mujoco.MjModel.from_xml_string(S.build_mjcf(factors, occluded=True))
        data = mujoco.MjData(model)
        ids = G.resolve_ids(model)
        angle = factors["approach_angle"]
        renderer = mujoco.Renderer(model, 224, 224)
        try:
            renderer.enable_segmentation_rendering()
            for offset in (-0.05, 0.02, 0.09, 0.16):
                mujoco.mj_resetDataKeyframe(model, data, 0)
                radius = factors["obj_radius"] + offset
                data.qpos[ids.target_qpos] = radius * np.cos(angle)
                data.qpos[ids.target_qpos + 1] = radius * np.sin(angle)
                mujoco.mj_forward(model, data)
                ray.append(G.occlusion_fraction(model, data, ids, 384,
                                                np.random.default_rng(index)))
                reference.append(G.occlusion_fraction_segmentation(renderer, model, data, ids))
        finally:
            renderer.close()

    ray, reference = np.asarray(ray), np.asarray(reference)
    assert reference.std() > 0.05, "reference has no variance; occluder may not be occluding"
    correlation = float(np.corrcoef(ray, reference)[0, 1])
    assert correlation > 0.95, f"ray-cast estimator only reaches r={correlation:.3f}"
    assert np.abs(ray - reference).mean() < 0.10

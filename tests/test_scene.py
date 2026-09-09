"""Scene construction, the mass trap, and friction (spec 5.1, 6.1)."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from physground import factors as F
from physground import ground_truth as G
from physground import scene as S


@pytest.mark.parametrize("index", [0, 1, 2, 7, 13])
@pytest.mark.parametrize("occluded", [False, True])
def test_model_compiles(index, occluded):
    model = mujoco.MjModel.from_xml_string(
        S.build_mjcf(F.sample_factors(index, 0), occluded=occluded))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    assert model.ngeom > 0


@pytest.mark.parametrize("index", range(12))
def test_mass_is_applied_not_derived_from_density(index):
    """Spec 6.1: the single most expensive failure in the project.

    If mass came from density x volume it would be a deterministic function of
    obj_size, decorrelation would be gone, nothing would error, and the renders
    would look perfect.
    """
    factors = F.sample_factors(index, 0)
    model = mujoco.MjModel.from_xml_string(S.build_mjcf(factors, occluded=False))
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    assert abs(float(model.body_mass[body]) - float(factors["mass"])) < 1e-6


def test_mass_is_independent_of_size_across_the_corpus():
    """The consequence that matters, stated directly."""
    table = F.factors_table(range(2000), 0)
    matrix = np.column_stack([table["mass"], table["obj_size"]])
    rho = F.spearman_matrix(matrix)[0, 1]
    assert abs(rho) < 0.10, f"mass tracks obj_size at rho={rho:.3f}"


@pytest.mark.parametrize("geom_type", ["box", "cylinder", "sphere"])
def test_sampled_friction_governs_stopping_distance(geom_type):
    """MuJoCo takes the element-wise MAX of two geoms' friction.

    If the floor kept its default 1.0, max(1.0, f_obj) would be 1.0 for every
    scene and friction_slide -- a probe target -- would have no effect on the
    physics, while every render still looked perfect. Asserted end to end rather
    than by trusting the combination rule.
    """
    distances = []
    for friction in (0.05, 0.4, 1.2):
        factors = F.sample_factors(0, 0)
        factors.update(friction_slide=friction, obj_geom_type=geom_type,
                       obj_size=0.05, spawn_height=0.0, distractor_present=0)
        model = mujoco.MjModel.from_xml_string(S.build_mjcf(factors, occluded=False))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        ids = G.resolve_ids(model)
        angle = factors["approach_angle"]
        data.qvel[ids.target_qvel:ids.target_qvel + 2] = [0.5 * math.cos(angle), 0.5 * math.sin(angle)]
        start = data.xpos[ids.target_body].copy()
        for _ in range(1500):
            mujoco.mj_step(model, data)
        distances.append(float(np.linalg.norm(data.xpos[ids.target_body][:2] - start[:2])))

    assert distances[0] > distances[1] > distances[2], (
        f"{geom_type} stopping distance not monotone in friction: {distances}")
    assert distances[0] > 3 * distances[2], (
        f"{geom_type} friction barely matters: {distances}")


def test_target_is_visible_under_default_render_options():
    """MjvOption defaults to geomgroup [1,1,1,0,0,0].

    A target parked in group 3 would be absent from every frame in the corpus
    while the renders otherwise looked entirely normal.
    """
    assert S.TARGET_GEOM_GROUP <= 2
    option = mujoco.MjvOption()
    assert option.geomgroup[S.TARGET_GEOM_GROUP] == 1


@pytest.mark.parametrize("radius,phi", [(0.09, 0.0), (0.2, -0.5), (0.35, 0.5), (0.5, 0.3)])
def test_ik_round_trips(radius, phi):
    q1, q2 = S.planar_ik(radius, phi)
    x, y = S.forward_kinematics(q1, q2)
    assert abs(math.hypot(x, y) - radius) < 1e-9
    assert abs(math.atan2(y, x) - phi) < 1e-9


def test_ik_stays_inside_the_reachable_annulus():
    """A saturated position actuator buzzes rather than erroring."""
    for radius in (-1.0, 0.0, 10.0):
        q1, q2 = S.planar_ik(radius, 0.0)
        x, y = S.forward_kinematics(q1, q2)
        assert S.ARM.min_reach <= math.hypot(x, y) <= S.ARM.reach


def test_distractor_is_never_on_the_push_ray():
    """It is a control for object presence only if it is never contacted."""
    worst = np.inf
    for index in range(400):
        factors = F.sample_factors(index, 0)
        if not factors["distractor_present"]:
            continue
        position = S.distractor_position(factors)
        angle = factors["approach_angle"]
        ray = np.array([math.cos(angle), math.sin(angle)])
        perpendicular = abs(float(np.cross(ray, position[:2])))
        clearance = perpendicular - factors["distractor_size"] * math.sqrt(2) - S.ARM.finger_radius
        worst = min(worst, clearance)
    assert worst > 0.01, f"distractor comes within {worst:.4f} m of the push ray"

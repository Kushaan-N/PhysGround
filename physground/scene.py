"""Procedural MJCF construction and the arm's planar kinematics (spec 5.1).

One parameterised model: ground plane, 2-DOF planar pusher, a target free body,
an optional distractor, and — in the occlusion condition — a static slab between
the camera and the contact region.

Layout rationale
----------------
The arm links ride at ``LINK_Z`` = 0.24 m, above the tallest possible object
(half-extent 0.09, so a top at 0.18). Only the vertical finger capsule reaches
down into the object layer. This means *the only geom that can ever touch an
object is the finger*, which is what lets ``contact_state`` be a clean
end-effector/target predicate rather than an accident of which link brushed
what.

Friction
--------
MuJoCo combines the friction of two colliding geoms element-wise by taking the
**maximum**. If the floor's sliding friction were left at the default 1.0, then
``max(1.0, f_obj)`` would be 1.0 for every object and the sampled
``friction_slide`` — a probe target — would have no effect on the physics at all,
while every render still looked perfect. Floor and finger therefore get
near-zero friction so the object's sampled value always dominates.
``tests/test_scene.py`` asserts this end-to-end via stopping distance rather
than trusting the combination rule.

All three friction components are scaled by the single sampled coefficient. A
sphere barely slides, so with a *constant* rolling friction the sampled value
would be unrecoverable for a third of the corpus — the exact concern spec 9.1
raises about geom type. Scaling the whole triple keeps the target estimable
within every geom type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .factors import hue_to_rgb

__all__ = [
    "ArmSpec",
    "ARM",
    "CameraSpec",
    "CAMERA",
    "RENDER_HW",
    "planar_ik",
    "forward_kinematics",
    "camera_xyaxes",
    "camera_position",
    "resting_height",
    "object_position",
    "distractor_position",
    "contact_point",
    "occlusion_anchor",
    "occluder_params",
    "build_mjcf",
]

RENDER_HW = (224, 224)


# --------------------------------------------------------------------------- #
# Fixed geometry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ArmSpec:
    l1: float = 0.32          # shoulder link length, m
    l2: float = 0.30          # elbow link length, m
    link_z: float = 0.24      # height of the link plane, m
    link_radius: float = 0.022
    finger_radius: float = 0.012
    finger_bottom: float = 0.006   # world z of the finger's lower tip
    finger_top: float = 0.200      # world z of the finger's upper end
    ee_site_z: float = 0.103       # world z of the end-effector reference site
    home_radius: float = 0.090     # retracted radius, m
    home_angle: float = 0.0        # canonical rest direction, rad

    @property
    def reach(self) -> float:
        return self.l1 + self.l2

    @property
    def min_reach(self) -> float:
        return abs(self.l1 - self.l2)


ARM = ArmSpec()


#: Geom group holding the target, and nothing else. Groups are a pure
#: visualisation attribute in MJCF -- collision is governed by contype and
#: conaffinity -- so isolating the target in its own group lets a renderer draw
#: its unoccluded silhouette without perturbing the physics. That is what makes
#: an exact occlusion reference measurement possible (see
#: ``ground_truth.occlusion_fraction_segmentation``).
#:
#: It must be 2, not some unused group like 3. ``MjvOption`` defaults to
#: geomgroup [1,1,1,0,0,0], so groups 3 and above are hidden unless a caller
#: opts in -- a target placed there would be missing from every rendered frame
#: in the corpus while the renders otherwise looked completely normal.
#: ``tests/test_scene.py`` asserts the target is actually drawn under the
#: default options.
TARGET_GEOM_GROUP = 2


@dataclass(frozen=True)
class CameraSpec:
    """Nominal camera pose. Per-scene jitter is applied on top (spec 5.3).

    The azimuth is not arbitrary. The link plane sits 0.19 m above a resting
    object, and at a low elevation that height difference projects to a large
    parallax offset, so a camera placed across the workspace from the arm sees
    the links draped over the target. Measured over representative poses
    spanning the push, an azimuth of 235 deg hid 38% of the target's
    camera-facing surface *in the nominally unoccluded condition*. That would
    have compressed the occlusion covariate H4 depends on and undercut the
    Experiment A positive control, which requires R^2 > 0.90 on object position.

    Viewing from ~305 deg puts the arm behind the target rather than across it
    and drops that to 2%, with the arm, its base, and the end-effector all still
    in frame at every pose (spec 5.1 requires the arm be visible throughout).
    """
    lookat: tuple[float, float, float] = (0.34, 0.0, 0.07)
    azimuth_deg: float = 305.0
    elevation_deg: float = 34.0
    distance: float = 1.25
    fovy_deg: float = 45.0


CAMERA = CameraSpec()


# --------------------------------------------------------------------------- #
# Kinematics
# --------------------------------------------------------------------------- #

def planar_ik(radius: float, phi: float, arm: ArmSpec = ARM, elbow_up: bool = True) -> tuple[float, float]:
    """Joint angles placing the finger axis at polar ``(radius, phi)``.

    Standard 2-link planar inverse kinematics. ``radius`` is clipped just inside
    the annulus: at full extension the Jacobian is singular and a position
    actuator commanded to an unreachable pose saturates and buzzes, which would
    show up as a jittering arm in the renders rather than as an error.
    """
    r = float(np.clip(radius, arm.min_reach + 1e-3, arm.reach - 1e-3))
    cos_q2 = (r * r - arm.l1 ** 2 - arm.l2 ** 2) / (2.0 * arm.l1 * arm.l2)
    q2 = math.acos(float(np.clip(cos_q2, -1.0, 1.0)))
    if not elbow_up:
        q2 = -q2
    q1 = phi - math.atan2(arm.l2 * math.sin(q2), arm.l1 + arm.l2 * math.cos(q2))
    return q1, q2


def forward_kinematics(q1: float, q2: float, arm: ArmSpec = ARM) -> tuple[float, float]:
    """Finger-axis ``(x, y)`` for a joint configuration. Inverse of :func:`planar_ik`."""
    x = arm.l1 * math.cos(q1) + arm.l2 * math.cos(q1 + q2)
    y = arm.l1 * math.sin(q1) + arm.l2 * math.sin(q1 + q2)
    return x, y


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #

def camera_position(factors: dict, cam: CameraSpec = CAMERA) -> np.ndarray:
    """World camera position including this scene's jitter."""
    az = math.radians(cam.azimuth_deg + float(factors["cam_daz"]))
    el = math.radians(cam.elevation_deg + float(factors["cam_del"]))
    dist = cam.distance * (1.0 + float(factors["cam_ddist"]))
    offset = np.array([
        math.cos(el) * math.cos(az),
        math.cos(el) * math.sin(az),
        math.sin(el),
    ]) * dist
    return np.asarray(cam.lookat, dtype=float) + offset


def camera_xyaxes(cam_pos: Sequence[float], lookat: Sequence[float]) -> np.ndarray:
    """MJCF ``xyaxes`` (camera right and up in world frame) for a look-at pose.

    A MuJoCo camera looks down its own -Z, with +X right and +Y up, so the frame
    is built from the *negated* view direction.
    """
    cam_pos = np.asarray(cam_pos, dtype=float)
    forward = np.asarray(lookat, dtype=float) - cam_pos
    forward /= np.linalg.norm(forward)
    cam_z = -forward
    world_up = np.array([0.0, 0.0, 1.0])
    cam_x = np.cross(world_up, cam_z)
    norm = np.linalg.norm(cam_x)
    if norm < 1e-8:  # looking straight down; any right vector will do
        cam_x = np.array([1.0, 0.0, 0.0])
    else:
        cam_x = cam_x / norm
    cam_y = np.cross(cam_z, cam_x)
    return np.concatenate([cam_x, cam_y])


# --------------------------------------------------------------------------- #
# Object placement
# --------------------------------------------------------------------------- #

def resting_height(size: float, geom_type: str) -> float:
    """Centre height of an object resting on the plane.

    Equal to the half-extent for all three types as parameterised below: a box
    of ``size`` on a side-half, a sphere of radius ``size``, and a cylinder of
    radius and half-height ``size``.
    """
    return float(size)


def object_position(factors: dict) -> np.ndarray:
    """Target spawn position. ``spawn_height`` is added above the resting pose."""
    r = float(factors["obj_radius"])
    theta = float(factors["approach_angle"])
    z = resting_height(float(factors["obj_size"]), factors["obj_geom_type"]) + float(factors["spawn_height"])
    return np.array([r * math.cos(theta), r * math.sin(theta), z])


def distractor_position(factors: dict) -> np.ndarray:
    """Distractor spawn position, offset to one side of the push ray."""
    side = 1.0 if int(factors["distractor_side"]) else -1.0
    angle = float(factors["approach_angle"]) + side * float(factors["distractor_angle"])
    r = float(factors["distractor_radius"])
    z = resting_height(float(factors["distractor_size"]), factors["distractor_geom_type"])
    return np.array([r * math.cos(angle), r * math.sin(angle), z])


def contact_point(factors: dict) -> np.ndarray:
    """Where the finger first touches the target. Predicted from layout, not measured."""
    size = float(factors["obj_size"])
    theta = float(factors["approach_angle"])
    r = float(factors["obj_radius"]) - size - ARM.finger_radius
    return np.array([r * math.cos(theta), r * math.sin(theta), resting_height(size, factors["obj_geom_type"])])


def occlusion_anchor(factors: dict) -> np.ndarray:
    """Midpoint of the contact interface's travel during the push.

    The occluder is anchored here rather than at the first contact point. The
    interface does not stay put: it advances a full ``push_dist`` (0.08-0.18 m)
    while contact is held, which is several object widths. Measured with the
    slab at first contact, occlusion over the contact frames fell from 0.58 at
    first contact to 0.06 at last -- so the frames H4 is actually about were
    barely occluded at all, and the treatment was strongest on frames where
    contact was absent.

    Anchoring at the midpoint and widening the slab to span the traversal keeps
    the contact interface behind it for the whole push.
    """
    size = float(factors["obj_size"])
    theta = float(factors["approach_angle"])
    r = float(factors["obj_radius"]) - size + 0.5 * float(factors["push_dist"])
    return np.array([r * math.cos(theta), r * math.sin(theta), resting_height(size, factors["obj_geom_type"])])


# --------------------------------------------------------------------------- #
# Occluder
# --------------------------------------------------------------------------- #

#: Occluder shape constants, tuned by measuring the realised occlusion fraction
#: at the contact pose over 50 scenes. The pair below gives a mean of ~0.52 with
#: sd ~0.20 and essentially no fully-hidden frames. Fully-hidden frames are the
#: thing to avoid: at occlusion ~1.0 object position stops being decodable too,
#: and H4 rests on the *contrast* between contact collapsing and position
#: surviving. An earlier setting produced a mean of 0.76 with 28% of frames
#: above 0.9, which would have erased that contrast.
OCCLUDER_CUT_FRAC = 0.42       # fraction of the object's height to hide up to
OCCLUDER_SIZE_MULT = 1.25      # half-width contribution per object half-extent
OCCLUDER_TRAVEL_MULT = 0.55    # half-width contribution per unit of push travel
OCCLUDER_STAND_FRAC = 0.15     # position along the anchor -> camera segment


def occluder_params(factors: dict, cam: CameraSpec = CAMERA,
                    cut_frac: float = OCCLUDER_CUT_FRAC,
                    stand_frac: float = OCCLUDER_STAND_FRAC) -> dict:
    """Pose and size of the occluding slab (spec 5.5).

    The slab sits on the floor a quarter of the way from the contact point
    toward the camera, with its face perpendicular to the camera ray. Its top
    edge is placed on the line of sight from the camera to a point ``cut_frac``
    of the way up the target, so it hides the lower part of the object *and* the
    finger tip — the contact interface — while the object's upper body stays
    visible. That asymmetry is the whole point of H4: object position must
    remain decodable while contact does not.

    Being nearer the camera than the object, the slab has to be taller than the
    height it hides; that is what the interpolation below computes. The realised
    occlusion is measured per frame rather than assumed (see
    ``ground_truth.occlusion_fraction``), so these numbers only need to land in
    the right neighbourhood.
    """
    cam_pos = camera_position(factors, cam)
    target = occlusion_anchor(factors)
    size = float(factors["obj_size"])

    z_cut = cut_frac * 2.0 * size          # height on the object to hide up to
    f = float(stand_frac)                  # 0 = at the object, 1 = at the camera

    top_sight = np.array([target[0], target[1], z_cut])
    slab_top = top_sight + f * (cam_pos - top_sight)
    half_height = max(float(slab_top[2]) / 2.0, 0.01)

    centre_xy = target[:2] + f * (cam_pos[:2] - target[:2])

    # Perspective foreshortening: the slab is (1-f) of the way from camera to
    # object, so a given world width there subtends more than the same width at
    # the object. Scale the half-width to keep the covered angular span constant.
    # Wide enough to span the interface's whole traversal, not just the object.
    # A slab sized to the object alone is passed in a fraction of the push.
    half_width = (OCCLUDER_SIZE_MULT * size
                  + OCCLUDER_TRAVEL_MULT * float(factors["push_dist"])) * (1.0 - f)

    ray = cam_pos[:2] - target[:2]
    yaw = math.atan2(ray[1], ray[0]) + math.pi / 2.0   # face normal along the ray

    # Radians throughout: the MJCF compiler is configured with angle="radian",
    # and a degrees-vs-radians mix here would misorient the slab by a factor of
    # 57 without erroring.
    return {
        "pos": (float(centre_xy[0]), float(centre_xy[1]), half_height),
        "size": (float(half_width), 0.008, float(half_height)),
        "euler_rad": (0.0, 0.0, float(yaw)),
    }


# --------------------------------------------------------------------------- #
# MJCF
# --------------------------------------------------------------------------- #

def _geom_size(geom_type: str, size: float) -> str:
    """MJCF size attribute for our three types, all keyed to one half-extent."""
    s = float(size)
    if geom_type == "box":
        return f"{s:.6f} {s:.6f} {s:.6f}"
    if geom_type == "sphere":
        return f"{s:.6f}"
    if geom_type == "cylinder":
        return f"{s:.6f} {s:.6f}"
    raise ValueError(f"unknown geom type {geom_type!r}")


def _rgba(hue: float, alpha: float = 1.0) -> str:
    r, g, b = hue_to_rgb(hue)
    return f"{r:.4f} {g:.4f} {b:.4f} {alpha:.3f}"


def _yaw_quat(yaw: float) -> str:
    half = 0.5 * float(yaw)
    return f"{math.cos(half):.6f} 0 0 {math.sin(half):.6f}"


def build_mjcf(factors: dict, *, occluded: bool, arm: ArmSpec = ARM,
               cam: CameraSpec = CAMERA, render_hw: tuple[int, int] = RENDER_HW,
               timestep: float = 0.002) -> str:
    """Full MJCF for one scene.

    ``occluded`` is the only thing that differs between a matched pair, which is
    what makes the H4 comparison paired: identical mass, friction, colours,
    camera jitter, and scripted trajectory, with and without the slab.
    """
    height, width = render_hw
    obj_pos = object_position(factors)
    obj_size = float(factors["obj_size"])
    obj_type = factors["obj_geom_type"]
    mass = float(factors["mass"])
    fric = float(factors["friction_slide"])

    cam_pos = camera_position(factors, cam)
    xyaxes = camera_xyaxes(cam_pos, cam.lookat)

    gray = float(factors["floor_gray"])
    light = (
        cam.lookat[0] + float(factors["light_x"]),
        cam.lookat[1] + float(factors["light_y"]),
        float(factors["light_z"]),
    )

    # Object friction: one sampled coefficient scaling slide, spin, and roll.
    # See the module docstring for why all three move together.
    obj_friction = f"{fric:.6f} {0.005 * fric:.8f} {0.006 * fric:.8f}"
    # Near-zero on everything else so the element-wise max always yields the
    # object's sampled value.
    inert_friction = "0.000001 0.000001 0.000001"

    finger_lo = arm.finger_bottom - arm.link_z
    finger_hi = arm.finger_top - arm.link_z
    site_z = arm.ee_site_z - arm.link_z

    parts: list[str] = []
    parts.append(f"""<mujoco model="physground">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic"/>

  <!-- offwidth/offheight must equal the render size exactly: MuJoCo's default
       offscreen buffer is 640x480, rendering above it errors and rendering far
       below it wastes memory. shadowsize=0 removes the extra full-scene shadow
       pass, the single largest cost under software rasterisation (spec 17.4). -->
  <visual>
    <global offwidth="{width}" offheight="{height}" fovy="{cam.fovy_deg}"/>
    <quality shadowsize="0" offsamples="0"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/>
    <map znear="0.05" zfar="12"/>
  </visual>

  <default>
    <joint armature="0.01" damping="1.0"/>
    <!-- Soft, well-damped contact. A stiff contact (solref "0.008 1") driven by
         a near-rigid position actuator turns the push into a train of impacts:
         the finger drives in, a large impulse launches the light object, contact
         is lost, the finger catches up. Measured over 10 scenes that gave peak
         contact forces 20x the steady push force mu*m*g and 157 on/off
         transitions per push, which is spec 6.2's contact flicker at a rate no
         3-step debounce can absorb. Softening the contact and lowering the
         actuator gain (see <actuator/>) brings that to 4.4x and 12 transitions. -->
    <geom condim="6" solref="0.03 2" solimp="0.85 0.95 0.03"/>
    <default class="arm">
      <geom rgba="0.30 0.32 0.36 1" friction="{inert_friction}"/>
    </default>
  </default>

  <worldbody>
    <light name="key" pos="{light[0]:.4f} {light[1]:.4f} {light[2]:.4f}"
           dir="0 0 -1" diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1" castshadow="false"/>
    <geom name="floor" type="plane" size="12 12 0.1" pos="0 0 0"
          rgba="{gray:.4f} {gray:.4f} {gray + 0.01:.4f} 1" friction="{inert_friction}" condim="6"/>
    <camera name="main" pos="{cam_pos[0]:.6f} {cam_pos[1]:.6f} {cam_pos[2]:.6f}"
            xyaxes="{' '.join(f'{v:.6f}' for v in xyaxes)}" fovy="{cam.fovy_deg}"/>

    <body name="shoulder" pos="0 0 {arm.link_z}">
      <joint name="q1" type="hinge" axis="0 0 1" range="-2.6 2.6"/>
      <geom class="arm" name="link1" type="capsule"
            fromto="0 0 0 {arm.l1} 0 0" size="{arm.link_radius}" mass="1.5"/>
      <body name="elbow" pos="{arm.l1} 0 0">
        <joint name="q2" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
        <geom class="arm" name="link2" type="capsule"
              fromto="0 0 0 {arm.l2} 0 0" size="{arm.link_radius}" mass="1.0"/>
        <geom class="arm" name="finger" type="capsule"
              fromto="{arm.l2} 0 {finger_lo:.6f} {arm.l2} 0 {finger_hi:.6f}"
              size="{arm.finger_radius}" mass="0.4" rgba="0.85 0.55 0.15 1"/>
        <site name="ee" pos="{arm.l2} 0 {site_z:.6f}" size="0.006" rgba="0 0 0 0"/>
      </body>
    </body>

    <body name="target" pos="{obj_pos[0]:.6f} {obj_pos[1]:.6f} {obj_pos[2]:.6f}"
          quat="{_yaw_quat(float(factors['obj_yaw']))}">
      <freejoint name="target_free"/>
      <!-- mass set explicitly. Leaving it to density would make it a
           deterministic function of obj_size and void the decorrelation the
           entire experiment depends on (spec 6.1). rollout.py asserts that
           model.body_mass came out equal to the sampled value. -->
      <geom name="target_geom" type="{obj_type}" size="{_geom_size(obj_type, obj_size)}"
            mass="{mass:.8f}" friction="{obj_friction}" rgba="{_rgba(float(factors['obj_hue']))}"
            condim="6" group="{TARGET_GEOM_GROUP}"/>
    </body>
""")

    if int(factors["distractor_present"]):
        d_pos = distractor_position(factors)
        d_size = float(factors["distractor_size"])
        d_type = factors["distractor_geom_type"]
        parts.append(f"""    <body name="distractor" pos="{d_pos[0]:.6f} {d_pos[1]:.6f} {d_pos[2]:.6f}">
      <freejoint name="distractor_free"/>
      <geom name="distractor_geom" type="{d_type}" size="{_geom_size(d_type, d_size)}"
            mass="0.5" friction="{inert_friction}" rgba="{_rgba(float(factors['distractor_hue']))}"
            condim="6"/>
    </body>
""")

    if occluded:
        occ = occluder_params(factors, cam)
        parts.append(f"""    <geom name="occluder" type="box"
          pos="{occ['pos'][0]:.6f} {occ['pos'][1]:.6f} {occ['pos'][2]:.6f}"
          size="{occ['size'][0]:.6f} {occ['size'][1]:.6f} {occ['size'][2]:.6f}"
          euler="{occ['euler_rad'][0]:.6f} {occ['euler_rad'][1]:.6f} {occ['euler_rad'][2]:.6f}"
          rgba="0.45 0.45 0.47 1" friction="{inert_friction}"/>
""")

    q1_home, q2_home = planar_ik(arm.home_radius, arm.home_angle, arm)
    parts.append(f"""  </worldbody>

  <actuator>
    <!-- kp=1000, not the 3000 a tracking-error argument alone would suggest. A
         near-rigid finger cannot absorb the contact impulse, so the push
         degenerates into repeated impacts (see the <default> geom comment).
         Measured across kp and contact stiffness, 1000 with the soft contact
         above holds mean radial tracking error to 3.1 mm during the push while
         keeping contact continuous. Ground truth reads the realised site
         position, never the command, so that residual error is measured rather
         than assumed away. -->
    <position name="a_q1" joint="q1" kp="1000" dampratio="1" ctrlrange="-2.6 2.6"/>
    <position name="a_q2" joint="q2" kp="1000" dampratio="1" ctrlrange="-2.9 2.9"/>
  </actuator>

  <keyframe>
    <key name="home" qpos="{_home_qpos(q1_home, q2_home, factors)}" ctrl="{q1_home:.6f} {q2_home:.6f}"/>
  </keyframe>
</mujoco>
""")
    return "".join(parts)


def _home_qpos(q1: float, q2: float, factors: dict) -> str:
    """qpos for the home keyframe: two hinges then each free body's 7-vector."""
    values = [f"{q1:.6f}", f"{q2:.6f}"]
    obj = object_position(factors)
    values += [f"{v:.6f}" for v in obj] + [_yaw_quat(float(factors["obj_yaw"]))]
    if int(factors["distractor_present"]):
        d = distractor_position(factors)
        values += [f"{v:.6f}" for v in d] + ["1 0 0 0"]
    return " ".join(values)

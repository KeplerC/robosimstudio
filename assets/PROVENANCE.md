# Vendored assets

Each directory here is a copy of geometry authored elsewhere, vendored so a task
suite is self-contained and reproducible. This file records where each came
from and what was changed.

Robot descriptions (embodiments) live under `embodiments/`. Geometry that one
task declares lives beside that task, in `<collection>/tasks/<name>/assets/`,
with its own `PROVENANCE.md`; `register_asset` rewrites a record's `files` to
absolute paths at registration. An export that a collection's documents stage
lives with that collection, with its own `PROVENANCE.md`; a document names its
path in `(:staging (:export …))`.

`robosimstudio.paths.resolve_path` searches the checkout's `assets/`, then each
collection's `assets/`, then the copy a wheel force-includes at
`robosimstudio/assets/`. `ROBOSIMSTUDIO_ASSETS` overrides the search outright,
for a checkout that keeps large binaries out of the package.

Files marked *generated* below were written by development scripts that are not
in this repository and checked against their source; the checked-in files are
the source of truth.

---

## `embodiments/yam/` — I2RT YAM bimanual arena

**Source:** the arena's `sim_models/` directory, in a read-only reference
checkout of the Isaac Lab Arena YAM stage
([isaac-sim/IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena),
Apache-2.0, Copyright 2025 NVIDIA Corporation and the Isaac Lab Arena Project
Developers). The arm geometry and meshes in it are i2rt's YAM (MIT). `LICENSE`
beside `yam_bimanual_arms.xml` is the Apache License 2.0 text that covers the
arena-authored files; `i2rt_yam/LICENSE` is i2rt's MIT licence for the meshes.

**Files**

| File | What it is |
| --- | --- |
| `yam_bimanual_arms.xml` | *Generated.* The arena's two YAM arms and nothing else, at their own base origin: the scene as authored (two YAM arms, a gate, cameras, a table, seven free-jointed props) with every free-jointed body and the home keyframe removed, the cell's scenery deleted, and the arm roots moved to the origin. **What the embodiment imports.** |
| `yam_bimanual_arm.urdf` | One arm's kinematic chain (`base_link`, `link_1`..`link_6`, joints `joint1`..`joint6`) transcribed from `yam_bimanual_arms.xml`'s bodies: each body's `pos`/`quat` as the joint origin, each joint's axis and range, in chain order. No geometry or inertia. For the PyRoKi IK solver, which checks it against the MJCF at construction and refuses drift. Named by `yam_bimanual` (`robots/embodiments/yam_bimanual.py`). |
| `i2rt_yam/` | The meshes the MJCF names. Unmodified. Carries its own `LICENSE`. |
| `LICENSE` | Isaac Lab Arena's `LICENSE.md` (Apache License 2.0), under which the arena distributes the files this directory derives from; fetched from `isaac-sim/IsaacLab-Arena` @ `9abd5dd0` on 2026-10-13. |

Not shipped: i2rt's per-arm `yam.urdf` and its meshes under `i2rt_yam/urdf/`
(unused; the IK drift check refuses it, since it lists `joint6` first), and the
arena's 9 MB `base_visual_gate.stl` (unreferenced after stripping).

**Stripped:** the free-jointed props (`bottle_1` … `bottle_6`,
`bin_container`) and the home keyframe, whose `qpos` indexed them; a task
supplies its own objects. The strip deletes only `<body>` elements carrying a
`<freejoint>` and the `<keyframe>` block, and `meshdir` still resolves because
the file sits beside `i2rt_yam/`. `robosimstudio.scene.build.imports.strip_free_bodies`
does the same deletion at runtime; the result is checked in so the import reads
this file.

**Scenery removed:** a light, a floor plane, a gantry mesh with its collision
posts and a camera bar, three wall panels, a visual table slab, and an
invisible `table_plane` collider at z = 0.728 (22 mm below the surface of a
0.750 bench). The arm roots moved from `(0.2525, ±0.31, 0.75)` to the origin.
The ±0.31 shoulder separation is kept; 0.2525 and 0.75 became
`EmbodimentSpec.base_offset` against a scene's work surface.

**Arena constants**, restated in `robosimstudio/robots/embodiments/yam_bimanual.py` rather than
imported from the reference: table top at `z = 0.728 m`, table centred at
`(0.6, 0.0)` with half-extents `(0.30, 0.40)`, per-arm home
`[0, 1.047, 1.047, 0, 0, 0]`, grasp site `0.1347 m` along the wrist's `+z` with
a −90° twist about `z`.

---

## `embodiments/panda/` — Franka Panda: robosuite's model, and a kinematics URDF

### The physics model — robosuite 1.4.0's Panda, as the VAB tool-sort export carries it

robosuite 1.4.0 (MIT) arm and hand, cut from VAB-industrial's `ToolSortPickPlace`
export; mesh geometry from Franka's `franka_description` (Apache-2.0).
`LICENSE` is robosuite's, fetched on 2026-10-04 and byte-identical to the
`v1.4.0` tag's. Source, files, changes and the measurements across the copy:
[`embodiments/panda/README.md`](embodiments/panda/README.md).

### The kinematics URDF — for full-orientation IK

**Source:** `example-robot-data` @ `d0d9098d752014aec3725b07766962acf06c5418`
(`robots/panda_description/urdf/panda.urdf`, via the machine's
`robot_descriptions` cache). BSD 3-Clause, Copyright (C) 2018-2020 LAAS-CNRS,
University of Edinburgh, INRIA: `LICENSE.example-robot-data` is that
repository's `LICENSE` at the same commit, and `NOTICE` credits it.

**Files**

| File | What it is |
| --- | --- |
| `panda_kinematics.urdf` | The URDF with every `<visual>` and `<collision>` element removed — joints, links, limits and inertials only. |

**Stripped** because the PyRoKi IK backend (`robosimstudio.solvers.ik.pyroki`)
reads kinematics only, and a kinematics-only file has no mesh paths to break.
The physics Panda is the MJCF above; this file is never simulated, only solved
against.

**Tool-frame check:** the PyRoKi backend measures `T_hand→tcp =
fk_urdf(q)⁻¹ · T_base⁻¹ · fk_sim(q)` at two configurations at construction and
refuses drift over 1 mm / 0.1°. Measured on the tool-sort scene: ≤0.001 mm,
0.0000°.

**Per-link parity, for a planner.** The tool-frame check is enough for IK,
which needs only the tool point, but not for a planner, which also needs the
elbow. Measured link by link against the tool-sort MJCF at 64 configurations
(seed 0, home, both limit corners, 61 uniform), with a per-link frame
comparator that has since been removed:

| Frame | vs MJCF | Verdict |
| --- | --- | --- |
| `panda_link0` … `panda_link7` | ≤ **0.0016 mm** / ≤ 0.00002° | the same arm, to float32 |
| `panda_hand` | 0.4994 mm along the link's own −z, spread 0.0025 mm | a constant |
| `panda_leftfinger`, `panda_rightfinger` | 6.4989 mm along −z, spread 0.0031 mm | a constant |

The hand and finger offsets are frame-origin conventions: robosuite places the
hand's origin half a millimetre from franka_description's, and inserts a
`gripper0_right_gripper` body between hand and fingers that accounts for the
further 6 mm. The offsets are constant in the link's own axes to five decimal
places over the whole sample, and the orientations agree to 1e-5°. The spread
is recorded beside each magnitude because a small spread is what shows the
offset is constant.

cuRobo's hand and fingertip spheres are authored in franka_description's frames
and must be re-anchored, or inflated to stay enclosing, before they are
trusted. The arm spheres need neither.

**Before pointing a planner at this file:**

* `panda_finger_joint2` is a `<mimic>` of joint 1 (franka_description declares
  two independent prismatics). cuRobo folds a mimic joint out of the
  independent set, so `franka.yml`'s `lock_joints` entry for joint 2 names a
  missing joint; lock joint 1 alone.
* The MJCF mirrors the fingers by sign (`joint_q` holds `+x, −x`), the URDF by
  axis. A parity check must lock the URDF finger where the scene holds it, or
  it reports a phantom 35 mm error.

---

## `embodiments/yam_i2rt/` — i2rt YAM bimanual pair with linear_4310 grippers

**Source:** `i2rt-robotics/i2rt` @ `c40b4125` (read from a mirror of that
repository), files
`i2rt/robot_models/arm/yam/{yam.xml,assets/*.stl}` and
`i2rt/robot_models/gripper/linear_4310/{linear_4310.xml,assets/*.stl}`. MIT
licensed; `LICENSE` copied beside the meshes.

**Files**

| File | What it is |
| --- | --- |
| `yam_i2rt_bimanual.xml` | *Generated.* Two composed YAM+linear_4310 arms, `left_`/`right_` prefixed, bases baked at the reference workcell's mounts. |
| `yam_i2rt_arms.xml` | *Generated.* The same two arms with the baked mount taken out: the arm roots moved from `(0.2525, ±0.31, 0.75)` to the origin, the ±0.31 shoulder separation kept, 0.2525 and 0.75 moved to `EmbodimentSpec.base_offset`. **What `yam_i2rt` imports.** |
| `assets/` | The 9 STLs the MJCF references. Byte-identical copies. |
| `LICENSE` | i2rt's MIT license. |
| `yam_i2rt_arm.urdf` | *Generated.* One arm's chain (`base_link`, `link_1`..`link_6`, `joint1`..`joint6`) transcribed from `yam_i2rt_arms.xml`'s `right_` bodies: each body's `pos`/`quat` as the joint origin, each joint's own axis and range, in chain order; no geometry or inertia. Used by PyRoKi IK, which refuses i2rt's stock `yam.urdf` (not shipped; it lists `joint6` first and puts `joint1` at 63.1 mm where the MJCF has 67, and the check read 268.58 mm / 10.524° tool-to-TCP drift between two configurations on the crate-washing scene, measured 2026-10-03) and accepts this one on both arms; its drift check holds this file to the MJCF. Named by `yam_i2rt`; `yam_i2rt_arena_narrow` has the same chain but names no URDF (see `robots/embodiments/yam_i2rt_arena.py`). |

**What changed on the way in** (all in the generator, none in the meshes):

* The single-arm composition replicates the YAM builder in the original reference sims
  (itself a re-implementation of i2rt's `combine_arm_and_gripper_xml()`):
  fixed `base` wrap, linear_4310 mounted on `link6` with the transform from
  i2rt's `linear_4310.yml`, `joint6` axis overridden to `0 0 -1`.
* Actuator physics the stock MJCF omits, from that builder verbatim:
  reflected armature `N²·I_rotor`, damping, frictionloss and true DM4340/DM4310
  torque limits per joint; position actuators kp=300/kv=30 (arms) and
  kp=600/kv=20 (fingers).
* `gravcomp="1"` baked onto every body — the sims set `mujoco:gravcomp = 1.0`
  on every robot body in code; baking it lets the importer harvest it.
* **Both fingers actuated and the stock `joint7 == joint8` equality dropped**:
  Newton's MJCF import applies that equality with inverted polarity
  (`joint8 = -joint7`), which translates the jaw pair sideways instead of
  closing it. The original reference sims carry the same fix
  (`skip_equality_constraints=True` plus a second actuator); both fingers are
  always commanded to the same value, so the coupling is preserved by
  construction.
* Two prefixed copies merged into one worldbody at bases
  `(0.2525, ∓0.31, 0.75)` (right = active), per-side contact excludes kept.

---

## `embodiments/yam_i2rt_arena/` — the composed i2rt YAM wearing the arena's hand

**Source:** the two directories above. `yam_i2rt_arena_narrow_arms.xml`
(*generated*) is `yam_i2rt/`'s links, meshes, inertias and joint physics with
the arena's hand grafted onto the wrist unchanged: the `{side}_link_6` subtree
of `yam/yam_bimanual_arms.xml` — wrist inertial, palm capsules, D405 body and
its collider, both finger linkages with their `<equality>` coupling — hung
under the composed `{side}_link6` as a fixed child, rotated by `quat 0 -1 1 0`
and not translated. Shoulders at ±0.17 m. `assets/` holds the STLs the file
names; `LICENSE` is i2rt's MIT licence, as for `yam_i2rt/`; the grafted hand
comes from Isaac Lab Arena (Apache-2.0, see `embodiments/yam/` above and
`NOTICE`). Placement rationale and measurements:
`robosimstudio/robots/embodiments/yam_i2rt_arena.py`.

---

## `embodiments/robotiq_2f85/` — Robotiq 2F-85, from MuJoCo Menagerie

**Source:** `google-deepmind/mujoco_menagerie` @ `affef083`, folder
`robotiq_2f85`. The commit is Newton's own `MENAGERIE_REF`, read off
`newton._src.utils.download_assets`, and fetched by a development script not in
this repository. The model is **BSD 2-Clause**: `LICENSE` is the file
Menagerie ships beside it, "Copyright (c) 2013, ROS-Industrial", and the
vendored `README.md` says the same.

**Files:** Menagerie's bytes, unmodified (`2f85.xml`, `assets/*.stl`,
`LICENSE`), and its `README.md` with the two links to files not vendored here
(`CHANGELOG.md`, `2f85.png`) pointed at Menagerie.

**At build**, `EntityEmbodiment._attach_hands` imports it with `parent_body` set
to the arm's flange, switches its MuJoCo actuators to `JOINT_TARGET` (a live
tendon actuator fights a joint target; Newton's composer test uses the same
setting), and puts `right_driver_joint` / `left_driver_joint` in position mode.
Nothing on disk changes.

**Measured** on a tool-sorting scene, welded to a UR5e, closing under physics:
pad centres 0.09793 m apart at full open and 0.01252 m at full closed, i.e.
**0.0854 m of travel against the published 0.085 m stroke**. Closing axis
tool-local ±x, a right angle from the `GripperSpec` default. `housing_half`
measures 0.0424 here against the Isaac stage export's 0.0425.

---

## `embodiments/suction/` — a single-cup vacuum hand, authored here

**Source:** none; `suction_cup.xml` was written for this repository and is
Apache-2.0 like the code. Four cylinders: an 18 mm flange and a 6 mm stem on
`suction_mount`, an 11 mm bellows and a 15 mm face on `suction_pad`, whose
origin is the face, 60 mm from the mount along +z. Masses are stated per geom,
70 g in all.

**At build**, `EntityEmbodiment._attach_hands` imports it with `parent_body`
set to the arm's wrist at the arm's `ArmSpec.gripper_mount`, and
`after_scene_built` adds one disabled MuJoCo weld row from `suction_pad` to
each part with a flat face; `robosimstudio.core.suction` switches them.
Nothing on disk changes.

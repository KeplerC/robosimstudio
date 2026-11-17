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

## `embodiments/yam/` — I2RT YAM bimanual arena: the one YAM, `yam`

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
| `yam_bimanual_arm.urdf` | One arm's kinematic chain (`base_link`, `link_1`..`link_6`, joints `joint1`..`joint6`) transcribed from `yam_bimanual_arms.xml`'s bodies: each body's `pos`/`quat` as the joint origin, each joint's axis and range, in chain order. No geometry or inertia. For the PyRoKi IK solver, which checks it against the MJCF at construction and refuses drift. Named by `yam` (`robots/embodiments/yam.py`). |
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

**What changed on the way in** -- the hand, edited in the checked-in file on
2026-10-26 so it grips like the robot it models (`i2rt-robotics/i2rt` @
`120c3c8`, v1.3.6). Measured on a top-down grasp of a free box (mu 0.3) with the
right hand, the box in the arm's MuJoCo island (`MUJOCO_500HZ`):

* **Collider friction 1.0**, from 2.0 on the `collision` class and 4.0 on the
  0.6 mm pad nubs (`sphere_collision`). MuJoCo gives a contact the larger of its
  two geoms' frictions, so every finger contact ran at 2 to 4 whatever the
  object said: the mu 0.3 box read `[2.0, 4.0]` in `mjw_data.contact.friction`,
  and reads `[1.0]` now. 1.0 is what i2rt's own `linear_4310.xml` leaves on its
  tips (MuJoCo's default).
* **The finger equality is stiff** (`solref="0.004 1" solimp="0.95 0.99 0.001"`,
  from MuJoCo's default 0.02). At the default the carried finger gave 4 mm under
  a 24 N squeeze, so the hand pinched off centre and lost force to the coupling;
  the box now sits within 0.2 mm of the grasp point.
* **The finger actuator's `forcerange="-100 100"`** (it had none): the driver's
  limit. i2rt runs the real arm with `limit_gripper_force=50.0`
  (`i2rt/robots/get_robot.py`), and its `GripperForceLimiter`
  (`i2rt/robots/utils.py`) sees a blocked jaw and holds the motor at the torque
  that makes 50 N of grip, whatever the width. The one actuator drives both
  fingers through the equality, which splits its force onto the two jaws, so
  100 N is 50 N a jaw.

**The finger gains are not the file's.** The file keeps the arena's, kp 200 N/m,
kv 14: a spring whose squeeze grew with the object, 0.3 N a jaw on a 6 mm box,
1.5 N on 30 mm and 2.5 N on 50 mm. The library drives the hand with its spec's
own servo instead (`yam_linear_arena.CLAMP`, a `GripperSpec.actuation`), so a
page that states its own servo replaces it and gets this file's joint physics
back, under the same cap:

* **A stiff servo**, kp 300000, kv 300, the real one's stiffness in kind
  (`i2rt/robots/config/linear_4310.yml`: a DM4310 at kp 20, 6.57 rad of motor per
  0.096 m of jaw, so about 94 kN/m at the jaw before the limiter): the cap is
  reached 0.33 mm past contact, and 50 N a jaw measured on boxes of 1, 2, 6, 30
  and 60 mm.
* **230 N*s/m of damping on each finger joint.** It sets the closing speed: a
  clamp this stiff and undamped slams the pads, and at kp 200 kN/m they crossed
  56 mm through each other shut on nothing and went through a 1-2 mm plate. The
  equality ties the two joints, so the pair carries 460: pulling at the 100 N
  cap a finger runs at 100/460 = 0.22 m/s and the jaw at 0.43 m/s, the DM4310 at
  30 rad/s (the velocity range i2rt's motor table gives it) through 6.57 rad per
  96 mm. That is an upper bound, since no closing time is published for the
  hardware: shut on nothing from full open takes 0.20 s to 90 % at 50, 60 and
  240 Hz control (0.16 s on the spring). On both joints rather than the driven
  one: with the carried finger left at 0.5 it rang against the stiff equality,
  flung a held box under `GENERIC` and crossed a 6 mm plate at 240 Hz.

After: every box from 1 to 60 mm held to the 2 kg the probe stops at (the YAM's
payload; the arm itself sags 28 mm at 2 kg), where the arena hand dropped the
6 mm box at 0.29 kg and the 30 mm one at 1.31 kg on its mu 2 to 4. Under
`GENERIC`, the box on VBD behind the proxy coupler (the finger contacts resolved
by VBD, which mixes friction as the geometric mean), the 6, 30 and 60 mm boxes
lifted and held to 1.9, 1.5 and 1.8 kg, where the arena hand never lifted the
6 mm box and dropped the 30 mm one at 0.21 kg. Two limits: a 1-2 mm plate
standing on edge is not reached under `GENERIC` by either hand (the tool centre
stops 11 mm high over it), and at 240 Hz (`rigid_mujoco@v1`'s rate) with the
plate in MuJoCo the pads cross it in two steps.

The arm's gains stay the arena's, kp 80 kv 6 on joints 1-3 and kp 30 kv 2 on
the wrist, though i2rt's `yam_v1.yml` runs kp 80/80/80/10/10/10 and kd
5/5/5/1.5/1.5/1.5. Decided by measurement: with i2rt's gains in the classes,
the medium tier lost six shipped tasks that pass on the arena's -- the towel
example's edge slipped out of the jaws (0 of 10 particles held), the
fold-the-towel demo scored 0.74 against 0.85 (its near half dragged 27 mm), a
scrub-a-stain stroke inside its force band wiped 0.56 of the stain against 0.8,
the suction hand stopped 7.5 mm short of its plate, the paper fold kept
2.54 rad against 2.8 and the stuck label bonded 0.07 against 0.8 -- and the arm
sagged 32 mm, not 28, under the 2 kg probe box. The routines were written on
the stiffer wrist; until they are re-planned, it stays.

**Arena constants**, restated in `robosimstudio/robots/embodiments/yam.py` rather than
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

## Kept on the Hugging Face Hub, not in git

These used to be vendored here. Their files, with this file's old entries for
them, are in the dataset `keplerccc/robosimstudio-agentic-real2sim`:

| Dataset folder | What it is | Why it left git |
|---|---|---|
| `robots/robotiq_2f85/` | Robotiq 2F-85, from MuJoCo Menagerie (BSD-2-Clause); the hand `panda_robotiq` wears | No example uses it. `resolve_path` finds a fetched copy under `~/.cache/robosimstudio/hub/<revision>/`, and until then names the command: `python -m robosimstudio.hub fetch robots/robotiq_2f85 --dest <that folder>`. |
| `robots/archive/yam_i2rt/` | The composed i2rt YAM with linear_4310 blade-tip grippers (MIT), and its spec modules | Retired: RoboSimStudio has one YAM, `yam` (above). |
| `robots/archive/yam_i2rt_arena/` | The composed i2rt YAM wearing the arena's hand, and its spec module | Retired with it. |

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

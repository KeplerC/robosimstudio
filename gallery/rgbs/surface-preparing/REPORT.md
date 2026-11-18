# Task 5: Surface Preparing

**Changed for the gallery after review (render only, no physics):** the added marker stripe is the clip's mid blue (0.12, 0.32, 0.85) rather than navy, and `photo` renders at the calibration's own 1920 x 1080 (same intrinsics, twice the pixels), so the wipe can be seen on film. The takes in the gallery were filmed on this checkout after those two changes.

The revised contact routine is filmed through the held, calibrated [photo camera](run/hero.mp4) and the re-aimed [wide camera](run/hero_wide.mp4), using the page's RTX path tracer. Each take records 30.6 s of routine time, with 307 samples at 10 fps including both endpoints. The sponge now bends over the rail, wipes in five strokes, and returns released to the mat. The left hand reaches and holds the left elbow. A procedural blue marker stripe is **added**, then removed by measured contact exposure.

The current page's **check and physics suite pass**. The 1.08 px photo calibration is unchanged. I read the final [photo strip](run/hero_strip.png) and [wide strip](run/hero_wide_strip.png) beside the clip and inspected the full-size frames linked below. These measurements supersede round 2 and the intermediate fitting trials in `run/round3/`.

## Results beside the frames

| Measurement | Photo hero | Wide hero |
|---|---:|---:|
| Pressed coverage of the 340 mm straight run | 100.0% | 100.0% |
| Bar drift | 0.00 mm | 0.00 mm |
| First crown press | 11.4 s | 11.4 s |
| First ≥90% coverage | 22.4 s | 22.4 s |
| Mean crown force in loaded-stroke phases | 3.48 N | 3.47 N |
| Median loaded-stroke arc, nonzero samples | 88.9° | 89.9° |
| Peak sampled crown force | 5.40 N | 5.46 N |
| Fingers clear of sponge | 26.6 s | 26.6 s |
| Final sponge-to-foot gap | 45.5 mm | 45.2 mm |
| Final long-axis angle from across-bar direction | 3.7° | 3.8° |
| Final bottom above mat | 0.83 mm | 0.83 mm |
| Final finger contact force | 0.000 N | 0.000 N |
| Final centre speed | 1.5e-06 m/s | 8.3e-07 m/s |

Sources: [photo result](run/hero_result.json), [trace](run/hero_trace.json), [summary](run/hero_summary.json), [bin exposures](run/hero_exposure.json); [wide result](run/hero_wide_result.json), [trace](run/hero_wide_trace.json), [summary](run/hero_wide_summary.json), [bin exposures](run/hero_wide_exposure.json). Source hashes for the films are [photo](run/hero_sources.json) and [wide](run/hero_wide_sources.json).

- [Initial photo, 0 s](run/hero_frames/hero_00.0.png): both feet, the blue crown stripe and the scalloped sponge outside the right foot.
- [Left hand, 5 s](run/hero_frames/hero_05.0.png): stock fingers at the left elbow. The bar is an anchored fixture; this hand carries nothing. It stays there through the wipe and opens before the home motion.
- [Grasp, 8 s](run/hero_frames/hero_08.0.png) and [carry, 10 s](run/hero_frames/hero_10.0.png): the right fingers pinch near the middle of the sponge's length. Foam protrudes on both sides of the jaws along the long, across-bar direction.
- Loaded wrap: [13 s](run/hero_frames/hero_13.0.png), [15.4 s](run/hero_frames/hero_15.4.png), [18 s](run/hero_frames/hero_18.0.png), [23 s](run/hero_frames/hero_23.0.png). The near edge hangs below the rail's widest point. The stripe progressively clears from right to left.
- [16 s](run/hero_frames/hero_16.0.png) is deliberately **in the air**, matching `ref/t016.00.jpg`. It cannot simultaneously be a loaded-wrap frame; 15.4 s supplies the adjacent loaded view. At 16 s the photo probe reads 0.000 N on the bar, 0 pressed samples, and 34.9 mm clearance above the crown. Wide reads 0.000 N and 35.1 mm clearance.
- [Put-down, 26.8 s](run/hero_frames/hero_26.8.png) and [settled, 30 s](run/hero_frames/hero_30.0.png): foam flat on the mat, fingers free, long scalloped edges toward the camera. The small bottom clearance is soft-contact padding. The bar remains fixed; zero drift is not a free-standing stability claim.
- Wide checks: [grasp](run/hero_wide_frames/hero_wide_08.0.png), [loaded stroke](run/hero_wide_frames/hero_wide_13.0.png), [lift](run/hero_wide_frames/hero_wide_16.0.png), [left-end stroke](run/hero_wide_frames/hero_wide_23.0.png), [put-down](run/hero_wide_frames/hero_wide_26.8.png). Both arms and the bar are in view, with the foam visible below the fingers.

## Foam fit and grasp

Counting the 25 mm grid in `ref/t028.00.jpg` against the 84 mm foot face gives approximately five squares along the long edge, 2.5 squares across the near edge, and two squares of clear gap to the foot. The rest mesh is now **120 × 62.5 × 25 mm**, long axis across the rail, starting at world `(0.640, -0.3453)` m. Thickness remains an estimate. Four 30 mm scallop periods have 8 mm peak-to-trough depth. The unbent, ungrooved mesh has 1,950 vertices and 8,640 tetrahedra. All of its surface belongs to the simulated soft body; it has no pins, attachment, hidden support or scripted node motion.

The previous 80 kPa material did not fit the clip's loaded shape. The final homogeneous SuperDex foam uses **Young's modulus 12 kPa**, Poisson ratio 0.3, density 110 kg/m³ (about 18.0 g), stiffness damping 0.008 s, friction 0.9 and friction falloff 0.1 mm/s. Density, damping, Poisson ratio, friction and contact parameters are assumptions, not measured foam specifications. Twenty nonlinear iterations and four soft substeps give a stable grasp in the tested runs; this is not a formal convergence study.

The press is now displacement-driven, with robot targets 20–42 mm below the free-foam seating target, rather than the old 1.5 N hover. These are **commanded TCP overdrives**, not measured foam indentation: stock arm compliance and grasp deformation absorb part of the command. The old benchmark's 3/8 mm indentation figures therefore cannot be equated with these target offsets. The fit is judged by actual contact and the deformed surface at the clip's loaded moments:

| Time | Photo pressed arc | Wide pressed arc | Photo crown force | Near edge below bar axis, photo |
|---|---:|---:|---:|---:|
| [Photo 13 s](run/hero_frames/hero_13.0.png) | 96.2° | [98.2°](run/hero_wide_frames/hero_wide_13.0.png) | 4.44 N | 2.7 mm |
| [Photo 18 s](run/hero_frames/hero_18.0.png) | 129.2° | [128.7°](run/hero_wide_frames/hero_wide_18.0.png) | 4.90 N | 3.5 mm |
| [Photo 23 s](run/hero_frames/hero_23.0.png) | 108.3° | [104.9°](run/hero_wide_frames/hero_wide_23.0.png) | 4.17 N | 3.5 mm |

The [photo wrap probe](run/hero_wrap_probe.json) and [wide wrap probe](run/hero_wide_wrap_probe.json) measure the angular span of bar contact samples at ≥300 Pa on the straight run. This is a sampled span, not a guarantee of equal pressure at every angle. The bar axis is z = 0.880 m; the near edge is measured on the camera side, x < 0.6275 m. Particle/contact arrays are preserved in `run/hero_wrap_13.npz`, `_18.npz`, `_23.npz` and the corresponding wide files. The [16 s probe](run/hero_lift_probe.json) separately verifies clearance.

Against the grasp in `ref/t008.00.jpg`/`ref/t010.00.jpg`, the **inner facing pads** of the right hand meet the two **long scalloped side faces**, near their upper portion and mid-length. The right hand's left finger meets the +y side and pushes toward −y; its right finger meets the −y side and pushes toward +y. They do not pinch the two short ends or hold the foam by its underside. At 10 s the [contact evidence](run/hero_grasp_contacts.json) reads `right_link_left_finger`: 2.79 N, centre (0.5355, -0.2379, 0.9733) m; `right_link_right_finger`: 2.76 N, centre (0.5367, -0.2822, 0.9729) m. Both contacts lie near the foam's mid-length, with material extending fore and aft of the jaw pair.

The carried grip uses a 5 N per-finger feedback target; the loaded strokes hold a 44 mm commanded jaw gap. Feedback resumes for the slower return, allowing a commanded gap down to 32 mm if necessary. The catalog's gains and force limits are untouched. Actual forces vary with deformation and differ from that target. The left hand commands a 22 mm jaw gap around the elbow through the stock fingers; the fixture does not depend on its squeeze.

## Strokes, stripe and timing

The clip was read every 0.5 s from 11–24.5 s ([reference sheet](run/round3/clip_strokes.jpg)). Five right-to-left loaded strokes replace the former constant-speed pass, separated by lifts and small backward resets. No inset, heat map or captions are rendered.

| Phase | Twin time |
|---|---:|
| Left hand reaches elbow; holds | 0.8–3.0 s; through 26.85 s |
| Right approach, lower, close and pick up | 3.0–9.5 s |
| Seat on right end | 10.0–11.5 s |
| Stroke 1 / lift | 11.5–13.1 / 13.1–13.65 s |
| Repress, stroke 2 / lift | 13.65–15.55 / 15.55–16.0 s |
| Repress, stroke 3 / lift | 16.0–18.05 / 18.05–18.65 s |
| Repress, stroke 4 / lift | 18.65–20.7 / 20.7–21.3 s |
| Repress, stroke 5 | 21.3–23.0 s |
| Clear bar, return, lower to mat | 23.0–26.25 s |
| Open fingers, retreat | 26.35–27.6 s |
| Both arms home | 27.6–29.1 s |
| Settled end | 30.6 s |

The first press is about 1.4 s later than the clip's nominal 10 s; release is about 1.6 s later than its nominal 25 s, and home is about 0.5–1.1 s late. The return moves in front of the bar to clear the catalog wrist. These are motion departures within the requested approximate two-second phase tolerance.

**The blue stripe was added procedurally**, following the visible marker along the clip's crown. It is a 3.6 mm wide visual coating represented by 136 non-colliding cells across the full 340 mm straight run. It fades bin by bin from the same measured exposure used for coverage: crown contact z > 0.889 m, |x−0.640| < 8 mm, contact force norm/area ≥300 Pa, axial relative sliding >2 mm/s, and accumulated exposure ≥25 ms within 3 mm of a bin. These are numerical cleaning thresholds, not a calibrated ink-removal model. `coverage.py` reads contact and changes colour only. There is no erasure from time, hand position or waypoints.

## Current-page validation

The following are the revised scene's own page runs, not CPU-preview readings. The suite's [normal replay result](run/round3/physics_written_result.json) records 100.0% coverage, 0.00 mm bar drift, 0.000 N final finger force and a 45.6 mm foot gap, consistent with the filmed frames above. The [stock-limit replay](run/round3/physics_built_result.json) also records 100.0% coverage, 0.00 mm drift, 0.000 N final finger force and a 45.7 mm gap. The bar legs now end flat at the steel plate tops, without a buried terminal sphere. The accepted bar/foot placement and photo calibration stay fixed.

`./ctl check --camera photo --object bar --object sponge --steps 180` ([output](run/round3/check.txt), [JSON](run/check.json)):

```text
PASS  builds       the last rebuild landed
PASS  finite       every number in the state is finite
PASS  camera       photo 960x540, fov 45 deg; clip 1920x1080
PASS  objects      all of bar, sponge are in the scene
PASS  penetration  worst overlap 0.08 mm between left_rf_down_geom_7 and bar (bar 1 mm)
PASS  rest         nothing moved more than 0.01 mm in 180 steps (bar 2 mm)
PASS  finite_after still finite after 180 steps
```

`./ctl physics routine --wait-limit 1200` ([full output](run/round3/physics.txt), [JSON](run/physics.json)):

```text
PASS  contact  nothing ties a thing to the robot, no motor on a thing's joints, nothing welded
PASS  robot    the registered yam, its own file, hands and servos
PASS  model    gravity 9.81 m/s²; the handled things' densities are ones objects have
PASS  written  finished; moved sponge
PASS  frozen   with the robot frozen nothing moved more than 0.0 mm
PASS  open     with the fingers held open the task did not happen (RuntimeError: Sponge did not clear the bar in the fingers)
PASS  built    finished with every friction at most 1 and the arm's torque at 27/10 Nm, the things ending where the written run left them
NOTE  heavy    not read: what the routine moved is a deformable (sponge), which heavy makes no heavier
PASS  grip     no finger force on a rigid thing (a deformable's is not read)
NOTE  drop     not read: what the routine moved is a deformable (sponge), and drop follows a rigid thing
PASS  plain    played with nothing watching, the things ended where the replay left them
```

Read-only samples from the [frozen control](run/round3/frozen_midpoint.json) and [open-finger pickup](run/round3/open_pickup.json) both show 0% coverage and the sponge still on the mat.

The generic heavy, grip and drop readings do not measure deformable mass scaling or soft-body grasp margin. The routine's own SuperDex contact measurements establish holding and final release. An additional current-scene [open-at-height probe](run/soft_drop_probe.json) records a 194.4 mm fall after opening, with 0.000 N final finger force. No prop pose, node position or attachment is written by the routine. All commanded motion is through the registered YAM's joints and gripper openings; every friction is at most 1.

## Calibration, provenance and remaining differences

The held photo fit still uses nine 25 mm grid crossings outside the overlays: **1.08 px RMS**, 1.50 px maximum at 1920 × 1080 ([fit](run/camera_fit.txt), [overlay](run/camfit_photo.png)). The [final initial-state comparison](run/round3/final_photo_vs_ref.png) shows the current twin beside the first clip frame. Intrinsics are scaled to 960 × 540. It is a planar fit with assumed grid origin and mat height, without independent non-coplanar validation. The real phone moves later; the twin's camera stays fixed. Wide is now on the phone/front-left side above the arms, at `(−0.65, 0.35, 1.55)` looking at `(0.50, −0.05, 0.95)`, 45° vertical field of view. It is an explanatory view, not a second calibrated phone camera.

Old benchmark claims retained after comparison are the 25 mm diameter rail, 340 mm straight run and 32 mm elbow radius. The accepted feet are 60 × 84 × 70 mm; bar centre is world `(0.640, −0.012, 0.880)` m. Arm spacing remains 601.6 mm from calibration. Base x, height and headings remain placement assumptions rather than the complete hand-eye transform. The old 50 × 75 mm sponge, along-bar orientation, stiff material, constant-speed pass and omitted stripe have been replaced. Runtime reads no files from `/inputs/old/`. [Asset provenance](assets/PROVENANCE.md) documents the procedural geometry.

The bench, mat, glass and slats are modeled, with no clip pixels or clean plate anywhere. The background remains darker, the wood/light warmer, and the room details less exact than the clip. The catalog YAM's wrists and fingers differ from the real rig's, and external purple cables are omitted. The right wrist uses a 60° approach to make the catalog hand pinch the resized, correctly oriented foam; its silhouette and return path differ from the real hand. The homogeneous foam approximates the visible wrap but is not a measured material identification; scallops are regularized, and edge bulging is simpler than the real sponge. Elbows are segmented geometry and mounting bores are omitted. The bar is an anchored fixture.

The page was not restarted. The photo and wide recordings completed in 548.1 and 515.3 seconds wall time respectively; each records 30.6 seconds of simulated motion. Both use RTX. No individual path-traced render exceeded ten minutes; longer control replays were advancing physics, not waiting on a still render. [Final status](run/rtx_final_status.json) records the final loaded scene and renderer. The page is left paused at the initial state for replay.

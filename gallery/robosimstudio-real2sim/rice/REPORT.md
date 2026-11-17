# Rice cleanup twin

This is the stock-photo spill rebuilt as a live 6,420-particle Newton MPM scene, with a contact-only dustpan cleanup by the catalog `yam_bimanual`. The left hand pinches the pan's back wall, the right hand pinches a wooden pusher and makes one sweep, then returns and releases the pusher on the counter. The empty right hand parks while the left lifts, tips and presents the loaded pan.

## Preserved

The original twin supplies the spill mesh, particle count, material, cup shell and dimensional scale. Its calibrated `photo` camera remains fixed: 1920×1280, 21.4° vertical FOV, eye `(0.405,0.041,1.530)`, target `(0.600,0.041,0.750)`, about 76° down. The spill anchor remains bench `(-0.029947,-0.010920,0.000300)` m. Each elongated grain follows a live MPM particle; it is not a rigid grain with individually simulated rotation.

The free cup retains the assumed 70 mm mouth, 62 mm base, 60 mm height and 2 mm wall. The tools follow the old film's shapes: a 180×150 mm inner pan with 10 mm floor, 35 mm ramp, 10 mm side walls, 20 mm back wall and 55 mm back height, and a 170×20×40 mm pusher. The left hand still pinches across the back wall, with one finger inside and one outside. The robot uses its registered file, original fingers, servo gains and force limits.

The closer `wide` camera is held at 1280×960, 42° FOV, eye `(0.18,-0.80,1.65)`, target `(0.55,0.03,0.98)`. Both cameras, the bake, the sweep and the left grasp/lift/tip are unchanged in this final revision.

## Rebuilt and departures

The old film swept 11,144 grains on a separate bare bench and composited the twin's surroundings afterward. This port performs the chore directly in the photo twin, with its 6,420 particles, cup and marble. The current material uses `GRANULAR_60HZ`, a 3.6 mm grid and 240 Hz material steps, with 60 Hz robot control. The tools have watertight particle colliders matching their visible shapes, with 4 mm particle margins; their rigid contact pieces remain the visible floor and walls. No object is attached, welded, scripted into place or driven by a motor.

`assets/settled.npz` is a ten-second re-settle of the original particles under this solver with the final cup placement, not the exact old bake. The largest change was 10.90 mm, RMS 1.19 mm (`run/rebake.json`). It is restored only on initialization/reset. The cup is explicitly in the rice's `containers` and remains free. It is moved **45 mm in bench +x** from the photo pose to clear the stroke and wrist; it is not moved during the chore. Its whole silhouette remains in frame, but its centre is visibly above the reference position.

The pan's lip is **0.2 mm rather than the old film's 2 mm**: the thicker lip failed the collection threshold. The other tool dimensions are retained. Pan/pusher centres start at bench x −0.037 m, with the pan lip y 0.095 m and pusher y −0.130 m. The pusher follows the actual ramp height and its own tilt during the sweep.

There is **one stroke**. The discarded second stroke previously gained only one grain, from 82.24299% to 82.25857% (+0.01558 percentage points) over 6.13 s. Two narrow side trails remain in addition to the lip line; the cleanup does not collect them. The single stroke is kept unchanged here.

The right arm stands on the counter's front edge facing the left arm; both stood behind the counter in the old film. Its final mount is bench `(0.27,-0.12,0.022)`, yaw 90°, with a steel support. Moving it 20 mm outward removed the static housing sliver from the photo without losing reach. IK solutions for the approach, pinch, sweep and return had position errors below 0.001 mm; the contact replays test the actual motion. The left wrist inclines 20° outward and grips 65 mm off the pan centre; the right wrist inclines 35° and grips 30 mm toward the cup. These placements separate the wrists and keep the sweeping arm near the frame edge.

After the sweep and the existing 80 mm rise, the right hand carries the pusher above its starting place, lowers it to about 3 mm above the counter, opens, waits for it to settle, rises and goes home empty. The left holds the pan down throughout placement, then performs the same 60 mm commanded rise, 25° tip and 70 mm presentation toward the photo centre. Placement adds 3.3 s; the routine lasts 20.7 s, versus the old film's roughly 11.7 s.

The grain palette is newly narrowed toward terracotta, and the procedural marble has a deeper grey vein. Neither uses reference pixels. The designed kitchen consists of painted walls, floor, cupboards, backsplash and an oak shelf. The end wall is extended and a painted +x wall closes the wide view's former black void. These surroundings, grain finish, light and shadow softness are approximations to the photo, not recovered measurements.

Masses (cup 90 g, pan 150 g, pusher 100 g) and friction (counter 0.4, cup 0.8, pusher 0.3, pan particle floor 0.15) are assumptions tested by the probes and replays. Grain meshes are 6–7.2 mm long and 1.8–2.15 mm wide with fixed random visual orientations. The scale itself rests on the assumed 70 mm cup mouth.

## Final readings

`run/final_compare.json`, `run/final_compare_beside.png` and `run/final_compare_look.png`: rice IoU **0.7069** (0.71 rounded), size ratio 0.88, centre offset [-3.5, -25.1] px. Cup IoU **0.1936**, centre offset [3.0, -178.5] px; the complete cup is visible. The photo's cup position remains a deliberate mismatch. Whole-picture dE 21.9, background dE 21.6. The marble and lighting remain approximate.

The unchanged bake's separate three-second probe (`run/rest.json`) measured maximum rice movement 0.418 mm, p99 0.133 mm and RMS 0.036 mm. Cup/pan/pusher moved 0.013/0.033/0.107 mm. The fresh final gate below measures its own 60-step rest window.

Both new RTX takes are independent runs from reset at 15 fps, with a 20.7 s routine (20.8 s video rounded):

| Take | Retained | Lip-origin rise | Actual grip rise |
| --- | ---: | ---: | ---: |
| `run/hero.mp4` | 79.66% | 118.03 mm | 52.96 mm |
| `run/hero_wide.mp4` | 79.63% | 118.05 mm | 52.97 mm |

The origin is the lip. About 65.1 mm of its rise comes from tipping about the back-wall grip, so this is not a 118 mm vertical hand lift. The strict inner-footprint cross-check counts **5,112 / 6,420 (79.63%)** in the pan's own live frame, within 180×150 mm, below 55 mm, with a 1.5 mm particle-centre floor allowance (`run/final_strict_measure.json`). The pass line is 75% with the pan raised more than 25 mm.

The final pusher is 0.22 mm from its initial x/y position, with top-normal tilt 0.000° and lower face -0.108 mm relative to the counter (small contact penetration). The empty right gripper is 95.00 mm open (`run/pusher_placed.json`). It stays on the counter after release.

Photo-camera robot segmentation at two-second intervals covers 13.55–17.52% during the sweep and 5.49% at the end (`run/hero_visibility/readings.json`, masks beside it). The initial photo has no clipped right-base housing. The wide first frame shows painted walls in the former black corner.

The filmed run records no left-finger contact with the pusher above 0.1 N (`run/left_pusher_contacts.json`). The audit's grip line below is the independent reading.

## Final check and contact audit

`./ctl check --camera photo --object rice --object cup --object dustpan --object pusher --object yam_bimanual` was run after filming. This explicitly requests the object-presence check. The following is copied verbatim from the final `run/check.json` (`run/check_final.txt` has the CLI output):

```text
PASS  builds       the last rebuild landed
PASS  finite       every number in the state is finite
PASS  camera       photo 1920x1280, fov 21 deg
PASS  objects      all of rice, cup, dustpan, pusher, yam_bimanual are in the scene
PASS  penetration  worst overlap 0.51 mm between left_lf_down_geom_5 and dustpan (bar 1 mm)
PASS  rest         nothing moved more than 0.11 mm in 60 steps (bar 2 mm)
PASS  finite_after still finite after 60 steps
```

`./ctl physics routine --only heavy` ran written and heavy first, then `run/complete_audit.py` invoked the same control API for frozen, open, built, drop and plain on that build. All seven modes are included. `run/physics.json` reports `passed: True`. Its report is copied verbatim below; `run/physics_final.txt` contains the remaining replay log and final report; `run/physics_written_heavy.txt` contains the initial written/heavy command output.

```text
PASS  contact  nothing ties a thing to the robot, no motor on a thing's joints, nothing welded
PASS  robot    the registered yam_bimanual, its own file, hands and servos
PASS  model    gravity 9.81 m/s²; the handled things' densities are ones objects have
PASS  settle   nothing starts inside anything (deepest 0.4 mm); the scene's settle moved nothing more than 0.0 mm
PASS  written  finished; moved dustpan, pusher, rice
PASS  frozen   with the robot frozen nothing moved more than 0.1 mm
PASS  open     with the fingers held open the task did not happen (RuntimeError: left pinch missed the 20 mm face: 95.0 mm)
PASS  built    finished with every friction at most 1 and the arm's torque at 27/10 Nm, the things ending where the written run left them
PASS  heavy    finished with the rigid handled things 2 times as heavy (rice: not), the things ending where the written run left them
PASS  grip     the fingers pressed dustpan 100 N (p95 100.4) by hand 1; pusher 100 N (p95 100.5) by hand 2; the solver used friction 1 to 1 at the finger contacts
PASS  drop     with the fingers opened at the top of the carry (step 783) dustpan, pusher fell
PASS  plain    played with nothing watching, the things ended where the replay left them
```

Retained shares in written / stock-limit / doubled-mass / unwatched replays were 79.63% / 79.56% / 78.97% / 79.58%, respectively (`run/final_audit_*_result.json`). The doubled-mass grip rise was 49.08 mm.

The drop check opens both hands at step 783 (13.05 s), during the pusher's rise. The pusher drops; the pan remains on the counter and cannot be lifted afterward. The audit uses the word ‘fell’ for an object ending away from its normal carried result as well as for a measured downward fall. Its drop replay ends with pan lift −0.1 mm, not a successful cleanup.

## Film inspection and folder

Inspected `run/hero_frames/contact.jpg` and `run/hero_wide_frames/contact.jpg` at two-second intervals, including the sweep, return, release, lift and last frame. Both final frames show the pusher on the counter and rice inside the lifted pan. The wide frame also shows the empty right gripper at home; that hand is outside the photo frame. Exact measurements and phase timings are in the two takes' `_result.json` and `_progress.json` files.

Runtime code is at the top level; meshes, marble texture and baked data are under `assets/`; the source photo and its two masks are under `ref/`. No runtime data is read from `/inputs` or another folder. The installed simulation library and catalog robot remain dependencies. The obsolete old film, unused `PARK_Q`, three unused provenance JSON files and Python cache directories were removed. Diagnostics, intermediate work and films remain in `run/`. Ruff check and format checks pass for the four runtime Python files. The page is reset and paused at the baked first state, with the normal `routine` restored.

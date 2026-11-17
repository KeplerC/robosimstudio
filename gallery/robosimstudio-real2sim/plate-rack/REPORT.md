# Plate rack — round 1 revision

**Gallery clip: [run/hero_pair.mp4](run/hero_pair.mp4)**, front and rack side by side. Both views are path traced and show the same physical take. Six plates reach six separate slots without a drop in **118.52 simulated seconds**, down from 196.07 seconds. Final plane tilts are **1.77–1.89°** from vertical. The scene check passes; the drawn-surface measurement finds no penetration in any of the **3,555 recorded states**. The full physics audit, including its NOTEs, is quoted below.

## Room and materials

The kitchen now has floor-to-ceiling return walls at both ends of the counter, an opposite wall, a continuous ceiling, and pale blue sky geometry behind the garden. A tiled kitchen floor replaces the bare floor slab. Interior fill and garden lighting keep the enclosed room bright. Every camera background remains empty: the picture comes from geometry and its own materials, never a reference photograph or projected photo pixels.

Backsplash grout is now pale and only 1 mm behind the tile faces, with 2 mm joints. Cream-white tile replaces the grey-block appearance. Cabinet colours and fill light produce pale mint in the render. The counter uses a new procedural honey wood texture with 100 mm strips and restrained grain. The sink's brighter, rougher metal finish reads grey steel rather than near-black. These choices were judged in the path-traced `run/front_first.png` and `run/rack_first.png`, then in both complete films' first and last frames. Neither view shows the former void corners; `run/room_materials.json` records the sampled material patches.

The room is still a designed interpretation of `ref/kitchen_pexels33375679.jpg`, not a calibrated reconstruction. Window/cabinet/room dimensions and lighting are estimates. The garden foliage, sink and faucet remain simplified; the window has no glazing, the sconce is not an emitting light, and glassware, towel, hose and small decorations are omitted. The counter-level task shelf differs from the photograph's larger wall rack. The rack view shows the counter, cream tile and return wall; the front view shows the sink, mint cabinets and window as well.

## Motion and result

The catalog **`yam_bimanual` (two arms)** is unchanged: original robot file, fingertips, colliders, collision filtering, servos and force limits. It is placed with `at=`/`arms_at=` on the same plinths. World base positions remain `(0.1525, ±0.21, 0.87)` and the counter remains 750 mm high. There are no pads, welds, attachments or object pose writes in the routine. The unused `arm_gains` and `finger_gains` arguments have been removed.

Each grasp is still at the top rim, with one finger inside the flange and one outside. The 7 mm flange depth, 1.4-second close, and jaw command lead of at most 0.25 mm are unchanged. Loaded carry stays at 35 mm/s, lift/fine motion/insertion at 25 mm/s, and set-down at 12 mm/s. These are controller nominal speeds; smooth interpolation eases each leg.

Empty-hand approach, retreat and home are twice as fast. When one arm reaches the shelf, the next begins its own approach, grip, lift and outer carry. It waits at `y=±320 mm` until the shelf corridor is free. After release and settling, its slow inward carry overlaps the other hand's empty retreat. Only one arm inserts at a time. An initial unrestricted overlap disturbed the opposing forearm; the outer waiting point fixes that interference without changing the layout or hand. The new pace is close to the requested 100–120 seconds, though still slower than the old approximately 66-second film.

| Plate | Arm | Slot | Release (s) | Final plane tilt |
|---|---|---|---|---|
| 0 | right | 5 | 27.58 | 1.89° |
| 1 | left | 4 | 44.98 | 1.89° |
| 2 | right | 3 | 62.37 | 1.86° |
| 3 | left | 2 | 79.78 | 1.88° |
| 4 | right | 1 | 97.18 | 1.80° |
| 5 | left | 0 | 114.60 | 1.77° |

Slots are numbered 0–5 from nearest to farthest from the robot. `run/hero_motion/meta.json` contains phase times, contacts and final slot offsets; `run/hero_motion/pose_summary.json` independently computes the plane tilt from recorded quaternions. The routine rejects a failed lift, a dropped plate, or a final placement outside its slot/3° tolerance.

## Checks and physical margin

`run/check.txt` and `run/check.json` record `./ctl check --steps 180` with all six plates named: build, finite state, presence, overlap and three-second rest checks pass. The starting poses in `assets/resting.json` are the same physically settled rack poses. The authored 180 mm plates, 25.532 mm depth, 3.8 mm flange, approximately 297.9 g mass, texture, mesh-derived inertia, and 97 convex pieces with 0.3 mm skin are retained. Shelf and rack geometry and all slot positions are unchanged. `run/asset_check.json` verifies that all 100 authored plate files are byte-for-byte unchanged.

`run/hero_motion/penetration.json` (with completion log `run/penetration_log.txt`) uses the original `check.py` method on every 30 Hz recorded state: 40,000 sampled points plus all vertices per drawn plate (44,994 points), 80,000 points plus vertices per fingertip for the reverse test, exact fixture primitive distances, drawn robot links and plate-to-plate tests. `measure_take.py` only divides this same work into resumable batches. Open robot shells are checked as convex hulls. This is a sampled measurement, not a proof between recorded states.

| Drawn pair category | Minimum measured clearance |
|---|---|
| shelf | 0.331 mm |
| racks | 0.328 mm |
| bench | 20.469 mm |
| fingers | 0.219 mm |
| fingers_reverse | 0.198 mm |
| Plate to plate | 20.229 mm |

Other robot links never entered the check's 1 mm expanded plate bounds. The raw motion and exported drawn model are retained in `run/hero_motion/take.npz` and `geometry.pkl` so the measurement can be repeated.

**Correction to the previous report:** the reviewer's round 1 gate did run `heavy`, `drop` and `plain`; the previous report incorrectly omitted those results. At twice the plate mass, plate_0 ended at 5.55°, exceeding the chore's 3° limit. Forced finger opening at step 422 dropped plate_0. Those were NOTEs; `plain` passed. The complete earlier reading is retained as `run/round1_physics.txt`.

The new audit is `./ctl physics routine --wait-limit 7200`, with **all default modes enabled**. `run/physics.txt` and `run/physics.json` contain every line:

```text
passed: True
report: 
    PASS  contact  nothing ties a thing to the robot, no motor on a thing's joints, nothing welded
    PASS  robot    the registered yam_bimanual, its own file, hands and servos
    PASS  model    gravity 9.81 m/s²; the handled things' densities are ones objects have
    PASS  settle   nothing starts inside anything (deepest 0.0 mm); the scene's settle moved nothing more than 0.0 mm
    PASS  written  finished; moved plate_0, plate_1, plate_2, plate_3, plate_4, plate_5
    PASS  frozen   with the robot frozen nothing moved more than 0.0 mm
    PASS  open     with the fingers held open the task did not happen (RuntimeError: plate_0: plate did not lift in the fingertips)
    PASS  built    finished with every friction at most 1 and the arm's torque at 27/10 Nm, the things ending where the written run left them
    NOTE  heavy    with the handled things 2 times as heavy the routine stopped: RuntimeError: Plate 0 is not upright in its slot: {'dx_foot_mm': -4.74, 'dx_top_mm': 4.63, 'tilt_deg': 5.65, 'yaw_deg': 0.01, 'dy_mm': np.float64(0.0)} (the grasp's margin)
    PASS  grip     the fingers pressed plate_0 24.1 N (p95 25.1) by hand 2; plate_1 23 N (p95 24) by hand 1; plate_2 23.8 N (p95 24.9) by hand 2; plate_3 22.9 N (p95 24) by hand 1; plate_4 23.8 N (p95 24.6) by hand 2; plate_5 22.9 N (p95 24.3) by hand 1; the solver used friction 1 to 1 at the finger contacts
    NOTE  drop     with the fingers opened at the top of the first carry (step 301) plate_0 fell; plate_1, plate_2, plate_3, plate_4, plate_5, carried later, was not released: one drop tests the carries under way at its release
    PASS  plain    played with nothing watching, the things ended where the replay left them
json: /work/repo/examples/live/twins/plate-rack/run/physics.json
logged: 12
```

- **heavy:** with the handled things 2 times as heavy the routine stopped: RuntimeError: Plate 0 is not upright in its slot: {'dx_foot_mm': -4.74, 'dx_top_mm': 4.63, 'tilt_deg': 5.65, 'yaw_deg': 0.01, 'dy_mm': np.float64(0.0)} (the grasp's margin)
- **drop:** with the fingers opened at the top of the first carry (step 301) plate_0 fell; plate_1, plate_2, plate_3, plate_4, plate_5, carried later, was not released: one drop tests the carries under way at its release
- **plain:** played with nothing watching, the things ended where the replay left them

The new twice-mass result, 5.65°, exceeds the 3° chore limit. The heavy NOTE is a real limit of the demonstrated grasp margin: success at the authored plate mass does not establish success at twice that mass. A drop NOTE records loss of support when the fingers are deliberately opened, rather than a drop in the normal chore. Neither is silently counted as chore success. Loaded speeds were preserved because of this limited margin.

Plate friction 0.7, fixture friction 0.8, contact stiffness 200,000 N/m, damping 400 N·s/m, and torsional/rolling friction radii 10/8 mm remain inherited modeling assumptions, not values measured from the kitchen photo. Physics remains 60 Hz control with eight substeps, elliptic cones at impedance ratio 30, and the same solver budgets. The audit's grip line states the effective contact friction and force actually used.

## Delivered files and inspection

- **`run/hero_pair.mp4` is the gallery film**, 1920×540, front and rack together.
- `run/hero.mp4`: front, 1280×720, recorded by `./ctl play routine --record hero --camera front --every 4 --wait`.
- `run/hero_rack.mp4`: rack, 960×540, rendered from the same saved physical motion using a separate render state in `render_take.py`; it does not move live objects. `run/rack_render_log.txt` records this render.
- `run/hero_strip.jpg` and `run/hero_rack_strip.jpg`: inspected phase sheets, including first and last frames. `run/film_check.json` records complete decoding and image-brightness checks.
- `run/front_first.png`, `run/rack_first.png`, `run/front_last.png`, `run/rack_last.png`: inspected path-traced stills, including the filled shelf in its room.
- `run/hero_log.txt` and `run/hero.json`: recording completion, metadata and contact/placement events. `run/final_status.json`: clean final page status. `run/serve.log` and `run/log.jsonl` are the live page's operational records.

`front` retains the old eye `(1.40, -0.78, 1.38)` and target `(0.52, 0, 0.86)`, with the established 55° field of view to include both arms. `rack` retains `(0.98, 0.40, 1.10)` toward `(0.55, -0.02, 0.84)`, 36°. The shelf close-up is essential to reading all six separate slots.

The scene and helpers are at the top level, scene assets and settled poses under `assets/`, and the reference under `ref/`. Nothing reads `/inputs/old`. Obsolete probes, old takes, intermediate builds, profiling files, segmentation dumps and batch scratch have been removed from `run/`; delivered films, cited readings and required live-server control files remain. `run/cleanup.json` records the retained file list and removed size. The page is left paused and reset to its six-plate starting state. Run `./ctl reset` then `./ctl play routine --wait` to repeat it.

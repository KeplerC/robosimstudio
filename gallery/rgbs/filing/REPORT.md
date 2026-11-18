# Task 3: Filing — round 3

This revision adds the 87.5–89.5 s press and the earlier intermittent contacts revealed by denser frame reading. Right pickup closure now precedes left closure. The right wrist pitches down for the one-hand put-down. Pickup remains near-level after steep-wrist probes failed to preserve the required grasp and face alignment. The calibrated `photo`, original prop placement, shipping YAM mechanics and contact-only abrasion are unchanged.

## Reference and timing

[reference_timing.md](reference_timing.md) lists **every two-second sample from 8 through 80 s**, plus every second from 81 through 90 s, as visually contacting or clear. Full frames are `run/round_3/ref_samples/`; inspected crops are `run/round_3/contacts_{00,12,24,36}.jpg`. The old `run/round_2/reference_timing.md` has also been corrected. The main view, not the acoustic inset, determines these readings; a visual seat cannot measure normal force. Near-touch boundaries are uncertain by millimetres and transition times by the sampling interval.

The earlier claim that the strip stayed clear throughout 82–89 s was wrong. It is clear at 84, rising at 87, seated at 88 and 89, and clear again at 90. There are earlier pauses too: the 8–80 s sample table shows distinct passes, not continuous contact.

| Event | Routine timing | Reference interpretation |
|---|---:|---|
| Right pinch closes | 3.6–4.2 s | Right at the strip around 4 s, left still waiting |
| Left pinch closes | 5.4–6.0 s | Both held by approximately 6–7 s |
| Lift under tool | 6–8 s | Raised by about 8 s; still clear of drum |
| Initial contacts | 15–17.2, 19–21 s | Contact at 16 and 20, clear at 18 and 22 s |
| Subsequent seated passes | 27.5–32.5, 39–41, 47.5–50.5, 57.5–60.5, 67.5–70.5, 77.5–80.5 s | Contact/clear samples listed individually in the timing notes |
| Pause below drum | 81–86.5 s | Clear at 81–86 s, about 25 mm beneath drum |
| Rise and second press | Rise 86.5–87.5; press 87.5–89.5 s | Rising/near touch at 87; seated at 88 and 89 |
| Drop clear | 89.5–90 s | Clear at 90 s |
| Return above mat | 90–92 s | Forward/lowering motion at 90–92 s |
| Left release | Opening ends 92.5 s | Left clear and raised at 93 s |
| Right put-down | 93–94.3 s | Right places strip around 94–94.3 s |
| Right release | Opening ends 94.8 s | Right releases between 94 and 95 s |
| Home | 99 s | Retreat during the final seconds |

The earlier passes rise over one second and descend over half a second. These are interpolated robot trajectories, not measured subsecond transition times. The shared clear height approximates the clip's variable gaps; its brief bounces are not identical vertical trajectories. Loaded passes use the same 2 N feedback, with a ±6 mm lengthwise sweep centred on the mark and a five-second period. The old continuous routine's biased sweep was replaced to cover the face during these shorter passes. **The abrasion coefficient and motor are not retuned.** Restoring the pauses changes removal progress: the round-2 claim of 100% by 70 s no longer describes this run. The late press can now remove remaining pigment as well as reproduce contact.

The one-hand carry after left release is intentional and matches the clip. It knowingly overrides the initial prompt's “two hands the whole time” line. Both pinches are required throughout the preceding carry, and at least one complete two-jaw pinch at every airborne sample. Airborne means the lowest transformed wood corner is more than 0.5 mm above the mat; tilted edge touchdown is not counted as airborne.

## Wrist probes and remaining motion departures

The front/back end pinch is retained: the jaws close across the strip's 15 mm width, one hand near each end. Robot targets alone move the strip. No workpiece pose, attachment or auxiliary finger collider is used.

The steep pickup was tested with shipping fingers and gains:

- `run/round_3/pitched_horizontal_pickup.json`: 35° left / 40° right with an inward approach pushed the strip approximately 29 mm sideways. The right pinch never formed; 0 of 78 airborne samples had both pinches.
- `pitched_vertical_pickup.json` and `pitched_grasp6.png`: the downward approach gave the right two jaw forces around 49 N, but the left missed. At 6 s its actual end-effector was about 37 mm above the target. There were 0 of 75 airborne samples with both pinches. Static IK alone found these orientations reachable (`pitched_ik.json`); reach alone did not establish a working contact path.
- `mixed_pickup.json`: near-level left / 40° right held both pinches in 75/75 short-probe airborne samples. However, the full run (`mixed_unlevel_probe.json`, `mixed_unlevel_phases.json`) retained about 16° strip tilt under the drum, touched only a small portion of the marked face, and cleared 0%.
- `mixed_trim_probe.json`: changing the relative filing hand heights by ±8 mm to level that grasp lost the right pinch; only 1,552/4,266 airborne samples retained both hands. The trial was rejected.

As permitted by the review, the final pickup uses the proven near-level angles: left 1.15° down, right 10.2° down. This remains a departure from the clip's steep pickup. The right still closes first. No robot limit, force, friction or geometry was changed to rescue the failed trials.

During put-down, after the left opens, the right wrist target blends to 25° by 93 s and 40° by 94.3 s. These are commanded angles; the stock servos yield under contact load. The probe retained a full right pinch. The strip lands on an edge before settling flat on release, rather than maintaining a perfectly horizontal face throughout placement. This is a remaining motion difference. In the final film, first logged edge contact is at 93.66 s. At the 94.3 s command endpoint the strip is still tilted about 25.8°, then settles flat by the 94.8 s release endpoint. The lowest-corner diagnostic reaches 1.54 mm below the nominal mat during this loaded transition: a finite-compliance contact penetration, not a claim of perfectly rigid nonpenetrating landing. It is retained as a limitation; the final resting strip is flat on the mat. Evidence: `run/round_3/paired_phases.json`, `paired_contact_samples.json`, `run/detail_t094.png` and `detail_t099.png`.

The higher elbows and imperfect left-home silhouette remain. The left wrist centre is held at about `(0.43117, 0.24999, 0.95962)` m, but some white link geometry remains above the reference inset boundary. The lower filing-elbow branches require coordinates −1.787 rad left / −1.827 rad right, outside the stock lower limit 0 rad; the left alternate elbow is also below the 0.750 m bench. `run/round_2/ik_solutions.json`, `unbounded_ik.json` and the home-candidate files retain this evidence. The shipping link shells and triangular fingers differ visibly from the real rig.

## Filmed result and checks

The final paired run reaches **96.47% cleared**, with mean residual intensity **0.00649**. Three of 85 cells remain above the 0.1 threshold; a small pale residual streak/dot pattern is visible in the close final frame. These are the filmed run's own readings (`run/paired_capture.json`, `run/round_3/final_readings.json`), not substituted replay numbers.

| Time | Cleared | Mean intensity | Drum normal force | Judged frames |
|---:|---:|---:|---:|---|
| 0 s | 0.00% | 1.00000 | 0.000 N | `run/wide_t000.png`, `run/detail_t000.png` |
| 16 s | 0.00% | 0.89823 | 2.191 N | `run/wide_t016.png`, `run/detail_t016.png` |
| 40 s | 8.24% | 0.37305 | 2.191 N | `run/wide_t040.png`, `run/detail_t040.png` |
| 70 s | 70.59% | 0.07359 | 1.463 N | `run/wide_t070.png`, `run/detail_t070.png` |
| 84 s | 87.06% | 0.02694 | 0.000 N | `run/wide_t084.png`, `run/detail_t084.png` |
| 85 s | 87.06% | 0.02694 | 0.000 N | `run/wide_t085.png`, `run/detail_t085.png` |
| 88 s | 96.47% | 0.01248 | 2.175 N | `run/wide_t088.png`, `run/detail_t088.png` |
| 89 s | 96.47% | 0.00783 | 1.458 N | `run/wide_t089.png`, `run/detail_t089.png` |
| 90 s | 96.47% | 0.00649 | 0.000 N | `run/wide_t090.png`, `run/detail_t090.png` |
| 99 s | 96.47% | 0.00649 | 0.000 N | `run/wide_t099.png`, `run/detail_t099.png` |

The 84 s and 90 s frames show a clear gap. Both 88 s and 89 s frames show the seated drum, and their measured forces exceed 0.1 N. The routine asserts those four samples. The spindle reads 58.88 and 59.16 rad/s at 88 and 89 s. The final face remains visibly much cleaner than the full initial blue ellipse, with the residual noted above.

“Cleared” means cell intensity ≤0.1, not that the visual surface is perfectly featureless. The zero-mass paint boxes remain as geometry and can leave a faint cell-dot pattern even when recoloured to wood. Wood and paint share roughness 0.8, but that does not eliminate their microscopic geometry. The report's earlier “renders uniformly as wood” wording was too strong. The larger blue mark disappears to the measured degree below; small residual pigment and/or cell texture must be judged in the final close view.

Final `./ctl physics routine`: **passed: True**, every line PASS (`run/physics.json`, `run/round_3/physics_console.txt`).

```text
PASS  contact  nothing ties a thing to the robot, no motor on a thing's joints, nothing welded
PASS  robot    the registered yam, its own file, hands and servos
PASS  model    gravity 9.81 m/s²; the handled things' densities are ones objects have
PASS  settle   nothing starts inside anything (deepest 0.0 mm); the scene's settle moved nothing more than 0.0 mm
PASS  written  finished; moved strip
PASS  frozen   with the robot frozen nothing moved more than 0.3 mm
PASS  open     with the fingers held open the task did not happen (RuntimeError: Strip did not lift in the fingers)
PASS  built    finished with every friction at most 1 and the arm's torque at 27/10 Nm, the things ending where the written run left them
PASS  heavy    finished with the handled things 2 times as heavy, the things ending where the written run left them
PASS  grip     the fingers pressed strip 100 N (p95 100.6) by hand 1; strip 101.1 N (p95 103.1) by hand 2; the solver used friction 1 to 1 at the finger contacts
PASS  drop     with the fingers opened at the top of the carry (step 725) strip fell
PASS  plain    played with nothing watching, the things ended where the replay left them
```

The grip values sum each hand's two jaws. Actual pad friction is 1. The 0.3 mm frozen drift is small startup settling. The stand has no movable body or scripted update and stays fixed.

| Replay | Cleared at end | Full pinch / airborne samples | Both pinches |
|---|---:|---:|---:|
| written | 96.47% | 4364/4364 | 4290 |
| built | 98.82% | 4365/4365 | 4290 |
| heavy | 96.47% | 4365/4365 | 4290 |
| plain | 98.82% | 4364/4364 | 4290 |

All four completed runs had no unexpected airborne supports. Files `run/round_3/physics_{written,built,heavy,plain}_{contacts,phases}.json` retain the measurements. The written replay had a single zero force sample at 70.00 s, with contact immediately before and after and a 69.90–70.10 s mean of 1.40 N. All required 84/88/89/90 s checks passed. Final filmed readings below are kept separate from these replay results.

Final `./ctl check --camera photo --steps 150 --object strip --object stand_base --object stand_post`: **passed: True** (`run/check.json`, `run/round_3/check_console.txt`).

```text
PASS  builds       the last rebuild landed
PASS  finite       every number in the state is finite
PASS  camera       photo 960x540, fov 31 deg; clip 1920x1080
PASS  objects      all of strip, stand_base, stand_post are in the scene
PASS  penetration  worst overlap 0.00 mm between table and strip (bar 1 mm)
PASS  rest         nothing moved more than 0.30 mm in 150 steps (bar 2 mm); not measured (no position to read): stand_base, stand_post
PASS  finite_after still finite after 150 steps
```

This checks initial/resting overlap; it does not certify zero transient penetration during a loaded landing. Lint and format checks pass for all five Python source files.

Final movies are **`run/hero.mp4`** (`photo`, 960×540) and **`run/wide.mp4`** (1920×1440). Both have **991 frames at 10 fps**, covering simulation 0–99 s at real-time playback. Capture began 2026-11-18T11:46:41.691790+00:00 and finished normally at step 4,950. The waiting CLI disconnected partway through; the page continued the same recording and the client reattached without restarting it. `run/round_3/paired_console.txt` records completion. `run/round_3/review_final.py` verifies exact equality of all frame-state records and unchanged source hashes.

The filmed run had **4364/4364** airborne samples supported by a full pinch, **4290** with both hands, and no unexpected airborne support contacts. Minimum individual jaw forces in the two-hand phase were 49.75, 49.00, 49.52, 49.16 N. Measured contact loss is **92.16 s left**, **94.50 s right**. The stand remains fixed. The strip travels 273 mm and ends about 4 mm from its initial centre, released on the mat.

I inspected `run/take_sheet.jpg`, both `run/round_3/{hero,wide}_take_strip.png` comparisons, `press_release_comparison.jpg`, and the native wide/detail frames named in the table. The comparison timeline now includes 84, 88, 89, 93 and 94 s. `run/detail_t004.png` and `detail_t006.png` show the sequential pickup; `detail_t094.png` shows the steep put-down, and `detail_t099.png` the final residual. These final films and their contact/phase files replace the prior take-named versions.

`filming.py` uses a persistent four-camera RTX rig to draw `photo`, `wide` and two supplementary detail views at the same states. It imports `robosimstudio_real2sim_agent` and replaces the two Recorder instances' frame-read methods; it does not change physics or advance the simulation. `scene.py` does not depend on this helper. Source hashes and exact equality of the two films' state records verify that both takes are from one unchanged run. Detail snapshots supplement the actual wide film; no caption or acoustic inset is added to either movie.

## Cameras, scene and assumptions

`photo` is held unchanged: twelve 25 mm mat-grid crossings outside overlays gave 1.7 px full-resolution RMS, focal length 1963.61 px at 1920×1080, vertical field of view 30.75°. The 960×540 take halves the intrinsics. Evidence: `assets/camera_calibration.json`, `run/camfit_photo.png`. This is a planar pinhole solve with assumed principal point; height lacks independent elevated calibration points. The handheld source camera drifts later; the twin camera does not.

`wide` stays at 1920×1440, field of view 55°, eye `(-0.05, -0.12, 1.32)`, look-at `(0.57, 0, 0.85)` m. Both arms are visible, with outer parts cropped at home. The stock jaws cover the grasped ends. Final visible strip bounding widths are 154 px at 0 s, 104 px at 16 s, 118 px at 40 s, 128 px at 70 s, 104 px at 84 s, 130 px at 88 s, 128 px at 89 s, 152 px at 99 s (`run/round_3/visibility.json`).

The strip remains 80 × 15 × 12 mm, mass 7.2 g, friction 0.65, contact stiffness 60,000 and damping 500. Its size was estimated against the grid; material properties are assumptions checked by the probes, not measurements from the clip. The 15 × 7 mm blue ellipse uses 85 one-millimetre cells less than 0.05 mm above the colliding wood. The drum remains 15 mm diameter × 12 mm high, at `(0.680, 0, 0.896)` m.

Pigment fades only where the upward-facing marked surface lies in the drum footprint and within 0.8 mm of its bottom face, with measured strip–drum normal force above 0.1 N and measured spindle speed above 10 rad/s. The increment remains `0.14 × force × angular_speed × radius × dt`. This is an assumed effective abrasion law using pair force, not a measured pressure distribution or a wood-loss model. Dust, heat, wood removal and abrasive wear are omitted. The routine never sets pigment or score.

The only non-robot motor is the fixed tool's spindle: 60 rad/s target, gain 0.01, 0.08 N m limit, armature 0.0002. These values cannot be measured from the clip. Control is 50 Hz with 1,000 Hz integration and a 0.2 mm contact gap. All robot gains, torques and finger colliders are stock YAM. Static stand components are fixed to the bench; they have no dynamic body for `check` to measure as drift.

Base spacing retains the supplied 601.6 mm transverse separation, 17 mm x and 29 mm z offsets. The right base sits on an inferred 29 mm riser, not in mid-air. Calibration spread was approximately 25 mm, so the riser height is uncertain. Absolute base depth is inferred. World bases remain `(0.250, 0.3008, 0.750)` and `(0.267, -0.3008, 0.779)` m.

The room, mat and grid are procedural geometry/materials, without clip pixels or a clean plate. The room remains darker than the clip, the mark is more saturated blue than the real periwinkle, and reflections, furniture, cables, tool details and the overhead cage are simplified. Assumptions include 140 mm slat pitch/60 mm width, a 680 mm pale round table at `(1.7, -0.48, 0.735)` m, glass opacity 0.12, lounge light intensity 1800, bench key 2.2 and ambient RGB `(1.1, 1.1, 1.1)`.

The unchanged first-frame scene's comparison remains `run/round_2/final_beside.png`, `final_look.png`, `final.json`: strip width 0.99× reference, height 0.90×, IoU 0.85; stand-base width 0.97×, height 0.79×, IoU 0.71. Whole-frame dE 29.2 includes the intentionally absent inset/title/speed overlays. Mask-excluded dE was 24.13 (`overlay_excluded_comparison.json`). The background mismatch remains substantial.

## Provenance and handoff

The old benchmark's `polish/machine.py`, `layout.py`, `records.py`, `marks.py`, routine and graph informed the fixtures, spatial abrasion field and grasp–press–stroke–return plan. `staging.py` and `look.py` informed the room. The old 150 × 25 × 4 mm plastic strip, sideways tool, cradles/rails and grasp locks were not retained. No runtime path imports `/inputs/old`. Code is at the top, data under `assets/`, clip/mask under `ref/`, and diagnostics under `run/`; no library or test files changed. Old round-2 measurements are historical unless explicitly identified as unchanged above. All `__pycache__/` directories are removed before handoff. The page is left at the reset, paused initial scene; both completed films are on its board.

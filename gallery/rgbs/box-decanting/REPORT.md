# Task 2: Box Decanting

A YAM picks a safety cutter up off the mat while its left fingertip braces the carton's corner, draws the blade along the yellow-taped seam from left to right, and sets the cutter down; then the left hand lifts the far flap and the right hand the near one, each from a short end. In the final `photo` take 100% of the tape is cut, both flaps stand at 95 degrees, the cutter is released at 24 s, and the carton slides at most 8.8 mm; every phase lands within about 2 s of the clip's. All seven `check` lines and all twelve `physics` lines pass.

**Departures from the clip, in short:**
- The tape is a material law of this twin's own (below), because the library ships no tape that seals two flaps and can be cut.
- The carton's inner flaps hang folded down (26 mm) where the clip's lie flat (about 50 mm): no catch the YAM's fingertip could make under a flap was found with flat inner flaps. The attempts are listed below.
- The seam gap is 12 mm against the clip's 2 to 3 mm.
- Both flap catches come in pitched 40 degrees down, where the clip's grippers enter level.

## Inner-flap attempt and remaining departures

`ref/t042.00.jpg`, `ref/t045.00.jpg` and `run/flap_reference/t027.00.jpg` show the pale inner bands and central notch. Their approximate 49/56/49 mm partition across the 154 mm length supports two approximately 50 mm folded-in panels. I built those panels with passive 0–95° hinges, a lower stop at flat and no spring. Their top faces sat 0.5 mm below the closed major panels. Each panel was assigned 3.85 g, scaling the prior assumed areal mass; height and mass remain modelling assumptions. The attempted construction is saved in `run/flat_carton_attempt.py`. It passed the three-second resting check (`run/flat_initial_check.json`), and the unchanged cutter stroke still severed every tape station.

I moved the catch from approximately 52 mm along the seam toward the centre to approximately 10 mm, inside the dust-flap tips. `run/flat_catch_probe.py` runs the actual pickup and cut before probing; it does not assign object pose or tape state. Trials included a rolled level hand, a lower-finger orientation with an alternative wrist branch, vertical finger separation, and 15° and 40° downward pitch. The latter trials approach above the inner panel, lower into the central seam and move under the major edge.

| Flat-panel probe | Contact result | Far flap at end | Maximum shell slide |
|---|---|---:|---:|
| `run/level_inboard.json` | native finger rail pressed opposite major panel; peak 35.1 N on tape | −0.06° | 83.66 mm |
| `run/level_lower_flip_inboard.json` | alternate wrist branch reached; rail pressed opposite panel, 43.6 N peak | 0.02° | 40.60 mm |
| `run/level_vertical_inboard.json` | rear finger hardware pressed far panel, 31.3 N peak | −0.12° | 199.50 mm |
| `run/pitch15_vertical_inboard.json` | rear finger hardware pressed far panel, 26.6 N peak | −0.01° | 17.07 mm |
| `run/pitch40_vertical_inboard.json` + `run/pitch40_vertical_lift.json` | fingertip landed on top of tape, 8.08 N peak in insertion; no underside catch | −0.01° | 8.93 mm |

These are failed diagnostic runs, not delivered takes. `run/pitch40_catch.png` shows the last insertion; the actual selected fingertip was 8.1 mm above its requested underside target. `run/catch_ik_scan.json` finds valid level wrist branches with sub-millimetre residuals, so reach alone does not explain the failures. The saved source variants and contact traces show failure of these tested approaches, not proof that every possible flat-box catch is impossible. So the twin keeps the hanging panels, which open reliably, rather than deliver a worse result.

**Seam-width departure:** the clip's exposed seam is approximately **2–3 mm** wide, inferred from the few-pixel slot against the roughly 25-pixel, 18 mm tape in `run/stroke_reference/detail0.jpg`. The twin retains a **12 mm** gap inherited from the old benchmark and sized for the native YAM fingertip. It is visibly wider, especially in the untaped end slots.

**Grasp departure:** the delivered catch still uses 40° downward pitch and 30° initial yaw, with native finger contact from each short end. It is steeper than the clip's nearly level entry. The left hand opens the far panel and the right opens the near one. No additional finger collider, dust-flap brace, opposite-hand press or altered robot servo is used. On the delivered hanging-panel geometry, `run/hanging_level_contact.json` repeats a 0°-pitch entry at the existing catch location. It hits the short wall at 19.57 N, leaves the far flap at approximately 0°, and slides the box 334.99 mm. This is the contact evidence for retaining 40° on the delivered path; it is not an IK reach claim.

## Reference timing and controller

The half-second stills from 6–21 s are in `run/stroke_reference/`, especially `detail0.jpg` and `detail1.jpg`. The cutter nose points image-left; the blade remains at the left end through roughly 16.5 s, pulls toward image-right from 17 s, finishes around 19–19.5 s, and lifts around 20 s. The right fingertips alone hold its rear housing through friction. The left fingertip braces the starting corner from about 5 to 19.5 s.

| Event | Clip time (s) | Twin time (s) |
|---|---:|---:|
| Pickup complete | about 4 | 4.0 |
| Blade seated | about 8 | 7.1 |
| Pull begins | about 17 | 17.0 |
| Pull ends | 19–19.5 | 19.5 |
| Tool lifts | about 20 | 19.5–20.3 |
| Cutter set down/released | 23–25 | 22.4–24.0 |
| Far flap starts lifting | about 25 | 24.6–24.7 |
| Far flap upright | about 27 | 27.1 |
| Right hand waiting beside short end | about 28–30 | by 29 |
| Near flap starts lifting | 37–38 | 36.4–36.5 |
| Near flap upright | about 39 | 38.9 |
| Both hands clear, beside short ends | about 42 | 40.3 |
| Final state | 45.4 | 45.4 |

The near lift and final withdrawal remain slightly early, within approximately two seconds. The small spontaneous far-flap opening after the real cut is not reproduced.

The integrated run completed at 45.4 s with 100% cut, both panels at 95°, 8.62 mm maximum shell slide, and the cutter released (`run/round4_integrated_result.json`). `run/round4_final_pose.png` shows the lowered final hands.

The lower idle poses put the lower native fingertip about 120 mm from the carton centre along the seam, at approximately lid height, outside the 77 mm half-length. The right hand waits there until 33.6 s, then makes a clearance move before entering the existing catch at 36.4 s. A 0.6 s settling interval follows the wrist transition; without it the near catch missed despite the correct target pose. After the lifts, both hands withdraw outside the short ends before turning level. No parked finger presses the open far flap.

The final left opening path caps the commanded fingertip arc at 110° instead of 115° and stops advancing once the measured panel passes 94°. This reduces pushing against the unchanged 95° passive stop. It does not change the object hinge, motorize it or write its state.

## Tape law, assumptions and picture

The **twin-local tape material law** is a departure from the shipped materials library. Two 84 × 9 × 1 mm rigid half-strips meet at the seam, with 28 longitudinal stations and two intact values per station. A station counts as cut only when both values are below 0.05; reported percentage is over the 84 mm tape length.

`d(intact)/dt = −90 × max(press − 2 N, 0) × |v_draw|`

Damage requires solver-measured blade contact and actual relative drawing motion within a 7 mm kerf. Press is averaged over five 20 ms steps. Each hinge receives internal cohesion `mean(intact) × (−120 q − 0.35 qdot)`, capped at `±0.20 N m × mean(intact)`. Fully cut tape holds nothing. The routine commands only the robot. The 1 mm drawn slit, colliding tape cells and yellow halves are unchanged. The prior `run/cohesion_round3_probe.json` measured 0.0453 N m peak opening contact moment, 0.1120 N m peak resisting cohesion and only 0.01134° opening on an uncut flap. Those are simulation probes, not real tape-force measurements.

The outer carton remains 154 × 100 × 78.5 mm. It is a rigid-board approximation, not deformable cardboard. Major panels are 2 mm thick and 42 mm deep, mass 5 g each, with passive 0–95° stops, damping 0.0015 N m s/rad and armature 0.0005 kg m². The dust panels are 26 mm deep, initially −90°, with −95–0° limits, mass 2 g each, damping 0.0015 and armature 0.00005. Shell mass is 85 g and cutter mass 25 g. These are assumptions. Frictions are shell 0.65, mat 0.75, housing 0.8, blade 0.25 and tape 0.4; shipped fingers remain at 1. Stepping is 50 Hz control / 500 Hz rigid, 0.2 mm contact gap, stiffness 60,000 and damping 500. Native robot file, gains and torque limits are retained.

`photo` is held at its accepted twelve-point grid fit: 0.44 px RMS, 0.65 px maximum, fx=fy=1799.45 px, principal point (960,540). Evidence is in `assets/calibration_points.json` and `run/camfit_photo.png`. The planar fit does not prove equivalent accuracy throughout the room. Both bases remain on the bench, with 601.648 mm transverse spacing and 17.044 mm fore-aft offset. Cutter centre remains (0.59063, −0.17726, 0.7667) m; its 45 × 14 × 25 mm housing has a grey body, red slider and black end, with a box approximation to the rounded real shape. The closer `wide` camera is unchanged.

Room geometry, reflections, lighting and arm hardware remain approximate. Cables and the overhead camera are omitted. The later handheld camera motion is intentionally not followed. No clip pixels, overlay captions, acoustic inset or clean plate appear in the simulated scene.

The old benchmark supplied the basic shell/flap construction, cutter geometry and contact-driven tape/cohesion law. The dimensions, camera, pull direction, timings and native-finger motion were checked or rebuilt locally. Nothing imports `/inputs/old`. The routine creates `run/`, finds contact shapes by label, verifies that only right fingertips touch the cutter, and posts its own measured end result on the page. The pre-revision source and successful takes are preserved under `run/round3/`.

## Final verification

`./ctl check --camera photo --object carton --object cutter --steps 150` passes all seven checks. Worst initial overlap is 0.01 mm and maximum drift over three seconds is 0.10 mm (`run/check_round4_reply.json`, `run/check.json`).

`./ctl physics routine` passes all twelve lines on the delivered controller (`run/physics_round4_reply.json`, full detail in `run/physics.json`):

```text
PASS  contact  nothing ties a thing to the robot, no motor on a thing's joints, nothing welded
PASS  robot    the registered yam, its own file, hands and servos
PASS  model    gravity 9.81 m/s²; the handled things' densities are ones objects have
PASS  settle   nothing starts inside anything (deepest 0.0 mm); the scene's settle moved nothing more than 0.0 mm
PASS  written  finished; moved cutter, carton#3, carton#4
PASS  frozen   with the robot frozen nothing moved more than 0.2 mm
PASS  open     with the fingers held open the task did not happen (RuntimeError: Cutter pickup failed)
PASS  built    finished with every friction at most 1 and the arm's torque at 27/10 Nm, the things ending where the written run left them
PASS  heavy    finished with the handled things 2 times as heavy, the things ending where the written run left them
PASS  grip     the fingers pressed cutter 100.2 N (p95 108.5) by hand 2; the solver used friction 1 to 1 at the finger contacts
PASS  drop     with the fingers opened at the top of the carry (step 1006) cutter fell
PASS  plain    played with nothing watching, the things ended where the replay left them
```

The routine’s own outcomes for those replays are in `run/physics_round4_task_results.json`; these verify tape, flap and slide criteria beyond the generic audit’s end-position comparison.

| Replay | Tape length cut | Near / far angles | Maximum shell slide | Cutter released |
|---|---:|---:|---:|---|
| written | 100% | 95.00° / 95.00° | 8.33 mm | yes |
| built | 100% | 95.00° / 95.00° | 9.20 mm | yes |
| heavy | 100% | 95.00° / 95.00° | 5.22 mm | yes |
| plain | 100% | 95.00° / 95.00° | 8.73 mm | yes |

All 56 tape intact values finish at zero in these completed replays. Only `right_lf_down_*` and `right_rf_down_*` shapes contact the cutter. The audit’s 100.2 N grip reading is the summed native finger reaction, not blade pressure. Run-to-run slide varies; the final filmed readings are reported separately below.

The fresh RTX comparison (`run/round4_initial_reply.json`, `run/round4_initial_beside.png`, `run/round4_initial_look.png`) retains carton IoU 0.9846 and cutter IoU 0.7585. Their centre offsets are −0.4/−0.9 px and −2.5/+1.5 px respectively. Whole-picture colour difference is 34.1 CIE76 dE, 34.4 behind the keyed props. This includes the reference’s excluded overlay regions and does not indicate a photographic room match.

## Delivered films and measured result

Both final takes use RTX, 10 frames/s and 455 frames covering simulation time 0–45.4 s. Each starts from a fresh reset of the same audited source.

| Take | Camera | Cut length | Near / far angles | Maximum shell slide | Released |
|---|---|---:|---:|---:|---|
| [hero](run/hero.mp4) | photo | 100% | 95.00° / 95.00° | 8.806 mm | yes, 0 N finger force |
| [hero_wide](run/hero_wide.mp4) | wide | 100% | 95.00° / 95.00° | 8.906 mm | yes, 0 N finger force |

For the no-regression comparison, the archived previous photo and wide takes measured 8.837 and 8.913 mm respectively (`run/round3/hero_result.json`, `run/round3/hero_wide_result.json`). The new takes measure 8.806 and 8.906 mm. The photo remains 8.8 mm to the review's one-decimal precision; the wide take remains 8.9 mm. Both preserve the prior cut, open angles, release and phase times. These tiny slide differences are below the observed replay variation; they are not evidence of a substantial sliding improvement.

In `hero`, the left corner brace measures 5.39 N median downward reaction (5th–95th percentiles 5.17–6.57 N). Blade pressure during the draw measures 5.27 N median and 6.25 N p95. The first whole tape station parts at 17.74 s, 50% at 18.32 s, at least 95% at 19.02 s and 100% at 19.08 s. The hand finishes the stroke at 19.5 s. All 56 intact values and both cohesion torques finish at zero. Detailed readings are in `run/hero_result.json`, `run/hero_trace.json`, `run/hero_seal_history.json` and `run/hero_seal_final.json`, with matching `hero_wide_*` files.

I inspected `run/hero_inspection.jpg`, `run/hero_wide_inspection.jpg`, the full-resolution `run/hero_frames/t29.0.jpg`, and the reference comparison strip `run/hero_reference_strip.png`. The 29/33 s frames show the right hand waiting beside the short end; 40.3/42/45 s show both hands beside the carton, clear of both upright panels. `run/hero_wide_reference_strip.png` provides the other camera at the same reference times. The final solver reading contains no hand–carton contact pairs (`run/final_hand_carton_contacts.json`).

The comparison tool's carton-motion number includes the largest three-dimensional displacement of any articulated panel; it is not shell sliding. The routine measures horizontal root displacement every control step. Its own end-of-take cut percentage, flap angles, maximum slide and released state appear in the page's **Measured decant result** panel. This is not a typed caption.

Ruff check and format check pass for all five top-level Python modules. `./ctl status --wait` reports generation 59, RTX, finite state and a finished routine at 45.4 s. No simulated source changed after the full successful audit and final films. The page is paused in the final state with `hero_wide` as the latest take. The tape law, prop geometry, scene and room modules are byte-for-byte unchanged from the archived accepted take; the delivered simulated change is the robot motion.

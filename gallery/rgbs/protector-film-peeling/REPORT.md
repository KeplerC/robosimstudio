# Task 4: Protector Film Peeling

Both new path-traced takes finish with 100% of the film's bonds broken and the film hanging from the raised right hand. Maximum phone movement is **0.598 mm in hero** and **0.544 mm in hero_wide**. Every executed physics replay, including heavy, stays below 5 mm. The right close breaks 0% of bonds in both takes and every replay that executes it. The standard physics audit and scene check pass; the material NOTES and grip-reader limitation are quoted below.

## Seat robustness correction

The accepted round-4 result did not have enough margin: the later fresh-page check reported 10.36 mm of phone movement and a heavy failure. Those later numbers are the user's retest report, not measurements reproduced here. The previous delivered source and evidence are preserved under `run/round4/`.

The left grip was loading a corner unevenly and creeping during its long hold. Easing the original closure alone still moved the phone 3.81 mm by 23 s (`run/seat_eased912.json`). Its contact export also found the wrist camera touching the phone's far edge (`run/seat_eased912_points.json`). Attempts to level the jaws met another obstruction: the stock wrist body touched the mat and held the fingertips above their requested pose (`run/seat_flat_all_contacts.json`). These are observed contacts; the probes do not isolate a single force as the sole cause.

The final grasp moves the flat ends of the fingers onto the exposed side rim. Its phone-local TCP target is (-65, +48, +1) mm; the TCP lies outside the phone outline because the stock fingers extend inward from it. The approach yaw is 110° rather than 105°, the downward pitch is 3.5° rather than 7.5°, and the left closing command is 0.92 rather than 0.93 (requested gap 7.6 mm rather than 6.65 mm). A known IK branch supplies the approach and grasp joint waypoints. This clears the wrist from the phone and mat and reduces contact load without changing the robot's gains, force limits, fingers, colliders, prop geometry, masses or frictions.

In the isolated final hold, maximum phone movement was 0.211 mm at 12 s, 0.356 mm at 20 s and 0.407 mm at 23 s, with positive force on both phone faces and no bond break (`run/seat_clear110.json`). The complete preflight reached 100% peeled with 0.513 mm maximum phone movement (`run/seat_full_metrics.json`, `run/seat_full_probe.json`). The final movies and replays below test the complete routine again. The routine's original `if d.max_phone_mm >= 10` seat assertion is unchanged; the stricter 5 mm delivery criterion is checked against the recorded maxima.

## Measured result in every run

| Run | Maximum phone movement | Final phone movement | Peeled after close | Final peeled |
|---|---:|---:|---:|---:|
| hero | 0.598 mm | 0.505 mm | 0.0% | 100.0% |
| hero_wide | 0.544 mm | 0.379 mm | 0.0% | 100.0% |
| written | 0.502 mm | 0.420 mm | 0.0% | 100.0% |
| frozen | 0.200 mm | 0.164 mm | 0.0% | 0.0% |
| open | 0.143 mm | 0.143 mm | 0.0% | 0.0% |
| built | 0.516 mm | 0.443 mm | 0.0% | 100.0% |
| heavy | 0.489 mm | 0.481 mm | 0.0% | 100.0% |
| plain | 0.549 mm | 0.468 mm | 0.0% | 100.0% |

Routine maxima are sampled at every control step (60 Hz), not only the video frames. Frozen's maximum comes from the complete stock audit (rounded there to 0.1 mm) because its routine can stop before the full replay duration. Frozen's fingers never physically close. The stock drop mode does not execute a tape carry replay and therefore has no new phone-maximum reading; its accepted NOTE is quoted below.

| Run | First 95% crossing | Tab–right TCP | Drift in retained pinch | Right pad forces |
|---|---:|---:|---:|---|
| hero | 33.983 s | 17.08 mm | 9.28 mm | 58.76 N / 49.27 N |
| hero_wide | 33.983 s | 16.85 mm | 9.68 mm | 59.04 N / 49.47 N |
| written | 33.983 s | 17.05 mm | 9.38 mm | 58.74 N / 49.32 N |
| built | 33.983 s | 16.80 mm | 9.70 mm | 49.36 N / 58.97 N |
| heavy | 33.983 s | 16.76 mm | 9.80 mm | 49.24 N / 58.68 N |
| plain | 33.983 s | 16.91 mm | 9.58 mm | 58.99 N / 49.37 N |

At each 95% crossing, both right fingers have positive contacts on opposite broad handle faces inside its outline. The routine checks those contacts again at the raised finish. Drift is retained sliding inside the pinch, not a reset measurement. The historical 6 mm drift cutoff was already replaced with this direct contact test in round 4. `pull_N` is still saved but is not described as glass peel load: the tape API returns its handle-clamp spring resultant.

Tab travel during the two-second close is hero: 0.230 mm; hero_wide: 0.166 mm. All the film remains bonded through the open hover and close. Individual bond-break events are in each `*_metrics.json` file.

## Left contact and timing

| Photo take time | Finger body | Normal force | Phone face |
|---|---|---:|---|
| 12 s | `left_link_left_finger` | 0.949 N | bottom, local z -4.000 to -4.000 mm |
| 12 s | `left_link_right_finger` | 0.837 N | top, local z 4.000 to 4.000 mm |
| 34.8 s | `left_link_left_finger` | 1.203 N | bottom, local z -4.000 to -4.000 mm |
| 34.8 s | `left_link_right_finger` | 1.084 N | top, local z 4.000 to 4.000 mm |

The lower pad still contacts the phone's bottom edge rather than sitting 8 mm inside the outline; this remains a departure from the real fingers. The upper pad now bears on the exposed side rim. Full contact coordinates are in `run/hero_metrics.json` and `run/hero_wide_metrics.json`. The left jaws finish closing again at 33.5 s; the measured first renewed top/bottom contact is hero: 33.517 s; hero_wide: 33.517 s. The film's near rows continue peeling with this earlier re-grasp, and all bonds still break in the right pinch.

The reference reading is unchanged: the main sheet is flat through 20 s, the right jaws are open at 22 s and closed by 23 s, and the corner rises by 24 s (`ref/close/t018.00.jpg`–`t026.00.jpg`, `run/r4_reference_close_sheet.jpg`). The new routine keeps the accepted right-hand timing and raised finish:

| Phase | Simulation time |
|---|---:|
| Left approach / initial close | 0.5–3.75 s |
| Right approach / arrival | 4.75–8.25 s |
| Open hover | 8.25–20 s |
| Final corner alignment | 20–21 s |
| Right close | 21–23 s |
| First lift / progressive peel | 23–27.25 s |
| Left release and withdrawal | 27.25–28.75 s |
| Left return and close | 32.5–33.5 s |
| Final near-row pull | 33.5–34.25 s |
| Finishing arc | 34.75–36.25 s |
| Raised finish | 39.25 s |

| Take | 20 s peeled | 27 s peeled | 30 s peeled | 35 s peeled |
|---|---:|---:|---:|---:|
| hero | 0.0% | 37.6% | 45.6% | 100.0% |
| hero_wide | 0.0% | 37.6% | 45.9% | 100.0% |

## Takes and visible measurement

`run/hero.mp4` and `run/hero_wide.mp4` are new RTX path-traced recordings from the final source. `annotate_take.py` adds the on-screen bond percentage afterwards from measured samples; it changes no simulation state. The unannotated movies are retained as `*_raw.mp4`. Frame indices below are zero-based.

| Take | Tab above bench | Lowest film above mat | Film–phone clearance | First video frame ≥95% |
|---|---:|---:|---:|---|
| hero | 247.2 mm | 101.1 mm | 56.5 mm | frame 340, 34.0 s, 95.9% |
| hero_wide | 247.3 mm | 102.3 mm | 57.2 mm | frame 340, 34.0 s, 96.2% |

The equal-time comparison strips are `run/close_timing_strip.png` and `run/absolute_timing_strip.png`. The initial comparison is `run/final_initial_vs_ref.png`; final frames are `run/hero_final_frame.png` and `run/hero_wide_final_frame.png`. The camera is held while the reference camera drifts. These frames and the takes were inspected, not just their readings.

## Checks

`run/final_check.txt` / `run/check.json`:

```text
PASS  builds       the last rebuild landed
PASS  finite       every number in the state is finite
PASS  camera       photo 1920x1080, fov 45 deg; clip 1920x1080
SKIP  hand         photo rides no body: the hand line holds a wrist camera to its clip
PASS  objects      all of phone, film are in the scene
PASS  penetration  worst overlap 0.00 mm (bar 1 mm)
PASS  rest         nothing moved more than 1.87 mm in 180 steps (bar 2 mm)
PASS  finite_after still finite after 180 steps
```

`run/final_physics.txt` / `run/physics.json`:

```text
PASS  contact  nothing ties a thing to the robot, no motor on a thing's joints, nothing welded
PASS  robot    the registered yam, its own file, hands and servos
PASS  model    gravity 9.81 m/s²; the handled things' densities are ones objects have
PASS  settle   nothing starts inside anything (deepest 0.0 mm); the scene's settle moved nothing more than 0.0 mm
PASS  written  finished; moved film
PASS  frozen   with the robot frozen nothing moved more than 0.2 mm
PASS  open     with the fingers held open the task did not happen (RuntimeError: Only 0.0% of film bonds broke)
PASS  built    finished with every friction at most 1 and the arm's torque at 27/10 Nm, the things ending where the written run left them
NOTE  heavy    not read: what the routine moved is a deformable (film), which heavy makes no heavier
PASS  grip     no finger force on a rigid thing (a deformable's is not read)
NOTE  drop     not read: what the routine moved is a deformable (film), and drop follows a rigid thing
PASS  plain    played with nothing watching, the things ended where the replay left them
```

The standard server physics replays were run one mode per client through `run/audit_mode.py`, using the same engine as `./ctl physics`; their verdict and readings were not patched. Heavy executes the routine but does not increase the deformable sheet/embedded handle's mass, so its aggregate line is a NOTE. Drop follows a rigid carry apex and skips tape. The old supplemental heavy/drop probes remain under `run/round2/`; they are not claimed as new tests of this revision. The stock grip line does not read a deformable's forces; its PASS is not evidence for the right pinch. `contact_readings.py` reads the public VBD contact export for the measurements above. Those forces are solver readings, not independent load-cell measurements.

## Retained geometry, appearance and departures

The 20-point photo calibration is held: 5.4 px RMS, focal length 1990.4 px, largest residual 14.28 px (`assets/photo_calibration.json`, `assets/calibration_points.json`, `run/r2_camfit_spread.*`). The final keyed comparison (`run/final_compare.txt`) reads width 1.08×, height 1.15×, centre 9 px right / 30 px low, and visible-phone IoU 0.06; film covers most of the phone, so this is not a full silhouette IoU. Whole-frame colour difference remains 34.6 CIE76 dE, including the reference overlays. The 601.6 mm arm spacing and 17 mm longitudinal base offset are unchanged. The stock YAM is still used with only home joint angles, colour and appearance changed. Its elbow bulk, wrist housing and fingers differ from the real rig, and the initial right wrist remains about 100 px left/high relative to the clip.

The phone remains 147 × 72 × 8 mm and 200 g, leaning 2° on the fixed 105 × 57 × 35 mm white block. Those dimensions and the simplified slot are inferred from the still, not an independent measurement. The phone is free. Film is the shipped tape material with its own adhesion: 60 × 145 mm, 12 × 29 cells, 60 µm thickness, 0.06 kg/m² areal density, 0.5 mm particle radius, tack 2e6, break opening 1.5 mm and bending coefficient 0.1. These are simulation assumptions; the real film stiffness was not independently identified. Film friction is 0.3, phone/cradle/handle friction 0.8, and stock pad frictions at most 1.

The review-authorized 62 × 8 × 24 mm, 30 g handle with `tab_hold=3e3` remains a numerical handle for frictional gripping, not a literal thin film corner. It retains the film's display colour and opacity but still reads thicker than the reference. `film_appearance.py` changes display finish only and inherits the tape physics. The scratched sheet, grey 25 mm grid, foam seams, slats and glass use generated geometry/textures with no clip pixels. Lighting, reflections, cabling, rounded phone details and minor fixtures remain approximate; the mat renders darker than the reference. Whole-frame comparison includes the three burned-in reference overlay boxes, which the twin does not recreate.

The old benchmark supplied dimensions and material starting ideas; its latch and servo customization are absent. Only robot joint commands move the task. No prop pose, bond flag or prop joint is driven, and no finger attachment, extra collider or stronger servo was introduced. The final gallery binding contains only `routine`. Runtime telemetry and annotation writes create `run/` first. Code is at the top level, assets/calibration under `assets/`, references under `ref/`, and probes/takes under `run/`. Runtime scene inputs are local. No library or test files changed. The page is left reset and paused; bytecode caches are removed before handoff.

# Task 1: Cable Routing

The left catch and pull pass two consecutive physics invocations on unchanged source: `run/physics_final1_r5.json` (written and built) and `run/physics_final2_r5.json` (all replay modes). Built uses the real 27/10 Nm arm torque and every friction at most 1. The paired path-traced `hero` and `hero_wide` show the same successful execution. The full goal assertion remains enabled and unchanged.

## Catch correction and evidence

Two instrumented baseline invocations passed, so the reviewer's intermittent fresh-page failure was not reproduced exactly (`run/physics_baseline_r5.json`, `run/physics_baseline_repeat_r5.json`). I cannot identify the exact losing tick of that external execution from its final outcome alone. The baseline traces do show a marginal contact: the cord sits about 14.9 mm along a native pad ending at 16.5 mm, with about 4.5–4.9 N per finger during the initial lift. The stronger 0.975 closing target previously arrived only after that lift. Native collider dimensions are recorded in `run/native_pads_r5.txt`.

A controlled closer-to-neck probe exposed the contact failure directly: in `run/neck_built_r5.json` the thicker connector receives up to 57.7/64.2 N during the initial lift and later loses the pinch during the turn/pull; final cord normals are zero. Its written counterpart loses contact during the first lift. A shallow angled approach also collided with the exposed shelf. Neither diagnostic is a successful take or the delivered routine.

The final catch targets uniform cord approximately 16 mm behind the rod endpoint (about 10 mm behind the connector's neck), instead of 12 mm. The 12 mm-wide native pad then has nominal clearance from the thicker neck. It corrects the measured hand position before closing, uses the existing **0.975 closing target before the lift**, and lifts clear over **0.80 s rather than 0.60 s**. The draw and return still finish at about 25 s. No gains, torque limits, material stiffness, friction, dimensions, finger geometry or object attachments were changed. The camera, first state and emergence-terminated feed are retained.

The final measured pinch is farther inside the native pads and stays loaded through the lift and turn. With the 2.25 mm cord radius, the baseline centre at 14.9 mm puts part of its section beyond the 16.5 mm pad end; the final built centres remain inside that edge. The table tracks rod node −3, inside the contact, in the actual hand frame from completed catch to 25.8 s. Motion of adjacent unheld nodes includes bending and must not be mistaken for slip.

| Replay | Catch time (s) | Minimum cord normals (N) | Maximum held-point motion in hand (mm) | Pad-depth coordinate, start → end (mm) |
|---|---:|---:|---:|---:|
| final1 built | 21.24 | 10.31 / 10.45 | 0.148 | 13.72 → 13.82 |
| final2 built | 20.32 | 11.38 / 11.63 | 0.140 | 12.17 → 12.06 |

`motion.py` records hand pose, jaw opening, last eight rod nodes, finger normals, contact forces by native shape and joint tracking error every 20 ms during handoff/pull. Per-replay traces are `run/physics_final1_built_r5.json` and `run/physics_final2_built_r5.json`. Two passing runs are evidence of improved margin, not a proof over every possible numerical execution.

## Coverage and final results

`reading()` now uses the requested bore-interior centreline band with approximately 0.5 mm tolerance: **|x| < 6.75 mm, 1.5 mm < z < 15.5 mm**, in the box frame. It still measures the union of occupied channel-length intervals; the goal is still **80%**, not a changed threshold. The footer uses this live reading.

All distances below are millimetres. Height is above the box-top/channel-floor frame; “past left” is past the physical left end, not merely the lid exit. Finger columns are measured rod-contact normals.

| Execution | Coverage | End height | Past left | Tail on mat | Left normals (N) | Max box corner |
|---|---:|---:|---:|---:|---:|---:|
| final1 written | 100.0% | 169.4 | 130.3 | 386.7 | 11.76 / 11.62 | 0.642 |
| final1 built | 100.0% | 171.4 | 129.3 | 386.7 | 10.55 / 10.99 | 0.642 |
| final2 written | 100.0% | 171.1 | 128.9 | 386.7 | 10.72 / 10.66 | 0.642 |
| final2 built | 100.0% | 170.5 | 130.3 | 380.6 | 11.69 / 11.84 | 0.642 |
| final2 plain | 100.0% | 172.1 | 130.5 | 386.7 | 14.46 / 12.16 | 0.642 |
| final2 heavy | 100.0% | 170.7 | 130.5 | 386.7 | 11.69 / 11.70 | 0.642 |
| hero / hero_wide | 100.0% | 171.5 | 128.3 | 386.7 | 10.02 / 9.92 | 0.476 |

The paired take's emerged arc is **390.0 mm**, compared with the conditional 368 mm t025 reference trace; tail outside the right mouth is **638.8 mm**, including **386.7 mm on the mat**. “On mat” sums outside-mouth segments whose centres are within 5 mm of the bench height. The connector remains beyond the left fingertips.

Both physics reports explicitly read:

```text
PASS  built    finished with every friction at most 1 and the arm's torque at 27/10 Nm, the things ending where the written run left them
```

The first invocation selected written/built to test the repair. The immediate second invocation selected built, plain, frozen, open, heavy and drop after written. No runtime source or asset changed between them or before filming; `run/source_manifest_r5.json` records their hashes. Full tool detail is `run/physics_detail_final1_r5.json` and `run/physics_detail_final2_r5.json`.

The tool's rigid-box movement and rise, distinct from the routine's corner/yaw measurement:

| Replay | Tool movement (mm) | Tool rise (mm) |
|---|---:|---:|
| final1 written | 0.6 | 0.0 |
| final1 built | 0.6 | 0.0 |
| final2 written | 0.6 | 0.0 |
| final2 built | 0.6 | 0.0 |
| final2 plain | 0.6 | 0.0 |
| final2 frozen | 0.6 | 0.0 |
| final2 open | 0.6 | 0.0 |
| final2 heavy | 0.6 | 0.0 |

Contact, robot, model, settle, written, built, plain and grip pass. Frozen retains the accepted settling NOTE: the trailing connector drifts 3.2 mm over 25.8 s. Open rejects the task. The open/frozen jaw guard idles rather than playing the arm path; this limitation is retained. Heavy does not increase deformable mass, and drop follows rigid carried objects, so their NOTE lines are not heavier-cord or release validation. The rigid-grip report does not measure the rod pinch; the material normals above do.

## Timing, feed and film

In the paired take, pickup closes at **5.98 s**, insertion ends at **8.36 s**, **6 feed strokes** finish at **19.52 s**, and the left catch closes at **21.24 s**. Lift-clear ends at **22.04 s**, draw at **23.96 s**, return at about 25 s, and the take at 25.8 s. The emergence condition produces five or six feeds across replays; the recorded count above is the film's actual count. The right hand stays open and unloaded at the entry during the handoff.

Measured paired-take feed travel along the channel:

| End time (s) | Hand (mm) | Tip (mm) | Right normals (N) | Flare clearance (mm) |
|---|---:|---:|---:|---:|
| 8.36 | 52.92 | 52.94 | — | 4.01 |
| 10.22 | 64.14 | 64.12 | 7.96 / 7.90 | 4.01 |
| 12.08 | 64.43 | 64.07 | 7.92 / 7.89 | 4.10 |
| 13.94 | 61.46 | 61.39 | 7.86 / 7.82 | 2.78 |
| 15.80 | 65.16 | 64.19 | 8.56 / 8.55 | 4.88 |
| 17.66 | 66.95 | 66.74 | 11.17 / 11.13 | 4.24 |
| 19.52 | 65.26 | 64.20 | 9.60 / 9.56 | 3.66 |

`run/hero.mp4` and `run/hero_wide.mp4` are two cameras of **one execution**, recorded with shared path-traced renderers in `capture.py`; this checkout's `play --also` selects renderer names, not camera names. `run/paired_capture_r5.log`, `run/paired_telemetry_r5.json` and `run/paired_verification_r5.json` preserve the outcome and matching times, steps, rigid positions and sampled material positions. Both films have 646 frames at 25 fps. Their identical metadata spans 0–25.78 s relative to recording start; the live simulation footer spans 0.02–25.80 s. One control tick elapsed between reset and recording, which also explains the film's slightly smaller corner displacement relative to its own starting pose. Physics measures from reset and bounds the full displacement at 0.642 mm. The footer displays measured coverage, box corner movement and mat tail.

The inspected frames include `run/hero_wide_r5_10.png`, `run/hero_wide_r5_21.png`, `run/hero_wide_r5_25.png`, `run/hero_r5_25.png` and `run/cmp_photo_r5_strip.png` / `run/cmp_wide_r5_strip.png`. Wide shows the entering segment and the raised left loop with retained mat tail. **Photo's native right upper link hides the feed and its held framing crops much of the final left wrist.** Wide should lead the gallery presentation; photo stays at the calibrated first-second pose rather than following the phone.

## Rest and held-end motion

The round-3 rest failure was measured at the hanging final state, not reset. The new reset-state check (`run/check_final_r5.log`) passes: **0.24 mm maximum drift over 150 steps / 3 s**, **0.04 mm worst overlap**, finite before and after. The camera/build lines pass; object lookup is skipped because no names were supplied. These reset readings, rather than hanging-state drift, judge the initial bake.

After filming, the robot holds the same targets for three seconds (`run/hold_final_r5.json`). The held material moves at most **0.116 mm** relative to the hand; tip-to-hand distance ranges **19.216–19.300 mm**. Connector centroid motion is **1.151 mm**. Left normal ranges are **9.89–10.02 N** and **9.91–9.95 N**. These readings distinguish free-end flex/swing from loss of the pinch. The page is reset before delivery.

## Retained calibration, materials and departures

The camera remains the **31-point grid-only solve, 3.037 px RMS** at 1920 × 1080 (`assets/camera_grid_r3.json`), avoiding overlays and using no box corners. The box fit (`assets/box_fit_r3.json`) is 480 × 260 × 110 mm, with a 17 mm bore and 380 mm covered run. The final initial keyed comparison (`run/compare_initial_r5.txt`, `run/cmp_initial_r5_beside.png`) reads box size 1.03×, width/height 1.10×/1.02×, centre 3 px left/2 px low, IoU **0.89**; cable size 1.33×, centre 94 px left/37 px low, IoU **0.00**. The assumed return loop occupies extra visible mat area: this first-state mismatch remains. Whole-frame dE 33.8 / PSNR 8.9 includes the burnt-in overlays. The mat, slats, glass and room are procedural, with no clip pixels, inset or captions.

The 1.413 m, 236-node cord and both colliding connectors remain the settled round-4 bake (`assets/cord_long_settled_r4.npy`, `connector_long_settled_r4.npz`, `tail_connector_long_settled_r4.npz`). The measured 25 mm grid supports approximately **4.5 mm cord diameter**. The complete off-frame length/return is assumed. The rod has a straight stress-free shape independent of its laid curve, density 1050 kg/m³, axial rigidity 2000 N, E = 10 MPa and friction 0.65. Each connector has a 6 mm neck and 30 mm barrel/sleeve, maximum diameter 12 mm, effective E = 2 MPa and density 1000 kg/m³. Uniform deformable SDF volumes approximate the distinct barrel and foam; the trailing connector's same dimensions are assumed. Manufactured rod-to-connector joins attach neither end to the robot or world.

The unchanged axial component probe (`run/axis_sdf_r3.json`) measured 27.035 mm hand / 27.039 mm tip travel, median sliding resistance 0.01910 N and p95 0.02269 N. SDF connector surfaces avoid the prior oversized contact-envelope jam. Rod mirror padding plus threshold represents its 2.25 mm radius. Direct LDLT with eight nonlinear iterations is retained.

**EI remains unvalidated against the clip:** the scene uses 2.013e−4 N·m². The conditional planar fit in `run/bend_fit_r3.json` gives 3.81e−6 N·m² but has condition number 1277, unknown depth/boundary loads and a moving phone. Its 368 mm traced arc is a scale-based comparison, not reliable stiffness validation.

The old benchmark supplied the box/channel construction and general plan. Its size claims were revised after grid calibration; its grasp locks and other non-contact shortcuts were not retained. Unresolved physical assumptions are a 750 mm bench, 1.20 kg box, exterior friction 0.65, channel floor/walls 0.30, lid 0.40 and flares 0.15. **The four 15 mm / 20° lead-ins are absent in the clip and are an insertion-assisting departure**; they collide and are lid-white. The separate white lid has a dark seam; exposed left top is black mesh.

The catalog YAM keeps its own file, gains, limits and fingers; only appearance/home poses differ. Base spacing is the measured 601.6 mm with supplied translations; small base rotations are omitted. Native fingers differ from the clip. The press is about 35 mm from the lid's left end, nearer the end than the clip's approximately quarter-length press. Right regrips are about 90 mm from the mouth versus roughly 40–50 mm in the clip, and five/six strokes replace roughly three/four. The right initial upper link remains higher and its hand more horizontal than the clip's folded-low/upright pose. The final left approach is diagonal, about 15° below horizontal; its explicit 100 mm return forms the raised loop. These are motion departures. Wiring, reflections, socket detail and white mains-lead shape remain approximate. The mains lead is colliding rigid geometry carried by the box, and the black cord is not electrically joined to the strip.

All runtime paths are anchored to the twin; output directories are created if missing. `assets/` contains only runtime bakes and cited calibration/provenance records. Historical attempts live under `run/`, including `run/accepted_r4/`. Nothing in the routine writes object pose/velocity, creates a grasp attachment or adds a finger collider. `run/final_status_r5.txt` records the reset delivery state.

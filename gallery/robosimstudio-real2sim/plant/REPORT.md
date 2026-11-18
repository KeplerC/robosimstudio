# Alocasia watering — round 2 revision

The return, mounting collision and large plant sweep are corrected. **The uninterrupted-stream visual requirement remains open:** the lip and upper stream are visible in profile, but the central leaf hides part of the lower fall and the MPM surface breaks into droplets. The six pour stills below document this limitation; they are not claimed to show an unbroken lip-to-soil stream.

Review files: [photo film](run/hero.mp4), [wide film](run/hero_wide.mp4), [photo pour stills](run/hero_pour_three.jpg), [wide pour stills](run/hero_wide_pour_three.jpg), [photo return 22–34 s](run/hero_return_sheet.jpg), [wide return 22–34 s](run/hero_wide_return_sheet.jpg). The return finishes at 42.5 s; the corresponding `*_return_finish_sheet.jpg` files show release and park.

## What changed

The active right arm now pinches the camera-side rim of the white cup, with one native finger inside and one outside. It lifts clear, levels the small pickup tilt, carries above the foliage while turning toward the pouring pose, and meters the pour about world X. The mouth faces +Y, toward the photograph's left, so the lip is seen in profile. The final lip target is world `(0.585, -0.100, 1.085)`; the pre-tilt is 45°, the carry 75°, and the four-second controlled pour ranges from 80° to 105°. As the angle grows past 85°, the lip rises by `0.08 × (cos(85°) − cos(angle))` metres, up to 27.7 mm at 105°, compensating for the wrist body’s descent during the roll. The controller reads particle counts and outflow rate and adjusts only the robot's wrist target. Nothing prescribes a water trajectory or changes a particle's pose during the routine.

The earlier return error was an IK branch error, not a deliberate alternative put-down point. The baseline trace showed about 140–150 mm of hand target error while the cup's position relative to the hand changed only about 2 mm (`run/round2/baseline_hand_trace.json`). The new return sends the measured outbound arm joint path backwards through the stock servos. It then opens the fingers, withdraws along the lift path, returns to `home_q`, and holds for two seconds. The take-specific hand JSON and return-tracking CSV files record commanded and measured hand positions throughout the return.

The right base moved to the image's left. Its `arms_at` pose is bench `(-0.22, 0.55, 0.302, 0, 0, -0.5, 0.866025403784)`, hence world translation `(0.38, 0.55, 1.052)`. The tabletop riser has bench centre `(-0.22, 0.57, 0.141)` and half extents `(0.052, 0.025, 0.140)`. Its bottom is the drawn tabletop at bench z=0.001; it occupies no apron, leg or under-table volume. The old floor stand intersecting the table is gone. Both bases and both parked hands are physically outside the held photo camera; the near plane remains 10 mm. The left arm stays upright on its separate pedestal outside the table.

The two canvas prints were restored from the rectified photographic prints with image inpainting, replacing the mirrored patches and lamp glare. Their obscured regions are reconstructed artwork, not recovered detail. Only the printed canvas faces use photo-derived pixels. The wall, table, chair, lamps, floor and room remain geometry with their own materials. `assets/teal_yellow_restored.png` and `assets/flower_restored.png` are the live prints. The wood tint is deeper reddish amber, its finish is glossier, and warm lights illuminate the wall below the pendants.

## Kept and assumed

The photo camera is unchanged: eye `(-0.50, 0, 1.18)`, look `(0.6, 0, 1.03)`, vertical FOV 38.5°, 1920×1280. The original seven settled blade meshes, UVs, vein and normal textures, upper stalk centrelines, pot and saucer dimensions, and calibrated room layout remain. The wide camera shows the robot, tabletop, cup and plant.

The initial composition probe rejected MuJoCo + MPM + VBD: the proxy coupler supports only two islands (`run/composition_probe.txt`). The permitted fallback remains on the robot's MuJoCo island: eight capsule segments per stalk on passive spring joints and a 1 mm shell over each old blade mesh on a passive hinge. Stiffness is 3 N·m/rad at roots, 1.5 at intermediate joints and 0.10 at blades; damping is 0.025 / 0.006, armature 1e-6, stalk density 550 kg/m³ and leaf areal density 0.15 kg/m². These are fitted assumptions, not measurements of the living plant. There are no plant motors or changing spring targets. A blade rotates as a shell; it does not bend internally like cloth. Plant self-contact remains disabled.

The catalog `yam_bimanual` is unchanged apart from placement and `home_q`. Its file, fingers, servos, torque limits and friction are the registered ones. The cup is 82 mm across and 90 mm high, with a 66 mm mouth, 14 mm base and assumed 120 g mass representing hollow double-wall ceramic. It retains the 24 rigid wall sectors, base collider and watertight liquid/render mesh. Dry contact friction is 0.8 and wet interior friction is 0.2. No attachment, weld, latch, added finger collider, object motor or object pose write carries it.

The 150 mm saucer scale, 750 mm table height, 36 mm table depth, 100 mm apron, 50 mm square legs and 320×340 mm chair seat at 465 mm above the floor remain modelling assumptions. The cup, robot, riser and watering motion are designed additions: the source photograph has no watering action, and the old Alocasia film is a photo-to-twin wipe.

## Liquid boundaries and measurement

The liquid is `rss.liquid(material="water")` under the unchanged `rss.LIQUID_120HZ`, with projection off. The initial cylindrical sample is 38 mm in diameter and 45 mm high, sampled at 1.5 mm, and settles for 600 control frames. Dose and capture use particle identities and counts; the MPM water preset's loss of apparent volume makes height unsuitable for measuring the dose.

The liquid-only closed boundary now joins pot, soil and saucer. Its lower plug lies inside the wooden table and its saucer outer wall is vertical, eliminating buried pot/saucer undersides and the lower bevel from the liquid query. The old thin saucer shell remains for rendering and rigid contacts. Four particle traces in the preceding trial reached the saucer edge and then the tabletop interior (`run/round2/saucer_contact_probe.json`); those failures are not counted as successful captures. The liquid margins are 10 mm on the cup, 16 mm on the pot/saucer union and 26 mm on the table. These numerical contact buffers are an approximation, not wall thicknesses.

The visible soil remains 15 mm below the old model's soil: bench z=124.5 mm, below the unchanged 145 mm rim. Root centres start at 126.5 mm so the 2 mm capsule ends meet that soil surface; upper stalk nodes retain the old shape. The liquid boundary omits internal soil faces, drainage notches and small ribs. Water does not absorb into soil or drain out of the pot.

Capture means an escaped particle ends within 58 mm of the pot centre, world z=0.870–0.910 m, outside the cup. The audit separately tracks identities crossing the pot wall below the rim, passing through soil/table solids, penetrating the soil surface, or going below the drawn tabletop at world z=0.751 m. Misses remain misses; no water is removed or projected into the pot.

## Final takes and particle audit

Both uninterrupted RTX takes use `--every 2` at 60 Hz: **30 fps**, 1920×1280 for `run/hero.mp4` and 1920×1080 for `run/hero_wide.mp4`. They include the full return, open-finger withdrawal and two-second parked ending.

| Measurement | Photo | Wide |
|---|---:|---:|
| Initial particles | 14670 | 14670 |
| Particles ever leaving cup | 14670 | 14670 |
| Final particles captured inside pot | 12870 | 12732 |
| Particles returned in cup | 10 | 8 |
| Through table / pot / soil | 0 / 0 / 0 | 0 / 0 / 0 |
| Ever below drawn tabletop | 0 | 0 |
| Maximum soil surface penetration, mm | 0.000 | 0.000 |
| Put-down error, mm | 8.32 | 8.13 |
| Cup upright cosine | 1.000000 | 1.000000 |
| 10–90% discharge interval, s | 2.53 | 2.50 |
| Capture of all escaped identities | 87.730% | 86.789% |

The put-down target is the measured initial cup position, not an alternative location.

- Photo: target `[0.3800010681152344, 0.3000006675720215, 0.750993013381958]`, final `[0.3816014230251312, 0.308163583278656, 0.750993013381958]` (world metres).
- Wide: target `[0.3800010681152344, 0.3000006675720215, 0.750993013381958]`, final `[0.38152527809143066, 0.307989239692688, 0.7509930729866028]` (world metres).

The take-specific `*_audit.json`, `*_phases.json`, `*_flow.json`, `*_hand.json`, `*_contacts.json`, `*_return_tracking.csv`, `*_pool.json`, and `*_source_sha256.json` accompany each movie. The CSV samples the commanded and measured hand and cup positions through the entire return; the JSON retains every control step.

| Leaf | Photo peak tip, mm | Wide peak tip, mm | Photo peak stalk junction, mm | Wide peak stalk junction, mm | Photo final tip, mm | Wide final tip, mm |
|---|---:|---:|---:|---:|---:|---:|
| leaf_back | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 |
| leaf_left | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 |
| leaf_front_left | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 |
| leaf_middle_right | 20.59 | 20.52 | 9.66 | 9.70 | 0.000 | 0.000 |
| leaf_right | 7.00 | 6.79 | 10.03 | 9.85 | 0.000 | 0.000 |
| leaf_front_right | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 |
| leaf_edge | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 |

## Final checks

`./ctl check --camera photo --json`:

```text
PASS  builds       the last rebuild landed
PASS  finite       every number in the state is finite
PASS  camera       photo 1920x1280, fov 38 deg
SKIP  objects      no names given (--object or --key), so nothing was looked for; the scene has 45 things
PASS  penetration  worst overlap 0.01 mm between tabletop and cup (bar 1 mm)
PASS  rest         nothing moved more than 1.98 mm in 60 steps (bar 2 mm); not measured (no position to read): table_leg_0_1, lamp_left, table_apron_x0.26, table, chair, soil, table_apron_x-0.26, table_leg_0_0, right_mounting_plate, lamp_left_cord, canvas_right, canvas_left, room_floor, lamp_right_inside, pot, clock, wide, lamp_right, lamp_left_inside, chair_seat, table_apron_y-0.61, chair_back_post_0, wall, wall_return, photo, chair_leg_1_0, chair_leg_1_1, table_leg_1_1, table_apron_y0.61, wall_opposite, lamp_right_cord, wall_below_photo, left_pedestal, tabletop, table_leg_1_0, chair_leg_0_0, chair_back_post_1, pot_liquid_boundary, table_liquid_boundary, chair_leg_0_1, saucer
PASS  finite_after still finite after 60 steps
```

All seven standard replays were rerun: written, frozen, open, built, heavy, drop and plain. `run/round2/physics_final.json` contains the model, every run and the verdict. Exact verdict:

```text
PASS  contact  nothing ties a thing to the robot, no motor on a thing's joints, nothing welded
PASS  robot    the registered yam_bimanual, its own file, hands and servos
PASS  model    gravity 9.81 m/s²; the handled things' densities are ones objects have
PASS  settle   nothing starts inside anything (deepest 0.0 mm); the scene's settle moved nothing more than 0.0 mm
PASS  written  finished; moved cup, water
NOTE  frozen   with the robot frozen no rigid thing moved, but water drifted 2.2 mm: still settling
PASS  open     with the fingers held open the task did not happen (RuntimeError: Cup did not lift in the fingers)
PASS  built    finished with every friction at most 1 and the arm's torque at 27/10 Nm, the things ending where the written run left them
NOTE  heavy    with the rigid handled things 2 times as heavy (water: not) the routine stopped: RuntimeError: Water crossed a supporting surface (the grasp's margin)
PASS  grip     the fingers pressed cup 109.3 N (p95 112.2) by hand 2; the solver used friction 1 to 1 at the finger contacts
PASS  drop     with the fingers opened at the top of the carry (step 725) cup fell
PASS  plain    played with nothing watching, the things ended where the replay left them
```

`./ctl compare --camera photo --ref ref/plant_a_unsplash-Mfjb8E7LjWQ.jpg`:

```text
no --key: the pair and the look sheet are written; key a thing that keys cleanly to read its size, place and overlap
picture: colour difference 11.7 (mean CIE76 dE) over the frame, 11.7 behind the keyed things; PSNR 17.0 dB, 17.0 dB behind them (identical pixels read dE 0)
```

The held-camera comparison was dE 13.5 in round 1 and 12.5 at round 2 review. The final value above is from this delivered source, including the new prints, deeper wood tint and warm lamp pools. Files judged: `run/round2/compare_final_beside.png`, `compare_final_look.png`, and `photo_final_vs_ref.png` from the required `snapshot --beside`. PSNR is 17.0 dB versus 17.3 dB at review, so the lower mean colour difference does not imply every image metric improved. It remains an approximation: leaf outlines are faceted, the pot ribs and chair differ, and the photograph’s soft exposure and wall gradients are not exact.

## Cup-only drainage probe

The isolated cup starts fixed at 115°, with the final fill and wet friction, under `LIQUID_120HZ`. It has no moving hand and no scripted flow.

| Time | Particles left in cup | Fraction |
|---|---:|---:|
| 1 s | 0 / 14670 | 0.00% |
| 2 s | 0 / 14670 | 0.00% |
| 4 s | 0 / 14670 | 0.00% |

First exits below the upper 15 mm of the cup: **0**. Evidence: `run/round2/cup_115_final.json`, `cup_probe.py` and `cup_probe_console.txt`. The old square-cup probe retained 13.07% after one second and 10.41% after four.

## Gallery layout

The folder contains the runtime code at top level, 32 live mesh/texture/baked-data files in `assets/`, the source photograph in `ref/`, and evidence and scratch work in `run/`. The runtime asset is `assets/table_veneer.png`; there are no hash-suffixed asset duplicates. Cache output goes to `run/cache`, and all generated assets and measurements resolve from the twin’s `ROOT`. No runtime code reads `/inputs/old` or another twin. `__pycache__` directories were removed. The eleven root Python files pass scoped Ruff checks and formatting.

## Frames and contact review

For each of `hero` and `hero_wide`, `run/<take>_pour_1.png`, `_2.png`, `_3.png` are unaltered full-resolution frames at **18, 19.5 and 21 s**. `run/<take>_pour_three.jpg` puts them together. `run/<take>_pour_sheet.jpg` covers 13–22.5 s; the individual half-second frames are in `run/round2/`. `run/<take>_return_sheet.jpg` covers **22–34 s**, and `_return_finish_sheet.jpg` covers **34–42 s**, including set-down, release, withdrawal and park. The whole movies decode without FFmpeg errors.

`hero`: the blade-surface intersection test found **0 crossing frames in 661 control steps**, 13–24 s. Non-finger contacts with cup, pot, saucer or table: **0**. The wrist brushes the central leaf; maximum recorded robot–leaf force is 1.277 N, numerical contact overlap 1.619 mm. The cup moves up to 9.23 mm relative to the hand during the carry/tip/return: some settling or slip remains in the contact grasp. Immediately before resuming the Cartesian cursor at set-down, hand error against the last commanded joint pose’s FK target is 10.416 mm. The phase-mark cursor error becomes zero when resume resets the cursor to the measured hand; that zero is not evidence of perfect tracking. The residual cup placement error is 8.32 mm, within the requested 20 mm.

`hero_wide`: the blade-surface intersection test found **0 crossing frames in 661 control steps**, 13–24 s. Non-finger contacts with cup, pot, saucer or table: **0**. The wrist brushes the central leaf; maximum recorded robot–leaf force is 1.281 N, numerical contact overlap 1.626 mm. The cup moves up to 9.11 mm relative to the hand during the carry/tip/return: some settling or slip remains in the contact grasp. Immediately before resuming the Cartesian cursor at set-down, hand error against the last commanded joint pose’s FK target is 10.341 mm. The phase-mark cursor error becomes zero when resume resets the cursor to the measured hand; that zero is not evidence of perfect tracking. The residual cup placement error is 8.13 mm, within the requested 20 mm.

The full-resolution 13–19 s frames were inspected for blade/hand crossings. Image overlap with the arm is largely occlusion; the logged wrist contact bends the blade and its stalk, and both return. Self-contact is still disabled, so the zero blade–blade result is an observed result for these takes, not a general self-collision guarantee.

## Pool, flow and misses

| Measurement | Photo | Wide |
|---|---:|---:|
| Initial particle mass, g | 50.840 | 50.840 |
| Captured below the rim | 12308 | 12123 |
| Below-rim capture / initial dose | 83.899% | 82.638% |
| Median captured-particle clearance below rim, mm | 5.120 | 5.143 |
| 95th percentile of below-rim pool: clearance, mm | 3.579 | 3.504 |
| Captured-footprint particles above rim | 562 | 609 |
| 5–99% discharge interval, s | 3.700 | 3.683 |

The bulk pool sits below the rim, but isolated beads on stalks and at the rim rise above it. The rendered surface can still look like a pale disc or rounded bead: this is a visible departure. The MPM preset compresses water; its mass and particle identities, not its drawn height, measure the dose. Missed particles are retained in the simulation and remain counted as misses. The table contact margin can leave those small beads visibly above the wood. No particle is deleted or teleported to improve capture.

## Replay variation

The routine audits below belong to the ordinary and load-test modes. Frozen, open and drop are deliberate task-breaking interventions; their verdicts are in the exact seven-mode report above.

| Replay | Captured / ever left | Capture | In cup at end | Through table / pot / soil | Below drawn tabletop |
|---|---:|---:|---:|---:|---:|
| written | 12746 / 14670 | 86.885% | 11 | 0 / 0 / 0 | 0 |
| built | 12811 / 14670 | 87.328% | 12 | 0 / 0 / 0 | 0 |
| heavy | 12965 / 14670 | 88.378% | 9 | 1 / 0 / 0 | 1 |
| plain | 12905 / 14670 | 87.969% | 11 | 0 / 0 / 0 | 0 |

Written captured 12746 particles and plain captured 12905, a difference of 159 particles (1.084% of the initial dose). The wrist angle is count-controlled, so MPM nondeterminism changes both the measured flow and the resulting small angle corrections. These are repeated executions of the same code, not prescribed water motion. The final plain verdict is PASS. The counts quantify the remaining particle-level variation despite that passing endpoint check.

The doubled-mass stress audit must also be read separately from the generic grasp verdict: its boundary counts are shown above. A nonzero count there is a failed liquid-containment result for that stress run, even if the cup itself returns upright.

The count/deflection plot is [final_flow_and_leaf.png](run/final_flow_and_leaf.png), with an exportable PDF beside it. The images below are the three-moment pour sheets; the lower-stream interruption is visible.

![Photo pour](run/hero_pour_three.jpg)

![Wide pour](run/hero_wide_pour_three.jpg)

The final frozen NOTE is 2.2 mm of water drift after the 600-frame initial settle; rigid objects stay still. The heavy NOTE is a real liquid-containment failure: one particle tunnels through the table with the cup doubled in mass (the water mass is unchanged). The ordinary photo, written and built runs have zero such crossings; the wide and plain counts are recorded separately above. Native finger contacts solve at friction 1, as the grip line says, despite the cup surface setting of 0.8. The stock-limit replay caps every contact at 1 and still finishes.

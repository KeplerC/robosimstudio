# Brush RSS — round 3

This revision shortens the return, restores visible powder grain, and refines the workshop. Final source: **`4fea04d6a9`**. Both replacement path-traced films complete the routine in **26.317 simulated seconds**, including a **9.500-second return**. Both film-end checks pass `rest` at **0.14 mm** maximum movement over 60 steps. The handle ends about **0.40 mm** from its start, with the bristles on the rest. The complete seven-mode final physics audit passes on this source; its grip-force line is a NOTE.

The panel, rest, brush, letters and writing plan are retained: 170 × 320 × 10 mm panel, 60 mm square rest, 22 × 22 × 60 mm handle, 7 × 7 nylon bristles, 20 mm bristle length and 0.09 rad flare. `letters.py` keeps the original centrelines and 14 mm target width. Scoring uses the original panel window, 3,233-cell target, per-letter boxes, and measured clearing after the final lift-off. Writing remains at 120 mm/s with the same grasp, 4 mm press and inter-letter lifts. The port's 10 mm tangent lead remains a departure from the old take's 8 mm, compensating the lag on this checkout.

The robot is the unchanged catalog `rss.robot('yam_bimanual')`, positioned with `at=` and `arms_at=`. No model, finger collider, gains, torque or force limits were altered. The handle moves only through contact. `surface.FineBrush` supplies the instance's `ScrubGrid.around(..., reach=.4, cell=.002)`; the brush's dynamics and measurement kernel are inherited. No library function is patched.

The return now uses a 2.4-second jaw ramp, concentrated around the 22 mm handle width. Seating corrections and preload relief end when the measured handle stays below 0.005 mm translation per tick for 20 consecutive ticks. After opening, the same quiet condition also requires a measured jaw width of at least 30 mm. Only then does the hand lift vertically 100 mm at 100 mm/s and move aside. Two one-second windows of bristle-body movement below 0.5 mm are measured during this withdrawal. The last two seconds include the vertical lift, movement aside and any remaining fraction of the stability window.

| Return milestone | Simulation time, both films |
| --- | ---: |
| Letters clear; return begins | 16.817 s |
| Hand over rest | 17.967 s |
| Seating complete | 20.333 s |
| Preload relieved; opening begins | 21.583 s |
| Jaws clear and handle quiet | 24.317 s |
| Vertical withdrawal complete | 25.317 s |
| Hand aside; routine finished | 26.317 s |

The old film's `brush_reveal` return spans **sim 15.2–22.2 s, about 7.0 s** (`ref/old/v3_notes.json`). The new return spans **16.817–26.317 s, 9.500 s**, a **2.5 s increase** over that old segment. It is substantially shorter than the reviewed 45.75-second take. The fixed macro still holds on completed RSS while the brush returns outside its view; it does not reproduce the old moving reveal edit.

The brush is first seated with a 2.2 mm compression target. The hand then eases the bristle preload through contact before opening, targeting the handle 0.3 mm below its initial position. Leaving the preload compressed caused continued bristle rearrangement. Precision return commands hold the arm cursor so the page's arrival tolerance cannot cancel these small corrections. These are robot commands, never object pose writes. The brush kind's handle gravity compensation is unchanged: VBD does not return bristle support forces to the MuJoCo handle, so the kind compensates its weight. The hand consequently does not feel the bristle press or drag. Small released-handle drift remains measurable; this is not an uncompensated-weight calibration.

**The dust is drawn from the brush kind's own measurement, `brush.scrub_grid()`.** Neither renderer reads the target letter mask. The dust is a visual height field over the solid contact panel, not simulated granular powder. It adds no physical bodies or forces and writes no simulation state.

The shared-vertex dust mesh has 121,693 vertices and 241,920 triangles at **3× lattice resolution, 0.667 mm spacing**. Gaussian smoothing has **sigma 0.70 mm**, nominal cutoff **1.40 mm**, and discrete kernel radius **1.33 mm**. The compact smooth transition rolls cleared edges, and the outer edge rolls over 1.4 mm. Cleared regions lie 0.1 mm below the panel; intact dust is nominally 1 mm thick, with texture relief up to **±0.30 mm**.

Texture mapping is corrected: the original **1280 × 680** image's **u axis follows the panel's 320 mm Y direction**, and **v follows its 170 mm X direction**. This gives an undistorted full-panel mapping at approximately 0.25 mm per source pixel. Height is sampled bilinearly from the original image, without resizing or averaging it first. Colour uses the same UV mapping. Normals combine the geometric height gradient with independent, seeded noise at roughly **2–4 mm feature size**: a difference of Gaussian filters with sigmas 0.8 and 2.0 mm, normalized to a 0.12 mm base bump, then differentiated with **strength 4 × bump gain 0.9**. This follows the old film's strong fine bump treatment; the noise changes normals only and cannot change the cleared footprint.

The final macro's surface measurement finds at most **0.67 mm extra clearing** and **1.33 mm remaining dust into measured clearing**, both within one 2 mm cell, excluding the outside rim. Maximum height above the panel is **1.30 mm** (`run/dust_surface_measurement.json`, source `4fea04d6a9`). Success is still scored on the unsmoothed lattice. `dust_render.DustRTX` draws the textured surface in the path tracer; the browser free view uses the same geometry with flat colour through `page_dust.DustPanel`.

The workshop is entirely geometry with its own textures: worn wood, concrete, painted plaster, cardboard and pegboard under `assets/`. There is no HDRI, photograph, clean plate or projected room image. The beam remains 3 mm below the bench top, boxes sit on their shelf, and the pegboard is in frame. The door now has a 120 mm lever of 20 mm thickness, about 64 mm proud of the door, on a small plate. The tools are a hammer, red and blue screwdrivers, an open wrench and a graduated square. Ambient was raised from 0.20 to **0.26**, and key intensity from 2.0 to **2.4**, retaining directional shadows. The local RTX renderer orients its 0.55 × 0.40 m emitter along `light_direction=(-.35,.45,-1)` because the stock LAB emitter does not consume that setting.

The macro retains the old final eye and aim with the port's 28° field of view. Wide framing remains unchanged from the accepted revision: eye `(1.65,-.65,1.40)`, aim `(.40,-.04,1.03)`, 46° vertical field of view. Both whole arms stay in frame in the sampled phases. This remains a newly designed workshop, with different room geometry and lighting from the old photographed set. The old moving edit, slow motion, depth of field and dashed stroke guide are omitted.

Both new films are **1280 × 720, 528 frames at 20 fps** (26.4 seconds encoded; routine duration 26.317 seconds), recorded with `dust_render.DustRTX`. Their logs, saved scores, return measurements and checks all use **`4fea04d6a9`**.

| Reading | `hero` (macro) | `hero_wide` |
| --- | ---: | ---: |
| Raw lattice IoU | 0.842504 | 0.842458 |
| R | 0.871476 | 0.871476 |
| S1 | 0.831978 | 0.831075 |
| S2 | 0.821747 | 0.822480 |
| Excess / missed cells | 202 / 339 | 201 / 340 |
| Handle offset from start | 0.403 mm | 0.405 mm |
| Movement after jaws cleared | 0.128 mm | 0.162 mm |
| Bristle drift in withdrawal windows | 0.091, 0.146 mm | 0.095, 0.087 mm |
| End-state `rest`, 60 steps | PASS, 0.14 mm | PASS, 0.14 mm |
| Detected penetration | 0.00 mm | 0.00 mm |
| Record / check log entries | 30 / 31 | 32 / 33 |

The actual end-state check line for each film is: `PASS rest nothing moved more than 0.14 mm in 60 steps (bar 2 mm)`. Builds, finite state, camera, objects, penetration and finite-after also pass. These checks were run directly after their respective films, before resetting. Files: `run/hero_end_check.json`, `run/hero_wide_end_check.json` and canonical `run/check.json`; scores and return readings are `run/hero_score.json`, `run/hero_return.json`, `run/hero_wide_score.json`, `run/hero_wide_return.json`, with each take's measured mask saved separately.

I inspected `run/hero_contact.png`, `run/hero_final.png`, `run/take_sheet.jpg`, and `run/hero_wide_final.png`, and compared the grain with `run/old_macro.png`. RSS reads clearly in the final macro, the brush casts a shadow while writing, the dust has visible grain, and the final wide frame shows the bristles on the rest. The take sheet shows the shortened release and full arm framing. These current frames are displayed on the page.

The final audit command was the full `./ctl physics routine`, after both films and their canonical end-state checks. **All seven replay modes pass their audit criteria** on **`4fea04d6a9`**; grip is a NOTE. `run/physics.json` is the complete output, matched by log entry **34**, not a partial rerun. Its report follows, abbreviating only the written-run internal body list.

```text
PASS  contact  nothing ties a thing to the robot, no motor on a thing's joints, nothing welded; 147 stiff joint(s) on things held at their targets (their elasticity)
PASS  robot    the registered yam_bimanual, its own file, hands and servos
PASS  model    gravity 9.81 m/s²; the handled things' densities are ones objects have
PASS  settle   nothing starts inside anything (deepest 0.1 mm); the scene's settle moved nothing more than 0.0 mm
PASS  written  finished; internal brush-body list omitted here
PASS  frozen   with the robot frozen nothing moved more than 0.1 mm
PASS  open     with the fingers held open the task did not happen (RuntimeError: brush not lifted: 0.0 mm)
PASS  built    finished with every friction at most 1 and the arm's torque at 27/10 Nm, the things ending where the written run left them
PASS  heavy    finished with the handled things 2 times as heavy, the things ending where the written run left them
NOTE  grip     the fingers pressed handle 100.2 N (p95 100.3) by hand 2; brush (stepped by SolverVBD): not touched by the fingers, its force not read (its contacts are not in MuJoCo's list); the solver used friction 1 to 1 at the finger contacts
PASS  drop     with the fingers opened at the top of the carry (step 1063) handle fell
PASS  plain    played with nothing watching, the things ended where the replay left them
```

The 147 stiff joints are the brush's own elasticity. The measured 100.2 N median grip is the shipped hand's force, with no force override. The frozen replay deliberately cannot reach its commanded hand target; its criterion is that the brush stays still. The open-finger replay cannot lift the brush. The drop replay opens at step 1063, after writing, and the handle falls; the routine then raises `handle did not become quiet with the required clearance`. Its PASS is a successful release control, not completion of the writing-and-return routine with a forced-open hand. Written, built, heavy and plain all finish at 1,579 steps. The brush-kind compensation limitation described above still applies.

After the audit’s automatic reset, the normal routine was replayed once to leave the page **paused at completed RSS, with the brush on its rest**. That display replay finishes in 26.317 s, with a 9.500 s tail, **0.402 mm** handle offset and final bristle drift windows of **0.093 and 0.098 mm** (`run/handoff_return.json`, source `4fea04d6a9`). The last audit remains the complete seven-mode result in `run/physics.json`; `run/check.json` remains the wide film’s canonical end-state check. `run/audit_manifest.json` links the current source and log entries. Previous handoff readings are archived under `run/previous_round_2/`.

Inherited material values remain design assumptions, tested by contact and coverage rather than fitted from a real photograph: handle mass 29.04 g, density 1000 kg/m³ and friction 1; nylon bend stiffness 0.01 N·m/rad, bend damping 0.0002 N·m·s/rad, stretch stiffness 10,000 N/m, stretch damping 0.1 N·s/m, friction 0.6, density 1150 kg/m³, bristle radius 0.75 mm and four segments per bristle. None was retuned.

Reproduce from this folder with `./ctl reset`, then `./ctl play routine --record hero --camera macro --every 3 --wait`; repeat from reset using `--record hero_wide --camera wide`. Run `./ctl check` at each routine's end, then the full `./ctl physics routine` without `--only`. All runtime assets and references are local under `assets/` and `ref/old/`; no code reads the old input directory. No library or tests were changed.

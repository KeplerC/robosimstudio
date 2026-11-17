# Workshop brush — round 3

The revised side view shows the bristles below the wrist and keeps the idle arm out of the foreground. The routine now returns the right arm to its folded start pose after releasing the brush. Both takes were re-filmed with the path tracer and inspected at the start, during the strokes and after the final hold.

The main physical correction is the negative-x stroke endpoint. The old path overlapped the **visible base foot**, although the catalog's smaller collision geometry did not detect it. The endpoint is now 25 mm shorter. This deliberately reduces coverage from the round-2 94.69%; the new measured result remains above the requested 80%. The brush, patch, START, robot bases, photo camera, grasp, positive-x endpoint and stroke line are unchanged.

## Final results

| Run | Scrubbed | Maximum slip | Return error | Final tilt |
| --- | ---: | ---: | ---: | ---: |
| Photo film | 84.27% | 1.64 mm | 1.78 mm | 0.057° |
| Wide film | 84.27% | 1.63 mm | 1.77 mm | 0.053° |
| Written replay | 84.17% | 1.64 mm | 1.78 mm | 0.048° |
| Robot as built | 84.27% | 1.64 mm | 1.79 mm | 0.059° |
| Double mass | 83.96% | 1.75 mm | 1.95 mm | 0.066° |

These are fresh measurements, not the inherited round-2 readings. The routine takes 1917 control steps, 31.95 simulated seconds. Slip is measured in the actual right-hand frame every control step. Return error is the final handle displacement from its starting position. Every completed task run in the table meets the coverage, 5 mm slip, 30 mm return and upright limits.

The right hand pinches the narrow sides of the D grip's top bar, lifts, makes six alternating pressed strokes, returns, holds still and releases. It then lifts clear, reverses the approach, folds to its original joint pose and holds for **1.2 seconds**. The final photo frame contains the brush alone, as the first frame does. The final state remains on the page.

## Clearance from the robot

`run/clearance_probe.py` makes a separate, never-stepped MuJoCo distance-query model from the shipped geometry. At **every control step**, it reads the live Newton transforms and computes signed capsule/convex-shape distances. It checks all bristle capsules and handle colliders against robot base/link colliders. The static visible base meshes are checked separately as conservative convex hulls. The shipped brush bristles do not collide with the robot, so this external clearance check is necessary. No robot or brush state is changed. Positive values mean separation; negative values mean overlap. Intentional finger contact is separated from unwanted base/link contact.

The old path (`run/round3_clearance_before.json`) had **−17.57 mm bristle-to-foot** and **−18.14 mm handle-to-foot** clearance against that visible hull. Its collision-only clearances were positive, explaining why the earlier physics checks passed. Shortening the negative endpoint from `dx=-0.075` to `dx=-0.050` clears the foot without changing the registered robot or its colliders.

The final written replay (`run/round3_clearance_final.json` and its per-step `.npz`) reads:

| Pair | Minimum signed clearance |
| --- | ---: |
| Bristle capsules / robot colliders | 43.69 mm |
| Bristle capsules / visible base-foot hulls | 7.37 mm |
| Handle colliders / robot links, excluding fingers | 39.44 mm |
| Handle colliders / visible base-foot hulls | 6.80 mm |
| Handle / fingers, intended grasp contact | -1.97 mm |

The negative finger value is the existing compliant pinch contact, not base overlap. The retained fine-fibre skins can extend 5.2 mm beyond the capsule chains; the final capsule-to-foot clearance also exceeds that visual allowance. The collision probe was restored after the written replay, so normal routines and the physics replays do not carry this diagnostic observer.

## Cameras, surroundings and light

The original photo `CAMERA` in `workshop_layout.py` is unchanged. The wide camera is **1280 × 960**, eye `(-0.35, 0.10, 1.60)`, target `(0.66, -0.065, 0.91)`, vertical field of view **25°**. Its elevation is approximately 34°, and its eye is 0.70 m horizontally from the idle left base. This frames the task from the side, rather than looking along the wrist.

Recorded segmentation measures brush widths of **306, 346 and 304 px** at the beginning, middle and end, respectively (23.9%, 27.0%, 23.8%), in a **1280 px frame** (`run/round3_brush_pixels.json`). The 20% framing requirement is 256 px. The side view shows the bristles at the paving contact; the robot base and forearm do not sit over the patch in projection. Paving and wall fill the background, with no grey void or bench sides.

The grey band now continues along both wall extensions. The skirting extensions are copies of the original board dimensions, orientation and finish, so their texture mapping matches the original run. The original terrace, wall and paver geometry remains, and no photograph is used as a background or surface texture.

The warm directional sun is **1800** under the unchanged **3000** dome, between the previous 900 and original 9000. The rendered photo view again has a brush shadow and warm right-edge illumination; the wide plaster retains visible relief. The warm sun direction and photo camera remain unchanged.

The round-2 review gate read **dE 15.7**, not the 15.3 quoted in the previous report. Fresh `./ctl compare` readings against `ref/workshop_a_unsplash-mHdaRVIBq0g.jpg` are **dE 15.0 initially** and **dE 15.0 in the final state**, with PSNR 15.5 / 15.5 dB. See `run/round3_compare_start.json` and `run/round3_compare_end.json`, their beside/look images, and `run/round3_photo_start_vs_ref.png`.

## Dirt and the paint hook

**The grime is drawn from the brush kind's own `scrub_grid` measurement.** The decal lattice uses the actual entity's `origin`, `cell`, `nx` and `ny`. It is not a separately positioned grid. Only a cell marked by a bristle tip is removed; lowering that display decal below the paver reveals the original textured relief. Cleaned cells are never repainted beige.

The initial dirt is an irregular connected blob formed from a smooth two-dimensional noise field, with approximately 20–40 mm features. The same field controls its brown tint and alpha, fading gradually at its edge. There are no random single-cell holes or opaque rectangular borders. `assets/grime_blob.png` and `assets/grime_field.npz` contain this procedural field and aggregate-based appearance; no reference pixels are used. Recessed joints remain uncovered. The measured coverage denominator still includes the entire 200 × 120 mm patch, independently of the initial dirt distribution.

**The paint hook touches decals only.** It launches one Warp kernel that reads the kind's device lattice and writes `body_q[decal_id]` for each display anchor. It never reads back or writes the whole body-pose array, and it never writes a handle, bristle or robot row. Reset also uses that indexed kernel. The private `_grid_device` gives the kernel the same lattice whose public `scrub_grid()` is read for reporting. The anchors are massless, fixed to the world and noncolliding; they cannot support or move the brush. This is contact-coverage visualization, not physical dirt transport or abrasion.

## Checks and evidence

`./ctl check` passes for the final scene (0.00 mm overlap and 0.45 mm drift in the 180-step rest probe); the complete reading is in `run/check.json` and `run/round3_check_console.txt`. The build is clean, the state is finite, and the scene uses the unchanged registered `yam_bimanual`, fingers, gains, force limits and frictions at most 1.

The full final physics suite reports **passed: true** in `run/physics.json`. Only its long moved-body list is abbreviated:

```text
PASS  contact  nothing ties a thing to the robot, no motor on a thing's joints, nothing welded; 960 stiff joint(s) on things held at their targets (their elasticity)
PASS  robot    the registered yam_bimanual, its own file, hands and servos
NOTE  model    brush: drawn parts that do not collide reach 5.2 mm outside its colliders: say what they are
PASS  settle   nothing starts inside anything (deepest 0.0 mm); the scene's settle moved nothing more than 0.0 mm
PASS  written  finished; moved handle, brush and its segments (body list abbreviated)
NOTE  frozen   with the robot frozen no rigid thing moved, but brush#619 drifted 4.1 mm: still settling
PASS  open     with the fingers held open the task did not happen (RuntimeError: Brush did not lift in the fingers)
PASS  built    finished with every friction at most 1 and the arm's torque at 27/10 Nm, the things ending where the written run left them
PASS  heavy    finished with the handled things 2 times as heavy, the things ending where the written run left them
NOTE  grip     the fingers pressed handle 100 N (p95 100.1) by hand 2; brush (stepped by SolverVBD): not touched by the fingers, its force not read (its contacts are not in MuJoCo's list); the solver used friction 1 to 1 at the finger contacts
PASS  drop     with the fingers opened at the top of the carry (step 398) handle fell
PASS  plain    played with nothing watching, the things ended where the replay left them
```

The model note concerns retained fine fibre skins, the frozen note concerns elastic bristle settling with the rigid handle stationary, and the grip note is the measured solver force/friction. The named-handle load-coupling limitation below remains relevant to the drop result.

Evidence inspected:

- `run/hero_wide.mp4`, `run/hero_wide_review.jpg` and the extracted first, mid-stroke and last frames; `run/round3_wide_start.png`, `run/round3_wide_mid.png`, `run/round3_wide_last.png`; `run/round3_wide_before_after.jpg`.
- `run/hero.mp4`, `run/hero_review.jpg` and the first and last frames; `run/round3_photo_before_after.jpg`. The original close photo crop includes only part of the forward patch; the wide view shows the full scrub.
- Each take's `.json` and `_seg.npz` contain frame timing and segmentation. Both films decode successfully. Their encoded duration differs slightly from simulated duration because the recorder samples the routine.
- `run/round3_clearance_before.json`, `run/round3_clearance_final.json`, `run/round3_brush_pixels.json`, the two compare JSONs and `run/physics.json` contain the quoted measurements.

Ruff lint and format checks pass. No library test tiers were run for this twin. The brush colour now reads `p["color"]` directly. Unused previous dirt assets and stale content hashes were removed, and the top-level bytecode cache was removed at handoff. The geometry, motion and hook changes were batched into one rebuild. A subsequent lighting-only correction lowered the sun after the first wide render revealed clipping. The two loads took 124.0 and 94.2 seconds, respectively.

## Retained design and remaining differences

The original LAYOUT, brush/patch placements, 180 × 90 × 12 mm block, 71 mm D grip, 30 mm bristles, paver sizes and textures remain. The brush is `rss.brush(handle="handle")`, nylon `BRUSH_60HZ`, with 320 bristles and four segments each. The named handle has the block collider and 18 convex D-grip sections, preserving its opening. The total estimated mass remains 204 g (54.1022 g handle plus approximately 149.8978 g strips/bristles). The photos do not establish mass, stiffness, friction or timing; nylon is a surrogate for natural tan fibre. No object is attached to the robot, and only the fingers' pinch carries it.

**Compatibility adapters:** `rss.brush` has no yaw. `workshop_brush.py` temporarily patches `EntityBrush._sole_xform`, `build` and `apply_display` during construction to preserve the photo yaw/sole height and retained fine-fibre/block-mark appearance. `workshop_rest.py` writes `_rest_body_q`, `_rest_joint_q` and `sim.settled_body_q` only at construction to restore the physically settled initial state consistently with that yaw; it does not move the brush during the routine. `assets/settled.npz` is the 420-step physical bake for this checkout; the old archive is provenance only. Separately, `workshop_assets.py` temporarily patches `EntityStatic._add_box` for the retained worn paver relief. All method patches are restored after building; no library file is edited.

**Named-handle coupling:** the shipped brush kind gravity-compensates its named handle. VBD does not return strip-joint reactions to the MuJoCo handle, so the robot does not feel the full bristle load or scrub drag. Release therefore does not validate realistic full-weight support. This is the prompt's prescribed named-handle design, not a latch or extra force introduced by the twin. The nominal 4 mm press is a displacement, not a calibrated normal force.

Some of the brightest plaster grains still clip: 12.8% of pixels in the lit-wall sampling rectangle reach RGB 255, while its median is 236 and its relief remains visible (`run/round3_plaster_highlights.json`). The parked working arm is partly cropped at the wide frame’s right/top edge; the idle arm is outside this task view. Remaining differences include regular tuft geometry, simplified wear marks, approximate wood/plaster detail and lighting, and designed terrace geometry beyond the original photo crop. Fine fibre skins follow 320 colliding chains rather than 11,520 independently colliding hairs. Dirt removal remains quantized to 5 mm measured cells at the newly cleaned edge. The safer stroke gives less coverage than round 2, as quantified above. The old film was a still comparison; the scrub motion is specified by the requested chore rather than recovered from that film.

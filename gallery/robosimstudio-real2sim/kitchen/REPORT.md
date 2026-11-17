# Kitchen sweep — response to review round 3

The wall/tile sampling and the uninterrupted physics audit are corrected. **The side residue line remains unresolved.** Two added cleanup strokes retained about 90% but left the line, so the review's guard was triggered: the original `sweep.py` was restored byte-for-byte. No second stroke is in the delivered routine. Both final takes were regenerated with that motion and the retained material correction.

`routine` remains the contact-only bimanual sweep with the registered `yam_bimanual`, its standard mounts, stock fingers, gains and force limits. Only `home_q` sets the parked first pose. The left hand pinches the pan's back wall with one finger inside and one outside; the right hand pinches a free pusher. Grounds and beans are Newton MPM particles. There are no object-pose writes, attachments, tool motors, added finger colliders or force overrides.

The off-lip measure counts grounds outside the same pan-frame wall/floor bounds used for retention, then selects those more than 20 mm from the lip's horizontal ground-plane projection. Distance is perpendicular to the infinite lip line, not vertical distance to the lifted pan. The table gives both the share of all grounds and the share of the outside grounds. This separates collection percentage from the incorrect residue pattern.

| Motion probe | Retained | Off-lip count | Off-lip / all grounds | Off-lip / outside grounds |
|---|---:|---:|---:|---:|
| Original stroke, before change | 89.963% | 235 | 2.721% | 27.105% |
| Added second stroke | 89.847% | 326 | 3.774% | 37.172% |
| Second stroke, near end leading | 89.986% | 261 | 3.022% | 30.173% |

The first trial raised the pusher 80 mm, returned to 135 mm before the lip, shifted 25 mm toward −X, lowered and swept at 0.10 m/s with the existing 3 mm ramp clearance. It recentered before the pan walls. The second trial added −9° yaw so the near end led, lowered the TCP another 2 mm and recentered earlier. The left hand kept the pan on the counter throughout. Trial phases were cleanup return 12.8 s, cleanup sweep 14.6 s, cleanup swept 17.1 s, lift 18.3 s, done 23.9 s. Both images still show the side trail. These trials failed the visual guard before further carrying replays were warranted; they are not claimed to pass the full audit. Evidence: `run/round4/baseline_result.json`, `baseline_particles.npz`, `cleanup_probe_result.json`, `cleanup_probe.png`, `cleanup_probe2_result.json`, `cleanup_probe2.png`; the read-only diagnostic is `run/round4/measure_residue.py`.

The requested RGB rectangles gave the following medians in the 1920×1280 reference and uncompressed path-traced photo-A snapshots:

| Sample rectangle | Photo RGB | Before RGB | Retained RGB |
|---|---|---|---|
| Tiles, (1300,150)–(1500,300) | 184,189,192 | 184,189,189 | 184,189,190 |
| Plain wall, (50,100)–(250,400) | 211,217,217 | 227,232,231 | 215,220,220 |
| Counter, (980,1210)–(1300,1270) | 165,171,168 | 164,171,166 | 163,170,166 |

The tile rectangle was already close; the wall rectangle was too bright. Accordingly the retained wall albedo is `#d7dddf`, lower than the previous `#e3e8ea`; tiles use `#bcc1c4` and roughness 0.34 rather than 0.36. The camera, concrete and lighting are unchanged. Tiles are within 2 RGB levels and the wall within 4. Whole-frame **ΔE improves from 12.3 to 12.1**. `ctl compare` rounds both PSNR readings to **15.0 dB**; the unrounded full-resolution RGB calculation improves slightly from **15.006254 to 15.032426 dB**, only 0.026 dB. This is a small improvement, not a complete photographic match. Evidence: `run/round4/colours.json`, `before.png`, `material_final.png`, `photo_before.json`, `photo_final.json`, `photo_final_beside.png`, `photo_final_look.png`.

The teapot remains size ratio 1.000, IoU 0.7714. The top plate reads size ratio 0.929, width ratio 1.051, IoU 0.5182 in the new CLI comparison. The lower four plate meshes and colliders retain the corrected 234.6 mm diameter, and the top plate remains 195 mm. The refreshed whole-stack union from the final take reads width ratio 1.000, size ratio 0.947, IoU 0.6184 and offset [42.5, −0.7] pixels at 960×640 (`run/round4/plate_stack.json`). It uses the shipped `compare_masks` on the union of the five native body labels. The clearance translation still offsets the stack from the photograph; its shallow profiles remain approximate. The whole-stack mask is `ref/plate_stack_mask.png` and the keyed masks are `ref/stack_top_mask.png` and `ref/teapot_mask.png`.

The seam-free `assets/concrete_v8.png` remains periodic: every Gaussian-filter octave uses `mode="wrap"`. The robot bases retain their colliding 120×220×44 mm support pads, centred in the surface frame at (−347.5,±310,−10) mm, and their vertical brackets. These designed supports do not change catalog mount coordinates. Cabinets, floor and extended walls remain geometry. Jar reflections and tile/dish profiles still differ from the photographs.

The full, unmodified `./ctl physics routine` ran once, uninterrupted, and exited 0. **The tool itself wrote `run/physics.json`; it has not been assembled or edited.** It is newer than every retained Python file. `run/round4/audit_job.py` only launched that CLI and copied completed per-run diagnostics; it did not step the simulation or generate a verdict. Its transcript, exit code and diagnostic copies are `audit.log`, `audit_exit.json` and `audit_retention.json` in the same directory.

| Carrying replay | Grounds retained | Beans | Lip lift |
|---|---:|---:|---:|
| written | 89.975% | 17/30 | 105.37 mm |
| built | 89.766% | 17/30 | 105.37 mm |
| heavy | 84.985% | 14/30 | 98.70 mm |
| plain | 89.720% | 17/30 | 105.38 mm |

```text
PASS  contact  nothing ties a thing to the robot, no motor on a thing's joints, nothing welded
PASS  robot    the registered yam_bimanual, its own file, hands and servos
NOTE  model    utensil_jar: drawn parts that do not collide reach 5.9 mm outside its colliders: say what they are; mortar: drawn parts that do not collide reach 4.3 mm outside its colliders: say what they are
PASS  settle   nothing starts inside anything (deepest 0.6 mm); the scene's settle moved nothing more than 0.0 mm
PASS  written  finished; moved pusher, dustpan, beans, grounds
PASS  frozen   with the robot frozen nothing moved more than 1.2 mm
PASS  open     with the fingers held open the task did not happen (RuntimeError: Sweep missed the 75% / 25 mm bar: {'in_pan_pct': 7.154433896735355, 'beans_in_pan': 0, 'beans': 30, 'grounds': 8638, 'lifted_mm': np.float64(-0.025451183319091797), 'pan_pose': [0.5799710750579834, -0.04493635147809982, 0.7499745488166809, -1.3396759641182143e-05, 1.3877468063583365e-07, -2.99030289)
PASS  built    finished with every friction at most 1 and the arm's torque at 27/10 Nm, the things ending where the written run left them
PASS  heavy    finished with the rigid handled things 2 times as heavy (beans, grounds: not), the things ending where the written run left them
PASS  grip     the fingers pressed pusher 100 N (p95 107.4) by hand 2; dustpan 100 N (p95 102.5) by hand 1; the solver used friction 1 to 1 at the finger contacts
PASS  drop     with the fingers opened at the top of the carry (step 791) pusher, dustpan fell
PASS  plain    played with nothing watching, the things ended where the replay left them
```

The following three-second `./ctl check` also passed, with worst overlap 0.17 mm between table and nuts jar and maximum drift 0.98 mm (beans). Dish-brush drift was 0.37 mm. Builds, finite state, camera, expected objects and finite state after 180 steps all passed. Evidence: `run/check.json`.

| Final RTX take | Retention | Beans | Lip lift | Off-lip / all grounds | Off-lip / outside grounds |
|---|---:|---:|---:|---:|---:|
| hero | 89.928% | 17/30 | 105.37 mm | 2.582% | 25.632% |
| hero_wide | 90.183% | 17/30 | 105.35 mm | 2.802% | 28.538% |

The score transforms particles into the current pan frame and checks its wall/floor bounds with particle-radius tolerance. The pass bar remains 75% of grounds and a lift above 25 mm. The films are [photo A](run/hero.mp4) and [wide](run/hero_wide.mp4), each 285 frames at 15 fps, 19 seconds including the four-second hold. Their phase results are `run/hero_result.json` and `run/hero_wide_result.json`. Fresh residue readings and reproducible final particle/pose arrays are `run/round4/hero_residue.json`, `hero_particles.npz`, `hero_wide_residue.json` and `hero_wide_particles.npz`.

I inspected both fresh frame folders (`run/hero_frames/`, `run/hero_wide_frames/`), their contact sheets and first/last frames. Their timelines end at `t018.93.jpg`; no round-1 frames remain. The unscaled full last frames are `run/hero_last_full.png` and `run/hero_wide_last_full.png`; crops are `run/hero_pan_crop.png` and `run/hero_wide_pan_crop.png`. Crop coordinates and pan widths (444 and 282 pixels) are in the matching `_crop.json` files. `run/round4/crop_take.py` extracts these from the completed film and native segmentation. Each take's `.json` and `_seg.npz` preserve recording metadata and body labels. The heap is visible through the hold; the remaining side line is also visible. Neither film establishes the requested lip-only cleanup.

The retained irregular frustum spill has seed 29, X/Y half-extents 62/83 mm, height 11.5 mm and a top outline at 70% linear scale. Its centre is 10 mm toward the pusher relative to the earlier prism. There are 8,638 grounds at 2.5 mm spacing, density 420 kg/m³, internal friction 0.78, totaling 56.69 g. Thirty warmer brown beans at 6 mm spacing and density 650 kg/m³ total approximately 4.21 g; sixteen start on the counter and fourteen over the patch. These are coarse MPM particles, not grooved rigid bean meshes. The room/support view is `run/round4/room_final.png`.


The retained model notes concern unhandled hollow props: mortar and utensil-jar surfaces extend 4.3 and 5.9 mm beyond their segmented wall proxies. Handled tools have full visible collision geometry. The backsplash remains static, without a weld. The pusher's 4 mm MPM margin warning remains: the shared Newton margin feeds the particle solver, while MuJoCo native CCD zeros its own copy and uses the real rigid faces. There is no separate public per-solver margin field. Sparse bean sampling, capsule/mesh multicontact limits and the shipped finite-difference IK fallback for 191 joint coordinates remain disclosed approximations.

The clearance translations, measured from the original photo layout in work-surface coordinates (+X into the counter, +Y left in photo A), are now:

| Props | Translation |
|---|---|
| Mortar, saucer and pestle | +110 mm X, +14 mm Y (round 1: +151 mm X) |
| Plate stack | +5 mm X, −45 mm Y (round 1: −130 mm Y), plus its roughly 3 mm settling shift toward the camera |
| Board and its wooden utensils | −120 mm Y, as before |
| Utensil jar, whisk and brushes | −130 mm Y, as before, plus saved settling offsets |
| Coffee jar | +20 mm Y, as before |
| Teapot | −5 mm X, −6 mm Y centring correction |

The loose foreground plate and fork remain omitted to make room, and the old 27 rigid beans remain replaced by the granular spill. The pestle still leans differently from the photo, and the whisk/brushes retain their earlier naturally settled rotations. The round-1 offset records are kept in `assets/rest_adjustments_mm.json` and `assets/rest_rotations_deg.json`. The dish brush retains its saved free-body resting pose in `assets/resting_props.json`; it was not fixed or held. The new three-second check measures 0.37 mm of brush drift.


`photo_a` retains the October pose/FOV: eye (−0.15,0,1.07), look-at (0.85,0,0.87), FOV 41.2563366786°, 1920×1280. `wide` remains at (−0.30,0,1.38), looking at (0.52,0,0.87), FOV 39°, 1920×1080. `room` exposes the cabinets, floor and supports. `photo_b` remains removed because its fit failed. The room, cabinets, floor and extended walls are geometry, with no camera background or projected photo. Unphotographed room dimensions and cabinet design are assumptions. Jar lids are fused, bristles rigid, and whisk wires use segment proxies.

Assumptions retained: the photo scale anchor is an unsurveyed 180 mm side plate; work surface is 595×1300 mm at z=750 mm. Pan inner floor is 180×150 mm, with 2 mm lip, ramp to 10 mm over 35 mm, 10 mm side walls, 20 mm back wall, 150 g mass. Pusher is 170×20×40 mm, 136 g. Friction is counter 0.4, pan floor 0.15, walls 1, pusher 0.3, other props 0.7; finger-contact friction is the measured 1. Generic background masses remain 400 g for most props and 80 g for utensils/brushes, not weighed values. MPM runs at 120 Hz with 8 mm exterior-projection gap and the shipped 1000× rigid-proxy mass approximation; actual rigid masses and returned contact impulses are unchanged. Granular stiffness is the shipped numerical preset (Young modulus 1e15 Pa, Poisson ratio 0.3, yield pressure 1e12 Pa), not a measured coffee fit. Robot control is 60 Hz; contact capacities remain 8192/32768.

The preserved approach pinches at TCP height 17 mm above the counter, uses 3 mm ramp clearance, sweeps at 0.12 m/s and commands 90 mm travel into the pan. Phase times remain moves 1.833 s, reach 4.433 s, sweep 9.233 s, swept 12.083 s, lift 13.283 s and done 18.883 s. The lift requests −5° about X and −12° about Y, rises 90 mm and holds four seconds. Relative to October, sweep/lift start about 0.55 s later and the hold is two seconds longer.


The folder retains runtime assets, references and only the current evidence cited here. Historical `run/round1/`, `round2/`, `round3/`, `revision/`, probe-small-buffer files, old comparison sheets, old presentation exports, old audits, guard backups and Python caches were removed. All `.pid` files and `run/ctl.sock` were removed as requested after the final page commands. The page remains running, paused and reset; the removed socket will need recreating for a future CLI session. `run/round4/status_delivery.json` records the final page state. Ruff lint and formatting pass for all six source files. All kitchen assets are read locally from this folder; the robot remains the shipped catalog model. The reference photos and October sweep evidence are under `ref/`.

Optional beans-only and water routines remain unimplemented; no water-volume claim is made. No library test tiers were run.

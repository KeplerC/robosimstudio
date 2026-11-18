# Anthurium watering

## Final take: read this first

The rest of this report was written after round 2. The take that was filmed and accepted came from round 3 (`run/hero.mp4` and `run/hero_wide.mp4`). Where a later section disagrees with this one, this one holds. Readings are from `run/hero_water_balance.csv` and `run/hero_containment.json`.

| Reading | Final take |
|---|---:|
| In pot | 83.80% |
| On table | 16.10% |
| Below table | 0.00% |
| Leaf-2 peak tip bend | 40.3 mm, mostly sideways (y −31.7 mm, z −17.1 mm) |
| Leaf-2 final residual | 0.146 mm |
| Final cup axis from vertical | 2.87° |
| Take length | 26.8 s, 269 frames |

- **None of the leaf dose reaches the pot.** The pot reads 0% until the pour over it begins. The stream runs off leaf 2's tip onto the table, which already holds 14.1% at 13.25 s. The leaf's bow is more a sideways swing than a droop.
- **The wide camera does not show the chore.** It sits at 43° from eye (−0.45, −0.95, 1.48), as `scene.py` sets it, not where the round-2 text below puts it. In its film the right arm stands between the lens and the cup during both pours. `run/hero.mp4` is the film that shows the chore.
- **`heavy` is a NOTE, not a PASS.** At the final gate, with the cup at twice its mass, the solver blew up at step 1167 (about 19.4 s), during the pour over the pot. The likely cause is the lagged MPM proxy coupling at mass_scale 1000 under the heavier cup. On current main the same replay stops sooner: the heavier cup sits 12 mm off its grasp, so the pour pose is out of reach at `lift_out`. The `drop` test opened the fingers at step 531.
- **The settled volume is out of date.** The 51.98 mL figure below comes from the previous water bake. `assets/water_rest_480.json` was re-baked for the final take, and its settled volume was not measured again.
- **On current main (3f3970f):** the routine finishes in 1,611 steps. The pot holds 85.2% of the water and the table 14.6%, with none below the table. Leaf 2 bends 44.5 mm at its peak, and the cup ends 2.9° from vertical. `check` passes. The physics replays show no FAIL. They give the `heavy` NOTE above plus the contact, model and settle NOTEs that the physics section below explains: the stalk roots welded into the soil, the pot-wall and bench colliders, and the 1.7 mm spathe junction.

## Round 2 report

The reviewed films are `run/hero.mp4` and `run/hero_wide.mp4`, both path-traced at `quality="final"`. The wide film renders the same measured states through a second camera. The cup is moved only by the catalog hand's finger contact. The large outer leaf now receives a sustained stream, bows, and recovers.

## Measured take (round 2)

| Reading | Round 2 hero | Old film target |
|---|---:|---:|
| In pot | 80.74% (52.32 g nominal) | about 86% |
| On table | 19.08% | about 14% |
| Below table | 0.00% | 0% required |
| In cup | 0.00% | empty |
| Other water | 0.18% | — |
| Target leaf-2 peak tip bend | 25.62 mm | 34.6 mm; revision target ≥25 mm |
| Target leaf-2 final residual | 0.136 mm | <1 mm |
| Largest final leaf-tip residual | 0.136 mm | <1 mm |
| Final cup axis from vertical | 6.37° | upright |

Bend is total 3D tip displacement. At the largest sampled frame, the vertical component is -13.0 mm and the photo displacement is (33.4, 18.0) pixels; the blade curls sideways as well as downward.

The capture rate is lower than the old film: the longer leaf dose runs onto the table before the remaining fill goes into the pot. Nothing was knocked over. `run/hero_result.json`, `run/hero_events.json`, `run/hero_leaf_trace.csv` and `run/hero_water_balance.csv` contain the readings. `run/hero_cup_fill_trace.csv` records fill at 10 Hz through the sustained stream. `run/hero_containment.json` checks every particle in the 10 Hz trajectory: maximum below-table fraction is 0.00%, minimum bench-relative height 6.49 mm. The balance counts all 3,920 particles without filtering.


## Water containment and rest

The old 6 mm tabletop was thinner than two 4 mm MPM voxels. Its finite signed-distance query band did not guarantee continuous collision. It is now 40 mm thick, with its top unchanged at bench z=6 mm. The four water-facing containers (cup, table, room floor and garden ground) use a 5 mm liquid collision margin. Rigid MuJoCo margins are explicitly zero in that solver's view, as its native CCD already ignores these liquid offsets.

There was also a collider-set problem: `collide=False` decoration still retained `COLLIDE_PARTICLES` in this checkout. Overlapping visual boards, pots, foliage and the stock solid bench display entered MPM's nearest-surface queries. The scene-local coupler now filters those particle flags, retaining the actual physical surfaces. The room floor and garden ground remain real collision geometry. The duplicate stock ground plane is excluded from the MPM view.

Thickening and filtering alone did not eliminate all tunnelling: trial C still recorded one inside-table crossing (`run/r2_trial_c_spill_paths.json`). A continuous swept impact solve now checks every water-particle segment against MPM's actual fixed collider mesh. It resolves entering impacts with zero restitution and Coulomb friction from the contacted face (capped at 1). This is collision response, not a z clamp or a prescribed water path. No particles are removed, reset during a run, or excluded from measurements. The fixed table/floor receive the reaction. `run/swept_static_probe.json` separately tests a bare-table traversal and a floor traversal: both stop at their surfaces with zero normal velocity. This added collision implementation is in `anthurium_coupling.py`; the library was not changed.

`run/table_drop_crossings_r2.json` and `run/containment_probe_r2.json` distinguish real spillover beyond the tabletop edge from tunnelling through its footprint. The floor catches the off-edge drop probe. These diagnostic drops are not the chore and do spill below table height. The chore/replay measurements below count every particle, including any off-edge spill.

Genesis is absent. `rss.liquid(material="water")` uses implicit MPM, PIC collider basis, P1d strain basis, 4 mm voxels, and 480 Hz stepping. The nominal fill is 64.8 mL / 64.8 g, represented by 3,920 equal-mass particles. All are retained. After 10.6 simulated seconds of settling, the convex-envelope volume is **51.98 mL (80.2% of nominal)**. This is spatial compression, not a claim of incompressibility. `assets/water_rest_480.json` was re-baked with positions, elastic strain and plastic-volume state. Liquid post-step projection remains off.

The unchanged 180-step rest gate passes: water drift **1.16 mm** against 2 mm; cup 0.11 mm; leaf 2 centre 0.16 mm. The final gate includes every green leaf and spathe; their largest centre drift is 0.25 mm. Worst penetration is 0.03 mm between table and cup, against 1 mm. All finite/build/camera/object checks pass (`run/check_final.txt`, `run/check.json`).

## Leaf response and chore

The stream is aimed at leaf 2's free outer blade, facing the photo camera. The tilted opening is held over it for 1.8 s. The previous 0.25 s touch was too short. Early longer-dose trials also missed much of leaf 2; the current path was corrected using actual particle crossings (`run/leaf_stream_c.json`). It withdraws toward the camera before lifting across the canopy, avoiding the spathes.

Free-sheet exponential velocity damping was reduced from 8/s to **2/s**. Leaf 2's bending parameter is **20** rather than 40; membrane parameters remain 14,000. Leaf 9 retains the earlier 100 setting that prevents its inverted snap-through; other green leaves remain 40 and spathes 60. These are fitted numerical choices, not botanical measurements supplied by the photograph. The unloaded leaf-2 re-bake changed its centre by 4.18 mm, with a 10.92 mm 95th-percentile particle displacement relative to the previous bake (`run/r2_plant_rest_c.json`). The camera is unchanged.

Water/sheet contact is a swept particle/triangle contact with frictionless, inelastic normal impulse. Equal and opposite impulses are distributed barycentrically to live VBD vertices. No scheduled force or displacement produces the bow. There is no wetting, adhesion, absorption or distributed water-to-leaf pressure model beyond those particle impulses.

The routine unfolds the right arm outside the table before approaching the cup. It grips the cup's outer walls, lifts, makes the leaf pour, empties into the pot, rights the cup and withdraws. The extra unfolding and safe canopy clearance make this take about 32.6 s, slower than the old film's approximately 22.8 s. The final upright orientation is approximate, quantified above. The remaining capture-rate difference is disclosed rather than hidden by moving particles into the pot.

## Native physics audit

The complete audit below used the same physics, meshes, baked initial state and routine as the final scene. Only the return-wall paint changed afterward. A fresh final-quality take was then recorded; the full audit was not rerun for a paint change.

The full native audit is **PASS** (`run/physics.json`, `run/physics_cli.txt`). Its lines are quoted unchanged:

```text
NOTE  contact  1 step hook(s) added while the routine played: say in the report what each does (one that moves a thing is an attachment); leaf_01_stalk welded to the world; leaf_02_stalk welded to the world; leaf_03_stalk welded to the world; leaf_04_stalk welded to the world; leaf_05_stalk welded to the world; leaf_06_stalk welded to the world; leaf_07_stalk welded to the world; leaf_08_stalk welded to the world; leaf_09_stalk welded to the world; leaf_10_stalk welded to the world; leaf_11_stalk welded to the world; leaf_12_stalk welded to the world; spathe_01_stalk welded to the world; spathe_02_stalk welded to the world; spathe_03_stalk welded to the world; spathe_04_stalk welded to the world: a welded thing cannot move, so say in the report why it is welded
PASS  robot    the registered yam_bimanual, its own file, hands and servos
NOTE  model    pot_wall_10, pot_wall_11, pot_wall_12, pot_wall_13, pot_wall_14, pot_wall_15, pot_wall_16, pot_wall_17, pot_wall_18, pot_wall_19, pot_wall_20, pot_wall_21, pot_wall_22, pot_wall_23, pot_wall_24, pot_wall_6, pot_wall_7, pot_wall_8, pot_wall_9, table: a collider nothing draws touched a handled thing: say what it is in the clip
NOTE  settle   spathe_02_stalk rests 1.7 mm into spathe_02_spadix and stays there with the robot frozen: a contact that soft, say why
PASS  written  finished; moved cup, water
PASS  frozen   with the robot frozen nothing moved more than 1.4 mm
PASS  open     with the fingers held open the task did not happen (RuntimeError: The cup did not lift in the fingers)
PASS  built    finished with every friction at most 1 and the arm's torque at 27/10 Nm, the things ending where the written run left them
PASS  heavy    finished with the rigid handled things 2 times as heavy (water: not), the things ending where the written run left them
PASS  grip     the fingers pressed cup 100.1 N (p95 100.5) by hand 2; the solver used friction 1 to 1 at the finger contacts
PASS  drop     with the fingers opened at the top of the carry (step 1313) cup fell
PASS  plain    played with nothing watching, the things ended where the replay left them
```

| Replay | Pot | Cup | Below table | Leaf-2 peak | Leaf-2 residual |
|---|---:|---:|---:|---:|---:|
| written | 81.20% | 0.00% | 0.00% | 27.200 mm | 0.148 mm |
| frozen | 0.00% | 100.00% | 0.00% | — | — |
| open | 0.00% | 100.00% | 0.00% | — | — |
| built | 82.12% | 0.08% | 0.00% | 27.901 mm | 0.141 mm |
| heavy | 81.91% | 0.08% | 0.00% | 25.355 mm | 0.159 mm |
| drop | 0.26% | 25.99% | 0.00% | 25.982 mm | 0.155 mm |
| plain | 81.10% | 0.20% | 0.00% | 28.211 mm | 0.149 mm |

`run/physics_case_metrics.json` maps each row to its timestamped measurement archive. Frozen/open runs correctly fail to lift the cup; their routine archives are taken at that failure, while the native audit continues observing the prescribed duration. `run/frozen_open_containment_bound.json` also uses the native all-particle maximum displacement to bound water height during the full frozen/open observation periods: both remain above the below-table threshold. The heavy test doubles the rigid cup mass, not the deformable water mass. These are distinct replays of the same settings, not selected repetitions.

The contact NOTE names sixteen stalk roots welded to the world. These are permanent roots in soil, not robot-to-object attachments. Each blade also retains its fifteen-vertex basal attachment to its own stalk. The attachment belongs to the plant and is retained. Its coupling is one-way: sheet loads do not flex the rod, so the measured bow comes from the free blade. Pollen spikes similarly follow their stalk tips. Rod/particle and plant self-contact are disabled to avoid embedded botanical joints repelling their own parts. Water loads the blades, not the thin stalks directly. This is not a fully coupled botanical model.

The one routine-added step hook copies arrays for the second film; it never changes state. The model NOTE identifies the pot-wall collision segments underneath the detailed terracotta/cachepot meshes and the standard bench collision block underneath the wooden dressing. These represent the real pot interior and bench; they are not extra finger colliders. The bench display was hidden, not its rigid collision block.

The stable 1.7 mm overlap at the spathe-2 stalk/spadix joint is an intrinsic botanical junction, explaining the settle NOTE. Reset restoration writes the settled initial state only at reset. The cup has no weld, attachment, latch, dragger, joint motor or routine-written pose.

The robot remains the registered `yam_bimanual`, its standard mounts, file, colliders, servo gains and torque limits. Only arm home joint angles change. No additional finger colliders were added. The scene-local three-island solver advances MuJoCo, VBD and MPM once per substep, with lagged proxy reactions. The MPM numerical proxy mass scale is 1000; actual cup mass remains 180 g. Proxy inertia caches are refreshed after construction and model changes. No library test tiers were run or added.

Newton's inertia-floor warning identifies 66 thin stalk segments, not robot bodies (`run/inertia_audit.json`). Avoidable rigid-margin warnings were removed. RTX occasionally logs a renderer teardown warning after closing a secondary renderer; it does not indicate a failed scene build.

## Held camera, room and gallery assets

Kept the October plant meshes and fitted layout: twelve green cloth leaves, four cloth spathes, sixteen rod stalks, terracotta pot, honeycomb cachepot, mint saucer, pale lid board and steel windows. The available wood/plaster textures and plant calibration data remain local. This revision rebuilt the water-facing collision setup, pouring path, parked poses, garden and visible room.

The October `photo` camera stays at **1920×1280, 36.5° vertical FOV**, world eye (-0.6, 0.24, 1.26), look-at (0.6, 0.24, 1.035). The 140 mm pot rim remains the scale reference. `run/photo_final_beside.png`, `run/photo_final_look.png` and `run/photo_compare.txt` show the final comparison. Mean CIE76 colour difference is **28.5**, improved from round 1's 30.2. PSNR is 11.2 dB. The broad table, bright return wall, missing window haze/reflections and simplified lighting remain visible differences.

`run/garden_bounds.json` verifies a minimum foliage x of 1.061 m, beyond the required 0.45 m. The garden is overlapping solid hedge crowns with individual leaf geometry. All foliage vertices are beyond bench x=0.45 m; none float inside the room. Pergola beams are above the window head, with no low duplicate rail or extra posts. The room has plaster returns, a side wall, window head, under-window wall and floor. Four wooden legs, an apron and drawer case the standard bench. Its black display plinth is hidden visually; its rigid bench contacts and mounts remain. No clean plate, background photograph or projected photo pixels are used.

`wide` is 1280×960, 55° vertical FOV, eye (-0.65, 1.2, 1.6), look-at (0.45, 0, 0.8). It shows both full arms, cup and plant inside the modelled room. The right arm parks out of the photo. A small left shoulder/upper-link portion remains at the bottom of the photo: this does **not** fully meet the requested arm-free first frame. `run/photo_initial.png` is the time-zero snapshot and records that limitation; `run/hero_0.0s.jpg` shows the first film frame. The standard mounts were not moved to hide it.

`run/hero_strip.jpg` and `run/hero_wide_strip.jpg` show pickup, leaf pour, pot pour, withdrawal and recovery. Full-size pour stills are `run/hero_17.6s.jpg`, `run/hero_18.6s.jpg`, `run/hero_22.6s.jpg` and `run/hero_24.3s.jpg`. Both films contain 327 frames at 10 Hz over 32.6 simulated seconds and use the final 12-pass RTX setting. Water remains a surface render at opacity 0.3. The gripper partly occludes the incoming leaf stream; runoff is visible below the free tip. The leaf bow is visible below the hand. The cup lip remains inside the held photo frame. The wide film is read-only playback of `run/hero_states.npz` into a separate render state, never a pose write to the live simulation.

The dead backdrop material hookup, panel branch, garden/flyscreen props and unused asset generators were removed. Unread backdrop images/meshes and superseded leaf textures were deleted (`run/asset_cleanup.json`). Runtime meshes, material textures and baked data live under `assets/`; references under `ref/`; helpers at the top level. Nothing at runtime reads `/inputs` or another external asset directory. The installed RoboSimStudio library and catalog robot remain dependencies. Round-1 evidence is retained under `run/round1/`; failed trial files are diagnostics, not claimed successes.

## Assumptions retained from the port

The glass cup is a five-slab square approximation with a 50×50×70 mm inner cavity and 8 mm walls, not a measured reconstruction of the old film's glass. Its physical slabs collide. The pot interior uses 32 wall segments and a nonporous soil surface at bench z=188 mm; water collects above soil rather than soaking in. The pot, cachepot and saucer are fixed.

Leaf areal density is 0.23 kg/m² and nominal thickness 0.45 mm. Stalk radius is 1.45 mm, density 650 kg/m³, stretch 100,000, bend 5 and twist 3. Cup friction is 0.7, with stock hand contacts. Water/leaf contact offset is 1.5 mm. These values and room depths/light intensities are inferred, not established by the stock photo. The old chore used a different rigid-blade plant on a kitchen counter; its shape and material response are not the October cloth twin.

The page is left paused at the initial state. `run/final_status.json` records the final build and renderer status.

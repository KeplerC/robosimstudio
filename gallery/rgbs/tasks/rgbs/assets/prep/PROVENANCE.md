# `rgbs` prep assets (`wipe`) — provenance

**Everything here was authored for this corpus.** No upstream meshes, no
imported BDDL, no vendored geometry: the suite ships no binary asset at all, and
every shape is a Newton primitive drawn by the entity that owns it.

The reference is the **Surface Preparing** clip in `sample_videos/` (1920×1080,
2× speed, with the real rig's overhead "RGBS View" inset in its bottom-left
640×360) and the stills cut from it under
`outputs/rgbs_reference/surface_preparing/`. The scene was measured off those
frames on 2026-09-09/10 for `rgbs/scripts/surface_cleaning.py` and backported
here; the grid on the mat is 25 mm on 300 mm tiles, which is the ruler for every
number below.

## The scene, as measured (2026-09-10)

Everything in the cell's work-surface frame; `layout.py` is the single
statement of it and everything else derives.

| | |
|---|---|
| mat | 10 mm EVA over the whole bench top, 25 mm grid on 300 mm tiles |
| tube | Ø25 mm (one grid square), straight run 340 mm between elbow tangents |
| elbows | 32 mm centreline radius, five cylinders and six spheres each |
| legs | drop 50 mm from the axis to the block's top (two tube diameters), buried 15 mm |
| blocks | 76 × 80 × 70 mm, black; the leg enters 18 mm in from the front face, 20 mm in from the inner |
| bar axis | x 0.000, y −0.020 (seven to eight grid rows from the arm brackets), z 0.130 = mat 0.010 + block 0.070 + drop 0.050 |
| crown | z 0.1425 — the rig's aim, mid-run |
| stripe | the middle 200 mm of the crown, 9° half-angle (~4 mm), 2.5 mm cells: 243 of them |
| sponge | 50 × 75 × 30 mm, (9, 13, 6) = 980 particles, long axis along the bar |
| sponge at | 70 mm of clear mat past the near block's outer face, 35 mm beyond the bar's line: (0.035, −0.3895), on the mat |
| support hold | the far elbow's tangent point, u = +0.170 |

**The bar is a fixture with a handle.** The real bar is bolted into its blocks
and never leaves the bench, so `SpecGrabBar` is `static=True` — an `ObjectSpec`
still, because only the `objects` bin gets an `EntityHandle` and the goal reads
`hands-clear bar` off it. A static object is a jointless link, which MuJoCo's
converter refuses ("outside articulations and no standalone joint to world");
`EntityGrabBar.build` welds it with a FIXED joint from the world, which the
converter accepts as a standalone root (it says so in a warning: `converting 1
joint(s) outside articulations as standalone world roots: ['bar_weld']`). Under
GENERIC the bar builds without the weld, so a rigid-only check never sees the
failure.

**The stripe is shorter than the photograph's on purpose.** The working hand's
far stroke has to stay a jaw's width clear of the hand steadying the far elbow
(`HAND_CLEAR`, 25 mm), and a stroke to the end of a longer stripe shoved that
hand off the bar on the live page. With a 75 mm sponge the strokes run from
u = −0.0875 to +0.1075 (`stroke_span`): 25 mm past the stripe's near end and
45 mm past its far end, because the leaning hand's contact patch trails the
pinned centre — measured on the page, the end tenths of the stripe were what
stayed blue (38 % and 8 % cleared) when the strokes turned at the block's own
edge.

**The rig cannot hold the whole scene from the calibrated overhead.** Aimed at
the crown, the blocks sit at ±0.77 of the overhead's half-frame and the sponge
at +1.18 — past its edge (+0.94 in `top`). The clip's bar is 404 mm leg to leg
and the calibrated overhead frames 540 mm at the crown's depth; moving the aim
toward the sponge would move the `inset` view off the bar it was measured
against. `tests/test_rgbs.py`'s framing ledger fails on `wipe` for exactly
this, and it is right to: the `top`, `photo` and `inset` views hold everything.

### The sponge's speedometer

The goal's set-down clause was `(< (settle-speed sponge) 0.02)`, and it reads
**0.0 for the whole episode**, a 0.1 m/s carry included: the library's measure
is over Newton's particle velocities and the host island never writes them. On
the 22 mm layout that was survivable — the sponge started 145 mm from the bar
and the set-down could be raced inside the 120-step `hold-for` window. Here it
starts 370 mm away and the verdict would latch with the sponge still in the
jaws. So the taint entity, which already holds the bridge and the sponge's
name, publishes `sponge_speed`: the centroid speed off the node positions the
bridge writes back every control step (its actor entry's `start`/`count`), and
the goal thresholds that beside the dead clause, plus `(not (sponge-on-bar
taint))` so the far-end re-seat — sponge inching down a millimetre at a time,
hand just outside the 100 mm sphere — cannot start the clock.

### The shipped take

`RoutineSurfacePrep` drives `Wiper.steps()`, the same generator the live page
plays through `Wiper.play()`; one choreography, two drivers. Results are in the
section below, patched in from the two fresh-process rollouts.

Two fresh-process rollouts on 2026-09-10 (`scripts/bench_run.py --task rgbs/wipe
--solve routine`, no step cap, the document's own 12000 budget), and the
live page's own driver (`Wiper.play()`, headless) on the same scene:

| | run A | run B | example headless |
|---|---|---|---|
| verdict | SUCCESS, reward 1.00 | SUCCESS, reward 1.00 | (no evaluator) |
| length | 2154 control steps | 2153 control steps | 2420 (the take's own tail included) |
| cleaned | 0.971 | 0.971 | 0.971 |
| legs refused | 0 | 0 | 0 |
| grasp | 176 nodes pinned, underside 22.4 mm below the TCP | 176 nodes, 22.4 mm | 176 / 22.4 mm |
| press | seated 37 mm down at 4.18 N; re-seat 9 mm at 4.07 N; 2.7-8.1 N through the strokes | 37 mm at 4.19 N; re-seat 9 mm at 4.08 N; 2.8-9.2 N | same |
| wall | 246 s (8.7 steps/s, CUDA graph captured) | 247 s | 280 s |

What the take does: the support hand's hold at the far elbow solves to
0.0 mm / 1.0°; the sponge pick is 0.1 mm / 1.7°; the two lifts to the
clearance height straight over the sponge spot are refused by the IK check
(75 mm upright, 134 mm leaning -- the corner of the envelope) and the take
lifts 60 mm instead, then crosses to the stripe's near end at clearance with
0.0 mm / 1.7° of residual. The force loop spends 37 of its 70 mm budget
finding 4 N: the 35° lean lifts the sponge's near edge, so the foam meets the
crown 3 mm later than a flat-grasp measurement predicts. One outward stroke
cleans 0.90, the far-end re-seat finds 4 N again 9 mm down, and the return
stroke ends at 0.97 with the press drifting up to 8 N as the foam beds in;
the take exits at `PATCH_CLEARED` (0.96) after one cycle of the three it has.
The verdict latches 120 steps after the sponge is set down and released,
during the work hand's retreat -- the steadying hand at the far elbow reads
170 mm from the bar's centre and so is "clear" by the atom's own measure
throughout (see the document's note on `hands-clear`).


## History: the 22 mm bar with end cubes (2026-09-02 to 2026-09-10, retired)

The suite's earlier twin of the same photograph: a free Ø22 mm bar, 140 mm of
barrel between two 30 mm cubes, 200 mm overall, 0.30 kg, resting on two
40 × 32 × 21 mm pads on a 4 mm work mat, with three 30 mm × 40° taint patches at
axial −45/0/+45 mm and 0/+40/−40° off the crown, and a 70 × 45 × 30 mm sponge
at (0.0, −0.245). The cubes were the grasp feature so the support arm *could*
lift the bar; the clip's bar is bolted down, and the measurements below are why
the lift was never shipped. Every number here was measured on that geometry and
is kept because the removal law, the pressure floor, the grasp and the pins
were all calibrated on it and carried over unchanged.

### The wrap curve, which is what the removal law was fitted on

Measured on the 22 mm bar and 70 × 45 mm sponge (`rgbs/scripts/rgbs_prep_bench.py
wrap`), with the sponge's underside pinned so it cannot squirt out from under a
narrow presser:

| overdrive | 0 mm | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| force | 0.00 N | 0.09 | 0.46 | 1.06 | 1.82 | 3.60 | 5.64 |
| contact wrap | 43° | 53 | 73 | 87 | 95 | 112 | 126 |
| median pressure | 0.00 kPa | 0.06 | 0.19 | 0.69 | 1.16 | 2.57 | 2.63 |

A rigid pad on a bar would touch a line at every one of those depths. That
difference is the reason the abrasive is foam and the reason the bar is round.

**The zero-overdrive row is not noise, it is the padding band.** SuperDex's
penalty contact reports 292 samples spanning 43° carrying 0.002 N *before
anything is touching*. `p_min = 300 Pa` is what stops a policy cleaning the bar
by holding the sponge near it — a correctness gate, not a tuning knob, and it
sits an order of magnitude above the 1 mm reading and an order below the 3 mm
one.

**The effective reach is narrower than the geometric wrap**, and that is the
pressure floor rather than the contact: at 4 mm the contact spans ±43° but the
low-pressure edge falls under `p_min`, so a 4 mm press cleared the crown patch
(0.94) and left the ±40° ones at exactly 0.00. They needed ~8 mm. The first
authored layout put patches 30° wide at 40° off the crown, so their outer cells
sat at 70° against a full press's 63°: a full 8 mm press cleared one patch to
exactly **0.80 and stayed there for six more strokes**, failing as a stall
rather than as an error. `REACHABLE_HALF_ANGLE` (60°) refuses the mistake at
build time and still guards the stripe.

**The lattice does not change the physics, only the readback**: (18, 12, 8),
(8, 12, 6) and (10, 7, 5) read 1.82, 1.87 and 1.91 N at 4 mm with the same wrap,
and 5.6, 2.3 and 1.8 ms to read the contact patch, because SuperDex hands
per-sample contacts over as Python objects with no buffer protocol. The
kitchen sponge's (9, 13, 6) is chosen the same way.

**Pressure is measured, not nominal.** The engine reports each contact sample's
own quadrature weight; checked on a sponge resting on a bench, the weights sum
to 7.000e-3 m² against a 100 × 70 mm footprint and the forces to 0.1442 N
against a mass of 14.7 g. Ships `k_p = 1.2e-2 (Pa·m)⁻¹`.

### The rig error that made all of that read wrong once

The first version of the bench measured its press standoff from the extent of
the **pinned** nodes rather than the whole block. The waist band is 13 mm of a
33 mm sponge, so "4 mm of overdrive" was really 14 mm: every press read four to
ten times too hard (37 N against a calibrated 5.6 N), the foam wrapped 174° of
a bar the calibration rig wraps 126° of at its deepest, and a sponge commanded
to hover a millimetre clear cleaned 0.71 of the bar. None of it raised. The gap
now runs from the pinned band down to the block's own underside, and is checked
at three depths: −8 mm gives zero contact samples, 0 mm gives 22 of them
carrying 0.000 N — the padding band, correctly refused by `p_min` — and +4 mm
gives 220 carrying 0.635 N.

**A press should be force-servoed rather than positioned**, and this is why:
the `wrap` rig drove a bar into a bottom-pinned sponge, the `wipe` rig lowered a
waist-pinned sponge onto a bar, same law, same materials, same commanded
depths — and forces that differed by a factor of two, because what holds the
foam decides how much of the press becomes compression. The shipped take
presses to 4 N by force for exactly this reason.

### The grasp: what was wrong, and what was the engine's

**The arm is fine and the YAM is correctly tuned.** The same code, arm and
orientation helper picked a 30 mm rigid cube off this bench at z 0.79 and
lifted it 65–83 mm at every yaw tried.

**A contact-only pinch does not carry the foam.** At the wrist roll the arm
could reach, the foam was untouched (45 → 46 mm on closing, −0.1 mm of lift);
at the roll that squeezed it (45 → 62 mm) the pose was 64.3° off anything the
IK solved. So the grasp is `bridge.pin_nodes` re-aimed from the pad frame every
control step, and the jaws are friction — driven through the tools it lifted
the sponge 100.4 mm on a 100 mm lift, slipped 0.15 mm on a 60 mm lateral drag
and dropped 103.8 mm on release.

**And the pinch was half an engine bug.** `robosimstudio/engines/superdex/mirror.py`
built every mirrored collider with `penalty_smoothing_half_distance` at the
engine's default of **5 mm**, wider than the ~3 mm at which the penalty engages,
so a fingertip pressing foam never left the soft part of the ramp: the pads sat
**2.06 mm inside the block** and pressed it with **700 Pa**. Mirrors are now
built at 0.5 mm. Same scene, same 39 mm gap:

| | before | after |
|---|---|---|
| deepest contact gap | −2.06 mm | +0.35 mm |
| peak deformation | 1.95 mm | 4.73 mm |
| grip force | 0.93 N | 5.53 N |

It has to be the **collider** side: the same value on the soft actor alone
changed nothing (−2.081 mm). Two "gap band" rules recorded in between (grips
37–42 mm, extrudes below 36; "the foam takes 7–8 mm of jaw travel") were the
contact bug wearing a material's clothes and are retracted: every gap from 41
down to 26 mm carries, grip rising monotonically 2.28 → 13.16 N. `GRIP_GAP` is
chosen for the grip it gives.

**Do not try to measure the tool geometry off the model.** Fingertip *body
origins* sit ~80 mm behind the gripping surface, and `model.shape_transform`
put the pads 210 mm out on the wrong side of the hand — all fifty decomposed
pieces of one fingertip report the same value. `PAD_CONTACT_OFFSET` is read
off MuJoCo's contact table on a working grasp: the pads straddle the TCP at
±15 mm on the jaw axis and lead it by ~7 mm, so the TCP is the grasp point. A
reach map built on the 210 mm figure claimed the hand could not place its pads
below z 0.83 anywhere on the bench; that was false. Any map built on a computed
tool offset inherits that offset's error — swept with the TCP as the grasp
point, a top-down grasp solves to 0.0 mm everywhere on the mat, and it is the
100 mm standoff *over* a pick point that runs out of envelope toward the far
side of the bench (34.9 mm out over world (0.685, −0.14)).

**Driving an arm goes through the action vector**, not `control.joint_target_q`:
writing the targets directly and stepping with `idle_action()` leaves the arm at
home while IK reports 0.0 mm of error.

### One hand at one end did not get the bar airborne

The support grasp on the cube was sound (0.1–0.2 mm of bar motion on closing,
never dropped). Lifting the near cube inside the reach-measured band (the
support arm solved 0.80–0.84 and failed at 0.86 by 45 mm):

| commanded hold z | bar rose | tilt | far end |
|---|---|---|---|
| 0.80 | 5.2 mm | 3.0° | on its pad |
| 0.82 | 22.5 mm | 4.7° | on its pad |
| 0.84 | 33.4 mm | 10.9° | on its pad |

170 mm of cube-to-cube span at 10.9° is 32.1 mm of near-end rise against the
33.4 measured — the far end never moved; the bar pivoted on its far pad. A
firmer grip barely helped (14.5° at 26 mm, 10.9° at 20, identical at 12), so
this was the grasp yielding to the bar's own moment (0.30 kg at 85 mm), which
is polish's finding in another shape. The clip's bar is bolted down; this one
is now welded.

### The wipe in place, as driven by the arm

The shipped 22 mm routine reached **0.980** against the 0.95 gate in 2278
control steps (38.0 s) of a 12000 budget, 0 legs refused, 166 of 819 nodes
pinned, the block hanging 22.9 mm below the TCP, pressing 23 N at an outer
station at 8 mm of commanded overdrive (four times the 5.6 N a free block
takes, for the "what holds the foam" reason above), the underside ending at
z 0.7546 on a mat top of 0.754, verdict SUCCESS. Getting there cost five takes:

| take | cleaned | what was wrong |
|---|---|---|
| 1 | 0.485 | `over` aimed at the patch's **surface** centre, parking the sponge r·sin(40°) = 7.1 mm off the bar's axis |
| 2 | 0.687 | a centred press at an outer station rode up on an **end cube**, 4 mm proud of the crown |
| 3 | 0.606 | the standoff was measured **free-hanging**, where the foam sags and creeps — 17.4 → 21.6 mm over three stations |
| 4 | 0.646 | force polling drove the dead station to 16 mm of depth chasing contact that was geometrically impossible |
| 5 | 0.970 | **the arm could not reach the stations at all** at the 90 mm crossing height: 39–41 mm of miss, executed anyway |

`_plan` computed `worst_position_error` and nobody read it. **A plan nobody
checks is a plan that gets executed**, so `_plan` refuses a path that misses by
more than `REACH_TOLERANCE`, and the crossing height is 40 mm rather than 90.
The standoff lesson is worth keeping: what matters is not *when* the gap between
hand and foam is measured but whether the foam was **supported** when it was —
free-hanging it sags and keeps sagging, pressed it is steady at 17.1–17.4 mm.

### `settle-speed sponge` reads 0.0 while the sponge is being carried

On the 22 mm layout the `hold-for` clock started the instant `hands-clear`
passed 100 mm, partway through the crossing home: at 0.04 m/s the clock started
at step 2158, the episode latched at 2278, and touchdown would not have come
until 2329 — a SUCCESS with the sponge 40 mm up and still in the jaws. The
set-down was then sized to fit the window (0.08 m/s, ~105 steps end to end).
The grab-bar layout cannot do that (370 mm of crossing against a 120-step
window), which is why the taint field now publishes `sponge_speed`; see above.

## records.py

Declarations only. Six records, each naming the factory that places it; the bar
declares no grasp feature any more, because there is no cube to close on and
the steadying hold is on the tube.

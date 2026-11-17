# `rgbs` decant assets (`seam`, `decant`)

**Everything here is authored for this corpus**, from `sample_videos/box_decanting.mp4`.
No geometry is vendored: the carton, the cutter and the tape are built from box
primitives by the entities in this directory, so there are no mesh files and
nothing to attribute.

Two vendored meshes were considered and not used, and it is worth recording why
so the question is not reopened by accident:

* `vabar/assets/vab_tool_sort/assets/industrial/plastic_box_cutter/` — a real box
  cutter, CC-BY-4.0 by Kasra_ftn via Sketchfab. It is **×1.6 thickened along its
  resting axis** for gripper clearance (see `assets/PROVENANCE.md`), its blade is
  not a separable collider, and it is a 150 mm retractable knife where the clip's
  tool is a palm-sized safety cutter. MuJoCo collides a mesh geom as its convex
  hull, so handle and blade would arrive as one solid wedge with the blade's
  clearance filled in. Since the cut law integrates along the blade's **edge**, a
  hulled or fitted edge is not a cosmetic loss — it is a law whose domain is an
  artefact.
* `utility_knife` in the `vab_tool_sort` export — declared, but export-only (no
  `files.mesh`) and its collider there is a plain 10 × 50 × 206 mm slab.

## What the clip shows, and what was measured off it

`sample_videos/box_decanting.mp4` (1920 × 1080, 22.8 s at 2× speed; the
bottom-left 640 × 360 is the real rig's overhead "RGBS View" inset). Measured
against the mat's 25 mm grid on 300 mm tiles and the arms' 620 mm mount spacing:

| What | Measured | Modelled |
| --- | --- | --- |
| Carton | ~200 mm along the bench (eight grid squares of front face), ~150 across it (the top face reads 0.53 of the front face at a ~28° phone elevation, which is 150 mm, not 100), ~100 tall; long axis along the bench, centre within a square of y = 0 | `SpecShippingCarton.outer` 200 × 150 × 100, yawed a quarter turn so its seam runs along world y, at surface (−0.06, 0) |
| Seam | a visible dark join along the whole long axis | `seam_gap` 12 mm: the join is open, and a 7 mm fingertip blade fits it |
| Tape | one short yellow strip across the middle of the join, ~100 × 25 mm, sRGB 255/252/174 lit | two 100 × 12.5 × 1 mm rigid strips, one per flap, meeting at the centreline; `tape_color` (0.92, 0.88, 0.36) |
| Cutter | a palm-sized safety cutter — slate oval body, red round slider, charcoal grip block — about four squares long and a bit over one wide; at t = 0 it stands on its butt beside the carton and is taken at its nose end, at the end it lies flat on the mat with the slider up | `SpecBoxCutter` 100 × 30 × 20 mm, 15 mm of blade out of the nose, lying flat on the mat nose toward the working arm. **Deliberate deviation:** the standing start and the side pick are not reproduced; the tool lies flat as it does at the end of the clip and is picked top-down across its body, which is the grasp the arms can hold tilted along the whole draw (below) |
| Where the cutter lies | to the right (−y) of the carton and a little beyond it (+x), about two squares past its corner | surface (+0.05, −0.165) relative to the carton's x: closer than the clip so the rig's overhead keeps it inside 0.60 of the half-frame (`tests/test_rgbs.py`) |
| Where the carton stands | the inset puts its centre ~0.13 of the frame above the frame's middle, i.e. some 100–150 mm on the far side of the bench's centre | surface x = −0.06: **140 mm nearer the arms than the clip's**, because that is where the working arm can hold the tilted draw *and* hook the far flap (below). The `photo` view therefore shows the carton larger and nearer than the clip's frame 0 |
| Choreography | right arm picks the cutter (0–3 s); left arm pins the carton's far end while the right seats the blade at the right end of the tape and draws it toward the left arm (4–8 s); the cutter goes back down (9–11 s); the right arm puts a fingertip into the join near the right end and lifts the far flap up and over (12–16 s); the left hand lets go (17–22 s) | `routine_decant.py::RoutineBoxDecant`, in that order |

## What was measured, and where

No constant in this suite is a preference. The measurements are reproducible
with `rgbs/scripts/rgbs_decant_bench.py` and the reach probes described on
`layout.py`.

| Constant | Value | How it was set |
| --- | --- | --- |
| `SpecShippingCarton.outer` | 200 × 150 × 100 mm | The clip's parcel (above). The seam is the long axis; only the middle 100 mm of it is taped, so the draw is 100 mm rather than the seam's length and the reach question moved from "how long a seam" to "where the carton stands". |
| carton centre | surface (−0.06, 0), world (0.54, 0) | A chained full-pose sweep of the working arm over the draw at the tape's height, 21 stations, 4 mm / 6° gate, the knife in its tilted hold: solves at 0.0 mm / 1.7° for every tilt 20–45° at x = −0.10, −0.06, −0.02 and at 30–45° at +0.02; the nose-LEADING mirror is 93–194 mm out everywhere. The far-flap hook arc (a top-down hand, the +x fingertip on a 55 mm radius round the far hinge to 110°) solves at 0.0 mm at x ≤ −0.06 and is 33–69 mm out from −0.02 on. The pin, the pick and the carry solve at every candidate. |
| `seam_gap` | **12 mm** | A YAM fingertip's low band is a 7 mm blade and the errand ends with one put into the join; asserted at build against 9 mm. The shipped zero was itself measured (a 2 mm slot swallowed a 1.5 mm blade: press 0.00 N at every overdrive) and still rules where the blade goes — hence the strips. |
| tape strips | 100 × 12.5 × 1 mm each, meeting at the centreline | The clip's strip; the width reaches 6.5 mm onto the flap's top face and 6 mm across the gap. 1 mm thick because a collider under a millimetre against a 0.2 mm contact gap is the ill-conditioned pair the blade's own 1.5 mm floor avoids. |
| `SpecBoxCutter.body_half`, `blade_at` | 100 × 30 × 20 mm; blade 15 × 1.5 × 12 mm out of the nose, lower edge 1 mm above the underside | Lying flat the blade must not touch the mat (measured on the built scene: body bottom 0.754 m, blade bottom 0.755 m, mat top 0.754 m). Held level the underside lands first, so the hold is tilted; `tip_lead()` gives 7.0 / 7.8 / 8.9 mm of tip below the nose corner at 30 / 35 / 40°, asserted ≥ 6 mm at `cut_tilt` = 35°. |
| `hold_tau_max` | 6.0 N·m | Re-measured on this carton (`--hold`, lever 65 mm): sealed sags 6.3° at 92 N at the edge and yields at 138 N (a YAM makes ~40); half-cut yields at 92 N; **20 %-cut yields at 23 N**; cut opens at 0.8 N. The third is why the goal's gate is 0.95. |
| `flap_armature` | 5.0e-4 kg·m² | The hold is an explicit generalized force, so `kp < I_eff/dt²`. At the flap's physical inertia (1.4e-5) that bound is 13 N·m/rad and the tape needs ~120. |
| `f_pierce` | **2.0 N** | Set by what the ARM can put through the tool, not by what the holder can. Seating a millimetre at a time and reading the blade against the flaps and the *body* against the flaps apart: the blade climbs 1.8 → 2.8 → 4.7 → 5.2 N as the tip goes 5.4 → 9.1 mm down, and one millimetre later the body carries 1.4 → 6.7 → 13.6 → 20.3 N while the blade reads 0.00 and the knife is levered round in the jaws (its tip went from 38 mm below the TCP to 12 mm above). So the grasp delivers **3–5 N** and a 6.0 N gate was unreachable however good the aim. Bench knee re-measured at the new gate (`--cut`, pinned, tilted holder): 0.5 mm → 0.69 N cuts 0.000 over 128 mm of skate, 0.75 → 2.29 N 0.000, 1.0 → 3.50 N 0.000, 1.5 → 13.0 N cuts 0.750, 2.0 → 19.2 N 0.708, 3.0 → 12.4 N 0.875. Static (`--press`, pinned): 0.5/1/1.5/2/3/4 mm → 0.46/2.9/5.9/8.2/12.8/17.2 N, i.e. ~4.5 N/mm against 1.3 N/mm on the level 150 mm knife — the tilted holder's stiff angular servo (`kp_ang` 80) reacts the press couple instead of pitching out of it. |
| `k_c` | **90.0** | `83/d` is the analytic single-pass value at `d` newtons of overdrive (a cell sits in the 12 mm kerf window for 0.24 s at the design draw). At 90 a press **one newton over the gate** parts a cell in one pass, which is the margin the arm has. Was 40 against the 6 N gate, where the same sentence needed 2 N. |
| carton shove | 0.5 mm at 2.5 N, **96 mm** at 4.7 N | Measured unpinned over one full draw (`--shove`): the tilted blade's drag walks the 160 g carton off the blade at a press still *under* the gate. What the support hand is for. |
| `KnifeRigSpec.kp_ang` | 80 N·m/rad | Was 8. At 8 the tilted holder pitched 4° under a 10 N press and lifted the tip 5 mm out of its own press; at 80 it is 0.4° and half a millimetre. |

`control.joint_f` reaching an **unactuated** revolute was the one assumption the
flap-hold mechanism rested on, and it is measured rather than assumed: 0.05 N·m
swings a cut flap to its 110° stop.

## What the arms measured, once the routine drove them

The bench holds the knife exactly; an arm does not. Everything here was found
driving `rgbs/tasks/rgbs/routine_decant.py` against the real scene.

| Finding | Measurement |
| --- | --- |
| **A blade seated on the strip's own END reads nothing** | The strip is a 100 mm box and the blade a 15 mm one held nose-down, so its only low point is the tip's corner, and over the strip's end that corner meets a box *edge*: the pair is degenerate. Seated exactly at the end, the force search descended its whole 20 mm reading 0.0 N until the tip was 5 mm through, and the draw that followed read 3.5 / 0.1 / 0.0 / 0.0 N and cut nothing. Seated 10 mm in (`DRAW_INSET`) the same search stops at **0.6 mm of penetration with 4.7 N** and the first pass cuts **0.81**. The law's window runs from the tip 15 mm toward the tool's NOSE, and the nose points against the draw, so a forward pass sweeps `[first tip, last tip + 15 mm]` — which is why the far inset is free and the near one is paid back by one short backward draw. |
| **The fingertip is a tapered blade, and 6 mm of assumption killed the flap hook** | Its outer surface stands at TCP + 48 mm at a shut flap's height and TCP + 43 mm at the pad's bottom 12 mm lower. The arithmetic guess (47.5 mm of travel + a 7 mm blade = 54) put the pad 1 mm *short* of the flap's free edge, and the lift touched the flap on one waypoint in twenty-four (`flap_far_open` 0.036, then 0.000). Corrected to the measured 48, a vertical finger still cannot get under: driven in until its bottom is under the edge, its own tapered side is jammed against the 3 mm edge face, and the recorded result is the flap **and the carton with it shoved 20 mm along +x** at `flap_far_open` 0.000. The hand has to lean so the pad's bottom corner leads — and leaning it the other way, which is what shipped first, is the same sideways push by a different route. |
| **The tilted hold is reachable one way round only** | With the jaws across the body and the hand tilted about the closing axis, the nose must point at the working arm: every station of the draw solves to 0.0 mm / 1.7° that way and is 93–194 mm out with the nose leading. The tip trails the +y draw; the cut law reads speed along the edge. |
| **The cutter pitches further in the jaws** | Lifted and tilted 35°, the tip measured 17° further nose-down than commanded and 11 mm off along the closing axis; every pass re-measures the tip in hand and aims with it, so this is absorbed rather than corrected. |
| **The jaws close at 1.0, open at 0.0** (`action_closed_at_high`) | Inverted, the hand descends shut, knocks the cutter across the mat and opens on empty air. |
| **A leg must be interpolated, and gated** | One IK solve held for N steps is a joint target the servo slews to at full rate; a carried knife is thrown. An unreachable waypoint returns a solution tens of millimetres off with nothing raised, so every leg is solved end to end, its orientation slerped, and refused before it is driven. |
| **Seating IS the press** | Descending past contact in a search ejects the knife from the hand within a few tenths of a millimetre. The approach is planned to stop at the cutting depth. |
| **A return stroke cuts nothing** | The blade rides over what it has already opened. Passes go one way, with a lift and a re-seat between. |
| **Seating is the press, and it has a ceiling as well as a floor** | The search descends a millimetre at a time until the blade reads `SEAT_FORCE`, and stops early if the *body* reads more than `BODY_TOUCH` — past that the tool rides on its own nose. Descending past contact in a search also ejects the knife from the hand within a few tenths of a millimetre. |
| **A return stroke cuts nothing** | The blade rides over what it has already opened. Passes go one way, with a lift and a re-seat between. |
| **The carton has to be re-read between passes** | The pin is a down-force and its lateral hold is only `mu * (weight + pin)`, so the draw walks the box a few millimetres a pass. Reading it once at the start, a scored episode finished **30 mm adrift with `cut_fraction` 0.604**; re-reading it at the top of every pass and before the flap, with the pin at 12 N rather than 6, gave **0.750 with 14.5 mm and no refused legs**. |
| **A dotted cut is a press that dips, not an aim that wanders** | `cut_fraction` 0.750 against a `cut_run` of 0.208 looks like a blade running to one side of the seam — but the two rows came back `near_intact` 0.250 and `far_intact` 0.225, cut **equally**, so the gaps are along the stroke. One commanded height across 80 mm of an arm that sags differently along it reads 4.5 N at the seat, 0.0 N a third of the way and 7 N two thirds, and the law cuts *nothing* under its gate, so every dip leaves tape. `prep.py`'s four-segment trim is what took it from 0.604 to 0.750; polling the field's own force every six control steps and correcting by 0.6 mm is the same idea at ten times the rate. |
| **A force-following stroke is what makes the front travel** | Polling the field's own blade force every six control steps and correcting the commanded height by 0.6 mm took the scored episode from `cut_fraction` 0.750 / `cut_run` 0.208 to **0.917 / 0.417**, with both rows nearly clean (`near_intact` 0.042, `far_intact` 0.064). Fraction and run moved together for the first time. `prep.py`'s four-segment trim is the same idea at a tenth the rate and was worth 0.604 → 0.750 on its own. |
| **What a forward pass can never reach** | The law's window runs from the tip 15 mm toward the tool's nose, and the nose points against the draw, so the band a pass sweeps is `[first tip, last tip + 15 mm]` — the tape *behind* the first tip is unreachable however many passes are run. A run whose four passes all began 10 mm in finished at 0.917 with exactly the two near-end stations intact. Every pass now backs up over that 14 mm before it draws forward. |
| **The pin descends by force, not by depth** | The support arm sags ~7-8 mm at this reach under the 1600 gain, so a pin commanded 4 mm through the flap's top holds nothing. It now steps down a millimetre at a time until its own pads read `PIN_FORCE` against the carton: measured 7.6-16.4 N at 7-8 mm of travel. |

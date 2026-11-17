# `rgbs` peel — the peel, as a graph

The graph behind **one** of `rgbs`'s eight variants, `handle`. It was
`tape_peel/gt` and then `peel/gt`; the suite merged and the graph came with
it, one directory instead of two. It drove a fourth variant, `along_x`, by
refusing it; that one has since been withdrawn from the suite and what the
refusal measured is kept below.

**It drove `panel` and `panel_gentle` until 2026-09-10 and no longer does.**
Those two now peel a protective film off a phone leaning back 30 degrees in a
cradle, with the left hand holding the phone and the right drawing the film
back over its face in an arc -- which is what the reference clip shows and
what this graph cannot express: its haul is `motion.plan_linear`, one straight
line, and its perception body reads a tab on an axis-aligned box top. They
declare `routine:RoutineFilmPeel` instead. Everything below is `handle`'s, and
the `panel` rows are kept as the measurement they were rather than deleted.

Peels the strip off the box. There is no upstream workflow to retarget here — VAB
ships none for a deformable task — so the structure is this suite's own.

## The shape

```
START → perceive_tab ─oracle→ perceive_tab_oracle_sg ─┐
                     └vision→ perceive_tab_vision_sg ─┴→ peel_sg → verify_sg → done
                                                                 └→ abort(sim.detach, open)

peel_sg:  reach ─reachable→ open → approach → sink → descend ─moved→ close
                                                          → latch ─held→ lift → lifted
          reach ─unreachable→ unreachable      latch ─missed→ missed
```

25 nodes across five scopes, eight scripts. It was 16 nodes and four scripts, of
which one computed the entire peel and three replayed it.

## What each node decides

| Node | Calls | The decision it owns |
| --- | --- | --- |
| `perceive_tab` | — | which perception body runs (`--perception`) |
| `..._oracle_sg.read` | `sim.get_tab` | where the tab is, read off the particles |
| `..._vision_sg.read` | GDINO + SAM3 + depth | the same, from a camera — **skipped here**, see below |
| `peel_sg.reach` | `robot.grasp_frame`, `motion.plan_joint` | pinch or top-down; which wrist twin; can this arm do it at all |
| `peel_sg.approach` | `motion.plan_joint` | swing to the clearance via — a posture, not a pose |
| `peel_sg.sink` | `motion.plan_joint` | down to the pre-grasp |
| `peel_sg.descend` | `motion.plan_linear` | straight onto the tab |
| `peel_sg.latch` | `sim.attach`, `sim.release_anchor` | did the grasp happen, and *then* let the layout's pin go |
| `peel_sg.lift` | `motion.plan_linear` | the haul, which is the peel |
| `verify_sg.check` | `sim.get_states` | whether to keep working |

## The three things that are load-bearing

**Transit is a posture problem and manipulation is a path problem.** This is
measured, not preferred. The leg from the arm's home posture to the tab's
pre-grasp cannot be flown as a Cartesian line at all on this arm: every
intermediate point is **162 mm out of reach** with the wrist roll *free*, because
the straight line passes through a region the arm cannot occupy with its tool
pointing along the grasp's advance. Both endpoints are exact. The two short legs
at the tab — pre-grasp to grasp, grasp to lift — solve to **0.00 mm** as straight
lines with the whole rotation held. So `approach` and `sink` are
`motion.plan_joint`, which pins only its endpoints and eases between them, and
`descend` and `lift` are `motion.plan_linear`, which pins every point on the way.

**The wrist twin is a decision, and gating position alone hides it.** A parallel
jaw closing across +z and across −z is the same physical jaw, but only one of the
two rolls is inside this wrist's range. Measured on `handle`: one twin converges
to **1.53°**, the other parks at its roll limit **109°** out — with a *perfect*
position score. `reach` picks by rotation error first and the winner carries the
whole chain.

**The latch order is now three lines instead of a welded tool.** `sim.attach`
seizes the film at the grasp site and re-aims the pin at the hand every control
step; then, and only once the hold is proven, `sim.release_anchor` lets the
layout's own `HandlePin` go. Both ways round fail: release first and the lift
pulls against a flap; leave the pin on and two kinematic holds fight over the
same particles — the pin wins, the tab does not move, and **not a single bond
breaks**. That was this graph's first working run: every node green, 220 lift
waypoints executed, `broken_fraction` 0.0.

## Which layout it can do

`reach` gates on the worst of `pre`, `grasp` and `lift` — the waypoints that have
to be accurate — and **not** on the clearance via, which solves **161 mm** short
on `handle` (and solved 155 mm short on the withdrawn `along_x`) while the peel
`handle` leads into solves to a millimetre. Gating on the via would refuse a
motion that works.

| Layout | grasp | worst gated pos | worst gated rot | route |
| --- | --- | ---: | ---: | --- |
| `handle` | pinch (rigid pull-tab) | 0.00 mm | 0.86° | `reachable` |
| `along_x` (withdrawn) | top-down (drooping flap) | 71.8 mm | — (roll free) | `unreachable` |

**Those are `yam_i2rt` numbers and they are the second set.** These variants
were authored on `yam_bimanual` and read 0.0001 mm / 1.53° and 137.6 mm there.
The `rgbs` merge changed the robot; see *What the robot change cost* below.

`along_x` routed to failure and that was correct, not a gap. Its tab drooped
down the box face at the edge of the workspace; a graph that drove that plan
anyway would drag the strip off the box from centimetres away and call it a
peel. Its own document said so: `(:provenance (:tab-reachable false))`. The
variant was withdrawn from the suite on 2026-09-03 — not because the refusal
was wrong, but because the suite did not want the row. The gate that produced
it is unchanged and still guards the three layouts that remain: a future
layout whose tab lands out of reach gets refused the same way, and this
paragraph is the record of what that looks like when it happens.

## The vision body: declared, and skipped with a reason

`perceive_tab_vision_sg` ships and does not run in the measured
configuration. It needs `grounding-dino.detect`, `sam3.segment_box` and
`geometry.mask_to_world_points`, all of which come from `open-robot-skills` —
the sibling registry `[tool.gap]` pins, whose RPC bundles serve those tools only
once `gap skills install` has built their environments; no `uv sync` here
supplies them. The script checks the registry **before** it does anything and
raises a message naming the three tools and the reason.

That check is the point. The alternative was measured in this repo and written
down: with the bundle absent, a graph reached its detector node, took the
`not_found` exit for every object in turn, and reported **0.00 with a
plausible-looking trace**. A body that cannot run must fail as a missing
dependency, never as a perception result.

```bash
--perception vision   # → node error naming the three tools; exit code non-zero
--perception oracle   # → the default, and what every number here was taken with
```

Two of the oracle's eight keys have no camera answer and the vision body says so
rather than inventing one: `tab_feature` names a particle set the *asset*
declares (empty from a camera, meaning "seize whatever is inside the radius") and
`tab_anchored` is a fact about the staging (always false — a real cell has no
pin). The contract between the pair is the *keys*, never where the values came
from, and `tests/test_perception_bodies.py` is what enforces it, corpus-wide.

**`--no-privileged` still cannot run this graph, and no perception fixes that.**
`sim.attach` is the honest privileged core of every deformable task: no real hand
seizes particles. What the perception split buys is that the *seeing* half
transfers, not the holding half.

## What the robot change cost

**One constant, and it is `PINCH_TILT_DEG`.** The `rgbs` merge restaged these
four variants from `yam_bimanual` onto `yam_i2rt` — a different arm model, a
different hand (`linear_4310` blades against the arena's fingers), and a mount
sitting 22 mm lower relative to the work surface. Nothing in any peel *layout*
moved: every constant those carry is stated in the cell's work-surface frame, so
the box, the panel, the strip and the grasp station are exactly where they were.

What did not survive is the angle the pinch comes in at. Measured with
`rgbs/scripts/rgbs_peel_bench.py tilt` on the `handle` layout as staged:

| tilt | worst position | worst rotation | |
| ---: | ---: | ---: | --- |
| 0–20° | 68 → 8.0 mm | 53 → 45° | refused, and refused on **rotation** throughout |
| 25° | 0.00 mm | 0.92° | |
| **35°** | **0.00 mm** | **0.86°** | shipped |
| 45° | 0.00 mm | 1.02° | |
| 50° | 1.06 mm | 1.16° | |
| 55–75° | 15 → 73 mm | 51 → 59° | refused |

The old 20° sits five degrees outside the near edge of the new arm's band, and
it fails on the wrist roll rather than on position — 44.6° against a 5° gate,
with the hand 8 mm from the tab. So this is not a nudge: the two arms want
different *bands*, and 35 is the middle of this one. Every value from 25 to 50
solves to 0.00 mm and about a degree, so what the middle buys is margin if the
layout ever moves, not accuracy.

`rgbs/scripts/rgbs_peel_bench.py` is what settled it and is written over the same two tools
this node calls, so a layout it reports as reachable is one the graph accepts.

## Status

**`rgbs/handle`: solved.** Measured on `yam_i2rt` after the tilt above,
`--renderer ""` on an RTX PRO 6000, one episode each in a fresh serial process.
The `panel` rows are the layout this graph drove until 2026-09-10 -- a film on
a flat glossy block -- and are not a claim about the phone layout that replaced
it:

| variant | result | reward | steps |
| --- | --- | ---: | ---: |
| `handle` | SUCCESS | 1.00 | 468 (468 on a repeat) |
| `panel` (old flat-panel layout) | SUCCESS | 1.00 | 504 |
| `panel_gentle` (old flat-panel layout) | SUCCESS | 1.00 | 488 |
| `along_x` (withdrawn) | refused at `reach`, without moving | 0.00 | 107 |

Re-measured 2026-09-03 on the same machine, three fresh single-episode
processes per variant: `handle` 3/3 at 446–474 steps, `panel` 3/3 at 494–499,
`panel_gentle` 3/3 at 493–499, and `along_x` 3/3 refusals at 107–110 steps
before it was withdrawn. Every one of those runs scored reward 1.00 or 0.00
with no intermediate values.

The same commit run on the *old* pairing — `peel/handle` on `yam_bimanual`,
before the suites were deleted — gives SUCCESS 1.00 at **427** steps, inside the
425–427 band recorded below. So the arm change costs about 40 steps on this
layout and nothing else.

**One episode is one episode.** Everything below this line is the eight-process
`yam_bimanual` record, kept because it is the only distribution anybody has
measured on this graph and because the single runs above are consistent with it.
Re-measure before quoting a rate.

**`tape_peel/handle`: solved, 8/8.** Measured 2026-08-14, `--renderer ""` on an
RTX PRO 6000, eight fresh processes, one episode each:

| | median | IQR | values |
| --- | ---: | --- | --- |
| `broken_fraction` | **1.000** | [1.000, 1.000] | 1.0 × 8 |
| `peel_front` | **1.000** | [1.000, 1.000] | 15/15 rows × 8 |
| `clear` | **0.947** | [0.947, 0.947] | 0.947 × 8 |
| `episode_length` | 425 | [425, 427] | 423–427 |

The pre-refactor figure taken the same day on the same box, one episode: reward
1.00 at 410 steps, `broken_fraction` 0.956, `peel_front` 0.93 (14/15),
`clear` 0.84, `held_count` 6. The graph now finishes the last row of the strip
every time where the replayed plan left one row bonded; the extra ~16 steps are
the straight-line legs, which are longer paths than the joint interpolation they
replace.

**G2 re-measured both arms properly before deleting the routine** — seven fresh
processes each, medians and IQRs rather than success counts, because success is
at ceiling on this layout for both:

| `tape_peel/handle` | routine (n=7) | graph (n=7) |
| --- | --- | --- |
| success | 7/7 | 7/7 |
| `broken_fraction` | 0.956 [0.956, 0.956] | **1.000** [1.000, 1.000] |
| `peel_front` | 0.933 [0.933, 0.933] | **1.000** [1.000, 1.000] |
| `clear` | 0.825 [0.825, 0.825] | **0.947** [0.947, 0.947] |
| `episode_length` | 340 [340, 340] | 425 [424, 426] |

Neither arm has any spread at all on this layout — seven processes, one number
each — which is what makes a 4.4-point gap on `broken_fraction` a difference
rather than noise. The graph is 85 steps longer and takes the whole strip.

**Run the two arms at the same concurrency, or do not compare them.** The graph
numbers above are from *serial* runs. Four concurrent graph episodes on one GPU
score differently and much worse: every one of eleven runs at concurrency 4 came
back `broken_fraction` 0.000 with the bus closing mid-`lift`, where the same
build run alone gives 1.000. The scripted arm is immune — its seven concurrent
runs match a serial one exactly, 340 steps and 0.956 — because it has no bus and
no wall-clock coupling. So a campaign that oversubscribes the CPU does not
measure a slower graph, it measures a broken one, and it does it silently on
only one of the two arms.

**`tape_peel/along_x`: refused, 7/7, without moving.** The variant is
withdrawn; this is what it measured while it shipped. `reach` routes
`unreachable` at 137.6 mm and only two nodes ever run — `read` and `reach`.
`broken_fraction` 0.0, `peel_front` 0.0, reward 0.0, 106–108 steps over seven
fresh processes, with zero spread on every metric. That is the
recorded behaviour and the correct one: the document itself declares
`(:provenance (:tab-reachable false))`. That clause read
`(:policy-params (routine-solves false))` until G2 renamed it, and the rename is
worth knowing about here: the old spelling was a verdict on a policy, and
measured over seven fresh processes that policy took this tab 1 time in 7 — so
the graph's flat refusal is the *more* stable reading of the layout, not a
weaker one. Refusing *before* the first motion node matters here —
an arm that swung at this tab and then retreated could disturb the strip, and a
disturbed strip on a latching goal is a success nobody performed.

### Run this one episode at a time, and that is not a style note

**A three-episode run reports 3/3 and two of them are peels.** Measured:

| run | episode lengths | real peels |
| --- | --- | --- |
| `--episodes 0:3` | 410, 411, **24** | 2 of 3 |
| `--episodes 0:4` | 410, 410, **38**, **35** | 2 of 4 |
| `--episodes 0:3`, after the hook-leak fix | 412, **58**, **41** | 1 of 3 |
| `--episodes 2:3`, fresh process | **410** | 1 of 1 |

Every one of those episodes scored `SUCCESS reward 1.00`. The short ones peeled
nothing. Episode 2 run *alone* is a real peel, so nothing is wrong with that
episode's start state; what is wrong is running it third. From the third episode
onward every episode in the process is false, and the rate the runner prints is
100%.

**This is open. The obvious explanation was tested and is wrong.** At step 24 of
the third episode, with the workflow still inside the approach and the gripper
never once closed, `broken_fraction` reads 1.0 and `held_count` reads 6 — six tab
particles within 20 mm of a tool centre that has not arrived. Ruled out: the
environment's own reset (an idle two-episode probe comes back clean, and
`Env.reset` restores `particle_q` and the bond field bit-exactly), and leaked
substep/post-step hooks (a real defect, fixed, and the false successes survived
the fix).

Quote pooled rates from **separate processes**, the way the certifier's solve
stage already does for the unrelated reset-drift reason. An in-process batch on this
suite does not merely drift; it manufactures successes.

Expect the workflow's own exit status to read `abort` on a successful run. The
evaluator is authoritative and terminates the episode the moment the peel passes
its gate — typically mid-lift — which closes the bus under whatever node was
still running. The episode row is the measurement; the graph's exit status is
not.

### What made this hard, before the refactor

Six defects, each hidden behind the last, and only two were in the graph:

| Defect | Where |
| --- | --- |
| plan gate included the clearance via | this graph |
| trajectories not streamed (step inputs) | GaP facade |
| 480-step settle per stage | tool call params |
| handle pin never released | this graph's latch |
| worker thread stepping the sim | GaP facade |
| **substep hook added after capture, never re-captured** | this graph's latch |

The last is the one that mattered, and it fails *silently and plausibly*: the
grab bookkeeping is host-side, so the tool reports `latched: True` while the
pin's kernel is simply absent from the captured block. What little peeling then
happened was the fingers dragging the film by friction — 2-15% of bonds against
the routine's 100%. `env.pipeline.invalidate_graph()` after appending the hook is
the fix, and `sim.attach` carries it now.

## Known gaps

**`robot.close_gripper` closes *fully*.** The scripted routine deliberately
pinched to about 0.9 instead, because driving the pads to zero squeezes the
particles out of the jaws right before the latch. `robot.set_grip(fraction_closed=...)`
exists and would express it; the `close` node has not been moved onto it, because
the full close measured at or above the routine on this layout and changing it
would change what every number here means. The routine was deleted in G2, so the
comparison is now a record rather than something re-runnable — which is an
argument for moving the node deliberately and re-measuring, not for leaving it
unstated.

**`(:policy-params (lift-steps ...))` has no reader.** The `lift` node carries
its waypoint count in the workflow, so a document cannot ask for a faster haul.
`film_peel`'s routine read the parameter and is deleted, so as of G2 nothing
reads it at all. Both peel variants declare 220 and the graph flies 220, so no
number here is wrong; what is missing is the knob, and `film_peel/panel_gentle`
is written around the idea that a peel can be pulled too hard.

## Running it

```bash
python scripts/bench_run.py --solve graph --task rgbs/handle --graph gt --episodes 0:1
```

Needs the `[gap]` extra. `--episodes 0:1` is deliberate — see above.

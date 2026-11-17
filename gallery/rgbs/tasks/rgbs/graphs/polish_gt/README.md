# `rgbs/strip` — the shipped gt graph

The graph behind one of `rgbs`'s nine variants. It was `polish/strip`'s, in
`tasks/polish/graphs/gt`; the suite merged and the graph came with it unchanged.

```
uv run python scripts/bench_run.py --task rgbs/strip --solve graph --graph gt --episodes 0:3
```

## Schedule

```
setup   wait out the spindle's ~0.15 s re-spin, verify it runs (dead drum -> abort)
pick    both hands: stage high and outboard of the strip's ends, descend, then
        slide in END-ON along its axis TOGETHER, close in lockstep on measured
        pad force, verify, capture the real grasp offsets
spotc   raise to the drum -> press the CENTRE mark -> RUB it along the spindle
        axis, three out-and-back cycles a round, polling the mark's cleared
        fraction between rounds and stopping the moment it is finished
spotl   SHIFT the whole strip 33 mm -- both hands travelling with it -- and
        press-and-rub the -33 mm mark the same way
spotr   shift the other way, press-and-rub the +33 mm mark
finish  carry to the rest, lower, open both hands, both retreat, settle 150 steps
verify  the evaluator's own gate, with the field fractions in the trace
```

**Both hands hold the strip from the pick to the set-down.** There is no
regrasp, no ledge and no one-handed phase anywhere in this schedule, and that
is the change that made the task finish — see *The schedule that did not work*.

## The rub, and why it runs the way it does

The drum is clamped **sideways**, nose forward at the arms and fifteen degrees
down, so it works on its **barrel**. That makes the contact a band rather than
a disc: 10 mm along the spindle (the drum's own length) and 12–15 mm across it
(set by how far the strip rides into the soft contact — measured at 2.4 mm for
the 3 mm of overdrive this schedule presses with). The marks are **8 mm along
the strip and 18 mm across it**, which is nearly twice the band's length.

So most of every mark is out of reach of any one press, and it stays out of
reach however long the press is held. Measured on the bench rig
(`rgbs/scripts/rgbs_polish_bench.py --press`), at every overdrive from 1 to 6 mm:

    overdrive    press    ride-in    band     mark cleared
       1 mm      1.71 N   0.68 mm   8.1 mm       0.68
       2 mm      2.65 N   1.54 mm  12.0 mm       0.68
       3 mm      3.29 N   2.40 mm  14.7 mm       0.68
       4 mm      4.46 N   3.21 mm  16.7 mm       0.68
       6 mm      6.74 N   4.84 mm  19.8 mm       0.68

Two columns move and the last one does not, which is the shape of the claim:
pressing harder widens the band ALONG the strip, where there was nothing left
to reach, and does nothing at all across it, where the drum's own 10 mm length
is the limit. 0.68 against a gate of 0.95.

The rest is reached by rubbing the strip **along the spindle axis**, toward and
away from the arms. `rgbs/scripts/rgbs_polish_bench.py --rub --cycles 9`, nine out-and-back
rubs of ±6 mm at 0.02 m/s, at the same 3.1 N press:

    rub direction                          holder force    mark cleared
    ALONG the spindle                          0.04 N          1.00
    ACROSS it (along the strip's length)       3.39 N          0.68

Both halves of that table are the point.

*The cleared column.* Across the spindle reaches **no new cells at all** — the
band is already 15 mm in that direction against an 8 mm mark, so sliding along
it re-rubs what is already gone. And the across figure does not move when `k_p`
is swept across a factor of ten: it is where the band ends, not where the rate
does. The claim cannot be tuned into or out of existence.

*The force column.* 3.39 N is `mu * press` almost exactly — 1.10 × 3.1 — because
across the spindle is the direction the drum's own surface runs in at 0.38 m/s.
Along the spindle costs 0.04 N, eighty-five times less, and that is Coulomb
friction rather than a modelling choice: the contact is already sliding at full
speed in one direction, so the cone has nothing left to spend opposing a slow
perpendicular slide. In the cell that 3.39 N would be delivered along the
strip's own long axis, straight out of an end-on pinch.

The solver says the same thing without a holder in the way at all.
`rgbs/scripts/rgbs_polish_bench.py --drag` reads the strip↔drum pairwise force straight off
MuJoCo's contact table and resolves it on the machine's own frame:

    overdrive    normal    along the spindle    across it    mu*N
       1 mm      1.69 N         0.00 N           2.03 N     1.85 N
       3 mm      3.32 N         0.00 N           3.45 N     3.64 N
       6 mm      6.64 N         0.00 N           3.13 N     7.27 N

**Zero**, along the axis, at every press. The drag is not merely smaller in the
rub's direction; the contact has none to give there.

A press-and-rub is therefore what the schedule does, and the direction is not a
free parameter. Shifts **between** marks are still lift-and-shift: 33 mm of
in-contact drag would spend the whole friction budget in the one direction that
costs it.

## What makes it work

**The grasp is real friction — there is no latch anywhere.** Both hands take
the strip on their pads and nothing couples it to them. Five things are
load-bearing, and removing any one of them breaks the episode.

*Both hands enter at the same time.* One at a time, whichever hand arrives
first shoves the free strip 14 mm and both hands end up gripping its literal
tips.

*The jaws stand 46 mm open on a 4 mm strip.* The pads are far deeper than the
part is thick, so the jaw gap — not the strip — decides whether a hand can slide
in end-on without touching it. Swept: at 16 mm the pads meet the strip's own END
and drive it like a vice (81-97 N of contact with the hands still 65 mm out and
still closing, which reads exactly like a grasp on a probe and has no vertical
friction in it at all); at 46 mm the strip does not move by a measurable amount
and both hands arrive where they were asked.

*The close is gated on measured pad force, not on a width.* Ramped to a width,
the jaws keep travelling long after the pads are on the part — `jaw_gap` reads
25 mm while they already press at 66 N — and the over-squeeze walks both hands
from ±58 mm out to ±83 mm, half a pad off the end.

*The gated close ARMS A FORCE HOLD, and the hold is what carries the episode.*
A position-servoed finger has no grip force of its own: what it presses with is
the standing position error the part holds its jaws open by. Every press erodes
that error — the strip settles a few tenths of a millimetre deeper into the
pads, the servo arrives at the target it was given, and having arrived it stops
pushing. Left alone, gated at 30 N: **40 N at the close, ~20 N at the bit, 8 N
after the first mark, 2 N after the second, and the strip is on the bench during
the third.** Nothing slips visibly first; the pads simply stop pressing.
`ToolsPolish.GripHold` closes each hand by 0.04 mm whenever its pads read under
the commanded force, six control steps apart, capped at 8 mm of extra travel.
With it the same carry holds a pad force for the whole episode instead of
bleeding it away.

The hold is set **above** the close gate — 45 N against the gate's 30 N — and
the two numbers are bounded by different things. The gate is a ramp, and what
limits it is what the close may do to the strip's pose: past ~60 N the ramp
walks both hands outboard and drags the part. The hold only ever nudges, 0.04 mm
at a time, against a part already captured. Measured over twenty episodes each:
**30 N holds 19, 45 N holds 20**, at the same episode length.

*The hand grips on its blades, not on their convex hull.* MuJoCo collides every
mesh geom as its convex hull, and this fingertip is an open A-frame whose hull is
**3.13x its volume** with an inner face ~13 mm proud of the pad. Probed at the
instant this pick reported a grasp, before the fix: all four pads **13-14 mm
clear of the strip**, their hulls 1.2-2.3 mm inside it. The task ran that way --
it is a consistent phantom and friction on it carries the part — but every grip
number in the suite was then about a surface that is not drawn and is not where
the rubber is. `yam_i2rt` now sets
`EmbodimentSpec.decompose_colliders`, and the same close reads a real
characteristic: first touch at a 3.0 mm finger command on a 4 mm strip, ~20 N per
millimetre after that, 44.5 N at full close, and the strip moving 0.01 mm while
the jaws shut.

*Grip force is measured per pad, and the weaker one.* `pairwise_force` sums
impulse *vectors*, so a whole-hand reading is the hand's NET push on the part --
which goes to zero exactly when the two pads balance, i.e. when the grasp is
good. With each pad pressing at 44-45 N the whole-hand number reads 0.16 N. The
suite gated on that number for its whole life and got away with it only because
the hull's faces are not parallel, so the grasp was permanently unbalanced.

*It captures the grasp it actually got.* Commanding both hands to their nominal
stations when they landed a few millimetres off makes two stiff arms pull
against each other through a rigid strip for the length of every carry.
Measured: the strip let go two thirds of the way into the first raise.

**Every press aligns on the mark first.** The plan puts the mark under the
drum's contact band and the strip arrives a few millimetres off; the band is
only 10 mm along the spindle, so an uncorrected press reads real force and
polishes nothing — the cells sit outside it. `do_move.py` reads the mark's true
centre and shifts the target by the residual, projected into the pressed plane
so an alignment error cannot turn into a deeper press.

**It rubs in rounds and polls between them.** Three out-and-back cycles a
round, up to three rounds, with `sim.wait_polish` reading the mark's cleared
fraction in between. Splitting it that way costs a few control steps and buys
two things: a finished mark is not rubbed on, and a mark that is going nowhere
is abandoned after a round instead of after nine cycles.

## The schedule that did not work

This graph used to put the strip down between marks and hand it to one arm —
`spotl` parked it on the rest, opened the work hand and pressed the -33 mm spot
with the support hand alone; `spotr` re-picked with the work hand and released
the support one. It scored 0.00-0.70 and never finished, and the reason is
worth keeping because it is a fact about the cell rather than about tuning.

*One hand cannot react the press torque.* Note the claim, because an earlier
version of this file made a different and **wrong** one: that a single YAM hand
could not hold the strip at all, on the evidence that opening either hand
dropped the other's pad force to 0.1 N and the strip fell 18 mm. Both halves of
that were artifacts — the grasp was on the fingertip's convex hull, and the
0.1 N was a whole-hand NET contact force, which reads ~0 for any *balanced* grip
whether one hand holds the part or two.

Re-measured on the blade, with grip force read per pad: a single hand holds this
strip **comfortably**. Released in free air after a two-handed pick and left for
a thousand control steps, the strip fell 0.0 mm (work) and 0.2 mm (support),
with 30-38 N steady on the pads and 0.6-2.7 mm of slip. The regrasp works too:
the work hand opens, retreats, and the support hand carries the strip alone to
the drum at 49 N.

What it cannot do is hold it *while pressing*. The mark sits ~78 mm from the
grasp, so even a 4 N press is ~0.3 N·m about it, and the grip's couple is only
as wide as the strip is **thick** — 4 mm. Reacting that needs ~75 N a pad
against the ~35 N a hand makes. So the strip rotates out of contact:

    two-handed press    5.9-8.0 N at touchdown, 7.6-7.9 N through the dwell,
                        1 mm of slip, mark clears
    one-handed press    2.8-3.3 N at touchdown, 0.00 N through the dwell,
                        slip growing 4 -> 64 mm per cycle, nothing clears

Two hands fix it because the second grasp is 116 mm from the first, which is a
wide couple rather than a 4 mm one. It is marginal rather than impossible: of
two one-handed press attempts the work arm did clear its mark (5.6 N held
through the dwell) and the support arm did not.

**The marks were never out of reach of one grasp.** That was the premise of the
regrasp and it was wrong: the hands hold at ±58 mm and the marks sit at 0 and
±33 mm, so reaching the outer two is a 33 mm translation of the whole rigid
body, both hands with it. Measured, it costs nothing at all — no clearance
violation, no reach failure, and the arms stay well inside the machine's
corridor.

## Measured behaviour

**Unchanged by the `rgbs` merge, and checked rather than assumed.** The merge
moved this variant into a suite and swapped its camera rig; it did not touch the
cell, the robot, the arrangement, the layout or a line of this graph. Built specs
before and after differ in the cameras, the suite name and nothing else. Run
either side of it on one episode: `polish/strip` SUCCESS 1.00 at 5124 steps,
`rgbs/strip` SUCCESS 1.00 at 5392 — both inside the 5068–5945 band below, which
is where run-to-run variance on this task lives.

**10 of 10 episodes score, at reward 1.00**, in a mean 5588 of the 12000
allowed control steps (5068–5945): every mark **fully** cleared —
`polished-fraction` is 1.000 in all ten, not 0.96 scraping a 0.95 gate — the
strip back on its rest at its settled height, both hands 200 mm clear against a
100 mm gate.

That is on the sideways drum, with a press-and-rub schedule. The nose-down
build it replaces measured 18-20 of 20 at a mean ~5300 steps across three
twenty-episode sweeps, and the spread between those was noise rather than
signal (19/20 with the force hold at the close gate's 30 N, 20/20 at 45 N,
18/20 once the fingertip collider was decomposed and no grip constant was
re-tuned for it). Its failures all looked the same: the strip lost after the
second mark, episode scores 0.65.

Ten trials of one layout, so the number is run-to-run variance and not
generalisation; the scene does not vary. It is real variance, though: this stack
is not deterministic run to run, and two identical probes measured 27 N and 39 N
of pad force at the same control step. Nothing here was concluded from a single
run.

**Rubbing costs the grasp, so the schedule buys as little of it as it can.**
Three cycles a round, three rounds, and a per-mark target of 0.99 that lets a
finished mark route `done` and skip the rest. Four rounds and a target no mark
could reach was tried and is the instructive failure: 10084 control steps
instead of 6300, and what broke was not the polishing — `polished-fraction`
still cleared its gate and the episode still scored reward 1.00 — but the extra
rubbing eroded the grip past `GripHold`'s 8 mm budget and the strip came down
BESIDE its rest. Reward 1.00, `success` false. The lesson is the same one the
press force teaches: on this rig the grasp is the scarce resource, and every
second in contact spends it.

That failure was against 20 mm marks, and it does not reproduce on the 18 mm
ones that ship now: the same four-round config re-run against them scores in
4944 steps. No mark needs a fourth round any more, so it is never spent. **The
guard is the reach margin, not the round count** — which is worth knowing before
someone widens a mark again and wonders why the episode got long.

The decomposed collider costs about **29% of the step rate** (91 -> 65 steps/s)
for ~212 extra collision geoms across the four fingers, and buys a hand that
grips where it is drawn.

The press runs at 3-4 N, which is **half** what this first shipped with, and the
halving is the second change that made the difference. The drum's drag on a
pressed strip is `mu * press` in the strip's own plane, so a 9 N press puts ~9 N
of sideways load into a grasp whose entire friction budget is the pads' normal
force. At 6 mm of overdrive the strip walks across the pads about a millimetre
per press and is gone by the third mark; at 3 mm it does not walk.

Against a barrel that number gained a floor as well as a ceiling, and they do
not fight. Pressing deeper widens the band ALONG the strip — 8.1 mm at 1 mm of
overdrive, 14.7 mm at 3 — and the mark is only 8 mm in that direction, so below
about 2 mm the mark's own length stops being covered and a rub along the strip
would start to gain cells. That is the direction that must gain nothing. 3 mm
is where the grasp's ceiling and the band's floor both hold.

Repositioning **between marks** is always lift-and-shift, never an in-contact
drag. The rub itself is an in-contact drag and has to be — but it runs along the
spindle, where the measured Coulomb cost is 0.04 N. A 33 mm reposition would run
along the strip, where it is the full `mu * press`.

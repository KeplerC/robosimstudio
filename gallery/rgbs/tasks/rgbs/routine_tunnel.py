# SPDX-License-Identifier: Apache-2.0
"""A scripted solution: hold the box, pinch the cable behind the mouth and walk it in.

**Feeding is not carrying, and the difference is the whole routine.** A weave
takes hold of a rod and moves it somewhere; the rod goes where the hand goes. A
feed drives a rod *into* something that resists, so the hand is not transporting
a load -- it is loading a column, and a column has a limit that has nothing to do
with the arm. Three consequences, and each one is a number below rather than a
preference:

1. **The grip is taken as close to the mouth as the box allows.** The free span
   between the pinch and the entry face is the column's length and the buckling
   load goes as its inverse square, so the 45 mm the flare and the hand's own
   width cost is not clearance -- it is most of the margin.
2. **The stroke is short and there are several.** One long push has the free span
   at its longest exactly when the resistance is highest, because the resistance
   grows with how much rod is in the bore. Short strokes keep the worst case at
   the start of each one rather than at the end of all of them.
3. **The support hand holds the box, not the cable.** The channel is on a free
   body now, and nothing but the pinch on its lid keeps the bore where the
   regions were measured. It takes the box first and keeps it for the whole
   feed, which is what the reference clip shows a person doing.
4. **Letting go is a move, not an instant.** A commanded-open hand is still a
   closed hand for a hundred steps while its pads unload a cable they grip at
   ``mu`` 60, and everything the arm does in that window it does to the cable --
   which, threaded through a bore on a free box, means to the box as well. So
   the release waits for the fingers to read open (:data:`OPEN_WAIT`), then
   comes off sideways if one is stuck (:meth:`RoutineTunnel._free_jaws`), and
   only then moves back down the cable and lifts. Fed by a fixed 30-step hold
   instead, the same routine dragged the rod back out on every stroke and
   finished at -143 mm of advance with the box 90 mm out of place.

**The bore is 100 mm up, and the cable comes to it up a ramp.** The conduit's
floor was the bench, so the feeding hand pinched a rod lying on a table at a
height it could assume. Here the cable's rest shape is the clip's own drape --
flat on the mat behind the box, up the 130 mm hang the rod actually holds, then
along the bore axis (``tunnel/layout.py::cable_nodes``) -- so the pinch
is neither on a table nor in mid-air: 90 mm back from the entry the cable lies
88 mm under the axis and 10 mm above the mat, nearly flat, and the push drives
it up the hang and in. Each regrasp reads where the cable actually is
(:meth:`RoutineTunnel._pinch_capsule`) rather than assuming a height -- the hang
the first stroke pinches becomes a riser beside the box once the slack is fed
in -- and the push carries it up to the bore's own line clear of the flare,
which keeps the rod ahead of the pinch a straight column rather than levering
it against the lip.

Nothing here uses the latch, and the document's cable is built without one
(``tunnel/layout.py``, ``allow_grasp_latch``). ``cable_route`` found that its own latch was
covering for a hand that never touched the rod -- with the pin disabled its weave
made zero gripper-vs-rod contacts and moved the cable 0.1 mm -- and a pin that
does all the work cannot be told apart from a grasp by looking at the score. A
pinch at the depth that puts the pads around a rod carries one; this asks for the
same pinch and reads the contacts back through :meth:`row_extras` so that "it
never touched it" and "it touched it and could not move it" stay different
sentences.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from robosimstudio.arms import arm_slot, arm_slots
from robosimstudio.core.math import quat_to_mat, transform_matrix
from robosimstudio.core.render.visual import shape_geometry
from robosimstudio.robots.embodiments import top_down_grasp_site_quat, work_gripper

from .tunnel.box import BORE, BOX_HEIGHT, FLARE, ROOF, RUN, WALL
from .tunnel.plug import PLUG_W
from .tunnel.rod import RADIUS

__all__ = ["RoutineTunnel"]

PINCH_ENGAGE = 0.0045
"""How far each pad presses into the rod when closed [m].

It was ``cable_route``'s half millimetre, and half a millimetre into a proxy
contact is a pinch the cable slides through: the take's first stroke moved the
hand 41 mm and the capsule between the pads 29, its second 54 and 31, and
every stroke after that 31 and 0.0 -- the fingers read their commanded 13.9 mm
throughout (``pinch_q_mm``), which is exactly what a pinch that holds nothing
reads. Two millimetres a side (measured beside it on the same scene: hand
42 mm, capsule 41, tip 35) is a grip on the hang; 4.5 -- the faces asked to
half a millimetre of the axis, which the closed blades cannot reach, so the
servo stalls on the cable at full force -- is the grip that also carried the
capsule 29 mm for 30 of hand with the cable ten millimetres off the mat,
where two millimetres carried it nothing. It is what ``PAD_INBOARD`` exists
to warn about, and the release is eased and unloaded (:data:`RELEASE_UNLOAD`)
because of it."""

HOLD_ENGAGE = 0.003
"""How far each pad presses into the LID when the support hand takes the box [m].

Its own number rather than :data:`PINCH_ENGAGE`'s, because the two pinches
are asked for opposite things: the cable pinch must not squeeze a rod that is
about to be released, the lid pinch must not let a 350 g box go anywhere. At
half a millimetre the pads read 1.9 N on the lid, and the take showed what
that buys: the first stroke's release levered the jaws 7 mm open and the box
went 15 mm sideways with three degrees of yaw, so the cable met the mouth at
an angle and every stroke after the second slipped."""

LIVE_BOX = True
"""Whether each stroke reads the mouth, the entry and the flare clearance off
the box's LIVE pose rather than the layout's constants.

The layout's numbers are where the box was placed; the box is a free body and
the take had it 17 mm along the run and 17 mm across it from there by the
second stroke, so a push aimed at the layout's mouth was a push aimed 17 mm to
one side of the real one. False reproduces the shipped behaviour."""

PINCH_UNDER = 0.016
"""How far below the cable's centre the pad tips reach when the feeding hand
pinches it [m].

**Sixteen on ``yam``.** The arena hand's pads are parallel faces 37 mm tall,
not a V, so nothing below argues for a shallow pinch on it, and a capsule held
10 mm over the tips was pushed out under them when the stroke carried the
cable up a steep hang. Sixteen keeps the capsule in the middle of the faces
and the tips :data:`TIP_CLEAR` off the bench (see
:meth:`RoutineTunnel._pinch_capsule`). The i2rt reasoning below is kept for
the blades it was measured on.

**Ten, and not deeper, because the closed blades are a V.** Read off the
finger colliders in 10 mm bands at a closed pinch
(:meth:`RoutineTunnel._report_pinch`): commanded to a 6 mm gap, the pads stand
8-10 mm apart in the band at the tips and converge by about 2 mm a band above
it. A cable whose equator sits at the tips is held by the narrowing V above
it, and a hand that lifts wedges it tighter -- which is what carried it on
every hang pinch measured (capsule 41 mm for 42 of hand). Taken 20 mm deep,
with the equator in a band already narrower than the cable, the same V
ejects it downward through the open tips the moment the hand rises: 0.6 mm
of capsule for 19 mm of lift, the contact rows falling from 74 to 30 as the
pads left it. So the pinch is shallow on purpose, and it is taken where
there is air under the cable (:data:`PINCH_OUT`).

**The conduit's pinch had a table under it; this one has air.** On the bench
the site was sent to ``surface + pad_drop`` -- tips on the table, a radius under
the cable's centre -- and the table is what kept a squeezed cable between the
pads. Sent to the same height off a hanging cable the arm settled 5.6 mm above
its command (free air, no contact to sag against), the tips arrived level with
the cable's centre, and the closing pads overlapped only its upper half and
squeezed it out *under* them: the fingers stalled at q 6.6 mm with their faces
inside the cable's own radius, one pad on top of the cable. The feed still
advanced 46 mm (that pad dragging the cable along), and the open command then
carried the cable 16 mm sideways on that pad and shoved the box 10 mm through
the bore. Measured, ``rgbs/scripts/rgbs_tunnel_bench.py grip``. Ten millimetres puts the
tips a diameter under the centre with the arm's settle taken out, so the cable
is between two vertical faces and has nowhere to go but stay."""

RELEASE_ACROSS = 0.010
"""How far the opened hand steps ACROSS the cable before it moves along it [m].

The first move after the grip opens, and :meth:`RoutineTunnel._free_jaws` only
makes it when a finger is actually still shut on something -- a hand that
opened cleanly has nothing to get off."""

RELEASE_UNLOAD = 0.005
"""How far the still-closed hand backs off along its push line before it opens [m].

**Five on ``yam``**, whose pads end a push 15 mm off the flare and never touch
the box (the contact table reads 0.0 N there): fifteen only drew the cable
back, 11 mm of the first stroke's 48. The i2rt reasoning below is what the 15
was for.

Read off the MuJoCo contact table: at the end of a push the finger carriage
stands on the flare (:data:`CARRIAGE_REACH`), pressing the box with 9 N,
and a hand opened there presses it with 17 N by the time the jaws are wide,
because the finger against the box is the one that cannot open. Fifteen
millimetres back along the line takes the carriage clear (it meets the
flare over the last 15 mm of a full push) and lets the column's compression
out through pads that are still closed, so the open that follows moves
nothing. The cable comes back with the hand by less than this -- the pads
slip before the bore does -- and what it gives back is the price of a
release that leaves the box where it was."""

RELEASE_BACK = 0.030
"""How far the opened hand slides back down the cable before it lifts [m].

Along the cable, which is the one direction a pad still touching it cannot drag
it in. Three capsule pitches: far enough that the pads are clear of the segment
they pinched, short enough to stay inside the stroke just made."""

STALL_GRACE = 45
STALL_WINDOW = 24
STALL_ADVANCE = 0.0015
BOX_BUDGET = 0.0015
"""When to stop pushing: less than 1 mm of tip travel over 12 steps, or 1.5 mm
of box movement [steps, m, m].

**A push that is not advancing the cable is a push into the box.** The hand
drives 0.5 mm a step and a sliding cable takes about 0.35 of it; when the run
stops taking cable -- the head meeting a wall, the column bowing, the bore's
friction winning -- the rest of the stroke loads the rod against the bore, and
that load comes out on the box, which is a free body held by a pinch. Measured
on the stroke that showed it: the hand held its pose at the end of the push and
the box moved 18 mm in the next 48 steps, with the yaw going with it. Reading
the tip is what tells the two apart, and it is the same reader the advance
telemetry uses.

**A stroke starts by taking up slack, so the check waits.** The first 45 steps
of a push move the hand 22 mm and the tip barely at all -- the pads settle on
the cable, the hang straightens, the capsules take up -- and a stall check
running from step one reads that as a jam and stops every stroke dead: measured,
the feed made three strokes and moved the tip 0.0 mm. :data:`STALL_GRACE` is
that take-up, and the check runs after it.

**And the window has to be short, which is the second measurement.** At 36
steps it was 18 mm of hand travel after the jam, by which point the arm was
pressing the box through the support hand's pads at **19 N** -- read directly
off ``env.contacts`` (:meth:`RoutineTunnel._hold_force`), against the 1.8-2.5 N
that same pinch reads at rest -- and the box had gone 20 mm. Twelve steps is
6 mm of hand travel, and a sliding cable still moves 3-4 mm in that window
against the 1 mm asked.

**The box's own displacement is the other stop, and it is the honest one.** The
thing the task promises is that a held box does not move; so the push watches
it, and gives up the moment it has spent :data:`BOX_BUDGET` of the 2 mm the
document is scored against. A control law that keeps pushing after the work has
started moving the fixture is not feeding a cable, it is bulldozing."""

EASE_OPEN = 120
"""Over how many steps the feeding hand eases its grip open [steps].

**A pinch is a loaded spring and a snapped release is what it does to the
scene.** The pads squeeze the cable, the cable is bent around the mouth and
compressed along the run, and commanding the grip from shut to wide in one step
lets all of that go at once: measured across four rollouts, the box sat within
1.5 mm through the whole push and then moved 15-17 mm in the sixty steps that
followed the open command, with both arms standing still and the arm's own
command frozen at its measured pose. Nothing the arm did caused it; it was the
cable.

Eased over two seconds the same energy leaves as creep rather than as a kick.
This is also what a hand does: a person feeding a cable relaxes the pinch, they
do not spring it."""

OPEN_WAIT = 180
"""How long the hand waits for its fingers to ACTUALLY open, before it moves [steps].

**The bug this exists for, and it cost a whole rollout.** Commanding the grip
open does not open it: the pads are squeezing a cable whose own friction against
them is ``mu_gripper`` 60, and a finger unloads at about 0.4 mm a step --
measured, one finger was still at 9 mm of its 47.5 mm travel thirty steps after
the open command, while the other was already wide. Everything the hand did in
that window it did to the cable: a rollout that moved on after a fixed 30-step
hold dragged the rod back out of the bore on every stroke and finished at
**-143 mm of advance** with the box shoved 90 mm, with eight strokes that each
fed and then took back more than they fed.

So the hold is a condition rather than a count: :meth:`RoutineTunnel._await_open`
waits until both fingers read open, and this is only the cap that keeps a stuck
jaw from hanging the episode -- at which point :meth:`_free_jaws` gets it off
the cable sideways. 180 steps is three seconds, against the ~120 a finger takes
to travel the full 47.5 mm under load."""

OPEN_ENOUGH = 0.85
"""What share of full travel counts as open, for :data:`OPEN_WAIT`. The last
few millimetres are the slowest and the cable is long clear by then."""

PINCH_SETTLE = 30
"""Steps the feeding hand waits at the pinch height before closing. A free leg
lags its command by a few millimetres at the end; the close waits it out."""

GRIPPER_CHANNEL_IS_NORMALIZED = True
"""The action's gripper slot is a 0-to-1 command, not finger travel in metres.

0.0 is **open** and 1.0 is **shut**, and a width in metres reaches it through
``cable_route/routine.py:487``'s conversion::

    command = 1.0 - travel / arm.gripper_open

Everything in this module that names a grip is therefore per-finger travel in
metres -- ``arm.gripper_open`` (0.0475) wide open, 0.0 shut -- and :meth:`_action`
is the single place it becomes a command. A pad gap is twice a travel.

Keeping width and command in one variable went wrong twice here, in opposite
directions, and both times the hand looked plausible in the viewer:

* **Metres fed straight into the channel.** ``gripper_open`` (0.0475) and
  ``radius - PINCH_ENGAGE`` (0.0045) both clamp to the low end, so open and shut
  emit the same command and the hand never actuates at all. Measured: 47.80 mm
  against 48.03 mm of jaw-body separation, a fifth of a millimetre apart.
* **The repair, which inverted the sign.** A calibration table was built by
  driving commands and reading ``|tip_left - tip_right|``, then read as if the
  smaller separation were the shut hand. It is the open one: on this gripper the
  fingertip *origins* travel **apart** as the blades converge -- 48 mm open,
  102 mm shut. ``scripts/measure_gripper.py`` states this, and the i2rt module says
  "never read ``|tip_left - tip_right|`` as a gap". So every ``_shut()`` resolved
  to about 0.06 -- commanded wide open at every grasp. Measured there, the
  nearest pad *collider* vertex stood 17 mm clear of the cable's surface on both
  jaws, and the episode made zero gripper-versus-rod contacts.

The lesson that outlived both: a jaw gap is measured from the pad **colliders**,
never from body origins. ``rgbs/scripts/rgbs_tunnel_bench.py grip`` reports collider
clearance for that reason."""


HAND_HALF_Y = 0.010
"""How far the closed hand reaches along the run, toward the box, from its own grasp site [m].

Measured off ``yam``'s colliders in the grasp the routine takes (wrist away from
the box, :data:`WRIST_AWAY`): the fingers span 32 mm behind the site and 7 mm
ahead of it, and nothing of the hand within 120 mm above the site reaches more
than 10 mm ahead. The i2rt hand's blades reached 25-27 mm, which is what this
was. Used where only the pads are near anything -- the end of a push and the
plug grab."""

CARRIAGE_REACH = 0.045
"""How far the finger CARRIAGE reaches along the run from the grasp site [m],
40 to 90 mm above the pad tips. Measured off the colliders, and the number the
whole failing take turned on.

Each finger link is a blade on a carriage, and the carriage of one finger
stands 41-45 mm along +y from the site (the other's 22). A push that ends
with the pads :data:`PUSH_CLEAR` from the flare -- the old ``HAND_HALF_Y`` of
20 mm -- ends with that carriage at the flare's lip: read directly off the
MuJoCo contact table (:meth:`RoutineTunnel._work_box_force`), the feeding hand
was pressing the box with 9 N when the box budget stopped the push and with
17 N by the time its jaws had opened, because the finger leaning on the box is
the finger that cannot open, and a servo dragging it open drags the box: 14 mm
and three degrees of yaw per release, the support hand's own jaws pried 7 mm
apart. Every stroke after that fed a cable bent 17 mm sideways at the mouth
through a pinch that could not hold it. Ending every push this far out was
tried and leaves a 20 mm stroke from a pinch that then sits where the cable
has no air under it; so the push runs to the pads' own clearance and the
hand backs off by :data:`RELEASE_UNLOAD` before it opens, which is what the
contact-table reading says the carriage needs."""

PUSH_CLEAR = 0.005
"""Air between the pads' reach and the flare's outer lip [m]. The arm settles
within 3 mm of a commanded point at this reach (measured)."""

STROKE = 0.080
"""The most one push may travel along the cable [m]. 80 on ``yam``, where a push
runs from the riser to the bore's line (see :data:`PINCH_OUT_MIN`); it was 55.
And, with
:data:`FEED_PAST_EXIT`, what keeps the feed alone from reaching the exit
region (``tests/test_rgbs_tunnel.py``). A 55 mm push along the 45-degree line from
:data:`PINCH_OUT` advances the hand 39 mm along the run, to the pads'
clearance from the flare. Measured on the built scene a push nets about 85
per cent of its hand travel at the tip, and the unloaded release gives some
of it back."""

STROKE_YIELD = 0.6
"""What a stroke nets, as a share of :data:`STROKE`. Measured: 40 mm pushes
netted 19-33 mm of tip travel each. What is lost is the rod's own springback
when the jaws open and the drape's pull -- the hanging cable draws the rod back
out of the bore at about a millimetre a second while nothing holds it, which
over a regrasp is 5-8 mm. Not driven by anything; it is what
:data:`STROKES_MAX` and the fast tier's arithmetic are stated against."""

STROKES_MAX = 12
"""The most feed strokes the routine will take.

**The feed stops when the tip is out, not after a count** -- see
:data:`FEED_PAST_EXIT` -- so this is a ceiling against a feed that is not
advancing, not the number of strokes an episode has. It was nine; the
release now backs off before it opens (:data:`RELEASE_UNLOAD`) and gives
some of each push back, so the net stroke is nearer 30 mm than 45 and the
tip has 197 mm to go from where the settle leaves it to
:data:`FEED_PAST_EXIT`. Twelve strokes at roughly 550 steps each, the hold
and the draw fit the document's 8000."""

FEED_PAST_EXIT = 0.020
"""How far past the exit face the tip has to be for the feed to stop [m].

Read live off the rod's last capsule before every stroke. Enough that the plug
-- whose overmold's centre leads the tip by 16 mm -- is out on the ledge for the
second hand to take, and, with one more :data:`STROKE` at full yield, still
short of ``beyond_exit`` 80 mm out: the feed alone must not satisfy the goal,
and this plus a whole stroke is 75."""

PINCH_OUT_MIN = 0.045
"""The nearest behind the entry the feeding hand pinches the cable [m].

The far end of the window :meth:`RoutineTunnel._pinch_capsule` searches, with
:data:`PINCH_OUT` the near one. **Why a window and not a place**: the feed draws
the slack in, and the hang the first stroke pinches 90 mm out becomes a riser
standing 40-60 mm off the entry within two strokes, with the cable at 90 mm
lying on the table. Pinched there, a push compresses the riser like a spring:
measured, the third stroke moved the pinched capsule 35 mm and the tip 9, and
stalled. Pinched on the riser, the column is the riser itself. 45 mm keeps the
pads (which reach :data:`HAND_HALF_Y` ahead of the site) 15 mm off the flare's
outer lip."""

TIP_CLEAR = 0.004
"""How far above the bench the pad tips must stay at a pinch [m]: what tells a
capsule on the hang from one lying on the table."""

PINCH_OUT = 0.090
"""How far behind the entry face the feeding hand pinches the cable [m].

**On the hang, 26 mm above the mat, and that is what places it now.** The
pinch wants air under the cable: taken 125 mm out, where the cable lies on
the mat, the same closed pads carried 0.2 mm of capsule for 31 mm of hand
(34 contact rows against the hang pinch's 66-75) -- a V that can only push
a cable down cannot hold one that is already on a table. Here the tips
graze the mat and the cable can wedge upward into the blades. The conduit's
pinch was 90 mm out too, on a bench.

**The column arithmetic, kept for the record.** The rod between the pinch and
the mouth is the column the push loads: from 90 mm out the cable is 88 mm
under the axis (``layout.py::HANG_AT_PINCH``), the chord to the flare's lip
116 mm and its Euler limit 41 mN against a 21 mN load
(``tests/test_rgbs_tunnel.py`` states the sum). Measured, the feed runs two
orders above that arithmetic: the support pinch reads 4-7 N through the box
during a push with the pinched capsule tracking the hand and the column
intact, so at material parity this rod is not the column the figure
describes. The stroke no longer runs along that chord anyway -- see the
choreography: it lifts first."""

ENTRY_HOLD = 0.035
"""Where along the lid the feeding hand pinches it once the feed is done [m],
in from the entry face: far enough that its fingers, which reach 32 mm back
from the grasp site toward the wrist, stay over the lid rather than the flare.
It holds the box while the other hand draws the connector."""

HOLD_FROM_EXIT = 0.015
"""Where along the lid the support hand pinches it [m], short of the exit.

**The second hand holds the box, not the cable.** Two hands on the rod is the
obvious reading of a bimanual feed and it does not fit this bench: the only place
a hand can take the rod is the 90 mm between the flare and the free span behind
it, and two grippers do not share 90 mm. One hand on the work and one on the box
is what the reference clip shows, and now that the box is free it is what keeps
the bore where the document measured it.

At the lid's far end rather than its middle, as the clip has it, and as close
to the exit as the pads allow: they are 25 mm wide along the run, so 15 mm
short of the exit puts their far edge at the lid's own end, where the head
emerging onto the ledge cannot meet them -- and every millimetre toward the
exit is a millimetre nearer the left arm, which is at the far end of its reach
here (``rgbs/scripts/rgbs_tunnel_bench.py reach``). The pinch is across x: one pad in front
of the box's front face, the other down the slot behind the lid."""

LID_CLEAR = 0.014
"""How much wider than the lid the jaws are set for the descent [m], per side.

The jaws cannot come down open: the back pad would land on the mat, which is
what stands behind the slot. Set to the lid's half-width plus this, the pads'
faces come down 4 mm clear of the lid's faces (:data:`PAD_INBOARD`), the back
pad's outer face 13 mm short of the mat's edge, and the pinch closes the rest.
**It was 4 mm, and 4 mm was the whole failure of the first held run**: the
faces were 6 mm inside the lid's width, the front pad landed on the lid's
front edge, the hand stopped 20 mm high with its pads closed on air above the
strip, and the box -- pressed rather than pinched -- was shoved 30 mm by the
first stroke's lift. The descent is also corrected by the offset the hand
actually settled to at its approach point (:meth:`RoutineTunnel._settled_offset`,
3 mm at this reach)."""

HEAD_DESCENT_CLEAR = 0.014
"""The same descent margin for the plug [m]: the jaws come down at the head's
half-width plus this, so the back pad is over the slot rather than the mat.
See :data:`PAD_INBOARD` for why 14 and not 4."""

PAD_INBOARD = 0.0
"""How far inboard of the finger's travel coordinate the pad's contact face
sits, in the top-down grasp [m].

**Measured off ``yam``'s finger colliders in the TCP frame**, finger by finger
against the commanded coordinate: on the arena hand a pad's face stands at the
finger's own ``q`` from the grasp axis, to within a millimetre over its lowest
30 mm (``q - 0.7`` mm at the nubs, ``q + 0.1`` at the tips), so nothing is
inboard. It was 9.4 mm on the i2rt hand this suite was first written for, whose
blades stood 9.4 mm inside their coordinate; asked for that hand's widths, the
arena hand's pads closed 9.4 mm wide of every one of them -- 26.5 mm of travel
around a 16.5 mm half-lid, a hold that touched nothing.

**Every width this module commands goes through it, and that is a repair.** The
gripper channel takes finger travel, and this routine read "half the cable, less
half a millimetre of interference" straight into it -- 4.5 mm on a 10 mm cable.
With the faces 9.4 mm further in, that is not half a millimetre of interference,
it is **nine millimetres**: a command to crush a 10 mm cable to nothing, which
the blades answer by stalling on each other around 7 mm with the servo at full
force and the rod's contact loaded like a spring. Measured, that spring is what
moved the box -- the instant the jaws were told to open, the box jumped 13-18 mm
and took three degrees of yaw with it while both arms stood still.

The same lesson this file already carries one level up ("a jaw gap is measured
from the pad **colliders**, never from body origins"), applied to the grip and
not only to the clearances. :meth:`RoutineTunnel._finger_for` is the crossing."""

CARRY_CLEAR = 0.060
"""How high the feeding hand lifts off the cable's axis to travel back for the
next stroke [m]."""

SUPPORT_OVER = 0.030
"""How high above the lid the support hand's approach sits [m].

Not :data:`CARRY_CLEAR`: the lid's end is at the far edge of the left arm's
envelope, and a waypoint 60 mm over it comes back 25.6 mm short with the
wrist straight down (``rgbs/scripts/rgbs_tunnel_bench.py reach``); 30 mm solves. The
approach is a free leg either way, but a free leg that lands short comes down
diagonally onto the lid with the jaws set 4 mm wider than it."""

WRIST_AWAY = 0.02
"""How much grasp score [unitless] a hold may give up to stand its wrist away
from the box.

``yam``'s grasp point is 44 mm off the wrist axis and the wrist carries a D405
camera 80 mm further out the same way, so the two holds that pinch a rod across
equally well differ by where the hand's bulk stands. Turned toward the box, the
feeding hand landed on it 15 mm behind the cable and stopped 69 mm over the
pinch, pressing the box with 49 N; turned away, it reaches the cable. On the
i2rt hand, whose grasp point is on the wrist axis, the two tie exactly and
this changes nothing."""

PULL = 0.070
"""How far the support hand draws the connector along the ledge [m].

Enough that the head and a working length of cable are out in the open where the
picture shows it, short enough to stay inside the arm's envelope and on the
ledge -- the box's left end is 170 mm past the exit."""

PULL_SEAT = 0.030
"""How far the connector is drawn before the hand lifts it [m].

Short of what carries ``cable.end_b`` into the exit region, on purpose -- see the
ordering note in :meth:`RoutineTunnel._choreography`. It also buys the lift some
free cable: lifting a head that is only just clear of the mouth levers it against
the exit lip instead of raising it."""

LIFT = 0.060
"""How high the support hand raises the connector off the ledge once it is out [m].

**The move that distinguishes having the connector from being next to it.** The
pull alone is horizontal, and a horizontal draw along a ledge is what a hand
would produce whether it had closed on the head or merely barged it -- measured
on the conduit's first version, the connector's height varied by 0.1 mm across
an entire episode. Picking it up cannot be done by barging.

Set down again before the hand opens: the goal asks whether the cable has come to
rest, and a connector released from 60 mm is still falling when it is asked."""

HEAD_CLEAR = 0.006
"""How far past the exit face the head must be before the hand goes for it [m].

A grasp taken flush with the mouth closes the pads on the lid's own end. Not
measured -- it is a keep-out, and the cost of it being generous is a few
millimetres of pull."""

MAX_POS, MAX_ROT = 0.006, math.radians(50.0)
"""What a planned leg may be off by before it is refused rather than driven.

An unreachable waypoint comes back as a solution tens of millimetres out with
nothing raised, and streaming one with a rod in the jaws is a whip. **Position is
the gate that means something here; the rotation term is a backstop against a
wild flip and nothing finer**. It was loose for the i2rt hand, whose grasp was
chosen against a legacy closing axis that wrist could not hold (a standing 35-39
degrees); ``yam``'s holds solve to zero rotation, and 50 degrees still catches a
basin flip."""

VERBOSE = True
"""Whether the routine narrates each phase to stdout.

**What the take could not say, the routine now says for itself.** A 50-minute
episode that ends at reward 0 with ``stalls 7`` and ``advance_mm 60`` is a
verdict, not a diagnosis; the tape had to be replayed off-line to learn that
the second stroke fed 48 mm and every stroke after it moved the hand 31 mm
over a cable that did not move at all. So each phase prints what it is trying,
where the tip is against the live bore, where the pads landed against the
capsule they closed on, and why a push stopped. Off, the routine is silent."""

SETTLE = 90
"""Steps of stillness at the open and the close of the episode.

The opening one is where the cable's tail falls off the box's end and lies down
on the mat, and where the box settles its half-millimetre onto the table; the
box's rest pose is read at its end. The closing one is the goal's own:
``(<= (speed cable) 0.05)`` and ``(> (tcp-distance cable) 0.05)`` are read
wherever the episode leaves the rod, and a verdict taken on the release frame
is a verdict taken mid-spring."""


class RoutineTunnel:
    """Hold the box, feed the cable through its channel one short stroke at a
    time, then draw the plug out along the ledge.

    Args:
        env: The environment to drive. Its task must stage a cable box and a rod.
        hold: Whether the support hand takes the box before the feed. ``False``
            is the measurement the hold is justified by -- the same feed against
            an unheld box -- and nothing a document stages.
    """

    MODE = "joint_position"
    """The action space this emits into. Read by ``SolutionRoutine``, which is
    what lets a Cartesian routine and a joint-space one share a scene."""

    def __init__(self, env, *, hold: bool = True, **kwargs: Any) -> None:
        del kwargs
        self.env = env
        self.hold = bool(hold)
        slots = arm_slots(env)
        self.work = env.handles.work_arm or slots[0]
        self.support = env.handles.support_arm
        self.dof = len(env.idle_action(self.MODE)) // len(slots)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        env = self.env
        slots = arm_slots(env)
        self.held = {name: self._q(name) for name in slots}
        self.grip = {name: self._open(name) for name in slots}
        self.home_seed = np.asarray(env.state.joint_q.numpy(), dtype=np.float64)
        """The joint vector every stroke re-plans from, and the one grasp
        orientations are chosen against.

        A leg chain picks the wrist basin and the basin picks the roll: solved
        from here either hand puts its pads across the rod, and chained across
        six strokes the same hand arrives a quarter turn out, shoving the cable
        rather than holding it. Nothing raises -- the residual is a rotation the
        gate then refuses, so the push is skipped too."""
        self.seed = {name: self.home_seed.copy() for name in slots}
        self.refused: list[str] = []
        self.strokes_done = 0
        self.advance = 0.0
        self.grasp_contacts = 0
        self._plug: int | None = None
        self.head_pulled_mm = 0.0
        self.head_contacts = 0
        self.head_lifted_mm = 0.0
        self.box_shift_mm = 0.0
        self.box_shift_final_mm = 0.0
        self.box_yaw_deg = 0.0
        self.hold_depth_mm = 0.0
        self.hold_force_n = 0.0
        self.hold_force_feed_n = 0.0
        self.pinch_q_mm = 0.0
        self.box_shift_pinch_mm = 0.0
        self.jaw_frees = 0
        self.jaw_stuck_mm = 0.0
        self.open_waits: list[int] = []
        self.stalls = 0
        self.shoves = 0
        self._box_rest: np.ndarray | None = None
        self._watching_box = False
        self._quat_cache: dict[str, np.ndarray] = {}
        self._quat_report: dict[str, tuple[float, float, float]] = {}
        self._box = self.env.handles.entity("box")
        self._tip0 = self._tip_y()
        self._t = 0
        self._resolve_contact()
        self._steps = self._choreography()

    # --------------------------------------------------------------- narration
    def _say(self, msg: str) -> None:
        """One line of progress, stamped with the control step. See :data:`VERBOSE`."""
        if VERBOSE:
            print(f"[tunnel {self._t:5d}] {msg}", flush=True)

    def _in_box(self, point) -> np.ndarray:
        """A world point in the box's LIVE frame [m]: origin on the bore axis at
        floor level mid-run, y along the run. The frame the document's regions
        are written in, and the one a shoved or yawed box is still honest in."""
        q = self.env.state.body_q.numpy()[int(self._box.body)]
        return quat_to_mat(np.asarray(q[3:7], dtype=np.float64)).T @ (np.asarray(point, dtype=np.float64) - q[:3])

    def _from_box(self, point) -> np.ndarray:
        """A point in the box's live frame, in world [m]. The inverse of :meth:`_in_box`."""
        q = self.env.state.body_q.numpy()[int(self._box.body)]
        return quat_to_mat(np.asarray(q[3:7], dtype=np.float64)) @ np.asarray(point, dtype=np.float64) + q[:3]

    def _fmt(self, point) -> str:
        p = self._in_box(point) * 1e3
        return f"({p[0]:+.1f}, {p[1]:+.1f}, {p[2]:+.1f})"

    def _pad_faces(self, name: str) -> list[tuple[np.ndarray, np.ndarray]]:
        """Each finger's collider bounding box in world [m], as ``(min, max)``.

        The pads' rubber is a bounding box of convex pieces on the finger body,
        and where those pieces stand against the capsule the hand closed on is
        the one reading that says "the cable is between the pads" rather than
        "the fingers reached their command"; the latter is what the take's
        ``pinch_q_mm 13.95`` said, and it was true of a pinch that had nothing.
        """
        model = self.env.scene.index_model
        body_q = self.env.state.body_q.numpy()
        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        shape_scale = model.shape_scale.numpy()
        shape_xform = model.shape_transform.numpy()
        out = []
        for side in self.env.handles.arms[name].finger_groups[:2]:
            fingers = {int(b) for b, _ in side}
            lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
            for shape in range(model.shape_count):
                if int(shape_body[shape]) not in fingers:
                    continue
                geometry = shape_geometry(model.shape_source[shape], int(shape_type[shape]), shape_scale[shape])
                if geometry is None:
                    continue
                world = transform_matrix(body_q[int(shape_body[shape])]) @ transform_matrix(shape_xform[shape])
                points = geometry[0] @ world[:3, :3].T + world[:3, 3]
                lo, hi = np.minimum(lo, points.min(axis=0)), np.maximum(hi, points.max(axis=0))
            out.append((lo, hi))
        return out

    def _report_pinch(self, name: str, rod, label: str) -> None:
        """Where the closed pads actually stand against the capsule they were sent to."""
        if not VERBOSE:
            return
        site, _ = self.env.arm(name).ee_pose(self.env.state)
        q = self._finger_q(name) * 1e3
        rod = np.asarray(rod, dtype=np.float64)
        parts = [
            f"{label}: site-rod dx {(site[0] - rod[0]) * 1e3:+.1f} dy {(site[1] - rod[1]) * 1e3:+.1f} "
            f"dz {(site[2] - rod[2]) * 1e3:+.1f} mm, fingers q {q[0]:.1f}/{q[1]:.1f} mm"
        ]
        for k, (lo, hi) in enumerate(self._pad_faces(name)):
            inner = lo[0] if lo[0] > rod[0] else hi[0]
            parts.append(
                f"pad{k} inner face x {(inner - rod[0]) * 1e3:+.1f} mm off the capsule, "
                f"z [{(lo[2] - rod[2]) * 1e3:+.1f}, {(hi[2] - rod[2]) * 1e3:+.1f}] about its centre, "
                f"y [{(lo[1] - rod[1]) * 1e3:+.1f}, {(hi[1] - rod[1]) * 1e3:+.1f}]"
            )
        parts.append(f"finger-vs-rod contact rows {self._contact_rows(name)}")
        self._say("; ".join(parts))
        # The finger links' colliders in the box's frame, in 10 mm bands of
        # height: how far along the run each finger reaches, band by band, so
        # a blade leaning on the box's end face can be seen for what it is.
        model = self.env.scene.index_model
        body_q = self.env.state.body_q.numpy()
        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        shape_scale = model.shape_scale.numpy()
        shape_xform = model.shape_transform.numpy()
        for k, side in enumerate(self.env.handles.arms[name].finger_groups[:2]):
            fingers = {int(b) for b, _ in side}
            pts = []
            for shape in range(model.shape_count):
                if int(shape_body[shape]) not in fingers:
                    continue
                geometry = shape_geometry(model.shape_source[shape], int(shape_type[shape]), shape_scale[shape])
                if geometry is None:
                    continue
                world = transform_matrix(body_q[int(shape_body[shape])]) @ transform_matrix(shape_xform[shape])
                pts.append(geometry[0] @ world[:3, :3].T + world[:3, 3])
            if not pts:
                continue
            local = np.asarray([self._in_box(p) for p in np.concatenate(pts)]) * 1e3
            bands = []
            for z0 in range(
                int(np.floor(local[:, 2].min() / 10.0)) * 10, int(np.ceil(local[:, 2].max() / 10.0)) * 10, 10
            ):
                sel = (local[:, 2] >= z0) & (local[:, 2] < z0 + 10)
                if sel.any():
                    bands.append(
                        f"z[{z0:+d},{z0 + 10:+d}) x[{local[sel, 0].min():+.0f},{local[sel, 0].max():+.0f}] y[{local[sel, 1].min():+.0f},{local[sel, 1].max():+.0f}]"
                    )
            self._say(f"  finger {k} colliders, box frame [mm]: " + " | ".join(bands))

    def _q(self, name: str) -> np.ndarray:
        ik = self.env.arm(name)
        return np.asarray(self.env.state.joint_q.numpy(), dtype=np.float64)[ik.arm_coords]

    def _open(self, name: str) -> float:
        """That hand's full opening [m of per-finger travel].

        Read per arm rather than off the work arm: both YAM hands are the same
        gripper here, but sizing one hand's pads from the other's handle is the
        class of copy ``cable_route`` keeps finding.
        """
        return float(self.env.handles.arms[name].gripper_open)

    def _guide_gap(self) -> float:
        """The rod's own diameter: pads on it, free to slide."""
        return 2.0 * float(self.env.task.deformables[0].radius)

    def _stop_pressing(self, name: str) -> None:
        """Stop driving into whatever the hand has run out of travel against.

        **Breaking out of a leg is not the same as stopping.** The command that
        stands is the last waypoint the leg streamed, and a position servo keeps
        pressing toward it for as long as it stands -- so a push abandoned at a
        jam goes on loading the cable, and through it the box: measured, a
        stroke that stopped itself at 1.5 mm of box travel had let the box go a
        further 15 mm by the time the jaws opened, sixty steps later.

        Setting the command to where the arm actually *is* takes the error to
        zero without moving anything. Only the streamed command is rewritten;
        the planning seed stays the last solved vector, which is what the next
        leg needs -- see :meth:`_leg` on why a leg planned from a sagged arm
        refuses itself.
        """
        self.held[name] = self._q(name)

    def _finger_for(self, face: float) -> float:
        """The finger command [m] that puts each pad FACE at ``face`` from the
        grasp axis.

        The one crossing from "where I want the rubber" to "what the channel
        takes", and the reason it exists is :data:`PAD_INBOARD`.
        """
        return max(0.0, float(face) + PAD_INBOARD)

    def _guide(self) -> float:
        """The command that makes a hand a guide ring rather than a grip [m].

        The rod's own radius, with none of :data:`PINCH_ENGAGE`'s interference:
        the pads sit at the cable's diameter, so it is held on its axis and still
        free to slide through. Not used by the choreography -- the support hand
        holds the box -- and kept because the medium tier states the pinch and
        the guide against the built rod's gauge.
        """
        return self._finger_for(0.5 * self._guide_gap())

    def _shut(self) -> float:
        """The close command [m]: the rod's own radius, less the engagement.

        Asked of the task's rod rather than written down, so a variant that
        changes the gauge does not silently keep a pinch calibrated for the old
        one.
        """
        return self._finger_for(0.5 * self._guide_gap() - PINCH_ENGAGE)

    def row_extras(self) -> dict[str, float]:
        return {
            "strokes": float(self.strokes_done),
            "legs_refused": float(len(self.refused)),
            "advance_mm": float(self.advance * 1e3),
            "grasp_contacts": float(self.grasp_contacts),
            "jaw_across": float(self._quat_report.get(self.work, (0.0, 0.0, 0.0))[0]),
            "jaw_along": float(self._quat_report.get(self.work, (0.0, 0.0, 0.0))[1]),
            "tip_y": float(self._tip_y()),
            "head_y": float(self._plug_xyz()[1]),
            "head_pulled_mm": float(self.head_pulled_mm),
            "head_contacts": float(self.head_contacts),
            "head_lifted_mm": float(self.head_lifted_mm),
            "box_shift_mm": float(self.box_shift_mm),
            "box_shift_final_mm": float(self.box_shift_final_mm),
            "box_yaw_deg": float(self.box_yaw_deg),
            "hold_depth_mm": float(self.hold_depth_mm),
            "hold_force_n": float(self.hold_force_n),
            "hold_force_feed_n": float(self.hold_force_feed_n),
            "pinch_q_mm": float(self.pinch_q_mm),
            "box_shift_pinch_mm": float(self.box_shift_pinch_mm),
            "jaw_frees": float(self.jaw_frees),
            "jaw_stuck_mm": float(self.jaw_stuck_mm),
            "open_wait_max": float(max(self.open_waits) if self.open_waits else 0.0),
            "stalls": float(self.stalls),
            "shoves": float(self.shoves),
            "held": float(self.hold),
        }

    # ------------------------------------------------------------------ drive
    def act(self) -> np.ndarray:
        try:
            next(self._steps)
        except StopIteration:
            pass
        self._record_box()
        self._t += 1
        return self._action()

    def _grip_command(self, name: str) -> float:
        """That arm's stored width [m of travel] as the action's 0-to-1 command.

        The single crossing between the two, so a width can never be emitted raw.
        ``cable_route/routine.py:487`` does the same arithmetic inline; it is a
        method here because two suites got the direction wrong reading it off a
        jaw-body separation. See :data:`GRIPPER_CHANNEL_IS_NORMALIZED`.
        """
        travel = float(self.env.handles.arms[name].gripper_open)
        if travel <= 0.0:
            return 0.0
        return float(np.clip(1.0 - float(self.grip[name]) / travel, 0.0, 1.0))

    def _action(self) -> np.ndarray:
        a = np.asarray(self.env.idle_action(self.MODE), dtype=np.float64)
        for name in arm_slots(self.env):
            s = arm_slot(self.env, name) * self.dof
            a[s : s + self.dof - 1] = self.held[name][: self.dof - 1]
            a[s + self.dof - 1] = self._grip_command(name)
        return a

    # ------------------------------------------------------------------- legs
    def _hold(self, n: int):
        for _ in range(n):
            yield

    def _leg(self, name, target, quat, *, speed=0.06, grip=None, label="", gate=True):
        """Solve the whole leg, gate it, then drive it one waypoint per step.

        ``box_decant``'s leg, and its reasoning: a single IK solve held for N
        control steps is one joint target the servo slews to at full rate, which
        flings whatever is in the hand. ``gate`` is off for a free approach,
        where the residual is large at the first waypoint and shrinks as the arm
        converges, and on for anything holding the rod.

        **The leg starts from the command that stands, not from where the arm
        measures.** Read off the live state instead -- which is the obvious
        spelling, and what this did first -- every one of the feed strokes was
        refused and the cable moved 0.0 mm across a whole episode. The cause is
        not a reach problem: a hand that has closed on a rod is sitting ~14 mm
        below its own commanded pose, carried there on the position servo's
        compliance by fingertip-first contact, while IK solves the same target
        to zero. Planning the next leg from that sag makes its first waypoint a
        jump the gate correctly refuses, and refusing every stroke reads exactly
        like a task that cannot be done. ``cable_route`` seeds its pay-outs from
        the last solved vector for the same reason.
        """
        ik = self.env.arm(name)
        target = np.asarray(target, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        # Planned on the host chain walk. A leg is a few hundred waypoints and
        # each one is an IK solve, which on the device is a full-model eval_fk
        # and a body_q readback apiece; the same scope took cable_route's station
        # planner from minutes to seconds.
        with ik.host_fk():
            start, _ = ik.fk_ee(np.asarray(self.seed[name], dtype=np.float32))
            steps = max(2, int(np.linalg.norm(target - start) / speed * 60.0))
            solutions, worst_p, worst_r, seed = [], 0.0, 0.0, self.seed[name]
            for k in range(steps):
                waypoint = start + (target - start) * (k + 1) / steps
                solution = ik.solve_q_full(seed, waypoint, quat)
                p, r = ik.pose_errors(solution, waypoint, quat)
                worst_p, worst_r = max(worst_p, p), max(worst_r, r)
                solutions.append(solution)
                seed = solution
        if gate and (worst_p > MAX_POS or worst_r > MAX_ROT):
            self.refused.append(f"{label or name}: {worst_p * 1e3:.1f} mm / {math.degrees(worst_r):.1f} deg")
            return
        if grip is not None:
            self.grip[name] = grip
        for solution in solutions:
            self.seed[name] = solution
            self.held[name] = np.asarray(solution)[ik.arm_coords]
            yield

    # -------------------------------------------------------------- measuring
    def _grasp_quat(self, name: str, at=None) -> np.ndarray:
        """A grasp orientation whose **achieved** pose puts the pads across the rod.

        Chosen by measurement at build, not by a constant, and that is a
        correction rather than a refinement. Every fixed answer tried here was
        wrong in a way nothing raised: the yaw the API's wording implies, the yaw
        a rod-displacement probe rewards (which cannot tell a grasp from a
        shove), and the closing axis ``cable_route`` grips with. Each was right
        for one pose and wrong at the pose the routine's own leg chain reaches,
        because the chain picks the wrist basin and the basin picks the roll.

        So the routine asks the arm. For each candidate it solves the grasp pose,
        runs the arm's own FK, and reads where the jaws close in the pose the arm
        actually reached: the hand's declared closing axis (its TCP's y, measured
        off the pads in the gripper module) carried into the world, wanted along
        **x**, across a rod that runs in y, and not along it. The residual is
        scored alongside, because an orientation the wrist cannot hold arrives
        somewhere else entirely. It used to read the line between the two jaw
        groups' body origins, which on ``yam``'s arena hand do not move with the
        pads (they stand 83 mm apart open and 108 shut).

        Of the holds that score within :data:`WRIST_AWAY` of the best, the one
        whose wrist stands away from the box's middle is taken.

        The same orientation serves the support hand: it pinches the lid across
        x, exactly as the work hand pinches the rod.

        Cached per arm: it is a property of the hand and the place, and solving
        it costs a few dozen IK calls.
        """
        cached = self._quat_cache.get(name)
        if cached is not None:
            return cached
        env = self.env
        ik = env.arm(name)
        hand = env.scene.embodiment.hands.get(name) or work_gripper(env)
        target = np.asarray(at if at is not None else self._grasp_target(name), dtype=np.float64)
        rot_site = quat_to_mat(np.asarray(hand.tcp_offset_rot, dtype=np.float64))
        close_site = rot_site.T @ np.asarray(hand.close_axis, dtype=np.float64)

        # Which way the wrist should stand off the grip point: away from the
        # box's middle, along the run. See WRIST_AWAY.
        _, y_mid, _ = self._layout()
        away = np.array([0.0, float(np.sign(target[1] - y_mid) or 1.0), 0.0])
        wrist_local = self._wrist_from_tcp(name)

        candidates = []
        with ik.host_fk():
            for yaw in range(0, 360, 15):
                quat = np.asarray(top_down_grasp_site_quat(hand, float(yaw)), dtype=np.float64)
                solution = ik.solve_q_full(self.home_seed, target, quat)
                _, rot = ik.pose_errors(solution, target, quat)
                _, reached = ik.fk_ee(np.asarray(solution, dtype=np.float32))
                line = quat_to_mat(np.asarray(reached, dtype=np.float64)) @ close_site
                norm = float(np.linalg.norm(line))
                if norm < 1e-6:
                    continue
                # Across the run and level: a pair that separates vertically
                # closes above and below a rod rather than onto it.
                across = abs(float(line[0])) / norm
                along = abs(float(line[1])) / norm
                score = across - along - 0.5 * float(rot)
                side = float(np.dot(quat_to_mat(quat) @ wrist_local, away))
                candidates.append((score, side, (quat, across, along, float(rot), float(yaw))))
        top = max(c[0] for c in candidates)
        _, _, best = max((c for c in candidates if c[0] >= top - WRIST_AWAY), key=lambda c: (c[1], c[0]))
        quat, across, along, rot, yaw = best
        self._quat_report[name] = (across, along, math.degrees(rot), yaw)
        self._quat_cache[name] = quat
        return quat

    def _wrist_from_tcp(self, name: str) -> np.ndarray:
        """The wrist body's origin from the TCP, in the TCP's own frame [m]."""
        hand = self.env.scene.embodiment.hands.get(name) or work_gripper(self.env)
        offset = np.asarray(hand.tcp_offset, dtype=np.float64)
        return quat_to_mat(np.asarray(hand.tcp_offset_rot, dtype=np.float64)).T @ -offset

    def _layout(self):
        """The layout's own constants, through the package the routine lives in."""
        from .tunnel.layout import BORE_FLOOR_Z, TUNNEL_X, TUNNEL_Y

        return TUNNEL_X, TUNNEL_Y, BORE_FLOOR_Z

    def _grasp_target(self, name: str):
        """Where this hand takes hold, for choosing its orientation.

        The support hand's is the lid pinch; the work hand's is the first
        stroke's pinch at the bore axis' height. A centimetre of height does
        not move the wrist's roll, which is all this target is for."""
        x, y_mid, floor = self._layout()
        surface = float(self.env.handles.surface_z)
        if name == self.support:
            return [x, y_mid + RUN / 2 - HOLD_FROM_EXIT, surface + floor + (BORE + ROOF) / 2]
        entry = y_mid - RUN / 2
        return [x, entry - PINCH_OUT, surface + floor + RADIUS - 0.06]

    def _pad_drop(self, name: str) -> float:
        """How far the pad tips reach below the grasp site, along the approach [m].

        ``cable_route``'s ``pad_tip_drop``, restated -- two tasks may not reach
        each other -- and it is restated rather than skipped because it is what
        stands between a pinch and a hand driven into the ledge. The grasp site
        rides at the pad nub field and the tips ride this far under it, so a
        site commanded at the rod's own centre puts the tips half a diameter
        *below* whatever the rod lies on. About 10.7 mm on the i2rt hand.

        Measured off that hand's own finger colliders at whatever pose the arm
        holds right now: the fingers of a linear jaw translate across the
        approach axis, so the projection onto "down, in the grasp orientation" is
        invariant to both pose and opening.
        """
        env = self.env
        ik = env.arm(name)
        site_pos, site_quat = ik.ee_pose(env.state)
        down_in_site = quat_to_mat(self._grasp_quat(name)).T @ np.array([0.0, 0.0, -1.0])
        live_rot = quat_to_mat(np.asarray(site_quat, dtype=np.float64))

        # The model indices are resolved against, which replicated is the
        # TEMPLATE: ``body_q`` below is this world's slice, indexed like the
        # template, so the shape table walked here has to be the same one. Off
        # the stepping model this walks every world's shapes to use one world's.
        model = env.scene.index_model
        body_q = env.state.body_q.numpy()
        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        shape_scale = model.shape_scale.numpy()
        shape_xform = model.shape_transform.numpy()
        fingers = {int(b) for side in env.handles.arms[name].finger_groups for b, _ in side}

        drop = 0.0
        for shape in range(model.shape_count):
            if int(shape_body[shape]) not in fingers:
                continue
            geometry = shape_geometry(model.shape_source[shape], int(shape_type[shape]), shape_scale[shape])
            if geometry is None:
                continue
            vertices, _ = geometry
            world = transform_matrix(body_q[int(shape_body[shape])]) @ transform_matrix(shape_xform[shape])
            points = vertices @ world[:3, :3].T + world[:3, 3]
            drop = max(drop, float(((points - site_pos) @ live_rot @ down_in_site).max()))
        return drop

    def _tip_y(self) -> float:
        """Where the fed end of the rod is right now [m], along the run.

        The last capsule's own body origin, which sits half a pitch behind the
        rod's geometric end -- 5.8 mm on this rod. The same place ``cable.end_b``
        reads, so an advance measured here and a goal read there are measuring one
        thing.
        """
        bodies = self.env.handles.entity("cable").bodies
        return float(self.env.state.body_q.numpy()[bodies[-1]][1])

    def _rod_at(self, y: float) -> np.ndarray:
        """Where the cable actually is at ``y`` along the run [m, world].

        The capsule whose centre is nearest ``y``, read live. **This is what a
        feed at height needs that a feed on a bench did not**: the rod behind
        the entry hangs off the box's end and drapes to the mat, so where the
        hand should close is a fact about the cable's current shape, not a
        constant -- measured, 90 mm out it lies about 30 mm under the bore axis
        before the first stroke, and less as the tail is used up.
        """
        bodies = list(self.env.handles.entity("cable").bodies)
        pose = self.env.state.body_q.numpy()[bodies][:, :3]
        return np.asarray(pose[int(np.argmin(np.abs(pose[:, 1] - float(y))))], dtype=np.float64)

    def _pinch_capsule(self, entry_y: float, bench_z: float) -> int:
        """The capsule body the feeding hand pinches this stroke.

        The one furthest from the mouth inside the window :data:`PINCH_OUT_MIN`
        to :data:`PINCH_OUT` behind the entry whose pinch keeps the pad tips
        :data:`TIP_CLEAR` off the bench -- so on the hang, never on the table.
        The first stroke finds it at :data:`PINCH_OUT`, as it always did; once
        the feed has drawn the slack in and the hang has become a riser beside
        the box, it finds it on the riser."""
        bodies = list(self.env.handles.entity("cable").bodies)
        pose = np.asarray(self.env.state.body_q.numpy()[bodies][:, :3], dtype=np.float64)
        lo, hi = entry_y - PINCH_OUT, entry_y - PINCH_OUT_MIN
        ok = (pose[:, 1] >= lo) & (pose[:, 1] <= hi) & (pose[:, 2] - PINCH_UNDER >= bench_z + TIP_CLEAR)
        if not ok.any():
            return int(bodies[int(np.argmin(np.abs(pose[:, 1] - (entry_y - PINCH_OUT))))])
        rows = np.flatnonzero(ok)
        return int(bodies[int(rows[int(np.argmin(pose[rows, 1]))])])

    def _settled_offset(self, name: str, commanded) -> np.ndarray:
        """How far the hand actually sits from where it was sent [m, world xy].

        A position servo at the far end of its reach settles a few millimetres
        off the pose IK solved to zero -- 3 mm in x at the lid pinch, measured
        -- and a pinch with 9 mm of clearance a side does not have 3 mm to
        give twice. Read at the approach point, where nothing is in contact, so
        the offset is the servo's and not a collision's; the descent below is
        sent to the commanded point less this, and the *plan* still starts from
        the commanded seed (see :meth:`_leg`).
        """
        site, _ = self.env.arm(name).ee_pose(self.env.state)
        offset = np.asarray(site[:2], dtype=np.float64) - np.asarray(commanded, dtype=np.float64)[:2]
        return np.clip(offset, -0.008, 0.008)

    def _ease_open(self, name: str, shut: float, wide: float):
        """Let the grip out over :data:`EASE_OPEN` steps instead of releasing it.

        See :data:`EASE_OPEN` for the measurement. The width is what this module
        speaks in; :meth:`_grip_command` is still the only place it becomes a
        command.
        """
        for step in range(EASE_OPEN):
            self.grip[name] = shut + (wide - shut) * (step + 1) / EASE_OPEN
            yield
        self.grip[name] = wide

    def _await_open(self, name: str):
        """Hold still until this hand's fingers have actually opened.

        The grip command is already open when this is called; what it waits for
        is the *state*. Reads the finger coordinates rather than counting steps,
        because how long a pad takes to come off a cable is a fact about the
        load on it. Gives up after :data:`OPEN_WAIT` and lets the caller's
        :meth:`_free_jaws` deal with what is left.
        """
        open_q = self._finger_open_q(name)
        for step in range(OPEN_WAIT):
            q = self._finger_q(name)
            if q.size and float(np.min(q / np.maximum(open_q[: q.size], 1e-9))) >= OPEN_ENOUGH:
                self.open_waits.append(step)
                return
            yield
        self.open_waits.append(OPEN_WAIT)

    def _finger_q(self, name: str) -> np.ndarray:
        """Each finger's live coordinate [m], 0 shut, one per jaw.

        A hand whose second jaw is carried by an equality (``yam_bimanual``'s
        arena hand drives one finger joint) reports its one driven coordinate
        for both jaws: the equality is what the other one reads."""
        coords = list(self.env.handles.arms[name].gripper_coords)
        q = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float64)[coords]
        return np.repeat(q, 2) if q.size == 1 else q

    def _finger_open_q(self, name: str) -> np.ndarray:
        """Each jaw's fully open coordinate [m], one per jaw as :meth:`_finger_q` reports them."""
        q = np.asarray(self.env.handles.arms[name].gripper_open_q, dtype=np.float64)
        return np.repeat(q, 2) if q.size == 1 else q

    def _free_jaws(self, name: str, at, quat):
        """After an open command, get the pad that is still on the cable off it.

        **The finding this exists for**: commanded open with the cable pinched
        on the drape, one finger opens at once and the other creeps at 0.4 mm a
        step -- the cable, pulled sideways by its own hang, presses on that pad,
        and pad-on-cable friction is the rod's ``mu_gripper`` of 60, so the
        finger cannot slide off it. Lifted like that the hand takes the cable
        with it, feeds it by levering it over the lip, and shoves the box 30 mm
        through the bore (measured, ``box_shift_mm`` in lockstep with the lift).
        So the hand is moved a centimetre *toward the stuck finger's side* --
        away from the cable, which stays -- and the finger is free to open.
        Nothing to do when both fingers report open.
        """
        open_q = self._finger_open_q(name)
        q = self._finger_q(name)
        stuck = [i for i in range(min(2, len(q))) if q[i] < 0.85 * open_q[i]]
        if not stuck:
            return
        self.jaw_stuck_mm = max(self.jaw_stuck_mm, float(open_q[stuck[0]] - q[stuck[0]]) * 1e3)
        site, _ = self.env.arm(name).ee_pose(self.env.state)
        body_q = self.env.state.body_q.numpy()
        group = self.env.handles.arms[name].finger_groups[stuck[0]]
        finger_x = float(np.mean([body_q[int(b)][0] for b, _ in group]))
        side = 1.0 if finger_x >= float(site[0]) else -1.0
        self.jaw_frees += 1
        target = [float(at[0]) + side * RELEASE_ACROSS, float(at[1]), float(at[2])]
        yield from self._leg(name, target, quat, gate=False, speed=0.04, label="free jaws")
        yield from self._hold(15)

    def _box_origin(self) -> np.ndarray:
        """The box's origin -- the bore axis at floor level, mid-run -- right now [m, world]."""
        return np.asarray(self.env.state.body_q.numpy()[int(self._box.body)][:3], dtype=np.float64)

    def _box_yaw(self) -> float:
        q = self.env.state.body_q.numpy()[int(self._box.body)][3:7]
        x, y, z, w = (float(v) for v in q)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _record_box(self) -> None:
        """How far the box has moved off its rest pose, in the plane [mm].

        ``box_shift_mm`` is the worst excursion *during the feed* -- from the
        pose the box had when the support hand's pinch closed on it, or from
        where it settled when nothing holds it -- and it is the number the hold
        is justified by from both sides: a held box that moves has a hold that
        is not holding, an unheld box that does not move has a coupling that
        is not two-way. ``box_shift_pinch_mm`` is what the pinch itself did to
        the box as it closed: the pads centre the lid between them, and a hand
        that settled a few millimetres off the bore axis drags the box that
        far as the first pad meets the lid before the second (measured, 5.7 mm
        and a degree of yaw; the descent's offset correction takes the
        approach's settle out, not the pinch's). ``box_shift_final_mm`` is
        where it ended up, off the same rest.
        """
        if self._box_rest is None:
            return
        shift = float(np.linalg.norm(self._box_origin()[:2] - self._box_rest[:2])) * 1e3
        self.box_shift_final_mm = shift
        self.box_yaw_deg = math.degrees(self._box_yaw() - self._rest_yaw)
        if self._watching_box:
            self.box_shift_mm = max(self.box_shift_mm, shift)

    def _plug_xyz(self) -> np.ndarray:
        """Where the connector's overmold is right now [m].

        Read off the built model -- the overmold's own shape transform on
        whichever body carries it, composed with that body's live pose -- rather
        than recomputed from the rod's tip and a constant. The head is rigid on
        the rod's last capsule, so the two agree; reading the shape is what keeps
        them agreeing if the head is ever re-authored, and it is the pose a hand
        is actually sent to.
        """
        # The template, for :meth:`_pad_drop`'s reason: a replicated model
        # carries one ``_overmold`` label per world and the body index taken off
        # it is applied to this world's slice of ``body_q``.
        model = self.env.scene.index_model
        if self._plug is None:
            labels = [str(x).split("/")[-1] for x in model.shape_label]
            self._plug = next(i for i, lab in enumerate(labels) if lab.endswith("_overmold"))
        shape = self._plug
        body = int(model.shape_body.numpy()[shape])
        world = transform_matrix(self.env.state.body_q.numpy()[body]) @ transform_matrix(
            model.shape_transform.numpy()[shape]
        )
        return np.asarray(world[:3, 3], dtype=np.float64)

    def _resolve_contact(self) -> None:
        """Shapes to count a hand-versus-cable contact over, resolved once.

        **Per arm, because two different hands take two different holds here**:
        the work hand grips the bare cable to feed it, and the support hand grips
        the connector to draw it through. ``_head`` is the plug's own shapes
        rather than the whole rod, so "did the hand get the connector" is
        answerable separately from "did it get the cable somewhere".
        """
        # The template again: the contact view these sets are asked of rebases
        # its shapes to it, so a set naming every world's rod is the wrong
        # question asked of the right table.
        model = self.env.scene.index_model
        labels = [str(label).split("/")[-1] for label in model.shape_label]
        shape_body = model.shape_body.numpy()
        self._pads = {}
        for name in (self.work, self.support):
            if not name:
                continue
            fingers = {int(b) for side in self.env.handles.arms[name].finger_groups for b, _ in side}
            self._pads[name] = {i for i in range(model.shape_count) if int(shape_body[i]) in fingers}
        self._rod = {i for i, label in enumerate(labels) if label.startswith("cable")}
        # The box is a free body in the ARMS' island and so is the hand holding
        # it, so this pair needs no proxy: ``env.contacts`` sees it, the way
        # ``routine_decant.py`` reads its blade against the flaps.
        self._box_shapes = [i for i in range(model.shape_count) if int(shape_body[i]) == int(self._box.body)]
        self._head = {i for i, label in enumerate(labels) if label.endswith(("_overmold", "_nose", "_cap"))}

    def _record_pinch(self) -> None:
        """Where the feeding fingers stalled when they closed on the cable [mm].

        Telemetry, not a gate, and worth reading with its limits: a pinch with
        the pads' hooks under the cable and their faces on its flanks stalls
        the fingers at about 7 mm (the proxy contact is soft and the faces sink
        a couple of millimetres into the rod), and the pinch that squeezed the
        cable out from under the pads stalled at 6.6 -- the number does not
        tell the two apart. What does is behavioural and is measured beside
        it: the tip's advance per stroke, and a box that stays put through the
        release (``box_shift_mm``). The smallest over the episode is kept.
        """
        q = float(np.mean(self._finger_q(self.work))) * 1e3
        self.pinch_q_mm = q if self.pinch_q_mm <= 0.0 else min(self.pinch_q_mm, q)

    def _hold_force(self) -> float:
        """What the support hand is actually pressing the box with [N].

        **The measurement that tells a weak hold from a missed one**, and the
        one number this routine was reasoning about without ever reading. The
        contacts it counts elsewhere are finger-versus-CABLE rows off the proxy
        pipeline and say nothing about the box; this pair is MuJoCo's own --
        both bodies are in the arms' island -- so the probe reads it directly.
        A hold in the tens of newtons that still lets the box slide is a
        friction problem; a hold near zero is a hand closing beside the lid,
        which is a different fix entirely.

        The pads alone against the box alone: ``pairwise_force`` sums impulse
        vectors, so folding in the arm's other links brings back the
        cancellation that makes a balanced pair read zero.
        """
        probe = self.env.contacts
        if probe is None or not self.support:
            return 0.0
        try:
            probe.update(step=None)
            pads = sorted(self._pads.get(self.support, ()))
            if not pads or not self._box_shapes:
                return 0.0
            return float(np.linalg.norm(probe.pairwise_force(self._box_shapes, pads)))
        except Exception:  # telemetry must never take an episode down
            return 0.0

    def _work_box_force(self) -> float:
        """What the FEEDING hand is pressing the box with [N], if anything.

        The number that separates a box moved by the cable from a box moved by
        the hand pushing it: the fingers of a hand driven to the flare's
        clearance are long, and a blade leaning on the box's end face is a
        shove no cable telemetry would show."""
        probe = self.env.contacts
        if probe is None:
            return 0.0
        try:
            probe.update(step=None)
            pads = sorted(self._pads.get(self.work, ()))
            if not pads or not self._box_shapes:
                return 0.0
            return float(np.linalg.norm(probe.pairwise_force(self._box_shapes, pads)))
        except Exception:  # telemetry must never take an episode down
            return 0.0

    def _record_hold(self) -> None:
        """Whether the support hand's pinch went down AROUND the lid.

        ``hold_depth_mm`` is how far the pad tips reach below the lid's top when
        the pinch has closed -- positive is a hand whose pads are beside the
        strip, and about zero is a hand sitting on it with its jaws shut on
        air, which is what the first held run did and what no contact count
        would have said. Read off the site and the measured pad drop.
        """
        if not self.support:
            return
        site, _ = self.env.arm(self.support).ee_pose(self.env.state)
        lid_top = float(self._box_origin()[2]) + BORE + ROOF
        self.hold_depth_mm = (lid_top - (float(site[2]) - self._pad_drop(self.support))) * 1e3
        self.hold_force_n = max(self.hold_force_n, self._hold_force())

    def _record_contact(self, name: str = "", *, head: bool = False) -> None:
        """How many finger-versus-cable contacts the closed hand is making.

        **The check that the hand actually has the cable**, and it has to be read
        off the *proxy* pipeline: ``env.contacts`` is the MuJoCo island's probe
        and the rod is solved in the VBD one, so it reports zero for this pair
        however hard the hand is squeezing. Measured, ``cable_route/port_easy`` --
        the same rod and hand, gripping successfully -- carries 38 rows at its
        own grasp, so tens of rows is a grasp and zero is a hand closing beside
        the cable.

        Kept per episode rather than gating anything: a routine that refused to
        continue on a bad grasp would hide the failure this number exists to
        show.
        """
        rows = self._contact_rows(name or self.work, head=head)
        if head:
            self.head_contacts = max(self.head_contacts, rows)
        else:
            self.grasp_contacts = max(self.grasp_contacts, rows)

    def _contact_rows(self, name: str, *, head: bool = False) -> int:
        """The finger-versus-cable (or -connector) rows in the proxy pipeline, now."""
        pads = self._pads.get(name)
        if not pads:
            return 0
        want = self._head if head else self._rod
        solver = self.env.scene.pipeline.solver
        getter = getattr(solver, "get_proxy_contacts", None)
        if getter is None:
            return 0
        try:
            contacts = getter("mujoco", "vbd")
            if contacts is None:
                return 0
            count = int(np.asarray(contacts.rigid_contact_count.numpy()).reshape(-1)[0])
            if not count:
                return 0
            a = np.asarray(contacts.rigid_contact_shape0.numpy())[:count]
            b = np.asarray(contacts.rigid_contact_shape1.numpy())[:count]
            return sum(
                1
                for i in range(count)
                if (int(a[i]) in pads and int(b[i]) in want) or (int(b[i]) in pads and int(a[i]) in want)
            )
        except Exception:  # telemetry must never take an episode down
            return 0

    def _record_advance(self) -> None:
        """How far the tip has come since the episode began [m].

        Where the tip actually is, per stroke, which answers the question a
        contact count is a proxy for -- and ``env.contacts`` cannot answer it
        here anyway, being the MuJoCo island's probe while the rod is solved in
        the VBD one.
        """
        self.advance = self._tip_y() - self._tip0

    # ---------------------------------------------------------- choreography
    def _choreography(self):
        env = self.env
        x_bore, y_mid, _floor_spec = self._layout()
        entry_y, exit_y = y_mid - RUN / 2, y_mid + RUN / 2
        push_to = entry_y - FLARE - HAND_HALF_Y - PUSH_CLEAR
        push_from = entry_y - PINCH_OUT
        quat = self._grasp_quat(self.work)
        shut, open_ = self._shut(), self._open(self.work)

        yield from self._hold(SETTLE)

        # Read, not assumed: the box has settled its half-millimetre onto the
        # table and the cable's tail has fallen. Everything below is measured
        # from where the bore floor actually is.
        self._box_rest = self._box_origin().copy()
        self._rest_yaw = self._box_yaw()
        floor_z = float(self._box_rest[2])
        axis_z = floor_z + RADIUS
        # The pinch height off the cable's live axis rather than off a bench
        # under it: the tips PINCH_UNDER below the axis, so the site sits
        # (pad_drop - PINCH_UNDER) above it. See PINCH_UNDER for the pinch that
        # squeezed the cable out under the pads.
        pinch_above = self._pad_drop(self.work) - PINCH_UNDER
        carry_z = axis_z + CARRY_CLEAR
        self._watching_box = True
        rod0 = self._rod_at(push_from)
        self._say(
            f"settled: box origin {np.round(self._box_rest, 4).tolist()} yaw {math.degrees(self._rest_yaw):+.2f} deg; "
            f"tip {self._fmt(self.env.state.body_q.numpy()[self.env.handles.entity('cable').bodies[-1]][:3])} in the box "
            f"frame ({(self._in_box(self.env.state.body_q.numpy()[self.env.handles.entity('cable').bodies[-1]][:3])[1] + RUN / 2) * 1e3:.1f} mm "
            f"inside the entry); cable at the pinch y {push_from:+.3f}: {self._fmt(rod0)}, "
            f"{(axis_z - rod0[2]) * 1e3:.0f} mm under the axis; pad drop {self._pad_drop(self.work) * 1e3:.1f} mm, "
            f"shut {shut * 1e3:.1f} mm/finger, push_to y {push_to:+.3f}"
        )

        # --- the left hand takes the box by its lid, once, and keeps it -------
        if self.support and self.hold:
            hold_quat = self._grasp_quat(self.support)
            hold_y = exit_y - HOLD_FROM_EXIT
            hold_z = floor_z + (BORE + ROOF) / 2
            over_z = floor_z + BORE + ROOF + SUPPORT_OVER
            # Down with the jaws just wider than the lid, not open: open, the
            # back pad lands on the mat behind the slot. See LID_CLEAR.
            lid_half = BORE / 2 + WALL
            yield from self._leg(
                self.support,
                [x_bore, hold_y, over_z],
                hold_quat,
                grip=self._finger_for(lid_half + LID_CLEAR),
                gate=False,
                speed=0.20,
                label="steady over",
            )
            yield from self._hold(15)
            off = self._settled_offset(self.support, [x_bore, hold_y])
            yield from self._leg(
                self.support,
                [x_bore - off[0], hold_y - off[1], hold_z],
                hold_quat,
                gate=False,
                speed=0.04,
                label="steady down",
            )
            yield from self._hold(15)
            self.grip[self.support] = self._finger_for(lid_half - HOLD_ENGAGE)
            yield from self._hold(40)
            self._record_hold()
            # The feed is measured from where the pinch left the box, and the
            # pinch's own centring shove is recorded on its own.
            self.box_shift_pinch_mm = float(np.linalg.norm(self._box_origin()[:2] - self._box_rest[:2])) * 1e3
            self._box_rest = self._box_origin().copy()
            self._rest_yaw = self._box_yaw()
            self.box_shift_mm = 0.0
            lq = self._finger_q(self.support) * 1e3
            self._say(
                f"hold: lid pinched at box y {self._in_box(self.env.arm(self.support).ee_pose(self.env.state)[0])[1] * 1e3:+.0f} mm, "
                f"fingers q {lq[0]:.1f}/{lq[1]:.1f} mm (asked {self.grip[self.support] * 1e3:.1f}), depth {self.hold_depth_mm:.1f} mm, "
                f"force {self.hold_force_n:.2f} N, pinch shoved the box {self.box_shift_pinch_mm:.1f} mm"
            )

        # --- the right hand feeds the cable ---------------------------------
        for stroke in range(STROKES_MAX):
            # Until the tip is out, not for a count: read live, and the head is
            # on the ledge for the second hand once the tip is FEED_PAST_EXIT
            # past the lid's end.
            if self._tip_y() >= exit_y + FEED_PAST_EXIT:
                self._say(
                    f"feed done: tip y {self._tip_y():+.4f} is past the exit {exit_y:+.4f} + {FEED_PAST_EXIT * 1e3:.0f} mm"
                )
                break
            self._say(
                f"stroke {stroke}: tip {self._fmt(self.env.state.body_q.numpy()[self.env.handles.entity('cable').bodies[-1]][:3])} "
                f"box frame, box shift {self.box_shift_final_mm:.1f} mm yaw {self.box_yaw_deg:+.2f} deg"
            )
            # Back to the basin the grasp orientation was chosen in, so the
            # last stroke grips the way the first one was measured to.
            self.seed[self.work] = self.home_seed.copy()
            if LIVE_BOX:
                # The mouth, the entry and the flare clearance where the box
                # IS, not where the layout put it. See LIVE_BOX.
                lip = self._from_box([0.0, -RUN / 2, RADIUS])
                x_bore, entry_y = float(lip[0]), float(lip[1])
                axis_z = float(lip[2])
                push_to = entry_y - FLARE - HAND_HALF_Y - PUSH_CLEAR
                push_from = entry_y - PINCH_OUT
            # Where the cable rises, read live. See PINCH_OUT_MIN.
            pinched = self._pinch_capsule(entry_y, float(self._box_origin()[2]) - BOX_HEIGHT)
            push_from = float(self.env.state.body_q.numpy()[pinched][1])
            yield from self._leg(
                self.work,
                [x_bore, push_from, carry_z],
                quat,
                grip=open_,
                gate=False,
                speed=0.20,
                label=f"over {stroke}",
            )
            yield from self._hold(15)
            # Where the cable is, not where the axis is: it hangs. And where
            # the hand actually settled, not where it was sent.
            rod = np.asarray(self.env.state.body_q.numpy()[pinched][:3], dtype=np.float64)
            off = self._settled_offset(self.work, [x_bore, push_from])
            grasp_z = min(float(rod[2]) + pinch_above, axis_z + pinch_above)
            pinch = np.array([float(rod[0]) - off[0], float(rod[1]) - off[1], grasp_z])
            yield from self._leg(
                self.work,
                [float(v) for v in pinch],
                quat,
                gate=False,
                speed=0.06,
                label=f"down {stroke}",
            )
            yield from self._hold(PINCH_SETTLE)
            self.grip[self.work] = shut
            # Long enough for the jaws to travel and stall on the rod, and it is
            # checked rather than assumed, twice: `_record_contact` reads the
            # proxy pipeline, which is the only place a hand-versus-rod contact
            # is visible, and `_record_pinch` reads where the fingers stalled,
            # which is the only thing that says the pads are BESIDE the cable
            # rather than on top of it -- the contact rows are counted inside
            # the rod's 10 mm detection gap and cannot tell the two apart.
            yield from self._hold(60)
            self._record_contact()
            self._record_pinch()
            self._report_pinch(
                self.work, rod, f"stroke {stroke} pinch (sent to {self._fmt(pinch)}, cable {self._fmt(rod)})"
            )

            # The push goes ALONG the hanging cable toward the mouth, so the rod
            # ahead of the pinch stays a straight column in compression. Driven
            # level, the chord between the hand and the lip shortens and the
            # cable simply hangs deeper -- gravity, not the bore, takes the
            # stroke. And it ENDS where the finger carriage would otherwise
            # meet the flare (CARRIAGE_REACH): measured, that is a shorter push
            # than the cable would take, so the feed is more strokes rather
            # than a closer one. A straight lift and a level push were tried
            # instead and the pads carried nothing: this hand's pinch drags a
            # cable along its own line and cannot lift it off it.
            #
            # It aims at the push's end level with the bore, clear of the flare
            # (`push_to`), rather than at the mouth: the cable pinched low on
            # the riser is carried up it and stops on the bore's own line, so
            # the rod ahead of the pads ends straight rather than bent into the
            # mouth from below.
            aim = np.array([x_bore, push_to, axis_z + pinch_above])
            direction = aim - np.array([float(rod[0]), float(rod[1]), grasp_z])
            reach = min(STROKE, float(np.linalg.norm(direction)))
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            end = pinch + max(reach, 0.0) * direction
            # Driven waypoint by waypoint rather than streamed, so the push can
            # STOP when the cable stops taking it -- see STALL_ADVANCE. Breaking
            # out of the leg leaves the arm holding its last commanded pose,
            # which is exactly what a person does when a cable will not go in.
            history: list[float] = [self._tip_y()]
            box_at_push = self._box_origin()[:2].copy()
            grip_body = pinched
            cap0 = self.env.state.body_q.numpy()[grip_body][:3].copy()
            site0, _ = self.env.arm(self.work).ee_pose(self.env.state)
            site0 = np.asarray(site0, dtype=np.float64).copy()
            self._say(
                f"stroke {stroke} push: {reach * 1e3:.0f} mm along {np.round(direction, 2).tolist()} from {self._fmt(pinch)} "
                f"to {self._fmt(end)} (box frame)"
            )
            stopped = "ran its length"
            for k, _ in enumerate(
                self._leg(self.work, [float(v) for v in end], quat, speed=0.03, label=f"feed {stroke}")
            ):
                history.append(self._tip_y())
                if VERBOSE and k % 15 == 14:
                    site, _ = self.env.arm(self.work).ee_pose(self.env.state)
                    cap = self.env.state.body_q.numpy()[grip_body][:3]
                    self._say(
                        f"  push {k + 1:3d}: hand moved {np.linalg.norm(site - site0) * 1e3:5.1f} mm, gripped capsule "
                        f"{np.linalg.norm(cap - cap0) * 1e3:5.1f} mm, tip +{(history[-1] - history[0]) * 1e3:5.1f} mm; "
                        f"box shift {np.linalg.norm(self._box_origin()[:2] - box_at_push) * 1e3:.1f} mm, hold {self._hold_force():.1f} N, "
                        f"hand-on-box {self._work_box_force():.1f} N, rows {self._contact_rows(self.work)}, "
                        f"fingers q {np.round(self._finger_q(self.work) * 1e3, 1).tolist()}"
                    )
                if history[-1] >= exit_y + FEED_PAST_EXIT:
                    self._stop_pressing(self.work)
                    stopped = "put the tip out"
                    break
                if (
                    len(history) > STALL_GRACE + STALL_WINDOW
                    and history[-1] - history[-1 - STALL_WINDOW] < STALL_ADVANCE
                ):
                    self.stalls += 1
                    self._stop_pressing(self.work)
                    stopped = "STALLED (tip not advancing)"
                    break
                if float(np.linalg.norm(self._box_origin()[:2] - box_at_push)) > BOX_BUDGET:
                    self.shoves += 1
                    self._stop_pressing(self.work)
                    stopped = "SHOVED the box"
                    break
                yield
            yield from self._hold(20)
            self.strokes_done += 1
            self.hold_force_feed_n = max(self.hold_force_feed_n, self._hold_force())
            self._record_advance()
            self._record_contact()
            site, _ = self.env.arm(self.work).ee_pose(self.env.state)
            cap = self.env.state.body_q.numpy()[grip_body][:3]
            self._say(
                f"stroke {stroke} {stopped}: hand moved {np.linalg.norm(site - site0) * 1e3:.1f} mm, "
                f"gripped capsule {np.linalg.norm(cap - cap0) * 1e3:.1f} mm, tip +{(history[-1] - history[0]) * 1e3:.1f} mm "
                f"(now {self._fmt(self.env.state.body_q.numpy()[self.env.handles.entity('cable').bodies[-1]][:3])}); "
                f"box shift {np.linalg.norm(self._box_origin()[:2] - box_at_push) * 1e3:.1f} mm, hold {self._hold_force():.1f} N"
            )
            # BACK OFF BEFORE LETTING GO. The push ends with the finger
            # carriage on the flare (CARRIAGE_REACH), and a hand that opens
            # there opens against the box: the finger leaning on it cannot
            # slide open, the servo drags it, and the box goes with it. So the
            # closed hand first comes back RELEASE_UNLOAD along the line it
            # pushed on, which takes the carriage off the flare and the
            # column's compression out of the cable, and only then opens.
            unload = np.asarray(self.env.arm(self.work).ee_pose(self.env.state)[0], dtype=np.float64)
            unload = unload - RELEASE_UNLOAD * direction
            tip_at_unload = self._tip_y()
            yield from self._leg(
                self.work, [float(v) for v in unload], quat, gate=False, speed=0.03, label=f"unload {stroke}"
            )
            yield from self._hold(10)
            self._say(
                f"stroke {stroke} unload: hand back {RELEASE_UNLOAD * 1e3:.0f} mm, tip {(self._tip_y() - tip_at_unload) * 1e3:+.1f} mm, "
                f"hand-on-box {self._work_box_force():.1f} N, box shift {np.linalg.norm(self._box_origin()[:2] - box_at_push) * 1e3:.1f} mm"
            )
            end = unload
            tip_before_release = self._tip_y()
            box_before_release = self._box_origin()[:2].copy()
            yaw_before_release = self._box_yaw()

            for k, _ in enumerate(
                itertools.chain(self._ease_open(self.work, shut, open_), self._await_open(self.work))
            ):
                if VERBOSE and k % 20 == 19:
                    self._say(
                        f"  open {k + 1:3d}: asked {self.grip[self.work] * 1e3:.1f}, fingers q {np.round(self._finger_q(self.work) * 1e3, 1).tolist()} mm; "
                        f"tip {(self._tip_y() - tip_before_release) * 1e3:+.1f} mm, box {np.linalg.norm(self._box_origin()[:2] - box_before_release) * 1e3:.1f} mm "
                        f"yaw {math.degrees(self._box_yaw() - yaw_before_release):+.2f} deg, hold {self._hold_force():.1f} N, "
                        f"hand-on-box {self._work_box_force():.1f} N, rows {self._contact_rows(self.work)}, "
                        f"support q {np.round(self._finger_q(self.support) * 1e3, 1).tolist() if self.support else '-'}"
                    )
                yield
            self._say(
                f"stroke {stroke} release: jaws read open after {self.open_waits[-1]} steps, fingers q "
                f"{np.round(self._finger_q(self.work) * 1e3, 1).tolist()} mm; tip moved {(self._tip_y() - tip_before_release) * 1e3:+.1f} mm, "
                f"box {np.linalg.norm(self._box_origin()[:2] - box_before_release) * 1e3:.1f} mm during the open"
            )
            # **Sideways first, and the order is the finding.** An open hand
            # that moves off a pinch still has the cable on one pad --
            # pad-on-rod friction is the rod's own `mu_gripper` of 60, and one
            # finger measurably creeps open at 0.4 mm a step while the other is
            # already wide -- so whatever the hand does next, it does to the
            # cable too. ACROSS the cable is the one direction that neither
            # feeds it nor pulls it back out: the rod is held on its axis by the
            # bore at one end and lies on the mat at the other, so 10 mm of x is
            # taken up by a bend it springs back from, while 30 mm along the
            # cable is 30 mm of rod dragged back through the run. Measured with
            # the retreat first: the box was shoved 16 mm on the first release.
            yield from self._free_jaws(self.work, end, quat)
            # Then back down the cable, which unloads the pads before the lift
            # rather than after it.
            back = end - RELEASE_BACK * direction
            yield from self._leg(
                self.work, [float(v) for v in back], quat, gate=False, speed=0.05, label=f"release {stroke}"
            )
            yield from self._hold(15)
            yield from self._leg(
                self.work, [x_bore, float(back[1]), carry_z], quat, gate=False, speed=0.10, label=f"lift {stroke}"
            )
            yield from self._hold(10)
            self._say(
                f"stroke {stroke} done: tip moved {(self._tip_y() - tip_before_release) * 1e3:+.1f} mm through the release, "
                f"box moved {np.linalg.norm(self._box_origin()[:2] - box_before_release) * 1e3:.1f} mm; jaws freed {self.jaw_frees}x; "
                f"advance so far {self.advance * 1e3:.1f} mm, refused {self.refused}"
            )

        # --- the feeding hand takes the box by its lid at the entry end --------
        # So the draw pulls the cable through the box rather than the box along
        # the table: the box is a free 350 g body on a slick foot, and drawn with
        # nothing holding it, it rode 152 mm with the connector and the cable
        # never moved through it. The same pinch the support hand took at the
        # exit end, taken at the entry end once the feed is done.
        entry_hold = bool(self.support and self.hold)
        if entry_hold:
            lid_half = BORE / 2 + WALL
            lid = self._from_box([0.0, -RUN / 2 + ENTRY_HOLD, (BORE + ROOF) / 2])
            lid_over = self._from_box([0.0, -RUN / 2 + ENTRY_HOLD, BORE + ROOF + SUPPORT_OVER])
            here = np.asarray(self.env.arm(self.work).ee_pose(self.env.state)[0], dtype=np.float64)
            yield from self._leg(
                self.work,
                [float(here[0]), float(here[1]), float(lid_over[2]) + 0.03],
                quat,
                gate=False,
                speed=0.10,
                label="up off the riser",
            )
            yield from self._leg(
                self.work,
                [float(v) for v in lid_over],
                quat,
                grip=self._finger_for(lid_half + LID_CLEAR),
                gate=False,
                speed=0.15,
                label="over the entry lid",
            )
            yield from self._hold(15)
            off = self._settled_offset(self.work, lid_over)
            yield from self._leg(
                self.work,
                [float(lid[0]) - off[0], float(lid[1]) - off[1], float(lid[2])],
                quat,
                gate=False,
                speed=0.04,
                label="down on the entry lid",
            )
            yield from self._hold(15)
            self.grip[self.work] = self._finger_for(lid_half - HOLD_ENGAGE)
            yield from self._hold(40)
            lq = self._finger_q(self.work) * 1e3
            self._say(f"entry hold: fingers q {lq[0]:.1f}/{lq[1]:.1f} mm (asked {self.grip[self.work] * 1e3:.1f})")
        else:
            # --- the feeding hand clears --------------------------------------
            yield from self._leg(
                self.work, [x_bore - 0.16, push_from, carry_z + 0.06], quat, gate=False, speed=0.20, label="retreat"
            )

        # --- the left hand lets go of the box and takes the connector ----------
        # The half of the job the feed cannot do. A pushed cable stops where the
        # friction of the run balances the push; drawing the plug out along the
        # ledge is what a person does next, and it is the only move here that
        # acts on the head rather than on the cable behind it.
        if self.support:
            hold_quat = self._grasp_quat(self.support)
            over_z = floor_z + BORE + ROOF + SUPPORT_OVER
            self._watching_box = False
            if self.hold:
                self.grip[self.support] = self._open(self.support)
                # Until the jaws READ open: the finger servo this document states closes and opens at
                # 50 mm/s, and lifted after a fixed 20 steps the hand carried the box 100 mm up and away.
                yield from self._await_open(self.support)
                yield from self._leg(
                    self.support,
                    [x_bore, exit_y - HOLD_FROM_EXIT, over_z],
                    hold_quat,
                    gate=False,
                    speed=0.20,
                    label="release box",
                )
                yield from self._hold(15)

            # Read, not computed. The head is where the feed left it, and the
            # ledge is where the box is now.
            head = self._plug_xyz()
            box_now = self._box_origin()
            ledge_z = float(box_now[2])
            grab_y = max(float(head[1]), float(box_now[1]) + RUN / 2 + HEAD_CLEAR + HAND_HALF_Y)
            head_z = max(ledge_z + RADIUS, ledge_z + self._pad_drop(self.support))
            head_grip = self._finger_for(PLUG_W / 2 - PINCH_ENGAGE)
            self.head_at_grab = grab_y
            self._say(
                f"draw: head at {self._fmt(head)} (box frame), exit face y {(box_now[1] + RUN / 2):+.4f}; grabbing at y {grab_y:+.4f} "
                f"({(grab_y - float(head[1])) * 1e3:+.1f} mm from the head), z {head_z:.4f}"
            )

            # Across the run the pads go where the HEAD is, not where the
            # box's origin line would put it: a box that has yawed a degree
            # carries its ledge 4 mm off that line by here.
            head_x = float(head[0])
            yield from self._leg(
                self.support,
                [head_x, grab_y, ledge_z + BORE + ROOF + SUPPORT_OVER],
                hold_quat,
                grip=self._finger_for(PLUG_W / 2 + HEAD_DESCENT_CLEAR),
                gate=False,
                speed=0.20,
                label="over head",
            )
            yield from self._hold(15)
            off = self._settled_offset(self.support, [head_x, grab_y])
            yield from self._leg(
                self.support,
                [head_x - off[0], grab_y - off[1], head_z],
                hold_quat,
                gate=False,
                speed=0.04,
                label="down on head",
            )
            yield from self._hold(15)
            # The pads close across PLUG_W (15 mm), not across the cable's 10 --
            # the head is the wider thing and a pinch calibrated for the gauge
            # would crush it by 2.5 mm a side.
            self.grip[self.support] = head_grip
            yield from self._hold(60)
            self._record_contact(self.support, head=True)
            self._report_pinch(self.support, self._plug_xyz(), "draw pinch on the head")

            # DRAW, THEN LIFT, THEN FINISH THE DRAW -- and the order is forced
            # rather than chosen. An episode ends the instant its goal is met,
            # and this goal's release clause is `tcp_distance`, which measures
            # **the work arm only**. The feeding hand retreated legs ago, so that
            # clause is already true while the support hand still has the
            # connector: the goal fires the moment `end_b` crosses the exit
            # region, part way through this phase. Anything sequenced after that
            # crossing does not happen -- measured on the conduit, a lift placed
            # after a single 70 mm draw ran zero times in a recorded episode.
            # So the lift goes before the crossing: the first draw is short
            # enough to leave `end_b` short of the region, the lift happens with
            # free cable to bend, and the second draw carries it over. The
            # telemetry is updated every step for the same reason.
            before = float(self._plug_xyz()[1])

            def _drawn() -> float:
                return (float(self._plug_xyz()[1]) - before) * 1e3

            def _lifted() -> float:
                return (float(self._plug_xyz()[2]) - ledge_z - RADIUS) * 1e3

            for _ in self._leg(
                self.support, [head_x, grab_y + PULL_SEAT, head_z], hold_quat, speed=0.03, label="draw head"
            ):
                self.head_pulled_mm = _drawn()
                yield
            for _ in range(20):
                self.head_pulled_mm = _drawn()
                yield
            self._say(
                f"draw: head drawn {self.head_pulled_mm:+.1f} mm, head contact rows {self._contact_rows(self.support, head=True)}"
            )

            # Up, held, and measured off the connector rather than the wrist: a
            # hand that lifts while the head stays down has slipped, and that
            # reads as a lift from the outside.
            for _ in self._leg(
                self.support,
                [head_x, grab_y + PULL_SEAT, ledge_z + LIFT],
                hold_quat,
                speed=0.05,
                label="lift head",
            ):
                self.head_lifted_mm = _lifted()
                yield
            for _ in range(45):
                self.head_lifted_mm = _lifted()
                yield
            self._say(
                f"draw: head lifted {self.head_lifted_mm:+.1f} mm, tip {self._fmt(self.env.state.body_q.numpy()[self.env.handles.entity('cable').bodies[-1]][:3])}"
            )

            # The rest of the draw, carried at height. `end_b` crosses the exit
            # region somewhere in here and the episode ends with the connector
            # held up, which is the state the task is named for.
            for _ in self._leg(
                self.support,
                [head_x, grab_y + PULL, ledge_z + LIFT],
                hold_quat,
                speed=0.03,
                label="draw clear",
            ):
                self.head_pulled_mm = _drawn()
                self.head_lifted_mm = _lifted()
                yield
            for _ in range(25):
                self.head_pulled_mm = _drawn()
                yield
            self._say(
                f"draw clear: head drawn {self.head_pulled_mm:+.1f} mm, lifted {self.head_lifted_mm:+.1f} mm, tip {self._fmt(self.env.state.body_q.numpy()[self.env.handles.entity('cable').bodies[-1]][:3])}"
            )

            # Only reached if the goal has not already closed the episode.
            yield from self._leg(
                self.support, [head_x, grab_y + PULL, head_z], hold_quat, speed=0.05, label="set head down"
            )
            yield from self._hold(20)

            self.grip[self.support] = self._open(self.support)
            # Until the jaws READ open: the finger servo this document states closes and opens at
            # 50 mm/s, and lifted after a fixed 20 steps the hand carried the box 100 mm up and away.
            yield from self._await_open(self.support)
            yield from self._leg(
                self.support,
                [x_bore - 0.10, grab_y + PULL, ledge_z + BORE + ROOF + SUPPORT_OVER + 0.04],
                hold_quat,
                gate=False,
                speed=0.20,
                label="head retreat",
            )
        if entry_hold:
            # The entry hand lets go last and goes well clear: the goal's
            # release clause reads this hand's distance from the cable.
            self.grip[self.work] = self._open(self.work)
            # Until the jaws READ open: the finger servo this document states closes and opens at
            # 50 mm/s, and lifted after a fixed 20 steps the hand carried the box 100 mm up and away.
            yield from self._await_open(self.work)
            yield from self._leg(
                self.work, [float(v) for v in lid_over], quat, gate=False, speed=0.10, label="off the entry lid"
            )
            yield from self._leg(
                self.work,
                [float(lid_over[0]) - 0.16, float(lid_over[1]) - 0.04, float(lid_over[2]) + 0.06],
                quat,
                gate=False,
                speed=0.20,
                label="retreat",
            )
        yield from self._hold(SETTLE)
        env.request_termination("tunnel feed script complete")

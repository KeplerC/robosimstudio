# Provenance — rgbs/tasks/rgbs/assets/polish (`strip`)

Everything here is **authored for this corpus**; no upstream meshes, no
imported BDDL. The reference is a video of the real rig (`IMG_2462.MOV`,
2026-08) and a still from it: two I2RT YAM arms on the composed workcell
mounts, a Dremel-style rotary tool clamped **sideways** to a vertical stand
over a work mat — its nose pointing forward at the arms and leaning about
fifteen degrees down — a ~25 mm polishing drum on a mandrel in the collet, a
small rest ledge on the column, and a white plastic strip (~150 x 25 x 4 mm)
with blue marks that the arms press up against the running drum's **barrel**
and rub back and forth along the spindle, lifting and shifting between marks.

The tool was modelled **nose-down** in the first build of this task,
presenting the drum's flat end face to the work. That is not how a polishing
drum is used and it is not what the still shows; more to the point it made the
task's subject unmeasurable, because a spindle pointing down puts the drum's
drag in a swirl rather than in a direction, and no rub of the strip changed
anything.

## machine.py — `rotary_polisher`

Procedural geometry at the video's proportions (nothing measured to better
than ~5 mm; the numbers are stated once in `SpecPolishMachine` and asserted
where they are load-bearing). `tilt_deg = 75` — fifteen degrees off horizontal
— is the mounting, and everything else derives from it through
`machine_frame()`, the single place the working contact point, the spindle axis
and the working radial are computed. The tool's barrel forward of the clamp
collides; the motor housing behind it is drawn and does not, because it lies
behind the column where no arm reaches and colliding it would push the stand
off the back of the table.

The drum collider is a convex 16-gon prism
mesh, **never** a cylinder primitive: mujoco_warp emits at most one contact
point for cylinder pairs (its own runtime warning) and the single point rocks;
box/mesh pairs under `enable_multiccd` get a face manifold. That matters more
for a barrel contact than it did for a face one — a pressed strip rides on one
or two facets at a time. The spindle drive
is Newton's native VELOCITY joint actuator with a torque ceiling
(`effort_limit`), which measured on this stack as: exact speed hold under
contact load, honest spin-up (~0.15 s), ~4% droop under an 8 N press, stall
only above ~24 N — above anything the task commands.

## marks.py — `polish_markfield`

The removal law is **Preston's equation** (`rate = k_p * pressure * |v_rel|`),
the standard empirical model for glass/CMP polishing, applied per 2 mm cell on
the strip's top face. Pressure comes from the solver itself: a private
`ContactProbe` reads the strip-against-bit pairwise force once per control
step and a 10-step mean-of-nonzero (the placement handler's dropout treatment)
feeds the in-graph kernel. Structure — device arrays, irreversible per-element
state, bit-exact reset from a recorded mask, cached published states — is the
peel tape bond field's, inverted: peel breaks on *opening* displacement so
pressing cannot unstick it; polishing removes *by* pressing and sliding.

`k_p = 6.0e-4 (Pa*m)^-1` is fitted on `rgbs/scripts/rgbs_polish_bench.py --rub`, against
the ~3.1 N press the arms deliver and the nine out-and-back rubs the shipped
schedule spends on a mark. What makes a mark need rubbing at all is geometry
rather than the coefficient: the barrel's contact band is 10 mm along the
spindle against a mark 18 mm across the strip, so a press alone clears 0.68 of
a mark at every overdrive from 1 to 6 mm and then stops. Rubbing along the
spindle finishes it; rubbing across the spindle stays at 0.68 while `k_p` is
swept over a factor of ten, because 0.68 is where the band ends rather than
where the rate does.

The marks are `8 x 18 mm` for the same reason. Round 10 mm spots — what this
first shipped with — sit entirely inside the band at any real press, so a spot
cleared standing still and the technique was decoration.

The drag is measurable straight off the solver, with no holder in the way:
`--drag` resolves the strip↔drum pairwise force on the machine's frame and
reads 3.45 N across the spindle against **0.00 N along it**, at a 3.3 N normal
force. The contact has no drag to give in the direction the rub runs.

## records.py

Declarations only. The strip's grasp box restates the procedural box's own
half extents; the `spinning` state thresholds the published spindle speed at
80% of command (below neither the press droop nor anything but a dead drum).
That threshold read `48.0` — 80% of a 60 rad/s command — for as long as the
drum was 13 mm, and was not carried down when the spindle was halved to 30, so
the state could never be true. Nothing on the shipped path resolved it, which
is exactly why it survived; it is 24.0 now.

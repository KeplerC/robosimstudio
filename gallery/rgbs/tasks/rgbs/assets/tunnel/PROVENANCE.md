# `rgbs` tunnel assets (`thread`)

## `cable_box`

**Sized off `channel routing.mp4`, authored as boxes.** No upstream mesh, no
vendored CAD. The clip shows a cream plastic cable-management box standing on
the gridded mat with its long axis along the bench, a black dimpled anti-slip
mat over its top, a cream lid strip standing proud along the top-front edge
with a dark slot behind it, a white six-outlet power strip lying on the mat
toward the far edge with its cord looping off the right end, and a dark notch
low in the front face. That is what `box.py` builds, on one free body.

What was measured, and how (all off `outputs/rgbs_reference/channel_routing/`):

* **Length**: counting the mat's 25 mm grid along the box's front-bottom edge
  in frame 0 gives 18-20 squares (450-500 mm); projecting the same edge through
  the suite's `photo` camera gives 390. Built at 420, the value the bench admits
  -- the cable's tail has to land on the table behind the entry, and the entry
  is the box's right end (`layout.py::TUNNEL_Y`).
* **Depth**: the overhead inset shows mat on both sides of the 50 mm power
  strip and the top running out of the frame, so at least 200 mm; the phone's
  foreshortening reads it as about half that. Built at 200.
* **Height**: the front face is 235 px tall in frame 0 against 55 px grid
  squares at the phone's 25-degree look-down, 105-115 mm. Built at 105 to the
  mat top, with the lid 10 mm proud of it.
* **Lid strip**: about two thirds of the box's length in the inset, offset to
  the right; 250 of 420 here, from the right end -- the run is set by the rod
  (`box.py::RUN`).
* **Power strip**: 64 % of the box's length in the inset, 50 mm deep, 28 mm
  tall, centred on the box near its far edge; 270 x 50 x 28 here.
* **Colours**: cream sampled by the suite as `look.DUCT_CREAM`; the mat's base
  is sRGB 40/40/42 between the dimples, raised toward 70 on the crowns
  (`box._dimple`); the strip is white satin.

The bore stays the conduit's: 17 mm square, sized to the plug, drawn as walls
and a roof (the lid strip) on the box's own shelf, with the flared entry at the
right end face -- and now a flare floor, because the bench is no longer under
the mouth. Boxes rather than a mesh for the reason the conduit was: MuJoCo
collides a mesh geom as its convex hull and the hull of a channel is a solid
block. See `box.py`'s module docstring.

The box is a **free body** of 350 g (`rigid` domain, served by the arms' own
MuJoCo backend -- `staging.py`'s arrangement names that one spec object under
both `articulated` and `rigid`, which is what puts the box in the arms' island
rather than a second MuJoCo one the proxy coupler would refuse), mirrored into
the cable's VBD island two ways. Its foot meets the table at whatever the rod's
`scene_friction` pass leaves it (`mu_base` 2.0 against `MU_SURFACE`), which is
what keeps a 350 g box still under a feed of tens of millinewtons; the support
hand's pinch on the lid is what keeps it still under anything else.

## `tunnel_cable`

**Authored here, from cable_route's weave rod.** The 10 mm gauge and the ~11.6 mm
capsule pitch are `vabar/tasks/cable_route/task.py`'s `WEAVE_CABLE_RADIUS` and its
segment pitch, reused so that the bend stiffness this suite derives at material
parity is derived by the same arithmetic against the same reference. The length
is this suite's own: 0.44 m, set together with the run by the bench (the tail
lands on the table) and the route (the rod covers the run, the exit clearance
and a tail a hand can hold). The plug is a black moulded strain relief -- an
overmold, a nose and a rounded cap -- as the clip's is, and it is the binding
dimension in the bore.

The rod mechanism itself is the rope kind's (`examples/10_rope/rod/`), which is Newton's
`add_rod` at the `newton_tasks/sim/yam_cable_sim.py` reference defaults.

**What the bore's height costs, and how the seed answers it.** The channel is
100 mm up, so the cable has to get there from the mat, and at material parity
this rod's section (EI 5.6e-5 N*m^2 against 27.5 g/m) cannot bridge that in
air. Seeded straight along the bore axis it free-falls and levers the tip 33 mm
back out before a hand has moved; seeded on a 250 mm ramp it collapses within
90 steps and drags the tip 24 mm. So the rest shape is the hang the rod itself
settles into, measured off the built scene and reproduced by a smoothstep over
130 mm (`layout.py::RAMP_RUN`):

    back from the entry   -8    24    46    64    89   111   133 mm
    above the mat        101    89    63    33    10     2     0 mm

Seeded that way the tip holds its 80 mm for the settle and creeps ~5 mm. The
routine never assumes the shape: it reads the cable live where it means to
pinch, and drives the push along the line from there to the mouth.

**A frame trap that cost an afternoon, recorded because nothing raised.** The
polyline is stated in the cell's WORK-SURFACE frame while `TUNNEL_X`/`TUNNEL_Y`
are world coordinates; this cell's surface frame is offset 0.6 m in x. Handing
the raw world x to the entity put every node 0.6 m beyond the bench's far edge,
where the cable free-fell 850 mm to the floor at exactly g with no contact at
all -- which reads as a solver that has lost the rod's collisions, not as a rod
in the wrong place. `cable_nodes` builds in world and converts with
`point_from_surface`, as every `start=` in this tree already did.

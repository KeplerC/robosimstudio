# SPDX-License-Identifier: Apache-2.0
"""The arrangements a particle scene stages: ``GENERIC`` with the rigid props beside the robot.

:data:`~robosimstudio.specs.GENERIC` already routes ``granular``, ``liquid`` and
``dough`` to the ``mpm`` backend at its registered defaults, and a heap alone
beside a robot composes under it. What it does not do is hold a free rigid
tool: ``GENERIC`` puts free bodies on VBD behind the proxy coupler, which makes a
robot, a paddle and a heap **three** Newton islands, and the proxy coupler
couples two. So the arrangements here say ``("rigid", backend("mujoco"))``, the
same spec as the arm's, and a paddle, a rolling pin or a cup is a body of the
robot's island -- which is also the island a hand can hold a rigid body in
(:data:`~robosimstudio.specs.MUJOCO_500HZ` makes the same choice for the same
reason).

The rate is ``GENERIC``'s: sixteen substeps at 60 Hz. The heap and the lump
step the MPM once per control step (``step_hz=60``, the registered default),
the pool twice (``step_hz=120``). Measured on an RTX PRO 6000
with Newton 1.6.0, graph-captured: a 28,812-particle rice bed beside the YAM
bimanual on ``bench_075`` costs 15.6 ms per control step at 60 Hz and 25.9 ms
at 120 Hz (the robot alone is 4.2 ms). What the rate buys is travel per MPM
step: the SDF band is ~1.7 voxels, so at 4 mm voxels and 60 Hz a tool has to
stay under ~0.36 m/s or it tunnels (``probe_b``), and a liquid that leaves its
cup is past that speed after 7 mm of free fall, which is why the pool pays for
120 Hz (:data:`LIQUID_120HZ`) and a heap, which nothing drops, does not.
Another rate is one ``replace`` away: ``GRANULAR_60HZ.with_backend("granular",
backend("mpm", step_hz=120.0))``; 60, 120, 240, 480 and 960 fit the sixteen
substeps.

Three tables rather than one because three things differ. ``project_outside``
after each MPM step is what keeps a thin tool from leaving grains inside it
(``probe_b``: 9.9% of the swept rice left behind with it, 14-16% without), and
it is wrong for a liquid: it pushes particles into the container wall and cost
6x in the cup probe. So :data:`GRANULAR_60HZ` and :data:`DOUGH_60HZ` state it and
:data:`LIQUID_120HZ` states it off. The two that project do so at
:data:`PROJECT_GAP` rather than Newton's default, because the default gap is
the same ~1.7 voxels as the solve's own band, so a grain that crossed the band
in one step -- rice let go from a hand's height -- was through the bench for
good; a 30 mm gap reaches it and puts it back. A plastic solid needs more solver
iterations than grains or it loses volume at rest (:data:`DOUGH_ITERATIONS`),
and a liquid needs the rate above or it falls through what it is poured into.
Each difference makes that kind's ``mpm`` spec unequal to the others', so a
scene holding two kinds gets two MPM islands, and the proxy coupler refuses
that beside a robot. One particle kind per robot scene under these tables; a
scene that wants two states one spec for both.

All three mirror the rigid bodies into the MPM at :data:`PROXY_MASS_SCALE`
times their mass, which is what lets a free tool hold a liquid or push a heap
rather than be moved aside inside the MPM's own step.
"""

from __future__ import annotations

from robosimstudio.specs.physics import GENERIC, ArrangementSpec, BackendSpec, backend, coupler

__all__ = [
    "DOUGH_60HZ",
    "DOUGH_ITERATIONS",
    "GRANULAR_60HZ",
    "LIQUID_120HZ",
    "PROJECT_GAP",
    "PROXY_MASS_SCALE",
    "with_particles_on",
]

PROXY_MASS_SCALE = 1000.0
"""How much heavier the MPM sees a mirrored rigid body than the rigid solver does.

The implicit MPM moves its colliders inside its own step by the impulse the
particles put on them, divided by the collider's effective mass -- which for a
free rigid body is that body's own mass, and nothing holds it. Measured with a
50 mm cube of honey (0.175 kg) on an 8 mm free plate and in a free cup, both
resting on the bench, the MPM at 60 Hz (``probe4``, re-run with its output
archived, RTX PRO 6000, Newton 1.6.0, 2 s at rest): at the coupler's default
scale 4,757 of 8,000 particles were through the plate by the end of the settle
and 7,863 after 2 s; at 20x, 3,544 and 6,682; with the plate weighing 2 kg at
the default scale, 1,860 and 6,613; at 1000x the plate kept 7,055 after 2 s
and the cup 7,969 of 8,000. World-fixed slabs, which the MPM holds as
immovable, kept 7,702. The arm's hand needs none of this (its effective mass through the
chain is 5 kg), and the scale changes only what the MPM assumes while it
solves: the wrench the rigid solver receives is the harvested impulse either
way (``probe_b``: identical particle outcomes and wrenches at 1 and 1e6 for a
paddle welded to the arm)."""


def with_particles_on(arrangement: ArrangementSpec, spec: BackendSpec) -> ArrangementSpec:
    """``arrangement`` with the three particle domains on ``spec``, free rigid bodies on the arm's solver, and the proxies heavy.

    One ``BackendSpec`` for ``granular``, ``liquid`` and ``dough`` is one island
    by the composer's equality rule. The rigid entry copies the ``articulated``
    one, so a tool is a body of the robot's island; stated explicitly rather
    than relying on the fallback, which is VBD under ``GENERIC``. The coupler
    is the arrangement's own with ``mass_scale`` set to :data:`PROXY_MASS_SCALE`,
    unless the arrangement already states one.
    """
    arm = arrangement.backend_for("articulated") or backend("mujoco")
    table = arrangement
    for domain in ("granular", "liquid", "dough"):
        table = table.with_backend(domain, spec)
    table = table.with_backend("rigid", arm)
    link = table.coupling if table.coupling.method == "proxy" else coupler("proxy")
    if link.get("mass_scale") is None:
        link = link.with_(mass_scale=PROXY_MASS_SCALE)
    return table.with_(coupling=link)


PROJECT_GAP = 0.03
"""How far the projection after each MPM step looks for a collider surface [m]: 7.5 of the registered 4 mm voxels.

Newton's default is ``voxel * sqrt(3)``, 6.9 mm, which is also the width of the
band the solve itself sees a collider in. A particle that crosses that band
inside one step is beyond both: it sees no collider, is not projected, and
falls through the bench top and the ground. At 60 Hz the band is crossed at
0.42 m/s, which rice reaches 9 mm into a free fall, so a heap dropped by a
hand, a paddle tipped over a tray or grains thrown up by a tool all lose
particles. Measured with ``probe_drop`` (GPU 3, RTX PRO 6000, Newton 1.6.0,
graph-captured, beside the YAM on ``bench_075``): a 27,648-particle block of
the rice preset let go 80 mm over the bench put 10,183 particles through the
bench and the ground with the default gap, 1,429 from 30 mm and 309 from
150 mm; with a 30 mm gap none at any height, and none from a block built on
the bench. 30 mm covers a step at 1.8 m/s, a fall of 165 mm, and costs
nothing measurable: 16.6 ms per control step against 16.8 with the default on
that scene. A wider gap would also start projecting a particle out through
the far side of a thin wall, which at 30 mm a two-voxel wall is already prone
to under a hand's push, so a tray a heap is swept into has walls of five."""

GRANULAR_60HZ = with_particles_on(GENERIC, backend("mpm", project_outside=True, project_gap=PROJECT_GAP))
"""A heap beside the robot, its tools on the robot's island, the MPM at 60 Hz with the projection on.

``GENERIC``'s rates and coupler; ``project_outside=True`` is the registered
default restated so that the spec reads as what it is, at :data:`PROJECT_GAP`
rather than the registered default gap. 15.6 ms per 60 Hz control step for a
28,812-particle rice bed and the YAM bimanual on ``bench_075`` (RTX PRO 6000,
Newton 1.6.0, graph-captured)."""

DOUGH_ITERATIONS = 150
"""Solver iterations per MPM step for a plastic solid, against the registered 50.

A soft solid (E 5e6) is not converged at 50 iterations and 1e-4, and what an
unconverged step loses is volume: a 60 mm by 40 mm puck of the dough preset at
rest on the bench, nothing touching it, lost 17% of its height half a second
after settling and 24% after 2 s at 50 iterations (``probe5``, RTX PRO 6000, Newton
1.6.0, graph-captured, beside the YAM). 100 iterations: 2.7% in 2 s at 19.5 ms
per control step; 150: 1.1% at 23.4 ms; 250: 0.2% at 31.1 ms; 50 cost 15.7 ms.
150 is the last rung inside the 25 ms the GENERIC scene was budgeted. A
granular (E 1e15) does not creep this way and keeps the 50."""

DOUGH_60HZ = with_particles_on(
    GENERIC, backend("mpm", project_outside=True, project_gap=PROJECT_GAP, max_iterations=DOUGH_ITERATIONS)
)
"""A lump beside the robot: the granular table at :data:`DOUGH_ITERATIONS`.

The same projection as the heap's, at the same gap (it keeps a rolling pin
from leaving dough inside it), three times the iterations, so the lump holds its height while
nothing presses it. A different ``mpm`` spec from :data:`GRANULAR_60HZ`, so a
heap and a lump in one robot scene are two MPM islands and refused by the
proxy coupler; a scene wanting both states one spec for both
(``GRANULAR_60HZ.with_backend("dough", GRANULAR_60HZ.backend_for("granular"))``
and accepts the creep, or the other way round and pays the iterations)."""

LIQUID_120HZ = with_particles_on(GENERIC, backend("mpm", project_outside=False, step_hz=120.0))
"""A pool beside the robot, the projection off and the MPM stepped twice per control step.

The projection pushes a liquid's particles into the walls that hold it and
cost 6x in the cup probe (``probe_c_water``), so it is off. What keeps the
liquid in a thin container instead is a signed-distance margin on the
container's shapes, which :class:`~robosimstudio_materials.mpm.spec.ParticleBodySpec`
writes for the entities named in ``containers``; without it, walls thinner
than about two voxels (8 mm at the registered 4 mm) leak under the capturable
``S2`` collider basis.

The rate is the pour's. The 0.36 m/s rule a tool obeys at 60 Hz binds a
falling liquid too, and honey is past it 7 mm below the lip it left
(``v = sqrt(2 g h)``): the pour example's first scene, its cup carried by the drag
handle to 100 mm over the bench beside the bowl, tilted 130 degrees over 2 s
and held 3 s (``pour_probe``, RTX PRO 6000, Newton 1.6.0, graph-captured),
had at 60 Hz 17 of 8,000 particles below the bench before the cup had tilted,
85 after the hold, 944 on the bench outside the bowl and 4.0% in it; at
120 Hz none below the bench at any stage, 34 outside and 42.5% in the bowl;
at 240 Hz 55.2% in the bowl and 12 outside, for 53.5 ms per control step
against 30.0 at 120 Hz and 18.2 at 60. From 170 mm, 60 Hz put 406 through
after the hold and 0.2% in the bowl; 120 Hz put 5 through and 65.5% in. So
120 Hz is the first rung at which a pour from a hand's height lands, not the
rate at which nothing ever tunnels, and the puddle example pays it too rather
than ship two liquid tables."""

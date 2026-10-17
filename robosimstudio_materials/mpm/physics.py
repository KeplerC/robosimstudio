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

The rate is ``GENERIC``'s: sixteen substeps at 60 Hz, the MPM stepping once per
control step (``step_hz=60``, the registered default). Measured by the core
agent on an RTX PRO 6000 with Newton 1.6.0, graph-captured: a 28,812-particle
rice bed beside the YAM bimanual on ``bench_075`` costs 15.6 ms per control
step at 60 Hz and 25.9 ms at 120 Hz (the robot alone is 4.2 ms). 120 Hz halves
a tool's travel per MPM step -- the SDF band is ~1.7 voxels, so at 4 mm voxels
and 60 Hz a tool has to stay under ~0.36 m/s or it tunnels (``probe_b``) -- and
is one ``replace`` away: ``GRANULAR_60HZ.with_backend("granular", backend("mpm",
step_hz=120.0))``; 60, 120, 240, 480 and 960 fit the sixteen substeps.

Three tables rather than one because two things differ. ``project_outside``
after each MPM step is what keeps a thin tool from leaving grains inside it
(``probe_b``: 9.9% of the swept rice left behind with it, 14-16% without), and
it is wrong for a liquid: it pushes particles into the container wall and cost
6x in the cup probe. So :data:`GRANULAR_60HZ` and :data:`DOUGH_60HZ` state it and
:data:`LIQUID_60HZ` states it off. And a plastic solid needs more solver
iterations than grains or it loses volume at rest (:data:`DOUGH_ITERATIONS`).
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

__all__ = ["DOUGH_60HZ", "DOUGH_ITERATIONS", "GRANULAR_60HZ", "LIQUID_60HZ", "PROXY_MASS_SCALE", "with_particles_on"]

PROXY_MASS_SCALE = 1000.0
"""How much heavier the MPM sees a mirrored rigid body than the rigid solver does.

The implicit MPM moves its colliders inside its own step by the impulse the
particles put on them, divided by the collider's effective mass -- which for a
free rigid body is that body's own mass, and nothing holds it. Measured with a
50 mm cube of honey (0.175 kg) on an 8 mm free plate and in a free cup, both
resting on the bench (``probe4``, RTX PRO 6000, Newton 1.6.0, 2 s at rest):
at the coupler's default scale 3,545 of 8,000 particles were through the plate
by the end of the settle and 6,682 after 2 s; at 20x, the same; with the plate
weighing 2 kg, the same; at 1000x the plate kept 7,056 after 2 s and the cup
7,970 of 8,000. World-fixed slabs, which the MPM holds as immovable, kept
7,702. The arm's hand needs none of this (its effective mass through the
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


GRANULAR_60HZ = with_particles_on(GENERIC, backend("mpm", project_outside=True))
"""A heap beside the robot, its tools on the robot's island, the MPM at 60 Hz with the projection on.

``GENERIC``'s rates and coupler; ``project_outside=True`` is the registered
default restated so that the spec reads as what it is. 15.6 ms per 60 Hz
control step for a 28,812-particle rice bed and the YAM bimanual on
``bench_075`` (RTX PRO 6000, Newton 1.6.0, graph-captured)."""

DOUGH_ITERATIONS = 150
"""Solver iterations per MPM step for a plastic solid, against the registered 50.

A soft solid (E 5e6) is not converged at 50 iterations and 1e-4, and what an
unconverged step loses is volume: a 60 mm by 40 mm puck of the dough preset at
rest on the bench, nothing touching it, lost 17% of its height in 1.7 s and
28% in 4.2 s at 50 iterations (``probe2``/``probe5``, RTX PRO 6000, Newton
1.6.0, graph-captured, beside the YAM). 100 iterations: 2.7% in 2 s at 19.5 ms
per control step; 150: 1.1% at 23.4 ms; 250: 0.2% at 31.1 ms; 50 cost 15.7 ms.
150 is the last rung inside the 25 ms the GENERIC scene was budgeted. A
granular (E 1e15) does not creep this way and keeps the 50."""

DOUGH_60HZ = with_particles_on(GENERIC, backend("mpm", project_outside=True, max_iterations=DOUGH_ITERATIONS))
"""A lump beside the robot: the granular table at :data:`DOUGH_ITERATIONS`.

The same projection as the heap's (it keeps a rolling pin from leaving dough
inside it), three times the iterations, so the lump holds its height while
nothing presses it. A different ``mpm`` spec from :data:`GRANULAR_60HZ`, so a
heap and a lump in one robot scene are two MPM islands and refused by the
proxy coupler; a scene wanting both states one spec for both
(``GRANULAR_60HZ.with_backend("dough", GRANULAR_60HZ.backend_for("granular"))``
and accepts the creep, or the other way round and pays the iterations)."""

LIQUID_60HZ = with_particles_on(GENERIC, backend("mpm", project_outside=False))
"""A pool beside the robot, the projection off.

The projection pushes a liquid's particles into the walls that hold it and
cost 6x in the cup probe (``probe_c_water``), so it is off. What keeps the
liquid in a thin container instead is a signed-distance margin on the
container's shapes, which :class:`~robosimstudio_materials.mpm.spec.ParticleBodySpec`
writes for the entities named in ``containers``; without it, walls thinner
than about two voxels (8 mm at the registered 4 mm) leak under the capturable
``S2`` collider basis."""

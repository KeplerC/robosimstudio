# SPDX-License-Identifier: Apache-2.0
"""A strip held down by rows of bonds and drawn up at 90 degrees loses its rows in order, one at a time.

Eight 10 mm plates in a row, each a free body, hinged to the next along their lower edges by two
bonds that never break. The strip's free end and every hinge line is held to the world by a row of
breakable bonds of 2 N, so the strip is a tape with no bending stiffness stuck down every 10 mm. A
0.2 kg puller, bonded to the free end, is driven straight up at 10 mm/s by a spring and is free along
the strip, so the hanging plates draw it over the front row and the pull there stays vertical.

A tape with no bending stiffness drawn up at 90 degrees loads only the row at the front, so the pull
when row k goes is the row's strength plus the weight already off the base:

    P_k = bonds per row x strength + (k + 1/2) x plate weight

Between rows the pull falls back to the hanging weight: rigid plates and bonds that do not stretch
store nothing. Checked here: the rows go in order, one row at a control step at most, a pitch over the
speed apart; the puller's spring reads within 20% of P_k as each goes; and no row the rule broke is
ever on again in the solver, also after it is told the constraints changed, which copies the model's
flags back over its own.

The plates carry a light drag (0.05 N s/m, 0.5 mN at the peel speed): nothing else damps the hanging
chain, and its swing loads the front row. They weigh 4 g because a taut chain of light plates under a
pull P rings at sqrt(3 P / (m L)), and at 0.5 g that is past what a 500 Hz step integrates.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

TIER = "medium"
"""Two strips of eight plates on one GPU, each peeled for nine seconds of scene time."""

PLATES, PITCH, STRENGTH, SPEED = 8, 0.010, 2.0, 0.010
"""Plates, their length [m], a bond's strength [N] and the puller's rise [m/s]."""
PLATE_MASS, PULLER_MASS = 0.004, 0.2
THICK, HALF_WIDTH, BASE = 0.002, 0.010, 0.3
"""A plate's thickness and half-width, and the height of the plane the strip is stuck to [m]."""
STIFFNESS = 3000.0
"""The puller's spring [N/m]: 30 N/s at the peel speed, 0.06 N a substep."""
GRAVITY = 9.81


@wp.kernel
def _rig_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    puller: int,
    plates: wp.array[int],
    stiffness: float,
    damping: float,
    speed: float,
    weight: float,
    dt: float,
    clock: wp.array[float],
    pull: wp.array[float],
    body_f: wp.array[wp.spatial_vector],
):
    """The puller's spring toward a height rising at ``speed``, its hold across the strip, and the plates' drag.

    ``pull[0]`` is the spring's force over the puller's own weight now, ``pull[1]`` the largest since a
    test last zeroed it.
    """
    i = wp.tid()
    if i == 0:
        clock[0] = clock[0] + dt
        q = wp.transform_get_translation(body_q[puller])
        v = wp.spatial_top(body_qd[puller])
        lift = stiffness * (clock[1] + speed * clock[0] - q[2]) + damping * (speed - v[2])
        across = -stiffness * q[1] - damping * v[1]
        wp.atomic_add(body_f, puller, wp.spatial_vector(wp.vec3(-0.5 * v[0], across, weight + lift), wp.vec3()))
        pull[0] = lift
        pull[1] = wp.max(pull[1], lift)
    else:
        b = plates[i - 1]
        qd = body_qd[b]
        wp.atomic_add(body_f, b, wp.spatial_vector(-0.05 * wp.spatial_top(qd), -2.0e-7 * wp.spatial_bottom(qd)))


def _peel(per_row: int) -> dict:
    import newton

    import robosimstudio as rss
    from robosimstudio_materials.bond import BondPoint

    scene = rss.Scene(cell="void", name="peel", physics=rss.MUJOCO_500HZ)
    for i in range(PLATES):
        half = (0.5 * PITCH - 1.0e-4, HALF_WIDTH, 0.5 * THICK)
        scene.add(
            rss.box(name=f"plate_{i}", half=half, at=((i + 0.5) * PITCH, 0.0, BASE + 0.5 * THICK), mass=PLATE_MASS)
        )
    scene.add(rss.sphere(name="puller", radius=0.003, at=(0.0, 0.0, BASE), mass=PULLER_MASS))
    across = [0.0] if per_row == 1 else list(np.linspace(-0.6 * HALF_WIDTH, 0.6 * HALF_WIDTH, per_row))
    rows = [
        BondPoint(a=f"plate_{i}", at=(i * PITCH, y, BASE), normal=(0.0, 0.0, 1.0), strength=STRENGTH, group=f"row_{i}")
        for i in range(PLATES)
        for y in across
    ]
    hinges = [
        BondPoint(a=f"plate_{i}", b=f"plate_{i + 1}", at=((i + 1) * PITCH, y, BASE), strength=1.0e9, group="hinge")
        for i in range(PLATES - 1)
        for y in (-HALF_WIDTH, HALF_WIDTH)
    ]
    ends = [
        BondPoint(a=f"plate_{PLATES - 1}", at=(PLATES * PITCH, y, BASE), strength=1.0e9, group="end")
        for y in (-HALF_WIDTH, HALF_WIDTH)
    ]
    link = BondPoint(a="puller", b="plate_0", at=(0.0, 0.0, BASE), strength=1.0e9, group="link")
    scene.add(rss.bond(name="strip", points=[*rows, *hinges, *ends, link]))
    sim = scene.build()
    strip = sim["strip"].entity
    pipeline = sim.pipeline
    device = pipeline.model.device
    puller = int(sim["puller"].entity.body)
    plates = wp.array([int(sim[f"plate_{i}"].entity.body) for i in range(PLATES)], dtype=wp.int32, device=device)
    clock = wp.array([0.0, BASE], dtype=wp.float32, device=device)
    pull = wp.zeros(2, dtype=wp.float32, device=device)
    damping = 2.0 * float(np.sqrt(STIFFNESS * PULLER_MASS))

    def rig(state, dt: float) -> None:
        wp.launch(
            _rig_kernel,
            dim=PLATES + 1,
            inputs=[state.body_q, state.body_qd, puller, plates, STIFFNESS, damping, SPEED, PULLER_MASS * GRAVITY, dt],
            outputs=[clock, pull, state.body_f],
            device=device,
        )

    pipeline.hooks.rigid_force.append(rig)
    pipeline.invalidate_graph()
    n = len(rows)
    row_of = np.repeat(np.arange(PLATES), per_row)
    was = np.zeros(n, dtype=bool)
    order, pulls, times, together, back_on, told = [], [], [], 0, 0, False
    while sim.time < PLATES * PITCH / SPEED + 2.0:
        sim.run(1)
        gone = strip.let_go()[:n]
        back_on += int((strip.active()[:n] & gone).sum())
        new = sorted(set(row_of[gone & ~was].tolist()))
        if new:
            together = max(together, len(new))
            for row in new:
                order.append(row)
                times.append(sim.time)
                pulls.append(float(pull.numpy()[1]))
            pull.zero_()
        was = gone
        if not told and gone.sum() >= n // 2:
            pipeline.program.notify(int(newton.ModelFlags.CONSTRAINT_PROPERTIES))
            told = True
        if gone.all():
            break
    return {
        "order": order,
        "pulls": pulls,
        "times": times,
        "together": together,
        "back_on": back_on,
        "told": told,
        "predicted": [per_row * STRENGTH + (k + 0.5) * PLATE_MASS * GRAVITY for k in range(PLATES)],
        "others": strip.let_go()[n:],
        "captured": bool(pipeline.captured),
    }


class TestAStripPeelsRowByRow(unittest.TestCase):
    def test_the_rows_go_in_order_at_their_strength_and_stay_gone(self) -> None:
        for per_row in (1, 2):
            with self.subTest(bonds_per_row=per_row):
                got = _peel(per_row)
                self.assertEqual(list(range(PLATES)), got["order"], "the rows did not go in order, each once")
                self.assertEqual(1, got["together"], "two rows went in one control step")
                gaps = np.diff(got["times"])
                np.testing.assert_allclose(gaps, PITCH / SPEED, atol=0.15, err_msg="a row every pitch over speed")
                error = np.abs(np.array(got["pulls"]) / np.array(got["predicted"]) - 1.0)
                self.assertLess(
                    error.max(), 0.20, f"pulls {np.round(got['pulls'], 3)} against {np.round(got['predicted'], 3)}"
                )
                self.assertEqual(0, got["back_on"], "a broken row was on again in the solver")
                self.assertTrue(got["told"], "the solver was never told the constraints changed")
                self.assertFalse(got["others"].any(), "a hinge, an end or the link let go")
                self.assertTrue(got["captured"])


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""The contact-force readout's pure parts, on the CPU: the per-shape sum and the force-gated wipe.

The one-way coupler asks VBD for one wrench per rigid-soft contact record
(Newton 1.7's ``CONTACT_F``) and :meth:`SolverOneWay.shape_contact_forces` sums
the live records by the shape each was against;
:class:`robosimstudio_materials.cloth.ForceGatedWipe` turns the hand's push and
the particles it touches into a wiped fraction. Both are checked here against
hand-written records, so what the GPU tests measure rests on bookkeeping already
known to be right: records past the live count and records with no shape count
for nothing, the force rows start after the rigid ones, an edge or face record
touches each of its corners, and a near miss (a record carrying no force) touches
nothing. The physics -- that the push is zero off contact, grows with the press
and reaches the bench -- is :mod:`tests.test_scrub_a_stain`'s.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

TIER = "fast"
"""Warp on the CPU and numpy; no scene is built."""


def _coupler(records, *, shape_count=4, rigid=3, capacity=6):
    """A one-way coupler holding ``records``: ``(corners, shape, force)`` rows, past the ``rigid`` rows."""
    import warp as wp

    from robosimstudio.scene.physics.couplers import SolverOneWay

    corners = np.full((capacity, 3), -1, dtype=np.int32)
    shapes = np.full(capacity, -1, dtype=np.int32)
    rows = np.zeros((rigid + capacity, 6), dtype=np.float32)
    for i, (c, s, f) in enumerate(records):
        corners[i, : len(c)] = c
        shapes[i] = s
        rows[rigid + i, :3] = f
        rows[rigid + i, 3:] = (9.0, 9.0, 9.0)  # a torque, which the sums must leave out
    rows[0, :3] = (100.0, 100.0, 100.0)  # a rigid row, which the soft sums must not reach
    solver = object.__new__(SolverOneWay)
    solver.device = "cpu"
    solver.model = SimpleNamespace(shape_count=shape_count)
    solver.contacts = SimpleNamespace(
        soft_contact_count=wp.array([len(records)], dtype=wp.int32, device="cpu"),
        soft_contact_shape=wp.array(shapes, dtype=wp.int32, device="cpu"),
        soft_contact_indices=wp.array(corners, dtype=wp.vec3i, device="cpu"),
        soft_contact_max=capacity,
        rigid_contact_max=rigid,
    )
    solver.observables = SimpleNamespace(contact_f=wp.array(rows, dtype=wp.spatial_vector, device="cpu"))
    solver._shape_forces = wp.zeros(shape_count, dtype=wp.vec3, device="cpu")
    return solver


class TestTheCouplerSumsTheLiveRecordsByShape(unittest.TestCase):
    def test_each_shape_gets_its_records_and_nothing_else(self) -> None:
        solver = _coupler([((0,), 1, (0.0, 0.0, 2.0)), ((1, 2), 1, (0.0, 1.0, 3.0)), ((3, 4, 5), 2, (1.0, 0.0, -5.0))])
        np.testing.assert_allclose(
            solver.shape_contact_forces(), [[0, 0, 0], [0, 1, 5], [1, 0, -5], [0, 0, 0]], atol=1e-6
        )

    def test_records_past_the_live_count_and_without_a_shape_count_for_nothing(self) -> None:
        solver = _coupler([((0,), 1, (0.0, 0.0, 2.0)), ((1,), -1, (0.0, 0.0, 7.0))])
        solver.contacts.soft_contact_count.assign(np.array([1], dtype=np.int32))
        np.testing.assert_allclose(solver.shape_contact_forces()[1], [0, 0, 2], atol=1e-6)
        solver.contacts.soft_contact_count.assign(np.array([2], dtype=np.int32))
        np.testing.assert_allclose(solver.shape_contact_forces().sum(axis=0), [0, 0, 2], atol=1e-6)

    def test_the_records_come_back_trimmed_to_the_live_ones(self) -> None:
        solver = _coupler([((0,), 1, (0.0, 0.0, 2.0)), ((3, 4, 5), 2, (1.0, 0.0, -5.0))])
        corners, shape, force = solver.soft_contact_records()
        np.testing.assert_array_equal(corners, [[0, -1, -1], [3, 4, 5]])
        np.testing.assert_array_equal(shape, [1, 2])
        np.testing.assert_allclose(force, [[0, 0, 2], [1, 0, -5]], atol=1e-6)

    def test_only_the_live_records_come_off_the_device(self) -> None:
        # The buffers hold soft_contact_max rows, about a million under the scrub example's towel: copied
        # whole, a read took 1.7 ms, sliced to the 11,522 live ones 0.13 ms (p8_cost.py, GPU 1).
        from unittest import mock

        import warp as wp

        solver = _coupler([((0,), 1, (0.0, 0.0, 2.0))], capacity=64)
        copied, numpy = [], wp.array.numpy

        def counting(array):
            copied.append(int(array.shape[0]))
            return numpy(array)

        with mock.patch.object(wp.array, "numpy", counting):
            corners, shape, force = solver.soft_contact_records()
        self.assertEqual(1, max(copied), f"rows copied, per array: {copied}")
        np.testing.assert_allclose(force, [[0, 0, 2]], atol=1e-6)
        corners, shape, force = _coupler([], capacity=64).soft_contact_records()
        self.assertEqual(((0, 3), (0,), (0, 3)), (corners.shape, shape.shape, force.shape), "none live")

    def test_a_coupler_staged_without_forces_says_how_to_stage_them(self) -> None:
        solver = _coupler([])
        solver.observables = None
        with self.assertRaisesRegex(RuntimeError, r"contact_forces=True"):
            solver.shape_contact_forces()


def _scene(solver, *, stain=(10, 11, 12, 13)):
    """The least of a sim that :class:`ForceGatedWipe` reads: a hand of shapes 1 and 2 on bodies 5 and 6."""
    robot = SimpleNamespace(arms={"right": None}, gripper_bodies=lambda arm: [5, 6])
    model = SimpleNamespace(
        shape_body=SimpleNamespace(numpy=lambda: np.array([-1, 5, 6, 7])), shape_label=["table", "a", "b", "c"]
    )
    towel = SimpleNamespace(entity=SimpleNamespace(sets={"towel/stain": np.asarray(stain)}))
    return SimpleNamespace(
        scene=SimpleNamespace(embodiment=robot),
        model=model,
        pipeline=SimpleNamespace(program=SimpleNamespace(solver=solver)),
        towel=towel,
    )


class TestTheSlideFeedback(unittest.TestCase):
    """``slide_feedback``: each record's push along the slide joint of the body it is on, as a pair across it.

    A hand (body 0) carries a finger (body 1) on a slide whose axis is the finger's local x; the finger is
    turned a quarter about z, so the slide runs along world y. A record on the finger pushing (3, 4, 5) N
    gives the finger 4 N along y and the hand the opposite at the finger's centre of mass; a record on the
    hand, which no slide carries, and a record past the live count give nothing.
    """

    def test_the_finger_gets_the_push_along_its_slide_and_the_hand_the_reaction(self) -> None:
        import warp as wp

        from robosimstudio.scene.physics.couplers import _slide_feedback

        def a(values, dtype):
            return wp.array(values, dtype=dtype, device="cpu")

        rigid, quarter = 2, (0.0, 0.0, float(np.sin(np.pi / 4)), float(np.cos(np.pi / 4)))
        forces = np.zeros((rigid + 3, 6), dtype=np.float32)
        forces[rigid] = (3.0, 4.0, 5.0, 9.0, 9.0, 9.0)
        forces[rigid + 1] = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        forces[rigid + 2] = (7.0, 7.0, 7.0, 0.0, 0.0, 0.0)
        body_f = wp.zeros(2, dtype=wp.spatial_vector, device="cpu")
        wp.launch(
            _slide_feedback,
            dim=3,
            inputs=[
                a([2], wp.int32),  # live records: the third is past the count
                a([1, 0, 1], wp.int32),  # each record's shape
                rigid,
                a(forces, wp.spatial_vector),
                a([0, 1], wp.int32),  # shape 0 is on the hand, shape 1 on the finger
                a([-1, 1], wp.int32),  # body 1 rides joint 1, a slide; body 0 on none
                a([-1, 0], wp.int32),  # joint 1's parent is the hand
                a([(0, 0, 0, 0, 0, 0, 1)] * 2, wp.transform),
                a([(1.0, 0.0, 0.0)] * 2, wp.vec3),
                a([0, 1], wp.int32),
                a([(0, 0, 0, 0, 0, 0, 1), (0.1, 0.0, 0.0, *quarter)], wp.transform),
                a([(0.0, 0.0, 0.0)] * 2, wp.vec3),
            ],
            outputs=[body_f],
            device="cpu",
        )
        hand, finger = body_f.numpy()
        np.testing.assert_allclose(finger, (0.0, 4.0, 0.0, 0.0, 0.0, 0.0), atol=1e-5)
        # The reaction at the finger's centre, 0.1 m along x from the hand's: no net force, no net moment.
        np.testing.assert_allclose(hand, (0.0, -4.0, 0.0, 0.0, 0.0, -0.4), atol=1e-5)


class TestTheSoftSolverIsAskedForForcesOnlyWhenTheyAreRead(unittest.TestCase):
    """A deformable solver whose ``step`` predates Newton 1.7 takes no ``observables``: it is passed only when
    the coupler reads contact forces."""

    def test_no_keyword_without_forces_and_the_forces_with_them(self) -> None:
        coupler = _coupler([])
        self.assertEqual({"observables": coupler.observables}, coupler._observing())
        coupler.observables = None
        self.assertEqual({}, coupler._observing())


class TestAResetForgetsTheLastPushes(unittest.TestCase):
    """The coupler's reset empties the live rigid-soft records, which the next substep would otherwise feed back."""

    def test_a_reset_reads_no_force_and_still_resets_both_solvers(self) -> None:
        coupler = _coupler([((10,), 1, (0.0, 0.0, 12.0)), ((11, 12), 2, (0.0, 3.0, 0.0))])
        self.assertGreater(float(np.abs(coupler.shape_contact_forces()).sum()), 0.0)
        calls = []
        coupler.rigid = SimpleNamespace(reset=lambda state, world_mask=None: calls.append("rigid"))
        coupler.soft = SimpleNamespace(reset=lambda state, world_mask=None: calls.append("soft"))
        coupler.reset(None)
        np.testing.assert_array_equal(coupler.shape_contact_forces(), 0.0)
        self.assertEqual(len(coupler.soft_contact_records()[1]), 0)
        self.assertEqual(calls, ["rigid", "soft"])


class _Sim(SimpleNamespace):
    def __getitem__(self, name):
        return getattr(self, name)


class TestTheWipeIsGatedOnThePush(unittest.TestCase):
    def _wipe(self, records):
        from robosimstudio_materials.cloth import ForceGatedWipe

        sim = _Sim(**vars(_scene(_coupler(records, capacity=8))))
        return ForceGatedWipe(sim, "towel", "stain", "right", (20.0, 60.0), rub=0.0)  # every touch counts

    def test_the_hand_is_its_tool_link_and_fingers_and_the_bench_is_found_by_name(self) -> None:
        from robosimstudio_materials.cloth import contact_shapes

        sim = _Sim(**vars(_scene(_coupler([]))))
        np.testing.assert_array_equal(contact_shapes(sim, "right"), [1, 2])
        np.testing.assert_array_equal(contact_shapes(sim, "table"), [0])
        with self.assertRaises(KeyError):
            contact_shapes(sim, "nothing")

    def test_a_push_under_the_band_wipes_nothing(self) -> None:
        wipe = self._wipe([((10,), 1, (0.0, 0.0, 12.0))])
        reading = wipe.update()
        self.assertAlmostEqual(reading["push"], 12.0, places=4)
        self.assertEqual((reading["wiped"], reading["crushed"]), (0.0, 0.0))

    def test_a_push_inside_the_band_wipes_every_corner_it_touches(self) -> None:
        # 30 N on the hand from a particle record and a face record; the bench's record is not the hand's.
        wipe = self._wipe(
            [((10,), 1, (0.0, 0.0, 10.0)), ((11, 12, 99), 2, (0.0, 0.0, 20.0)), ((13,), 0, (0.0, 0.0, -40.0))]
        )
        reading = wipe.update()
        self.assertAlmostEqual(reading["push"], 30.0, places=4)
        np.testing.assert_array_equal(wipe.wiped, [True, True, True, False])
        self.assertEqual(reading["crushed"], 0.0)

    def test_a_near_miss_touches_nothing(self) -> None:
        wipe = self._wipe([((10,), 1, (0.0, 0.0, 30.0)), ((13,), 2, (0.0, 0.0, 0.0))])
        wipe.update()
        np.testing.assert_array_equal(wipe.wiped, [True, False, False, False])

    def test_one_step_of_a_jump_does_not_mark_a_stroke_too_hard(self) -> None:
        import warp as wp

        wipe = self._wipe([((10,), 1, (0.0, 0.0, 30.0))])
        rows = wipe.sim.pipeline.program.solver.observables.contact_f
        for push in (30.0, 30.0, 30.0, 500.0, 30.0):
            values = rows.numpy()
            values[3, 2] = push  # the first soft row, after the coupler's three rigid ones
            rows.assign(wp.array(values, dtype=wp.spatial_vector, device="cpu"))
            reading = wipe.update()
        self.assertAlmostEqual(reading["push"], 30.0, places=4)
        self.assertEqual(reading["crushed"], 0.0, "a step's jump is not a press")
        self.assertEqual(reading["wiped"], 0.25)
        self.assertEqual(max(wipe.raw), 500.0, "the raw readings keep it")

    def test_the_raw_readings_keep_the_last_ones_only(self) -> None:
        from robosimstudio_materials.cloth import ForceGatedWipe

        wipe = self._wipe([((10,), 1, (0.0, 0.0, 12.0))])
        for _ in range(ForceGatedWipe.KEEP + 5):
            wipe.update()
        self.assertEqual(ForceGatedWipe.KEEP, len(wipe.raw), "a page updates it every step, for as long as it is open")
        wipe.raw.append(-1.0)  # the newest goes in at the end and the oldest drops off the front
        self.assertEqual(-1.0, wipe.raw[-1])
        self.assertEqual(ForceGatedWipe.KEEP, len(wipe.raw))

    def test_lifting_a_hard_press_back_through_the_band_wipes_nothing(self) -> None:
        import warp as wp

        from robosimstudio_materials.cloth import ForceGatedWipe

        sim = _Sim(**vars(_scene(_coupler([((10,), 1, (0.0, 0.0, 90.0))], capacity=8))))
        wipe = ForceGatedWipe(sim, "towel", "stain", "right", (20.0, 60.0), window=1, rub=0.0)
        rows = sim.pipeline.program.solver.observables.contact_f

        def push(newtons: float) -> dict:  # the hand's one record, on the stain's first particle
            values = rows.numpy()
            values[3, 2] = newtons
            rows.assign(wp.array(values, dtype=wp.spatial_vector, device="cpu"))
            return wipe.update()

        reading = push(90.0)
        self.assertEqual((reading["wiped"], reading["crushed"]), (0.0, 0.25))
        reading = push(40.0)  # coming off, the push passes back through the band
        self.assertEqual((reading["wiped"], reading["crushed"]), (0.0, 0.25), "still the hard press")
        push(5.0)  # off: the press is over
        self.assertFalse(wipe.hard)
        reading = push(40.0)  # a fresh press inside the band wipes
        self.assertEqual(reading["wiped"], 0.25)

    def test_only_a_hand_sliding_along_the_sheet_wipes(self) -> None:
        from robosimstudio_materials.cloth import ForceGatedWipe

        sim = _Sim(**vars(_scene(_coupler([((10,), 1, (0.0, 0.0, 30.0))], capacity=8))))
        body_q = np.zeros((8, 7), dtype=np.float32)
        sim.state = SimpleNamespace(body_q=SimpleNamespace(numpy=body_q.copy))
        wipe = ForceGatedWipe(sim, "towel", "stain", "right", (20.0, 60.0), window=1)
        np.testing.assert_array_equal(wipe.bodies, [5, 6])
        self.assertEqual(wipe.update()["wiped"], 0.0, "the first reading has nothing to compare with")
        body_q[[5, 6], 2] += 0.01  # straight up 10 mm: a lift, not a rub
        self.assertEqual(wipe.update()["wiped"], 0.0)
        body_q[[5, 6], 0] += 0.0001  # 0.1 mm along: under the default 0.2
        self.assertEqual(wipe.update()["wiped"], 0.0)
        body_q[[5, 6], 0] += 0.0005
        self.assertEqual(wipe.update()["wiped"], 0.25)

    def test_a_scene_reset_brings_the_stain_back_and_the_jump_home_is_no_rub(self) -> None:
        from robosimstudio_materials.cloth import ForceGatedWipe

        sim = _Sim(**vars(_scene(_coupler([((10,), 1, (0.0, 0.0, 30.0))], capacity=8))), step_count=40)
        body_q = np.zeros((8, 7), dtype=np.float32)
        sim.state = SimpleNamespace(body_q=SimpleNamespace(numpy=body_q.copy))
        wipe = ForceGatedWipe(sim, "towel", "stain", "right", (20.0, 60.0), window=1)
        wipe.update()

        def stroke(along: float, step: int) -> float:
            body_q[[5, 6], 0] += along
            sim.step_count = step
            return wipe.update()["wiped"]

        self.assertEqual(stroke(0.001, 41), 0.25)
        # The page's Reset: the count back to 0, the hand 50 mm away at home, the record still on the stain.
        self.assertEqual(stroke(0.05, 0), 0.0, "the towel a reset puts back is clean, and the jump wipes nothing")
        self.assertEqual(stroke(0.001, 1), 0.25, "a stroke after it wipes again")
        wipe.reset()  # Clean stain
        self.assertEqual(stroke(0.05, 2), 0.0, "the jump after Clean stain is no rub either")

    def test_a_push_past_the_band_is_too_hard_and_wipes_nothing(self) -> None:
        wipe = self._wipe([((10, 11), 1, (0.0, 0.0, 90.0))])
        reading = wipe.update()
        self.assertEqual(reading["wiped"], 0.0)
        self.assertEqual(reading["crushed"], 0.5)
        wipe.reset()
        self.assertEqual(wipe.readout()["crushed"], 0.0)


if __name__ == "__main__":
    unittest.main()

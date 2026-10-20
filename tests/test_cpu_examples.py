# SPDX-License-Identifier: Apache-2.0
"""Every material example and scored variant on a CPU: it builds there, or it is refused by name as it builds.

The examples a reader runs first for each kind -- one script a kind in
``examples/03_materials``, the scored variants in ``examples/06_tasks`` and the
suction cup in ``examples/04_extending`` -- run here as a reader runs them, with
``device`` left as the script leaves it, on a machine whose GPUs are hidden
(``scripts/testrun.py --tier cpu``). Each has one verdict in :data:`VERDICTS`:

* **Built**: on the CPU, with every island on MuJoCo's C library or SuperDex
  and the example's own thing on the one :data:`BUILT` names.
* **Refused**: the compiler names the thing and its kind and says it needs a
  GPU, before an entity is made (:data:`REFUSED`).
* **Needs CUDA**: the capsule chain's Dahl memory and twist joint are VBD's, so
  those two examples build with ``device="cuda"`` and a machine without one
  refuses them by the device's name (:data:`NEEDS_CUDA`).

A script added to ``03_materials`` or a scored variant added to ``06_tasks``
without a verdict fails :meth:`TestEveryExampleHasAVerdict.test_the_folders_and_the_table_agree`.
What a built kind does once stepped is :mod:`tests.test_cpu_scenes`'s and
:mod:`tests.test_cpu_materials`'s. Run by hand on a GPU box without the runner,
the module sets ``ROBOSIMSTUDIO_DEVICE=cpu`` for itself, as they do.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import traceback
import unittest
import unittest.mock

TIER = "cpu"
"""No GPU, real scenes; the SuperDex ones need the ``[cpu]`` extra. About 30 s warm, 26 of it the two cutting builds."""

HAS_SUPERDEX = importlib.util.find_spec("superdex") is not None

BUILT = {
    "03_materials/00_rope.py": ("cable", "superdex"),
    "03_materials/03_knot.py": ("rope", "superdex"),
    "03_materials/04_band.py": ("band", "superdex"),
    "03_materials/06_bag.py": ("bag", "superdex"),
    "03_materials/10_cutting.py": ("tofu", "mujoco"),
    "04_extending/02_suction.py": ("plate", "mujoco"),
    "06_tasks/05_suction_scored.py": ("plate", "mujoco"),
    "06_tasks/06_bag_scored.py": ("bag", "superdex"),
    "06_tasks/07_band_scored.py": ("band", "superdex"),
    "06_tasks/09_cutting_scored.py": ("tofu", "mujoco"),
}
"""Example -> (its thing, the engine that steps it on a CPU)."""

REFUSED = {
    "03_materials/05_paper.py": ("paper", "paper"),
    "03_materials/07_brush.py": ("brush", "brush"),
    "03_materials/08_tape.py": ("label", "tape"),
    "03_materials/09_adhesion.py": ("label", "tape"),
    "03_materials/11_granular.py": ("rice", "granular"),
    "03_materials/12_liquid.py": ("honey", "liquid"),
    "03_materials/13_dough.py": ("dough", "dough"),
    "03_materials/14_water.py": ("water", "water"),
    "06_tasks/03_granular_scored.py": ("rice", "granular"),
    "06_tasks/04_brush_scored.py": ("brush", "brush"),
    "06_tasks/08_tape_scored.py": ("label", "tape"),
    "06_tasks/10_water_scored.py": ("water", "water"),
}
"""Example -> (the thing the refusal names, its kind)."""

NEEDS_CUDA = ("03_materials/01_rope_memory.py", "03_materials/02_twist.py")
"""The rope examples whose physics is the capsule chain's on VBD, built with ``device="cuda"``."""

VERDICTS = {**dict.fromkeys(BUILT, "built"), **dict.fromkeys(REFUSED, "refused"), **dict.fromkeys(NEEDS_CUDA, "cuda")}
"""Every example this module holds to the CPU story, and which way it goes there."""

_hidden = unittest.mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_DEVICE": "cpu"})
_device_before: list = []


def setUpModule() -> None:
    """Hide the GPU by the rule's own variable when the runner did not hide it by the driver's."""
    import warp as wp

    _device_before.append(wp.get_device())
    if wp.is_cuda_available():
        _hidden.start()


def tearDownModule() -> None:
    import warp as wp

    with contextlib.suppress(RuntimeError):  # never started
        _hidden.stop()
    if _device_before:
        wp.set_device(_device_before.pop())


def _cuda_visible() -> bool:
    import warp as wp

    return bool(wp.is_cuda_available())


NAMED_DEVICE_WINS = "a device named in code wins over ROBOSIMSTUDIO_DEVICE: hide the GPUs to run this"


class TestEveryExampleHasAVerdict(unittest.TestCase):
    def test_the_folders_and_the_table_agree(self) -> None:
        from tests.fixtures.examples import EXAMPLES

        shipped = {
            *(f"03_materials/{p.name}" for p in (EXAMPLES / "03_materials").glob("*.py")),
            *(f"06_tasks/{p.name}" for p in (EXAMPLES / "06_tasks").glob("*_scored.py")),
            "04_extending/02_suction.py",
        }
        self.assertEqual(sorted(shipped - set(VERDICTS)), [], "a script with no CPU verdict")
        self.assertEqual(sorted(set(VERDICTS) - shipped), [], "a verdict for a script that is not there")
        self.assertEqual(len(VERDICTS), len(BUILT) + len(REFUSED) + len(NEEDS_CUDA), "a script with two verdicts")


class TestEachExampleOnACpu(unittest.TestCase):
    """Each example's ``main()`` against a studio that is not there, on a CPU."""

    def test_the_built_ones_are_on_the_cpu_engines(self) -> None:
        from tests.fixtures.examples import _run

        for rel, (thing, engine) in BUILT.items():
            if engine == "superdex" and not HAS_SUPERDEX:
                continue  # the [cpu] extra's; a base install is tests.test_cpu_scenes's
            with self.subTest(example=rel):
                sim, _ = _run(rel)
                try:
                    self.assertTrue(sim.platform.is_cpu, sim.platform)
                    program = sim.pipeline.program
                    engines = {island.backend.solver for island in program.islands}
                    self.assertLessEqual(engines, {"mujoco", "superdex"}, program.summary())
                    owner = next(island for island in program.islands if island.owns(thing))
                    self.assertEqual(owner.backend.solver, engine, program.summary())
                finally:
                    sim.close()

    def test_the_refused_ones_name_the_thing_and_its_kind(self) -> None:
        from robosimstudio.scene.physics import CompositionError
        from tests.fixtures.examples import _run

        for rel, (name, kind) in REFUSED.items():
            with self.subTest(example=rel):
                try:
                    _run(rel)
                except CompositionError as refused:  # by hand: assertRaises drops the traceback this reads
                    message, frames = str(refused), traceback.extract_tb(refused.__traceback__)
                else:
                    self.fail(f"{rel} built on a CPU")
                self.assertIn(f"'{name}' ({kind}) needs a GPU", message)
                extra = " with the [genesis] extra" if kind == "water" else ""
                self.assertTrue(message.endswith(f"Build it on a GPU{extra}."), message)
                self.assertEqual(frames[-1].name, "_for_platform", [f.name for f in frames])

    @unittest.skipIf(_cuda_visible(), NAMED_DEVICE_WINS)
    def test_the_capsule_chain_ones_are_refused_for_want_of_cuda(self) -> None:
        from tests.fixtures.examples import _run

        for rel in NEEDS_CUDA:
            with self.subTest(example=rel), self.assertRaises(ValueError) as caught:
                _run(rel)
            self.assertIn("device='cuda' (there is no CUDA device here)", str(caught.exception))


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""What lets the paper fold be done by two hands, by contact, each on its own side, read off the scene the example declares.

``examples/03_materials/05_paper`` folds an A4 sheet in half with both of the robot's hands. A parallel
jaw cannot get a pad under a sheet lying flat (seven top-down pinches raised one 0 mm), so the sheet lies
on a box with its near edge 35 mm past the box's edge, where each hand slides its lower pad under it
from its own corner. The sheet lies across the middle so the two corners are one in each hand's half;
on the bench edge the old layout hung it over, only the right arm reached and the left hand crossed
into its space to help. These checks hold that layout and its realism settings in place without
building anything: the overhang the pads close on, the sheet on the box and across the middle, the arms
that collide with each other, a grip capped at 15 N or less and fed the sheet's push, a pad friction
under 1, and nothing pinned to a hand.

The example's own ``main()`` runs, with ``Scene.build`` replaced: what it declared is
:attr:`Scene.request`, and what it then does with the built sim and on its page --
including inside the callbacks the page registers, each run once -- is recorded
against a stand-in that remembers every name it reached for or wrote.
"""

from __future__ import annotations

import inspect
import math
import re
import unittest
from collections.abc import Callable
from unittest import mock

import numpy as np

from tests.fixtures.examples import _load, _run, scripts_module

TIER = "fast"
"""Runs the example's ``main()`` with the build replaced: nothing is built."""

EXAMPLE = "03_materials/05_paper.py"

ATTACHMENTS = {"grab_set_requested", "follow_arm", "particle_sets"}
"""Names that are attachments whole."""
ATTACHMENT_WORDS = {"pin", "pins", "pinned", "attach", "attached", "attachment", "attachments"}
"""Words that make a name one: matched as ``_``-separated words, so ``stepping`` or ``mapping`` is not ``pin``."""


class _Log:
    """What the stand-ins saw: every name reached or written, and every callback handed to them."""

    def __init__(self) -> None:
        self.seen: list[str] = []
        self.callbacks: list[tuple[str, Callable]] = []

    def run_callbacks(self, limit: int = 100) -> int:
        """Run every registered callback once, and any they register in turn; how many ran.

        A callback's required arguments are stand-ins too (a button handler's
        event). It has to run to its end: a write after a line the stand-in
        cannot answer would otherwise go unseen, so the exception is the test's.
        """
        ran = 0
        while ran < len(self.callbacks):
            if ran == limit:
                raise AssertionError(f"more than {limit} callbacks registered: the page keeps registering more")
            path, fn = self.callbacks[ran]
            required = [
                p
                for p in inspect.signature(fn).parameters.values()
                if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            fn(*(_Recorder(f"{path}:{p.name}", self) for p in required))
            ran += 1
        return ran


class _Recorder:
    """Whatever ``main()`` reaches for on the built sim or the studio, remembering every name it reached or wrote.

    Arrays read back to the host are empty, so the example's numpy over them runs; a
    number read off one rounds to 0; a callable handed to one (``each_step``, a
    button's ``on_click``) is kept, for :meth:`_Log.run_callbacks` to run.
    """

    def __init__(self, path: str, log: _Log) -> None:
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_log", log)

    def __getattr__(self, name: str) -> _Recorder:
        if name.startswith("__"):
            raise AttributeError(name)
        self._log.seen.append(f"{self._path}.{name}")
        return _Recorder(f"{self._path}.{name}", self._log)

    def __setattr__(self, name: str, value) -> None:
        self._log.seen.append(f"{self._path}.{name}=")

    def __call__(self, *args, **kwargs):
        for value in (*args, *kwargs.values()):
            if callable(value) and not isinstance(value, _Recorder):
                self._log.callbacks.append((self._path, value))
        return np.zeros(0) if self._path.endswith(".numpy") else _Recorder(f"{self._path}()", self._log)

    def __getitem__(self, key) -> _Recorder:
        return _Recorder(f"{self._path}[{key!r}]", self._log)

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def __round__(self, ndigits=None) -> float:
        return 0.0

    def __enter__(self) -> _Recorder:
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _names(path: str) -> list[str]:
    """The attribute names along a recorded path: ``sim.model.shape_material_mu=`` -> ``[..., "shape_material_mu"]``."""
    return [re.sub(r"\(\)|\[.*?\]|=$|^.*:", "", part) for part in path.split(".")]


READOUTS = ("crease_angle", "cloth_fold")
"""The measures the page's readouts call each step: they read a real sheet, so they are stand-ins too."""


def _declared():
    """``main()`` and its page run against a build that builds nothing: ``(request, things, log)``.

    ``things`` maps a declared thing's name to its spec; ``log`` holds every name
    ``main()``, its page and the callbacks the page registered reached for or wrote.
    """
    import robosimstudio as rss

    log = _Log()
    scenes = []

    def build(scene, **kwargs):
        scenes.append(scene)
        return _Recorder("sim", log)

    module = _load(EXAMPLE)
    readouts = {name: _Recorder(name, log) for name in READOUTS}
    with mock.patch.object(rss.Scene, "build", build), mock.patch.multiple(module, **readouts):
        _, kwargs = _run(EXAMPLE)
        kwargs["page"](_Recorder("studio", log))
        log.run_callbacks()
    (scene,) = scenes
    request = scene.request
    return request, {spec.name: spec for spec in request.entities}, log


class TestTheFoldIsLaidOutForTwoHands(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.request, cls.things, cls.log = _declared()
        cls.names = [_names(path) for path in cls.log.seen]
        sheet, box = cls.things["paper"], cls.things["box"]
        cls.size, cls.corner = sheet.size, sheet.pose
        half, at = box.shape.half_extents, box.xform[:3]
        cls.box = tuple(2.0 * h for h in half)
        cls.box_near = at[0] - half[0]
        cls.box_top = at[2] + half[2]

    def test_the_pads_close_on_the_overhang_clear_of_the_box(self) -> None:
        takes = scripts_module("recording.takes")

        overhang = self.box_near - self.corner[0]
        far_side = takes.FOLD_PINCH + 0.006  # the 12 mm pads' far side, in from the edge
        self.assertGreaterEqual(
            overhang - far_side, 0.005, f"the pads end {far_side * 1e3:.0f} mm in, the box at {overhang * 1e3:.0f}"
        )
        self.assertLessEqual(overhang, 0.05, f"the edge runs {overhang * 1e3:.0f} mm past the box")

    def test_the_sheet_lies_on_the_box_across_the_middle(self) -> None:
        size, box, corner = self.size, self.box, self.corner
        overhang = self.box_near - corner[0]
        self.assertAlmostEqual(corner[2], self.box_top + 0.005, places=6, msg="laid 5 mm over the box's top")
        self.assertAlmostEqual(corner[1], -size[1] / 2, places=6, msg="one corner in each hand's half")
        self.assertGreaterEqual(box[0], size[0] - overhang, "the rest of the sheet is on the box")
        self.assertGreater(box[1], size[1], "and the box is wider than it")

    def test_the_arms_collide_with_each_other(self) -> None:
        from robosimstudio.robots.registry import get_embodiment

        self.assertTrue(get_embodiment("yam_bimanual").cross_arm_collisions, "the YAM's own setting")
        self.assertTrue(self.request.embodiment.cross_arm_collisions, "and the example keeps it")

    def test_the_grip_is_capped_and_feels_the_sheet(self) -> None:
        hands = self.request.gripper_actuation
        # One motor drives a hand's two fingers through the asset's joint equality: its cap bounds the driven
        # finger's slide, not either finger's contact push, which test_fold_paper_robot holds to 15 N.
        self.assertLessEqual(hands.effort_limit, 15.0, "the motor's cap")
        # Stiff enough that a finger held 0.2 mm off shut already asks the whole cap: Close stalls at the cap.
        self.assertGreaterEqual(hands.target_ke * 0.0002, hands.effort_limit)
        coupling = dict(self.request.physics.coupling.params)
        self.assertTrue(coupling["slide_feedback"], "the sheet's push reaches the fingers' slides")
        self.assertTrue(coupling["contact_forces"])

    def test_no_pad_grips_paper_harder_than_friction_one(self) -> None:
        """VBD mixes paper against a pad as the geometric mean: read the pads' friction off the hand's asset."""
        import xml.etree.ElementTree as ET

        from robosimstudio.paths import resolve_path

        asset = ET.parse(resolve_path("embodiments/yam/yam_bimanual_arms.xml"))
        pads = [float(g.get("friction").split()[0]) for g in asset.iter("geom") if g.get("friction")]
        self.assertTrue(pads)
        self.assertLessEqual(math.sqrt(self.things["paper"].material.friction * max(pads)), 1.0)
        friction = [".".join(names) for names in self.names if any("mu" in name.split("_") for name in names)]
        self.assertEqual([], friction, "and the example writes no friction of its own")

    def test_nothing_on_the_page_pins_the_sheet_to_a_hand(self) -> None:
        sheet = self.things["paper"]
        self.assertEqual((), sheet.particle_sets, "the fold is carried by contact; a particle set is an attachment's")
        self.assertEqual("", sheet.pinned)
        self.assertEqual({"box", "paper"}, set(self.things), "and nothing else is declared to hold it")
        for names in self.names:
            words = {word for name in names for word in name.split("_")}
            with self.subTest(".".join(names)):
                self.assertFalse(ATTACHMENTS & set(names), "the fold is carried by contact; that is an attachment's")
                self.assertFalse(ATTACHMENT_WORDS & words, "the fold is carried by contact; that is an attachment's")

    def test_the_pages_callbacks_ran(self) -> None:
        """The readouts' step callback ran against the stand-ins, so what it reaches is in what these checks read."""
        self.assertIn("studio.each_step", [path for path, _ in self.log.callbacks])
        self.assertIn("studio.gui.add_number().value=", self.log.seen, "the readout written each step")


if __name__ == "__main__":
    unittest.main()

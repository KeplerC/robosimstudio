# SPDX-License-Identifier: Apache-2.0
"""A label on a box beside a towel, the YAM over both: it stays stuck at rest.

``examples/19_tape/00_peel_a_label`` has nothing else made of particles in it. Here a towel is, and
what the label's own example cannot show is held:

* **Another thing's settle.** A towel settles for 60 frames, stepping the whole scene, label and box
  included, and a reset restores that snapshot while it puts the box back at its spawn pose. The
  label came back 1.5 to 1.7 mm inside the box top, sprang off and every one of its 144 bonds broke
  within 30 control steps, nothing touching it (``p1c.py``, GPU 1). A reset puts the sheet back where
  it was built instead, so it lies where its bonds hold it.
"""

from __future__ import annotations

import unittest

TIER = "medium"
"""One GPU: one scene built, about 11 s, and 60 control steps."""


def _build():
    import robosimstudio as rss

    scene = rss.Scene(cell="bench_0728", physics=rss.TAPE_60HZ, name="tape_beside_a_towel")
    scene.add(rss.robot("yam_bimanual"))
    towel = rss.sheet(
        name="towel", size=(0.1, 0.1), cells=(10, 10), at=(0.1, 0.1, 0.02), material=rss.materials.TERRY_TOWELLING
    )
    scene.add(towel)
    scene.add(rss.box(name="box", half=(0.05, 0.06, 0.025), at=(0.0, -0.08, 0.025), mass=0.5))
    scene.add(rss.tape(name="label", on="box", at=(0.02, -0.025, 0.0505), quat=(0.0, 0.0, 1.0, 0.0)))
    return scene.build(control_mode="joint_position")


class TestALabelBesideATowel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = sim = _build()
        label = cls.label = sim["label"].entity
        sim.run(30)
        cls.at_rest = (label.peeled(), label.held())
        sim.reset()
        sim.run(30)
        cls.after_reset = (label.peeled(), label.held())

    def test_the_towels_settle_leaves_the_label_stuck(self) -> None:
        self.assertEqual((0.0, len(self.label.bonds)), self.at_rest)
        self.assertEqual((0.0, len(self.label.bonds)), self.after_reset, "and so does every reset after it")


if __name__ == "__main__":
    unittest.main()

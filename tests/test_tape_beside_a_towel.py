# SPDX-License-Identifier: Apache-2.0
"""A label on a box beside a towel, the YAM over both: it stays stuck at rest, and a bond that let go stays gone.

``examples/19_tape/00_peel_a_label`` has nothing else made of particles in it. Here a towel is, and
two things the label's own example cannot show are held:

* **Another thing's settle.** A towel settles for 60 frames, stepping the whole scene, label and box
  included, and a reset restores that snapshot while it puts the box back at its spawn pose. The
  label came back 1.5 to 1.7 mm inside the box top, sprang off and every one of its 144 bonds broke
  within 30 control steps, nothing touching it (``p1c.py``, GPU 1). A reset puts the sheet back where
  it was built instead, so it lies where its bonds hold it.
* **A notify after a break.** With the YAM in the scene the label's VBD island holds its own copy of
  the attachment arrays, and Newton copies the model's over it on a ``CONSTRAINT_PROPERTIES`` notify.
  The break kernel cleared only the island's ``enabled``, so that notify stuck every broken bond back
  on: 6 of 6 in ``p2_probe.py``. It clears the model's too.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "medium"
"""One GPU: one scene built, about 11 s, and 61 control steps."""


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
        import newton

        from robosimstudio.core.tuning import notify

        cls.sim = sim = _build()
        label = cls.label = sim["label"].entity
        cls.view = view = sim.scene.program.view_for("label")
        sim.run(30)
        cls.at_rest = (label.peeled(), label.held())
        # The three bonds furthest from the fold, their particles lifted 5 mm: those let go on the next step.
        far = label.bond_particles[np.argsort(label.along)[-3:]]
        for state in (sim.pipeline.state_0, sim.pipeline.state_1):
            q = state.particle_q.numpy()
            q[far, 2] += 0.005
            state.particle_q.assign(q)
        sim.run(1)
        cls.gone = label.broken.numpy().astype(bool)
        notify(sim.pipeline, int(newton.ModelFlags.CONSTRAINT_PROPERTIES))
        cls.stuck_again = view.attachment_particle_body_enabled.numpy()[label.bonds][cls.gone]
        sim.reset()
        sim.run(30)
        cls.after_reset = (label.peeled(), label.held())

    def test_the_label_island_holds_its_own_attachments(self) -> None:
        self.assertIsNot(self.view, self.sim.model, "with the YAM the label steps on a coupled island's view")

    def test_the_towels_settle_leaves_the_label_stuck(self) -> None:
        self.assertEqual((0.0, len(self.label.bonds)), self.at_rest)
        self.assertEqual((0.0, len(self.label.bonds)), self.after_reset, "and so does every reset after it")

    def test_a_bond_that_let_go_stays_gone_through_a_constraint_notify(self) -> None:
        self.assertTrue(self.gone.any(), "lifting three particles 5 mm breaks their bonds")
        self.assertLess(int(self.gone.sum()), len(self.gone), "and not the whole label")
        self.assertFalse(self.stuck_again.any(), f"{int(self.stuck_again.sum())} broken bonds stuck back on")


if __name__ == "__main__":
    unittest.main()

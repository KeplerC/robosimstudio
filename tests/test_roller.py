# SPDX-License-Identifier: Apache-2.0
"""``rss.roller`` owns the handle the jaws close on, and refuses a handle it cannot build.

The roller is two bodies, the drum and the handle on its axle, and the handle's
four boxes are what a gripper touches: a contact measure between the hand and the
roller, the **contact** tunable and the segmentation all go through the shapes
the entity collects. Collected off the drum's body alone, the handle was in none
of them. The handle's shoulder stands ``grip`` minus 18 mm over the axle, so a
grip too low for the drum puts the shoulder inside it, or under the axle. No
device: a builder, the entity and a CPU model.
"""

from __future__ import annotations

import unittest

import newton
import numpy as np

from robosimstudio_materials.dough import EntityRoller, roller

TIER = "fast"
"""Builds one roller into a builder and finalizes it on the CPU."""


def _built(spec):
    entity = EntityRoller(spec)
    builder = newton.ModelBuilder()
    entity.build_into(builder, episode=None)
    model = builder.finalize(device="cpu")
    entity.bind(model)
    return entity, builder, model


class TestTheRollerOwnsItsHandle(unittest.TestCase):
    def test_the_handles_boxes_are_the_rollers_shapes(self) -> None:
        for spins in (True, False):
            with self.subTest(spins=spins):
                entity, builder, model = _built(roller(at=(0.0, 0.0, 0.02), spins=spins))
                labels = [builder.shape_label[int(s)] for s in entity.handle(model).collision_shapes]
                self.assertIn("roller", labels, "the drum")
                for k in range(4):
                    self.assertIn(f"roller_handle_{k}", labels, "what the jaws close on")
                self.assertEqual(len(entity.shapes), builder.shape_count)

    def test_the_contact_tunable_reaches_the_handle(self) -> None:
        entity, builder, _ = _built(roller(at=(0.0, 0.0, 0.02)))
        contact = next(t for t in entity.tunables() if t.name == "contact")
        self.assertIn(f"{builder.shape_count} shape(s)", contact.summary)
        self.assertEqual(entity.body, int(np.asarray(builder.shape_body)[0]), "the pose is still the drum's")


class TestAHandleThatCannotBeBuiltIsRefused(unittest.TestCase):
    def test_the_defaults_build(self) -> None:
        roller()
        roller(radius=0.04, grip=0.07)

    def test_a_grip_that_puts_the_shoulder_under_the_axle(self) -> None:
        with self.assertRaisesRegex(ValueError, "grip"):
            roller(grip=0.015)

    def test_a_drum_that_buries_the_shoulder(self) -> None:
        with self.assertRaisesRegex(ValueError, "grip"):
            roller(radius=0.04)


if __name__ == "__main__":
    unittest.main()

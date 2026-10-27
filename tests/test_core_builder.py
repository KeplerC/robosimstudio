# SPDX-License-Identifier: Apache-2.0
"""The model-builder helpers in ``core.builder``: substep parity, attribute re-keying, attribute writes.

An odd substep count leaves the frame in the wrong buffer; ``collapse_fixed_joints``
renumbers bodies and leaves a BODY-frequency attribute on the wrong ones unless it
is re-keyed; ``set_body_attribute`` counts what it wrote.
"""

from __future__ import annotations

import types
import unittest

import newton

from robosimstudio.core.builder import (
    assert_even_substeps,
    rekey_custom_attributes,
    restate_body_attribute,
)

TIER = "fast"
"""No GPU: a ``newton.ModelBuilder`` that is never finalized, and stand-ins."""


class TestSubstepGuard(unittest.TestCase):
    """An odd substep count leaves the frame in the wrong buffer."""

    def test_odd_substeps_raise(self):
        """Odd substep counts are rejected."""
        for n in (1, 3, 7):
            with self.assertRaises(ValueError):
                assert_even_substeps(n)

    def test_even_substeps_pass(self):
        """Even substep counts are returned unchanged."""
        self.assertEqual(assert_even_substeps(20), 20)

    def test_zero_and_negative_raise(self):
        """A non-positive substep count is rejected before the parity check."""
        for n in (0, -2):
            with self.assertRaises(ValueError):
                assert_even_substeps(n)


class TestRekeyCustomAttributes(unittest.TestCase):
    """The collapse_fixed_joints BODY-attribute bug, and its fix.

    ``collapse_fixed_joints`` renumbers bodies and re-keys only attributes that
    *reference* an entity; a BODY-frequency attribute keeps its pre-collapse
    keys and its values land on whichever body now holds that index. On a real
    arm that silently removes gravity compensation from half the links.
    """

    def _builder_with_gravcomp(self, values: dict[int, float]) -> newton.ModelBuilder:
        builder = newton.ModelBuilder()
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="gravcomp",
                dtype=float,
                frequency=newton.Model.AttributeFrequency.BODY,
                namespace="mujoco",
                default=0.0,
                values=dict(values),
            )
        )
        return builder

    def test_body_frequency_values_are_rekeyed(self):
        """Values follow their body through the collapse's renumbering."""
        builder = self._builder_with_gravcomp({0: 1.0, 3: 1.0, 7: 1.0})
        results = {"body_remap": {0: 0, 3: 1, 7: 2}, "joint_remap": {}, "body_merged_parent": {}}
        counts = rekey_custom_attributes(builder, results)
        self.assertEqual(counts["mujoco:gravcomp"], 3)
        self.assertEqual(builder.custom_attributes["mujoco:gravcomp"].values, {0: 1.0, 1: 1.0, 2: 1.0})

    def test_merged_body_does_not_displace_its_parent(self):
        """A body merged away hands its value up only if the parent has none."""
        builder = self._builder_with_gravcomp({0: 1.0, 1: 9.0, 2: 5.0})
        # Body 1 merges into body 0, which survives as index 0 and has its own
        # value; body 2 merges into a surviving body 3 that declares none.
        results = {
            "body_remap": {0: 0, 3: 1},
            "joint_remap": {},
            "body_merged_parent": {1: 0, 2: 3},
        }
        rekey_custom_attributes(builder, results)
        values = builder.custom_attributes["mujoco:gravcomp"].values
        self.assertEqual(values[0], 1.0, "the surviving body's own value must win over a merged child's")
        self.assertEqual(values[1], 5.0, "a merged child hands its value to a parent that declares none")

    def test_joint_frequency_values_are_rekeyed(self):
        """A JOINT-frequency attribute follows the joint remap, dropping removals."""
        builder = newton.ModelBuilder()
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="stiffness",
                dtype=float,
                frequency=newton.Model.AttributeFrequency.JOINT,
                namespace="test",
                default=0.0,
                values={0: 1.0, 4: 2.0, 9: 3.0},
            )
        )
        rekey_custom_attributes(builder, {"body_remap": {}, "joint_remap": {0: 0, 9: 1}, "body_merged_parent": {}})
        self.assertEqual(builder.custom_attributes["test:stiffness"].values, {0: 1.0, 1: 3.0})

    def test_restate_writes_every_body(self):
        """The blanket fallback sets the attribute on all surviving bodies."""
        builder = self._builder_with_gravcomp({0: 1.0})
        for _ in range(4):
            builder.add_body()
        written = restate_body_attribute(builder, "mujoco:gravcomp", 1.0)
        self.assertEqual(written, builder.body_count)
        self.assertEqual(set(builder.custom_attributes["mujoco:gravcomp"].values), set(range(builder.body_count)))

    def test_restate_on_a_missing_attribute_is_a_no_op(self):
        """Restating an attribute the import never declared writes nothing."""
        self.assertEqual(restate_body_attribute(newton.ModelBuilder(), "mujoco:gravcomp", 1.0), 0)


class TestSetBodyAttribute(unittest.TestCase):
    def test_set_body_attribute_counts_a_generator(self) -> None:
        from robosimstudio.core.builder import set_body_attribute

        attr = types.SimpleNamespace(values={})
        builder = types.SimpleNamespace(custom_attributes={"k": attr})
        self.assertEqual(set_body_attribute(builder, "k", 1.0, (b for b in (3, 4))), 2)
        self.assertEqual(attr.values, {3: 1.0, 4: 1.0})


if __name__ == "__main__":
    unittest.main()

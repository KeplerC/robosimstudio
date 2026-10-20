# SPDX-License-Identifier: Apache-2.0
"""A model with a body and no joint finalizes into one VBD can reset.

A static box was a body with no joint (a bare ``add_link``) until it was welded to the world
(``robosimstudio.domains.static.welded.weld_to_world``), so a sheet and a static box with no robot
made a model with one body and none; a bare ``add_link`` still does, and this builds one. Newton
1.7 builds ``joint_dof_dim`` as ``wp.array(np.array([]), ndim=2)``, which warp turns into a
one-dimensional empty array, and VBD's ``reset`` launches ``reset_rigid_state`` with it whenever it
has a body to integrate: the scene failed at its first reset, "argument 'joint_dof_dim' expects an
array with 2 dimension(s)", with the sheet on VBD or on SuperDex alike (``c5b_joints.py``). The
compiler's finalize keeps it two-dimensional. Here on the CPU, with no scene: a box on a bare link
alone, finalized both ways, and VBD built over it and reset. ``tests/test_superdex_sheet.py``
builds the sheet over a box.
"""

from __future__ import annotations

import unittest

TIER = "fast"
"""One body on the CPU and a VBD solver over it, in about a second."""


def _a_jointless_body():
    """A box on a bare ``add_link``: one body and no joint, which a static box no longer is (it is welded)."""
    import newton
    import warp as wp

    builder = newton.ModelBuilder()
    body = builder.add_link(xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()), label="box")
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder.add_ground_plane()
    builder.color()
    return builder


class TestAModelWithoutJoints(unittest.TestCase):
    def test_newton_alone_hands_vbd_a_one_dimensional_joint_dof_dim(self) -> None:
        """What the compiler works around: if Newton fixes it, this fails and the workaround can go."""
        model = _a_jointless_body().finalize(device="cpu")
        self.assertEqual((1, 0, 1), (model.body_count, model.joint_count, model.joint_dof_dim.ndim))

    def test_the_compilers_finalize_resets_on_vbd(self) -> None:
        import newton
        import warp as wp

        from robosimstudio.scene.compiler import _finalize

        builder = _a_jointless_body()
        with wp.ScopedDevice("cpu"):
            model = _finalize(builder)
        self.assertEqual((0, 2), tuple(model.joint_dof_dim.shape))
        solver = newton.solvers.SolverVBD(model)
        solver.reset(model.state())  # raised "expects an array with 2 dimension(s)"


if __name__ == "__main__":
    unittest.main()

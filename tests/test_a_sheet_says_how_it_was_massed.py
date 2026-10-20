# SPDX-License-Identifier: Apache-2.0
"""A sheet's actor states how Newton massed it in a word the Genesis mass check reads.

``HostActor.lumping`` is ``"grid"`` (``add_cloth_grid``) or ``"mesh"`` (``add_cloth_mesh``), and the Genesis
bridge weighs a sheet Newton masses only in part by it (``robosimstudio.engines.genesis.bridge._weighed_share``).
A way it went quiet, here on the CPU with no engine:

* **A word it does not know.** ``"Mesh"`` was declared as readily as ``"mesh"``, and the bridge, reading it as
  saying nothing, let a record of twice the built density through: it is refused at the declaration now.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "fast"
"""Specs, a Newton model on the CPU and the Genesis bridge's checks before ``gs.init``, under a second."""

BRIDGE = "robosimstudio.engines.genesis.bridge"
"""The logger the bridge's mass check speaks on."""


def _rim_held_mesh():
    """An 11 x 11 ``add_cloth_mesh`` sheet 200 mm square at 0.3 kg/m^2, its 40 rim particles massless as a hand round
    it leaves them: Newton weighs it only in part, so the check needs its lumping."""
    import newton
    import warp as wp

    n, side = 11, 0.2
    ticks = np.linspace(0.0, side, n)
    vertices = [wp.vec3(x, y, 0.0) for y in ticks for x in ticks]
    cells = [(a, a + 1, a + n + 1, a + n) for a in (j * n + i for j in range(n - 1) for i in range(n - 1))]
    faces = [v for a, b, c, d in cells for v in (a, b, c, a, c, d)]
    builder = newton.ModelBuilder()
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0), vertices=vertices, indices=faces,
        density=0.3,
    )  # fmt: skip
    model = builder.finalize(device="cpu")
    mass = model.particle_mass.numpy()
    rest = model.particle_q.numpy()
    edge = 1.0e-6
    rim = (rest[:, 0] < edge) | (rest[:, 0] > side - edge) | (rest[:, 1] < edge) | (rest[:, 1] > side - edge)
    mass[rim] = 0.0
    model.particle_mass.assign(mass)
    return model


def _open(model, records: list[dict]) -> None:
    """The Genesis bridge over ``records``, stopped where it would first ask for the engine."""
    from unittest import mock

    from robosimstudio.engines.genesis import bridge
    from robosimstudio.engines.genesis.solver import SolverGenesis

    recipe = {"island": "cloth", "actors": records, "mirrors": []}
    with mock.patch.object(bridge, "ensure_initialized", side_effect=LookupError("asked for the engine")):
        SolverGenesis(model, recipe=recipe, substeps=2)


class TestALumpingIsAWordTheReaderKnows(unittest.TestCase):
    def test_grid_mesh_and_none_are_declared(self) -> None:
        from robosimstudio.specs.physics import HostActor

        for lumping in ("grid", "mesh", None):
            with self.subTest(lumping=lumping):
                record = HostActor(kind="shell", lumping=lumping).record(name="s", particle_start=0, particle_count=4)
                self.assertEqual(lumping, record.get("lumping"))

    def test_any_other_word_is_refused_at_the_declaration(self) -> None:
        """The bridge reads ``"Mesh"`` as saying nothing: a record stating it and twice the sheet's built density
        reached the engine unweighed, where ``"mesh"`` is refused at ratio 2.000."""
        from robosimstudio.specs.physics import HostActor

        model = _rim_held_mesh()
        record = {"kind": "shell", "name": "sheet", "particle_start": 0, "particle_count": 121}
        twice = {**record, "params": {"areal_density": 0.6}}
        with self.assertRaisesRegex(ValueError, "ratio 2.000"):
            _open(model, [{**twice, "lumping": "mesh"}])
        with self.assertRaisesRegex(LookupError, "asked for the engine"):
            _open(model, [{**twice, "lumping": "Mesh"}])
        for word in ("Mesh", "cloth_mesh", "", "grid "):
            with self.subTest(lumping=word):
                with self.assertRaises(ValueError) as caught:
                    HostActor(kind="shell", lumping=word)
                self.assertIn(repr(word), str(caught.exception))


if __name__ == "__main__":
    unittest.main()

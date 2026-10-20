# SPDX-License-Identifier: Apache-2.0
"""A sheet's actor states how Newton massed it, and the Genesis mass check is never silent about one that does not.

``HostActor.lumping`` is ``"grid"`` (``add_cloth_grid``) or ``"mesh"`` (``add_cloth_mesh``), and the Genesis
bridge weighs a sheet Newton masses only in part by it (``robosimstudio.engines.genesis.bridge._weighed_share``).
Three ways it went quiet, each here on the CPU with no engine:

* **A word it does not know.** ``"Mesh"`` was declared as readily as ``"mesh"``, and the bridge, reading it as
  saying nothing, let a record of twice the built density through: it is refused at the declaration now.
* **A record that says nothing**, the composer's default for an entity declaring no actor and a record replayed
  from before records stated it: the sheet was not weighed and only a debug line said so. A warning says so now,
  once an actor a build.
* **A bag**, read as a shell from its name, range and params alone: its ``"mesh"`` reached no reader.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "fast"
"""Specs, three Newton models on the CPU and the Genesis bridge's checks before ``gs.init``, under a second."""

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
        with self.assertRaisesRegex(LookupError, "asked for the engine"), self.assertLogs(BRIDGE, "WARNING"):
            _open(model, [{**twice, "lumping": "Mesh"}])
        for word in ("Mesh", "cloth_mesh", "", "grid "):
            with self.subTest(lumping=word):
                with self.assertRaises(ValueError) as caught:
                    HostActor(kind="shell", lumping=word)
                self.assertIn(repr(word), str(caught.exception))


class TestASheetNotWeighedIsWarnedOf(unittest.TestCase):
    def test_once_an_actor_a_build_at_warning(self) -> None:
        """Two sheets whose records state no lumping, one at twice its density: neither weighed, each named once at
        WARNING, and once again when the island is built again. It was a debug line, which no default log shows."""
        model = _rim_held_mesh()
        record = {"kind": "shell", "particle_start": 0, "particle_count": 121}
        records = [
            {**record, "name": "sheet", "params": {"areal_density": 0.3}},
            {**record, "name": "twice", "params": {"areal_density": 0.6}},
        ]
        for build in range(2):
            with self.subTest(build=build):
                with self.assertLogs(BRIDGE, "WARNING") as said:
                    with self.assertRaisesRegex(LookupError, "asked for the engine"):
                        _open(model, records)
                self.assertEqual(2, len(said.records))
                for line, name in zip(said.output, ("'sheet'", "'twice'"), strict=True):
                    for word in (name, "not weighed", "81 of its 121 particles", "no lumping (None)"):
                        self.assertIn(word, line)

    def test_a_sheet_newton_weighs_whole_is_weighed_without_one(self) -> None:
        """Every particle massed, the share is 1 whatever the builder: weighed, refused at twice, and nothing warned."""
        import newton
        import warp as wp

        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=4, dim_y=4, cell_x=0.05, cell_y=0.05,
            mass=0.3 * 0.04 / 25,
        )  # fmt: skip
        model = builder.finalize(device="cpu")
        record = {"kind": "shell", "name": "sheet", "particle_start": 0, "particle_count": 25}
        with self.assertNoLogs(BRIDGE, "WARNING"), self.assertRaisesRegex(LookupError, "asked for the engine"):
            _open(model, [{**record, "params": {"areal_density": 0.3}}])
        with self.assertRaisesRegex(ValueError, "ratio 2.000"):
            _open(model, [{**record, "params": {"areal_density": 0.6}}])


class TestABagsLumpingReachesItsReader(unittest.TestCase):
    def test_a_built_bag_reads_as_the_mesh_it_was_massed_as(self) -> None:
        """``EntityBag.build`` masses it with ``add_cloth_mesh`` and its record says ``"mesh"``; read back through the
        ``bag`` kind's reader it said ``None``."""
        import newton

        import robosimstudio as rss
        from robosimstudio.engines.geometry import actor_geometry
        from robosimstudio_materials.bag import EntityBagSuperdex

        entity = EntityBagSuperdex(rss.bag(name="pouch", size=(0.08, 0.06), cells=(8, 6)))
        builder = newton.ModelBuilder()
        entity.build_into(builder, None)
        model = builder.finalize(device="cpu")
        particles = entity.owned_particles()
        record = entity.host_actor().record(
            name="pouch", particle_start=int(particles[0]), particle_count=len(particles)
        )
        self.assertEqual(("bag", "mesh"), (record["kind"], record["lumping"]))
        geometry = actor_geometry(model, record)
        self.assertEqual("mesh", geometry.lumping)
        self.assertGreater(len(geometry.tris), 0, "still read as a shell: its triangles")
        self.assertIsNone(actor_geometry(model, {k: v for k, v in record.items() if k != "lumping"}).lumping)


if __name__ == "__main__":
    unittest.main()

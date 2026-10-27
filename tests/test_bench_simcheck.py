# SPDX-License-Identifier: Apache-2.0
"""``goals.simcheck``: whether a staged arrangement can serve every asset a document places.

Static and cheap: an asset's ``sim`` declaration against the lowered task's
arrangement, read off data. Each refusal names the instance and what it asked
for, and an asset nothing declares is the typechecker's to report, not this.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from robosimstudio_bench.goals.simcheck import sim_problems
from robosimstudio_bench.taskbuild.lower import lower
from tests.fixtures import taskdocs

TIER = "fast"
"""Lowered documents and a fake catalog: no scene, no GPU."""

TOWEL = taskdocs.DOCUMENT.replace(
    f"(:objects cube - {taskdocs.CUBE})", f"(:objects cube - {taskdocs.CUBE} towel - towel_probe)"
)
"""The cube document on its rigid-only arrangement, with a towel beside the cube."""


def _sim(domain: str, *, backend=None, problems=()):
    return SimpleNamespace(domain=domain, backend=backend, problems=lambda: list(problems))


class TestTheStagedArrangementServesTheAssets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with taskdocs.cube_asset():
            cls.doc = taskdocs.doc(TOWEL)
            cls.spec = lower(cls.doc, suite="t", name="t", resolve_artifact=lambda reference: None).spec

    def _problems(self, **assets) -> list[str]:
        catalog = taskdocs.FakeCatalog({name: taskdocs.Asset(sim=sim) for name, sim in assets.items()})
        return sim_problems(self.doc, self.spec, catalog)

    def test_a_cloth_on_a_rigid_engine_cannot_have_its_particles_owned(self):
        (problem,) = self._problems(towel_probe=_sim("cloth"))
        self.assertTrue(problem.startswith("towel is a cloth"), problem)
        self.assertIn("cannot own ['particles']", problem)

    def test_a_domain_the_arrangement_names_no_backend_for_is_refused_unless_static(self):
        rigid_only = SimpleNamespace(
            physics=SimpleNamespace(backend_for=lambda domain: None, domains=lambda: ("rigid",))
        )
        catalog = taskdocs.FakeCatalog({"towel_probe": taskdocs.Asset(sim=_sim("cloth"))})
        (problem,) = sim_problems(self.doc, rigid_only, catalog)
        self.assertIn("names no backend for that domain (it serves ['rigid'])", problem)
        catalog = taskdocs.FakeCatalog({"towel_probe": taskdocs.Asset(sim=_sim("static"))})
        self.assertEqual(sim_problems(self.doc, rigid_only, catalog), [])

    def test_the_records_own_problems_come_through_under_the_instance_name(self):
        self.assertEqual(self._problems(towel_probe=_sim("static", problems=["no mesh"])), ["towel: no mesh"])

    def test_an_asset_nothing_declares_is_left_to_the_typechecker(self):
        self.assertEqual(self._problems(), [])

    def test_a_record_without_a_sim_declaration_is_not_checked(self):
        self.assertEqual(self._problems(towel_probe=None), [])

    def test_a_backend_nobody_registered_is_named(self):
        (problem,) = self._problems(towel_probe=_sim("cloth", backend=SimpleNamespace(solver="no_such_solver")))
        self.assertTrue(problem.startswith("towel: "), problem)
        self.assertIn("no_such_solver", problem)

    def test_a_rigid_body_on_a_rigid_engine_is_served(self):
        self.assertEqual(self._problems(towel_probe=_sim("rigid")), [])


if __name__ == "__main__":
    unittest.main()

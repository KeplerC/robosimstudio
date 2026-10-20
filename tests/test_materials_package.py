# SPDX-License-Identifier: Apache-2.0
"""The shipped kinds are a sibling package, found by scan, and the library is whole without it.

``cloth``, ``soft``, the rope, ``paper``, the brush and the three particle kinds are the
modules of ``robosimstudio_materials``: ``cloth.py``, ``soft.py``, ``paper.py``, ``brush.py``,
``granular.py``, ``liquid.py`` and ``dough.py`` one file each, ``rod/`` a package, and ``mpm/`` the
implementation the particle kinds share. The library never imports that package. It names it as a string in two places -- the registries'
scan list (:data:`robosimstudio.roots.DOMAIN_PACKAGES`) and the flat-name table
(``robosimstudio._PACKS``) -- and everything else follows from the scan: a kind
registers itself as it is imported, so a fresh process that asks any kind
registry finds every one without anybody importing them by name. Five things
have to hold, and each fails silently when it does not:

* **A fresh process finds the kinds by scan**, and the scan imports no
  numbered example script (none lies in the package; the rule that skips one is
  tested on a third-party root in ``tests/test_registry_discovery``).
* **Each base kind stands alone**: none of ``cloth``, ``soft``, ``rod``,
  ``granular``, ``liquid`` and ``dough`` imports another at any scope, and the
  package imports nothing from the studio, the benchmark or the agents. With
  the other kinds left out of the package, one still registers and hands out
  its spec and its entity. ``paper``, ``bag`` and ``brush`` are the declared
  exceptions: a sheet of paper is a cloth whose hinges yield and a bag is a cloth
  sewn into a pocket, so each imports ``cloth`` and nothing else, and a brush's
  bristles are the rope's capsule chains, so it imports ``rod``; each stands with its
  base kind alone. The particle kinds share ``mpm/``, which
  is not a kind and stays when any of them is left out.
* **The library names the sibling only as a string.** An ``import
  robosimstudio_materials`` anywhere under ``robosimstudio/`` is the sibling
  law broken, and the day a tree without the package stops importing.
* **The wheel ships the package** like any other, and no copy of the kinds
  under another name (``_domain_packs`` was that copy).
* **Without the package, nothing is half there.** ``import robosimstudio``
  works, every scan answers, a rigid scene builds and steps on the CPU, and
  the names the package would have provided are ``AttributeError``s that say
  which package is missing.

``tests/test_rod_is_a_pack`` holds the rope's own, finer version of the last
two claims. Subprocesses throughout, because every one of these is a claim about
what a fresh *process* sees.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tomllib
import unittest
from contextlib import ExitStack
from pathlib import Path
from textwrap import dedent

from tests.fixtures.packs import MATERIALS, NEEDS, PYTHON_FLAGS, env_without

TIER = "fast"
"""No GPU. Imports the library in subprocesses; the one scene is built on the CPU."""

REPO = Path(__file__).resolve().parents[1]
LIBRARY = REPO / "robosimstudio"
SIBLINGS = ("robosimstudio_studio", "robosimstudio_bench", "robosimstudio_policy_agent", "robosimstudio_real2sim_agent")
"""The packages beside the library that the materials may not name."""

KINDS = {
    "cloth": ("sheet", "ClothSpec", "robosimstudio_materials.cloth", "cloth"),
    "soft": ("sponge", "SoftSpec", "robosimstudio_materials.soft", "soft"),
    "rod": ("rod", "RodSpec", "robosimstudio_materials.rod.entity", "rod"),
    "paper": ("paper", "PaperSpec", "robosimstudio_materials.paper", "paper"),
    "bag": ("bag", "BagSpec", "robosimstudio_materials.bag", "bag"),
    "granular": ("heap", "ParticleBodySpec", "robosimstudio_materials.granular", "heap"),
    "liquid": ("liquid", "ParticleBodySpec", "robosimstudio_materials.liquid", "liquid"),
    "dough": ("dough", "ParticleBodySpec", "robosimstudio_materials.dough", "dough"),
    "brush": ("brush", "BrushSpec", "robosimstudio_materials.brush", "brush"),
    "tape": ("tape", "TapeSpec", "robosimstudio_materials.tape", "tape"),
    "cutting": ("cuttable", "CuttableSpec", "robosimstudio_materials.cutting.entity", "cuttable"),
}
"""Kind -> its constructor, its spec, the module its entity is defined in, and the entity's registered name.

The entity's name is the kind's except for ``granular``, whose entity is a ``heap``, and ``cutting``, whose
block is a ``cuttable`` (its knife is a ``knife``)."""

FLAT = {
    "sheet": "robosimstudio_materials.cloth",
    "ClothSpec": "robosimstudio_materials.cloth",
    "sponge": "robosimstudio_materials.soft",
    "SoftSpec": "robosimstudio_materials.soft",
    "rod": "robosimstudio_materials.rod.make",
    "RodSpec": "robosimstudio_materials.rod.spec",
    "paper": "robosimstudio_materials.paper",
    "PaperSpec": "robosimstudio_materials.paper",
    "bag": "robosimstudio_materials.bag",
    "BagSpec": "robosimstudio_materials.bag",
    "heap": "robosimstudio_materials.granular",
    "liquid": "robosimstudio_materials.liquid",
    "dough": "robosimstudio_materials.dough",
    "roller": "robosimstudio_materials.dough",
    "ParticleBodySpec": "robosimstudio_materials.mpm.spec",
    "brush": "robosimstudio_materials.brush",
    "BrushSpec": "robosimstudio_materials.brush",
    "tape": "robosimstudio_materials.tape",
    "TapeSpec": "robosimstudio_materials.tape",
    "cuttable": "robosimstudio_materials.cutting.spec",
    "CuttableSpec": "robosimstudio_materials.cutting.spec",
    "knife": "robosimstudio_materials.cutting.spec",
    "KnifeSpec": "robosimstudio_materials.cutting.spec",
}
"""Flat name -> the module the thing behind it is defined in: each kind's constructor and spec, and the dough's roller.

The arrangement tables (``GRANULAR_60HZ`` and the like) are instances and carry
no module of their own, so they are checked by absence only, with every other
name ``robosimstudio._PACKS`` provides."""

LISTED = [
    "bag",
    "bags",
    "brush",
    "cuttable",
    "cuttables",
    "dough",
    "heap",
    "knife",
    "liquid",
    "materials",
    "paper",
    "papers",
    "rod",
    "roller",
    "sheet",
    "sponge",
    "tape",
    "tapes",
]
"""The lower-case flat names, as ``dir(rss)`` lists them when the package is installed."""


def _run(script: str, env: dict[str, str] | None = None, *, flags: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *flags, "-c", script],
        cwd=REPO,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", **(env or {})},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def _last_json(proc: subprocess.CompletedProcess):
    return json.loads([line for line in proc.stdout.splitlines() if line.startswith("{")][-1])


def _imports(path: Path) -> list[tuple[str, int]]:
    """Every module an ``import`` statement in ``path`` names, at any scope, with its line."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append(("." * node.level + (node.module or ""), node.lineno))
    return found


def _package_files() -> list[Path]:
    return sorted(p for p in MATERIALS.rglob("*.py") if "__pycache__" not in p.parts)


SCANNED = dedent(
    """
    import json, os, sys
    import robosimstudio as rss
    from robosimstudio.specs.physics import list_domains
    from robosimstudio.scene.registry import get_entity
    domains = list_domains()
    loaded = sorted(m for m in sys.modules if m.startswith("robosimstudio_materials"))
    stray = [m for m in loaded if not all(part.isidentifier() for part in m.split("."))]
    from_examples = sorted(
        name for name, module in list(sys.modules.items())
        if getattr(module, "__file__", None) and os.path.realpath(module.__file__).startswith(__EXAMPLES__)
    )
    print(json.dumps({
        "domains": domains, "loaded": loaded, "stray": stray, "from_examples": from_examples,
        "library": os.path.realpath(rss.__file__),
        "entities": {kind: get_entity(entity).__module__ for kind, entity in __ENTITIES__},
        "flat": {name: getattr(rss, name).__module__ for name in __FLAT__},
        "dir": sorted(n for n in dir(rss) if n in __LISTED__),
    }))
    """
).strip()


class TestAFreshProcessFindsTheKinds(unittest.TestCase):
    """The scan, in a process that imported nothing by name."""

    @classmethod
    def setUpClass(cls) -> None:
        script = (
            SCANNED.replace("__ENTITIES__", repr([(kind, entity) for kind, (_, _, _, entity) in KINDS.items()]))
            .replace("__FLAT__", repr(tuple(FLAT)))
            .replace("__LISTED__", repr(tuple(LISTED)))
            .replace("__EXAMPLES__", repr(f"{REPO / 'examples'}{os.sep}"))
        )
        proc = _run(script)
        if proc.returncode != 0:
            raise AssertionError(proc.stderr[-4000:])
        cls.got = _last_json(proc)

    def test_the_kinds_register(self) -> None:
        self.assertLessEqual(set(KINDS), set(self.got["domains"]))
        self.assertEqual({kind: module for kind, (_, _, module, _) in KINDS.items()}, self.got["entities"])

    def test_the_flat_names_are_the_packages_own(self) -> None:
        self.assertEqual(FLAT, self.got["flat"])
        self.assertEqual(LISTED, self.got["dir"])

    def test_the_scan_imports_no_example_script(self) -> None:
        """Of this checkout's ``examples/``, so the library it scanned with must be this checkout's too."""
        self.assertEqual(
            str((LIBRARY / "__init__.py").resolve()), self.got["library"], "robosimstudio came from another checkout"
        )
        self.assertEqual([], self.got["stray"], "a non-identifier name was imported under robosimstudio_materials")
        self.assertEqual([], self.got["from_examples"], "the scan imported a file out of examples/")
        self.assertLessEqual({f"robosimstudio_materials.{kind}" for kind in KINDS}, set(self.got["loaded"]))

    def test_each_registry_finds_a_kind_when_it_is_asked_first(self) -> None:
        """Nothing in the library imports the package, so each of these would answer
        differently before and after something else happened to import it -- unless
        the lookup itself runs the scan. A fresh process per question, so "first"
        is true of every one of them."""
        asks = {
            "the entities": "from robosimstudio.scene.registry import get_entity; print(get_entity('soft').__module__)",
            "the materials": (
                "from robosimstudio.assetlib.materials import get_material; "
                "print(get_material('terry_towelling').spec.__class__.__module__)"
            ),
            "the material kinds": (
                "from robosimstudio.assetlib.materials import MATERIAL_KINDS; print(MATERIAL_KINDS.get('cloth').__module__)"
            ),
            "the atoms": (
                "from robosimstudio.assetlib.semantics.atoms import ATOMS; "
                "print('robosimstudio_materials.cloth' if 'fold_score' in ATOMS else 'missing')"
            ),
        }
        for what, script in asks.items():
            with self.subTest(first_asked=what):
                proc = _run(script)
                self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
                self.assertTrue(
                    proc.stdout.strip().splitlines()[-1].startswith("robosimstudio_materials."),
                    f"{what} did not find the kinds: {proc.stdout.strip().splitlines()[-1:]}",
                )


ALONE = dedent(
    """
    import importlib, importlib.util, json, sys
    import robosimstudio as rss
    from robosimstudio.scene.registry import get_entity
    from robosimstudio.specs.physics import list_domains

    kind = importlib.import_module("robosimstudio_materials.__KIND__")
    spec = rss.__MAKE__(name="thing")
    print(json.dumps({
        "domains": list_domains(), "file": kind.__file__, "spec": type(spec).__name__,
        "entity": get_entity(spec.entity).__module__,
        "others_findable": [o for o in __OTHERS__ if importlib.util.find_spec("robosimstudio_materials." + o) is not None],
        "loaded": sorted(m for m in sys.modules if m.startswith("robosimstudio_materials.")),
    }))
    """
).strip()


class TestEachKindStandsAlone(unittest.TestCase):
    """Cloth without soft, soft without cloth, the rope without either, a heap without the pool: one module, one kind, nothing borrowed.

    ``soft.py`` used to reach into ``cloth.py`` for ``resolve_selector`` inside
    two functions, so a tree holding soft alone imported, registered, and then
    failed at the first build. What the kinds share is the library's
    (:mod:`robosimstudio.domains.selection`, ``capabilities``), and the claim is
    static *and* dynamic: no import names another kind at any scope, and with
    the other two left out of the package a kind still imports, registers and
    hands out its spec and its entity.
    """

    def test_no_kind_imports_another_at_any_scope(self) -> None:
        """Beyond what :data:`NEEDS` declares: paper, the bag and the tape may import cloth and the brush the rope, and that is the whole list."""
        for kind in KINDS:
            with self.subTest(kind=kind):
                module = MATERIALS / f"{kind}.py"
                files = [module] if module.is_file() else sorted((MATERIALS / kind).rglob("*.py"))
                self.assertTrue(files, f"{kind} is not in the package")
                offenders = [
                    f"{path.relative_to(REPO)}:{line}: {named}"
                    for path in files
                    for named, line in _imports(path)
                    for other in KINDS
                    if other != kind
                    and other not in NEEDS.get(kind, ())
                    and (
                        named == f"robosimstudio_materials.{other}"
                        or named.startswith(f"robosimstudio_materials.{other}.")
                    )
                ]
                self.assertEqual([], offenders, f"{kind} imports another kind")

    def test_a_declared_need_is_a_real_import(self) -> None:
        """The exception is not a blank cheque: paper names cloth because it uses it."""
        for kind, needs in NEEDS.items():
            for other in needs:
                with self.subTest(kind=kind, needs=other):
                    named = [m for m, _ in _imports(MATERIALS / f"{kind}.py")]
                    self.assertIn(f"robosimstudio_materials.{other}", named)

    def test_the_package_imports_only_the_library(self) -> None:
        """No file in the package names the studio, the benchmark or an agent, at any scope.

        The package may name itself: ``paper.py`` imports ``robosimstudio_materials.cloth``."""
        offenders = [
            f"{path.relative_to(REPO)}:{line}: {module}"
            for path in _package_files()
            for module, line in _imports(path)
            if module.split(".")[0] in SIBLINGS
        ]
        self.assertEqual([], offenders, "robosimstudio_materials imports a package beside the library")
        named = sorted(
            {
                module.split(".")[0]
                for path in _package_files()
                for module, _ in _imports(path)
                if module.startswith("robosimstudio") and not module.startswith(".")
            }
        )
        self.assertLessEqual(
            set(named), {"robosimstudio", "robosimstudio_materials"}, "the package names a package beside the library"
        )

    def test_a_kind_registers_and_builds_its_spec_with_the_others_absent(self) -> None:
        for kind, (make, spec, entity_module, _) in KINDS.items():
            others = tuple(sorted(set(KINDS) - {kind} - set(NEEDS.get(kind, ()))))
            with self.subTest(kind=kind, without=others), ExitStack() as stack:
                script = ALONE.replace("__KIND__", kind).replace("__MAKE__", make).replace("__OTHERS__", repr(others))
                proc = _run(script, env_without(stack, *others), flags=PYTHON_FLAGS)
                self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
                got = _last_json(proc)
                self.assertIn(kind, got["domains"])
                self.assertFalse(set(others) & set(got["domains"]), "the other kinds were meant to be left out")
                self.assertEqual(
                    [], got["others_findable"], "the other kinds were still importable: this proved nothing"
                )
                self.assertEqual(spec, got["spec"])
                self.assertEqual(entity_module, got["entity"])
                self.assertEqual(
                    f"{kind}.py" if kind not in ("rod", "cutting") else "__init__.py", Path(got["file"]).name
                )
                self.assertEqual(
                    [], [m for m in got["loaded"] if m.split(".")[1] in others], "a module of another kind was imported"
                )


class TestTheLibraryNamesTheSiblingOnlyAsAString(unittest.TestCase):
    """The sibling law, both halves readable off the source."""

    def test_no_library_module_imports_the_package(self) -> None:
        offenders = [
            f"{path.relative_to(REPO)}:{line}: {module}"
            for path in sorted(p for p in LIBRARY.rglob("*.py") if "__pycache__" not in p.parts)
            for module, line in _imports(path)
            if module == "robosimstudio_materials" or module.startswith("robosimstudio_materials.")
        ]
        self.assertEqual([], offenders, "the library imports the materials package; it may only name it as a string")

    def test_every_kind_registry_scans_both_packages(self) -> None:
        """One list, twelve scans: a registry a kind writes to on import and that
        scanned only ``robosimstudio.domains`` would answer without the shipped
        kinds whenever it was asked first."""
        from robosimstudio import roots
        from robosimstudio.assetlib.materials import MATERIAL_KINDS
        from robosimstudio.assetlib.materials import MATERIALS as MATERIAL_RECORDS
        from robosimstudio.assetlib.registry import ASSETS
        from robosimstudio.assetlib.semantics.atoms import ATOMS
        from robosimstudio.assetlib.semantics.handlers import HANDLERS
        from robosimstudio.assetlib.semantics.kinds import KINDS as FEATURE_KINDS
        from robosimstudio.domains.capabilities import CAPABILITIES
        from robosimstudio.engines.geometry import ACTOR_BUILDERS, ACTOR_KINDS
        from robosimstudio.scene.registry import ENTITIES
        from robosimstudio.specs.registries import DOMAINS

        self.assertEqual(("robosimstudio.domains", "robosimstudio_materials"), roots.DOMAIN_PACKAGES)
        self.assertEqual((), roots.DEFAULT_DOMAIN_ROOT_NAMES, "the checkout has no domain root of its own any more")
        self.assertEqual([], roots.DOMAINS._checkouts())
        for name, registry in {
            "DOMAINS": DOMAINS,
            "ENTITIES": ENTITIES,
            "MATERIALS": MATERIAL_RECORDS,
            "MATERIAL_KINDS": MATERIAL_KINDS,
            "ASSETS": ASSETS,
            "ATOMS": ATOMS,
            "HANDLERS": HANDLERS,
            "KINDS": FEATURE_KINDS,
            "CAPABILITIES": CAPABILITIES,
            "ACTOR_KINDS": ACTOR_KINDS,
            "ACTOR_BUILDERS": ACTOR_BUILDERS,
        }.items():
            with self.subTest(registry=name):
                self.assertEqual(roots.DOMAIN_PACKAGES, registry.packages)

    def test_the_flat_name_table_keys_the_kinds(self) -> None:
        import robosimstudio as rss

        self.assertEqual(
            [
                "robosimstudio_materials.bag",
                "robosimstudio_materials.brush",
                "robosimstudio_materials.cloth",
                "robosimstudio_materials.cutting",
                "robosimstudio_materials.dough",
                "robosimstudio_materials.granular",
                "robosimstudio_materials.liquid",
                "robosimstudio_materials.paper",
                "robosimstudio_materials.rod",
                "robosimstudio_materials.soft",
                "robosimstudio_materials.tape",
                "robosimstudio_materials.water",
            ],
            sorted(rss._PACKS),
        )
        for name in (
            "sheet",
            "sponge",
            "rod",
            "paper",
            "papers",
            "materials",
            "ClothSpec",
            "SoftSpec",
            "RodSpec",
            "heap",
            "liquid",
            "dough",
            "PaperSpec",
            "bag",
            "bags",
            "BagSpec",
            "brush",
            "BrushSpec",
        ):
            self.assertNotIn(name, rss._EXPORTS, f"{name} is a kind's and belongs in _PACKS")
            self.assertIn(name, rss._PACK_EXPORTS)


class TestTheWheelShipsThePackage(unittest.TestCase):
    """``pyproject.toml`` and the publish workflow, read as text; ``uv build`` is the gate that runs them."""

    def test_pyproject_lists_the_package_and_no_copy_of_the_kinds(self) -> None:
        text = (REPO / "pyproject.toml").read_text()
        config = tomllib.loads(text)
        wheel = config["tool"]["hatch"]["build"]["targets"]["wheel"]
        self.assertIn("robosimstudio_materials", wheel["packages"])
        self.assertIn("/robosimstudio_materials", config["tool"]["hatch"]["build"]["targets"]["sdist"]["include"])
        self.assertNotIn("_domain_packs", text)
        self.assertEqual({"assets"}, set(wheel.get("force-include", {})), "only the robot assets are force-included")

    def test_the_publish_workflow_checks_for_the_package(self) -> None:
        text = (REPO / ".github" / "workflows" / "publish.yml").read_text()
        for path in (
            "robosimstudio_materials/cloth.py",
            "robosimstudio_materials/soft.py",
            "robosimstudio_materials/rod/__init__.py",
            "robosimstudio_materials/paper.py",
            "robosimstudio_materials/bag.py",
        ):
            self.assertIn(path, text)
        self.assertNotIn("_domain_packs", text)


WITHOUT = dedent(
    """
    import json, sys
    sys.modules["robosimstudio_materials"] = None   # the import system now says the package is not there
    import robosimstudio as rss
    out = {"errors": {}, "resolved": []}
    for name in __NAMES__:
        try:
            getattr(rss, name)
            out["errors"][name] = "resolved"
        except AttributeError as exc:
            out["errors"][name] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            out["errors"][name] = f"WRONG {type(exc).__name__}: {exc}"
    out["listed"] = [n for n in dir(rss) if n in out["errors"]]
    for name in sorted(rss._EXPORTS):
        try:
            getattr(rss, name)
        except Exception as exc:
            out["resolved"].append(f"{name}: {type(exc).__name__}: {exc}")
    from robosimstudio.specs.physics import list_domains
    from robosimstudio.scene.registry import ENTITIES
    from robosimstudio.assetlib.materials import MATERIALS
    out["domains"] = list_domains()
    out["entities"] = ENTITIES.names()
    out["materials"] = MATERIALS.names()
    scene = rss.Scene(cell="void")
    cube = scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0, 0, 0.2)))
    sim = scene.build(device="cpu")
    sim.step(5)
    out["cube_z"] = float(cube.position[2])
    out["finite"] = bool(sim.state_is_finite())
    out["materials_loaded"] = [m for m in sys.modules if m.startswith("robosimstudio_materials") and sys.modules[m] is not None]
    print(json.dumps(out))
    """
).strip()


class TestTheLibraryIsWholeWithoutThePackage(unittest.TestCase):
    """A process in which ``robosimstudio_materials`` cannot be imported, by the import system's own word."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        cls.names = sorted(rss._PACK_EXPORTS)
        proc = _run(WITHOUT.replace("__NAMES__", repr(tuple(cls.names))))
        if proc.returncode != 0:
            raise AssertionError(f"the library did not come up without robosimstudio_materials:\n{proc.stderr[-6000:]}")
        cls.got = _last_json(proc)

    def test_the_package_really_was_absent(self) -> None:
        self.assertEqual([], self.got["materials_loaded"])
        self.assertEqual(["articulated", "rigid", "static"], self.got["domains"])
        self.assertFalse({entity for (_, _, _, entity) in KINDS.values()} & set(self.got["entities"]))
        self.assertNotIn("rice", self.got["materials"])
        self.assertNotIn("terry_towelling", self.got["materials"])

    def test_the_kinds_names_say_which_package_is_missing(self) -> None:
        self.assertEqual(self.names, sorted(self.got["errors"]))
        self.assertLessEqual(
            set(FLAT) | set(LISTED), set(self.names), "every name the tests know is one the table provides"
        )
        for name, message in self.got["errors"].items():
            with self.subTest(name=name):
                self.assertTrue(message.startswith("AttributeError: "), message)
                self.assertIn("is not installed here", message)
                self.assertIn("robosimstudio_materials", message)
                self.assertIn(f"rss.{name}", message)
        self.assertEqual([], self.got["listed"], "dir(rss) must not list a name that does not resolve")

    def test_every_core_name_resolves_and_a_rigid_scene_builds(self) -> None:
        self.assertEqual([], self.got["resolved"])
        self.assertTrue(self.got["finite"])
        self.assertGreater(self.got["cube_z"], 0.15, "the cube was dropped from 200 mm and stepped five times")


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""Content can reach every registry that says it can.

Four registries -- ``DOMAINS``, ``MATERIALS``, ``ATOMS`` and ``HANDLERS`` --
advertised a content path and did not have one. Each scanned a fixed package or
nothing at all and carried no populator, so a domain, a fabric, an atom or a
handler declared in a task directory registered **if and only if** something else
had already imported that task. ``specs/registries.py`` stated the promise outright:
*"A task registers its own the same way, from its assets/ package."*

The usage counts are what a promise nobody could keep looks like from outside.
Against thirty-six assets and fifteen entity classes declared by content, these
four had **eight domains, seven materials, nineteen atoms and thirty handlers --
every one of them the library's own, and none from a task.**

So this module asks the question that was never asked: put a domain and a
material in a throwaway task root, execute it, and see whether the registries
have them. And separately, that each of the four now carries a populator at all
-- because a registry that scans nothing and is told nothing finds nothing, and
that failure looks exactly like an empty corpus.

Static: no scene, no GPU.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent

from robosimstudio import roots
from robosimstudio.assetlib.materials import MATERIALS
from robosimstudio.assetlib.semantics.atoms import ATOMS
from robosimstudio.assetlib.semantics.handlers import HANDLERS
from robosimstudio.specs.physics import DOMAINS
from robosimstudio_bench.taskbuild.discovery import scan
from tests.fixtures.processes import Processes, last_json

TIER = "fast"
"""No GPU. Gates every commit."""

DOCUMENT = dedent(
    """
    (define (problem PROBE_DISCOVERY)
      (:domain robosimstudio/probe)
      (:language a probe)
      (:objects)
      (:goal)
    )
    """
).strip()

LOADER = dedent(
    '''
    """A task directory that declares a physics domain and a fabric."""
    from robosimstudio.assetlib.materials import MaterialRecord, register_material
    from robosimstudio_materials.cloth import ClothMaterialSpec
    from robosimstudio.specs.physics import register_domain

    register_domain(
        "probe_gel",
        state=("particles",),
        reads="cloth",
        summary="A probe domain declared by content.",
    )
    register_material(
        MaterialRecord(
            name="probe_denim",
            spec=ClothMaterialSpec(),
            notes="A probe fabric declared by content.",
        )
    )
    '''
).strip()


class _Root:
    """A throwaway task root, cleaned up and unregistered afterwards."""

    def __init__(self) -> None:
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name)
        directory = self.path / "probe_discovery"
        directory.mkdir(parents=True)
        (directory / f"probe_discovery{roots.TASK_SUFFIX}").write_text(DOCUMENT)
        (directory / "task.py").write_text(LOADER)

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, *exc: object) -> None:
        for registry, name in ((DOMAINS, "probe_gel"), (MATERIALS, "probe_denim")):
            registry.forget(name)
        roots.clear_task_roots()
        self._tmp.cleanup()


class TestContentReachesTheseRegistries(unittest.TestCase):
    def test_a_task_directory_can_declare_a_domain_and_a_material(self) -> None:
        """The promise ``specs/registries.py`` made, executed.

        A ``register_domain`` from a task directory was reachable only by luck of
        import order before this; the domain here is declared by a root that
        nothing else has any reason to import.
        """
        with _Root() as root:
            self.assertNotIn("probe_gel", DOMAINS._items, "the probe leaked from an earlier run")
            scan(root)
            self.assertIn("probe_gel", DOMAINS._items, "a domain declared by content did not register")
            self.assertIn("probe_denim", MATERIALS._items, "a material declared by content did not register")
            self.assertEqual(("cloth",), DOMAINS.get("probe_gel").reads)


class TestEveryContentRegistryHasAPopulator(unittest.TestCase):
    """A registry that scans nothing and is told nothing finds nothing.

    That failure is invisible: it looks exactly like a corpus with none of the
    thing in it, which is the state all four of these were in.
    """

    def test_the_four_that_had_none_now_have_one(self) -> None:
        for name, registry in (
            ("DOMAINS", DOMAINS),
            ("MATERIALS", MATERIALS),
            ("ATOMS", ATOMS),
            ("HANDLERS", HANDLERS),
        ):
            with self.subTest(registry=name):
                self.assertTrue(
                    registry.discovers,
                    f"{name} scans no content and is told about none; a task's entry would never be found",
                )


KIND = dedent(
    '''
    """A kind of thing declared on somebody else's root."""
    from robosimstudio.scene.entity import Entity
    from robosimstudio.scene.registry import register_entity
    from robosimstudio.specs.physics import register_domain

    register_domain("probe_granular", state=("particles",), reads="cloth",
                    role="deformables", summary="A bed of grains.")

    @register_entity("probe_granular")
    class EntityProbeGranular(Entity):
        domain = "probe_granular"
    '''
).strip()

REPORT = dedent(
    """
    import json
    from robosimstudio.specs.physics import list_domains, get_domain
    from robosimstudio.scene.registry import get_entity
    cls = get_entity("probe_granular")
    print(json.dumps({"domains": list_domains(), "summary": get_domain("probe_granular").summary,
                      "module": cls.__module__, "entity": cls.__name__}))
    """
).strip()


def _kind_root(processes: Processes, name: str = "probe_granular", *, module: bool = False, scripts=()) -> Path:
    """A root holding one kind -- a package, or with ``module`` a single ``<name>.py`` -- and example ``scripts``."""
    root = processes.directory()
    source = (
        KIND.replace("probe_granular", name).replace("ProbeGranular", "Probe") if name != "probe_granular" else KIND
    )
    if module:
        (root / f"{name}.py").write_text(source)
    else:
        (root / name).mkdir()
        (root / name / "__init__.py").write_text(source)
    for script in scripts:
        (root / script).write_text(NUMBERED)
    return root


def _start(processes: Processes, key: str, script: str, *rootpaths: Path) -> None:
    """A fresh interpreter with ``rootpaths`` as its domain roots."""
    processes.start(key, script, env={roots.DOMAIN_ROOTS_ENV: os.pathsep.join(str(r) for r in rootpaths)})


class TestAKindMayLiveOutsideTheCheckout(unittest.TestCase):
    """A domain package on somebody else's root, reached by our own import name.

    The task roots execute a directory under a synthetic parent
    (``rstask.<slug>_<hash>``) precisely so a task never becomes a top-level
    importable name. A domain package wants the opposite: a task directory
    writes ``from robosimstudio.domains.granular import EntityHeap``, and that
    line has to keep working whether the kind is in the library or on a root.
    So ``robosimstudio.domains`` extends its own ``__path__`` instead, and the
    registries that scan it -- ``DOMAINS``, ``ENTITIES``, ``MATERIALS``, every
    one walking ``pkgutil.iter_modules(__path__)`` -- pick up both in one sweep
    with no registry change at all. (The kinds the repository ships are not on
    a root: they are the package ``robosimstudio_materials``, which the same
    registries scan beside this one -- ``tests/test_materials_package``.)

    Run in a subprocess because ``__path__`` is built when the package is first
    imported, so this is a claim about what a *process* sees -- one process per
    claim, all of them started together.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.processes = Processes()
        _start(cls.processes, "ours", REPORT, _kind_root(cls.processes))
        cls.first, cls.second = _kind_root(cls.processes, "probe_dup"), _kind_root(cls.processes, "probe_dup")
        _start(cls.processes, "two roots", "import robosimstudio.domains", cls.first, cls.second)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.processes.close()

    def test_it_registers_and_reads_as_ours(self) -> None:
        proc = self.processes.result("ours")
        self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
        got = last_json(proc)
        self.assertIn("probe_granular", got["domains"], "a domain on a root did not register")
        self.assertIn(
            "cloth", got["domains"], "the library's own kinds are gone -- the path was replaced, not extended"
        )
        self.assertEqual("A bed of grains.", got["summary"])
        self.assertEqual(
            "robosimstudio.domains.probe_granular",
            got["module"],
            "an out-of-tree kind must be reachable under OUR import name -- that is what lets a task's "
            "`from robosimstudio.domains.<kind> import ...` name a kind the library never shipped",
        )

    def test_two_roots_holding_one_name_refuse(self) -> None:
        """``pkgutil.iter_modules`` keeps the first and never imports the second,
        which hides exactly the collision the registries exist to report."""
        proc = self.processes.result("two roots")
        self.assertNotEqual(0, proc.returncode, "a duplicate kind across two roots was accepted")
        self.assertIn("probe_dup", proc.stderr)
        self.assertIn(str(self.second), proc.stderr, "the message must name both roots")


NUMBERED = 'raise RuntimeError("a numbered example was imported as a domain pack")'
"""An example script as the scan must see it: not at all. Importing it is the failure."""

SCANNED = dedent(
    """
    import json, sys
    from robosimstudio.specs.physics import list_domains
    domains = list_domains()
    stray = sorted(m for m in sys.modules if m.startswith("robosimstudio.domains.") and not m.rsplit(".", 1)[-1].isidentifier())
    print(json.dumps({"domains": domains, "stray": stray}))
    """
).strip()


class TestARootMayHoldAModuleOrAnExampleFolder(unittest.TestCase):
    """What :func:`robosimstudio.roots.extend_path` accepts on a root of somebody's own.

    ``<root>/<name>.py`` is a kind exactly as ``<root>/<name>/__init__.py`` is;
    a numbered script beside it is not a kind and is never imported; and one
    name as a module on one root and a package on another is the clash the
    duplicate rule exists to catch, which ``pkgutil.iter_modules`` would hide
    by keeping the first. The repository's own kinds no longer travel this way
    (they are a package), so these claims are made on throwaway roots, a fresh
    process each, all started together.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.processes = processes = Processes()
        _start(processes, "module", REPORT, _kind_root(processes, module=True))
        cls.pairs = {}
        for order in ("module first", "package first"):
            module, package = _kind_root(processes, "probe_dup", module=True), _kind_root(processes, "probe_dup")
            cls.pairs[order] = pair = (module, package) if order == "module first" else (package, module)
            _start(processes, order, "import robosimstudio.domains", *pair)
        leftover = processes.directory()
        (leftover / "probe_granular" / "__pycache__").mkdir(parents=True)
        (leftover / "probe_granular" / "__pycache__" / "__init__.cpython-312.pyc").write_bytes(b"")
        _start(processes, "emptied", REPORT, leftover, _kind_root(processes, module=True))
        root = _kind_root(processes, module=True, scripts=("00_drop.py", "01_fold.py"))
        _start(processes, "beside scripts", SCANNED, root)
        first = _kind_root(processes, module=True, scripts=("00_intro.py",))
        second = _kind_root(processes, "probe_gel", module=True, scripts=("00_intro.py",))
        _start(processes, "two intros", SCANNED, first, second)
        first = _kind_root(processes, module=True, scripts=("00_intro.py",))
        second = _kind_root(processes, module=True, scripts=("01_other.py",))
        _start(processes, "one pack twice", "import robosimstudio.domains", first, second)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.processes.close()

    def test_a_single_module_on_a_root_registers_and_reads_as_ours(self) -> None:
        proc = self.processes.result("module")
        self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
        got = last_json(proc)
        self.assertIn("probe_granular", got["domains"], "a one-module kind on a root did not register")
        self.assertIn("cloth", got["domains"], "the shipped kinds are gone -- the scan lost a package")
        self.assertEqual("robosimstudio.domains.probe_granular", got["module"])

    def test_a_module_and_a_package_of_one_name_refuse(self) -> None:
        for order, pair in self.pairs.items():
            with self.subTest(order=order):
                proc = self.processes.result(order)
                self.assertNotEqual(0, proc.returncode, "one name as a module and as a package was accepted")
                self.assertIn("probe_dup", proc.stderr)
                self.assertIn(str(pair[1]), proc.stderr, "the message must name both roots")

    def test_an_emptied_directory_is_not_a_second_copy(self) -> None:
        """What a moved pack leaves behind in a checkout that has run it: the
        directory, holding nothing but ``__pycache__``. It is not a package -- the
        scan would not yield it -- so it must not be refused as one."""
        proc = self.processes.result("emptied")
        self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
        self.assertEqual("robosimstudio.domains.probe_granular", last_json(proc)["module"])

    def test_the_scan_imports_the_pack_and_not_the_scripts_beside_it(self) -> None:
        proc = self.processes.result("beside scripts")
        self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
        got = last_json(proc)
        self.assertIn("probe_granular", got["domains"], "the pack beside the scripts did not register")
        self.assertEqual([], got["stray"], "a numbered script was imported under robosimstudio.domains")

    def test_two_folders_may_each_hold_a_script_of_one_name(self) -> None:
        """Two ``00_intro.py`` are two examples, not one kind declared twice -- which
        is what the duplicate rule would say of two files called ``intro.py``."""
        proc = self.processes.result("two intros")
        self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
        got = last_json(proc)
        self.assertLessEqual({"probe_granular", "probe_gel"}, set(got["domains"]))
        self.assertEqual([], got["stray"])
        # The counter-test: the exemption is for names no import can spell, and for nothing else.
        proc = self.processes.result("one pack twice")
        self.assertNotEqual(0, proc.returncode, "one pack name on two roots was accepted")
        self.assertIn("probe_granular", proc.stderr)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""A rope routed to SuperDex names the capsule chain's fields it states and the Cosserat rod never reads.

Measured before this: a rope stating ``dahl_eps_max=0.5``, ``twist_ke=1e-3``,
``shear_ke=1e5``, ``bend_ke=0.2`` and ``stretch_ke=3e5``, put on SuperDex by its
arrangement, built as ``EntityRodSuperdex`` and stepped with no warning at all,
while a cloth's lost fields are named (``tests/test_material_loss_is_loud``).
The composer's entity side read ``spec.material``, and a ``RodSpec`` has none:
its matter is its own fields, and its actor's params are derived from some of
them (``axial_stiffness`` from ``young_modulus``), so no key says which went in.
The rod now names the fields only the capsule chain reads
(``EntityRodSuperdex.unread_fields``), and the composer says which of them the
spec states away from its default, in the same ``MaterialLossWarning``.

No engine and no scene: the composer's check over the rope's own actor, against
the ``superdex`` alias, which registers without the wheel. The list is checked
against the source of every module the rope's spec and actor pass through, and
against a rope built into a model on the CPU whose spec notes each read of it.
"""

from __future__ import annotations

import ast
import dataclasses
import sys
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

from tests.fixtures.packs import MATERIALS

TIER = "fast"
"""The composer's warning over a constructed entity, and one rope built into a CPU model: no GPU."""

HAS_ROD = (MATERIALS / "rod" / "__init__.py").is_file()
ROOT = MATERIALS.parent
READERS = (
    *("robosimstudio_materials/rod/superdex.py", "robosimstudio_materials/rod/centerline.py"),
    *("robosimstudio_materials/rod/actor.py", "robosimstudio_materials/rod/measures.py"),
    *(f"robosimstudio/engines/superdex/{m}.py" for m in ("__init__", "bridge", "mirror", "solver")),
    *("robosimstudio/scene/entity.py", "robosimstudio/domains/capabilities.py"),
    "robosimstudio_studio/tuning/binder.py",
)
"""What a SuperDex rope's spec and actor pass through: the rope and its ``Centerline``, the rod's
actor module and measures, the SuperDex bridge, ``Entity`` and ``ParticleMixin``, and the studio's
``SpecBinder``. Not ``rod/entity.py`` or ``RodSpec.drive_gains``: the capsule chain's, which read
every one of them."""


def _lost(**fields) -> list[str]:
    """The ``MaterialLossWarning`` messages the composer gives a SuperDex rope stating ``fields``."""
    from robosimstudio.scene.physics.compose import _warn_lost_material
    from robosimstudio.scene.physics.errors import MaterialLossWarning
    from robosimstudio_materials.rod import RodSpec
    from robosimstudio_materials.rod.superdex import EntityRodSuperdex

    entity = EntityRodSuperdex(RodSpec(name="cable", length=0.4, segments=18, **fields))
    node = SimpleNamespace(name="cable", entity=entity)
    island = SimpleNamespace(backend=SimpleNamespace(solver="superdex"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_lost_material(node, entity.host_actor(), island)
    return [str(w.message) for w in caught if issubclass(w.category, MaterialLossWarning)]


@unittest.skipUnless(HAS_ROD, "the rod pack is not in this tree")
class TestARopeOnSuperdexSaysWhatItDrops(unittest.TestCase):
    def test_the_capsule_chains_stated_fields_are_named(self) -> None:
        messages = _lost(dahl_eps_max=0.5, twist_ke=1.0e-3, shear_ke=1.0e5, bend_ke=0.2, stretch_ke=3.0e5)
        self.assertEqual(1, len(messages), messages)
        message = messages[0]
        for name in ("dahl_eps_max", "twist_ke", "shear_ke", "bend_ke", "stretch_ke"):
            with self.subTest(field=name):
                self.assertIn(f"'{name}'", message)
        self.assertIn("superdex", message)
        self.assertIn("flexural_stiffness", message, "the message says what the engine is handed instead")
        self.assertIn("5 stated RodSpec field(s) never reach superdex", message, "the record the fields are on")
        self.assertNotIn("material", message, "a RodSpec has no material: its matter is its own fields")

    def test_a_field_left_at_its_default_is_not_named(self) -> None:
        """``dahl_tau`` is stated by every rope at 0.1 and says nothing until ``dahl_eps_max`` is set."""
        message = _lost(twist_ke=1.0e-3)[0]
        self.assertNotIn("dahl_tau", message)
        self.assertNotIn("bend_ke", message)

    def test_a_rope_stating_only_what_superdex_reads_is_quiet(self) -> None:
        """The knot example's rope: modulus, stretch, friction, penalty, quadrature."""
        self.assertEqual(
            [],
            _lost(
                young_modulus=1.0e5,
                axial_stiffness=300.0,
                mu_cable=0.25,
                friction_falloff=1.0e-3,
                contact_penalty=4.0e8,
                skin_samples="P1Q6",
                self_contact=True,
            ),
        )

    def test_a_default_rope_is_quiet(self) -> None:
        self.assertEqual([], _lost())

    def test_a_list_equal_to_its_default_tuple_is_not_named(self) -> None:
        """``surface_labels=["table"]`` and ``bow_range=[0.0, 0.0]`` are the defaults, given as lists: compared
        with ``!=``, a list never equals a tuple, and both were reported lost."""
        self.assertEqual([], _lost(surface_labels=["table"], bow_range=[0.0, 0.0]))
        self.assertIn("'surface_labels'", _lost(surface_labels=["table", "bin"])[0], "a list that differs is stated")


class TestANumpyValueAtItsDefaultIsNotStated(unittest.TestCase):
    """``_stated`` compares through ``_as_value``, which widened a float32 array's floats to float64."""

    def test_a_float32_array_of_the_default_is_not_stated(self) -> None:
        """An MPM material's ``color`` as a float32 array of ``(0.86, 0.8, 0.62)`` was named stated."""
        import numpy as np

        from robosimstudio.scene.physics.compose import _stated

        @dataclasses.dataclass
        class _Material:
            color: tuple = (0.86, 0.8, 0.62)
            yield_stress: float = 0.1

        self.assertEqual([], _stated(_Material(color=np.array((0.86, 0.8, 0.62), dtype=np.float32))))
        self.assertEqual([], _stated(_Material(yield_stress=np.float32(0.1))), "a float32 scalar")
        self.assertEqual([], _stated(_Material(yield_stress=np.array(0.1, dtype=np.float32))), "a 0-d array")
        self.assertEqual(["color"], _stated(_Material(color=np.array((0.86, 0.8, 0.6), dtype=np.float32))))
        self.assertEqual(["yield_stress"], _stated(_Material(yield_stress=np.float32(0.2))))


@unittest.skipUnless(HAS_ROD, "the rod pack is not in this tree")
class TestTheListIsTrue(unittest.TestCase):
    """Each name is a ``RodSpec`` field the SuperDex rope's own code never reads: a list that went stale
    would warn about a knob that works."""

    def test_every_name_is_a_rod_field(self) -> None:
        from robosimstudio_materials.rod import RodSpec
        from robosimstudio_materials.rod.superdex import EntityRodSuperdex

        fields = {f.name for f in dataclasses.fields(RodSpec)}
        self.assertEqual([], [n for n in EntityRodSuperdex.unread_fields if n not in fields])

    def test_the_superdex_rope_reads_none_of_them(self) -> None:
        """Named in the source of every module a SuperDex rope's spec or actor passes through: as an
        attribute, or as a string (``getattr(spec, "bend_ke")``, an actor param's key)."""
        from robosimstudio_materials.rod.superdex import EntityRodSuperdex

        unread = set(EntityRodSuperdex.unread_fields)
        read: list[str] = []
        for module in READERS:
            tree = ast.parse((ROOT / module).read_text())
            listed = {
                id(c)
                for node in ast.walk(tree)
                if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "unread_fields"
                for c in ast.walk(node.value)
            }
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in unread:
                    read.append(f"{module}:{node.lineno}: .{node.attr}")
                elif isinstance(node, ast.Constant) and node.value in unread and id(node) not in listed:
                    read.append(f"{module}:{node.lineno}: {node.value!r}")
        self.assertEqual([], read, "the SuperDex rope reads a field it says it drops")

    def test_a_built_rope_reads_none_of_them(self) -> None:
        """At run time, which a name computed in a ``getattr`` does not escape: a rope whose spec notes
        every read of a listed field, built into a model on the CPU, bound, readied, observed, published,
        reset and offered for tuning. A whole-record copy or comparison (``dataclasses.replace``, the
        generated ``__eq__``) is not a read of a knob, and is not noted. The engine's side is the actor's
        params, which name none of them; the source of what reads those is scanned above."""
        import newton

        from robosimstudio_materials.rod import RodSpec
        from robosimstudio_materials.rod.superdex import EntityRodSuperdex

        unread = set(EntityRodSuperdex.unread_fields)
        read: list[str] = []

        class _Noted(RodSpec):
            def __getattribute__(self, name: str):
                if name in unread:
                    code = sys._getframe(1).f_code
                    if code.co_filename not in (dataclasses.__file__, "<string>"):
                        read.append(f"{Path(code.co_filename).name}:{code.co_name}: .{name}")
                return super().__getattribute__(name)

        rope = EntityRodSuperdex(_Noted(name="cable", length=0.4, segments=18))
        rope.sim_requirements()
        self.assertEqual([], sorted(set(rope.host_actor().params) & unread), "the engine is handed one")
        builder = newton.ModelBuilder()
        rope.build_into(builder, None)
        rope.after_build(builder, None)
        rope.apply_display(builder)
        model = builder.finalize(device="cpu")
        rope.bind(model)
        rope.apply_model_materials(model)
        pipeline = SimpleNamespace(state_0=model.state(), state_1=model.state())
        rope.on_scene_ready(pipeline)
        rope.observe(pipeline.state_0, 0)
        snapshot = SimpleNamespace(particle_q=pipeline.state_0.particle_q.numpy())
        rope.published_states(snapshot)
        rope.curve_states(snapshot)
        rope.handle(model)
        rope.reset(pipeline, None)
        self.assertEqual((), rope.tunables(None), "the studio would bind a tunable through SpecBinder")
        self.assertEqual([], read, "the SuperDex rope reads a field it says it drops")


if __name__ == "__main__":
    unittest.main()

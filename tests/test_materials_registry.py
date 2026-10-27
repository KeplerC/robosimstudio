# SPDX-License-Identifier: Apache-2.0
"""What each shipped kind registers, as one table: its flat names, its domain, its entity, its records and materials.

Every kind of ``robosimstudio_materials`` makes the same five promises, and each
kind's own module used to write them its own way. Here they are rows of one
table checked by one test per promise, so a kind that drops a name, registers its
domain reading the wrong layouts, or files a material under another kind fails
by name -- and a kind added to the package with no row fails too, since the
library's own tables have to equal this one:

* **Its flat names** are on ``rss`` as the kind module's own objects, through
  the pack table (``rss._PACK_EXPORTS``), and never in the library's own table.
* **Its domain** registers, reading its own layout and the ones it borrows.
* **Its entity** registers under its name, and is the kind's class.
* **A record** made from one of its specs carries its domain and builds that
  very spec; handed another kind's spec, it refuses.
* **Its materials** register under its kind, and a named one is the module's own value.
"""

from __future__ import annotations

import importlib
import unittest
from typing import NamedTuple

TIER = "fast"
"""Registries, records and specs: nothing is built, so no GPU."""


class Kind(NamedTuple):
    module: str
    """The module the pack table names: ``robosimstudio_materials.<module>``."""
    names: tuple[str, ...]
    """Every flat name it provides on ``rss``."""
    reads: tuple[str, ...]
    """What its domain reads: its own layout first, then the ones it borrows."""
    entities: tuple[tuple[str, str], ...]
    """``(registered name, module.Class)`` for each entity it registers."""
    record: tuple[str, str, str] = ("", "", "")
    """``(record class, the constructor making a spec, the domain the record carries)``; empty for a kind with none."""


KINDS: dict[str, Kind] = {
    "cloth": Kind(
        "cloth",
        ("ClothMaterialSpec", "ClothSpec", "materials", "sheet"),
        ("cloth",),
        (("cloth", "cloth.EntityCloth"), ("held_cloth", "cloth.EntityHeldCloth")),
    ),
    "soft": Kind(
        "soft",
        ("SoftMaterialSpec", "SoftSpec", "sponge"),
        ("volume", "cloth"),
        (("soft", "soft.EntitySoft"), ("soft@superdex", "soft.EntitySoftSuperdex")),
    ),
    "rod": Kind(
        "rod",
        ("RodMaterialSpec", "RodSpec", "rod"),
        ("rod",),
        (("rod", "rod.entity.EntityRod"), ("rod@superdex", "rod.superdex.EntityRodSuperdex")),
        ("Rod", "rod", "rod"),
    ),
    "granular": Kind(
        "granular",
        ("GRANULAR_60HZ", "ParticleBodySpec", "ParticleMaterialSpec", "heap"),
        ("granular", "cloth"),
        (("heap", "granular.EntityHeap"),),
        ("Heap", "heap", "granular"),
    ),
    "liquid": Kind(
        "liquid",
        ("LIQUID_120HZ", "liquid"),
        ("liquid", "granular", "cloth"),
        (("liquid", "liquid.EntityLiquid"),),
        ("Liquid", "liquid", "liquid"),
    ),
    "dough": Kind(
        "dough",
        ("DOUGH_60HZ", "dough", "roller"),
        ("dough", "granular", "cloth"),
        (("dough", "dough.EntityDough"), ("roller", "dough.EntityRoller")),
        ("Dough", "dough", "dough"),
    ),
    "paper": Kind(
        "paper",
        ("PaperMaterialSpec", "PaperSpec", "paper", "papers"),
        ("paper", "cloth"),
        (("paper", "paper.EntityPaper"),),
    ),
    "bag": Kind(
        "bag",
        ("BAG_60HZ", "BagSpec", "bag", "bags"),
        ("bag", "cloth"),
        (("bag", "bag.EntityBag"), ("bag@superdex", "bag.EntityBagSuperdex")),
        ("Bag", "bag", "bag"),
    ),
    "brush": Kind(
        "brush",
        ("BRUSH_60HZ", "BrushMaterialSpec", "BrushSpec", "brush"),
        ("brush", "rod"),
        (("brush", "brush.EntityBrush"),),
        ("Brush", "brush", "brush"),
    ),
    "tape": Kind(
        "tape",
        ("TAPE_60HZ", "TapeMaterialSpec", "TapeSpec", "tape", "tapes"),
        ("tape", "cloth"),
        (("tape", "tape.EntityTape"),),
        ("Tape", "tape", "tape"),
    ),
    "cutting": Kind(
        "cutting",
        ("CUTTING_50HZ", "CuttableMaterialSpec", "CuttableSpec", "KnifeSpec", "cuttable", "cuttables", "knife"),
        ("cutting",),
        (("cuttable", "cutting.entity.EntityCuttable"), ("knife", "cutting.entity.EntityKnife")),
        ("Cuttable", "cuttable", "cutting"),
    ),
    "water": Kind(
        "water",
        ("WATER_60HZ", "WaterMaterialSpec", "WaterSpec", "water"),
        ("water", "granular", "cloth"),
        (("water", "water.EntityWater"), ("water@genesis", "water.EntityWaterGenesis")),
        ("Water", "water", "water"),
    ),
}

NAMED_MATERIALS: dict[str, tuple[str, str]] = {
    "terry_towelling": ("cloth", "TERRY_TOWELLING"),
    "terry_membrane": ("cloth", "TERRY_MEMBRANE"),
    "shirt_jersey": ("cloth", "SHIRT_JERSEY"),
    "shirt_fleece": ("cloth", "SHIRT_FLEECE"),
    "shirt_membrane": ("cloth", "SHIRT_MEMBRANE"),
    "tape": ("cloth", "TAPE_MATERIAL"),
    "soft_foam": ("soft", "SOFT_FOAM"),
    "firm_foam": ("soft", "FIRM_FOAM"),
    "paper": ("paper", "PAPER"),
    "cardboard": ("paper", "CARDBOARD"),
    "paper_label": ("tape", "PAPER_LABEL"),
    "label": ("tape", "LABEL"),
    "masking_tape": ("tape", "MASKING_TAPE"),
    "duct_tape": ("tape", "DUCT_TAPE"),
    "tofu": ("cutting", "TOFU"),
    "sponge_cake": ("cutting", "CAKE"),
    "pu_foam": ("cutting", "FOAM"),
}
"""Material -> (the kind it registers under, the module's name for it). The particle kinds' presets
live in the shared ``mpm`` package, and the brush's and the rope's in their records."""

KIND_OF: dict[str, tuple[str, ...]] = {
    "granular": (
        "rice",
        "sand",
        "beans",
        "flour",
        "honey",
        "ketchup",
        "water",
        "dough",
        "plasticine",
        "clay",
        "butter",
    ),
    "brush": ("nylon_bristle", "soft_bristle"),
    "rod": ("reference_cable_drive",),
}
"""The rest of the materials, by the kind they register under. The liquid and the dough read as
``granular``: one spec, :class:`ParticleMaterialSpec`, for the three particle kinds."""


def _kind_module(kind: str):
    return importlib.import_module(f"robosimstudio_materials.{KINDS[kind].module}")


def _resolve(dotted: str):
    module, _, name = f"robosimstudio_materials.{dotted}".rpartition(".")
    return getattr(importlib.import_module(module), name)


class TestTheTableIsTheLibrarys(unittest.TestCase):
    def test_the_pack_table_keys_every_kind_and_no_other(self) -> None:
        import robosimstudio as rss

        self.assertEqual(sorted(f"robosimstudio_materials.{k.module}" for k in KINDS.values()), sorted(rss._PACKS))

    def test_each_kind_provides_exactly_its_flat_names(self) -> None:
        import robosimstudio as rss

        provided: dict[str, set[str]] = {}
        for name, module in rss._PACK_EXPORTS.items():
            provided.setdefault(module, set()).add(name)
        for kind, row in KINDS.items():
            with self.subTest(kind=kind):
                self.assertEqual(set(row.names), provided[f"robosimstudio_materials.{row.module}"])


class TestTheFlatNames(unittest.TestCase):
    def test_each_flat_name_is_the_kinds_own_object(self) -> None:
        import robosimstudio as rss

        for kind, row in KINDS.items():
            module = _kind_module(kind)
            for name in row.names:
                with self.subTest(kind=kind, name=name):
                    self.assertIs(getattr(module, name), getattr(rss, name))
                    self.assertEqual(module.__name__, rss._PACK_EXPORTS[name])
                    self.assertNotIn(name, rss._EXPORTS, f"{name} is a kind's and belongs in _PACKS")
                    self.assertIn(name, dir(rss))

    def test_each_constructor_makes_a_spec_of_its_entity(self) -> None:
        import robosimstudio as rss

        constructors = {"sheet": "cloth", "sponge": "soft", "rod": "rod", "heap": "heap", "liquid": "liquid"}
        constructors |= {"dough": "dough", "roller": "roller", "paper": "paper", "bag": "bag", "brush": "brush"}
        constructors |= {"tape": "tape", "cuttable": "cuttable", "knife": "knife", "water": "water"}
        for name, entity in constructors.items():
            with self.subTest(constructor=name):
                self.assertEqual(entity, getattr(rss, name)(name="thing").entity)

    def test_a_kinds_default_material_is_its_registered_one(self) -> None:
        """Paper's spec, made with no material, is made of ``PAPER``, the material registered as ``paper``."""
        import robosimstudio as rss
        from robosimstudio.assetlib.materials import get_material

        self.assertIs(get_material("paper").spec, rss.PaperSpec().material)


class TestTheDomains(unittest.TestCase):
    def test_each_kind_registers_its_domain_with_what_it_reads(self) -> None:
        from robosimstudio.specs.physics import get_domain, list_domains

        self.assertLessEqual(set(KINDS), set(list_domains()))
        for kind, row in KINDS.items():
            with self.subTest(kind=kind):
                domain = get_domain(kind)
                self.assertEqual(row.reads, domain.reads)
                self.assertEqual("deformables", domain.role)


class TestTheEntities(unittest.TestCase):
    def test_each_kind_registers_its_entities_under_their_names(self) -> None:
        from robosimstudio.scene.registry import ENTITIES

        for kind, row in KINDS.items():
            for name, cls in row.entities:
                with self.subTest(kind=kind, entity=name):
                    self.assertIs(_resolve(cls), ENTITIES.get(name))


class TestTheRecords(unittest.TestCase):
    def test_a_record_carries_its_domain_and_builds_the_spec_it_was_made_from(self) -> None:
        import robosimstudio as rss

        for kind, row in KINDS.items():
            record, constructor, domain = row.record
            if not record:
                continue
            with self.subTest(kind=kind):
                spec = getattr(rss, constructor)(name="thing")
                made = getattr(_kind_module(kind), record).from_spec(spec, name=f"a_{kind}")
                self.assertEqual(domain, made.sim.domain)
                self.assertIs(spec, made.build())

    def test_a_record_refuses_a_spec_of_another_kind(self) -> None:
        import robosimstudio as rss

        records = [(kind, row.record[0]) for kind, row in KINDS.items() if row.record[0]]
        for (kind, record), (other, _) in zip(records, records[1:] + records[:1], strict=True):
            with self.subTest(kind=kind, handed=other):
                cls = getattr(_kind_module(kind), record)
                with self.assertRaises(TypeError):
                    cls.from_spec(getattr(rss, KINDS[other].record[1])(name="thing"), name="not_mine")
                with self.assertRaises(TypeError):
                    cls.from_spec(rss.box(name="cube"), name="not_mine")


class TestTheMaterials(unittest.TestCase):
    def test_a_named_material_is_the_kinds_own_value_and_registers_under_it(self) -> None:
        from robosimstudio.assetlib.materials import get_material, kind_of

        for name, (kind, constant) in NAMED_MATERIALS.items():
            with self.subTest(material=name):
                spec = getattr(_kind_module(kind), constant)
                self.assertIs(spec, get_material(name).spec)
                self.assertEqual(kind, kind_of(spec))

    def test_every_registered_material_is_in_the_table(self) -> None:
        from robosimstudio.assetlib.materials import MATERIALS, kind_of

        expected = {name: kind for name, (kind, _) in NAMED_MATERIALS.items()}
        expected |= {name: kind for kind, names in KIND_OF.items() for name in names}
        got = {name: kind_of(MATERIALS.get(name).spec) for name in MATERIALS.names()}
        self.assertEqual(expected, got)


if __name__ == "__main__":
    unittest.main()

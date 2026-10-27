# SPDX-License-Identifier: Apache-2.0
"""The catalog's JSON forms: a material written and read back, and an asset as an agent reads it.

A material file is what the studio writes when somebody tunes one and what a
task loads; the claim worth holding is the round trip -- every shipped material
written to JSON and read back is the record it was, down to its type and its
hash -- and the three refusals that keep a file from loading as something it
was not: a schema version the code does not know, a kind nothing registered,
and a field the spec does not have.

An asset's catalog form is pasted into a model's prompt, so it says what was
stated and nothing else: an unset part is absent rather than empty, and a
requirement reads as the same words it was declared in.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from robosimstudio.assetlib.materials import (
    SCHEMA_VERSION,
    CalibrationSpec,
    get_material,
    kind_of,
    list_materials,
    material_from_dict,
    material_to_dict,
    read_material,
    write_material,
)

TIER = "fast"
"""Frozen dataclasses through ``json``: no model, no GPU."""


def _through_json(record):
    return material_from_dict(json.loads(json.dumps(material_to_dict(record))))


class TestAMaterialSurvivesItsFile(unittest.TestCase):
    def test_every_shipped_material_survives_a_json_round_trip(self) -> None:
        """Every kind's spec, read back from the JSON it was written as, equals the spec it was."""
        names = list_materials()
        self.assertGreater(len(names), 20, "the shipped materials did not load")
        for name in names:
            with self.subTest(material=name):
                self.assertEqual(get_material(name), _through_json(get_material(name)))

    def test_a_material_read_back_hashes_as_the_one_written(self) -> None:
        """A spec is a frozen dataclass: one read back with a list in it cannot be hashed at all."""
        for name in list_materials():
            with self.subTest(material=name):
                self.assertEqual(hash(get_material(name).spec), hash(_through_json(get_material(name)).spec))

    def test_write_then_read_is_the_record(self) -> None:
        kinds = {kind_of(get_material(name).spec): name for name in list_materials()}
        with tempfile.TemporaryDirectory() as tmp:
            for kind, name in sorted(kinds.items()):
                with self.subTest(kind=kind):
                    path = write_material(get_material(name), Path(tmp) / kind / f"{name}.json")
                    self.assertEqual(get_material(name), read_material(path))

    def test_a_calibration_comes_back_as_the_tuples_it_was(self) -> None:
        name = next(n for n in list_materials() if kind_of(get_material(n).spec) == "cloth")
        calibration = CalibrationSpec(
            fabric="denim", knob="bend_ke", target_cm=(3.1, 3.4), sweep=((0.1, 2.0), (0.2, 3.2)), notes="a probe"
        )
        record = replace(get_material(name), calibration=calibration)
        back = _through_json(record)
        self.assertEqual(calibration, back.calibration)
        self.assertIsInstance(back.calibration.sweep[0], tuple)


class TestAFileThatIsNotAMaterialIsRefused(unittest.TestCase):
    def _data(self, **changes):
        name = next(n for n in list_materials() if kind_of(get_material(n).spec) == "cloth")
        return {**material_to_dict(get_material(name)), **changes}

    def test_a_schema_version_it_does_not_know_is_refused_by_name(self) -> None:
        with self.assertRaises(ValueError) as caught:
            material_from_dict(self._data(version=SCHEMA_VERSION + 1), where="towel.json")
        self.assertIn(f"towel.json: schema version {SCHEMA_VERSION + 1}", str(caught.exception))

    def test_a_kind_nothing_registered_names_the_ones_that_are(self) -> None:
        with self.assertRaises(ValueError) as caught:
            material_from_dict(self._data(kind="jelly"))
        self.assertIn("'jelly'", str(caught.exception))
        self.assertIn("'cloth'", str(caught.exception))

    def test_a_misspelt_field_is_refused_rather_than_dropped(self) -> None:
        data = self._data()
        data["material"] = {**data["material"], "edge_kdd": 1.0}
        with self.assertRaises(ValueError) as caught:
            material_from_dict(data, where="towel.json")
        message = str(caught.exception)
        self.assertIn("towel.json.material", message)
        self.assertIn("'edge_kdd'", message)

    def test_a_misspelt_calibration_field_is_refused_where_it_is(self) -> None:
        with self.assertRaises(ValueError) as caught:
            material_from_dict(self._data(calibration={"fabrik": "denim"}))
        self.assertIn("material.calibration", str(caught.exception))

    def test_a_file_without_a_name_or_a_material_is_refused(self) -> None:
        for missing in ("name", "material"):
            with self.subTest(missing=missing):
                data = self._data()
                del data[missing]
                with self.assertRaisesRegex(ValueError, "needs both 'name' and 'material'"):
                    material_from_dict(data)

    def test_a_file_that_states_no_kind_is_a_cloth(self) -> None:
        """Written before the kinds were more than one, a material file was a cloth's."""
        data = self._data()
        del data["kind"]
        self.assertEqual("cloth", kind_of(material_from_dict(data).spec))


class TestTheCatalogFormSaysOnlyWhatWasStated(unittest.TestCase):
    def test_a_bare_record_is_four_keys(self) -> None:
        from robosimstudio.assetlib import AssetRecord, asset_to_dict

        self.assertEqual(
            {"name": "mug", "substrate": "rigid", "summary": "A mug.", "kinds": []},
            asset_to_dict(AssetRecord(name="mug", summary="A mug.")),
        )

    def test_kinds_are_listed_once_each_and_sorted(self) -> None:
        from robosimstudio.assetlib import AssetRecord, asset_to_dict
        from robosimstudio.assetlib.semantics import FeatureSpec

        features = tuple(FeatureSpec(name=f"f{i}", kind=kind) for i, kind in enumerate(("set", "grasp", "set")))
        out = asset_to_dict(AssetRecord(name="x", summary="s", features=features))
        self.assertEqual(sorted({"set", "grasp"}), out["kinds"])
        self.assertEqual(["f0", "f1", "f2"], [f["name"] for f in out["features"]])

    def test_requirements_left_unset_are_absent(self) -> None:
        from robosimstudio.specs import SimRequirements
        from robosimstudio.specs.declare import sim_to_dict

        self.assertEqual({"domain": "cloth"}, sim_to_dict(SimRequirements(domain="cloth")))

    def test_every_stated_requirement_reads_back_as_it_was_declared(self) -> None:
        from robosimstudio.specs import SimRequirements, backend, contacts_with
        from robosimstudio.specs.declare import sim_to_dict

        spec = SimRequirements(
            domain="cloth",
            contacts=contacts_with("surface", "hands", coupling="one_way"),
            contributes_hooks=("bonds",),
            self_contact=True,
            backend=backend("vbd", substeps=4, iterations=40),
            notes="measured",
        )
        out = json.loads(json.dumps(sim_to_dict(spec)))
        self.assertEqual(
            [{"with": "surface", "coupling": "one_way"}, {"with": "hands", "coupling": "one_way"}], out["contacts"]
        )
        self.assertEqual(["bonds"], out["contributes_hooks"])
        self.assertIs(True, out["self_contact"])
        self.assertEqual({"solver": "vbd", "params": {"iterations": 40}, "substeps": 4}, out["backend"])
        self.assertEqual("measured", out["notes"])


if __name__ == "__main__":
    unittest.main()

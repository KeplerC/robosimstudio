# SPDX-License-Identifier: Apache-2.0
"""The catalog's JSON forms: a material written and read back, and an asset's requirements as data.

A material file is what the studio writes when somebody tunes one and what a
task loads; the claim worth holding is the round trip -- every shipped material
written to JSON and read back is the record it was, down to its type and its
hash -- and the three refusals that keep a file from loading as something it
was not: a schema version the code does not know, a kind nothing registered,
and a field the spec does not have.
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


if __name__ == "__main__":
    unittest.main()

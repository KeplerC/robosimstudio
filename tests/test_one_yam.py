# SPDX-License-Identifier: Apache-2.0
"""There is one YAM, ``yam``; its old names say what became of them; a robot kept on the Hub is fetched on request.

``yam_bimanual`` was renamed, so it still resolves -- to ``yam``, with a warning,
and without being listed as a second robot. The composed i2rt builds were
retired rather than renamed (they stood on a different table and closed blade
tips), so asking for one is an error that names ``yam``. The Robotiq 2F-85
left git for the Hub: ``resolve_path`` finds a fetched copy in the hub cache and,
until there is one, names the command that fetches it, never downloading.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

TIER = "fast"
"""No GPU: registry lookups and path resolution."""


class TestOneYam(unittest.TestCase):
    def test_yam_is_listed_and_its_old_names_are_not(self) -> None:
        from robosimstudio.robots.registry import list_embodiments

        names = list_embodiments()
        self.assertIn("yam", names)
        self.assertIn("yam_suction", names)
        for old in ("yam_bimanual", "yam_bimanual_suction", "yam_i2rt", "yam_i2rt_arena", "yam_i2rt_arena_narrow"):
            self.assertNotIn(old, names)

    def test_the_old_name_resolves_to_yam_with_a_warning(self) -> None:
        from robosimstudio.robots.registry import EMBODIMENTS, get_embodiment

        EMBODIMENTS._warned.discard("yam_bimanual")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spec = get_embodiment("yam_bimanual")
        self.assertIs(get_embodiment("yam"), spec)
        self.assertTrue(any(issubclass(w.category, FutureWarning) and "'yam'" in str(w.message) for w in caught))

    def test_a_retired_robot_or_hand_names_yam(self) -> None:
        from robosimstudio.robots.registry import get_embodiment, get_gripper

        for name in ("yam_i2rt", "yam_i2rt_arena", "yam_i2rt_arena_narrow"):
            with self.subTest(robot=name), self.assertRaisesRegex(KeyError, r"retired.*'yam'"):
                get_embodiment(name)
        with self.assertRaisesRegex(KeyError, r"retired.*'yam'"):
            get_gripper("linear_4310")

    def test_the_old_module_still_imports(self) -> None:
        from robosimstudio.robots.embodiments.yam import YAM
        from robosimstudio.robots.embodiments.yam_bimanual import YAM_BIMANUAL

        self.assertIs(YAM, YAM_BIMANUAL)


class TestTheRobotiqIsFetchedOnRequest(unittest.TestCase):
    PATH = "embodiments/robotiq_2f85/2f85.xml"

    def test_until_it_is_fetched_the_error_names_the_command(self) -> None:
        from robosimstudio.paths import resolve_path

        with tempfile.TemporaryDirectory() as empty, mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_HUB_CACHE": empty}):
            with self.assertRaisesRegex(FileNotFoundError, r"python -m robosimstudio\.hub fetch robots/robotiq_2f85"):
                resolve_path(self.PATH)

    def test_a_fetched_copy_resolves_from_the_hub_cache(self) -> None:
        from robosimstudio import hub
        from robosimstudio.paths import resolve_path

        with tempfile.TemporaryDirectory() as cache, mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_HUB_CACHE": cache}):
            fetched = Path(cache) / hub.REVISION / "robots" / "robotiq_2f85" / "2f85.xml"
            fetched.parent.mkdir(parents=True)
            fetched.write_text("<mujoco/>", encoding="utf-8")
            self.assertEqual(fetched.resolve(), resolve_path(self.PATH))


if __name__ == "__main__":
    unittest.main()

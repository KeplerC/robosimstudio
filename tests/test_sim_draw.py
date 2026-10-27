# SPDX-License-Identifier: Apache-2.0
"""``rss.view`` and the page it serves: the config it is handed, and the tab's name.

``view`` keeps the fields of a config it is handed and lays its own keywords on
top; every page this package serves names its browser tab RoboSimStudio, not
Viser.
"""

from __future__ import annotations

import dataclasses
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from robosimstudio.sim.draw import title_page

TIER = "fast"
"""CPU only: a stand-in studio session and a stand-in viser server."""


@dataclasses.dataclass
class _Config:
    panels: object = "all"
    render_every: int = 4
    port: int = 8080
    share: bool = False
    handles: int = 1
    speed: str = "1x"
    dark_mode: bool = False
    cloth_look: str = "fluffy"
    lighting: object = None
    show_scenery: bool = False


class TestViewKeepsTheConfigItIsGiven(unittest.TestCase):
    def _view(self, **kwargs) -> _Config:
        from robosimstudio.sim.draw import view

        made: list[_Config] = []

        class Session:
            def __init__(self, sim, *, config) -> None:
                made.append(config)

            def run(self) -> None:
                pass

        fake = types.SimpleNamespace(StudioConfig=_Config, StudioSession=Session)
        with mock.patch.dict(sys.modules, {"robosimstudio_studio.session": fake}):
            view(object(), **kwargs)
        return made[0]

    def test_no_config_gets_this_doors_defaults_and_no_tunnel(self) -> None:
        config = self._view()
        self.assertEqual((config.panels, config.render_every, config.share), ("drive", 1, False))

    def test_a_given_config_keeps_its_fields(self) -> None:
        config = self._view(config=_Config(panels="all", render_every=4, share=True))
        self.assertEqual((config.panels, config.render_every, config.share), ("all", 4, True))

    def test_a_keyword_still_lands_on_top_of_it(self) -> None:
        config = self._view(config=_Config(), panels="drive", share=True)
        self.assertEqual((config.panels, config.render_every, config.share), ("drive", 4, True))


class TestTheTabIsNamed(unittest.TestCase):
    def test_the_title_is_queued_as_a_script_every_browser_runs(self) -> None:
        from viser import _messages

        queued = []
        server = SimpleNamespace(_websock_server=SimpleNamespace(queue_message=queued.append))
        title_page(server)
        (message,) = queued
        self.assertIsInstance(message, _messages.RunJavascriptMessage)
        self.assertEqual(message.source, 'document.title = "RoboSimStudio";')


if __name__ == "__main__":
    unittest.main()

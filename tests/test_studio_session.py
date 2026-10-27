# SPDX-License-Identifier: Apache-2.0
"""The studio session without a scene: a reset that recompiles rebinds the page, scenery re-reads the picker."""

from __future__ import annotations

import types
import unittest
import unittest.mock
from unittest import mock

from robosimstudio_studio import session as session_mod
from robosimstudio_studio.session import StudioSession

TIER = "fast"
"""No GPU: sessions built without their constructors."""


class TestAResetThatRecompilesRebindsThePage(unittest.TestCase):
    def _session(self, swap: bool):
        from robosimstudio_studio.session import StudioSession

        session = StudioSession.__new__(StudioSession)
        env = types.SimpleNamespace(model=object())

        def reset():
            if swap:
                env.model = object()
            return {}, {}

        env.reset = reset
        session.env = env
        session.panels = []
        session._drop_panels = unittest.mock.Mock()
        session._bind = unittest.mock.Mock()
        session.on_reset = unittest.mock.Mock()
        return session

    def test_a_new_model_rebuilds_the_page_around_it(self) -> None:
        session = self._session(swap=True)
        self.assertTrue(session.reset_now())
        session._drop_panels.assert_called_once_with()
        session._bind.assert_called_once_with(session.env)
        session.on_reset.assert_called_once()

    def test_the_same_model_leaves_the_page_alone(self) -> None:
        session = self._session(swap=False)
        self.assertTrue(session.reset_now())
        session._bind.assert_not_called()
        session._drop_panels.assert_not_called()


class SceneryInvalidatesPicker(unittest.TestCase):
    def test_apply_scenery_invalidates_the_picker(self):
        stub = mock.MagicMock()
        with (
            mock.patch.object(session_mod, "garment_visuals", return_value=()),
            mock.patch.object(session_mod, "scene_repaints_live", return_value=False),
        ):
            StudioSession.apply_scenery(stub)
        stub._set_scenery_visible.assert_called_once()
        stub.picker.invalidate.assert_called_once_with()

    def test_no_picker_is_fine(self):
        stub = mock.MagicMock()
        stub.picker = None
        with (
            mock.patch.object(session_mod, "garment_visuals", return_value=()),
            mock.patch.object(session_mod, "scene_repaints_live", return_value=False),
        ):
            StudioSession.apply_scenery(stub)


if __name__ == "__main__":
    unittest.main()

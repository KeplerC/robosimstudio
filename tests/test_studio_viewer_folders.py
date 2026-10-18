# SPDX-License-Identifier: Apache-2.0
"""A studio page's sidebar starts with the page's own folders, not Newton's.

Newton 1.7's ``ViewerViser`` opens a **Simulation** folder (Pause, Step, Reset) and a
fifteen-checkbox **Visualization** folder, both expanded, above everything a page adds. The
studio steps itself, so that Pause did not pause and that Reset had nothing to call, and the
Visualization folder pushed the example readouts below the fold: the press-dough clip filmed on
1.7 never showed its height readout. ``ViewerStudio`` adds neither. The bare Newton viewer is
built beside it, so the day Newton stops adding them this test says so rather than passing on
nothing.
"""

from __future__ import annotations

import unittest

TIER = "fast"

STUDIO_PORT = 9811
NEWTON_PORT = 9812
NEWTONS = {"Pause", "Step", "Reset", "Show Visual", "Show Collision", "Enable Picking"}
"""Some of what Newton 1.7 puts in its two folders, by label."""


def _sidebar(viewer) -> tuple[list[str], set[str]]:
    """The folder titles and the input labels a viewer's server holds."""
    gui = viewer._server.gui
    folders = [getattr(h, "label", None) for h in gui._container_handle_from_uuid.values()]
    inputs = {getattr(h, "label", "") for h in gui._gui_input_handle_from_uuid.values()}
    return [f for f in folders if f], inputs


class TestNewtonsFoldersStayOffTheStudioPage(unittest.TestCase):
    def test_the_studio_viewer_adds_neither_folder(self) -> None:
        from newton.viewer import ViewerViser

        from robosimstudio_studio.garments import viewer_studio

        bare = ViewerViser(port=NEWTON_PORT)
        try:
            folders, inputs = _sidebar(bare)
        finally:
            bare.close()
        self.assertIn("Visualization", folders, "Newton's own viewer opens a Visualization folder")
        self.assertTrue(NEWTONS <= inputs, sorted(NEWTONS - inputs))

        studio = viewer_studio(port=STUDIO_PORT, share=False)
        try:
            folders, inputs = _sidebar(studio)
            self.assertEqual([], folders, "a studio viewer opens no folder of its own")
            self.assertFalse(NEWTONS & inputs, sorted(NEWTONS & inputs))
            studio._sync_gui_controls()  # Newton's per-frame sync, with no handles to sync
        finally:
            studio.close()


if __name__ == "__main__":
    unittest.main()

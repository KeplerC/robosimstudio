# SPDX-License-Identifier: Apache-2.0
"""A studio page's sidebar starts with the page's own folders, not Newton's.

Newton 1.7's ``ViewerViser`` opens a **Simulation** folder (Pause, Step, Reset) and a
fifteen-checkbox **Visualization** folder, both expanded, above everything a page adds. The
studio steps itself, so that Pause did not pause and that Reset had nothing to call, and the
Visualization folder pushed the example readouts below the fold: the press-dough clip filmed on
1.7 never showed its height readout. ``ViewerStudio`` adds neither. The bare Newton viewer is
built beside it, so the day Newton stops adding them this test says so rather than passing on
nothing. The core tier's ``rss.viewer`` and ``rss.draw`` pages keep Newton's Simulation folder, the
only Pause and Reset they have: its Reset, greyed out with no callback, calls ``sim.reset()``.
"""

from __future__ import annotations

import unittest

TIER = "fast"

STUDIO_PORT = 9811
NEWTON_PORT = 9812
BARE_PORT = 9813
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


class TestTheBareViewersSimulationFolderDrivesTheScene(unittest.TestCase):
    """``rss.viewer`` and ``rss.draw`` keep Newton's Simulation folder: no other Pause or Reset is on them."""

    def test_reset_resets_the_scene_and_the_documented_loop_asks_before_it_steps(self) -> None:
        from types import SimpleNamespace

        import newton

        from robosimstudio.sim.draw import Viewer, viewer

        resets: list[int] = []
        sim = SimpleNamespace(
            model=newton.ModelBuilder().finalize(device="cpu"), name="bare", reset=lambda: resets.append(1)
        )
        page = Viewer(sim, port=BARE_PORT)
        try:
            self.assertFalse(page._viewer._simulation_gui_handles["reset"].disabled, "Reset is live")
            page._viewer._interaction_events.put(("reset",))  # what a click on it queues
            self.assertTrue(page.should_step())
            self.assertEqual([1], resets)
        finally:
            page.close()
        self.assertIn("if viewer.should_step():", viewer.__doc__, "the loop the docs show honours Pause")


if __name__ == "__main__":
    unittest.main()

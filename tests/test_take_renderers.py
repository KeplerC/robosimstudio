# SPDX-License-Identifier: Apache-2.0
"""``play --record TAKE --also R,R``: one run filmed by the page's renderer and drawn by each other renderer named.

Needs the GPU: it builds a scene and draws it with gl and the tiled tracer.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

TIER = "medium"


def _frames(path: Path) -> int:
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        return sum(1 for _ in reader)
    finally:
        reader.close()


class AlsoRenderers(unittest.TestCase):
    def test_one_run_is_drawn_by_each_renderer(self) -> None:
        import robosimstudio as rss
        from robosimstudio_real2sim_agent import checks

        scene = rss.Scene(cell="bench_075", name="take_renderers")
        scene.add(rss.box(name="cube", half=(0.02,) * 3, at=(0.0, 0.0, 0.2)))  # falls onto the bench as it is filmed
        with mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_RENDERER": "gl"}), redirect_stdout(io.StringIO()):
            sim = scene.build()
            with tempfile.TemporaryDirectory() as tmp:
                server = types.SimpleNamespace(sim=sim, scene=scene, run_dir=tmp)
                rec = checks.Recorder(server, "fall", every=2, also="tiled,gl")  # gl is the page's own: filmed once
                self.assertEqual(list(rec.also), ["tiled"])
                rec.probe()
                rec.start()
                for _ in range(10):
                    sim.run(1)
                    rec.maybe()
                out = rec.finish()
                meta = json.loads((Path(tmp) / "fall.json").read_text())
                take, tiled, grid = Path(out["video"]), Path(out["also"]["tiled"]), Path(out["renderers"])
                n = len(meta["frames"])
                self.assertGreater(n, 3)
                self.assertEqual((_frames(take), _frames(tiled), _frames(grid)), (n, n, n), "one frame each per state")
                import imageio.v2 as imageio

                first = imageio.get_reader(str(grid), "ffmpeg").get_data(0)
                self.assertEqual(first.shape[1], 2 * int(rec.cam.width), "the two renderers side by side")
                self.assertTrue(np.isfinite(first).all())

    def test_an_unknown_renderer_is_refused_before_filming(self) -> None:
        import robosimstudio as rss
        from robosimstudio_real2sim_agent import checks

        scene = rss.Scene(cell="bench_075", name="take_renderers_refused")
        scene.add(rss.box(name="cube", half=(0.02,) * 3, at=(0.0, 0.0, 0.03)))
        with mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_RENDERER": "gl"}), redirect_stdout(io.StringIO()):
            sim = scene.build()
            with tempfile.TemporaryDirectory() as tmp:
                server = types.SimpleNamespace(sim=sim, scene=scene, run_dir=tmp)
                with self.assertRaises(ValueError):
                    checks.Recorder(server, "nope", also="povray")
                self.assertFalse(any(Path(tmp).iterdir()), "nothing written for a take refused")


if __name__ == "__main__":
    unittest.main()

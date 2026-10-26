# SPDX-License-Identifier: Apache-2.0
"""Which renderers draw on this machine, how fast, and whether they agree on where things are.

A machine with no NVIDIA GPU -- a Mac -- steps scenes on the CPU and cannot
run ``rtx``; this says what it can do instead. It builds a small bench (two
YAMs, a cube, a cloth) where the machine puts it, steps it, and asks each
renderer for a 640x360 frame through a calibrated camera (principal point off
centre) of one state: how long the first frame took (building the renderer)
and the next, and how far its depth is from the tiled tracer's, which every
machine runs.
The frames go to ``--out``.

    uv run python scripts/check_renderers.py                       # every renderer
    uv run python scripts/check_renderers.py --renderers gl,blender --out /tmp/frames

On a Mac: ``gl`` opens Apple's CGL headless (``MUJOCO_GL=glfw`` for a window
instead), and ``blender`` finds ``/Applications/Blender.app`` by itself and
renders Cycles on Metal (``ROBOSIMSTUDIO_BLENDER_DEVICE=cpu`` for the CPU).
What the live page then needs is ``./ctl up --renderer gl`` or
``--renderer blender`` in a twin's folder.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--renderers", default="tiled,gl,blender,rtx", help="comma-separated, tiled first")
    parser.add_argument("--out", default="outputs/check_renderers", help="where the frames go")
    parser.add_argument("-v", "--verbose", action="store_true", help="the traceback of each renderer that fails")
    args = parser.parse_args(argv)

    import robosimstudio as rss
    from robosimstudio.machine import platform as machine_platform
    from robosimstudio_materials.cloth import sheet

    print(f"[check] {machine_platform()}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    scene = rss.Scene(cell="bench_075", name="check")
    scene.add(rss.robot("yam_bimanual"))
    scene.add(rss.box(name="cube", half=(0.03, 0.03, 0.03), at=(0.05, 0.08, 0.03), color=(0.2, 0.4, 0.9)))
    scene.add(sheet(name="towel", at=(-0.15, -0.15, 0.06), size=(0.14, 0.14), cells=(10, 10), color=(0.8, 0.2, 0.2)))
    cam = rss.CameraSpec(
        name="cam",
        width=640,
        height=360,
        fx=470.0,
        fy=470.0,
        cx=300.0,
        cy=190.0,
        eye=(0.15, -0.55, 1.25),
        look_at=(0.6, 0.0, 0.76),
    )
    scene.add(cam)
    sim = scene.build()
    built = time.perf_counter() - t
    t = time.perf_counter()
    sim.run(60).sync()
    stepped = time.perf_counter() - t
    print(f"[check] built in {built:.1f} s; 60 control steps, {sim.time:.2f} s simulated, in {stepped:.1f} s")

    reference = None
    rows = []
    for name in [r.strip() for r in args.renderers.split(",") if r.strip()]:
        try:
            t = time.perf_counter()
            renderer = sim.make_renderer(name, cameras=[cam], channels=["rgb", "depth"])
            try:
                renderer.render(sim.state)
                first = time.perf_counter() - t
                t = time.perf_counter()  # the same state for every renderer, so the depths compare
                frame = renderer.render(sim.state)["cam"]
                second = time.perf_counter() - t
            finally:
                renderer.close()
        except Exception as exc:  # what this machine cannot run, said and passed over
            rows.append((name, "no", f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"))
            if args.verbose:
                traceback.print_exc()
            continue
        from PIL import Image

        Image.fromarray(frame["rgb"]).save(out / f"{name}.png")
        note = f"first frame {first:.2f} s, next {second:.3f} s, mean {frame['rgb'].mean():.0f}"
        depth = frame["depth"]
        if reference is None and name == "tiled":
            reference = np.where(depth <= cam.far, depth, 0.0)
        elif reference is not None:
            both = (reference > 0) & (depth > 0)
            if both.any():
                gap = float(np.median(np.abs(reference[both] - depth[both]))) * 1000.0
                note += (
                    f", depth {gap:.2f} mm from tiled's, {((reference > 0) == (depth > 0)).mean():.1%} of hits agree"
                )
        rows.append((name, "yes", note))
    sim.close()
    width = max(len(r[0]) for r in rows) if rows else 8
    for name, ok, note in rows:
        print(f"[check] {name:<{width}}  {ok:<3}  {note}")
    print(f"[check] frames in {out.resolve()}")
    return 0 if any(ok == "yes" for _, ok, _ in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

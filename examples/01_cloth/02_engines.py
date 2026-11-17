# SPDX-License-Identifier: Apache-2.0
"""Cloth on another engine

The same sheet on Newton, SuperDex or Genesis: change one string at the top of ``main()`` and run it again.

The spec is unchanged and the arm stays on MuJoCo; the printed island table names the solver
each thing runs on. ``newton`` runs on a base install; ``[superdex]`` (Python 3.12) and
``[genesis]`` are optional extras.
"""

import robosimstudio as rss

SOLVERS = {"newton": "vbd", "superdex": "superdex", "genesis": "genesis"}
"""Which solver each engine name means. Newton is the host; VBD is its solver for deformables."""


def main():
    engine = "newton"  # or "superdex", "genesis"

    cell = "bench_075"
    embodiment = "yam"
    size = 0.3
    cells = 20

    # Foreign engines take 2 substeps per control step.
    cloth = rss.backend("vbd") if engine == "newton" else rss.backend(SOLVERS[engine], substeps=2)
    physics = rss.replace(
        rss.GENERIC,
        backends=(("articulated", rss.backend("mujoco")), ("cloth", cloth), ("*", rss.backend("vbd"))),
    )

    scene = rss.Scene(cell=cell, physics=physics, name=f"cloth_{engine}")
    scene.add(rss.robot(embodiment))
    # ``at`` places the sheet's origin corner, not its middle.
    scene.add(rss.sheet(name="towel", size=(size, size), cells=(cells, cells), at=(-size / 2, -size / 2, 0.06)))
    sim = scene.build(control_mode="joint_position")
    print(sim.pipeline.program.summary())

    rss.view(sim)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Hook a band, scored

The same pegboard and band as a task document, with a goal that the band encircles both pegs: hook it from the upper peg onto the lower, and the goal turns true.

The pegs are the example's posts declared with a ``rod`` feature, their two
ends, so ``encircles`` holds the band to each peg's length: hanging in front of a
peg's tip, round only the line through it, does not count. The embodiment is the
YAM with ``03_materials/04_band``'s right-arm home, the hand in front of the band, and
``tests/test_band_on_pegs`` drives the hook there: the goal turns true. Needs ``[superdex]``.
"""

import contextlib

import robosimstudio as rss
import robosimstudio_bench as bench
from robosimstudio.assetlib import register_asset
from robosimstudio.assetlib.semantics.features import FeatureSpec
from robosimstudio.domains.rigid import Body
from robosimstudio.robots.registry import get_embodiment, register_embodiment
from robosimstudio_materials.rod import Rod
from robosimstudio_materials.rod.band import RUBBER, band

TASK = """
(define (problem band_on_pegs)
  (:language hook the rubber band from the upper peg onto the lower one)
  (:staging (:scene bench_075) (:embodiment yam_pegboard) (:physics band_superdex@v1))
  (:fixtures board - board upper - upper_peg lower - lower_peg tip - peg_tip)
  (:objects band - rubber_band)
  (:goal (and (encircles band upper.span) (encircles band lower.span))))
"""


def main() -> None:
    face, y, side = 0.10, -0.20, 0.006  # the board's face, the pegs' line, each peg's radius [m]
    pegs = {"upper": (0.26, 0.08), "lower": (0.135, 0.05)}  # height over the bench, length out of the board [m]
    diameter = 0.08  # the band at rest [m]
    in_front = (0.5819, 1.1118, 0.3863, 0.7255, 0.5819, 0.0)  # the right arm's home, as in 03_materials/04_band [rad]

    yam = get_embodiment("yam_bimanual")
    arms = (yam.arms[0], rss.replace(yam.arms[1], home_q=in_front))
    with contextlib.suppress(ValueError):  # registered by an earlier run in this process
        register_embodiment(rss.replace(yam, name="yam_pegboard", arms=arms))

    engine = rss.backend("superdex", substeps=2, feedback="off")
    physics = rss.replace(rss.GENERIC, backends=(("rod", engine), *rss.GENERIC.backends))
    bench.register_arrangement("band_superdex@v1", physics)
    board = rss.slab(name="board", half=(0.005, 0.012, 0.15), at=(face + 0.005, y, 0.15))
    register_asset(Body.from_spec(board, name="board"))
    along_x = (0.0, 0.70710678, 0.0, 0.70710678)  # a post's axis turned from z onto x
    for name, (z, length) in pegs.items():
        peg = rss.post(name=name, radius=side, half_height=length / 2, at=(face - length / 2, y, z, *along_x))
        ends = [[0, 0, -length / 2], [0, 0, length / 2]]  # tip and root, in the post's own frame (its axis is z)
        span = FeatureSpec(name="span", kind="rod", payload={"axis": ends, "radius": side})
        register_asset(Body.from_spec(peg, name=f"{name}_peg", features=(span,)))
    z, length = pegs["lower"]  # its tip turns up 18 mm, as a pegboard hook's does
    tip = rss.post(name="tip", radius=0.004, half_height=0.009, at=(face - length + 0.004, y, z + 0.009))
    register_asset(Body.from_spec(tip, name="peg_tip"))
    hang = (0.035, y, pegs["upper"][0] + side + RUBBER.cord_radius + 0.001 - diameter / 2)
    loop = band(name="band", diameter=diameter, at=hang, normal=(1, 0, 0), segments=64, skin_samples="P1Q6")
    register_asset(Rod.from_spec(loop, name="rubber_band"))

    env = bench.Env(bench.task(TASK))
    env.reset()
    rss.view(env)


if __name__ == "__main__":
    main()

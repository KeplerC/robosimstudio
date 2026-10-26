# SPDX-License-Identifier: Apache-2.0
"""Cut tofu, scored

The same knife, rest, board and tofu as a task document, with a goal that the tofu is in two pieces: take the knife and push it down through the tofu as in ``03_materials/10_cutting``, and the goal turns true.

The tofu is a ``Cuttable.from_spec`` record, the knife a ``Cuttable.knife`` one carrying its edge as a
``blade`` feature, and the rest and the board are fixtures. The embodiment is the YAM with
``03_materials/10_cutting``'s right-arm home, the hand over the handle, and the physics ``rss.CUTTING_50HZ``
under a task-language name. Before the hand comes the goal is unmet; ``tests/test_cut_tofu_scored``
drives the cut through the page and ``evaluate`` reads ``pieces`` 2 and says success.
"""

import contextlib

import robosimstudio as rss
import robosimstudio_bench as bench
from robosimstudio.assetlib import register_asset
from robosimstudio.domains.rigid import Body
from robosimstudio.robots.registry import get_embodiment, register_embodiment
from robosimstudio_materials.cutting import Cuttable

TASK = """
(define (problem cut_tofu)
  (:language cut the tofu in two with the knife)
  (:staging (:scene bench_075) (:embodiment yam_knife) (:physics cutting_50hz@v1)
            (:gripper-actuation (:ke 1500) (:kd 60)))
  (:fixtures rest_a - knife_rest_a rest_b - knife_rest_b board - cutting_board)
  (:objects knife - kitchen_knife tofu - firm_tofu)
  (:goal (>= (pieces tofu) 2)))
"""


def main() -> None:
    knife_y, tofu_y, blade_x = -0.31, -0.16, 0.04  # as in 03_materials/10_cutting [m]
    over_knife = (0.0, 1.792, 1.376, -1.154, 0.0, 0.0)  # the right arm's home, as in 03_materials/10_cutting [rad]

    yam = get_embodiment("yam_bimanual")
    with contextlib.suppress(ValueError):  # registered by an earlier run in this process
        register_embodiment(
            rss.replace(yam, name="yam_knife", arms=(yam.arms[0], rss.replace(yam.arms[1], home_q=over_knife)))
        )
    bench.register_arrangement("cutting_50hz@v1", rss.CUTTING_50HZ)
    knife = rss.knife(name="knife", at=(blade_x - 0.105, knife_y, 0.04))
    register_asset(Cuttable.knife(knife, name="kitchen_knife"))
    walls = (0.05, 0.004, 0.012)
    for name, side in (("a", -1.0), ("b", 1.0)):
        wall = rss.slab(
            name=f"rest_{name}", half=walls, at=(blade_x, knife_y + side * 0.00625, walls[2]), color=(0.35, 0.25, 0.18)
        )
        register_asset(Body.from_spec(wall, name=f"knife_rest_{name}"))
    board = rss.slab(name="board", half=(0.07, 0.08, 0.006), at=(blade_x, tofu_y, 0.006), color=(0.62, 0.45, 0.28))
    register_asset(Body.from_spec(board, name="cutting_board"))
    tofu = rss.cuttable(
        name="tofu", knife=knife, size=(0.06, 0.10, 0.04), at=(blade_x, tofu_y, 0.012), material=rss.cuttables.TOFU
    )
    register_asset(Cuttable.from_spec(tofu, name="firm_tofu"))

    env = bench.Env(bench.task(TASK))
    env.reset()
    rss.view(env)


if __name__ == "__main__":
    main()

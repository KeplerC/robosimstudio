# SPDX-License-Identifier: Apache-2.0
"""Pick up a plate, scored

The same plate, cup and square as a task document, with a goal on the plate resting in the square: carry it over, untick **Vacuum**, lift the hand away, and the goal turns true.

The plate is a ``Body.from_spec`` record the document places under its own
name, the robot is ``yam_bimanual_suction`` and the physics the one MuJoCo
island under its task-language name, ``mujoco_500hz@v1``. The square's height
window, ``(:z-range 0.0 0.01)``, counts the plate's centre only within 10 mm of
the bench, so a plate carried over the square does not count, and
``hands-clear`` asks every tool centre to be 40 mm from the plate's, so a plate
still held down on the square does not either. ``tests/test_suction_pick``
drives the hand through the pick and the goal turns true.
"""

import robosimstudio as rss
import robosimstudio_bench as bench
from robosimstudio.assetlib import register_asset
from robosimstudio.domains.rigid import Body

TASK = """
(define (problem pick_a_plate)
  (:language put the plate in the square)
  (:staging (:scene bench_075) (:embodiment yam_bimanual_suction) (:physics mujoco_500hz@v1))
  (:objects plate - plate_150mm)
  (:regions (square (:ranges ((-0.11 -0.22 0.11 0.0))) (:z-range 0.0 0.01)))
  (:goal (once (hold-for 0.5 (and (>= (containment plate square) 1.0) (<= (speed plate) 0.02)
                                  (> (hands-clear plate) 0.04))))))
"""


def main() -> None:
    # The document places a "plate_150mm"; this says what one is: 150 x 150 x 6 mm, 0.3 kg, under the cup.
    plate = rss.box(
        name="plate", half=(0.075, 0.075, 0.003), at=(0.0, -0.31, 0.003), mass=0.3, color=(0.88, 0.88, 0.85)
    )
    register_asset(Body.from_spec(plate, name="plate_150mm"))

    env = bench.Env(bench.task(TASK))
    env.reset()
    rss.view(env)


if __name__ == "__main__":
    main()

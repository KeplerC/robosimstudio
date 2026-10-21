# SPDX-License-Identifier: Apache-2.0
"""Many demos

Film scripted bimanual YAM demos in parallel: five 1280x720 cameras at 20 Hz.

``collect`` starts worker processes that each build the scene once and run
their share of the episodes. Each worker steps the physics on a CPU core
(MuJoCo's C library, one call per control period), rasterises the cameras with
the ``gl`` renderer on a GPU, and encodes each camera in its own ``ffmpeg``
while the next period steps. A 20 s demo takes about two seconds in one worker.

``make_policy`` gets the episode number after each reset and returns ``act``,
which gives one ``joint_position`` action per control step. This one reaches
the right arm out and back, a little differently every episode, and solves
nothing: it is where your scripted demo goes. Each episode lands in
``outputs/many_demos/<episode>/``, with ``cameras/<name>.mp4``, the actions and
joint positions in ``steps.npz``, and the body poses behind every frame in
``privileged.npz``. For hundreds, ``python -m robosimstudio.record.collect``
runs the same two references from the shell; the line is below.
"""

from pathlib import Path

import numpy as np

import robosimstudio as rss
from robosimstudio.record.collect import CollectSpec, collect

W, H = 1280, 720


def build() -> rss.Scene:
    """The bench, the two arms, a cube, and the rig: three fixed cameras and one on each wrist."""
    scene = rss.Scene(cell="bench_075", physics=rss.MUJOCO_500HZ)
    scene.add(rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.0, 0.1, 0.03)))
    scene.add(rss.robot("yam_bimanual"))
    for camera in (
        rss.CameraSpec(name="overhead", width=W, height=H, fov_y=60.0, eye=(0.0, -0.25, 1.75), look_at=(0.0, 0.0, 0.75)),
        rss.CameraSpec(name="side_left", width=W, height=H, fov_y=60.0, eye=(0.45, -0.75, 1.2), look_at=(0.0, 0.0, 0.8)),
        rss.CameraSpec(name="side_right", width=W, height=H, fov_y=60.0, eye=(-0.45, -0.75, 1.2), look_at=(0.0, 0.0, 0.8)),
        rss.CameraSpec(name="left_wrist", width=W, height=H, fov_y=80.0, parent_frame="left", xform=(0, 0, 0.05, 0, 0, 0, 1)),
        rss.CameraSpec(name="right_wrist", width=W, height=H, fov_y=80.0, parent_frame="right", xform=(0, 0, 0.05, 0, 0, 0, 1)),
    ):  # fmt: skip
        scene.add(camera)
    return scene


def make_policy(sim, episode: int):
    """Reach the right arm out and back every two seconds, closing and opening its jaws at the far end."""
    home = np.asarray(sim.idle_action("joint_position"), dtype=np.float32)
    per_arm = sim.controllers["joint_position"].per_arm
    right = slice(per_arm, 2 * per_arm - 1)  # the right arm's joints; its gripper is the slot after
    reach = home.copy()
    reach[right] += np.random.default_rng(episode).uniform(-0.4, 0.4, per_arm - 1).astype(np.float32)
    period = int(2.0 * sim.request.physics.control_hz)

    def act(sim, step: int):
        phase = (step % period) / period
        out = home + (reach - home) * float(0.5 - 0.5 * np.cos(2.0 * np.pi * phase))
        out[2 * per_arm - 1] = 1.0 if 0.4 < phase < 0.6 else 0.0  # shut at the far end
        return out

    return act


def main() -> None:
    episodes, workers = 8, 4
    here = Path(__file__).resolve()
    summary = collect(
        CollectSpec(
            scene=f"{here}:build",
            policy=f"{here}:make_policy",
            out="outputs/many_demos",
            episodes=episodes,
            workers=workers,
        )
    )
    print(f"{summary['ran']} demos at {summary['episodes_per_min']:.0f} a minute, in outputs/many_demos")


if __name__ == "__main__":
    main()

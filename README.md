# <img alt="" src=".github/media/logo.png" height="44" align="top" /> RoboSimStudio

[Tech report](https://arxiv.org/abs/2610.18835) | [Project Page](https://robosimstudio.github.io/)

RoboSimStudio is an open, interactive simulation toolkit for rebuilding real
robot manipulation tasks in simulation, by hand or with a coding agent. It runs
on an NVIDIA GPU, or on a CPU alone.

<p align="center">
  <img alt="Forty-eight manipulation tasks built with RoboSimStudio, playing side by side" src=".github/media/splash.webp" width="100%" />
</p>

## Install

From a checkout, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/robosimstudio/robosimstudio.git && cd robosimstudio
uv run examples/00_getting_started/00_hello_world.py      # with an NVIDIA GPU

export UV_NO_GROUP=rtx                                     # without one: leave out the 1.8 GB GPU path tracer
uv run --extra cpu examples/00_getting_started/00_hello_world.py
```

Or with pip: `pip install "robosimstudio @ git+https://github.com/robosimstudio/robosimstudio"`,
and `"robosimstudio[cpu] @ git+..."` on a machine without a GPU.

Nothing else changes between the two. A scene picks its device when it is built
and prints what it picked, and you can ask before building anything:

```console
$ python -m robosimstudio.machine
cpu (no CUDA device) · CUDA graphs off · renderer tiled (cpu is not a CUDA device)
```

On a CPU, MuJoCo's C library steps the arm and the rigid bodies, SuperDex steps
cloth, rope and soft bodies, and Newton's tiled tracer renders. A scene left on
the default physics runs as `rss.CPU`, and a solver that has no CPU path is
replaced, with the change printed. A CPU runs one world per process. It cannot
use signed-distance-field contact (`full_surface_contact` against a mesh,
`hydroelastic`). SuperDex needs Python 3.12.

You can also choose instead of letting it pick:
- `scene.build(device="cpu")` in code.
- `ROBOSIMSTUDIO_DEVICE` for a whole shell: `auto`, `cpu`, `cuda` or `cuda:1`.
- `ROBOSIMSTUDIO_RENDERER` likewise: `auto`, `rtx` or `tiled`.

## Quick start

We provide comprehensive examples in [`examples/`](https://github.com/robosimstudio/robosimstudio/tree/main/examples) to help you get started quickly. Alternatively, you can use a coding agent such as Codex or Claude Code to create a simulation environment by simply providing the following prompt:

```
Set up environment for https://github.com/robosimstudio/robosimstudio
and create a simulation environment for [Your task instruction]
based on the video at [Your Video Path].
```

## Acknowledgements

RoboSimStudio is built on [Newton](https://github.com/newton-physics/newton),
[Warp](https://github.com/NVIDIA/warp) and
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), and its studio
on [viser](https://github.com/viser-project/viser). It draws on
[Isaac Lab](https://github.com/isaac-sim/IsaacLab),
[Isaac Lab Arena](https://github.com/isaac-sim/IsaacLab-Arena),
[PyRoKi](https://github.com/chungmin99/pyroki),
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis),
[SuperDex](https://github.com/facebookresearch/project_superdex) and
[LeRobot](https://github.com/huggingface/lerobot). Robot models come from
[i2rt](https://github.com/i2rt-robotics/i2rt) and
[robosuite](https://github.com/ARISE-Initiative/robosuite); see `NOTICE`.
Thanks to their authors.

## Citation

```bibtex
@misc{robosimstudio2026,
  title         = {RoboSimStudio: An Interactive Toolkit for Rapid Real2Sim of Robot Manipulation Tasks},
  author        = {Chen, Kaiyuan and Xie, Shuangyu and Yoo, Uksang and Zhu, Xinyue and Magner, Cael and Zhu, Alvin and Kim, Daehwa and Heiden, Eric and Coumans, Erwin and Macklin, Miles and Huang, Spencer and Goldberg, Ken},
  year          = {2026},
  eprint        = {2610.18835},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2610.18835},
}
```

## License

RoboSimStudio is released under the [Apache License 2.0](LICENSE).

# Installation

This guide covers the recommended ways to install RoboSimStudio: with pip into an
environment of your own, or with [uv](https://docs.astral.sh/uv/) from a checkout
of the repository, which is also how the examples run. For working on
RoboSimStudio itself, see [Contributing](../CONTRIBUTING.md).

## System Requirements

### Minimum Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.12 | 3.12 is what CI tests, and the only version SuperDex ships wheels for |
| OS | Linux x86-64 | The platform RoboSimStudio is developed and tested on |
| GPU | Optional | An NVIDIA GPU with a driver that Warp 1.18 supports (see Newton's [system requirements](https://newton-physics.github.io/newton/latest/guide/installation.html#system-requirements)). Without one, scenes run on the CPU |

### Platform-Specific Requirements

**Linux**

The OVRTX path tracer (the `rtx` extra) ships wheels for glibc 2.35 or newer,
such as Ubuntu 22.04 and later.

**Linux aarch64 (ARM64)**

SuperDex ships no wheel for ARM64, so the `superdex` and `cpu` extras install
nothing there. Sponges cannot be built, and on a CPU neither can cloth or rope.

**GPUs without RT cores**

The path tracer needs RT cores: a T4, A10, A40, L4, L40S or an RTX card. On an
A100, H100 or B200, pictures are drawn by the `tiled` renderer, which draws the
colliders alone.

> [!NOTE]
> macOS and Windows are not tested.

## Installing RoboSimStudio

RoboSimStudio installs from its Git repository. It is not on PyPI, because one
of its dependencies (PyRoKi) is only available from Git:

```bash
pip install "robosimstudio @ git+https://github.com/robosimstudio/robosimstudio"
```

On a machine without an NVIDIA GPU, install the `cpu` extra:

```bash
pip install "robosimstudio[cpu] @ git+https://github.com/robosimstudio/robosimstudio"
```

We recommend installing RoboSimStudio inside a virtual environment to avoid
conflicts with other packages:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install "robosimstudio @ git+https://github.com/robosimstudio/robosimstudio"
```

### From a Checkout with uv

The examples live in the repository. To run them, clone it and let uv build the
environment:

```bash
git clone https://github.com/robosimstudio/robosimstudio.git
cd robosimstudio
uv run examples/00_getting_started/00_hello_world.py   # builds the environment on first run
```

`uv run` and `uv sync` also install the `rtx` dependency group by default:
NVIDIA's OVRTX path tracer, 1.8 GB from NVIDIA's package index. A GPU without RT
cores never uses it; leave it out with `--no-group rtx`, or for every uv command
in a shell with `UV_NO_GROUP=rtx`.

### Without a GPU

```bash
export UV_NO_GROUP=rtx
uv run --extra cpu examples/00_getting_started/00_hello_world.py
```

> [!NOTE]
> `UV_NO_GROUP=rtx` is `--no-group rtx` for every uv command. Set it in the
> shell rather than passing the flag once: `uv run` syncs the environment each
> time, and a plain `uv run` would install the path tracer again. pip never
> installs dependency groups, so a pip install needs neither.

Nothing else changes between the two: a scene built on a machine with no CUDA
device runs on the CPU by itself.

- MuJoCo's C library steps the arm and the rigid bodies.
- SuperDex steps cloth, rope and soft bodies.
- Newton's tiled tracer renders.
- A scene left on the default physics (`rss.GENERIC`) runs as `rss.CPU`, and a
  solver with no CPU path is replaced by `rss.CPU`'s for that kind of thing, with
  the change printed.
- One world per process, and no signed-distance-field contact
  (`full_surface_contact` against a mesh, `hydroelastic`).

### Verifying the Installation

```bash
python -m robosimstudio.machine
```

It prints what a scene built in this environment would run on, and why:

```console
cuda:0 (NVIDIA L40S (sm_89)) · CUDA graphs on · renderer rtx (NVIDIA L40S (sm_89))
cpu (no CUDA device) · CUDA graphs off · renderer tiled (cpu is not a CUDA device)
```

Every build prints the same line once, and `sim.platform` keeps it.

## Choosing the Device and Renderer

The device and the renderer are picked automatically. A name given in code wins
over the environment, and the environment wins over the machine:

| | In code | For a shell | Values |
|---|---|---|---|
| Device | `scene.build(device="cpu")` | `ROBOSIMSTUDIO_DEVICE` | `auto`, `cpu`, `cuda`, `cuda:1` |
| Renderer | `renderer="tiled"` | `ROBOSIMSTUDIO_RENDERER` | `auto`, `rtx`, `tiled` |

The physics follows the device and has no setting of its own. The renderer runs
on the same device as the physics.

### A CPU-Only Material in a GPU Scene

The device belongs to the scene and the engine belongs to the material. SuperDex
runs on the CPU in any scene, so a GPU scene can put one kind of thing, or one
object, on it while everything else stays on the GPU:

```python
physics = rss.GENERIC.with_backend("cloth", rss.backend("superdex", substeps=2))  # every cloth

rss.sheet(name="towel", size=(0.2, 0.2),
          sim=rss.SimRequirements(domain="cloth", backend=rss.backend("superdex", substeps=2)))  # one sheet
```

## Extra Dependencies

| Extra | Adds |
|---|---|
| `cpu` | What a machine without a GPU uses: SuperDex, for cloth, rope and soft bodies |
| `superdex` | SuperDex, the CPU FEM engine behind `rss.sponge` (Python 3.12, no ARM64 wheel) |
| `rtx` | The OVRTX path tracer; uv installs it by default as the `rtx` group |
| `usd` | USD scene import |
| `genesis` | Genesis's IPC coupler as a second foreign engine (NVIDIA GPU only) |
| `gap` | The graph runtime `robosimstudio_policy_agent` runs policies with; large, as it brings torch and cuRobo |
| `curobo` | cuRobo, the collision-aware planner behind `motion="plan"` |
| `lerobot-feetech` | The Feetech motor bus behind leader-arm teleoperation |
| `dev` | ruff and the test tooling |

`cpu` includes `superdex`, so install the most specific one. From a checkout,
name an extra with `--extra`:

```bash
uv sync --extra superdex
```

With pip, the path tracer comes from NVIDIA's index:

```bash
pip install "robosimstudio[rtx] @ git+https://github.com/robosimstudio/robosimstudio" \
    --extra-index-url https://pypi.nvidia.com
```

> [!NOTE]
> Install `genesis` in a virtual environment of its own: it moves about twenty
> packages, and pins trimesh below 5.

## Next Steps

- Build a first scene with [Getting started](00_getting_started/README.md).
- Browse every example in the [examples index](README.md).
- Rebuild a real scene with a coding agent on the [live page](live/README.md).
- Run the tests before changing code: see [Contributing](../CONTRIBUTING.md).

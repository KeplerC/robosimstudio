# SPDX-License-Identifier: Apache-2.0
"""Where a scene runs: the device, the physics that follows from it, and the renderer -- decided in one place.

Two settings, one vocabulary, one rule::

    scene.build(device="auto")          # "auto" | "cpu" | "cuda" | "cuda:1"
    ROBOSIMSTUDIO_DEVICE=cpu            # the same, for a whole shell or machine
    ROBOSIMSTUDIO_RENDERER=tiled        # "auto" | "rtx" | "tiled" | a dotted path

A name given in code wins over the environment, the environment wins over the
machine, and ``auto`` (or nothing) names nothing at either level. A device chosen
in code around the build -- ``with wp.ScopedDevice("cuda:1"):`` or
``wp.set_device`` -- is a name given in code, except one: Warp keeps no record of
an open scope, so a scope naming the device Warp would pick anyway
(``ScopedDevice("cuda:0")`` on a GPU machine) cannot be told from no scope, and
with the variable set it is the variable that wins. Pass ``device="cuda:0"`` to
choose that one. The machine's answer is CUDA where there is a GPU and the CPU
where there is not.

**Physics has no setting of its own: it follows the device.** On a CPU every
MuJoCo island runs MuJoCo's C library, nothing is captured into a CUDA graph,
a scene left on :data:`~robosimstudio.specs.physics.GENERIC` builds on
:data:`~robosimstudio.specs.physics.CPU`, and a backend that cannot run on a
CPU is replaced by the one ``CPU`` uses for its domain. The compiler applies
that (:func:`robosimstudio.scene.compiler.compile_scene`); this module says
where, and the build prints it.

**The renderer runs where the physics does**, because it reads the model's
arrays where they live: ``rtx`` where ovrtx is installed and the GPU has RT
cores, ``tiled`` -- Newton's tracer, which runs on Warp's CPU device too --
everywhere else. :func:`~robosimstudio.solvers.render.choose_renderer` holds
that rule and every renderer in the tree is chosen through
:meth:`Platform.renderer_for`, so a still, an observation rig, a camera preview
and a video export cannot disagree about it.

Nothing is hidden: :func:`platform` answers before anything is built,
``python -m robosimstudio.machine`` answers from a shell, a build prints the
line once, and ``sim.platform`` keeps it, with whatever the build adapted.

Called ``machine`` rather than ``platform`` so the module neither shadows the
standard library's ``platform`` -- which Warp imports -- for a script run from
this directory, nor replaces the ``rss.platform`` function with itself the
moment someone imports it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

__all__ = ["AUTO", "DEVICE_ENV", "Platform", "announce", "apply", "gpu_label", "platform", "platform_of"]

AUTO = "auto"
"""The name for "pick for me", for the device and the renderer alike."""

DEVICE_ENV = "ROBOSIMSTUDIO_DEVICE"
"""The environment variable that names the device for a whole shell or machine.

``auto``, ``cpu``, ``cuda`` or ``cuda:N``. A device given in code wins over it.
Set to a CPU on a machine with a GPU, it runs what a CPU-only machine would. (The
``cpu`` test tier hides the GPUs from the driver instead, so it proves the
automatic path.)"""

_RTX_HINT = "ovrtx is installed and unused on a CPU: uv sync --no-group rtx frees 1.8 GB"


@dataclass(frozen=True)
class Platform:
    """What a scene runs on, and why each part of it was chosen.

    Attributes:
        device: The Warp device alias: ``"cuda:0"``, ``"cpu"``.
        device_why: Why that device: ``"asked for"``,
            ``"ROBOSIMSTUDIO_DEVICE=cpu"``, the GPU's name, or ``"no CUDA device"``.
        renderer: The renderer a frame asked for with no name gets, or ``""``
            when :data:`~robosimstudio.solvers.render.ENV` names one that cannot
            be built -- which raises only when a frame is actually asked for, so
            a scene without cameras still builds.
        renderer_why: Why that renderer, or what is wrong with the one named.
        notes: What a build adapted to the device, one line each -- an
            arrangement swapped, a backend replaced. Empty from :func:`platform`;
            the compiler fills it in on the platform a scene keeps.
        rtx_installed: Whether ovrtx is importable, for the readout's hint on a
            CPU, where it is never used.
        from_env: Whether the device came from :data:`DEVICE_ENV`, which a
            build then makes Warp's default (:func:`apply`).
        graphs: Whether the build will capture its substep loop into a CUDA
            graph, once a build has said -- off on a CPU, and off on a GPU when
            the arrangement or ``build(use_cuda_graph=False)`` says so. ``None``
            from :func:`platform`, which only knows the device allows one.
    """

    device: str
    device_why: str
    renderer: str
    renderer_why: str
    notes: tuple[str, ...] = ()
    rtx_installed: bool = False
    graphs: bool | None = None
    from_env: bool = False

    @property
    def is_cpu(self) -> bool:
        """Whether this is Warp's CPU device."""
        return self.device == "cpu"

    @property
    def cuda_graphs(self) -> bool:
        """Whether the substep loop may be captured into a CUDA graph: never on a CPU."""
        return not self.is_cpu

    def renderer_for(self, requested: str = "") -> tuple[str, str]:
        """The renderer to build for ``requested``, and why, on this platform's device.

        A name in code first, then :data:`~robosimstudio.solvers.render.ENV`,
        then the machine; ``auto`` and ``""`` name nothing. A renderer that
        cannot run on this device raises naming it, rather than becoming a
        different picture.
        """
        from robosimstudio.solvers.render import choose_renderer

        return choose_renderer(requested or "", device=self.device)

    def with_notes(self, notes: tuple[str, ...]) -> Platform:
        """This platform with what a build adapted to it."""
        return replace(self, notes=tuple(notes))

    def readout(self) -> str:
        """Everything on one line: device, what was adapted, graphs, renderer."""
        parts = [f"{self.device} ({self.device_why})", *self.notes]
        captured = self.cuda_graphs if self.graphs is None else self.graphs
        parts.append("CUDA graphs " + ("on" if captured else "off"))
        if self.renderer:
            parts.append(f"renderer {self.renderer} ({self.renderer_why})")
        else:
            parts.append(f"renderer unavailable ({self.renderer_why})")
        if self.is_cpu and self.rtx_installed:
            parts.append(_RTX_HINT)
        return " · ".join(parts)

    def __str__(self) -> str:
        return self.readout()


def _named(value: Any) -> Any:
    """``value`` if it names something, else ``None``: empty, ``None`` and ``auto`` name nothing."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return None if not text or text == AUTO else text
    return value  # a warp.Device


def gpu_label(device: Any) -> str:
    """A CUDA device as the readout names it: ``NVIDIA L40S (sm_89)``. Shared with the renderer's reasons."""
    return f"{device.name} (sm_{device.arch})"


_SET_FOR_THE_VARIABLE: list[str] = []
"""The device :func:`apply` last made Warp's default for :data:`DEVICE_ENV`, so the
rule can tell that apart from a device a caller chose in code."""


def _chosen_in_code(wp: Any) -> Any:
    """Warp's current device when code chose it -- a ``ScopedDevice`` or ``set_device`` -- else ``None``.

    Read as "Warp's current device is not the one it would pick", which is all
    Warp lets anyone see: a scope naming its preferred device looks like none.
    """
    current = wp.get_device()
    if current.alias == wp.get_preferred_device().alias:
        return None
    if _SET_FOR_THE_VARIABLE and current.alias == _SET_FOR_THE_VARIABLE[-1]:
        return None
    return current


def _device(device: Any) -> tuple[str, str, bool]:
    """``(alias, why, from_env)`` by the rule in the module docstring."""
    import warp as wp

    named, source = _named(device), "asked for"
    if named is None:
        scoped = _chosen_in_code(wp)
        if scoped is not None:
            return scoped.alias, "Warp's current device", False
        named = _named(os.environ.get(DEVICE_ENV, ""))
        source = f"{DEVICE_ENV}={named}"
    if named is not None:
        where = source if source != "asked for" else f"device={named!r}"
        try:
            resolved = wp.get_device(named)
        except Exception as exc:
            hint = " (there is no CUDA device here)" if "cuda" in str(named) and not wp.is_cuda_available() else ""
            raise ValueError(
                f"unknown device {where}{hint}: choose {AUTO!r}, 'cpu', 'cuda' or 'cuda:N' ({exc})"
            ) from exc
        return resolved.alias, source, source != "asked for"
    current = wp.get_device()
    if _SET_FOR_THE_VARIABLE and current.alias == _SET_FOR_THE_VARIABLE[-1]:
        # Our own leftover from a build under the variable, which is gone now:
        # not a choice anyone made, so the machine decides again.
        current = wp.get_preferred_device()
    if current.is_cuda:
        return current.alias, gpu_label(current), False
    if wp.is_cuda_available():
        # A CPU while a GPU exists: an enclosing ``wp.ScopedDevice`` or
        # ``wp.set_device`` chose it, and that is honoured, as it always was.
        return current.alias, "Warp's current device", False
    return current.alias, "no CUDA device", False


def platform(device: Any = AUTO, renderer: str = AUTO) -> Platform:
    """What a scene built here would run on, and why -- before anything is built.

    Args:
        device: ``"auto"`` (or ``None``/``""``), ``"cpu"``, ``"cuda"``,
            ``"cuda:N"``, or a ``warp.Device``.
        renderer: A renderer to resolve instead of the machine's pick, for the
            readout: ``"auto"``, ``"rtx"``, ``"tiled"`` or a dotted path.

    Raises:
        ValueError: If the device named -- in code or by :data:`DEVICE_ENV` --
            is not one this machine has. A bad renderer name does not raise
            here; it is reported in :attr:`Platform.renderer_why` and raised
            when a frame is asked for.

    Example::

        >>> print(rss.platform())
        cpu (no CUDA device) · CUDA graphs off · renderer tiled (cpu is not a CUDA device)
    """
    from robosimstudio.solvers.render import choose_renderer, rtx_available

    alias, why, from_env = _device(device)
    try:
        name, because = choose_renderer(renderer if _named(renderer) else "", device=alias)
    except (ValueError, ImportError) as exc:
        name, because = "", str(exc)
    return Platform(
        device=alias,
        device_why=why,
        renderer=name,
        renderer_why=because,
        rtx_installed=rtx_available(),
        from_env=from_env,
    )


def apply(plat: Platform) -> None:
    """What a *build* on ``plat`` does to the process: nothing, unless the device came from :data:`DEVICE_ENV`.

    The variable is a setting for the whole process, so a build made under it
    makes its device Warp's default too: an array a caller allocates with no
    device named lands beside the model rather than on a GPU nobody asked for.
    A later build with the variable gone undoes it. Here and not in
    :func:`platform`, which only answers.
    """
    import warp as wp

    if plat.from_env:
        wp.set_device(plat.device)
        _SET_FOR_THE_VARIABLE[:] = [plat.device]
        return
    if _SET_FOR_THE_VARIABLE and wp.get_device().alias == _SET_FOR_THE_VARIABLE[-1] != plat.device:
        # The variable that set Warp's default is gone and this build is elsewhere:
        # give Warp back the default it would pick.
        wp.set_device(wp.get_preferred_device())
        _SET_FOR_THE_VARIABLE.clear()


def platform_of(env: Any) -> Platform:
    """The platform something with a model runs on: its own, or one resolved from its model's device.

    For the studio's panels, which are handed a ``Sim``, a bench ``Env`` or a
    fleet and should not care which.
    """
    known = getattr(env, "platform", None)
    if isinstance(known, Platform):
        return known
    model = getattr(env, "model", None)
    return platform(getattr(model, "device", None))


_said: set[str] = set()


def announce(plat: Platform, *, verbose: bool = False) -> None:
    """Print the platform line: every time when ``verbose``, else once per process per distinct line.

    Once, because a studio rebuilds its scene on every save and the line does
    not change; again whenever it does, because then it says something new.
    """
    line = plat.readout()
    if verbose or line not in _said:
        _said.add(line)
        print(f"[rss] {line}")


if __name__ == "__main__":
    print(platform())

# SPDX-License-Identifier: Apache-2.0
"""The ``genesis`` backend: Genesis's IPC coupler, or its SPH liquid, co-simulated beside Newton.

Importing this registers the alias and nothing else --- the engine itself loads
the first time a scene actually builds a Genesis island, so a checkout without
the ``[genesis]`` extra composes, lists and runs every shipped task untouched.

Named ``genesis`` inside ``robosimstudio.engines``, which is
``robosimstudio.engines.genesis`` and shadows nothing: the adapter's own
``import genesis`` is absolute and resolves to the wheel.
"""

from __future__ import annotations

from robosimstudio.specs.physics import register_backend

__all__ = ["BACKEND"]

BACKEND = register_backend(
    "genesis",
    cls="robosimstudio.engines.genesis.solver.SolverGenesis",
    supports=("particles",),
    summary="Genesis, co-simulated on the host: cloth and FEM bodies on its IPC coupler (libuipc), "
    "liquids on SPH under its legacy coupler; Newton islands lead, reactions feed back.",
    host_engine=True,
    # There is no CPU Genesis here: the runtime initialises Genesis on its GPU
    # backend (runtime.py), the SPH liquid with it, and libuipc builds
    # Engine("cuda"). A CPU scene routes cloth to SuperDex instead, and refuses
    # water by name (robosimstudio.scene.physics.devices.unserved).
    devices=("cuda",),
    cpu_note="Genesis is initialised on its GPU backend here, and IPC's libuipc builds a CUDA engine",
    extra="genesis",
    # What each builder reads is declared beside it, per kind, in ``bridge.py``.
)

from . import bridge as _bridge  # noqa: E402, F401 -- registers the shell and tet builders
from . import sph as _sph  # noqa: E402, F401 -- registers the sph kind and its builder

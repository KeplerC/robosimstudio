# SPDX-License-Identifier: Apache-2.0
"""A stand-in particle thing for the settle: it counts its settles and its rest snapshots.

The settle and the studio's **Re-settle** are both read off the same record of
things and the runs they joined with, so both test modules hand them this.
"""

from __future__ import annotations

from types import SimpleNamespace

__all__ = ["particle_thing"]


def particle_thing(name: str = "towel", log: list | None = None, run: tuple[int, int] | None = None):
    """A particle thing; ``run`` is the ``(settle_frames, settle_substeps)`` its spec states.

    And the run it joined the settle with, as ``join_settle`` keeps it, until it joins again.
    """
    from robosimstudio.domains.capabilities import ParticleMixin

    class _Stub(ParticleMixin):
        def __init__(self) -> None:
            self.name = name
            self.settles = 0
            self.snapshots = 0
            if run is not None:
                self.spec = SimpleNamespace(settle_frames=run[0], settle_substeps=run[1])
            self.settle_run = run

        def settle(self, pipeline) -> None:
            del pipeline
            self.settles += 1
            if log is not None:
                log.append(f"settle {self.name}")

        def snapshot_rest(self, pipeline) -> None:
            del pipeline
            self.snapshots += 1
            if log is not None:
                log.append(f"snapshot {self.name}")

    return _Stub()

# SPDX-License-Identifier: Apache-2.0
"""A wrench on chosen bodies of a built scene, applied in every substep of the captured graph.

The bond tests load a joint the way a test rig does: a force on a block, ramped or held, with nothing
touching it. A scene has no such thing to offer, so this is one ``rigid_force`` stage that adds a
device array of wrenches to ``state.body_f``; a test writes the array between control steps.
"""

from __future__ import annotations

import numpy as np
import warp as wp

__all__ = ["Wrenches"]


@wp.kernel
def _add_wrenches(wrench: wp.array[wp.spatial_vector], body_f: wp.array[wp.spatial_vector]):
    b = wp.tid()
    body_f[b] = body_f[b] + wrench[b]


class Wrenches:
    """``set`` a body's force and torque [N, N m, world, about its centre of mass]; they hold until set again."""

    def __init__(self, sim) -> None:
        self.pipeline = sim.pipeline
        model = self.pipeline.model
        self._host = np.zeros((model.body_count, 6), dtype=np.float32)
        self._device = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
        self.pipeline.hooks.rigid_force.append(self._apply)
        self.pipeline.invalidate_graph()

    def _apply(self, state, dt: float) -> None:
        del dt
        wp.launch(
            _add_wrenches,
            dim=len(self._device),
            inputs=[self._device],
            outputs=[state.body_f],
            device=self._device.device,
        )

    def set(self, body: int, force=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0)) -> None:
        self._host[int(body), :3] = force
        self._host[int(body), 3:] = torque
        self._device.assign(self._host)

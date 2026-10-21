# SPDX-License-Identifier: Apache-2.0
"""A body welded to the world that an **episode** places -- per world, at reset.

Lifted out of the surface-preparation suite (which still re-exports it)
once the static domain needed the same channel: a peg board drawn per episode,
a bolted receptacle jittered in its slots, a hook row moved a few millimetres --
every one of them a welded fixture that used to be ``body=-1`` scenery, and so
world 0's for every world of a replicated model. See
:class:`~robosimstudio.domains.static.entity.EntityStatic` for how a task
static takes this channel in a replicated build.

---------------------------------------------------------------------------
**Why a weld looked immovable, and what actually makes it so**

MuJoCo-Warp runs ``W`` Newton worlds as ``W`` copies of **one** MuJoCo model --
it converts the first world's shapes and replicates (``separate_worlds``, which
a multi-world model may not switch off). So anything whose pose lives in the
MuJoCo *model* is world 0's for everybody, and a static body is exactly that:
``_convert_to_mjc`` bakes its pose into ``body_pos``/``body_quat`` at conversion.
Writing ``joint_X_p`` afterwards moves nothing, in any world.

The exception is a **mocap** body. Its pose lives in ``mjw_data.mocap_pos`` --
per world -- and ``notify_model_changed(JOINT_PROPERTIES)`` is what refreshes it
from ``model.joint_X_p``. An *un*articulated fixed-to-world joint -- which is
what a welded fixture is -- gets that channel only when the child body is
kinematic: ``mocap = child_is_kinematic``. So the whole of the mechanism is:

1. mark the welded body :attr:`newton.BodyFlags.KINEMATIC` at build, which is
   already true of it -- a bolted fixture does not respond to forces -- and is
   what asks the converter for the per-world channel rather than the baked one;
2. write that world's ``joint_X_p`` row and tell the solver, at reset.

**A kinematic body is not a weaker weld.** It is immovable by construction
rather than by an infinite mass, and MuJoCo integrates neither. What changes is
who may move it: a mocap body is moved by whoever writes ``mocap_pos``, which is
this module, once per episode, before the first step.

**What the caller still owes.** Newton's ``body_q`` is *not* refreshed by the
notification -- a mocap body has no MuJoCo joint, so the ``qpos`` write-back
that carries every articulated body home skips it. Everything in the studio
that asks where a fixture is reads ``body_q``: handles, features, the renderer.
So :meth:`WeldedFixture.place_welded` writes both, and they are the same
transform by construction rather than by agreement.
"""

from __future__ import annotations

from typing import Any

import newton
import numpy as np

__all__ = ["WeldedFixture", "weld_to_world"]


def weld_to_world(builder: newton.ModelBuilder, xform: Any, *, label: str) -> tuple[int, int]:
    """A body bolted to the world at ``xform``: ``(body, joint)``, both indices into ``builder``.

    The one recipe for a body that stands still, whichever entity builds it: a
    link added kinematic, and a fixed joint from the world at the link's own pose,
    labelled ``<label>_weld``. Both halves are needed.

    * **The joint** is what MuJoCo converts. Outside any articulation it becomes a
      standalone world root, and the converter says so once in a ``UserWarning``.
      A link with no joint at all is refused: "cannot convert bodies that are
      outside articulations and have no standalone joint to world".
    * **The kinematic flag** is what VBD reads. It gives the body an inverse mass
      of zero, where the joint alone let a static block move 0.05 mm under a
      pressing hand and a static cube 0.01 mm under a towel (``s4_weld_only.py``).
      It is also what asks MuJoCo's converter for a mocap body, the per-world
      channel :meth:`WeldedFixture.place_welded` writes, rather than a pose baked
      into the model (the module docstring).

    The joint's parent transform is the body's pose and its child transform is
    identity, so ``joint_X_p`` **is** the body's world pose, which is what makes
    :meth:`WeldedFixture.place_welded` a single row write rather than a frame
    composition. The joint has no coordinates, so a reset has none to write, and
    no articulation, so forward kinematics leaves ``body_q`` where it was put.
    """
    body = int(builder.add_link(xform=xform, label=label, is_kinematic=True))
    joint = int(builder.add_joint_fixed(-1, body, parent_xform=builder.body_q[body], label=f"{label}_weld"))
    return body, joint


class WeldedFixture:
    """Mixin: a body welded to the world, re-placeable per world at reset.

    Two calls, one at each end of the lifecycle. :meth:`weld_in_place` builds the
    welded body (:func:`weld_to_world`) and records its joint, and
    :meth:`place_welded` is what its ``reset`` calls.

    Mixed in *before* the entity base so ``self.body``, ``self.worlds`` and
    ``self.stepping_model`` are the entity's own.
    """

    weld_joint: int = -1
    """Template index of the fixed joint holding this body to the world.

    ``-1`` until :meth:`weld_in_place` runs. The **template's**, like every index
    an entity resolves at build: :attr:`~robosimstudio.scene.entity.Entity.worlds`
    is what turns it into one world's row.
    """

    def weld_in_place(self, builder: newton.ModelBuilder, xform: Any, *, label: str) -> int:
        """A body bolted to the world at ``xform``, on the per-world channel; returns its index.

        :func:`weld_to_world`, with the joint kept as :attr:`weld_joint` for
        :meth:`place_in_model` and :meth:`place_welded` to write.
        """
        body, self.weld_joint = weld_to_world(builder, xform, label=label)
        return body

    def place_in_model(self, model: Any, pose: np.ndarray, *, world: int = 0) -> None:
        """Bolt this world's copy to ``pose`` in the MODEL, before anything reads it.

        The compile-time half of :meth:`place_welded`, and the reason it exists
        is that ``_assemble`` is not a passive reader: it CONVERTS the model, and
        MuJoCo converts the first world's geometry into a model every world then
        shares. A weld written afterwards still moves the body -- the mocap
        channel is exactly that -- but whatever the conversion derived from where
        the bar stood was derived from where it stood *then*.

        Measured: a scene rebuilt for an episode cleans its stripe, and the same
        scene re-posed into that episode presses the bar 26 mm off the tube, with
        every static quantity -- Newton's ``body_q``, MuJoCo's ``xpos`` and
        ``xquat``, all 25 mirror colliders, every particle -- identical between
        the two. So the difference is something conversion-time, and this is
        where a placement gets in front of it.

        No solver notification: there is no solver yet, which is the point.
        """
        if self.weld_joint < 0:
            raise RuntimeError(f"{getattr(self, 'name', type(self).__name__)} was never welded")
        row = np.asarray(pose, dtype=np.float64).reshape(-1)[:7]
        worlds = getattr(self, "worlds", None)
        joint = worlds.joint(self.weld_joint, world) if worlds is not None else self.weld_joint
        body = worlds.body(int(self.body), world) if worlds is not None else int(self.body)  # type: ignore[attr-defined]
        parents = model.joint_X_p.numpy()
        parents[joint] = row
        model.joint_X_p.assign(parents)
        bodies = model.body_q.numpy()
        bodies[body] = row
        model.body_q.assign(bodies)

    def place_welded(self, pipeline: Any, pose: np.ndarray, *, world: int = 0) -> None:
        """Move this world's copy of the fixture to ``pose`` (xyz + xyzw).

        Writes the three places a welded pose is read from and keeps them equal:
        the model's ``joint_X_p`` row (which MuJoCo re-reads into ``mocap_pos``
        for this world alone), and both states' ``body_q`` row (which is what
        every reader in the studio actually asks).

        The notification goes straight to the solver rather than through
        :func:`robosimstudio.core.tuning.notify`, and that is deliberate:
        the helper drops a captured graph, because most model edits are host work
        a replay does not perform. This one is not. It writes into the MuJoCo
        arrays the graph already holds pointers to, so the replay reads the new
        pose without being recaptured -- measured at 1.2 ms against a live graph,
        which is what makes this affordable once per world per wave rather than a
        recapture per wave.
        """
        if self.weld_joint < 0:
            raise RuntimeError(f"{getattr(self, 'name', type(self).__name__)} was never welded; build() must run first")
        model = getattr(self, "stepping_model", None) or self.model  # type: ignore[attr-defined]
        row = np.asarray(pose, dtype=np.float64).reshape(-1)[:7]
        worlds = getattr(self, "worlds", None)
        joint = worlds.joint(self.weld_joint, world) if worlds is not None else self.weld_joint
        body = worlds.body(int(self.body), world) if worlds is not None else int(self.body)  # type: ignore[attr-defined]

        # One host round trip each, at reset only. A per-row device write would
        # need a kernel; a wave resets once and steps thousands of times.
        parents = model.joint_X_p.numpy()
        parents[joint] = row
        model.joint_X_p.assign(parents)

        for state in (pipeline.state_0, pipeline.state_1):
            if state is None:
                continue
            bodies = state.body_q.numpy()
            bodies[body] = row
            state.body_q.assign(bodies)

        # Through the program, which on a CPU also carries the mocap rows into
        # the C data MuJoCo's C library steps (core.mjdata.after_notify). Not
        # core.tuning.notify: that drops a captured graph, and a reset that moves
        # a fixture must not cost a re-capture -- the mocap rows are device
        # memory the graph already reads.
        flags = int(newton.ModelFlags.JOINT_PROPERTIES)
        program = getattr(pipeline, "program", None)
        if program is not None:
            program.notify(flags)
        elif getattr(pipeline, "solver", None) is not None:
            from robosimstudio.core.mjdata import after_notify

            pipeline.solver.notify_model_changed(flags)
            after_notify(pipeline.solver, flags)

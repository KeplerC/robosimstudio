# SPDX-License-Identifier: Apache-2.0
"""Adhesion: the tape's own physics, owned by the tape.

A discrete cohesive-zone model, after AutoPeel (ICRA 2025). Every tape particle
resting on the box is a penalty spring to a fixed anchor, and a spring breaks
*permanently* once the particle has lifted past a threshold. Pull the free tab
and the bonds nearest it break first, so the peel front travels across the box
on its own -- nothing scripts the peel, which is the entire point of modelling
it this way.

The version of this module that lived in ``tasks/tape_peel/`` made the argument
for its own move and did not notice:

    This lives in the suite rather than in core or in the generic cloth entity
    because it is not general deformable physics: it is what adhesive tape *is*.
    A cloth entity that knew about bonds would be carrying one task's model into
    every scene that drapes a shirt.

Both halves of that are right, and together they name a layer that did not exist
then. Bonds are not general cloth physics and they are not a fact about
*peeling* -- a second task over the same tape would need every one of them. They
are a fact about the tape, so they live with the tape, in the corpus, beside the
record that declares them. The fear the original expresses is *satisfied* by the
move rather than violated: the generic cloth entity still knows nothing about
bonds.

Three departures from the reference implementation, all deliberate:

* **The bonds run as a Warp kernel.** The reference evaluates them in numpy,
  round-tripping ``particle_q``/``particle_f`` to the host twice per substep.
  At 20 substeps that is 40 synchronizes a frame, and it makes the step
  uncapturable. On device it is neither.
* **Mode-I break criterion.** The reference breaks on total displacement; this
  breaks on *opening* displacement -- how far the particle has lifted off the
  surface. With a real servoed arm straddling the strip, particles routinely get
  pressed down or dragged a few millimetres along the box, and tape pressed onto
  its surface does not unstick. Only being pulled *off* it peels. The reference
  never notices the difference because its kinematic arm never touches the strip.
* **Metres, not centimetres.** Lengths convert directly; the stiffness is set to
  20x a garment's, which is the ratio the reference uses against its own cloth.

And one generalisation over it, added for the ``panel`` layouts (2026-09-10),
which bond the film to a phone leaning back at 30 degrees: the bonding surface
is a box with an **orientation** rather than an axis-aligned one, and a bond
breaks on the opening displacement along the surface **normal** rather than
along world z. With the identity orientation both reduce to the old
expressions bit for bit -- the selection compares ``q @ I`` against
``centre @ I``, and the kernel's ``dot(delta, (0, 0, 1))`` is ``delta[2]`` --
so every axis-aligned caller (``handle``) is untouched; ``tests/test_rgbs_peel_phone.py``
holds it to that.

The handle pin is here for the same reason: it holds a rigid pull-tab at its
as-laid pose until a gripper takes it, which is a property of *this* task's
layout. It is a device kernel gated by a device-side flag, launched
unconditionally, so the substep block stays capturable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.core.math import quat_to_mat

from ...batched import HostField

__all__ = [
    "BOND_BREAK",
    "BOND_MARGIN",
    "BOND_STIFFNESS",
    "PIN_MAX_STEP",
    "AdhesionBonds",
    "BondState",
    "HandlePin",
]

BOND_STIFFNESS = 3.0
"""Bond stiffness [N/m]. The reference's 3000 dyn/cm, converted."""
BOND_BREAK = 0.005
"""Opening displacement at which a bond lets go [m].

A bond resists ``BOND_STIFFNESS * BOND_BREAK`` = 0.015 N before releasing, so a
row of seven across the strip's width costs 0.105 N to lift -- which a gripper
pulls comfortably, the tape's own 3.8 g never reaches, and a careless drag does
not.
"""
BOND_MARGIN = 0.015
"""How close to the surface a particle must rest to bond at all [m]."""

PIN_MAX_STEP = 0.0006
"""Ceiling on how far a kinematically held particle moves per substep [m].

One number and one reason, and it was three copies before this: the handle pin
here, the scripted routine's latch, and the graph tool's latch all walk
particles toward a target and all three would detonate the solver if they
teleported. Teleported to its rest pose in one substep the tab yanks every bond
in the strip past threshold at once and the tape rips off whole; teleported to
the hand, a latch injects the entire grasp error in one substep. Same failure,
same fix, so it is stated once and imported.
"""


@wp.kernel
def _bond_force_kernel(
    particle_q: wp.array[wp.vec3],
    bond_idx: wp.array[wp.int32],
    bond_anchor: wp.array[wp.vec3],
    bond_active: wp.array[wp.int32],
    bond_k: float,
    break_distance: float,
    normal: wp.vec3,
    particle_f: wp.array[wp.vec3],
):
    """Pull each still-stuck particle back to its anchor, and break it once the
    opening displacement exceeds the threshold.

    Force and break test in one pass, on device. A broken bond never re-forms --
    that irreversibility *is* the peel.

    ``normal`` is the bonding face's outward normal in world, so "opening" is
    the component of the displacement along it: world z on a flat box, the
    face normal on a tilted one. Bit-for-bit the old ``delta[2]`` test when the
    normal is ``(0, 0, 1)`` -- the two zero products add nothing.
    """
    i = wp.tid()
    if bond_active[i] == 0:
        return
    p = bond_idx[i]
    delta = particle_q[p] - bond_anchor[i]
    if wp.dot(delta, normal) > break_distance:
        bond_active[i] = 0
        return
    wp.atomic_add(particle_f, p, -delta * bond_k)


@wp.kernel
def _handle_pin_kernel(
    pin_idx: wp.array[wp.int32],
    pin_rest: wp.array[wp.vec3],
    pin_enabled: wp.array[wp.int32],
    max_step: float,
    inv_dt: float,
    pins_per_world: int,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
):
    """Hold the rigid pull-tab at its as-laid pose, rate-limited.

    Gated by a *device-side* flag rather than by a host-side ``if`` around the
    launch: the block is captured into a CUDA graph, so a launch skipped on the
    host is simply absent from the graph and releasing the pin later would do
    nothing.

    The rate limit matters for the same reason it does on a carry: teleported to
    its rest pose in one substep, the tab yanks every bond in the strip past
    threshold at once and the tape rips off whole.

    **The flag is one per world, and the launch covers every world's tab.**
    ``pin_idx`` and ``pin_rest`` hold each world's own particles and each
    world's own as-laid pose, laid out world-major; the thread id names the row
    and ``i / pins_per_world`` names the world whose flag gates it. One shared
    flag would mean the first hand to take its tab let go of every world's --
    and the tapes the other worlds were still working on would fall off the box
    on their own.
    """
    i = wp.tid()
    if i >= pin_idx.shape[0]:
        return
    if pin_enabled[i / pins_per_world] == 0:
        return
    p = pin_idx[i]
    goal = pin_rest[i]
    delta = goal - particle_q[p]
    dist = wp.length(delta)
    if max_step > 0.0 and dist > max_step:
        goal = particle_q[p] + delta * (max_step / dist)
    particle_qd[p] = (goal - particle_q[p]) * inv_dt
    particle_q[p] = goal


@dataclass
class BondState:
    """The bond field's state, as a recorder or an evaluator reads it.

    Attributes:
        active: Per-bond intact flag, shape ``[bond_count]``, dtype uint8.
        s: Broken fraction in [0, 1] -- the authoritative peel quantity.
        row: Per width-row broken fraction, shape ``[rows]``. A peel front is a
            *contiguous* run of broken rows from the tab, which is what
            distinguishes a peel from the tape being torn in the middle.
    """

    active: np.ndarray
    s: float
    row: np.ndarray


class AdhesionBonds:
    """A field of breakable springs holding a strip to a surface.

    Args:
        model: The finalized model.
        particles: The strip's particle indices.
        surface_center: Centre of the box the strip is stuck to [m].
        surface_half: Its half-extents [m].
        k: Bond stiffness [N/m].
        break_distance: Opening displacement at which a bond lets go [m].
        margin: How close to the surface top a particle must rest to bond [m].
        row_axis: Which axis of the *surface frame* runs along the strip, for
            the per-row breakdown a peel front is measured on. The world axis
            when the surface is axis-aligned, which is every layout but the
            phone's.
        surface_quat: The box's orientation, xyzw. Its local +z is the face
            the strip bonds to and the axis a bond opens along; identity (the
            default) is the axis-aligned box every earlier layout stated, and
            leaves every expression below exactly as it was.
        worlds: The build's :class:`~robosimstudio.core.worlds.WorldMap`, or
            ``None`` at one world. The field is **per world**: one block of
            bonds for each copy of the strip, over that copy's particles. A
            single block would bond world 0's strip and leave every other
            world's lying loose on its box -- which looks like a peel that was
            already finished rather than like a bug.
    """

    def __init__(
        self,
        model: newton.Model,
        particles: Sequence[int],
        *,
        surface_center: Sequence[float],
        surface_half: Sequence[float],
        k: float = BOND_STIFFNESS,
        break_distance: float = BOND_BREAK,
        margin: float = BOND_MARGIN,
        row_axis: int = 0,
        surface_quat: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        worlds: Any = None,
    ) -> None:
        self.model = model
        self.worlds = worlds
        self.particles = np.asarray(particles, dtype=np.int32)
        self.surface_center = np.asarray(surface_center, dtype=np.float64)
        self.surface_half = np.asarray(surface_half, dtype=np.float64)
        self.surface_quat = np.asarray(surface_quat, dtype=np.float64)
        self.surface_rot = quat_to_mat(self.surface_quat)
        """Columns are the surface frame's axes in world; ``[:, 2]`` is the face normal."""
        self.normal = np.array(self.surface_rot[:, 2], dtype=np.float64)
        self.k = float(k)
        self.break_distance = float(break_distance)
        self.margin = float(margin)
        self.row_axis = int(row_axis)

        self.bond_idx: wp.array | None = None
        self.bond_anchor: wp.array | None = None
        self.bond_active: wp.array | None = None
        self.rows: np.ndarray = np.zeros(0, dtype=np.int32)
        """Which row each bond belongs to, ordered from the tab inwards.

        One world's worth, because every world's strip is the same strip laid
        the same way: the rows are geometry and only what has broken is per
        world."""
        self.per_world: int = 0
        """Bonds in one world's block of the field."""
        self.idx_host: np.ndarray = np.zeros(0, dtype=np.int64)
        """``bond_idx`` on the host. Fixed from :meth:`create`, so it is copied
        once rather than per read: only which bonds are still intact changes."""
        self.anchor_host: np.ndarray = np.zeros((0, 3))
        """``bond_anchor`` on the host, likewise fixed from :meth:`create`."""
        self._active_host: HostField | None = None
        """``bond_active``, brought to the host once per instant rather than once
        per world. See :mod:`tasks.rgbs.assets.batched`."""
        self._rest_active: np.ndarray | None = None

    # ------------------------------------------------------------------ create
    def create(self, state: newton.State, *, verbose: bool = False) -> int:
        """Bond every strip particle currently resting on the surface.

        Call **once, on the flat just-laid strip**, before settling. The
        reference bonds at frame 0 and only then settles, with the adhesion
        holding the film in place -- and its own docstring claims otherwise
        while its code does not wait. Settling first instead sends the strip
        sliding across a deliberately low-friction box, and the bonds then form
        against wherever it slid to.

        The overhanging tab is over empty space either way and bonds to nothing,
        which is what leaves it graspable.

        **Every world's strip, in one field.** ``state`` is the model that steps,
        so its particle array holds one block per world; ``self.particles`` are
        the template's indices, which is the first block. Each world's bonds are
        chosen against *its own* particles and anchored at *its own* positions --
        the same test, run per block, rather than world 0's answer tiled. The
        positions are identical at this instant (replication superimposes the
        worlds and nothing has stepped yet), so this costs one host copy and
        makes the arithmetic say what it means instead of relying on that.

        Returns:
            How many bonds formed, over the whole field.
        """
        q = np.asarray(state.particle_q.numpy(), dtype=np.float64)
        # In the surface frame, where the box is axis-aligned whatever its
        # tilt: ``v @ R`` is ``R^T v``, the world vector read along the box's own
        # axes. Rotating the particles and the centre alike, rather than
        # translating first, is what keeps the identity case bit-identical to
        # the world-frame comparison this was before the phone layout -- both
        # sides pass through ``@ I`` unchanged.
        rot = self.surface_rot
        q_s = q @ rot
        centre, half = self.surface_center @ rot, self.surface_half
        top = centre[2] + half[2]
        on_surface = (
            (q_s[:, 0] >= centre[0] - half[0])
            & (q_s[:, 0] <= centre[0] + half[0])
            & (q_s[:, 1] >= centre[1] - half[1])
            & (q_s[:, 1] <= centre[1] + half[1])
            & (np.abs(q_s[:, 2] - top) <= self.margin)
        )
        num_worlds = int(getattr(self.worlds, "num_worlds", 1) or 1)
        stride = int(getattr(self.worlds, "particles_per_world", 0) or 0)
        blocks: list[np.ndarray] = []
        for world in range(num_worlds):
            mask = np.zeros(len(q), dtype=bool)
            mask[self.particles + world * stride] = True
            blocks.append(np.flatnonzero(on_surface & mask).astype(np.int32))
        counts = {len(b) for b in blocks}
        if len(counts) != 1:
            # Identical builds must bond identically. A world that bonded a
            # different number of particles means the worlds did not start alike
            # -- and a ragged field cannot be sliced per world at all.
            raise RuntimeError(
                f"adhesion bonded different counts per world ({sorted(counts)}); the worlds are "
                "meant to be one strip replicated, so this is a build that did not replicate"
            )
        idx = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.int32)
        self.per_world = len(blocks[0]) if blocks else 0

        device = self.model.device
        self.bond_idx = wp.array(idx, dtype=wp.int32, device=device)
        self.bond_anchor = wp.array(q[idx].astype(np.float32), dtype=wp.vec3, device=device)
        self.bond_active = wp.array(np.ones(len(idx), dtype=np.int32), dtype=wp.int32, device=device)
        self._active_host = HostField(self.bond_active.numpy)
        self.idx_host = idx.astype(np.int64)
        # Through float32 on the way, because the device array is float32 and a
        # host mirror that skipped the narrowing would make every load read
        # slightly differently from the one this replaced.
        self.anchor_host = np.asarray(q[idx].astype(np.float32), dtype=np.float64)
        self._rest_active = np.ones(len(idx), dtype=np.int32)
        # Rows are grouped along the surface's own axis, so a tilted film's
        # width-rows are its grid rows and not their projection on world x -- and
        # off the first world's block, because every world's copy of the strip is
        # the same grid and the rows are an index into it.
        self.rows = self._assign_rows(q_s[blocks[0]]) if blocks else np.zeros(0, dtype=np.int32)
        if verbose:
            print(
                f"[adhesion] bonded {self.per_world} of {len(self.particles)} strip particles "
                f"in each of {num_worlds} world(s) "
                f"(break {self.break_distance * 1000:.0f} mm, k {self.k} N/m)"
            )
        return len(idx)

    def _assign_rows(self, anchors: np.ndarray) -> np.ndarray:
        """Group bonds into width-rows along the strip.

        ``anchors`` are in the surface frame (world, for an axis-aligned box).
        Rows are ordered so row 0 is nearest the tab, which is what makes
        "contiguous broken rows from the tab" a meaningful peel-front measure.
        The tab overhangs the surface, so it is at whichever end of
        ``row_axis`` lies outside the box -- taken here as the far end, matching
        how the strip is laid.
        """
        if not len(anchors):
            return np.zeros(0, dtype=np.int32)
        axis = anchors[:, self.row_axis]
        # One row per distinct anchor coordinate along the strip, tolerant of
        # float noise in the grid spacing.
        order = np.unique(np.round(axis, 5))
        rows = np.searchsorted(order, np.round(axis, 5))
        # Row 0 nearest the tab: the tab is at the high end of row_axis.
        return (len(order) - 1 - rows).astype(np.int32)

    # ------------------------------------------------------------------ runtime
    def apply(self, state: newton.State, dt: float | None = None) -> None:
        """Add this substep's adhesion forces. A ``pre_solve`` hook.

        Runs after ``clear_forces`` (which wipes ``particle_f``) and before the
        deformable solve, so the bonds are integrated like any other external
        force.
        """
        del dt
        if self.bond_idx is None or len(self.bond_idx) == 0:
            return
        wp.launch(
            _bond_force_kernel,
            dim=len(self.bond_idx),
            inputs=[
                state.particle_q,
                self.bond_idx,
                self.bond_anchor,
                self.bond_active,
                self.k,
                self.break_distance,
                wp.vec3(*[float(v) for v in self.normal]),
            ],
            outputs=[state.particle_f],
            device=self.model.device,
        )

    def reset(self, world: int | None = None) -> None:
        """Restore bonds to intact -- bit-exact, from the recorded field.

        Args:
            world: Which world's block, or ``None`` for the whole field. A wave
                resets its worlds one at a time, so restoring everything on each
                call would re-intact the strips of the worlds already reset --
                the same answer only because every world starts alike, and wrong
                the moment one does not.
        """
        if self.bond_active is None or self._rest_active is None:
            return
        if world is None or self.per_world <= 0:
            self.bond_active.assign(self._rest_active)
            if self._active_host is not None:
                self._active_host.invalidate()
            return
        block = slice(int(world) * self.per_world, (int(world) + 1) * self.per_world)
        field = np.asarray(self.bond_active.numpy(), dtype=np.int32)
        field[block] = self._rest_active[block]
        self.bond_active.assign(field)
        self._active_host.invalidate()

    # ------------------------------------------------------------------ queries
    @property
    def count(self) -> int:
        """Bonds in the whole field -- every world's, at more than one."""
        return 0 if self.bond_idx is None else len(self.bond_idx)

    def read(self, world: int = 0, *, at: Any = None) -> BondState:
        """One world's bond field, read back to the host. One copy per instant.

        The copy is the whole array and the slice is taken on the host: it is
        one transfer either way. ``at`` is the scoring instant, and it is what
        makes a wave pay for one transfer rather than one per world -- see
        :mod:`tasks.rgbs.assets.batched`. Without it every call re-reads, which
        is what a caller outside the scoring path wants.
        """
        if self.bond_active is None:
            return BondState(active=np.zeros(0, dtype=np.uint8), s=0.0, row=np.zeros(0))
        active = np.asarray(self.active_rows(at), dtype=np.uint8)
        if self.per_world > 0:
            active = active[int(world) * self.per_world : (int(world) + 1) * self.per_world]
        total = len(active)
        broken = float(total - int(active.sum())) / total if total else 0.0
        n_rows = int(self.rows.max()) + 1 if len(self.rows) else 0
        row = np.zeros(n_rows)
        for r in range(n_rows):
            in_row = self.rows == r
            row[r] = float(1.0 - active[in_row].mean()) if in_row.any() else 0.0
        return BondState(active=active, s=broken, row=row)

    def active_rows(self, at: Any = None) -> np.ndarray:
        """``bond_active`` on the host, copied only once per instant."""
        if self.bond_active is None:
            return np.zeros(0, dtype=np.int32)
        return self._active_host.rows(at)

    def intact(self, world: int | None = None) -> int:
        """How many bonds are still holding -- one world's, or the whole field's."""
        if self.bond_active is None:
            return 0
        active = self.bond_active.numpy()
        if world is not None and self.per_world > 0:
            active = active[int(world) * self.per_world : (int(world) + 1) * self.per_world]
        return int(active.sum())


class HandlePin:
    """Holds a rigid pull-tab at its as-laid pose until a gripper takes it.

    Some layouts present the tab as a stiff horizontal handle rather than a
    drooping flap, which is what makes them graspable by a top-down or a pinch
    approach. The reference keeps such a tab pinned from the first frame until
    the demonstrator's gripper closes on it; this is that pin, in the post-solve
    slot every kinematic hold uses.

    Args:
        model: The finalized model.
        indices: The tab-tip particles to hold, as **template** indices.
        positions: The **whole model's** particle positions [m], as
            ``particle_q`` hands them over -- every world's block, not one.
            Each world's rest rows are gathered out of it at its own indices.
        max_step: Ceiling on how far a pinned particle moves per substep [m].
        worlds: The build's :class:`~robosimstudio.core.worlds.WorldMap`, or
            ``None`` at one world. Every world's tab is pinned, each at its own
            particles, and each is released on its own: a hand closing on one
            world's tab is not a reason for another world's to let go.
    """

    def __init__(
        self,
        model: newton.Model,
        indices: Sequence[int],
        positions: np.ndarray,
        *,
        max_step: float = PIN_MAX_STEP,
        worlds: Any = None,
    ) -> None:
        self.model = model
        self.worlds = worlds
        self.max_step = float(max_step)
        device = model.device
        self.num_worlds = int(getattr(worlds, "num_worlds", 1) or 1)
        stride = int(getattr(worlds, "particles_per_world", 0) or 0)
        template = np.asarray(indices, dtype=np.int64).reshape(-1)
        self.per_world = len(template)
        # Each world's own tab, at its own as-laid pose. The poses are identical
        # at this instant -- replication superimposes the worlds and the settle
        # has not run -- so gathering per world costs nothing and is the
        # difference between arithmetic that says what it means and arithmetic
        # that is right because of a fact stated somewhere else.
        blocks = [template + world * stride for world in range(self.num_worlds)]
        spread = np.concatenate(blocks) if self.per_world else np.zeros(0, dtype=np.int64)
        rows = (
            np.asarray(positions, dtype=np.float32).reshape(-1, 3)[spread]
            if self.per_world
            else np.zeros((0, 3), dtype=np.float32)
        )
        self._idx = wp.array(spread.astype(np.int32), dtype=wp.int32, device=device)
        self._rest = wp.array(rows, dtype=wp.vec3, device=device)
        self._enabled = wp.array(np.ones(self.num_worlds, dtype=np.int32), dtype=wp.int32, device=device)
        self._enabled_host = np.ones(self.num_worlds, dtype=np.int32)
        self.count = len(spread)

    @property
    def enabled(self) -> bool:
        """Whether world 0's tab is still pinned. See :meth:`enabled_for`."""
        return self.enabled_for(0)

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.set_enabled(bool(value), world=0)

    def enabled_for(self, world: int = 0) -> bool:
        """Whether one world's tab is still pinned."""
        return bool(self._enabled_host[int(world) % self.num_worlds])

    def set_enabled(self, value: bool, *, world: int = 0) -> None:
        """Pin or release one world's tab, and upload the whole flag row.

        One ``assign`` of ``num_worlds`` int32s rather than a per-world write:
        the transfer is the array whichever way it is spelled.
        """
        self._enabled_host[int(world) % self.num_worlds] = 1 if value else 0
        self._enabled.assign(self._enabled_host)

    def apply(self, state: newton.State, dt: float | None = None) -> None:
        """Hold every world's tab for one substep. A ``post_solve`` hook.

        A kinematic write has to land after the solve, on the state the solve
        just wrote, or the solver simply undoes it.
        """
        if not self.count:
            return
        wp.launch(
            _handle_pin_kernel,
            dim=self.count,
            inputs=[
                self._idx,
                self._rest,
                self._enabled,
                self.max_step,
                0.0 if not dt else 1.0 / dt,
                max(1, self.per_world),
            ],
            outputs=[state.particle_q, state.particle_qd],
            device=self.model.device,
        )

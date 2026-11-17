# SPDX-License-Identifier: Apache-2.0
"""A moulded plug on the leading end of the fed cable.

**What makes this task a cable-routing task rather than a rod-pushing one.** A
bare rod threads a bore by being thinner than it everywhere; a terminated cable
has one rigid, oversized, non-round place that has to clear the run, and that
place is the thing an operator actually worries about. The connector is the
binding dimension here by construction -- :data:`PLUG_W` is 15 mm in a 17 mm
bore, 1 mm of clearance a side, against the cable's own 3.5 mm.

**Black, as the clip's is.** The reference cable is a black round cable with a
small black moulded head -- a strain relief that swells from the cable's gauge
to the overmold and ends in a rounded nose -- so the head is the overmold box, a
narrower nose box and a rounded cap, all painted the cable's black at a satin
finish. It was a blue USB-C shell (``ACCENT_PLUG``) inside a cream tunnel; the
colour told the tunnel and the plug apart in a picture, and the picture is
what this now matches instead.

**The head is shapes on the rod's last capsule, not a body of its own, and that
is a correction.** The first version welded the connector to a separate
``add_link`` body with ``add_joint_fixed``, and built that way it **never
moved**: over a whole episode the cable advanced 209.9 mm and the plug body sat
at its spawn pose. ``Ref.bodies`` and the island composition both go through
``owned_indices(entity, "owned_bodies", "bodies")``, and
``EntityCable.owned_bodies`` returns ``_rod_spans[0]`` -- the rod's own capsules,
closed before the weld is built -- so the plug body belonged to no entity, was
not placed in the rod's VBD island, and the fixed joint spanned two solvers with
nothing to enforce it. Putting the shapes on the terminal capsule makes all of
it moot: the head rides a body the rod already owns, the proxy channel picks its
shapes up through ``shape_body``, and it is rigid with respect to the cable's
end by construction. The plug is scored only through where it drags the cable's
last capsule, which is what a connector does.

**Boxes, not a mesh.** MuJoCo collides a mesh as its convex hull, and what
matters here is a stated cross-section a test can compare against the bore;
:func:`fits_bore` asserts it, which "approximately its bounding shape" cannot.
"""

from __future__ import annotations

import newton
import warp as wp

from robosimstudio.core.render.finish import SATIN
from robosimstudio.scene.registry import register_entity
from robosimstudio_materials.rod import EntityCable

from ..look import BLACK
from .rod import RADIUS

__all__ = [
    "CAP_R",
    "NOSE_H",
    "NOSE_L",
    "NOSE_W",
    "PLUG_H",
    "PLUG_L",
    "PLUG_W",
    "EntityTunnelCable",
    "head_length",
    "plug_boxes",
    "plug_cap",
]

PLUG_W = 0.015
"""The overmold's width [m] -- across the bore, and the dimension that binds.

Sized *from* the bore rather than to a datasheet: 1 mm of clearance a side at
``box.BORE = 0.017``. Anything under about 12 mm and the head passes more easily
than the cable does, which would make the connector decoration.
"""

PLUG_H = 2.0 * RADIUS
"""The overmold's height [m]: the cable's own diameter, so the head rides the
bore floor exactly as the cable does and the weld introduces no step for the
rod to climb. It is why the plug is wide rather than merely bigger."""

PLUG_L = 0.020
"""The overmold's length along the cable [m]. The part a hand grips."""

NOSE_W = 0.011
NOSE_H = 0.008
NOSE_L = 0.008
"""The moulded nose ahead of the overmold [m]: the strain relief narrowing
toward the rounded end. Proportioned so it stays inside the overmold's section
and never binds before the overmold does."""

CAP_R = 0.004
"""The rounded end [m]: a sphere on the nose's tip, the same height as the
nose."""


def head_length() -> float:
    """How far the head reaches past the terminal capsule's centre [m]: the
    overmold, the nose and the cap's radius."""
    return PLUG_L + NOSE_L + CAP_R


def plug_boxes(pitch: float) -> tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...]:
    """``(label, centre, half_extents)`` for each box, in the terminal capsule's frame.

    The rod's chain runs along its own local **+z**, and the terminal *body*
    sits half a capsule short of the rod's actual end, so the head starts at
    ``pitch / 2`` rather than at 0. Getting that wrong buries the overmold's
    first 6 mm inside the last capsule, which reads as a plug that has been on
    the cable all along and collides with it forever.
    """
    tip = 0.5 * float(pitch)
    return (
        ("overmold", (0.0, 0.0, tip + PLUG_L / 2), (PLUG_W / 2, PLUG_H / 2, PLUG_L / 2)),
        ("nose", (0.0, 0.0, tip + PLUG_L + NOSE_L / 2), (NOSE_W / 2, NOSE_H / 2, NOSE_L / 2)),
    )


def plug_cap(pitch: float) -> tuple[tuple[float, float, float], float]:
    """The rounded end as ``(centre, radius)`` in the terminal capsule's frame:
    a sphere whose centre is the nose's far face, so it rounds it off rather
    than standing ahead of it as a bead."""
    tip = 0.5 * float(pitch)
    return (0.0, 0.0, tip + PLUG_L + NOSE_L), CAP_R


def fits_bore(bore: float) -> bool:
    """Whether a square bore of side ``bore`` passes the head at all.

    The task's own premise, as a function rather than a comment, so
    ``tests/test_rgbs_tunnel.py`` can assert it and a variant that retunes
    either number cannot quietly stop being possible.
    """
    return max(PLUG_W, PLUG_H, NOSE_W, NOSE_H, 2.0 * CAP_R) < float(bore)


@register_entity("tunnel_cable")
class EntityTunnelCable(EntityCable):
    """The library rod, with a moulded plug fixed to the end that leads."""

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        super().build(builder, episode)
        spec = self.spec
        # The rod's last capsule -- verified to be the +y leading end, which is
        # the one that enters the tunnel. A plug on the other end would thread
        # the tail through and leave the connector on the bench.
        terminal = int(self._rod_spans[0][1]) - 1
        self.plug_body = terminal
        pitch = float(spec.length) / int(spec.segments)
        cfg = builder.default_shape_cfg.copy()
        cfg.density = 1400.0
        cfg.mu = float(spec.mu_cable)
        cfg.gap = float(spec.contact_gap)
        cfg.mu_torsional = 0.0
        cfg.mu_rolling = 0.0
        self.plug_shapes: list[int] = []
        for label, centre, half in plug_boxes(pitch):
            self.plug_shapes.append(
                builder.add_shape_box(
                    terminal,
                    xform=wp.transform(wp.vec3(*[float(v) for v in centre]), wp.quat_identity()),
                    hx=float(half[0]),
                    hy=float(half[1]),
                    hz=float(half[2]),
                    cfg=cfg,
                    label=f"{spec.name}_{label}",
                )
            )
        centre, radius = plug_cap(pitch)
        self.plug_shapes.append(
            builder.add_shape_sphere(
                terminal,
                xform=wp.transform(wp.vec3(*[float(v) for v in centre]), wp.quat_identity()),
                radius=float(radius),
                cfg=cfg,
                label=f"{spec.name}_cap",
            )
        )

    def apply_display(self, builder: newton.ModelBuilder) -> None:
        """The rod's colour on the rod, black satin on the head.

        ``EntityCable.apply_display`` gives everything outside ``_rod_spans[2]``
        ``ROBOT_DARK`` -- correct for the fingertip hulls it was written for, and
        wrong for a connector, which would come out the same colour as the hand
        holding it. Repainted here rather than left to build order for the
        reason that method already gives: unpainted is not neutral. Satin rather
        than the cable's gloss: a moulded strain relief is duller than the
        jacket it grips.
        """
        super().apply_display(builder)
        for shape in getattr(self, "plug_shapes", ()):
            builder.shape_color[int(shape)] = BLACK
            self.finishes[int(shape)] = SATIN

    def apply_model_materials(self, model: newton.Model) -> None:
        """The plug is the cable, as far as friction is concerned.

        ``EntityCable.apply_model_materials`` writes ``mu_cable`` to the shapes in
        ``_rod_spans[2]`` and ``mu_base`` (2.0) to everything it does not
        recognise. The plug is built after that span closes, so it lands in the
        ``else`` branch -- and 2.0 against the box's ``mu_surface`` is a
        connector that slides through the bore at mu 0.047 while the cable behind
        it drags at 0.26, which is backwards and which nothing raises.

        Written to ``mu_cable`` rather than named in ``surface_labels``: the head
        has to be *grippable*, and a surface label would give it the tabletop's
        friction and a hand that cannot hold it.
        """
        super().apply_model_materials(model)
        shapes = tuple(int(s) for s in getattr(self, "plug_shapes", ()))
        if not shapes:
            return
        mu = model.shape_material_mu.numpy()
        ke = model.shape_material_ke.numpy()
        kd = model.shape_material_kd.numpy()
        for shape in shapes:
            mu[shape] = float(self.spec.mu_cable)
            ke[shape] = float(self.spec.contact_ke)
            kd[shape] = float(self.spec.contact_kd)
        model.shape_material_mu.assign(mu)
        model.shape_material_ke.assign(ke)
        model.shape_material_kd.assign(kd)

    def plug_offset(self) -> float:
        """How far the overmold's centre sits ahead of the terminal capsule [m].

        The head is rigid on that capsule, so this offset plus the capsule's own
        pose *is* where the connector is -- there is no second body to read. The
        routine's pull targets it.
        """
        pitch = float(self.spec.length) / int(self.spec.segments)
        return 0.5 * pitch + PLUG_L / 2.0

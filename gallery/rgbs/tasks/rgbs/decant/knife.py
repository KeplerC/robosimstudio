# SPDX-License-Identifier: Apache-2.0
"""The safety cutter: a compact body the jaws close on, and a short blade at its nose.

**Box primitives on one free body, not a mesh.** MuJoCo collides every mesh
geom as its **convex hull**, so a body-plus-blade authored as one mesh arrives
as a solid wedge with the blade's clearance filled in -- the failure
``EmbodimentSpec.decompose_colliders`` exists to undo on this very robot's
fingertips, where the hull stood 13 mm proud of the drawn pad and the hand
gripped air. ``add_shape_box`` calls are MuJoCo ``box`` geoms and there is no
hull anywhere near them; box-box under ``enable_multiccd`` is also the
best-conditioned manifold this stack produces.

The other two constructions were considered and rejected. A **mesh with
``collider="slabs"``** *fits* the collider from the mesh's vertices, so the
cutting edge would land wherever a band decomposition put it -- and the edge is
not decoration here, it is the **domain of the cut law** (``t`` in
``[0, edge_len]``, ``|perp| <= kerf`` are both evaluated against it). A law whose
support is an artefact of a decomposition heuristic is not a law. A **compound
collider** (``tool_hanging``'s scissors) exists for a mesh with a cavity a hull
would seal; a cutter has no cavity, so it buys nothing for 30 files.

**The tool is the clip's.** ``sample_videos/box_decanting.mp4`` cuts with a
palm-sized safety cutter -- a slate-blue oval body with a red round slider on
its top face and a charcoal grip block at the butt, the blade a few millimetres
out of the nose -- not the 150 mm retractable knife on a rest block this
shipped with. It lies flat on the mat beside the carton and goes back down on
the mat, which is what removed the rest and its record.

This is an ``ObjectSpec`` subclass rather than an extras entity, and that is
load-bearing: only the ``objects`` bin gets a full ``EntityHandle``, so only this
way can a goal say ``(gripped knife.body)`` at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import newton
import numpy as np
import warp as wp

from robosimstudio.core.render.finish import Finish
from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.build.bodies import set_body_mass
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.asset import ObjectSpec, ShapeSpec

from ..look import BLACK_PLASTIC, STEEL

__all__ = [
    "BLADE_STEEL",
    "CUTTER_CHARCOAL",
    "CUTTER_RED",
    "CUTTER_SLATE",
    "EntityBoxCutter",
    "SpecBoxCutter",
    "add_blade",
    "add_body",
    "add_front",
    "blade_edge_local",
    "front_half",
    "front_offset",
    "tip_lead",
]

BLADE_STEEL = replace(STEEL, metallic=0.45, roughness=0.22)
"""How the blade reflects light.

The suite's :data:`~..look.STEEL` is 0.95 metallic, which is right for a bar of
stock and wrong for a 1.5 mm plate seen edge-on: a near-fully metallic surface
shows almost nothing but its REFLECTION, and against this cell's dim dome the
blade came back as a black spike sticking out of a grey tool -- darker than the
mat behind it, in every camera. Dropping the metallic to 0.45 lets
:attr:`.SpecBoxCutter.blade_color`'s near-white albedo carry the brightness and
keeps a tight specular on top of it, which is what the eye reads as "steel"
here."""

CUTTER_SLATE = (0.115, 0.130, 0.175)
"""The cutter's body, sampled off ``sample_videos/box_decanting.mp4``.

The tool in that clip is a **slate blue-grey** moulding. Its lit face reads
sRGB 132/148/174 (linear 0.23/0.30/0.42) and its shaded one 99/93/111, so the
albedo sits well under the lit sample -- this is the shaded end of that range,
which is what a satin plastic wants under the lab dome.

Stated here rather than in :mod:`..look` because it is one asset's moulding
rather than a surface the bench shares. ``look.CUTTER_GREY`` (0.30/0.31/0.33)
is what shipped before and is a neutral grey with no blue in it at all, which
is why the tool read as a concrete offcut."""

CUTTER_CHARCOAL = (0.030, 0.030, 0.033)
"""The grip block at the butt: a charcoal foam pad, sampled at sRGB 50/50/52 on
its lit face (linear 0.033) and near-black in shadow. Slightly above
:data:`~..look.BLACK` on purpose -- the block is a textured foam, not a print,
and under the dome it reads as dark grey rather than as a hole."""

CUTTER_RED = (0.55, 0.06, 0.05)
"""The round slider on the top face, sampled at sRGB 212/79/78. Drawn as one
thin non-colliding cylinder: two square centimetres of colour, but it is the
one mark that says which way up the tool is in every frame of the clip."""

GRIP_SPAN = 0.09396
"""YAM ``linear_4310`` fingertip separation at full open [m]."""


@dataclass(frozen=True, kw_only=True)
class SpecBoxCutter(ObjectSpec):
    """A palm-sized safety cutter: a 100 x 30 x 20 mm body with 15 mm of blade out of its nose.

    The body's long axis is local ``+x`` with the nose at ``+x``; ``+z`` is the
    top face the slider sits on. Lying flat it stands on its underside at
    ``z = -body_half[2]``.
    """

    entity: str = "box_cutter"
    category: str = "box_cutter"
    name: str = "knife"

    body_half: tuple[float, float, float] = (0.050, 0.015, 0.010)
    """Body half-extents [m] -- 100 x 30 x 20 mm, the clip's tool against the
    mat's 25 mm grid: four squares long, a bit over one wide, lying on the mat.

    30 mm across is what the jaws close on, against a 94 mm span: comfortably
    open, and well clear of the stall band the ``gripped`` atom reads. The
    shipped 150 x 28 x 18 mm knife was a grey ruler laid across the parcel; a
    100 mm body had been tried against the bench's D6 holder and reverted, and
    that failure was the holder's servo gains (fitted to the 150 mm knife),
    which ``rgbs/scripts/rgbs_decant_bench.py`` re-fits for this body."""

    butt_len: float = 0.040
    """How much of the body, from the butt, is the charcoal grip block [m]. The
    rest is the slate moulding. Appearance only: the two are one collider."""

    blade_half: tuple[float, float, float] = (0.0075, 0.00075, 0.006)
    """Blade half-extents [m] -- 15 x 1.5 x 12 mm.

    **1.5 mm of thickness is a numerical floor, not a real blade** (a real one
    is 0.45 mm). It is 7.5x the arrangement's 0.2 mm contact activation gap, so
    contact opens on a fifth of the blade rather than on six times it, and
    box-box stays conditioned. The deviation is absorbed by
    :attr:`.SpecSealTape.kerf`, which is a technique tolerance rather than a
    slit width -- sharpness lives in the pierce gate, not in the geometry, the
    same division polish makes between a drum's abrasive and Preston's law.

    15 mm out of the nose is a safety cutter's exposure. It is short on purpose
    twice over: one press cannot reach more than a sliver of the tape, so the
    DRAW is the task; and a tilted hold puts the tip well below the body's nose
    corner, which is the whole clearance argument in :attr:`blade_at`."""

    blade_at: tuple[float, float, float] = (0.0575, 0.0, -0.003)
    """Where the blade's centre sits in the body's frame [m]: out of the NOSE
    along the body's axis, its lower edge 1 mm above the body's underside.

    Two constraints and one number. Lying flat on the mat the blade must not
    touch it, so the edge stays inside the body's own thickness: bottom at
    ``z = -0.009`` against an underside at ``-0.010``. And in the hand the edge
    must reach the tape with the body clear of the flaps, which a level hold
    cannot give (the underside would land first) -- so the hold is TILTED
    nose-down about the jaw's closing axis, and the tip leads. At 30 degrees the
    tip stands 7.0 mm below the body's nose corner, at 35 degrees 7.8, at 45
    degrees 9.9 and at 50 degrees 11.2 -- computed from these numbers by
    :func:`tip_lead`, against the 5-10 mm the seat drives the tip down through
    an arm that gives most of it back. Read through ``blade_edge_local``;
    nothing restates them."""

    cut_tilt: float = 50.0
    """The nose-down tilt the blade is designed to cut at [deg], about the jaw's
    closing axis, and the tilt the build's clearance assertion is stated at.

    **Steep, and both reasons are measurements.** :func:`tip_lead` gives the
    tip's stand-off below the body's own nose corner as 7.8 mm at 35 degrees,
    9.9 at 45 and 11.2 at 50 -- and the seat drives the tip 5-10 mm down
    through an arm that gives most of that back, so at 35 the body's corner
    reaches the tape while the blade is still working. And the press's moment
    on the grasp goes as the HORIZONTAL run from the pads to the tip,
    ``a*cos(tilt) + b*sin(tilt)``: at the routine's 30 mm grip station that is
    42 mm at 35 degrees and 32 mm at 50, which is a fifth off what the pads
    have to hold. ``routine_decant.py::CUT_TILT`` reads it from the
    document."""

    button_radius: float = 0.008
    button_at: float = -0.030
    """The red slider: an 16 mm disc on the top face, 20 mm in from the butt."""

    mass: float | None = 0.060
    """A safety cutter is 50-70 g.

    Below ``f_pierce / g`` = 0.61 kg, deliberately: a knife *set down* on the
    seam cannot pierce it under its own weight, so the cut law needs no special
    case for a resting blade."""

    friction: float | None = 0.35
    """Hardened steel on tape and kraft liner. Its real job is to set the drag
    ``mu * press`` that shoves an unpinned carton, so it is one of the two
    numbers ``rgbs/scripts/rgbs_decant_bench.py --shove`` fits."""

    ke: float | None = 6.0e4
    kd: float | None = 500.0
    shape: ShapeSpec | None = None
    color: tuple[float, float, float] | None = CUTTER_SLATE
    """The body: the slate blue-grey moulding the reference clip shows."""
    finish: Finish | None = BLACK_PLASTIC
    """Satin ABS. Stated rather than left to ``finish_for``, which keys off the
    colour and would give a dark body the same treatment as the matte black
    print the rest sits on -- so the body went out with no sheen at all and
    read as a slab rather than a moulding."""
    blade_color: tuple[float, float, float] = (0.88, 0.90, 0.93)
    """Fresh blade: near-white, because a polished edge under a dome light is a
    mirror and its albedo is the wrong lever for how bright it reads."""

    def __post_init__(self) -> None:
        if self.shape is None:
            # The slate front is what `EntityRigid._add_primitive` builds; the
            # butt block, the slider and the blade are this entity's.
            object.__setattr__(self, "shape", ShapeSpec(kind="box", half_extents=front_half(self)))
        super().__post_init__()


def blade_edge_local(spec: SpecBoxCutter) -> tuple[np.ndarray, np.ndarray, float]:
    """``(p0, unit_dir, length)`` of the CUTTING EDGE in the knife's own frame.

    The blade box's lower edge, running along the knife's long axis with the
    TIP at ``p0 + length * unit_dir``. This is the one place the edge is
    derived: :class:`~.seam.EntitySealTape` integrates its cut law along it, the
    routine aims with its tip, and ``rgbs/scripts/rgbs_decant_bench.py`` drives to
    it -- so nothing can measure a cut against an edge the knife is not carrying.
    """
    cx, cy, cz = (float(v) for v in spec.blade_at)
    hx, _, hz = (float(v) for v in spec.blade_half)
    p0 = np.array([cx - hx, cy, cz - hz], dtype=np.float64)
    return p0, np.array([1.0, 0.0, 0.0]), 2.0 * hx


def tip_lead(spec: SpecBoxCutter, tilt_deg: float) -> float:
    """How far the blade's tip stands below the body's lowest nose corner [m]
    when the body is tilted ``tilt_deg`` nose-down about its own ``y``.

    The clearance argument of a tilted cut, as a number: positive means the tip
    lands first and the seat can drive it that deep before the body touches.
    """
    p0, u, length = blade_edge_local(spec)
    tip = p0 + length * u
    t = np.radians(float(tilt_deg))
    s, c = float(np.sin(t)), float(np.cos(t))

    def drop(x: float, z: float) -> float:
        return -x * s + z * c

    corner = drop(float(spec.body_half[0]), -float(spec.body_half[2]))
    return corner - drop(float(tip[0]), float(tip[2]))


def add_blade(builder, body: int, spec: SpecBoxCutter, cfg, *, name: str) -> int:
    """Draw the blade onto ``body``, returning its shape index.

    The ONE place the blade is built. :class:`EntityBoxCutter` calls it and so
    does ``rgbs/scripts/rgbs_decant_bench.py``'s holder, so a bench that measures a cut made
    by a blade the task is not carrying is not constructible -- ``polish_bench``'s
    rule, which is why it takes its geometry from ``machine_frame()``.
    """
    return builder.add_shape_box(
        body,
        xform=wp.transform(wp.vec3(*(float(v) for v in spec.blade_at)), wp.quat_identity()),
        hx=float(spec.blade_half[0]),
        hy=float(spec.blade_half[1]),
        hz=float(spec.blade_half[2]),
        cfg=cfg,
        color=spec.blade_color,
        label=f"{name}_blade",
    )


def front_half(spec: SpecBoxCutter) -> tuple[float, float, float]:
    """The slate front's half-extents [m]: the body less the butt block."""
    return (
        0.5 * (2.0 * float(spec.body_half[0]) - float(spec.butt_len)),
        float(spec.body_half[1]),
        float(spec.body_half[2]),
    )


def front_offset(spec: SpecBoxCutter) -> float:
    """Where the slate front's centre sits along the body [m]: forward of the
    body's origin by half the butt block, so the whole tool is centred on it."""
    return 0.5 * float(spec.butt_len)


def add_front(builder, body: int, spec: SpecBoxCutter, cfg, *, name: str) -> int:
    """The slate front, onto a bare body. The entity gets its front from
    ``EntityRigid``'s primitive and re-seats it by :func:`front_offset`; the
    bench's holder has no primitive and builds it here, from the same numbers."""
    hx, hy, hz = front_half(spec)
    return builder.add_shape_box(
        body,
        xform=wp.transform(wp.vec3(front_offset(spec), 0.0, 0.0), wp.quat_identity()),
        hx=hx,
        hy=hy,
        hz=hz,
        cfg=cfg,
        color=spec.color,
        label=name,
    )


def add_body(builder, body: int, spec: SpecBoxCutter, cfg, *, name: str) -> tuple[int, int]:
    """The butt block and the slider, onto a body whose slate front already
    exists. Returns ``(butt_shape, button_shape)``.

    Shared with the bench's holder for the same reason as :func:`add_blade`: the
    holder must carry the body the task carries, or the press it measures is
    reacted by a different tool.
    """
    bx, by, bz = (float(v) for v in spec.body_half)
    butt = builder.add_shape_box(
        body,
        xform=wp.transform(wp.vec3(-bx + 0.5 * float(spec.butt_len), 0.0, 0.0), wp.quat_identity()),
        hx=0.5 * float(spec.butt_len),
        hy=by,
        hz=bz,
        cfg=cfg,
        color=CUTTER_CHARCOAL,
        label=f"{name}_butt",
    )
    decal = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
    # Newton's cylinder stands along its local z, so an unrotated one on the
    # top face is a disc.
    button = builder.add_shape_cylinder(
        body,
        xform=wp.transform(wp.vec3(float(spec.button_at), 0.0, bz + 0.001), wp.quat_identity()),
        radius=float(spec.button_radius),
        half_height=0.001,
        cfg=decal,
        color=CUTTER_RED,
        label=f"{name}_button",
    )
    return int(butt), int(button)


@register_entity("box_cutter")
class EntityBoxCutter(EntityRigid):
    """The slate front (built by ``EntityRigid``), the butt block, the slider and the blade."""

    spec: SpecBoxCutter

    def __init__(self, spec: SpecBoxCutter, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.front_shape: int = -1
        self.butt_shape: int = -1
        self.button_shape: int = -1
        self.blade_shape: int = -1

    def apply_display(self, builder: newton.ModelBuilder) -> None:
        """The body's finish on the mouldings, steel on the blade; the butt and
        the slider keep the colours :func:`add_body` painted."""
        super().apply_display(builder)
        if self.blade_shape >= 0:
            self.finishes[self.blade_shape] = BLADE_STEEL
        if self.butt_shape >= 0:
            builder.shape_color[self.butt_shape] = CUTTER_CHARCOAL
        if self.button_shape >= 0:
            builder.shape_color[self.button_shape] = CUTTER_RED

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        # `_add_primitive` builds the free body, the free joint, the articulation
        # and the spec's `shape` -- the slate front -- centred on the body's
        # origin. It is re-seated forward here so the body's origin stays at the
        # whole tool's centre and the butt block fills the rest.
        super().build(builder, episode)
        spec = self.spec
        self.front_shape = builder.shape_count - 1
        builder.shape_transform[self.front_shape] = wp.transform(
            wp.vec3(front_offset(spec), 0.0, 0.0), wp.quat_identity()
        )
        cfg = self._contact_cfg(builder)
        self.butt_shape, self.button_shape = add_body(builder, self.body, spec, cfg, name=spec.name)
        self.blade_shape = add_blade(builder, self.body, spec, cfg, name=spec.name)
        # `_add_primitive` already set the declared mass, on the front alone --
        # the rest did not exist yet. `set_body_mass` rescales and preserves the
        # distribution, so re-stating it here is what makes the declared mass the
        # last word AND the inertia include the whole tool.
        if spec.mass is not None:
            set_body_mass(builder, self.body, float(spec.mass))
        self._assert_geometry()

    def _assert_geometry(self) -> None:
        spec = self.spec
        p0, _, edge_len = blade_edge_local(spec)
        underside = -float(spec.body_half[2])
        assert float(p0[2]) > underside, (
            f"the blade's edge sits at z={p0[2] * 1e3:.1f} mm and the body's underside at "
            f"{underside * 1e3:.1f} mm: lying flat on the mat the blade rests on it -- raise `blade_at[2]`"
        )
        assert float(p0[0]) >= float(spec.body_half[0]) - 1.0e-9, (
            f"the blade starts at x={p0[0] * 1e3:.1f} mm, inside a body whose nose is at "
            f"{spec.body_half[0] * 1e3:.1f} mm; the edge the law integrates along must be OUT of the tool"
        )
        lead = tip_lead(spec, float(spec.cut_tilt))
        assert lead >= 0.006, (
            f"tilted {spec.cut_tilt:.0f} degrees nose-down the tip leads the body's nose corner by only "
            f"{lead * 1e3:.1f} mm; the seat drives it 4-6 mm through the tape and the body must not "
            "land first -- lengthen the blade, lower it, or tilt further"
        )
        assert 2.0 * float(spec.body_half[1]) <= GRIP_SPAN - 0.020, (
            f"the body is {2 * spec.body_half[1] * 1e3:.0f} mm across a "
            f"{GRIP_SPAN * 1e3:.0f} mm jaw span, with under 20 mm to spare"
        )
        assert 2.0 * float(spec.blade_half[1]) >= 5.0 * 2.0e-4, (
            f"the blade is {2 * spec.blade_half[1] * 1e3:.2f} mm thick against the arrangement's "
            "0.2 mm contact gap; below ~5x the gap the contact activates on a large fraction of "
            "the blade's own thickness and the press force stops meaning what it says"
        )
        assert edge_len <= 0.25 * 0.100, (
            f"the blade's edge is {edge_len * 1e3:.0f} mm against 100 mm of tape. It must be a "
            "small fraction of it, or one press cuts the whole strip and the DRAW -- which is "
            "what this task is about -- becomes decoration"
        )
        assert float(spec.butt_len) < 2.0 * float(spec.body_half[0]), "the butt block is the whole body"

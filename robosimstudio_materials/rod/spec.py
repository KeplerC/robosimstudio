# SPDX-License-Identifier: Apache-2.0
"""What a rod *is*, as data: the cable and the matter it is made of.

Both names re-export from :mod:`robosimstudio_materials.rod` and as
``rss.RodSpec`` / ``rss.RodMaterialSpec``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from robosimstudio.core.render.finish import Finish
from robosimstudio.core.render.palette import DEFORMABLE
from robosimstudio.specs.params import LIVE, RECAPTURE, ui
from robosimstudio.specs.physics import SimRequirements

__all__ = ["RodMaterialSpec", "RodSpec"]


@dataclass(frozen=True, kw_only=True)
class RodSpec:
    """The rod, its material, and the wrap variant's anchor. Reference defaults."""

    stage: ClassVar[str] = "deformables"
    """Built in the deformables pass, so ``TaskSpec.deformables`` and the studio's material panel see it."""
    name: str = "cable"
    entity: str = "rod"
    """Which registered class builds it: the library's generic ``"rod"``. A task's own
    name for it (``"cable_rod"``) resolves to the same class."""
    start: tuple[float, float, float] = (-0.02, -0.4572, 0.011)
    """Where the first capsule goes [m], in the cell's **work-surface frame**: the
    reference's start, 20 mm short of the bench centre, one radius above the top."""
    direction: tuple[float, float, float] = (0.0, 1.0, 0.0)
    length: float = 0.9144
    """36 inches [m]."""
    radius: float = 0.0110
    segments: int = 28
    """Capsule count. Segment spacing must stay above 2*radius."""
    density: float = 350.0
    stretch_ke: float = 2.0e5
    stretch_kd: float = 1.0
    bend_ke: float = 0.04
    bend_kd: float = 0.06
    shear_ke: float | None = None
    """Per-joint transverse shear stiffness [N/m], or ``None`` for :attr:`stretch_ke`.

    Newton's rod joint has four slots -- stretch, shear, bend, twist -- and the
    capsule chain used to state two numbers and let the builder pair them
    (shear from stretch, twist from bend). ``None`` keeps that pairing, so a
    spec that says nothing builds the model it always built; a number splits
    the pair. See :meth:`drive_gains` for the eight resolved values."""
    shear_kd: float | None = None
    """Shear damping [N*s/m], or ``None`` for :attr:`stretch_kd`."""
    twist_ke: float | None = None
    """Per-joint twist stiffness [N*m/rad], or ``None`` for :attr:`bend_ke`.

    The number that decides whether a roll put into one end reaches the other:
    a data cable's jacket twists far more easily than it bends, a stranded
    steel cable the other way round."""
    twist_kd: float | None = None
    """Twist damping [N*m*s/rad], or ``None`` for :attr:`bend_kd`."""
    dahl_eps_max: float = 0.0
    """How much bend the cable may keep, as the Dahl hysteresis envelope [rad]:
    the bending (and twisting) moment friction can hold at a joint is
    ``bend_ke * dahl_eps_max``, so a cable loaded past this curvature slips and
    one loaded short of it springs back only as far as the friction lets it.
    ``0.0`` is elastic: the joint stores no plastic memory and VBD never runs
    its Dahl kernels. **REBUILD-arms the model**: the solver decides whether to
    run those kernels once, at construction, from whether any joint was built
    with both this and :attr:`dahl_tau` positive -- a cable built at 0 cannot be
    given memory by a slider, and one built positive can be tuned live, down to
    anything above 0. Newton's own cable-bundle example uses 2.0."""
    dahl_tau: float = 0.1
    """Over how much curvature change the friction builds up to its envelope
    [rad]: the Dahl transition length. Small and the memory is sharp -- the cable
    keeps nearly every bend put into it; large and it is a slow drag that lets
    most of a bend spring back. Read only where :attr:`dahl_eps_max` is positive;
    Newton's own example uses 0.1."""
    contact_ke: float = 1.0e4
    contact_kd: float = 0.1
    contact_gap: float = 0.01
    mu_base: float = 2.0
    mu_cable: float = 60.0
    mu_gripper: float = 60.0
    mu_surface: float = 0.02
    surface_labels: tuple[str, ...] = ("table",)
    color: tuple[float, float, float] = DEFORMABLE
    finish: Finish | None = None
    """How the rod's surface reflects light under the path-traced backend, or
    ``None`` for the finish keyed to :attr:`color`. Display only."""
    anchor_grasp: tuple[float, float, float] | None = None
    """Where the anchoring arm pre-grips the cable [m], in the work-surface frame, or ``None``.
    The wrap variant's second hand: posed kinematically before the first step,
    holding at :attr:`anchor_grip`, its targets never written again."""
    anchor_arm: str = ""
    """Which arm takes :attr:`anchor_grasp`; empty means the embodiment's last
    declared arm. The entity cannot see :class:`~robosimstudio.specs.Binding`, so a
    bimanual task states the name it gives ``Binding.support_arm`` (the shipped
    cable route does)."""
    anchor_grip: float = 0.010
    """Anchor finger command [m]: ~1.5 mm of interference a side. The working
    arm squeezes at 0.003 because it must drag the rod around a peg; commanding
    that here loads the anchor wrist with the full ±80 N squeeze reaction
    through the 20x proxy and deflects the pinch 30 mm off the rod. The anchor
    only has to pin its spot -- at mu 60, a light pinch does."""
    scene_friction: bool = False
    """Whether this rod rewrites every shape's friction at bind: all to
    :attr:`mu_base`, then the cable, the gripper and the surface. The two shipped
    rod suites, which own their cells, turn it on. Off by default because it
    retunes the rest of the scene, including a second rod placed beside it for
    comparison (whichever binds last sets both ``mu_cable``)."""
    allow_grasp_latch: bool = True
    """Whether a robot hand may kinematically latch a cable segment.

    Contact-only benchmark variants disable this so their grasps are carried by
    collision and friction. World-fixed :attr:`anchors` are independent fixture
    constraints.
    """
    filter_fingertip_scenery: bool = True
    """Whether the legacy fingertip/static collision exclusions are installed.

    Contact-only variants disable this so fingers collide with the bench, pegs,
    and terminal fixtures.
    """
    anchors: tuple[str, ...] = ()
    """Which ends are **clamped** to the world where they were built: ``"start"``,
    ``"end"`` or both. A cable strung between two posts says ``("start", "end")``.

    A clamp on both engines -- the end's position, its tangent and its twist
    are held, so a cable anchored at one end is a cantilever, not a pendulum.
    The capsule chain (VBD) welds the end capsule to the world with a fixed
    joint. The Cosserat rod (SuperDex) holds the end node and its neighbour
    where they were built, which fixes the end element's tangent, and holds
    that element's twist at zero.

    Held by its end node alone, the SuperDex rod is a pin: a level 300 mm cable
    anchored at its start swung down 74 degrees in 120 steps (288 mm of tip
    drop). Clamped, it sags 6 mm, and the capsule chain 15-26 mm. There is no
    pin option."""
    bow_range: tuple[float, float] = (0.0, 0.0)
    """Lateral bow at spawn, sampled per episode as ``[lo, hi]`` metres.

    Zero-width (straight) by default. A non-zero range bends each episode's rod
    by a seeded amount, so a policy has to read the rod's shape.

    The bow is a half-cosine in the plane perpendicular to :attr:`direction` and
    the world up: maximal at the middle, zero at both tips, which is the shape a
    rod laid down by hand actually takes. Amplitude is signed, so a range
    spanning zero gives bows both ways.

    Read per episode off ``episode.overrides["cable_bow"]`` as a unit fraction,
    so the draw is the sampler's and the metres are the spec's. The four spawn
    ranges are inert unless a sampler supplies their fractions, and a spec that
    leaves them zero-width spawns byte-identically to one that never had them.
    """
    shift_range: tuple[float, float] = (0.0, 0.0)
    """Lateral offset drawn per episode [m], applied to the WHOLE rod.

    Zero-width by default. A bow leaves the tips in place; this moves them."""
    wave_range: tuple[float, float] = (0.0, 0.0)
    """Second-harmonic amplitude drawn per episode [m].

    Adds an S on top of the bow's single arc, so the spawn looks laid down
    rather than pulled taut. Zero-width by default."""
    yaw_range: tuple[float, float] = (0.0, 0.0)
    """In-plane rotation drawn per episode [deg], about the rod's own midpoint.

    Zero-width by default. Each station then sees a different standoff, which a
    shift cannot produce."""
    nodes: tuple[tuple[float, float, float], ...] | None = None
    """A polyline rest shape, in the work-surface frame -- or ``None`` for straight.

    For a coiled cable, a pre-routed harness or one digitised from a real cable.
    At least two nodes. When given it is the source of truth: ``length`` becomes
    its arc length, ``segments`` one fewer than its nodes, ``start`` its first
    node and ``direction`` its first leg, so every reader of those fields (mass,
    the latch's force ceiling, the foreign engine's node count) is unchanged.

    ``segments`` is the capsule count in the Newton implementation and the
    segment count between nodes in the Cosserat one, so both discretise the same
    polyline at the same resolution.
    """
    young_modulus: float = 5.0e6
    """Elastic modulus [Pa] for the Cosserat (SuperDex) rod's ``EA``, ``EI`` and
    ``GJ``. The capsule chain uses :attr:`stretch_ke` / :attr:`bend_ke` instead.
    5.0e6 is the order of a rubber-jacketed cord."""
    poisson_ratio: float = 0.45
    """Near-incompressible, as an elastomer is. Only ``G = E / 2(1 + v)`` uses it."""
    stiffness_damping: float = 1.0e-3
    """Damping proportional to stiffness, for a continuum implementation."""
    axial_stiffness: float | None = None
    """``EA`` [N] when the cable is not one homogeneous section, or ``None`` to
    derive it from :attr:`young_modulus` like ``EI`` and ``GJ``.

    A data cable is a soft jacket round copper: its bending is the jacket's and
    its stretch the copper's. One modulus cannot state both -- a Cat6 bending
    stiffness of 2e-4 N*m^2 at r = 3 mm is E = 3.1 MPa, which derived EA = 89 N
    from, so the cable stretched 1 % under the weight of a plug and a hand."""
    contact_penalty: float = 8.0e7
    """Contact penalty [Pa/m] for an implementation that has one (SuperDex).

    The default is the foam-calibrated figure every SuperDex actor shipped
    with. Against a stiff part it is a thousand times too soft: a 6 mm cable
    pressed into a polymer clip's lips needs ~16 N to open them, and at 8e7 it
    simply sank 2 mm into the lips instead (measured; the clip never moved).
    The engine's own tendon-through-eyelet example uses 1e11."""
    contact_threshold: float | None = None
    """Penalty onset distance [m] when this rod is the collider, or ``None`` for the engine's 1 mm."""
    contact_smoothing: float | None = None
    """``penalty_smoothing_half_distance`` [m], or ``None`` for the engine's."""
    friction_falloff: float | None = None
    """Slip speed [m/s] below which friction is proportional to speed (no
    static friction), or ``None`` for the engine's 10 mm/s. A cable held by a
    pinch creeps at roughly ``load / (mu N) * friction_falloff``."""
    skin: bool = True
    """Contact on a tubular surface of :attr:`radius` rather than on the axis.

    A Cosserat centerline samples contact on its own nodes, so it rests with
    its *axis* on whatever it touches -- one radius inside the bench. True
    builds the engine's tubular skin, which is what its own rod examples do."""
    skin_segments: int = 8
    """Sides of the tubular skin (see :attr:`skin`)."""
    skin_samples: str | None = None
    """Contact quadrature per skin triangle (``"P1Q3"``, ``"P1Q6"``...), or ``None``
    for the engine's. A narrow feature pressing the cable -- a clip's lip -- needs
    more samples than a bench does, or it slips between them."""
    stress_free: str | tuple[tuple[float, float, float], ...] | None = None
    """The shape in which the rod stores no elastic energy, when it is not
    :attr:`nodes`: ``"straight"``, or a polyline with the same number of nodes
    and the same leg lengths (in any frame -- only its shape is read). ``None``
    rests at :attr:`nodes`.

    A rod with no rest/initial split rests in whatever shape it was laid out
    in: a cable built in an S on the bench, lifted, bowed back into the S (it
    hung 26 mm off the bench at a clip's foot, measured). Read by the SuperDex
    implementation, which builds the rod at this shape and moves it onto
    :attr:`nodes` before the first step."""
    self_contact: bool = False
    """Whether the rod touches itself: a strand pulled across another is pushed back
    instead of passing through it, so a knot stays a knot.

    Read by the foreign-engine rod (SuperDex tests the rod's skin against itself,
    segments within one and a half radii of each other along it left out); the
    capsule chain's own collision filter decides for it. A declared :attr:`sim`
    decides in its place, by its ``self_contact``. Off by default: it costs a broad phase over
    the rod's own samples every Newton iteration, which a cable that never folds
    back on itself does not need."""
    closed: bool = False
    """Whether the rod is a **loop**: :attr:`nodes` go round once and the last
    joins the first, with no ends -- a rubber band, an O-ring, a drive belt.

    Needs :attr:`nodes` (three or more, the first not repeated at the end), and
    then :attr:`segments` is the number of nodes, one element per node, and
    :attr:`length` the perimeter, closing leg included. Read by the foreign-engine
    rod, whose polyline can be periodic (SuperDex closes it with an element from
    the last node to the first, and its tubular skin with it); the capsule chain
    has two ends and refuses one. No :attr:`anchors`, :attr:`joins` or
    :attr:`stress_free`: a loop has no end to clamp, mould or straighten."""
    joins: tuple[tuple[str, str, float], ...] = ()
    """``((end, entity, length), ...)``: the last ``length`` metres of the rod at
    ``end`` (``"start"`` or ``"end"``) are moulded into ``entity`` -- a cable
    into its plug's boot -- welded node by node to that body's nearest nodes,
    with contact between the two switched off. Read by the foreign-engine
    implementation, which is the one whose bodies can be joined this way."""
    sim: SimRequirements | None = None
    """The asset's physics declaration, or ``None`` for the rod default."""

    def __post_init__(self) -> None:
        # The polylines are held as plain tuples whatever was given (``overhand()`` returns an
        # array), so the spec compares and hashes as the value a ``SpecFactory`` takes it for.
        if self.stress_free is not None and not isinstance(self.stress_free, str):
            free = np.asarray(self.stress_free, dtype=np.float64)
            if free.ndim == 2:
                object.__setattr__(self, "stress_free", tuple(map(tuple, free.tolist())))
        if self.nodes is None:
            if self.closed:
                raise ValueError(f"rod {self.name!r}: a closed rod needs nodes, the loop it goes round")
            return
        pts = np.asarray(self.nodes, dtype=np.float64).reshape(-1, 3)
        object.__setattr__(self, "nodes", tuple(map(tuple, pts.tolist())))
        if len(pts) < (3 if self.closed else 2):
            raise ValueError(
                f"rod {self.name!r}: nodes needs at least {3 if self.closed else 2} points, got {len(pts)}"
            )
        if self.closed:
            if self.anchors or self.joins or self.stress_free is not None:
                raise ValueError(f"rod {self.name!r}: a closed rod has no end to anchor, join or straighten")
            pts = np.vstack([pts, pts[:1]])  # the closing leg counts toward the length and the elements
        legs = np.diff(pts, axis=0)
        norms = np.linalg.norm(legs, axis=1)
        if not np.all(norms > 0.0):
            raise ValueError(
                f"rod {self.name!r}: nodes has a zero-length leg (a closed rod's first node is not repeated)"
            )
        # Frozen, so the echoes are written the one way a frozen dataclass allows.
        object.__setattr__(self, "length", float(norms.sum()))
        object.__setattr__(self, "segments", int(len(pts) - 1))
        object.__setattr__(self, "start", tuple(float(v) for v in pts[0]))
        object.__setattr__(self, "direction", tuple(float(v) for v in legs[0] / norms[0]))

    def drive_gains(self) -> dict[str, float]:
        """The eight numbers the cable joints are built with, the ``None`` pairs resolved.

        Keyed as :class:`RodMaterialSpec` names them (``stretch_ke`` ...
        ``twist_kd``), so the spec, the builder call and the tuning view agree
        on one spelling.
        """
        stretch_ke, stretch_kd = float(self.stretch_ke), float(self.stretch_kd)
        bend_ke, bend_kd = float(self.bend_ke), float(self.bend_kd)
        return {
            "stretch_ke": stretch_ke,
            "stretch_kd": stretch_kd,
            "shear_ke": stretch_ke if self.shear_ke is None else float(self.shear_ke),
            "shear_kd": stretch_kd if self.shear_kd is None else float(self.shear_kd),
            "bend_ke": bend_ke,
            "bend_kd": bend_kd,
            "twist_ke": bend_ke if self.twist_ke is None else float(self.twist_ke),
            "twist_kd": bend_kd if self.twist_kd is None else float(self.twist_kd),
        }

    def rest_polyline(self) -> np.ndarray:
        """The rest nodes in the work-surface frame: :attr:`nodes`, or the straight line.

        ``(segments + 1, 3)`` for an open rod; a closed one's ``segments`` is its node count, so it gives
        ``(segments, 3)``, the seam left for the reader to repeat.
        """
        if self.nodes is not None:
            return np.asarray(self.nodes, dtype=np.float64).reshape(-1, 3)
        start = np.asarray(self.start, dtype=np.float64)
        axis = np.asarray(self.direction, dtype=np.float64)
        axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
        return start + np.outer(np.linspace(0.0, float(self.length), int(self.segments) + 1), axis)


# ---------------------------------------------------------------------------
# Rod material
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class RodMaterialSpec:
    """A rod's constitutive stiffness, as the built model holds it.

    A view of ``joint_target_ke``/``kd`` over the cable joints' four dof rows --
    stretch and shear linear, bend and twist angular -- one pair of fields per
    slot. A :class:`RodSpec` that leaves ``shear_*``/``twist_*`` at ``None``
    builds shear equal to stretch and twist equal to bend, and this view then
    shows the pairs equal; sliding one row moves that row alone.

    RECAPTURE rather than LIVE: the assignment is live, but the rod solves inside
    the coupled solver's *view* of the model and the write reaches it through
    ``notify_model_changed`` -- host work a captured graph does not replay.

    The two plastic fields are a view of ``model.vbd.dahl_eps_max`` /
    ``dahl_tau`` over the same joints, and those VBD reads by pointer every
    substep, so they are LIVE -- **where the rod was built with them positive**.
    Built at 0 the solver never runs its Dahl kernels, and the slider moves a
    number nothing reads (see :attr:`RodSpec.dahl_eps_max`).

    Attributes:
        bend_ke: Per-joint bend stiffness [N*m/rad]. The number that answers
            "why is this cable so stiff".
        bend_kd: Bend damping [N*m*s/rad].
        stretch_ke: Per-joint stretch stiffness [N/m].
        stretch_kd: Stretch damping [N*s/m].
        shear_ke: Per-joint transverse shear stiffness [N/m].
        shear_kd: Shear damping [N*s/m].
        twist_ke: Per-joint twist stiffness [N*m/rad]: whether a roll at one
            end reaches the other.
        twist_kd: Twist damping [N*m*s/rad].
        dahl_eps_max: The bend the cable keeps, as a curvature [rad];
            :attr:`RodSpec.dahl_eps_max`.
        dahl_tau: The curvature change the friction builds up over [rad];
            :attr:`RodSpec.dahl_tau`.
    """

    bend_ke: float = field(
        default=0.04, metadata=ui(tier=RECAPTURE, unit="N*m", range=(1e-4, 10.0), log=True, group="bend")
    )
    bend_kd: float = field(
        default=0.06, metadata=ui(tier=RECAPTURE, unit="N*m*s", range=(1e-4, 10.0), log=True, group="bend")
    )
    stretch_ke: float = field(
        default=2.0e5, metadata=ui(tier=RECAPTURE, unit="N/m", range=(1e3, 1e7), log=True, group="stretch")
    )
    stretch_kd: float = field(
        default=1.0, metadata=ui(tier=RECAPTURE, unit="N*s/m", range=(1e-2, 1e3), log=True, group="stretch")
    )
    shear_ke: float = field(
        default=2.0e5, metadata=ui(tier=RECAPTURE, unit="N/m", range=(1e3, 1e7), log=True, group="shear")
    )
    shear_kd: float = field(
        default=1.0, metadata=ui(tier=RECAPTURE, unit="N*s/m", range=(1e-2, 1e3), log=True, group="shear")
    )
    twist_ke: float = field(
        default=0.04, metadata=ui(tier=RECAPTURE, unit="N*m", range=(1e-4, 10.0), log=True, group="twist")
    )
    twist_kd: float = field(
        default=0.06, metadata=ui(tier=RECAPTURE, unit="N*m*s", range=(1e-4, 10.0), log=True, group="twist")
    )
    dahl_eps_max: float = field(
        default=0.0,
        metadata=ui(
            tier=LIVE,
            unit="rad",
            log=True,
            range=(1e-3, 5.0),
            group="plastic",
            doc="Bend the cable keeps: the Dahl friction envelope as a curvature, times bend_ke for the moment. "
            "Inert when built at 0 -- the solver arms its Dahl kernels at construction.",
        ),
    )
    dahl_tau: float = field(
        default=0.1,
        metadata=ui(
            tier=LIVE,
            unit="rad",
            log=True,
            range=(1e-3, 5.0),
            group="plastic",
            doc="Curvature change over which the friction builds up: small keeps every bend, large lets most spring back.",
        ),
    )

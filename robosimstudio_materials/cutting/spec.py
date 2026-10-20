# SPDX-License-Identifier: Apache-2.0
"""What a cuttable block, its material and the knife are, as data.

Three frozen specs and the materials. :class:`CuttableMaterialSpec` is a block's matter: the
cage solver's St. Venant-Kirchhoff constants, a density and a toughness. :class:`KnifeSpec` is a
rigid body, a handle and a blade on one body, whose blade edge is the ``blade`` feature. And
:class:`CuttableSpec` is one block standing on the work surface, its base held by the board,
naming the knife that cuts it. Every number on the presets is sourced in its comment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import ClassVar

from robosimstudio.assetlib.semantics.features import FeatureSpec
from robosimstudio.specs import ObjectSpec
from robosimstudio.specs.asset import ShapeSpec
from robosimstudio.specs.base import Quat, Vec3
from robosimstudio.specs.params import REBUILD, ui

__all__ = [
    "CAKE",
    "FOAM",
    "TOFU",
    "CuttableMaterialSpec",
    "CuttableSpec",
    "KnifeSpec",
    "cuttable",
    "cuttables",
    "knife",
]


@dataclass(frozen=True, kw_only=True)
class CuttableMaterialSpec:
    """A cuttable block's matter.

    Attributes:
        young_modulus: The cage solver's E [Pa]: St. Venant-Kirchhoff with the rest strain the
            solver keeps so its rest pose is force free.
        poisson_ratio: The cage solver's nu.
        density: [kg/m^3]. Sets the block's weight on the board and its inertia.
        toughness: The push per metre of edge a sharp blade needs before the block parts [N/m],
            the material's cutting toughness (a wire-cutting ``G_c`` is this number in J/m^2).
            The knife's :attr:`KnifeSpec.sharpness` divides it: the blade's threshold is
            ``toughness / sharpness`` per metre of edge in the block. REBUILD: the block offers no
            tunable, so a new toughness comes with a new spec; the knife's sharpness is the live
            knob (``EntityCuttable.set_sharpness``).
        color: Display RGB, the block's skin and its cut faces alike.
    """

    young_modulus: float = field(
        default=23.5e3, metadata=ui(tier=REBUILD, unit="Pa", log=True, range=(1.0e3, 1.0e6), group="material")
    )
    poisson_ratio: float = field(default=0.45, metadata=ui(tier=REBUILD, range=(0.0, 0.49), group="material"))
    density: float = field(
        default=1050.0, metadata=ui(tier=REBUILD, unit="kg/m^3", range=(10.0, 2000.0), group="material")
    )
    toughness: float = field(
        default=3.0,
        metadata=ui(
            tier=REBUILD,
            unit="N/m",
            log=True,
            range=(0.1, 1.0e4),
            group="material",
            doc="Push per metre of edge a sharp blade needs before the block parts.",
        ),
    )
    color: Vec3 = (0.93, 0.91, 0.82)

    def __post_init__(self) -> None:
        if self.young_modulus <= 0.0 or self.density <= 0.0:
            raise ValueError("a cuttable material needs a positive young_modulus and density")
        if not 0.0 <= self.poisson_ratio < 0.5:
            raise ValueError(f"poisson_ratio must be in [0, 0.5), got {self.poisson_ratio}")
        if self.toughness <= 0.0:
            raise ValueError(f"toughness must be positive, got {self.toughness}")


TOFU = CuttableMaterialSpec(
    # Firm tofu's shear modulus is 8.1 kPa (dynamic mechanical analysis at 14 Hz; Pattison et al.,
    # IEEE Trans. Med. Imaging 33(6), 2014, Table I); E = 2 G (1 + nu) = 23.5 kPa at nu 0.45.
    young_modulus=23.5e3,
    # A water-swollen protein gel, nearly incompressible; 0.45 keeps the cage solve well conditioned.
    poisson_ratio=0.45,
    # Mostly water: 1050 kg/m^3, a few per cent over water for the protein.
    density=1050.0,
    # Soft food gels part at a few J/m^2 under a wire: a rate-corrected toughness G0 of 1.96 and 2.87
    # for 15% and 20% modified sago starch gels (1.9-3.1 across its cutting speeds; Gamonpilas,
    # Charalambides, Williams, Dooling and Gibbon, "On the study of wire cutting of starch gels", ICF12,
    # Ottawa 2009), and, quoted there, G0 3.7 for Gruyere and 6.0 for mild Cheddar (Kamyab, Chakrabarti
    # and Williams, "Cutting cheese with wire", J. Mater. Sci. 33, 1998, doi:10.1023/A:1017517332112).
    toughness=3.0,
    color=(0.93, 0.91, 0.82),
)
"""Firm tofu: E 23.5 kPa, nu 0.45, 1050 kg/m^3, 3 N/m."""

CAKE = CuttableMaterialSpec(
    # Sponge cake reads an apparent E of about 5 kPa in compression (Assad-Bustillos et al.,
    # "Chewing mechanisms investigated using finite element modeling for two soft cereal foods",
    # COMSOL Conference).
    young_modulus=5.0e3,
    # An open-cell foam: nu about 1/3 (Gibson and Ashby, Cellular Solids).
    poisson_ratio=0.3,
    # Sponge cake's bulk density is 0.25-0.35 g/cm^3 (food cellular solids tables).
    density=300.0,
    # Crumb parts like the softer food gels above: 2 N/m.
    toughness=2.0,
    color=(0.95, 0.80, 0.48),
)
"""Sponge cake: E 5 kPa, nu 0.3, 300 kg/m^3, 2 N/m."""

FOAM = CuttableMaterialSpec(
    # A 30-45 kg/m^3 flexible polyurethane foam takes 80-200 N over 314 cm^2 at 25% compression
    # (JIS K6400): 2.5-6.4 kPa at a strain of 0.25, a secant E of 10-25 kPa; 15 kPa here.
    young_modulus=15.0e3,
    # Open-cell foam: nu about 1/3 (Gibson and Ashby).
    poisson_ratio=0.3,
    density=30.0,
    # Low-density flexible foam tears at 160-376 N/m (ASTM D3574 / NBR 8515 figures in a foam
    # patent, US 9228047); 250 N/m here, which a knife has to push before the foam parts.
    toughness=250.0,
    color=(0.96, 0.86, 0.36),
)
"""Flexible polyurethane foam: E 15 kPa, nu 0.3, 30 kg/m^3, 250 N/m."""

cuttables = SimpleNamespace(TOFU=TOFU, CAKE=CAKE, FOAM=FOAM)
"""The materials above, which is what ``rss.cuttables`` resolves to."""


@dataclass(frozen=True, kw_only=True)
class KnifeSpec(ObjectSpec):
    """A kitchen knife: a handle and a thin blade on one rigid body.

    The knife's own frame has ``x`` along it (handle at ``-x``, blade toward ``+x``), ``z`` from the
    edge to the spine, and ``y`` across the blade, which is the way a hand's jaws close on the
    handle. The origin is the handle's middle, which is where :attr:`pose` puts it. The blade
    starts at the handle's end, its spine flush with the handle's top, so the edge runs
    ``blade_length`` along ``x`` at ``z = handle_size[2] / 2 - blade_height``: that segment is the
    ``blade`` feature (:meth:`blade_feature`). At the defaults the handle weighs 25 g and the blade
    71 g, so the knife's centre of mass is over the blade and it stands on its edge in a rest of two
    walls; a 50 mm blade stands 10 mm proud of a 40 mm block with its edge on the board.

    Attributes:
        handle_size: The handle box [m]: along the knife, across the jaws, tall.
        blade_length: The edge's length [m].
        blade_height: Spine to edge [m].
        blade_thickness: [m]. The blade box's real thickness; the edge's sharpness is not geometry.
        sharpness: Divides the block's toughness into the blade's threshold: 1 is a sharp kitchen
            knife, 0.001 a blade that has to push a thousand times harder (a butter knife's back).
        handle_density: [kg/m^3]. Wood or plastic, light, so the knife's mass sits over the blade.
        blade_density: [kg/m^3]. Steel.
        handle_color: Display RGB of the handle; :attr:`color` is the blade's.
    """

    entity: str = "knife"
    shape: ShapeSpec | None = None
    handle_size: Vec3 = (0.09, 0.02, 0.02)
    blade_length: float = 0.12
    blade_height: float = 0.05
    blade_thickness: float = 0.0015
    sharpness: float = 1.0
    handle_density: float = 700.0
    blade_density: float = 7850.0
    friction: float | None = 1.0
    color: Vec3 | None = (0.78, 0.80, 0.83)
    handle_color: Vec3 = (0.20, 0.13, 0.08)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.blade_length <= 0.0 or self.blade_height <= 0.0 or self.blade_thickness <= 0.0:
            raise ValueError(f"knife {self.name!r}: the blade needs a positive length, height and thickness")
        if not 0.0 < self.sharpness <= 1.0e3:
            raise ValueError(f"knife {self.name!r}: sharpness must be in (0, 1000], got {self.sharpness}")
        if self.shape is None:
            half = tuple(0.5 * float(v) for v in self.handle_size)
            object.__setattr__(self, "shape", ShapeSpec(kind="box", half_extents=half))

    @property
    def edge(self) -> tuple[Vec3, Vec3, float]:
        """The edge as ``(origin, direction, length)`` in the knife's frame [m]."""
        hx, _, hz = (0.5 * float(v) for v in self.handle_size)
        return (hx, 0.0, hz - float(self.blade_height)), (1.0, 0.0, 0.0), float(self.blade_length)

    @property
    def edgeward(self) -> Vec3:
        """The way the blade cuts, spine to edge, in the knife's frame: ``-z``."""
        return (0.0, 0.0, -1.0)

    def blade_feature(self, name: str = "edge") -> FeatureSpec:
        """The edge as a ``blade`` feature, for the knife's asset record."""
        origin, direction, length = self.edge
        return FeatureSpec(
            name=name,
            kind="blade",
            payload={"origin": origin, "direction": direction, "length": length},
            provenance="KnifeSpec.edge: the blade box's lower long edge, in the knife's frame.",
        )


def knife(*, name: str = "knife", at: Vec3 = (0.0, 0.0, 0.0), **kwargs) -> KnifeSpec:
    """A kitchen knife, its handle's middle at ``at`` in the cell's work-surface frame.

    ``quat`` turns it; every other :class:`KnifeSpec` field may be passed. Standing on its edge, the
    handle's middle is ``blade_height - handle_size[2] / 2`` over what the edge stands on.
    """
    return KnifeSpec(name=name, pose=at, **kwargs)


@dataclass(frozen=True, kw_only=True)
class CuttableSpec:
    """A block on the work surface that a knife can cut, its base held by the board under it.

    The block is a box, ``size`` along its own ``x``, ``y`` and ``z``; :attr:`at` is the middle of
    its base in the cell's work-surface frame and :attr:`quat` turns it about that point. Its base
    face is held where it stands: the board is a fixture.

    Attributes:
        knife: The knife that cuts it, the same spec added to the scene before the block.
        material: One of :data:`cuttables` or a :class:`CuttableMaterialSpec`.
        color: Display RGB; ``None`` draws the block in its material's colour.
        walks: Walk-on-spheres walks per point, the cutter's Monte Carlo budget: 1,024, the paper's.
        quad_points: Interior quadrature points the elastic solve integrates over: 800 asked, 785
            sampled (seeded, so every build) in ``03_materials/10_cutting``'s 60 x 100 x 40 mm block.
        mesh_cell: The drawn skin's triangle size [m]: 994 skin points at 5 mm on that block.
        push: The edge law's stiffness as a multiple of ``E`` per metre of edge in the block (see
            ``law.py``).
        lateral: How far from the cut's plane the edge may be and still count as in the cut [m].
        floor: How close the edge must come to the block's base for the last of it to part [m]:
            the knife stops on the board a contact gap above it. Driven to the board through the arm
            panel, the knife's heel ended 0.2 mm into the board's contact gap (``measure.py``).
        damping: The cage's Rayleigh mass damping [1/s]; ``0`` leaves the damping to backward Euler.
    """

    stage: ClassVar[str] = "deformables"
    """Built after the objects, so the knife it names already has a body."""
    name: str = "block"
    entity: str = "cuttable"
    size: Vec3 = (0.06, 0.10, 0.04)
    at: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (0.0, 0.0, 0.0, 1.0)
    knife: KnifeSpec | None = None
    material: CuttableMaterialSpec = field(default_factory=lambda: TOFU)
    color: Vec3 | None = None
    walks: int = 1024
    quad_points: int = 800
    mesh_cell: float = 0.005
    push: float = 1.0
    lateral: float = 0.005
    floor: float = 0.0025
    damping: float = 0.0

    def __post_init__(self) -> None:
        if min(self.size) <= 0.0:
            raise ValueError(f"cuttable {self.name!r}: size must be positive, got {self.size}")
        if not isinstance(self.material, CuttableMaterialSpec):
            raise ValueError(f"cuttable {self.name!r} needs a CuttableMaterialSpec, got {type(self.material).__name__}")
        if self.knife is not None and not isinstance(self.knife, KnifeSpec):
            raise ValueError(f"cuttable {self.name!r}: knife must be a KnifeSpec, got {type(self.knife).__name__}")
        if self.walks < 16 or self.quad_points < 8:
            raise ValueError(f"cuttable {self.name!r}: walks >= 16 and quad_points >= 8")


def cuttable(
    *,
    name: str = "block",
    knife: KnifeSpec | None = None,
    size: Vec3 = (0.06, 0.10, 0.04),
    at: Vec3 = (0.0, 0.0, 0.0),
    material: CuttableMaterialSpec | None = None,
    **kwargs,
) -> CuttableSpec:
    """A block of ``material`` the ``knife`` can cut, its base's middle at ``at`` on the work surface.

    Args:
        name: Key in handles and the page.
        knife: The :func:`knife` spec that cuts it, added to the scene too, before the block.
        size: Metres along the block's own ``x``, ``y`` and ``z``; ``z`` is its height.
        at: The middle of its base, in the cell's work-surface frame.
        material: One of :data:`cuttables`; ``None`` is :data:`TOFU`.
        **kwargs: Any other :class:`CuttableSpec` field.
    """
    extra = {} if material is None else {"material": material}
    return CuttableSpec(name=name, knife=knife, size=size, at=at, **extra, **kwargs)

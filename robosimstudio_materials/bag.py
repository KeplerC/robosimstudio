# SPDX-License-Identifier: Apache-2.0
"""Bags in one file: two layers of cloth sewn into a pocket, from the spec to the registration.

This file is what ``rss.bag`` builds, imported as ``robosimstudio_materials.bag``. A bag is cloth
(``cloth.py``) sewn on three edges, simulated by SuperDex as one shell that collides with itself.
Run ``examples/03_materials/06_bag.py`` first.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, ClassVar

import newton
import numpy as np
import warp as wp

import robosimstudio.engines.superdex  # noqa: F401 -- registers the shell builder's keys (section 7), not the wheel
from robosimstudio.assetlib.record import AssetRecord, SpecFactory
from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomCalibration, AtomSpec, register_atom
from robosimstudio.assetlib.semantics.handlers import handler
from robosimstudio.assetlib.sim import contacts_with
from robosimstudio.domains.capabilities import host_bridge
from robosimstudio.engines.geometry import actor_consumes, actor_geometry, register_actor_builder, register_actor_kind
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.arrangement import GENERIC, backend
from robosimstudio.specs.base import Vec3
from robosimstudio.specs.params import REBUILD, ui
from robosimstudio.specs.physics import HostActor, SimRequirements, register_domain
from robosimstudio_materials.cloth import ClothMaterialSpec, ClothSpec, EntityCloth

__all__ = [
    "BAG_60HZ",
    "CANVAS",
    "Bag",
    "BagSpec",
    "EntityBag",
    "EntityBagSuperdex",
    "bag",
    "bag_mesh",
    "bags",
    "layer_gap",
    "mouth_opening",
]


# =============================================================================
# 1. What this file is, and how a bag touches itself
# =============================================================================
# Importing this file registers the bag. The library finds it by scanning ``robosimstudio_materials``,
# like cloth. Like paper, a bag imports cloth: its spec, material and entity are cloth's, built as one
# surface whose two layers share the nodes of the three sewn edges.
#
# Self-contact. SuperDex samples a shell's surface as a point cloud of spheres of radius
# ``particle_radius`` and, with ``point_cloud_collider.self_contact``, pushes apart two samples of the
# same shell that come closer than that radius, except pairs already that close at rest (closer than
# 1.5 radii: neighbours on one layer). So the two layers rest one radius apart: 4.02 mm at a 4 mm
# radius, and with self-contact off the top layer falls onto the bottom one, the two coinciding to
# 0.03 mm (``p4_rawbag.py``, a 0.24 x 0.20 m bag of 10 mm cells on a box). The radius must cover the
# samples: on a 10 mm cell a 1 mm radius let a folded flap fall straight through the sheet under it,
# 2 mm held it 1.5 mm up and 4 mm held it 4.4 mm up (``p1_raw.py``). Hence the layers rest a radius
# apart at build too (``bag_mesh``'s ``gap`` is two radii, above the 1.5-radius exclusion), or
# they would never be told apart.
#
# The frame. The engine is float32 and solves in the coordinates it is given. A bag lying on a
# 0.728 m bench, built in world coordinates as the bridge's shell builder then built a sheet,
# never converged: four Newton iterations every engine step, 24 ms each, against 1.6 ms for the same
# bag built about its own centroid with the offset in its root transform (``p4_rawbag.py`` at the
# origin, at (0.6, 0, 0.728) and in a local frame; 52 ms per control step on the YAM scene against
# 8.9 ms, ``p8_prof.py``). So that builder now builds every sheet in its own frame, and a bag is
# its own actor kind only for what it is built with where its record says nothing (section 6).


# =============================================================================
# 2. The surface: two layers and a lip
# =============================================================================


def bag_mesh(
    length: float,
    width: float,
    cells: tuple[int, int],
    lip_cells: int,
    gap: float,
    lip_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The bag as one surface: ``(vertices, triangles, top, bottom)``.

    Local frame: ``x`` runs from the closed end (``x = 0``) to the mouth (``x = length``), ``y``
    across, ``z`` up. ``top`` and ``bottom`` are index grids ``[rows, cols]`` into the vertices; the
    two share the nodes of the three sewn edges (``x = 0``, ``y = 0``, ``y = width``), which sit at
    ``z = 0``, and the rest of each layer sits ``gap / 2`` above or below. The top layer runs on for
    ``lip_cells`` rows past the mouth, turned up by ``lip_angle`` [rad]: a tab to take hold of.
    Bottom triangles wind the other way, so every normal points out of the bag.
    """
    nx, ny = (int(c) for c in cells)
    dx, dy = length / nx, width / ny
    top = np.zeros((nx + 1 + int(lip_cells), ny + 1), dtype=np.int64)
    bottom = np.zeros((nx + 1, ny + 1), dtype=np.int64)
    vertices: list[tuple[float, float, float]] = []
    for grid, side in ((top, 1.0), (bottom, -1.0)):
        for i in range(grid.shape[0]):
            for j in range(ny + 1):
                if i <= nx and (i == 0 or j in (0, ny)):
                    if side > 0:  # a sewn node, made once with the top layer and shared
                        grid[i, j] = len(vertices)
                        vertices.append((i * dx, j * dy, 0.0))
                    else:
                        grid[i, j] = top[i, j]
                    continue
                grid[i, j] = len(vertices)
                if i <= nx:
                    vertices.append((i * dx, j * dy, side * gap / 2))
                else:  # the lip, turned up about the mouth
                    run = (i - nx) * dx
                    vertices.append(
                        (length + run * np.cos(lip_angle), j * dy, side * gap / 2 + run * np.sin(lip_angle))
                    )
    triangles = []
    for grid, outward in ((top, True), (bottom, False)):
        for i in range(grid.shape[0] - 1):
            for j in range(ny):
                a, b, c, d = grid[i, j], grid[i + 1, j], grid[i, j + 1], grid[i + 1, j + 1]
                # The closed corner's cell has three sewn nodes, which would make one triangle twice
                # (once per layer, and SuperDex's solve went to NaN on the first step): cut it the other way.
                pair = [(a, b, d), (a, d, c)] if i == 0 and j == 0 else [(a, b, c), (b, d, c)]
                triangles += pair if outward else [t[::-1] for t in pair]
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64), top, bottom


# =============================================================================
# 3. The material and the spec
# =============================================================================


CANVAS = ClothMaterialSpec(
    shell="membrane",
    # A cotton canvas pouch, 1 mm and 300 g/m^2.
    thickness=1.0e-3,
    areal_density=0.3,
    # Stiff enough that a 30 mm lip turned up 75 degrees stands, its edge 37 mm over the bench: at
    # 1e5 Pa one turned up 83 degrees sags to 16 mm, under the pads' reach (``p11_fold.py``). Soft
    # enough that the top layer lies on the bottom one under its own weight: with self-contact off
    # it sinks onto it, 2 mm up at the median, held off near the seams by its own bending, and
    # crossing it at the worst (``p4_rawbag.py``; -0.24 mm on the example's bench).
    shell_young=1.0e6,
    shell_poisson=0.3,
    # The self-contact radius: the layers rest this far apart. See section 1.
    particle_radius=4.0e-3,
    friction=0.5,
)
"""A cotton canvas pouch: 1 mm, 300 g/m^2, a 4 mm contact radius."""

bags = SimpleNamespace(CANVAS=CANVAS)
"""The bag fabrics under one name."""


@dataclass(frozen=True, kw_only=True)
class BagSpec(ClothSpec):
    """A flat bag lying on its bottom layer, its top layer running on past the mouth as a turned-up lip.

    Every field is ``REBUILD``: SuperDex copies a shell's material and contact when it builds it.

    Attributes:
        size: ``(length, width)`` [m]: closed end to mouth, and across.
        cells: Quads along each, per layer.
        pose: The middle of the bag's footprint, in the cell's work-surface frame. ``quat`` turns
            it about there; the mouth faces local ``+x``.
        lip: How far the top layer runs on past the mouth [m].
        lip_angle: How far the lip is turned up from the bench [rad].
        self_contact: Whether the shell collides with itself. Off, the layers pass through each other.
        friction_falloff: SuperDex ``friction_falloff_vel`` [m/s] of the bag. Below it friction grows
            with slip; against a gripper pad the deformable's value is the one that counts.
        mass_damping: SuperDex ``mass_damping_coefficient`` [1/s], a drag on every node.
        stiffness_damping: SuperDex ``stiffness_damping_coefficient`` [s].
        settle_substeps: Newton substeps per settle frame, or 0 for the run's own, as for a soft
            body: SuperDex steps the bag at its backend's rate either way.
    """

    entity: str = "bag"
    name: str = "bag"
    size: tuple[float, float] = (0.24, 0.20)
    cells: tuple[int, int] = (24, 20)
    material: ClothMaterialSpec = field(default_factory=lambda: CANVAS)
    lip: float = field(default=0.03, metadata=ui(tier=REBUILD, unit="m", range=(0.0, 0.06), group="geometry"))
    lip_angle: float = field(default=1.3, metadata=ui(tier=REBUILD, unit="rad", range=(0.0, 1.57), group="geometry"))
    self_contact: bool = field(default=True, metadata=ui(tier=REBUILD, group="contact"))
    friction_falloff: float = field(
        default=1.0e-2, metadata=ui(tier=REBUILD, unit="m/s", log=True, range=(1.0e-4, 1.0e-1), group="contact")
    )
    # 2/s: a 40 mm lip still swung 1.5 s after a 60-frame settle without it, at five times the cost
    # per step while it did; with it, settled after one second (``p8_prof.py``).
    mass_damping: float = field(default=2.0, metadata=ui(tier=REBUILD, unit="1/s", range=(0.0, 20.0), group="damping"))
    stiffness_damping: float = field(default=0.0, metadata=ui(tier=REBUILD, unit="s", group="damping"))
    settle_frames: int = 90
    settle_substeps: int = 0
    color: Vec3 = (0.80, 0.68, 0.48)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.material.is_slab:
            raise ValueError(f"bag {self.name!r}: a bag is a membrane, not a slab")
        if not self.material.particle_radius > 0.0:
            raise ValueError(f"bag {self.name!r}: the contact radius sets the layer gap and must be positive")
        if int(self.cells[0]) < 1 or int(self.cells[1]) < 2:
            # One cell across sews every node up to the mouth: the two layers would be the same
            # triangles twice, which sent SuperDex's solve to NaN on its first step.
            raise ValueError(
                f"bag {self.name!r}: cells needs at least 1 along and 2 across (with 1 across, every node "
                f"up to the mouth is on a sewn side and the two layers are one), got {tuple(self.cells)}"
            )

    @property
    def cell(self) -> float:
        """The cell length along the bag [m]."""
        return float(self.size[0]) / int(self.cells[0])

    @property
    def lip_cells(self) -> int:
        """Rows of the lip, the nearest whole number of cells to :attr:`lip`."""
        return int(round(self.lip / self.cell))


# =============================================================================
# 4. The entity
# =============================================================================


class EntityBag(EntityCloth):
    """One bag: one surface on the model, one self-colliding SuperDex shell in the engine.

    Newton holds the particles, so the probe, the renderer and the read-back work as for any sheet;
    they are built inactive, so no Newton solver integrates them alongside the engine.
    """

    spec: BagSpec
    domain = "bag"
    hosted: ClassVar[bool] = False
    """Whether SuperDex integrates this bag, as :class:`EntityBagSuperdex` says. Only it builds one:
    on a Newton solver the inactive particles would never move, and VBD has no self-contact for
    the two layers. Decided by the class the compiler picks for the arrangement, as for a soft body."""

    def __init__(self, spec: BagSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.top = np.zeros((0, 0), dtype=np.int64)
        """The top layer as indices into this bag's own particles, ``[rows, cols]``, closed end first."""
        self.bottom = np.zeros((0, 0), dtype=np.int64)
        """The bottom layer, the same way; it shares the sewn edges' nodes with :attr:`top`."""
        self.bridge: Any = None
        """The SuperDex bridge, once the scene is ready."""

    def sim_requirements(self) -> SimRequirements:
        declared = self.spec.sim
        if isinstance(declared, SimRequirements):
            return declared
        return SimRequirements(domain="bag", self_contact=bool(self.spec.self_contact))

    def surface(self) -> tuple[np.ndarray, np.ndarray]:
        spec = self.spec
        vertices, faces, self.top, self.bottom = bag_mesh(
            float(spec.size[0]),
            float(spec.size[1]),
            spec.cells,
            spec.lip_cells,
            2.0 * float(spec.material.particle_radius),
            float(spec.lip_angle),
        )
        # About the middle of the footprint, which is what ``pose`` names.
        return vertices - np.array([float(spec.size[0]) / 2, float(spec.size[1]) / 2, 0.0]), faces

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        if not self.hosted:
            # Refused rather than built frozen: no Newton solver would move the inactive layers.
            raise NotImplementedError(
                f"bag {spec.name!r}: a bag is a self-colliding SuperDex shell; route the 'bag' kind to "
                "backend('superdex'), as rss.BAG_60HZ does"
            )
        material = spec.material
        vertices, faces = self.surface()
        placed = self.placed((*spec.pose, *spec.quat))
        start = int(builder.particle_count)
        self.surface_count = len(vertices)
        self.lumping = "mesh"
        builder.add_cloth_mesh(
            pos=wp.vec3(*[float(v) for v in placed[:3]]),
            rot=wp.quat(*[float(v) for v in placed[3:7]]),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(*[float(c) for c in v]) for v in vertices],
            indices=[int(i) for i in faces.reshape(-1)],
            density=material.areal_density,
            tri_ke=material.tri_ke,
            tri_ka=material.tri_ka,
            tri_kd=material.tri_kd,
            edge_ke=material.edge_ke,
            edge_kd=material.edge_kd,
            particle_radius=material.particle_radius,
            label=spec.name,
        )
        flags = builder.particle_flags
        for index in range(start, int(builder.particle_count)):
            flags[index] = int(flags[index]) & ~int(newton.ParticleFlags.ACTIVE)

    def host_actor(self) -> HostActor:
        """The engine's ``bag`` actor: a shell in its own frame (section 6), with its contact stated."""
        spec, material = self.spec, self.spec.material
        params: dict = {
            "particle_radius": float(material.particle_radius),
            # As cloth reads it: a declared ``sim`` is the scene's answer, so it is the engine's too.
            "self_contact": bool(self.sim_requirements().self_contact),
            "thickness": float(material.thickness),
            "areal_density": float(material.areal_density),
            "friction": float(material.friction),
            "friction_falloff": float(spec.friction_falloff),
            "mass_damping": float(spec.mass_damping),
            "stiffness_damping": float(spec.stiffness_damping),
        }
        # Under the shell's names, as cloth states them (``ClothMaterialSpec.actor_names``).
        if material.shell_young is not None:
            params["young"] = float(material.shell_young)
        if material.shell_poisson is not None:
            params["poisson"] = float(material.shell_poisson)
        return HostActor(kind="bag", params=params, lumping=self.lumping)

    def apply_model_materials(self, model: newton.Model) -> None:
        """Nothing: SuperDex reads the bag's contact off :meth:`host_actor`.

        Cloth writes its contact as the model's one ``soft_contact_ke``/``kd``/``mu``, which only a
        sheet a Newton solver integrates reads. A bag that wrote CANVAS there would hand a VBD towel
        in the same scene the bag's values whenever it came after the towel.
        """
        del model

    def tunables(self, pipeline=None):
        """None: the engine copied the material when it built the shell, so nothing here is live."""
        del pipeline
        return ()

    def on_scene_ready(self, pipeline) -> None:
        self.bridge = host_bridge(pipeline, self.name)
        super().on_scene_ready(pipeline)

    def settle(self, pipeline) -> None:
        """Let the top layer come down onto the bottom one. The engine ticks inside each step, as for a soft body."""
        spec = self.spec
        if spec.settle_frames <= 0:
            return
        if spec.settle_substeps > 0:
            with pipeline.substeps_at(int(spec.settle_substeps)):
                for _ in range(int(spec.settle_frames)):
                    pipeline.step()
        else:
            for _ in range(int(spec.settle_frames)):
                pipeline.step()
        # A settle is scene construction, not simulated time.
        pipeline.sim_time = 0.0
        pipeline.step_count = 0


# =============================================================================
# 5. The constructor and the measures
# =============================================================================


def bag(
    *,
    name: str = "bag",
    size: tuple[float, float] = (0.24, 0.20),
    cells: tuple[int, int] = (24, 20),
    at: Vec3 = (0.0, 0.0, 0.0),
    material: Any = None,
    **kwargs,
) -> BagSpec:
    """A flat bag lying on its bottom layer, its mouth facing ``+x`` until ``quat`` turns it.

    Needs SuperDex: :data:`BAG_60HZ` routes ``bag`` to it, and a build without the ``[superdex]``
    extra raises and names it.

    Args:
        name: Key in handles and particle-set names.
        size: ``(length, width)`` [m], closed end to mouth and across.
        cells: Quads along each, per layer: the default is 10 mm cells, 1,044 particles.
        at: The middle of the bag's footprint, in the cell's work-surface frame.
        material: A :class:`~robosimstudio_materials.cloth.ClothMaterialSpec`; ``None`` is :data:`CANVAS`.
        **kwargs: Any other :class:`BagSpec` field: ``quat``, ``lip``, ``lip_angle``, ``self_contact``.
    """
    extra = {} if material is None else {"material": material}
    return BagSpec(name=name, size=size, cells=cells, pose=at, **extra, **kwargs)


class Bag:
    """A bag as a task document's type: ``register_asset(Bag.from_spec(rss.bag(...), name="canvas_bag"))``."""

    @classmethod
    def from_spec(cls, spec: BagSpec, *, name: str, summary: str = "") -> AssetRecord:
        """The bag a constructor returned, as the :class:`AssetRecord` a document's ``(:objects ...)`` names.

        Its ``at`` and ``quat`` are where the bag starts; the document's object name replaces its
        ``name``. The record states the requirements the bag runs, as
        :meth:`EntityBag.sim_requirements` reads them: the spec's declared ``sim`` when it has one,
        else the bag's contacts with the spec's self-contact.
        """
        if not isinstance(spec, BagSpec):
            raise TypeError(f"Bag.from_spec takes a BagSpec, got {type(spec).__name__}")
        declared = spec.sim
        return AssetRecord(
            name=name,
            summary=summary or f"a bag, {spec.size[0]:.2f} x {spec.size[1]:.2f} m",
            tags=("bag",),
            sim=declared
            if isinstance(declared, SimRequirements)
            else SimRequirements(
                domain="bag",
                contacts=contacts_with("surface", "fixtures", "objects", "hands"),
                self_contact=bool(spec.self_contact),
            ),
            build=SpecFactory(spec),
        )


def _grids(points: np.ndarray, entity: EntityBag) -> tuple[np.ndarray, np.ndarray]:
    """The two layers' nodes, closed end to mouth, the lip left off: ``[rows, cols, 3]`` each."""
    q = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rows = int(entity.spec.cells[0]) + 1
    return q[entity.top[:rows]], q[entity.bottom[:rows]]


def layer_gap(points: np.ndarray, entity: EntityBag) -> float:
    """How far the top layer stays inside the bottom one at worst [m]: negative where they crossed.

    Each top node is measured from the bottom node it was built over, along the bottom layer's
    normal into the bag, over every row but the sewn end and every column but the two sewn sides,
    and where the two lie together: within four contact radii of each other. So it is a signed
    distance wherever the bag lies, and a pair that passed through each other reads below zero;
    a mouth held open, its layers a hand's width apart, has no say in it. ``inf`` when no pair
    lies together.

    Args:
        points: This bag's particles ``[n, 3]``, in the model's order.
        entity: The bag, for its layer grids.
    """
    top, bottom = _grids(points, entity)
    normal = np.cross(np.gradient(bottom, axis=0), np.gradient(bottom, axis=1))
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1.0e-12)
    apart = (top - bottom)[1:, 1:-1]
    gap = np.einsum("ijk,ijk->ij", apart, normal[1:, 1:-1])
    together = np.linalg.norm(apart, axis=-1) < 4.0 * float(entity.spec.material.particle_radius)
    return float(gap[together].min()) if together.any() else float("inf")


def mouth_opening(points: np.ndarray, entity: EntityBag) -> dict[str, float]:
    """How wide the mouth stands open: ``height`` [m], the widest top-to-bottom span across it, and
    ``area`` [m^2], the area its rim encloses (the rim's vector area, so a mouth seen at a slant
    still reads its true opening)."""
    top, bottom = _grids(points, entity)
    upper, lower = top[-1], bottom[-1]
    rim = np.concatenate([upper, lower[::-1][1:-1]])
    area = 0.5 * float(np.linalg.norm(np.cross(rim, np.roll(rim, -1, axis=0)).sum(axis=0)))
    return {"height": float(np.linalg.norm(upper - lower, axis=1).max()), "area": area}


register_atom(
    AtomSpec(
        name="mouth_opening",
        summary="How far a bag's mouth stands open: the widest span across it and the area its rim encloses.",
        args=(ArgSpec(name="bag", substrates=("bag",), reads="handle"),),
        directions=("check", "observe"),
        returns=("height", "area"),
        unit="m",
        cost="cheap",
        reads_scene=True,
        provenance="Designed here, on the rim the bag is built with: the top and bottom layers' mouth rows.",
        calibration=AtomCalibration(
            positives=(
                "The example's canvas bag, its lip pinched by the YAM's pads 25 mm over the bench, lifted to "
                "80 mm and drawn 60 mm back, reads 74.9-76.6 mm and 66-67 cm^2 in seven drives; pinched 12-30 mm "
                "over the bench, 66-90 mm (f_bag.py, GPUs 1 and 3).",
            ),
            negatives=(
                "The bag lying shut reads 7.7 mm and 10.2 cm^2: its rows a contact radius apart. A hand that closes "
                "over the lip's top, 45 mm up, lifts nothing and the bag reads the same 7.7 mm (f_bag.py).",
            ),
            gate=0.03,
            bench="examples/03_materials/06_bag driven through its right arm panel, tests/test_open_a_bag.py",
            notes=(
                "The height is the widest span between the top and bottom rows of the mouth, the area the "
                "rim's vector area, so a mouth tilted toward the hand reads its true opening. The gate sits "
                "four times over a shut bag and under half the least a pinched lip gave. It reads the mouth, "
                "not the layers: with self-contact off the same drive reads 75 mm, so layer_gap is the "
                "measure that tells the two apart."
            ),
        ),
    )
)


@handler("mouth_opening", substrates=("bag",), summary="The mouth rim's widest span [m] and enclosed area [m^2].")
def _mouth_opening(snapshot, bag: str) -> dict[str, float]:
    entity = snapshot.entities.get(bag)
    if entity is None or not entity.top.size:
        return {"height": 0.0, "area": 0.0, f"{bag}/mouth_opening": 0.0}
    out = mouth_opening(snapshot.particles[bag].points(), entity)
    return {**out, f"{bag}/mouth_opening": out["height"]}


# =============================================================================
# 6. The engine's bag actor
# =============================================================================


_BAG_SHELL = {
    "thickness": float(CANVAS.thickness),
    "areal_density": float(CANVAS.areal_density),
    "young": float(CANVAS.shell_young),
    "poisson": float(CANVAS.shell_poisson),
    "particle_radius": float(CANVAS.particle_radius),
    "self_contact": True,
    "friction": float(CANVAS.friction),
    "mass_damping": 0.0,
    "stiffness_damping": 0.0,
}
"""What a bag is built with where its record says nothing: :data:`CANVAS`, colliding with itself, undamped."""

_OLD_NAMES = {"shell_young": "young", "shell_poisson": "poisson"}
"""The names a bag record stated its modulus under before it took the shell's, to the shell's: a record written
then, which a repro bundle replays verbatim, still states the bag's own modulus."""


def build_bag_actor(bridge: Any, record: dict, particle_q=None, tet_indices=None, tet_materials=None) -> dict:
    """The bag as a SuperDex shell: the bridge's shell builder, under the bag's own defaults.

    ``bridge.build_shell`` builds every sheet about its own centroid with the offset in the root
    transform (section 1), states contact (friction, falloff) and damping from the params, and
    reads the nodes back through the root, as ``place_actor`` writes them. What is the bag's own is
    what it is built with where its record says nothing: :data:`CANVAS`, self-contact on. Its
    modulus is the shell's ``young``/``poisson``, as a sheet's is.
    """
    del particle_q, tet_indices, tet_materials
    # A record from before the bag stated the shell's names says ``shell_young``/``shell_poisson``
    # (``_OLD_NAMES``): read as the shell's, or a replayed bag is built at the canvas's modulus, not the one
    # it states. A shell name stated beside one wins.
    params = dict(record.get("params", {}))
    for old, new in _OLD_NAMES.items():
        if old in params:
            params.setdefault(new, params.pop(old))
    return bridge.build_shell({**record, "params": params}, defaults=_BAG_SHELL)


def _read_bag(model: Any, base: Any) -> Any:
    """A bag reads off the model as a shell does: its triangles, re-indexed to zero, and its record's ``lumping``.

    Read again as a shell from a record of the name, range and params alone, the bag's ``"mesh"``
    (:meth:`EntityBag.build`) reached no reader: its geometry said ``None``."""
    record = {"name": base.name, "kind": "shell", "particle_start": base.start, "particle_count": base.count}
    return replace(actor_geometry(model, {**record, "params": base.params}), lumping=base.lumping)


# =============================================================================
# 7. Registration
# =============================================================================


_BAG_HOST = backend("superdex", substeps=2, feedback_dv_max=20.0)
"""Two engine steps per control step, GENERIC's for the soft kind; the soft kind takes the same backend, so a
sponge and a bag share the one host island."""

BAG_60HZ = GENERIC.with_backend("soft", _BAG_HOST).with_backend("bag", _BAG_HOST)
"""GENERIC with ``bag`` on SuperDex beside the soft kind: the arm on MuJoCo, 60 Hz control, the particle
materials on GENERIC's implicit MPM and everything else on its VBD. A sponge here takes the bag's
``feedback_dv_max`` of 20, not GENERIC's default, since one host island serves both."""

register_actor_kind("bag", _read_bag, summary="A bag: a shell's triangles.")
register_actor_builder(
    "superdex",
    "bag",
    build_bag_actor,
    # What the shell builder reads, off its own declaration: a key ``build_shell`` learns is the bag's too. And
    # ``_OLD_NAMES``, which ``build_bag_actor`` reads as the shell's: a record stating them is not told they are unread.
    consumes=(*actor_consumes("superdex", "shell"), *_OLD_NAMES),
    summary="SuperDex builds a bag as a self-colliding shell in its own frame.",
)

DOMAIN = register_domain(
    "bag",
    state=("particles",),
    reads=("bag", "cloth"),
    role="deformables",
    contacts=("surface", "fixtures", "objects", "hands", "deformables"),
    summary="Two layers of cloth sewn into a pocket: one SuperDex shell that collides with itself.",
)

register_entity("bag")(EntityBag)


class EntityBagSuperdex(EntityBag):
    """:class:`EntityBag` for SuperDex, which integrates the bag itself: Newton must leave it alone."""

    hosted = True


register_entity("bag", engine="superdex")(EntityBagSuperdex)


# Import-only; running it directly would register the bag twice.

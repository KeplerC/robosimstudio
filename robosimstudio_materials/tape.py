# SPDX-License-Identifier: Apache-2.0 AND MIT
"""Tape and labels in one file: a sheet stuck to a body by bonds that let go, with a pull tab.

This file is what ``rss.tape`` builds, imported as ``robosimstudio_materials.tape``. It is cloth
(``cloth.py``) plus three things: Newton's particle-body attachments, which stick the sheet to a
body named in the scene; kernels that make and break those bonds, by one of two laws; and a rigid
pull tab clamped on one end, which a hand closes on. Under opening adhesion the sheet is built stuck
and a bond lets go once the sheet lifts off it; under contact adhesion, ipc-toolkit's adhesion law,
a bond forms where the sheet is pressed onto the body and holds up to a strength. Run
``examples/03_materials/08_tape.py`` and ``03_materials/09_adhesion.py`` first.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.assetlib.materials import CalibrationSpec, MaterialRecord, register_material, register_material_kind
from robosimstudio.assetlib.record import AssetRecord, SpecFactory
from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomCalibration, AtomSpec, register_atom
from robosimstudio.assetlib.semantics.handlers import handler
from robosimstudio.assetlib.semantics.states import StateSpec
from robosimstudio.assetlib.sim import SimRequirements, contacts_with
from robosimstudio.core.pipeline import ScenePipeline
from robosimstudio.core.shell import grid_surface
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.arrangement import GENERIC, backend
from robosimstudio.specs.base import Vec3
from robosimstudio.specs.params import LIVE, REBUILD, ui
from robosimstudio.specs.physics import register_domain
from robosimstudio_materials.cloth import ClothMaterialSpec, ClothSpec, EntityCloth

__all__ = [
    "CONTACT_FIELDS",
    "DUCT_TAPE",
    "LABEL",
    "MASKING_TAPE",
    "PAPER_LABEL",
    "TAPE_60HZ",
    "EntityTape",
    "Tape",
    "TapeMaterialSpec",
    "TapeSpec",
    "adhesion_curve",
    "bonded_rows",
    "clamped_rows",
    "opening",
    "tape",
    "tape_surface",
    "tapes",
]


# =============================================================================
# 1. What this file is, and why it is built the way it is
# =============================================================================
# Importing this file registers the tape. The library finds it by scanning ``robosimstudio_materials``,
# like cloth; like paper and the bag, it extends ``cloth.py``.
#
# Opening adhesion, the bonds :data:`PAPER_LABEL` is built with. Newton 1.7's
# ``ModelBuilder.add_attachment_particle_body`` ties one particle to a point fixed in a body by a
# spring (stiffness [N/m], no force cap, no break), which ``SolverVBD`` applies to both ends. Every
# particle of the flat part that lies over a face of the body ``on`` names gets one, at the point it
# was built over, the sheet laid where VBD's contact rests it, the box's margin past a particle radius
# off the face; past the face's edge nothing is under a particle, and it gets none.
# Newton has no strength: a bond holds at any load. Strength is the model stated here, the cohesive
# zone of the peel literature in its simplest form: a bond breaks once the sheet has lifted
# off the body by more than :attr:`TapeMaterialSpec.break_opening` along the face's normal, and a
# broken bond stays broken. Shear does not break a bond: tape pushed along a face does not come
# unstuck, and only being pulled off it peels. The kernel (section 4) runs after every substep inside
# the captured graph and writes the attachment's own ``enabled`` flag, which VBD reads on every launch,
# into the array the solver holds (the coupled solver gives the VBD entry its own copy of the
# attachment arrays) and into the model's, which Newton copies over the solver's again whenever it is
# told the constraints changed. A bond's force at its break is ``tack x cell area x break_opening``:
# 0.1 N on a 5 mm cell of :data:`PAPER_LABEL`, 0.9 N across a 40 mm label's row.
#
# Contact adhesion (:attr:`TapeMaterialSpec.adhesion` ``"contact"``, :data:`LABEL` and the presets after
# it). A bond forms where the sheet is pressed onto the body and holds by ipc-toolkit's adhesion law
# (Fang, Li et al., IEEE TVCG 2024; ported in section 4), so a loose label sticks only where a hand presses it.
# ipc-toolkit adds its potentials to an implicit solve. Added instead as forces on ``particle_f``
# before the solve, the ``pre_solve`` phase of ``core/schedule.py``, with the reaction on the body's
# ``body_f``, they cannot work here: SolverVBD folds ``particle_f`` into the step's predicted velocity,
# so a force f moves a 2 mg particle's starting iterate by dt^2 f/m, 0.54 mm a millinewton at 1/960 s,
# and the body-particle contact turns that depth into its penalty on the body. A constant 1 mN pushed
# down on each of a label's 153 particles shoved the 0.5 kg box 49 mm, and the ipc pull
# at 4 kPa held a tab pulled up and back at 21 N without a row lifting, dragging the box 12 mm
# (``p2_pull.py``, GPU 6). So the law rides on the attachments, which VBD solves implicitly. Every
# bondable particle is given an attachment to the body at the build, disabled unless the sheet is
# built stuck. After every substep one kernel makes a bond where its particle presses into the body
# harder than :attr:`~TapeMaterialSpec.press` (VBD's own penalty for the pair, the mean of the sheet's
# ``contact_ke`` and the box's ``ke``, times the depth inside the contact thickness, over the particle's
# share of the sheet), anchored where the particle would rest on that face. A second sets every held bond's spring to the secant of the pull at the lift the
# substep starts at, so the spring pulls with ipc-toolkit's force there, and lets the bond go past
# :attr:`~TapeMaterialSpec.reach`. Along the face the anchor slides once the spring pulls harder than
# ``stiction`` times the bond's strength, a hold that fades as ipc-toolkit's tangential mollifier
# does once the slide passes ``stiction_speed``. A bond let go is made again where the sheet is
# pressed back. The two warp functions match ipctk 1.6.0 to float32 (``tests/test_tape_adhesion_law.py``).
#
# Both ways, within the tape's island. VBD applies an attachment to both its ends, so the box carries
# what the label carries: drawn up by its tab, :data:`DUCT_TAPE` lifted and dragged the 0.5 kg box 80 mm
# (``p4_law.py``), and the hand feels the tab through the proxy coupler. A body on another solver cannot
# be stuck to (Newton drops the attachment; section 5 raises), so contact adhesion does not reach a
# body MuJoCo owns: that would need the reaction carried through ``body_f``, the route the explicit
# forces above took. The press is measured against the body's boxes, so ``on`` must be an ``rss.box``.
#
# The pull tab, and why it is rigid. Peeling needs the hand to hold the sheet's end, and every way of
# holding it by contact was measured. The attachments need VBD to integrate the body too, so they
# need Newton's proxy coupler; Newton raises outright under RSS's one-way coupler, the one under
# which the YAM's pads pinch a towel. Under the proxy coupler the pads close on a cloth tab turned up
# 3 mm thick and stall on it at 3.2-3.8 mm, then slide up off it: the tab rose 0.2 mm or less in
# every variant tried (friction 0.3 and 0.8, a finger effort of 8 and 20 N, staggered and lagged
# proxies; ``p1_pinch.py``, Newton 1.7, GPUs 1 and 6). The same pads lift a 10 g rigid block on VBD
# 54 mm (``p7_rigid_tab.py``). So the sheet's end is turned up into a rigid tab: a block on the same
# solver, held to the end's particles by unbreakable attachments (:attr:`TapeSpec.tab_hold`), that
# does not collide with the sheet, so the end can lie inside it.
#
# Why the sheet is soft. A body tied to a cloth by attachments answers a pull only as fast as VBD's
# five iterations spread it through the cloth. Tied to a free particle, a 0.1 kg body pulled up by
# 1.98 N rises 307 mm in 0.25 s; tied to one particle of a 5 x 5 sheet at ``tri_ke`` 1e4 it sinks
# 83 mm instead, and with ``tri_ke`` 0 it rises again (``p10_one.py``, pure Newton 1.7). A label of
# copy paper (``edge_ke`` 30, ``tri_ke`` 1e4) held the tab still under the hand's pull until the pads
# slipped, and nothing peeled; at ``tri_ke`` 1e3 and ``edge_ke`` 0.01 the same pull peels it
# (``p8_peel.py``). The presets are soft in plane and in bending for that reason, not because tape is.
#
# The solver. RSS's VBD default keeps Newton's legacy rigid path (``rigid_compliant_alm=False``,
# pinned for the rod tests). On it a sheet lying on a free VBD body pushes the body into what it
# stands on: a 0.3 g, 189-particle label sank a 0.75 kg box 16 mm into the ground and slid it 8 mm/s
# (``p6_pure.py``, pure Newton 1.7; 14 mm in RSS on 1.6 and 1.7, ``p4_sheet_on_box.py``). With
# compliant ALM the box rests where it rests alone and slides 0.2 mm in 2 s. :data:`TAPE_60HZ` is
# ``GENERIC`` with compliant ALM on its VBD island, which the box must share with the tape: a bond
# whose body is on another solver is dropped by Newton without a word, so the entity checks the
# solver's copy and raises instead.
#
# The cost. VBD launches its attachment kernel once per particle colour per iteration, whatever the
# count, so bonds cost launches rather than arithmetic: a 216-particle label with 180 bonds on a box
# beside the YAM steps in 38.7 ms a control step and the same label with none in 28.9 ms, graph-captured
# at ``GENERIC``'s 16 substeps; the break kernel is one launch a substep (``p11_cost.py``, GPU 1).
# Contact adhesion is two launches a substep, one thread a bond: ``03_materials/09_adhesion``'s 126 bonds beside
# the YAM step in 35.1-35.5 ms a control step through the page, carried, laid and rubbed
# (``m1_screen.py``, five drives, GPU 6). Either law adds one launch a control step, outside the graph,
# that lays the bonds again where the box's contact margin has moved (``EntityTape._relay``): 35.6 ms
# a control step with it in two drives, 35.2-35.5 without, laid and rubbed (``step_ms.py``, GPU 5).


# =============================================================================
# 2. The spec: what a tape is, as data
# =============================================================================
# Exported as ``rss.TapeMaterialSpec`` and ``rss.TapeSpec``.


@dataclass(frozen=True, kw_only=True)
class TapeMaterialSpec(ClothMaterialSpec):
    """Cloth constitutive parameters plus an adhesive.

    Every :class:`~robosimstudio_materials.cloth.ClothMaterialSpec` field applies. The ones below are
    the adhesive. ``adhesion`` picks the law at the build; the rest are LIVE, written into the
    attachment arrays VBD reads on every launch or into arrays the bond kernels read.

    Attributes:
        tack: Opening adhesion: the bonds' stiffness per unit area of sheet [N/m per m^2]. Each bonded
            particle gets ``tack`` times its cell's area: 100 N/m on a 5 mm cell at 4e6.
        break_opening: Opening adhesion: how far the sheet lifts off the face under a bond before the
            bond lets go [m]. ``0`` never lets go: the sheet stays stuck whatever pulls on it.
        adhesion: ``"opening"``, bonds made at the build that let go past ``break_opening``; or
            ``"contact"``, bonds made where the sheet is pressed, held by ipc-toolkit's law (section 1).
        strength: Contact adhesion: the largest pull a bond makes per unit area [Pa], ipc-toolkit's
            ``Y * eps_c``, reached at a lift of ``peak_opening``. ``0`` is adhesion off: no bond forms
            and every bond lets go.
        peak_opening: Contact adhesion: the lift past contact at which a bond pulls hardest [m],
            ipc-toolkit's ``dhat_p``.
        reach: Contact adhesion: the lift past contact at which a bond lets go [m], ``dhat_a``.
        stiction: Contact adhesion: a bond's hold along the face, as a multiple of its strength.
        stiction_speed: Contact adhesion: sliding along the face faster than this [m/s], a bond's hold
            fades, gone at twice it (ipc-toolkit's ``eps_a``, as a speed).
        press: Contact adhesion: the contact pressure that makes a bond [Pa], VBD's penalty over the
            particle's share of the sheet: the depth times ``(contact_ke + ke) / 2``, the sheet's
            soft-contact stiffness and the box shape's, the stiffness ``SolverVBD`` gives the pair.
    """

    tack: float = field(
        default=4.0e6,
        metadata=ui(
            tier=LIVE,
            unit="N/m^3",
            log=True,
            range=(1.0e5, 1.0e8),
            group="adhesive",
            doc="Bond stiffness per square metre of sheet. A bond's strength is this x its cell's area x the opening.",
        ),
    )
    break_opening: float = field(
        default=1.0e-3,
        metadata=ui(
            tier=LIVE,
            unit="m",
            range=(0.0, 5.0e-3),
            group="adhesive",
            doc="Lift a bond's particle this far off the face and the bond lets go. 0 never lets go.",
        ),
    )

    adhesion: str = field(
        default="opening",
        metadata=ui(
            tier=REBUILD,
            choices=("opening", "contact"),
            group="adhesive",
            doc="opening: bonds made at build, let go past break_opening. contact: bonds form where pressed.",
        ),
    )
    strength: float = field(
        default=0.0,
        metadata=ui(
            tier=LIVE,
            unit="Pa",
            range=(0.0, 1.0e5),
            group="adhesive",
            doc="Contact adhesion's largest pull per square metre of sheet. 0 is adhesion off: nothing sticks.",
        ),
    )
    peak_opening: float = field(
        default=5.0e-4,
        metadata=ui(
            tier=LIVE, unit="m", range=(1.0e-5, 5.0e-3), group="adhesive", doc="Lift at which a bond pulls hardest."
        ),
    )
    reach: float = field(
        default=1.0e-3,
        metadata=ui(tier=LIVE, unit="m", range=(2.0e-5, 1.0e-2), group="adhesive", doc="Lift at which a bond lets go."),
    )
    stiction: float = field(
        default=1.0,
        metadata=ui(
            tier=LIVE,
            range=(0.0, 4.0),
            group="adhesive",
            doc="A bond's hold along the face, as a share of its strength.",
        ),
    )
    stiction_speed: float = field(
        default=0.01,
        metadata=ui(
            tier=LIVE,
            unit="m/s",
            log=True,
            range=(1.0e-4, 1.0),
            group="adhesive",
            doc="Sliding faster than this, a bond's hold along the face fades; at twice it, it is gone.",
        ),
    )
    press: float = field(
        default=5.0e3,
        metadata=ui(
            tier=LIVE,
            unit="Pa",
            log=True,
            range=(1.0e2, 1.0e6),
            group="adhesive",
            doc="Contact pressure that makes a bond, under contact adhesion.",
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tack < 0.0:
            raise ValueError(f"tape tack cannot be negative, got {self.tack}")
        if self.break_opening < 0.0:
            raise ValueError(f"tape break_opening cannot be negative, got {self.break_opening}")
        if self.adhesion not in ("opening", "contact"):
            raise ValueError(f"tape adhesion must be 'opening' or 'contact', got {self.adhesion!r}")
        if min(self.strength, self.stiction, self.press) < 0.0:
            raise ValueError("tape strength, stiction and press cannot be negative")
        if not 0.0 < self.peak_opening < self.reach:
            raise ValueError(f"tape needs 0 < peak_opening < reach, got {self.peak_opening} and {self.reach}")
        if self.stiction_speed <= 0.0:
            raise ValueError(f"tape stiction_speed must be positive, got {self.stiction_speed}")


ADHESIVE_FIELDS: tuple[str, ...] = (
    "tack",
    "break_opening",
    "strength",
    "peak_opening",
    "reach",
    "stiction",
    "stiction_speed",
    "press",
)
"""The adhesive fields, which the entity writes itself rather than into the cloth's material arrays."""

CONTACT_FIELDS: tuple[str, ...] = ("strength", "peak_opening", "reach", "stiction", "stiction_speed", "press")
"""Contact adhesion's fields, in the order its parameter array holds them."""


# =============================================================================
# 3. Materials: a paper label, and three that stick where pressed
# =============================================================================
# Section 8 registers it with what it was measured on.


PAPER_LABEL = TapeMaterialSpec(
    shell="membrane",
    # 80 g/m^2 label stock, 0.1 mm; a membrane's contact thickness is its particle radius.
    areal_density=0.08,
    thickness=1.0e-4,
    particle_radius=5.0e-4,
    # Soft in plane and in bending so the tab can pull it at all: section 1.
    tri_ke=1.0e3,
    tri_ka=1.0e3,
    tri_kd=1.0e-2,
    edge_ke=1.0e-2,
    edge_kd=1.0e-4,
    # The built shape is the rest shape: flat on the face, its end turned up into the tab.
    zero_rest_angle=False,
    friction=0.3,
    contact_ke=1.0e4,
    contact_kd=1.0e-2,
    # 0.1 N a bond at a 5 mm cell. At twice this (200 N/m a bond) the hand peeled the label too, then
    # lost the tab as the last row went, 58 mm of slip in two drives; at this, the tab stayed within
    # 6.4 mm of where the pads took it through the peel (``p16_peel.py``, a diagonal pull).
    tack=4.0e6,
    break_opening=1.0e-3,
)
"""A paper label: soft enough for its tab to peel it, 0.9 N across a 40 mm row at the break."""

# Contact adhesion's presets: PAPER_LABEL's sheet, soft for section 1's reason, with three adhesives.
# Each was pressed onto the example's 0.5 kg box with 5 N over the whole label, then its tab drawn
# straight up at 30 mm/s by a 300 N/m spring (``p4_law.py``, GPU 6, the 126 bonds of a 12 mm dry
# margin); the pull is the tab's clamps' summed force, and a label stuck by nothing reads 1.07 N on the
# same rig, the drag of VBD's slow answer through the cloth.
LABEL = replace(
    PAPER_LABEL,
    adhesion="contact",
    strength=4.0e3,
    peak_opening=5.0e-4,
    reach=1.0e-3,
    # A bond's hold along the face, 0.4 N on a 5 mm cell. At 1, with stiction_speed 0.01 m/s, the shut
    # fingertips rubbing at 30 mm/s, 5 mm under the label, dragged it along with them (``p18_film.py``,
    # one drive); at these two, four strokes 3 mm under it made 85-93% of the bonds in twelve drives.
    stiction=4.0,
    stiction_speed=0.1,
    # 0.125 N on a 5 mm cell: a label laid by the hand makes 1-6 of 126 bonds (``m1_screen.py``, five
    # drives), the shut fingertips rubbing it 85-93% in four strokes (``stick_ab.py`` and ``step_ms.py``,
    # twelve drives, GPUs 1 and 5). Measured as 8 kPa when the press was read off the sheet's 10 kN/m
    # alone; the pair's mean with the box's 2.5 kN/m is 6.25 kN/m, so 5 kPa is the same depth, 20 um on
    # a 5 mm cell.
    press=5.0e3,
)
"""A paper label that sticks where it is pressed: it peeled whole at 2.9 N drawn up on the rig, and drawn
up by the YAM's pads at 2.15-2.38 N, all but 0-3 of its 126 bonds (``stick_ab.py``, eight drives)."""

MASKING_TAPE = replace(LABEL, strength=1.0e4)
""":data:`LABEL`'s sheet with 2.5 times the strength: it peeled whole at 4.2 N on the rig. Not driven by
the YAM, whose pads let a tab slide out at 2-3.5 N pulled along them (``p8_peel.py``)."""

DUCT_TAPE = replace(LABEL, strength=4.0e4)
""":data:`LABEL`'s sheet with 10 times the strength: drawn up on the rig it did not peel, and lifted and
dragged the 0.5 kg box 80 mm at 4.6-7.8 N."""

tapes = SimpleNamespace(PAPER_LABEL=PAPER_LABEL, LABEL=LABEL, MASKING_TAPE=MASKING_TAPE, DUCT_TAPE=DUCT_TAPE)
"""The materials above under one name, which is what ``rss.tapes`` resolves to."""


@dataclass(frozen=True, kw_only=True)
class TapeSpec(ClothSpec):
    """A strip of tape or a label stuck to a body, its end turned up into a rigid pull tab.

    Every cloth field applies; ``mesh`` is always a grid. The sheet's own frame has ``x`` across,
    ``y`` along and ``z`` off the face; ``pose`` puts its origin corner, at the tab end, and ``quat``
    turns it. ``rss.tape`` returns one.

    Attributes:
        on: The body the sheet sticks to, by name: a free body built before the tape, on the same
            solver (``rss.box`` under :data:`TAPE_60HZ`). Every particle of the flat part can bond to
            it except the row at the fold; built stuck, the ones over a face of its box are bonded.
            ``""`` sticks it to nothing: a loose strip, its tab still on.
        tab: How much of the sheet's length is turned up, at ``y = 0``, into the pull tab [m]. ``0``
            builds no tab: the sheet is stuck along its whole length and nothing holds its end.
        tab_size: The tab's extents [m]: across the sheet, its thickness along the sheet (what the
            jaws close across) and its height off the face.
        tab_mass: [kg]. 30 g, with :attr:`tab_hold` at 3 kN/m: in 24 drives of the example, its jaws
            damped, the tab stayed within 3.4 mm of where the pads took it through the whole peel
            (``peel_probe.py``, GPUs 1 and 3). A 10 g tab clamped at 10 kN/m slid up to 9.5 mm, and out
            of the pads with 44% of the label off in one drive of seven (``m9_drives.py``, GPUs 1 and 6).
        tab_friction: The tab's friction against what touches it: the pads and the face.
        tab_hold: Stiffness of each attachment that clamps the sheet's turned-up end in the tab [N/m].
            These never break. Softer than the 10 kN/m Newton defaults to, so the tab's own inertia, not
            the clamp, sets how it answers the pads (section 1's "Why the sheet is soft").
        tab_color: Display RGB of the tab.
        dry: How far past the fold the sheet carries no adhesive [m], like a label's lift tab. ``0``
            bonds every row on the face but the fold's. Without a tab it runs from the sheet's end.
        stuck: Built stuck to ``on``, laid the box's contact margin further off the face than
            :attr:`pose` puts it, where VBD's contact rests it: a bond made at the build and again at
            every reset wherever the sheet lies over a face of the body's box, none past its edge,
            where nothing is under it. Opening adhesion gives a vertex past the edge no bond at all,
            contact adhesion one made only once it is pressed onto a face. ``False`` builds the sheet
            loose, its bonds made only where it is pressed onto ``on``, which needs contact adhesion
            (:attr:`TapeMaterialSpec.adhesion`).
        settle_frames: ``0``, unlike cloth's 60: a reset puts the body back at its spawn pose and the
            tab where it was built, so the sheet goes back where it was built too, on that pose.
            Settled for 30 frames first, the box came back to its spawn pose with the settled label
            0.7 mm inside its top, the label sprang off and every bond broke within 30 control steps
            (``d1.py``). Another thing's settle steps the label too, so a reset does not take the
            sheet from that snapshot: beside a towel settled 60 frames the label came back 1.5 to
            1.7 mm inside the box top and every bond broke within 30 control steps, untouched
            (``p1c.py``, GPU 1).
    """

    entity: str = "tape"
    name: str = "tape"
    size: tuple[float, float] = (0.04, 0.095)
    cells: tuple[int, int] = (8, 19)
    material: TapeMaterialSpec = field(default_factory=lambda: PAPER_LABEL)
    color: Vec3 = (0.95, 0.94, 0.88)
    on: str = ""
    tab: float = field(default=0.015, metadata=ui(tier=REBUILD, unit="m", range=(0.0, 0.05), group="tab"))
    tab_size: Vec3 = (0.044, 0.008, 0.024)
    tab_mass: float = 0.03
    tab_friction: float = 0.8
    tab_hold: float = 3.0e3
    tab_color: Vec3 = (0.16, 0.36, 0.78)
    settle_frames: int = 0
    stuck: bool = True
    dry: float = field(default=0.0, metadata=ui(tier=REBUILD, unit="m", range=(0.0, 0.05), group="tab"))

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.material, TapeMaterialSpec):
            raise ValueError(f"tape {self.name!r} needs a TapeMaterialSpec, got {type(self.material).__name__}")
        if self.mesh != "grid":
            raise ValueError(f"tape {self.name!r} is a grid; got mesh={self.mesh!r}")
        length = float(self.size[1])
        if not 0.0 <= self.tab < length:
            raise ValueError(f"tape {self.name!r}: tab {self.tab} m must be in [0, {length}) m, the sheet's length")
        if self.tab > 0.0 and (min(self.tab_size) <= 0.0 or self.tab_mass <= 0.0):
            raise ValueError(f"tape {self.name!r}: a tab needs positive tab_size and tab_mass")
        if self.tab > 0.0 and self.tab_size[2] < self.tab:
            raise ValueError(
                f"tape {self.name!r}: the tab ({self.tab_size[2]} m tall) must hold the {self.tab} m turned up"
            )
        if self.dry < 0.0 or self.tab + self.dry >= length:
            raise ValueError(f"tape {self.name!r}: dry {self.dry} m leaves nothing to stick past the {self.tab} m tab")
        if not self.stuck and self.material.adhesion != "contact":
            raise ValueError(
                f"tape {self.name!r}: only contact adhesion makes bonds after the build, so a loose "
                f"(stuck=False) tape needs a material with adhesion='contact', not {self.material.adhesion!r}"
            )

    @property
    def cell(self) -> tuple[float, float]:
        """The grid's cell, ``(across, along)`` [m]."""
        return float(self.size[0]) / int(self.cells[0]), float(self.size[1]) / int(self.cells[1])


# =============================================================================
# 4. The geometry, the selections and the kernel
# =============================================================================


def tape_surface(
    size: tuple[float, float], cells: tuple[int, int], tab: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The sheet in its own frame, its first ``tab`` metres turned straight up: ``(vertices, faces, up)``.

    ``up`` marks the vertices turned up. The fold is the row at ``y = tab``, which stays on the face.
    """
    vertices, faces = grid_surface(size, cells)
    vertices = vertices.copy()
    up = vertices[:, 1] < float(tab) - 1.0e-9
    vertices[up, 2] = float(tab) - vertices[up, 1]
    vertices[up, 1] = float(tab)
    return vertices, faces, up


def bonded_rows(vertices: np.ndarray, tab: float, cell_along: float, dry: float = 0.0) -> np.ndarray:
    """The vertices stuck to the body: on the face, past the fold row and past ``dry`` metres beyond it.

    Without a tab there is no fold: all of them past ``dry`` metres from the sheet's end at ``y = 0``.
    """
    on_face = vertices[:, 2] <= 1.0e-9
    if tab <= 0.0:
        return np.flatnonzero(on_face & (vertices[:, 1] >= float(dry) - 1.0e-9))
    past = float(tab) + max(0.5 * float(cell_along), float(dry) - 1.0e-9)
    return np.flatnonzero(on_face & (vertices[:, 1] >= past))


def clamped_rows(vertices: np.ndarray, tab: float) -> np.ndarray:
    """The vertices of the turned-up end held in the tab: its upper half, so the end bends at the fold."""
    if tab <= 0.0:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(vertices[:, 2] >= 0.5 * float(tab) - 1.0e-9)


def sheet_areas(size: tuple[float, float], cells: tuple[int, int]) -> np.ndarray:
    """Each vertex's share of the flat sheet's area [m^2]: a third of every triangle it is a corner of."""
    vertices, faces = grid_surface(size, cells)
    corners = vertices[faces]
    tri = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    out = np.zeros(len(vertices))
    for k in range(3):
        np.add.at(out, faces[:, k], tri / 3.0)
    return out


def contact_stiffness(material: TapeMaterialSpec, dmin: float) -> float:
    """Contact adhesion's spring at contact, per square metre of sheet [N/m per m^2]: the slope of the pull.

    The same number :func:`adhesion_stiffness` gives on the device, for a contact thickness ``dmin``.
    """
    peak, reach, strength = float(material.peak_opening), float(material.reach), float(material.strength)
    big_p = 2.0 * dmin * peak + peak * peak
    big_a = 2.0 * dmin * reach + reach * reach
    a2 = strength / (4.0 * (peak + dmin) * (big_p - big_a))
    return 8.0 * a2 * (1.0 - big_a / big_p) * dmin * dmin


def opening(points: np.ndarray, anchors: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """How far each point has lifted off its anchor along the face's normal [m]; negative is into the face."""
    return np.einsum("ij,ij->i", np.asarray(points, float) - np.asarray(anchors, float), np.asarray(normals, float))


@wp.kernel
def _break_kernel(
    rows: wp.array(dtype=wp.int32),
    particle: wp.array(dtype=wp.int32),
    body: wp.array(dtype=wp.int32),
    anchor: wp.array(dtype=wp.vec3),
    normal: wp.vec3,
    limit: wp.array(dtype=float),
    particle_q: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    enabled: wp.array(dtype=wp.bool),
    enabled_model: wp.array(dtype=wp.bool),
    broken: wp.array(dtype=wp.int32),
):
    """Break each bond whose particle has lifted off its anchor past the limit. One thread per bond.

    Launched every substep, unconditionally, so the block stays capturable; ``limit[0] <= 0`` is the
    switch for a sheet that never lets go. ``normal`` is the face's, in the body's frame. ``enabled``
    is the solver's copy and ``enabled_model`` the model's, the same array when the solver reads the
    model's: a ``CONSTRAINT_PROPERTIES`` notify copies the model's over the solver's, and a bond the
    model still held would stick again.
    """
    i = wp.tid()
    if broken[i] != 0:
        return
    lift = limit[0]
    if lift <= 0.0:
        return
    pose = body_q[body[i]]
    off = particle_q[particle[i]] - wp.transform_point(pose, anchor[i])
    if wp.dot(off, wp.transform_vector(pose, normal)) > lift:
        broken[i] = 1
        enabled[rows[i]] = False
        enabled_model[rows[i]] = False


# Contact adhesion. The force law is ipc-toolkit's (https://github.com/ipc-sim/ipc-toolkit, v1.6.0;
# Fang, Li et al., "Augmented Incremental Potential Contact for Sticky Interactions", IEEE TVCG 30(8),
# 2024, doi:10.1109/TVCG.2023.3295656): ``src/ipc/adhesion/adhesion.cpp`` and the argument shift in
# ``src/ipc/potentials/normal_adhesion_potential.cpp``, translated below into warp functions and, on the
# host, into :func:`contact_stiffness`. That code is ipc-toolkit's under the MIT License, Copyright (c)
# 2020-2025 Zachary Ferguson. ipc-toolkit evaluates its potential on the squared distance less
# ``dmin^2``; the pull here is that potential's derivative along the distance itself, so it peaks at
# ``strength`` (``Y * eps_c``) per unit area when the gap reaches ``peak``, and is zero at contact and
# past ``reach``. Section 1 says how it is applied.


@wp.func
def adhesion_pull(dist: float, dmin: float, peak: float, reach: float, strength: float) -> float:
    """ipc-toolkit's normal adhesion force per unit area at centre-to-surface distance ``dist`` [Pa].

    ``dmin`` is the contact thickness, ``peak`` and ``reach`` the gaps past it where the pull is largest
    and where it ends (``dhat_p`` and ``dhat_a``). Positive pulls the point onto the surface.
    """
    big_d = dist * dist - dmin * dmin
    big_p = 2.0 * dmin * peak + peak * peak
    big_a = 2.0 * dmin * reach + reach * reach
    if big_d <= 0.0 or big_d >= big_a:
        return 0.0
    a2 = strength / (4.0 * (peak + dmin) * (big_p - big_a))
    slope = float(0.0)
    if big_d < big_p:
        slope = 2.0 * a2 * (1.0 - big_a / big_p) * big_d
    else:
        slope = 2.0 * a2 * (big_d - big_a)
    return 2.0 * dist * slope


@wp.func
def adhesion_stiffness(dmin: float, peak: float, reach: float, strength: float) -> float:
    """The pull's slope at contact, per unit area [Pa/m]: what a bond starts as, and holds at below contact."""
    big_p = 2.0 * dmin * peak + peak * peak
    big_a = 2.0 * dmin * reach + reach * reach
    a2 = strength / (4.0 * (peak + dmin) * (big_p - big_a))
    return 8.0 * a2 * (1.0 - big_a / big_p) * dmin * dmin


@wp.func
def stiction_mollifier(y: float, eps: float) -> float:
    """ipc-toolkit's ``tangential_adhesion_f1``: 1 at a slide of ``eps`` a step, 0 at none and past ``2 eps``."""
    if y <= 0.0 or y >= 2.0 * eps:
        return 0.0
    z = y / eps
    return z * (2.0 - z)


@wp.kernel
def _curve_kernel(
    dist: wp.array(dtype=float),
    params: wp.array(dtype=float),
    pull: wp.array(dtype=float),
    mollified: wp.array(dtype=float),
):
    i = wp.tid()
    pull[i] = adhesion_pull(dist[i], params[0], params[1], params[2], params[3])
    mollified[i] = stiction_mollifier(dist[i], params[4])


def adhesion_curve(
    dist: np.ndarray, *, dmin: float, peak: float, reach: float, strength: float, eps: float = 1.0e-3
) -> tuple[np.ndarray, np.ndarray]:
    """The two warp functions above evaluated on the CPU: ``(pull [Pa], stiction_mollifier)`` at each ``dist``.

    For checking the port against ipc-toolkit itself (``tests/test_tape_adhesion_law.py``); the
    mollifier takes ``dist`` as the slide ``y``.
    """
    d = wp.array(np.asarray(dist, dtype=np.float32), dtype=float, device="cpu")
    params = wp.array(np.array([dmin, peak, reach, strength, eps], dtype=np.float32), dtype=float, device="cpu")
    pull = wp.zeros(len(d), dtype=float, device="cpu")
    mollified = wp.zeros(len(d), dtype=float, device="cpu")
    wp.launch(_curve_kernel, dim=len(d), inputs=[d, params], outputs=[pull, mollified], device="cpu")
    return pull.numpy().astype(np.float64), mollified.numpy().astype(np.float64)


@wp.func
def _box_gap(point: wp.vec3, half: wp.vec3) -> wp.vec4:
    """Signed distance from a box centred at the origin, and the outward normal there: ``(nx, ny, nz, d)``."""
    q = wp.vec3(wp.abs(point[0]) - half[0], wp.abs(point[1]) - half[1], wp.abs(point[2]) - half[2])
    if q[0] > 0.0 or q[1] > 0.0 or q[2] > 0.0:
        outside = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
        clamped = wp.vec3(
            wp.clamp(point[0], -half[0], half[0]),
            wp.clamp(point[1], -half[1], half[1]),
            wp.clamp(point[2], -half[2], half[2]),
        )
        n = wp.normalize(point - clamped)
        return wp.vec4(n[0], n[1], n[2], wp.length(outside))
    if q[0] >= q[1] and q[0] >= q[2]:
        return wp.vec4(wp.sign(point[0]), 0.0, 0.0, q[0])
    if q[1] >= q[2]:
        return wp.vec4(0.0, wp.sign(point[1]), 0.0, q[1])
    return wp.vec4(0.0, 0.0, wp.sign(point[2]), q[2])


@wp.func
def _margin_of(shape_margin: wp.array(dtype=float), s: int) -> float:
    """A shape's contact margin [m], 0 for none or with no margins on the model."""
    if s < 0 or shape_margin.shape[0] == 0:
        return 0.0
    return shape_margin[s]


@wp.func
def _nearest_box(
    local: wp.vec3,
    boxes: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_margin: wp.array(dtype=float),
) -> int:
    """The body's box nearest a point in the body's frame, its margin counted; -1 with no boxes."""
    best = float(1.0e9)
    hit = int(-1)
    for j in range(boxes.shape[0]):
        s = boxes[j]
        g = _box_gap(wp.transform_point(wp.transform_inverse(shape_transform[s]), local), shape_scale[s])
        if g[3] - _margin_of(shape_margin, s) < best:
            best = g[3] - _margin_of(shape_margin, s)
            hit = s
    return hit


_FLUSH = wp.constant(wp.float32(1.0e-5))
"""How far past a box's half extent a point still lies over the face [m], 0.01 mm: a sheet laid flush with an
edge lies on it. Its float32 vertices land up to 8e-9 m past the edge, and tested with no allowance a label
covering the example's box top lost a row of 11 of its 143 bonds, and on a box off the origin a row and a
column, 23; bonded on the edge, all 143 held 60 substeps of CPU VBD (``flush.py``, ``flush_step.py``). Seated
on the edge 0.01 mm past it, 0.5 mm off the face, a bond's normal tilts 1.1 degrees and it starts 0.1 um into
its lift."""


@wp.struct
class _Seat:
    """Where a point would rest on the box under it, in the body's frame: what :func:`_seat_on` finds.

    ``shape`` is the box, -1 with none; ``over`` whether the point lies over the face, within its
    extent and :data:`_FLUSH`, ``False`` with no box, so it alone says a bond can be placed there;
    ``face`` the face's outward normal; ``gap`` the point's distance off the box's surface less the
    box's margin [m], so a particle presses ``radius - gap`` into the contact; ``thick`` the contact
    thickness, the particle's radius plus that margin [m]; and ``anchor`` the point on the surface
    under it, ``thick`` off it along ``face``, where VBD's contact rests the particle.
    """

    shape: int
    over: wp.bool
    face: wp.vec3
    gap: float
    thick: float
    anchor: wp.vec3


@wp.func
def _seat_on(
    local: wp.vec3,
    radius: float,
    boxes: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_margin: wp.array(dtype=float),
) -> _Seat:
    """The seat on the body's box nearest a point in the body's frame: every kernel that places a bond calls this.

    A point lies over a face when at most one of its coordinates in the box's frame is past the box's
    half extent, by more than :data:`_FLUSH`: over that face, or inside the box. Past an edge or a corner
    nothing is under it, and the box's nearest point is that edge, its normal tilted between the faces:
    a vertex 5 mm past the top's edge and 0.5 mm over it gets ``(0.995, 0, 0.1)``. With no box nothing is.
    """
    seat = _Seat()
    seat.shape = _nearest_box(local, boxes, shape_transform, shape_scale, shape_margin)
    margin = _margin_of(shape_margin, seat.shape)
    seat.thick = radius + margin
    if seat.shape < 0:
        seat.over = False
        return seat
    frame = shape_transform[seat.shape]
    half = shape_scale[seat.shape]
    inside = wp.transform_point(wp.transform_inverse(frame), local)
    g = _box_gap(inside, half)
    n_shape = wp.vec3(g[0], g[1], g[2])
    past = int(0)
    for i in range(3):
        if wp.abs(inside[i]) > half[i] + _FLUSH:
            past += 1
    seat.over = past <= 1
    seat.face = wp.transform_vector(frame, n_shape)
    seat.gap = g[3] - margin
    seat.anchor = wp.transform_point(frame, inside - n_shape * g[3]) + seat.face * seat.thick
    return seat


def over_a_face(
    points: np.ndarray, transforms: np.ndarray, halves: np.ndarray, margins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Which points lie over a face of a body's boxes, within its extent, and the box nearest each.

    :func:`_seat_on`'s rule, in numpy: what the build bonds a stuck sheet by. The nearest box is the
    one whose surface less its margin is nearest, and a point lies over it when at most one of its
    coordinates in the box's frame is past the half extent by more than :data:`_FLUSH`. ``points`` are
    in the body's frame [m]; ``transforms`` are the boxes' in it, ``(x, y, z, qx, qy, qz, qw)``,
    ``halves`` their half extents and ``margins`` their contact margins [m]. Returns ``(over, box)``,
    ``box`` an index into the boxes given, -1 for none. In numpy because through the warp function the
    build compiled the tape's kernels for the CPU, inside a GPU process: a stuck :data:`LABEL` on the
    example's box built in 1.77-1.81 s on a cold kernel cache, and builds in 8 ms (``a4_build_time.py``).
    ``tests/test_tape_kind`` checks the two agree.
    """
    from robosimstudio.core.math import quat_to_mat

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    transforms = np.asarray(transforms, dtype=np.float64).reshape(-1, 7)
    halves = np.asarray(halves, dtype=np.float64).reshape(-1, 3)[:, None, :]
    margins = np.asarray(margins, dtype=np.float64).reshape(-1)[:, None]
    if not len(transforms) or not len(points):
        return np.zeros(len(points), dtype=bool), np.full(len(points), -1, dtype=np.int64)
    # Every point in every box's frame, (boxes, points, 3), and :func:`_box_gap`'s distance off each box.
    inside = np.stack([(points - t[:3]) @ quat_to_mat(t[3:7]) for t in transforms])
    q = np.abs(inside) - halves
    gap = np.where((q > 0.0).any(axis=2), np.linalg.norm(np.maximum(q, 0.0), axis=2), q.max(axis=2))
    box = np.argmin(gap - margins, axis=0)
    past = (np.abs(inside) > halves + float(_FLUSH))[box, np.arange(len(points))].sum(axis=1)
    return past <= 1, box.astype(np.int64)


@wp.kernel
def _stick_kernel(
    rows: wp.array(dtype=wp.int32),
    particle: wp.array(dtype=wp.int32),
    target: int,
    boxes: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_margin: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    area: wp.array(dtype=float),
    params: wp.array(dtype=float),
    soft_ke: float,
    shape_ke: wp.array(dtype=float),
    particle_q: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    enabled: wp.array(dtype=wp.bool),
    enabled_model: wp.array(dtype=wp.bool),
    body_point: wp.array(dtype=wp.vec3),
    body_point_model: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    dmin: wp.array(dtype=float),
    broken: wp.array(dtype=wp.int32),
):
    """Make each bond whose particle is pressed onto the body harder than ``press``. One thread per bond.

    The press is what VBD's penalty pushes back with: the depth inside the contact thickness, the
    particle's radius plus the shape's margin, against the nearest of the body's boxes, times the pair's
    stiffness, the mean of the particles' ``soft_ke`` and that shape's ``shape_ke``, as
    ``init_body_particle_contacts`` mixes them. ``SolverVBD`` holds the pair's penalty at that mean when
    its linear beta is 0, :data:`TAPE_60HZ`'s; a ramped penalty starts lower, and this reads its
    ceiling. ``soft_ke`` is a float frozen into a captured graph, as VBD's own launches take it, and
    cloth's ``contact_ke`` is RECAPTURE; ``shape_ke`` is the model's array, read live.

    The anchor goes where the particle would rest on that face, under it (:func:`_seat_on`). A
    particle past the face's edge has nothing under it and makes no bond: seated on the edge, along
    the tilted normal there, a bond would start most of its distance past the edge into its lift.
    ``enabled`` and ``body_point`` are the solver's copies and the ``_model`` ones the model's, written
    alike as :func:`_break_kernel` writes them: a ``CONSTRAINT_PROPERTIES`` notify copies the model's
    over the solver's.
    """
    k = wp.tid()
    row = rows[k]
    if enabled[row] or params[0] <= 0.0:
        return
    p = particle[k]
    pose = body_q[target]
    local = wp.transform_point(wp.transform_inverse(pose), particle_q[p])
    seat = _seat_on(local, particle_radius[p], boxes, shape_transform, shape_scale, shape_margin)
    if not seat.over:
        return
    pair = 0.5 * (soft_ke + shape_ke[seat.shape])
    if pair * (particle_radius[p] - seat.gap) < params[5] * area[k]:
        return
    body_point[row] = seat.anchor
    body_point_model[row] = seat.anchor
    normal[k] = seat.face
    dmin[k] = seat.thick
    enabled[row] = True
    enabled_model[row] = True
    broken[k] = 0


@wp.kernel
def _bond_kernel(
    rows: wp.array(dtype=wp.int32),
    particle: wp.array(dtype=wp.int32),
    target: int,
    area: wp.array(dtype=float),
    normal: wp.array(dtype=wp.vec3),
    dmin: wp.array(dtype=float),
    params: wp.array(dtype=float),
    dt: float,
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    enabled: wp.array(dtype=wp.bool),
    enabled_model: wp.array(dtype=wp.bool),
    body_point: wp.array(dtype=wp.vec3),
    body_point_model: wp.array(dtype=wp.vec3),
    stiffness: wp.array(dtype=float),
    stiffness_model: wp.array(dtype=float),
    broken: wp.array(dtype=wp.int32),
):
    """Each held bond's spring set to ipc-toolkit's law at its lift, and let go past ``reach``. One thread per bond.

    The spring is the secant of the pull at the lift the step starts at, so VBD carries the law
    implicitly. Along the face the anchor slides once the spring pulls harder than ``stiction`` times
    the strength, which fades as ipc-toolkit's tangential mollifier does once the slide passes
    ``stiction_speed``. Every write goes into the solver's copy and the model's (:func:`_stick_kernel`).
    """
    k = wp.tid()
    row = rows[k]
    if not enabled[row]:
        return
    strength = params[0]
    if strength <= 0.0:
        enabled[row] = False
        enabled_model[row] = False
        broken[k] = 1
        return
    peak = params[1]
    reach = params[2]
    pose = body_q[target]
    anchor = wp.transform_point(pose, body_point[row])
    n = wp.transform_vector(pose, normal[k])
    x = particle_q[particle[k]]
    off = x - anchor
    lift = wp.dot(off, n)
    if lift >= reach:
        enabled[row] = False
        enabled_model[row] = False
        broken[k] = 1
        return
    w = area[k]
    d0 = dmin[k]
    k0 = w * adhesion_stiffness(d0, peak, reach, strength)
    spring = k0
    if lift > 1.0e-7:
        spring = w * adhesion_pull(d0 + lift, d0, peak, reach, strength) / lift
    stiffness[row] = spring
    stiffness_model[row] = spring
    side = off - lift * n
    pulled = spring * wp.length(side)
    if pulled <= 0.0:
        return
    com = wp.transform_point(pose, body_com[target])
    twist = body_qd[target]
    v_face = wp.spatial_top(twist) + wp.cross(wp.spatial_bottom(twist), anchor - com)
    slide = particle_qd[particle[k]] - v_face
    slide = slide - wp.dot(slide, n) * n
    y = wp.length(slide) * dt
    eps = params[4] * dt
    hold = params[3] * w * strength
    if y > eps:
        hold = hold * stiction_mollifier(y, eps)
    if pulled > hold:
        moved = wp.transform_point(wp.transform_inverse(pose), anchor + side * (1.0 - hold / pulled))
        body_point[row] = moved
        body_point_model[row] = moved


@wp.kernel
def _seat_kernel(
    rows: wp.array(dtype=wp.int32),
    particle: wp.array(dtype=wp.int32),
    anchor: wp.array(dtype=wp.vec3),
    boxes: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_margin: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    area: wp.array(dtype=float),
    params: wp.array(dtype=float),
    dmin: wp.array(dtype=float),
    normal: wp.array(dtype=wp.vec3),
    stiffness: wp.array(dtype=float),
    stiffness_model: wp.array(dtype=float),
    body_point: wp.array(dtype=wp.vec3),
    body_point_model: wp.array(dtype=wp.vec3),
):
    """Each bond's contact thickness, its spring at contact and its anchor, as :func:`_stick_kernel` would
    make the bond over its built anchor, through the same :func:`_seat_on`: the radius and the margin of
    the box nearest it, as the model holds them now, and the anchor that far off the face, where VBD's
    contact rests the particle. One thread per bond; the spring and the anchor go into the solver's copy
    and the model's."""
    k = wp.tid()
    seat = _seat_on(anchor[k], particle_radius[particle[k]], boxes, shape_transform, shape_scale, shape_margin)
    dmin[k] = seat.thick
    spring = area[k] * adhesion_stiffness(seat.thick, params[1], params[2], params[0])
    stiffness[rows[k]] = spring
    stiffness_model[rows[k]] = spring
    if seat.shape < 0:
        return
    normal[k] = seat.face
    body_point[rows[k]] = seat.anchor
    body_point_model[rows[k]] = seat.anchor


@wp.kernel
def _relay_kernel(
    rows: wp.array(dtype=wp.int32),
    particle: wp.array(dtype=wp.int32),
    target: int,
    contact: int,
    under: wp.array(dtype=wp.int32),
    boxes: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_margin: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    normal: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    enabled: wp.array(dtype=wp.bool),
    laid: wp.array(dtype=float),
    anchor: wp.array(dtype=wp.vec3),
    body_point: wp.array(dtype=wp.vec3),
    body_point_model: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
):
    """Each held bond whose contact thickness has moved since it was laid, laid again where VBD's contact rests
    its particle now: the particle and the anchor moved off the face by the change. One thread per bond.

    ``laid`` is the thickness each was laid at, the particle's radius plus a margin [m]. Under contact
    adhesion (``contact`` 1) it is the bond's ``dmin``, against the margin of the box under its anchor
    (:func:`_seat_on`); under opening adhesion, against the largest margin of the boxes ``under`` the
    sheet, as a reset lays it (:meth:`EntityTape.margin_moved`), and ``anchor`` is the break kernel's
    copy of the anchors. The anchor goes into the solver's copy and the model's (:func:`_stick_kernel`).
    """
    k = wp.tid()
    row = rows[k]
    if not enabled[row]:
        return
    p = particle[k]
    thick = laid[k]
    if contact != 0:
        seat = _seat_on(body_point[row], particle_radius[p], boxes, shape_transform, shape_scale, shape_margin)
        if seat.shape >= 0:
            thick = seat.thick
    elif under.shape[0] > 0:
        margin = _margin_of(shape_margin, under[0])
        for j in range(1, under.shape[0]):
            margin = wp.max(margin, _margin_of(shape_margin, under[j]))
        thick = particle_radius[p] + margin
    moved = thick - laid[k]
    if moved == 0.0:
        return
    n = normal[k]
    point = body_point[row] + n * moved
    body_point[row] = point
    body_point_model[row] = point
    if contact == 0:
        anchor[k] = point
    laid[k] = thick
    particle_q[p] = particle_q[p] + wp.transform_vector(body_q[target], n) * moved


# =============================================================================
# 5. The entity: cloth, plus the bonds, the tab and the break kernel
# =============================================================================


def _transform(values: Any) -> wp.transform:
    v = [float(c) for c in values]
    return wp.transform(wp.vec3(*v[:3]), wp.quat(*v[3:7]))


class EntityTape(EntityCloth):
    """One strip of tape or label: a cloth bonded to a body, its end in a rigid tab.

    Building, settling and reading the sheet are :class:`EntityCloth`'s. This adds the attachments
    (section 1) and the tab at build, the bond kernels after every substep, and the bonds' and the
    tab's place in a reset: every bond back as built, stuck or loose, and the tab where it was built.
    """

    spec: TapeSpec
    domain = "tape"

    def __init__(self, spec: TapeSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.body: int = -1
        """The body the sheet is stuck to."""
        self.tab_body: int = -1
        """The pull tab's body, or -1 without one."""
        self.bonds: np.ndarray = np.zeros(0, dtype=np.int32)
        """Attachment index of each bond, in the model's order."""
        self.bond_particles: np.ndarray = np.zeros(0, dtype=np.int32)
        """Global particle index of each bond."""
        self.made: np.ndarray = np.zeros(0, dtype=bool)
        """Per bond, whether it is made at the build and again at every reset: on a stuck sheet, where its
        particle lies over a face of the body's box; on a loose one, none."""
        self.anchors: np.ndarray = np.zeros((0, 3))
        """Each bond's anchor in the body's frame [m]."""
        self.along: np.ndarray = np.zeros(0)
        """Each bond's distance from the fold, along the sheet [m]."""
        self.normal: np.ndarray = np.array([0.0, 0.0, 1.0])
        """The face's normal in the body's frame: the sheet's ``+z`` as it was built."""
        self.clamps: np.ndarray = np.zeros(0, dtype=np.int32)
        """Attachment index of each particle the tab holds."""
        self.tab_shapes: list[int] = []
        self.coord_start: int = -1
        self.tab_q0: np.ndarray = np.zeros(0)
        self.built: np.ndarray = np.zeros((0, 3))
        """The sheet's particles where it was built, on the body's spawn pose [m, world]: what a reset restores."""
        self.up: np.ndarray = np.array([0.0, 0.0, 1.0])
        """The sheet's ``+z`` in world as it was built: the way off the face."""
        self.lift: float = 0.0
        """How far past its particle radius a stuck sheet was laid off the face [m]: the margin of the box
        under it, where VBD's contact rests it (:meth:`_lay`). 0 for a loose sheet."""
        self.lift_shapes: np.ndarray = np.zeros(0, dtype=np.int64)
        """The shapes whose margin :attr:`lift` is: the body's boxes under the sheet's bonds."""
        self._model: Any = None
        """The model the bonds are bound on: the margins, radii and shapes every bond kernel reads."""
        self._under: Any = None
        self._radius: np.ndarray = np.zeros(0, dtype=np.float32)
        """The sheet's particle radii as the bonds were bound [m], by offset into the sheet (:attr:`_start`): what each
        particle's change is measured from, a bond's by :meth:`radius_moved` and every particle's of the sheet by a
        reset. The sheet's alone: held whole, the model's array kept every other particle's radius for the life of
        the tape, 0.4-4 MB beside an MPM material of 10^5-10^6 particles."""
        self._start: int = 0
        """The sheet's first particle in the model, as built: a particle less it is its offset into :attr:`_radius`."""
        self._laid: Any = None
        """Opening adhesion: per bond, the contact thickness its particle and anchor are laid at [m], on the device:
        the particle's radius plus the margin the sheet is laid at (:meth:`_relay`). Contact adhesion's is its
        ``dmin``."""
        self._normals: Any = None
        self._arrays: dict[str, list[Any]] = {}
        self._rows: Any = None
        self._particle: Any = None
        self._body: Any = None
        self._anchor: Any = None
        self._limit: Any = None
        self.broken: Any = None
        """Per bond, 1 once it has let go, on the device."""
        self.contact: bool = False
        """Contact adhesion: bonds made where the sheet is pressed, held by ipc-toolkit's law (section 1)."""
        self.bond_area: np.ndarray = np.zeros(0)
        """Each bond's share of the sheet's area [m^2]."""
        self.clamp_particles: np.ndarray = np.zeros(0, dtype=np.int32)
        self.clamp_points: np.ndarray = np.zeros((0, 3))
        """Where the tab holds each clamped particle, in the tab's frame [m]."""
        self._contact: dict[str, Any] = {}

    # -------------------------------------------------------------- phase 1
    def _named_body(self, builder: newton.ModelBuilder, name: str) -> int:
        hits = [b for b, label in enumerate(builder.body_label) if str(label).split("/")[-1] == name]
        if len(hits) != 1:
            raise KeyError(
                f"tape {self.spec.name!r}: on={name!r} names {len(hits)} bodies; add the body to the scene before "
                "the tape, under a name nothing else uses"
            )
        return int(hits[0])

    def _lay(
        self, builder: newton.ModelBuilder, sheet: wp.transform, flat: np.ndarray
    ) -> tuple[wp.transform, np.ndarray]:
        """A stuck sheet laid where the box's contact rests it: the sheet's pose, and which of its bondable
        vertices lie over a face of the body's boxes.

        Over the face is within its extent, by :func:`over_a_face`, the test the stick kernel makes a bond
        by. Past the edge nothing is under a vertex. Bonded there, a contact bond was seated on the edge
        along its tilted normal and let go on the first substep, read as peeled before anything pulled,
        and an opening bond held its vertex where it was built, the overhang stuck level in the air
        (``t2_overhang.py``). On a body with no box every vertex is over it: the face is the sheet's own.

        The sheet is placed a particle radius off the face, and VBD's contact rests a particle the box's
        margin further out, so it is laid that much further, as far as the largest margin of the boxes
        under its bonds (:attr:`lift`; with no box, of the body's shapes). Built a radius off a face
        with a margin, a stuck label was pushed out through it, overshot, and let its bonds go: of
        :data:`LABEL`'s 180, 178 held at a 0.8 mm margin and none 2-5 substeps in at 1 mm and up; and
        :data:`PAPER_LABEL`'s rested 0.47 mm lifted at 0.5 mm, 54 held at 0.8 mm and none past the first
        substep at 1.2 mm. Laid here, all 180 of either held 60 substeps at every margin from 0.3 to
        3 mm, none lifted (``z2b.py``, ``z2_opening.py``, VBD on the CPU).
        """
        to_body = wp.transform_inverse(_transform(builder.body_q[self.body]))
        local = [wp.transform_point(to_body, wp.transform_point(sheet, wp.vec3(*[float(c) for c in v]))) for v in flat]
        box = int(newton.GeoType.BOX)
        mine = [s for s in range(builder.shape_count) if builder.shape_body[s] == self.body]
        boxes = [s for s in mine if int(builder.shape_type[s]) == box]
        over, under = np.ones(len(flat), dtype=bool), np.asarray(mine, dtype=np.int64)
        if boxes:
            over, nearest = over_a_face(
                np.asarray(local, dtype=np.float64).reshape(-1, 3),
                [[float(c) for c in builder.shape_transform[s]] for s in boxes],
                [[float(c) for c in builder.shape_scale[s]] for s in boxes],
                [float(builder.shape_margin[s]) for s in boxes],
            )
            under = np.asarray(sorted({boxes[int(b)] for b in nearest[over]}), dtype=np.int64)
        self.lift_shapes = under
        self.lift = max((float(builder.shape_margin[s]) for s in under), default=0.0)
        up = wp.transform_vector(sheet, wp.vec3(0.0, 0.0, 1.0))
        laid = wp.transform(wp.transform_get_translation(sheet) + up * self.lift, wp.transform_get_rotation(sheet))
        return laid, over

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec, material = self.spec, self.spec.material
        vertices, faces, _ = tape_surface(spec.size, spec.cells, spec.tab)
        placed = self.placed((*spec.pose, *spec.quat))
        sheet = _transform(placed[:7])
        across, along = spec.cell
        rows, made = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
        if spec.on:
            self.body = self._named_body(builder, spec.on)
            self.contact = material.adhesion == "contact"
            rows = bonded_rows(vertices, spec.tab, along, spec.dry)
            made = np.full(len(rows), bool(spec.stuck))
            if spec.stuck:
                sheet, made = self._lay(builder, sheet, vertices[rows])
                if not self.contact:
                    # Opening adhesion makes its bonds at the build alone: a row with nothing under it has none.
                    rows, made = rows[made], made[made]
        self.up = np.array(wp.transform_vector(sheet, wp.vec3(0.0, 0.0, 1.0)), dtype=np.float64)
        start = int(builder.particle_count)
        self._start = start
        self.surface_count = len(vertices)
        self.lumping = "mesh"
        builder.add_cloth_mesh(
            pos=wp.transform_get_translation(sheet),
            rot=wp.transform_get_rotation(sheet),
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

        if spec.tab > 0.0:
            self._build_tab(builder, vertices, start, sheet)
        if not spec.on:
            return
        to_body = wp.transform_inverse(_transform(builder.body_q[self.body]))
        self.normal = np.array(wp.transform_vector(to_body, wp.transform_vector(sheet, wp.vec3(0.0, 0.0, 1.0))))
        self.bond_area = sheet_areas(spec.size, spec.cells)[rows]
        stiffness = np.full(len(rows), float(material.tack) * across * along)
        if self.contact:
            # At the radius alone: the bind seats each at the box's margin too, spring and anchor, as the
            # kernels read it.
            stiffness = self.bond_area * contact_stiffness(material, float(material.particle_radius))
        bonds, anchors = [], []
        for i, k, on in zip(rows, stiffness, made, strict=True):
            p = start + int(i)
            local = np.array(wp.transform_point(to_body, wp.vec3(*builder.particle_q[p])))
            bonds.append(
                builder.add_attachment_particle_body(
                    p, self.body, body_point=wp.vec3(*local), stiffness=float(k), enabled=bool(on)
                )
            )
            anchors.append(local)
        self.made = made
        self.bonds = np.asarray(bonds, dtype=np.int32)
        self.bond_particles = (start + rows).astype(np.int32)
        self.anchors = np.asarray(anchors, dtype=np.float64).reshape(-1, 3)
        self.along = vertices[rows, 1] - float(spec.tab)

    def _build_tab(self, builder: newton.ModelBuilder, vertices: np.ndarray, start: int, sheet: wp.transform) -> None:
        """The pull tab: a block standing on the fold, the turned-up end clamped inside it."""
        spec = self.spec
        half = [0.5 * float(v) for v in spec.tab_size]
        centre = wp.transform_point(sheet, wp.vec3(0.5 * float(spec.size[0]), float(spec.tab), half[2]))
        pose = wp.transform(centre, wp.transform_get_rotation(sheet))
        self.tab_body = builder.add_body(xform=pose, label=f"{spec.name}_tab")
        self.coord_start = int(builder.joint_q_start[builder.joint_count - 1])
        self.tab_q0 = np.asarray(builder.joint_q[self.coord_start : self.coord_start + 7], dtype=np.float64)
        cfg = builder.default_shape_cfg.copy()
        cfg.mu = float(spec.tab_friction)
        cfg.density = float(spec.tab_mass) / (8.0 * half[0] * half[1] * half[2])
        # The end lies inside the tab, so the tab must not push it out.
        cfg.has_particle_collision = False
        self.tab_shapes = [int(builder.add_shape_box(self.tab_body, hx=half[0], hy=half[1], hz=half[2], cfg=cfg))]
        to_tab = wp.transform_inverse(pose)
        clamps, held, points = [], [], []
        for i in clamped_rows(vertices, spec.tab):
            p = start + int(i)
            local = wp.transform_point(to_tab, wp.vec3(*builder.particle_q[p]))
            clamps.append(
                builder.add_attachment_particle_body(p, self.tab_body, body_point=local, stiffness=spec.tab_hold)
            )
            held.append(p)
            points.append(np.array(local, dtype=np.float64))
        self.clamps = np.asarray(clamps, dtype=np.int32)
        self.clamp_particles = np.asarray(held, dtype=np.int32)
        self.clamp_points = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    def apply_display(self, builder: newton.ModelBuilder) -> None:
        """The sheet's colour everywhere but the tab, which keeps its own."""
        super().apply_display(builder)
        from robosimstudio.core.render.finish import finish_for

        color = tuple(float(v) for v in self.spec.tab_color)
        for s in self.tab_shapes:
            builder.shape_color[s] = color
            self.finishes[s] = finish_for(color, "objects")

    # -------------------------------------------------------------- phase 2
    def tunables(self, pipeline=None):
        """Cloth's material tunable, with the adhesive's fields routed to the bonds.

        Only the law the tape was built with reads its fields. Under contact adhesion the law sets each
        bond's spring every substep and no break kernel runs, so ``tack`` and ``break_opening`` move
        nothing; under opening adhesion no contact kernel runs, so :data:`CONTACT_FIELDS` move nothing.
        A strip stuck to nothing has no bonds for any of them to reach, so there every adhesive field
        moves nothing. Those are the tunable's ``inert`` fields, shown read-only, and a write to one
        does not land. Its ``write_many``, what a whole material's write takes, writes contact
        adhesion's fields as one set (:meth:`write_adhesive_all`).
        """
        base = super().tunables(pipeline)
        if not base:
            return base
        (material,) = base
        inner = material.write
        idle = ("tack", "break_opening") if self.contact else CONTACT_FIELDS
        if not len(self.bonds):
            idle = ADHESIVE_FIELDS

        def write(name: str, value: Any) -> bool:
            if name in idle:
                return False
            if name == "tack":
                self.write_tack(float(value))
                return True
            if name == "break_opening":
                self.write_break_opening(float(value))
                return True
            if name in CONTACT_FIELDS:
                return self.write_adhesive(name, float(value))
            return inner(name, value)

        def write_many(values: dict[str, Any]) -> tuple[str, ...]:
            # Contact adhesion's fields as one set, and first, so a set refused writes nothing.
            law = {name: float(v) for name, v in values.items() if name in CONTACT_FIELDS and name not in idle}
            landed = list(law) if law and self.write_adhesive_all(law) else []
            return (*landed, *(name for name, v in values.items() if name not in law and write(name, v)))

        return (replace(material, write=write, write_many=write_many, inert=(*material.inert, *idle)),)

    def write_break_opening(self, opening: float) -> None:
        """Set how far a bond's particle may lift before it lets go [m]; 0 never lets go. LIVE: the kernel
        reads it every substep. Bonds already broken stay broken."""
        if self._limit is not None:
            self._limit.assign(np.array([float(opening)], dtype=np.float32))

    def write_adhesive(self, name: str, value: float) -> bool:
        """Set one of contact adhesion's fields (:data:`CONTACT_FIELDS`). LIVE: the kernels read them every
        substep. ``strength`` 0 lets every bond go and makes none.

        Returns:
            Whether it landed: ``False`` before the kernels' parameters are bound, which a tape with no
            bonds never has, and nothing is written.

        Raises:
            KeyError: For a name that is not one of :data:`CONTACT_FIELDS`.
            ValueError: For a value :class:`TapeMaterialSpec` refuses beside the other fields as the
                kernels hold them, and nothing is written. A ``peak_opening`` at the ``reach`` makes
                every held bond's spring NaN; one past it makes a law that no longer peaks at
                ``strength``.
        """
        return self.write_adhesive_all({name: value})

    def write_adhesive_all(self, values: dict[str, float]) -> bool:
        """Set several of contact adhesion's fields as one set: checked together, then written together.

        What a whole material's write takes, the material tunable's ``write_many``. One field at a time,
        a law the spec allows can pass through one it refuses on its way to another: from a
        ``peak_opening`` of 0.1 mm and a ``reach`` of 0.4 mm back to :data:`LABEL`'s 0.5 and 1 mm, the
        peak lands first, past the old reach. The fields not in ``values`` are as the kernels hold them.
        Returns and raises as :meth:`write_adhesive` does, for every field in ``values``.
        """
        for name in values:
            if name not in CONTACT_FIELDS:
                raise KeyError(f"{name!r} is not one of contact adhesion's fields {CONTACT_FIELDS}")
        params = self._contact.get("params")
        if params is None:
            return False
        # A copy: on the CPU ``numpy()`` is the array itself, and a refused value must not land.
        held = np.array(params.numpy(), copy=True)
        for name, value in values.items():
            held[CONTACT_FIELDS.index(name)] = float(value)
        try:
            replace(self.spec.material, **{f: float(v) for f, v in zip(CONTACT_FIELDS, held, strict=True)})
        except ValueError as exc:
            written = ", ".join(f"{name} = {value}" for name, value in values.items())
            raise ValueError(f"tape {self.spec.name!r}: {written} not written: {exc}") from None
        params.assign(held)
        return True

    def write_tack(self, tack: float) -> None:
        """Set every bond's stiffness from a tack [N/m per m^2], in every array a solver reads. LIVE.

        Opening adhesion only: under contact adhesion the law sets each bond's spring every substep."""
        if self.contact:
            return
        across, along = self.spec.cell
        for array in self._arrays.get("stiffness", ()):
            values = array.numpy()
            values[self.bonds] = float(tack) * across * along
            array.assign(values)

    # -------------------------------------------------------------- phase 3
    def contribute_hooks(self, hooks) -> None:
        if len(self.bonds):
            hooks.add("before", self._relay, name=f"relay:{self.spec.name}")
        if len(self.bonds) and self.contact:
            hooks.add("post_solve", self._stick, name=f"stick:{self.spec.name}")
        elif len(self.bonds):
            hooks.add("post_solve", self._break, name=f"bonds:{self.spec.name}")

    def _relay(self, state: newton.State, dt: float) -> None:
        """Once a control step, before its substeps: every held bond laid again where VBD's contact rests its
        particle, the particle and the anchor moved off the face by how far the contact thickness has moved
        since the bond was laid. What a reset does for the sheet (:meth:`margin_moved`), between resets.

        The box's contact ``thickness`` writes its margin live, and the next collide rests a particle that
        much further off the face; a sheet left where it lay was pushed out through it and overshot. A
        tabbed label stuck at no margin beside the YAM, the margin written live with no reset, kept none of
        :data:`PAPER_LABEL`'s 144 bonds at 1 and 2 mm and 5 and 8 of :data:`LABEL`'s; laid again here, all
        144 of either at 1 mm, and at 2 mm all but the row by the fold or part of it, 135-140 in two drives
        each (``scene_live.py``, ``scene_live2.py``, GPU 1). That row goes as the tab, which rests on the box
        by its own contact and is not moved here, is thrown up 16 mm by the jump; written in 0.25 mm steps a
        control step apart, as a dragged slider can write it, all 144 of either held to 2 mm, the tab rising
        4.6 mm at most. Moved by its pose too, the tab came off worse, VBD reading the jump as a velocity
        against its own pose history, and with the anchors moved alone 46 and 9 bonds held at 1 mm
        (``a2_variants.py``, VBD on the CPU). It runs once a control step, outside the graph, because a
        write lands between control steps: in a substep phase the first solve after it has already pushed
        the sheet out. A particle radius written live moves the thickness too, and is laid again alike: at
        1.5 mm, from 0.5, all 144 held on the CPU where 2 and none did (``a2_cpu2.py``); a reset lays it too
        (:meth:`radius_moved`)."""
        del dt
        if self._laid is None:
            return
        c = self._contact
        model = self._model
        arrays = (c["dmin"], c["normal"]) if c else (self._laid, self._normals)
        body_point = self._arrays["body_point"]
        wp.launch(
            _relay_kernel,
            dim=len(self.bonds),
            inputs=[
                self._rows,
                self._particle,
                self.body,
                1 if c else 0,
                self._under,
                c["boxes"] if c else self._under,
                model.shape_transform,
                model.shape_scale,
                model.shape_margin,
                model.particle_radius,
                arrays[1],
                state.body_q,
                self._arrays["enabled"][0],
            ],
            outputs=[arrays[0], self._anchor, body_point[0], body_point[-1], state.particle_q],
            device=state.particle_q.device,
        )

    def _stick(self, state: newton.State, dt: float) -> None:
        """Contact adhesion's substep: make bonds where pressed, then hold or release each by the law."""
        c = self._contact
        if not c:
            return
        model = self._model
        device = state.particle_q.device
        # The solver's copy and the model's, the same array when the solver reads the model's.
        keys = ("enabled", "body_point", "stiffness")
        enabled, body_point, stiffness = ((self._arrays[key][0], self._arrays[key][-1]) for key in keys)
        wp.launch(
            _stick_kernel,
            dim=len(self.bonds),
            inputs=[
                self._rows,
                self._particle,
                self.body,
                c["boxes"],
                model.shape_transform,
                model.shape_scale,
                model.shape_margin,
                model.particle_radius,
                c["area"],
                c["params"],
                float(model.soft_contact_ke),
                model.shape_material_ke,
                state.particle_q,
                state.body_q,
            ],
            outputs=[*enabled, *body_point, c["normal"], c["dmin"], self.broken],
            device=device,
        )
        wp.launch(
            _bond_kernel,
            dim=len(self.bonds),
            inputs=[
                self._rows,
                self._particle,
                self.body,
                c["area"],
                c["normal"],
                c["dmin"],
                c["params"],
                float(dt),
                state.particle_q,
                state.particle_qd,
                state.body_q,
                state.body_qd,
                model.body_com,
            ],
            outputs=[*enabled, *body_point, *stiffness, self.broken],
            device=device,
        )

    def _break(self, state: newton.State, dt: float) -> None:
        del dt
        if self._rows is None:
            return
        wp.launch(
            _break_kernel,
            dim=len(self.bonds),
            inputs=[
                self._rows,
                self._particle,
                self._body,
                self._anchor,
                wp.vec3(*[float(v) for v in self.normal]),
                self._limit,
                state.particle_q,
                state.body_q,
            ],
            outputs=[self._arrays["enabled"][0], self._arrays["enabled"][-1], self.broken],
            device=state.particle_q.device,
        )

    # -------------------------------------------------------------- phase 4
    def on_scene_ready(self, pipeline: ScenePipeline) -> None:
        """Find the attachment arrays the solver reads, then settle and snapshot like cloth."""
        self.built = np.asarray(pipeline.model.particle_q.numpy()[self.particles], dtype=np.float32).copy()
        if len(self.bonds) or len(self.clamps):
            self._bind_bonds(pipeline)
        super().on_scene_ready(pipeline)

    def _bind_bonds(self, pipeline: ScenePipeline) -> None:
        """Point the break kernel and the tunables at the solver's own copy of the attachments.

        Raises:
            RuntimeError: When the solver holds fewer attachments than the model, which is Newton
                dropping the ones whose body is on another solver, or holds them reordered.
        """
        model = pipeline.model
        view = model
        program = getattr(pipeline, "program", None)
        view_for = getattr(program, "view_for", None)
        if callable(view_for):
            view = view_for(self.spec.name)
        held, built = int(view.attachment_particle_body_count), int(model.attachment_particle_body_count)
        if held != built:
            raise RuntimeError(
                f"tape {self.spec.name!r}: its solver holds {held} of the model's {built} attachments. Newton drops "
                f"an attachment whose body is on another solver, so {self.spec.on!r} and the tape must share one: "
                "stage rss.TAPE_60HZ and make the body a free rss.box"
            )
        mine = model.attachment_particle_body_particle.numpy()
        if view is not model and not np.array_equal(view.attachment_particle_body_particle.numpy(), mine):
            raise RuntimeError(f"tape {self.spec.name!r}: its solver holds the attachments in another order")
        self._arrays = {
            key: [getattr(view, f"attachment_particle_body_{key}")]
            + ([getattr(model, f"attachment_particle_body_{key}")] if view is not model else [])
            for key in ("enabled", "stiffness", "body_point")
        }
        device = model.device
        self._rows = wp.array(self.bonds, dtype=wp.int32, device=device)
        self._particle = wp.array(self.bond_particles, dtype=wp.int32, device=device)
        self._body = wp.array(np.full(len(self.bonds), self.body, dtype=np.int32), dtype=wp.int32, device=device)
        self._anchor = wp.array(self.anchors.astype(np.float32), dtype=wp.vec3, device=device)
        self._limit = wp.array([float(self.spec.material.break_opening)], dtype=wp.float32, device=device)
        self._model = model
        self._under = wp.array(self.lift_shapes.astype(np.int32), dtype=wp.int32, device=device)
        self._radius = self._radii_now()
        self._laid = wp.array(self._laid_at(self.lift, self._radius), dtype=float, device=device)
        self._normals = wp.array(
            np.tile(self.normal, (len(self.bonds), 1)).astype(np.float32), dtype=wp.vec3, device=device
        )
        self.broken = wp.zeros(len(self.bonds), dtype=wp.int32, device=device)
        if self.contact and len(self.bonds):
            self._bind_contact(pipeline)

    def _bind_contact(self, pipeline: ScenePipeline) -> None:
        """Contact adhesion's device arrays, and the body's boxes it measures presses against.

        Raises:
            ValueError: When the body has a collision shape that is not a box: the press is measured
                against boxes only.
        """
        model = pipeline.model
        device = model.device
        material = self.spec.material
        bodies = model.shape_body.numpy()
        types = model.shape_type.numpy()
        mine = np.flatnonzero(bodies == self.body)
        boxes = mine[types[mine] == int(newton.GeoType.BOX)]
        if len(boxes) != len(mine) or not len(boxes):
            raise ValueError(
                f"tape {self.spec.name!r}: contact adhesion measures a press against boxes, and {self.spec.on!r} "
                f"has shapes of types {sorted(set(types[mine].tolist()))}; stick it to an rss.box"
            )
        self._contact = {
            "boxes": wp.array(boxes.astype(np.int32), dtype=wp.int32, device=device),
            "area": wp.array(self.bond_area.astype(np.float32), dtype=float, device=device),
            "normal": wp.array(
                np.tile(self.normal, (len(self.bonds), 1)).astype(np.float32), dtype=wp.vec3, device=device
            ),
            "dmin": wp.zeros(len(self.bonds), dtype=float, device=device),
            "params": wp.array(
                np.array([float(getattr(material, f)) for f in CONTACT_FIELDS], dtype=np.float32),
                dtype=float,
                device=device,
            ),
        }
        self._seat_bonds()

    def _seat_bonds(self) -> None:
        """Every contact bond's contact thickness, its spring at contact and its anchor, in every array a
        solver reads.

        From the particle's radius and the margin of the box nearest the bond's built anchor, as the model
        holds them now: a margin moved live since the build is the one the stick kernel makes a bond with.
        The anchor goes that far off the face, as the stick kernel puts it, where VBD's contact rests the
        particle and where a stuck sheet is laid at the build and at a reset (:meth:`_lay`,
        :meth:`margin_moved`). Seated there over a sheet built a radius off the face, every bond started
        the margin into its lift, and from 1 mm the push out through the margin overshot ``reach``.
        """
        c = self._contact
        model, stiffness, body_point = self._model, self._arrays["stiffness"], self._arrays["body_point"]
        wp.launch(
            _seat_kernel,
            dim=len(self.bonds),
            inputs=[
                self._rows,
                self._particle,
                self._anchor,
                c["boxes"],
                model.shape_transform,
                model.shape_scale,
                model.shape_margin,
                model.particle_radius,
                c["area"],
                c["params"],
            ],
            outputs=[c["dmin"], c["normal"], stiffness[0], stiffness[-1], body_point[0], body_point[-1]],
            device=c["params"].device,
        )

    # -------------------------------------------------------------- phase 5
    def reset(self, pipeline: ScenePipeline, episode, *, world: int = 0) -> None:
        """The sheet, the tab and every bond back as they were built.

        A particle thing's reset restores its whole-array snapshot, taken after the scene's settle, and
        a settle steps the label too. So this sheet's rows are put back where it was built in every
        such snapshot, the ones later things restore included (``TapeSpec.settle_frames``). A stuck
        sheet whose box's margin has moved since the build goes back that much further off the face,
        its tab and its opening bonds' anchors with it (:meth:`margin_moved`), and each particle whose
        radius has moved goes that much further again, by its own change, a bond's anchor with it and the
        tab and the end it clamps not, as :meth:`_relay` lays each bond between resets
        (:meth:`radius_moved`). Lifted by the largest change, a radius written to half the sheet laid the
        other half that much over the thickness each of its bonds is seated at. A sheet laid on the face with
        no bond, held only by its tab, is laid so too: stuck to nothing, it still rests where the contact
        rests it. A tabbed label with no bonded row on a cylinder, its radius written from 0.5 to 1.5 mm and
        the scene reset, kept its face row within 0.21 mm of that over 32 substeps; left where it was built,
        as a reset left it while the radius was held per bond, the row started 1 mm inside the contact and
        was thrown 2.3 mm past it by the eighth (``f6_probe.py``, VBD on the CPU).
        """
        # Once, for the sheet, the bonds and their laying; not at all for a loose sheet, laid at no radius,
        # whose change is all 0 without it (:meth:`_radius_change`).
        radii = None if self._model is None or not len(self.lift_shapes) else self._radii_now()
        moved, change = self.margin_moved(), self._radius_change(radii)
        lifts = (moved + change[self.particles - self._start].astype(np.float64))[:, None]
        lifts[np.isin(self.particles, self.clamp_particles)] = moved
        for thing in getattr(pipeline, "particle_settles", {}).get("things", [self]):
            rest = getattr(thing, "rest", None)
            if rest and len(self.built):
                rest["particle_q"][self.particles] = self.built + (lifts * self.up).astype(np.float32)
                rest["particle_qd"][self.particles] = 0.0
        super().reset(pipeline, episode, world=world)
        if self.broken is not None:
            self._reset_bonds(moved, change[self.bond_particles - self._start], radii)
        if self.coord_start >= 0:
            c, tab = self.coord_start, self.tab_q0.copy()
            tab[:3] += moved * self.up
            for state in (pipeline.state_0, pipeline.state_1):
                state.joint_q[c : c + 7].assign(tab.astype(np.float32))

    def margin_moved(self) -> float:
        """How far the margin a stuck sheet was laid at has moved since the build [m]: what a reset lays it
        further off the face by. The box's contact ``thickness`` writes the margin live; 0 for a loose sheet.
        Between resets :meth:`_relay` lays the held bonds again by the change, at the next control step.

        Built over no margin, the margin then written and the scene reset with the sheet where it was
        built, a tabbed label's 144 bonds let go: :data:`LABEL` kept 8 at 1 mm and none from 1.5 mm,
        :data:`PAPER_LABEL` 2 at 1 mm and none from 1.5 mm. Laid this much further, all 144 of either
        held 60 substeps at 0.5 to 2 mm, none lifted (``z2_reset.py``, VBD on the CPU)."""
        if not len(self.lift_shapes) or self._model is None:
            return 0.0
        return float(self._model.shape_margin.numpy()[self.lift_shapes].max()) - self.lift

    def radius_moved(self, radii: np.ndarray | None = None) -> np.ndarray:
        """How far each bond's particle radius has moved since a stuck sheet was laid [m], per bond: what a reset
        lays that bond's particle and anchor further off the face by, past :meth:`margin_moved`. The sheet's
        material ``particle_radius`` writes it live; all 0 on a loose sheet. Off ``radii``
        (:meth:`_radii_now`), read when not given.

        Reset at the radius it was built at after a live write from 0.5 to 1.5 mm, a contact bond was seated
        at the new thickness over a particle laid at the old, 1 mm inside the contact, and the relay, finding
        the bond already at the new thickness, moved nothing: :data:`LABEL` kept 8 of its 144 bonds one control
        step in. An opening bond was laid at the old thickness and moved by the relay at the next control
        step. Laid here, all 144 of either held (``r2_probe.py``, VBD on the CPU)."""
        return self._radius_change(radii)[self.bond_particles - self._start]

    def _radii_now(self) -> np.ndarray:
        """The sheet's particle radii as the model holds them now [m], by offset into the sheet, in float32 as
        :func:`_relay_kernel` sums them: the sheet's slice alone copied off the device, once a bind or a reset.
        A reset copied the model's whole array three times under opening adhesion, for the bonds' change, the
        sheet's and the bonds' laying, and twice under contact adhesion, which seats its bonds on the device."""
        start = self._start
        return self._model.particle_radius[start : start + self.surface_count].numpy().astype(np.float32)

    def _radius_change(self, radii: np.ndarray | None = None) -> np.ndarray:
        """How far each of the sheet's particle radii has moved since the bonds were bound [m], by offset into the
        sheet, in float32 as :func:`_relay_kernel` sums it, off ``radii`` (:meth:`_radii_now`, read when not
        given); all 0 on a loose sheet, which is laid at no radius."""
        if not len(self.lift_shapes) or self._model is None or not len(self._radius):
            return np.zeros(self.surface_count, dtype=np.float32)
        return (self._radii_now() if radii is None else radii) - self._radius

    def _laid_at(self, margin: float, radii: np.ndarray) -> np.ndarray:
        """Each bond's contact thickness laid at ``margin`` over its radius in ``radii`` (:meth:`_radii_now`) [m], in
        float32 as :func:`_relay_kernel` sums it."""
        return radii[self.bond_particles - self._start] + np.float32(margin)

    def _reset_bonds(
        self, moved: float | None = None, raised: np.ndarray | None = None, radii: np.ndarray | None = None
    ) -> None:
        """Every bond as built, in every array a solver reads: made or not (:attr:`made`), at its anchor, and
        under contact adhesion seated as a bond starts (:meth:`_seat_bonds`), not with the secant, the anchor
        and the face the last episode left. The seat writes a contact bond's anchor into every copy itself;
        an opening bond's goes ``moved`` and its own ``raised`` further off the face (:meth:`margin_moved`
        and :meth:`radius_moved`, per bond), and is laid over its radius in ``radii`` (:meth:`_radii_now`), each
        read when not given."""
        self.broken.zero_()
        rows = {"enabled": self.made}
        if not self._contact:
            radii = self._radii_now() if radii is None else radii
            moved = self.margin_moved() if moved is None else moved
            raised = self.radius_moved(radii) if raised is None else raised
            lift = moved + np.asarray(raised, dtype=np.float64)[:, None]
            anchors = (self.anchors + lift * self.normal).astype(np.float32)
            rows["body_point"] = anchors
            self._anchor.assign(anchors)
            self._laid.assign(self._laid_at(self.lift + moved, radii))
        for key, value in rows.items():
            for array in self._arrays.get(key, ()):
                values = array.numpy()
                values[self.bonds] = value
                array.assign(values)
        if self._contact:
            self._seat_bonds()

    # ---------------------------------------------------------------- runtime
    def _counted(self, holding: np.ndarray | None = None, broken: np.ndarray | None = None) -> np.ndarray:
        """Per bond, whether :meth:`peeled` and :meth:`bonded` count it: on a stuck sheet the bonds made at the
        build (:attr:`made`) and any made since, holding now or let go since; on a loose one every bond. A
        stuck sheet's count reads both flags, each copied off the device here only when not passed in; a loose
        one's reads neither.

        Opening adhesion gives a stuck sheet no bond past the box's edge, and contact adhesion one left
        unmade there until it is pressed onto a face. Counted over every bond, an untouched stuck
        :data:`LABEL` half over the edge read 100/180 bonded, and peeled at most that, where the same
        sheet under :data:`PAPER_LABEL` read 1.0 and 0.0 (``tests/test_tape_kind``). Counted over the
        build's alone, its overhang pressed onto the box's side made 80 bonds that held and read nothing:
        peeled whole first, it read 0.0 bonded with all 80 holding."""
        if not self.made.any():
            return np.ones(len(self.bonds), dtype=bool)
        holding = self._holding() if holding is None else holding
        broken = self._let_go() if broken is None else broken
        return self.made | holding | broken

    def _holding(self) -> np.ndarray:
        """Per bond, whether it holds now: its attachment enabled in the solver's copy, or as built unbound."""
        enabled = self._arrays.get("enabled")
        return enabled[0].numpy()[self.bonds].astype(bool) if enabled else self.made.copy()

    def _let_go(self) -> np.ndarray:
        """Per bond, whether it has let go since it was last made."""
        if self.broken is None:
            return np.zeros(len(self.bonds), dtype=bool)
        return self.broken.numpy().astype(bool)

    def peeled(self) -> float:
        """The share of the bonds counted (:meth:`_counted`) that have let go, in ``[0, 1]``."""
        return self._peeled_of(self._let_go())

    def _peeled_of(self, gone: np.ndarray) -> float:
        """:meth:`peeled` off the let-go flags already copied (:meth:`_let_go`)."""
        if self.broken is None or not len(gone):
            return 0.0
        return float(gone[self._counted(broken=gone)].mean())

    def held(self) -> int:
        """How many bonds hold now: under contact adhesion, the ones made and not yet let go."""
        return int(self._holding().sum())

    def bonded(self) -> float:
        """The share of the bonds counted (:meth:`_counted`) holding now, in ``[0, 1]``: on a stuck sheet, under
        either law, one less :meth:`peeled`."""
        return self._bonded_of(self._holding())

    def _bonded_of(self, holding: np.ndarray) -> float:
        """:meth:`bonded` off the holding flags already copied (:meth:`_holding`)."""
        if not len(holding):
            return 0.0
        return float(holding[self._counted(holding=holding)].mean())

    def pull(self, state: newton.State) -> float:
        """How hard the sheet pulls on its tab [N]: the tab's clamps' springs, summed as vectors."""
        if self.tab_body < 0 or not len(self.clamp_particles):
            return 0.0
        pose = state.body_q.numpy()[self.tab_body].astype(np.float64)
        from robosimstudio.core.math import quat_to_mat

        held = self.clamp_points @ quat_to_mat(pose[3:7]).T + pose[:3]
        points = state.particle_q.numpy()[self.clamp_particles].astype(np.float64)
        return float(np.linalg.norm(float(self.spec.tab_hold) * (points - held).sum(axis=0)))

    def peel_front(self) -> float:
        """How far from the fold the furthest broken bond lay [m]; 0 before any lets go."""
        return self._front_of(self._let_go())

    def _front_of(self, gone: np.ndarray) -> float:
        """:meth:`peel_front` off the let-go flags already copied (:meth:`_let_go`)."""
        return float(self.along[gone].max()) if gone.any() else 0.0

    def tab_pose(self, state: newton.State) -> np.ndarray:
        """The tab's pose ``(x, y, z, qx, qy, qz, qw)`` in world, or empty without a tab."""
        if self.tab_body < 0:
            return np.zeros(0)
        return np.asarray(state.body_q.numpy()[self.tab_body], dtype=np.float64)


# =============================================================================
# 6. The constructor and the record
# =============================================================================


def tape(
    *,
    name: str = "tape",
    on: str = "",
    size: tuple[float, float] = (0.04, 0.095),
    cells: tuple[int, int] = (8, 19),
    at: Vec3 = (0.0, 0.0, 0.0),
    material: Any = None,
    **kwargs,
) -> TapeSpec:
    """A strip of tape or a label, stuck to the body ``on`` names, its first ``tab`` metres in a pull tab.

    Args:
        name: Key in handles and particle-set names.
        on: The body to stick to: a free body added to the scene before the tape. ``""`` is a loose strip.
        size: Metres, ``(across, along)``, the turned-up end included.
        cells: Quads along each axis. The default is a 5 mm cell, which every number here was
            measured at; a bond's stiffness follows the cell's area, but the peel was driven at 5 mm.
        at: Where the sheet's origin corner goes, at the tab end, in the cell's work-surface frame:
            on the face, a particle radius off it. Built stuck, the sheet is laid the box's contact
            margin further out, where VBD's contact rests it (the box's ``thickness``, 0 by default).
        material: A :class:`TapeMaterialSpec` such as one of ``rss.tapes``; ``None`` is :data:`PAPER_LABEL`.
        **kwargs: Any other :class:`TapeSpec` field: ``quat`` turns the sheet, ``tab=0`` builds no tab.
    """
    extra = {} if material is None else {"material": material}
    return TapeSpec(name=name, on=on, size=size, cells=cells, pose=at, **extra, **kwargs)


TOUCHES = ("surface", "fixtures", "objects", "hands", "deformables")
"""What a tape contacts unless it says otherwise."""


class Tape:
    """Constructors for a tape's asset record, from the spec that places one."""

    @staticmethod
    def from_spec(
        spec: TapeSpec,
        *,
        name: str,
        summary: str = "",
        tags: tuple[str, ...] = ("tape", "label"),
        states: tuple[StateSpec, ...] = (),
        notes: str = "",
    ) -> AssetRecord:
        """The tape ``rss.tape`` returned, as the :class:`AssetRecord` a task document places by type.

        Its ``on`` is the document object of that name, which the document must place too.

        Raises:
            TypeError: For a spec that is not a :class:`TapeSpec`.
        """
        if not isinstance(spec, TapeSpec):
            raise TypeError(f"Tape.from_spec takes a TapeSpec, got {type(spec).__name__}")
        return AssetRecord(
            name=name,
            summary=summary or f"a {spec.size[0] * 1e3:.0f} x {spec.size[1] * 1e3:.0f} mm strip stuck to {spec.on}",
            tags=tuple(tags),
            states=tuple(states),
            sim=SimRequirements(domain="tape", contacts=contacts_with(*TOUCHES), notes=notes),
            build=SpecFactory(spec),
        )


# =============================================================================
# 7. The measures: how much has peeled, and how much holds
# =============================================================================


register_atom(
    AtomSpec(
        name="peeled_fraction",
        summary="What share of a tape's bonds have let go, and how far from its tab the peel has reached.",
        args=(ArgSpec(name="tape", substrates=("tape",), reads="handle"),),
        directions=("check", "observe"),
        returns=("fraction", "front"),
        cost="cheap",
        reads_scene=True,
        provenance=(
            "Designed here, on the tape's own bonds: the share the bond kernels have marked broken, of a stuck "
            "sheet's bonds made at the build or since or of every bond of a loose one, a history over the "
            "episode that a reset clears. Under opening adhesion a broken bond stays broken, so a label laid back down on "
            "the face still reads peeled; under contact adhesion one pressed back on is made again."
        ),
        calibration=AtomCalibration(
            positives=(
                "examples/03_materials/08_tape, its tab taken by the right hand's pads, drawn up 30 mm and then "
                "back 20 mm and up 10 mm seven times, reads 1.0 in 24 drives of 24, all 144 bonds (peel_probe.py, "
                "GPUs 1 and 3).",
            ),
            negatives=(
                "The same label untouched reads 0, and so does the same drive with break_opening 0, a label that "
                "never lets go: the pull goes into the box instead, the tab sliding out of the pads in 10 drives "
                "of 24 or, in the other 14, the hand dragging the box 46 mm by its label (peel_probe.py, GPUs 1 "
                "and 3; tests/test_peel_a_label).",
            ),
            gate=None,
            bench="examples/03_materials/08_tape driven through its right arm panel, tests/test_peel_a_label.py",
            notes="No gate: a task states its own share. The front is in metres from the fold, along the sheet.",
        ),
    )
)


@handler("peeled_fraction", substrates=("tape",), summary="The share of the bonds broken, and the peel front [m].")
def _peeled_fraction(snapshot, tape: str) -> dict[str, float]:
    entities = getattr(snapshot, "entities", None) or {}
    entity = entities.get(tape) if hasattr(entities, "get") else None
    if not isinstance(entity, EntityTape):  # its private reads, which an entity offering peeled() lacks
        raise KeyError(f"{tape!r} is not a tape the scene carries an entity for")
    gone = entity._let_go()  # once, for the share and the front both
    fraction = entity._peeled_of(gone)
    return {"fraction": fraction, "front": entity._front_of(gone), f"{tape}/peeled_fraction": fraction}


register_atom(
    AtomSpec(
        name="bonded_fraction",
        summary="What share of a tape's bonds hold now: under contact adhesion, the ones made where it was pressed.",
        args=(ArgSpec(name="tape", substrates=("tape",), reads="handle"),),
        directions=("check", "observe"),
        returns=("fraction", "held"),
        cost="cheap",
        reads_scene=True,
        provenance=(
            "Designed here, on the tape's own bonds: the share whose attachment is enabled in the solver's "
            "copy, read this step, of a stuck sheet's bonds made at the build or since or of every bond of a "
            "loose one. "
            "Under contact adhesion a bond is made where the sheet is pressed onto its body and let go past its "
            "reach, so the share rises as a hand rubs the sheet down and falls as it peels; on a stuck sheet, "
            "under either law, it is one less peeled_fraction."
        ),
        calibration=AtomCalibration(
            positives=(
                "examples/03_materials/09_adhesion, the loose label carried by its tab, laid on the box and "
                "rubbed by the right hand's shut fingertips in four strokes 10 mm apart, reads 0.85-0.93 in twelve "
                "drives (stick_ab.py, GPU 1, and step_ms.py, GPU 5; tests/test_stick_a_label), 0.25-0.29 after "
                "the first stroke (m1_screen.py and step_ms.py, nine drives, GPUs 6 and 5).",
            ),
            negatives=(
                "The same label laid and not rubbed reads 0.008-0.048, 1-6 of its 126 bonds; rubbed with the "
                "page's Adhesive off, 0 in two drives; drawn up by its tab after the rub, 0-0.024 in eight.",
            ),
            gate=None,
            bench="examples/03_materials/09_adhesion driven through its right arm panel, tests/test_stick_a_label.py",
            notes="No gate: a task states its own share. held is a count of bonds.",
        ),
    )
)


@handler("bonded_fraction", substrates=("tape",), summary="The share of the bonds holding now, and how many.")
def _bonded_fraction(snapshot, tape: str) -> dict[str, float]:
    entities = getattr(snapshot, "entities", None) or {}
    entity = entities.get(tape) if hasattr(entities, "get") else None
    if not isinstance(entity, EntityTape):  # its private reads, which an entity offering bonded() lacks
        raise KeyError(f"{tape!r} is not a tape the scene carries an entity for")
    holding = entity._holding()  # once, for the share and the count both
    fraction = entity._bonded_of(holding)
    return {"fraction": fraction, "held": float(holding.sum()), f"{tape}/bonded_fraction": fraction}


# =============================================================================
# 8. Registration
# =============================================================================


TAPE_60HZ = GENERIC.with_backend("*", backend("vbd", rigid_body_contact_buffer_size=256, rigid_compliant_alm=True))
"""``GENERIC`` with compliant ALM on the VBD island the tape and the body it sticks to share (section 1).

Sixteen substeps, ``GENERIC``'s, because the pads' hold on the tab needs them: at twelve (29 ms a
control step) and at three VBD iterations (29 ms) the tab slid out of the pads with 58% and 63% of the
label peeled, and with self-contact off (31 ms) 81% and 56%; at sixteen it stayed within 6.4 mm through
94% (``p16_peel.py``, GPUs 1 and 6). Measured on :data:`PAPER_LABEL` beside the YAM."""

register_material_kind("tape", TapeMaterialSpec)

_PEEL = "examples/03_materials/08_tape: the right hand takes the tab and draws it up and back over the label"

register_material(
    MaterialRecord(
        name="paper_label",
        spec=PAPER_LABEL,
        notes="A paper label on a box, 40 x 80 mm, peeled by a hand on its pull tab.",
        calibration=CalibrationSpec(
            fabric="paper label stock (80 gsm), removable adhesive",
            bench=_PEEL,
            knob="tack",
            cell=0.005,
            notes=(
                "Not a drape-bench material. Tack and opening were chosen for the hand's hold on the tab: at "
                "4e6 N/m^3 and 1 mm (0.9 N across a 40 mm row at the break) the example's seven-step pull took "
                "the whole label off in eight drives, the tab within 3.6 mm of where the pads took it "
                "(m10_screen.py); "
                "at 8e6 a diagonal pull peeled 94% and then lost the tab, 58 mm of slip in two drives "
                "(p16_peel.py). So the label lets go at about 0.2 N per cm of width, a weak adhesive: the pads "
                "lift a free 0.45 kg tab (p7_rigid_tab.py), but pulled along the jaws by a peel, the tab slid "
                "out of them at 2-3.5 N (p8_peel.py). Newton 1.7, GPUs 1 and 6."
            ),
        ),
    )
)

_STICK = (
    "examples/03_materials/09_adhesion: the right hand carries the label by its tab, lays it on the box, rubs it "
    "down and draws the tab straight up"
)

for _name, _spec, _notes in (
    (
        "label",
        LABEL,
        "Rubbed down by the YAM's shut fingertips, 85-93% of the bonds held in twelve drives (stick_ab.py, "
        "step_ms.py), and drawn up by its tab all but 0-3 of 126 let go at 2.15-2.38 N in eight (stick_ab.py); "
        "on the pull rig it peeled whole at 2.9 N (p4_law.py).",
    ),
    ("masking_tape", MASKING_TAPE, "On the pull rig it peeled whole at 4.2 N (p4_law.py). Not driven by the YAM."),
    (
        "duct_tape",
        DUCT_TAPE,
        "On the pull rig it did not peel; the tab lifted and dragged the 0.5 kg box 80 mm at 4.6-7.8 N (p4_law.py).",
    ),
):
    register_material(
        MaterialRecord(
            name=_name,
            spec=_spec,
            notes="A 40 x 95 mm sheet that sticks to a box where it is pressed, its end in a pull tab.",
            calibration=CalibrationSpec(
                fabric="paper label stock (80 gsm) with a pressure-sensitive adhesive",
                bench=_STICK,
                knob="strength",
                cell=0.005,
                notes=_notes + " Newton 1.7, GPU 6.",
            ),
        )
    )

DOMAIN = register_domain(
    "tape",
    state=("particles", "bodies", "joints", "shapes"),
    reads=("tape", "cloth"),
    role="deformables",
    contacts=TOUCHES,
    summary="A sheet stuck to a body by bonds made at the build or where it is pressed, its end in a rigid pull tab.",
    engine="vbd",
    staged_by="TAPE_60HZ",
)

register_entity("tape")(EntityTape)


# Import-only; running it directly would register the tape twice.

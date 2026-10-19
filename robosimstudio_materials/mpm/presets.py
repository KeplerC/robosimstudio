# SPDX-License-Identifier: Apache-2.0
"""The particle materials, named: grains, liquids and plastic solids on one spec.

One material kind, ``"granular"``, for all three particle kinds: a heap, a pool
and a lump are one :class:`~robosimstudio_materials.mpm.spec.ParticleMaterialSpec`
with different numbers, and the registry's kind is the *spec type*, not the
domain. So ``rss.heap(material="honey")`` is accepted and does what it says --
the domain decides the backend's projection and the measures' reading, the
material decides the matter.

Where the numbers come from, in two groups. **Newton's own**: rice is the
shipped granular example's defaults (E 1e15, nu 0.3, friction 0.68, yield
pressure 1e12); water, honey and ketchup are the water-like, viscous and
multi-material examples' (friction 0, tensile ratio 1; viscosity 50; yield
stress 3e2 with viscosity 100). **Measured here**, in the probes on an
RTX PRO 6000 with Newton 1.6.0 (``probe_c_dough``, an 80x80x30 mm block pressed
15 mm by a plate): the dough's yield stress of 1e3 Pa at E 5e6 spreads to 1.15x
its footprint at the peak of the press and keeps 83% of the compression after
release; 3e3 and 1e4 barely flow (spread 1.02 and 1.01) and are the plasticine
and the clay. The rest -- sand, beans, flour, butter -- are named variants of
those with one or two numbers moved and the reason stated; none was measured
against the real thing, and the records say so.
"""

from __future__ import annotations

from robosimstudio.assetlib.materials import MaterialRecord, register_material, register_material_kind

from .spec import ParticleMaterialSpec

__all__ = [
    "BEANS",
    "BUTTER",
    "CLAY",
    "DOUGH",
    "FLOUR",
    "HONEY",
    "KETCHUP",
    "PLASTICINE",
    "RICE",
    "SAND",
    "WATER",
    "presets",
]

register_material_kind("granular", ParticleMaterialSpec)

_UNMEASURED = "Not measured against the real material; a named starting point."

# ---------------------------------------------------------------------- grains
RICE = register_material(
    MaterialRecord(
        name="rice",
        spec=ParticleMaterialSpec(
            density=1500.0,
            young_modulus=1.0e15,
            poisson_ratio=0.3,
            friction=0.68,
            yield_pressure=1.0e12,
            tensile_yield_ratio=0.0,
            color=(0.93, 0.90, 0.80),
        ),
        notes=(
            "Newton's shipped granular example, as written: rigid grains, friction 0.68, no cohesion. "
            "The probes' bed; a 12x12x3 cm heap of it at 2.5 mm spacing is 27,648 particles."
        ),
    )
)

SAND = register_material(
    MaterialRecord(
        name="sand",
        spec=ParticleMaterialSpec(
            density=1600.0,
            young_modulus=1.0e15,
            poisson_ratio=0.3,
            friction=0.75,
            yield_pressure=1.0e12,
            tensile_yield_ratio=0.0,
            color=(0.80, 0.72, 0.52),
        ),
        notes=f"Rice with the friction raised to 0.75 and the density to dry sand's. {_UNMEASURED}",
    )
)

BEANS = register_material(
    MaterialRecord(
        name="beans",
        spec=ParticleMaterialSpec(
            density=800.0,
            young_modulus=1.0e15,
            poisson_ratio=0.3,
            friction=0.55,
            yield_pressure=1.0e12,
            tensile_yield_ratio=0.0,
            color=(0.45, 0.22, 0.14),
        ),
        notes=(
            "Rice with a lower bulk density and friction, for grains larger than the voxel. "
            f"The MPM smears grains into a continuum, so a bean is a colour here, not a shape. {_UNMEASURED}"
        ),
    )
)

FLOUR = register_material(
    MaterialRecord(
        name="flour",
        spec=ParticleMaterialSpec(
            density=600.0,
            young_modulus=1.0e15,
            poisson_ratio=0.3,
            friction=0.6,
            yield_pressure=2.0e3,
            tensile_yield_ratio=0.25,
            hardening=5.0,
            dilatancy=1.0,
            color=(0.97, 0.96, 0.92),
        ),
        notes=(
            "Cohesive and compactable: a finite yield pressure with a tensile ratio of 0.25 and "
            "hardening, the shape of Newton's snow in its multi-material example (2e4 Pa, 0.2, 10) "
            f"made softer, so a pressed heap keeps the print. {_UNMEASURED}"
        ),
    )
)

# --------------------------------------------------------------------- liquids
WATER = register_material(
    MaterialRecord(
        name="water",
        spec=ParticleMaterialSpec(
            density=1000.0,
            young_modulus=1.0e15,
            poisson_ratio=0.3,
            friction=0.0,
            yield_pressure=1.0e15,
            tensile_yield_ratio=1.0,
            viscosity=0.0,
            color=(0.35, 0.60, 0.90),
        ),
        notes=(
            "Newton's water-like preset. In Newton 1.6.0 it loses volume at tabletop scale: a 60 mm "
            "column in a thick tank at 4 mm voxels was 34% of its volume after 2 s at rest "
            "(probe_c_water; P1d strain at twice the cost keeps 69%; 240 Hz keeps 52%). Count particles "
            "in a region to measure it; a height or a volume is not a measurement of this preset."
        ),
    )
)

HONEY = register_material(
    MaterialRecord(
        name="honey",
        spec=ParticleMaterialSpec(
            density=1400.0,
            young_modulus=1.0e15,
            poisson_ratio=0.3,
            friction=0.0,
            yield_pressure=1.0e15,
            tensile_yield_ratio=1.0,
            viscosity=50.0,
            color=(0.90, 0.62, 0.15),
        ),
        notes=(
            "Newton's viscous example (viscosity 50 Pa*s) at honey's density. Slumps plausibly under a "
            "15 mm press in probe_c_dough (spread 1.23 at the peak, 1.32 after release) at 9 ms per MPM "
            "step for 83k particles, and keeps its particles inside a cup where the water preset does not."
        ),
    )
)

KETCHUP = register_material(
    MaterialRecord(
        name="ketchup",
        spec=ParticleMaterialSpec(
            density=1100.0,
            young_modulus=1.0e15,
            poisson_ratio=0.3,
            friction=0.0,
            yield_pressure=1.0e10,
            tensile_yield_ratio=1.0,
            yield_stress=3.0e2,
            viscosity=100.0,
            color=(0.72, 0.10, 0.08),
        ),
        notes=(
            "Newton's multi-material example's mud: a Bingham fluid with a 3e2 Pa yield stress and "
            "viscosity 100, which holds a puddle's edge until pushed."
        ),
    )
)

# ------------------------------------------------------------- plastic solids
DOUGH = register_material(
    MaterialRecord(
        name="dough",
        spec=ParticleMaterialSpec(
            density=1100.0,
            young_modulus=5.0e6,
            poisson_ratio=0.45,
            damping=1.0e-3,
            friction=0.0,
            yield_pressure=1.0e15,
            tensile_yield_ratio=1.0,
            yield_stress=1.0e3,
            color=(0.93, 0.86, 0.70),
        ),
        notes=(
            "E 5e6, nu 0.45, yield stress 1e3 Pa, P0 strain basis. Measured in probe_c_dough (80x80x30 mm "
            "block, 15 mm plate press): footprint 1.15x at the peak, 1.21x after release, 83% of the "
            "compression kept. The Q1 strain basis blew up on this material; the arrangement states P0."
        ),
    )
)

PLASTICINE = register_material(
    MaterialRecord(
        name="plasticine",
        spec=ParticleMaterialSpec(
            density=1800.0,
            young_modulus=5.0e6,
            poisson_ratio=0.45,
            damping=1.0e-3,
            friction=0.0,
            yield_pressure=1.0e15,
            tensile_yield_ratio=1.0,
            yield_stress=3.0e3,
            color=(0.30, 0.55, 0.80),
        ),
        notes=(
            "The dough at a 3e3 Pa yield stress: footprint 1.02x at the peak of the same 15 mm press, "
            "1.05x after release, 75% of the compression kept (probe_c_dough). Holds a print, barely flows."
        ),
    )
)

CLAY = register_material(
    MaterialRecord(
        name="clay",
        spec=ParticleMaterialSpec(
            density=1900.0,
            young_modulus=5.0e6,
            poisson_ratio=0.45,
            damping=1.0e-3,
            friction=0.0,
            yield_pressure=1.0e15,
            tensile_yield_ratio=1.0,
            yield_stress=1.0e4,
            color=(0.62, 0.40, 0.30),
        ),
        notes=(
            "The dough at a 1e4 Pa yield stress: footprint 1.01x at the peak of the 15 mm press, 66% of "
            "the compression kept (probe_c_dough). Stiff; a press leaves a dent and nothing spreads."
        ),
    )
)

BUTTER = register_material(
    MaterialRecord(
        name="butter",
        spec=ParticleMaterialSpec(
            density=910.0,
            young_modulus=5.0e6,
            poisson_ratio=0.45,
            damping=1.0e-3,
            friction=0.0,
            yield_pressure=1.0e15,
            tensile_yield_ratio=1.0,
            yield_stress=3.0e2,
            viscosity=20.0,
            color=(0.98, 0.88, 0.45),
        ),
        notes=(
            "The dough's elasticity with ketchup's yield stress (3e2 Pa) and a viscosity of 20: spreads "
            f"under a knife and stays where it is left. {_UNMEASURED}"
        ),
    )
)


def presets() -> dict[str, MaterialRecord]:
    """Every particle material record here, by name."""
    return {
        record.name: record
        for record in (RICE, SAND, BEANS, FLOUR, WATER, HONEY, KETCHUP, DOUGH, PLASTICINE, CLAY, BUTTER)
    }

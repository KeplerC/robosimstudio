# SPDX-License-Identifier: Apache-2.0
"""The Newton solvers as a bundle: ``mujoco``, ``vbd``, ``xpbd``, ``featherstone``, ``kinematic`` and ``mpm``.

Importing this registers the six aliases and nothing else -- every ``cls`` is a
dotted path, so resolving an alias imports no solver, exactly as the two foreign
bundles beside this one work. Four are Newton's own; ``kinematic`` is ours, a
body that follows its joint targets exactly, and it integrates Newton's state on
the device like the rest, which is why it is here and not under a host engine;
``mpm`` is Newton's implicit MPM behind a constructor that takes flat kwargs
(:mod:`robosimstudio.engines.newton.mpm`), for granular, liquid and plastic
materials as particles.

Now every alias arrives the same way: a miss scans ``robosimstudio.engines``,
and this bundle is found as ``superdex`` and ``genesis`` are. An arm-only scene
still resolves ``mujoco`` with nothing imported by hand;
``tests/test_engine_roots.py`` proves it in a fresh subprocess with only
``robosimstudio.specs`` imported first.

Not a foreign engine, and it lives in ``engines/`` anyway because an alias has
one kind of home now. The schemas sit beside the registrations because they are
the registrations': ``MUJOCO_SCHEMA`` says what each of MuJoCo's constructor
kwargs is and what changing it costs, and it has no reader that does not go
through ``get_backend("mujoco").schema``.
"""

from __future__ import annotations

from robosimstudio.specs.params import RECAPTURE
from robosimstudio.specs.physics import ParamRule, register_backend, rule

__all__ = ["FEATHERSTONE", "KINEMATIC", "MPM", "MPM_SCHEMA", "MUJOCO", "MUJOCO_SCHEMA", "VBD", "VBD_SCHEMA", "XPBD"]

MUJOCO_SCHEMA: dict[str, ParamRule] = {
    "nconmax": rule(group="buffers", doc="Contact ceiling, sized for the busiest frame."),
    "njmax": rule(group="buffers", doc="Constraint ceiling."),
    "jacobian": rule(choices=("sparse", "dense", "auto"), group="solver"),
    "iterations": rule(tier=RECAPTURE, apply="custom:mujoco_opt", range=(1.0, 200.0), group="solver"),
    "ls_iterations": rule(tier=RECAPTURE, apply="custom:mujoco_opt", range=(1.0, 50.0), group="solver"),
    "cone": rule(choices=("pyramidal", "elliptic"), group="solver"),
    "impratio": rule(range=(1.0, 100.0), group="solver"),
    "solver": rule(choices=("newton", "cg", "pgs"), group="solver"),
    "integrator": rule(choices=("euler", "implicitfast", "implicit"), group="solver"),
    "enable_multiccd": rule(group="contact", doc="Several contact points per pair; stops a resting mesh walking."),
    "use_mujoco_contacts": rule(group="contact", doc="Off routes collision through a Newton pipeline."),
}

VBD_SCHEMA: dict[str, ParamRule] = {
    "iterations": rule(tier=RECAPTURE, apply="attr:iterations", range=(1.0, 100.0), group="solver"),
    "friction_epsilon": rule(
        tier=RECAPTURE, apply="attr:friction_epsilon", log=True, range=(1e-4, 1e-1), group="solver"
    ),
    "particle_enable_self_contact": rule(group="self-contact"),
    "particle_self_contact_radius": rule(unit="m", range=(1.0e-4, 2.0e-2), group="self-contact"),
    "particle_self_contact_margin": rule(unit="m", range=(1.0e-4, 2.0e-2), group="self-contact"),
    "particle_topological_contact_filter_threshold": rule(range=(0.0, 5.0), step=1.0, group="self-contact"),
    "particle_rest_shape_contact_exclusion_radius": rule(unit="m", range=(0.0, 3.0e-2), group="self-contact"),
    "particle_vertex_contact_buffer_size": rule(range=(8.0, 512.0), group="buffers"),
    "particle_edge_contact_buffer_size": rule(range=(8.0, 512.0), group="buffers"),
    "rigid_avbd_beta": rule(log=True, range=(1.0, 1.0e4), group="rigid"),
    "rigid_contact_k_start": rule(log=True, range=(1.0, 1.0e6), group="rigid"),
    "rigid_contact_history": rule(group="rigid"),
    "rigid_body_contact_buffer_size": rule(range=(16.0, 1024.0), group="buffers"),
    "rigid_body_particle_contact_buffer_size": rule(range=(16.0, 2048.0), group="buffers"),
}

MPM_SCHEMA: dict[str, ParamRule] = {
    # The numbers in the docs are the Phase 0 probes' (RTX PRO 6000, Newton
    # 1.6.0, a 29-33k-particle rice bed): see robosimstudio.engines.newton.mpm.
    "voxel_size": rule(
        unit="m",
        log=True,
        range=(1.0e-3, 1.0e-1),
        group="grid",
        doc="Background grid cell. Step time is collider-bound, not particle-bound (20k, 30k and 50k "
        "particles cost the same, 3 to 5 mm voxels cost the same); what the voxel buys is the thinnest "
        "tool a heap resolves: a blade must be 1.5-2 voxels thick to push rather than pass through.",
    ),
    "grid_type": rule(
        choices=("sparse", "fixed", "dense"),
        group="grid",
        doc="A capacity-bounded sparse grid rebuilds in place and captures; a particle that leaves a fixed "
        "grid turns the whole state to NaN within two steps with no error.",
    ),
    "grid_padding": rule(range=(0.0, 128.0), step=1.0, group="grid", doc="Empty cells around the particles."),
    "max_active_cell_count": rule(
        group="buffers",
        doc="Reserved sparse-grid cells. Overflow is reported by the after-step grid check, not by a crash.",
    ),
    "max_leaf_node_count": rule(group="buffers"),
    "max_lower_node_count": rule(
        group="buffers",
        doc="NanoVDB lower nodes: spatial spread headroom. 4096 removed the overflow a poured slab hit at step 164.",
    ),
    "max_upper_node_count": rule(group="buffers"),
    "max_iterations": rule(tier=RECAPTURE, apply="attr:max_iterations", range=(1.0, 500.0), step=1.0, group="solver"),
    "tolerance": rule(tier=RECAPTURE, apply="attr:tolerance", log=True, range=(1.0e-6, 1.0e-2), group="solver"),
    "strain_basis": rule(
        choices=("P0", "P1d", "Q1", "Q1d"),
        group="solver",
        doc="P0 is the measured default; P1d costs 2x and keeps a water column's volume better (0.69 vs 0.41 "
        "after 2 s); Q1 blew up on the dough press.",
    ),
    "velocity_basis": rule(choices=("Q1", "B2", "B3"), group="solver"),
    "collider_basis": rule(
        choices=("S2", "Q1", "S3", "pic"),
        group="contact",
        doc="S2 is the thin-tool default (14-16% of swept rice left behind a 3 mm paddle; Q1 doubles that). "
        "pic leaks least from a thin cup (1-2% vs 16-81%) but copies to the host inside its step and CANNOT be "
        "graph-captured: CUDA error 906 during capture, and the context is poisoned after it. Use it only with "
        "use_cuda_graph=False on the arrangement.",
    ),
    "collider_velocity_mode": rule(choices=("forward", "backward"), group="contact"),
    "transfer_scheme": rule(choices=("apic", "pic"), group="solver"),
    "integration_scheme": rule(choices=("pic", "gimp"), group="solver"),
    "warmstart_mode": rule(choices=("auto", "none", "particles", "grid", "smoothed"), group="solver"),
    "critical_fraction": rule(range=(0.0, 1.0), group="material"),
    "air_drag": rule(range=(0.0, 10.0), group="material"),
    "separate_worlds": rule(
        group="worlds",
        doc="One grid per world. On by default for a replicated model: with a shared grid rice tunnelled "
        "through another world's ground plane.",
    ),
    "project_outside": rule(
        group="contact",
        doc="Project particles out of the colliders after each step: 9.9% of swept rice left behind a thin "
        "paddle instead of 16%. Harmful for a liquid, which it pushes into the container wall at 6x the cost.",
    ),
    "project_gap": rule(
        unit="m",
        range=(0.0, 0.1),
        group="contact",
        doc="Query distance of the projection; None is voxel*sqrt(3). 30 mm stopped a 2.4 m/s pour passing "
        "through a 10 cm slab at 120 Hz.",
    ),
    "step_hz": rule(
        unit="Hz",
        range=(30.0, 960.0),
        group="rates",
        doc="The solver's own rate, as a ceiling under the substep rate; it must divide control_hz*substeps. "
        "Per 60 Hz control step, a 29k rice bed with the YAM on bench_075 under GENERIC, captured: 15.6 ms at 60, "
        "25.9 ms at 120 (21.1 with 20 iterations), 163 ms at 960.",
    ),
}

MUJOCO = register_backend(
    "mujoco",
    cls="newton.solvers.SolverMuJoCo",
    supports=("bodies", "joints", "shapes"),
    native_contacts=True,
    schema=MUJOCO_SCHEMA,
    summary="MuJoCo-Warp: articulated rigid bodies, the robot's solver.",
    # ``jacobian`` sparse rather than MuJoCo's "auto" is measured: five stacked
    # boxes stand every time under sparse and three times in eight under auto
    # (a stacking benchmark run during development); the ceilings are sized for the busiest frame,
    # because an overflow escalates into an illegal memory access.
    defaults={
        "nconmax": 2048,
        "njmax": 2048,
        "jacobian": "sparse",
        "iterations": 20,
        "ls_iterations": 20,
        "cone": "elliptic",
        "impratio": 10.0,
        "enable_multiccd": False,
        "use_mujoco_contacts": True,
    },
)
VBD = register_backend(
    "vbd",
    cls="newton.solvers.SolverVBD",
    supports=("bodies", "joints", "particles", "shapes"),
    maximal_coordinates=True,
    schema=VBD_SCHEMA,
    summary="Vertex Block Descent: cloth, soft bodies, and chains of capsules on cable joints.",
    defaults={"iterations": 5},
    # Millimetre-scale self-contact, on: Newton's own defaults are 0.2 m radii
    # sized for its centimetre-scale demos, and a sheet that does not collide
    # with itself is not a sheet that folds.
    particle_defaults={
        "particle_enable_self_contact": True,
        "particle_self_contact_radius": 0.002,
        "particle_self_contact_margin": 0.002,
        "particle_topological_contact_filter_threshold": 2,
        "particle_rest_shape_contact_exclusion_radius": 0.0,
        "particle_vertex_contact_buffer_size": 32,
        "particle_edge_contact_buffer_size": 64,
    },
)
XPBD = register_backend(
    "xpbd",
    cls="newton.solvers.SolverXPBD",
    supports=("bodies", "joints", "particles", "shapes"),
    maximal_coordinates=True,
    summary="Extended position-based dynamics.",
)
FEATHERSTONE = register_backend(
    "featherstone",
    cls="newton.solvers.SolverFeatherstone",
    supports=("bodies", "joints", "shapes"),
    summary="Featherstone articulated bodies.",
)
KINEMATIC = register_backend(
    "kinematic",
    cls="robosimstudio.scene.physics.kinematic.KinematicSolver",
    supports=("bodies", "joints", "shapes"),
    summary="Bodies that follow their joint targets exactly: a robot as a moving boundary.",
)
MPM = register_backend(
    "mpm",
    cls="robosimstudio.engines.newton.mpm.SolverMPM",
    supports=("particles",),
    # Its colliders are the shapes of its view, rasterised onto its own grid:
    # it reads no contact buffer, so the composer routes it no pairs and the
    # proxy coupler builds it no pipeline (which would otherwise enumerate a
    # soft pair per particle per shape).
    native_contacts=True,
    steps_in_place=True,
    schema=MPM_SCHEMA,
    summary="Implicit MPM: granular, liquid and plastic materials as particles on a background grid.",
    # The Phase 0 probes' choices, each measured in robosimstudio.engines.newton.mpm's docstring:
    # 4 mm voxels, a capturable sparse grid sized for a tabletop heap with
    # spread headroom, 50 iterations at 1e-4, the thin-tool collider
    # configuration, and one MPM step per 60 Hz control step -- the rate at
    # which a 29k bed with the YAM steps in 15.6 ms; two steps cost 25.9 ms,
    # which is over the 25 ms the GENERIC scene was budgeted.
    defaults={
        "voxel_size": 0.004,
        "grid_type": "sparse",
        "grid_padding": 0,
        "max_active_cell_count": 1 << 15,
        "max_lower_node_count": 4096,
        "max_upper_node_count": 256,
        "max_iterations": 50,
        "tolerance": 1.0e-4,
        "strain_basis": "P0",
        "collider_basis": "S2",
        "collider_velocity_mode": "forward",
        "project_outside": True,
        "step_hz": 60.0,
    },
)

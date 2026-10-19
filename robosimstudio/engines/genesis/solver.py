# SPDX-License-Identifier: Apache-2.0
"""The ``genesis`` backend class. One method, because the contract is shared.

The one method picks the bridge: an island of cloth and soft bodies runs Genesis's
IPC coupler (:class:`~robosimstudio.engines.genesis.bridge.GenesisBridge`), an island
of liquids its legacy coupler (:class:`~robosimstudio.engines.genesis.sph.SphBridge`),
and a Genesis scene holds one coupler, so an island holding both is refused
(:func:`~robosimstudio.engines.genesis.bridge.coupler_for`).
"""

from __future__ import annotations

from typing import Any

from robosimstudio.engines.host import SolverHost

__all__ = ["SolverGenesis"]


class SolverGenesis(SolverHost):
    """The host island's solver for Genesis: opens the bridge, inherits the rest.

    Everything a host engine owes the composer --- the no-op step contract, the
    wrench feed, the pipeline attach, the rest capture, the reset, the tick ---
    is :class:`~robosimstudio.engines.host.SolverHost`'s. This class is the one
    thing that is Genesis's: how to build its bridge.

    Args:
        model: The finalized Newton model.
        recipe: The manifest's host-island record.
        substeps: Genesis steps per control step. A liquid island taking SPH
            substeps longer than one of its liquids holds is refused, naming the
            substeps it needs (:func:`~robosimstudio.engines.genesis.sph.longest_substep`):
            20 at 60 Hz for the ``water`` preset at 4 mm.
        precision: ``"64"``, and changing it is a correctness decision rather
            than a performance one for the IPC coupler --- see
            :mod:`robosimstudio.engines.genesis.runtime`. A liquid island reads
            its reaction as a momentum change and states ``"32"``, which is 8x
            cheaper there (:mod:`robosimstudio.engines.genesis.sph`); one that
            initialises Genesis at 64 is warned.
        feedback: ``"auto"`` or ``"off"`` --- the IPC constraint-reaction harvest
            for two-way mirrors.
        feedback_alpha: EMA coefficient for a *rising* harvested force; 1.0
            disables the ramp. See
            :class:`~robosimstudio.engines.feedback.WrenchAccumulator`.
        feedback_dv_max: The most velocity [m/s] one control step's harvested
            wrench may add to a body; scales the whole wrench, torque with it.
        pose_source: ``"fresh"`` or ``"snapshot"``.
        shell_young: Cloth elastic modulus [Pa], island-wide. A cloth spec speaks
            its Newton solver's stiffnesses, not Young and Poisson, so these ride
            the backend rather than the entity.
        shell_poisson: Cloth Poisson ratio, island-wide.
        bending_stiffness: Genesis's own cloth bending term. It has no Newton
            equivalent at all, which is exactly why it belongs on the arrangement.
        contact_d_hat: IPC barrier distance [m].
        constraint_strength_translation: Stiffness of the soft transform
            constraint holding a mirror to its aim. Note the reaction is
            ``(k/dt^2) * m * dx`` --- raising this shrinks the displacement that
            encodes the force, which is harmless at float64 and destroys the
            measurement at float32.
        constraint_strength_rotation: The rotational twin.
        pressure_solver: A liquid island's SPH pressure solve, ``"WCSPH"``. Genesis's
            ``"DFSPH"`` is accepted and is not what the liquid kinds measured: it
            held a resting cup's water at 1.96 times its weight with 2 ms substeps
            and 1.97 with 0.4 ms ones, which cost 171 ms a control step
            (``p1_rest.py``).
        sph_reach: How far [m] the SPH solver's box reaches past the liquids' rest
            bounds, sideways and up.
        sph_reach_down: And down: where water that falls off the bench comes to rest.
        wall_friction: Genesis's ``coup_friction`` for every mirror a liquid touches
            (its own default). A Newton shape's friction is a rigid-rigid
            coefficient, and a cup gripped through rubber pads is not a wall the
            water clings to.
    """

    def open(
        self,
        model: Any,
        recipe: dict,
        *,
        substeps: int = 1,
        precision: str = "64",
        feedback: str = "auto",
        feedback_alpha: float = 0.35,
        feedback_dv_max: float = 0.5,
        pose_source: str = "fresh",
        shell_young: float = 1.0e5,
        shell_poisson: float = 0.3,
        bending_stiffness: float = 10.0,
        contact_d_hat: float = 1.0e-3,
        constraint_strength_translation: float = 100.0,
        constraint_strength_rotation: float = 100.0,
        pressure_solver: str = "WCSPH",
        sph_reach: float = 0.5,
        sph_reach_down: float = 0.25,
        wall_friction: float = 0.1,
    ) -> Any:
        """The Genesis bridge, with the island-wide scalars folded into the recipe.

        They ride the recipe rather than arguments for the same reason SuperDex's
        do: the manifest records the recipe, so a twin rebuilds the same scene
        from it and the model alone.
        """
        from .bridge import GenesisBridge, coupler_for

        shared = {**recipe, "feedback_alpha": float(feedback_alpha), "feedback_dv_max": float(feedback_dv_max)}
        options = {
            "substeps": int(substeps),
            "precision": str(precision),
            "feedback": str(feedback),
            "pose_source": str(pose_source),
        }
        if coupler_for(recipe) == "legacy":
            from .sph import SphBridge

            liquid = {
                "pressure_solver": str(pressure_solver),
                "sph_reach": float(sph_reach),
                "sph_reach_down": float(sph_reach_down),
                "wall_friction": float(wall_friction),
            }
            return SphBridge(model, {**shared, **liquid}, **options)
        cloth = {
            "shell_young": float(shell_young),
            "shell_poisson": float(shell_poisson),
            "bending_stiffness": float(bending_stiffness),
            "contact_d_hat": float(contact_d_hat),
            "constraint_strength_translation": float(constraint_strength_translation),
            "constraint_strength_rotation": float(constraint_strength_rotation),
        }
        return GenesisBridge(model, {**shared, **cloth}, **options)

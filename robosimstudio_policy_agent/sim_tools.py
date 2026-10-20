# SPDX-License-Identifier: Apache-2.0
"""Privileged ``sim.*`` tools: ground truth, for the in-process mode only.

These read the simulator directly -- object poses, grasp boxes, particle state,
whether the evaluator says the task is done. A graph that calls them is being
*co-developed* or generating data, not evaluated: it is being handed answers a
real perception stack would have to earn.

That is a legitimate and useful mode, and it is why the privilege boundary is
structural rather than a convention. These tools exist only in the in-process
executor. A policy served over a socket is to receive only camera frames,
proprioception and intrinsics, and
:func:`~robosimstudio_policy_agent.contract.assert_non_privileged` is the check a
bridge sender runs to refuse anything else (none ships in this package) -- so a graph developed against ``sim.get_object_pose`` simply
cannot run over the bridge until it has been rewritten to look.

Every tool reads the simulator on the owner thread: off the
:class:`~robosimstudio.core.bus.SimSnapshot` it published where that carries
the answer (poses, frames), which costs nothing, and otherwise through
``bus.call``, so the read runs between steps rather than racing one from the
graph thread. A tool that marshals waits for the owner's next pump; none of
them reads live simulator state from the graph worker.

No tool takes a parameter called ``name``: the registry dispatches through
``ToolRegistry.invoke(name, ctx, **kwargs)``, so a tool with its own ``name``
argument raises "multiple values for argument 'name'" the first time a graph
calls it. Hence ``object_name`` / ``region_name``, which is also gap's own
house style (``robot.get_camera_pose(camera_name)``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.core.bus import Bus
from robosimstudio.core.math import quat_to_mat, transform_matrix
from robosimstudio.core.render.visual import shape_meshes
from robosimstudio.core.streams import reading
from robosimstudio.robots.embodiments import top_down_grasp_site_quat
from robosimstudio.sim.sim import scene_deformables
from robosimstudio.solvers import grasp_from_obb, top_down_grasp_quat
from robosimstudio_bench.goals.goal import about_key, leaves, placements

from .refusal import ToolRefusal
from .types import quat_wxyz, se3_of_xform, vec3

__all__ = ["GoalCursor", "Vacuum", "register_goal_cursor", "register_sim_tools", "register_vacuum"]


def _grasp_heading_deg(long_axis_yaw: float) -> float:
    """The heading a top-down grasp closes along, given the long axis's yaw.

    Square to it, always. Separate from ``grasp_yaw`` because the two are one
    step apart and the step is silent: ``robot.grasp_frame`` takes a
    ``close_heading_deg``, ``grasp_yaw`` becomes degrees with one conversion,
    and handing one to the other closes the jaws down the length of the object
    without erroring anywhere. Measured on the hammer grid -- holding the roll
    at ``grasp_yaw`` scores 0/12, holding it here scores 8/12.
    """
    return float(np.degrees(long_axis_yaw) + 90.0)


def register_sim_tools(registry, env, bus: Bus) -> None:
    """Register the generic privileged tools on a GaP tool registry.

    A suite contributes its own task tools separately -- the tape suite
    registers ``sim.get_tab`` beside its adhesion module -- so nothing here needs
    to know what task is running.

    Args:
        registry: A ``gap_core.tools.ToolRegistry``.
        env: The toolkit-layer environment.
        bus: The bus whose snapshot the tools read.
    """
    tools = SimTools(env, bus)
    for name, fn, summary in tools.entries():
        # register_callable is the only path allowed to claim the reserved
        # ``sim.``/``robot.`` prefixes, and it raises on a duplicate name rather
        # than letting a second registration win silently.
        registry.register_callable(name, fn, summary=summary, tags=("privileged",))
    register_attachment(registry, env, bus)
    register_vacuum(registry, env, bus)


def register_attachment(registry, env, bus: Bus) -> ClothAttachment | None:
    """Register ``sim.attach``/``sim.detach``, if the scene has anything to hold.

    Conditional on the scene rather than opt-in per task, which is the opposite
    of :func:`register_goal_cursor` and for the opposite reason. A goal cursor
    over a peel would invent a pair that does not exist, so a task has to ask
    for it; but "does this cell contain a deformable" is a fact about the cell,
    and on a cell that has one there is nothing ambiguous about attaching to it.
    A rigid-only scene gets no tool, which is the honest state -- rather than a
    tool that raises the first time a graph believes the registry.

    Returns the attachment so a suite registering further tools can share it,
    or ``None`` when the scene has no deformables (:func:`_deformables`).
    """
    if not _deformables(env):
        return None
    attachment = ClothAttachment(env, bus)
    registry.register_callable(
        "sim.attach",
        attachment.attach,
        summary="Seize a deformable's declared feature at one hand and carry it.",
        tags=("privileged",),
    )
    registry.register_callable(
        "sim.detach",
        attachment.detach,
        summary="Let a held deformable go, at one hand or at every hand.",
        tags=("privileged",),
    )
    return attachment


def _deformables(env) -> dict[str, Any]:
    """The scene's deformables, by :func:`~robosimstudio.sim.sim.scene_deformables`: what ``Sim.deformables`` answers.

    Asked of the scene rather than of ``env``, which may be one world of a fleet (a
    ``WorldView``, with no ``deformables``). Every tool here that lists, defaults or
    looks one up goes through this, so registering ``sim.attach``, its default target
    and ``get_particle_state`` agree on what a deformable is.
    """
    return scene_deformables(getattr(env, "scene", None))


def _deformable(env, name: str) -> tuple[str, Any]:
    """The deformable called ``name``, or the first one for ``""``; refused when there is none.

    The refusal is ``get_rgbd``'s for a camera it does not have: the name asked for and the ones
    there are. A name off the list -- a cuttable, a capsule-chain rope, a typo -- raised a bare
    ``KeyError`` out of ``get_particle_state``, which a graph cannot route on.
    """
    deformables = _deformables(env)
    if not deformables:
        raise ToolRefusal("this scene has no deformables")
    name = name or next(iter(deformables))
    if name not in deformables:
        raise ToolRefusal(f"no deformable {name!r}; have {sorted(deformables)}")
    return name, deformables[name]


def _probe(entity, env) -> Any:
    """``entity``'s particle probe for the world ``env`` is: a fleet ``WorldView`` carries its ``world``, an ``Env`` is 0.

    ``entity.probe`` alone is world 0's, so a graph driving world 3 of a wave read world 0's
    particles back from ``get_particle_state`` -- a centroid, a height band, a footprint, all of
    the wrong towel, with nothing to say so. Each probe-per-world thing opens one per world
    (``ParticleMixin.bind``) and the fleet refreshes each one as it scores that world
    (``FleetEnv._particles``); a thing holding one probe is kept out of a wave by
    ``scoring_refusal``, and keeps its one here.
    """
    probe_for = getattr(entity, "probe_for", None)
    return probe_for(int(getattr(env, "world", 0) or 0)) if callable(probe_for) else entity.probe


def _model_state(env) -> Any:
    """The whole model's state, which a probe reads: a fleet ``WorldView``'s ``state`` is its world's slice of it.

    A probe holds every particle of the model it was opened on and addresses its world's by model
    row, as :func:`_first_particle_row` maps them, so it is refreshed from the fleet's own state.
    """
    fleet = getattr(env, "fleet", None)
    return fleet.pipeline.state_0 if fleet is not None else env.state


def _particle_rows(env) -> slice | None:
    """The model rows of the world ``env`` is, on a fleet's ``WorldView``; ``None`` (every row) for an ``Env``.

    What a probe of that world reads (:func:`_probe`): its sets are in its world's block of the
    fleet's model, which holds every world's particles.
    """
    fleet = getattr(env, "fleet", None)
    return fleet.worlds.particles(int(getattr(env, "world", 0) or 0)) if fleet is not None else None


def _first_particle_row(env) -> int:
    """The model row of particle 0 of the world ``env`` is: 0 for an ``Env``, world ``w``'s first on a fleet's ``WorldView``.

    The world is :func:`_probe`'s, the view's ``world``. A view's ``state.particle_q`` is its own
    world's slice, indexed like the template, so a deformable's template indices read that world's
    particles out of it; a pin runs on the fleet's whole state, where the same indices are world
    0's rows. ``sim.attach`` on world 1 of a two-world granular wave seized eight of world 0's rice
    grains and moved them 13 mm toward world 1's hand in one step.
    """
    world = int(getattr(env, "world", 0) or 0)
    return env.fleet.worlds.particle(0, world) if world else 0


def register_vacuum(registry, env, bus: Bus) -> Vacuum | None:
    """Register ``sim.vacuum``, if the robot carries a suction hand.

    Conditional on the robot for the reason :func:`register_attachment` is
    conditional on the cell: "does this robot have a cup" is a fact about the
    hardware, and a robot with jaws only gets no tool rather than one that
    raises the first time a graph believes it.

    Returns the tool, or ``None`` when no arm has a cup.
    """
    embodiment = getattr(getattr(env, "scene", None), "embodiment", None)
    if not getattr(embodiment, "suctions", None):
        return None
    tool = Vacuum(env, bus)
    registry.register_callable(
        "sim.vacuum",
        tool.vacuum,
        summary="Switch a suction hand's vacuum on or off, and say whether it sealed and on what.",
        tags=("privileged",),
    )
    return tool


def register_goal_cursor(registry, env, bus: Bus) -> GoalCursor:
    """Register ``sim.next_goal`` and ``sim.skip_goal`` on a GaP tool registry.

    **Not part of :func:`register_sim_tools`, and that is the whole design.** The
    generic set answers questions about *objects* and *regions*, which every
    scene has; "which of this problem's goals should I do next" is a question
    about a goal tree, so a task that wants it contributes it through
    :attr:`~robosimstudio.specs.TaskSpec.tools` -- the same opt-in the
    tape suite's bond tools take. A task whose goal is a peel or a wrap has no
    placements at all, and a cursor registered over them would hand a graph an
    invented pair.

    It is *here* rather than in a suite because it reads nothing but the
    compiled goal and the evaluator's metrics, which is exactly what the rest of
    this module reads.

    Args:
        registry: A ``gap_core.tools.ToolRegistry``.
        env: The toolkit-layer environment.
        bus: The bus whose snapshot carries the evaluator's metrics.

    Returns:
        The cursor, so a suite registering further tools of its own can share
        its notion of which goal is current.
    """
    cursor = GoalCursor(env, bus)
    registry.register_callable(
        "sim.next_goal",
        cursor.next_goal,
        summary="The next unsatisfied (object, region) goal of this variant, or done.",
        tags=("privileged",),
    )
    registry.register_callable(
        "sim.skip_goal",
        cursor.skip_goal,
        summary="Give up on one goal for the rest of the episode and move to the next.",
        tags=("privileged",),
    )
    return cursor


class SimTools:
    """The generic privileged tool implementations."""

    def __init__(self, env, bus: Bus) -> None:
        self.env = env
        self.bus = bus
        # The home end-effector orientation, from the embodiment's *declared*
        # home rather than the live arm: constructed on the main thread before
        # the graph starts, and constant thereafter -- the live arm is only at
        # home before the first goal of a loop.
        self._home_ee_quat = self._compute_home_ee_quat()

    def _compute_home_ee_quat(self):
        embodiment = self.env.scene.embodiment
        handles = self.env.handles
        if embodiment is None or not embodiment.spec.arms or not handles.arms:
            return None
        # The work arm: this quaternion is the frame every commanded grasp is
        # composed about, so reading it off whichever arm happens to be first
        # yaws every grasp by the idle hand's wrist.
        name = handles.work_arm if handles.work_arm in handles.arms else next(iter(handles.arms))
        arm_spec = next((a for a in embodiment.spec.arms if a.name == name), embodiment.spec.arms[0])
        ik = self.env.arm(name)
        home = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float32).copy()
        if arm_spec.home_q:
            # strict: a home posture that does not match the arm's coordinate
            # count is a spec error, and zip's default would silently seed only
            # the joints the shorter one covers.
            for coord, value in zip(ik.arm_coords, arm_spec.home_q, strict=True):
                home[coord] = value
        _, quat_xyzw = ik.fk_ee(home)
        return np.asarray(quat_xyzw, dtype=np.float64)

    def _grasp_quat(self, yaw: float):
        """A top-down grasp at ``yaw``, in the grasp-site frame, as a wxyz dict.

        Built from the hand's **measured** approach and closing axes rather than
        by yawing its home wrist about world +z. The heuristic this replaces
        reads neither axis, which is fine for as long as every robot's hand
        happens to sit square to its wrist and silently wrong otherwise -- the
        Panda's measured closing axis is local *x*, and a grasp frame composed on
        the default local *y* closes the jaws along the object rather than
        across it.

        The ``[-90, 90)`` wrap is kept. Without it the last wrist joint runs into
        its limit on grasps near a right angle, which is measured elsewhere in
        this package as 40.9 degrees of residual roll against a hard stop.
        """
        hand = self._work_hand()
        if hand is None:
            if self._home_ee_quat is None:
                return None
            q = top_down_grasp_quat(self._home_ee_quat, yaw)
            return {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])}
        wrapped = (float(yaw) + np.pi / 2.0) % np.pi - np.pi / 2.0
        q = top_down_grasp_site_quat(hand, float(np.degrees(wrapped)))
        return {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])}

    def _work_hand(self):
        """The :class:`GripperSpec` on the work arm, or ``None`` for a bare cell."""
        embodiment = self.env.scene.embodiment
        if embodiment is None:
            return None
        handles = self.env.handles
        name = handles.work_arm if handles.work_arm in handles.arms else next(iter(handles.arms), "")
        return embodiment.hands.get(name)

    def entries(self) -> list[tuple[str, Any, str]]:
        return [
            ("sim.list_objects", self.list_objects, "Every manipulable object in the scene, with its category."),
            (
                "sim.get_object_pose",
                self.get_object_pose,
                "One object's ground-truth world pose (position + wxyz quaternion).",
            ),
            ("sim.get_object_obb", self.get_object_obb, "One object's oriented bounding box and the grasp it implies."),
            ("sim.list_regions", self.list_regions, "Every goal region, with its world centre."),
            (
                "sim.get_region_obb",
                self.get_region_obb,
                "One goal region as an oriented bounding box, for aiming a place.",
            ),
            (
                "sim.get_particle_state",
                self.get_particle_state,
                "A deformable's summary: centroid, height band, footprint.",
            ),
            ("sim.get_rgbd", self.get_rgbd, "The most recent camera frame: rgb, depth and intrinsics."),
            (
                "sim.get_collision_world",
                self.get_collision_world,
                "Exact simulator collider triangles, excluding the robot and selected object categories.",
            ),
            (
                "sim.get_object_bounds",
                self.get_object_bounds,
                "One object's world axis-aligned bounds, as it lies now.",
            ),
            ("sim.check_grasp", self.check_grasp, "Whether an arm is actually holding one object right now."),
            (
                "sim.get_states",
                self.get_states,
                "Every state an entity publishes, with how observable each one is.",
            ),
            ("sim.check_success", self.check_success, "What the task's evaluator says right now."),
            (
                "sim.query",
                self.query_tool,
                "Ask one relation, or a whole family of them, about named things right now.",
            ),
            (
                "sim.list_relations",
                self.list_relations_tool,
                "What can be asked about these things: the applicable relations, by family.",
            ),
            (
                "sim.clearance",
                self.clearance,
                "How close two things are [m] -- now, or at a joint configuration the arm is not in.",
            ),
        ]

    # ------------------------------------------------------------------ tools
    def list_objects(self) -> dict[str, Any]:
        handles = self.env.handles
        return {
            "objects": [
                {"name": name, "category": handle.category, "static": bool(handle.static)}
                for name, handle in sorted(handles.objects.items())
            ]
        }

    def get_object_pose(self, object_name: str) -> dict[str, Any]:
        body_q = self._body_q()
        handle = self.env.handles.entity(object_name)
        if getattr(handle, "body", -1) < 0:
            # A bodiless static indexes body_q at -1, which is the LAST body's
            # pose -- a silent lie discovered by the tool_hanging cell, where a
            # policy asked a pegboard fixture where it was and got the second
            # wrench's live pose back with an ok status. Refusing is the
            # honest answer; the fixture's geometry is reachable through the
            # region and container tools that know about welded scenery, and
            # through ``sim.query("geometry", ...)`` for a declared feature.
            raise ToolRefusal(f"{object_name!r} is a bodiless static; it has no body pose to read")
        return {"name": object_name, "pose": se3_of_xform(handle.pose(body_q))}

    def get_object_obb(self, object_name: str, handle_frac: float = 0.0) -> dict[str, Any]:
        """The grasp box, in the shape the shared geometry tools consume.

        The returned ``obb`` is a ``gap_core.types.OrientedBoundingBox``, so it
        drops straight into ``geometry.top_down_grasp_candidates`` where a
        perception-driven graph would put the output of a point-cloud fit. That
        substitutability is the point: a ground-truth graph and a perception
        graph differ in one node, not in their structure.

        ``extent`` is a **half**-extent, which is gap's convention and also what
        :class:`~robosimstudio.solvers.ObbGrasp` stores. The box is the
        body-frame AABB, so its world orientation is the body's own.

        Two boxes come back, and which one to plan against is a real choice:

        ``obb``
            The whole object. Correct for anything a gripper can close on at its
            centroid, and what a point-cloud fit would produce.
        ``grasp_obb``
            The **graspable part**, slid ``handle_frac`` along the long axis
            towards the light end and shortened to a segment the fingers span.
            An object wider than the gripper at its centroid -- a claw hammer is
            13.2 cm across against an 8 cm span -- cannot be grasped there at
            all, and planning against the whole box makes the fingers straddle
            the claw and catch it by friction: it holds sometimes, and drops
            mid-transport the rest of the time. This is what a grasp-proposal
            network would emit, so a graph that plans against it is not doing
            anything a perception graph could not.

        Two headings come back and they are perpendicular. ``grasp_yaw`` is the
        **long axis's** own heading; ``grasp_heading_deg`` is what a top-down
        grasp should close **along**, and is the one to hand to
        ``robot.grasp_frame(close_heading_deg=...)``. Passing ``grasp_yaw``
        there instead closes the jaws down the length of the object -- it
        validates and executes, so the only symptom is a grasp that lands on
        target and holds nothing.

        ``fits_gripper`` says which is needed; ``handle_frac`` defaults to 0 so
        the whole box is unchanged unless a caller asks.

        **A declared ``grasp_box`` is not consulted here, and that is a gap
        rather than a decision.** This uses the derived box unconditionally.
        Latent today -- the VAB tools deliberately declare none, since a rigid
        asset's grasp geometry is derived and re-declaring it would be
        transcribing a computation, and ``usb_c_plug`` declares one that no
        shipped graph asks for -- but an asset that declared a box precisely
        because the derived one is wrong would be ignored right where it
        mattered.

        Closing it is not a substitution. Everything below the box is
        ``ObbGrasp``'s: ``long_axis``, ``gripper_span``, ``grasp_width``,
        ``fits_gripper``, and the handle slide. A declared payload carries a half
        extent and a centre and no ``handle_sign`` -- which end is the light one
        is a fact about mass, not about the box -- so honouring a declaration
        means deciding what the handle slide even means for an asset that has
        already said where to close. That is a design question and wants asking
        rather than guessing.
        """
        handle = self.env.handles.entity(object_name)
        if handle.obb is None:
            raise ToolRefusal(f"object {object_name!r} has no grasp box")
        # By the body the box was fit on: a roller's is its handle, which stays upright while the drum it names rolls.
        pose = np.asarray(self._body_q()[handle.obb.body], dtype=np.float64)
        rotation = quat_to_mat(pose[3:7])
        centre = pose[:3] + rotation @ handle.obb.center_local
        point, yaw = grasp_from_obb(handle.obb, pose, handle_frac)
        orientation = quat_wxyz(pose[3:7])

        # The full commanded grasp orientation, not just the yaw angle: the
        # reference's own composition (yaw the *home* gripper so the fingers
        # cross the handle), computed here because only the environment knows
        # the home end-effector orientation -- and computed from the
        # embodiment's declared home rather than the live arm, because on the
        # second goal of a loop the arm is wherever the first goal left it,
        # and a "home" read there yaws the grasp by the leftover wrist angle.
        grasp_quat = self._grasp_quat(yaw)

        # The graspable segment: the long axis clipped to what the fingers can
        # straddle, the rest of the box unchanged so the planner still derives a
        # correct top surface to descend to.
        span = float(handle.obb.gripper_span)
        grasp_half = np.asarray(handle.obb.half, dtype=np.float64).copy()
        grasp_half[handle.obb.long_axis] = min(grasp_half[handle.obb.long_axis], span * 0.5)

        bounds = self.get_object_bounds(object_name)
        return {
            "name": object_name,
            "obb": {"center": vec3(centre), "extent": vec3(handle.obb.half), "orientation": orientation},
            "grasp_obb": {"center": vec3(point), "extent": vec3(grasp_half), "orientation": orientation},
            "grasp_position": vec3(point),
            "grasp_yaw": float(yaw),
            # The heading a top-down grasp should close ALONG, which is square
            # to the long axis -- and therefore square to `grasp_yaw`, which is
            # the long axis's own heading.
            #
            # Both are here because the two are a step apart and the step is
            # easy to miss: `robot.grasp_frame` takes a `close_heading_deg`,
            # `grasp_yaw` is in degrees once converted, and handing one to the
            # other type-checks, validates, plans, executes, and closes the
            # jaws down the length of the object. Measured on `tool_sort`'s
            # hammer: handle at 175 deg, `grasp_yaw` -5 deg -- the same line.
            #
            # The cost of the missing step is not subtle. Holding the roll at
            # `grasp_yaw` scores 0/12 on the hammer grid; holding it here
            # scores 8/12, and is the only configuration ever to pick the
            # hammer up at the second staged yaw at all. The shipped `gt`
            # graph's 5/12 comes entirely from the one yaw where a free wrist
            # happens to land square by accident.
            "grasp_heading_deg": _grasp_heading_deg(yaw),
            # The full commanded orientation the yaw implies -- see _grasp_quat.
            # NOTE: composed from `grasp_yaw`, so it inherits the same step:
            # it is identical to `robot.grasp_frame(close_heading_deg=
            # degrees(grasp_yaw))`, measured 0.0 deg apart. Kept as it is
            # because graphs depend on it; prefer `grasp_heading_deg`.
            "grasp_quat": grasp_quat,
            "grasp_width": float(handle.obb.grasp_width),
            "fits_gripper": bool(handle.obb.fits_gripper),
            "long_axis": int(handle.obb.long_axis),
            # Where the object actually is in z, as it lies: what a caller
            # needs to decide how deep the fingers may go before they bottom
            # out on whatever the object is resting on.
            "top_z": float(bounds["highest_z"]),
            "bottom_z": float(bounds["lowest_z"]),
        }

    def get_region_obb(self, region_name: str, height: float = 0.05) -> dict[str, Any]:
        """A goal region as an OBB, so a transport node can aim at it.

        Regions are axis-aligned rectangles in a frame, which is a box with an
        identity rotation -- expressing them this way lets the same drop-pose
        script serve a region and a physical container.

        The box sits on the region's own floor, ``z_range[0]`` above the frame
        origin, not on the frame origin itself. For a bin compartment the two
        coincide -- the window opens at the bin floor -- which is why this went
        unnoticed; for a region on a container they do not, because the
        container's cavity floor is offset from ``bin2``'s origin. Aiming a
        drop at the frame origin instead of the cavity floor put the release
        16 mm out on the syringe box.
        """
        handle = self.env.handles.regions[region_name]
        body_q = self._body_q()
        origin = handle.origin_now(body_q)
        x0, y0, x1, y1 = handle.xy_ranges
        centre = origin + np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, handle.z_range[0] + height * 0.5])
        return {
            "name": region_name,
            "obb": {
                "center": vec3(centre),
                "extent": vec3([abs(x1 - x0) * 0.5, abs(y1 - y0) * 0.5, height * 0.5]),
                "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            },
        }

    def list_regions(self) -> dict[str, Any]:
        body_q = self._body_q()
        return {
            "regions": [
                {"name": name, "frame": handle.frame, "center": vec3(handle.center(body_q))}
                for name, handle in sorted(self.env.handles.regions.items())
            ]
        }

    def get_particle_state(self, cloth: str = "", particle_set: str = "") -> dict[str, Any]:
        name, entity = _deformable(self.env, cloth)
        probe = _probe(entity, self.env)
        key = f"{name}/{particle_set}" if particle_set else name

        def read() -> dict[str, Any]:
            # Refreshed from the state as it stands, every call: nothing refreshes a probe for a
            # tool. A fleet refreshes world k's only as it scores that world, every eval_every
            # steps, and a bare Sim's holds what snapshot_rest left, so the summary alone was the
            # sheet where it lay at the last score. Only this world's rows, at this step, so that
            # neither a fleet nor an env reads them again until it steps.
            step = getattr(self.env, "step_count", None)
            with reading():
                probe.update(_model_state(self.env), step, rows=_particle_rows(self.env))
            return probe.summary(key)

        return {"cloth": name, "set": key, **self.bus.call(read)}

    def get_rgbd(self, camera: str = "") -> dict[str, Any]:
        frames = self.bus.snapshot.frames
        if not frames:
            raise ToolRefusal("no camera frames have been rendered; build the env with a renderer")
        name = camera or next(iter(frames))
        if name not in frames:
            raise ToolRefusal(f"no camera {name!r}; have {sorted(frames)}")
        spec = next((c for c in self.env.cameras if c.name == name), None)
        frame = frames[name]
        return {
            "camera": name,
            "rgb": frame.get("rgb"),
            "depth": frame.get("depth"),
            "intrinsics": None if spec is None else spec.intrinsics.tolist(),
        }

    def get_collision_world(self, exclude_categories: list[str] | None = None) -> dict[str, Any]:
        """Export the simulator's authored collision shapes as a CuRobo world.

        This is deliberately privileged and diagnostic-only.  Unlike the RGB-D
        path it performs no alpha-shape reconstruction: every returned triangle
        comes from the exact Newton collider used by physics.  Robot shapes and
        the requested manipulated-object categories are omitted so they are not
        duplicated as static world obstacles.
        """
        import newton  # local: simulator-only dependency

        excluded_categories = {str(value).lower() for value in (exclude_categories or [])}
        embodiment = self.env.scene.embodiment
        excluded_bodies = {int(body) for body in getattr(embodiment, "bodies", ())}
        excluded_shapes = set()
        if embodiment is not None:
            excluded_shapes.update(range(*embodiment.range.shapes))
        for object_name, handle in self.env.handles.objects.items():
            # `.category`, not `.kind`: the field was renamed and this call
            # site was not, so every call to sim.get_collision_world raised
            # AttributeError -- on a tool the `full` tier admits, which means
            # an agent reaching for the collision world lost its episode to a
            # crash rather than a refusal. Found by the surface smoke.
            identity = {str(object_name).lower(), str(handle.category).lower()}
            # A bodiless handle (body -1) has nothing to exclude, and -1 in the
            # set would drop every world-static shape below.
            if int(handle.body) >= 0 and any(
                token in value or value in token for token in excluded_categories for value in identity
            ):
                excluded_bodies.add(int(handle.body))

        model = self.env.scene.index_model
        body_q = self._body_q()
        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        labels = list(getattr(model, "shape_label", None) or ())
        meshes = []
        for shape in shape_meshes(model, skip_planes=True, flag=collide):
            if shape.index in excluded_shapes or shape.body in excluded_bodies:
                continue
            world = shape.local if shape.body < 0 else transform_matrix(body_q[shape.body]) @ shape.local
            vertices = shape.vertices @ world[:3, :3].T + world[:3, 3]
            meshes.append(
                {
                    "name": (f"shape_{shape.index}_{labels[shape.index] if shape.index < len(labels) else 'collider'}"),
                    "vertices": vertices.astype(np.float32),
                    "faces": shape.faces.astype(np.int32),
                    "pose": {
                        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    },
                }
            )
        if not meshes:
            raise ValueError("simulator collision export produced no static collider meshes")
        return {"config": {"meshes": meshes}, "mesh_names": [mesh["name"] for mesh in meshes]}

    def get_object_bounds(self, object_name: str) -> dict[str, Any]:
        """One object's world axis-aligned bounds, in its *current* orientation.

        The port of the reference's ``tool_lowest_z``, and it exists for one
        job: posting a held tool through a hole. How far the tool hangs below
        the gripper once it has been stood on end cannot be predicted from the
        grasp -- it settles differently in the fingers depending on where along
        it they closed -- and the reference is explicit that a centimetre of
        that error is the difference between dropping cleanly through an
        aperture and catching its rim. So it is measured live.

        Derived by transforming the body-frame box's eight corners and taking
        the extremes, which is the world AABB of the oriented box rather than of
        the mesh. For a tool held on end that is what the caller wants: the
        lowest point of the thing hanging out of the hand.
        """
        handle = self.env.handles.entity(object_name)
        if handle.obb is None:
            raise ToolRefusal(f"object {object_name!r} has no bounding box")
        pose = np.asarray(self._body_q()[handle.obb.body], dtype=np.float64)  # the box's own body, as above
        rotation = quat_to_mat(pose[3:7])
        half = np.asarray(handle.obb.half, dtype=np.float64)
        centre = np.asarray(pose[:3], dtype=np.float64) + rotation @ np.asarray(
            handle.obb.center_local, dtype=np.float64
        )
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
        corners = centre + (signs * half) @ rotation.T
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        return {
            "name": object_name,
            "min": vec3(lo),
            "max": vec3(hi),
            "lowest_z": float(lo[2]),
            "highest_z": float(hi[2]),
            # Which world axis the object is longest along, so a caller can tell
            # whether it is currently lying flat or standing on end.
            "long_axis": int(np.argmax(hi - lo)),
        }

    def check_grasp(self, object_name: str, arm: str = "", max_angle: float = 80.0) -> dict[str, Any]:
        """Whether the hand is actually holding ``object_name``.

        A ``robot.close_gripper`` that returns tells you the fingers finished
        travelling, not that anything is between them -- so without this a graph
        closes on air, routes to its transport branch, and carries nothing to
        the goal. The whole episode then reads as a clean end-to-end run scoring
        zero, which is the least diagnosable failure a benchmark can produce.

        The test is :meth:`~robosimstudio.core.probes.ContactProbe.check_grasp`,
        the same one the placement predicates and the SimplerEnv evaluators
        use: a contact impulse on *each* side of the hand whose direction is
        within ``max_angle`` of the direction that side opens in. The angle
        test is what stops an object resting on top of a closed gripper from
        reading as held.

        Returns ``held`` false rather than raising when the scene has no contact
        probe or the arm declared no finger bodies, since a task may legitimately
        run without either. An ``arm`` string that names no arm is different:
        that is a caller error, and it refuses rather than answers.
        """
        arm_names = self.env.handles.arms
        if arm and arm not in arm_names:
            # ``.get(arm)`` used to swallow this and answer a confident
            # held=False -- the tool_hanging cells burned their final
            # iterations guessing role words ("work", "support") against it,
            # and a silent False can never teach the right vocabulary.
            raise ToolRefusal(
                f"no arm named {arm!r}; this cell's arms are {sorted(arm_names)} "
                f"(robot.describe_arm's 'arm' field is the string this parameter takes)"
            )

        def read() -> bool:
            contacts = getattr(self.env, "contacts", None)
            if contacts is None:
                return False
            arms = self.env.handles.arms
            # The *work* arm when none is named, not the first one declared. The
            # two YAM cells list their arms in opposite orders, so declaration
            # order picks the idle hand on one of them -- and an unqualified
            # check_grasp then reports the working hand holding nothing while it
            # is holding the object.
            handle = arms.get(arm) if arm else arms.get(self.env.handles.work_arm)
            if handle is None or not handle.finger_groups:
                return False
            body_q = np.asarray(self.env.state.body_q.numpy(), dtype=np.float64)
            return bool(
                contacts.check_grasp(
                    handle.finger_groups,
                    self.env.handles.entity(object_name).collision_shapes,
                    body_q,
                    open_axis_signs=handle.open_axis_signs,
                    max_angle=max_angle,
                )
            )

        return {"object_name": object_name, "held": bool(self.bus.call(read))}

    def check_success(self) -> dict[str, Any]:
        info = self.bus.snapshot.info
        return {
            "success": bool(info.get("success", False)),
            "reward": float(info.get("reward", 0.0)),
            "metrics": info.get("metrics", {}),
        }

    # ------------------------------------------------------------- describing
    def get_states(self, entity: str = "") -> dict[str, Any]:
        """Every state an entity publishes, tagged with how observable each is.

        The generic form of a tool four suites wrote by hand. An asset declares
        what it publishes -- ``tape`` and ``film`` declare ten each, with a
        summary, a unit and an observability tier per entry -- and the entity
        computes them. Between those two facts there is nothing left for a
        per-suite tool to decide, so asking the declaration serves every asset
        that has one, including the ones whose suite never wrote a tool.

        ``film_peel`` is why this is worth having: it registers ``tape_peel``'s
        tools unchanged, so a graph peeling *film* could read the tape's four
        hand-derived numbers and none of the film's own ten.

        Args:
            entity: One entity, or ``""`` for every entity that publishes
                anything.

        Returns:
            ``{entity: {state: {"value", "observability"} | value}}``. The tier
            rides along rather than being stripped: a bond fraction is
            ``internal``, no camera reads one, and a policy that learned from it
            has learned something no real robot can. Untiered values -- an
            entity publishing something its record does not declare -- come
            through bare rather than being dropped, because a number nobody
            declared is still a number somebody is about to depend on.
        """
        from robosimstudio.assetlib.registry import ASSETS
        from robosimstudio.assetlib.semantics.snapshot import Snapshot
        from robosimstudio_bench.goals.bind import _published

        doc = self._document()
        types = {o.name: o.type for o in doc.objects} if doc is not None else {}

        def read() -> dict[str, Any]:
            snapshot = Snapshot(ctx=self.env.make_context()).freeze()
            wanted = [entity] if entity else sorted(snapshot.entities)
            out: dict[str, Any] = {}
            for one in wanted:
                published = _published(snapshot, one)
                if not published:
                    continue
                # The tier table is read off the asset record directly. It
                # used to come from the narration layer's tier helpers, which
                # this checkout deleted -- the tiering is a fact the catalog
                # already declares, and reading it here keeps the value
                # structured rather than routing it through prose.
                try:
                    record = ASSETS.get(types[one]) if types.get(one) else None
                except KeyError:
                    record = None
                tiers = {p.name.replace("-", "_"): p.tier for p in (record.published if record else ())}
                values: dict[str, Any] = {}
                for key, value in sorted(published.items()):
                    item = float(value) if isinstance(value, (int, float, np.floating)) else value
                    tier = tiers.get(str(key).replace("-", "_"))
                    values[key] = {"value": item, "observability": tier} if tier else item
                out[one] = values
            return out

        return self.bus.call(read)

    @staticmethod
    def _refs(refs: Any) -> tuple[str, ...]:
        """A graph's ``refs`` argument as the tuple the query layer wants.

        ``query(atom, *refs)`` is the right Python signature and the wrong tool
        signature. A graph reaches a tool through ``ctx.tool(name, **kwargs)``,
        which passes keywords only, and a keyword cannot bind to ``*args`` --
        so ``sim.query`` was *uncallable from a workflow*, in either spelling:
        ``refs=[...]`` raised "unexpected keyword argument 'refs'" and the
        positional workaround raised "tool() takes 2 positional arguments".
        Both were observed, twelve episodes at a time.

        Nobody hit it because nobody needed it: the suites whose fixtures have
        no body -- a pegboard hook, a rack hole -- shipped their own
        ``sim.plan_*`` tools that handed the location over, and this is the
        generic path those tools were standing in front of. Withhold them and
        the only documented route to a welded fixture is a dead one.

        So the registered tool takes ``refs`` as a real parameter and accepts
        what an author will plausibly write: one ref, a list of them, or the
        several-in-a-string spelling ("wrench_1.ring, wrench_hook_1.shank")
        that a model reaches for when a list looks unwelcome.
        """
        if refs is None:
            return ()
        if isinstance(refs, str):
            parts = [part.strip() for part in refs.replace(",", " ").split()]
            return tuple(part for part in parts if part)
        if isinstance(refs, (list, tuple)):
            out: list[str] = []
            for item in refs:
                out.extend(SimTools._refs(item))
            return tuple(out)
        return (str(refs),)

    def query_tool(self, atom: str, refs: Any = ()) -> dict[str, Any]:
        """:meth:`query`, with ``refs`` as one keyword argument. See :meth:`_refs`."""
        return self.query(atom, *self._refs(refs))

    def list_relations_tool(self, refs: Any = ()) -> dict[str, Any]:
        """:meth:`list_relations`, with ``refs`` as one keyword argument."""
        return self.list_relations(*self._refs(refs))

    def query(self, atom: str, *refs: str) -> dict[str, Any]:
        """Ask the scene one question, in the vocabulary the goal is written in.

        Returns the handler's structured measurement without rendering it into
        language.

        Args:
            atom: A relation or property name (``"on"``, ``"covers"``,
                ``"winding_turns"``), or a family (``"SpatialRelation"``) to ask
                every relation that admits these operands.
            *refs: What to ask about, in the document's own reference grammar:
                ``"hammer_0"``, ``"port.bore"``, ``"cable.point(0.63)"``.

        Returns:
            The structured reading. For a family, one entry per applicable
            relation.

        **The same measurement the goal uses**, resolved the same way and
        dispatched to the same handler -- so an agent is never reasoning about a
        world slightly different from the one being scored. No threshold comes
        back: what counts as enough is the task's, and an agent told only
        somebody else's boolean has been told less than it asked.
        """
        from robosimstudio.assetlib.semantics.snapshot import Snapshot
        from robosimstudio_bench.goals.bind import BindError
        from robosimstudio_bench.goals.query import QueryRefused
        from robosimstudio_bench.goals.query import query as _query

        def ask() -> dict[str, Any]:
            snapshot = Snapshot(ctx=self.env.make_context())
            return _query(snapshot, atom, *refs, doc=self.env.doc)

        try:
            # On the owner thread: the handlers read live state and probes.
            return self.bus.call(ask)
        except (QueryRefused, BindError, LookupError, ValueError) as exc:
            # **A refusal is an answer, and it is returned rather than raised.**
            # "This thing does not afford that question" is a legitimate and
            # common reply during authoring -- it is the catalog-gap signal the
            # whole asset layer is built to give -- and a tool that raised it
            # would end the episode over an agent asking a reasonable question.
            # The message already names what *is* available, which is the half
            # worth having.
            return {"refused": str(exc)}

    def list_relations(self, *refs: str) -> dict[str, Any]:
        """What may be asked about these things, by family.

        The discovery half. An agent holding two names rarely knows which
        relation it wants; it wants to know which are answerable, and asking
        each and catching the refusals is a worse way to find out.
        """
        from robosimstudio.assetlib.semantics.atom import leaves_of
        from robosimstudio.assetlib.semantics.snapshot import Snapshot
        from robosimstudio_bench.goals.bind import BindError
        from robosimstudio_bench.goals.query import QueryRefused
        from robosimstudio_bench.goals.query import query as _query

        # ``env.doc``, where the document lives -- the same one :meth:`query`
        # resolves against; ``env.task`` carries none.
        doc = self.env.doc

        def ask() -> dict[str, list[dict[str, Any]]]:
            snapshot = Snapshot(ctx=self.env.make_context())
            out: dict[str, list[dict[str, Any]]] = {}
            for family in ("SpatialRelation", "Relation", "Property"):
                readings = None
                for leaf in leaves_of(family):
                    if len(leaf.args) != len(refs):
                        continue
                    if readings is None:
                        readings = _query(snapshot, family, *refs, doc=doc)["readings"]
                    if leaf.name in readings:
                        out.setdefault(family, []).append(
                            {"atom": leaf.name, "summary": leaf.summary, "returns": list(leaf.returns)}
                        )
            return out

        try:
            return {"families": self.bus.call(ask)}
        except (QueryRefused, BindError, LookupError, ValueError) as exc:
            # Returned, not raised, for the reason :meth:`query` gives.
            return {"refused": str(exc)}

    def clearance(self, group_a: str, group_b: str = "", joint_q: list[float] | None = None) -> dict[str, Any]:
        """How close two things are, now or at a configuration the arm is not in.

        **The motion-feedback question**, and the reason it takes ``joint_q``: an
        agent about to move an arm from here to there wants to know what the
        move would touch *before* making it, and forward kinematics into the
        probe's own state answers that without stepping the scene or disturbing
        anything.

        Args:
            group_a: ``"arm:work"``, ``"arm:left"``, or an object name.
            group_b: The same, or empty for everything else in the scene.
            joint_q: A joint configuration to ask about. Omitted asks about the
                arm where it is.

        Returns:
            ``gap`` [m] -- negative where the two overlap -- and which shapes
            made it. ``inf`` means the model has filtered the pair against each
            other, which is the honest answer for two parts that cannot collide.
        """
        probe = self.env.clearance
        first = self._shape_group(group_a)
        second = self._shape_group(group_b) if group_b else self._complement(first)
        # On the owner thread: the probe holds simulator state of its own, and
        # the live body poses are read where they are written.
        if joint_q is not None:
            gap = self.bus.call(probe.at, list(joint_q), first, second)
            return {"gap": float(gap), "at_configuration": True}
        gap, shape_a, shape_b = self.bus.call(lambda: probe.closest(first, second, self.env.state.body_q.numpy()))
        return {"gap": float(gap), "shape_a": int(shape_a), "shape_b": int(shape_b), "at_configuration": False}

    def _shape_group(self, name: str) -> list[int]:
        """Collision shapes for an arm role or an object name.

        ``"arm:work"`` and ``"arm:support"`` are *roles* rather than arm names,
        because a graph is written against the role its binding assigns and the
        arm behind it differs per robot. Resolved through the same role order the
        runtime's ``arm_id`` counts in, so a tool and a graph mean one thing.
        """
        if name.startswith("arm:"):
            from robosimstudio.arms import arm_name, arm_order

            role = name.partition(":")[2] or "work"
            order = arm_order(self.env)
            resolved = role if role in order else arm_name(self.env, 0 if role == "work" else 1)
            return [int(s) for s in self.env.handles.arms[resolved].shapes]
        return [int(s) for s in self.env.handles.entity(name).collision_shapes]

    def _complement(self, shapes: list[int]) -> list[int]:
        """Everything else in the scene.

        The default second group, and it is what "will this move hit anything"
        means when the asker does not yet know what it might hit.
        """
        every: list[int] = []
        for handle in self.env.handles.objects.values():
            every.extend(int(s) for s in handle.collision_shapes)
        for handle in self.env.handles.statics.values():
            every.extend(int(s) for s in handle.collision_shapes)
        held = set(shapes)
        return [s for s in every if s not in held]

    def _document(self):
        """The task's document, when it was defined by one.

        ``None`` for a task assembled some other way: its description degrades to
        positions and containments rather than failing, which is the honest
        reading -- there is no document to type its objects against.
        """
        return self.env.doc

    # ---------------------------------------------------------------- helpers
    def _body_q(self) -> np.ndarray:
        snapshot = self.bus.snapshot
        if snapshot.body_q is not None:
            return snapshot.body_q
        return self.bus.call(lambda: np.asarray(self.env.state.body_q.numpy(), dtype=np.float64))


class ClothAttachment:
    """Holding a deformable at a hand, and letting it go.

    Three suites wrote this: ``sim.latch_cloth``/``release_cloth``,
    ``sim.latch_tape``/``release_tape`` and ``sim.hold_cable``/``release_cable``,
    whose return shapes already agreed. What differed between them was only
    *which particles* -- and a declared set name says that, so there is nothing
    left for a per-suite copy to decide. The tape pair is gone and calls this.

    **This stays privileged and always will.** No real hand seizes particles;
    a graph that attaches is a graph that will not transfer, and no amount of
    perception makes it transfer. It is here rather than hidden in a suite so
    that fact is stated once, loudly, in the module whose docstring is about
    ground truth -- rather than three times, quietly, in files a portability
    scan does not read.

    Four behaviours are load-bearing, and each of the three copies paid for at
    least one of them:

    * **The graph must be invalidated after the hook is appended.** The substep
      block is captured into a CUDA graph and a hook added afterwards is simply
      not in it. Skipping this is *silent*: the grab bookkeeping is host-side,
      so the tool reports the particles it seized while nothing holds them. On
      the tape suite what little peeling then happened was the fingers dragging
      the film by friction -- 2-15% of bonds against the routine's 100%.
    * **The pin must be re-aimed every control step.** It holds particles at
      fixed offsets from a *frame*, and a graph only acts between tool calls, so
      without a per-step follow the edge stays pinned in mid-air where it was
      seized.
    * **Through the arm's grasp frame, never the wrist body.** The i2rt YAM's
      grasp site is offset from that body by a translation *and* a quarter turn;
      hand the pin the untwisted quaternion and the seized edge rides along
      rotated out of alignment with the hand carrying it.
    * **No nearest-particle fallback.** A latch that missed its declared set and
      took whatever was closest would seize the middle of a sheet, drag it into
      a heap, and report a fold.

    Args:
        env: The toolkit-layer environment.
        bus: The bus whose owner thread every grab runs on.
    """

    def __init__(self, env, bus: Bus) -> None:
        self.env = env
        self.bus = bus
        self.draggers: dict[str, Any] = {}
        self._sets: dict[str, str] = {}
        self._following = False

    def attach(
        self,
        entity: str = "",
        arm: str = "",
        feature: str = "",
        radius: float = 0.03,
        max_particles: int = 128,
        pin_max_step: float = 0.0005,
    ) -> dict[str, Any]:
        """Seize a deformable's declared feature at one hand and carry it.

        Args:
            entity: Which deformable, of the ones ``get_particle_state`` lists
                (:func:`_deformables`). Empty takes the first of them. It took the
                task's first ``deformables``-stage spec, and a cuttable is staged
                there: a tofu declared before a towel had the hand pin the eight
                corners of its cage and report ``attached`` while it held nothing.
            arm: Which hand. Empty takes the work arm -- *not* the first arm
                declared, since the two YAM cells list theirs in opposite orders
                and declaration order names the idle hand on one of them.
            feature: The declared particle set to seize: a towel's edge row, a
                shirt side's strip, a film's handle column. Empty takes the
                whole body, which is right for a cable and wrong for a sheet.
            radius: Miss radius [m]. A candidate further than this from the
                grasp frame is not seized, and ``attached`` comes back false.
            max_particles: Ceiling on the seized count. Size it to the declared
                strip: the default cuts a rim grasp down to a centre patch on a
                garment whose edge row is longer.
            pin_max_step: How far a pinned particle may be moved per substep [m].

        Returns:
            ``attached``, ``n_grabbed``, ``nearest_m``, ``arm``, ``feature``.
            A miss is ``attached`` false with the distance that missed --
            the honest answer to "did the grasp happen", and the one a graph
            routes on rather than lifting against nothing.
        """

        def grab() -> dict[str, Any]:
            from robosimstudio.core.draggers import ParticleDragger

            env = self.env
            name, body = _deformable(env, entity)
            arm_name = arm or env.handles.work_arm or next(iter(env.handles.arms))
            if arm_name not in env.handles.arms:
                raise ToolRefusal(f"no arm {arm_name!r}; have {sorted(env.handles.arms)}")

            dragger = self.draggers.get(arm_name)
            if dragger is None:
                dragger = ParticleDragger(
                    env.model,
                    radius=float(radius),
                    max_particles=int(max_particles),
                    mode="pin",
                    pin_max_step=float(pin_max_step),
                )
                self.draggers[arm_name] = dragger
                env.pipeline.hooks.post_solve.append(dragger.apply)
                if not self._following:
                    env.post_step_hooks.append(self._follow)
                    self._following = True
                env.pipeline.invalidate_graph()
            dragger.radius = float(radius)

            self._sets[arm_name] = str(feature)
            candidates = body.set_indices(feature) if feature else body.particles
            position, quat = env.arm(arm_name).ee_pose(env.state)
            seized = int(
                dragger.grab(
                    env.state,
                    position,
                    quat,
                    candidates=candidates,
                    fallback_nearest=False,
                    first_row=_first_particle_row(env),
                )
            )
            return {
                "attached": bool(seized),
                "n_grabbed": seized,
                "nearest_m": float(dragger.last_distance),
                "arm": arm_name,
                "entity": name,
                "feature": str(feature),
            }

        return self.bus.call(grab)

    def detach(self, arm: str = "") -> dict[str, Any]:
        """Let go. Harmless if nothing was held; empty ``arm`` releases every hand.

        Ordering is the caller's business and it is measured: release before the
        withdrawal, because a latch still on while the hand lifts takes the
        sheet with it.
        """

        def let_go() -> dict[str, Any]:
            names = [arm] if arm else list(self.draggers)
            for name in names:
                dragger = self.draggers.get(name)
                if dragger is not None:
                    dragger.release()
            return {"released": True, "arms": names}

        return self.bus.call(let_go)

    def _follow(self, env) -> None:
        """Re-aim each pin at its own hand. Every control step, owner thread."""
        for arm_name, dragger in self.draggers.items():
            if not dragger.active:
                continue
            position, quat = env.arm(arm_name).ee_pose(env.state)
            dragger.set_target(position, quat)


class Vacuum:
    """A suction hand's vacuum, switched by a graph: the rigid counterpart of :class:`ClothAttachment`.

    The same :class:`~robosimstudio.core.suction.Suction` the page's
    **Vacuum** box switches, so a graph and a person hold a plate the same way:
    the cup seals onto the part under its face when the face is within 2 mm of
    it and square to it within 15 degrees, and lets go when switched off.

    **Privileged, though a vacuum is real hardware.** A real cup reports one
    bit, its pressure switch, which is ``sealed``. The gap, the tilt and the
    name of what it sealed on are read off the simulator, and the hold itself is
    a weld or a spring rather than air. A graph that routes on ``sealed`` alone
    is the part that would transfer.

    Args:
        env: The toolkit-layer environment.
        bus: The bus whose owner thread every switch runs on.
    """

    def __init__(self, env, bus: Bus) -> None:
        self.env = env
        self.bus = bus

    def vacuum(self, on: bool = True, arm: str = "", object_name: str = "", tier: str = "auto") -> dict[str, Any]:
        """Switch one cup's vacuum on or off.

        Args:
            on: True seals onto the part under the face (or ``object_name``) and
                holds it; False lets go.
            arm: Which hand. Empty takes the first arm with a cup, *not* the
                work arm: on ``yam_bimanual_suction`` the work arm is the left,
                which has jaws.
            object_name: The part to seal on. Empty takes whatever is under
                the cup, which is what a real vacuum does.
            tier: ``"auto"``, ``"weld"`` or ``"wrench"``; see
                :meth:`~robosimstudio.core.suction.Suction.hold`.

        Returns:
            ``on``, ``sealed``, ``held`` (the part's name, or empty), ``gap_m``,
            ``tilt_deg``, ``tier``, ``arm``, ``why``. A refusal is ``sealed``
            false with the gap and tilt that refused it and ``why`` in a
            sentence, which is what a graph routes on rather than lifting air.
        """

        def switch() -> dict[str, Any]:
            suctions = self.env.scene.embodiment.suctions
            name = arm or next(iter(suctions))
            if name not in suctions:
                raise ToolRefusal(f"no suction hand on arm {name!r}; the cups are on {sorted(suctions)}")
            if tier not in ("auto", "weld", "wrench"):
                raise ToolRefusal(f"tier must be 'auto', 'weld' or 'wrench', not {tier!r}")
            cup = suctions[name]
            if not on:
                cup.release()
                return {
                    "on": False,
                    "sealed": False,
                    "held": "",
                    "gap_m": float("nan"),
                    "tilt_deg": float("nan"),
                    "tier": "",
                    "arm": name,
                    "why": "",
                }
            body = None
            if object_name:
                entity = self.env.scene.by_name.get(object_name)
                body = getattr(entity, "body", None)
                if body is None or int(body) < 0:
                    raise ToolRefusal(f"{object_name!r} is not a rigid body in this scene")
            try:
                report = cup.hold(body, tier=tier)
            except RuntimeError as error:
                raise ToolRefusal(str(error)) from error
            return {
                "on": True,
                "sealed": bool(report.sealed),
                "held": self._name_of(cup),
                "gap_m": float(report.gap_m),
                "tilt_deg": float(np.degrees(report.tilt_rad)),
                "tier": cup.tier,
                "arm": name,
                "why": report.why,
            }

        return self.bus.call(switch)

    def _name_of(self, cup) -> str:
        """The held part's entity, by the cup's own map of who built each part it can seal on.

        By an entity's :attr:`body` alone, a cup sealed on a roller's handle or a
        brush's strip held "": neither is the body its entity's pose places. A
        part named to :meth:`vacuum` is that body, so it is the fallback.
        """
        body = cup.held
        if body < 0:
            return ""
        if int(body) in cup.owners:
            return cup.owners[int(body)]
        for name, entity in self.env.scene.by_name.items():
            if getattr(entity, "body", None) == body:
                return name
        return ""


class GoalCursor:
    """Which of a placing task's goals is outstanding, and giving up on one.

    A sorting or packing problem is a *conjunction*, so a workflow that runs
    pick-transport-place once solves a single-goal variant and leaves an
    eight-goal one at 0.125. What turns one workflow into a loop over a whole
    suite is this: the goal tree, read against the evaluator's current verdict,
    answered as "work on this object, put it there".

    **Satisfaction is read from the evaluator, never recomputed.** The evaluator
    is authoritative about whether a goal is met -- a cursor that re-derived
    containment could disagree with the verdict the episode is actually scored
    on, and then a graph would place an object, be told it had not, and place it
    again. The metric key comes from :func:`~robosimstudio_bench.goals.goal.about_key`,
    which is also what the binder writes it with. Hard-coding that shape is how
    a shipped registry once spent a whole rollout concluding no goal was ever
    satisfied: it looked for a key the evaluator had stopped emitting, found
    nothing, re-attempted the same tool until the episode truncated, and raised
    nothing at all.

    **The pair is the containment clause's, and doneness is too.** Both are
    choices, and each has an alternative that a different document shape would
    want:

    * *The pair.* A goal here names a labelled **region** -- a compartment of a
      bin, a basket's cavity -- so the destination is the region the containment
      clause measures, and a graph aims at it with ``sim.get_region_obb``. A
      document whose goals instead name a destination **body** (an item must
      come to rest against *that tote*, and one shared cavity is where every
      item's containment is checked) states the pair on its ``contact-force``
      clause, and would need ``placements(..., atom="contact_force")`` and an
      object name in the payload rather than a region. That is a real
      difference and not a parameter: the two answers go to different tools.
    * *Doneness.* Containment, even where an object's goal is three clauses.
      Containment is monotone as an object descends into a container, where the
      contact force only appears once it lands -- so a cursor waiting for the
      force would keep re-picking something it had already dropped in.

    Attributes:
        given_up: Objects abandoned this episode. Episode-scoped because the
            connector -- and with it this object -- is rebuilt per episode by
            :meth:`~robosimstudio_policy_agent.graph_solution.SolutionGraph.reset`.
            Held here rather than threaded through the graph because a subgraph
            re-entered by a loop edge starts with a fresh scope: there is
            nowhere in a workflow for "the ones I have already failed on" to
            live.
        attempts: Passes charged against each goal, for :data:`MAX_ATTEMPTS`.
    """

    MAX_ATTEMPTS = 2
    """Passes at one goal before it is abandoned.

    The livelock this prevents is specific and does not look like a failure: an
    object is transported, released over its destination, bounces off a divider
    or a rim and lands outside. Nothing raised, so the graph loops, picks the
    same goal, and does it again -- forever, until the episode truncates with
    every *other* goal untouched. One retry is worth having because a bounce is
    often not repeatable; a third never paid.
    """

    def __init__(self, env, bus: Bus) -> None:
        self.env = env
        self.bus = bus
        self.given_up: set[str] = set()
        self.attempts: dict[str, int] = {}
        self.current: str = ""
        """The goal the loop gate last handed out -- the pass now in flight.

        What :meth:`skip_goal` abandons when its caller names nothing, which a
        failure branch never does: it ran because the pass failed, and the
        object that pass was working on is the one to give up on.
        """

    @property
    def goals(self) -> tuple[tuple[str, str], ...]:
        """``(object, region)`` for every goal, in the document's own order.

        Read off the compiled goal tree, not off a copy beside it. The suite
        this came from read the pairs out of ``TaskSpec.metadata`` first -- a
        copy, hand-written beside the goal that already stated them -- and a
        copy that disagrees with its original produces a policy diligently
        solving a task nobody is scoring.

        Through ``env.goal`` rather than ``env.evaluator.goal``, which is the
        same tree on a single world and the difference between running and not
        on a fleet: a world's connector is built against a ``WorldView``, and a
        view over one world of a wave has no evaluator of its own to reach
        through. Empty for a hand-written evaluator with no tree, which is the
        same answer :func:`placements` gives for a goal that places nothing.
        """
        goal = self.env.goal
        if goal is None:
            return ()
        # Only the objects this episode runs. The goal is already masked to the
        # episode's active set where it is built; a hand-written evaluator's
        # tree is not, and the handles are the authority on what is here.
        objects = self.env.handles.objects
        return tuple((obj, region) for obj, region in placements(goal) if obj in objects)

    def next_goal(self, exclude: Any = None, count_attempt: bool = False) -> dict[str, Any]:
        """The first goal not yet satisfied, in the document's own order.

        Args:
            exclude: Object names to pass over -- what a graph that has already
                failed on one object uses to get on with the rest instead of
                retrying it until the episode truncates. Anything iterable of
                strings, or ``None``.
            count_attempt: Charge this call as an attempt at whatever it
                returns, abandoning the goal once it has had
                :data:`MAX_ATTEMPTS`. Only the loop gate passes this: the other
                nodes in a pass ask the same question again to learn *which*
                object they are working on, and counting those would abandon
                every goal on its first pass.

        Returns:
            ``object_name`` and ``region_name`` for the goal to work on and
            ``done`` false; or empty names and ``done`` true when every goal is
            satisfied or abandoned. ``remaining`` counts the outstanding ones,
            so a graph can report progress without asking twice, and ``attempt``
            says which pass at this goal the episode is on (1-based) -- what a
            node varies its strategy over, the way ``tool_sort``'s grasp yaw
            alternates on it.
        """
        skip = set(exclude or ()) | self.given_up
        satisfied = self._satisfied()
        outstanding = [(obj, region) for obj, region in self.goals if not satisfied.get((obj, region), False)]
        if count_attempt:
            self.current = ""
        for obj, region in outstanding:
            if obj in skip:
                continue
            if count_attempt:
                self.attempts[obj] = self.attempts.get(obj, 0) + 1
                if self.attempts[obj] > self.MAX_ATTEMPTS:
                    self.given_up.add(obj)
                    continue
                self.current = obj
            return {
                "object_name": obj,
                "region_name": region,
                "remaining": len(outstanding),
                "done": False,
                "attempt": int(self.attempts.get(obj, 0) or 1),
            }
        return {"object_name": "", "region_name": "", "remaining": len(outstanding), "done": True, "attempt": 0}

    def skip_goal(self, object_name: str = "") -> dict[str, Any]:
        """Abandon one goal for the rest of the episode.

        What a graph calls when a grasp or a transport failed: the alternative is
        retrying the same one until the episode truncates, which on an
        eight-goal variant means one ungraspable object costs the other seven.
        Several ``tool_sort`` variants ship a deliberately impossible tool -- the
        utility knife is 10.8 mm thick against fingers that reach the bin floor
        first -- so giving up on one and getting on with the rest is the
        behaviour that scores them honestly.

        Args:
            object_name: Which to abandon. Empty abandons the goal the loop
                gate last handed out (:attr:`current`) -- the pass that just
                failed, which is the caller a failure branch is -- and falls
                back to whichever :meth:`next_goal` would return only when no
                gate call preceded it.

        The fallback used to be the whole rule, and it abandoned the wrong
        object whenever the failed pass had left its own goal satisfied: the
        shipped ``tool_sort/gt`` failure branch opens the hand *before* it
        gives up, a carry that failed short of the compartment drops the object
        into it, and the first outstanding goal at give-up time is then a
        never-attempted one. Measured on the level-1 sorting tier: 14 of 27
        attributed losses across 16 scenes were objects abandoned without a
        single attempt, each on the pass of a different object.
        """
        if not object_name:
            object_name = self.current or str(self.next_goal()["object_name"])
        if object_name:
            self.given_up.add(object_name)
        return {"object_name": object_name, "given_up": sorted(self.given_up)}

    def _satisfied(self) -> dict[tuple[str, str], bool]:
        """``(object, region) -> bool``, from the goal's containment clauses.

        Keyed by the pair rather than by the object, because a document is
        entitled to name one object twice -- and the containment clause alone,
        because a goal that groups three clauses per object (landed, at rest,
        in the right place) states the pair on exactly one of them. Reading
        every clause with two operands instead would also pick up
        ``contact-force object bin2``, which is a *different* pair and would
        never match a goal, so it is filtered here rather than relied on to
        miss.

        Through ``env.goal`` for the reason :attr:`goals` gives.
        """
        goal = self.env.goal
        if goal is None:
            return {}
        metrics = self.bus.snapshot.info.get("metrics") or {}
        return {
            (operands[0], operands[1]): bool(metrics.get(about_key(name, operands), False))
            for leaf in leaves(goal)
            for name, operands in (leaf.about,)
            if name == "containment" and len(operands) >= 2
        }


# ---------------------------------------------------------------- wire types
# gap's types are TypedDicts of plain floats, not arrays: a tool that hands back
# a numpy array serialises fine in process and fails the moment the same graph
# is pointed at an RPC tool bundle. Coercing here keeps that difference from
# ever being the thing under test.

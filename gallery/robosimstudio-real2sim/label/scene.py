"""A YAM labels a parcel by contact, in a geometry-only packing workshop."""

import json
from dataclasses import replace
from pathlib import Path

import lighting  # noqa: F401 -- registers the workshop light and material UVs
import numpy as np
import parcel  # noqa: F401 -- registers the native textured entities
from drive import lay, press_down
from geometry import label_pose
from motion import go, tick
from workshop import WOOD, add_workshop

import robosimstudio as rss
from robosimstudio.core.render.finish import Finish
from robosimstudio.core.render.room import LAB
from robosimstudio.robots.registry import get_embodiment

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
OVER_TAB = (0.3322, 2.1402, 1.8639, -1.2957, 0.0003, 0.3322)
yam = get_embodiment("yam")
left, right = yam.arms
yam = replace(yam, arms=(left, replace(right, home_q=OVER_TAB)))
cell = rss.get_cell("bench_0728")
wood = WOOD
table = replace(
    cell.statics[0],
    half_extents=(0.3, 0.4, 0.03),
    xform=(0, 0.2, -0.03, 0, 0, 0, 1),
    color=(1.0, 1.0, 1.0),
    finish=wood,
)
cell = replace(cell, name="label_workbench", center=(0.0, 0.2, 0.0), surface_z=0.0, ground_z=-0.728, statics=(table,))
from robosimstudio.scene.registry import CELLS

if "label_workbench" in CELLS:
    CELLS.replace("label_workbench", cell)
else:
    CELLS.register("label_workbench", cell)
physics = replace(
    rss.TAPE_60HZ,
    backends=tuple(
        (
            domain,
            rss.backend("mujoco", iterations=100, ls_iterations=50)
            if domain == "articulated"
            else rss.backend(
                "vbd",
                iterations=20,
                rigid_compliant_alm=True,
                rigid_body_contact_buffer_size=256,
                rigid_body_particle_contact_buffer_size=1024,
                friction_epsilon=0.01,
            )
            if domain == "*"
            else backend,
        )
        for domain, backend in rss.TAPE_60HZ.backends
    ),
)
scene = rss.Scene(
    cell=cell.name,
    physics=physics,
    name="label",
    render=replace(LAB, scenery="label_window_light", key_intensity=4.0, ambient=(0.20, 0.20, 0.20), quality="final"),
)
scene.add(rss.robot(yam))
scene.add(
    rss.box(
        name="box",
        half=(0.05, 0.06, 0.025),
        at=(0, -0.08, 0.025),
        mass=0.5,
        color=(1.0, 1.0, 1.0),
        entity="printed_parcel",
    )
)
scene.add(
    rss.tape(
        name="label",
        entity="printed_label",
        on="box",
        at=(0.02, -0.16, 0.0005),
        quat=(0, 0, 1, 0),
        material=rss.tapes.LABEL,
        stuck=False,
        dry=0.012,
        tab_color=(0.93, 0.92, 0.88),
        finish=Finish(roughness=0.9, texture=str(ASSETS / "label_face_r3.png"), uv_scale=(1 / 0.095, 1 / 0.04)),
    )
)
for x in (-0.26, 0.26):
    for y in (-0.35, 0.35):
        scene.add(
            rss.slab(
                name=f"leg_{x}_{y}",
                half=(0.035, 0.035, 0.334),
                at=(x, y, -0.394),
                color=(0.27, 0.15, 0.07),
                finish=wood,
            )
        )
add_workshop(scene)
scene.add(
    rss.CameraSpec(
        name="macro", eye=(0.30, -0.09, 0.212), look_at=(0.0, 0.05, 0.052), fov_y=34, width=1920, height=1080
    )
)
scene.add(
    rss.CameraSpec(
        name="overview", eye=(1.5, -1.1, 1.122), look_at=(-0.45, 0.2, 0.272), fov_y=48, width=1920, height=1080
    )
)
scene.add(
    rss.CameraSpec(
        name="wide", eye=(0.95, -0.25, 0.512), look_at=(-0.08, 0.14, 0.172), fov_y=40, width=1920, height=1080
    )
)


class Marks:
    def __init__(self, sim):
        self.sim = sim
        self.events = []

    def mark(self, name, **kw):
        self.events.append(dict(name=name, t=self.sim.time, **kw))
        print(name, kw, flush=True)


def routine(sim, session):
    label = sim["label"].entity
    box0 = np.asarray(sim["box"].position)
    rec = Marks(sim)
    yield from tick(session, 20)
    laid = yield from lay(session, label, rec)
    print("laid", laid, label_pose(label, sim.state), flush=True)
    rubbed = yield from press_down(session, label, rec)
    yield from go(
        session, "right", np.array(session.arms["right"].cursor_pos) + np.array([-0.06, 0, 0.10]), speed=0.08, steps=30
    )
    yield from tick(session, 45)
    result = dict(
        bonded_fraction=label.bonded(),
        lines=rubbed,
        **label_pose(label, sim.state),
        box_moved_mm=float(1000 * np.linalg.norm(np.array(sim["box"].position) - box0)),
        lay=laid,
        events=rec.events,
    )
    print("RESULT", json.dumps(result), flush=True)
    (HERE / "run" / "latest_result.json").write_text(json.dumps(result, indent=2))
    assert result["bonded_fraction"] >= 0.8, "Less than 80% bonded"
    assert result["off_mm"] <= 10, "Label missed its target"
    assert abs(result["yaw_deg"]) <= 5, "Label is not square"
    assert result["box_moved_mm"] < 10, "Box moved too far"

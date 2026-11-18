"""Rebuild assets/label_rest.npz on the page's GPU.

Run: ./ctl eval - --wait < settle_label.py
The unchanged stock label settles freely for 200 seconds in a robot-free
scene. Only initial positions are baked, not a trajectory or new rest lengths.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import robosimstudio as rss
from robosimstudio.scene.registry import CELLS


def settle_label():
    original = rss.get_cell("bench_0728")
    shift = np.array([0.6, -0.2, 0.728])
    statics = tuple(
        replace(v, xform=tuple(np.asarray(v.xform[:3]) - shift) + tuple(v.xform[3:])) for v in original.statics
    )
    cell = replace(
        original, name="label_rest_cell", center=(0, 0.2, 0), surface_z=0.0, ground_z=-0.728, statics=statics
    )
    if cell.name in CELLS:
        CELLS.replace(cell.name, cell)
    else:
        CELLS.register(cell.name, cell)
    phys = replace(
        rss.TAPE_60HZ,
        backends=tuple(
            (
                d,
                rss.backend(
                    "vbd",
                    rigid_compliant_alm=True,
                    rigid_body_contact_buffer_size=256,
                    rigid_body_particle_contact_buffer_size=1024,
                    friction_epsilon=0.000001,
                )
                if d == "*"
                else b,
            )
            for d, b in rss.TAPE_60HZ.backends
        ),
    )
    sc = rss.Scene(cell=cell.name, physics=phys, name="label_rest_bake")
    sc.add(rss.box(name="box", half=(0.05, 0.06, 0.025), at=(0, -0.08, 0.025), mass=0.5))
    sc.add(
        rss.tape(
            name="label",
            on="box",
            at=(0.02, -0.16, 0.0005),
            quat=(0, 0, 1, 0),
            material=rss.tapes.LABEL,
            stuck=False,
            dry=0.012,
        )
    )
    ss = sc.build(device="cuda:0")
    try:
        for _ in range(200):
            ss.run(60).sync()
            yield
        en = ss["label"].entity
        q = ss.state.particle_q.numpy()[np.asarray(en.particles)]
        tab = en.tab_pose(ss.state).astype(float)
        R = Rotation.from_quat(tab[3:])
        yaw = np.arctan2(R.as_matrix()[1, 0], R.as_matrix()[0, 0])
        rz = Rotation.from_euler("z", np.pi - yaw)
        anchor = np.array([0.0, 0.025, tab[2]])
        q = rz.apply(q - tab[:3]) + anchor
        tab = np.r_[anchor, (rz * R).as_quat()]
        out = Path("assets/label_rest.npz")
        np.savez(out, q=q.astype(np.float32), tab=tab.astype(np.float32))
        print("Saved freely settled initial label:", out)
    finally:
        ss.close()


settle_label()

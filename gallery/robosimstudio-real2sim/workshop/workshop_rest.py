"""Restore only the brush's physically settled initial deformation, before play."""

import numpy as np


def restore_brush(sim, brush, path):
    e = brush.entity
    with np.load(path, allow_pickle=False) as data:
        labels = np.array([sim.model.body_label[int(b)] for b in e.bodies])
        if not np.array_equal(labels, data["body_labels"]):
            raise ValueError("Brush topology changed; settle and bake it again")
        rest = data["body_q"].copy()
        grip = data["handle_pose"].copy()
    e._rest_body_q[e.bodies] = rest
    e._rest_body_q[e.handle_body] = grip
    index = {int(b): i for i, b in enumerate(e.bodies)}
    for coord, body in zip(e.strip_coords, e.strip_bodies, strict=True):
        e._rest_joint_q[int(coord) : int(coord) + 7] = rest[index[int(body)]]
    sim.settled_body_q[e.bodies] = rest
    sim.reset()

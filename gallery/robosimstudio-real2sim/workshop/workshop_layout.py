import numpy as np

PAVER_PITCH = 0.200
PAVER_GAP = 0.005
PAVING_TOP = 0.008
PHYSICS = "BRUSH_60HZ"
PARK_Q = (0.0, 0.20, 2.6, 0.0, 0.0, 0.0)
# A clear 350 x 500 mm task patch; the brush is at its back, the wall beyond it.
TASK_AREA = (-0.125, -0.250, 0.225, 0.250)

# One machine-readable table for every added prop, including each paving slab.
# Brush at is the construction footprint; settled_pose is the block underside
# centre after the measured drop. The NPZ also preserves every bent bristle.
LAYOUT = {
    "brush": {
        "kind": "simulated material",
        "size": (0.090, 0.180, 0.113),
        "at": (0.184, -0.064, 0.008),
        "quat": (0, 0, -0.0972264, 0.9952623),
        "settled_pose": (
            0.1792406678,
            -0.0587502308,
            0.0374366045,
            0.0013906813,
            0.0074458146,
            -0.0849203393,
            0.9963589907,
        ),
        "color": "#bd823d",
        "finish": "rough tan-orange fibre bundles following simulated capsule chains",
        "rows": 16,
        "columns": 20,
        "segments": 4,
        "length": 0.030,
        "radius": 0.0006,
        "flare": 0.25,
        "material": "nylon",
        "sole": (0.090, 0.180, 0.012),
        "grip": "own VBD D-grip",
        "grip_size": (0.040, 0.151, 0.071),
        "plastic": "#363a42",
        "initial_drop": 0.003,
        "density": 700,
        "visual_fibres_per_chain": 36,
        "visual_fan_half_width": 0.0053,
    },
    "brush_plastic": {
        "kind": "rigid parts of simulated brush",
        "size": (0.090, 0.180, 0.083),
        "at": (0.1792406678, -0.0587502308, 0.0374366045),
        "quat": (0.0013906813, 0.0074458146, -0.0849203393, 0.9963589907),
        "color": "#363a42",
        "finish": "dusty scuffed black plastic; rounded block, rubbed cartouche with plaster flecks, D grip and moulded channels",
    },
    "paving_bed": {
        "kind": "fixed",
        "size": (0.595, 1.30, 0.002),
        "at": (0, 0, 0.001),
        "quat": (0, 0, 0, 1),
        "color": "#514c42",
        "finish": "dark recessed joint backing",
    },
    "skirting": {
        "kind": "fixed",
        "size": (0.012, 1.30, 0.100),
        "at": (0.285, 0, 0.058),
        "quat": (0, 0, 0, 1),
        "color": "#a06535",
        "finish": "varnished orange-brown straight wood grain",
    },
    "wall": {
        "kind": "fixed",
        "size": (0.030, 1.30, 0.800),
        "at": (0.306, 0, 0.508),
        "quat": (0, 0, 0, 1),
        "color": "#e1dfdc",
        "finish": "rough white render, CC0 PaintedPlaster017 colour and normal",
    },
    "wall_band": {
        "kind": "fixed",
        "size": (0.001, 1.30, 0.025),
        "at": (0.2905, 0, 0.1205),
        "quat": (0, 0, 0, 1),
        "color": "#c8c9c6",
        "finish": "cooler, darker plaster repair band",
    },
    "skirting_joint": {
        "kind": "visual",
        "size": (0.0005, 0.0015, 0.100),
        "at": (0.2788, -0.0295, 0.058),
        "quat": (0, 0, 0, 1),
        "color": "#3c3028",
        "finish": "dark butt joint between board lengths",
    },
    "sun_occluder": {
        "kind": "visual",
        "size": (0.020, 4.124, 2.20),
        "at": (-0.900, -0.062, 1.95),
        "quat": (0, 0, 0, 1),
        "color": "#575552",
        "finish": "off-camera building edge casting the sunlight boundary",
    },
}
for i, x in enumerate(np.arange(-0.2975, 0.2975, PAVER_PITCH)):
    for j, y in enumerate(np.arange(-0.650, 0.650, PAVER_PITCH)):
        lo = np.maximum((x, y), (-0.2975, -0.65))
        hi = np.minimum((x + PAVER_PITCH - PAVER_GAP, y + PAVER_PITCH - PAVER_GAP), (0.2975, 0.65))
        LAYOUT[f"paver_{i}_{j}"] = {
            "kind": "fixed",
            "size": (*(hi - lo), 0.006),
            "at": (*((lo + hi) / 2), 0.005),
            "quat": (0, 0, 0, 1),
            "color": "#a69e8c",
            "finish": "dusty exposed aggregate; worn visual relief over a planar contact slab",
            "uv_offset": (i * 0.17, j * 0.21),
        }

CAMERA = {
    "width": 1920,
    "height": 1280,
    "fov_y": 22.4545,
    "eye": (0.132885, 0.184432, 1.181095),
    "look_at": (0.654745, 0.044560, 0.923439),
    "up": (0.420058, -0.112750, 0.900466),
    "near": 0.005,
    "far": 4,
}

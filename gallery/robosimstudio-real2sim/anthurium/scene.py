# SPDX-License-Identifier: Apache-2.0
"""Photo twin: simulated Anthurium, 140 mm terracotta rim as metric anchor.

All measured layout dimensions/poses/colours are in LAYOUT below, in the
bench_075 work-surface frame. The last capsule of each rooted rss.rod carries
a basal held patch of its rss.sheet. This kinematic attachment transfers no
sheet weight to the rod; the remaining cloth vertices and rod joints simulate.
"""

import json
import math
from pathlib import Path

import anthurium_assets as aa
import anthurium_coupling
import anthurium_material
import anthurium_outlines as ao
import numpy as np
from scipy.spatial.transform import Rotation

import robosimstudio as rss
from robosimstudio.core.render.finish import Finish
from robosimstudio.core.render.scenery import register_scenery
from robosimstudio.core.render.spec import RenderSpec
from robosimstudio.robots.registry import get_embodiment



# fmt: off
BUILD = {}  # this box: the page, its sim and its path tracer stay on GPU 5
# fmt: on
ROOT = Path(__file__).resolve().parent
ORIGIN = np.array((0.6, 0, 0.75))
PHOTO = {
    "name": "photo",
    "width": 1920,
    "height": 1280,
    "fov_y": 36.5,
    "eye": (-0.60, 0.24, 1.26),
    "look_at": (0.6, 0.24, 1.035),
}
# LIVE_LAYOUT_BEGIN
# Measured live layout: metres and xyzw poses in the bench work-surface frame.
# This literal table is intended for AST readers. Construction parameters follow.
# Full material parameters, all particles/nodes, and comparisons: anthurium_layout.json.
# fmt: off
LAYOUT = {
    "table_wood": {"kind": "fixed", "size_m": [0.595, 1.3, 0.040], "pose": [0.0, 0.0, -0.014, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#a77b43", "finish": "worn varnished wood"},
    "lid_board": {"kind": "fixed", "size_m": [0.3, 0.42, 0.04], "pose": [0.0, -0.025, 0.029, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#bb9363", "finish": "worn pale plywood"},
    "lid_seam": {"kind": "fixed", "size_m": [0.308, 0.428, 0.005], "pose": [0.0, -0.025, 0.009, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#573b24", "finish": "dark wood seam"},
    "saucer": {"kind": "fixed", "size_m": [0.2, 0.2, 0.014], "pose": [0.0, 0.0, 0.049, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#a5d4ba", "finish": "mint ceramic glaze"},
    "cachepot": {"kind": "fixed", "size_m": [0.14, 0.14, 0.124], "pose": [0.0, 0.0, 0.061, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#d7dece", "finish": "off-white glazed raised hexagons"},
    "terracotta": {"kind": "fixed", "size_m": [0.14, 0.14, 0.087], "pose": [0.0, 0.0, 0.125, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#a14f28", "finish": "terracotta satin"},
    "soil": {"kind": "fixed", "size_m": [0.131, 0.131, 0.01], "pose": [0.0, 0.0, 0.203, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#322919", "finish": "rough soil with grit"},
    "drawer_front": {"kind": "fixed", "size_m": [0.02, 0.66, 0.105], "pose": [-0.28, 0.0, -0.085, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#4c301b", "finish": "aged varnished oak"},
    "table_apron": {"kind": "fixed", "size_m": [0.55, 1.25, 0.065], "pose": [0.0, 0.0, -0.04, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#53351e", "finish": "aged varnished oak"},
    "plaster_wall": {"kind": "fixed", "size_m": [0.04, 2.8, 0.65], "pose": [0.36, 0.32, -0.225, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#7e9896", "finish": "rough blue-grey plaster"},
    "window_sill": {"kind": "fixed", "size_m": [0.08, 2.8, 0.09], "pose": [0.325, 0.32, 0.085, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#737544", "finish": "olive painted steel"},
    "window_crossbar": {"kind": "fixed", "size_m": [0.04, 2.8, 0.027], "pose": [0.36, 0.32, 0.429, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#505b2e", "finish": "olive painted steel"},
    "mullion_0": {"kind": "fixed", "size_m": [0.041, 0.022, 1.25], "pose": [0.35, -0.525, 0.72, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#515a31", "finish": "olive painted steel"},
    "mullion_1": {"kind": "fixed", "size_m": [0.041, 0.022, 1.25], "pose": [0.35, -0.215, 0.72, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#515a31", "finish": "olive painted steel"},
    "mullion_2": {"kind": "fixed", "size_m": [0.041, 0.022, 1.25], "pose": [0.35, -0.055, 0.72, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#515a31", "finish": "olive painted steel"},
    "mullion_3": {"kind": "fixed", "size_m": [0.041, 0.022, 1.25], "pose": [0.35, 0.257, 0.72, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#515a31", "finish": "olive painted steel"},
    "mullion_4": {"kind": "fixed", "size_m": [0.041, 0.022, 1.25], "pose": [0.35, 0.423, 0.72, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#515a31", "finish": "olive painted steel"},
    "mullion_5": {"kind": "fixed", "size_m": [0.041, 0.022, 1.25], "pose": [0.35, 1.03, 0.72, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#515a31", "finish": "olive painted steel"},
    "table_leg_0": {"kind": "fixed", "size_m": [0.06, 0.065, 0.66], "pose": [0.21, -0.55, -0.36, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#412b1c", "finish": "aged varnished oak"},
    "table_leg_1": {"kind": "fixed", "size_m": [0.06, 0.065, 0.66], "pose": [0.21, 0.55, -0.36, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#412b1c", "finish": "aged varnished oak"},
    "leaf_01_stalk": {"kind": "simulated material: rod", "size_m": [0.2199817, 0.0029, 0.0029], "pose": [0.01995, 0.0074921, 0.2161968, -0.2899339, -0.0406361, 0.0, 0.9561836], "colour_hex": "#42691b", "finish": "satin living green stalk", "material_kind": "rod", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_02_stalk": {"kind": "simulated material: rod", "size_m": [0.1877539, 0.0029, 0.0029], "pose": [-0.015461, 0.0216171, 0.2153608, -0.262298, 0.0010464, -0.0, 0.9649864], "colour_hex": "#42691b", "finish": "satin living green stalk", "material_kind": "rod", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_03_stalk": {"kind": "simulated material: rod", "size_m": [0.253379, 0.0029, 0.0029], "pose": [0.0024957, -0.0185944, 0.2202619, -0.1373205, 0.0209339, 0.0, 0.9903054], "colour_hex": "#42691b", "finish": "satin living green stalk", "material_kind": "rod", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_04_stalk": {"kind": "simulated material: rod", "size_m": [0.2591866, 0.0029, 0.0029], "pose": [0.0161368, 0.0186905, 0.2208921, -0.0134959, 0.1040098, -0.0, 0.9944847], "colour_hex": "#42691b", "finish": "satin living green stalk", "material_kind": "rod", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_05_stalk": {"kind": "simulated material: rod", "size_m": [0.288541, 0.0029, 0.0029], "pose": [-0.0158942, -0.0032292, 0.2219637, -0.0223302, 0.1369157, 0.0, 0.990331], "colour_hex": "#42691b", "finish": "satin living green stalk", "material_kind": "rod", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_06_stalk": {"kind": "simulated material: rod", "size_m": [0.2006381, 0.0029, 0.0029], "pose": [0.018085, -0.0148986, 0.2180722, 0.0964248, 0.0137233, -0.0, 0.9952456], "colour_hex": "#42691b", "finish": "satin living green stalk", "material_kind": "rod", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_07_stalk": {"kind": "simulated material: rod", "size_m": [0.1780072, 0.0029, 0.0029], "pose": [-0.0069335, 0.0187297, 0.216384, 0.1469679, -0.0623812, 0.0, 0.9871722], "colour_hex": "#42691b", "finish": "satin living green stalk", "material_kind": "rod", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_08_stalk": {"kind": "simulated material: rod", "size_m": [0.1701487, 0.0029, 0.0029], "pose": [-0.0049399, -0.0212599, 0.2152666, 0.0382212, 0.2139112, -0.0, 0.9761051], "colour_hex": "#42691b", "finish": "satin living green stalk", "material_kind": "rod", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_09_stalk": {"kind": "simulated material: rod", "size_m": [0.1242175, 0.0029, 0.0029], "pose": [0.0164872, 0.007599, 0.2134988, 0.0166114, -0.1808694, 0.0, 0.9833668], "colour_hex": "#42691b", "finish": "satin living green stalk", "material_kind": "rod", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_10_stalk": {"kind": "simulated material: rod", "size_m": [0.0777686, 0.0029, 0.0029], "pose": [-0.0204425, 0.0080132, 0.2118756, 0.0543123, -0.0739133, -0.0, 0.9957846], "colour_hex": "#42691b", "finish": "satin living green stalk", "material_kind": "rod", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_11_stalk": {"kind": "simulated material: rod", "size_m": [0.2701342, 0.0029, 0.0029], "pose": [0.0119627, -0.0236241, 0.2212345, 0.0840922, 0.0918841, 0.0, 0.9922126], "colour_hex": "#42691b", "finish": "satin living green stalk", "material_kind": "rod", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_12_stalk": {"kind": "simulated material: rod", "size_m": [0.1246495, 0.0029, 0.0029], "pose": [0.0079631, 0.0233232, 0.2141621, -0.0734961, 0.0901586, -0.0, 0.9932119], "colour_hex": "#42691b", "finish": "satin living green stalk", "material_kind": "rod", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "spathe_01_stalk": {"kind": "simulated material: rod", "size_m": [0.3007068, 0.0029, 0.0029], "pose": [-0.0160064, -0.0072095, 0.2226133, -0.119643, 0.0596174, 0.0, 0.9910254], "colour_hex": "#42691b", "finish": "satin living green stalk", "material_kind": "rod", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "spathe_01_spadix": {"kind": "visual (kinematic spadix)", "size_m": [0.0045, 0.0045, 0.05], "pose": [0.0015143, 0.0724214, 0.522668, -0.0196303, -0.4328595, 0.002728, 0.9012435], "colour_hex": "#c8b74b", "finish": "pale yellow pollen"},
    "spathe_02_stalk": {"kind": "simulated material: rod", "size_m": [0.2948153, 0.0029, 0.0029], "pose": [0.0229869, -0.0051207, 0.2229254, 0.0053411, 0.0685355, -0.0, 0.9976344], "colour_hex": "#42691b", "finish": "satin living green stalk", "material_kind": "rod", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "spathe_02_spadix": {"kind": "visual (kinematic spadix)", "size_m": [0.0045, 0.0045, 0.05], "pose": [0.0654265, -0.0127689, 0.5187732, 0.0674081, -0.1052412, -0.0002878, 0.9921595], "colour_hex": "#c8b74b", "finish": "pale yellow pollen"},
    "spathe_03_stalk": {"kind": "simulated material: rod", "size_m": [0.2843422, 0.0029, 0.0029], "pose": [-0.0124823, 0.0221245, 0.2222079, -0.0950213, -0.0113398, -0.0, 0.9954106], "colour_hex": "#42691b", "finish": "satin living green stalk", "material_kind": "rod", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "spathe_03_spadix": {"kind": "visual (kinematic spadix)", "size_m": [0.0045, 0.0045, 0.05], "pose": [-0.0240156, 0.0923856, 0.5044783, -0.0954913, -0.0664332, 0.0002513, 0.9932111], "colour_hex": "#c8b74b", "finish": "pale yellow pollen"},
    "spathe_04_stalk": {"kind": "simulated material: rod", "size_m": [0.2576683, 0.0029, 0.0029], "pose": [-0.0058032, -0.021159, 0.2207733, -0.0514144, -0.0969786, 0.0, 0.9939576], "colour_hex": "#42691b", "finish": "satin living green stalk", "material_kind": "rod", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "spathe_04_spadix": {"kind": "visual (kinematic spadix)", "size_m": [0.0045, 0.0045, 0.05], "pose": [-0.0752526, 0.0116021, 0.4748279, -0.0110464, -0.1528204, 0.0003633, 0.9881922], "colour_hex": "#c8b74b", "finish": "pale yellow pollen"},
    "leaf_01": {"kind": "simulated material: cloth", "size_m": [0.1539859, 0.0983211, 0.00045], "pose": [-0.0040277, 0.1875207, 0.3392383, -0.3582305, 0.0570083, 0.7113066, 0.6020497], "colour_hex": "#419c13", "finish": "glossy deep green veined lamina", "material_kind": "sheet", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_02": {"kind": "simulated material: cloth", "size_m": [0.1454562, 0.0855625, 0.00045], "pose": [-0.0399701, 0.1729916, 0.321456, -0.2277097, 0.2021482, 0.8432017, 0.4430523], "colour_hex": "#316b17", "finish": "glossy deep green veined lamina", "material_kind": "sheet", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_03": {"kind": "simulated material: cloth", "size_m": [0.126889, 0.0746406, 0.00045], "pose": [-0.0084822, 0.0828463, 0.4113946, -0.3634895, 0.2842529, 0.8240637, 0.3286254], "colour_hex": "#174e24", "finish": "glossy deep green veined lamina", "material_kind": "sheet", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_04": {"kind": "simulated material: cloth", "size_m": [0.1413671, 0.0812883, 0.00045], "pose": [0.0727938, -0.0001451, 0.4383627, -0.3881676, 0.1698054, 0.696667, -0.5789189], "colour_hex": "#174624", "finish": "glossy deep green veined lamina", "material_kind": "sheet", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_05": {"kind": "simulated material: cloth", "size_m": [0.1455101, 0.0864286, 0.00045], "pose": [0.1059297, -0.0325026, 0.4705324, 0.4180622, -0.1418464, -0.4447518, 0.7792942], "colour_hex": "#0b3d19", "finish": "glossy deep green veined lamina", "material_kind": "sheet", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_06": {"kind": "simulated material: cloth", "size_m": [0.1248646, 0.0748035, 0.00045], "pose": [0.0395591, -0.0873766, 0.3920936, 0.2701938, 0.0285529, -0.6172884, 0.7383326], "colour_hex": "#255616", "finish": "glossy deep green veined lamina", "material_kind": "sheet", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_07": {"kind": "simulated material: cloth", "size_m": [0.1158382, 0.0581934, 0.00045], "pose": [-0.0224583, -0.08475, 0.3543708, 0.3510326, -0.0486033, -0.5608938, 0.7482058], "colour_hex": "#174e13", "finish": "glossy deep green veined lamina", "material_kind": "sheet", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_08": {"kind": "simulated material: cloth", "size_m": [0.1140487, 0.06, 0.00045], "pose": [0.105134, -0.0427465, 0.3088066, -0.0528451, 0.5448259, -0.0496615, 0.8354076], "colour_hex": "#315d15", "finish": "glossy deep green veined lamina", "material_kind": "sheet", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_09": {"kind": "simulated material: cloth", "size_m": [0.0846042, 0.0445396, 0.00045], "pose": [-0.0573552, -0.0040982, 0.2933239, -0.4106741, -0.0712263, 0.8829418, -0.2160728], "colour_hex": "#0e4531", "finish": "glossy deep green veined lamina", "material_kind": "sheet", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_10": {"kind": "simulated material: cloth", "size_m": [0.0824559, 0.047, 0.00045], "pose": [-0.0625922, -0.0049424, 0.2604308, -0.3510084, -0.1821688, 0.9184187, -0.0107132], "colour_hex": "#164728", "finish": "glossy deep green veined lamina", "material_kind": "sheet", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_11": {"kind": "simulated material: cloth", "size_m": [0.1182589, 0.0695641, 0.00045], "pose": [0.0976084, -0.0975284, 0.439382, 0.3520004, 0.2368319, -0.3421525, 0.838414], "colour_hex": "#4d681c", "finish": "glossy deep green veined lamina", "material_kind": "sheet", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "leaf_12": {"kind": "simulated material: cloth", "size_m": [0.0946028, 0.054, 0.00045], "pose": [0.0585013, 0.0552659, 0.2950765, -0.1000228, 0.4871399, 0.1403135, 0.8561555], "colour_hex": "#295d1e", "finish": "glossy deep green veined lamina", "material_kind": "sheet", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "spathe_01": {"kind": "simulated material: cloth", "size_m": [0.106254, 0.0685017, 0.00045], "pose": [0.0435909, 0.0786819, 0.5152261, -0.0519751, -0.5189403, 0.0019984, 0.8532265], "colour_hex": "#d70c17", "finish": "glossy veined spathe", "material_kind": "sheet", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "spathe_02": {"kind": "simulated material: cloth", "size_m": [0.087665, 0.0535024, 0.00045], "pose": [0.0982807, -0.0266585, 0.513025, 0.3232323, -0.4280532, -0.1243747, 0.8347588], "colour_hex": "#bf111c", "finish": "glossy veined spathe", "material_kind": "sheet", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "spathe_03": {"kind": "simulated material: cloth", "size_m": [0.1173946, 0.0411713, 0.00045], "pose": [-0.0296763, 0.0553863, 0.4924171, -0.0461024, 0.4053505, 0.792237, -0.4537908], "colour_hex": "#8b331b", "finish": "glossy veined spathe", "material_kind": "sheet", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "spathe_04": {"kind": "simulated material: cloth", "size_m": [0.0801094, 0.0397639, 0.00045], "pose": [-0.0629158, 0.0185785, 0.4504616, -0.4719817, -0.1217015, 0.8275413, 0.2785631], "colour_hex": "#ab203b", "finish": "glossy veined spathe", "material_kind": "sheet", "preset": "ANTHURIUM_VBD_60HZ", "backend": "vbd"},
    "table": {"kind": "fixed cell", "size_m": [0.595, 1.3, 0.75], "pose": [0.0, 0.0, -0.375, 0.0, 0.0, 0.0, 1.0], "colour_hex": "#a77b43", "finish": "wood overlay table_wood"},
}
# fmt: on
# LIVE_LAYOUT_END

# The prescribed bench remains 0.595 x 1.30 m. All decoration outside the pot/board is peripheral.
# name: shape, full size[m], centre[bench m], colour hex, finish.
PROPS = {
    "table_wood": {
        "shape": "box",
        "size": (0.595, 1.30, 0.040),
        "at": (0, 0, -0.014),
        "color": "#a77b43",
        "finish": "worn varnished wood",
    },
    "lid_board": {
        "shape": "beveled_box",
        "size": (0.30, 0.42, 0.040),
        "at": (0.0, -0.025, 0.029),
        "color": "#bb9363",
        "finish": "worn pale plywood",
    },
    "lid_seam": {
        "shape": "box",
        "size": (0.308, 0.428, 0.005),
        "at": (0, -0.025, 0.009),
        "color": "#573b24",
        "finish": "dark wood seam",
    },
    "saucer": {
        "shape": "lathe",
        "size": (0.200, 0.200, 0.014),
        "at": (0, 0, 0.049),
        "color": "#a5d4ba",
        "finish": "mint ceramic glaze",
    },
    "cachepot": {
        "shape": "lathe_hex",
        "size": (0.140, 0.140, 0.124),
        "at": (0, 0, 0.061),
        "color": "#d7dece",
        "finish": "off-white glazed raised hexagons",
    },
    "terracotta": {
        "shape": "lathe",
        "size": (0.140, 0.140, 0.087),
        "at": (0, 0, 0.125),
        "color": "#a14f28",
        "finish": "terracotta satin",
    },
    "soil": {
        "shape": "cylinder",
        "size": (0.131, 0.131, 0.010),
        "at": (0, 0, 0.203),
        "color": "#322919",
        "finish": "rough soil with grit",
    },
    "drawer_front": {
        "shape": "box",
        "size": (0.020, 0.66, 0.105),
        "at": (-0.280, 0, -0.085),
        "color": "#4c301b",
        "finish": "aged varnished oak",
    },
    "table_apron": {
        "shape": "box",
        "size": (0.55, 1.25, 0.065),
        "at": (0, 0, -0.040),
        "color": "#53351e",
        "finish": "aged varnished oak",
    },
    "plaster_wall": {
        "shape": "plaster_mesh",
        "size": (0.04, 2.8, 0.65),
        "at": (0.36, 0.32, -0.225),
        "color": "#7e9896",
        "finish": "rough blue-grey plaster",
    },
    "window_sill": {
        "shape": "box",
        "size": (0.08, 2.8, 0.090),
        "at": (0.325, 0.32, 0.085),
        "color": "#737544",
        "finish": "olive painted steel",
    },
    "window_crossbar": {
        "shape": "box",
        "size": (0.040, 2.8, 0.027),
        "at": (0.36, 0.32, 0.429),
        "color": "#505b2e",
        "finish": "olive painted steel",
    },

}
for i, y in enumerate((-0.525, -0.215, -0.055, 0.257, 0.423, 1.03)):
    PROPS[f"mullion_{i}"] = {
        "shape": "box",
        "size": (0.041, 0.022, 1.25),
        "at": (0.35, y, 0.72),
        "color": "#515a31",
        "finish": "olive painted steel",
    }
for i, (x, y) in enumerate(((.21,-.55),(.21,.55),(-.24,-.55),(-.24,.55))):
    PROPS[f"table_leg_{i}"] = {
        "shape": "box",
        "size": (0.06, 0.065, 0.66),
        "at": (x, y, -0.36),
        "color": "#412b1c",
        "finish": "aged varnished oak",
    }
# Blade pose is authored by basal and apical photo landmarks PLUS independent depths,
# and a normal hint from backlight/foreshortening. These are 3D, not camera-facing cards.
# name, base px, tip px, base x, tip x, width, normal hint, colour, cue
BLADES = [
    {
        "name": "leaf_01",
        "base": (1110, 508),
        "tip": (915, 553),
        "depth": (0.00, -0.045),
        "width": 0.068,
        "normal": (-0.65, 0.10, 0.75),
        "color": "#419c13",
        "cue": "left glowing underside; long drooping midrib, broad silhouette",
    },
    {
        "name": "leaf_02",
        "base": (1127, 550),
        "tip": (958, 596),
        "depth": (-0.015, -0.085),
        "width": 0.0855624900659411,
        "normal": (0.005124997356272296, 0.16661013844726447, 0.986009531479631),
        "color": "#316b17",
        "cue": "lower left blade narrow/foreshortened, below and in front of leaf 01; depth tilt uses a 1.7:1 physical blade aspect prior",
    },
    {
        "name": "leaf_03",
        "base": (1244, 387),
        "tip": (1156, 496),
        "depth": (0.015, -0.055),
        "width": 0.07464060083995634,
        "normal": (-0.2357443746558236, 0.5343322485372557, 0.8117349555065712),
        "color": "#174e24",
        "cue": "oblique central-left glossy blade, overlaps lit left leaf; depth tilt uses a 1.7:1 physical blade aspect prior",
    },
    {
        "name": "leaf_04",
        "base": (1286, 362),
        "tip": (1368, 424),
        "depth": (0.08, 0.00),
        "width": 0.096,
        "normal": (0.10, 0.30, 0.94),
        "color": "#174624",
        "cue": "upright rear leaf seen obliquely through petioles",
    },
    {
        "name": "leaf_05",
        "base": (1312, 329),
        "tip": (1448, 345),
        "depth": (0.075, 0.04),
        "width": 0.104,
        "normal": (-0.65, -0.15, 0.74),
        "color": "#0b3d19",
        "cue": "dark upper canopy behind front spathe; broad upward face; depth tilt uses a 1.7:1 physical blade aspect prior",
    },
    {
        "name": "leaf_06",
        "base": (1447, 433),
        "tip": (1513, 471),
        "depth": (0.025, -0.055),
        "width": 0.070,
        "normal": (-0.40, -0.35, 0.84),
        "color": "#255616",
        "cue": "right upper blade strongly foreshortened, rear basal lobe visible; depth tilt uses a 1.7:1 physical blade aspect prior",
    },
    {
        "name": "leaf_07",
        "base": (1424, 503),
        "tip": (1540, 514),
        "depth": (-0.035, -0.10),
        "width": 0.063,
        "normal": (-0.47, -0.28, 0.84),
        "color": "#174e13",
        "cue": "right horizontal arrow blade, broad upper surface toward window",
    },
    {
        "name": "leaf_08",
        "base": (1376, 532),
        "tip": (1370, 624),
        "depth": (0.085, 0.15),
        "width": 0.060,
        "normal": (0.18, 0.15, 0.97),
        "color": "#315d15",
        "cue": "narrow rear blade mostly obscured by vertical stems",
    },
    {
        "name": "leaf_09",
        "base": (1354, 602),
        "tip": (1390, 665),
        "depth": (-0.045, -0.105),
        "width": 0.038,
        "normal": (-0.60, -0.20, 0.77),
        "color": "#0e4531",
        "cue": "front glossy leaf with highlight, folds down toward viewer",
    },
    {
        "name": "leaf_10",
        "base": (1364, 669),
        "tip": (1398, 718),
        "depth": (-0.04, -0.11),
        "width": 0.047,
        "normal": (-0.48, -0.40, 0.78),
        "color": "#164728",
        "cue": "small basal front leaf occludes stems above soil",
    },
    {
        "name": "leaf_11",
        "base": (1449, 352),
        "tip": (1519, 399),
        "depth": (0.07, 0.14),
        "width": 0.06956406127839401,
        "normal": (-0.08786207920794681, -0.528056104112673, 0.8446520028666257),
        "color": "#4d681c",
        "cue": "back right leaf edge and yellowing tip, behind canopy; depth tilt uses a 1.7:1 physical blade aspect prior",
    },
    {
        "name": "leaf_12",
        "base": (1255, 575),
        "tip": (1210, 638),
        "depth": (0.04, 0.10),
        "width": 0.054,
        "normal": (0.35, 0.18, 0.92),
        "color": "#295d1e",
        "cue": "small lower-left rear leaf glimpsed between stalks",
    },
    {
        "name": "spathe_01",
        "base": (1225, 309),
        "tip": (1219, 221),
        "depth": (0.025, 0.060),
        "width": 0.092,
        "normal": (-0.87, 0.10, 0.48),
        "color": "#d70c17",
        "cue": "highest scarlet heart facing forward, tilts back about 25 degrees",
        "spathe": True,
    },
    {
        "name": "spathe_02",
        "base": (1346, 293),
        "tip": (1401, 251),
        "depth": (0.07, 0.11),
        "width": 0.055,
        "normal": (-0.52, -0.60, 0.61),
        "color": "#bf111c",
        "cue": "upper right crimson heart turned partly away behind canopy",
        "spathe": True,
    },
    {
        "name": "spathe_03",
        "base": (1218, 325),
        "tip": (1287, 286),
        "depth": (-0.020, -0.070),
        "width": 0.082,
        "normal": (-0.40, 0.55, 0.73),
        "color": "#8b331b",
        "cue": "left front rust-red spathe, heavily foreshortened and shaded",
        "spathe": True,
    },
    {
        "name": "spathe_04",
        "base": (1350, 383),
        "tip": (1314, 352),
        "depth": (-0.065, -0.035),
        "width": 0.096,
        "normal": (-0.90, -0.06, 0.43),
        "color": "#ab203b",
        "cue": "front pink-crimson heart, visible notch and darker central cup",
        "spathe": True,
    },
]
PHYSICS_NAME = "ANTHURIUM_VBD_60HZ: GENERIC proxy, 16 substeps, 20 VBD iterations, cloth self-contact disabled"
PHYSICS = rss.GENERIC.with_backend(
    "*",
    rss.backend(
        "vbd",
        iterations=20,
        particle_enable_self_contact=False,
        rigid_body_contact_buffer_size=1024,
        particle_self_contact_radius=0.0003,
        particle_self_contact_margin=0.0007,
        particle_rest_shape_contact_exclusion_radius=0.004,
    ),
)
ARM_BACKEND = rss.GENERIC.backend_for("articulated").with_(ls_iterations=80)
PHYSICS = PHYSICS.with_backend("articulated", ARM_BACKEND).with_backend("rigid", ARM_BACKEND)
PHYSICS = PHYSICS.with_backend("liquid", rss.backend("mpm", project_outside=False, step_hz=480.0, collider_basis="pic", strain_basis="P1d"))
PHYSICS = PHYSICS.with_(coupling=rss.coupler("anthurium_water"), use_cuda_graph=False, substeps=16)
LEAF_MATERIAL = rss.ClothMaterialSpec(
    tri_ke=14000,
    tri_ka=14000,
    tri_kd=0.025,
    edge_ke=40.0,
    edge_kd=0.02,
    areal_density=0.23,
    particle_radius=0.0006,
    thickness=0.00045,
    zero_rest_angle=False,
    contact_ke=5000,
    contact_kd=0.1,
)
ROD_PARAMS = {
    "radius": 0.00145,
    "density": 650,
    "stretch_ke": 100000,
    "stretch_kd": 0.8,
    "bend_ke": 5.0,
    "bend_kd": 0.08,
    "twist_ke": 3.0,
    "twist_kd": 0.05,
    "contact_gap": 0.0005,
    "anchors": ("start",),
    "allow_grasp_latch": False,
    "filter_fingertip_scenery": False,
    "mu_cable": 0.7,
}


def rgb(h):
    return tuple(int(h[i : i + 2], 16) / 255 for i in (1, 3, 5))


def norm(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


CAM_FORWARD = norm(np.array(PHOTO["look_at"]) - PHOTO["eye"])
CAM_RIGHT = norm(np.cross(CAM_FORWARD, (0, 0, 1)))
CAM_UP = np.cross(CAM_RIGHT, CAM_FORWARD)
FOCAL = 640 / math.tan(math.radians(PHOTO["fov_y"] / 2))


def unproject(pixel, x):
    ray = CAM_FORWARD + (pixel[0] - 960) / FOCAL * CAM_RIGHT - (pixel[1] - 640) / FOCAL * CAM_UP
    eye = np.array(PHOTO["eye"])
    p = eye + ray * ((x + 0.6 - eye[0]) / ray[0])
    return p - ORIGIN


def mesh_spec(name, path, at, color, finish, collide=False):
    return rss.StaticSpec(
        name=name,
        asset=rss.AssetSpec(path=path),
        xform=(*at, 0, 0, 0, 1),
        color=rgb(color),
        finish=finish,
        collide=collide,
    )


SUN_DIRECTION = (-1.0, 0.20, -0.80)

def scenery(kit):
    # The path tracer's generic lighting doesn't expose a positioned key. A local
    # scenery builder adds actual USD window lights; geometry still uses core specs.
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    stage = kit.stage
    dome = UsdLux.DomeLight.Define(stage, "/root/GardenAmbient")
    dome.GetIntensityAttr().Set(150)
    dome.GetColorAttr().Set(Gf.Vec3f(.68,.78,1.0))
    sun = UsdLux.DistantLight.Define(stage, "/root/AnthuriumSun")
    sun.GetIntensityAttr().Set(2500)
    sun.GetAngleAttr().Set(2.5)
    sun.GetColorAttr().Set(Gf.Vec3f(1.0, 0.86, 0.65))
    sunq = Rotation.align_vectors([norm(SUN_DIRECTION)], [[0, 0, -1]])[0].as_quat()
    UsdGeom.Xformable(sun).AddOrientOp().Set(Gf.Quatf(float(sunq[3]), Gf.Vec3f(*sunq[:3])))
    # Window light directed toward -x and down.
    panel = UsdLux.RectLight.Define(stage, "/root/AnthuriumWindow")
    panel.GetWidthAttr().Set(1.4)
    panel.GetHeightAttr().Set(0.9)
    panel.GetIntensityAttr().Set(700)
    panel.GetColorAttr().Set(Gf.Vec3f(0.88, 1.0, 0.88))
    xf = UsdGeom.Xformable(panel)
    xf.AddTranslateOp().Set(Gf.Vec3d(0.88, 0.4, 1.7))
    xf.AddRotateYOp().Set(65)
    sky = UsdLux.RectLight.Define(stage, "/root/GardenSky")
    sky.GetWidthAttr().Set(3)
    sky.GetHeightAttr().Set(3)
    sky.GetIntensityAttr().Set(12000)
    sky.GetColorAttr().Set(Gf.Vec3f(0.92, 1, 1))
    UsdGeom.Xformable(sky).AddTranslateOp().Set(Gf.Vec3d(1.8, 0, 3.0))
    garden_bounce=UsdLux.RectLight.Define(stage,"/root/GardenBounce")
    garden_bounce.GetWidthAttr().Set(2.5)
    garden_bounce.GetHeightAttr().Set(1.5)
    garden_bounce.GetIntensityAttr().Set(15000)
    garden_bounce.GetColorAttr().Set(Gf.Vec3f(1.0,.98,.82))
    gx=UsdGeom.Xformable(garden_bounce)
    gx.AddTranslateOp().Set(Gf.Vec3d(2.0,.3,1.65))
    gx.AddRotateYOp().Set(-90)
    near = UsdLux.RectLight.Define(stage, "/root/SillBounce")
    near.GetWidthAttr().Set(2.5)
    near.GetHeightAttr().Set(1.5)
    near.GetIntensityAttr().Set(7000)
    near.GetColorAttr().Set(Gf.Vec3f(1.0,.91,.75))
    nx=UsdGeom.Xformable(near)
    nx.AddTranslateOp().Set(Gf.Vec3d(1.05,.3,1.65))
    nx.AddRotateYOp().Set(-90)
    # Soft room bounce.
    fill = UsdLux.RectLight.Define(stage, "/root/RoomFill")
    fill.GetWidthAttr().Set(1)
    fill.GetHeightAttr().Set(1)
    fill.GetIntensityAttr().Set(6000)
    xf = UsdGeom.Xformable(fill)
    xf.AddTranslateOp().Set(Gf.Vec3d(-0.3, 0.8, 1.65))
    xf.AddRotateYOp().Set(-45)
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        sh = UsdShade.Shader(prim)
        if sh.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        src = sh.GetInput("diffuseColor").GetConnectedSource()
        if not src:
            continue
        texture = UsdShade.Shader(src[0].GetPrim())
        file = texture.GetInput("file")
        if file:
            normal_files = {
                "plaster_blue": "PaintedPlaster017/PaintedPlaster017_2K-JPG_NormalGL.jpg",
                "pale_board": "plywood/plywood_nor_gl_2k.jpg",
            }
            texture_path = Path(file.Get().path)
            if texture_path.name.startswith(("leaf_", "spathe_", "table_board")):
                sh.GetInput("useSpecularWorkflow").Set(1)
                reflectance = 0.0 if texture_path.name.startswith("table_board") else 0.008
                sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(reflectance))
                sh.GetInput("roughness").Set(0.85)
                # The diffuse vein texture supplies the blade detail on both sides.
                # Do not apply tangent-space normals to these two-sided sheets.
                continue
            for token, relative in normal_files.items():
                if token not in str(file.Get()):
                    continue
                normal = UsdShade.Shader.Define(stage, prim.GetPath().GetParentPath().AppendChild("surface_normal"))
                normal.CreateIdAttr("UsdUVTexture")
                normal.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                    Sdf.AssetPath(
                        relative
                        if Path(relative).is_absolute()
                        else str(ROOT / "assets" / "tex" / relative)
                    )
                )
                normal.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
                normal.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                normal.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
                st = texture.GetInput("st").GetConnectedSource()
                if st:
                    normal.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st[0], st[1])
                normal.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(normal.ConnectableAPI(), "rgb")


register_scenery("anthurium_window_v10", scenery)
from robosimstudio.scene.registry import get_cell, register_cell
bench = get_cell("bench_075")
bench = rss.replace(bench, name="anthurium_bench", statics=tuple(rss.replace(s, visible=False, entity="anthurium_hidden_fixture") if s.name == "table" else s for s in bench.statics))
register_cell(bench)
scene = rss.Scene(
    cell=bench.name,
    name="anthurium",
    physics=PHYSICS,
    render=RenderSpec(
        environment="none",
        scenery="anthurium_window_v10",
        ambient=(0.13, 0.13, 0.12),
        light_direction=SUN_DIRECTION,
        key_intensity=0.9,
        enable_textures=True,
        quality="final",
    ),
)
yam = get_embodiment("yam")
PARK_LEFT = (3.05, 3.66, 2.7, -1.23, 0.0, 0.0)
PARK_RIGHT = (-1.76, 3.56, 1.66, -0.5, 0.0, 0.0)
yam = rss.replace(
    yam, arms=tuple(rss.replace(a, home_q=PARK_LEFT if a.name == "left" else PARK_RIGHT) for a in yam.arms)
)
scene.add(rss.robot(yam))
scene.add(rss.CameraSpec(**PHOTO))
scene.add(rss.CameraSpec(name="wide", width=1280, height=960, fov_y=43, eye=(-0.45,-0.95,1.48), look_at=(0.60,-0.08,1.02)))
# Standard five-wall glass cup from the old watering take.
import os
os.environ["ROBOSIMSTUDIO_CACHE"] = str(ROOT / "assets" / "cache")
from robosimstudio_materials.liquid import cup
CUP_AT = (0.12, -0.31, 0.006)
scene.add(cup(name="cup", entity="anthurium_glass_cup", inner=(0.05,0.05,0.07), wall=0.008, at=CUP_AT, quat=(0,0,0.5,0.8660254037844386),
              mass=0.18, friction=0.7, color=(0.88,0.94,0.98),
              finish=Finish(roughness=0.05, opacity=0.22)))

texroot = ROOT / "assets" / "tex"
wood = texroot / "wood_table_worn/wood_table_worn_diff_2k.jpg"
if not wood.exists():
    wood = texroot / "wood_table_001/wood_table_001_diff_2k.jpg"
ply = texroot / "plywood/plywood_diff_2k.jpg"
plaster = texroot / "PaintedPlaster017/PaintedPlaster017_2K-JPG_Color.jpg"
for name, p in PROPS.items():
    size = p["size"]
    at = p["at"]
    color = p["color"]
    shape = p["shape"]
    fin = Finish(roughness=0.45)
    if "wood" in p["finish"] or name in ("table_wood", "drawer_front", "table_apron"):
        fin = Finish(roughness=0.45, texture=aa.worn_board(str(ply if name == "lid_board" else wood)), uv_scale=(2, 2))
    if name == "plaster_wall":
        fin = Finish(
            roughness=0.96, texture=aa.tinted_plaster(str(plaster)) if plaster.exists() else "", uv_scale=(2, 2)
        )
    if shape == "box":
        scene.add(
            rss.slab(
                name=name,
                half=tuple(np.array(size) / 2),
                at=at,
                color=rgb(color),
                finish=fin,
                collide=name in ("table_wood", "lid_board", "lid_seam"),
            )
        )
    elif shape == "beveled_box":
        scene.add(mesh_spec(name, aa.beveled_board(size), at, color, fin, collide=True))
    elif shape == "plaster_mesh":
        scene.add(mesh_spec(name, aa.plaster_mesh(size), at, color, fin))
    elif name == "saucer":
        path = aa.obj(
            name,
            *aa.lathe(
                [
                    (0, 0),
                    (0.074, 0),
                    (0.091, 0.002),
                    (0.100, 0.008),
                    (0.099, 0.012),
                    (0.092, 0.014),
                    (0.085, 0.006),
                    (0.070, 0.005),
                    (0, 0.005),
                ]
            ),
        )
        scene.add(mesh_spec(name, path, at, color, Finish(roughness=0.22)))
    elif name == "cachepot":
        scene.add(mesh_spec(name, aa.cachepot(), at, color, Finish(roughness=0.24)))
    elif name == "terracotta":
        path = aa.obj(
            name,
            *aa.lathe(
                [
                    (r, z * 0.87)
                    for r, z in [
                        (0.046, 0),
                        (0.050, 0.004),
                        (0.067, 0.092),
                        (0.070, 0.096),
                        (0.070, 0.100),
                        (0.065, 0.100),
                        (0.064, 0.093),
                        (0.047, 0.008),
                        (0.046, 0),
                    ]
                ]
            ),
        )
        scene.add(mesh_spec(name, path, at, color, Finish(roughness=0.54)))
    elif name == "soil":
        scene.add(
            rss.post(
                name=name,
                radius=size[0] / 2,
                half_height=0.030,
                at=(0, 0, 0.158),
                color=rgb(color),
                finish=Finish(roughness=1, texture=aa.soil(), uv_scale=(7.6, 7.6)),
                collide=True,
            )
        )

# Terracotta walls: 32 short slabs follow the retained tapered mesh.
for j in range(32):
    theta = 2*np.pi*j/32
    q = Rotation.from_euler("z", theta).as_quat()
    scene.add(rss.slab(name=f"pot_wall_{j}", half=(0.004,0.0075,0.042),
        at=(0.059*np.cos(theta),0.059*np.sin(theta),0.167,*q), visible=False))

import anthurium_garden
anthurium_garden.add(scene, ROOT)

# Every leaf uses 111 cloth particles and a ten-capsule, eleven-node petiole.
# Separate roots prevent all stalks intersecting at a single point.
SETTLED_PATH = ROOT / "assets" / "plant_final_settled.json"
BAKED = json.loads(SETTLED_PATH.read_text()) if SETTLED_PATH.exists() else {}
COMPENSATION = (
    json.loads((ROOT / "assets" / "anthurium_sag_calibration.json").read_text())
    if (ROOT / "assets" / "anthurium_sag_calibration.json").exists()
    else BAKED
)
CORRECTION = (
    json.loads((ROOT / "assets" / "anthurium_outline_correction.json").read_text())
    if (ROOT / "assets" / "anthurium_outline_correction.json").exists()
    else {}
)
LEAF_COMPENSATION = json.loads((ROOT / "assets" / "anthurium_leaf_sag_calibration.json").read_text())
for k, b in enumerate(BLADES):
    name = b["name"]
    sp = b.get("spathe", False)
    if name in ao.OUTLINES:
        b.update(base=ao.OUTLINES[name]["base"], tip=ao.OUTLINES[name]["tip"])
    base = unproject(b["base"], b["depth"][0])
    tip = unproject(b["tip"], b["depth"][1])
    u = norm(tip - base)
    n = norm(np.array(b["normal"]) - u * np.dot(b["normal"], u))
    v = norm(np.cross(n, u))
    n = np.cross(u, v)
    R = np.column_stack((u, v, n))
    L = float(np.linalg.norm(tip - base) / 0.87)
    verts, faces, held = aa.leaf_mesh(L, b["width"], sp)
    points = verts @ R.T + base
    if name in ao.OUTLINES:
        points, n = ao.mesh(
            ao.OUTLINES[name], base, np.asarray(PHOTO["eye"]), CAM_FORWARD, CAM_RIGHT, CAM_UP, FOCAL, ORIGIN, sp
        )
        b["target_vertices"] = points.tolist()
        if not sp and name in LEAF_COMPENSATION:
            measured = LEAF_COMPENSATION[name]
            points -= 0.8 * (np.asarray(measured["particles"]) - np.asarray(measured["rest_vertices"]))
        # Compensate the measured gravity deflection, then solve the new mesh again.
        if sp and name in COMPENSATION and "rest_vertices" in COMPENSATION[name]:
            points -= 0.75 * (
                np.asarray(COMPENSATION[name]["particles"]) - np.asarray(COMPENSATION[name]["rest_vertices"])
            )
        if sp and name in CORRECTION and not ao.OUTLINES[name].get("interior_hold"):
            points -= np.asarray(CORRECTION[name])
        if not sp:
            held = [i * 11 + j for i in range(5) for j in (4, 5, 6)]
        if ao.OUTLINES[name].get("interior_hold"):
            held = np.argsort(np.linalg.norm(points - base, axis=1))[:15].tolist()
        _, spans, axes = np.linalg.svd(points - points.mean(0), full_matrices=False)
        extent = np.ptp((points - points.mean(0)) @ axes.T, axis=0)
        L = float(extent[0])
        b["width"] = float(extent[1])
    root = np.array((0.021 * np.cos(k * 2.4), 0.023 * np.sin(k * 2.4), 0.205))
    t = np.linspace(0, 1, 11)[:, None]
    nodes = root + (base - root) * t
    nodes[:, 2] += 0.022 * np.sin(np.pi * t[:, 0])
    stalk = name + "_stalk"
    # Preserve elastic rest geometry. The baked live shape is restored after build.
    b.update(
        size=(L, b["width"], 0.00045),
        at=tuple(points.mean(0)),
        quat=tuple(Rotation.from_matrix(R).as_quat()),
        length=L,
        vertices=points.tolist(),
        faces=faces.tolist(),
        held=held,
        nodes=nodes.tolist(),
        inferred_normal=n.tolist(),
    )
    PROPS[stalk] = {
        "shape": "rod",
        "size": (float(np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum()), 0.0029, 0.0029),
        "at": tuple(nodes[0]),
        "color": "#42691b",
        "finish": "satin living green stalk",
    }
    scene.add(
        rss.rod(
            name=stalk,
            entity="anthurium_petiole_v3",
            nodes=tuple(map(tuple, nodes)),
            color=rgb("#42691b"),
            finish=Finish(roughness=0.36),
            **ROD_PARAMS,
        )
    )
    # PLY loading preserves unique mesh vertex positions; OBJ triangles retain the same topology.
    path = aa.obj(name, points, faces)
    texture, scale = aa.blade_texture(name, points, b["color"], sp)
    held_targets = points[held].copy()
    scene.add(
        rss.sheet(
            name=name,
            entity="anthurium_leaf_v1",
            mesh="mesh",
            asset=rss.AssetSpec(path=path),
            size=(L, b["width"]),
            cells=(10, 10),
            at=(0, 0, 0),
            held_by=f"{stalk}_edge_body_9",
            particle_sets=(
                (
                    "held",
                    lambda rest, targets=held_targets: (
                        np.min(np.linalg.norm(rest[:, None, :] - targets[None, :, :], axis=2), axis=1) < 1e-5
                    ),
                ),
            ),
            material=rss.replace(LEAF_MATERIAL, edge_ke=60.0 if sp else (100.0 if name == "leaf_09" else (20.0 if name == "leaf_02" else 40.0))),
            settle_frames=0,
            color=rgb(b["color"]),
            finish=Finish(roughness=0.72 if sp else 0.75, texture=texture, uv_scale=scale),
        )
    )
    if sp:
        # Visual rigid spadix at the inferred spathe base. Its pose is included in PROPS.
        direction = norm(n * 0.4 + np.array((0, 0, 1)) * 0.9)
        a = base + n * 0.002
        c = a + direction * 0.023
        if name == "spathe_01":
            # Visible spike (235..272 px) is foreshortened; its lower end sits
            # in the upper spathe cup behind the rust-red foreground spathe.
            direction = norm(0.465 * norm(CAM_UP - CAM_RIGHT * (2 / 37)) - 0.885 * CAM_FORWARD)
            c = unproject((1241, 253.5), 0.0)
        quat = Rotation.align_vectors([direction], [[0, 0, 1]])[0].as_quat()
        sn = name + "_spadix"
        PROPS[sn] = {
            "shape": "capsule",
            "size": (0.0045, 0.0045, 0.050),
            "at": tuple(c),
            "quat": tuple(quat),
            "color": "#c8b74b",
            "finish": "pale yellow pollen",
        }
        scene.add(
            anthurium_material.spadix(
                held_by=f"{stalk}_edge_body_9",
                name=sn,
                radius=0.00225,
                half_height=0.023,
                at=tuple(c),
                quat=tuple(quat),
                static=True,
                color=rgb("#c8b74b"),
                finish=Finish(roughness=0.65),
            )
        )

scene.add(rss.liquid(name="water", material="water", size=(0.036,0.036,0.050),
              at=(CUP_AT[0],CUP_AT[1],0.022), particle_spacing=0.0025,
              containers=("cup", "table_wood", "room_floor", "garden_ground"), container_margin=0.005, settle_frames=0, render="surface", surface_voxel=0.001, finish=Finish(roughness=0.02, opacity=0.65, ior=1.33), color=(0.24,0.47,0.56)))

# The live page accepts an explicitly built sim. The stored state retains the
# elastic rest shapes above, avoiding another gravity sag on each rebuild.
import anthurium_state

sim = scene.build(control_mode="joint_position", **BUILD)
BAKED_OBJECTS = anthurium_state.install_initial_state(sim, BLADES, ORIGIN, BAKED) if BAKED else []

# Density isosurfaces omit isolated airborne particles. Reconstruct every water
# particle with volume-normalized ellipsoids, using its measured position only.
# 1.24 converts the MPM cube-equivalent radius to an equal-volume sphere.
from robosimstudio.core.render.particles import ParticleSurfacer
_water_surface = sim["water"].entity.surface
sim["water"].entity.surface = ParticleSurfacer(
    _water_surface.radius, device=_water_surface.device, world_count=1,
    voxel_size=0.001, kernel_radius=0.012, surface_method="particle_sdf",
    particle_sdf_radius_scale=1.24, anisotropic=True, anisotropy_ratio=8,
    anisotropy_min_neighbors=2, smooth_lambda=0.25,
    mesh_smooth_iterations=2, mesh_smooth_lambda=0.2,
)

from watering import routine

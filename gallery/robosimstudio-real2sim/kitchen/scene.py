# SPDX-License-Identifier: Apache-2.0
"""Kitchen twin of counter_a, metres in bench_075's work-surface frame.

Scale: foreground side plate 180 mm; the photo has no surveyed dimension.
Kitchen props are free rigid bodies; the spill is MPM granular. Closed lids travel with
jars. Table origin is world (0.6, 0, 0.75); +x points into the backsplash.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import trimesh
from kitchen_geometry import (
    ASSETS,
    asset,
    bevel_tile,
    blade,
    box,
    concrete,
    ellipsoid,
    fork,
    jar,
    kitchen_look,
    lathe,
    plate,
    rgb,
    rigid,
    rotate,
    teapot,
    tile_edge,
    tube,
)

import robosimstudio as rss
from robosimstudio.core.render.finish import Finish, register_color_finish
from robosimstudio.core.render.spec import RenderSpec

ROOT = Path(__file__).resolve().parent
TEX = ROOT / "assets" / "tex"

# The complete readable layout table. Mesh origins are at the supporting bottom,
# except box primitives, whose origins are at their centre. q is scalar-last.
# size is a body-local enclosing size in metres; pose is surface-frame xyz, xyzw.
# Rigid poses are baked from the live resting state (run/live_settled_19.json).
LAYOUT = [
    {
        "name": "table",
        "kind": "fixed",
        "shape": "cell_table",
        "size": [0.595, 1.3, 0.75],
        "pose": [0, 0, -0.375, 0, 0, 0, 1],
        "color": "#9a9e97",
        "finish": "bench_075 collision support; covered by concrete visual finish",
    },
    {
        "name": "counter",
        "kind": "visual",
        "shape": "counter",
        "size": [0.595, 1.3, 0.03],
        "pose": [0, 0, -0.0147, 0, 0, 0, 1],
        "color": "#898b85",
        "finish": "matte concrete; 0.3 mm visual depth bias above physical table",
    },
    {
        "name": "wall",
        "kind": "fixed",
        "shape": "wall",
        "size": [0.025, 2.6, 1.2],
        "pose": [0.3065, 0, 0.6, 0, 0, 0, 1],
        "color": "#d1d4d6",
        "finish": "painted white plaster",
    },
    {
        "name": "backsplash",
        "kind": "fixed",
        "shape": "tiles",
        "size": [0.005, 1.3099697000000001, 1.2099663],
        "pose": [0.291, 0, 0, 0, 0, 0, 1],
        "color": "#d2dade",
        "finish": "glossy bevelled ceramic / grey grout",
    },
    {
        "name": "spaghetti_jar",
        "kind": "rigid",
        "shape": "spaghetti",
        "size": [0.10717, 0.116195, 0.284022],
        "pose": [0.229415023, 0.328871012, -4.649e-06, 8e-07, -1.167e-06, -0.0, 1.0],
        "color": "#edf3eb",
        "finish": "clear glass / steel clip / dry pasta",
    },
    {
        "name": "oats_jar",
        "kind": "rigid",
        "shape": "oats",
        "size": [0.092593, 0.092593, 0.188129],
        "pose": [0.01106199, 0.260280013, -4.649e-06, 5.4e-08, -1e-09, 0.0, 1.0],
        "color": "#edf3eb",
        "finish": "clear glass / walnut lid / pale oats",
    },
    {
        "name": "coffee_jar",
        "kind": "rigid",
        "shape": "coffee",
        "size": [0.103117, 0.103117, 0.128153],
        "pose": [0.131028974, 0.147845998, -4.649e-06, 4.06e-07, -4.2e-07, 0.0, 1.0],
        "color": "#edf3eb",
        "finish": "clear glass / walnut lid / coffee",
    },
    {
        "name": "nuts_jar",
        "kind": "rigid",
        "shape": "nuts",
        "size": [0.109084, 0.11857, 0.098679],
        "pose": [
            0.228458011,
            0.082521997,
            -4.649e-06,
            3.143e-06,
            -2.826e-06,
            -0.0,
            1.0,
        ],
        "color": "#edf3eb",
        "finish": "clear glass / steel clip / walnuts",
    },
    {
        "name": "small_clip_jar",
        "kind": "rigid",
        "shape": "smalljar",
        "size": [0.078316, 0.085775, 0.096688],
        "pose": [0.233742976, 0.223311007, -4.649e-06, 1.4e-08, -6e-09, -0.0, 1.0],
        "color": "#edf3eb",
        "finish": "clear glass / steel clip / biscuits",
    },
    {
        "name": "teapot",
        "kind": "rigid",
        "shape": "teapot",
        "size": [0.111973, 0.144208, 0.14835],
        "pose": [-0.123548007, 0.260957986, -4.649e-06, -2.732e-06, -5e-09, -0.0, 1.0],
        "color": "#15190f",
        "finish": "matte cast iron",
    },
    {
        "name": "single_plate",
        "kind": "rigid",
        "shape": "plate",
        "size": [0.18, 0.18, 0.0065],
        "pose": [-0.092999971, 0.098842531, -4.649e-06, -1e-09, -4e-09, 0.0, 1.0],
        "color": "#0c0e08",
        "finish": "matte black stoneware",
    },
    {
        "name": "stack_plate_1",
        "kind": "rigid",
        "shape": "plate",
        "size": [0.255, 0.255, 0.0175],
        "pose": [0.160500014, -0.286164194, -2.3067e-05, 0.0, 0.0, 0.0, 1.0],
        "color": "#080904",
        "finish": "matte black stoneware",
    },
    {
        "name": "stack_plate_2",
        "kind": "rigid",
        "shape": "plate",
        "size": [0.255, 0.255, 0.0175],
        "pose": [0.160500014, -0.286164194, 0.016967118, 0.0, 0.0, -0.0, 1.0],
        "color": "#080904",
        "finish": "matte black stoneware",
    },
    {
        "name": "stack_plate_3",
        "kind": "rigid",
        "shape": "plate",
        "size": [0.255, 0.255, 0.0175],
        "pose": [0.160500014, -0.286164194, 0.033966303, 0.0, 0.0, 0.0, 1.0],
        "color": "#080904",
        "finish": "matte black stoneware",
    },
    {
        "name": "stack_plate_4",
        "kind": "rigid",
        "shape": "plate",
        "size": [0.255, 0.255, 0.0175],
        "pose": [0.160500014, -0.286164194, 0.050974667, 0.0, 0.0, -0.0, 1.0],
        "color": "#080904",
        "finish": "matte black stoneware",
    },
    {
        "name": "stack_plate_5",
        "kind": "rigid",
        "shape": "plate",
        "size": [0.195, 0.195, 0.0065],
        "pose": [0.160500014, -0.286164194, 0.067971647, 0.0, 0.0, 0.0, 1.0],
        "color": "#0c0e08",
        "finish": "matte black stoneware",
    },
    {
        "name": "mortar_saucer",
        "kind": "rigid",
        "shape": "saucer",
        "size": [0.102226, 0.102226, 0.022],
        "pose": [
            0.049181008,
            -0.073948853,
            -1.3649e-05,
            -7.0025e-05,
            2.8e-08,
            -0.0,
            0.999999998,
        ],
        "color": "#81502d",
        "finish": "oiled walnut",
    },
    {
        "name": "mortar",
        "kind": "rigid",
        "shape": "mortar",
        "size": [0.102723, 0.102723, 0.07],
        "pose": [
            0.049181008,
            -0.073945731,
            0.004780769,
            -0.000420329,
            9e-09,
            -0.0,
            0.999999912,
        ],
        "color": "#aa7a4a",
        "finish": "striped end-grain wood",
    },
    {
        "name": "pestle",
        "kind": "rigid",
        "shape": "pestle",
        "size": [0.025, 0.025, 0.1],
        "pose": [
            0.049181008,
            -0.05586144,
            0.021534562,
            -0.129481177,
            -1.5e-08,
            -2.7e-08,
            0.99158188,
        ],
        "color": "#a8784d",
        "finish": "warm hardwood",
    },
    {
        "name": "cutting_board",
        "kind": "rigid",
        "shape": "board",
        "size": [0.15720214, 0.295, 0.0221],
        "pose": [
            -0.12454859,
            -0.290022016,
            -8.345e-06,
            -1.939e-06,
            -3.05e-07,
            -0.244636533,
            0.969614855,
        ],
        "color": "#794125",
        "finish": "worn walnut",
    },
    {
        "name": "tong_left",
        "kind": "rigid",
        "shape": "tong",
        "size": [0.16627789074825006, 0.013, 0.008],
        "pose": [
            -0.157990283,
            -0.216343373,
            0.022085011,
            -2.8101e-05,
            2.2311e-05,
            0.927658098,
            -0.373430652,
        ],
        "color": "#d8c49e",
        "finish": "pale beech; straight rounded rigid blade",
    },
    {
        "name": "tong_right",
        "kind": "rigid",
        "shape": "tong",
        "size": [0.1685933698982964, 0.013, 0.008],
        "pose": [
            -0.135438716,
            -0.239642739,
            0.023884237,
            -0.011599518,
            0.000504839,
            0.952864239,
            -0.303174767,
        ],
        "color": "#d8c49e",
        "finish": "pale beech; straight rounded rigid blade",
    },
    {
        "name": "spatula",
        "kind": "rigid",
        "shape": "spatula",
        "size": [0.24, 0.06, 0.006],
        "pose": [
            -0.142229772,
            -0.422950566,
            0.022089601,
            1.0511e-05,
            1.0519e-05,
            0.449999799,
            0.893028656,
        ],
        "color": "#e1d4b4",
        "finish": "pale beech",
    },
    {
        "name": "fork_front",
        "kind": "rigid",
        "shape": "fork",
        "size": [0.1042339431936608, 0.019, 0.0045],
        "pose": [
            -0.172392672,
            -0.105025321,
            0.004025459,
            -0.015914305,
            0.01404363,
            0.361356155,
            0.932186269,
        ],
        "color": "#303022",
        "finish": "dark brushed steel",
    },
    {
        "name": "fork_board",
        "kind": "rigid",
        "shape": "fork",
        "size": [0.11416734766781457, 0.019, 0.0045],
        "pose": [
            -0.049655116,
            -0.302381605,
            0.022086203,
            -0.00012406,
            -0.000110062,
            0.779217234,
            0.626753919,
        ],
        "color": "#303022",
        "finish": "dark brushed steel",
    },
    {
        "name": "utensil_jar",
        "kind": "rigid",
        "shape": "utensiljar",
        "size": [0.101408, 0.101408, 0.10705],
        "pose": [
            0.170092607,
            -0.509635389,
            -7.391e-06,
            -3.588e-06,
            6.607e-06,
            6.3203e-05,
            0.999999998,
        ],
        "color": "#edf3eb",
        "finish": "clear open glass",
    },
    {
        "name": "whisk",
        "kind": "rigid",
        "shape": "whisk",
        "size": [0.072, 0.072, 0.225],
        "pose": [
            0.16230222,
            -0.525525212,
            0.006210208,
            -0.251723351,
            0.025517879,
            0.091636055,
            0.963113195,
        ],
        "color": "#6b7164",
        "finish": "stainless wire / wood handle",
    },
    {
        "name": "dish_brush",
        "kind": "rigid",
        "shape": "brush",
        "size": [0.047, 0.047, 0.212],
        "pose": [
            0.179066026,
            -0.521602392,
            0.002920091,
            4.539e-06,
            -1.443e-06,
            -2.171e-06,
            1.0,
        ],
        "color": "#c0a57d",
        "finish": "beech / natural bristles",
    },
    {
        "name": "long_brush",
        "kind": "rigid",
        "shape": "longbrush",
        "size": [0.024, 0.1, 0.225],
        "pose": [
            0.186066175,
            -0.506602407,
            0.002919793,
            1.658e-06,
            0.000196539,
            -2.374e-06,
            0.999999981,
        ],
        "color": "#c0a57d",
        "finish": "beech / natural bristles",
    },
    {
        "name": "bean_01",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [0.006926858, -0.052062061, 0.001717269, -2.645e-06, -0.0, 0.0, 1.0],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_02",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            0.00334909,
            -0.074923128,
            0.001717269,
            -2.424e-06,
            -1.715e-06,
            0.651833772,
            0.758361875,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_03",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            0.002163374,
            -0.119514801,
            0.001717269,
            -5.76e-07,
            -3.452e-06,
            0.988651762,
            0.150225477,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_04",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.026399052,
            -0.034054168,
            0.001717269,
            -1.598e-06,
            2.502e-06,
            -0.847677844,
            0.530511331,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_05",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.05405674,
            -0.028477373,
            0.001717269,
            -3.045e-06,
            6.63e-07,
            -0.297041355,
            0.954864615,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_06",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.070131338,
            -0.021442229,
            0.001717269,
            -2.808e-06,
            -1.55e-06,
            0.397148154,
            0.917754512,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_07",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.063183224,
            -0.058000296,
            0.001717269,
            -1.487e-06,
            -3.172e-06,
            0.899405416,
            0.437115429,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_08",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.048889911,
            -0.117327951,
            0.001717269,
            8.95e-07,
            -3.097e-06,
            0.967001378,
            -0.254771143,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_09",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.039410329,
            -0.137537554,
            0.001717269,
            -2.382e-06,
            1.739e-06,
            -0.567268547,
            0.823532875,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_10",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.049928939,
            -0.135668829,
            0.001717269,
            -3.088e-06,
            2.2e-08,
            0.106611692,
            0.994300733,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_11",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.068159318,
            -0.124260671,
            0.001717269,
            -1.963e-06,
            -2.379e-06,
            0.728969047,
            0.684546659,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_12",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.094205952,
            -0.050333709,
            0.001699209,
            0.000406184,
            0.000704356,
            0.999989731,
            0.004458234,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_13",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.091154313,
            -0.073339894,
            0.001717269,
            -1.806e-06,
            2.247e-06,
            -0.786287969,
            0.617860202,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_14",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.100322133,
            -0.069117859,
            0.001717269,
            -2.861e-06,
            5.99e-07,
            -0.193548695,
            0.98109067,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_15",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.114516324,
            -0.073387578,
            0.001717269,
            -3.048e-06,
            -1.711e-06,
            0.492728078,
            0.870183338,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_16",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.116250968,
            -0.086076766,
            0.001717269,
            -8.47e-07,
            -2.903e-06,
            0.940881075,
            0.338737071,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_17",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.126480734,
            -0.081247076,
            0.001717269,
            9.93e-07,
            -2.577e-06,
            0.934328605,
            -0.356412764,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_18",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.150052822,
            -0.094975837,
            0.001646221,
            -0.010377153,
            -0.002181654,
            -0.477551104,
            0.878539981,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_19",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.162300831,
            -0.099189483,
            0.001717269,
            -2.936e-06,
            -3.89e-07,
            0.211950446,
            0.977280414,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_20",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.1118981,
            0.006880781,
            0.001717269,
            -1.588e-06,
            -2.299e-06,
            0.79779512,
            0.602928641,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_21",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.125639743,
            0.005158859,
            0.001717269,
            1.6e-07,
            -2.856e-06,
            0.998026653,
            -0.062791719,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_22",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.12731967,
            -0.01068777,
            0.001717269,
            -1.939e-06,
            2.387e-06,
            -0.715935601,
            0.698166324,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_23",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.141274726,
            0.000775602,
            0.001717269,
            -3.312e-06,
            1.73e-07,
            -0.087849876,
            0.996133726,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_24",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.150005972,
            -0.001913752,
            0.001717269,
            -2.3e-06,
            -1.554e-06,
            0.582691608,
            0.812693355,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_25",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.156214839,
            -0.008342351,
            0.001717269,
            -8.36e-07,
            -2.729e-06,
            0.971632073,
            0.236497598,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_26",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.020488179,
            -0.237828299,
            0.023808777,
            1.449e-06,
            9.17e-07,
            0.891005852,
            -0.45399182,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
    {
        "name": "bean_27",
        "kind": "rigid",
        "shape": "bean",
        "size": [0.006, 0.0043, 0.0036],
        "pose": [
            -0.045919394,
            -0.245558858,
            0.023808777,
            -2.591e-06,
            -2.75e-07,
            -0.379777669,
            0.925077793,
        ],
        "color": "#654029",
        "finish": "satin roasted coffee",
    },
]
CAMERA = {
    "name": "photo_a",
    "width": 1920,
    "height": 1280,
    "fov_y": 41.2563366786,
    "eye": (-0.15, 0, 1.070),
    "look_at": (0.850, 0, 0.870),
}

register_color_finish(rgb("#d2dade"), Finish(roughness=0.18))
register_color_finish(rgb("#c1c5bc"), Finish(roughness=0.95))

# Finishes are mapped to authored mesh-material colours, via the public finish API.
walnut = str(TEX / "wood_table_worn/wood_table_worn_diff_2k.jpg")
lidwood = str(ASSETS / "lid_walnut_v3.png")
register_color_finish(rgb("#98643f"), Finish(roughness=0.56, texture=lidwood, uv_scale=(1, 1)))
oak = str(TEX / "oak_veneer_01/oak_veneer_01_diff_2k.jpg")
boardwood = str(ASSETS / "board_walnut_v4.png")
for c in ["#81502d", "#794125"]:
    register_color_finish(
        rgb(c),
        Finish(
            roughness=0.58,
            texture=(boardwood if c == "#794125" else walnut),
            uv_scale=(1.0, 1.0),
        ),
    )
for c in ["#aa7a4a", "#a8784d", "#c0a57d"]:
    register_color_finish(rgb(c), Finish(roughness=0.55, texture=oak, uv_scale=(1, 1)))
register_color_finish(rgb("#edf3eb"), Finish(roughness=0.045, opacity=0.003, ior=1.47))
register_color_finish(rgb("#f0f2e8"), Finish(roughness=0.08, opacity=0.30, ior=1.0))
register_color_finish(rgb("#d9dfd2"), Finish(roughness=0.08, opacity=0.13, ior=1.0))
register_color_finish(rgb("#b4b9af"), Finish(roughness=0.18, metallic=0.85))
register_color_finish(rgb("#878b7e"), Finish(roughness=0.22, metallic=0.82))
register_color_finish(rgb("#6b7164"), Finish(roughness=0.25, metallic=0.90))
register_color_finish(rgb("#e1dfd6"), Finish(roughness=0.90))
for col, tex in [
    ("#a27043", oak),
    ("#c7ac83", str(TEX / "kitchen_wood/kitchen_wood_diff_2k.jpg")),
    ("#80502d", walnut),
]:
    register_color_finish(rgb(col), Finish(roughness=0.58, texture=tex, uv_scale=(1, 1)))
register_color_finish(rgb("#15190f"), Finish(roughness=0.32))
for c in ["#080904", "#0c0e08", "#303022"]:
    register_color_finish(rgb(c), Finish(roughness=0.48))

# Sealed concrete as in the filmed sweep; standard robot mounts unchanged.
base_cell = rss.get_cell("bench_075")
cell = replace(
    base_cell,
    name="kitchen_counter",
    statics=tuple(replace(st, friction=0.4) if st.name == "table" else st for st in base_cell.statics),
)
from robosimstudio.scene.registry import register_cell

register_cell(cell)
rigid_backend = rss.backend("mujoco", nconmax=8192, njmax=32768, iterations=100, ls_iterations=50, enable_multiccd=True)
physics = rss.GRANULAR_60HZ.with_backend("rigid", rigid_backend).with_backend("articulated", rigid_backend)
physics = physics.with_backend("granular", rss.backend("mpm", step_hz=120.0, project_outside=True, project_gap=0.008))
physics = physics.with_(collision=replace(physics.collision, rigid_contact_max=32768))
scene = rss.Scene(
    cell=cell.name,
    name="kitchen",
    physics=physics,
    render=RenderSpec(
        environment="none",
        scenery=kitchen_look,
        ambient=(1.2, 1.2, 1.2),
        key_intensity=0.95,
        quality="final",
    ),
)
robot = rss.robot("yam")
# Keep the specified embodiment and standard near-edge mounts. A compact home
# pose parks the two forearms below the photographed prop region.
from robosimstudio.robots.registry import get_embodiment

robot = replace(
    robot,
    name=replace(
        get_embodiment("yam"),
        arms=tuple(
            replace(a, home_q=((1.57 if a.name == "left" else -1.57), 3.15, 3.15, 0, 0, 0))
            for a in get_embodiment("yam").arms
        ),
    ),
)
scene.add(robot)
scene.add(rss.CameraSpec(**CAMERA))
# The inaccurate secondary photo_b camera was removed after round-1 review.

MOVE = {
    **dict.fromkeys(("mortar_saucer", "mortar", "pestle"), (0.151, 0.014)),
    **dict.fromkeys(("cutting_board", "tong_left", "tong_right", "spatula", "fork_board"), (0, -0.12)),
    **{f"stack_plate_{i}": (0.005, -0.13) for i in range(1, 6)},
    **dict.fromkeys(("utensil_jar", "whisk", "dish_brush", "long_brush"), (0, -0.13)),
    "coffee_jar": (0, 0.02),
}
# Refit the teapot geometry and room, keeping photo_a fixed.
for row in LAYOUT:
    if row["name"] == "wall":
        row["size"] = [0.025, 8, 3.2]
        row["pose"][2] = 0.85
        row["color"] = "#d7dddf"
    if row["name"] == "backsplash":
        row["color"] = "#bcc1c4"
register_color_finish(rgb("#bcc1c4"), Finish(roughness=0.34))
RESTING_PROPS = json.loads((ASSETS / "resting_props.json").read_text())
for row in LAYOUT:
    if row["name"] in ("single_plate", "fork_front") or row["name"].startswith("bean_"):
        continue
    if row["name"] in MOVE:
        dx, dy = MOVE[row["name"]]
        row["pose"][0] += dx
        row["pose"][1] += dy
    if row["name"] in RESTING_PROPS:
        row["pose"] = RESTING_PROPS[row["name"]]
    if row["name"] == "teapot":
        row["pose"] = list(row["pose"])
        row["pose"][0] -= 0.005
        row["pose"][1] -= 0.006
    if row["name"].startswith("stack_plate_"):
        if row["name"] != "stack_plate_5":
            row["size"] = [row["size"][0] * 0.92, row["size"][1] * 0.92, row["size"][2]]
        row["pose"] = list(row["pose"])
        row["pose"][1] += 0.0850
    if row["name"] in ("mortar", "mortar_saucer", "pestle"):
        row["pose"] = list(row["pose"])
        row["pose"][0] -= 0.041
    n = row["name"]
    s = row["shape"]
    sx, sy, sz = row["size"]
    p = row["pose"]
    c = row["color"]
    parts = []
    colliders = []
    extra_meshes = []
    if s == "cell_table":
        continue
    if s == "counter":
        scene.add(
            rss.slab(
                name=n,
                half=(sx / 2, sy / 2, sz / 2),
                at=p[:3],
                color=rgb(c),
                collide=False,
                finish=Finish(roughness=0.91, texture=concrete(), uv_scale=(1.6, 1.6)),
            )
        )
        continue
    if s == "wall":
        scene.add(
            rss.slab(
                name=n,
                half=(sx / 2, sy / 2, sz / 2),
                at=p[:3],
                color=rgb(c),
                finish=Finish(roughness=0.93),
            )
        )
        continue
    if s == "tiles":
        # Tessellated 280 x 93.3 mm herringbone parquet rotated 45 degrees.
        tiles = []
        grout = []
        L = 0.28
        w = L / 3
        gap = 0.0011
        for i in range(-12, 13):
            for j in range(-8, 9):
                # Standard interlocking horizontal/vertical parquet pair.
                a = i * (L + w) + j * w
                b = i * (L - w) - j * w
                for u, v, du, dv in [(a, b, L, w), (a + L, b, w, L)]:
                    uc = u + du / 2
                    vc = v + dv / 2
                    yy = (uc - vc) / np.sqrt(2) - 0.38
                    zz = (uc + vc) / np.sqrt(2) + 0.55
                    if -0.86 < yy < 0.36 and -0.08 < zz < 1.17:
                        m = bevel_tile(du - gap, dv - gap, 0.004, 0.003)
                        m = rotate(m, 45, axis=(1, 0, 0))
                        m.apply_translation((0, yy, zz))
                        m = tile_edge(m)
                        if len(m.faces):
                            tiles.append(m)
                        g = rotate(box((0.001, du, dv)), 45, axis=(1, 0, 0))
                        g.apply_translation((0.0025, yy, zz))
                        g = tile_edge(g)
                        if len(g.faces):
                            grout.append(g)
        # Local origin on the wall plane; visual tile solids, one fixed body.
        parts = [
            (trimesh.util.concatenate(tiles), c),
            (trimesh.util.concatenate(grout), "#c1c5bc"),
        ]
        tile_asset = asset(n, parts, [box((0.002, 1.12, 1.10), (0, -0.24, 0.55))])
        scene.add(
            rss.StaticSpec(
                name=n, asset=tile_asset, color=rgb(c), finish=Finish(roughness=0.34), xform=(0.291, 0, 0, 0, 0, 0, 1)
            )
        )
        continue
    if s in ["spaghetti", "oats", "coffee", "nuts", "smalljar", "utensiljar"]:
        conf = {
            "spaghetti": (0.060, 0.355, "clip", "spaghetti", "#ddd1b3"),
            "oats": (0.057, 0.230, "wood", "oats", "#e1dfd6"),
            "coffee": (0.052, 0.131, "wood", "coffee", "#3f2319"),
            "nuts": (0.054, 0.120, "clip", "nuts", "#b49c70"),
            "smalljar": (0.047, 0.117, "clip", "biscuits", "#9a6840"),
            "utensiljar": (0.054, 0.139, "open", "", c),
        }
        r, h, style, fill, fc = conf[s]
        parts = jar(r, h, style, fill, fc, len(n))
        if s != "utensiljar":
            colliders = [box((r * 1.5, r * 1.5, h), (0, 0, h / 2))]
        else:
            colliders = [box((0.075, 0.075, 0.004), (0, 0, 0.002))]
            for a in np.arange(12) * 2 * np.pi / 12:
                m = box((0.011, 0.025, 0.134), (0.049, 0, 0.071))
                m = rotate(m, a * 180 / np.pi)
                colliders.append(m)
    elif s == "teapot":
        parts = teapot()
        colliders = [box((0.12, 0.12, 0.12), (0, 0, 0.06))]
    elif s == "plate":
        parts = [(plate(sx), c)]
        colliders = [box((sx * 0.68, sy * 0.68, 0.018), (0, 0, 0.009))]
    elif s == "saucer":
        parts = [
            (
                lathe(
                    [
                        (0, 0),
                        (0.047, 0),
                        (0.061, 0.006),
                        (0.070, 0.023),
                        (0.0705, 0.030),
                        (0.067, 0.032),
                        (0.060, 0.012),
                        (0.042, 0.007),
                        (0, 0.007),
                    ]
                ),
                c,
            )
        ]
        colliders = [box((0.085, 0.085, 0.007), (0, 0, 0.0035))]
        # Convex sectors preserve the saucer recess; a hull over the entire
        # saucer would fill the hollow beneath the mortar.
        profile = [
            (0.047, 0),
            (0.061, 0.006),
            (0.070, 0.023),
            (0.0705, 0.030),
            (0.067, 0.032),
            (0.060, 0.012),
            (0.042, 0.007),
        ]
        for i in range(24):
            angles = np.linspace(i * 2 * np.pi / 24, (i + 1) * 2 * np.pi / 24, 3)
            vertices = [(r * np.cos(a), r * np.sin(a), z) for a in angles for r, z in profile]
            extra_meshes.append(trimesh.convex.convex_hull(np.asarray(vertices)))
    elif s == "mortar":
        parts = [
            (
                lathe(
                    [
                        (0, 0),
                        (0.025, 0),
                        (0.039, 0.008),
                        (0.056, 0.040),
                        (0.065, 0.078),
                        (0.066, 0.095),
                        (0.064, 0.103),
                        (0.059, 0.105),
                        (0.054, 0.100),
                        (0.052, 0.079),
                        (0.044, 0.040),
                        (0.026, 0.026),
                        (0, 0.026),
                    ]
                ),
                c,
            )
        ]
        m = parts[0][0]
        bands = (np.arange(len(m.faces)) // 2 % 96) // 16
        parts = [
            (
                m.submesh([np.flatnonzero(bands % 3 == j)], append=True, repair=False),
                col,
            )
            for j, col in enumerate(["#a27043", "#c7ac83", "#80502d"])
        ]
        colliders = [box((0.050, 0.050, 0.026), (0, 0, 0.013))]
        for a in np.arange(12) * 2 * np.pi / 12:
            colliders.append(rotate(box((0.015, 0.027, 0.075), (0.053, 0, 0.063)), a * 180 / np.pi))
    elif s == "pestle":
        parts = [
            (
                lathe(
                    [
                        (0, 0),
                        (0.011, 0.003),
                        (0.016, 0.020),
                        (0.015, 0.040),
                        (0.010, 0.090),
                        (0.011, 0.145),
                        (0.012, 0.157),
                        (0.009, 0.166),
                        (0, 0.170),
                    ]
                ),
                c,
            )
        ]
        colliders = [
            box((0.016, 0.016, 0.025), (0, 0, 0.014)),
            box((0.010, 0.010, 0.115), (0, 0, 0.087)),
        ]
    elif s == "board":
        parts = [(box((sx, sy, sz), (0, 0, sz / 2), 0.007), c)]
        colliders = [box((sx - 0.008, sy - 0.008, sz), (0, 0, sz / 2))]
    elif s == "tong":
        # Each tong blade is a separate pickable utensil.
        parts = [(blade(sx, sy, sz), c)]
        colliders = [box((sx - 0.008, sy * 0.95, sz), (0, 0, sz / 2))]
    elif s == "spatula":
        parts = [
            (box((0.17, 0.017, 0.007), (-0.057, 0, 0.0035), 0.003), c),
            (box((0.115, 0.065, 0.007), (0.085, 0, 0.0035), 0.003), c),
        ]
        colliders = [
            box((0.17, 0.017, 0.007), (-0.057, 0, 0.0035)),
            box((0.115, 0.065, 0.007), (0.085, 0, 0.0035)),
        ]
    elif s == "fork":
        parts = fork()
        colliders = [box((0.158, 0.024, 0.005), (0.08, 0, 0.0025))]
    elif s == "bean":
        parts = [
            (ellipsoid((sx, sy, sz)), c),
            (
                tube(
                    [
                        (-sx * 0.30, 0, sz * 0.44),
                        (0, 0.0002, sz * 0.49),
                        (sx * 0.30, 0, sz * 0.44),
                    ],
                    0.00023,
                    6,
                ),
                "#392316",
            ),
        ]
        colliders = [box((sx * 0.74, sy * 0.8, sz))]
    elif s == "whisk":
        wire_colliders = []
        parts = [(tube([(0, 0, 0), (0, 0, 0.13)], 0.007), "#c0a57d")]
        for a in np.arange(6) * np.pi / 6:
            t = np.linspace(0, 2 * np.pi, 64)
            z = 0.13 + 0.14 * (1 - np.cos(t)) / 2
            rad = 0.040 * np.sin(t)
            pts = np.stack([rad * np.cos(a), rad * np.sin(a), z], 1)
            parts.append((tube(pts, 0.0008, 6), c))
            for j in range(0, len(pts) - 1, 3):
                wire_colliders.append(tube(pts[j : min(j + 4, len(pts))], 0.0008, 6))
        colliders = [box((0.014, 0.014, 0.13), (0, 0, 0.065)), *wire_colliders]
    elif s in ["brush", "longbrush"]:
        h = 0.205 if s == "brush" else 0.265
        parts = [(tube([(0, 0, 0), (0, 0, h - 0.02)], 0.008), c)]
        if s == "brush":
            parts.append((ellipsoid((0.048, 0.048, 0.034), (0, 0, h)), c))
            width = 0.047
            dep = 0.047
        else:
            parts.append((box((0.018, 0.110, 0.045), (0, 0, h - 0.003), 0.006), c))
            width = 0.024
            dep = 0.110
        br = []
        for x in np.arange(-width / 2, width / 2, 0.003):
            for y in np.arange(-dep / 2, dep / 2, 0.003):
                points = (
                    [(x, y, h + 0.007), (x * 1.08, y * 1.08, h + 0.036)]
                    if s == "brush"
                    else [(0.009, y, h + x), (0.034, y * 1.02, h + x)]
                )
                br.append(tube(points, 0.0008, 5))
        parts.append((trimesh.util.concatenate(br), "#cec6ac"))
        colliders = [box((0.014, 0.014, h - 0.03), (0, 0, (h - 0.03) / 2))]
    scene.add(
        rigid(
            n,
            parts,
            colliders,
            row["size"],
            extra_meshes=extra_meshes,
            pose=p[:3],
            quat=p[3:],
            mass=0.003
            if s == "bean"
            else 0.08
            if s in ["fork", "tong", "spatula", "whisk", "brush", "longbrush"]
            else 0.4,
            friction=0.7,
            ke=1e5,
            kd=1000,
            contact_gap=0.0002,
        )
    )


scene.add(
    rss.CameraSpec(name="room", width=1600, height=1000, eye=(-1.1, -1.25, 1.55), look_at=(0.55, 0, 0.55), fov_y=50)
)
from kitchen_room import add_room

add_room(scene)

from coffee_spill import add_spill

add_spill(scene)
scene.add(rss.CameraSpec(name="wide", width=1920, height=1080, eye=(-0.30, 0, 1.38), look_at=(0.52, 0, 0.87), fov_y=39))
import sweep

routine = sweep.routine

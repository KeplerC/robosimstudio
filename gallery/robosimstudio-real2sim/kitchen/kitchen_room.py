"""Kitchen cabinet enclosure and tiled room, in the bench's surface frame."""

import robosimstudio as rss
from robosimstudio.core.render.finish import Finish


def add_room(scene):
    def slab(name, half, at, color, rough=0.7):
        scene.add(rss.slab(name=name, half=half, at=at, color=color, finish=Finish(roughness=rough), friction=0.6))

    # Light painted cabinet carcass, inset toe-kick and four inset fronts.
    slab("cabinet_case", (0.292, 0.649, 0.315), (0.003, 0, -0.345), (0.85, 0.84, 0.80))
    slab("cabinet_plinth", (0.265, 0.638, 0.045), (0.027, 0, -0.705), (0.61, 0.59, 0.54))
    for i in range(4):
        y = -0.4875 + i * 0.325
        slab(f"cabinet_front_{i}", (0.012, 0.160, 0.292), (-0.300, y, -0.347), (0.90, 0.89, 0.85), 0.5)
        slab(f"cabinet_pull_{i}", (0.017, 0.052, 0.006), (-0.326, y, -0.12), (0.35, 0.33, 0.29), 0.28)
    slab("worktop_edge", (0.012, 0.661, 0.014), (-0.298, 0, -0.015), (0.76, 0.76, 0.72), 0.82)
    slab("cabinet_left_cheek", (0.292, 0.012, 0.315), (0.003, 0.661, -0.345), (0.90, 0.89, 0.85), 0.5)
    slab("cabinet_right_cheek", (0.292, 0.012, 0.315), (0.003, -0.661, -0.345), (0.90, 0.89, 0.85), 0.5)

    # Stock mount plate bottoms are 12 mm above the work-surface frame.
    # These metal edge brackets bridge the cabinet/worktop to the catalog mounts.
    for side, y in (("left", 0.31), ("right", -0.31)):
        slab(f"mount_pad_{side}", (0.060, 0.110, 0.022), (-0.3475, y, -0.010), (0.37, 0.37, 0.35), 0.35)
        slab(f"mount_bracket_{side}", (0.018, 0.080, 0.065), (-0.327, y, -0.097), (0.37, 0.37, 0.35), 0.35)

    # Large physical walls close the room in every declared camera.
    slab("side_wall_left", (2.4, 0.04, 1.6), (-2.05, 3.0, 0.85), (0.87, 0.87, 0.84), 0.9)
    slab("side_wall_right", (2.4, 0.04, 1.6), (-2.05, -3.0, 0.85), (0.87, 0.87, 0.84), 0.9)
    slab("floor_base", (4, 4, 0.018), (-1, 0, -0.772), (0.58, 0.56, 0.51), 0.9)
    # Separate ceramic tiles and fine grout lines are geometry, no photo texture.
    for i in range(11):
        for j in range(14):
            tint = ((i * 17 + j * 13) % 7 - 3) * 0.004
            slab(
                f"floor_tile_{i}_{j}",
                (0.298, 0.298, 0.004),
                (-5.7 + i * 0.6, -3.9 + j * 0.6, -0.750),
                (0.72 + tint, 0.70 + tint, 0.65 + tint),
                0.75,
            )

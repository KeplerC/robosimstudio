import containment  # noqa: F401 -- registers the scenery entity before Scene.add
import numpy as np
from plant_assets import (
    canvas_texture,
    lathe,
    make_canvas,
    make_chair,
    make_clock,
    make_pot,
    make_saucer,
    make_shade,
    rgb,
    soil_texture,
    wood_texture,
)
from plant_layout import LAYOUT

import robosimstudio as rss
from robosimstudio.core.render import Finish


def add_room(scene):
    WOOD = wood_texture()
    for row in LAYOUT:
        name = row["name"]
        kind = row["kind"]
        at = row.get("at", (0, 0, 0))
        color = rgb(row["color"])
        row.setdefault("quat", [0, 0, 0, 1])
        finish = Finish(roughness=row["roughness"])
        if name == "wall":
            color = (0.72, 0.75, 0.79)
        if kind in ("cloth", "rod", "cell", "articulated"):
            continue
        if name == "tabletop":
            finish = Finish(roughness=0.07, texture=WOOD, uv_scale=(0.75, 0.75))
            scene.add(
                rss.StaticSpec(
                    name=name,
                    entity="plant_rigid_scenery",
                    half_extents=tuple(np.array(row["size"]) / 2),
                    xform=(*at, 0, 0, 0, 1),
                    color=color,
                    finish=finish,
                    collide=True,
                )
            )
        elif name in ("pot", "saucer"):
            asset = make_pot() if name == "pot" else make_saucer()
            scene.add(
                rss.StaticSpec(
                    name=name,
                    entity="plant_rigid_scenery",
                    asset=rss.AssetSpec(path=asset),
                    xform=(*at, 0, 0, 0, 1),
                    color=color,
                    finish=finish,
                    collide=True,
                )
            )
        elif name == "soil":
            finish = Finish(roughness=0.95, texture=soil_texture(), uv_scale=(8, 8))
            scene.add(
                rss.StaticSpec(
                    name=name,
                    entity="plant_rigid_scenery",
                    asset=rss.AssetSpec(
                        path=lathe(
                            "soil_volume", [(0, 0.0195), (0.045, 0.0195), (0.057, 0.1225), (0.057, 0.1245), (0, 0.1245)]
                        )
                    ),
                    xform=(at[0], at[1], 0, 0, 0, 0, 1),
                    color=color,
                    finish=finish,
                    collide=True,
                )
            )
        elif name.startswith("canvas"):
            asset = make_canvas(name, row["size"][1], row["size"][2])
            finish = Finish(
                roughness=0.91,
                texture=canvas_texture(name, name.endswith("right")),
                uv_scale=(1 / row["size"][1], 1 / row["size"][2]),
                uv_offset=(0.5, 0.5),
            )
            scene.add(
                rss.StaticSpec(
                    name=name,
                    asset=rss.AssetSpec(path=asset),
                    xform=(*at, 0, 0, 0, 1),
                    color=(1, 1, 1),
                    finish=finish,
                    collide=False,
                )
            )
        elif name.endswith("_inside"):
            scene.add(
                rss.post(
                    name=name,
                    radius=0.122,
                    half_height=0.001,
                    at=(*at, *row["quat"]),
                    color=color,
                    finish=finish,
                    collide=False,
                )
            )
        elif name.startswith("lamp"):
            quat = [0, np.sin(0.13 / 2), 0, np.cos(0.13 / 2)]
            row["quat"] = quat
            asset = make_shade(name)
            scene.add(
                rss.StaticSpec(
                    name=name,
                    asset=rss.AssetSpec(path=asset),
                    xform=(*at, *quat),
                    color=color,
                    finish=finish,
                    collide=False,
                )
            )
        elif name == "clock":
            asset, tex = make_clock()
            finish = Finish(roughness=0.7, texture=tex, uv_scale=(6.25, 6.25), uv_offset=(0.5, 0.5))
            scene.add(
                rss.StaticSpec(
                    name=name,
                    asset=rss.AssetSpec(path=asset),
                    xform=(*at, 0, 0, 0, 1),
                    color=(1, 1, 1),
                    finish=finish,
                    collide=False,
                )
            )
        elif name == "chair":
            q = [np.sin(0.19), 0, 0, np.cos(0.19)]
            row["quat"] = q
            asset = make_chair()
            finish = Finish(roughness=0.3, texture=WOOD, uv_scale=(2, 2))
            scene.add(
                rss.StaticSpec(
                    name=name,
                    asset=rss.AssetSpec(path=asset),
                    xform=(*at, *q),
                    color=color,
                    finish=finish,
                    collide=False,
                )
            )
        else:
            scene.add(
                rss.slab(
                    name=name, half=tuple(np.array(row["size"]) / 2), at=at, color=color, finish=finish, collide=True
                )
            )
        row["finish"] = {"roughness": finish.roughness, "texture": finish.texture, "uv_scale": list(finish.uv_scale)}

    # Wooden dining table: the calibrated top stays at the same height.
    wood_finish = Finish(roughness=0.24, texture=WOOD, uv_scale=(0.75, 0.75))
    scene.add(
        rss.StaticSpec(
            name="table",
            entity="plant_rigid_scenery",
            half_extents=(0.2975, 0.65, 0.018),
            xform=(0, 0, -0.019, 0, 0, 0, 1),
            color=(0.58, 0.39, 0.23),
            finish=wood_finish,
        )
    )
    for ix, x in enumerate([-0.245, 0.245]):
        for iy, y in enumerate([-0.59, 0.59]):
            scene.add(
                rss.slab(
                    name=f"table_leg_{ix}_{iy}",
                    half=(0.025, 0.025, 0.356),
                    at=(x, y, -0.394),
                    color=(0.50, 0.32, 0.18),
                    finish=wood_finish,
                )
            )
    for x in [-0.26, 0.26]:
        scene.add(
            rss.slab(
                name=f"table_apron_x{x}",
                half=(0.015, 0.60, 0.05),
                at=(x, 0, -0.085),
                color=(0.53, 0.35, 0.20),
                finish=wood_finish,
            )
        )
    for y in [-0.61, 0.61]:
        scene.add(
            rss.slab(
                name=f"table_apron_y{y}",
                half=(0.26, 0.015, 0.05),
                at=(0, y, -0.085),
                color=(0.53, 0.35, 0.20),
                finish=wood_finish,
            )
        )
    # Complete side chair, with the reference's original back rail retained.
    rail = np.array(next(r["at"] for r in LAYOUT if r["name"] == "chair"))
    cx, cy = rail[0] + 0.10, rail[1]
    scene.add(
        rss.slab(
            name="chair_seat",
            half=(0.16, 0.17, 0.018),
            at=(cx, cy, -0.285),
            color=(0.49, 0.30, 0.17),
            finish=wood_finish,
        )
    )
    for i, x in enumerate([cx - 0.13, cx + 0.13]):
        for j, y in enumerate([cy - 0.14, cy + 0.14]):
            scene.add(
                rss.slab(
                    name=f"chair_leg_{i}_{j}",
                    half=(0.015, 0.015, 0.2235),
                    at=(x, y, -0.5265),
                    color=(0.49, 0.30, 0.17),
                    finish=wood_finish,
                )
            )
    for j, side in enumerate([-1, 1]):
        y = cy + side * 0.077 * np.cos(0.38)
        top = rail[2] + side * 0.077 * np.sin(0.38)
        scene.add(
            rss.slab(
                name=f"chair_back_post_{j}",
                half=(0.013, 0.013, (top + 0.285) / 2),
                at=(rail[0], y, (top - 0.285) / 2),
                color=(0.49, 0.30, 0.17),
                finish=wood_finish,
            )
        )

    scene.add(
        rss.StaticSpec(
            name="table_liquid_boundary",
            entity="plant_liquid_boundary",
            half_extents=(0.2975, 0.65, 0.018),
            xform=(0, 0, -0.017, 0, 0, 0, 1),
            visible=False,
        )
    )

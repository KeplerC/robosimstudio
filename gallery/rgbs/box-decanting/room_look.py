"""Procedural room adjustments from the main view; no clip pixels."""

from robosimstudio.core.render.room import lab
from robosimstudio.core.render.scenery import register_scenery


class RoomKit:
    def __init__(self, kit):
        self.kit = kit

    def __getattr__(self, name):
        return getattr(self.kit, name)

    def box(self, name, center, half, **kwargs):
        if name.startswith("slat_"):
            i = int(name.split("_")[-1])
            if i >= 23:
                return
            center = (center[0], -1.5 + i * 0.135 + 0.0375, center[2])
            half = (half[0], 0.0375, half[2])
        if name == "wall_backing":
            kwargs["color"] = (0.25, 0.27, 0.28)
        if name.startswith(("lounge_wall", "lounge_side")):
            kwargs["color"] = (0.70, 0.69, 0.66)
        self.kit.box(name, center, half, **kwargs)


@register_scenery("decant_room")
def decant_room(kit):
    lab(RoomKit(kit))

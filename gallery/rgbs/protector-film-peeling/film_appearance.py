"""The stock tape material with a translucent display finish on its handle."""

from dataclasses import replace

from robosimstudio.scene.registry import register_entity
from robosimstudio_materials.tape import EntityTape


@register_entity("protector_film")
class ProtectorFilm(EntityTape):
    def apply_display(self, builder):
        super().apply_display(builder)
        for shape in self.tab_shapes:
            self.finishes[shape] = replace(self.spec.finish, texture=None, roughness_texture=None, roughness=0.22)

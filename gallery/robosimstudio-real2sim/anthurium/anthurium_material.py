"""Stock rod with fingertip collider preprocessing performed once per builder.

The shipped EntityRod repeats this preprocessing for every rod. Sixteen rooted
plant stalks share the same robot, so only the first needs to do it. Geometry,
joints, anchoring, integration and reads otherwise remain the stock rod kind.
"""

import newton

import robosimstudio as rss
from robosimstudio_materials.rod.entity import EntityRod
from robosimstudio.domains.static.entity import EntityStatic


@rss.register_entity("anthurium_hidden_fixture")
class HiddenFixture(EntityStatic):
    """Keep the stock bench collision volume, draw the wooden furniture instead."""
    def build(self, builder, episode):
        start=len(builder.shape_flags)
        super().build(builder, episode)
        for i in range(start,len(builder.shape_flags)):
            builder.shape_flags[i]=int(builder.shape_flags[i]) & ~int(newton.ShapeFlags.VISIBLE)


@rss.register_entity("anthurium_petiole_v3")
class AnthuriumPetiole(EntityRod):
    def _decompose_working_tips(self, builder):
        pass  # The catalog robot must retain its shipped colliders.

    def build(self, builder, episode):
        super().build(builder, episode)
        # The basal leaf patch is embedded in its petiole. A kinematic held
        # patch must not repel its own holder; otherwise it chases it forever.
        earlier = getattr(builder, "_anthurium_stalk_shapes", [])
        own = list(range(*self._rod_spans[2]))
        for a in own:
            for b in earlier:
                builder.add_shape_collision_filter_pair(a, b)
        builder._anthurium_stalk_shapes = earlier + own
        for shape in own:
            builder.shape_flags[shape] = int(builder.shape_flags[shape]) & ~int(newton.ShapeFlags.COLLIDE_PARTICLES)


# A pollen spike is visual, carried by the terminal petiole capsule. It adds
# neither mass nor collision to the plant, and follows on the GPU each substep.
from dataclasses import dataclass, fields

import warp as wp

from robosimstudio.domains.rigid.entity import EntityRigid


@dataclass(frozen=True, kw_only=True)
class SpadixSpec(rss.ObjectSpec):
    entity: str = "anthurium_spadix_v1"
    held_by: str = ""


@wp.kernel
def _follow_spadix(
    src: int, dst: int, offset: wp.transform, q: wp.array[wp.transform], qd: wp.array[wp.spatial_vector]
):
    q[dst] = wp.transform_multiply(q[src], offset)
    qd[dst] = qd[src]


@rss.register_entity("anthurium_spadix_v1")
class Spadix(EntityRigid):
    def build(self, builder, episode):
        first = len(builder.joint_type)
        super().build(builder, episode)
        if len(builder.joint_type) > first:
            builder.add_articulation(list(range(first, len(builder.joint_type))), label=self.name + '_root')

    def bind(self, model):
        super().bind(model)
        self.holder = next(i for i, n in enumerate(model.body_label) if str(n).split("/")[-1] == self.spec.held_by)
        poses = model.body_q.numpy()
        a = wp.transform(*[float(x) for x in poses[self.holder]])
        b = wp.transform(*[float(x) for x in poses[self.body]])
        self.offset = wp.transform_multiply(wp.transform_inverse(a), b)
        flags = model.shape_flags.numpy()
        flags[self.shapes] = int(newton.ShapeFlags.VISIBLE | newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
        model.shape_flags.assign(flags)

    def contribute_hooks(self, hooks):
        hooks.add("post_solve", self.follow, name="spadix:" + self.name)

    def follow(self, state, dt):
        wp.launch(
            _follow_spadix,
            dim=1,
            inputs=[self.holder, self.body, self.offset],
            outputs=[state.body_q, state.body_qd],
            device=self.model.device,
        )


def spadix(*, held_by, **kwargs):
    spec = rss.capsule(**kwargs)
    return SpadixSpec(
        **{f.name: getattr(spec, f.name) for f in fields(rss.ObjectSpec) if f.name != "entity"}, held_by=held_by
    )


from robosimstudio_materials.cloth import EntityHeldCloth


@wp.kernel
def _air_damping(indices: wp.array[wp.int32], rate: float, dt: float, velocity: wp.array[wp.vec3]):
    p = indices[wp.tid()]
    velocity[p] = velocity[p] * wp.exp(-rate * dt)


@rss.register_entity("anthurium_leaf_v1")
class AnthuriumLeaf(EntityHeldCloth):
    def bind(self, model):
        super().bind(model)
        held = set(self.held.tolist())
        self.free_particles = wp.array(
            [int(i) for i in self.particles if i not in held], dtype=wp.int32, device=model.device
        )

    def contribute_hooks(self, hooks):
        super().contribute_hooks(hooks)
        hooks.add("post_solve", self.damp, name="leaf_air:" + self.name)

    def damp(self, state, dt):
        wp.launch(
            _air_damping,
            dim=len(self.free_particles),
            inputs=[self.free_particles, 2.0, dt],
            outputs=[state.particle_qd],
            device=self.model.device,
        )


@rss.register_entity('anthurium_glass_cup')
class GlassCup(EntityRigid):
    """The hollow cup's five slabs collide; its duplicate display mesh does not."""
    def bind(self, model):
        super().bind(model)
        flags = model.shape_flags.numpy()
        for shape in self.shapes:
            if flags[shape] & int(newton.ShapeFlags.VISIBLE) and not flags[shape] & int(newton.ShapeFlags.COLLIDE_SHAPES):
                flags[shape] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        model.shape_flags.assign(flags)

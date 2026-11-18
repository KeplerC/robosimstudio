"""Scene-local three-domain proxy composition and water/lamina contact.

Newton's proxy machinery is indexed by island name. This subclass initializes
its mappings for three entries, keeping the stock stepping implementation.
The extra lamina contact is a frictionless, inelastic particle/triangle impulse;
its reaction is distributed barycentrically to the live VBD vertices.
"""
import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy
from robosimstudio.scene.physics.couplers import register_coupler_builder, _pipeline_factory, _null_pipeline


@wp.kernel
def contact(mesh: wp.uint64, water: wp.array[int], old: wp.array[wp.vec3],
            q: wp.array[wp.vec3], v: wp.array[wp.vec3], im: wp.array[float],
            dq: wp.array[wp.vec3], dv: wp.array[wp.vec3], impulse: wp.array[float]):
    i = water[wp.tid()]
    p = q[i]
    travel = p - old[i]
    length = wp.length(travel)
    face = int(-1)
    u = float(0.0)
    w = float(0.0)
    n = wp.vec3(0.0)
    point = p
    if length > 1.0e-8:
        hit = wp.mesh_query_ray(mesh, old[i], travel / length, length + 0.0015)
        if hit.result:
            face = hit.face
            u = hit.u
            w = hit.v
            n = hit.normal
            if wp.dot(n, travel) > 0.0:
                n = -n
            point = wp.mesh_eval_position(mesh, face, u, w)
    if face < 0:
        near = wp.mesh_query_point_no_sign(mesh, p, 0.002)
        if near.result:
            point = wp.mesh_eval_position(mesh, near.face, near.u, near.v)
            gap = p - point
            if wp.length(gap) < 0.0015 and wp.length(gap) > 1.0e-8:
                face = near.face
                u = near.u
                w = near.v
                n = wp.normalize(gap)
    if face >= 0:
        a = wp.mesh_get_index(mesh, face * 3)
        b = wp.mesh_get_index(mesh, face * 3 + 1)
        c = wp.mesh_get_index(mesh, face * 3 + 2)
        z = 1.0 - u - w
        rel = v[i] - (u * v[a] + w * v[b] + z * v[c])
        vn = wp.dot(rel, n)
        if vn < 0.0:
            eff = im[i] + u*u*im[a] + w*w*im[b] + z*z*im[c]
            j = -vn / wp.max(eff, 1.0e-8)
            wp.atomic_add(dv, i, j * im[i] * n)
            wp.atomic_add(dv, a, -j * u * im[a] * n)
            wp.atomic_add(dv, b, -j * w * im[b] * n)
            wp.atomic_add(dv, c, -j * z * im[c] * n)
            wp.atomic_add(dq, i, point + 0.0015 * n - p)
            wp.atomic_add(impulse, 0, j)


@wp.kernel
def apply(q: wp.array[wp.vec3], v: wp.array[wp.vec3], dq: wp.array[wp.vec3], dv: wp.array[wp.vec3]):
    i = wp.tid()
    q[i] = q[i] + dq[i]
    v[i] = v[i] + dv[i]


@wp.kernel
def swept_static_contact(mesh: wp.uint64, water: wp.array[int], old: wp.array[wp.vec3],
                         q: wp.array[wp.vec3], v: wp.array[wp.vec3],
                         friction: wp.array[float], hits: wp.array[int]):
    """Continuous impact against the actual fixed collider triangles.

    The MPM grid solve has a finite SDF band. This impact solve catches a
    segment that traverses it, with zero restitution and Coulomb friction.
    The fixed room/table receive the reaction; no particle is removed.
    """
    i=water[wp.tid()]
    travel=q[i]-old[i]
    length=wp.length(travel)
    if length>1.0e-8:
        direction=travel/length
        hit=wp.mesh_query_ray(mesh,old[i],direction,length+0.0005)
        if hit.result and wp.dot(hit.normal,direction)<-1.0e-5:
            n=hit.normal
            vn=wp.dot(v[i],n)
            if vn<0.0:
                tangent=v[i]-vn*n
                speed=wp.length(tangent)
                mu=wp.min(friction[hit.face],1.0)
                v[i]=tangent*wp.max(0.0,1.0+mu*vn/wp.max(speed,1.0e-8))
                q[i]=old[i]+direction*hit.t+0.0005*n
                wp.atomic_add(hits,0,1)


class PlantWaterSolver(SolverCoupledProxy):
    def __init__(self, model, entries, coupling, water_ids):
        self.shared_static = set(np.flatnonzero(model.shape_body.numpy() < 0).tolist())
        self.global_shape_count = model.shape_count
        self._validate_config(entries, coupling)
        bs = {e.name: set(map(int, e.bodies)) for e in entries}
        ps = {e.name: set(map(int, e.particles)) for e in entries}
        js = {e.name: set(map(int, e.joints)) for e in entries}
        self._proxy_mappings = self._build_proxy_mappings(model, coupling, bs)
        self._proxy_particle_mappings = self._build_proxy_particle_mappings(model, coupling, ps)
        self._proxy_joint_mappings = self._build_proxy_joint_mappings(model, coupling, bs, js, self._proxy_body_sets_by_destination())
        self._proxy_collision_configs = self._build_proxy_collision_configs(coupling)
        self._proxy_groups = self._build_proxy_groups()
        SolverCoupled.__init__(self, model=model, entries=entries, coupling=coupling)
        self._clean_mpm_colliders()
        self._refresh_mpm_inertia()
        self.water_ids = wp.array(water_ids, dtype=int, device=model.device)
        self.old_points = wp.empty_like(model.particle_q)
        self.mesh_points = wp.clone(model.particle_q)
        self.lamina = wp.Mesh(points=self.mesh_points, indices=wp.array(model.tri_indices.numpy().reshape(-1), dtype=int, device=model.device))
        self.dq_contact = wp.zeros_like(model.particle_q)
        self.dv_contact = wp.zeros_like(model.particle_q)
        self.contact_impulse = wp.zeros(1, dtype=float, device=model.device)
        self.static_contact_hits = wp.zeros(1,dtype=int,device=model.device)
        self.inverse_mass = model.particle_inv_mass

    def _clean_mpm_colliders(self):
        """Use physical surfaces, not the renderer's overlapping decoration.

        StaticSpec(collide=False) clears rigid collision only in this checkout.
        Its residual particle flag used to put visual boards, plant pots and
        foliage in the MPM's nearest-surface query. The stock bench's solid
        display plinth also conflicts with its dressed tabletop and the floor.
        """
        import newton
        for entry in self._entries.values():
            mpm = getattr(entry.solver, '_mpm_model', None)
            if mpm is None:
                continue
            view=entry.view
            flags=view.shape_flags.numpy().copy()
            for i,label in enumerate(view.shape_label):
                if not flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES) or str(label) in ('table','ground_plane'):
                    flags[i] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
            view.shape_flags=wp.array(flags,dtype=int,device=view.device)
            mpm.setup_collider(model=view,body_mass=view.body_mass,body_inv_inertia=view.body_inv_inertia)
            self.static_collision_mesh=mpm._collider_meshes[0]
            count=len(self.static_collision_mesh.indices)//3
            materials=mpm.collider.face_material_index.numpy()[:count]
            self.static_friction=wp.array(mpm.collider.material_friction.numpy()[materials],dtype=float,device=view.device)

    def _refresh_mpm_inertia(self):
        # Proxy initialization replaces the ModelView inertia arrays after MPM
        # has cached them. Keep its collision solve on the same effective mass
        # used to harvest reaction forces. The source rigid mass is unchanged.
        for entry in self._entries.values():
            mpm = getattr(entry.solver, '_mpm_model', None)
            if mpm is not None:
                mpm.collider_body_mass = wp.clone(entry.view.body_mass)
                mpm.collider_body_inv_inertia = wp.clone(entry.view.body_inv_inertia)
                mpm.notify_collider_changed()

    def notify_model_changed(self, flags):
        from newton import ModelFlags
        super().notify_model_changed(flags)
        if flags & ModelFlags.SHAPE_PROPERTIES:
            self._clean_mpm_colliders()
        self._refresh_mpm_inertia()

    def _build_owner_map(self, count, owned_by_entry):
        # Scenery has no state to reconcile; both deformable views need it.
        owner = [-1] * count
        for entry_idx, ids in enumerate(owned_by_entry):
            for i in map(int, ids):
                if owner[i] != -1:
                    if count == self.global_shape_count and i in self.shared_static:
                        continue
                    raise ValueError(f"Dynamic index {i} has two owners")
                owner[i] = entry_idx
        return owner

    def _step_proxy(self, state_in, control, contacts, dt, iteration_restart=False):
        """Advance each island once, summing feedback from every destination.

        The shipped two-entry method advances its source once per channel.
        A branched three-entry graph instead needs this topological pass.
        """
        from newton import StateFlags
        from newton._src.solvers.coupled.proxy_utils import sync_proxy_states_kernel
        pending=set(self._entries)
        done=set()
        while pending:
            ready=[name for name in pending if all(a in done for a,b in self._proxy_groups if b==name)]
            if not ready:
                raise RuntimeError('The plant coupling must be acyclic')
            for name in sorted(ready):
                entry=self._entries[name]
                outgoing=[p for p in self._proxy_mappings if p.src_name==name]
                incoming=[p for p in self._proxy_mappings if p.dst_name==name]
                if entry.has_body_force_input:
                    self._clear_body_force_input(entry)
                    self._add_body_force_input(entry,entry.body_local_to_global,state_in.body_f)
                    for proxy in outgoing:
                        self._add_body_force_input(entry,proxy.source_local_to_proxy_global,proxy.coupling_forces)
                    self._notify_input_state_update(entry,StateFlags.BODY_F,dt=dt)
                if entry.has_particle_force_input:
                    self._clear_particle_force_input(entry)
                    self._add_particle_force_input(entry,entry.particle_local_to_global,state_in.particle_f)
                    self._notify_input_state_update(entry,StateFlags.PARTICLE_F,dt=dt)
                for proxy in incoming:
                    self._stash_proxy_feedback(proxy)
                    src=self._entries[proxy.src_name]
                    wp.launch(sync_proxy_states_kernel,dim=proxy.source_local_to_proxy_local.shape[0],
                        inputs=[src.state_1.body_q,src.state_1.body_qd,proxy.source_local_to_proxy_local,
                                entry.state_0.body_q,entry.state_0.body_qd],device=self.model.device)
                    self._notify_input_state_update(entry,StateFlags.BODY_Q|StateFlags.BODY_QD,dt=dt)
                    wp.copy(proxy.proxy_qd_before,entry.state_0.body_qd)
                    proxy.coupling_forces.zero_()
                    entry.solver.coupling_rewind_proxy_body(proxy.destination_local_to_proxy_global,
                        entry.state_0,proxy.coupling_forces,entry.body_gravity_acceleration,dt)
                    self._notify_input_state_update(entry,StateFlags.BODY_QD|StateFlags.BODY_F,dt=dt)
                dst_contacts=contacts
                freshly=not iteration_restart
                filter_contacts=True
                config=next((self._proxy_collision_configs.get((p.src_name,name)) for p in incoming
                             if self._proxy_collision_configs.get((p.src_name,name)) is not None),None)
                if config is not None:
                    dst_contacts,freshly=self._proxy_collision_contacts(config,entry.state_0,iteration_restart=iteration_restart)
                    filter_contacts=False
                if incoming:
                    dst_contacts=entry.solver.coupling_prepare_proxy_contacts(entry.state_0,dst_contacts,
                        contacts_freshly_detected=freshly)
                used=self._step_entry(entry,control,dst_contacts,dt,filter_contacts=filter_contacts)
                for proxy in incoming:
                    entry.solver.coupling_harvest_proxy_wrenches(proxy.destination_local_to_proxy_global,
                        proxy.coupling_forces,body_qd_before=proxy.proxy_qd_before,state=entry.state_0,
                        state_out=entry.state_1,contacts=used,dt=dt)
                    self._blend_proxy_feedback(proxy)
                pending.remove(name)
                done.add(name)

    def step(self, state_in, state_out, control, contacts, dt, **kwargs):
        wp.copy(self.old_points, state_in.particle_q)
        super().step(state_in, state_out, control, contacts, dt, **kwargs)
        wp.copy(self.mesh_points, state_out.particle_q)
        self.lamina.refit()
        self.dq_contact.zero_()
        self.dv_contact.zero_()
        wp.launch(contact, len(self.water_ids), inputs=[self.lamina.id, self.water_ids, self.old_points,
                  state_out.particle_q, state_out.particle_qd, self.inverse_mass,
                  self.dq_contact, self.dv_contact, self.contact_impulse], device=self.model.device)
        wp.launch(apply, self.model.particle_count, inputs=[state_out.particle_q, state_out.particle_qd,
                  self.dq_contact, self.dv_contact], device=self.model.device)
        wp.launch(swept_static_contact,len(self.water_ids),inputs=[self.static_collision_mesh.id,
                  self.water_ids,self.old_points,state_out.particle_q,state_out.particle_qd,
                  self.static_friction,self.static_contact_hits],device=self.model.device)



def cached_factory(factory, rigid=False):
    """Capture only VBD's fixed launch sequence; PIC MPM remains eager."""
    def construct(view):
        if rigid:
            # Liquid margins are SDF offsets, not rigid contact margins. MuJoCo
            # already ignores these with native CCD; keep that view explicit.
            margins = view.shape_margin.numpy().copy()
            margins[:] = 0.0
            view.shape_margin = wp.array(margins, dtype=float, device=view.device)
        inner = factory(view)
        original = inner.step
        graphs = {}
        warmed = set()
        def step(state_in, state_out, control, contacts, dt, **kw):
            key = (id(state_in), id(state_out), id(control), id(contacts), float(dt), tuple(kw))
            if key not in warmed:
                original(state_in,state_out,control,contacts,dt,**kw)
                warmed.add(key)
                return
            if key not in graphs:
                with wp.ScopedCapture(device=view.device) as capture:
                    original(state_in,state_out,control,contacts,dt,**kw)
                graphs[key] = capture.graph
            wp.capture_launch(graphs[key])
        notify = inner.notify_model_changed
        def changed(flags):
            notify(flags)
            graphs.clear()
            warmed.clear()
        inner.notify_model_changed = changed
        inner.step = step
        return inner
    return construct


def build(ctx):
    by_name = {i.name: i for i in ctx.islands}
    proxies = []
    for c in ctx.channels:
        dest = by_name[c.destination]
        proxies.append(SolverCoupledProxy.Proxy(source=c.source, destination=c.destination,
            bodies=list(map(int, c.bodies)), mass_scale=1000.0 if dest.backend.solver == 'mpm' else 1.0,
            mode='staggered', collision_pipeline=_null_pipeline if dest.uses_native_contacts() else
            _pipeline_factory(ctx, c.pairs, particles=bool(dest.particles.size))))
    rod_island=next(i for i in ctx.islands if i.backend.solver=="vbd")
    fluid_island=next(i for i in ctx.islands if i.backend.solver=="mpm")
    proxies.append(SolverCoupledProxy.Proxy(source=rod_island.name,destination=fluid_island.name,
        bodies=list(map(int,rod_island.bodies)),mass_scale=1000.0,mode="staggered",collision_pipeline=_null_pipeline))
    water_ids = np.concatenate([i.particles for i in ctx.islands if i.backend.solver == 'mpm']).astype(np.int32)
    entries = ctx.entries(SolverCoupled.Entry)
    from dataclasses import replace
    entries = [replace(e, solver=cached_factory(e.solver, rigid=by_name[e.name].backend.solver == 'mujoco')) if by_name[e.name].backend.solver in ("vbd", "mujoco") else e for e in entries]
    return PlantWaterSolver(ctx.model, entries,
                           SolverCoupledProxy.Config(proxies=proxies), water_ids)


register_coupler_builder('anthurium_water', build)
from robosimstudio.specs.registries import register_coupler
register_coupler('anthurium_water', summary='Three-domain plant watering with physical particle/lamina impulses.')

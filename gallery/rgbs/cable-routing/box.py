"""Free cable box. Dimensions adapted from the supplied tunnel box; all pieces collide."""
from dataclasses import dataclass
import numpy as np
import math
import newton
import warp as wp
from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.build.bodies import set_body_mass
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.asset import ObjectSpec, ShapeSpec
from robosimstudio.core.render.finish import SATIN, Finish, register_texture


def dimples():
    y, x = np.mgrid[:512,:512] / 512
    row = np.round(y * 28)
    dx = (x*24 - (row%2)*.5 + .5)%1-.5
    dy = y*28-row
    dome = np.clip(1-(dx*dx+dy*dy)/.21,0,1)
    return np.repeat((.10+.23*dome)[...,None],3,axis=2)

register_texture('cable_box_mesh', dimples, version=1)

@dataclass(frozen=True, kw_only=True)
class CableBox(ObjectSpec):
    entity: str = 'gallery_cable_box'
    name: str = 'box'
    mass: float = 1.20
    friction: float = 0.65
    color: tuple = (.95,.92,.83)
    finish: object = SATIN
    shape: object = ShapeSpec(kind='box', half_extents=(.1,.2,.05))

@register_entity('gallery_cable_box')
class BoxEntity(EntityRigid):
    def build(self, builder, episode):
        pose = self._placed(self._episode_pose(episode))
        self.body = builder.add_body(xform=wp.transform(pose[:3],pose[3:]),label=self.spec.name)
        joint=builder.joint_count-1
        self.coord_start=int(builder.joint_q_start[joint])
        self.dof_start=int(builder.joint_qd_start[joint])
        self.colors={}
        self.textures={}
        cfg=self._contact_cfg(builder)
        cfg.margin=0.; cfg.gap=.001
        def block(label, center, half, color=None, friction=None, finish=None, quat=(0.,0.,0.,1.)):
            c=cfg.copy() if hasattr(cfg,'copy') else self._contact_cfg(builder)
            c.margin=0.; c.gap=.0002
            if friction is not None: c.mu=friction
            s=builder.add_shape_box(self.body,xform=wp.transform(center,wp.quat(*quat)),hx=half[0],hy=half[1],hz=half[2],cfg=c,label='box_'+label)
            if color is not None: self.colors[s]=color
            if finish is not None: self.textures[s]=finish
        # Origin is channel centre at floor height; x back, y left.
        # Real notch, not a black rectangle on a solid face.
        block('body',(.1135,0.,-.042),(.13,.24,.039))
        block('foot_left',(.1135,.15,-.0955),(.13,.09,.0145))
        block('foot_right',(.1135,-.15,-.0955),(.13,.09,.0145))
        block('notch_back',(.1535,0.,-.0955),(.09,.06,.0145))
        # 17 mm bore, 380 mm run, a 23 mm high cap.
        block('front_wall',(-.0125,-.05,.0085),(.004,.19,.0085),friction=.30)
        block('rear_wall',(.0125,-.05,.0085),(.004,.19,.0085),friction=.30)
        block('lid',(0,-.05,.020),(.0165,.19,.003),(1.,.99,.96),friction=.4)
        block('seam',(-.0166,-.05,.0155),(.0001,.19,.0015),(.12,.13,.14),friction=.3)
        block('floor',(0,-.05,-.0015),(.0165,.19,.0015),friction=.30)
        block('mesh_left',(.012,.19,-.0015),(.0285,.05,.0015),(.06,.065,.07),friction=.30,finish=Finish(roughness=.5,texture='cable_box_mesh',uv_scale=(6,6)))
        # Four 15 mm, 20-degree lead-in panels, outside the right mouth.
        angle=math.radians(20);spread=.015*math.tan(angle);wall=.003
        for side in (-1,1):
            normal=np.array([side*math.cos(angle),math.sin(angle),0.])
            center=np.array([side*(.0085+spread/2),-.2475,.0085])+wall/2*normal
            block('flare_side_'+str(side),tuple(center),(wall/2,.015/math.cos(angle)/2,.014),color=(1.,.99,.96),friction=.15,
                  quat=(0,0,math.sin(side*angle/2),math.cos(angle/2)))
        for sign in (-1,1):
            # sign +1 roof, -1 floor; local y rotates outwards/up or down.
            z=.017 if sign==1 else 0.
            center=(0.,-.24-.0075*math.cos(angle)+wall/2*math.sin(angle),
                    z+sign*.0075*math.sin(angle)+sign*wall/2*math.cos(angle))
            block('flare_'+('roof' if sign==1 else 'floor'),center,(.017,.0075,wall/2),color=(1.,.99,.96),friction=.15,
                  quat=(math.sin(-sign*angle/2),0,0,math.cos(angle/2)))
        # Dark top ends behind the 24 mm slot.
        block('mesh',(.142,0.,-.001),(.1015,.24,.001),(.06,.065,.07),finish=Finish(roughness=.5,texture='cable_box_mesh',uv_scale=(6,6)))
        block('strip',(.15,0.,.016),(.025,.15,.014),(1.,.98,.95))
        for k in range(6):
            for dx in (-.005,.005):
                block(f'socket_{k}_{dx}',(.15+dx,-.11+k*.040,.0303),(.001,.003,.0003),(.20,.20,.21))
        # Drawn white mains lead: colliding capsules, attached to the strip/body.
        for xshift in (-.0022,.0022):
            path=np.array([[.15,-.15,.018],[.15,-.19,.014],[.16,-.24,.014],[.18,-.28,.020],[.21,-.29,.04],[.22,-.275,.048]])
            path[:,0]+=xshift
            for i,(a,b) in enumerate(zip(path[:-1],path[1:])):
                d=b-a;length=float(np.linalg.norm(d));d=d/length
                q=np.array([-d[1],d[0],0.,1+d[2]]);q/=np.linalg.norm(q)
                shape=builder.add_shape_capsule(self.body,xform=wp.transform(tuple((a+b)/2),wp.quat(*q)),radius=.002,half_height=length/2,cfg=cfg,label='mains_lead')
                self.colors[shape]=(1.,.98,.94)
        set_body_mass(builder,self.body,float(self.spec.mass))
    def apply_display(self,builder):
        super().apply_display(builder)
        for s,c in self.colors.items(): builder.shape_color[s]=c
        self.finishes.update(self.textures)

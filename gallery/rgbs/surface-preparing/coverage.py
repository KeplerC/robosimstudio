"""Read-only coverage of the full bar crown from SuperDex contact quadrature."""
from dataclasses import dataclass
from collections import deque
import numpy as np
import newton
import warp as wp
from bar import STEEL
import robosimstudio as rss
from robosimstudio.domains.capabilities import host_bridge
from robosimstudio.scene.entity import label_index

@wp.kernel
def paint(exposure: wp.array(dtype=float), start: int, colors: wp.array(dtype=wp.vec3)):
    i=wp.tid()
    a=wp.clamp(1.0-exposure[i]/0.025,0.0,1.0)
    colors[start+i]=wp.vec3(.8,.8,.8)*(1.0-a)+wp.vec3(.12,.32,.85)*a

@dataclass(frozen=True, kw_only=True)
class CoverageSpec:
    entity: str = 'bar_coverage'
    name: str = 'coverage'

@rss.register_entity('bar_coverage')
class Coverage(rss.Entity):
    recolors_live=True
    def build(self, builder, episode):
        self.bar_body=label_index(list(builder.body_label),'bar')
        self.centres=np.linspace(-.012-.170+.00125,-.012+.170-.00125,136)
        self.exposure=np.zeros(136)
        self.decal_start=builder.shape_count
        cfg=newton.ModelBuilder.ShapeConfig(density=0.,has_shape_collision=False,has_particle_collision=False)
        for i,y in enumerate(self.centres):
            builder.add_shape_box(self.bar_body,xform=wp.transform(wp.vec3(.01265,0.,float(y+.012)),wp.quat_from_axis_angle(wp.vec3(0.,1.,0.),np.pi/2)),hx=.00125,hy=.0018,hz=.00008,cfg=cfg,color=(.12,.32,.85),label=f'ink_{i}')
        self.bridge=None
        self.press=0.
        self.recent=deque(maxlen=12)
        self.peak=0.
        self.wrap_arc=0.
        self.max_wrap_arc=0.
    def apply_display(self,builder):
        for i in range(self.decal_start,self.decal_start+136): self.finishes[i]=STEEL
    def bind(self,model):
        super().bind(model)
        self.device_exposure=wp.zeros(136,dtype=float,device=model.device)
    def recolor(self):
        model=self.stepping_model if self.stepping_model is not None else self.model
        self.device_exposure.assign(self.exposure.astype(np.float32))
        wp.launch(paint,dim=136,inputs=[self.device_exposure,self.decal_start,model.shape_color],device=model.device)
    def contribute_hooks(self,hooks):
        hooks.add('after',self.read_contacts,name='bar-coverage',cadence=1)
    def on_scene_ready(self,pipeline):
        self.bridge=host_bridge(pipeline,'sponge')
    @property
    def fraction(self):
        return float(np.mean(self.exposure>=.025))
    @property
    def held_press(self):
        return float(np.mean(self.recent)) if self.recent else 0.
    def read_contacts(self,state,dt):
        if self.bridge is None:return
        c=self.bridge.contact_points('sponge',against=self.bar_body)
        p=c['pos']; f=np.linalg.norm(c['force'],axis=1)
        pressure=f/np.maximum(c['area'],1e-9)
        upper=(pressure>=300.)&(p[:,2]>.88)&(p[:,1]>-.182)&(p[:,1]<.158)
        angles=np.degrees(np.arctan2(p[upper,0]-.64,p[upper,2]-.88))
        self.wrap_arc=float(np.ptp(angles)) if len(angles)>1 else 0.
        self.max_wrap_arc=max(self.max_wrap_arc,self.wrap_arc)
        # Upper crown band, approximately +/-40 degrees. No proximity-only credit: >=300 Pa and sliding.
        crown=(p[:,2]>.889)&(np.abs(p[:,0]-.64)<.008)&(p[:,1]>-.182)&(p[:,1]<.158)
        self.press=float(f[crown].sum()); self.recent.append(self.press)
        self.peak=max(self.peak,self.press)
        live=crown & (pressure>=300.) & (np.abs(c['v_soft'][:,1]-c['v_other'][:,1])>.002)
        if live.any():
            reached=np.min(np.abs(self.centres[:,None]-p[live,1][None,:]),axis=1)<=.003
            self.exposure[reached]+=float(dt)
            self.recolor()
    def reset(self,pipeline,episode,*,world=0):
        self.exposure[:]=0.;self.press=0.;self.peak=0.;self.recent.clear()
        self.wrap_arc=0.;self.max_wrap_arc=0.
        self.recolor()

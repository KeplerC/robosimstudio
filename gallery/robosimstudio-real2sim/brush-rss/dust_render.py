"""RTX extension drawing one dust mesh from the brush's measured lattice.

No simulation state is changed here. The registered RTX camera, lights,
materials, robot and geometry are drawn by the inherited renderer.
"""
from pathlib import Path
import numpy as np
import warp as wp
from robosimstudio.solvers.render.rtx import RendererRTX
from surface import dust_mesh

class DustRTX(RendererRTX):
    @classmethod
    def look_kwargs(cls,look):
        out=super().look_kwargs(look)
        out['light_direction']=look.light_direction
        return out

    def __init__(self,model,cameras,*,light_direction=None,**kwargs):
        from materials import finish
        finishes=dict(kwargs.pop('finishes',None) or {})
        # Dress the existing bench; no overlay, coincident face or collider.
        for i,name in enumerate(model.shape_label):
            if name=='table': finishes[i]=finish('worn_wood',1.4)
        super().__init__(model,cameras,finishes=finishes,**kwargs)
        original=self.viewer._decorate
        def decorate(viewer):
            from pxr import Gf,UsdGeom,UsdLux
            from robosimstudio.solvers.render.rtx import LAB_PANEL_PER_KEY
            original(viewer)
            if light_direction is not None:
                direction=np.asarray(light_direction,float)
                direction/=np.linalg.norm(direction)
                # The stock LAB panel is deliberately shadowless. Give it the
                # spec's direction, a smaller emitter and a warm workshop tint.
                xf=UsdGeom.Xform.Get(viewer.stage,'/root/_LabPanel')
                xf.ClearXformOpOrder()
                position=np.array([.56,-.2,.76])-direction*1.65
                xf.AddTranslateOp().Set(Gf.Vec3d(*position))
                xf.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0,0,-1),Gf.Vec3d(*direction)).GetQuat()))
                light=UsdLux.RectLight.Get(viewer.stage,'/root/_LabPanel/light')
                light.GetWidthAttr().Set(.55)
                light.GetHeightAttr().Set(.40)
                light.GetColorAttr().Set(Gf.Vec3f(1.,.93,.80))
                light.GetIntensityAttr().Set(LAB_PANEL_PER_KEY*float(kwargs.get('key_intensity',1.))*4.)
        self.viewer._decorate=decorate

    def _render_one(self,state,world):
        body_q=state.body_q.numpy() if any(c.parent_body>=0 for c in self.cameras) else None
        self.viewer.poses={c.name:c.pose_on_body(body_q,body_offset=int(world)*self.bodies_per_world) for c in self.cameras}
        self._push_live_colors()
        self._time+=1.e-3
        self.viewer.begin_frame(self._time)
        self.viewer.log_state(self._world_view(state,world))
        brush=getattr(self.model,'rss_measured_brush',None)
        if brush is not None:
            verts,tris,uv,normals=dust_mesh(brush)
            dev=self.model.device
            self.viewer.log_mesh('/measured_dust',wp.array(verts,dtype=wp.vec3,device=dev),
                wp.array(tris.ravel(),dtype=wp.int32,device=dev),
                wp.array(normals,dtype=wp.vec3,device=dev),
                uvs=wp.array(uv,dtype=wp.vec2,device=dev),texture=str(Path(__file__).parent/'assets/dust_tex.png'),
                roughness=.96,metallic=0.,dynamic=True,backface_culling=False)
        self.viewer.end_frame()
        for _ in range(self.accumulate-1): self.viewer.step_rig()
        return {c.name:self._read(c,i) for i,c in enumerate(self.cameras)}

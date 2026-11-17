"""A read-only studio panel showing the same measured dust as the RTX camera."""
import numpy as np
from robosimstudio_studio.panels import Panel
from surface import dust_mesh

class DustPanel(Panel):
    def __init__(self,session):
        super().__init__(session)
        self.mesh=None
        self.last=None
        self.draw()
    def draw(self):
        brush=self.env['brush'].entity
        cells=brush.scrub_grid().cells
        if self.last is not None and np.array_equal(cells,self.last):
            return
        self.last=cells.copy()
        vertices,faces,_,_=dust_mesh(brush)
        if self.mesh is None:
            self.mesh=self.server.scene.add_mesh_simple('/measured_dust',vertices=vertices,faces=faces,
                color=(194,179,145),side='double')
        else:
            self.mesh.vertices=vertices
    def drain(self):
        if self.env.step_count%6==0:
            self.draw()
    def on_reset(self,obs,info):
        self.last=None
        self.draw()
    def close(self):
        if self.mesh is not None:
            self.mesh.remove()
            self.mesh=None

def install_view(session):
    if not any(isinstance(p,DustPanel) for p in session.panels):
        session.panels.append(DustPanel(session))

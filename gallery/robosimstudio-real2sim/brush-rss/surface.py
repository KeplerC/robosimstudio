"""The shipped brush with a scene-local, 2 mm measurement lattice."""
import numpy as np
from robosimstudio.scene.registry import register_entity
from robosimstudio_materials.brush import EntityBrush, ScrubGrid
CELL=.002
ORIGIN=(.16,-.805)
CX=ORIGIN[0]+(np.arange(400)+.5)*CELL
CY=ORIGIN[1]+(np.arange(400)+.5)*CELL
IX=np.flatnonzero((CX>.485)&(CX<.655))
IY=np.flatnonzero((CY>-.355)&(CY<-.035))

@register_entity('rss_fine_brush')
class FineBrush(EntityBrush):
    def _lattice(self):
        if self._grid is None:
            self._grid=ScrubGrid.around((.56,-.405),reach=.4,cell=CELL)
        return self._grid
    def on_scene_ready(self,pipeline):
        super().on_scene_ready(pipeline)
        pipeline.model.rss_measured_brush=self

# Renderer-only height field: three samples per lattice cell (0.667 mm),
# Gaussian sigma 0.70 mm truncated at 1.40 mm. No target mask is consulted.
_UP=3
_SPACING=CELL/_UP
_cache=None

def dust_mesh(brush):
    from pathlib import Path
    from PIL import Image
    from scipy.ndimage import map_coordinates, gaussian_filter
    global _cache
    clean=brush.scrub_grid().cells[np.ix_(IY,IX)]>0
    if _cache is not None and np.array_equal(clean,_cache[0]):
        return _cache[1]
    nx,ny=len(IX)*_UP+1,len(IY)*_UP+1
    xx,yy=np.meshgrid(np.arange(nx)/_UP-.5,np.arange(ny)/_UP-.5)
    coverage=map_coordinates((~clean).astype(float),[yy,xx],order=1,mode='nearest')
    coverage=gaussian_filter(coverage,.00070/_SPACING,truncate=2.,mode='nearest')
    # Compact rolled boundary. Fully cleared regions lie beneath the panel.
    coverage=np.clip((coverage-.15)/.70,0,1)
    coverage=coverage*coverage*(3-2*coverage)
    edge=np.minimum.reduce([xx+.5,len(IX)-.5-xx,yy+.5,len(IY)-.5-yy])*CELL
    edge=np.clip(edge/.0014,0,1)
    coverage*=edge*edge*(3-2*edge)
    x=CX[IX[0]]+xx*CELL
    y=CY[IY[0]]+yy*CELL
    # Landscape texture's long axis follows the panel's long (Y) axis.
    # Read the original pixels directly, without resizing or box averaging.
    u=(y+.355)/.320
    vv=(x-.485)/.170
    pixels=np.asarray(Image.open(Path(__file__).parent/'assets/dust_tex.png').convert('L'),float)
    tex=map_coordinates(pixels,[vv*(pixels.shape[0]-1),u*(pixels.shape[1]-1)],order=1,mode='nearest')
    tex=(tex-pixels.mean())/max(float(pixels.std()),1.)
    relief=np.clip(tex,-2,2)*.00015
    h=.7599+coverage*(.0011+relief)
    v=np.stack([x,y,h],axis=-1).reshape(-1,3).astype(np.float32)
    dhdy,dhdx=np.gradient(h,_SPACING)
    # Independent 2–4 mm granular normal detail. This never alters clearing.
    noise=np.random.default_rng(837).standard_normal((ny,nx))
    fine=gaussian_filter(noise,1.2,mode='wrap')-gaussian_filter(noise,3.,mode='wrap')
    fine/=fine.std()
    bump=fine*.00012
    by,bx=np.gradient(bump,_SPACING)
    strength,bump_gain=4.,.9
    normals=np.stack([-dhdx-strength*bump_gain*bx*coverage,
                      -dhdy-strength*bump_gain*by*coverage,np.ones_like(h)],axis=-1).reshape(-1,3)
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    k=(np.arange(ny-1)[:,None]*nx+np.arange(nx-1)[None,:]).ravel()
    tris=np.stack([k,k+1,k+nx+1,k,k+nx+1,k+nx],axis=-1).reshape(-1,3).astype(np.int32)
    uv=np.stack([u.ravel(),vv.ravel()],axis=1).astype(np.float32)
    result=(v,tris,uv,normals.astype(np.float32))
    _cache=(clean.copy(),result)
    return result

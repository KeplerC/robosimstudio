"""Deterministic, seamless workshop materials; generated assets travel with the twin."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from robosimstudio.core.render.finish import Finish

ASSETS=Path(__file__).parent/'assets'

def make_textures():
    rng=np.random.default_rng(845)
    n=768
    y,x=np.mgrid[:n,:n]/n
    noise=rng.normal(size=(n,n))
    def field(sigma):
        a=gaussian_filter(noise,sigma,mode='wrap')
        return a/max(float(a.std()),1e-8)
    def save(name,base,variation):
        rgb=np.asarray(base)[None,None,:]+variation[:,:,None]
        Image.fromarray(np.uint8(np.clip(rgb,0,255))).save(ASSETS/(name+'.png'))
    # Worn hardwood: long grain, plank seams, tiny pale abrasions.
    grain=np.sin(2*np.pi*(x*55+.18*np.sin(2*np.pi*y*2)+.22*field((24,2))))
    wood=7*grain+4*field((35,1))+3*field(6)+1.4*noise
    seams=np.minimum((x*4)%1,1-(x*4)%1)<.007
    wood[seams]-=25
    scratches=(rng.random((n,n))>.998).astype(float)
    wood+=18*gaussian_filter(scratches,(15,.3),mode='wrap')/.03
    save('worn_wood',(139,106,68),wood)
    concrete=9*field(38)+5*field(6)+3*noise
    concrete-=13*np.clip(field(16)-1,0,2)
    save('concrete',(143,143,135),concrete)
    save('painted_wall',(169,182,174),4*field(42)+2*field(5)+1.2*noise)
    fibers=3*field((.5,5))+1.5*noise+3*field(20)
    save('cardboard',(170,133,85),fibers)
    peg=4*field(12)+1.5*noise
    holes=((x*24+.5)%1-.5)**2+((y*24+.5)%1-.5)**2<.075**2
    peg[holes]-=75
    save('pegboard',(141,105,61),peg)

def finish(name,scale=1.,roughness=.85):
    return Finish(texture=str(ASSETS/(name+'.png')),uv_scale=(scale,scale),roughness=roughness)

if __name__=='__main__':
    make_textures()

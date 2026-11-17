"""Procedural honey butcher-block grain; no reference image pixels."""
from pathlib import Path
import numpy as np
from PIL import Image

def make_wood():
    n=1024
    y,x=np.mgrid[:n,:n]/n
    rng=np.random.default_rng(41)
    strip=(x*8).astype(int)
    base=np.array([205.,163.,104.])
    shifts=rng.uniform(-12,12,(8,1))
    tint=shifts[strip]
    grain=2.5*np.sin(2*np.pi*(x*75+1.1*np.sin(y*9)+.4*np.sin(y*25)))+1.2*np.sin(x*1400+y*4)
    noise=rng.normal(0,1.25,(n,n))
    rgb=base+tint+(grain+noise)[...,None]
    # Eight 100 mm strips per repeat, with staggered butt joints.
    joints=((x*8)%1<.009)|(((y*2+(strip%3)*.333)%1)<.002)
    rgb[joints]-=13
    Image.fromarray(np.clip(rgb,0,255).astype('uint8')).save(Path(__file__).parent/'assets/wood.png')
if __name__=='__main__':make_wood()

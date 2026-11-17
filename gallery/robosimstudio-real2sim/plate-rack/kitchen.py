"""Kitchen geometry inspired by the supplied Pexels photograph; no photo textures."""
from dataclasses import replace
from pathlib import Path
import math
import numpy as np
import robosimstudio as rss
from robosimstudio.core.render.room import LAB
from robosimstudio.core.render.finish import Finish
LOOK = replace(LAB, scenery="plate_kitchen_lights", quality="final", key_intensity=1.25, ambient=(1.1,1.1,1.1))
WOOD = Finish(roughness=.34, texture=str(Path(__file__).parent/'assets/wood.png'), uv_scale=(1.25,1.25))
METAL = Finish(roughness=.48, metallic=.55)
MINT=(.62,.82,.70)
CREAM=(.86,.85,.75)

def add_kitchen(scene):
    def box(name,p,h,color,finish=None):
        scene.add(rss.StaticSpec(name=name, half_extents=h, xform=(*p,0,0,0,1), world_xform=True, color=color, finish=finish))
    def rod(name,a,b,r,color,finish=None):
        from model import _quat_z_to
        a,b=np.array(a),np.array(b); d=b-a
        scene.add(rss.StaticSpec(name=name,shape=rss.ShapeSpec(kind='capsule',radius=r,half_height=max(0,float(np.linalg.norm(d))/2-r)), xform=(*((a+b)/2),*_quat_z_to(d/np.linalg.norm(d))), world_xform=True,color=color,finish=finish))
    for y in (-.21,.21): box('robot_plinth'+str(y),(.1525,y,.81),(.075,.065,.06),(.30,.32,.30),Finish(roughness=.4))
    # A 750 mm work surface, with the original bench footprint and an adjoining sink.
    box('counter',(.44,-.52,.725),(.57,1.15,.025),(1,1,1),WOOD)
    box('counter_sink_front',(.935,1.15,.725),(.075,.52,.025),(1,1,1),WOOD)
    box('counter_sink_back',(-.025,1.15,.725),(.105,.52,.025),(1,1,1),WOOD)
    for y in (.67,1.65): box('counter_sink_end'+str(y),(.47,y,.725),(.39,.02,.025),(1,1,1),WOOD)
    # Mint panelled cabinetry and brushed pulls.
    box('cabinet_body',(.47,-.52,.36),(.50,1.15,.35),(.45,.65,.53))
    box('sink_cabinet_base',(.47,1.15,.23),(.50,.52,.22),(.45,.65,.53))
    for j,y in enumerate(np.linspace(-1.43,1.43,7)):
        box(f'door{j}',(.978,y,.375),(.016,.196,.30),MINT,Finish(roughness=.45))
        box(f'door_inset{j}',(.996,y,.37),(.006,.155,.245),(.56,.76,.63))
        for dy in (-.17,.17): box(f'door_stile{j}_{dy}',(1.005,y+dy,.37),(.01,.018,.265),MINT)
        for z in (.105,.635): box(f'door_rail{j}_{z}',(1.005,y,z),(.01,.185,.018),MINT)
        rod(f'pull{j}',(1.032,y-.055,.58),(1.032,y+.055,.58),.007,(.6,.62,.61),METAL)
    box('kickboard',(.94,0,.055),(.015,1.66,.055),(.36,.50,.40))
    # Backsplash and plaster around the double sash window (wall behind arms).
    box('backsplash_grout',(-.146,0,.985),(.03,2.12,.235),(.89,.88,.82))
    for row in range(5):
        for col in range(22):
            y=-2.0+col*.19+(row%2)*.095
            box(f'tile{row}_{col}',(-.124,y,.794+row*.09),(.009,.094,.044),(.98,.975,.94),Finish(roughness=.2))
    for y in (-1.73,1.73): box(f'wall{y}',(-.18,y,1.65),(.06,.32,.9),(.77,.75,.63))
    box('wall_above',(-.18,0,2.47),(.06,1.42,.13),(.77,.75,.63))
    box('window_sill',(-.095,0,1.24),(.11,1.43,.035),CREAM,Finish(roughness=.36))
    for y in (-1.40,0,1.40): box(f'window_post{y}',(-.14,y,1.82),(.055,.035,.57),CREAM)
    for z in (1.28,1.82,2.38): box(f'sash_horizontal{z}',(-.12,0,z),(.04,1.38,.022),CREAM)
    for y in (-.70,.70): box(f'sash_vertical{y}',(-.12,y,1.83),(.035,.018,.53),CREAM)
    for y in (-1.04,-.35,.35,1.04): box(f'window_latch{y}',(-.065,y,1.32),(.014,.035,.015),(.44,.45,.42),METAL)
    # Garden beyond: actual branches/leaves and brickwork, not a billboard.
    box('garden_ground',(-1.3,0,.06),(1.0,2.1,.04),(.23,.31,.12))
    box('brick_mortar',(-1.25,1.66,1.25),(.06,1.0,1.25),(.55,.51,.42))
    for row in range(24):
        for col in range(9):
            box(f'brick{row}_{col}',(-1.18,.74+col*.22+(row%2)*.11,.10+row*.10),(.025,.106,.046),(.47+.04*((row+col)%3),.24+.02*(col%2),.16),Finish(roughness=.85))
    rng=np.random.default_rng(9)
    for i in range(65):
        p=(-.65-rng.random()*.8,-1.5+rng.random()*2.6,.7+rng.random()*1.7)
        scene.add(rss.StaticSpec(name=f'foliage{i}',shape=rss.ShapeSpec(kind='sphere',radius=.07+rng.random()*.09),xform=(*p,0,0,0,1),world_xform=True,color=(.13+rng.random()*.10,.29+rng.random()*.15,.09),finish=Finish(roughness=.9)))
    rod('tree',(-.8,-.6,.1),(-.8,-.5,2.2),.045,(.27,.20,.12))
    # Two steel basins. The opening is hollow, with a floor and four sloped-looking sides.
    for i,y in enumerate((.91,1.40)):
        box(f'sink_floor{i}',(.47,y,.54),(.35,.215,.008),(.96,.98,1.),METAL)
        for x in (.115,.825): box(f'sink_side{i}_{x}',(x,y,.635),(.009,.225,.095),(.96,.98,1.),METAL)
        for yy in (y-.224,y+.224): box(f'sink_end{i}_{yy}',(.47,yy,.635),(.36,.009,.095),(.96,.98,1.),METAL)
        for x in (.10,.84): box(f'sink_lip{i}_{x}',(x,y,.751),(.018,.24,.004),(.96,.98,1.),METAL)
        for yy in (y-.235,y+.235): box(f'sink_lip_end{i}_{yy}',(.47,yy,.751),(.37,.012,.004),(.96,.98,1.),METAL)
        scene.add(rss.StaticSpec(name=f'drain{i}',shape=rss.ShapeSpec(kind='cylinder',radius=.037,half_height=.003),xform=(.47,y,.553,0,0,0,1),world_xform=True,color=(.14,.16,.16),finish=METAL))
    for y in (.97,1.33):
        rod('faucet_mount'+str(y),(-.11,y,.99),(.04,y,.99),.022,(.65,.68,.70),METAL)
    rod('mixer',(.04,.97,.99),(.04,1.33,.99),.021,(.65,.68,.70),METAL)
    path=[(.04,1.15,.99),(.04,1.15,1.10),(.08,1.15,1.18),(.38,1.15,1.18),(.44,1.15,1.13),(.44,1.15,1.08)]
    for i,(a,b) in enumerate(zip(path,path[1:])): rod(f'spout{i}',a,b,.013,(.70,.73,.75),METAL)
    box('sconce_back',(-.075,-1.65,1.69),(.025,.07,.12),(.48,.36,.18),METAL)
    box('sconce_shade',(.0,-1.65,1.72),(.07,.11,.14),(.97,.86,.56),Finish(roughness=.5))
    # Enclosed room: both return walls, opposite wall, continuous ceiling.
    plaster=(.88,.86,.76)
    for y in (-2.10,2.10):
        box(f'return_wall{y}',(1.43,y,1.35),(1.63,.06,1.35),plaster,Finish(roughness=.75))
        box(f'return_skirt{y}',(1.43,y-(.065 if y>0 else -.065),.07),(1.63,.012,.07),CREAM)
    box('opposite_wall',(3.05,0,1.35),(.06,2.16,1.35),plaster,Finish(roughness=.75))
    box('ceiling',(1.43,0,2.73),(1.68,2.16,.04),(.96,.95,.90),Finish(roughness=.85))
    box('window_wall_header',(-.18,0,2.60),(.06,2.12,.13),plaster)
    box('window_wall_lower',(-.18,0,.38),(.06,2.12,.38),plaster)
    # Pale sky is a solid backdrop beyond the garden, never a camera background.
    box('garden_sky',(-2.6,0,2.0),(.04,8,4),(.60,.79,.90),Finish(roughness=1))
    box('floor_base',(1.43,0,-.028),(1.68,2.16,.025),(.68,.66,.58),Finish(roughness=.8))
    for i in range(7):
        for j in range(9):
            tone=.025*((i+2*j)%3)
            box(f'floor_tile{i}_{j}',(-.02+i*.46,-1.84+j*.46,-.001),(.228,.228,.003),(.74+tone,.72+tone,.64+tone),Finish(roughness=.6))


def kitchen_lights(kit):
    """Room fill and garden daylight, inside the modeled enclosure."""
    if not hasattr(kit, 'stage'):
        return
    from pxr import Gf, UsdGeom, UsdLux
    for name, pos, rotation, size, power in (
        ('room_fill', (2.75,0,1.7), (0,90,0), (2.0,3.5), 2600.),
        ('garden_daylight', (-.30,0,2.0), (0,90,0), (2.0,3.0), 1800.),
    ):
        xf=UsdGeom.Xform.Define(kit.stage, '/root/KitchenLight_'+name)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
        lamp=UsdLux.RectLight.Define(kit.stage, str(xf.GetPath())+'/light')
        lamp.GetWidthAttr().Set(size[0]); lamp.GetHeightAttr().Set(size[1])
        lamp.GetIntensityAttr().Set(power)
        lamp.GetColorAttr().Set(Gf.Vec3f(1.,.98,.94))

from robosimstudio.core.render.scenery import register_scenery
register_scenery('plate_kitchen_lights', kitchen_lights)

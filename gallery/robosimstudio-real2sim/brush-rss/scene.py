"""A YAM writes RSS in measured dust on a workshop bench."""
from pathlib import Path
from dataclasses import replace
import robosimstudio as rss
from robosimstudio.core.render.room import LAB
from robosimstudio.core.render.finish import Finish
import os
import surface
os.environ["ROBOSIMSTUDIO_RENDERER"]="dust_render.DustRTX"
from drive import routine, score
from page_dust import install_view
from materials import finish

HERE=Path(__file__).parent
scene=rss.Scene(cell='bench_075',name='brush-rss',physics=rss.BRUSH_60HZ,
                render=replace(LAB,scenery='',quality='final',key_intensity=2.4,ambient=(.26,.26,.26),light_direction=(-.35,.45,-1.)))
scene.add(rss.robot('yam', at=(-.3475,0,.022), arms_at={'left':(-.3475,.31,.022),'right':(-.3475,-.31,.022)}))
scene.add(rss.slab(name='panel',half=(.085,.16,.005),at=(-.03,-.195,.005),color=(.055,.064,.070),finish=Finish(roughness=.93)))
scene.add(rss.slab(name='rest',half=(.03,.03,.005),at=(-.04,-.405,.005),color=(.55,.40,.25)))
brush=rss.brush(name='brush',at=(-.04,-.405,.010),material='nylon',handle='handle',rows=7,columns=7,
                sole=(.010,.010,.006),bristle_length=.02,flare=.09,color=(.93,.80,.45),handle_color=(.30,.17,.08))
scene.add(rss.box(name='handle',half=(.011,.011,.03),at=(-.04,-.405,.010+brush.sole_top+.03),friction=1.,color=(.30,.17,.08)))
scene.add(replace(brush,entity='rss_fine_brush'))
# Workshop: all actual geometry, with no photograph or image dome.
def slab(name,half,at,color,material=None,scale=1.):
    return scene.add(rss.slab(name=name,half=half,at=at,color=color,finish=finish(material,scale) if material else Finish(roughness=.8)))
slab('rear_mount_beam',(.11,.65,.055),(-.3475,0,-.058),(.18,.20,.19))
for side,y in [('left',.31),('right',-.31)]:
    slab('mount_'+side,(.09,.10,.0125),(-.3475,y,.0095),(.24,.26,.25))
slab('concrete_floor',(3.,3.,.035),(0,0,-.784),(.31,.32,.30),'concrete',.7)
slab('back_wall',(.06,3.,1.5),(-1.4,0,.70),(.52,.55,.51),'painted_wall',.8)
slab('side_wall',(2.8,.06,1.5),(0,2.1,.70),(.60,.60,.54),'painted_wall',.8)
slab('door',(.025,.44,1.0),(-1.325,-1.20,.25),(.19,.26,.27))
for y in [-1.66,-.74]: slab('door_jamb_'+str(y),(.04,.035,1.04),(-1.29,y,.29),(.30,.33,.32))
slab('door_top',(.04,.49,.035),(-1.29,-1.2,1.33),(.30,.33,.32))
slab('door_handle_plate',(.006,.022,.065),(-1.294,-.87,.25),(.46,.48,.47))
slab('door_handle_spindle',(.025,.010,.010),(-1.269,-.87,.25),(.65,.65,.61))
slab('door_handle',(.010,.060,.010),(-1.246,-.92,.25),(.65,.65,.61))
for z in [-.48,.05]:
    slab('shelf_'+str(z),(.24,.60,.025),(-1.08,.55,z),(.40,.29,.17),'worn_wood',2.)
for y in [-.08,1.18]:
    slab('upright_'+str(y),(.03,.03,.85),(-1.08,y,.05),(.13,.16,.16))
for i in range(5):
    slab('storage_box_'+str(i),(.12,.09,.14),(-1.06,.12+i*.22,.215),(.44+.025*i,.35+.015*i,.23),'cardboard',3.)
slab('pegboard',(.025,.60,.32),(-1.29,.55,.55),(.40,.29,.16),'pegboard',1.5)
# Five recognisable tools, all geometry, hanging on short board pegs.
for i,(y,z) in enumerate([(.10,.62),(.32,.58),(.51,.63),(.73,.63),(1.0,.61)]):
    slab('tool_peg_'+str(i),(.014,.006,.006),(-1.252,y,z),(.25,.26,.25))
slab('hammer_handle',(.012,.016,.105),(-1.245,.10,.50),(.45,.25,.10),'worn_wood',3.)
slab('hammer_head',(.024,.062,.025),(-1.245,.10,.63),(.35,.37,.38))
for i,(y,z,length,color) in enumerate([(.32,.44,.055,(.60,.12,.07)),(.51,.49,.075,(.13,.27,.48))]):
    slab('driver_grip_'+str(i),(.014,.019,.045),(-1.245,y,z),color)
    slab('driver_shaft_'+str(i),(.005,.004,length),(-1.245,y,z+.045+length),(.65,.67,.68))
    slab('driver_tip_'+str(i),(.005,.008,.009),(-1.245,y,z+.045+2*length),(.58,.60,.61))
slab('wrench_shank',(.009,.012,.085),(-1.245,.73,.51),(.54,.57,.59))
slab('wrench_head_base',(.010,.033,.010),(-1.245,.73,.60),(.60,.63,.65))
for i,y in enumerate([.705,.755]):
    slab('wrench_jaw_'+str(i),(.010,.008,.025),(-1.245,y,.625),(.60,.63,.65))
slab('square_blade',(.007,.013,.100),(-1.245,.94,.52),(.43,.48,.50))
slab('square_stock',(.014,.075,.016),(-1.245,1.0,.432),(.64,.42,.08))
for i in range(6):
    slab('square_mark_'+str(i),(.001,.006,.001),(-1.237,.94,.46+i*.025),(.83,.85,.83))
scene.add(rss.CameraSpec(name='macro',width=1280,height=720,eye=(.85,-.31,.855),look_at=(.555,-.19,.772),fov_y=28))
scene.add(rss.CameraSpec(name='wide',width=1280,height=720,eye=(1.65,-.65,1.40),look_at=(.40,-.04,1.03),fov_y=46))

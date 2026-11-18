# SPDX-License-Identifier: Apache-2.0
"""Surface Preparing: freely deforming foam, held and wiped by contact."""
from dataclasses import replace
from pathlib import Path
import robosimstudio as rss
from robosimstudio.core.render.room import LAB, LAB_GRID_MAT, LAB_MAT_GREY
from robosimstudio.core.render.finish import MATTE, SATIN
from robosimstudio.robots import get_embodiment
from robosimstudio_materials.soft import FIRM_FOAM
from bar import BarSpec, STEEL

PHYSICS = rss.ArrangementSpec(control_hz=60, substeps=16, backends=(
    ('*', rss.backend('mujoco', use_mujoco_contacts=True, nconmax=2048, njmax=4096,
                      iterations=100, ls_iterations=50, cone='elliptic', impratio=10., enable_multiccd=True)),
    ('soft', rss.backend('superdex', substeps=4, feedback_dv_max=5., newton_iterations=20)),
), collision=rss.CollisionSpec(contact_ke=6e4, contact_kd=500., rigid_contact_max=8192, rigid_gap=.0002))
PHOTO = rss.CameraSpec(name='photo', width=1920, height=1080, fx=2091.49, fy=2091.49,
    cx=960.,cy=540.,eye=(-.1344,.063,1.3313),look_at=(.7194,.001,.8145),up=(.5169,-.0159,.8559))
WIDE = rss.CameraSpec(name='wide', width=960, height=540, eye=(-.65,.35,1.55),look_at=(.50,-.05,.95), fov_y=45)
scene=rss.Scene(cell='bench_075', name='surface-preparing', physics=PHYSICS, render=replace(LAB,quality='preview'))
scene.add(PHOTO)
scene.add(WIDE)
robot_spec=get_embodiment('yam')
HOME=(0.,1.0764551163,.2743743956,.8020805717,0.,0.)
robot_spec=replace(robot_spec,arms=tuple(replace(a,home_q=HOME) for a in robot_spec.arms))
scene.add(rss.robot(robot_spec,arms_at={'left':(-.3475,.3008,.022),'right':(-.3475,-.3008,.022)}))
fixture=rss.SimRequirements(domain='static',contacts=rss.contacts_with('objects','robot','deformables'))
scene.add(rss.StaticSpec(name='mat', half_extents=(.2975,.65,.005),xform=(0,0,.005,0,0,0,1),friction=.9,color=LAB_MAT_GREY,finish=LAB_GRID_MAT,sim=fixture))
for side,sign in [('left',1),('right',-1)]:
    scene.add(rss.StaticSpec(name='foot_'+side,half_extents=(.030,.042,.035),xform=(.040,-.012+sign*.210,.045,0,0,0,1),friction=.9,color=(.018,.018,.021),finish=SATIN,sim=fixture))
for side,sign in [('left',1),('right',-1)]:
    cx,cy=.040,-.012+sign*.210
    scene.add(rss.StaticSpec(name='plate_'+side,half_extents=(.029,.041,.001),xform=(cx,cy,.081,0,0,0,1),color=(.6,.61,.62),finish=STEEL,sim=fixture))
    for i,(dx,dy) in enumerate([(-.021,-.030),(-.021,.030),(.021,-.030),(.021,.030)]):
        scene.add(rss.post(name=f'screw_{side}_{i}',radius=.0027,half_height=.0005,at=(cx+dx,cy+dy,.0825),color=(.025,.025,.025)))
scene.add(BarSpec(pose=(.040,-.012,.130),quat=(-.5,-.5,-.5,.5),sim=rss.SimRequirements(domain='rigid',contacts=rss.contacts_with('surface','fixtures','robot','deformables'))))
SPONGE_AT=(.040,-.3453,.011)
scene.add(rss.sponge(name='sponge',size=(.120,.0625,.025),cells=(9,13,6),at=SPONGE_AT,asset=rss.AssetSpec(path=str(Path(__file__).resolve().parent / 'assets/sponge.npz')),
    material=replace(FIRM_FOAM,young=12000.,friction_falloff=1e-4),color=(.93,.94,.94),finish=replace(MATTE,roughness=.95),settle_frames=60,
    sim=rss.SimRequirements(domain='soft',contacts=rss.contacts_with('surface','hands','fixtures','objects'))))
from coverage import CoverageSpec
scene.add(CoverageSpec())
from motion import routine

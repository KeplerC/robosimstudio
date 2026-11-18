"""Cable Routing gallery twin. Contact-only YAM and library rod."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import robosimstudio as rss
from robosimstudio_materials.rod import rod
from robosimstudio_materials.soft import SoftSpec, SoftMaterialSpec
from scipy.spatial.transform import Rotation
from robosimstudio.core.render.room import LAB
from robosimstudio.robots.registry import get_embodiment
from box import CableBox

mj=rss.backend('mujoco',use_mujoco_contacts=False,iterations=60,ls_iterations=20,nconmax=1024,njmax=2048)
engine=rss.backend('superdex',substeps=3,feedback='auto',mirror_friction_falloff=1e-4,mirror_threshold=.0002,mirror_smoothing=.0001,mirror_padding=.00205,newton_iterations=8,linear_solver='ldlt')
physics=rss.replace(rss.GENERIC,control_hz=50,substeps=12,
    backends=(('articulated',mj),('rigid',mj),('rod',engine),('soft',engine)),
    collision=rss.CollisionSpec(rigid_gap=.0005))
scene=rss.Scene(cell='bench_075',name='cable-routing',physics=physics,render=replace(LAB,ambient=(2.,2.,2.),key_intensity=1.5))
yam=get_embodiment('yam')
def clip_hands(builder, shapes):
    for i in range(*shapes):
        label=builder.shape_label[i]
        if 'link_6' in label or 'finger' in label or 'camera_d405' in label:
            builder.shape_color[i]=(.008,.009,.012)
    return 0
yam=replace(yam,appearance=clip_hands)
home_left=(0.45721250772476196, 2.213111639022827, 1.471858024597168, -1.5707999467849731, 1.0100008249282837, -0.7255746126174927)
home_right=(-0.4891849160194397, 2.2620530128479004, 1.4035359621047974, -1.5707999467849731, -0.9799078702926636, 0.6150237917900085)
yam=replace(yam,arms=(replace(yam.arms[0],home_q=home_left,home_gripper_q=(.018,)),replace(yam.arms[1],home_q=home_right,home_gripper_q=(.018,))))
scene.add(rss.robot(yam,arms_at={'left':(-.3475,.3008,.022),'right':(-.3305,-.3008,.051)}))
box=scene.add(CableBox(pose=(-.031,0.,.1102)))
# The initial free cord lies beside the right end. No pre-threading or anchor.
# Authored initial polyline, in the bench surface frame; see assets/PROVENANCE.md.
nodes=tuple(map(tuple,np.load(Path(__file__).parent/'assets/cord_long_settled_r4.npy')))
arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(np.asarray(nodes),axis=0),axis=1))]
straight_rest=tuple(map(tuple,np.asarray(nodes[0])+arc[:,None]*np.array([0.,1.,0.])))
plug=scene.add(SoftSpec(name='connector',asset=rss.AssetSpec(path=str(Path(__file__).parent/'assets/connector_long_settled_r4.npz')),
    pose=nodes[-1],quat=(0.,0.,0.,1.),rigid=None,settle_frames=0,
    material=SoftMaterialSpec(young=2e6,density=1000,friction=.65,contact_penalty=1e9,contact_padding=.0002,friction_falloff=1e-4,collider='sdf',sdf_resolution=.001,contact_threshold=.0002,contact_smoothing=.0001),
    color=(.015,.015,.018)))
tail=scene.add(SoftSpec(name='tail_connector',asset=rss.AssetSpec(path=str(Path(__file__).parent/'assets/tail_connector_long_settled_r4.npz')),
    pose=nodes[0],quat=(0.,0.,0.,1.),rigid=None,settle_frames=0,
    material=SoftMaterialSpec(young=2e6,density=1000,friction=.65,contact_penalty=1e9,contact_padding=.0002,friction_falloff=1e-4,collider='sdf',sdf_resolution=.001,contact_threshold=.0002,contact_smoothing=.0001),
    color=(.015,.015,.018)))
cable=scene.add(rod(name='cable',nodes=nodes,radius=.00225,density=1050,
    young_modulus=1e7,axial_stiffness=2000.,mu_cable=.65,stress_free=straight_rest,joins=(('end','connector',.0061),('start','tail_connector',.0061)),
    contact_penalty=1e6,contact_gap=.0005,contact_threshold=.0002,contact_smoothing=.0001,
    skin=False,friction_falloff=1e-4,color=(.012,.012,.014)))
scene.add(rss.CameraSpec(name='photo',width=960,height=540,fov_y=33.83,fx=887.935,fy=887.935,cx=480.,cy=270.,eye=(-.2088,-.1759,1.2345),look_at=(.7047,-.037,.8521),up=(.3746,.0793,.9238)))
scene.add(rss.CameraSpec(name='wide',width=960,height=540,eye=(-.05,-.08,1.45),look_at=(.55,0,.85),fov_y=52))

from routing import routine, reading

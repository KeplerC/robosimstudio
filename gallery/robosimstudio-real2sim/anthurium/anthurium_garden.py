"""Layered hedge crowns and leafy branches beyond a modelled window and room."""
import numpy as np
import trimesh
import robosimstudio as rss
from robosimstudio.core.render.finish import Finish


def add(scene, root):
    rng=np.random.default_rng(7302)
    groups=[[] for _ in range(5)]
    # Overlapping crowns close the gaps; individual leaves break their silhouettes.
    for y in np.linspace(-1.7,1.9,10):
        for z in (.30,.76,1.20):
            centre=np.array([rng.uniform(1.5,2.8),y+rng.uniform(-.07,.07),z])
            radii=np.array([.31,.20,.28])*rng.uniform(.85,1.15,3)
            for k in range(700):
                normal=rng.normal(size=3);normal/=np.linalg.norm(normal)
                p=centre+normal*radii*rng.uniform(.45,1.12)
                length=rng.uniform(.03,.06);width=length*.58
                v=np.array([[0,0,0],[.25,-.5,.04],[.7,-.4,.07],[1,0,0],[.7,.4,.07],[.25,.5,.04],[.45,0,.13]])*[length,width,length]
                mesh=trimesh.Trimesh(v,[[6,i,(i+1)%6] for i in range(6)],process=False)
                mesh.apply_transform(trimesh.transformations.random_rotation_matrix(rng.random(3)))
                mesh.apply_translation(p)
                assert mesh.vertices[:,0].min()>.45
                groups[k%5].append(mesh)
    colors=[(.10,.30,.10),(.20,.42,.16),(.035,.14,.05),(.28,.48,.19),(.07,.25,.09)]
    for i,meshes in enumerate(groups):
        path=root/'assets'/f'hedge_branches_{i}.obj'
        if not path.exists():trimesh.util.concatenate(meshes).export(path)
        scene.add(rss.StaticSpec(name=f'garden_{i}',asset=rss.AssetSpec(path=str(path)),color=colors[i],finish=Finish(roughness=.64),collide=False))
    # Timbers at garden depth project into the upper right panes.
    for i,y in enumerate((-1.2,-.65,-.1)):
        scene.add(rss.slab(name=f'pergola_beam_{i}',half=(.8,.04,.045),at=(1.6,y,.78),color=(.45,.36,.21),finish=Finish(roughness=.85),collide=False))
    scene.add(rss.slab(name='pergola_crossbeam',half=(.055,1.0,.05),at=(2.3,-.65,.78),color=(.45,.36,.21),finish=Finish(roughness=.85),collide=False))
    scene.add(rss.slab(name='garden_boundary',half=(.05,7,2.4),at=(8.0,0,1.2),color=(.70,.80,.67),finish=Finish(roughness=1),collide=False))
    scene.add(rss.slab(name='planter_ledge',half=(.055,1.35,.02),at=(.54,.25,.13),color=(.50,.50,.44),finish=Finish(roughness=.8),collide=False))
    # Begonias are grouped around branching stems, not scattered over the room.
    flowers=[[],[],[]]
    for y in (-.40,.70,1.08):
        centre=np.array([.515,y,.15])
        for j in range(15):
            tip=centre+rng.uniform([-.012,-.10,.035],[.012,.10,.20])
            stem=trimesh.creation.cylinder(radius=.0015,segment=np.array([centre,tip]),sections=5);flowers[0].append(stem)
            for k in range(3):
                q=centre+(tip-centre)*(.3+.2*k)
                leaf=trimesh.creation.icosphere(subdivisions=1);leaf.vertices*=np.array([.028,.045,.004]);leaf.apply_transform(trimesh.transformations.random_rotation_matrix(rng.random(3)));leaf.apply_translation(q);flowers[0].append(leaf)
            for k in range(5):
                angle=k*2*np.pi/5
                petal=trimesh.creation.icosphere(subdivisions=1);petal.vertices*=np.array([.013,.011,.003]);petal.apply_translation(tip+[.009*np.cos(angle),.009*np.sin(angle),0]);flowers[1+j%2].append(petal)
    for i,meshes in enumerate(flowers):
        path=root/'assets'/f'begonias_front_{i}.obj'
        if not path.exists():trimesh.util.concatenate(meshes).export(path)
        scene.add(rss.StaticSpec(name=f'begonias_{i}',asset=rss.AssetSpec(path=str(path)),color=((.15,.29,.10),(.83,.31,.40),(.95,.58,.62))[i],finish=Finish(roughness=.78),collide=False))
    scene.add(rss.slab(name='garden_ground',half=(1.5,2.5,.08),at=(1.9,0,-.75),color=(.22,.28,.09),collide=True))
    scene.add(rss.slab(name='room_floor',half=(3,3,.08),at=(-.3,0,-.83),color=(.34,.27,.18),finish=Finish(roughness=.8),collide=True))
    plaster=Finish(roughness=.95)
    for name,half,at in [
        ('room_right_return',(.07,1.3,1.5),(.40,-2.35,.75)),
        ('room_left_return',(.07,1.2,1.5),(.40,2.92,.75)),
        ('room_window_head',(.07,3.,.85),(.40,.32,2.20)),
        ('room_ceiling',(2.7,3.,.06),(-2.1,0,3.0)),
        ('room_side_wall',(1.8,.07,1.5),(-1.35,-1.10,.75)),
        ('room_under_window',(.07,1.4,.35),(.40,.32,-.42)),
    ]:
        scene.add(rss.slab(name=name,half=half,at=at,color=(.46,.57,.55),finish=plaster,collide=True))
    face=np.array([[.65,1.25,-.7],[1.15,.43,-.7],[1.15,.43,.58],[.65,1.25,.80]])
    path=root/'assets'/'garden_return_wall_r3.obj'
    if not path.exists():trimesh.convex.convex_hull(np.concatenate([face,face+[.06,.07,0]])).export(path)
    scene.add(rss.StaticSpec(name='garden_return_wall',asset=rss.AssetSpec(path=str(path)),color=(.45,.48,.49),finish=Finish(roughness=.95, opacity=.90, ior=1.05),collide=True))

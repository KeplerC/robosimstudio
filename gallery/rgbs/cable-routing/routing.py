"""Measured contact-only insertion, short horizontal feeds, catch and pull."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from motion import Motion
from contacts import material_contacts
from robosimstudio.engines.geometry import read_shape_geometry

PINCH=(.70710678,.70710678,0.,0.)
BASE=Rotation.from_quat(PINCH)
PRESS=tuple((Rotation.from_euler('z',30,degrees=True)*Rotation.from_euler('x',-45,degrees=True)*BASE).as_quat())
CATCH=tuple((BASE*Rotation.from_euler('z',180,degrees=True)).as_quat())
MOUTH=-.24
EXIT=.14
LEFT=.24
AXIS=.0085

def normals(sim,name):
    c=sim[name].entity.bridge.contact_points(name)
    return {int(b):float(np.abs(np.sum(c['force'][c['body']==b]*c['normal'][c['body']==b],axis=1)).sum()) for b in np.unique(c['body'])}

def reading(sim,max_box=0.):
    points=sim['cable'].particle_q;bp=np.asarray(sim['box'].pose)
    local=Rotation.from_quat(bp[3:]).inv().apply(points-bp[:3]);intervals=[]
    for a,b in zip(local[:-1],local[1:]):
        for t in np.linspace(0,1,8,endpoint=False):
            p=a+(b-a)*t;q=a+(b-a)*(t+.125)
            if max(abs(p[0]),abs(q[0]))<.00675 and min(p[2],q[2])>.0015 and max(p[2],q[2])<.0155:
                lo=max(MOUTH,min(p[1],q[1]));hi=min(EXIT,max(p[1],q[1]))
                if hi>lo:intervals.append((lo,hi))
    covered=0.;edge=MOUTH
    for lo,hi in sorted(intervals):covered+=max(0.,hi-max(lo,edge));edge=max(edge,hi)
    ds=np.linalg.norm(np.diff(points,axis=0),axis=1);mid=(local[:-1]+local[1:])/2
    tail=mid[:,1]<MOUTH
    mat=((points[:-1,2]+points[1:,2])/2)<.755
    f=normals(sim,'cable')
    return dict(cord_length_mm=float(ds.sum()*1000),tail_outside_mouth_mm=float(ds[tail].sum()*1000),tail_on_mat_mm=float(ds[tail&mat].sum()*1000),emerged_arc_mm=float(ds[mid[:,1]>EXIT].sum()*1000),tip_above_box_mm=float(local[-1,2]*1000),time_s=round(sim.time,3),channel_coverage=round(covered/(EXIT-MOUTH),4),tip_world_m=points[-1].tolist(),tip_past_box_left_mm=float(local[-1,1]-LEFT)*1000,left_tip_distance_mm=float(np.linalg.norm(sim.arm('left').ee_pose(sim.state)[0]-points[-1]))*1000,left_finger_contact_N={'left':f.get(6,0.),'right':f.get(7,0.)},box_max_translation_mm=max_box*1000)

class Routing:
    def __init__(self,sim):
        self.sim=sim;self.m=Motion(sim);self.events=[];self.press_ready=False
        self.left_gap_indices=[int(sim.model.joint_q_start.numpy()[i]) for i,n in enumerate(sim.model.joint_label) if n.endswith('/left_left_finger') or n.endswith('/left_right_finger')]
        self.gap_indices=[int(sim.model.joint_q_start.numpy()[i]) for i,n in enumerate(sim.model.joint_label) if n.endswith('/right_left_finger') or n.endswith('/right_right_finger')]
    def points(self):return self.sim['cable'].particle_q
    def bp(self):return np.asarray(self.sim['box'].position)
    def until(self,t):yield from self.m.hold(max(0.,t-self.sim.time))
    def note(self,phase,**values):
        row=dict(phase=phase,time_s=self.sim.time,tip=self.points()[-1].tolist(),box_corner_mm=self.m.max_corner*1000,box_yaw_deg=np.rad2deg(self.m.max_yaw),**values)
        self.events.append(row);print('MEASURE '+json.dumps(row),flush=True)
    def gap(self):return float(np.abs(self.sim.state.joint_q.numpy()[self.gap_indices]).sum())
    def open_right(self):
        self.m.grips['right']=.65
        for i in range(15):
            yield from self.m.hold(.02)
            f=normals(self.sim,'cable')
            if i>=4 and self.gap()>.025 and max(f.get(14,0),f.get(15,0))<.02:break
        self.note('jaws open',gap_mm=self.gap()*1000)
    def pickup(self):
        m=self.m;bp=self.bp()
        yield from m.close('left',1.,.16)
        left_gap=float(np.abs(self.sim.state.joint_q.numpy()[self.left_gap_indices]).sum())
        self.press_ready=left_gap<.010
        if not self.press_ready:
            self.note('press unavailable: jaws did not close',gap_mm=left_gap*1000)
            return
        yield from m.move('left',bp+[-.008,.105,.080],.65,PRESS)
        yield from m.move('left',bp+[-.008,.105,.032],.50,PRESS)
        yield from self.open_right()
        p=self.points()[-5].copy();p[2]+=.085
        yield from m.move('right',p,.8,PINCH)
        yield from self.until(4.5)
        p=self.points()[-5].copy();p[2]=.7665
        yield from m.move('right',p,.60,PINCH)
        yield from m.hold(.10)
        for _ in range(3):
            h=m.arms['right'].ee_pose(self.sim.state)[0].copy();node=self.points()[-5].copy()
            p[:2]+=node[:2]-h[:2]
            yield from m.move('right',p,.16,PINCH)
        self.note('before pinch',ee=m.arms['right'].ee_pose(self.sim.state)[0].tolist(),node=self.points()[-5].tolist())
        yield from m.close('right',.970,.30)
        self.note('pinch',cord_normal_N=normals(self.sim,'cable'))
        p=m.arms['right'].ee_pose(self.sim.state)[0].copy();p[2]+=.12
        yield from m.move('right',p,.70,PINCH)
        yield from m.hold(.10)
        self.note('pickup',cord_normal_N=normals(self.sim,'cable'),connector_normal_N=normals(self.sim,'connector'))
    def finger_front(self):
        bp=self.bp();model=self.sim.model;body=self.sim.state.body_q.numpy();sb=model.shape_body.numpy();front=-1e9
        for i in np.where(np.isin(sb,[14,15]))[0]:
            geom=read_shape_geometry(model,int(i));mesh=geom.mesh()
            if mesh is None:continue
            b=body[sb[i]];t=geom.local;R=Rotation.from_quat(b[3:])*Rotation.from_quat(t[3:]);c=Rotation.from_quat(b[3:]).apply(t[:3])+b[:3]
            vertices=R.apply(mesh[0])+c
            if vertices[:,2].min()<bp[2]+.025:front=max(front,float(vertices[:,1].max()))
        return front
    def insert(self):
        m=self.m;bp=self.bp();a=m.arms['right'];p=a.ee_pose(self.sim.state)[0].copy();offset=p-self.points()[-1]
        target=bp+np.array([0,MOUTH-.055,AXIS]);goal=target+offset
        yield from m.move('right',goal,.65,PINCH)
        yield from m.hold(.15)
        for k in range(4):
            h,q=a.ee_pose(self.sim.state);tip=self.points()[-1].copy();direction=tip-self.points()[-2];direction/=np.linalg.norm(direction)
            rot=Rotation.align_vectors([[0.,1.,0.]],[direction])[0];rv=rot.as_rotvec();angle=np.linalg.norm(rv)
            if angle>np.deg2rad(35):rot=Rotation.from_rotvec(rv*np.deg2rad(35)/angle)
            goal=target+rot.apply(h-tip);orient=(rot*Rotation.from_quat(q)).as_quat()
            yield from m.move('right',goal,.20,orient)
            yield from m.hold(.10)
            self.note('align',ee=a.ee_pose(self.sim.state)[0].tolist(),axis=(self.points()[-1]-self.points()[-2]).tolist())
            d=self.points()[-1]-self.points()[-2];d/=np.linalg.norm(d)
            if np.linalg.norm(self.points()[-1]-target)<.001 and d[1]>.996:break
        start,orient=a.ee_pose(self.sim.state);start=start.copy();tip=self.points()[-1].copy()
        allowance=self.finger_front()-start[1];goal=start.copy();goal[1]=min(start[1]+.080,bp[1]+MOUTH-.019-allowance)
        yield from m.move('right',goal,.40,orient)
        yield from m.hold(.12)
        self.note('insert',hand_advance_mm=((a.ee_pose(self.sim.state)[0]-start)*1000).tolist(),tip_advance_mm=((self.points()[-1]-tip)*1000).tolist(),finger_clearance_mm=(bp[1]+MOUTH-.015-self.finger_front())*1000,contacts={n:material_contacts(self.sim,n) for n in ('cable','connector')})
    def feed(self):
        m=self.m;a=m.arms['right'];before=self.points()[-1].copy();bp=self.bp();mouth=bp[1]+MOUTH
        yield from self.open_right()
        pp=self.points()
        # Trace from the fed end to the first crossing of the regrip plane.
        # A long tail may share y with the entry strand; it is not that strand.
        ids=np.where(pp[:,1]<=mouth-.090)[0]
        ix=int(ids[-1])
        if ix+1<len(pp) and abs(pp[ix+1,1]-(mouth-.090))<abs(pp[ix,1]-(mouth-.090)):ix+=1
        node=pp[ix].copy()
        # Open jaws travel back around the cord; their measured opening clears it.
        direction=pp[min(ix+1,len(pp)-1)]-pp[max(ix-1,0)];direction/=np.linalg.norm(direction)
        tangent=Rotation.align_vectors([direction],[[0.,1.,0.]])[0]
        grasp_q=(tangent*BASE).as_quat()
        pad_offset=tangent.apply([0.,0.,.010])
        h,q=a.ee_pose(self.sim.state)
        yield from m.move('right',h+[0.,0.,.030],.10,q)
        p=node+pad_offset+[0.,0.,.030]
        yield from m.move('right',p,.14,grasp_q)
        for approach_step in range(3):
            pp=self.points();node=pp[ix].copy()
            direction=pp[min(ix+1,len(pp)-1)]-pp[max(ix-1,0)];direction/=np.linalg.norm(direction)
            tangent=Rotation.align_vectors([direction],[[0.,1.,0.]])[0]
            grasp_q=(tangent*BASE).as_quat();pad_offset=tangent.apply([0.,0.,.010])
            p=node+pad_offset
            yield from m.move('right',p,.12,grasp_q)
            yield from m.hold(.04)
            if np.linalg.norm(self.points()[ix]+pad_offset-a.ee_pose(self.sim.state)[0])<.001:break
        yield from m.close('right',.970,.14)
        contact=self.sim['cable'].entity.bridge.contact_points('cable')
        weights=np.abs(np.sum(contact['force']*contact['normal'],axis=1));keep=np.isin(contact['body'],[14,15])&(weights>.01)
        if not np.any(keep) or min(normals(self.sim,'cable').get(b,0.) for b in (14,15))<.1:
            self.note('regrip has no cord contact');return
        centre=np.average(contact['pos'][keep],axis=0,weights=weights[keep])
        ix=int(np.argmin(np.linalg.norm(self.points()-centre,axis=1)))
        ix=max(1,min(len(self.points())-2,ix))
        self.note('regrip contact',grip_index=ix,contact_centre=centre.tolist(),node=self.points()[ix].tolist(),ee=a.ee_pose(self.sim.state)[0].tolist(),grip_normal_N=normals(self.sim,'cable'))
        target=np.array([bp[0],self.points()[ix,1],bp[2]+AXIS])
        for correction_step in range(3):
            h,q=a.ee_pose(self.sim.state);node=self.points()[ix].copy()
            direction=self.points()[ix+1]-self.points()[ix-1];direction/=np.linalg.norm(direction)
            rot=Rotation.align_vectors([[0.,1.,0.]],[direction])[0]
            p=target+rot.apply(h-node)
            orient=(rot*Rotation.from_quat(q)).as_quat()
            yield from m.move('right',p,.12,orient)
            yield from m.hold(.04)
            d=self.points()[ix+1]-self.points()[ix-1];d/=np.linalg.norm(d)
            if np.linalg.norm(self.points()[ix]-target)<.001 and d[1]>.998:break
        self.note('feed aligned',ee=a.ee_pose(self.sim.state)[0].tolist(),grip_node=self.points()[ix].tolist(),axis=(self.points()[ix+1]-self.points()[ix-1]).tolist(),grip_normal_N=normals(self.sim,'cable'))
        start=a.ee_pose(self.sim.state)[0].copy();tip=self.points()[-1].copy();goal=start.copy();goal[1]=mouth-.019-(self.finger_front()-start[1])
        f0=normals(self.sim,'cable');grip_start=self.points()[ix].copy()
        yield from m.move('right',goal,.32,orient)
        yield from m.hold(.12)
        self.note('feed',finger_clearance_mm=(mouth-.015-self.finger_front())*1000,grip_node_advance_mm=((self.points()[ix]-grip_start)*1000).tolist(),hand_advance_mm=((a.ee_pose(self.sim.state)[0]-start)*1000).tolist(),tip_advance_mm=((self.points()[-1]-tip)*1000).tolist(),regrip_tip_mm=((tip-before)*1000).tolist(),grip_normal_N=f0,contacts={n:material_contacts(self.sim,n) for n in ('cable','connector')})
    def run(self):
        m=self.m
        yield from self.pickup()
        if not self.press_ready:
            yield from self.until(25.8)
        else:
            held=max(min(normals(self.sim,n).get(b,0) for b in (14,15)) for n in ('cable','connector'))>.02 and self.points()[-1,2]>.8
            if held:
                yield from self.insert()
                while self.sim.time<19.9 and self.points()[-1,1]<self.bp()[1]+EXIT+.035 and m.max_corner<.005:
                    before=self.points()[-1,1]
                    yield from self.feed()
                    if self.points()[-1,1]-before<.005:
                        self.note('feed stalled');break
            else:self.note('pickup failed')
            yield from self.until(18.6)
            # Remain at the entry, open and unloaded, while the left hand catches.
            yield from self.open_right()
            p=m.arms['left'].ee_pose(self.sim.state)[0].copy()
            self.note('left release start')
            yield from m.move('left',p+[0,.035,.045],.45,PRESS)
            yield from m.close('left',.65,.16)
            self.note('left release clear')
            if self.points()[-1,1]>self.bp()[1]+EXIT+.030 and m.max_corner<.005:
                # Seat the 12 mm-wide native pads on uniform cord, with 4 mm
                # clearance from the connector neck. Track before closing.
                def catch_point():
                    pp=self.points()
                    return pp[-3]+(pp[-4]-pp[-3])*(2./3.)
                p=catch_point();p[2]+=.045
                yield from m.move('left',p,.35,CATCH)
                p=catch_point();p[2]=max(p[2]+.012,self.bp()[2]+.015)
                yield from m.move('left',p,.20,CATCH)
                # Correct actual tracking error before the jaws close.
                h=m.arms['left'].ee_pose(self.sim.state)[0]
                p[:2]+=catch_point()[:2]-h[:2]
                yield from m.move('left',p,.16,CATCH)
                yield from m.close('left',.975,.30)
                self.note('left catch',cord_normal_N=normals(self.sim,'cable'))
                yield from self.until(21.)
                p=self.bp()+np.array([.005,.250,.077])
                yield from m.move('left',p,.80,CATCH)
                self.note('lift clear',cord_normal_N=normals(self.sim,'cable'))
                yield from m.close('left',.975,.12)
                p=self.bp()+np.array([.005,.475,.192])
                # Reachable diagonal approach, with the jaw closing axis horizontal.
                horizontal=Rotation.from_euler('z',60,degrees=True)*Rotation.from_euler('x',-75,degrees=True)*Rotation.from_quat(CATCH)
                yield from m.move('left',p,max(.5,24.-self.sim.time),tuple(horizontal.as_quat()))
                self.note('drawn out',cord_normal_N=normals(self.sim,'cable'))
                p[1]-=.100
                yield from m.move('left',p,max(.5,25.-self.sim.time),tuple(horizontal.as_quat()))
            else:self.note('no handoff: end has not emerged')
            yield from self.until(25.8)
        result=reading(self.sim,m.max_box);result.update(box_max_corner_mm=m.max_corner*1000,box_max_yaw_deg=np.rad2deg(m.max_yaw))
        print('OUTCOME '+json.dumps(result,sort_keys=True),flush=True)
        run_dir=Path(__file__).resolve().parent/'run';run_dir.mkdir(parents=True,exist_ok=True)
        telemetry=dict(outcome=result,events=self.events,contacts=m.history)
        with (run_dir/'latest_telemetry.json').open('w') as f:json.dump(telemetry,f,indent=2)
        with (run_dir/'telemetry_history.jsonl').open('a') as f:f.write(json.dumps(telemetry)+'\n')
        passed=result['tail_on_mat_mm']>20 and result['emerged_arc_mm']>330 and result['tip_above_box_mm']>140 and result['channel_coverage']>=.8 and result['tip_past_box_left_mm']>0 and min(result['left_finger_contact_N'].values())>.02 and result['left_tip_distance_mm']<35 and m.max_corner<.005
        if not passed:raise RuntimeError('Routing goal not met; see OUTCOME. Contact-only attempt is incomplete.')

def routine(sim):yield from Routing(sim).run()

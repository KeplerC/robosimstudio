"""Timed arm targets. The routine never writes a sponge node or a prop pose."""
import json
from pathlib import Path
import math
import numpy as np
from robosimstudio.core.math import quat_multiply, quat_to_mat

GRIP_FORCE=5.

def wipe_quat(tilt=0,lean=-60):
    down=quat_multiply((0,math.sin(math.radians(lean)/2),0,math.cos(math.radians(lean)/2)),(0,1,0,0))
    t=math.radians(tilt)
    return np.asarray(quat_multiply((math.sin(t/2),0,0,math.cos(t/2)),down))

def slerp(a,b,t):
    a=np.asarray(a);b=np.asarray(b)
    d=float(np.dot(a,b))
    if d<0:b=-b;d=-d
    if d>.9995:
        q=a+t*(b-a);return q/np.linalg.norm(q)
    w=math.acos(np.clip(d,-1,1))
    return (math.sin((1-t)*w)*a+math.sin(t*w)*b)/math.sin(w)

class Take:
    def __init__(self,sim):
        self.sim=sim
        self.names=list(sim.handles.arms)
        self.ik={n:sim.arm(n) for n in self.names}
        self.seed={n:sim.state.joint_q.numpy().copy() for n in self.names}
        self.action=np.asarray(sim.idle_action('joint_position'),dtype=float)
        self.home=self.action.copy()
        self.field=sim['coverage'].entity
        self.q=wipe_quat(0)
        self.trace=[]
        self.phase='start'
        self.n=0
        self.bar_start=np.asarray(sim['bar'].position)
        self.bar_drift=0.
        self.result={}
        self.grip_feedback=False
        self.grip_gap=.054
        self.grip_min=.044
        self.grip_ema=GRIP_FORCE
        self.grip_target=GRIP_FORCE
        self.finger_forces={'left':0.,'right':0.}
    def points(self):return np.asarray(self.sim['sponge'].particle_q)
    def read_fingers(self):
        contacts=self.sim['sponge'].entity.bridge.contact_points('sponge')
        forces={'left':0.,'right':0.}
        for body in np.unique(contacts['body']):
            label=str(self.sim.model.body_label[body]) if body>=0 else ''
            if '/right_arm/' not in label:continue
            for side in forces:
                if f'link_{side}_finger' in label:
                    forces[side]+=float(np.linalg.norm(contacts['force'][contacts['body']==body],axis=1).sum())
        return forces
    def tick(self):
        self.sim.step(self.action,mode='joint_position')
        self.n+=1
        self.bar_drift=max(self.bar_drift,float(np.linalg.norm(np.asarray(self.sim['bar'].position)-self.bar_start)))
        if self.n%3==0:
            self.finger_forces=self.read_fingers()
            if self.grip_feedback:
                f=min(self.finger_forces.values())
                self.grip_ema=.5*self.grip_ema+.5*f
                self.grip_gap=float(np.clip(self.grip_gap+np.clip((self.grip_ema-self.grip_target)*.00008,-.001,.001),self.grip_min,.065))
                self.action[self.names.index('right')*7+6]=1-self.grip_gap/self.sim.handles.arms['right'].gripper_span
        if self.n%6==0:
            p=self.points()
            self.trace.append(dict(t=self.n/60,phase=self.phase,centre=p.mean(0).tolist(),low=float(p[:,2].min()),press=self.field.press,coverage=self.field.fraction,wrap_arc_deg=self.field.wrap_arc,tcp=self.ik['right'].ee_pose(self.sim.state)[0].tolist(),grip_left_n=self.finger_forces['left'],grip_right_n=self.finger_forces['right'],finger_q=self.sim.state.joint_q.numpy()[list(self.sim.handles.arms['right'].gripper_coords)].tolist(),command_gap=self.sim.handles.arms['right'].gripper_span*(1-self.action[-1])))
        yield
    def hold(self,seconds):
        for _ in range(round(seconds*60)):yield from self.tick()
    def leg(self,target,seconds,quat=None,arm='right',label='move'):
        self.phase=label
        ik=self.ik[arm]; slot=self.names.index(arm)*7
        pos,rot=ik.ee_pose(self.sim.state)
        target=np.asarray(target);quat=self.q if quat is None else quat
        n=round(seconds*60);prev=self.sim.state.joint_q.numpy().copy();seed=self.seed[arm]
        worst=0.
        for k in range(0,n,6):
            end=min(k+6,n); t=end/n
            f=t if label.startswith('stroke') else t*t*(3-2*t)
            if label.startswith('return'):
                e=.1
                f=t*t/(2*e*(1-e)) if t<e else (1-(1-t)**2/(2*e*(1-e)) if t>1-e else (t-e/2)/(1-e))
            point=pos+(target-pos)*f; qr=slerp(rot,quat,f)
            sol=ik.solve_q_full(seed,point,qr)
            err=ik.pose_errors(sol,point,qr)[0];worst=max(worst,err)
            for j in range(k+1,end+1):
                v=prev+(sol-prev)*(j-k)/(end-k)
                self.action[slot:slot+6]=v[ik.arm_coords]
                yield from self.tick()
            seed=sol;prev=sol
        self.seed[arm]=seed
        print(f'{label}: t={self.n/60:.2f}, IK max {worst*1000:.1f} mm; sponge {self.points().mean(0)}',flush=True)
    def grip(self,gap,seconds=1.):
        self.grip_feedback=False
        self.phase='grasp' if gap else 'release'
        slot=self.names.index('right')*7+6
        start=self.action[slot]
        target=0. if gap is None else np.clip(1-gap/self.sim.handles.arms['right'].gripper_span,0,1)
        n=round(seconds*60)
        for k in range(1,n+1):
            self.action[slot]=start+(target-start)*k/n
            yield from self.tick()
        if gap is not None:
            self.grip_gap=float(gap);self.grip_ema=GRIP_FORCE;self.grip_feedback=True
    def pick(self):
        # Put the foam farther into the stock jaws, rather than pinching only its top edge.
        p=self.points().mean(0)
        self.pickpos=p.copy()
        yield from self.hold(.8)
        yield from self.leg([.64,.190,.860],2.2,quat=quat_multiply((math.sin(math.radians(-35)/2),0,0,math.cos(math.radians(-35)/2)),quat_multiply((0,0,-math.sin(math.pi/4),math.cos(math.pi/4)),(1,0,0,0))),arm='left',label='left hand to elbow')
        self.action[self.names.index('left')*7+6]=1-.022/self.sim.handles.arms['left'].gripper_span
        yield from self.leg(p+[0,0,.08],2.,label='approach')
        yield from self.leg(p,1.5,label='lower')
        yield from self.grip(.054,1.)
        yield from self.hold(.25)
        yield from self.leg([p[0],p[1],.945],1.,label='vertical lift')
        self.q=wipe_quat(0)
        yield from self.leg([.54,-.26,.975],.75,label='lift clear of foot')
        yield from self.hold(.5)
        low=self.points()[:,2].min()
        print('lift bottom',low,'gripper',self.action[-1],flush=True)
        if low<.89:raise RuntimeError('Sponge did not clear the bar in the fingers')
    def run(self):
        yield from self.pick()
        tcp=self.ik['right'].ee_pose(self.sim.state)[0]
        offset=self.points().mean(0)-tcp
        # At 10 s, seat near the right end. Actual foam underside sets clearance.
        self.offset=offset.copy()
        y0=-.162-offset[1]
        points=self.points()
        mid=points[np.abs(points[:,0]-points[:,0].mean())<.012]
        drop=tcp[2]-float(np.quantile(mid[:,2],.05))
        z=.8925+drop
        yield from self.leg([.64-offset[0],y0,z+.040],.8,label='over right end')
        yield from self.grip(.044,.2)
        self.grip_feedback=False
        yield from self.leg([.64-offset[0],y0,z-.020],.5,label='seat')
        # Five short loaded strokes, separated by actual clearance and re-seating.
        # Waypoints read at 0.5 s intervals from the real clip, in real seconds.
        strokes=[(13.1,-.09,-.025,'stroke 1'),(13.65,-.15,.040,'lift 1'),
                 (14.1,-.15,-.025,'repress 2'),(15.55,-.050,-.025,'stroke 2'),
                 (16.0,-.065,.045,'lift 2'),(16.6,-.065,-.032,'repress 3'),
                 (18.05,.025,-.032,'stroke 3'),(18.65,.005,.045,'lift 3'),
                 (19.1,.005,-.037,'repress 4'),(20.7,.100,-.037,'stroke 4'),
                 (21.3,.070,.040,'lift 4'),(21.75,.070,-.042,'repress 5'),
                 (23.0,.160,-.042,'stroke 5')]
        for end,y,dz,label in strokes:
            self.q=wipe_quat(0.)
            yield from self.leg([.64-offset[0],y-offset[1],z+dz],end-self.n/60,quat=self.q,label=label)
            if label.startswith('lift') and sum(self.read_fingers().values())<.05 and self.points()[:,2].min()<.88:
                raise RuntimeError('Sponge lost at '+label)
        ik=self.ik['right']
        print(f'coverage {self.field.fraction:.3f}, peak {self.field.peak:.3f} N',flush=True)
        self.grip_target=GRIP_FORCE
        self.grip_min=.032
        self.grip_ema=GRIP_FORCE
        self.grip_feedback=True
        tcp=ik.ee_pose(self.sim.state)[0]
        yield from self.leg([.51,min(tcp[1],.05),tcp[2]+.080],.75,label='off bar')
        yield from self.leg([.51,-.3453,.90],1.75,quat=self.q,label='return in front of bar')
        if min(self.read_fingers().values())<.1 or self.points()[:,2].min()<.82:
            raise RuntimeError('Sponge lost before the put-down')
        p=self.points(); tcp=ik.ee_pose(self.sim.state)[0]
        target=tcp+np.array([.64-p[:,0].mean(),-.3453-p[:,1].mean(),.761-p[:,2].min()])
        yield from self.leg(target,.75,quat=self.q,label='set down')
        yield from self.hold(.1)
        yield from self.grip(None,.5)
        self.action[self.names.index('left')*7+6]=0.
        yield from self.leg(self.pickpos+[0,0,.12],.75,label='retreat')
        self.phase='home';start=self.action.copy()
        for k in range(90):
            t=(k+1)/90;f=t*t*(3-2*t);self.action=start+(self.home-start)*f
            yield from self.tick()
        yield from self.hold(1.5)
        p=self.points()
        contacts=self.sim['sponge'].entity.bridge.contact_points('sponge')
        finger_force=0.
        for body in np.unique(contacts['body']):
            label=str(self.sim.model.body_label[body]) if body>=0 else ''
            if 'right' in label and 'finger' in label:
                finger_force+=float(np.linalg.norm(contacts['force'][contacts['body']==body],axis=1).sum())
        speed=float(np.linalg.norm(np.asarray(self.trace[-1]['centre'])-np.asarray(self.trace[-6]['centre']))/.5)
        _,axes=np.linalg.eigh(np.cov(p[:,:2].T));axis=axes[:,-1]
        long_angle=(float(np.degrees(np.arctan2(axis[1],axis[0])))+90)%180-90
        self.result=dict(final_bounds_min=p.min(0).tolist(),final_bounds_max=p.max(0).tolist(),
            final_long_axis_deg=long_angle,foot_gap_mm=float((-.264-p[:,1].max())*1000),coverage=self.field.fraction,peak_crown_force_n=self.field.peak,max_wrap_arc_deg=self.field.max_wrap_arc,
            bar_drift_mm=1000*self.bar_drift,final_centre=p.mean(0).tolist(),
            bottom_mm_above_mat=1000*(float(p[:,2].min())-.760),final_finger_force_n=finger_force,
            final_speed_m_s=speed,duration_s=self.n/60)
        print('RESULT '+json.dumps(self.result),flush=True)
        self.save()
        if self.field.fraction<.90:raise RuntimeError(f'Coverage {self.field.fraction:.1%} below 90%')
        if self.bar_drift>.002:raise RuntimeError('Bar moved more than 2 mm')
        if abs(float(p[:,2].min())-.760)>.005 or p.mean(0)[1]>-.30:
            raise RuntimeError('Sponge did not end on the mat to the right')
        if np.linalg.norm(p.mean(0)[:2]-np.array([.64,-.3453]))>.020 or abs(long_angle)>12:
            raise RuntimeError('Sponge missed the clip put-down footprint')
        if finger_force>.05 or speed>.02:raise RuntimeError('Sponge is not released and settled')
    def save(self):
        Path('run').mkdir(exist_ok=True)
        Path('run/last_trace.json').write_text(json.dumps(self.trace,indent=2))
        Path('run/last_result.json').write_text(json.dumps(self.result,indent=2))
        Path('run/last_exposure.json').write_text(json.dumps(self.field.exposure.tolist()))

def routine(sim):
    # The same contact routine, with read-only evidence sampled at clip moments.
    from probes import wrap_probe
    yield from wrap_probe(sim)

def grasp_probe(sim):
    global last_take
    last_take=Take(sim)
    try:
        yield from last_take.pick()
        yield from last_take.hold(2.)
    finally:last_take.save()

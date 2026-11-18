"""Time-based joint targets from seeded IK. Only the robot receives commands."""
import numpy as np
from contacts import material_contacts
from scipy.spatial.transform import Rotation

DOWN=(0.,1.,0.,0.)

class Motion:
    def __init__(self,sim):
        self.sim=sim
        self.seed=sim.state.joint_q.numpy().copy()
        self.arms={n:sim.arm(n) for n in ('left','right')}
        self.targets={n:self.seed[a.arm_coords].copy() for n,a in self.arms.items()}
        self.grips={n:0. for n in self.arms}
        self.hz=float(sim.request.physics.control_hz)
        self.start=np.asarray(sim['box'].position)
        self.max_box=0.
        self.start_pose=np.asarray(sim['box'].pose).copy()
        self.local_corners=np.array([(x,y,z) for x in (-.0165,.2435) for y in (-.24,.24) for z in (-.11,.023)])
        self.start_corners=Rotation.from_quat(self.start_pose[3:]).apply(self.local_corners)+self.start_pose[:3]
        self.max_corner=0.;self.max_yaw=0.
        self.history=[]
        self.left_gap_indices=[int(sim.model.joint_q_start.numpy()[i]) for i,n in enumerate(sim.model.joint_label) if n.endswith('/left_left_finger') or n.endswith('/left_right_finger')]
    def step(self):
        action=[]
        for n in ('left','right'):
            action.extend(self.targets[n]); action.append(self.grips[n])
        self.sim.step(action,mode='joint_position')
        pose=np.asarray(self.sim['box'].pose)
        self.max_box=max(self.max_box,float(np.linalg.norm(pose[:3]-self.start)))
        corners=Rotation.from_quat(pose[3:]).apply(self.local_corners)+pose[:3]
        self.max_corner=max(self.max_corner,float(np.linalg.norm(corners-self.start_corners,axis=1).max()))
        delta=(Rotation.from_quat(pose[3:])*Rotation.from_quat(self.start_pose[3:]).inv()).as_matrix()
        self.max_yaw=max(self.max_yaw,abs(float(np.arctan2(delta[1,0],delta[0,0]))))
        if (18.4<self.sim.time<25.81) or round(self.sim.time*self.hz)%25==0:
            self.sim.contacts.update(self.sim.step_count)
            sb=self.sim.model.shape_body.numpy()
            contacts={}
            for side,bodyids in [('left',[6,7]),('right',[14,15]),('left_wrist',[4,5]),('right_wrist',[12,13])]:
                fingers=np.where(np.isin(sb,bodyids))[0]
                byshape={}
                for shape in np.where(sb==16)[0]:
                    force=self.sim.contacts.pairwise_force(fingers,[int(shape)])
                    if np.linalg.norm(force)>.01:byshape[self.sim.model.shape_label[shape]]=np.asarray(force).tolist()
                contacts[side+'_box_forces_N']=byshape
            grasp={}
            if self.sim.time>=19.0:
                hand,quat=self.arms['left'].ee_pose(self.sim.state)
                points=self.sim['cable'].particle_q
                force={}
                for name in ('cable','connector'):
                    c=self.sim[name].entity.bridge.contact_points(name)
                    force[name]={str(b):float(np.abs(np.sum(c['force'][c['body']==b]*c['normal'][c['body']==b],axis=1)).sum()) for b in (6,7)}
                grasp=dict(hand=hand.tolist(),quat=quat.tolist(),end_nodes=points[-8:].tolist(),node_in_hand=Rotation.from_quat(quat).inv().apply(points[-3]-hand).tolist(),jaw_mm=float(np.abs(self.sim.state.joint_q.numpy()[self.left_gap_indices]).sum()*1000),normal_N=force,shape_contacts={n:material_contacts(self.sim,n) for n in ('cable','connector')},joint_error=(self.targets['left']-self.sim.state.joint_q.numpy()[self.arms['left'].arm_coords]).tolist())
            self.history.append(dict(grasp=grasp,time_s=self.sim.time,box_corner_mm=self.max_corner*1000,box_rise_mm=(pose[2]-self.start_pose[2])*1000,**contacts))
    def hold(self,seconds):
        for _ in range(round(seconds*self.hz)):
            self.step(); yield
    def move(self,name,pos,seconds,quat=DOWN):
        arm=self.arms[name]
        p0,q0=arm.ee_pose(self.sim.state)
        # Seed all waypoint solves from the previous target to keep its elbow.
        knots=max(2,round(seconds*10))
        ticks=max(1,round(seconds*self.hz/knots))
        for k in range(1,knots+1):
            u=k/knots
            p=np.asarray(p0)*(1-u)+np.asarray(pos)*u
            endq=np.asarray(quat,dtype=float)
            startq=np.asarray(q0,dtype=float)
            if startq@endq<0: endq=-endq
            interp=startq*(1-u)+endq*u
            interp/=np.linalg.norm(interp)
            with arm.host_fk():
                q=arm.solve_q_full(self.seed,p,interp,polish_passes=3)
                actual, orient=arm.fk_ee(q)
                error=np.linalg.norm(actual-p)+.1*(1-abs(np.dot(orient,interp)))
                if error>.001:
                    for guess in ([0,1.6,1.2,-1.1,0,1.6],[0,1.6,1.2,-1.1,0,-1.6],[0,0,0,0,0,0]):
                        seed=self.seed.copy();seed[arm.arm_coords]=guess
                        candidate=arm.solve_q_full(seed,p,interp,polish_passes=3)
                        cp,cq=arm.fk_ee(candidate)
                        ce=np.linalg.norm(cp-p)+.1*(1-abs(np.dot(cq,interp)))
                        if ce<error:q=candidate;error=ce
                        if error<.0001:break
            prev=self.targets[name].copy()
            target=q[arm.arm_coords]
            self.seed[arm.arm_coords]=target
            for j in range(1,ticks+1):
                self.targets[name]=prev+(target-prev)*j/ticks
                self.step(); yield
    def close(self,name,value,seconds=.4):
        self.grips[name]=value
        yield from self.hold(seconds)
    def report(self,label):
        print(label,'t',round(self.sim.time,2),'box_max_mm',round(self.max_box*1000,2),
              'hands',{n:np.round(a.ee_pose(self.sim.state)[0],4).tolist() for n,a in self.arms.items()},flush=True)

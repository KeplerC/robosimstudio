"""Contact grasp and constant-speed brush strokes through the studio arm panel."""
import json
import math
from pathlib import Path
import numpy as np
import letters as Lt
from surface import IX, IY, CX, CY

LEAD = .010
TOOL_DOWN = np.array([0.,1.,0.,0.])

def tick(n):
    for _ in range(n):
        yield

def hand(session):
    return np.asarray(session.arms['right'].ik.ee_pose(session.env.state)[0])

def walk(session,pts,speed,each=None,settle=0,hold=False):
    arm=session.arms['right']
    seq=[np.asarray(arm.cursor_pos).copy()]+[np.asarray(p,float) for p in pts]
    length=np.linalg.norm(np.diff(seq,axis=0),axis=1)
    cum=np.r_[0,np.cumsum(length)]
    total=cum[-1]
    n=max(1,math.ceil(total/(speed/60)))
    for k in range(1,n+1):
        s=total*k/n
        j=min(int(np.searchsorted(cum,s,side='right'))-1,len(length)-1)
        u=0 if length[j]==0 else (s-cum[j])/length[j]
        arm.cursor_pos=seq[j]+(seq[j+1]-seq[j])*u
        arm.dragging=k<n or hold
        yield
        if each: each()
    arm.cursor_pos=seq[-1]; arm.dragging=hold
    yield from tick(settle)

def go(session,target,speed=.1,tol=.004,steps=240):
    yield from walk(session,[target],speed)
    for _ in range(steps):
        if np.linalg.norm(hand(session)-target)<tol: return
        yield
    raise RuntimeError(f'hand missed target by {1000*np.linalg.norm(hand(session)-target):.1f} mm')

def score(sim,save=False):
    g=sim['brush'].entity.scrub_grid()
    clean=g.cells[np.ix_(IY,IX)]>0
    strokes=[(n,Lt.to_world(uv,.620,-.311)) for n,uv in Lt.strokes_uv()]
    masks={n:Lt.mask([p],CX[IX],CY[IY]) for n,p in strokes}
    mask=np.logical_or.reduce(list(masks.values()))
    per={}
    for k,(n,m) in enumerate(masks.items()):
        lo=-.311+k*(Lt.L+Lt.GAP)-Lt.GAP/2
        box=((CY[IY]>=lo)&(CY[IY]<lo+Lt.L+Lt.GAP))[:,None]
        per[n]=Lt.iou(clean&box,m)
    out={'iou':Lt.iou(clean,mask),'letters':per,'outside_cells':int((clean&~mask).sum()),'missed_cells':int((mask&~clean).sum()),'mask_cells':int(mask.sum()),'finite':bool(sim.state_is_finite())}
    if save:
        path=Path(__file__).parent/'run'
        (path/'score.json').write_text(json.dumps(out,indent=2))
        np.savez(path/'measured_mask.npz',clean=clean,mask=mask,cx=CX[IX],cy=CY[IY])
    return out

def routine(sim,session):
    from page_dust import install_view
    install_view(session)
    entity=sim['brush'].entity
    arm=session.arms['right']
    start_step=sim.step_count
    yield from tick(35)
    arm.cursor_quat=TOOL_DOWN.copy()
    standing=np.asarray(sim['handle'].position).copy()
    grasp=np.array([standing[0],standing[1],entity.floor_z+entity.spec.sole_top+.020+.0165])
    yield from go(session,grasp+[0,0,.06],.25)
    yield from go(session,grasp,.08)
    arm.grip_request=1.
    yield from tick(45)
    width=float(sim.gripper('right').width(session.env.state))
    offset=np.asarray(sim['handle'].position)-hand(session)
    yield from walk(session,[grasp+[0,0,.04]],.08,settle=20)
    rise=float(sim['handle'].position[2]-standing[2])
    if rise<.025: raise RuntimeError(f'brush not lifted: {rise*1000:.1f} mm')
    tips=entity.tip_positions(session.env.state.body_q.numpy())
    held=tips[:,:2].mean(axis=0)-hand(session)[:2]
    print('BRUSH grasp',{'width_mm':width*1000,'rise_mm':rise*1000,'held_mm':(held*1000).tolist()},flush=True)
    zp=grasp[2]-.004; up=grasp[2]+.025
    entity.clear_scrub()
    stats={'slip_mm':0.,'down_min':1.}
    trace=[]
    def track():
        stats['slip_mm']=max(stats['slip_mm'],1000*float(np.linalg.norm(np.asarray(sim['handle'].position)-hand(session)-offset)))
        tips=entity.tip_positions(session.env.state.body_q.numpy())
        trace.append(np.r_[arm.cursor_pos,hand(session),tips.mean(axis=0)])
        stats['down_min']=min(stats['down_min'],float(((tips[:,2]-entity.floor_z)<=entity.spec.bristle_radius+.002).mean()))
    for name,uv in Lt.strokes_uv():
        pts=Lt.lead_path(Lt.to_world(uv,.620,-.311),LEAD)-held[None,:]
        s0=pts[0]
        if name=='R': yield from go(session,[*s0,up],.25,tol=.004)
        else: yield from walk(session,[[*s0,up]],.16)
        yield from walk(session,[[*s0,zp]],.07,settle=8)
        yield from walk(session,[[x,y,zp] for x,y in pts],.12,each=track)
        yield from tick(15)
        print('BRUSH',name,score(sim),flush=True)
        yield from walk(session,[[*pts[-1],up]],.08)
    np.save(Path(__file__).parent/'run/tip_trace.npy',np.asarray(trace))
    result=score(sim,True)
    result.update(stats)
    result.update(width_mm=width*1000,rise_mm=rise*1000,lead_mm=LEAD*1000)
    (Path(__file__).parent/'run/score.json').write_text(json.dumps(result,indent=2))
    print('BRUSH RESULT',result,flush=True)
    tail_start=sim.step_count
    times={'letters_clear':(tail_start-start_step)/60}
    def mark(name): times[name]=(sim.step_count-start_step)/60
    def quiet_handle(clearance=False,limit=90):
        previous=np.asarray(sim['handle'].position).copy()
        quiet=0
        for _ in range(limit):
            yield
            current=np.asarray(sim['handle'].position).copy()
            clear=not clearance or float(sim.gripper('right').width(session.env.state))>=.030
            quiet=quiet+1 if clear and np.linalg.norm(current-previous)<.000005 else 0
            previous=current
            if quiet>=20: return
        raise RuntimeError('handle did not become quiet with the required clearance')
    yield from go(session,[grasp[0],grasp[1],up+.02],.25)
    mark('over_rest')
    # Hold precision targets so the page's arrival tolerance cannot cancel them.
    yield from walk(session,[grasp+[0,0,-.0022]],.07,hold=True)
    yield from quiet_handle()
    desired=standing+np.array([0.,0.,-.0022])
    for _ in range(3):
        error=desired-np.asarray(sim['handle'].position)
        if np.linalg.norm(error)<.0005: break
        yield from walk(session,[np.asarray(arm.cursor_pos)+error],.03,hold=True)
        yield from quiet_handle()
    mark('seated')
    arm.grip_request=1.-.020/.095
    yield from quiet_handle()
    # Ease the bristle preload through the fingers; the kind does not return
    # that support force to its compensated handle.
    for _ in range(6):
        error=standing+np.array([0.,0.,-.0003])-np.asarray(sim['handle'].position)
        if np.linalg.norm(error)<.00015: break
        yield from walk(session,[np.asarray(arm.cursor_pos)+error],.02,hold=True)
        yield from quiet_handle()
    preopen=np.asarray(sim['handle'].position).copy()
    mark('preopen')
    # Spend most opening time where the pads actually leave the 22 mm handle.
    for lo,hi,count in [(.020,.0214,18),(.0214,.0228,108),(.0228,.024,18)]:
        for k in range(count):
            u=(k+1)/count
            width=lo+(hi-lo)*(.5-.5*math.cos(math.pi*u))
            arm.grip_request=1.-width/.095
            yield
    arm.grip_request=0.
    yield from quiet_handle(clearance=True,limit=150)
    release=np.asarray(sim['handle'].position).copy()
    mark('jaws_clear')
    # Observe the resting bristles while the hand withdraws, not afterwards.
    ids=np.asarray(entity.bodies,dtype=int)
    previous=sim.state.body_q.numpy()[ids,:3].copy()
    bristle_drift=[]
    observed=0
    quiet=0
    def watch_rest():
        nonlocal previous,observed,quiet
        observed+=1
        if observed%60: return
        current=sim.state.body_q.numpy()[ids,:3].copy()
        drift=float(np.linalg.norm(current-previous,axis=1).max())
        bristle_drift.append(drift*1000)
        quiet=quiet+1 if drift<.0005 else 0
        previous=current
    yield from walk(session,[np.asarray(arm.cursor_pos)+[0,0,.10]],.10,each=watch_rest)
    mark('hand_clear')
    yield from walk(session,[[.42,-.36,grasp[2]+.16]],.20,each=watch_rest)
    while quiet<2 and observed<600:
        yield
        watch_rest()
    if quiet<2: raise RuntimeError('bristles did not settle on the rest')
    mark('finished')
    final=np.asarray(sim['handle'].position)
    tips=entity.tip_positions(session.env.state.body_q.numpy())
    returned={'tail_seconds':(sim.step_count-tail_start)/60,'timings':times,'start':standing.tolist(),'preopen':preopen.tolist(),'release':release.tolist(),'end':final.tolist(),
              'distance_mm':float(1000*np.linalg.norm(final-standing)),
              'lift_drift_mm':float(1000*np.linalg.norm(final-release)),
              'bristle_drift_windows_mm':bristle_drift,
              'tip_height_mm':(1000*(tips[:,2]-entity.floor_z)).tolist(),
              'jaw_width_mm':float(1000*sim.gripper('right').width(session.env.state)),
              'end_speed_mm_s':float(1000*np.linalg.norm(np.asarray(sim['handle'].velocity)[:3]))}
    if returned['distance_mm']>.005*1000: raise RuntimeError(f"handle returned {returned['distance_mm']:.2f} mm from its start")
    (Path(__file__).parent/'run/return.json').write_text(json.dumps(returned,indent=2))
    print('BRUSH RETURN',{k:v for k,v in returned.items() if k!='tip_height_mm'},flush=True)
    if result['iou']<.8: raise RuntimeError(f"scrub IoU {result['iou']:.3f} below 0.8")

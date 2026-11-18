"""Read-only evidence for the same contact routine, plus an independent release test."""
import json
from pathlib import Path
import numpy as np
from motion import Take

def finger_evidence(take):
    sim=take.sim
    c=sim['sponge'].entity.bridge.contact_points('sponge')
    rows=[]
    for body in np.unique(c['body']):
        label=str(sim.model.body_label[body]) if body>=0 else ''
        if '/right_arm/' not in label or 'finger' not in label:continue
        keep=c['body']==body
        f=np.linalg.norm(c['force'][keep],axis=1)
        if f.sum()<1e-6:continue
        rows.append(dict(body=label.split('/')[-1],force_norm_sum_n=float(f.sum()),
            force_vector_n=c['force'][keep].sum(0).tolist(),
            contact_centre=np.average(c['pos'][keep],axis=0,weights=f).tolist()))
    return dict(time_s=take.n/60,foam_centre=take.points().mean(0).tolist(),fingers=rows)

def wrap_probe(sim):
    Path('run').mkdir(exist_ok=True)
    import motion
    take=Take(sim)
    motion.last_take=take
    readings=[]
    try:
        for _ in take.run():
            yield
            if take.n==600:
                Path('run/grasp_contacts.json').write_text(json.dumps(finger_evidence(take),indent=2))
            if take.n not in (780,960,1080,1380):continue
            c=sim['sponge'].entity.bridge.contact_points('sponge',against=take.field.bar_body)
            force=np.linalg.norm(c['force'],axis=1)
            live=(force/np.maximum(c['area'],1e-9)>=300)&(c['pos'][:,1]>-.182)&(c['pos'][:,1]<.158)
            angle=np.degrees(np.arctan2(c['pos'][live,0]-.64,c['pos'][live,2]-.88))
            points=take.points();near=points[points[:,0]<.6275]
            result=dict(time_s=take.n/60,contact_samples=int(live.sum()),
                pressed_arc_deg=[float(angle.min()),float(angle.max())] if len(angle) else [],
                arc_span_deg=float(np.ptp(angle)) if len(angle)>1 else 0.,
                bar_force_n=float(force.sum()),crown_force_n=take.field.press,
                finger_force_n=take.read_fingers(),foam_low_z=float(points[:,2].min()),
                near_side_low_z=float(near[:,2].min()) if len(near) else None)
            if take.n==960:
                Path('run/lift_probe.json').write_text(json.dumps(result,indent=2))
            else:
                readings.append(result)
                Path('run/wrap_probe.json').write_text(json.dumps(readings,indent=2))
            np.savez_compressed(f'run/wrap_probe_{take.n//60}.npz',nodes=points,contacts=c['pos'][live],force=c['force'][live])
            print(json.dumps(result),flush=True)
    finally:take.save()

def drop_probe(sim):
    take=Take(sim)
    yield from take.pick()
    start=take.points().mean(0).copy()
    yield from take.grip(None,.5)
    yield from take.hold(1.)
    end=take.points().mean(0)
    result={'released_drop_mm':float((start[2]-end[2])*1000),'start':start.tolist(),
            'end':end.tolist(),'finger_force_n':take.read_fingers()}
    Path('run/soft_drop_probe.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if result['released_drop_mm']<100 or sum(result['finger_force_n'].values())>.05:
        raise RuntimeError('Released foam did not fall clear of the fingers')

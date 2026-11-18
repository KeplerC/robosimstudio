"""Read contact forces by native collider; never changes a state."""
import numpy as np
from robosimstudio.engines.superdex.bridge import _handle_value

def material_contacts(sim,name):
    bridge=sim[name].entity.bridge
    entry=next(e for e in bridge.actors if e.get('entity',e['name'])==name)
    mine=_handle_value(entry['actor'])
    mirrors={_handle_value(m.actor):m for m in bridge.mirrors}
    result={}
    for c in entry['contacts']():
        first=c.actor_a.value==mine
        other=c.actor_b.value if first else c.actor_a.value
        mirror=mirrors.get(other)
        if mirror is None:continue
        label=sim.model.shape_label[mirror.shape]
        row=result.setdefault(label,dict(normal_N=0.,force_N=[0.,0.,0.],min_gap_mm=1e6))
        force=np.asarray(c.force)*(1 if first else -1);normal=np.asarray(c.normal)*(1 if first else -1)
        row['normal_N']+=float(abs(force@normal));row['force_N']=(np.array(row['force_N'])+force).tolist()
        row['min_gap_mm']=min(row['min_gap_mm'],float(c.distance)*1000)
    return result

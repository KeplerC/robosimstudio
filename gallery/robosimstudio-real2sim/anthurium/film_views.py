"""Render another view of measured states; never step or alter the live sim."""
from pathlib import Path
import json
import numpy as np
import imageio.v2 as imageio


def render_saved(sim, source, destination, camera='wide'):
    data=np.load(source)
    spec=next(c for c in sim.cameras if c.name==camera)
    renderer=sim.make_renderer('rtx',cameras=[spec],channels=['rgb'])
    state=sim.model.state()
    times=data['time']
    fps=1/float(np.median(np.diff(times)))
    try:
        with imageio.get_writer(str(destination),fps=fps,codec='libx264',quality=8,macro_block_size=1) as writer:
            for i,(body,particles) in enumerate(zip(data['body_q'],data['particle_q'])):
                state.body_q.assign(body)
                state.particle_q.assign(particles)
                if i==0:
                    renderer.render(state)  # RTX stage warm-up
                writer.append_data(renderer.render(state)[camera]['rgb'])
                if i%50==0:print('wide frame',i,'of',len(times),flush=True)
    finally:
        renderer.close()
    Path(destination).with_suffix('.json').write_text(json.dumps(dict(source=str(source),camera=camera,renderer='rtx',fps=fps,frames=len(times),simulation_time=[float(times[0]),float(times[-1])],method='read-only playback of measured rigid and particle states'),indent=2))
    return str(destination)

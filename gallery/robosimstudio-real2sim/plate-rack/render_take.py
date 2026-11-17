"""Path-trace a second view of a recorded take; never step or modify the live simulation."""
from pathlib import Path
import numpy as np
import imageio.v2 as imageio

def render_recording(sim, folder, output, camera='rack'):
    folder=Path(folder)
    data=np.load(folder/'take.npz')
    from demo import cameras
    cam=next(c for c in cameras() if c.name==camera).scaled(960,540)
    renderer=sim.make_renderer('rtx',cameras=[cam])
    state=sim.model.state()
    # The recorder saved 30 Hz; draw alternate frames at 15 Hz, matching hero.
    ids=list(range(0,len(data['body_q']),2))
    writer=imageio.get_writer(str(output),fps=15,codec='libx264',quality=8,macro_block_size=1,pixelformat='yuv420p')
    try:
        for n,i in enumerate(ids):
            state.body_q.assign(data['body_q'][i])
            state.joint_q.assign(data['joint_q'][i])
            fr=renderer.render(state)
            rgb=np.asarray(fr[camera]['rgb'])[...,:3]
            if rgb.dtype != np.uint8:
                rgb=np.clip(rgb*(255 if rgb.max()<=1 else 1),0,255).astype(np.uint8)
            writer.append_data(rgb)
            if n in (0,len(ids)-1): imageio.imwrite(str(Path(output).with_name(Path(output).stem+('_first.png' if n==0 else '_last.png'))),rgb)
            if n%40==0:
                print('rendered',n,'of',len(ids),flush=True)
                yield
    finally:
        writer.close()
        renderer.close()
    print('Wrote',output,flush=True)

"""Two path-traced cameras observing one live contact trajectory."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from robosimstudio_real2sim_agent.checks import Recorder
from robosimstudio_real2sim_agent import checks
from routing import Routing, reading
class SharedViews:
    def __init__(self, sim, cameras):
        self.sim=sim;self.step=None
        self.path=sim.make_renderer('rtx',cameras=cameras,channels=['rgb'])
        self.ids=sim.make_renderer('tiled',cameras=cameras,channels=['seg'])
        sim.sync();self.path.render(sim.state)
    def read(self, rec):
        if self.step!=int(self.sim.step_count):
            self.sim.sync()
            self.rgb=self.path.render(self.sim.state)
            self.seg=self.ids.render(self.sim.state)
            self.q=checks._particles(self.sim)
            self.step=int(self.sim.step_count)
        rgb=checks.stills.to_uint8(self.rgb[rec.cam.name]['rgb'][...,:3])
        seg=checks.name_deformables(self.sim,rec.cam,np.asarray(self.seg[rec.cam.name]['seg']).astype(np.int32),rec.deformable_labels,self.q)
        return rgb,seg,self.q,{}
    def close(self):self.path.close();self.ids.close()
class SharedRecorder(Recorder):
    def _look(self,strict=False):return self.shared.read(self)
class MeteredWriter:
    def __init__(self, writer, route):self.writer=writer;self.route=route
    def overlay(self, rgb):
        r=reading(self.route.sim,self.route.m.max_box)
        im=Image.fromarray(np.asarray(rgb));d=ImageDraw.Draw(im)
        d.rectangle((8,im.height-31,im.width-8,im.height-7),fill=(16,19,23))
        d.text((16,im.height-26),f"SIM  {self.route.sim.time:5.2f} s    measured channel {r['channel_coverage']*100:5.1f}% / 80%    box corner {self.route.m.max_corner*1000:4.1f} mm / 5 mm    tail on mat {r['tail_on_mat_mm']:4.0f} mm",fill=(240,240,240))
        return np.asarray(im)
    def append(self,rgb):self.writer.append(self.overlay(rgb))
    def try_encode(self,rgb):self.writer.try_encode(self.overlay(rgb))
    def close(self):self.writer.close()
def capture_pair(sim,live):
    run_dir=Path(__file__).resolve().parent/'run';run_dir.mkdir(parents=True,exist_ok=True)
    route=Routing(sim);recorders=[SharedRecorder(live,'hero',camera='photo',every=2),SharedRecorder(live,'hero_wide',camera='wide',every=2)]
    shared=SharedViews(sim,[rec.cam for rec in recorders])
    for rec in recorders:
        rec.renderer.close();rec.renderer=None;rec.shared=shared
        rec.writer=MeteredWriter(rec.writer,route);rec.probe();rec.start()
    try:
        for _ in route.run():
            for rec in recorders:rec.maybe()
            yield
    finally:
        summaries=[]
        for rec in recorders:
            meta=rec.finish();summaries.append((rec,meta))
            print('RECORDED',rec.name,meta['frames'],'frames',flush=True)
        shared.close()
        for rec,meta in summaries:
            motion=checks.take_motion(rec.frames)
            live.last_take=meta
            live.takes_written=[t for t in live.takes_written if t!=rec.name]+[rec.name]
            checks.log_append(live.run_dir,dict(cmd='record',scene=live.scene_id,renderer='rtx',same_run_pair=['hero','hero_wide'],**meta))
            live._show_take(rec.name,checks.motion_line(motion),motion)
        outcome=reading(sim,route.m.max_box)
        outcome.update(box_max_corner_mm=route.m.max_corner*1000,box_max_yaw_deg=float(np.rad2deg(route.m.max_yaw)))
        json.dump(outcome,(run_dir/'paired_outcome.json').open('w'),indent=2)

    return route

"""Six stoneware plates put away by the catalog YAM's unmodified fingertips."""
from model import *
from demo import cameras, Director, Recorder
from types import SimpleNamespace
from dataclasses import asdict
scene, plates = build_scene()
for cam in cameras():
    scene.add(cam)

def routine(sim):
    director = Director(sim, [sim[f"plate_{i}"] for i in range(6)], LAYOUT, SimpleNamespace(settle=0))
    recorder = Recorder(sim, [], None, every=2)
    out = HERE / "run" / "last_motion"
    out.mkdir(parents=True,exist_ok=True)
    try:
        yield from director.run(recorder=recorder)
    finally:
        if recorder.body_q:
            recorder.save(out,30)
            (out / "meta.json").write_text(json.dumps({"layout":asdict(LAYOUT),"record_every":2,"events":director.events},indent=2))

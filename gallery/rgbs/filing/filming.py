"""Two calibrated render streams observing exactly the same routine states."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from motion import routine

import robosimstudio as rss
from robosimstudio_real2sim_agent.checks import Recorder
from robosimstudio_real2sim_agent.stills import save_png, to_uint8


def record_pair(sim, live):
    """Use with ctl eval --record hero --camera photo --every 5.

    The page films photo. A second read-only Recorder films wide at the same
    routine yields; neither recorder changes the model or advances physics.
    """
    files = ("scene.py", "motion.py", "filing_model.py", "look.py", "filming.py", "assets/home_q.json")
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    wide = Recorder(live, "wide", camera="wide", every=5)
    readings = []

    detail = rss.CameraSpec(
        name="filing_detail", width=1280, height=960, fov_y=40, eye=(0.46, -0.06, 1.12), look_at=(0.68, 0, 0.88)
    )
    resting = rss.CameraSpec(
        name="strip_detail", width=1280, height=960, fov_y=45, eye=(0.29, -0.04, 0.98), look_at=(0.44, 0, 0.756)
    )
    primary = live.recorder
    if primary is None or primary.cam.name != "photo":
        raise RuntimeError("Start with --record hero --camera photo --every 5")
    # One persistent RTX rig: every camera observes the same state. Changing
    # Sim.render's single-camera target each frame rebuilds its RTX products.
    rig = sim.make_renderer("rtx", cameras=[primary.cam, wide.cam, detail, resting], channels=["rgb"])
    rig.render(sim.state)
    cached_step, cached = -1, None

    def frames():
        nonlocal cached_step, cached
        if cached_step != int(sim.step_count):
            sim.sync()
            cached = rig.render(sim.state)
            cached_step = int(sim.step_count)
        return cached

    def look(recorder):
        rgb = to_uint8(frames()[recorder.cam.name]["rgb"][..., :3])
        seg = recorder.renderer.render(sim.state)[recorder.cam.name]["seg"]
        return rgb, seg, None, {}

    primary_look = primary._look
    primary._look = lambda strict=False: look(primary)
    wide._look = lambda strict=False: look(wide)
    anchors = {0, 200, 300, 800, 2000, 3500, 4200, 4250, 4400, 4450, 4500, 4700, 4950}

    def reading():
        field = sim["apparatus"].entity
        if int(sim.step_count) in anchors:
            camera = resting if int(sim.step_count) in (0, 200, 300, 4700, 4950) else detail
            save_png(to_uint8(frames()[camera.name]["rgb"][..., :3]), Path(f"run/detail_t{round(sim.time):03d}.png"))
        readings.append(
            {
                "step": int(sim.step_count),
                "time": float(sim.time),
                "cleared": field.read()["cleared"],
                "mean_intensity": field.read()["mean_intensity"],
                "drum_normal_N": field.force,
                "spindle_rad_s": field.speed,
            }
        )

    started = datetime.now(UTC).isoformat()
    wide.probe()
    wide.start()
    reading()
    complete = False
    try:
        for _ in routine(sim):
            wide.maybe()
            reading()
            yield
        complete = True
    finally:
        summary = wide.finish()
        primary._look = primary_look
        rig.close()
        changed = [p for p, digest in hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest]
        Path("run/paired_capture.json").write_text(
            json.dumps(
                {
                    "started_utc": started,
                    "complete": complete,
                    "source_sha256": hashes,
                    "changed": changed,
                    "wide": summary,
                    "renderer": "persistent four-camera RTX rig",
                    "readings": readings,
                },
                indent=2,
            )
        )
    if changed:
        raise RuntimeError(f"Source changed during capture: {changed}")
    print("Same-state photo/wide recording complete", flush=True)

# SPDX-License-Identifier: Apache-2.0
"""Many demonstrations at once: worker processes that step, render and encode, writing one episode store.

A demonstration dataset is hundreds of episodes of the same scene, each driven by
a scripted policy and filmed by a fixed rig: the hang task is 500 of 20 s with
five 1280x720 cameras at 20 Hz, two thousand frames an episode. Three costs make
one, measured on a bimanual YAM:

* **Physics.** On a CPU each period is one call into MuJoCo's C library
  (:func:`robosimstudio.core.mjdata.step_block`): 0.6 ms, so 0.6 s an episode on
  one core. A single world on a GPU is launch-bound at 3.4 ms a period.
* **Frames.** The ``gl`` renderer rasterises the five cameras in 2.5 ms on the
  machine's GPU while the physics runs on its CPU; the tiled tracer on a CPU
  would take seconds a frame.
* **Encoding.** Each camera is an ``ffmpeg`` of its own fed by a thread
  (:class:`~robosimstudio.core.render.video.VideoWriter`), so the encode runs
  while the next period steps.

That is about two seconds an episode in one process, and the machine holds many
processes: each worker is one CPU core of physics, one GL context on a GPU
named round-robin, and its encoders. A worker builds the scene once and resets
it between episodes.

Output is the native :mod:`episode store <robosimstudio.record.store>`: an
episode directory per episode with ``meta.json``, ``steps.npz`` (action,
joint_q, sim_time per control step), ``privileged.npz`` (body_q per frame, the
step each frame follows, for an offline re-render) and ``cameras/<name>.mp4``,
and one row per episode in ``episode_results.jsonl``. An episode already on disk
is skipped, so an interrupted run resumes.

The two things a collection names are references, ``path/to/file.py:function``
or ``package.module:function``, because a worker is a fresh process that has to
import them itself:

* **scene**: ``build() -> rss.Scene`` (the worker builds it on its device) or a
  built ``Sim``.
* **policy**: ``make(sim, episode) -> act``, called after each reset, where
  ``act(sim, step)`` returns the next action, or ``None`` to end the episode
  early. It may instead return an object with ``act`` and, optionally,
  ``success(sim) -> bool``, read when the episode ends.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["CollectSpec", "collect", "load_reference", "main"]


@dataclass(frozen=True, kw_only=True)
class CollectSpec:
    """One collection.

    Attributes:
        scene: Reference to the scene builder (see the module docstring).
        policy: Reference to the policy factory.
        out: The run directory.
        episodes: How many.
        first: The first episode id; ids are ``first .. first + episodes - 1``
            and are what the policy gets to vary on.
        seconds: Simulated time an episode runs for, unless its policy ends it.
        fps: Frames per second of simulated time, the first at the reset.
        mode: The control mode the policy's actions are in.
        cameras: Which of the scene's cameras to film; empty films every one.
        device: Where the physics runs: ``"cpu"`` or ``"cuda"``.
        renderer: ``"gl"``, ``"tiled"``, or ``"auto"`` -- ``gl`` for a scene
            without particles, else ``tiled``.
        workers: Worker processes; 0 picks from the cores, about one per six.
        gpus: The GPUs workers render on (and step on, with ``device="cuda"``),
            round-robin. Empty uses every CUDA device; none at all leaves ``gl``
            to OSMesa.
        preset: libx264's speed preset.
        threads: Encoder threads per camera.
        keep_failures: Keep the files of an episode whose policy reported
            failure. Its row is written either way.
    """

    scene: str
    policy: str
    out: str
    episodes: int = 1
    first: int = 0
    seconds: float = 20.0
    fps: int = 20
    mode: str = "joint_position"
    cameras: tuple[str, ...] = ()
    device: str = "cpu"
    renderer: str = "auto"
    workers: int = 0
    gpus: tuple[int, ...] = ()
    preset: str = "veryfast"
    threads: int = 2
    keep_failures: bool = True
    extra: dict[str, Any] = field(default_factory=dict)
    """Anything the run should remember in ``run_meta.json``."""


def load_reference(reference: str) -> Any:
    """What ``path/to/file.py:name`` or ``package.module:name`` names."""
    left, _, name = reference.rpartition(":")
    if not left or not name:
        raise ValueError(f"reference {reference!r} must be 'path/to/file.py:name' or 'package.module:name'")
    if left.endswith(".py") or os.sep in left:
        path = Path(left).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"no file {path} for reference {reference!r}")
        module_name = f"_rss_collect_{abs(hash(str(path)))}"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
    else:
        module = importlib.import_module(left)
    try:
        return getattr(module, name)
    except AttributeError:
        raise AttributeError(f"{left} has no {name!r} (from reference {reference!r})") from None


# ---------------------------------------------------------------------- worker
def _episode_dir(out: Path, episode: int) -> Path:
    return out / f"{int(episode):04d}"


def _build(spec: CollectSpec):
    import robosimstudio as rss

    made = load_reference(spec.scene)()
    sim = made.build(device=spec.device if spec.device == "cpu" else "cuda:0") if isinstance(made, rss.Scene) else made
    cameras = [c for c in sim.cameras if not spec.cameras or c.name in spec.cameras]
    missing = sorted(set(spec.cameras) - {c.name for c in cameras})
    if missing or not cameras:
        raise ValueError(f"the scene has no camera {missing or 'at all'}; it has {[c.name for c in sim.cameras]}")
    name = spec.renderer
    if name == "auto":
        name = "tiled" if int(sim.model.particle_count) else "gl"
    try:
        renderer = sim.make_renderer(name, cameras=cameras, channels=("rgb",))
    except RuntimeError as exc:
        if spec.renderer != "auto" or name != "gl":
            raise
        print(f"[collect] gl unavailable ({exc}); rendering with tiled", flush=True)
        name, renderer = "tiled", sim.make_renderer("tiled", cameras=cameras, channels=("rgb",))
    return sim, renderer, name, cameras


def _run_episode(spec: CollectSpec, sim, renderer, renderer_name: str, cameras, episode: int) -> dict[str, Any]:
    """One episode, written; returns its results row."""
    from robosimstudio.core.render.video import VideoSet

    out = _episode_dir(Path(spec.out), episode)
    t0 = time.perf_counter()
    sim.reset()
    made = load_reference(spec.policy)(sim, int(episode))
    act = made.act if hasattr(made, "act") else made
    hz = float(sim.request.physics.control_hz)
    steps = max(1, round(float(spec.seconds) * hz))
    physics = render = encode = 0.0
    actions, joint_q, sim_time, body_q, frame_step = [], [], [], [], []
    videos = VideoSet(out / "cameras", fps=int(spec.fps), preset=spec.preset, threads=int(spec.threads))
    next_frame = 0.0
    try:
        for k in range(-1, steps):
            if k >= 0:
                action = act(sim, k)
                if action is None:
                    break
                a = time.perf_counter()
                sim.step(np.asarray(action, dtype=np.float32), spec.mode)
                physics += time.perf_counter() - a
                actions.append(np.asarray(action, dtype=np.float32))
                joint_q.append(sim.state.joint_q.numpy().copy())
                sim_time.append((k + 1) / hz)
            if (k + 1) / hz + 1e-9 < next_frame:
                continue
            next_frame += 1.0 / float(spec.fps)
            a = time.perf_counter()
            frames = renderer.render(sim.state)
            b = time.perf_counter()
            videos.append_frames(frames)
            render += b - a
            encode += time.perf_counter() - b
            body_q.append(sim.state.body_q.numpy().copy())
            frame_step.append(k)
    finally:
        a = time.perf_counter()
        videos.close()
        encode += time.perf_counter() - a
    success = getattr(made, "success", None)
    success = bool(success(sim)) if callable(success) else None
    np.savez_compressed(
        out / "steps.npz",
        action=np.array(actions, dtype=np.float32),
        joint_q=np.array(joint_q, dtype=np.float32),
        sim_time=np.array(sim_time, dtype=np.float64),
    )
    np.savez_compressed(
        out / "privileged.npz", body_q=np.array(body_q, dtype=np.float32), frame_step=np.array(frame_step)
    )
    row = {
        "episode_id": int(episode),
        "success": success,
        "episode_length": len(actions),
        "frames": len(frame_step),
        "renderer": renderer_name,
        "wall_s": round(time.perf_counter() - t0, 4),
        "physics_s": round(physics, 4),
        "render_s": round(render, 4),
        "encode_wait_s": round(encode, 4),
    }
    meta = {
        **row,
        "seed": int(episode),
        "fps": int(spec.fps),
        "control_hz": hz,
        "mode": spec.mode,
        "renderer": renderer_name,
        "device": str(sim.model.device),
        "cameras": {
            c.name: {"width": int(c.width), "height": int(c.height), "K": np.asarray(c.intrinsics).tolist()}
            for c in cameras
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if success is False and not spec.keep_failures:
        import shutil

        shutil.rmtree(out, ignore_errors=True)
    return row


def _work(spec_fields: dict[str, Any], episodes: list[int], results) -> None:
    """A worker process: build once, then every episode it was given; a row on ``results`` per episode."""
    spec = CollectSpec(**spec_fields)
    from robosimstudio.record.store import EpisodeStore

    store = EpisodeStore(spec.out)
    try:
        sim, renderer, name, cameras = _build(spec)
    except BaseException as exc:
        results.put({"error": f"{type(exc).__name__}: {exc}", "episodes": episodes})
        return
    try:
        for episode in episodes:
            try:
                row = _run_episode(spec, sim, renderer, name, cameras, episode)
            except BaseException as exc:
                results.put({"error": f"episode {episode}: {type(exc).__name__}: {exc}", "episodes": [episode]})
                continue
            store.append_result(row)
            results.put(row)
    finally:
        renderer.close()
        sim.close()


# ------------------------------------------------------------------ orchestrator
def _visible_gpus() -> tuple[int, ...]:
    named = os.environ.get("CUDA_VISIBLE_DEVICES")
    if named is not None:
        return tuple(int(g) for g in named.split(",") if g.strip().isdigit())
    try:
        import warp as wp

        return tuple(range(wp.get_cuda_device_count()))
    except Exception:
        return ()


def collect(spec: CollectSpec, *, verbose: bool = True) -> dict[str, Any]:
    """Run a collection; returns its summary, also written into ``run_meta.json``.

    Each worker is a spawned process whose environment pins it before anything
    imports Warp: ``MUJOCO_EGL_DEVICE_ID`` names its GPU for the GL context, and
    ``CUDA_VISIBLE_DEVICES`` is that GPU on ``device="cuda"`` and empty on a CPU,
    where a worker needs no CUDA context of its own.
    """
    import multiprocessing as mp

    from robosimstudio.record.store import EpisodeStore, RunMeta

    if spec.device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {spec.device!r}")
    out = Path(spec.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = list(range(int(spec.first), int(spec.first) + int(spec.episodes)))
    todo = [e for e in wanted if not (_episode_dir(out, e) / "meta.json").exists()]
    gpus = tuple(spec.gpus) or _visible_gpus()
    if spec.device == "cuda" and not gpus:
        raise ValueError("device='cuda' needs a GPU, and none is visible")
    workers = int(spec.workers) or max(1, (os.cpu_count() or 2) // 6)
    workers = max(1, min(workers, len(todo))) if todo else 0
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    store = EpisodeStore(out)
    t0 = time.perf_counter()
    done: list[dict[str, Any]] = []
    errors: list[str] = []
    if workers:
        context = mp.get_context("spawn")
        results = context.Queue()
        processes = []
        fields = asdict(spec)
        for w in range(workers):
            env = {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            gpu = gpus[w % len(gpus)] if gpus else None
            if gpu is not None:
                env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu) if spec.device == "cuda" else ""
            saved = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                process = context.Process(target=_work, args=(fields, todo[w::workers], results), daemon=True)
                process.start()
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            processes.append(process)
        expected = len(todo)
        accounted = 0
        while (accounted < expected and any(p.is_alive() for p in processes)) or not results.empty():
            try:
                item = results.get(timeout=1.0)
            except Exception:
                continue
            if "error" in item:
                errors.append(item["error"])
                accounted += len(item["episodes"])
                if verbose:
                    print(f"[collect] {item['error']}", flush=True)
                continue
            done.append(item)
            accounted += 1
            if verbose:
                rate = len(done) / max(time.perf_counter() - t0, 1e-9) * 60.0
                print(
                    f"[collect] {len(done)}/{expected} episode {item['episode_id']}: {item['wall_s']:.2f}s "
                    f"(physics {item['physics_s']:.2f}, render {item['render_s']:.2f}, "
                    f"encode wait {item['encode_wait_s']:.2f}) -- {rate:.1f} episodes/min",
                    flush=True,
                )
        for process in processes:
            process.join(timeout=60)
        if accounted < expected:
            errors.append(f"{expected - accounted} episodes never reported: a worker died without saying why")
    wall = time.perf_counter() - t0
    summary = {
        "episodes": len(wanted),
        "ran": len(done),
        "skipped": len(wanted) - len(todo),
        "failed": len(errors),
        "successes": sum(1 for r in done if r.get("success")),
        "workers": workers,
        "gpus": list(gpus),
        "wall_s": round(wall, 2),
        "episodes_per_min": round(len(done) / max(wall, 1e-9) * 60.0, 2),
        "frame_sets": sum(int(r["frames"]) for r in done),
        "errors": errors[:20],
    }
    store.write_run_meta(
        RunMeta(
            run_name=out.name,
            scene=spec.scene,
            solution=spec.policy,
            renderer=",".join(sorted({str(r["renderer"]) for r in done})) or spec.renderer,
            device=spec.device,
            started=started,
            finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
            episodes=len(wanted),
            successes=summary["successes"],
            provenance={"collect": {**asdict(spec), "summary": summary}},
        )
    )
    if verbose:
        print(
            f"[collect] {summary['ran']} episodes in {wall:.1f}s with {workers} workers on GPUs {list(gpus)}: "
            f"{summary['episodes_per_min']:.1f} episodes/min"
            + (f", {summary['skipped']} already on disk" if summary["skipped"] else "")
            + (f", {summary['failed']} failed" if summary["failed"] else ""),
            flush=True,
        )
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """``python -m robosimstudio.record.collect``: :func:`collect` from the shell, one flag per field."""
    import argparse

    defaults = CollectSpec(scene="", policy="", out="")
    parser = argparse.ArgumentParser(prog="python -m robosimstudio.record.collect", description=collect.__doc__)
    parser.add_argument("--scene", required=True, help="path/to/file.py:build or package.module:build")
    parser.add_argument("--policy", required=True, help="path/to/file.py:make_policy or package.module:make_policy")
    parser.add_argument("--out", required=True, help="the run directory")
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--first", type=int, default=defaults.first)
    parser.add_argument("--seconds", type=float, default=defaults.seconds)
    parser.add_argument("--fps", type=int, default=defaults.fps)
    parser.add_argument("--mode", default=defaults.mode)
    parser.add_argument("--cameras", nargs="*", default=[], help="camera names; none films every one")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=defaults.device)
    parser.add_argument("--renderer", default=defaults.renderer, help="gl, tiled or auto")
    parser.add_argument("--workers", type=int, default=defaults.workers, help="0 picks about one per six cores")
    parser.add_argument("--gpus", default="", help="comma-separated GPU indices; empty uses every one visible")
    parser.add_argument("--preset", default=defaults.preset, help="libx264 preset")
    parser.add_argument("--threads", type=int, default=defaults.threads, help="encoder threads per camera")
    parser.add_argument("--drop-failures", action="store_true", help="delete the files of failed episodes")
    args = parser.parse_args(argv)
    spec = CollectSpec(
        scene=args.scene,
        policy=args.policy,
        out=args.out,
        episodes=args.episodes,
        first=args.first,
        seconds=args.seconds,
        fps=args.fps,
        mode=args.mode,
        cameras=tuple(args.cameras),
        device=args.device,
        renderer=args.renderer,
        workers=args.workers,
        gpus=tuple(int(g) for g in args.gpus.split(",") if g.strip()),
        preset=args.preset,
        threads=args.threads,
        keep_failures=not args.drop_failures,
    )
    return collect(spec)


if __name__ == "__main__":
    main()

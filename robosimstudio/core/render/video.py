# SPDX-License-Identifier: Apache-2.0
"""Streaming mp4 writers, one per camera.

A recorder that buffers frames until close costs memory proportional to the
episode and loses everything if the run dies, so :class:`VideoWriter` opens the
file on the first frame and appends, to an ``ffmpeg`` of its own that encodes
while the caller steps. ``imageio-ffmpeg`` is imported only for its binary, and
only when a first frame arrives.

Depth and segmentation are written as colourised previews, never as the data: an
mp4 is 8-bit and lossy, so those videos are for looking at. The metric array and
the label ids belong in the episode's ``npz``. The two colourisers live here
because a live view in a browser wants exactly the same conversion -- a channel
that is not ``rgb`` has to become ``rgb`` before anything can display it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["FFMPEG_ENV", "VideoWriter", "colorize_depth", "colorize_labels", "ffmpeg_exe", "write_video"]

# Golden-ratio conjugate: successive multiples of it are maximally spread over
# the unit interval, which is what keeps consecutive label ids far apart in hue.
_HUE_STEP = 0.6180339887498949


def colorize_depth(depth: np.ndarray, *, near: float = 0.0, far: float = 0.0) -> np.ndarray:
    """Depth as a viewable 8-bit grey image, misses left black.

    Args:
        depth: Metric depth [m], 0 for a miss.
        near: Low end of the mapped range [m]. 0 uses the frame's own minimum
            over the hits.
        far: High end [m]. 0 uses the frame's own maximum.
    """
    d = np.asarray(depth, dtype=np.float32)
    hit = d > 0
    if not hit.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo = float(near) if near > 0 else float(d[hit].min())
    hi = float(far) if far > 0 else float(d[hit].max())
    span = max(hi - lo, 1.0e-6)
    grey = np.clip((d - lo) / span, 0.0, 1.0)
    grey = np.where(hit, 1.0 - grey, 0.0)  # near is bright, misses black
    return np.repeat((grey * 255).astype(np.uint8)[..., None], 3, axis=2)


def colorize_labels(labels: np.ndarray, *, background: int = 0) -> np.ndarray:
    """Label ids as distinguishable 8-bit colours, ``background`` left black.

    A hue walk rather than a fixed palette, because the ids are the scene's to
    assign: a task labels as many things as it has, so a colouriser cannot know
    how many colours it will be asked for, and *adjacent* ids have to land far
    apart -- consecutive integers are exactly how a scene labels two objects
    lying against each other, and a palette that walks hue linearly renders that
    pair as two shades of the same colour.

    Args:
        labels: Integer label ids, any shape.
        background: The id meaning "nothing here", rendered black.

    Returns:
        ``uint8`` RGB, shape ``(*labels.shape, 3)``.
    """
    ids = np.asarray(labels).astype(np.int64, copy=False)
    unique = np.unique(ids)
    colors = _hues_to_rgb(np.mod(unique.astype(np.float64) * _HUE_STEP, 1.0))
    colors[unique == int(background)] = 0
    # unique is sorted, so searchsorted is the lookup -- one pass over the image
    # rather than a Python loop over however many labels the scene declared.
    return colors[np.searchsorted(unique, ids)]


def _hues_to_rgb(hue: np.ndarray, *, saturation: float = 0.65, value: float = 1.0) -> np.ndarray:
    """HSV to ``uint8`` RGB for a 1-D array of hues in [0, 1)."""
    h = np.mod(np.asarray(hue, dtype=np.float64), 1.0) * 6.0
    sector = np.floor(h).astype(np.int64) % 6
    f = h - np.floor(h)
    v = np.full(h.shape, float(value))
    p = v * (1.0 - float(saturation))
    q = v * (1.0 - float(saturation) * f)
    t = v * (1.0 - float(saturation) * (1.0 - f))
    conditions = [sector == i for i in range(6)]
    r = np.select(conditions, [v, q, p, p, t, v])
    g = np.select(conditions, [t, v, v, q, p, p])
    b = np.select(conditions, [p, p, t, v, v, q])
    return (np.stack([r, g, b], axis=-1) * 255.0).round().astype(np.uint8)


FFMPEG_ENV = "ROBOSIMSTUDIO_FFMPEG"
"""An ffmpeg binary to encode with instead of the one ``imageio-ffmpeg`` ships."""


def ffmpeg_exe() -> str:
    """The ffmpeg every writer runs: :data:`FFMPEG_ENV` if set, else ``imageio-ffmpeg``'s."""
    import os

    named = os.environ.get(FFMPEG_ENV, "").strip()
    if named:
        return named
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


class VideoWriter:
    """One mp4, appended a frame at a time, encoded beside the caller rather than in its way.

    Each writer is its own ``ffmpeg`` process fed raw RGB through a pipe by a
    thread of its own, so :meth:`append` costs a queue put: the encode runs
    while the simulation steps, and five cameras encode on five processes at
    once. It used to be imageio's writer, called inline, which held the stepping
    thread for the whole encode -- 3.8 s of a 20 s demo's 11.8 s for five
    1280x720 cameras. The bytes are the same: libx264, ``yuv420p``, imageio's
    mapping from ``quality`` to a CRF.

    Args:
        path: Output file. Parent directories are created.
        fps: Frame rate stamped into the container.
        quality: 1..10, imageio's scale: 7 is CRF 15.
        preset: libx264's speed preset. ``"medium"`` is what imageio used;
            ``"veryfast"`` encodes about three times as fast for a larger file,
            and ``"ultrafast"`` faster again.
        queue: Frames that may wait for the encoder before :meth:`append`
            blocks, which is what bounds the memory a slow encoder can take.
        threads: Encoder threads, 0 for ffmpeg's own choice (about one and a
            half per core). A pool running a writer per camera per worker sets
            it, or the encoders oversubscribe the machine.
        copy: Copy each frame as it is queued. The encode happens later, so a
            caller that draws every frame into one buffer needs it; a caller
            whose every frame is a fresh array -- any renderer's -- can skip it.

    A writer dropped without :meth:`close` still finishes its file once it is
    collected, as imageio's did; :meth:`close` is what reports a failure.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fps: int = 30,
        quality: int = 7,
        preset: str = "medium",
        queue: int = 16,
        threads: int = 0,
        copy: bool = True,
    ) -> None:
        import queue as queue_mod

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.quality = int(quality)
        self.preset = str(preset)
        self.threads = max(0, int(threads))
        self.copy = bool(copy)
        self.frames = 0
        self._queue: queue_mod.Queue = queue_mod.Queue(maxsize=max(1, int(queue)))
        self._process = None
        self._thread = None
        self._feed: _Feed | None = None
        self._finish = None
        self._shape: tuple[int, int] | None = None

    def _open(self, height: int, width: int) -> None:
        """Start ffmpeg on the first frame, which is the first time the size is known."""
        import subprocess
        import tempfile
        import threading
        import weakref

        if not 1 <= self.quality <= 10:
            raise ValueError(f"quality must be 1..10, got {self.quality}")
        crf = int((1.0 - self.quality / 10.0) * 51)
        command = [
            ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(self.fps),
            "-i", "-", "-an",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", self.preset,
            *(["-threads", str(self.threads)] if self.threads else []),
            str(self.path),
        ]  # fmt: skip
        # ffmpeg's messages go to a file, not a pipe: a pipe nobody reads until
        # close fills at 64 KB, ffmpeg stops reading its input to wait on it, and
        # the feeding thread and then append() wait on ffmpeg -- a hang, not an error.
        self._log = tempfile.TemporaryFile()
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=self._log)
        self._shape = (height, width)
        # The thread holds the feed, never the writer, so a writer dropped
        # without close() is collected, and its finaliser ends the stream.
        self._feed = _Feed(self._queue, self._process.stdin)
        self._finish = weakref.finalize(self, _end_stream, self._queue, self._log)
        self._thread = threading.Thread(target=self._feed.run, name=f"video:{self.path.name}", daemon=True)
        self._thread.start()

    @property
    def _error(self) -> BaseException | None:
        return self._feed.error if self._feed is not None else None

    def _messages(self) -> str:
        """What ffmpeg has said so far, its last 2000 characters."""
        log = getattr(self, "_log", None)
        if log is None:
            return ""
        log.seek(0)
        return log.read().decode(errors="replace").strip()[-2000:]

    def _raise(self) -> None:
        if self._error is not None:
            detail = self._messages()
            raise RuntimeError(f"encoding {self.path} failed: {self._error}{': ' + detail if detail else ''}")

    def append(self, frame: np.ndarray) -> None:
        """Queue one RGB uint8 frame, top-left origin: a copy of it, or with ``copy=False`` the array itself."""
        frame = np.array(frame, dtype=np.uint8, order="C", copy=True if self.copy else None)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"a video frame is [H, W, 3] uint8, got {frame.shape}")
        if self._process is None:
            self._open(frame.shape[0], frame.shape[1])
        elif frame.shape[:2] != self._shape:
            raise ValueError(f"frame {frame.shape[:2]} does not match the video's {self._shape}")
        self._raise()
        self._queue.put(frame)
        self.frames += 1

    def close(self) -> None:
        """Finish the container: drain the queue, end ffmpeg, raise if it failed. Idempotent."""
        if self._process is None:
            return
        process, self._process = self._process, None
        self._finish.detach()  # this ends the stream instead, and reads the log first
        self._queue.put(None)
        self._thread.join()
        code = process.wait()
        messages = self._messages()
        self._log.close()
        if self._error is not None or code != 0:
            raise RuntimeError(f"encoding {self.path} failed (ffmpeg exited {code}): {messages or self._error}")

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _end_stream(queue: Any, log: Any) -> None:
    """What a writer collected without :meth:`VideoWriter.close` does: end ffmpeg's input, and let go of its log.

    The end-of-stream goes in from a thread of its own: the queue may be full,
    and the collector may be running this on the very thread that empties it.
    """
    import threading

    threading.Thread(target=queue.put, args=(None,), name="video:end", daemon=True).start()
    log.close()


class _Feed:
    """A writer's thread: frames off the queue into ffmpeg's stdin until the end-of-stream ``None``."""

    def __init__(self, queue: Any, stdin: Any) -> None:
        self.queue = queue
        self.stdin = stdin
        self.error: BaseException | None = None
        """Surfaced on the writer's next append or close."""

    def run(self) -> None:
        try:
            while True:
                frame = self.queue.get()
                if frame is None:
                    break
                # A pipe write releases the GIL; the stepping thread runs on.
                self.stdin.write(memoryview(frame).cast("B"))
        except BaseException as exc:
            self.error = exc
            while self.queue.get() is not None:  # let a blocked append through
                pass
        finally:
            try:
                self.stdin.close()  # flushes, and raises again if ffmpeg is already gone
            except OSError as exc:
                self.error = self.error or exc


class VideoSet:
    """A :class:`VideoWriter` per camera, opened lazily on first use.

    Args:
        root: Directory the files are written under.
        fps: Frame rate.
        suffix: Appended to each camera name, so one run can hold both
            ``wrist.mp4`` and ``wrist_depth.mp4``.
        preset: libx264's speed preset for every writer; see :class:`VideoWriter`.
        threads: Encoder threads for every writer; see :class:`VideoWriter`.
        copy: Whether every writer copies its frames; see :class:`VideoWriter`.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        fps: int = 30,
        suffix: str = "",
        preset: str = "medium",
        threads: int = 0,
        copy: bool = True,
    ) -> None:
        self.root = Path(root)
        self.fps = int(fps)
        self.suffix = suffix
        self.preset = str(preset)
        self.threads = int(threads)
        self.copy = bool(copy)
        self.writers: dict[str, VideoWriter] = {}

    def append(self, name: str, frame: np.ndarray) -> None:
        writer = self.writers.get(name)
        if writer is None:
            writer = VideoWriter(
                self.root / f"{name}{self.suffix}.mp4",
                fps=self.fps,
                preset=self.preset,
                threads=self.threads,
                copy=self.copy,
            )
            self.writers[name] = writer
        writer.append(frame)

    def append_frames(self, frames: Mapping[str, Mapping[str, np.ndarray]], *, depth: bool = False) -> None:
        """Append the ``rgb`` channel of every camera in a render result.

        Args:
            frames: A :data:`~robosimstudio.core.render.base.FrameSet`.
            depth: Also write a colourised depth preview per camera that has
                one, as ``<name>_depth.mp4``.
        """
        for name, frame in frames.items():
            if "rgb" in frame:
                self.append(name, frame["rgb"])
            if depth and "depth" in frame:
                self.append(f"{name}_depth", colorize_depth(frame["depth"]))

    def close(self) -> None:
        """Finish every writer; the first failure is raised once all have been closed."""
        failures = []
        for writer in self.writers.values():
            try:
                writer.close()
            except RuntimeError as exc:
                failures.append(exc)
        self.writers = {}
        if failures:
            raise failures[0]

    def __enter__(self) -> VideoSet:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_video(path: str | Path, frames, *, fps: int = 30) -> str:
    """Encode a whole sequence at once. Returns the path written."""
    with VideoWriter(path, fps=fps) as writer:
        for frame in frames:
            writer.append(frame)
    return str(path)

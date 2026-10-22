# SPDX-License-Identifier: Apache-2.0
"""The mp4 writer encodes beside its caller: every frame lands, in order, and a failure is not swallowed.

It used to be imageio's writer called inline, which held the stepping thread
for the whole encode; now each writer is an ``ffmpeg`` fed by a thread of its
own. What that must not change is the file: the frames, their order, the size,
and an error when the encoder fails rather than a short video.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

TIER = "fast"


def _frames(count: int, height: int = 48, width: int = 64) -> list[np.ndarray]:
    """Solid frames whose red channel counts up, so order and loss both show on reading back."""
    out = []
    for k in range(count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = (6 * k) % 256
        frame[..., 2] = 200
        out.append(frame)
    return out


class TestTheWriter(unittest.TestCase):
    def test_every_frame_lands_in_order(self) -> None:
        import imageio.v2 as imageio

        from robosimstudio.core.render.video import VideoWriter

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "clip.mp4"
            with VideoWriter(path, fps=10, preset="ultrafast") as writer:
                for frame in _frames(20):
                    writer.append(frame)
            self.assertEqual(writer.frames, 20)
            with imageio.get_reader(str(path)) as reader:
                red = [float(np.asarray(frame)[..., 0].mean()) for frame in reader]
        self.assertEqual(len(red), 20)
        np.testing.assert_allclose(red, [6.0 * k for k in range(20)], atol=4.0)

    def test_a_caller_that_draws_into_one_buffer_gets_every_frame(self) -> None:
        """The encode happens after append returns, so a reused canvas is copied as it is queued."""
        import imageio.v2 as imageio

        from robosimstudio.core.render.video import VideoWriter

        canvas = np.zeros((48, 64, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            with VideoWriter(path, fps=10, preset="ultrafast") as writer:
                for frame in _frames(20):
                    canvas[...] = frame
                    writer.append(canvas)
            with imageio.get_reader(str(path)) as reader:
                red = [float(np.asarray(frame)[..., 0].mean()) for frame in reader]
        np.testing.assert_allclose(red, [6.0 * k for k in range(20)], atol=4.0)

    def test_a_writer_dropped_without_close_still_finishes_its_file(self) -> None:
        """As imageio's did: the thread holds the feed, not the writer, so the writer is collected."""
        import gc
        import weakref

        import imageio.v2 as imageio

        from robosimstudio.core.render.video import VideoWriter

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            writer = VideoWriter(path, fps=10, preset="ultrafast")
            for frame in _frames(12):
                writer.append(frame)
            process, thread, alive = writer._process, writer._thread, weakref.ref(writer)
            del writer
            gc.collect()
            self.assertIsNone(alive(), "the writer was kept alive")
            thread.join(timeout=30)
            self.assertEqual(process.wait(timeout=30), 0)
            with imageio.get_reader(str(path)) as reader:
                self.assertEqual(sum(1 for _ in reader), 12)

    def test_a_frame_of_another_size_is_refused(self) -> None:
        from robosimstudio.core.render.video import VideoWriter

        with tempfile.TemporaryDirectory() as tmp:
            writer = VideoWriter(Path(tmp) / "clip.mp4", fps=10, preset="ultrafast")
            writer.append(_frames(1)[0])
            with self.assertRaisesRegex(ValueError, "does not match"):
                writer.append(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.close()

    def test_an_encoder_that_fails_is_reported_at_close(self) -> None:
        from robosimstudio.core.render.video import VideoWriter

        with tempfile.TemporaryDirectory() as tmp:
            writer = VideoWriter(Path(tmp) / "clip.mp4", fps=10, preset="no-such-preset")
            with self.assertRaisesRegex(RuntimeError, "encoding .*clip.mp4 failed"):
                for frame in _frames(40):
                    writer.append(frame)
                writer.close()

    def test_an_odd_size_is_padded_to_an_even_one_rather_than_refused(self) -> None:
        import imageio.v2 as imageio

        from robosimstudio.core.render.video import VideoWriter

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            with VideoWriter(path, fps=10, preset="ultrafast") as writer:
                writer.try_encode(_frames(1, height=47, width=63)[0])
                for frame in _frames(5, height=47, width=63):
                    writer.append(frame)
            self.assertEqual(imageio.get_reader(str(path)).get_data(0).shape[:2], (48, 64))

    def test_a_trial_encode_raises_what_the_encoder_says_and_writes_nothing(self) -> None:
        from robosimstudio.core.render.video import VideoWriter

        with tempfile.TemporaryDirectory() as tmp:
            writer = VideoWriter(Path(tmp) / "clip.mp4", fps=10, preset="no-such-preset")
            with self.assertRaisesRegex(RuntimeError, "cannot encode .*clip.mp4"):
                writer.try_encode(_frames(1)[0])
            with mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_FFMPEG": str(Path(tmp) / "no-ffmpeg")}):
                with self.assertRaisesRegex(RuntimeError, "could not be run"):
                    writer.try_encode(_frames(1)[0])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_an_ffmpeg_that_talks_a_lot_does_not_hang_the_writer(self) -> None:
        """More than a pipe's 64 KB of messages, and nobody reading them until close, must not stop the encode."""
        import os
        import stat
        import threading
        from unittest import mock

        from robosimstudio.core.render.video import FFMPEG_ENV, VideoWriter, ffmpeg_exe

        real = ffmpeg_exe()
        with tempfile.TemporaryDirectory() as tmp:
            chatty = Path(tmp) / "ffmpeg"
            chatty.write_text(f'#!/bin/sh\nhead -c 200000 /dev/zero | tr "\\\\0" x >&2\nexec "{real}" "$@"\n')
            chatty.chmod(chatty.stat().st_mode | stat.S_IEXEC)
            done = []

            def encode() -> None:
                with VideoWriter(Path(tmp) / "clip.mp4", fps=10, preset="ultrafast") as writer:
                    for frame in _frames(40):
                        writer.append(frame)
                done.append(writer.frames)

            with mock.patch.dict(os.environ, {FFMPEG_ENV: str(chatty)}):
                thread = threading.Thread(target=encode, daemon=True)
                thread.start()
                thread.join(timeout=60)
            self.assertEqual(done, [40], "the writer hung on ffmpeg's messages")

    def test_a_set_closes_every_writer_before_raising_the_first_failure(self) -> None:
        from robosimstudio.core.render.video import VideoSet, VideoWriter

        with tempfile.TemporaryDirectory() as tmp:
            videos = VideoSet(tmp, fps=10, preset="ultrafast")
            videos.writers["bad"] = VideoWriter(Path(tmp) / "bad.mp4", fps=10, preset="no-such-preset")
            for frame in _frames(5):
                videos.append("good", frame)
            with self.assertRaises(RuntimeError):
                videos.append("bad", _frames(1)[0])
                videos.close()
            videos.close()
            self.assertEqual(videos.writers, {})
            self.assertGreater((Path(tmp) / "good.mp4").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

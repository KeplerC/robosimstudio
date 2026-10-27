# SPDX-License-Identifier: Apache-2.0
"""The live page's socket: where it is, what it binds, and one line in for one line out.

A run directory names one socket path however it is spelled, and a deep one
still gets a path that fits ``sockaddr_un``. A live socket is never taken over
and a stale one is; a regular file at the path is never deleted. The client
says NoServer or NoReply, never a traceback, whatever the other end does.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from robosimstudio_real2sim_agent import wire
from tests.fixtures.sockets import chdir, short_dir

TIER = "fast"
"""No GPU and no page: sockets on temp paths, answered by plain functions."""


class TestTheWire(unittest.TestCase):
    def test_where_the_socket_is(self) -> None:
        with short_dir("w-") as here, chdir(here):  # `run/ctl.sock` is resolved against the cwd, so the cwd is short
            self.assertEqual(wire.socket_path(None), Path("run") / "ctl.sock")
            self.assertEqual(wire.socket_path("d"), Path("d") / "ctl.sock")
        deep = Path(tempfile.gettempdir()) / ("d" * 60) / ("e" * 60) / "run"
        short = wire.socket_path(deep)
        self.assertLessEqual(
            len(str(short)),
            wire.SUN_PATH_MAX,
            "a run directory too deep for sockaddr_un gets a short path",
        )
        self.assertEqual(
            short,
            wire.socket_path(deep),
            "the same one every time, so the server and its clients agree",
        )
        self.assertNotEqual(
            short,
            wire.socket_path(deep.with_name("run2")),
            "and a different one per run directory",
        )
        self.assertEqual(wire.socket_path("d", "s.sock"), Path("s.sock"), "an explicit socket wins")

    def test_the_ready_line_reads_back(self) -> None:
        line = wire.ready_line("http://localhost:8080", "/r/ctl.sock", "/r/scene.py")
        self.assertTrue(line.startswith(wire.READY_PREFIX))
        parsed = wire.parse_ready("noise\n" + line + "\nmore noise\n")
        self.assertEqual(
            parsed,
            {
                "url": "http://localhost:8080",
                "sock": "/r/ctl.sock",
                "scene": "/r/scene.py",
                "share": None,
            },
        )
        shared = wire.parse_ready(wire.ready_line("http://localhost:1", "/s", "/f", "https://x.y/z"))
        self.assertEqual(shared["share"], "https://x.y/z")
        self.assertIsNone(wire.parse_ready("nothing here"))

    def test_encode_spells_tolist_by_attribute(self) -> None:
        reply = json.loads(wire.encode({"a": np.arange(3), "b": np.float32(1.5), "c": Path("p")}))
        self.assertEqual(reply, {"a": [0, 1, 2], "b": 1.5, "c": "p"})

    def test_one_line_in_one_line_out(self) -> None:
        tmp = self.enterContext(short_dir("w-"))
        path = tmp / "ctl.sock"

        def handler(request: dict) -> dict:
            if request.get("cmd") == "boom":
                raise ValueError("no")
            return {"ok": True, "echo": request}

        server = wire.ControlSocket(path, handler)
        server.start()
        deadline = time.time() + 5.0
        while not path.exists() and time.time() < deadline:
            time.sleep(0.05)
        try:
            self.assertEqual(
                wire.request(path, {"cmd": "status", "n": 1}, 5.0),
                {"ok": True, "echo": {"cmd": "status", "n": 1}},
            )
            failed = wire.request(path, {"cmd": "boom"}, 5.0)
            self.assertFalse(failed["ok"])
            self.assertIn("ValueError: no", failed["error"])
            for raw in (b"not json\n", b"[1, 2]\n"):
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(5.0)
                    client.connect(str(path))
                    client.sendall(raw)
                    reply = json.loads(client.makefile("rb").readline())
                self.assertFalse(reply["ok"])
                self.assertIn("bad request", reply["error"])
        finally:
            server.close()
        self.assertFalse(path.exists(), "close unlinks the socket")
        with self.assertRaises(wire.NoServer):
            wire.request(path, {"cmd": "status"}, 1.0)

    def test_a_live_socket_is_refused_and_a_stale_one_replaced(self) -> None:
        path = self.enterContext(short_dir("w-")) / "ctl.sock"
        first = wire.ControlSocket(path, lambda request: {"ok": True, "who": "first"})
        first.start()
        try:
            second = wire.ControlSocket(path, lambda request: {"ok": True, "who": "second"})
            with self.assertRaises(FileExistsError, msg="a running server's socket is not taken over"):
                second.start()
            second.close()
            self.assertEqual(
                wire.request(path, {"cmd": "status"}, 5.0)["who"],
                "first",
                "and the refused one did not unlink it on the way out",
            )
        finally:
            first.close()
        # A socket file nobody answers on: what a server killed with -9 leaves.
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(str(path))
        stale.close()
        self.assertTrue(path.exists())
        third = wire.ControlSocket(path, lambda request: {"ok": True, "who": "third"})
        third.start()
        try:
            self.assertEqual(wire.request(path, {"cmd": "status"}, 5.0)["who"], "third")
        finally:
            third.close()

    def test_a_bind_failure_is_the_callers(self) -> None:
        """Raised by ``start`` on the caller's thread, so the server never prints a ready line over it."""
        path = self.enterContext(short_dir("w-")) / ("s" * 200 + ".sock")
        with self.assertRaises(OSError):
            wire.ControlSocket(path, lambda request: {"ok": True}).start()


class TestTheSocketPath(unittest.TestCase):
    def test_every_spelling_of_one_run_dir_agrees(self) -> None:
        deep = self.enterContext(short_dir("w-")) / ("d" * 60) / ("e" * 40)
        deep.mkdir(parents=True)
        with chdir(deep):
            self.assertEqual(wire.socket_path("run"), wire.socket_path(deep / "run"))
            self.assertEqual(wire.socket_path("run"), wire.socket_path(os.path.join(".", "run")))

    def test_a_long_spelling_of_a_short_run_dir_comes_back_short(self) -> None:
        short = self.enterContext(short_dir("w-")).resolve()
        padded = os.path.join(*([".."] * 40), *short.parts[1:], "run")  # from the cwd, past / and back down
        self.assertGreater(len(os.fsencode(padded)), wire.SUN_PATH_MAX)
        path = wire.socket_path(padded)
        self.assertLessEqual(len(os.fsencode(path)), wire.SUN_PATH_MAX, "the path returned can be bound")
        self.assertEqual(path, wire.socket_path(short / "run"))


class TestTheBind(unittest.TestCase):
    def test_a_regular_file_is_not_deleted(self) -> None:
        path = self.enterContext(short_dir("w-")) / "f.txt"
        path.write_text("precious")
        with self.assertRaises(FileExistsError):
            wire.ControlSocket(path, lambda request: {"ok": True}).bind()
        self.assertEqual(path.read_text(), "precious")


class TestTheClient(unittest.TestCase):
    def test_a_server_that_drops_the_request_is_no_reply(self) -> None:
        path = self.enterContext(short_dir("w-")) / "ctl.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        server.listen(1)

        def drop() -> None:
            conn, _ = server.accept()
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
            conn.close()

        thread = threading.Thread(target=drop, daemon=True)
        thread.start()
        try:
            with self.assertRaises((wire.NoReply, wire.NoServer)):
                wire.request(path, {"cmd": "status", "pad": "x" * 200_000}, 2.0)
        finally:
            thread.join(2.0)
            server.close()

    def test_a_frozen_server_is_no_reply_not_a_traceback(self) -> None:
        path = self.enterContext(short_dir("w-")) / "ctl.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        server.listen(0)
        try:
            for _ in range(4):  # fills the backlog: later connects fail with EAGAIN
                with self.assertRaises(wire.NoReply):
                    wire.request(path, {"cmd": "status"}, 0.2)
        finally:
            server.close()

    def test_an_unreadable_socket_is_no_server(self) -> None:
        path = self.enterContext(short_dir("w-")) / "ctl.sock"
        path.write_text("not a socket")
        with self.assertRaises(wire.NoServer):
            wire.request(path, {"cmd": "status"}, 0.5)


if __name__ == "__main__":
    unittest.main()

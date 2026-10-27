# SPDX-License-Identifier: Apache-2.0
"""Stand-ins for the live page's server, for the client's half and the server's bookkeeping.

``ctl`` talks to a page over a Unix socket and waits on what ``status`` says.
:class:`FakePage` is a real :class:`~robosimstudio_real2sim_agent.wire.ControlSocket`
answering ``status`` from a dict the test changes; :class:`FakeStatus` answers
from a script of replies, the last one forever. Neither builds a scene: the
client is standard library only, and what it is tested on is the words.
"""

from __future__ import annotations

from pathlib import Path

from robosimstudio_real2sim_agent import wire

__all__ = ["FakePage", "FakeStatus"]


class FakePage:
    """A socket answering ``status`` from a dict the test changes, for the client's half."""

    def __init__(self, path: Path, status: dict) -> None:
        self.status = dict(status)
        self.socket = wire.ControlSocket(path, lambda request: {"ok": True, **self.status})
        self.socket.start()

    def close(self) -> None:
        self.socket.close()


class FakeStatus:
    """A socket that answers ``status`` from a script of replies, the last one forever."""

    def __init__(self, path: Path, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.socket = wire.ControlSocket(path, self._answer)
        self.socket.start()

    def _answer(self, request: dict) -> dict:
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    def close(self) -> None:
        self.socket.close()

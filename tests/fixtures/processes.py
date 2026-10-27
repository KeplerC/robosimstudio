# SPDX-License-Identifier: Apache-2.0
"""Fresh interpreters, started together and read when a test asks.

Some claims are about what a *process* sees -- a package ``__path__`` extended
from an environment variable at first import, a bundle that must not import its
engine -- and the only honest way to make them is a new interpreter per claim.
Run one after another, each pays for importing the library, and two modules
that did that spent 25 s of a tier meant to take seconds. Nothing makes them
wait for each other, so :class:`Processes` starts every one a class needs in
``setUpClass`` and each test reads its own: the class costs its slowest
process, not the sum. At most :attr:`Processes.limit` run at once: a class
with seven, in a tier already run sixteen modules wide, would otherwise be
dozens of cold imports of the whole library racing for the same cores.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

__all__ = ["ROOT", "Processes", "last_json"]

ROOT = Path(__file__).resolve().parents[2]
"""The checkout, where every process starts so ``import tests`` and the library resolve as they do here."""


class Processes:
    """Named interpreters running at once, and the temporary directories they read."""

    def __init__(self, limit: int = 4) -> None:
        self.limit = limit
        self._stack = ExitStack()
        self._running: dict[str, subprocess.Popen] = {}
        self._done: dict[str, subprocess.CompletedProcess] = {}
        self._timed_out: dict[str, str] = {}

    def directory(self) -> Path:
        """A fresh directory that lives until :meth:`close`."""
        return Path(self._stack.enter_context(TemporaryDirectory()))

    def start(self, key: str, script: str, *, env: dict[str, str] | None = None) -> None:
        """Run ``python -c script`` from the checkout, GPUs hidden, with ``env`` over this process's own.

        With :attr:`limit` already running, the oldest is waited for first.
        """
        while len(self._running) >= self.limit:
            self._finish(next(iter(self._running)))
        self._running[key] = subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", **(env or {})},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def result(self, key: str, timeout: float = 600.0) -> subprocess.CompletedProcess:
        """What the process ``key`` printed and returned, waiting for it to end.

        A process that ran past ``timeout`` is killed and every later ask for it
        fails with the same message, not with a ``KeyError`` that hides why.
        """
        if key in self._running:
            self._finish(key, timeout)
        if key in self._timed_out:
            raise AssertionError(self._timed_out[key])
        return self._done[key]

    def _finish(self, key: str, timeout: float = 600.0) -> None:
        proc = self._running[key]
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, err = proc.communicate()
            self._timed_out[key] = f"{key!r} ran past {timeout} s:\n{err[-2000:]}"
        else:
            self._done[key] = subprocess.CompletedProcess(proc.args, proc.returncode, out, err)
        del self._running[key]

    def close(self) -> None:
        """Stop whatever no test read, and remove the directories."""
        for proc in self._running.values():
            proc.kill()
            proc.communicate()
        self._running.clear()
        self._stack.close()


def last_json(proc: subprocess.CompletedProcess, opener: str = "{") -> Any:
    """The last line of ``proc``'s output that starts with ``opener``, parsed."""
    return json.loads([line for line in proc.stdout.splitlines() if line.startswith(opener)][-1])

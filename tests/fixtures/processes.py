# SPDX-License-Identifier: Apache-2.0
"""Fresh interpreters, started together and read when a test asks.

Some claims are about what a *process* sees -- a package ``__path__`` extended
from an environment variable at first import, a bundle that must not import its
engine -- and the only honest way to make them is a new interpreter per claim.
Run one after another, each pays for importing the library, and two modules
that did that spent 25 s of a tier meant to take seconds. Nothing makes them
wait for each other, so :class:`Processes` starts every one a class needs in
``setUpClass`` and each test reads its own: the class costs its slowest
process, not the sum.
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

    def __init__(self) -> None:
        self._stack = ExitStack()
        self._running: dict[str, subprocess.Popen] = {}
        self._done: dict[str, subprocess.CompletedProcess] = {}

    def directory(self) -> Path:
        """A fresh directory that lives until :meth:`close`."""
        return Path(self._stack.enter_context(TemporaryDirectory()))

    def start(self, key: str, script: str, *, env: dict[str, str] | None = None) -> None:
        """Run ``python -c script`` from the checkout, GPUs hidden, with ``env`` over this process's own."""
        self._running[key] = subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", **(env or {})},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def result(self, key: str, timeout: float = 600.0) -> subprocess.CompletedProcess:
        """What the process ``key`` printed and returned, waiting for it to end."""
        if key not in self._done:
            proc = self._running.pop(key)
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
                raise AssertionError(f"{key!r} ran past {timeout} s:\n{err[-2000:]}") from None
            self._done[key] = subprocess.CompletedProcess(proc.args, proc.returncode, out, err)
        return self._done[key]

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

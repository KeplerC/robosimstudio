# SPDX-License-Identifier: Apache-2.0
"""Learning to fetch from your clicks: what the policy sees, which buttons it has, and how it is fitted.

The duck already walks on its own policy; what it learns here is the errand on top
of it, in the page's own terms. It sees one thing -- where the bowl is from its
beak, in its own frame -- and chooses one of the buttons a person has: turn 15
degrees either way, take a short step, walk a stretch, or drop. Each button is a
small move that finishes by itself, so a demonstration is simply the list of
(what it saw, which button) at each press, and the learned policy presses the
same buttons, one move at a time. :class:`Policy` is a small MLP classifier fitted
to those presses by behaviour cloning, in numpy, in a second or two.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
from duck import HELD, _apply, _rot

from robosimstudio.core.draggers import ObjectDragger

OBS = ("bowl ahead [m]", "bowl to the left [m]", "distance [m]")

BUTTONS = {
    # name: (forward [m/s], turn [rad/s], control steps it lasts)
    "Turn left": (0.0, 1.0, 13),  # about 15 degrees
    "Turn right": (0.0, -1.0, 13),
    "Step": (0.3, 0.0, 25),  # a short shuffle forward, a few centimetres
    "Walk": (0.35, 0.0, 60),  # a stretch, about 15 cm
    "Drop it": (0.0, 0.0, 0),
}
NAMES = tuple(BUTTONS)
SETTLE = 12  # control steps of standing between two moves, so each starts from a stand
REACH = 0.09  # drop once the beak is this close to the bowl's middle [m]: any closer and the toes are on its rim


def observe(duck, target_xy) -> np.ndarray:
    """Where ``target_xy`` (world, m) is from the beak, in the duck's heading frame."""
    T = duck.trunk()
    R = _rot(T[3:7])
    head = np.asarray(duck._scene.pipeline.snapshot.body_q[duck.entity.head], dtype=np.float64)
    beak = _apply(head, HELD)[:2]
    fwd = R[:2, 0] / max(np.linalg.norm(R[:2, 0]), 1e-9)
    left = np.array([-fwd[1], fwd[0]])
    to = np.asarray(target_xy, dtype=np.float64)[:2] - beak
    return np.array([fwd @ to, left @ to, np.linalg.norm(to)], dtype=np.float32)


def teacher(obs) -> str:
    """A careful person's rule: face the bowl, walk while it is far, shuffle when near, drop when there."""
    ahead, left, dist = (float(v) for v in obs)
    if dist < REACH:
        return "Drop it"
    bearing = math.atan2(left, ahead)
    if abs(bearing) > 0.2:
        return "Turn left" if bearing > 0 else "Turn right"
    return "Walk" if dist > 0.25 else "Step"


class Moves:
    """Runs one button's move on the duck at a time; ``step()`` once per control step."""

    def __init__(self, duck) -> None:
        self.duck = duck
        self.left = 0

    @property
    def idle(self) -> bool:
        return self.left == 0 and not self.duck.busy

    def press(self, name: str) -> None:
        if name == "Drop it":
            self.duck.walk(0, 0, 0)
            self.duck.drop()
            return
        speed, turn, steps = BUTTONS[name]
        self.duck.walk(speed, 0.0, turn)
        self.left = steps + SETTLE

    def step(self) -> None:
        if self.left > 0:
            self.left -= 1
            if self.left == SETTLE:
                self.duck.walk(0, 0, 0)


@dataclass
class Demos:
    """Recorded demonstrations: one list of (obs, button) rows per round, a row per press."""

    episodes: list = field(default_factory=list)

    def start(self) -> None:
        self.episodes.append([])

    def add(self, obs, button: str) -> None:
        if not self.episodes:
            self.start()
        self.episodes[-1].append((np.asarray(obs).tolist(), button))

    @property
    def rounds(self) -> int:
        """Rounds that ended in a drop."""
        return sum(1 for ep in self.episodes if ep and ep[-1][1] == "Drop it")

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        rows = [r for ep in self.episodes for r in ep]
        return np.array([o for o, _ in rows], np.float32), np.array([NAMES.index(b) for _, b in rows], np.int64)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.episodes))

    @classmethod
    def load(cls, path) -> Demos:
        return cls(json.loads(Path(path).read_text()))


class Policy:
    """obs -> which button: 3 -> 64 -> 64 -> 5, tanh hidden layers, softmax out."""

    def __init__(self, seed: int = 0, hidden: int = 64) -> None:
        rng = np.random.default_rng(seed)
        sizes = (len(OBS), hidden, hidden, len(NAMES))
        self.W = [rng.normal(0, 1 / math.sqrt(a), (a, b)).astype(np.float32) for a, b in pairwise(sizes)]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.mu, self.sd = np.zeros(len(OBS), np.float32), np.ones(len(OBS), np.float32)

    def _forward(self, x):
        h = [(x - self.mu) / self.sd]
        for i, (W, b) in enumerate(zip(self.W, self.b, strict=True)):
            z = h[-1] @ W + b
            h.append(np.tanh(z) if i < len(self.W) - 1 else z)
        return h

    def __call__(self, obs) -> str:
        logits = self._forward(np.asarray(obs, np.float32)[None])[-1][0]
        return NAMES[int(np.argmax(logits))]

    def fit(self, X, Y, *, steps: int = 2000, lr: float = 3e-3, seed: int = 0) -> list[float]:
        """Behaviour cloning: cross-entropy on the button pressed, every class weighed equally. Adam."""
        rng = np.random.default_rng(seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        counts = np.bincount(Y, minlength=len(NAMES)).astype(np.float32)
        weight = (len(Y) / (len(NAMES) * np.maximum(counts, 1)))[Y]
        params = [*self.W, *self.b]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        losses = []
        for t in range(1, steps + 1):
            idx = rng.integers(0, len(X), min(256, len(X)))
            h = self._forward(X[idx])
            z = h[-1] - h[-1].max(1, keepdims=True)
            p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
            w = weight[idx][:, None]
            onehot = np.eye(len(NAMES), dtype=np.float32)[Y[idx]]
            losses.append(float(np.mean(w[:, 0] * -np.log(p[np.arange(len(idx)), Y[idx]] + 1e-7))))
            g = w * (p - onehot) / len(idx)
            grads_W, grads_b = [], []
            for i in reversed(range(len(self.W))):
                grads_W.insert(0, h[i].T @ g)
                grads_b.insert(0, g.sum(0))
                if i:
                    g = (g @ self.W[i].T) * (1 - h[i] ** 2)
            for j, grad in enumerate([*grads_W, *grads_b]):
                m[j] = 0.9 * m[j] + 0.1 * grad
                v[j] = 0.999 * v[j] + 0.001 * grad**2
                params[j] -= lr * (m[j] / (1 - 0.9**t)) / (np.sqrt(v[j] / (1 - 0.999**t)) + 1e-8)
        return losses

    def save(self, path) -> None:
        np.savez(path, *self.W, *self.b, mu=self.mu, sd=self.sd)

    @classmethod
    def load(cls, path) -> Policy:
        d = np.load(path)
        pol = cls()
        n = len(pol.W)
        pol.W = [d[f"arr_{i}"] for i in range(n)]
        pol.b = [d[f"arr_{n + i}"] for i in range(n)]
        pol.mu, pol.sd = d["mu"], d["sd"]
        return pol


class Driver:
    """Press the buttons a policy (or the teacher) chooses, one move at a time, until it drops.

    ``record`` takes the (obs, button) of every press, so the teacher's rounds become demos.
    """

    def __init__(self, duck, choose, target, *, record: Demos | None = None, presses: int = 60) -> None:
        self.duck, self.choose, self.target, self.record = duck, choose, target, record
        self.moves = Moves(duck)
        self.left, self.done = presses, False

    def step(self) -> None:
        self.moves.step()
        if self.done or not self.moves.idle:
            return
        if self.left == 0:
            self.done = True
            return
        obs = observe(self.duck, self.target())
        button = self.choose(obs)
        if self.record is not None:
            self.record.add(obs, button)
        self.moves.press(button)
        self.left -= 1
        self.done = button == "Drop it"


class Rounds:
    """A round's set-up on the page: the bowl to a fresh spot, the ball into the beak.

    :meth:`setup` is a generator that does one control step's worth per ``next``, so
    the page can run it from its per-step callback while the scene keeps stepping.
    The bowl goes at least 0.25 m from where the duck stands, anywhere on the bench,
    and both moves are made with the same drag a mouse handle uses.
    """

    def __init__(self, sim, duck, ball, bowl, *, seed: int = 0) -> None:
        self.sim, self.duck, self.ball, self.bowl = sim, duck, ball, bowl
        self.rng = np.random.default_rng(seed)  # the same rounds every time the page opens
        self.dragger = ObjectDragger(sim.model)

    def _drag(self, body, points, frames):
        hooks = self.sim.pipeline.hooks.rigid_force
        if self.dragger.apply not in list(hooks):  # a Reset clears hooks added after the build
            hooks.append(self.dragger.apply)
            self.sim.pipeline.invalidate_graph()
        X = self.sim.pipeline.state_0.body_q.numpy()[body]
        self.dragger.attach(body, X[:3], X[3:7])
        start = X[:3].copy()
        for point, n in zip(points, frames, strict=True):
            end = np.asarray(point() if callable(point) else point)
            for i in range(n):
                a = (i + 1) / n
                self.dragger.set_target((1 - a) * start + a * end, X[3:7])
                yield
            start = end
        self.dragger.detach()

    def setup(self):
        for _ in range(10):  # the page resets the scene before this step's callbacks; let the duck settle
            yield
        T = self.duck.trunk()
        while True:
            p = np.array([0.6 + self.rng.uniform(-0.19, 0.19), self.rng.uniform(-0.55, 0.55)])
            if np.linalg.norm(p - T[:2]) > 0.25:
                break
        b = self.bowl.position
        yield from self._drag(
            int(self.bowl.handle.body), [b + np.array([0, 0, 0.03]), [*p, b[2] + 0.03], [*p, b[2]]], [15, 50, 15]
        )
        self.duck.mouth(1.0)
        top = self.ball.position + np.array([0, 0, 0.12])
        yield from self._drag(
            int(self.ball.handle.body), [top, lambda: self.duck.held_point(0.08), self.duck.held_point], [25, 45, 35]
        )
        self.duck.mouth(0.0)
        for _ in range(25):
            yield

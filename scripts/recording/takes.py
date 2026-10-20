# SPDX-License-Identifier: Apache-2.0
"""Every clip an example README shows, as the take that films it, or the reason it is kept.

A take is the example's own script, run unchanged, with a person on its page:
the gestures are :mod:`.beats`, the camera and the sidebar are set before the
first frame, and the expectations say what the physics must have done for the
clip to ship. ``scripts/record_examples.py`` films them.

A take marked ``pending`` is a draft: its example's physics is still being
changed, so it is filmed into ``outputs/record/`` for a look and never written
into a ``media/`` folder. Clear the mark once the take has been tuned on the
final physics and its sheet looked at.

Dataclasses only, and no import of the library or a browser:
``tests/test_record_takes.py`` reads this table on a machine with neither.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any

from .beats import (
    At,
    Beat,
    Bench,
    Camera,
    Check,
    Click,
    Corner,
    Do,
    Drag,
    DragTo,
    Expectation,
    Fold,
    Grasp,
    Hand,
    Held,
    Highest,
    Laid,
    Layout,
    Mid,
    Nearest,
    Orbit,
    Patch,
    Policy,
    Press,
    Rest,
    Rose,
    Run,
    Sank,
    Scroll,
    Select,
    Slider,
    Squashed,
    Toggle,
    Top,
    Turn,
    Wait,
    laid,
)

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Take:
    """One clip: which script, what happens on its page, and what must be true afterwards."""

    example: str
    """The script, from the repository root: ``examples/00_getting_started/00_hello_world.py``."""
    beats: tuple[Beat, ...] = ()
    name: str = ""
    """The clip's name; the script's stem by default, anything else an extra clip."""
    folder: str = ""
    """The folder whose ``media/`` gets it, from the repository root; the script's own by default."""
    argv: tuple[str, ...] = ()
    """Arguments to the script: ``("--simulate",)`` for a teleop example with no arm attached."""
    needs: tuple[str, ...] = ()
    """Extras the script needs: ``uv run --extra <x>``."""
    camera: Camera | Orbit | None = None
    """``None`` keeps the page's own opening view."""
    layout: Layout = field(default_factory=Layout)
    config: tuple[tuple[str, Any], ...] = ()
    """``StudioConfig`` fields set on top of the example's, each with its reason beside it."""
    warmup: float = 0.0
    """Seconds of scene stepped before the browser opens, unfilmed."""
    tail: float = 0.6
    """Seconds of clip filmed after the last beat."""
    fps: int = 15
    speed: float = 1.0
    """Seconds of scene per second of clip. Above 1 for a routine that runs past half a minute."""
    viewport: tuple[int, int] = (1152, 648)
    """CSS pixels: the sidebar is 324 of them, the 3D view the rest."""
    dpr: float = 2.0
    size: tuple[int, int] = (960, 540)
    """The clip: the page at twice its pixels, scaled down with Lanczos."""
    budget: int = 900_000
    cursor: bool = True
    expect: tuple[Expectation, ...] = ()
    pending: str = ""
    """Why this take is a draft. Non-empty: filmed into ``outputs/record/`` only."""

    @property
    def stem(self) -> str:
        return self.name or Path(self.example).stem

    @property
    def home(self) -> str:
        """The folder the clip belongs to, from the repository root."""
        return self.folder or str(Path(self.example).parent)

    @property
    def key(self) -> str:
        """``<folder>/<clip>``, ``examples/`` left off: how the command line and :data:`KEPT` name a clip."""
        return f"{self.home.removeprefix('examples/')}/{self.stem}"

    @property
    def media(self) -> Path:
        """Where the clip goes when it ships."""
        return REPO / self.home / "media" / f"{self.stem}.webp"


# =============================================================================
# Final: their physics is settled, and each ships when its expectations hold
# =============================================================================


def _spread(film, name: str, at: str, since: str = "start") -> float:
    return film.marks[at]["spreads"][name] - film.marks[since]["spreads"][name]


def _stiffened(film) -> tuple[bool, str]:
    """The tape stood out flat at ``stiff`` and fell back into its drape by ``end``."""
    out, back = _spread(film, "tape", "stiff"), _spread(film, "tape", "end")
    return out > 0.03 and abs(back) < 0.01, (
        f"the tape's footprint grew {out * 1000:.1f} mm stiffened (need 30) and ended {back * 1000:+.1f} mm from where "
        "it began (need within 10)"
    )


def _routed(film) -> tuple[bool, str]:
    """The cable demo's own goal terms (its ``board.routed``): seated in both spools, its end in the port."""
    import board  # the demo's folder is on sys.path while its script runs

    r = board.routed(film.sim)
    ok = r["seat0"] >= 0.04 and r["seat1"] >= 0.04 and r["port"] >= 0.06 and r["end"] >= 1.0
    return ok, f"seats {r['seat0']:.2f}/{r['seat1']:.2f} (need 0.04), port {r['port']:.2f} (0.06), end {r['end']:.0f}"


def _script(film):
    """The take's script as a module of its own, for the measures it defines; loaded once, its ``main()`` not run."""
    import importlib.util
    import pathlib
    import sys

    path = pathlib.Path(__file__).resolve().parents[2] / film.take.example
    name = "_take_" + path.stem
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return sys.modules[name]


def _folded(film) -> tuple[bool, str]:
    """The fold demo's own measures (its ``fold_measures``), against the cloth_fold suite's gate."""
    m = _script(film).fold_measures(film.sim)
    ok = m["fold_extent"] <= 0.6 and m["fold_score"] >= 0.85
    return ok, f"extent {m['fold_extent']:.3f} (need <= 0.6), score {m['fold_score']:.3f} (need >= 0.85)"


def _wiped(film) -> tuple[bool, str]:
    """The wipe demo's own measure: the share of the marker stripe the sponge cleaned."""
    cleaned = float(film.sim["stripe"].entity.cleaned)
    return cleaned >= 0.9, f"{cleaned:.3f} of the stripe clean (need 0.9)"


def _kept_bend(film) -> tuple[bool, str]:
    """The cable with plastic memory is still bent after the hand let go: straightness under 0.9."""
    import numpy as np

    from robosimstudio_materials.rod.measures import straightness

    ref = film.sim["cable"]
    measured = straightness(np.asarray(film.sim.pipeline.state_0.body_q.numpy())[ref.bodies, :3])
    return measured < 0.9, f"straightness {measured:.3f} after release (need under 0.9)"


def _peg(film, name: str) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """A peg of ``03_materials/04_band`` as the example lays it: its root on the board [world], the way out, its length.

    A post's middle and half-length off its spec, turned with it: the example turns each post's
    axis from z onto x, so its +z end is the root, on the board's face at bench x 0.10, and it
    points out along -x; the upper is 80 mm long at 260 mm up, the lower 50 at 135.
    """
    spec = film.sim[name].spec
    qx, qy, qz, qw = (float(v) for v in spec.xform[3:7])
    axis = (2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy))
    half = float(spec.shape.half_height)
    root = film.bench(tuple(p + half * a for p, a in zip(laid(film, name), axis, strict=True)))
    return root, tuple(-a for a in axis), 2.0 * half


def _band_on_both(film) -> tuple[bool, str]:
    """The band is round both pegs of the board and stretched there, the hand gone."""
    import numpy as np

    from robosimstudio_materials.rod.band import encircles

    ref = film.sim["band"]
    points = np.asarray(ref.particle_q, dtype=np.float64)
    on = {n: encircles(points, *_peg(film, n)) for n in ("upper", "lower")}
    stretch = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) / float(ref.entity.spec.length)
    return all(on.values()) and stretch > 1.1, f"round {on}, stretch {stretch:.3f} (need both, over 1.1)"


def _knot_tight(film) -> tuple[bool, str]:
    """The knot kept its three crossings, gave back rope, and both tails are still in their hands."""
    import numpy as np

    from robosimstudio_materials.rod.knot import crossings, knot_length

    rope = film.sim["rope"]
    points = np.asarray(rope.particle_q, dtype=np.float64)
    built = knot_length(np.asarray(rope.spec.nodes, dtype=np.float64))  # the knot as the example tied it
    now, cross = knot_length(points), crossings(points)
    hand = {a: film.sim.arm(a).ee_pose(film.sim.state)[0] for a in ("left", "right")}  # the page is shut by now
    held = max(float(np.linalg.norm(points[i] - hand[a])) for i, a in ((0, "right"), (-1, "left")))
    ok = cross == 3 and now < 0.95 * built and held < 0.06
    return (
        ok,
        f"{cross} crossings, {now * 1e3:.0f} of {built * 1e3:.0f} mm in the knot, tails {held * 1e3:.0f} mm off the hands",
    )


def _wrap_post(film) -> tuple[tuple[float, float, float], float]:
    """``03_materials/01_rope_memory``'s post as the example stands it, on the bench [world], and its radius plus the cable's [m].

    Bench (0, -86) mm, and 20 + 11 mm: what a radian of wrap lays on the post.
    """
    post = film.bench((*laid(film, "post")[:2], 0.0))
    return post, float(film.sim["post"].spec.shape.radius) + float(film.sim["cable"].spec.radius)


def _round_the_post(film) -> dict:
    """The memory take's carry: the right cursor walked 120 degrees round the post, its gizmo turning with it.

    What the gizmo's arrows and ring write, a film frame at a time: no beat drags
    an arc or turns a ring. The path is ``tests/test_cable_robot``'s, at 50 mm/s,
    about the post where the example stands it, shrinking by what the wrap lays on
    it: the post's radius plus the cable's, 31 mm a radian. The capsule's distance
    from the hand at the end is kept for :func:`_carried_round`.
    """
    import numpy as np

    from robosimstudio.core.math import quat_multiply

    arm, wrap = film.session.arms["right"], math.radians(120.0)
    post, lays = _wrap_post(film)
    x0, y0, z = (float(v) for v in arm.cursor_pos)
    start = (x0 - post[0], y0 - post[1])
    rho = math.hypot(*start)
    frames = round(rho * wrap / 0.05 * film.take.fps)
    for i in range(1, frames + 1):
        a, shrink = wrap * i / frames, 1.0 - wrap * lays / rho * i / frames
        x = (math.cos(a) * start[0] - math.sin(a) * start[1]) * shrink
        y = (math.sin(a) * start[0] + math.cos(a) * start[1]) * shrink
        quat = quat_multiply(np.array([0.0, 0.0, math.sin(a / 2), math.cos(a / 2)]), np.array([0.0, 1.0, 0.0, 0.0]))
        arm.aim((post[0] + x, post[1] + y, z), quat, moving=i < frames)
        film.hold(1.0 / film.take.fps)
    film.hold(0.5)
    cable = film.sim["cable"]
    body_q = film.sim.state.body_q.numpy()
    film.carried = min(math.dist(body_q[b, :3], film.hand("right")) for b in cable.bodies)
    return {"frames": frames, "carried_mm": round(film.carried * 1000, 1)}


def _carried_round(film) -> tuple[bool, str]:
    """The cable stayed in the right hand's jaws all the way round the post."""
    gap = getattr(film, "carried", math.inf)
    return gap < 0.02, f"the nearest capsule ended {gap * 1000:.1f} mm from the tool centre (need under 20)"


def _mid_twists(film) -> dict[str, float]:
    """The twist example's readouts: each cable's turns through the middle third of its capsules, unsigned."""
    import numpy as np

    from robosimstudio_materials.rod.measures import twist_turns

    body_q = np.asarray(film.sim.state.body_q.numpy())
    out = {}
    for name in ("soft_twist", "stiff_twist"):
        bodies = film.sim[name].bodies
        n = len(bodies)
        out[name] = abs(twist_turns(body_q[bodies[n // 3 : 2 * n // 3 + 1], 3:7]))
    return out


def _turn_both(film) -> dict:
    """The twist take's turn: each hand's cursor turned 120 degrees inward about the vertical, over four seconds.

    What each gizmo's ring writes; no beat turns a ring. ``tests/test_cable_robot``
    turns them the same way, a degree a control step. The mid-cable twists at the
    end are kept for :func:`_twisted`.
    """
    import numpy as np

    from robosimstudio.core.math import quat_multiply, quat_to_mat

    level = np.array([0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)])
    arms = {"right": 1.0, "left": -1.0}
    start = {}
    for name in arms:
        axis = quat_to_mat(np.asarray(film.session.arms[name].cursor_quat))[:, 2]
        start[name] = math.atan2(axis[1], axis[0])
    frames = round(4.0 * film.take.fps)  # 2 degrees a frame: a frame is four control steps, and a jump of
    # four degrees at once, twice the test's whole step, let the soft tip slip a fifth of its turn
    for i in range(1, frames + 1):
        for name, sign in arms.items():
            yaw = start[name] + sign * math.radians(120.0) * i / frames
            panel = film.session.arms[name]
            quat = quat_multiply(np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]), level)
            panel.aim(panel.cursor_pos, quat, moving=i < frames)
        film.hold(1.0 / film.take.fps)
    film.hold(1.0)
    film.twist_held = _mid_twists(film)
    return {"turned_deg": 120.0, "held": {n: round(v, 3) for n, v in film.twist_held.items()}}


def _twisted(film) -> tuple[bool, str]:
    """The turn reached both middles while held, and once let go only the stiff cable gave it back."""
    held = getattr(film, "twist_held", {"soft_twist": 0.0, "stiff_twist": 0.0})
    now = _mid_twists(film)
    ok = (
        held["soft_twist"] > 0.09
        and held["stiff_twist"] > 0.03
        and now["stiff_twist"] < 0.01
        and now["soft_twist"] > 0.06
    )
    return ok, (
        f"mid-cable turns held soft {held['soft_twist']:.3f} stiff {held['stiff_twist']:.3f} (need 0.09, 0.03); "
        f"let go soft {now['soft_twist']:.3f} stiff {now['stiff_twist']:.3f} (need over 0.06, under 0.01)"
    )


def _succeeded(film) -> tuple[bool, str]:
    """The task's own verdict on the scene as the take left it."""
    result = film.sim.evaluate()[-1]
    return bool(result["success"]), f"the task says success={result['success']}"


def _walled(film, walls: str, floor: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The inside of four slabs as the example lays them, bench frame [m]: between their faces, ``floor`` to their top.

    ``<walls>_near`` and ``_far`` close it in x, ``_right`` and ``_left`` in y. ``03_materials/12_liquid``'s
    bowl is 140 mm square about bench (0.12, -0.16) and 40 mm tall; ``03_materials/14_water``'s glass 60 mm
    square about (0.12, -0.13) and 55 mm tall.
    """
    near, far = laid(film, f"{walls}_near", (1, 0, 1)), laid(film, f"{walls}_far", (-1, 0, 0))
    right, left = laid(film, f"{walls}_right", (0, 1, 0)), laid(film, f"{walls}_left", (0, -1, 0))
    return (near[0], right[1], floor), (far[0], left[1], near[2])


def _poured(film) -> tuple[bool, str]:
    """The pour example's readout at the end: the share of the honey inside the bowl, 90% or more."""
    import numpy as np

    from robosimstudio_materials.mpm.measures import fill_fraction

    lo, hi = (np.array(film.bench(p)) for p in _walled(film, "bowl", -0.005))
    share = fill_fraction(np.asarray(film.sim["honey"].particle_q, dtype=np.float64), lo, hi)
    return share >= 0.9, f"{share:.3f} of the honey in the bowl (need 0.9)"


def _water_in_glass(film) -> tuple[bool, str]:
    """The water example's readout at the end: 90% of the water or more in the glass, standing past its 25 mm line."""
    import numpy as np

    from robosimstudio_materials.water import fill_level

    lo, hi = (np.array(film.bench(p)) for p in _walled(film, "glass", 0.0))
    filled = fill_level(np.asarray(film.sim["water"].particle_q, dtype=np.float64), lo, hi, 0.004)
    ok = filled["fraction"] >= 0.9 and filled["height"] >= 0.025
    return (
        ok,
        f"{filled['fraction']:.3f} of the water in the glass, {1e3 * filled['height']:.1f} mm deep (need 0.9, 25 mm)",
    )


def _bag_open(film) -> tuple[bool, str]:
    """The bag's mouth stands 30 mm open in the right hand, and its layers stayed apart."""
    import numpy as np

    from robosimstudio_materials.bag import layer_gap, mouth_opening

    bag = film.sim["bag"]
    points = np.asarray(bag.particle_q, dtype=np.float64)
    mouth, gap = mouth_opening(points, bag.entity), layer_gap(points, bag.entity)
    return mouth["height"] >= 0.03 and gap > 0.0015, (
        f"the mouth stands {mouth['height'] * 1000:.1f} mm open (need 30), {mouth['area'] * 1e4:.1f} cm^2, "
        f"the layers at least {gap * 1000:.2f} mm apart (need 1.5)"
    )


RIGHT = "Arm: right"
"""The right arm's fold; its title may carry a role, ``Arm: right (work)``, which a prefix still matches."""
PANDA_VIEW = Camera(eye=(0.95, -0.25, 1.00), look_at=(0.457, -0.115, 0.85))
"""The getting-started Panda's hand, which starts 231 mm over the cube, pointing down."""
TASK_VIEW = Camera(eye=(1.15, -0.03, 1.10), look_at=(0.65, -0.03, 0.82))
"""The cube task: the cube at bench (0, 0.1), the square around (0.1, -0.15), and the Panda between."""


def lift(example: str, *, before: tuple[Beat, ...] = (), camera: Any = PANDA_VIEW, **kw) -> Take:
    """The rigid tutorials' gesture: the hand down onto the cube, Close, up 150 mm, held, Open."""
    return Take(
        example,
        camera=camera,
        warmup=1.0,  # a released cube lands and the jaws finish opening before the first frame
        beats=(
            *before,
            Wait(seconds=0.5, label="the hand, open, over the cube"),
            Drag("right", "z", to=At("cube"), seconds=1.4, label="drag the gizmo down onto the cube"),
            Wait(seconds=0.3),
            Press(text="Close", within=RIGHT, label="Close"),
            Wait(seconds=0.9, mark="closed", label="friction holds it"),
            Drag("right", "z", 0.15, seconds=1.4, label="drag it up 150 mm"),
            Wait(seconds=0.8, mark="lifted", label="held"),
            Press(text="Open", within=RIGHT, label="Open: it drops"),
            Wait(seconds=1.2),
        ),
        # The rigid tutorials' own medium test holds the same gesture to a rise over 80 mm.
        expect=(Rose("cube", 0.10, at="lifted"), Held("cube", since="closed", at="lifted", tolerance=0.01)),
        **kw,
    )


def policy(example: str, **kw) -> Take:
    """A task page's Run: the policy carries the cube to the square on its own, and the task scores it."""
    return Take(
        example,
        camera=TASK_VIEW,
        warmup=0.5,
        cursor=True,
        beats=(
            Wait(seconds=0.5, label="the cube, the square, the Panda"),
            Policy(label="Run the policy"),
            Wait(seconds=1.0, label="scored"),
        ),
        # Not the jaw width: under a policy's carry the cube turns in the pads and reads 54 to 58 mm.
        expect=(Check(_succeeded),),
        budget=1_200_000,  # a policy's run is as long as it takes, and these take 15 to 20 s
        **kw,
    )


FINAL = (
    Take(
        "examples/00_getting_started/01_core_concepts.py",
        name="drag_cube",
        folder="examples/05_studio",
        camera=Orbit(At("ball", dz=0.08), distance=0.6, azimuth=-5.0, elevation=26.0, fov=50.0),
        warmup=1.0,
        layout=Layout(collapse=("Simulation", "Arm: right")),
        beats=(
            Wait(seconds=0.5, label="a ball on the bench"),
            Click(At("ball"), label="click it: a handle"),
            Wait(seconds=0.3),
            Drag("handle", "z", 0.12, seconds=1.2, label="drag the handle up"),
            Drag("handle", "y", -0.08, seconds=1.0, label="and across"),
            Wait(seconds=0.6, mark="carried", label="held by the handle"),
            Press(text="Release all", label="Release all: it falls"),
            Wait(seconds=1.4),
        ),
        expect=(Rose("ball", 0.08, at="carried"),),
    ),
    Take(
        "examples/00_getting_started/00_hello_world.py",
        name="drive_arm",
        folder="examples/05_studio",
        camera=PANDA_VIEW,
        warmup=1.0,
        beats=(
            Wait(seconds=0.5),
            Drag("right", "z", to=At("cube"), seconds=1.4, label="drag the hand down"),
            Press(text="Close", within=RIGHT, label="Close"),
            Wait(seconds=0.8, mark="closed"),
            Drag("right", "z", 0.12, seconds=1.2, label="lift"),
            Drag("right", "y", 0.10, seconds=1.0, label="carry"),
            Wait(seconds=0.4, mark="carried"),
            # The cube's middle 2 mm over where it rests: set down, not dropped.
            Drag("right", "z", to=Bench(z=0.027), seconds=1.2, label="set it down"),
            Press(text="Open", within=RIGHT, label="Open"),
            Drag("right", "z", 0.08, seconds=0.8, label="and away"),
            Wait(seconds=0.8),
        ),
        expect=(Rose("cube", 0.08, at="carried"), Held("cube", since="closed", at="carried")),
    ),
    Take(
        "examples/01_cloth/01_fabrics.py",
        # The tape in the middle, the jersey to its left, the fleece to its right.
        camera=Orbit(At("tape", dz=-0.04), distance=0.75, azimuth=-20.0, elevation=30.0, fov=50.0),
        warmup=3.0,  # the four sheets settle over their posts before anyone looks
        beats=(
            Wait(seconds=0.5, label="four fabrics on four posts"),
            Fold(title="Material", label="open Material"),
            Select(text="Cloth", option="tape", within="Material", label="pick the tape"),
            Scroll(pixels=360, seconds=0.6, label="to its bending stiffness"),
            # Bending, upwards. Measured on this scene, headless: the draped tape's
            # footprint goes 49 -> 98 mm across at edge_ke 100 N m/rad and back to
            # 49.3 at its own 0.1. Softening a settled sheet moves nothing -- the
            # fleece's in-plane stiffness down 100x lowered it 1.3 mm -- because
            # taking a stiffness away does not push; adding one does.
            Slider(text="edge_ke", value=2.0, seconds=1.4, within="Material", label="edge_ke up 1000x"),
            Wait(seconds=1.8, mark="stiff", label="the tape stands out flat"),
            Slider(text="edge_ke", value=None, seconds=1.4, within="Material", label="and back"),
            Wait(seconds=1.8, label="it drapes again"),
        ),
        expect=(Check(_stiffened, needs=("tape",)),),
    ),
    lift("examples/00_getting_started/00_hello_world.py"),
    lift("examples/00_getting_started/01_core_concepts.py"),
    lift("examples/04_extending/00_add_a_robot.py"),
    lift(
        "examples/05_studio/00_your_own_panel.py",
        camera=Camera(eye=(1.05, -0.20, 1.00), look_at=(0.55, -0.05, 0.85)),
        before=(
            Wait(seconds=0.5, label="a panel of the page's own: Cube"),
            Press(text="Toss the cube", within="Cube", label="Toss the cube"),
            Wait(seconds=1.4, label="up 11 cm at 1.5 m/s, and down"),
            DragTo("right", At("cube", dz=0.10), label="over it"),
        ),
    ),
    Take(
        "examples/09_teleop/00_leader_arm.py",
        argv=("--simulate",),
        cursor=False,  # a scripted leader arm drives the hands; no one is at the mouse, and the gizmos are hidden
        # Both hands: they work at about z 1.15, y -0.5 to +0.5, well over the cube.
        camera=Camera(eye=(1.55, -0.35, 1.30), look_at=(0.55, 0.0, 0.98)),
        warmup=0.5,
        # 9 s: both arms move through most of the frame the whole time, and 12 s was 968 KB at the bottom of the ladder.
        beats=(Wait(seconds=9.0, label="a simulated leader arm drives the follower"),),
    ),
    Take(
        "examples/06_tasks/00_your_first_task.py",
        camera=TASK_VIEW,
        warmup=1.0,
        budget=1_200_000,  # nine drags: the longest gesture of the examples
        beats=(
            Wait(seconds=0.5, label="the cube and the square"),
            DragTo("right", Bench(0.0, 0.1, 0.125), label="over the cube"),
            Drag("right", "z", to=At("cube"), seconds=1.2, label="down onto it"),
            Press(text="Close", within=RIGHT, label="Close"),
            Wait(seconds=0.8, mark="closed"),
            Drag("right", "z", 0.10, seconds=1.0, label="up"),
            DragTo("right", Bench(0.1, -0.15, 0.125), label="over the square"),
            Drag("right", "z", -0.10, seconds=1.0, label="down"),
            Press(text="Open", within=RIGHT, label="Open"),
            Drag("right", "z", 0.08, seconds=0.8, label="and away"),
            Wait(seconds=1.2, mark="placed", label="the goal turns true"),
        ),
        expect=(Check(_succeeded),),
    ),
    policy("examples/07_policies/00_pick_and_place.py"),
    policy("examples/07_policies/01_check_and_retry.py"),
    policy("examples/07_policies/02_without_privilege.py"),
)


# =============================================================================
# The rest: the grasps on cloth, rope and foam, and the pages a hand cannot show
# =============================================================================


def grasp(example: str, thing: str, *, at: Any = None, lift: float = 0.12, **kw) -> Take:
    """The gesture as one beat: the nearest hand over the thing, down, Close, up."""
    return Take(
        example,
        camera=kw.pop("camera", Orbit(Mid(At(thing), Nearest(thing)), distance=0.85, azimuth=-15.0, elevation=25.0)),
        warmup=kw.pop("warmup", 1.0),
        beats=(
            Wait(seconds=0.5),
            Grasp(thing=thing, at=at, lift=lift, label=f"down onto the {thing}, Close, up"),
            Wait(seconds=1.0),
        ),
        expect=kw.pop("expect", (Rose(thing, lift * 0.6, at="lifted"),)),
        **kw,
    )


def handle(example: str, thing: str, at: Any, *, camera: Any, lift: float = 0.10, across=("y", -0.06), **kw) -> Take:
    """A click on the thing and a drag of the handle it puts there, then Release all.

    For the engines a robot's grasp cannot honestly be shown on -- a cloth on
    another solver, a tet mesh, a kind of your own, a SuperDex rod -- the
    handle is the studio's own way in: a pin on the patch clicked, or a PD
    wrench on the body, and the thing dynamic under it.
    """
    radius = kw.pop("radius", 0.02)
    return Take(
        example,
        camera=camera,
        warmup=kw.pop("warmup", 1.0),
        layout=Layout(collapse=("Simulation", "Arm: left", "Arm: right")),
        beats=(
            Wait(seconds=0.5, label=f"the {thing}"),
            Click(at, label=f"click the {thing}: a handle"),
            Wait(seconds=0.3),
            Drag("handle", "z", lift, seconds=1.2, label="drag the handle up"),
            Drag("handle", across[0], across[1], seconds=1.0, label="and across"),
            Wait(seconds=0.8, mark="lifted", label="it hangs from the handle"),
            Press(text="Release all", label="Release all"),
            Wait(seconds=1.4, label="and it falls"),
        ),
        # What the handle took comes up with it; the rest of a sheet, a ring or a heap need not.
        expect=kw.pop("expect", (Rose(Patch(thing, near=at, radius=radius), lift * 0.5, at="lifted"),)),
        **kw,
    )


def routine(
    example: str,
    *,
    speed: float,
    success: str,
    done: Any = None,
    budget: int = 1_500_000,
    needs: tuple[str, ...] = (),
) -> Take:
    """A demo: the home pose, then its Run button, filmed to the routine's end.

    ``done(film)`` is the demo's own success measure; a take without one is still a draft.
    """
    return Take(
        example,
        needs=needs,
        fps=12,
        speed=speed,
        warmup=0.5,
        budget=budget,
        beats=(
            Wait(seconds=0.5, label="home, hands open"),
            Run(label="Run"),
            Wait(seconds=1.5, label="done"),
        ),
        expect=(
            Check(lambda film: film.ended.endswith(": done"), "the routine ran to its end:"),
            Check(done or (lambda film: True), success),
        ),
        pending="" if done else "the demo and its routine are being built on another branch",
    )


TOWEL = "examples/01_cloth/00_towel.py"
TOWEL_VIEW = Orbit(At("towel"), distance=0.8, azimuth=-50.0, elevation=35.0)
"""The towel from its corner: both its x and its y arrows have length on screen."""
TELEOP_VIEW = Camera(eye=(1.55, -0.35, 1.30), look_at=(0.55, 0.0, 0.98))
"""Both YAM hands: they work at about z 1.15, y -0.5 to +0.5, well over the bench."""

DRAFTS = (
    Take(
        TOWEL,
        camera=Orbit(Bench(0.0, -0.11, 0.08), distance=1.1, azimuth=-22.0, elevation=30.0, fov=40.0),
        warmup=1.0,  # the towel lands from 20 mm
        layout=Layout(collapse=("Simulation", "Arm: left")),
        beats=(
            Wait(seconds=0.5, label="the right hand, open, over the towel's near edge"),
            Drag("right", "z", to=(0.600, -0.200, 0.776), seconds=1.2, label="drag the gizmo down to the towel"),
            # It stops about 0.748, the fingertips on the bench at the edge.
            Drag("right", "z", to=(0.600, -0.200, 0.746), seconds=0.8, label="and down onto the bench"),
            Wait(seconds=0.3),
            Press(text="Close", within=RIGHT, label="Close: the pads take the edge"),
            Wait(seconds=1.0, mark="closed"),
            Drag("right", "z", to=(0.600, -0.200, 0.848), seconds=1.4, label="drag it up 100 mm"),
            Wait(seconds=2.0, mark="lifted", label="held by friction"),
        ),
        # The edge under the jaws, not the towel: its centroid barely moves. Measured 96 mm.
        expect=(Rose(Patch("towel", near=(0.600, -0.200, 0.730)), 0.06, at="lifted"),),
    ),
    Take(
        TOWEL,
        name="pinch_towel",
        camera=TOWEL_VIEW,
        warmup=1.0,
        beats=(
            Wait(seconds=0.5),
            Click(Corner("towel", -1, -1, 0.03), label="click a corner: a handle"),
            Drag("handle", "z", 0.12, seconds=1.2, label="lift it"),
            Drag("handle", "x", -0.08, seconds=1.0, label="carry it"),
            Wait(seconds=0.6, mark="lifted"),
            Press(text="Release all", label="let go"),
            Wait(seconds=1.2),
        ),
        expect=(Rose(Patch("towel", near=Corner("towel", -1, -1, 0.03)), 0.06, at="lifted"),),
    ),
    Take(
        TOWEL,
        name="move_together",
        camera=TOWEL_VIEW,
        warmup=1.0,
        beats=(
            Wait(seconds=0.5),
            # The edge nearest the camera: the far corner of the other edge is behind the
            # right arm from here, and a click there took the arm.
            Click(Corner("towel", 1, -1, 0.03), label="one corner"),
            Click(Corner("towel", 1, 1, 0.03), label="and the next"),
            Toggle(text="Move together", label="Move together"),
            Drag("handle", "z", 0.10, seconds=1.2, label="one drag lifts both"),
            Drag("handle", "x", -0.10, seconds=1.0, label="and carries the sheet flat"),
            Wait(seconds=0.6, mark="lifted"),
            Press(text="Release all"),
            Wait(seconds=1.2),
        ),
        expect=(
            Rose(Patch("towel", near=Corner("towel", 1, -1, 0.03)), 0.05, at="lifted"),
            Rose(Patch("towel", near=Corner("towel", 1, 1, 0.03)), 0.05, at="lifted"),
        ),
    ),
    handle(
        "examples/01_cloth/02_engines.py",
        "towel",
        Corner("towel", 1, -1, 0.03),
        camera=Orbit(At("towel"), distance=0.75, azimuth=-30.0, elevation=35.0),
    ),
    Take(
        "examples/01_cloth/04_your_own_cloth.py",
        camera=Camera(eye=(1.05, -0.38, 0.95), look_at=(0.60, -0.08, 0.83)),
        warmup=3.0,  # the page puts the sheet back flat over the ball, and it drapes for 1 to 3 s
        beats=(
            Wait(seconds=0.5, label="the Panda's hand over the sheet's hem"),
            Drag("right", "z", to=(0.600, -0.095, 0.860), seconds=1.0, label="drag the gizmo down"),
            Drag("right", "z", to=(0.600, -0.095, 0.800), seconds=0.8, label="and onto the hem"),
            Wait(seconds=0.3),
            # Not Close: shut, the pads squeezed the hem out and it slid 37 mm down the hand.
            # 0.95 of the slider leaves 5 mm, the sheet's contact thickness.
            Slider(text="Grip amount", value=0.95, seconds=1.0, within=RIGHT, label="Grip amount to 0.95"),
            Wait(seconds=1.0, mark="closed"),
            Drag("right", "z", to=(0.600, -0.095, 0.900), seconds=1.2, label="drag it up 100 mm"),
            Wait(seconds=2.0, mark="lifted", label="held by friction"),
        ),
        # Measured 94.6 mm.
        expect=(Rose(Patch("sheet", near=(0.600, -0.095, 0.800)), 0.06, at="lifted"),),
    ),
    Take(
        "examples/02_soft/00_sponge.py",
        needs=("superdex",),
        # The sponge under the Panda's tool-down hand, from the bench's open side.
        camera=Camera(eye=(0.80, -0.50, 1.02), look_at=(0.457, -0.115, 0.80)),
        warmup=1.0,  # the foam settles on the bench before the first frame
        beats=(
            Wait(seconds=0.5, label="the hand, open, over the sponge"),
            # The tool centre 20 mm over the bench: the pads span 11-27 mm of the 35 mm foam.
            Drag("right", "z", to=Bench(-0.143, -0.115, 0.020), seconds=1.6, label="drag the gizmo down onto it"),
            Wait(seconds=0.3),
            Press(text="Close", within=RIGHT, label="Close: the pads squeeze the foam"),
            Wait(seconds=1.0, mark="closed", label="friction holds it"),
            Drag("right", "z", 0.10, seconds=1.4, label="drag it up 100 mm"),
            Wait(seconds=1.5, mark="lifted", label="held"),
            Press(text="Open", within=RIGHT, label="Open: it drops"),
            Wait(seconds=1.5),
        ),
        # tests/test_example_grasps_soft.py measured 98.7 mm of rise and 1.7 mm of slip.
        expect=(Rose("sponge", 0.07, at="lifted"), Held("sponge", since="closed", at="lifted", tolerance=0.01)),
    ),
    Take(
        "examples/02_soft/00_sponge.py",
        name="press_sponge",
        needs=("superdex",),
        camera=Camera(eye=(0.80, -0.50, 1.02), look_at=(0.457, -0.115, 0.80)),
        warmup=1.0,
        beats=(
            Wait(seconds=0.5, label="the hand over the sponge"),
            # Open, the pads are 80 mm apart and pass either side of the 50 mm foam:
            # measured, the tool centre went 5 mm under its top and touched nothing.
            Slider(text="Grip amount", value=0.6, seconds=0.8, within=RIGHT, label="jaws in to 33 mm, inside its 50"),
            # The fingertips touch with the tool centre 40 mm over the bench; 12 mm on.
            Drag("right", "z", to=Bench(-0.143, -0.115, 0.028), seconds=1.6, label="down into the foam"),
            Wait(seconds=1.0, mark="pressed", label="it dents under the fingertips"),
            Drag("right", "z", 0.10, seconds=1.2, label="up"),
            Wait(seconds=1.2, label="and it springs back"),
        ),
        # The deepest point of the foam's top under the hand: its middle, between the two
        # fingertips, bulges up 2.3 mm, and the sponge's height grows with it. Measured
        # headless, the top layer within 20 mm sank 3.0 mm on average, all of it back.
        expect=(Sank(Patch("sponge", near=Top("sponge"), radius=0.025, skin=0.004, lowest=True), 0.002),),
    ),
    handle(
        "examples/02_soft/01_tet_mesh.py",
        "gasket",
        Highest("gasket"),
        camera=Orbit(At("gasket"), distance=0.5, azimuth=-20.0, elevation=35.0),
        radius=0.015,
        # Gently: lifted 100 mm by one patch and carried 60, the ring came down a crumpled
        # blob and stayed one.
        lift=0.05,
        across=("y", -0.03),
        pending="the O-ring folds over itself under a handle and stays folded, at 50 mm as at 100: "
        "a question for the soft body, not the take",
    ),
    handle(
        "examples/02_soft/02_your_own_soft.py",
        "block",
        Top("block"),
        camera=Orbit(At("block", dz=0.05), distance=0.6, azimuth=-20.0, elevation=30.0),
    ),
    handle(
        "examples/04_extending/01_new_kind.py",
        "chain",
        Highest("chain"),
        camera=Orbit(At("chain"), distance=0.7, azimuth=-15.0, elevation=30.0),
        warmup=3.0,  # the stack of loose links is still falling over at 1 s
        lift=0.12,
        radius=0.012,  # the one link the handle took: the links are loose, and the rest stay in the heap
    ),
    Take(
        "examples/09_teleop/01_force_feedback.py",
        argv=("--simulate",),
        cursor=False,  # a scripted leader arm drives the hands; no one is at the mouse
        camera=TELEOP_VIEW,
        warmup=1.0,
        beats=(Wait(seconds=9.0, label="a simulated leader meets the panel and the crate"),),
    ),
    grasp("examples/06_tasks/01_goals_and_rewards.py", "cube", camera=TASK_VIEW),
    grasp("examples/06_tasks/02_episodes.py", "cube", camera=TASK_VIEW),
    Take(
        "examples/03_materials/00_rope.py",
        camera=Orbit(Bench(-0.10, -0.08, 0.10), distance=0.9, azimuth=-30.0, elevation=30.0),
        warmup=1.0,
        layout=Layout(collapse=("Simulation", "Arm: left")),
        beats=(
            Wait(seconds=0.5, label="the right hand over the cable's middle"),
            # It settles about 3 mm over this, on the capsule.
            Drag("right", "z", to=(0.550, -0.011, 0.771), seconds=1.2, label="drag the gizmo down onto it"),
            Wait(seconds=0.3),
            Press(text="Close", within=RIGHT, label="Close"),
            Wait(seconds=1.0, mark="closed"),
            # Slowly: past 0.2 m/s a flick drops it. 120 mm over 1.8 s, eased, peaks at 0.1 m/s.
            Drag("right", "z", to=(0.550, -0.011, 0.891), seconds=1.8, label="drag it up, slowly"),
            # The loaded hand sits about 25 mm under its gizmo.
            Wait(seconds=2.0, mark="lifted", label="held by friction"),
        ),
        # The capsule in the jaws. Measured 92 mm.
        expect=(Rose(Patch("cable", near=(0.550, -0.011, 0.771)), 0.06, at="lifted"),),
    ),
    Take(
        "examples/03_materials/01_rope_memory.py",
        camera=Orbit(Bench(-0.04, -0.13, 0.05), distance=0.95, azimuth=-30.0, elevation=38.0, fov=50.0),
        warmup=1.0,
        budget=1_400_000,  # the whole arm in view: 1.2 MB held it only at quality 46
        layout=Layout(collapse=("Simulation", "Arm: left")),
        beats=(
            Wait(seconds=0.5, label="the right hand over a cable with memory, a post beside it"),
            # Capsule 13 lies at bench (-0.15, -0.05), 11 mm up; the tool centre closes 10 mm over its axis.
            Drag("right", "z", to=(0.450, -0.050, 0.771), seconds=1.2, label="drag the gizmo down onto the cable"),
            Press(text="Close", within=RIGHT, label="Close"),
            Wait(seconds=1.0, mark="closed"),
            Drag("right", "z", 0.03, seconds=0.8, label="lift it 30 mm"),
            Do(fn=_round_the_post, label="carry it round the post, turning the ring"),
            Wait(seconds=0.3, mark="wrapped"),
            Drag("right", "z", -0.03, seconds=0.8, label="lay it down"),
            Press(text="Open", within=RIGHT, label="Open"),
            Wait(seconds=0.5),
            Drag("right", "z", 0.09, seconds=1.0, label="lift away"),
            Wait(seconds=2.0, label="it stays wrapped round the post"),
        ),
        expect=(Check(_carried_round), Check(_kept_bend, needs=("cable",))),
    ),
    Take(
        "examples/03_materials/02_twist.py",
        camera=Orbit(Bench(0.02, 0.0, 0.17), distance=0.95, azimuth=12.0, elevation=34.0, fov=50.0),
        budget=1_500_000,  # two hands turning and two gizmos for 15 s
        warmup=1.0,
        layout=Layout(collapse=("Simulation",)),
        beats=(
            Wait(seconds=0.5, label="two cables hang, each hand level in front of a tip"),
            # Each arrow is the hand's own axis, turned 30 degrees out: 60 mm along it is the tip.
            Drag("right", "x", 0.06, seconds=1.0, label="drag the right hand onto its tip"),
            Drag("left", "x", 0.06, seconds=1.0, label="and the left onto the other"),
            Press(text="Close", within=RIGHT, label="Close"),
            Press(text="Close", within="Arm: left", label="Close"),
            Wait(seconds=0.8, mark="closed"),
            Do(fn=_turn_both, label="turn each hand a third of a turn about its cable"),
            Press(text="Open", within=RIGHT, label="Open: the stiff one unwinds"),
            Press(text="Open", within="Arm: left", label="Open: the soft one keeps it"),
            Drag("right", "y", -0.06, seconds=0.8, label="back the hands off"),
            Drag("left", "y", 0.06, seconds=0.8),
            Wait(seconds=1.0, label="the stiff reads 0, the soft keeps its turn"),
        ),
        expect=(Check(_twisted, needs=("soft_twist", "stiff_twist")),),
    ),
    Take(
        "examples/03_materials/03_knot.py",
        needs=("superdex",),
        budget=1_800_000,  # 24 s, both arms in view: 1.4 MB at quality 46
        camera=Orbit(Bench(0.0, 0.0, 0.08), distance=1.3, azimuth=-15.0, elevation=30.0, fov=40.0),
        warmup=1.0,  # the knot settles
        layout=Layout(collapse=("Simulation",)),
        beats=(
            Wait(seconds=0.5, label="a loose overhand knot, a hand over each tail"),
            # tests/test_knot_pull's gesture: each tool centre 10 mm over its tail's axis, the sliders to
            # 0.90, each hand up 120 mm, then each out 85 mm, one at a time at about 40 mm/s.
            Drag("right", "z", to=Bench(0.03, -0.1247, 0.0167), seconds=1.4, label="the right hand down onto its tail"),
            Drag("left", "z", to=Bench(0.03, 0.1247, 0.0167), seconds=1.4, label="the left onto the other"),
            Slider(text="Grip amount", value=0.90, seconds=0.8, within=RIGHT, label="Grip amount 0.90"),
            Slider(text="Grip amount", value=0.90, seconds=0.8, within="Arm: left", label="and on the left"),
            Wait(seconds=0.5),
            Drag("right", "z", 0.12, seconds=3.0, label="up 120 mm"),
            Drag("left", "z", 0.12, seconds=3.0),
            Drag("right", "y", -0.085, seconds=2.2, label="and out: the knot draws up"),
            Drag("left", "y", 0.085, seconds=2.2),
            Wait(seconds=2.0, label="three crossings still: a knot"),
        ),
        expect=(Check(_knot_tight, needs=("rope",)),),
    ),
    Take(
        "examples/03_materials/04_band.py",
        needs=("superdex",),
        budget=1_300_000,  # 17 s with the whole arm in view: 966 KB at quality 46 at the default
        camera=Orbit(Bench(0.03, -0.20, 0.17), distance=1.0, azimuth=-62.0, elevation=28.0, fov=40.0),
        warmup=1.5,  # the band settles on the upper peg
        layout=Layout(collapse=("Simulation", "Arm: left")),
        beats=(
            Wait(seconds=0.5, label="a band on the upper peg, the right hand in front of it"),
            Press(text="Close", within=RIGHT, label="Close: the shut jaws are the hook"),
            Wait(seconds=0.6),
            # tests/test_band_on_pegs's path, at about its 50 mm/s: through the loop, 5 mm short of the
            # band as hung, down to 30 mm under the lower peg, in under it to 18 mm off the board's face
            # (the fingertips 1 mm off it), up to 15 mm under the peg, and back out 60 mm short of the band.
            Drag("right", "x", to=Laid("band", dx=-0.005), seconds=1.2, label="into the loop"),
            Drag("right", "z", to=Laid("lower", dz=-0.03), seconds=2.2, label="down past the lower peg"),
            Drag("right", "x", to=Laid("board", face=(-1, 0, 0), dx=-0.018), seconds=1.2, label="in under it"),
            Drag("right", "z", to=Laid("lower", dz=-0.015), seconds=0.5, label="up"),
            Drag("right", "x", to=Laid("band", dx=-0.06), seconds=2.3, label="back out"),
            # The gizmo's arrows and arcs reach 0.14 m along +x from the hand: from 25 mm off they sat over
            # the lower peg and hid the band on it, so the hand goes on back until they end short of its tip.
            Drag("right", "x", to=Bench(x=-0.13), seconds=1.2, label="and clear of the pegs"),
            Wait(seconds=2.0, label="stretched round both pegs"),
        ),
        expect=(Check(_band_on_both, needs=("band",)),),
    ),
    # 60 s of scene: four strokes at 3x read as a wipe and fit 2 MB; at 2x the clip was 2.6 MB at quality 46.
    routine(
        "examples/08_demos/wipe_the_bar/wipe_the_bar.py",
        speed=3.0,
        success="the bar is wiped:",
        done=_wiped,
        budget=2_000_000,
        needs=("superdex",),
    ),
    routine(
        "examples/08_demos/route_the_cable/route_the_cable.py", speed=2.0, success="the cable is routed:", done=_routed
    ),
    routine(
        "examples/08_demos/fold_the_towel/fold_the_towel.py", speed=2.0, success="the towel is folded:", done=_folded
    ),
)

# =============================================================================
# The particle kinds, worked by the robot's hand
# =============================================================================
#
# Each take is the gesture its example's medium test drives through the arm panel,
# played on the page at the test's speeds: the right arm's gizmo dragged or turned,
# Close pressed. Nothing is clicked onto a handle, so what moves the rice, the
# honey and the dough is the hand.

TOOL_DOWN = (0.0, 1.0, 0.0, 0.0)
"""The YAM's tool pointing down, its jaws closing along world y (xyzw). Its home points the tool along +x."""


def _turn(arm: str, quat, seconds: float, label: str) -> Do:
    """The arm's gizmo turned to ``quat`` over ``seconds`` of clip, filmed, and the hand turning after it.

    What a drag of the gizmo's rotation ring writes, written through the panel's
    own ``aim``: no beat drags a ring, so the clip shows the gizmo turning with
    the pointer off it. To an orientation, eased, where :class:`~.beats.Turn`
    turns by an angle about an axis at an even rate.
    """

    def turn(film):
        import numpy as np

        from robosimstudio.core.math import quat_slerp

        from .page import ease

        panel = film.session.arms[arm]
        a = np.asarray(panel.cursor_quat, dtype=np.float64)
        b = np.asarray(quat, dtype=np.float64)
        half = math.acos(min(1.0, abs(float(a @ b))))
        frames = max(1, round(seconds * film.take.fps))
        for k in range(1, frames + 1):
            panel.aim(panel.cursor_pos, quat_slerp(a, b, ease(k / frames)), moving=k < frames)
            film.advance()
        return {"turned_deg": round(math.degrees(2.0 * half), 1)}

    return Do(fn=turn, label=label)


def _dolly(camera: Orbit | Camera, seconds: float, label: str) -> Do:
    """The view moved to ``camera`` over ``seconds`` of clip, filmed: what a person's scroll and orbit do.

    For a take whose first drag needs the hand in view at its home, far over
    what it then works on: the page opens wide and closes in once the hand is down.
    The beat's ``fn`` is a partial over ``camera``, so a test can read where it ends.
    """
    return Do(fn=partial(_dolly_to, camera, seconds), label=label)


def _dolly_to(camera: Orbit | Camera, seconds: float, film) -> dict:
    from .page import ease

    eye1, at1, fov1 = camera.resolve(film)
    now = film.page.js("camera")
    eye0, at0, fov0 = now["eye"], now["at"], float(now["fov"])
    frames = max(1, round(seconds * film.take.fps))
    for k in range(1, frames + 1):
        t = ease(k / frames)
        film.aim(
            [a + (b - a) * t for a, b in zip(eye0, eye1, strict=True)],
            [a + (b - a) * t for a, b in zip(at0, at1, strict=True)],
            fov0 + (fov1 - fov0) * t,
        )
        film.advance()
    return {"camera_to": {"eye": [round(v, 3) for v in eye1], "at": [round(v, 3) for v in at1]}}


def _note(key: str, read, label: str) -> Do:
    """Read something off the scene or the page between two frames and keep it in the timeline under ``key``."""
    return Do(fn=lambda film: {key: read(film)}, label=label)


def _noted(film, key: str):
    """What a :func:`_note` kept."""
    for span in film.spans:
        if key in span:
            return span[key]
    raise KeyError(key)


def _jaws(film) -> float:
    """The right hand's jaw opening [mm]."""
    return round(1000.0 * float(film.sim.gripper("right").width(film.sim.state)), 1)


def _jaws_on(width_mm: float, within_mm: float):
    """The jaws closed on a tool's faces: the opening noted after Close is the tool's own width."""

    def check(film):
        width = _noted(film, "jaws_mm")
        return abs(width - width_mm) <= within_mm, (
            f"the jaws closed to {width:.1f} mm on its {width_mm:.0f} mm faces (need within {within_mm:.0f})"
        )

    return check


def _carried(name: str, along_y: float, *, upright: float, at: str = "end"):
    """``name`` went at least ``along_y`` [m] toward +y from Close to mark ``at``, and kept its height within ``upright``."""

    def check(film):
        a, b = film.marks["closed"]["things"][name], film.marks[at]["things"][name]
        dy, dz = b[1] - a[1], b[2] - a[2]
        return dy >= along_y and abs(dz) <= upright, (
            f"the {name} went {dy * 1000:.0f} mm with the hand (need {along_y * 1000:.0f}) and "
            f"{dz * 1000:+.1f} mm in height (need within {upright * 1000:.0f})"
        )

    return check


def _tray(film) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """``03_materials/11_granular``'s tray inside as the example lays its walls, bench frame [m], 5 mm under the bench to the top.

    Between the side walls' inner faces, from their open ends to the back wall's face: 280 by 200 mm
    from bench (-0.14, -0.22), 30 mm tall.
    """
    left, right = laid(film, "wall_left", (1, -1, 0)), laid(film, "wall_right", (-1, 0, 0))
    back = laid(film, "wall_back", (0, -1, 1))
    return (left[0], left[1], -0.005), (right[0], back[1], back[2])


def _rice_in_tray(film) -> tuple[bool, str]:
    """``03_materials/11_granular``'s readout: the share of the rice inside the tray, boxed as the example boxes it."""
    import numpy as np

    from robosimstudio_materials.mpm.measures import fill_fraction

    lo, hi = (np.asarray(film.bench(p)) for p in _tray(film))
    fill = fill_fraction(np.asarray(film.sim["rice"].particle_q, dtype=np.float64), lo, hi)
    return fill >= 0.5, f"{100.0 * fill:.1f}% of the rice is in the tray (tests/test_sweep_rice asks half)"


def _dough_height(film) -> float:
    """``03_materials/13_dough``'s readout: the puck's height as a share of the height it settled at [%]."""
    from robosimstudio_materials.mpm.measures import heap_height

    lump = film.sim["dough"]
    rest, floor = lump.entity.rest["particle_q"][lump.particles], lump.entity.floor_z
    return round(100.0 * heap_height(lump.particle_q, floor) / heap_height(rest, floor), 1)


def _dent_kept(film) -> tuple[bool, str]:
    """The hand took the puck down, and most of the dent stayed once it lifted (tests/test_mpm_kinds asks 0.6)."""
    pressed, after = _noted(film, "height_pct"), _dough_height(film)
    kept = (100.0 - after) / max(1e-6, 100.0 - pressed)
    return pressed <= 90.0 and kept >= 0.6, (
        f"height {pressed:.1f}% of settled under the hand (need 90 or less) and {after:.1f}% after it lifted: "
        f"{kept:.2f} of the dent kept (need 0.6)"
    )


ROBOT_PARTICLES = (
    Take(
        "examples/03_materials/11_granular.py",
        # From the front and the tray's side: the heap comes toward the lens into the tray, and the arm stays behind it.
        camera=Orbit(Bench(0.0, -0.29, 0.08), distance=1.1, azimuth=40.0, elevation=35.0, fov=42.0),
        warmup=1.0,  # the heap settles before the first frame
        layout=Layout(collapse=("Simulation", "Arm: left")),
        budget=1_800_000,  # nine gestures and a sweep: 18 s of clip
        beats=(
            Wait(seconds=0.5, label="the right hand over the heap, pointing forward"),
            DragTo("right", Laid("pusher", dz=0.10), seconds_per_m=4.5, label="over the pusher"),
            _turn("right", TOOL_DOWN, 1.0, "turn the gizmo: the hand points down"),
            Drag("right", "z", to=At("pusher"), seconds=1.6, label="down onto the pusher"),
            Press(text="Close", within=RIGHT, label="Close: the jaws take its 20 mm faces"),
            Wait(seconds=1.0, mark="closed"),
            _note("jaws_mm", _jaws, "the jaw opening"),
            # tests/test_sweep_rice walks the hand at 0.15 m/s; eased over 3 s, these 290 mm peak at 0.145. It
            # stops the tool centre 140 mm short of the back wall's face, the pusher's own face 130 (its SWEEP_END).
            Drag(
                "right",
                "y",
                to=Laid("wall_back", face=(0, -1, 0), dy=-0.14),
                seconds=3.0,
                label="sweep into the tray at a walking pace",
            ),
            Wait(seconds=1.5, mark="swept", label="the panel reads the share in the tray"),
            Press(text="Open", within=RIGHT, label="Open"),
            Drag("right", "z", 0.10, seconds=1.0, label="lift away"),
            Drag("right", "y", -0.15, seconds=1.0, label="and back out of the tray"),
            # Under the push the rice packs into a strip a few millimetres high along the pusher's face, the
            # face this view looks at, and a tenth of it runs round the pusher's ends (sweep_where.py, GPU 0).
            Wait(seconds=1.2, label="the strip along the pusher's face, and what ran round its ends"),
        ),
        expect=(
            Check(_jaws_on(20.0, 4.0)),
            Check(_carried("pusher", 0.20, upright=0.006, at="swept"), needs=("pusher",)),
            Check(_rice_in_tray),
        ),
    ),
    Take(
        "examples/03_materials/13_dough.py",
        # From the front and the arm's outside: the hand at its home, 330 mm over the puck, and the puck both in view.
        # The particles are drawn unlit, so a dent shows only as a notch in the outline, seen along the blade.
        camera=Orbit(Bench(0.0, -0.31, 0.14), distance=1.0, azimuth=-35.0, elevation=28.0, fov=40.0),
        warmup=1.0,  # the puck settles before the first frame
        layout=Layout(collapse=("Simulation", "Arm: left")),
        beats=(
            Wait(seconds=0.5, label="the right hand, open, over the puck"),
            Press(text="Close", within=RIGHT, label="Close: the fingers make one blade"),
            Drag("right", "z", to=Bench(0.0, -0.31, 0.08), seconds=1.2, label="drag the gizmo down to the puck"),
            # 15 mm under the settled top, as tests/test_mpm_kinds presses it.
            Drag("right", "z", to=Top("dough", dz=-0.015), seconds=1.5, label="and 15 mm into it"),
            Wait(seconds=0.5, mark="pressed"),
            _note("height_pct", _dough_height, "the puck's height under the hand"),
            Drag("right", "z", 0.12, seconds=1.4, label="lift away"),
            Drag("right", "y", -0.12, seconds=1.0, label="and aside"),
            # Round to the front, along the blade, at the same distance: the dent's notch shows in the outline.
            _dolly(
                Orbit(Bench(0.0, -0.31, 0.10), distance=1.0, azimuth=-5.0, elevation=28.0, fov=36.0),
                1.5,
                "round to the front",
            ),
            Wait(seconds=1.5, label="the dent stays"),
        ),
        # The whole puck's top comes down 2 mm or more under the blade: 3.0 mm on Newton 1.6 and 2.6 mm on
        # 1.7 (its MPM converges to a slightly stiffer press at the same 150 iterations; the drive test reads
        # 0.779 of the height under the hand where 1.6 read 0.761), so the 3 mm this asked was 1.6's own reading.
        expect=(Check(_dent_kept), Squashed("dough", 0.002, at="pressed")),
    ),
    Take(
        "examples/03_materials/12_liquid.py",
        camera=Orbit(Bench(0.09, -0.24, 0.13), distance=1.1, azimuth=-30.0, elevation=30.0, fov=40.0),
        warmup=1.0,
        layout=Layout(collapse=("Simulation", "Arm: left")),
        beats=(
            Wait(seconds=0.5, label="the hand, open, over the cup"),
            Drag(
                "right",
                "z",
                to=At("cup", dz=0.055),
                seconds=1.4,
                label="drag the gizmo down: the jaws straddle the cup",
            ),
            Wait(seconds=0.3),
            Press(text="Close", within=RIGHT, label="Close"),
            Wait(seconds=1.0, mark="closed", label="the pads on its walls"),
            Drag("right", "z", 0.095, seconds=1.4, label="drag it up"),
            Drag("right", "y", 0.14, seconds=1.6, label="and over the bowl"),
            Wait(seconds=0.5, mark="carried"),
            Turn("right", "z", 40.0, seconds=1.0, label="turn the gizmo 40 degrees about the vertical"),
            Turn("right", "x", -140.0, seconds=2.0, label="and 140 about x: the wrist tips the cup"),
            Wait(seconds=1.6, mark="poured", label="the honey pours into the bowl"),
            Turn("right", "x", 140.0, seconds=2.0, label="back upright"),
            Wait(seconds=0.6),
        ),
        expect=(
            Rose("cup", 0.05, since="closed", at="carried"),
            Held("cup", since="closed", at="carried"),
            Check(_poured),
        ),
        budget=1_300_000,  # 19 s of a moving arm: 1,008 KB at the ladder's last quality, over the default 900
    ),
)

ROBOT_BAGS: tuple[Take, ...] = (
    # The right hand pinching the top layer's lip by contact and opening the mouth, nothing holding the bag
    # down; tests/test_open_a_bag.py drives the same path through the panel. Seen from the mouth's side and
    # a little in front, so the opening shows under the hand: the gizmo's arrows and arcs point +x, +y, +z.
    Take(
        "examples/03_materials/06_bag.py",
        needs=("superdex",),
        camera=Orbit(Bench(0.0, -0.10, 0.04), distance=0.95, azimuth=-80.0, elevation=25.0, fov=38.0),
        warmup=0.5,
        layout=Layout(collapse=("Simulation", "Arm: left")),
        beats=(
            Wait(seconds=0.4, label="a canvas bag, its lip turned up toward the right hand"),
            Drag(
                "right", "z", to=Bench(z=0.025), seconds=1.2, label="drag the hand down: its open jaws straddle the lip"
            ),
            Wait(seconds=0.3),
            Press(text="Close", within=RIGHT, label="Close: the pads pinch the lip"),
            Wait(seconds=1.0, mark="closed"),
            Drag("right", "z", to=Bench(z=0.08), seconds=1.2, label="drag it up to 80 mm"),
            Drag("right", "y", to=Bench(y=-0.065), seconds=1.5, label="and back over the bag: the mouth opens"),
            Wait(seconds=2.0, mark="open", label="nothing holds the bag down; its layers stay apart"),
        ),
        expect=(Check(_bag_open, needs=("bag",)),),
    ),
)


# The brush: the hand closes on its handle, presses the bristles to the bench and scrubs the square in
# three strokes, the gesture tests/test_scrub_bench drives through the same arm panel.
BRUSH_PAD_TIP = 0.0165
"""How far the YAM's pad tips reach below its tool centre [m] (tests/test_scrub_bench's ``PAD_TIP``)."""
BRUSH_CLOSE_OVER_SOLE = 0.008
"""Where the pad tips stop when the hand closes: this far over the brush's sole [m], as the test closes."""
BRUSH_PRESS = 0.004
"""How far below the closing height the hand scrubs [m], as the test presses."""
BRUSH_STROKE = 0.15
"""Each stroke along the bench's y [m]; three of them, 12 mm apart in x, cover the 100 mm square."""
BRUSH_LANE = 0.012


def _over_the_sole(film) -> tuple[float, float, float]:
    """Where the tool centre closes on the handle: over its middle, the pad tips 8 mm over the sole."""
    brush, handle = film.sim["brush"].entity, film.position("handle")
    z = float(brush.floor_z) + float(brush.spec.sole_top) + BRUSH_CLOSE_OVER_SOLE + BRUSH_PAD_TIP
    return (handle[0], handle[1], z)


def _scrubbed(film) -> tuple[bool, str]:
    """``03_materials/07_brush``'s readout: the share of the square's 5 mm cells a bristle tip touched while down.

    The square is the example's ``patch`` region as the scene built it: bench (-25, -270) to (75, -170) mm.
    """
    x0, y0, x1, y1 = (float(v) for v in film.sim.handles.regions["patch"].xy_ranges)
    lo, hi = film.bench((x0, y0, 0.0))[:2], film.bench((x1, y1, 0.0))[:2]
    share = float(film.sim["brush"].entity.scrubbed(lo, hi))
    return share >= 0.5, f"{100.0 * share:.1f}% of the square scrubbed (tests/test_scrub_bench asks half)"


ROBOT_BRUSH = (
    Take(
        "examples/03_materials/07_brush.py",
        # From in front of the bench, 1.15 m off, the brush end on: its bristles show under the fingers,
        # the strokes run left to right across the drawn square, and no link is near enough to loom.
        camera=Orbit(Bench(0.03, -0.25, 0.06), distance=1.15, azimuth=-20.0, elevation=32.0, fov=30.0),
        warmup=1.0,  # the field settles under its sole before the first frame
        layout=Layout(collapse=("Simulation", "Arm: left")),
        budget=1_500_000,
        beats=(
            Wait(seconds=0.5, label="the right hand, open, over the brush's handle"),
            Drag("right", "z", to=_over_the_sole, seconds=1.4, label="drag the gizmo down onto the handle"),
            Wait(seconds=0.3),
            Press(text="Close", within=RIGHT, label="Close: the jaws take the handle's 22 mm faces"),
            Wait(seconds=0.8, mark="closed"),
            _note("jaws_mm", _jaws, "the jaw opening"),
            Drag("right", "z", -BRUSH_PRESS, seconds=0.8, label="press the bristles 4 mm onto the bench"),
            Drag("right", "x", -BRUSH_LANE, seconds=0.6, label="the first lane"),
            # tests/test_scrub_bench walks the cursor at 0.10 m/s; eased over 2 s, these 150 mm peak at 0.11.
            Drag("right", "y", BRUSH_STROKE, seconds=2.0, label="scrub out across the square"),
            Drag("right", "x", BRUSH_LANE, seconds=0.6, label="over 12 mm"),
            Drag("right", "y", -BRUSH_STROKE, seconds=2.0, label="and back"),
            Drag("right", "x", BRUSH_LANE, seconds=0.6, label="over 12 mm"),
            Drag("right", "y", BRUSH_STROKE, seconds=2.0, label="and out again"),
            Wait(seconds=0.8, mark="scrubbed", label="the panel reads the share scrubbed"),
            Drag("right", "z", 0.04, seconds=1.0, label="lift: the brush comes up in the fingers"),
            Wait(seconds=1.0, mark="lifted", label="the bristles spring back"),
        ),
        expect=(
            Check(_jaws_on(22.0, 3.0)),
            Check(_scrubbed),
            Held("handle", since="closed", at="lifted", tolerance=0.005),
            Rose("handle", 0.03, at="lifted", since="scrubbed"),
        ),
    ),
)


# The suction hand: down onto the plate, the right arm's Vacuum box ticked, lift, carry, down, unticked.
# The gestures are tests/test_suction_pick's, through the same arm panel, with the gizmo put 3 mm into the
# plate rather than 4 and a lowering aimed to stop the held plate's lowest corner 2 mm over the bench: it
# hangs 1.7 degrees off level, so that corner stops 2.2 mm over and its middle 4.5 mm, and it drops that
# far when the box is unticked. What holds the plate is the vacuum: the Seal readout on the panel says
# sealed and the mass it holds.
INTO_PLATE = 0.003
"""How far under the plate's top the gizmo is put [m]. The cup stops on the plate 3 mm short of the gizmo,
inside the panel's 4 mm tolerance, so the panel puts the gizmo back on the hand (its EE readout says on
target) and the face rests on the plate: the Seal readout's gap is 0.0 mm before the box is ticked and once
it seals, inside the seal's 2 mm."""


def _plate_half(film) -> float:
    """Half the plate's height along world z as it is turned now [m]: 3 mm for ``04_extending/02_suction``'s, flat."""
    return film.height("plate") / 2.0


def _square(film) -> tuple[float, float, float, float]:
    """``04_extending/02_suction``'s square as the scene built its region, bench frame [m]: x0, y0, x1, y1.

    (-110, -220) to (110, 0) mm: 220 mm across, its centre 200 mm from the plate's.
    """
    return tuple(float(v) for v in film.sim.handles.regions["square"].xy_ranges)


def _seal(film) -> str:
    """The right arm's **Seal** readout, as the page shows it."""
    return film.page.text("Seal", RIGHT)


def _plate_over_bench(clearance: float):
    """Where the gizmo goes for the held plate's lowest corner to stop ``clearance`` [m] over the bench."""

    def there(film) -> tuple[float, float, float]:
        cursor = film.where(film.gizmo("right"))
        underside = film.position("plate")[2] - _plate_half(film)
        bench = film.bench((0.0, 0.0, 0.0))[2]
        return (cursor[0], cursor[1], cursor[2] - (underside - bench - clearance))

    return there


def _sealed(film) -> tuple[bool, str]:
    """The Seal readout said sealed once the box was ticked, holding the plate's 0.300 kg."""
    readout = str(_noted(film, "seal"))
    return readout.startswith("sealed") and "0.300 kg" in readout, f"the Seal readout said {readout!r}"


def _landed(film) -> tuple[bool, str]:
    """The cup let go and the plate lies flat on the bench inside the square."""
    import math as m

    import numpy as np

    plate = np.asarray(film.position("plate"))
    x0, y0, x1, y1 = _square(film)
    lo, hi = np.asarray(film.bench((x0, y0, 0.0))[:2]), np.asarray(film.bench((x1, y1, 0.0))[:2])
    inside = bool(np.all(plate[:2] >= lo) and np.all(plate[:2] <= hi))
    top = 1000.0 * (plate[2] + _plate_half(film) - film.bench((0.0, 0.0, 0.0))[2])
    body = int(film.sim.scene.by_name["plate"].body)
    w = abs(float(film.sim.state.body_q.numpy()[body][6]))
    tilt = m.degrees(2.0 * m.acos(min(1.0, w)))
    holding = bool(film.sim.suction("right").holding)
    ok = inside and top < 10.0 and tilt < 3.0 and not holding
    return ok, (
        f"the plate at ({plate[0]:.3f}, {plate[1]:.3f}) {'inside' if inside else 'outside'} the square, its top "
        f"{top:.1f} mm over the bench (need under 10), tilted {tilt:.2f} deg (need under 3), "
        f"{'still held' if holding else 'let go'}"
    )


ROBOT_SUCTION = (
    Take(
        "examples/04_extending/02_suction.py",
        # From in front of the bench, 1.2 m off: the plate, the drawn square and the carry between them.
        camera=Orbit(Bench(0.0, -0.18, 0.06), distance=1.2, azimuth=25.0, elevation=32.0, fov=36.0),
        warmup=1.0,  # the plate settles flat before the first frame
        layout=Layout(collapse=("Simulation", "Arm: left")),
        budget=1_200_000,
        beats=(
            Wait(seconds=0.5, label="the cup over the plate; Seal reads the gap to it"),
            Drag(
                "right", "z", to=Top("plate", dz=-INTO_PLATE), seconds=1.6, label="drag the gizmo down onto the plate"
            ),
            Wait(seconds=0.6, label="Seal: not sealed, the face on the plate"),
            Toggle(text="Vacuum", within=RIGHT, label="tick Vacuum"),
            Wait(seconds=0.8, mark="sealed", label="Seal: sealed, holding 0.3 kg"),
            _note("seal", _seal, "the Seal readout"),
            Drag("right", "z", 0.08, seconds=1.2, label="lift: the vacuum holds the plate"),
            Wait(seconds=0.5, mark="lifted"),
            Drag("right", "y", 0.20, seconds=2.0, label="carry it over the square"),
            Wait(seconds=0.5, mark="carried"),
            Drag("right", "z", to=_plate_over_bench(0.002), seconds=1.2, label="lower it to 2 mm over the bench"),
            Wait(seconds=0.5, mark="lowered"),
            Toggle(text="Vacuum", within=RIGHT, label="untick Vacuum: it lets go"),
            Wait(seconds=0.6),
            Drag("right", "z", 0.06, seconds=1.0, label="and the hand lifts away"),
            Wait(seconds=1.0, mark="placed", label="the plate rests in the square"),
        ),
        expect=(
            Check(_sealed),
            Rose("plate", 0.06, at="lifted"),
            Held("plate", since="sealed", at="lowered", tolerance=0.003),
            Check(_landed, needs=("plate",)),
        ),
    ),
)

# The tape: the right hand takes the label's pull tab by contact and peels the label off the box, the
# staircase tests/test_peel_a_label drives through the same arm panel: up 30 mm, then seven times back
# 20 mm toward the right arm and up 10 mm.


def _tab(film) -> tuple[float, float, float]:
    """The label's pull tab's middle where the scene built it, bench frame [m].

    It stands on the label's fold, half its 24 mm over the label's corner, which the example lays
    0.5 mm over the box's top: (0, -40, 62.5) mm in ``03_materials/08_tape``.
    """
    built, origin = film.sim["label"].entity.tab_q0[:3], film.bench((0.0, 0.0, 0.0))
    return tuple(float(b) - float(o) for b, o in zip(built, origin, strict=True))


def _over_the_tab(film) -> tuple[float, float, float]:
    """Where the tool centre closes: 4 mm over the tab's middle, the pads clear of the box top (the test's CLOSE)."""
    x, y, z = _tab(film)
    return film.bench((x, y, z + 0.004))


def _peel_stair(back: float, up: float):
    """A corner of the peel's staircase: ``back`` [m] toward the right arm from the tab, ``up`` [m] over the box top."""

    def there(film) -> tuple[float, float, float]:
        x, y, _ = _tab(film)
        return film.bench((x, y - back, laid(film, "box", (0, 0, 1))[2] + up))

    return there


def _peeled(film) -> tuple[bool, str]:
    """``03_materials/08_tape``'s readout: the share of the label's bonds that have let go."""
    label = film.sim["label"].entity
    share = float(label.peeled())
    front = 1000.0 * float(label.peel_front())
    return share >= 0.8, f"{100.0 * share:.1f}% of the bonds let go, the front {front:.0f} mm along (the test asks 80%)"


ROBOT_TAPE = (
    Take(
        "examples/03_materials/08_tape.py",
        # From in front of the bench, the label running left to right across the frame, so its lift shows.
        camera=Orbit(Bench(0.0, -0.10, 0.06), distance=0.9, azimuth=15.0, elevation=38.0, fov=36.0),
        warmup=0.5,
        # The page puts its Label folder first; expanding it last scrolls the sidebar back up to it.
        layout=Layout(collapse=("Simulation", "Arm: left"), expand=("Label",)),
        # 1.8 s of scene a second at 12 frames: the fourteen short drags and the waits for the hand
        # between them ran 25 s of clip at 15, and 1.47 MB at the bottom of the quality ladder.
        speed=1.8,
        fps=12,
        budget=1_200_000,
        beats=(
            Wait(seconds=0.4, label="a paper label on a box, its end in a blue pull tab under the right hand"),
            Drag("right", "z", to=_over_the_tab, seconds=1.2, label="drag the hand down: its jaws straddle the tab"),
            Wait(seconds=0.3),
            Press(text="Close", within=RIGHT, label="Close: the pads squeeze the tab"),
            Rest(seconds=0.4),  # off the button: left there, Close sat lit as hovered while the jaws shut
            Wait(seconds=0.4, mark="closed"),
            Drag("right", "z", to=_peel_stair(0.0, 0.03), seconds=1.0, label="drag it up 30 mm: the first rows let go"),
            *(
                beat
                for k in range(1, 8)
                for beat in (
                    Drag(
                        "right",
                        "y",
                        to=_peel_stair(0.02 * k, 0.03 + 0.01 * (k - 1)),
                        seconds=0.5,
                        approach=0.3,
                        label="back 20 mm toward the arm",
                    ),
                    Drag(
                        "right",
                        "z",
                        to=_peel_stair(0.02 * k, 0.03 + 0.01 * k),
                        seconds=0.3,
                        approach=0.3,
                        label="and up 10 mm",
                    ),
                )
            ),
            Wait(seconds=1.0, mark="peeled", label="the last bonds let go: the label is off the box, which stayed put"),
        ),
        expect=(Check(_peeled, needs=("label",)),),
    ),
)

# =============================================================================
# Contact: a fold carried by the pads, and a scrub read by its push
# =============================================================================
# Newton 1.7's VBD reports the force each rigid-soft contact puts on its shape, which the one-way
# coupler reads with ``contact_forces``. The fold is carried by contact alone; the scrub's readout
# is that force. tests/test_fold_paper_robot.py plays the fold's beats below headless, and
# tests/test_scrub_a_stain.py drives the scrub's path.

FOLD = "examples/03_materials/05_paper.py"

FOLD_OPEN = 0.79
"""The **Grip amount** the hands come in with: the pads 20 mm apart, the lower one clear of the bench."""
FOLD_GRIP = 0.986
"""The **Grip amount** that pinches the sheet: the pads stop 2.1 mm apart, the grip nubs 0.9 mm, on a
sheet whose contact thickness is 1.0 mm. **Close** (1.0) takes the pads to 0.8 mm and crushes it."""
FOLD_HANDS = 0.025
"""How far in from each corner the pads' middle closes [m]: the fingertips stay 70 mm apart."""
FOLD_GROW = 0.009
"""The carry's arc starts at the pinch and ends 18 mm further out than a circle about the fold line [m]:
on the circle the sheet bent one hinge column (10.6 mm) short of its middle and the edges missed by
21 to 25 mm; grown 6 to 12 mm, the bend lands on the middle column (``fold_b.py``)."""
FOLD_TURN = 88.0
"""How far each hand turns over the carry [deg], about the fold line: with the flap to nearly vertical, then
held. 90 here turned the hands 90.5 to 90.6 degrees, the servo's overshoot."""
FOLD_PRESS = 0.0035
"""How high the closed fingertip runs over the box top along the fold [m]: on the flap, 2 mm over the sheet
under it. At 3 mm one particle went 0.03 mm under the box's top."""
FOLD_SWEEP = (20.0, 30.0)
"""The sweeping hand's tool: tipped 20 degrees forward and 30 toward the sheet's middle, trailing the
stroke. Straight down, neither arm reaches the fold line at the box's height (13 to 48 mm short)."""


def _frame_quat(approach, close):
    """The tool's quaternion (x, y, z, w) with its +z along ``approach`` and its pads parting along ``close``.

    In the frame the arm panel's IK aims (the YAM's): the pads part along local y and the finger blades
    run back along local -x from them.
    """
    import numpy as np

    from robosimstudio.core.math import mat_to_quat

    z = np.asarray(approach, dtype=np.float64)
    z /= np.linalg.norm(z)
    y = np.asarray(close, dtype=np.float64)
    y = y - z * float(np.dot(y, z))
    y /= np.linalg.norm(y)
    return mat_to_quat(np.stack([np.cross(y, z), y, z], axis=1))


_FOLD_SIDE = {"right": -1.0, "left": 1.0}


def _fold_hand(arm: str, turn: float = 0.0):
    """A pinching hand: the tool along the near edge toward the sheet's middle, the pads parting vertically,
    the finger blades back toward the robot and off the box; then turned ``turn`` degrees about the fold line."""
    import numpy as np

    s = _FOLD_SIDE[arm]
    t = math.radians(turn)
    turn_y = np.array([[math.cos(t), 0.0, math.sin(t)], [0.0, 1.0, 0.0], [-math.sin(t), 0.0, math.cos(t)]])
    from robosimstudio.core.math import mat_to_quat, quat_to_mat

    return mat_to_quat(turn_y @ quat_to_mat(_frame_quat((0.0, -s, 0.0), (0.0, 0.0, s))))


def _fold_sweeper(arm: str):
    """The sweeping hand: fingertip down, tipped :data:`FOLD_SWEEP` forward and toward the sheet's middle."""
    import numpy as np

    s = _FOLD_SIDE[arm]
    forward, trail = (math.radians(v) for v in FOLD_SWEEP)
    tool = np.array([math.sin(forward) * math.cos(trail), -s * math.sin(trail), -math.cos(trail) * math.cos(forward)])
    return _frame_quat(tool, (0.0, -s * math.cos(trail), math.sin(trail)))


def _fold_layout(film) -> dict:
    """Where the fold works [world, m], off the specs ``03_materials/05_paper`` builds its sheet and box from.

    ``edge`` and ``far`` are the A4 sheet's two short edges along x, ``fold`` the line half way; ``pinch``
    the x the pads' middle closes at, 6 mm in from the edge so their 12 mm cover the overhang's first
    12 mm; ``hands`` each hand's y, :data:`FOLD_HANDS` in from its corner; ``box_edge`` and ``top`` the
    box's near face and top; ``line`` the fold line in the bench frame, where ``crease_angle`` reads it.
    """
    x, y, _ = laid(film, "paper")
    length, width = (float(v) for v in film.sim["paper"].spec.size)
    edge = film.bench((x, 0.0, 0.0))[0]
    return {
        "edge": edge,
        "far": edge + length,
        "fold": edge + length / 2,
        "pinch": edge + 0.006,
        "hands": {
            "right": film.bench((0.0, y + FOLD_HANDS, 0.0))[1],
            "left": film.bench((0.0, y + width - FOLD_HANDS, 0.0))[1],
        },
        "box_edge": film.bench(laid(film, "box", (-1.0, 0.0, 0.0)))[0],
        "top": film.bench(laid(film, "box", (0.0, 0.0, 1.0)))[2],
        "line": x + length / 2,
    }


def _fold_pads(film, arm: str):
    """The hand's two pad boxes (12 x 4 x 80 mm, not the slanted wing boxes beside them): centres and corners [world]."""
    import newton
    import numpy as np

    from robosimstudio.core.math import quat_to_mat

    sim = film.sim
    index = sim.scene.embodiment.arms[arm]
    body = sim.model.shape_body.numpy()
    kinds = sim.model.shape_type.numpy()
    scale = sim.model.shape_scale.numpy()
    body_q, xform = sim.state.body_q.numpy(), sim.model.shape_transform.numpy()
    centres, corners = [], []
    for side in index.finger_bodies:
        for s in np.flatnonzero(np.isin(body, side)):
            if int(kinds[s]) != int(newton.GeoType.BOX) or abs(float(scale[s][2]) - 0.04) > 1.0e-4:
                continue
            rot = quat_to_mat(body_q[body[s]][3:7])
            c = body_q[body[s]][:3] + rot @ xform[s][:3]
            r = rot @ quat_to_mat(xform[s][3:7])
            centres.append(c)
            corners += [c + r @ (np.array([i, j, k]) * scale[s][:3]) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
    return np.array(centres), np.array(corners)


def _fold_jaws(film, arm: str):
    """The middle between the two pads [world]: where the sheet goes."""
    return _fold_pads(film, arm)[0].mean(axis=0)


def _fold_tip(film, arm: str):
    """The closed pads' lowest corner [world]: what the sweep runs along the fold."""
    corners = _fold_pads(film, arm)[1]
    return corners[corners[:, 2].argmin()]


def _fold_targets(film, points: dict, quats: dict) -> dict:
    """Tool-centre targets that put each hand's pads' middle at its point, turned to its quaternion."""
    import numpy as np

    from robosimstudio.core.math import quat_to_mat

    out = {}
    for arm, point in points.items():
        tcp, q = film.session.arms[arm].ik.ee_pose(film.sim.state)
        local = quat_to_mat(q).T @ (_fold_jaws(film, arm) - tcp)
        out[arm] = np.asarray(point, dtype=np.float64) - quat_to_mat(quats[arm]) @ local
    return out


def _slerp(q0, q1, t: float):
    import numpy as np

    q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1.0 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def _go_both(film, targets: dict, quats: dict, speed: float, steps: int = 120, tolerance: float = 0.006):
    """:func:`_go` for several hands at once, turning each cursor as it walks: both drags end together."""
    import numpy as np

    panels = {arm: film.session.arms[arm] for arm in targets}
    starts = {arm: np.array(p.cursor_pos, dtype=np.float64) for arm, p in panels.items()}
    turns = {arm: np.array(p.cursor_quat, dtype=np.float64) for arm, p in panels.items()}
    stride = float(speed) / float(film.session.env.request.physics.control_hz)
    legs = max(1, *(int(np.ceil(np.linalg.norm(np.asarray(targets[a]) - starts[a]) / stride)) for a in targets))
    for k in range(1, legs + 1):
        for arm, panel in panels.items():
            point = starts[arm] + (np.asarray(targets[arm]) - starts[arm]) * (k / legs)
            panel.aim(point, _slerp(turns[arm], quats[arm], k / legs), moving=k < legs)
        yield
    for arm, panel in panels.items():
        panel.aim(np.asarray(targets[arm], dtype=np.float64), quats[arm], moving=False)
    for _ in range(steps):
        yield
        miss = [np.linalg.norm(p.ik.ee_pose(film.sim.state)[0] - np.asarray(targets[a])) for a, p in panels.items()]
        if max(miss) < tolerance:
            break


def _per_step(script, label: str) -> Do:
    """Run ``script(film)``, a generator that yields once a control step, from the page's per-step hook.

    What the tests do with :func:`tests.fixtures.grasp.go`, at the same resolution: the cursor is
    walked a control step at a time at the test's speeds, rather than jumped once a film frame, so
    the take is the test's drive and not an approximation of it. The drive moves the gizmo from the
    server, so the pointer is kept on it (:func:`_point_on_gizmo`), not left on the button or folder
    header it last pressed, which reads as a hand still on it.
    """

    def play(film):
        steps = script(film)
        state = {"done": False, "n": 0}
        memo: dict = {}

        def hook():
            if state["done"]:
                return
            try:
                next(steps)
                state["n"] += 1
            except StopIteration:
                state["done"] = True

        film.session.each_step(hook)
        while not state["done"]:
            _point_on_gizmo(film, memo)
            film.hold(1.0 / film.take.fps)
        return {"control_steps": state["n"]}

    return Do(fn=play, label=label)


def _point_on_gizmo(film, memo: dict, glide_px: float = 150.0) -> None:
    """Put the pointer on the arrow of the gizmo the drive is moving, the arrow most along its motion.

    Once a film frame, before it is stepped: the arm whose cursor moved most since the last frame
    is the one being driven, and the pointer sits 0.6 of the way along that gizmo's arrow, where a
    drag would press it, following it as it goes. It moves at most ``glide_px`` a frame, so it glides
    there from the sidebar and from arrow to arrow; an arrow is changed only when another lies at
    least 0.3 nearer the motion, so an arc does not flick it between two. No button is pressed: the
    drive, not the browser, moves the gizmo.
    """
    import numpy as np

    now = {arm: np.asarray(panel.cursor_pos, dtype=np.float64) for arm, panel in film.session.arms.items()}
    last, memo["at"] = memo.get("at", now), now
    moved = {arm: float(np.linalg.norm(now[arm] - last[arm])) for arm in now}
    arm = max(moved, key=moved.get)
    if moved[arm] > 1.0e-4:
        if memo.get("arm") != arm:
            memo.pop("axis", None)
        memo["arm"], memo["dir"] = arm, (now[arm] - last[arm]) / moved[arm]
    if "arm" not in memo:
        return
    axes = film.page.js("gizmoAxes", f"/studio/{memo['arm']}/cursor")
    if not axes:
        return
    along = [abs(float(np.dot(a["world_dir"], memo["dir"]))) for a in axes]
    best, held = int(np.argmax(along)), memo.get("axis")
    if held is None or held >= len(axes) or along[best] > along[held] + 0.3:
        memo["axis"] = held = best
    (ox, oy), (tx, ty) = axes[held]["origin"], axes[held]["tip"]
    x, y = ox + 0.6 * (tx - ox), oy + 0.6 * (ty - oy)
    left, top, width, height = film.page.js("canvasRect")
    if not (left <= x <= left + width and top <= y <= top + height):
        return
    mx, my = film.page.mouse
    gap = math.dist((mx, my), (x, y))
    k = min(1.0, glide_px / gap) if gap > 0.0 else 1.0
    film.page._move(mx + (x - mx) * k, my + (y - my) * k)


def _go(film, arm: str, target, speed: float, steps: int = 120, tolerance: float = 0.008, quat=None):
    """:func:`tests.fixtures.grasp.go` as a generator: walk the cursor there at ``speed``, then wait for the hand."""
    import numpy as np

    panel = film.session.arms[arm]
    # Through ``aim``, which moves the gizmo with the cursor: a cursor written behind the page's
    # back leaves the browser's gizmo where it was, and the page writes that back.
    quat = np.asarray(panel.cursor_quat if quat is None else quat, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    start = np.array(panel.cursor_pos, dtype=np.float64)
    stride = float(speed) / float(film.session.env.request.physics.control_hz)
    legs = max(1, int(np.ceil(np.linalg.norm(target - start) / stride)))
    for k in range(1, legs + 1):
        panel.aim(start + (target - start) * (k / legs), quat, moving=k < legs)
        yield
    panel.aim(target, quat, moving=False)
    for _ in range(steps):
        yield
        if float(np.linalg.norm(panel.ik.ee_pose(film.sim.state)[0] - target)) < tolerance:
            break


def _wait(n: int):
    for _ in range(n):
        yield


def _fold_reach(film):
    """Both hands, pads 20 mm apart, out past their corners at the overhang's height, then in along the edge;
    each then nudged up or down until its pads' middle is level with the sheet between them."""
    import numpy as np

    film.fold = lay = _fold_layout(film)
    points = np.asarray(film.sim["paper"].particle_q, dtype=np.float64)
    lay["strip"] = strip = float(np.median(points[points[:, 0] < lay["box_edge"]][:, 2]))
    for arm in _FOLD_SIDE:
        film.session.arms[arm].grip_request = FOLD_OPEN
    hands = {arm: _fold_hand(arm) for arm in _FOLD_SIDE}
    x, ys = lay["pinch"], lay["hands"]
    out = {arm: _FOLD_SIDE[arm] * 0.20 for arm in _FOLD_SIDE}
    yield from _go_both(
        film, _fold_targets(film, {a: (x, out[a], strip + 0.10) for a in hands}, hands), hands, 0.15, 300
    )
    out = {arm: _FOLD_SIDE[arm] * 0.165 for arm in _FOLD_SIDE}
    yield from _go_both(film, _fold_targets(film, {a: (x, out[a], strip) for a in hands}, hands), hands, 0.06, 200)
    targets = _fold_targets(film, {a: (x, ys[a], strip) for a in hands}, hands)
    yield from _go_both(film, targets, hands, 0.04, 200, 0.003)
    # A person looks: the hand sags a few millimetres under its own weight, so the cursor goes up by what is left.
    for _ in range(4):
        for arm, quat in hands.items():
            jaws = _fold_jaws(film, arm)
            points = np.asarray(film.sim["paper"].particle_q, dtype=np.float64)
            near = np.linalg.norm(points[:, :2] - jaws[:2], axis=1) < 0.007
            targets[arm] = targets[arm] + np.array([0.0, 0.0, float(np.median(points[near, 2])) - jaws[2]])
            film.session.arms[arm].aim(targets[arm], quat, moving=False)
        yield from _wait(40)


def _fold_pinch(film):
    """Both hands' **Grip amount** drawn from :data:`FOLD_OPEN` to :data:`FOLD_GRIP` over a second: the pads
    stop on the sheet. Set at once, the fingers met it at speed and it read 900 N a pad for a step."""
    for k in range(1, 61):
        for arm in _FOLD_SIDE:
            film.session.arms[arm].grip_request = FOLD_OPEN + (FOLD_GRIP - FOLD_OPEN) * k / 60
        yield
    yield from _wait(30)


def _fold_carry(film):
    """The near edge up and over on matching arcs, each hand turning with the flap to vertical, then holding
    its turn; lowered to 30 mm over the far edge."""
    lay = film.fold
    centre = lay["fold"] + FOLD_GROW
    radius = lay["fold"] - lay["pinch"] + FOLD_GROW
    for k in range(1, 13):
        turn = math.pi * k / 12
        quats = {arm: _fold_hand(arm, min(FOLD_TURN, math.degrees(turn))) for arm in _FOLD_SIDE}
        up = radius * math.sin(turn) + (0.03 if k == 12 else 0.0)
        points = {arm: (centre - radius * math.cos(turn), lay["hands"][arm], lay["strip"] + up) for arm in _FOLD_SIDE}
        yield from _go_both(film, _fold_targets(film, points, quats), quats, 0.06, 60)


def _fold_let_go(film):
    """Open to :data:`FOLD_OPEN`, up 50 mm, and out to each hand's own side, turning to the sweep's fingertip."""
    import numpy as np

    lay = film.fold
    yield from _wait(10)
    for arm in _FOLD_SIDE:
        film.session.arms[arm].grip_request = FOLD_OPEN
    yield from _wait(40)
    held = {arm: _fold_hand(arm, FOLD_TURN) for arm in _FOLD_SIDE}
    up = {arm: _fold_jaws(film, arm) + np.array([0.0, 0.0, 0.05]) for arm in _FOLD_SIDE}
    yield from _go_both(film, _fold_targets(film, up, held), held, 0.05, 60)
    sweep = {arm: _fold_sweeper(arm) for arm in _FOLD_SIDE}
    film.fold["park"] = park = {
        arm: np.array([lay["fold"] - 0.03, _FOLD_SIDE[arm] * 0.20, lay["top"] + 0.12]) for arm in _FOLD_SIDE
    }
    yield from _go_both(film, park, sweep, 0.10, 150, 0.01)
    for arm in _FOLD_SIDE:
        film.session.arms[arm].grip_request = 1.0
    yield from _wait(60)


def _fold_sweep(arm: str):
    """One hand runs its closed fingertip along the fold line, from the middle out past its own corner,
    :data:`FOLD_PRESS` over the box, 5 mm in from the fold's nose; then up and back to its side."""

    def script(film):
        import numpy as np

        lay, quat = film.fold, _fold_sweeper(arm)
        points = np.asarray(film.sim["paper"].particle_q, dtype=np.float64)
        x = float(points[np.abs(points[:, 1]) < 0.075][:, 0].min()) + 0.005  # the fold's nose, and 5 mm on
        s = _FOLD_SIDE[arm]
        tcp = film.session.arms[arm].ik.ee_pose(film.sim.state)[0]
        offset = _fold_tip(film, arm) - tcp
        start = np.array([x, s * 0.015, lay["top"] + FOLD_PRESS]) - offset
        yield from _go_both(film, {arm: start + np.array([0.0, 0.0, 0.07])}, {arm: quat}, 0.10, 120)
        yield from _go_both(film, {arm: start}, {arm: quat}, 0.02, 60, 0.002)
        for _ in range(3):  # the hand sags: nudge until the fingertip runs at the press height
            start = start + np.array([0.0, 0.0, lay["top"] + FOLD_PRESS - _fold_tip(film, arm)[2]])
            film.session.arms[arm].aim(start, quat, moving=False)
            yield from _wait(20)
        end = start + np.array([0.0, s * 0.105, 0.0])
        yield from _go_both(film, {arm: end}, {arm: quat}, 0.03, 30, 0.003)
        yield from _go_both(film, {arm: end + np.array([0.0, 0.0, 0.05])}, {arm: quat}, 0.05, 60)
        yield from _go_both(film, {arm: lay["park"][arm]}, {arm: quat}, 0.10, 60, 0.01)

    return script


def _fold_gizmos(film, scale: float = 0.07) -> None:
    """Each arm's gizmo drawn at half the page's 0.14 m, where the drive moves it from: at full size its rings
    lay over up to 19% of the folded sheet in the clip. The page's own size is unchanged."""
    for arm, panel in film.session.arms.items():
        if panel.gizmo is None:
            continue
        old = panel.gizmo
        panel.gizmo = film.session.server.scene.add_transform_controls(
            f"/studio/{arm}/cursor", scale=scale, line_width=3.0, position=old.position, wxyz=old.wxyz
        )


def _paper_folded(film) -> tuple[bool, str]:
    """Folded in half and creased: the edges laid within 10 mm of each other, the fold line keeping 2.2 rad."""
    import numpy as np

    from robosimstudio.assetlib.semantics.snapshot import Snapshot
    from robosimstudio.assetlib.vocabulary.measures import cloth_fold
    from robosimstudio_materials.paper import crease_angle

    sim = film.sim
    sim["paper"].entity.update_probe(sim.state, sim.step_count)
    snapshot = Snapshot(sim.make_context())
    lay = _fold_layout(film)
    angle = float(crease_angle(snapshot, "paper", {"crease": {"axis": 0, "at": lay["line"]}})["angle"])
    extent = float(cloth_fold(snapshot.ctx, "paper", axis=0)[0])
    entity = sim["paper"].entity
    rest = np.asarray(entity.rest["particle_q"], dtype=np.float64)[np.asarray(entity.particles)]
    now = np.asarray(sim["paper"].particle_q, dtype=np.float64)
    near, far = rest[:, 0] < rest[:, 0].min() + 0.003, rest[:, 0] > rest[:, 0].max() - 0.003
    gap = float(now[near, 0].mean() - now[far, 0].mean())
    return (
        angle >= 2.2 and abs(gap) <= 0.01 and extent < 0.65,
        f"the fold kept {angle:.2f} rad (need 2.2), the edges {gap * 1e3:+.1f} mm apart (need 10), span {extent:.3f}",
    )


SCRUB = "examples/01_cloth/03_scrub_a_stain.py"
SCRUB_TIP = 0.0167
"""The closed fingertips hang this far under the tool centre, pointing down [m].

A number of the YAM's, not of the example's layout: tests/test_scrub_a_stain's TIP."""
SCRUB_STAIN = (-0.05, 0.05, -0.08)
"""The stain's ends along x and its line in y, bench frame [m], as ``03_scrub_a_stain``'s selector draws it.

Restated, not read: the selector is a function in the example's ``main()``, and the particles it
picks lie on the towel's grid, the one row 3.3 mm off the line and 3.3 mm short of each end, so
nothing in the built scene carries these numbers (tests/test_scrub_a_stain's STAIN restates them too).
"""


def _scrub_reading(film) -> dict:
    labels = ("push (N)", "wiped (20-60 N)", "too hard")
    handles = film.session.server.gui._gui_input_handle_from_uuid.values()
    return {h.label: round(float(h.value), 2) for h in handles if getattr(h, "label", "") in labels}


def _scrub_press(reads: float):
    """Down from 10 mm over the towel until the page's **push** reads ``reads`` N, a control step at a time.

    What the page asks a person to do, and what tests/test_scrub_a_stain.py does: the cursor is lowered
    0.05 mm a control step, watching the readout. A height would not do: the push climbs 30 to 80 N a
    millimetre, and the same 6.0 mm read 32 N in the test and 116 N in a take.
    """

    def script(film):
        import numpy as np

        panel = film.session.arms["right"]
        x, y, _ = film.bench((SCRUB_STAIN[0] - 0.04, SCRUB_STAIN[2], 0.0))
        z, floor = film.bench((0.0, 0.0, SCRUB_TIP + 0.010))[2], film.bench((0.0, 0.0, SCRUB_TIP))[2]
        yield from _go(film, "right", (x, y, z), 0.05, steps=600, tolerance=0.004)
        while _scrub_reading(film).get("push (N)", 0.0) < reads and z > floor:
            z -= 0.00005
            panel.aim(np.array([x, y, z]), panel.cursor_quat, moving=True)
            yield
        panel.aim(np.array([x, y, z]), panel.cursor_quat, moving=False)
        yield from _wait(20)
        film.scrub_z = z

    return script


def _scrub_along(film):
    """Along the stain at the height the press stopped at, from 40 mm before it to 10 mm past, then on
    15 mm more while easing 8 mm up off the towel, 0.75 mm a control step: tests/test_scrub_a_stain's
    stroke. Stopped dead 10 mm past, the towel the stroke pushed ahead pressed back as the hand came to
    rest, and a light stroke wiped the stain's last particle (take_path.py)."""
    import numpy as np

    panel = film.session.arms["right"]
    (x0, y, _), flat = (
        film.bench((SCRUB_STAIN[0] - 0.04, SCRUB_STAIN[2], 0.0)),
        film.bench((SCRUB_STAIN[1] + 0.01, 0, 0))[0],
    )
    end = flat + 0.015
    n = round((end - x0) / 0.00075)
    for k in range(1, n + 1):
        x = x0 + (end - x0) * k / n
        z = film.scrub_z + 0.008 * max(0.0, x - flat) / 0.015
        panel.aim(np.array([x, y, z]), panel.cursor_quat, moving=k < n)
        yield


def _scrub_lift(film):
    """Straight up 30 mm from where the stroke left the towel."""
    import numpy as np

    panel = film.session.arms["right"]
    yield from _go(film, "right", np.asarray(panel.cursor_pos) + np.array([0.0, 0.0, 0.03]), 0.05)


def _keep_reading(name: str) -> Do:
    def keep(film):
        film.readings = {**getattr(film, "readings", {}), name: _scrub_reading(film)}
        return film.readings[name]

    return Do(fn=keep, label="")


def _stain_wiped(film) -> tuple[bool, str]:
    """The light stroke wiped nothing, the hard one marked the stain too hard and wiped none of it,
    lifted off included, and the stroke inside the band, on the stain cleaned back, wiped it: the
    clip ends on the wiped stain with nothing too hard."""
    seen = getattr(film, "readings", {})
    light, hard, off, band, done = (seen.get(k, {}) for k in ("light", "hard", "lifted", "band", "done"))
    passed = (
        light.get("wiped (20-60 N)", 1.0) == 0.0
        and hard.get("too hard", 0.0) >= 0.5
        and hard.get("wiped (20-60 N)", 1.0) == 0.0
        and off.get("wiped (20-60 N)", 1.0) == 0.0
        and band.get("wiped (20-60 N)", 0.0) >= 0.8
        and band.get("too hard", 1.0) == 0.0
        and done.get("wiped (20-60 N)", 0.0) >= 0.8
        and done.get("too hard", 1.0) == 0.0
    )
    return passed, f"light {light}, hard {hard}, lifted {off}, band {band}, at the end {done}"


ROBOT_CONTACT = (
    Take(
        FOLD,
        # From beyond the bench, a little to the robot's left and up, 0.95 m off: both hands, the box,
        # the matched arcs of the carry and the sweep along the fold, the sheet across 31 to 36% of the
        # view's width, folded or flat (from 1.10 m and 35 degrees round, the folded sheet took 26%).
        camera=Orbit(Bench(0.08, 0.0, 0.12), distance=0.95, azimuth=20.0, elevation=40.0, fov=38.0),
        warmup=0.5,
        fps=8,
        speed=2.0,  # a minute of scene: a reach, a carry and two sweeps
        budget=2_000_000,
        layout=Layout(collapse=("Simulation", "Material", "Physics")),
        beats=(
            Fold(title="Simulation", open=False),
            Fold(title="How to drive this", open=False),
            Do(fn=_fold_gizmos, label=""),
            Wait(seconds=0.4, label="an A4 sheet on a box, its near edge 35 mm over the box's edge"),
            _per_step(_fold_reach, "both hands out past their corners, then in along the edge: the pads straddle it"),
            _per_step(_fold_pinch, "Grip amount 0.986 on both: the pads stop on the sheet"),
            Scroll(pixels=-600, seconds=0.4),  # back to the top: the Crease readouts lead the panel
            Wait(seconds=0.4, mark="pinched"),
            _per_step(_fold_carry, "up and over together, each hand turning 88 degrees, each on its own side"),
            _per_step(_fold_let_go, "let go 30 mm over the far edge, and out to each hand's own side"),
            _per_step(_fold_sweep("right"), "the right fingertip runs along the fold, from the middle out"),
            _per_step(_fold_sweep("left"), "the left fingertip runs along the fold, from the middle out"),
            Wait(seconds=1.5, label="it stays folded in half"),
        ),
        expect=(Check(_paper_folded, needs=("paper",)),),
    ),
    Take(
        SCRUB,
        # In front of the bench and well to its right, 1.25 m off: the whole working arm, the closed
        # hand, the towel and the drawn stain, the Scrub folder's push and wiped beside them. From
        # azimuth -40 the idle left arm's upper links ran along the top edge, drawn see-through by the
        # page whether or not the room was drawn (cause not found); from -60 they are out of frame.
        camera=Orbit(Bench(0.04, -0.10, 0.02), distance=1.25, azimuth=-60.0, elevation=36.0, fov=38.0),
        warmup=1.0,  # the towel lands from 20 mm
        fps=8,
        speed=1.0,  # three strokes of 3.7 s; a stroke drawn faster heaps the towel and pushes harder
        budget=1_800_000,
        layout=Layout(collapse=("Simulation", "Arm: left")),
        # The two controls first and the task last, so the clip ends on the wiped stain: a light
        # stroke, a hard one marked red, Clean stain, then the stroke inside the band.
        beats=(
            Fold(title="Simulation", open=False),
            Fold(title="How to drive this", open=False),
            Wait(seconds=0.5, label="the right hand over the stain's near end"),
            Press(text="Close", within=RIGHT, label="Close"),
            # Straight off the sidebar. A Scroll here moved nothing (the sidebar fits) and glided the
            # pointer to its middle, onto the right arm's Open, lit as hovered until the drive began.
            Rest(seconds=0.4),
            Wait(seconds=0.4),
            _per_step(_scrub_press(0.5), "down until the fingers touch the towel"),
            _per_step(_scrub_along, "a light stroke: push under 20 N, nothing wiped"),
            _per_step(_scrub_lift, "up off the towel"),
            _keep_reading("light"),
            _per_step(_scrub_press(90.0), "down until push reads 90 N, past the band"),
            _per_step(_scrub_along, "too hard: marked red, not wiped"),
            _keep_reading("hard"),
            _per_step(_scrub_lift, "lifted off: still nothing wiped"),
            _keep_reading("lifted"),
            Press(text="Clean stain", within="Scrub", label="Clean stain: the red marks are cleared"),
            _per_step(_scrub_press(20.0), "down until push reads 20 N, inside the band"),
            _per_step(_scrub_along, "the stroke inside the band wipes the stain"),
            _keep_reading("band"),
            _per_step(_scrub_lift, "up off the towel"),
            _keep_reading("done"),
            Wait(seconds=1.5, mark="wiped", label="the stain is wiped: wiped 1.0, too hard 0"),
        ),
        expect=(Check(_stain_wiped),),
    ),
)

# =============================================================================
# Water, poured by the hand
# =============================================================================
#
# The pour tests/test_pour_water drives through the right arm's panel, at its speeds: down onto the
# cup, Close, up to 170 mm, 125 mm across, the gizmo turned 120 degrees about x, held, turned back.

ROBOT_WATER = (
    Take(
        "examples/03_materials/14_water.py",
        # From the front, a little to the glass's side: the cup tips across the view, toward the lens, and the
        # stream falls over the glass's near wall. 1.1 m keeps the raised wrist under a third of the frame.
        camera=Orbit(Bench(0.12, -0.20, 0.13), distance=1.1, azimuth=15.0, elevation=32.0, fov=38.0),
        warmup=1.0,
        layout=Layout(collapse=("Simulation", "Arm: left")),
        beats=(
            Wait(seconds=0.5, label="the hand, open, over the cup of water"),
            Drag(
                "right",
                "z",
                to=At("cup", dz=0.055),
                seconds=1.4,
                label="drag the gizmo down: the jaws straddle the cup",
            ),
            Wait(seconds=0.3),
            Press(text="Close", within=RIGHT, label="Close"),
            Rest(seconds=0.4),  # off the button, so Close does not sit lit as hovered while the jaws shut
            Wait(seconds=0.6, mark="closed", label="the pads on its walls"),
            Drag("right", "z", 0.115, seconds=1.6, label="drag it up to 170 mm"),
            Drag("right", "y", 0.125, seconds=1.6, label="and 125 mm across, over the glass"),
            Wait(seconds=0.5, mark="carried"),
            Turn("right", "x", -120.0, seconds=2.0, label="turn the gizmo 120 degrees about x: the wrist tips the cup"),
            Wait(seconds=2.5, mark="poured", label="the water runs into the glass, past the red line"),
            Turn("right", "x", 120.0, seconds=2.0, label="back upright"),
            Wait(seconds=1.0, label="the glass filled past its line"),
        ),
        expect=(
            Rose("cup", 0.05, since="closed", at="carried"),
            Held("cup", since="closed", at="carried"),
            Check(_water_in_glass),
        ),
        budget=1_300_000,  # 19 s of a moving arm, as the honey pour's
    ),
)

# The cutting kind: the right hand takes the knife off its rest by its handle, by contact, and pushes
# it down through the tofu until the edge meets the board, the drive tests/test_cut_tofu runs through
# the same arm panel. The hand closes on the handle's middle as the example stands the knife in its rest
# (``Laid("knife")``, 40 mm up), lifts it 70 mm and carries it across to the tofu's middle.


def _edge_into_board(film) -> tuple[float, float, float]:
    """Where the tool centre goes for the knife's edge to end 5 mm into the board, as tests/test_cut_tofu pushes.

    The board's top as the example lays it, 12 mm up, plus how far the edge is under the handle's
    middle, 40 mm off the knife's spec for its 50 mm blade.
    """
    x, y, top = laid(film, "board", (0, 0, 1))
    return film.bench((x, y, top - float(film.sim["knife"].spec.edge[0][2]) - 0.005))


def _parted(film) -> tuple[bool, str]:
    """``03_materials/10_cutting``'s readouts: the tofu's pieces and how far the cut ran."""
    tofu = film.sim["tofu"].entity
    pieces, fraction = tofu.pieces(), tofu.cut_fraction()
    return pieces == 2, f"the tofu is in {pieces} pieces, the cut {100.0 * fraction:.0f}% through (the test asks 2)"


ROBOT_CUTTING = (
    Take(
        "examples/03_materials/10_cutting.py",
        # From the front-right, 0.95 m off: the knife rest, the board and the hand in one frame.
        camera=Orbit(Bench(0.03, -0.22, 0.04), distance=0.95, azimuth=48.0, elevation=30.0, fov=37.0),
        warmup=0.5,
        layout=Layout(collapse=("Simulation", "Arm: left"), expand=("Tofu",)),
        budget=1_200_000,
        beats=(
            Wait(seconds=0.4, label="a knife standing in its rest under the right hand, a block of tofu on a board"),
            Drag("right", "z", to=Laid("knife"), seconds=1.2, label="drag the hand down onto the handle"),
            Wait(seconds=0.3),
            Press(text="Close", within=RIGHT, label="Close: the pads squeeze the handle"),
            Wait(seconds=0.8, mark="closed"),
            Drag("right", "z", to=Laid("knife", dz=0.07), seconds=1.2, label="up 70 mm: the blade leaves its rest"),
            Drag("right", "y", to=Laid("tofu"), seconds=1.5, label="across to the tofu"),
            Drag(
                "right",
                "z",
                to=_edge_into_board,
                seconds=3.0,
                label="down through the tofu until the edge meets the board",
            ),
            Wait(seconds=1.6, mark="cut", label="the edge is on the board: the tofu is in two pieces"),
        ),
        expect=(Check(_parted, needs=("tofu",)), Held("knife", since="closed", at="cut", tolerance=0.005)),
    ),
)

# The stuck label: the right hand carries a loose label by its pull tab, lays it on the box and rubs
# it down with its shut fingertips, the drive tests/test_stick_a_label makes through the same panel.
STICK = "examples/03_materials/09_adhesion.py"
STICK_AJAR = 0.6
"""The **Grip amount** the jaws open to around the tab: 38 mm, the far jaw clear of the box."""
STICK_TIP, STICK_PUSH, STICK_HOVER, STICK_CLEAR = 0.0165, 0.003, 0.05, 0.011
"""The tool centre over the shut fingertips, how far under the label the rub's cursor goes, how high
the fingertips move between strokes and how far past the tab's furthest corner a stroke starts [m]:
tests/test_stick_a_label's TIP, the YAM's, and its drive's PUSH, HOVER and CLEAR."""


def _stick_far(film) -> tuple[float, float, float]:
    """Where the tool centre is when the carried label's free end touches the box top [world].

    Over the middle of the box's far (-y) end as the example lays it, 15 mm in from it and 96 mm over
    its top: bench (0, -125, 146) mm (tests/test_stick_a_label's FAR).
    """
    x, y, z = laid(film, "box", (0, -1, 1))
    return film.bench((x, y + 0.015, z + 0.096))


def _stick_near(film) -> tuple[float, float, float]:
    """Where the tab is set down [world]: over the box's near (+y) end, 20 mm in from it and 19 mm over its top.

    Bench (0, -40, 69) mm (tests/test_stick_a_label's NEAR).
    """
    x, y, z = laid(film, "box", (0, 1, 1))
    return film.bench((x, y - 0.020, z + 0.019))


def _stick_column(label, state, c: int, along: float):
    """The label's grid column ``c``, ``along`` metres from its tab end, where it lies now [m, world]."""
    import numpy as np

    nx, ny = label.spec.cells
    cell = label.spec.cell[1]
    q = state.particle_q.numpy()[np.asarray(label.particles)]
    j = min(int(along // cell), ny - 1)
    t = along / cell - j
    return (1.0 - t) * q[j * (nx + 1) + c] + t * q[(j + 1) * (nx + 1) + c]


def _stick_tab_reach(label, state, away) -> float:
    """How far along ``away`` the tab's furthest corner reaches [m]."""
    import numpy as np

    from robosimstudio.core.math import quat_to_mat

    pose = label.tab_pose(state)
    corners = (
        np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * 0.5 * np.asarray(label.spec.tab_size)
    )
    return float(np.max((corners @ quat_to_mat(pose[3:7]).T + pose[:3]) @ away))


def _stick_onto_tab(film):
    """Over the tab and down until the jaws straddle it."""
    import numpy as np

    film.stick_tab = tab = film.sim["label"].entity.tab_pose(film.sim.state)[:3]
    yield from _go(film, "right", tab + np.array([0.0, 0.0, 0.05]), 0.08)
    yield from _go(film, "right", tab + np.array([0.0, 0.0, 0.004]), 0.04, tolerance=0.004)


def _stick_lay(film):
    """Up 150 mm, over the box's far end, then down and toward its near end together: it lies down."""
    import numpy as np

    tab, far, near = film.stick_tab, np.asarray(_stick_far(film)), np.asarray(_stick_near(film))
    yield from _go(film, "right", (tab[0], tab[1], film.bench((0.0, 0.0, 0.15))[2]), 0.04)
    yield from _go(film, "right", far, 0.03)
    yield from _wait(60)
    for k in range(1, 31):
        yield from _go(film, "right", far + (near - far) * k / 30, 0.03, steps=5)
    yield from _wait(20)


def _stick_off(film):
    """Up 50 mm off the tab."""
    import numpy as np

    yield from _go(film, "right", np.asarray(_stick_near(film)) + np.array([0.0, 0.0, 0.05]), 0.05)
    yield from _wait(30)


def _stick_rub(film):
    """Four strokes along the label's columns 1, 3, 5 and 7, where it lies, from past the tab to past its end."""
    import numpy as np

    label = film.sim["label"].entity
    a, b = _stick_column(label, film.sim.state, 4, 0.03), _stick_column(label, film.sim.state, 4, 0.09)
    away = np.array([b[0] - a[0], b[1] - a[1], 0.0])
    away /= np.linalg.norm(away)
    lift, hover = np.array([0.0, 0.0, STICK_TIP - STICK_PUSH]), np.array([0.0, 0.0, STICK_HOVER + STICK_PUSH])

    def ends(c):
        state = film.sim.state
        first = _stick_column(label, state, c, 0.03)
        start = first - away * (float(np.dot(first, away)) - _stick_tab_reach(label, state, away) - STICK_CLEAR)
        return start + lift, _stick_column(label, state, c, 0.098) + lift

    for c in (1, 3, 5, 7):
        start, end = ends(c)
        yield from _go(film, "right", start + hover, 0.08, steps=60)
        start, end = ends(c)
        yield from _go(film, "right", start, 0.05, steps=0)
        yield from _wait(10)
        yield from _go(film, "right", end, 0.03, steps=30)
        yield from _wait(10)
        yield from _go(film, "right", end + hover, 0.05, steps=30)


def _stick_clear(film):
    """Up 60 mm and back 60 mm toward the arm, off the label, so the stuck label shows."""
    import numpy as np

    panel = film.session.arms["right"]
    yield from _go(film, "right", np.asarray(panel.cursor_pos) + np.array([-0.06, 0.0, 0.06]), 0.06)
    yield from _wait(20)


def _keep_bonded(name: str) -> Do:
    def keep(film):
        film.bonded = {**getattr(film, "bonded", {}), name: float(film.sim["label"].entity.bonded())}
        return film.bonded

    return Do(fn=keep, label="")


def _stuck(film) -> tuple[bool, str]:
    """Laid, it barely stuck; rubbed, 80% of its bonds hold, read off the page's bonded_fraction."""
    seen = getattr(film, "bonded", {})
    laid, rubbed = seen.get("laid", 1.0), float(film.sim["label"].entity.bonded())
    return (
        laid < 0.1 and rubbed >= 0.8,
        f"bonded {laid:.3f} laid and {rubbed:.3f} rubbed (the test asks under 0.1, then 0.8)",
    )


ROBOT_TAPE = (
    *ROBOT_TAPE,
    Take(
        STICK,
        # In front of the bench and to its right: the label on the bench, the box, the carry 150 mm up
        # and the four strokes in one view, the Label folder's bonded_fraction beside them.
        camera=Orbit(Bench(0.0, -0.14, 0.07), distance=1.0, azimuth=38.0, elevation=34.0, fov=38.0),
        warmup=0.5,
        layout=Layout(collapse=("Simulation", "Arm: left"), expand=("Label",)),
        speed=2.0,
        fps=10,
        budget=1_800_000,
        beats=(
            Wait(seconds=0.4, label="a loose label on the bench beside the box, its tab under the right hand"),
            Slider(
                text="Grip amount",
                value=STICK_AJAR,
                seconds=0.5,
                within=RIGHT,
                label="Grip amount 0.6: the jaws part 38 mm",
            ),
            _per_step(_stick_onto_tab, "drag the hand down: the jaws straddle the tab"),
            Press(text="Close", within=RIGHT, label="Close: the pads take the tab"),
            Wait(seconds=0.6, mark="closed"),
            _per_step(_stick_lay, "up, over the box's far end, down toward its near end: the label lies down"),
            Slider(
                text="Grip amount",
                value=STICK_AJAR,
                seconds=0.5,
                within=RIGHT,
                label="Grip amount 0.6: the tab is let go",
            ),
            _per_step(_stick_off, "the hand lifts off: laid, not pressed, it barely sticks"),
            _keep_bonded("laid"),
            Press(text="Close", within=RIGHT, label="Close: the fingertips shut"),
            Scroll(pixels=-600, seconds=0.3),  # back to the top: the Label readouts lead the panel
            _per_step(_stick_rub, "four strokes along the label, 10 mm apart: it sticks where they pressed"),
            _per_step(_stick_clear, "the hand lifts away"),
            Wait(seconds=1.5, mark="stuck", label="stuck: bonded_fraction past 0.8"),
        ),
        expect=(Check(_stuck, needs=("label",)),),
    ),
)

# =============================================================================
# Scored: the same drives on the task pages, ending on the goal
# =============================================================================
#
# Each ``06_tasks/NN_<kind>_scored`` script is a material's scene as a task document with a goal,
# opened by ``rss.view(env)``: the page holds the same scene and arm panels, and its sidebar opens
# with the **Goal** folder, one line a clause, ticked once it holds. Its take is its material
# take's drive, beat for beat; a page whose hand starts at the YAM's own home, pointing ahead,
# first takes it over the thing and turns it down, as the scored script's medium test does. The
# take then waits for the goal to read true, holds on the tick, and keeps every check of the
# material take, with one more: the goal met at the end.

GOAL_WAIT = 8.0
"""The longest the clip waits for the goal after the drive [s of clip]: a ``hold-for`` banks its seconds here."""
GOAL_HOLD = 1.5
"""How long the clip holds on the ticked goal [s of clip]."""
TASK_FOLDS = ("Inspector", "Policy")
"""The task page's folds no drive uses, closed: open, the Policy folder's Run and the inspector's rows
overran the sidebar, and pressing **Close** or ticking **Vacuum** scrolled the Goal folder out of view."""


def _goal_reads_true(film) -> bool:
    """The page's own verdict on its last step: what the Goal folder ticks, scored once a step by its drain."""
    return bool(film.sim.last_result.success)


def _goal_met(film) -> tuple[bool, str]:
    """The task's goal met at the end, as the page's Goal folder says it: the verdict, and its first clause's line."""
    from robosimstudio_studio.goal_panel import TICK

    panel = film.session.goal_panel
    if panel is None:
        return False, "the page has no Goal folder"
    lines = [line.strip("- ").strip() for line in panel._tree_text().splitlines() if line.strip()]
    met = _goal_reads_true(film) and bool(lines) and lines[0].startswith(TICK)
    return met, f"the Goal folder reads {lines[0] if lines else 'nothing'!r} at the end (need {TICK}), success={met}"


def _material(example: str) -> Take:
    """The take that films ``example``'s own clip."""
    found = [t for t in (*DRAFTS, *ROBOT_PARTICLES, *ROBOT_BAGS, *ROBOT_BRUSH) if t.example == example and not t.name]
    found += [
        t for t in (*ROBOT_SUCTION, *ROBOT_TAPE, *ROBOT_WATER, *ROBOT_CUTTING) if t.example == example and not t.name
    ]
    (take,) = found
    return take


def _opening(view: Orbit) -> Orbit:
    """``view`` drawn back to hold the YAM's right hand at its home as well: aimed between the two, further off, wider."""
    return Orbit(
        Mid(Hand("right"), view.about),
        distance=min(1.4, view.distance + 0.3),
        azimuth=view.azimuth,
        elevation=view.elevation,
        fov=view.fov + 6.0,
    )


def scored(example: str, of: str, *, over: Any = None, turn: bool = False, what: str = "", **kw) -> Take:
    """``of``'s take on ``example``'s page, its beats one for one, and then the goal.

    ``over``: the scored page's right hand starts at the YAM's own home, pointing ahead, where the
    material page's starts over ``what``. The clip opens drawn back to hold the hand there, drags it
    over ``what``, turns it to point down if ``turn``, and closes in to the material take's view in
    place of that take's opening pause. The sidebar opens the Goal folder and closes what the
    material take closes; a folder only the material script's page adds is not there to open.
    With ``over``, ``of``'s take must open on a :class:`Wait`, the pause this stands in for; a
    take that opens on anything else is refused, since dropping its first beat drops a step of
    the drive.
    """
    material = _material(of)
    start: tuple[Beat, ...] = ()
    if over is not None:
        opening = material.beats[0] if material.beats else None
        if not isinstance(opening, Wait):
            raise ValueError(
                f"{of}'s take opens on {type(opening).__name__}, not a Wait: the drag over {what or 'the work'} "
                "stands in for an opening pause, and dropping this beat would drop a step of the drive"
            )
        start = (
            Wait(seconds=0.5, label=f"the right hand at the YAM's home, pointing ahead; {what}"),
            DragTo("right", over, seconds_per_m=4.5, label=f"drag the hand over {what}"),
            *((_turn("right", TOOL_DOWN, 1.0, "turn the gizmo: the hand points down"),) if turn else ()),
            _dolly(material.camera, 1.2, "and the view closes in"),
        )
        kw.setdefault("camera", _opening(material.camera))
    return replace(
        material,
        example=example,
        beats=(
            *start,
            *material.beats[1 if start else 0 :],
            Wait(seconds=GOAL_WAIT, until=_goal_reads_true, label="the Goal folder ticks: the goal turns true"),
            Wait(seconds=GOAL_HOLD, mark="goal", label="the goal met"),
        ),
        layout=Layout(collapse=(*material.layout.collapse, *TASK_FOLDS), expand=("Goal",)),
        expect=(*material.expect, Check(_goal_met)),
        **kw,
    )


def _over_the_lip(film) -> tuple[float, float, float]:
    """100 mm over the bag's lip where it crosses 25 mm up, as ``tests/test_open_a_bag_scored`` aims [world].

    The top layer's turned-up lip: its middle across the bag, and where along it the lip stands
    25 mm over the bench, the height the hand closes at. ``03_materials/06_bag``'s home is here.
    """
    import numpy as np

    bag = film.sim["bag"]
    lip = np.asarray(bag.particle_q, dtype=np.float64)[bag.entity.top[int(bag.entity.spec.cells[0]) + 1 :].ravel()]
    order, bench = np.argsort(lip[:, 2]), film.bench((0.0, 0.0, 0.0))[2]
    x, y = float(np.median(lip[:, 0])), float(np.interp(bench + 0.025, lip[order, 2], lip[order, 1]))
    return (x, y, bench + 0.10)


def _over_the_tab_high(film) -> tuple[float, float, float]:
    """50 mm over the label's tab, where ``tests/test_peel_a_label_scored`` takes the hand first [world]."""
    x, y, z = _tab(film)
    return film.bench((x, y, z + 0.05))


def _over_the_handle(film) -> tuple[float, float, float]:
    """60 mm over where the hand closes on the brush's handle [world]: ``03_materials/07_brush``'s home, about."""
    x, y, z = _over_the_sole(film)
    return (x, y, z + 0.06)


SCORED = (
    scored("examples/06_tasks/03_granular_scored.py", "examples/03_materials/11_granular.py"),
    scored(
        "examples/06_tasks/04_brush_scored.py",
        "examples/03_materials/07_brush.py",
        over=_over_the_handle,
        turn=True,
        what="the brush's handle",
        budget=1_800_000,
    ),
    scored("examples/06_tasks/05_suction_scored.py", "examples/04_extending/02_suction.py"),
    scored(
        "examples/06_tasks/06_bag_scored.py",
        "examples/03_materials/06_bag.py",
        over=_over_the_lip,
        turn=True,
        what="the bag's lip",
        budget=1_300_000,  # 20 s with the carry over from the YAM's home: 1,232 KB at quality 46, over the default 900
    ),
    scored("examples/06_tasks/07_band_scored.py", "examples/03_materials/04_band.py"),
    # Filmed 4 times with the page's (:effort-limit 12), the material page's ceiling: passed 4 of 4,
    # and 03_materials/08_tape 4 of 4. PR #9's filming without the clause passed 3 of 4. On every page
    # the pads stall 3.84 mm short of shut and hold the tab at 5.8 N; the clause caps only Close's
    # first push (71 N uncapped). After the last bonds let go, the tab fell from the jaws in 3 of the
    # scored films and 2 of the material ones, once the goal had turned: the drive, not the grip.
    scored(
        "examples/06_tasks/08_tape_scored.py",
        "examples/03_materials/08_tape.py",
        over=_over_the_tab_high,
        turn=True,
        what="the label's pull tab",
        budget=1_500_000,  # 30 s at 12 frames: 1,397 KB at quality 46, over the material take's 1,200
    ),
    scored("examples/06_tasks/09_cutting_scored.py", "examples/03_materials/10_cutting.py"),
    scored(
        "examples/06_tasks/10_water_scored.py",
        "examples/03_materials/14_water.py",
        over=Laid("cup", dz=0.155),
        what="the cup of water",
    ),
)

TAKES: tuple[Take, ...] = (
    *FINAL,
    *DRAFTS,
    *ROBOT_PARTICLES,
    *ROBOT_BAGS,
    *ROBOT_BRUSH,
    *ROBOT_SUCTION,
    *ROBOT_TAPE,
    *ROBOT_CONTACT,
    *ROBOT_WATER,
    *ROBOT_CUTTING,
    *SCORED,
)

KEPT: dict[str, str] = {
    "05_studio/01_render": "opens no page: it path traces a still and a video, which is what its clip shows",
    "07_policies/03_many_worlds": "opens no page: a fleet run of every world at once, filmed as a grid of tiles",
    "live/live_page": "the live page rebuilding on a save of scene.py: a file being edited, not a scene being driven",
}
"""Clips no take films, each with why its existing file stands. Keyed like :attr:`Take.key`."""


def find(key: str) -> Take:
    for take in TAKES:
        if take.key == key:
            return take
    raise KeyError(key)

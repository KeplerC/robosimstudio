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
from dataclasses import dataclass, field
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
    Held,
    Highest,
    Layout,
    Mid,
    Nearest,
    Orbit,
    Patch,
    Policy,
    Press,
    Rose,
    Run,
    Sank,
    Scroll,
    Segment,
    Select,
    Slider,
    Squashed,
    Toggle,
    Top,
    Turn,
    Wait,
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

    ref = film.sim["cable"]
    points = np.asarray(film.sim.pipeline.state_0.body_q.numpy())[ref.bodies, :3]
    path = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    straightness = float(np.linalg.norm(points[-1] - points[0]) / path) if path > 0 else 1.0
    return straightness < 0.9, f"straightness {straightness:.3f} after release (need under 0.9)"


def _rolled_flat(film) -> tuple[bool, str]:
    """The roll example's own readouts: the footprint 1.2 times the block's or more, the height 60% of it or less."""
    import numpy as np

    from robosimstudio_materials.mpm.measures import heap_height, spread

    ref = film.sim["dough"]
    rest = ref.entity.rest["particle_q"][ref.particles]
    points = np.asarray(ref.particle_q, dtype=np.float64)
    floor = float(ref.entity.floor_z)
    footprint, before, after = spread(points, rest), heap_height(rest, floor), heap_height(points, floor)
    return footprint >= 1.2 and after <= 0.6 * before, (
        f"footprint {footprint:.2f} of the block's (need 1.2), height {before * 1000:.1f} -> {after * 1000:.1f} mm "
        f"(need {0.6 * before * 1000:.1f})"
    )


def _flap_shut(film) -> tuple[bool, str]:
    """The box flap's hinge kept 1.8 rad once the hand had gone, and its free edge rests no higher than the rim."""
    import math

    import numpy as np

    from robosimstudio.assetlib.semantics.snapshot import Snapshot
    from robosimstudio_materials.paper import crease_angle

    sim, flap = film.sim, film.sim["flap"]
    line = {"crease": {"axis": 2, "at": 0.103 + 0.02 * math.cos(math.radians(15.0))}}  # the example's hinge rows
    flap.entity.update_probe(sim.state, sim.step_count)
    angle = float(crease_angle(Snapshot(sim.make_context()), "flap", line)["angle"])
    rest = np.asarray(flap.entity.rest["particle_q"], dtype=np.float64)[flap.particles]
    edge = np.asarray(flap.particle_q, dtype=np.float64)[rest[:, 2] > rest[:, 2].max() - 0.003].mean(axis=0)
    above = float(edge[2]) - (0.728 + 0.10)
    return angle >= 1.8 and above < 0.015, (
        f"the hinge kept {angle:.2f} rad (need 1.8) and the free edge is {above * 1000:+.0f} mm off the rim (need under 15)"
    )


def _band_on_both(film) -> tuple[bool, str]:
    """The band is round both pegs of the board and stretched there, the hand gone."""
    import numpy as np

    from robosimstudio_materials.rod.band import encircles

    ref = film.sim["band"]
    points = np.asarray(ref.particle_q, dtype=np.float64)
    pegs = {"upper": (0.26, 0.08), "lower": (0.135, 0.05)}  # the example's heights and lengths [m]
    on = {n: encircles(points, (0.7, -0.2, 0.75 + z), (-1.0, 0.0, 0.0), length) for n, (z, length) in pegs.items()}
    stretch = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) / float(ref.entity.spec.length)
    return all(on.values()) and stretch > 1.1, f"round {on}, stretch {stretch:.3f} (need both, over 1.1)"


def _knot_tight(film) -> tuple[bool, str]:
    """The knot kept its three crossings, gave back rope, and both tails are still in their hands."""
    import numpy as np

    from robosimstudio_materials.rod.knot import crossings, knot_length, overhand

    points = np.asarray(film.sim["rope"].particle_q, dtype=np.float64)
    built = knot_length(overhand(at=(0.03, 0.0), size=0.10, radius=0.006, tail=0.12, spacing=0.008))
    now, cross = knot_length(points), crossings(points)
    hand = {a: film.sim.arm(a).ee_pose(film.sim.state)[0] for a in ("left", "right")}  # the page is shut by now
    held = max(float(np.linalg.norm(points[i] - hand[a])) for i, a in ((0, "right"), (-1, "left")))
    ok = cross == 3 and now < 0.95 * built and held < 0.06
    return (
        ok,
        f"{cross} crossings, {now * 1e3:.0f} of {built * 1e3:.0f} mm in the knot, tails {held * 1e3:.0f} mm off the hands",
    )


def _sprang_back(film) -> tuple[bool, str]:
    """The span's middle is back where it hung before the hand took it."""
    key = str(Segment("cable", 0.5))
    off = abs(film.marks["end"]["things"][key][2] - film.marks["start"]["things"][key][2])
    return off < 0.002, f"the middle ended {off * 1000:.1f} mm from where it hung (need under 2)"


def _round_the_post(film) -> dict:
    """The memory take's carry: the right cursor walked 120 degrees round the post, its gizmo turning with it.

    What the gizmo's arrows and ring write, a film frame at a time: no beat drags
    an arc or turns a ring. The path is ``tests/test_cable_robot``'s, at 50 mm/s,
    shrinking by what the wrap lays on the post; the capsule's distance from the
    hand at the end is kept for :func:`_carried_round`.
    """
    import math

    import numpy as np

    from robosimstudio.core.math import quat_multiply

    arm, wrap = film.session.arms["right"], math.radians(120.0)
    post = film.bench((0.0, -0.086, 0.0))
    x0, y0, z = (float(v) for v in arm.cursor_pos)
    start = (x0 - post[0], y0 - post[1])
    rho = math.hypot(*start)
    frames = round(rho * wrap / 0.05 * film.take.fps)
    for i in range(1, frames + 1):
        a, shrink = wrap * i / frames, 1.0 - wrap * 0.031 / rho * i / frames
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


def _turn_both(film) -> dict:
    """The twist take's turn: each hand's cursor turned 120 degrees inward about the vertical, over four seconds.

    What each gizmo's ring writes; no beat turns a ring. ``tests/test_cable_robot``
    turns them the same way, a degree a control step. The mid-cable twists at the
    end are kept for :func:`_twisted`.
    """
    import math

    import numpy as np

    from robosimstudio.core.math import quat_multiply, quat_to_mat
    from robosimstudio_materials.rod.measures import twist_turns

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
    body_q = film.sim.state.body_q.numpy()
    film.twist_held = {
        n: abs(twist_turns(body_q[film.sim[n].bodies[8:17], 3:7])) for n in ("soft_twist", "stiff_twist")
    }
    return {"turned_deg": 120.0, "held": {n: round(v, 3) for n, v in film.twist_held.items()}}


def _twisted(film) -> tuple[bool, str]:
    """The turn reached both middles while held, and once let go only the stiff cable gave it back."""
    import numpy as np

    from robosimstudio_materials.rod.measures import twist_turns

    held = getattr(film, "twist_held", {"soft_twist": 0.0, "stiff_twist": 0.0})
    body_q = film.sim.state.body_q.numpy()
    now = {n: abs(twist_turns(np.asarray(body_q)[film.sim[n].bodies[8:17], 3:7])) for n in held}
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


def _poured(film) -> tuple[bool, str]:
    """The pour example's readout at the end: the share of the honey inside the bowl, 90% or more."""
    import numpy as np

    from robosimstudio_materials.mpm.measures import fill_fraction

    # The bowl's inside as the example states it: 140 mm square about bench (0.12, -0.16), from 5 mm under the top.
    lo, hi = np.array(film.bench((0.05, -0.23, -0.005))), np.array(film.bench((0.19, -0.09, 0.04)))
    share = fill_fraction(np.asarray(film.sim["honey"].particle_q, dtype=np.float64), lo, hi)
    return share >= 0.9, f"{share:.3f} of the honey in the bowl (need 0.9)"


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
    Take(
        "examples/03_rope/01_between_posts.py",
        camera=Orbit(At("cable", dz=0.08), distance=1.0, azimuth=-30.0, elevation=30.0, fov=50.0),
        warmup=1.0,
        layout=Layout(collapse=("Simulation", "Arm: left")),
        beats=(
            Wait(seconds=0.5, label="the right hand pointing at the span's middle"),
            # The hand leans 40 degrees, so its z arrow is the hand's own axis: 80 mm down it is the cable.
            Drag("right", "z", -0.08, seconds=1.2, label="drag the gizmo down onto the cable"),
            Press(text="Close", within=RIGHT, label="Close"),
            Wait(seconds=0.8, mark="closed"),
            Drag("right", "z", 0.03, seconds=1.0, label="pull it back up"),
            Wait(seconds=0.8, mark="pulled", label="the span hangs from the hand"),
            Press(text="Open", within=RIGHT, label="Open"),
            Drag("right", "z", 0.05, seconds=0.8, label="and away"),
            Wait(seconds=1.5, label="it springs back between the posts"),
        ),
        expect=(Rose(Segment("cable", 0.5), 0.008, at="pulled"), Check(_sprang_back, needs=(Segment("cable", 0.5),))),
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


def routine(example: str, *, speed: float, success: str, done: Any = None, budget: int = 1_500_000) -> Take:
    """A demo: the home pose, then its Run button, filmed to the routine's end.

    ``done(film)`` is the demo's own success measure; a take without one is still a draft.
    """
    return Take(
        example,
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
    # The hand, closed and pointing down, pushes the standing flap over and presses it into the box;
    # tests/test_box_flap_robot.py drives the same path through the panel.
    Take(
        "examples/13_paper/01_box_flap.py",
        camera=Orbit(Bench(-0.06, 0.05, 0.12), distance=1.05, azimuth=35.0, elevation=32.0),
        warmup=1.0,
        layout=Layout(collapse=("Simulation", "Arm: right")),
        beats=(
            Wait(seconds=0.5, label="the flap stands open, the left hand beside it"),
            Press(text="Close", within="Arm: left", label="Close"),
            Wait(seconds=0.6),
            Drag("left", "z", to=Bench(z=0.162), seconds=0.8, label="down, level with the flap"),
            Drag("left", "y", to=Bench(y=-0.01), seconds=1.8, label="across: the hand pushes it over"),
            Drag("left", "z", to=Bench(z=0.062), seconds=1.2, label="down: it presses the flap into the box"),
            Wait(seconds=0.5, mark="pressed"),
            Drag("left", "z", to=Bench(z=0.162), seconds=1.0, label="up: the hand leaves"),
            Wait(seconds=2.5, label="the flap stays shut"),
        ),
        expect=(Check(_flap_shut, needs=("flap",)),),
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
        "examples/03_rope/00_cable.py",
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
        "examples/03_rope/03_cable_memory.py",
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
        "examples/03_rope/04_twist.py",
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
        "examples/03_rope/02_superdex.py",
        needs=("superdex",),
        camera=Orbit(Bench(-0.10, -0.06, 0.10), distance=1.25, azimuth=-30.0, elevation=30.0, fov=40.0),
        warmup=1.0,
        layout=Layout(collapse=("Simulation", "Arm: left")),
        beats=(
            Wait(seconds=0.5, label="the right hand over the SuperDex cable's middle"),
            # The cable's axis lies 11.2 mm up; the tool centre closes 10 mm over it, as in 00_cable.
            Drag("right", "z", to=(0.550, -0.011, 0.771), seconds=1.2, label="drag the gizmo down onto it"),
            Wait(seconds=0.3),
            # Close would shut the jaws through it: SuperDex's push on the fingers is not fed back.
            Slider(text="Grip amount", value=0.80, seconds=0.8, within=RIGHT, label="Grip amount 0.80: jaws on it"),
            Wait(seconds=1.0, mark="closed"),
            Drag("right", "z", to=(0.550, -0.011, 0.891), seconds=1.8, label="drag it up, slowly"),
            Wait(seconds=2.0, mark="lifted", label="held by friction, nothing pinned"),
        ),
        # The node in the jaws. Measured 113 mm in tests/test_superdex_cable_robot.
        expect=(Rose(Patch("cable", near=(0.550, -0.011, 0.761)), 0.08, at="lifted"),),
    ),
    Take(
        "examples/17_knots/00_pull_knot_tight.py",
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
        "examples/18_bands/00_band_on_pegs.py",
        needs=("superdex",),
        budget=1_300_000,  # 17 s with the whole arm in view: 966 KB at quality 46 at the default
        camera=Orbit(Bench(0.03, -0.20, 0.17), distance=1.0, azimuth=-62.0, elevation=28.0, fov=40.0),
        warmup=1.5,  # the band settles on the upper peg
        layout=Layout(collapse=("Simulation", "Arm: left")),
        beats=(
            Wait(seconds=0.5, label="a band on the upper peg, the right hand in front of it"),
            Press(text="Close", within=RIGHT, label="Close: the shut jaws are the hook"),
            Wait(seconds=0.6),
            # tests/test_band_on_pegs's path, at about its 50 mm/s: through the loop, down past the
            # lower peg, in under it to 1 mm off the board, up, and back out along the jaws.
            Drag("right", "x", to=Bench(0.030, -0.20, 0.201), seconds=1.2, label="into the loop"),
            Drag("right", "z", to=Bench(0.030, -0.20, 0.105), seconds=2.2, label="down past the lower peg"),
            Drag("right", "x", to=Bench(0.082, -0.20, 0.105), seconds=1.2, label="in under it"),
            Drag("right", "z", to=Bench(0.082, -0.20, 0.120), seconds=0.5, label="up"),
            Drag("right", "x", to=Bench(-0.025, -0.20, 0.120), seconds=2.3, label="back out"),
            # The gizmo's arrows and arcs reach 0.14 m along +x from the hand: from 25 mm off they sat over
            # the lower peg and hid the band on it, so the hand goes on back until they end short of its tip.
            Drag("right", "x", to=Bench(-0.13, -0.20, 0.120), seconds=1.2, label="and clear of the pegs"),
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
# ketchup, the honey, the dough and the roller is the hand.

TOOL_DOWN = (0.0, 1.0, 0.0, 0.0)
"""The YAM's tool pointing down, its jaws closing along world y (xyzw). Its home points the tool along +x."""


def _turn(arm: str, quat, seconds: float, label: str) -> Do:
    """The arm's gizmo turned to ``quat`` over ``seconds`` of clip, filmed, and the hand turning after it.

    What a drag of the gizmo's rotation ring writes, written through the panel's
    own ``aim``: no beat drags a ring, so the clip shows the gizmo turning with
    the pointer off it.
    """

    def turn(film):
        import numpy as np

        from .page import ease

        panel = film.session.arms[arm]
        a = np.asarray(panel.cursor_quat, dtype=np.float64)
        b = np.asarray(quat, dtype=np.float64)
        b = -b if float(a @ b) < 0.0 else b
        half = math.acos(min(1.0, float(a @ b)))
        frames = max(1, round(seconds * film.take.fps))
        for k in range(1, frames + 1):
            t = ease(k / frames)
            q = b if half < 1e-6 else (math.sin((1.0 - t) * half) * a + math.sin(t * half) * b) / math.sin(half)
            panel.aim(panel.cursor_pos, q / np.linalg.norm(q), moving=k < frames)
            film.advance()
        return {"turned_deg": round(math.degrees(2.0 * half), 1)}

    return Do(fn=turn, label=label)


def _dolly(camera: Orbit | Camera, seconds: float, label: str) -> Do:
    """The view moved to ``camera`` over ``seconds`` of clip, filmed: what a person's scroll and orbit do.

    For a take whose first drag needs the hand in view at its home, far over
    what it then works on: the page opens wide and closes in once the hand is down.
    """

    def dolly(film):
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

    return Do(fn=dolly, label=label)


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


def _rice_in_tray(film) -> tuple[bool, str]:
    """``00_sweep_rice``'s readout: the share of the rice inside the tray, boxed as the example boxes it."""
    import numpy as np

    from robosimstudio_materials.mpm.measures import fill_fraction

    lo, hi = np.asarray(film.bench((-0.14, -0.22, -0.005))), np.asarray(film.bench((0.14, -0.02, 0.03)))
    fill = fill_fraction(np.asarray(film.sim["rice"].particle_q, dtype=np.float64), lo, hi)
    return fill >= 0.5, f"{100.0 * fill:.1f}% of the rice is in the tray (tests/test_sweep_rice asks half)"


def _ketchup_off(film) -> tuple[bool, str]:
    """``01_wipe_spill``'s readout: the share of the ketchup off the patch it settled on, at any height."""
    import numpy as np

    from robosimstudio_materials.mpm.measures import spill_fraction

    sauce = film.sim["ketchup"]
    rest = sauce.entity.rest["particle_q"][sauce.particles]
    pad = 0.5 * float(sauce.spec.particle_spacing)
    lo, hi = rest.min(axis=0) - np.array((pad, pad, 0.0)), rest.max(axis=0) + np.array((pad, pad, 0.03))
    off = spill_fraction(np.asarray(sauce.particle_q, dtype=np.float64), lo, hi)
    return off >= 0.5, f"{100.0 * off:.1f}% of the ketchup is off its patch (tests/test_wipe_spill asks half)"


def _dough_height(film) -> float:
    """``01_press_dough``'s readout: the puck's height as a share of the height it settled at [%]."""
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
        "examples/10_granular/00_sweep_rice.py",
        # From the front and the tray's side: the heap comes toward the lens into the tray, and the arm stays behind it.
        camera=Orbit(Bench(0.0, -0.29, 0.08), distance=1.1, azimuth=40.0, elevation=35.0, fov=42.0),
        warmup=1.0,  # the heap settles before the first frame
        layout=Layout(collapse=("Simulation", "Arm: left")),
        budget=1_800_000,  # nine gestures and a sweep: 18 s of clip
        beats=(
            Wait(seconds=0.5, label="the right hand over the heap, pointing forward"),
            DragTo("right", Bench(0.0, -0.45, 0.12), seconds_per_m=4.5, label="over the pusher"),
            _turn("right", TOOL_DOWN, 1.0, "turn the gizmo: the hand points down"),
            Drag("right", "z", to=At("pusher"), seconds=1.6, label="down onto the pusher"),
            Press(text="Close", within=RIGHT, label="Close: the jaws take its 20 mm faces"),
            Wait(seconds=1.0, mark="closed"),
            _note("jaws_mm", _jaws, "the jaw opening"),
            # tests/test_sweep_rice walks the hand at 0.15 m/s; eased over 3 s, these 290 mm peak at 0.145.
            Drag("right", "y", to=Bench(0.0, -0.16, 0.02), seconds=3.0, label="sweep into the tray at a walking pace"),
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
        "examples/11_liquid/01_wipe_spill.py",
        # From the front and the bench's middle: the push runs left to right toward the lens, the ridge forms on the
        # wiper's face this view looks at, and the arm stays side-on as the hand backs away at the end.
        camera=Orbit(Bench(0.0, -0.22, 0.09), distance=1.05, azimuth=50.0, elevation=36.0, fov=42.0),
        warmup=1.0,  # the puddle settles before the first frame
        layout=Layout(collapse=("Simulation", "Arm: left")),
        budget=1_200_000,
        beats=(
            Wait(seconds=0.5, label="the right hand over the wiper, pointing down"),
            # The pads' tips on the bench, where tests/test_wipe_spill closes: the tool centre 16.5 mm up.
            Drag("right", "z", to=At("wiper", dz=0.0015), seconds=1.4, label="drag the gizmo down onto the wiper"),
            Press(text="Close", within=RIGHT, label="Close: the jaws take its 20 mm faces"),
            Wait(seconds=0.8, mark="closed"),
            _note("jaws_mm", _jaws, "the jaw opening"),
            # The test walks the cursor at 0.09 m/s; eased over 3 s, these 180 mm peak at 0.09.
            Drag("right", "y", 0.18, seconds=3.0, label="push the puddle toward the middle of the bench"),
            Wait(seconds=0.8, mark="pushed", label="the ketchup is off its patch"),
            Press(text="Open", within=RIGHT, label="Open"),
            Drag("right", "z", 0.08, seconds=1.0, label="lift away"),
            Drag("right", "x", -0.10, seconds=1.0, label="and back: the ridge stays where the wiper left it"),
            Wait(seconds=1.0),
        ),
        expect=(
            Check(_jaws_on(20.0, 3.0)),
            Check(_carried("wiper", 0.15, upright=0.01, at="pushed"), needs=("wiper",)),
            Check(_ketchup_off),
        ),
    ),
    Take(
        "examples/12_dough/01_press_dough.py",
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
        "examples/11_liquid/00_pour_honey.py",
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
    Take(
        "examples/12_dough/00_roll_dough.py",
        camera=Orbit(Bench(-0.03, -0.31, 0.10), distance=1.0, azimuth=-40.0, elevation=32.0, fov=40.0),
        warmup=1.0,
        layout=Layout(collapse=("Simulation", "Arm: left")),
        # 1.5 s of scene a second: the three passes are 2.4 s of scene each at the test's pace, and at
        # real time the clip ran 22 s and 1255 KB at the bottom of the quality ladder.
        speed=1.5,
        budget=1_200_000,  # nine drags and a press, the hand in the middle of the frame throughout
        beats=(
            Wait(seconds=0.4, label="the right hand over the roller's handle"),
            Drag("right", "z", to=(0.460, -0.310, 0.820), seconds=1.0, label="drag the gizmo down onto the handle"),
            Wait(seconds=0.2),
            Press(text="Close", within=RIGHT, label="Close: the jaws on the handle"),
            Wait(seconds=0.5, mark="closed"),
            Drag("right", "z", to=(0.460, -0.310, 0.850), seconds=0.6, label="lift it out of its rest"),
            Drag("right", "x", to=(0.515, -0.310, 0.850), seconds=0.6, label="over the near edge of the block"),
            # The cursor 88 mm over the bench: the drum's underside runs about 11 mm over it (tests/test_roll_dough).
            Drag("right", "z", to=(0.515, -0.310, 0.838), seconds=0.4, label="down onto the dough"),
            # 160 mm in 2.4 s of scene, eased: under 0.11 m/s, a third of what the MPM's grid allows a tool.
            Drag("right", "x", to=(0.675, -0.310, 0.838), seconds=1.6, label="roll it across"),
            Drag("right", "x", to=(0.515, -0.310, 0.838), seconds=1.6, label="back"),
            Drag("right", "x", to=(0.675, -0.310, 0.838), seconds=1.6, label="and across again"),
            Wait(seconds=0.3, mark="rolled"),
            Drag("right", "z", to=(0.675, -0.310, 0.890), seconds=0.7, label="lift away: the dough stays flat"),
            Wait(seconds=0.7),
        ),
        expect=(Held("roller", since="closed", at="rolled"), Check(_rolled_flat, needs=("dough",))),
    ),
)

ROBOT_BAGS: tuple[Take, ...] = (
    # The right hand pinching the top layer's lip by contact and opening the mouth, nothing holding the bag
    # down; tests/test_open_a_bag.py drives the same path through the panel. Seen from the mouth's side and
    # a little in front, so the opening shows under the hand: the gizmo's arrows and arcs point +x, +y, +z.
    Take(
        "examples/16_bags/00_open_a_bag.py",
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
    """``00_scrub_the_bench``'s readout: the share of the square's 5 mm cells a bristle tip touched while down."""
    lo, hi = film.bench((-0.025, -0.27, 0.0))[:2], film.bench((0.075, -0.17, 0.0))[:2]
    share = float(film.sim["brush"].entity.scrubbed(lo, hi))
    return share >= 0.5, f"{100.0 * share:.1f}% of the square scrubbed (tests/test_scrub_bench asks half)"


ROBOT_BRUSH = (
    Take(
        "examples/14_brush/00_scrub_the_bench.py",
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
# plate rather than 4 and a lowering aimed to stop the held plate 2 mm over the bench (it stops 2.8 mm
# over and drops that far when the box is unticked). What holds the plate is the vacuum: the Seal readout
# on the panel says sealed and the mass it holds.
PLATE_HALF = 0.003
"""The plate's half-thickness [m]: 6 mm thick."""
INTO_PLATE = 0.003
"""How far under the plate's top the gizmo is put [m]. The cup stops on the plate with its face 0.1 to 0.2 mm
over it (the Seal readout's gap once ticked, inside the seal's 2 mm), and the panel reads the hand 6 mm behind
the gizmo: too far to put the gizmo back on the hand, so the arm keeps the face down on the plate."""
SQUARE = (-0.11, -0.22, 0.11, 0.0)
"""``00_pick_a_plate``'s square in the bench frame [m]: x0, y0, x1, y1."""


def _seal(film) -> str:
    """The right arm's **Seal** readout, as the page shows it."""
    return film.page.text("Seal", RIGHT)


def _plate_over_bench(clearance: float):
    """Where the gizmo goes for the held plate's underside to stop ``clearance`` [m] over the bench."""

    def there(film) -> tuple[float, float, float]:
        cursor = film.where(film.gizmo("right"))
        underside = film.position("plate")[2] - PLATE_HALF
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
    lo, hi = np.asarray(film.bench((*SQUARE[:2], 0.0))[:2]), np.asarray(film.bench((*SQUARE[2:], 0.0))[:2])
    inside = bool(np.all(plate[:2] >= lo) and np.all(plate[:2] <= hi))
    top = 1000.0 * (plate[2] + PLATE_HALF - film.bench((0.0, 0.0, 0.0))[2])
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
        "examples/15_suction/00_pick_a_plate.py",
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
TAPE_TOP = 0.05
"""The example's box top over the bench [m]."""
TAPE_CLOSE = TAPE_TOP + 0.0125 + 0.004
"""Where the tool centre closes: 4 mm over the middle of the 24 mm tab, the pads clear of the box top."""


def _peeled(film) -> tuple[bool, str]:
    """``00_peel_a_label``'s readout: the share of the label's bonds that have let go."""
    label = film.sim["label"].entity
    share = float(label.peeled())
    front = 1000.0 * float(label.peel_front())
    return share >= 0.8, f"{100.0 * share:.1f}% of the bonds let go, the front {front:.0f} mm along (the test asks 80%)"


ROBOT_TAPE = (
    Take(
        "examples/19_tape/00_peel_a_label.py",
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
            Drag(
                "right", "z", to=Bench(z=TAPE_CLOSE), seconds=1.2, label="drag the hand down: its jaws straddle the tab"
            ),
            Wait(seconds=0.3),
            Press(text="Close", within=RIGHT, label="Close: the pads squeeze the tab"),
            Wait(seconds=0.8, mark="closed"),
            Drag(
                "right", "z", to=Bench(z=TAPE_TOP + 0.03), seconds=1.0, label="drag it up 30 mm: the first rows let go"
            ),
            *(
                beat
                for k in range(1, 8)
                for beat in (
                    Drag(
                        "right",
                        "y",
                        to=Bench(y=-0.04 - 0.02 * k),
                        seconds=0.5,
                        approach=0.3,
                        label="back 20 mm toward the arm",
                    ),
                    Drag(
                        "right",
                        "z",
                        to=Bench(z=TAPE_TOP + 0.03 + 0.01 * k),
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
# is that force. tests/test_fold_paper_robot.py and tests/test_scrub_a_stain.py drive the same paths.

FOLD = "examples/13_paper/00_fold_paper.py"
FOLD_EDGE = -0.435
"""The sheet's near edge in the bench frame [m]: 35 mm past the bench's right edge."""
FOLD_CREASE = FOLD_EDGE + 0.297 / 3
FOLD_PINCH = (-0.02, FOLD_EDGE + 0.0135)
"""Where the jaws' middle closes on the overhang, bench frame [m]: 35 mm in from the corner, 13.5 in from the edge."""


def _q(a, b):
    """``a`` after ``b``, both xyzw."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _about(axis, degrees):
    """A quaternion (x, y, z, w) turning ``degrees`` about the unit ``axis``; not the arm-gizmo ``_turn`` above."""
    s, c = math.sin(math.radians(degrees) / 2), math.cos(math.radians(degrees) / 2)
    return (axis[0] * s, axis[1] * s, axis[2] * s, c)


FOLD_JAWS = _q(_q(_about((0, 1, 0), 10.0), (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))), _about((0, 0, 1), 90.0))
"""The right hand's pinch: the tool along +x tipped 10 degrees down, the jaws closing vertically."""


def _fold_jaws(film):
    """The right hand's jaws' middle [world]: the mean of its finger boxes, as tests/test_fold_paper_robot reads it."""
    import newton
    import numpy as np

    from robosimstudio.core.math import quat_to_mat

    sim = film.sim
    index = sim.scene.embodiment.arms["right"]
    body = sim.model.shape_body.numpy()
    kinds = sim.model.shape_type.numpy()
    boxes = [
        s
        for side in index.finger_bodies
        for s in np.flatnonzero(np.isin(body, side))
        if int(kinds[s]) == int(newton.GeoType.BOX)
    ]
    body_q, xform = sim.state.body_q.numpy(), sim.model.shape_transform.numpy()
    return np.mean([body_q[body[s]][:3] + quat_to_mat(body_q[body[s]][3:7]) @ xform[s][:3] for s in boxes], axis=0)


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


def _jaws_to(film, point, speed: float, steps: int = 120, quat=None):
    """Put the right hand's jaws' middle at ``point``: the cursor is the tool centre."""
    import numpy as np

    panel = film.session.arms["right"]
    offset = _fold_jaws(film) - panel.ik.ee_pose(film.sim.state)[0]
    yield from _go(film, "right", np.asarray(point) - offset, speed, steps, quat=quat)


def _wait(n: int):
    for _ in range(n):
        yield


def _fold_surface(film) -> float:
    import numpy as np

    rest = np.asarray(film.sim["paper"].particle_q, dtype=np.float64)
    return float(np.median(rest[rest[:, 1] < film.bench((0.0, -0.40, 0.0))[1]][:, 2]))


FOLD_STAMP = _q(_about((0, 0, 1), -30.0), (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)))
"""The left hand's stamp: the tool along +x turned 30 degrees toward the fold, which it reaches that way."""


def _fold_hold(film):
    """The left hand onto the far end, its flat underside 3 mm into the bench."""
    hold = film.bench((0.05, FOLD_EDGE + 0.262, 0.003))
    yield from _go(film, "left", (hold[0], hold[1], 0.80), 0.15, steps=600, tolerance=0.004)
    yield from _go(film, "left", hold, 0.06, steps=60, tolerance=0.004)


def _fold_in(film):
    """Jaws vertical, level with the overhang and short of its corner, then in along it."""
    film.surface = _fold_surface(film)
    panel = film.session.arms["right"]
    yield from _go(film, "right", panel.ik.ee_pose(film.sim.state)[0], 0.05, steps=600, tolerance=0.004, quat=FOLD_JAWS)
    corner, pinch = film.bench((-0.055 - 0.06, FOLD_PINCH[1], 0.0)), film.bench((*FOLD_PINCH, 0.0))
    yield from _jaws_to(film, (corner[0], corner[1], film.surface), 0.06)
    yield from _jaws_to(film, (pinch[0], pinch[1], film.surface), 0.05)


def _fold_over(film):
    """The carry on an arc about the fold line, the hand turning over with it; laid, drawn on, lifted clear."""
    import numpy as np

    pinch = film.bench((*FOLD_PINCH, 0.0))
    crease = film.bench((0.0, FOLD_CREASE, 0.0))[1]
    radius, z = crease - pinch[1], film.surface
    for k in range(1, 13):
        turn = math.pi * k / 12
        quat = _q(FOLD_JAWS, _about((0, 0, 1), -math.degrees(turn)))
        point = (pinch[0], crease - radius * math.cos(turn), z + (radius + 0.015) * math.sin(turn))
        yield from _jaws_to(film, point, 0.06, 120 if k < 12 else 60, quat=quat)
    yield from _jaws_to(film, (pinch[0], crease + radius, z + 0.03), 0.03)
    yield from _jaws_to(film, (pinch[0], crease + radius, z + 0.020), 0.02, 60)
    yield from _wait(30)
    yield from _jaws_to(film, (pinch[0], crease + radius + 0.03, z + 0.03), 0.03)
    yield from _wait(20)
    film.stamp_y = float(np.asarray(film.sim["paper"].particle_q)[:, 1].min()) + 0.02  # the loop's nose, plus 20 mm
    yield from _jaws_to(film, (pinch[0], crease + radius + 0.03, z + 0.05), 0.02)


def _fold_stamps(film):
    """Up off the far end, turned toward the fold, and three stamps along it 20 mm in from the loop's nose."""
    hold, y = film.bench((0.05, FOLD_EDGE + 0.262, 0.06)), film.stamp_y
    up, down = film.bench((0.0, 0.0, 0.06))[2], film.bench((0.0, 0.0, 0.009))[2]
    yield from _go(film, "left", hold, 0.08, quat=FOLD_STAMP)
    for x in (0.0, 0.055, 0.11):
        at = film.bench((x, 0.0, 0.0))[0]
        yield from _go(film, "left", (at, y, up), 0.10)
        yield from _go(film, "left", (at, y, down), 0.04, steps=40)
        yield from _wait(15)
        yield from _go(film, "left", (at, y, up), 0.06)


def _fold_up(film):
    """Up 25 mm with the jaws still shut: opening to half brings the lower pad 25 mm back down."""
    import numpy as np

    yield from _jaws_to(film, _fold_jaws(film) + np.array([0.0, 0.0, 0.025]), 0.02)


def _fold_out(film):
    """Out 40 mm past the edge with the jaws open, up 80 mm, and the left hand away."""
    import numpy as np

    yield from _jaws_to(film, _fold_jaws(film) + np.array([0.0, 0.04, 0.0]), 0.05)
    yield from _jaws_to(film, _fold_jaws(film) + np.array([0.0, 0.0, 0.08]), 0.08)
    yield from _go(film, "left", film.bench((0.11, -0.15, 0.12)), 0.08)


def _paper_folded(film) -> tuple[bool, str]:
    """The sheet's fold, a third of the way along it, kept 2.2 rad once the hands had gone, and it lies folded."""
    from robosimstudio.assetlib.semantics.snapshot import Snapshot
    from robosimstudio.assetlib.vocabulary.measures import cloth_fold
    from robosimstudio_materials.paper import crease_angle

    sim = film.sim
    sim["paper"].entity.update_probe(sim.state, sim.step_count)
    snapshot = Snapshot(sim.make_context())
    line = {"crease": {"axis": 1, "at": film.bench((0.0, FOLD_CREASE, 0.0))[1]}}
    angle = float(crease_angle(snapshot, "paper", line)["angle"])
    extent = float(cloth_fold(snapshot.ctx, "paper", axis=1)[0])
    return (
        angle >= 2.2 and extent < 0.83,
        f"the fold kept {angle:.2f} rad (need 2.2), span {extent:.3f} (need under 0.83)",
    )


SCRUB = "examples/01_cloth/03_scrub_a_stain.py"
SCRUB_TIP = 0.0167
"""The closed fingertips hang this far under the tool centre, pointing down [m]."""


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
        x, y, _ = film.bench((-0.09, -0.08, 0.0))
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
    (x0, y, _), flat = film.bench((-0.09, -0.08, 0.0)), film.bench((0.06, 0.0, 0.0))[0]
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
        # From in front of the bench and to its right, 1.35 m off: the overhang, both hands, the
        # carry's arc and the sheet in thirds all in view, the panel's Crease readouts beside them.
        camera=Orbit(Bench(0.0, -0.22, 0.12), distance=1.40, azimuth=-25.0, elevation=34.0, fov=38.0),
        warmup=0.5,
        fps=8,
        speed=2.0,  # two minutes of scene: two hands, a carry and three stamps
        budget=2_000_000,
        # The left arm's folder stays open: collapsed, its Close is not on the page to press, and the
        # left hand held the far end open-fingered; the sheet slid 37 mm and no take folded.
        layout=Layout(collapse=("Simulation", "Material", "Physics")),
        beats=(
            Fold(title="Simulation", open=False),
            Fold(title="How to drive this", open=False),
            Wait(seconds=0.4, label="an A4 sheet hangs 35 mm over the bench's right edge"),
            Press(text="Close", within="Arm: left", label="Close the left hand"),
            _per_step(_fold_hold, "it holds the far end down"),
            _per_step(_fold_in, "the right hand down and in: the jaws straddle the overhanging corner"),
            Press(text="Close", within=RIGHT, label="Close: the pads pinch the paper"),
            Scroll(pixels=-600, seconds=0.4),  # back to the top: the Crease readouts lead the panel
            Wait(seconds=0.6, mark="closed"),
            _per_step(_fold_over, "carry the edge up and over, turning the hand over, and lay it on the middle"),
            _per_step(_fold_stamps, "the left hand turns toward the fold and stamps it in three places"),
            _per_step(_fold_up, "up 25 mm"),
            Slider(text="Grip amount", value=0.5, seconds=0.6, within=RIGHT, label="Grip amount 0.5: the pads part"),
            Scroll(pixels=-600, seconds=0.3),
            Wait(seconds=0.3),
            _per_step(_fold_out, "drawn out past the edge, and away"),
            Wait(seconds=1.5, label="it stays folded"),
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
            Scroll(pixels=-600, seconds=0.4),  # back to the top: the Scrub readouts lead the panel
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

TAKES: tuple[Take, ...] = (
    *FINAL,
    *DRAFTS,
    *ROBOT_PARTICLES,
    *ROBOT_BAGS,
    *ROBOT_BRUSH,
    *ROBOT_SUCTION,
    *ROBOT_TAPE,
    *ROBOT_CONTACT,
)

KEPT: dict[str, str] = {
    "16_bags/01_open_a_bag_scored": "the same bag, hands and gesture as 00_open_a_bag with a goal beside them; its clip is that one's",
    "10_granular/01_sweep_rice_scored": "the same rice, pusher and tray as 00_sweep_rice with a goal beside them; its clip is that one's",
    "18_bands/01_band_on_pegs_scored": "the same pegboard, band and hook as 00_band_on_pegs with a goal beside them; its clip is that one's",
    "14_brush/01_scrub_scored": "the same brush, handle and square as 00_scrub_the_bench with a goal beside them; its clip is that one's",
    "15_suction/01_pick_scored": "the same plate, cup and square as 00_pick_a_plate with a goal beside them; its clip is that one's",
    "19_tape/01_peel_a_label_scored": "the same box, label and peel as 00_peel_a_label with a goal beside them; its clip is that one's",
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

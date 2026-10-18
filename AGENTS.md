# For the agent working in this repository

RoboSimStudio is a Newton-native simulation toolkit. Packages:

- `robosimstudio/`: the core API; builds and steps scenes.
- `robosimstudio_studio/`: a browser page over any scene.
- `robosimstudio_bench/`: tasks and their scoring.
- `robosimstudio_policy_agent/`: policies as graphs of robot tools.
- `robosimstudio_real2sim_agent/`: the live page that rebuilds a `scene.py`
  on every save, and the `./ctl` command line onto it.

## Read first

1. [`README.md`](README.md): what is here and how to run it.
2. [`examples/00_getting_started/README.md`](examples/00_getting_started/README.md)
   (`Scene`, `Sim`, `Ref`, the constructors) and each example folder's README. Poses
   are in the cell's work-surface frame: `at=(0, 0, 0.2)` is 200 mm over the
   middle of the bench, whatever bench (frames and units:
   [`examples/live/REFERENCE.md`](examples/live/REFERENCE.md#frames-and-units)).
3. `examples/`: numbered scripts, shortest first. `examples/01_cloth/cloth.py`,
   `examples/02_soft/soft.py` and `examples/03_rope/rod/` are the shipped cloth,
   soft-body and rope implementations; read them to see how each works.

## Checks

- `uv run python scripts/testrun.py --tier fast`: no GPU, gates every commit.
- `uv run --extra cpu --no-group rtx python scripts/testrun.py --tier cpu`: no
  GPU; builds and steps scenes on a CPU (MuJoCo's C library, SuperDex).
- `uv run --extra gap python scripts/testrun.py --tier medium`: needs the GPU;
  builds and steps scenes.
- `uvx ruff@0.16.1 check . && uvx ruff@0.16.1 format . --check`.
- `examples/live/run/` is not committed. A commit message is a lowercase
  `area:` prefix and one declarative sentence
  (`studio: the room, and lights that match the render`).

## Real to sim: a real bench, scene or clip, rebuilt on the live page

When asked to rebuild a real scene from a photo or video, match a reference,
twin a real cell, or build a scene interactively, use the live page in
[`examples/live/`](examples/live/README.md) instead of writing a script and
running it blind: edit `scene.py`, and the page rebuilds it on every save.
From `examples/live/`:

```bash
./ctl status                          # a page is up (names, positions, last error), or it exits 2
./ctl up                              # if not: start one; prints the URL (--share for a public one)
./ctl ref frames <clip> --at 0:start,4.5:lid_open,end:end   # stills named by their second, a contact sheet, ref/timeline.json
./ctl snapshot --camera <cam> --beside ref/<still>          # the render beside the still, full size
./ctl compare --camera <cam> --key case=red --key socket=@1240,410 --note "<why>"   # size, offset, overlap per thing, and a look sheet
./ctl check --camera <cam> --key ...  # the gate: PASS/FAIL with the number
./ctl eval "t.play()" --record take1 --camera <cam>         # film a routine; it is compared to the clip when it ends
./ctl play tidy --record take1        # the page's Run button: scene.py binds routine or routines={name: fn}
./ctl log                             # every reading so far, with the scene code it measured
./ctl catalog robots cells            # the names rss.robot() and Scene(cell=) accept
./ctl eval "sim['<name>'].position"   # python against the live scene
```

A snapshot is path traced and draws the room, the mat and the textures; the
page draws the bench and what you added.

The method and what counts as a finished twin are in
[`.claude/skills/real-to-sim/SKILL.md`](.claude/skills/real-to-sim/SKILL.md)
(the `real-to-sim` skill for Claude Code). The `scene.py` contract, every
command and every reading are in
[`examples/live/REFERENCE.md`](examples/live/REFERENCE.md).

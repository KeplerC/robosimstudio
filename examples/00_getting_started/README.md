# Getting started

How to assemble a scene, build it, step it and read it back.

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Hello world

![Hello world](media/00_hello_world.webp)

The smallest RoboSimStudio program: a bench, a robot arm, and a cube that lands under its hand.

`build()` compiles the scene, 150 steps let the cube land, and `rss.view`
opens the studio. Drag the hand down onto the cube, press **Close** and lift: the
fingers hold it by friction, which needs `rss.MUJOCO_500HZ`; under the default
physics a hand can push a body but not pinch it. Or click the cube to drag it by
a handle; it stays fully dynamic.

```bash
uv run examples/00_getting_started/00_hello_world.py
```

Source: [`00_hello_world.py`](00_hello_world.py)

## Core concepts

![Core concepts](media/01_core_concepts.webp)

A scene you declare, a sim that runs it, and a ref on each thing you added.

A `rss.Scene` is a list of specs in a named place;
`rss.Scene.build` compiles it into a `rss.Sim`,
which owns the state and steps it; each `add` returns a
`rss.Ref`, which reads the live state each time it is asked.
Every `at=`, the robot's mount included, is measured from the middle of the
place's work surface, so the same scene works on `"bench_075"` and `"void"`.

```bash
uv run examples/00_getting_started/01_core_concepts.py
```

Source: [`01_core_concepts.py`](01_core_concepts.py)

<!-- examples:end -->

## No GPU?

See [Installation](../install.md#without-a-gpu).

# Contributing to RoboSimStudio

## Set up

You need [uv](https://docs.astral.sh/uv/). An NVIDIA GPU is needed only for the
`medium` and `slow` tiers. Without one, `export UV_NO_GROUP=rtx` and add
`--extra cpu` to the `uv run` lines below; see [Installation](examples/install.md#without-a-gpu).

```bash
git clone https://github.com/robosimstudio/robosimstudio.git
cd robosimstudio
uv run examples/00_getting_started/00_hello_world.py   # builds the environment on first run
uvx pre-commit install                                  # ruff on every commit
```

## Check your change

```bash
uv run python scripts/testrun.py --tier fast     # no GPU; CI runs this
uv run --extra cpu --no-group rtx python scripts/testrun.py --tier cpu   # no GPU; real scenes on a CPU; CI runs this
uv run --extra gap python scripts/testrun.py --tier medium   # needs the GPU; builds and steps scenes
uvx ruff@0.16.1 check . && uvx ruff@0.16.1 format . --check
```

Every test module declares `TIER = "fast" | "medium" | "slow" | "cpu"`;
`scripts/testrun.py` fails, naming the module, on one without it. Tests that
build a scene on a GPU are `medium`, and tests that build one on a CPU are `cpu`
(the runner hides every GPU for `fast` and `cpu`, as CI's runners have none). CI runs `fast` and `cpu`, so run
`medium` on a GPU before changing simulation code.

## Where things go

- **A new example** is one script under `examples/<NN_category>/`. Line 1 of its
  docstring is the title, the next sentence says what it shows, and it defines
  `main()`. A looping clip of it goes in that folder's `media/` as
  `<script>.webp`; then `uv run python scripts/example_readmes.py` writes its
  section of the folder's README.
- **A clip** is a take in `scripts/recording/takes.py`: the example, the camera,
  what the cursor does and what the physics must have done; then
  `scripts/record_examples.py <name>` films it (see [`scripts/README.md`](scripts/README.md#media)).
- **New robots, kinds of thing, tasks and policies** live outside the library: a
  robot or kind registers itself (`examples/04_extending/`), a task is a document
  (`examples/06_tasks/`), and a policy is a graph folder (`examples/07_policies/`).
- **Imports go one way.** The library imports neither the studio nor the agent,
  and a lower layer never imports a higher one.
- **A number in a docstring was measured.** A new one should be too.

## Commits and pull requests

A commit message is a lowercase `area:` prefix and one declarative sentence:
`studio: the room, and lights that match the render`. Keep a pull request to one
change, say how you tested it, and include a picture when it changes what a
scene looks like.

By contributing you agree that your contribution is licensed under the
repository's Apache-2.0 licence.

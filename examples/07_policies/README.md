# Policies

A policy is a graph of tool calls and short Python scripts that drives a task's
robot, run by [graph-as-policy](https://github.com/KeplerC/graph-as-policy),
which defines the workflow format and the runtime. Every example here needs
`uv run --extra gap`.

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Pick and place

![Pick and place](media/00_pick_and_place.webp)

A policy graph is a folder holding a `workflow.json` and the scripts it names.

`pick_place/` reads where the cube and the square are, works out four hand
poses in one script node, and moves the arm with `robot.*` tools; `$ref`
passes one node's output to another's input. The task names the folder in
`(:graph pick "pick_place")`, resolved beside this script, and
`SolutionGraph` runs it for one episode, scored as the benchmark scores it.

```bash
uv run examples/07_policies/00_pick_and_place.py
```

Source: [`00_pick_and_place.py`](00_pick_and_place.py)

## Check and retry

![Check and retry](media/01_check_and_retry.webp)

A graph checks its own grasp and tries a missed one once more.

`pick_place_retry/` splits the job into two subgraphs. `pick` turns the jaws
to close across the cube's faces, lifts, and raises if the hand shut on air; its
`on_error` names that outcome, and `conditional_edges` route it to a second
copy of `pick`. `roll="held"` turns the wrist to the heading the graph asked
for. The cube starts somewhere new, turned, in each of four episodes.

```bash
uv run examples/07_policies/01_check_and_retry.py
```

Source: [`01_check_and_retry.py`](01_check_and_retry.py)

## Without privilege

![Without privilege](media/02_without_privilege.webp)

A graph finds the cube and the target through a camera, as a real robot would.

With `privileged=False` no `sim.*` tool is registered, so the graph has
camera frames and the arm's own state. `pick_place_camera/` back-projects the
overhead depth image, takes what stands 50 mm above the bench as the cube and
what stands 4 mm above it as the pad, and runs the retry example's pick, check
and place. `renderer="auto"` path traces the camera where the GPU can and
draws colliders where it cannot.

```bash
uv run examples/07_policies/02_without_privilege.py
```

Source: [`02_without_privilege.py`](02_without_privilege.py)

## Many worlds

![Many worlds](media/03_many_worlds.webp)

Run one graph in sixteen worlds at once and report its success rate.

A `JobSpec` names the task, the policy, the episodes and `num_worlds=16`.
The fleet runner builds sixteen copies of the scene in one model, steps them
together and batches the graphs' motions, each world running its own episode
and its own copy of the graph. It writes one results row per episode and the
rate with its 95% Wilson interval.

```bash
uv run examples/07_policies/03_many_worlds.py
```

Source: [`03_many_worlds.py`](03_many_worlds.py)

<!-- examples:end -->

## Nodes

A graph is a folder:

```
pick_place/
  workflow.json      # nodes, the edges between them, and subgraphs
  scripts/plan.py    # one script node: inputs in, a dict out
```

| Node | What it does |
| --- | --- |
| `tool` | Calls one tool with its `inputs` |
| `script` | Runs `run(ctx, **inputs)` from a file under `scripts/`; `ctx.tool(name, ...)` calls a tool |
| `subgraph` | Runs a named subgraph; its `on_error` names the outcome when a node in it raises |
| `noop`, `end` | A named outcome, and the end of the graph |

`{"$ref": "plan.grasp"}` passes one node's output to another's input. A
subgraph ends at a `noop` named for its outcome, and `conditional_edges` route
on it.

## Tools

| Family | Tools |
| --- | --- |
| `robot.*` | `go_to_pose`, `move_to_joints`, `execute_trajectory`, `go_home`, `set_grip`, `open_gripper`, `close_gripper`, `get_gripper`, `get_ee_pose`, `get_observation`, `get_camera_pose`, `grasp_frame`, `describe_arm`, `describe_gripper`, `describe_workspace`, `wait_steps` |
| `motion.*` | `plan_linear`, `plan_joint`, `describe_planner`; with a collision-aware planner (`motion="curobo"`), also `plan_to_pose` and `solve_ik_batch` |
| `sim.*` | `get_object_pose`, `get_object_obb`, `get_region_obb`, `check_grasp`, `check_success`, `query` and more: the simulator's own answers; `attach` on a scene with a deformable, `vacuum` on a robot with a suction hand |

## Privilege

`sim.*` tools are **privileged**: they answer from the simulator, which a real
robot cannot. `SolutionGraph(env, graph=..., privileged=False)` registers none
of them. That leaves camera frames, the arm's state, `robot.*`, `motion.*` and
the perception tools `geometry.*` and `sam3.*`.

## Running graphs

Besides a script, `rss.view(env)` opens the studio, whose policy panel lists the
task's graphs, and `scripts/bench_run.py --solve graph --graph pick
[--no-privileged]` runs one from a shell. `robosimstudio_policy_agent.SolutionGraphBatched`
is the batched runner behind many worlds.

A subgraph's `skill` may name a bundle from
[open-robot-skills](https://github.com/graph-robots/open-robot-skills). It is
optional: `generic` subgraphs need none. Clone it to `../open-robot-skills` (the
default in `pyproject.toml`'s `[tool.gap]`), or point `$GAP_SKILLS_PATH` or
`bench_run.py --skills` at it. `sam3.*` loads `facebook/sam3`, whose weights are
gated on Hugging Face.

[Tasks](../06_tasks/README.md) describes what a graph is scored against.

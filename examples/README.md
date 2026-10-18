# Examples

## [00 Getting Started](00_getting_started/README.md)

| [Hello world](00_getting_started/00_hello_world.py) | [Core concepts](00_getting_started/01_core_concepts.py) |
|---|---|
| <img src="00_getting_started/media/00_hello_world.webp" width="240"> | <img src="00_getting_started/media/01_core_concepts.webp" width="240"> |

## [01 Cloth Simulation](01_cloth/README.md)

| [Towel](01_cloth/00_towel.py) | [Fabrics](01_cloth/01_fabrics.py) | [Other engines](01_cloth/02_engines.py) |
|---|---|---|
| <img src="01_cloth/media/00_towel.webp" width="240"> | <img src="01_cloth/media/01_fabrics.webp" width="240"> | <img src="01_cloth/media/02_engines.webp" width="240"> |

## [02 Soft-Body Simulation](02_soft/README.md)

| [Sponge](02_soft/00_sponge.py) | [Tet mesh](02_soft/01_tet_mesh.py) | [Your own](02_soft/02_your_own_soft.py) |
|---|---|---|
| <img src="02_soft/media/00_sponge.webp" width="240"> | <img src="02_soft/media/01_tet_mesh.webp" width="240"> | <img src="02_soft/media/02_your_own_soft.webp" width="240"> |

## [03 Rope Simulation](03_rope/README.md)

| [Cable](03_rope/00_cable.py) | [Between posts](03_rope/01_between_posts.py) | [SuperDex](03_rope/02_superdex.py) |
|---|---|---|
| <img src="03_rope/media/00_cable.webp" width="240"> | <img src="03_rope/media/01_between_posts.webp" width="240"> | <img src="03_rope/media/02_superdex.webp" width="240"> |

| [A cable keeps its bend](03_rope/03_cable_memory.py) | [A cable carries a twist](03_rope/04_twist.py) |
|---|---|

## [04 Robots and Kinds of Your Own](04_extending/README.md)

| [Add a robot](04_extending/00_add_a_robot.py) | [A new kind](04_extending/01_new_kind.py) |
|---|---|
| <img src="04_extending/media/00_add_a_robot.webp" width="240"> | <img src="04_extending/media/01_new_kind.webp" width="240"> |

## [05 The Browser Studio](05_studio/README.md)

| [Your own panel](05_studio/00_your_own_panel.py) | [Render](05_studio/01_render.py) |
|---|---|
| <img src="05_studio/media/00_your_own_panel.webp" width="240"> | <img src="05_studio/media/01_render.webp" width="240"> |

## [06 Tasks and Scoring](06_tasks/README.md)

| [First task](06_tasks/00_your_first_task.py) | [Goals and rewards](06_tasks/01_goals_and_rewards.py) | [Episodes](06_tasks/02_episodes.py) |
|---|---|---|
| <img src="06_tasks/media/00_your_first_task.webp" width="240"> | <img src="06_tasks/media/01_goals_and_rewards.webp" width="240"> | <img src="06_tasks/media/02_episodes.webp" width="240"> |

## [07 Policies as Graphs](07_policies/README.md)

| [Pick and place](07_policies/00_pick_and_place.py) | [Check and retry](07_policies/01_check_and_retry.py) | [Many worlds](07_policies/03_many_worlds.py) |
|---|---|---|
| <img src="07_policies/media/00_pick_and_place.webp" width="240"> | <img src="07_policies/media/01_check_and_retry.webp" width="240"> | <img src="07_policies/media/03_many_worlds.webp" width="240"> |

## [08 Demos](08_demos/README.md)

| [Wipe the bar](08_demos/wipe_the_bar/wipe_the_bar.py) | [Route the cable](08_demos/route_the_cable/route_the_cable.py) | [Fold the towel](08_demos/fold_the_towel/fold_the_towel.py) |
|---|---|---|
| <img src="08_demos/wipe_the_bar/media/wipe_the_bar.webp" width="240"> | <img src="08_demos/route_the_cable/media/route_the_cable.webp" width="240"> | <img src="08_demos/fold_the_towel/media/fold_the_towel.webp" width="240"> |

## [09 Teleoperation](09_teleop/README.md)

| [Leader arm](09_teleop/00_leader_arm.py) | [Force feedback](09_teleop/01_force_feedback.py) |
|---|---|
| <img src="09_teleop/media/00_leader_arm.webp" width="240"> | <img src="09_teleop/media/01_force_feedback.webp" width="240"> |

## [10 Granular](10_granular/README.md)

| [Sweep rice into the tray](10_granular/00_sweep_rice.py) | [Sweep rice, scored](10_granular/01_sweep_rice_scored.py) |
|---|---|

## [11 Liquid](11_liquid/README.md)

| [Pour honey into a bowl](11_liquid/00_pour_honey.py) | [Wipe a spill](11_liquid/01_wipe_spill.py) |
|---|---|

## [12 Dough](12_dough/README.md)

| [Roll the dough flat](12_dough/00_roll_dough.py) | [Press a dough ball](12_dough/01_press_dough.py) |
|---|---|

## [13 Paper](13_paper/README.md)

| [Fold a sheet of paper](13_paper/00_fold_paper.py) | [Close a cardboard flap](13_paper/01_box_flap.py) |
|---|---|

## [16 Bags](16_bags/README.md)

| [Open a bag](16_bags/00_open_a_bag.py) | [Open a bag, scored](16_bags/01_open_a_bag_scored.py) |
|---|---|
| <img src="16_bags/media/00_open_a_bag.webp" width="240"> | |

## [Live Scene Editing](live/README.md)

| [The live page](live/README.md) |
|---|
| <img src="live/media/live_page.webp" width="240"> |

## Running them

Run one by path from the repository root; all but the headless render open the
studio in your browser:

```bash
uv run examples/01_cloth/00_towel.py
```

No script takes a flag (except `09_teleop/`, which takes the leader arm's port
and calibration); edit the locals at the top of `main()`. Folders are numbered
in reading order.

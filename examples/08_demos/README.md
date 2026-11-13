# Demos

Three bimanual YAM jobs from the [project page](https://robosimstudio.github.io/). Each opens the studio, and
**Run** under *Routine* plays the job by dragging the arms' gizmos and pressing
their **Grip** buttons the way a person would; you can stop it and take over by
hand.

| Demo | The job | What holds the thing |
| --- | --- | --- |
| [`wipe_the_bar/`](wipe_the_bar/README.md) | Take a sponge off the mat, wipe a marker stripe off a grab bar while the other arm steadies it, and put the sponge back. | The jaws close on the foam by contact; once they have, a latch holds it through the strokes and lets go on **Open**. |
| [`route_the_cable/`](route_the_cable/README.md) | Two arms lay a 600 mm cable past two spools and seat its end in a port. | Friction alone. |
| [`fold_the_towel/`](fold_the_towel/README.md) | Pinch the far edge of a terry towel, fold it over in a low arc and lay it down. | Friction alone. |

```bash
uv run examples/08_demos/route_the_cable/route_the_cable.py
```

Two more work a screw thread, which is real geometry here: a ridge on each part,
and nothing else joining them
([`robosimstudio_materials/thread.py`](../../robosimstudio_materials/thread.py)).
Their routines drive the arms' joints, since unscrewing is a turn of the wrist.

| Demo | The job | What holds the thing |
| --- | --- | --- |
| [`unscrew_the_cap/`](unscrew_the_cap/README.md) | One hand holds a bottle by its base and the other turns its cap off, two turns in four strokes, and lifts it away. | Friction alone, on the cap and on the bottle; the cap stays on by its thread until it is turned off. |
| [`screw_in_the_bulb/`](screw_in_the_bulb/README.md) | Take a household lamp from its tray, set it in a lamp holder and screw it in until its eyelet presses on the holder's centre contact and it lights. | Friction alone on the glass. The holder and the tray are fixtures. |

Each of the first three scenes follows a task from the project's real-to-sim
work; its README says where it differs and why. `wipe_the_bar`'s sponge is a
SuperDex body, so it needs the `[superdex]` extra (Python 3.12); the others run
on a base install, the two thread demos on a GPU only.

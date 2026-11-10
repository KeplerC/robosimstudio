# Screw in the bulb

![Screw in the bulb](media/screw_in_the_bulb.webp)

A YAM screws a lamp into its holder and stops when the lamp's base contact
presses on the holder's.

```bash
uv run examples/08_demos/screw_in_the_bulb/screw_in_the_bulb.py
```

Press **Screw in the bulb** under *Routine*; **Reset** stands the lamp loose
again.

## The scene

`rss.bulb` and `rss.socket` are the `thread` kind
([`robosimstudio_materials/thread.py`](../../../robosimstudio_materials/thread.py)),
the same thread as the bottle's turned over: 3 mm pitch, two turns, 27.6 mm
over the crests.

- The holder stands on a post 0.30 m in front of the left arm, its floor 90 mm
  over `bench_075`. **It is a fixture**: a lamp holder is screwed to something.
- The lamp is a free body of 20 g: a threaded base under a 34 mm glass tube. It
  starts 2.15 turns out, past the thread's two, so it stands loose on the end
  of the holder's ridge with its tip 6.2 mm over the floor.
- A contact in the middle of the lamp's tip meets a pad that stands 0.3 mm off
  the holder's floor.

The physics is `rss.THREAD_60HZ`, as in
[`unscrew_the_cap`](../unscrew_the_cap/README.md).

## The routine

`routine(sim)` is a generator, one yield per control step, driving the left
arm's joints itself.

1. **The hand comes down over the glass**, tool down, its `roll_axis` grip
   point on the lamp's axis.
2. **Strokes.** The jaws close in two stages, the hand is aimed 1.5 mm under
   where it took hold (about 1.5 N down, which keeps the loose lamp on the
   thread until the first ridge catches), the wrist turns 200 degrees clockwise
   in 3.5 s, the jaws open, and the wrist turns back.
3. **Stop on the contact.** Each control step of a stroke the routine reads the
   force between the lamp's contact and the pad from the scene's contact probe
   (`sim.contacts`), and it stops turning above 2 N. That is in the fourth
   stroke.
4. **Let go** and back off 60 mm.

The routine reads the lamp's pose from the simulation to aim the hand, marked
in the script, and writes none. It runs about 31 s of simulated time.

## Measured

Four runs of the routine, three through
`tests/test_demo_screw_in_the_bulb.py` and one by the clip's recorder:

| | turns | most force on the contact | tip over the floor, hand gone | force on the contact, hand gone |
| --- | --- | --- | --- | --- |
| test run 1 | 1.990 | 2.3 N | 0.288 mm | 0.18 N |
| test run 2 | 1.998 | 3.1 N | 0.287 mm | 0.18 N |
| test run 3 | 2.066 | 5.3 N | 0.294 mm | 0.20 N |
| the clip | | | 0.30 mm | 0.20 N |

The lamp goes down 3 mm a turn within 5%, and the glass does not slip in the
fingers (within 6 degrees of the wrist a stroke). The lamp weighs 0.19 N, which
is what rests on the contact once the hand has gone.

## Limits

- **Where it stops varies by a sixth of a turn.** The tip lands on the pad
  after 1.97 turns with about 1.7 N on it. Some runs pass 2 N there; in others
  the thread first takes up its play and then clamps the lamp down.
- **The pad is rigid.** A real holder's centre contact is a sprung tab.
- **The glass is short**, 22 mm under its round end. The wrist's own collider
  reaches 65 mm down the roll axis and the grip point is at 95.7 mm, so about
  30 mm of a part fits above the fingers. On a 45 mm tube the palm sat on the
  lamp and turned it back out on every return stroke.
- **It needs a GPU**, as `rss.THREAD_60HZ` does.

# Sharpa Wave — Sharpa's own MJCF, the `with_flange` variant

What the `sharpa_wave` hand (`robosimstudio/robots/grippers/sharpa_wave.py`)
describes, and what `scripts/compose_ur5e_sharpa.py` bolts onto the UR5e to
make `embodiments/ur5e_sharpa/`. `assets/PROVENANCE.md` records the source,
the two edits made here, and the measurements.

## Where it came from

* **Sharpa Robotics**, `sharpa-robotics/sharpa-urdf-usd-xml` @ `0d19cac`,
  Apache-2.0 (`LICENSE.txt` and `NOTICE.txt` beside this file are the
  repository's). Files `wave_01/{left,right}_sharpa_wave/{left,right}_sharpa_wave_with_flange.xml`
  and the STL meshes they name, from `wave_01/{left,right}_sharpa_wave/meshes/`.
  The `with_wrist`, float-base, dual-hand and high-fidelity variants, the URDFs
  and the USDs are not vendored.

## What it is

A 22-joint five-finger hand: thumb (five joints), index, middle and ring
(four each), little finger (five), every joint driven by a `<position>`
actuator whose `kp`, `dampratio`, `armature`, `frictionloss` and `damping`
Sharpa calibrated against the hardware. The root body `*_hand_C_MC` carries the
flange and wrist meshes, so its origin is the tool flange: `+z` along the
fingers, `+x` out of the palm, `+y` towards the thumb. Colliders are the
per-phalanx meshes (convex hulls to MuJoCo); the five fingertip elastomers
carry `solref="0.06 0.9"`, softer than the default.

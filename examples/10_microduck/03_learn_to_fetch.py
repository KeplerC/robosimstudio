# SPDX-License-Identifier: Apache-2.0
"""Learn to fetch

Click the duck to the bowl a few times, press Train, and it fetches on its own: behaviour cloning on your button presses.

**New round** resets the scene, moves the bowl and puts the ball in the beak.
Bring the duck over with **Turn left**, **Turn right** (15 degrees), **Step**
(a few centimetres) and **Walk** (about 15 cm), and press **Drop it** with the
beak over the bowl. Each press is kept with where the bowl was from the beak;
**Train** fits a small classifier to them (``learn.py``) and **Fetch on its
own** lets it press the buttons. Three to six rounds are enough, and **Show
me** lets a scripted helper do one. Needs ``[microduck]``.
"""

import threading

from duck import DUCK_200HZ, DuckSpec, frame
from learn import NAMES, Demos, Driver, Moves, Policy, Rounds, observe, teacher
from props import ball, bowl

import robosimstudio as rss


def main() -> None:
    scene = rss.Scene(cell="bench_075", physics=DUCK_200HZ, name="learn_to_fetch")
    duck = scene.add(DuckSpec(at=(0.0, -0.40), yaw=1.5708, on="surface"))
    toy, target = scene.add(ball(at=(0.12, -0.15))), scene.add(bowl(at=(0.0, 0.25)))
    sim = scene.build()
    demos, moves, rounds = Demos(), Moves(duck), Rounds(sim, duck, toy, target)
    now = {"policy": None, "driver": None, "setup": None, "press": None}

    def page(studio):
        frame(studio, eye=(-0.30, 0.0, 1.70), look_at=(0.62, 0.0, 0.74))  # the whole bench
        gui, go = studio.gui, lambda choose, rec=None: Driver(duck, choose, lambda: target.position, record=rec)
        with gui.add_folder("Rounds"):
            new = gui.add_button("New round")
            new.on_click(
                lambda _: (setattr(studio, "reset_requested", True), now.update(setup=rounds.setup(), driver=None))
            )
            record, info = gui.add_checkbox("Record", True), gui.add_text("recorded", "0 rounds", disabled=True)
        with gui.add_folder("Buttons"):
            for name in NAMES:
                gui.add_button(name).on_click(lambda _, n=name: now.update(press=n))
            gui.add_button("Show me").on_click(lambda _: now.update(driver=go(teacher, demos)))
        with gui.add_folder("Learn"):
            train, fetch = gui.add_button("Train"), gui.add_button("Fetch on its own")
            status = gui.add_text("policy", "not trained", disabled=True)

        def fit():
            X, Y = demos.arrays()
            policy = Policy()
            policy.fit(X, Y)
            now["policy"], status.value = policy, f"trained on {len(X)} presses from {demos.rounds} rounds"

        train.on_click(lambda _: demos.rounds and threading.Thread(target=fit, daemon=True).start())
        fetch.on_click(lambda _: now["policy"] and now.update(driver=go(now["policy"])))

        def each():
            moves.step()
            if now["setup"] is not None:
                if next(now["setup"], "done") == "done":
                    now["setup"] = None
                    demos.start()
            elif now["driver"] is not None:
                now["driver"].step()
                if now["driver"].done and not duck.busy:
                    now["driver"] = None
            elif now["press"] is not None:
                if record.value and demos.episodes:
                    demos.add(observe(duck, target.position), now["press"])
                moves.press(now["press"])
            now["press"], info.value = None, f"{demos.rounds} rounds"

        studio.each_step(each)

    rss.view(sim, page=page, scenery=True)


if __name__ == "__main__":
    main()

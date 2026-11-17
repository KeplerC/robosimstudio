# SPDX-License-Identifier: Apache-2.0
"""Starter scene for the live page; it rebuilds on every save.

Contract: when this file has run, ``scene`` is an ``rss.Scene`` (poses in the
work-surface frame). Details: examples/live/REFERENCE.md.
"""

import robosimstudio as rss

scene = rss.Scene(cell="void", name="live")
scene.add(rss.robot("yam"))  # a scene needs something that moves before it will build

if __name__ == "__main__":
    rss.view(scene.build(control_mode="joint_position"))

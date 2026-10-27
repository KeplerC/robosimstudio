# SPDX-License-Identifier: Apache-2.0
"""The viser stand-ins agree with viser 1.1.1 where a panel can tell the difference.

:mod:`tests.fixtures.viser_stubs` is what every studio panel test builds into,
so where it parts from viser a panel test passes against behaviour the browser
never has. What is held here is each point a panel can observe: an assigned
value is cast to the widget's own type before it is compared, so an equal value
sent as an array or a list is no change and calls nobody back; a scene node's
click handlers are not its drag handlers; and a node added again under its name
replaces the one before it. Each is read off viser's own source
(``_gui_handles.py``'s ``value`` setter, ``_gui_api.py``'s update handler,
``_scene_api.py``'s node table), not off a browser.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.fixtures.viser_stubs import _StubGui, _StubScene

TIER = "fast"
"""No GPU and no browser: the stubs are plain objects."""


def _heard(widget) -> list:
    """What the widget's ``on_update`` callbacks were handed, as they are called."""
    heard: list = []
    widget.on_update(heard.append)
    return heard


class TestAValueIsCastAsViserCastsIt(unittest.TestCase):
    def test_an_equal_value_assigned_as_an_array_is_no_change(self) -> None:
        widget = _StubGui().add_vector3("Where", initial_value=(0.0, 0.5, 1.0))
        heard = _heard(widget)
        widget.value = np.array([0.0, 0.5, 1.0])
        self.assertEqual([], heard)
        self.assertEqual((0.0, 0.5, 1.0), widget.value)

    def test_an_equal_value_from_the_browser_as_a_list_is_no_change(self) -> None:
        widget = _StubGui().add_vector3("Where", initial_value=(0.0, 0.0, 1.0))
        heard = _heard(widget)
        widget.fire([0, 0, 1])
        self.assertEqual([], heard)

    def test_a_new_value_is_kept_in_the_widgets_own_types(self) -> None:
        widget = _StubGui().add_vector3("Where", initial_value=(0.0, 0.0, 0.0))
        heard = _heard(widget)
        widget.value = np.array([1, 2, 3])
        self.assertEqual((1.0, 2.0, 3.0), widget.value)
        self.assertIs(float, type(widget.value[0]))
        self.assertEqual(1, len(heard))
        widget.fire([4, 5, 6])
        self.assertEqual((4.0, 5.0, 6.0), widget.value)
        self.assertEqual(2, len(heard))

    def test_a_number_given_as_text_is_the_widgets_number(self) -> None:
        widget = _StubGui().add_number("Steps", initial_value=4.0)
        heard = _heard(widget)
        widget.fire("4")
        self.assertEqual([], heard, "4.0 and '4' are one value once cast")
        widget.value = 8
        self.assertEqual((8.0, float), (widget.value, type(widget.value)))
        self.assertEqual(1, len(heard))


class TestANodesClickIsNotItsDrag(unittest.TestCase):
    def test_a_drag_calls_the_update_handlers_alone(self) -> None:
        node = _StubScene().add_camera_frustum("/cams/front", fov=1.0, aspect=1.5)
        drags, clicks = [], []
        node.on_update(drags.append)
        node.on_click(clicks.append)
        node.fire(position=(1.0, 0.0, 0.0))
        self.assertEqual((1, 0), (len(drags), len(clicks)))

    def test_a_click_calls_the_click_handlers_alone(self) -> None:
        node = _StubScene().add_icosphere("/handle")
        drags, clicks = [], []
        node.on_update(drags.append)
        node.on_click(clicks.append)
        node.click()
        self.assertEqual((0, 1), (len(drags), len(clicks)))
        self.assertIs(node, clicks[0].target)


class TestANodeAddedAgainReplacesItself(unittest.TestCase):
    def test_a_light_added_again_under_its_name_is_one_node(self) -> None:
        scene = _StubScene()
        scene.add_light_directional("/lighting/key", position=(1.0, 0.0, 1.0))
        again = scene.add_light_directional("/lighting/key", position=(0.0, 1.0, 1.0))
        named = [node for node in scene.nodes if node.name == "/lighting/key"]
        self.assertEqual([again], named)
        self.assertIs(again, scene._handle_from_node_name["/lighting/key"])
        self.assertEqual((0.0, 1.0, 1.0), scene.lights["/lighting/key"]["position"])

    def test_any_node_added_again_replaces_the_one_before(self) -> None:
        scene = _StubScene()
        scene.add_box("/regions/bin", color=(1, 0, 0))
        scene.add_camera_frustum("/cams/front")
        scene.add_box("/regions/bin", color=(0, 1, 0))
        scene.add_camera_frustum("/cams/front")
        self.assertEqual(["/regions/bin", "/cams/front"], [node.name for node in scene.nodes])


if __name__ == "__main__":
    unittest.main()

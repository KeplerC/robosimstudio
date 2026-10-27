# SPDX-License-Identifier: Apache-2.0
"""The renderer factory's public surface: what a caller can read off it without building a renderer."""

import typing
import unittest

TIER = "fast"
"""Type hints only: no renderer is opened."""


class TestCreateRendererHints(unittest.TestCase):
    def test_type_hints_resolve(self):
        from robosimstudio.solvers.render import create_renderer

        self.assertIn("cameras", typing.get_type_hints(create_renderer))


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""One world's view of a fleet: an array read copies that world's rows, not the fleet's."""

import unittest

import numpy as np

TIER = "fast"
"""A Warp array on the CPU device: no GPU."""


class TestAWorldReadCopiesOnlyItsWorld(unittest.TestCase):
    def test_numpy_reads_the_slice_not_the_fleet(self):
        import warp as wp

        from robosimstudio_bench.envs.world_view import _ArraySlice

        whole = wp.array(np.arange(12, dtype=np.float32), device="cpu")

        class _NoWholeRead:
            """The fleet's array, refusing to be copied whole."""

            def __getitem__(self, key):
                return whole[key]

            def numpy(self):
                raise AssertionError("copied every world to read one")

        np.testing.assert_array_equal(_ArraySlice(_NoWholeRead(), 2, 3).numpy(), [6, 7, 8])


if __name__ == "__main__":
    unittest.main()
